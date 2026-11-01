"""Issue screening and source-located calls, imports, types and configuration."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import PurePosixPath
import re
import warnings

from .types import Observation, ROLES

SUFFIXES = {".py", ".go", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts",
            ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".rs", ".rb", ".php",
            ".sh", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".xml",
            ".md", ".rst", ".txt", ".mod", ".sum", ".lock", ".conf", ".properties",
            ".mk", ".cmake", ".gradle", ".proto", ".sql"}
CONFIG_NAMES = {"Makefile", "GNUmakefile", "makefile", "Dockerfile", "Containerfile",
                "Jenkinsfile", "Rakefile", "Gemfile", "Procfile", "BUILD", "WORKSPACE",
                "CMakeLists.txt", "meson.build", "configure", ".editorconfig", ".gitignore"}
CONFIG_SUFFIXES = {".toml", ".yaml", ".yml", ".ini", ".cfg", ".json", ".mod", ".sum",
                   ".lock", ".conf", ".properties", ".mk", ".cmake", ".gradle", ".xml"}
SKIP = {".git", "node_modules", "vendor", "dist", "build", ".venv", "__pycache__"}


def words(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z][a-z0-9]+", text.lower().replace("_", " "))


@dataclass(frozen=True)
class Entry:
    path: str
    start: int
    end: int
    name: str
    kind: str
    text: str

    @property
    def key(self):
        return self.path, self.start, self.end

    def descriptor(self):
        return dict(path=self.path, start=self.start, end=self.end, name=self.name, kind=self.kind)


def python_entries(path, text):
    lines = text.splitlines()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return []
    result, stack = [], []
    class Visitor(ast.NodeVisitor):
        def record(self, node, kind):
            lo = min([node.lineno]+[d.lineno for d in getattr(node, "decorator_list", [])])
            hi = node.end_lineno or node.lineno
            result.append(Entry(path, lo, hi, ".".join(stack+[node.name]), kind,
                                "\n".join(lines[lo-1:hi])))
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()
        def visit_ClassDef(self, node):
            self.record(node, "class")
        def visit_FunctionDef(self, node):
            self.record(node, "function")
        visit_AsyncFunctionDef = visit_FunctionDef
    Visitor().visit(tree)
    return result


def brace_entries(path, text):
    """Recognize common Go/JS/TS/C-family declaration spans without executing code."""
    lines = text.splitlines()
    result = []
    patterns = [
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)",
        r"^\s*(?:export\s+)?(?:abstract\s+)?(?:class|interface|enum|struct|trait)\s+(\w+)",
        r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*(?:\[.*?\])?\s*\(",
        r"^\s*type\s+(\w+)\s+(?:struct|interface)\b",
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*[(<]",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=.*(?:=>|function)",
        r"^\s*(?:(?:public|private|protected|static|virtual|inline|async|final|override)\s+)*(?:[\w:<>\[\]*&]+\s+)?([A-Za-z_$]\w*)\s*\([^;]*\)\s*(?::[^=]+)?\s*\{",
    ]
    for i, line in enumerate(lines):
        match = next((m for pat in patterns if (m := re.search(pat, line))), None)
        if not match or match.group(1) in {"if", "for", "while", "switch", "catch", "with"}:
            continue
        balance, opened, end = 0, False, i
        for j in range(i, min(len(lines), i+2000)):
            # Strings and line comments cannot create braces.
            clean = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|\x60[^\x60]*\x60|//.*", "", lines[j])
            opened = opened or "{" in clean
            balance += clean.count("{") - clean.count("}")
            end = j
            if opened and balance <= 0:
                break
            if not opened and j > i+12:
                end = i
                break
        result.append(Entry(path, i+1, end+1, match.group(1), "declaration",
                            "\n".join(lines[i:end+1])))
    return result


def _normal_path(value: str) -> str | None:
    """Normalize a repository-relative module path, rejecting traversal above root."""
    parts = []
    for part in value.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _without_comments(text: str) -> str:
    """Remove C-family comments while preserving quoted import specifiers."""
    output = []
    index = 0
    quote = None
    while index < len(text):
        character = text[index]
        if quote:
            output.append(character)
            if character == "\\" and quote != "`" and index + 1 < len(text):
                index += 1
                output.append(text[index])
            elif character == quote:
                quote = None
        elif character in "\"'`":
            quote = character
            output.append(character)
        elif text.startswith("//", index):
            end = text.find("\n", index)
            if end < 0:
                break
            output.append("\n")
            index = end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
            output.append("\n" * text[index:end].count("\n"))
            index = end - 1
        else:
            output.append(character)
        index += 1
    return "".join(output)


def _go_module_roots(texts: dict[str, str]) -> dict[str, str]:
    """Map declared Go module names to exact repository directories."""
    result = {}
    for path, text in sorted(texts.items()):
        if PurePosixPath(path).name != "go.mod" or set(PurePosixPath(path).parts) & SKIP:
            continue
        match = re.search(r"(?m)^\s*module\s+(?:\"([^\"]+)\"|([^\s]+))", _without_comments(text))
        if match:
            parent = str(PurePosixPath(path).parent)
            result[match.group(1) or match.group(2)] = "" if parent == "." else parent
    return result


def _module_tokens(text: str) -> list[tuple[str, str]]:
    """Small lexical stream; words inside literals cannot become import statements."""
    pattern = r'''(?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`[^`]*`)|(?P<word>[A-Za-z_$][\w$]*)|(?P<punct>[^\s])'''
    result = []
    for match in re.finditer(pattern, _without_comments(text)):
        kind = match.lastgroup
        value = match.group()
        result.append((kind, value[1:-1] if kind == "string" else value))
    return result


def _python_import_paths(path: str, text: str, texts: dict[str, str]) -> set[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return set()
    # Resolve package paths, never a repository-wide basename match. Common src
    # layout is a separate import root; relative imports retain their source root.
    roots = [""]
    if any(candidate.startswith("src/") and candidate.endswith(".py") for candidate in texts):
        roots.append("src")
    result = set()

    def resolve(module_path: str) -> bool:
        normalized = _normal_path(module_path)
        if normalized is None or not normalized:
            return False
        for candidate in (normalized + "/__init__.py", normalized + ".py"):
            if candidate in texts:
                result.add(candidate)
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.replace(".", "/")
                for root in roots:
                    if resolve("/".join(part for part in (root, name) if part)):
                        break
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").replace(".", "/")
            if node.level:
                parent = PurePosixPath(path).parent.parts
                root_length = 1 if path.startswith("src/") and "src" in roots else 0
                if node.level > len(parent) - root_length:
                    continue
                retained = parent[:len(parent) - node.level + 1]
                bases = ["/".join((*retained, module)).rstrip("/")]
            else:
                bases = ["/".join(part for part in (root, module) if part) for root in roots]
            for base in bases:
                found = resolve(base)
                for alias in node.names:
                    if alias.name != "*":
                        found = resolve(base + "/" + alias.name.replace(".", "/")) or found
                if found:
                    break
    return result


def _javascript_import_paths(path: str, text: str, texts: dict[str, str]) -> set[str]:
    specifiers = set()
    tokens = _module_tokens(text)
    for index, token in enumerate(tokens):
        if token[0] != "word" or token[1] not in {"import", "export", "require"}:
            continue
        following = tokens[index + 1:]
        if not following:
            continue
        if token[1] == "import" and following[0][0] == "string":
            specifiers.add(following[0][1])
        elif token[1] in {"import", "require"} and len(following) > 2:
            if following[0][1] == "(" and following[1][0] == "string" and following[2][1] == ")":
                specifiers.add(following[1][1])
        if token[1] in {"import", "export"}:
            for offset, item in enumerate(following):
                if item[1] == ";" or item == ("word", "import"):
                    break
                if item == ("word", "from") and offset + 1 < len(following):
                    if following[offset + 1][0] == "string":
                        specifiers.add(following[offset + 1][1])
                    break
    extensions = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json")
    result = set()
    for specifier in specifiers:
        if not specifier.startswith(("./", "../")) or "${" in specifier:
            # Bare npm packages are external unless a workspace binding is
            # explicitly model-grounded; do not link same-named local files.
            continue
        base = _normal_path(str(PurePosixPath(path).parent) + "/" + specifier)
        if base is None:
            continue
        candidates = [base]
        suffix = PurePosixPath(base).suffix
        if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
            stem = base[:-len(suffix)]
            candidates.extend(stem + extension for extension in (".ts", ".tsx", ".mts", ".cts"))
        if not suffix:
            candidates.extend(base + extension for extension in extensions)
            candidates.extend(base + "/index" + extension for extension in extensions)
        for candidate in candidates:
            if candidate in texts:
                result.add(candidate)
                break
    return result


def _go_import_paths(path: str, text: str, texts: dict[str, str],
                     modules: dict[str, str]) -> set[str]:
    specifications = []
    tokens = _module_tokens(text)
    for index, token in enumerate(tokens):
        if token != ("word", "import") or index + 1 >= len(tokens):
            continue
        if tokens[index + 1][1] == "(":
            for item in tokens[index + 2:]:
                if item[1] == ")":
                    break
                if item[0] == "string":
                    specifications.append(item[1])
        else:
            next_index = index + 1
            if tokens[next_index][0] != "string":
                next_index += 1
            if next_index < len(tokens) and tokens[next_index][0] == "string":
                specifications.append(tokens[next_index][1])
    result = set()
    for specifier in specifications:
        directory = None
        if specifier.startswith(("./", "../")):
            directory = _normal_path(str(PurePosixPath(path).parent) + "/" + specifier)
        else:
            for name in sorted(modules, key=len, reverse=True):
                if specifier == name or specifier.startswith(name + "/"):
                    relative = specifier[len(name):].lstrip("/")
                    directory = _normal_path(modules[name] + "/" + relative)
                    break
        if directory is None:
            continue
        packages = defaultdict(list)
        for candidate, content in texts.items():
            parent = str(PurePosixPath(candidate).parent)
            parent = "" if parent == "." else parent
            if parent != directory or not candidate.endswith(".go") or candidate.endswith("_test.go"):
                continue
            declaration = re.search(r"(?m)^\s*package\s+([A-Za-z_]\w*)", _without_comments(content))
            if declaration:
                packages[declaration.group(1)].append(candidate)
        if packages:
            # A Go directory has one importable package; build-tag variants can
            # coexist with commands. Prefer the dominant declared package, with
            # the import path basename resolving equal-sized alternatives.
            basename = specifier.rsplit("/", 1)[-1]
            package = min(packages, key=lambda name: (-len(packages[name]), name != basename, name))
            result.update(packages[package])
    return result


def _import_paths(path: str, texts: dict[str, str], go_modules: dict[str, str]) -> set[str]:
    text = texts[path]
    suffix = PurePosixPath(path).suffix
    if suffix == ".py":
        return _python_import_paths(path, text, texts)
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"}:
        return _javascript_import_paths(path, text, texts)
    if suffix == ".go":
        return _go_import_paths(path, text, texts, go_modules)
    return set()


class RepositoryIndex:
    def __init__(self, texts: dict[str, str], settings):
        self.settings = settings
        self.texts = {p:t for p,t in texts.items()
                      if (PurePosixPath(p).suffix in SUFFIXES or PurePosixPath(p).name in CONFIG_NAMES)
                      and not set(PurePosixPath(p).parts).intersection(SKIP)
                      and len(t.encode()) <= settings.max_file_bytes}
        self._go_modules = _go_module_roots(texts)
        self._import_cache = {}
        self.entries = []
        for path, text in sorted(self.texts.items()):
            symbols = python_entries(path, text) if path.endswith(".py") else brace_entries(path, text)
            self.entries.extend(symbols)
            # Top-level imports/configuration and files without named declarations remain searchable.
            lines = text.splitlines()
            for start in range(0, len(lines), settings.read_lines):
                end = min(len(lines), start+settings.read_lines)
                self.entries.append(Entry(path, start+1, end, "<module>", "module",
                                          "\n".join(lines[start:end])))
        self.documents = [Counter(words(e.path+" "+e.name+" "+e.text)) for e in self.entries]
        self.df = Counter(w for d in self.documents for w in d)
        self.avg_len = sum(map(lambda d:sum(d.values()), self.documents))/max(1,len(self.documents))
        self.by_name = defaultdict(list)
        self.by_path = defaultdict(list)
        for e in self.entries:
            self.by_path[e.path].append(e)
            if e.name != "<module>":
                self.by_name[e.name.rsplit(".",1)[-1]].append(e)
        self.module_paths = {}
        for p in self.texts:
            key = str(PurePosixPath(p).with_suffix("")).replace("/", ".")
            self.module_paths[key.removesuffix(".__init__")] = p

    def rank(self, issue: str, limit=40):
        query = Counter(words(issue))
        explicit = set(re.findall(r"(?:[\w.-]+/)+[\w.-]+\.\w+", issue))
        n = len(self.documents)
        scored = []
        for e, doc in zip(self.entries, self.documents):
            size = sum(doc.values())
            score = 0.0
            for word, freq in query.items():
                tf = doc.get(word, 0)
                if tf:
                    idf = math.log(1+(n-self.df[word]+0.5)/(self.df[word]+0.5))
                    score += freq*idf*tf*2.2/(tf+1.2*(0.25+0.75*size/max(1,self.avg_len)))
            if any(e.path.endswith(p) or p.endswith(e.path) for p in explicit):
                score += 6
            scored.append((score,e))
        return [e for _,e in sorted(scored, key=lambda x:(-x[0],x[1].key))[:limit]]

    def summary(self, issue):
        top = self.rank(issue, 30)
        return dict(files=len(self.texts),
                    symbols=sum(e.kind != "module" for e in self.entries),
                    anchors=[e.descriptor() for e in top],
                    relations=[r for e in top[:8] for r in self.relations(e)[:8]])

    def relations(self, entry: Entry):
        targets = {}
        calls = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", entry.text))
        types = set(re.findall(r"\b([A-Z]\w+)\b", entry.text))
        for name in sorted(calls | types):
            matches = self.by_name.get(name, [])
            # Common short identifiers are only useful when their binding is local.
            for other in matches[:12]:
                if other.key != entry.key:
                    targets[other.key] = dict(**other.descriptor(),
                                             relation="call" if name in calls else "type")
        if entry.path not in self._import_cache:
            self._import_cache[entry.path] = _import_paths(entry.path, self.texts, self._go_modules)
        for p in sorted(self._import_cache[entry.path]):
            if p != entry.path and self.by_path[p]:
                e = self.by_path[p][0]
                targets[e.key] = dict(**e.descriptor(), relation="import")
        # Reverse call/import edges reveal callers whose contracts constrain the edit.
        short_name = entry.name.rsplit(".", 1)[-1]
        if short_name != "<module>":
            needle = re.compile(r"\b"+re.escape(short_name)+r"\s*\(")
            for other in self.entries:
                if other.path != entry.path and other.kind != "module" and needle.search(other.text):
                    targets[other.key] = dict(**other.descriptor(), relation="caller")
                    if len(targets) >= 40:
                        break
        # Binding names in configuration are inspected as supporting source.
        names = set(words(entry.name + " " + " ".join(types)))
        for p, text in self.texts.items():
            if PurePosixPath(p).suffix in CONFIG_SUFFIXES or PurePosixPath(p).name in CONFIG_NAMES:
                if names.intersection(words(text)) and self.by_path[p]:
                    e = self.by_path[p][0]
                    targets[e.key] = dict(**e.descriptor(), relation="configuration")
                    if len(targets) >= 48:
                        break
        return [targets[k] for k in sorted(targets)][:48]

    def locate(self, path, start, end):
        if (path not in self.texts or type(start) is not int or type(end) is not int
                or start < 1 or end < start):
            return None
        lines = self.texts[path].splitlines()
        if end > len(lines):
            return None
        choices = self.by_path.get(path, [])
        containing = next((e for e in choices if e.start <= start and e.end >= end), None)
        if containing is not None:
            return containing
        # Cross-symbol or cross-chunk candidates navigate from their own source.
        return Entry(path, start, end, "<module>", "interval", "\n".join(lines[start-1:end]))

    def validate(self, observation: Observation, repo, commit):
        if not observation.locations or not observation.roles or not set(observation.roles) <= ROLES:
            return False
        for l in observation.locations:
            if l.repo != repo or l.commit != commit or l.path not in self.texts:
                return False
            lines = self.texts[l.path].splitlines()
            if l.end > len(lines) or "\n".join(lines[l.start-1:l.end]) != observation.text:
                return False
        from hashlib import sha256
        return observation.id == sha256(observation.text.encode()).hexdigest()

