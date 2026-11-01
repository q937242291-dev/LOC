"""Static import resolution is tied to package paths and declared module roots."""
import unittest

from locus.config import Settings
from locus.repository import RepositoryIndex


class RepositoryImportsTests(unittest.TestCase):
    def imports(self, texts, path):
        index = RepositoryIndex(texts, Settings())
        entry = next(entry for entry in index.by_path[path] if entry.kind == "module")
        return {relation["path"] for relation in index.relations(entry)
                if relation["relation"] == "import"}

    def test_python_relative_levels_and_src_root(self):
        texts = {
            "src/pkg/__init__.py": "# package\n",
            "src/pkg/sub/__init__.py": "# subpackage\n",
            "src/pkg/sub/worker.py": "from ..helpers import value\nfrom . import local\nimport pkg.other\n",
            "src/pkg/helpers.py": "value = 1\n",
            "src/pkg/sub/local.py": "enabled = True\n",
            "src/pkg/other.py": "answer = 42\n",
            "src/unrelated/helpers.py": "value = 0\n",
        }
        self.assertEqual(self.imports(texts, "src/pkg/sub/worker.py"), {
            "src/pkg/helpers.py", "src/pkg/sub/__init__.py",
            "src/pkg/sub/local.py", "src/pkg/other.py"})

    def test_python_from_parent_module_and_no_basename_guess(self):
        texts = {
            "pkg/sub/worker.py": "from ..module import thing\nfrom ...missing import other\n",
            "pkg/module.py": "thing = 1\n",
            "pkg/sub/module.py": "thing = 2\n",
            "elsewhere/missing.py": "other = 3\n",
        }
        self.assertEqual(self.imports(texts, "pkg/sub/worker.py"), {"pkg/module.py"})

    def test_javascript_named_default_require_and_side_effect_import(self):
        texts = {
            "src/app.ts": "import Default from './default';\n"
                          "import {\n helper as h\n} from './tools/helper';\n"
                          "const legacy = require('../shared/legacy');\nimport './init';\n"
                          "// import Ignored from './ignored';\n"
                          "const misleading = \"require('./ignored')\";\n",
            "src/default.ts": "export default 1;\n",
            "src/tools/helper.ts": "export const helper = 1;\n",
            "shared/legacy.js": "module.exports = 2;\n",
            "src/init/index.ts": "export const ready = true;\n",
            "src/ignored.ts": "export default 0;\n",
            "unrelated/helper.ts": "export const helper = 9;\n",
        }
        self.assertEqual(self.imports(texts, "src/app.ts"), {
            "src/default.ts", "src/tools/helper.ts", "shared/legacy.js", "src/init/index.ts"})

    def test_typescript_js_extension_and_dynamic_import(self):
        texts = {
            "src/main.ts": "import type { T } from './types.js';\n"
                           "const lazy = import('./lazy');\nimport external from 'external';\n",
            "src/types.ts": "export type T = string;\n",
            "src/lazy/index.ts": "export const value = 1;\n",
            "external.ts": "export default 0;\n",
        }
        self.assertEqual(self.imports(texts, "src/main.ts"), {"src/types.ts", "src/lazy/index.ts"})

    def test_go_module_import_alias_block_and_relative(self):
        texts = {
            "go.mod": "module example.org/project\n\ngo 1.23\n",
            "cmd/main.go": 'package main\nimport renamed "example.org/project/internal/kit"\n'
                           'import (\n "fmt"\n "example.org/project/config"\n _ "../local"\n)\n',
            "internal/kit/a.go": "package tools\nfunc A() {}\n",
            "internal/kit/b.go": "package tools\nfunc B() {}\n",
            "internal/kit/a_test.go": "package tools_test\nfunc TestA() {}\n",
            "config/config.go": "package config\nvar Enabled = true\n",
            "local/local.go": "package local\nvar Value = 1\n",
            "other/kit/a.go": "package kit\nfunc A() {}\n",
            "fmt.go": "package fmt\nfunc Println() {}\n",
        }
        self.assertEqual(self.imports(texts, "cmd/main.go"), {
            "internal/kit/a.go", "internal/kit/b.go", "config/config.go", "local/local.go"})

    def test_nested_go_module_and_commented_import(self):
        texts = {
            "go.mod": "module example.org/root\n",
            "tools/go.mod": "module example.org/root/tools\n",
            "tools/cmd/main.go": 'package main\nimport "example.org/root/tools/pkg"\n'
                                 '// import "example.org/root/ignored"\n',
            "tools/pkg/a.go": "package pkg\nfunc Work() {}\n",
            "pkg/a.go": "package pkg\nfunc Work() {}\n",
            "ignored/a.go": "package ignored\nfunc Work() {}\n",
        }
        self.assertEqual(self.imports(texts, "tools/cmd/main.go"), {"tools/pkg/a.go"})


class RepositoryLocationTests(unittest.TestCase):
    def test_cross_chunk_and_symbol_interval_uses_its_own_relations(self):
        text = ("def unrelated():\n"
                "    return unrelated_dependency()\n"
                "\n"
                "import local_import\n"
                "\n"
                "def selected():\n"
                "    value: DesiredType = None\n"
                "    return desired_helper(value)\n"
                "\n"
                "def neighbor():\n"
                "    return desired_helper(None)\n")
        texts = {
            "module.py": text,
            "desired.py": "class DesiredType:\n    pass\n\ndef desired_helper(value):\n    return value\n",
            "unrelated.py": "def unrelated_dependency():\n    return 1\n",
            "local_import.py": "ready = True\n",
            "settings.json": '{"DesiredType": "enabled"}\n',
        }
        index = RepositoryIndex(texts, Settings(read_lines=4))
        self.assertEqual(index.by_path["module.py"][0].name,"unrelated")
        self.assertFalse(any(entry.start <= 7 and entry.end >= 11
                             for entry in index.by_path["module.py"]))
        located = index.locate("module.py",7,11)
        self.assertEqual(located.key,("module.py",7,11))
        self.assertEqual(located.text,"\n".join(text.splitlines()[6:11]))
        self.assertEqual(located.kind,"interval")
        relations = index.relations(located)
        by_kind = {}
        for relation in relations:
            by_kind.setdefault(relation["relation"],set()).add(relation["path"])
        self.assertIn("desired.py",by_kind["call"])
        self.assertIn("desired.py",by_kind["type"])
        self.assertIn("local_import.py",by_kind["import"])
        self.assertIn("settings.json",by_kind["configuration"])
        self.assertNotIn("unrelated.py",{relation["path"] for relation in relations})

    def test_containing_symbol_keeps_reverse_caller_navigation(self):
        texts = {"module.py":"def target():\n    return helper()\n",
                 "helper.py":"def helper():\n    return 1\n",
                 "caller.py":"def caller():\n    return target()\n"}
        index = RepositoryIndex(texts,Settings())
        located = index.locate("module.py",2,2)
        self.assertEqual(located.name,"target")
        relations = index.relations(located)
        self.assertTrue(any(r["path"] == "caller.py" and r["relation"] == "caller" for r in relations))
        self.assertTrue(any(r["path"] == "helper.py" and r["relation"] == "call" for r in relations))

    def test_interval_lookup_returns_none_for_unavailable_or_invalid_coordinates(self):
        index = RepositoryIndex({"module.py":"first = 1\nsecond = 2\n"},Settings(read_lines=1))
        for path,start,end in (("missing.py",1,1),("module.py",0,1),("module.py",2,1),
                               ("module.py",1,3),("module.py",True,2),("module.py",1,2.0)):
            with self.subTest(path=path,start=start,end=end):
                self.assertIsNone(index.locate(path,start,end))


if __name__ == "__main__":
    unittest.main()
