"""Build an anonymous source-only release from an explicit file allowlist."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import re
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

TOP_FILES = {"README.md","LICENSE","pyproject.toml","requirements.txt",".gitignore",".env.example"}
DIRECTORIES = {"src","tests","configs","docs","scripts"}
SUFFIXES = {".py",".md",".toml",".txt",".yml",".yaml"}
EXCLUDE = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules"}
ENGLISH_RELEASE_TEXT = re.compile(
    r"[\u2e80-\u303f\u3100-\u312f\u31a0-\u31bf\u31c0-\u31ef\u3400-\u4dbf"
    r"\u4e00-\u9fff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef\U00020000-\U000323af]")
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "API credential": re.compile(r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b"),
    "credential assignment": re.compile(
        r"""(?im)^\s*(?:api_key|secret_key|password|access_token)\s*=\s*["']([^"'\n]{12,})["']"""),
    "personal local path": re.compile(r"(?i)[A-Z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}


def release_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink() or set(relative.parts)&EXCLUDE:
            continue
        if len(relative.parts) == 1:
            if relative.name not in TOP_FILES:
                continue
        elif relative.parts[0] not in DIRECTORIES or path.suffix not in SUFFIXES:
            continue
        if ".egg-info" in str(relative):
            continue
        yield path, relative


def build(root: Path, output: Path) -> dict:
    root = root.resolve()
    files = list(release_files(root))
    if not (root/"README.md").is_file() or not files:
        raise ValueError("Choose the LOCUS source project")
    for path, relative in files:
        text = path.read_text(encoding="utf-8")
        if ENGLISH_RELEASE_TEXT.search(relative.as_posix()) or ENGLISH_RELEASE_TEXT.search(text):
            raise ValueError(f"English release text required: {relative.as_posix()}")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"Release scan: {label} in {relative.as_posix()}")
    output = output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,"w",compression=ZIP_DEFLATED,compresslevel=9) as archive:
        for path,relative in files:
            info = ZipInfo("LOCUS-ICDE/"+relative.as_posix(),date_time=(2026,1,1,0,0,0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info,path.read_bytes())
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Release ZIP failed its integrity check")
    return dict(files=len(files),bytes=output.stat().st_size,
                sha256=sha256(output.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    print(build(args.root,args.output))


if __name__ == "__main__":
    main()

