#!/usr/bin/env python3
"""Check a code-only release manifest; optionally archive exactly its files."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = "release_manifest.json"
SAFE_SUFFIXES = {".py", ".md", ".txt", ".json", ".cff"}
SECRET_PATTERNS = (
    re.compile(r"\b(?:ghp_|github_pat_|olp_)[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
)


def validate():
    manifest_path = ROOT / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("Manifest must not be a symlink.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files_sha256"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Missing file hash inventory.")
    allowed = set(expected) | {MANIFEST}
    discovered = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts:
            continue  # Git history is never part of the archive allowlist.
        if path.is_symlink():
            raise ValueError("Symlink not allowed: " + relative.as_posix())
        if path.is_file():
            discovered.add(relative.as_posix())
    if discovered != allowed:
        raise ValueError("Unexpected or missing files: " + str(sorted(discovered ^ allowed)))
    for name in sorted(expected):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError("Invalid inventory path.")
        if name not in {".gitignore", "LICENSE"} and relative.suffix not in SAFE_SUFFIXES:
            raise ValueError("Not a source/documentation file: " + name)
        raw = (ROOT / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError("Hash mismatch: " + name)
        content = raw.decode("utf-8")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError("Possible credential in " + name + "; do not publish.")
        private_prefix = "/" + "local/home/"
        if private_prefix in content:
            raise ValueError("Private absolute path in " + name)
        if relative.suffix == ".py":
            ast.parse(content, filename=name)
    print(f"Validated {len(expected)} source/documentation files: inventory, hashes, syntax, limited disclosure scan.")
    print("This check is not exhaustive privacy review or validation of scientific results.")
    return sorted(allowed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        help="Create an exclusive code-only ZIP outside this directory; never upload.")
    args = parser.parse_args()
    names = validate()
    if args.archive is not None:
        destination = args.archive.expanduser().resolve()
        if destination == ROOT or ROOT in destination.parents:
            raise ValueError("Archive must be outside the release directory.")
        if destination.exists():
            raise FileExistsError("Archive already exists; refusing to overwrite.")
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                member = zipfile.ZipInfo("fitbit-cvd-analysis/" + name, (2026, 9, 17, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                archive.writestr(member, (ROOT / name).read_bytes())
        print("Created local code-only archive; nothing uploaded.")


if __name__ == "__main__":
    main()
