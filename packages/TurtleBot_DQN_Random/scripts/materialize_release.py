#!/usr/bin/env python3
"""Materialize exactly one authenticated release into a new directory.

Only files named by RELEASE_MANIFEST.sha256, plus the manifest itself, are
copied. Forbidden caches, bytecode, build products, links, or undeclared files
make the operation fail before any release module is imported.
"""

import argparse
import hashlib
import os
import shutil
import stat
import sys
from typing import Dict, List, Tuple


sys.dont_write_bytecode = True

FORBIDDEN_NAMES = {"__pycache__", ".pytest_cache", ".git", "build", "install", "log"}
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".egg-info")
RELEASE_MANIFEST = "RELEASE_MANIFEST.sha256"


def _forbidden_paths(root: str) -> List[str]:
    found: List[str] = []
    for base, dirs, files in os.walk(root, followlinks=False):
        retained: List[str] = []
        for name in sorted(dirs):
            rel = os.path.relpath(os.path.join(base, name), root)
            if name in FORBIDDEN_NAMES or name.endswith(".egg-info"):
                found.append(rel + "/")
            else:
                retained.append(name)
        dirs[:] = retained
        for name in sorted(files):
            if name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES) or name.endswith(".ipynb"):
                found.append(os.path.relpath(os.path.join(base, name), root))
    return sorted(found)


def _fail_forbidden(root: str) -> None:
    forbidden = _forbidden_paths(root)
    if forbidden:
        raise ValueError("forbidden release paths: " + ", ".join(forbidden))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(root: str) -> Dict[str, str]:
    path = os.path.join(root, RELEASE_MANIFEST)
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{RELEASE_MANIFEST} is missing or is not a regular file")
    declared: Dict[str, str] = {}
    with open(path, encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.rstrip("\n")
            digest, separator, relative = line.partition("  ")
            if (
                separator != "  "
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not relative
                or os.path.isabs(relative)
                or os.path.normpath(relative) != relative
                or relative.startswith("..")
                or relative == RELEASE_MANIFEST
            ):
                raise ValueError(f"invalid manifest line {number}: {line!r}")
            if relative in declared:
                raise ValueError(f"duplicate manifest path: {relative}")
            declared[relative] = digest
    if not declared:
        raise ValueError("release manifest is empty")
    return declared


def _inventory(root: str) -> Tuple[List[str], List[str]]:
    regular: List[str] = []
    invalid: List[str] = []
    for base, dirs, files in os.walk(root, followlinks=False):
        retained: List[str] = []
        for name in sorted(dirs):
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root)
            if os.path.islink(path):
                invalid.append(f"symbolic-link directory {relative}")
            else:
                retained.append(name)
        dirs[:] = retained
        for name in sorted(files):
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root)
            if name == RELEASE_MANIFEST:
                continue
            try:
                mode = os.lstat(path).st_mode
            except OSError as exc:
                invalid.append(f"unreadable path {relative}: {exc}")
                continue
            if not stat.S_ISREG(mode):
                invalid.append(f"non-regular release path {relative}")
            else:
                regular.append(relative)
    return sorted(regular), sorted(invalid)


def _verify_without_import(root: str) -> List[str]:
    declared = _read_manifest(root)
    actual, invalid = _inventory(root)
    problems = list(invalid)
    problems.extend(f"undeclared file {path}" for path in sorted(set(actual) - set(declared)))
    problems.extend(f"missing file {path}" for path in sorted(set(declared) - set(actual)))
    for relative in sorted(set(actual) & set(declared)):
        if _sha256_file(os.path.join(root, relative)) != declared[relative]:
            problems.append(f"digest mismatch {relative}")
    return problems


def materialize(source: str, destination: str) -> int:
    source = os.path.realpath(source)
    destination = os.path.abspath(destination)
    if not os.path.isdir(source):
        raise ValueError(f"source is not a directory: {source}")
    if os.path.lexists(destination):
        raise ValueError(f"destination already exists: {destination}")
    _fail_forbidden(source)
    problems = _verify_without_import(source)
    if problems:
        raise ValueError("source release is not authenticated: " + "; ".join(problems))

    os.makedirs(destination, mode=0o755)
    declared = _read_manifest(source)
    relatives = sorted(declared) + [RELEASE_MANIFEST]
    for relative in relatives:
        source_path = os.path.join(source, relative)
        destination_path = os.path.join(destination, relative)
        if os.path.islink(source_path) or not os.path.isfile(source_path):
            raise ValueError(f"release entry is not a regular file: {relative}")
        os.makedirs(os.path.dirname(destination_path), mode=0o755, exist_ok=True)
        shutil.copy2(source_path, destination_path, follow_symlinks=False)

    _fail_forbidden(destination)
    copied_problems = _verify_without_import(destination)
    if copied_problems:
        raise ValueError("materialized release failed authentication: " + "; ".join(copied_problems))
    print(f"materialized {len(relatives)} authenticated files at {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        return materialize(args.source, args.destination)
    except (OSError, ValueError) as exc:
        print(f"materialize_release: FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
