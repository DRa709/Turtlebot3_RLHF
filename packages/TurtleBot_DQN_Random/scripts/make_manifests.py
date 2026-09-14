import posixpath
#!/usr/bin/env python3
"""Regenerate package manifests without importing release Python modules."""

import hashlib
import os
import stat
import sys
import tempfile
from typing import List


sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_MANIFEST = "RELEASE_MANIFEST.sha256"
SHARED_LAYER_LIST = "SHARED_LAYER_FILES.txt"
SHARED_LAYER_MANIFEST = "SHARED_LAYER_MANIFEST.sha256"
FORBIDDEN_NAMES = {"__pycache__", ".pytest_cache", ".git", "build", "install", "log"}
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".egg-info", ".ipynb")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _invalid_paths(root: str) -> List[str]:
    found: List[str] = []
    for base, dirs, files in os.walk(root, followlinks=False):
        retained: List[str] = []
        for name in sorted(dirs):
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root)
            if os.path.islink(path):
                found.append(f"symbolic-link directory {relative}")
            elif name in FORBIDDEN_NAMES or name.endswith(".egg-info"):
                found.append(f"forbidden path {relative}/")
            else:
                retained.append(name)
        dirs[:] = retained
        for name in sorted(files):
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root)
            try:
                mode = os.lstat(path).st_mode
            except OSError as exc:
                found.append(f"unreadable path {relative}: {exc}")
                continue
            if not stat.S_ISREG(mode):
                found.append(f"non-regular path {relative}")
            elif name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES):
                found.append(f"forbidden path {relative}")
    return sorted(found)


def _release_files(root: str) -> List[str]:
    paths: List[str] = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(dirs)
        for name in sorted(files):
            if name == RELEASE_MANIFEST:
                continue
            paths.append(os.path.relpath(os.path.join(base, name), root).replace('\\', '/'))
    return sorted(paths)


def _shared_files(root: str) -> List[str]:
    list_path = os.path.join(root, SHARED_LAYER_LIST)
    with open(list_path, encoding="utf-8") as stream:
        paths = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate path in {SHARED_LAYER_LIST}")
    for relative in paths:
        rel_posix = relative.replace('\\', '/')
        if posixpath.isabs(rel_posix) or posixpath.normpath(rel_posix) != rel_posix or rel_posix.startswith(".."):
            raise ValueError(f"unsafe shared-layer path: {relative}")
        target = os.path.join(root, relative)
        if os.path.islink(target) or not os.path.isfile(target):
            raise ValueError(f"shared-layer path is missing or not regular: {relative}")
    return paths


def _write_atomic(path: str, lines: List[str]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", dir=os.path.dirname(path), text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if hasattr(os, 'fchmod'): os.fchmod(stream.fileno(), 0o644)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    invalid = _invalid_paths(ROOT)
    if invalid:
        print("refusing to generate manifests with invalid release paths:", file=sys.stderr)
        for problem in invalid:
            print(f"  {problem}", file=sys.stderr)
        return 2
    try:
        shared_paths = _shared_files(ROOT)
        shared_lines = [f"{_sha256_file(os.path.join(ROOT, path))}  {path}" for path in shared_paths]
        _write_atomic(os.path.join(ROOT, SHARED_LAYER_MANIFEST), shared_lines)
        release_paths = _release_files(ROOT)
        release_lines = [f"{_sha256_file(os.path.join(ROOT, path))}  {path}" for path in release_paths]
        _write_atomic(os.path.join(ROOT, RELEASE_MANIFEST), release_lines)
    except (OSError, ValueError) as exc:
        print(f"manifest generation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"{SHARED_LAYER_MANIFEST}: {len(shared_lines)} files; "
        f"{RELEASE_MANIFEST}: {len(release_lines)} files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
