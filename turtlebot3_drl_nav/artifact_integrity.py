"""Immutable run-output inventory written before the COMPLETE marker."""

import hashlib
import os
import sys
from typing import List


MANIFEST = "RUN_FILES.sha256"
EXCLUDED_NAMES = {MANIFEST, "COMPLETE", "FAILED", "INTERRUPTED"}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_files(run_dir: str) -> List[str]:
    paths: List[str] = []
    for base, dirs, files in os.walk(run_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name in EXCLUDED_NAMES or name.endswith((".tmp", ".pyc")):
                continue
            paths.append(os.path.relpath(os.path.join(base, name), run_dir))
    return sorted(paths)


def symlink_directories(run_dir: str) -> List[str]:
    """Find directory symlinks without descending into them."""
    links: List[str] = []
    for base, dirs, _ in os.walk(run_dir, followlinks=False):
        for name in list(dirs):
            path = os.path.join(base, name)
            if os.path.islink(path):
                links.append(os.path.relpath(path, run_dir))
                dirs.remove(name)
    return sorted(links)


def write_manifest(run_dir: str) -> str:
    path = os.path.join(run_dir, MANIFEST)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to replace {path}")
    files = run_files(run_dir)
    links = symlink_directories(run_dir) + [rel for rel in files if os.path.islink(os.path.join(run_dir, rel))]
    if links:
        raise ValueError(f"run output contains symbolic links: {links}")
    special = [rel for rel in files if not os.path.isfile(os.path.join(run_dir, rel))]
    if special:
        raise ValueError(f"run output contains non-regular files: {special}")
    lines = [f"{sha256_file(os.path.join(run_dir, rel))}  {rel}" for rel in files]
    tmp = path + ".tmp"
    with open(tmp, "x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def verify_manifest(run_dir: str) -> List[str]:
    path = os.path.join(run_dir, MANIFEST)
    if not os.path.isfile(path):
        return [f"{MANIFEST} missing"]
    problems: List[str] = []
    problems.extend(f"symbolic-link directory {rel}" for rel in symlink_directories(run_dir))
    declared: List[str] = []
    with open(path, encoding="utf-8") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            expected, separator, rel = line.partition("  ")
            if not separator or len(expected) != 64 or os.path.isabs(rel) or os.path.normpath(rel) != rel or rel.startswith(".."):
                problems.append(f"invalid manifest line {line!r}")
                continue
            if rel in declared:
                problems.append(f"duplicate path {rel}")
                continue
            declared.append(rel)
            target = os.path.join(run_dir, rel)
            if not os.path.isfile(target):
                problems.append(f"missing {rel}")
            elif os.path.islink(target):
                problems.append(f"symbolic link {rel}")
            elif sha256_file(target) != expected:
                problems.append(f"digest mismatch {rel}")
    actual = run_files(run_dir)
    problems.extend(f"undeclared file {rel}" for rel in sorted(set(actual) - set(declared)))
    problems.extend(f"manifest-only path {rel}" for rel in sorted(set(declared) - set(actual)))
    return problems


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in ("write", "verify"):
        print("usage: python3 -m turtlebot3_drl_nav.artifact_integrity write|verify RUN_DIR", file=sys.stderr)
        return 2
    command, run_dir = args
    if command == "write":
        print(write_manifest(run_dir))
        return 0
    problems = verify_manifest(run_dir)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
