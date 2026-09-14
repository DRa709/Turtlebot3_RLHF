import posixpath
"""Run identity: configuration digest, shared-layer digest, container digest,
and the identity block every recorded row carries.

Shared layer. No ROS dependency. Usable as a module and as a command:

    python3 -m turtlebot3_drl_nav.identity config-digest <package_root>
    python3 -m turtlebot3_drl_nav.identity shared-layer-digest <package_root>
    python3 -m turtlebot3_drl_nav.identity verify-shared-layer <package_root>
"""

import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

IDENTITY_FIELDS = [
    "experiment", "run_id", "algorithm", "algorithm_version", "action_space", "arm",
    "learning_seed", "world_id", "world_seed", "initialization_seed", "dynamic_obstacle_seed",
    "evaluation_seed", "config_sha256", "container_sha256", "shared_layer_sha256",
    "release_sha256", "package_version", "phase_type", "phase_label",
]

CONFIG_DIGEST_GLOBS = ("config", "worlds")
SHARED_LAYER_LIST = "SHARED_LAYER_FILES.txt"
SHARED_LAYER_MANIFEST = "SHARED_LAYER_MANIFEST.sha256"
RELEASE_MANIFEST = "RELEASE_MANIFEST.sha256"
RELEASE_EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", "build", "install", "log"}


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _length_prefixed_digest(root: str, relative_paths: Sequence[str]) -> str:
    """Digest of sorted (name, bytes) pairs, each length-prefixed, so neither a
    rename nor a boundary shift between files can collide."""
    digest = hashlib.sha256()
    for rel in sorted(relative_paths):
        name = rel.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with open(os.path.join(root, rel), "rb") as stream:
            data = stream.read()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def config_files(package_root: str) -> List[str]:
    out: List[str] = []
    for folder in CONFIG_DIGEST_GLOBS:
        base = os.path.join(package_root, folder)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if os.path.isfile(path) and not name.startswith("."):
                out.append(f"{folder}/{name}")
    if not out:
        raise FileNotFoundError(f"no configuration files under {package_root}")
    return out


def config_digest(package_root: str) -> str:
    """Digest of config/* and worlds/* — the frozen experimental configuration."""
    return _length_prefixed_digest(package_root, config_files(package_root))


def shared_layer_files(package_root: str) -> List[str]:
    path = os.path.join(package_root, SHARED_LAYER_LIST)
    with open(path, encoding="utf-8") as stream:
        files = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    missing = [f for f in files if not os.path.isfile(os.path.join(package_root, f))]
    if missing:
        raise FileNotFoundError(f"shared-layer files missing: {missing}")
    return files


def shared_layer_manifest_lines(package_root: str) -> List[str]:
    return [f"{sha256_file(os.path.join(package_root, f))}  {f}" for f in shared_layer_files(package_root)]


def shared_layer_digest(package_root: str) -> str:
    """Digest of the shared-layer manifest text: identical across packages iff
    every common file is byte-identical."""
    return sha256_text("\n".join(shared_layer_manifest_lines(package_root)) + "\n")


def release_files(package_root: str) -> List[str]:
    out: List[str] = []
    for base, dirs, files in os.walk(package_root):
        dirs[:] = sorted(d for d in dirs if d not in RELEASE_EXCLUDE_DIRS and not d.endswith(".egg-info"))
        for name in sorted(files):
            if name == RELEASE_MANIFEST or name.endswith(".pyc"):
                continue
            out.append(os.path.relpath(os.path.join(base, name), package_root).replace("\\", "/"))
    return sorted(out)


def _symlink_directories(root: str) -> List[str]:
    """Find directory symlinks that ``os.walk(..., followlinks=False)`` would
    otherwise leave outside the file inventory."""
    links: List[str] = []
    for base, dirs, _ in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = os.path.join(base, name)
            if os.path.islink(path):
                links.append(os.path.relpath(path, root).replace("\\", "/"))
                dirs.remove(name)
    return sorted(links)


def release_digest(package_root: str) -> str:
    problems = verify_manifest(package_root, RELEASE_MANIFEST)
    if problems:
        raise ValueError("release manifest is not satisfied: " + "; ".join(problems))
    return sha256_file(os.path.join(package_root, RELEASE_MANIFEST))


def verify_manifest(package_root: str, manifest_name: str) -> List[str]:
    """Return a list of mismatches (empty when the manifest is satisfied)."""
    problems: List[str] = []
    path = os.path.join(package_root, manifest_name)
    if not os.path.isfile(path):
        return [f"{manifest_name} missing"]
    problems.extend(f"symbolic-link directory {rel}" for rel in _symlink_directories(package_root))
    declared: List[str] = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line:
                continue
            expected, _, rel = line.partition("  ")
            if not expected or not rel or posixpath.isabs(rel) or posixpath.normpath(rel) != rel or rel.startswith(".."):
                problems.append(f"invalid manifest line {line!r}")
                continue
            if rel in declared:
                problems.append(f"duplicate manifest path {rel}")
                continue
            declared.append(rel)
            target = os.path.join(package_root, rel)
            if not os.path.isfile(target):
                problems.append(f"missing {rel}")
            elif os.path.islink(target):
                problems.append(f"symbolic link {rel}")
            elif sha256_file(target) != expected:
                problems.append(f"digest mismatch {rel}")
    if manifest_name == RELEASE_MANIFEST:
        actual = release_files(package_root)
        extras = sorted(set(actual) - set(declared))
        if extras:
            problems.extend(f"undeclared file {rel}" for rel in extras)
        omitted = sorted(set(declared) - set(actual))
        if omitted:
            problems.extend(f"manifest path outside release inventory {rel}" for rel in omitted)
    return problems


def build_identity(
    experiment: str,
    algorithm: str,
    algorithm_version: str,
    arm: str,
    learning_seed: int,
    world_id: str,
    world_seed: int,
    initialization_seed: int,
    dynamic_obstacle_seed: int,
    evaluation_seed: int,
    config_sha256: str,
    container_sha256: str,
    shared_layer_sha256: str,
    release_sha256: str,
    package_version: str,
    phase_type: str,
    phase_label: str,
    job_token: str,
) -> Dict[str, object]:
    if phase_label not in ("calibration", "pilot", "controlled"):
        raise ValueError("phase_label must be calibration, pilot or controlled")
    if phase_type not in ("training", "evaluation"):
        raise ValueError("phase_type must be training or evaluation")
    if arm not in ("random",):
        raise ValueError("this release defines only the random-initial-state arm")
    for name, value in (
        ("experiment", experiment), ("algorithm", algorithm),
        ("algorithm_version", algorithm_version), ("world_id", world_id),
        ("package_version", package_version),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(value)):
            raise ValueError(f"identity {name} contains unsafe characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(job_token)):
        raise ValueError("job_token contains unsafe characters")
    for name, value in (
        ("learning_seed", learning_seed), ("world_seed", world_seed),
        ("initialization_seed", initialization_seed), ("dynamic_obstacle_seed", dynamic_obstacle_seed),
        ("evaluation_seed", evaluation_seed),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name, value in (
        ("config_sha256", config_sha256), ("container_sha256", container_sha256),
        ("shared_layer_sha256", shared_layer_sha256), ("release_sha256", release_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    run_id = f"{algorithm}-{arm}-{phase_label}-s{learning_seed}-{config_sha256[:8]}-{phase_type}-{job_token}"
    identity = {
        "experiment": experiment,
        "run_id": run_id,
        "algorithm": algorithm,
        "algorithm_version": algorithm_version,
        "action_space": "discrete",
        "arm": arm,
        "learning_seed": int(learning_seed),
        "world_id": world_id,
        "world_seed": int(world_seed),
        "initialization_seed": int(initialization_seed),
        "dynamic_obstacle_seed": int(dynamic_obstacle_seed),
        "evaluation_seed": int(evaluation_seed),
        "config_sha256": config_sha256,
        "container_sha256": container_sha256,
        "shared_layer_sha256": shared_layer_sha256,
        "release_sha256": release_sha256,
        "package_version": package_version,
        "phase_type": phase_type,
        "phase_label": phase_label,
    }
    assert list(identity.keys()) == IDENTITY_FIELDS
    return identity


def write_identity(path: str, identity: Dict[str, object]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_identity(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as stream:
        identity = json.load(stream)
    if not isinstance(identity, dict):
        raise ValueError("run identity must be a JSON object")
    missing = sorted(set(IDENTITY_FIELDS) - set(identity))
    extra = sorted(set(identity) - set(IDENTITY_FIELDS))
    if missing or extra:
        raise ValueError(f"run identity schema mismatch: missing={missing}, extra={extra}")
    integer_fields = (
        "learning_seed", "world_seed", "initialization_seed",
        "dynamic_obstacle_seed", "evaluation_seed",
    )
    for field in integer_fields:
        if type(identity[field]) is not int or identity[field] < 0:
            raise ValueError(f"run identity {field} must be a non-negative integer")
    string_fields = tuple(field for field in IDENTITY_FIELDS if field not in integer_fields)
    for field in string_fields:
        value = identity[field]
        if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
            raise ValueError(f"run identity {field} must be a non-empty single-line string")
    for field in ("config_sha256", "container_sha256", "shared_layer_sha256", "release_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", identity[field]):
            raise ValueError(f"run identity {field} must be a lowercase SHA-256 digest")
    if identity["action_space"] != "discrete" or identity["arm"] != "random":
        raise ValueError("run identity must name the discrete random-start arm")
    if identity["phase_type"] not in ("training", "evaluation"):
        raise ValueError("run identity phase_type must be training or evaluation")
    if identity["phase_label"] not in ("calibration", "pilot", "controlled"):
        raise ValueError("run identity phase_label must be calibration, pilot or controlled")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity["run_id"]):
        raise ValueError("run identity run_id contains unsafe characters")
    for field in ("experiment", "algorithm", "algorithm_version", "world_id", "package_version"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity[field]):
            raise ValueError(f"run identity {field} contains unsafe characters")
    return identity


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] not in ("config-digest", "shared-layer-digest", "verify-shared-layer", "verify-release"):
        print(__doc__)
        return 2
    command, root = argv
    if command == "config-digest":
        print(config_digest(root))
    elif command == "shared-layer-digest":
        print(shared_layer_digest(root))
    elif command == "verify-shared-layer":
        problems = verify_manifest(root, SHARED_LAYER_MANIFEST)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    else:
        problems = verify_manifest(root, RELEASE_MANIFEST)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
