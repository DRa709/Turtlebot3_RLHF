#!/usr/bin/env python3
"""Resolve one exact validated policy checkpoint row from checkpoints.csv."""

import csv
import hashlib
import os
import sys


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: select_checkpoint.py TRAIN_RUN_DIR CHECKPOINT_STEP")
    run_dir = os.path.realpath(sys.argv[1])
    step = int(sys.argv[2])
    with open(os.path.join(run_dir, "checkpoints.csv"), newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row for row in rows
        if row["kind"] == "policy_only" and int(row["env_step"]) == step and row["validated"] == "1"
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one validated policy checkpoint at step {step}, found {len(matches)}")
    row = matches[0]
    path = row["path"] if os.path.isabs(row["path"]) else os.path.join(run_dir, row["path"])
    path = os.path.realpath(path)
    checkpoint_root = os.path.realpath(os.path.join(run_dir, "checkpoints")) + os.sep
    if not path.startswith(checkpoint_root) or not os.path.isfile(path):
        raise SystemExit("checkpoint path is missing or outside the training checkpoint directory")
    actual = digest(path)
    if actual != row["sha256"]:
        raise SystemExit("checkpoint digest disagrees with its exact checkpoints.csv row")
    print(path)
    print(actual)
    return 0


if __name__ == "__main__":
    sys.exit(main())
