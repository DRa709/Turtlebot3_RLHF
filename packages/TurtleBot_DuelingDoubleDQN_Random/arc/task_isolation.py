#!/usr/bin/env python3
"""Atomic transport leases for concurrent ARC tasks. Shared layer.

Two tasks on one node must never share a ROS domain or a Gazebo master port.
`--exclusive` already prevents sharing a node; this allocator is the second
barrier: it acquires a (ROS_DOMAIN_ID, GAZEBO_MASTER_URI port) pair through an
exclusive-create lease file keyed by hostname, checks the liveness of the
holder of every existing lease, and never derives an identity by modulo.

    eval "$(python3 arc/task_isolation.py acquire <lease_dir>)"   # exports the identity
    python3 arc/task_isolation.py release <lease_path>
"""

import json
import os
import shlex
import socket
import sys
import time
from typing import Optional, Tuple

DOMAIN_IDS = range(1, 101)          # ROS 2 domain IDs safe on every platform
GAZEBO_PORTS = range(20100, 20200)  # one Gazebo master port per lease
STALE_AFTER_S = 3 * 24 * 3600


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lease_path(lease_dir: str, hostname: str, domain: int) -> str:
    return os.path.join(lease_dir, f"{hostname}.domain{domain}.lease")


def acquire(lease_dir: str, hostname: Optional[str] = None) -> Tuple[int, int, str]:
    hostname = hostname or socket.gethostname()
    os.makedirs(lease_dir, exist_ok=True)
    for index, domain in enumerate(DOMAIN_IDS):
        path = _lease_path(lease_dir, hostname, domain)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as stream:
                    holder = json.load(stream)
                stale = (time.time() - float(holder.get("time", 0))) > STALE_AFTER_S
                dead = not _alive(int(holder.get("pid", -1)))
            except (ValueError, OSError):
                stale, dead = True, True
            if stale or dead:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            else:
                continue
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            # the holder is the calling wrapper shell (our parent), which outlives this helper
            json.dump({"pid": os.getppid(), "helper_pid": os.getpid(), "host": hostname, "time": time.time(),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}, stream)
        port = GAZEBO_PORTS[index]
        return domain, port, path
    raise RuntimeError("no free ROS domain lease on this host")


def release(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def main(argv) -> int:
    if len(argv) == 2 and argv[0] == "acquire":
        domain, port, path = acquire(argv[1])
        print(f"export ROS_DOMAIN_ID={domain}")
        print(f"export GAZEBO_MASTER_URI=http://127.0.0.1:{port}")
        print(f"export TRANSPORT_LEASE={shlex.quote(path)}")
        return 0
    if len(argv) == 2 and argv[0] == "release":
        release(argv[1])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
