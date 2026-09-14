"""Parameter forwarding of the training/evaluation launch, as a pure function
so a unit test can assert that every launch argument reaches its consumer.
Shared layer."""

import os
from typing import Dict, List, Tuple

ALGORITHM_FILES: Dict[str, Tuple[str, str]] = {
    "DQN": ("phase1_dqn.yaml", "dqn_agent"),
    "DoubleDQN": ("phase1_doubledqn.yaml", "doubledqn_agent"),
    "DuelingDoubleDQN": ("phase1_duelingdoubledqn.yaml", "duelingdoubledqn_agent"),
    "RainbowDQN": ("phase1_rainbowdqn.yaml", "rainbowdqn_agent"),
    "DiscreteSAC": ("phase1_discretesac.yaml", "discretesac_agent"),
}


def package_algorithm(package_root: str) -> Tuple[str, str, str]:
    """(algorithm, algorithm config file name, agent executable) from the
    package's ALGORITHM marker file — the one algorithm-specific fact the shared
    launch file needs."""
    with open(os.path.join(package_root, "ALGORITHM"), encoding="utf-8") as stream:
        algorithm = stream.read().strip()
    if algorithm not in ALGORITHM_FILES:
        raise ValueError(f"unknown algorithm marker {algorithm!r}")
    config_name, executable = ALGORITHM_FILES[algorithm]
    return algorithm, config_name, executable


LAUNCH_ARGUMENTS: List[str] = [
    "package_root", "run_dir", "mode", "learning_seed", "world_seed", "initialization_seed",
    "dynamic_obstacle_seed", "evaluation_seed", "phase_label", "checkpoint_path",
]
REQUIRED_ARGUMENTS = ["package_root", "run_dir", "learning_seed", "world_seed", "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed", "phase_label"]
PHASE_LABELS = ("calibration", "pilot", "controlled")


def _int(value: str, name: str) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        raise ValueError(f"launch argument {name} must be an integer, got {value!r}")


def build_node_parameters(args: Dict[str, str]) -> Dict[str, Dict[str, object]]:
    """Return {"environment": {...}, "agent": {...}} parameter overrides.

    Every entry of LAUNCH_ARGUMENTS is consumed here; an unknown key is an error
    so a typo cannot be ignored silently."""
    unknown = set(args) - set(LAUNCH_ARGUMENTS)
    if unknown:
        raise ValueError(f"unknown launch arguments {sorted(unknown)}")
    missing = [name for name in REQUIRED_ARGUMENTS if not str(args.get(name, "")).strip()]
    if missing:
        raise ValueError(f"missing launch arguments {missing}")
    mode = str(args.get("mode", "train")).strip().lower()
    if mode not in ("train", "eval"):
        raise ValueError("mode must be train or eval")
    seeds = {name: _int(args[name], name) for name in ("world_seed", "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed")}
    for name, value in seeds.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    environment = {"package_root": str(args["package_root"]), "run_dir": str(args["run_dir"]), "use_sim_time": True}
    environment.update(seeds)
    agent = {
        "package_root": str(args["package_root"]), "run_dir": str(args["run_dir"]), "use_sim_time": True,
        "mode": mode, "learning_seed": _int(args["learning_seed"], "learning_seed"),
        "checkpoint_path": str(args.get("checkpoint_path", "")),
    }
    phase_label = str(args["phase_label"]).strip()
    if phase_label not in PHASE_LABELS:
        raise ValueError(f"phase_label must be one of {PHASE_LABELS}")
    agent["phase_label"] = phase_label
    if mode == "eval" and not agent["checkpoint_path"]:
        raise ValueError("eval mode requires checkpoint_path")
    if mode == "train" and agent["checkpoint_path"]:
        raise ValueError("train mode does not accept checkpoint_path (restarts are from scratch by policy)")
    return {"environment": environment, "agent": agent}
