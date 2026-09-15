# TurtleBot3 SD-SAC navigation 1.0.2

SD-SAC navigation for TurtleBot3 Burger using ROS 2 Foxy, Gazebo 11, and CPU PyTorch.
This directory contains the learner, environment configuration, simulation assets,
ARC job scripts, and analysis tools for the SDSAC experiment.

## Algorithm and environment

A categorical SAC adaptation with double-average Q targets, an entropy-change penalty, and Q clipping; `alpha=0.2`, `beta=0.5`, and `c=0.5`.

- One learner: `SDSAC`.
- A 41-value observation combines 36 LiDAR ranges with goal and previous-command features.
- The policy selects one of five discrete velocity commands.
- Robot starts are sampled from the configured random-start distribution.
- The arena contains static and moving obstacles, with a fixed goal at `(2, 0)`.
- The controlled training budget is 500,000 transitions for each of seeds 101, 202, 303, 404, and 505.
- Evaluation records goal reached, safety stop, physical contact, and timeout separately.

See [ALGORITHM.md](ALGORITHM.md) for the update equations and parameters,
[RANDOM_INIT_SPEC.md](RANDOM_INIT_SPEC.md) for the reset and evaluation protocol,
and [DATA_CONTRACT.md](DATA_CONTRACT.md) for recorded fields.

## Run on ARC

Follow [ARC_RUNBOOK.md](ARC_RUNBOOK.md) to build the runtime, run the tests,
complete calibration and the two-seed pilot, and submit training and evaluation.
## Package files

| Path | Purpose |
| --- | --- |
| `turtlebot3_drl_nav/sdsac.py` | Learner, updates, and checkpoints |
| `turtlebot3_drl_nav/sdsac_agent_node.py` | ROS learner interface |
| `config/phase1_sdsac.yaml` | Algorithm parameters |
| `config/common_environment.yaml` | Environment, seeds, and evaluation schedule |
| `worlds/phase1_mixed.world` | Gazebo arena |
| `arc/` | Training and evaluation jobs |
| `scripts/verify_package.sh` | Inventory, checksum, syntax, and scenario checks |
| `scripts/make_tables.py` | Tables from completed runs |
| `scripts/make_figures.py` | Figures from completed runs |
| `apptainer/tb3_phase1_foxy.def` | Runtime container recipe |

## Results and validation

Completed runs provide transition, episode, update, evaluation, and checkpoint
records. Analysis uses those records to build the tables and figures specified
in the data contract.

Run `bash scripts/verify_package.sh` to check the package files and
`bash scripts/run_tests.sh` inside the runtime to execute the test suite.
Simulation timing, container execution, and cluster scheduling are checked through
the calibration and pilot steps in the runbook.
