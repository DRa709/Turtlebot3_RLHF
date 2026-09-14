# TurtleBot Discrete SAC Random 1.0.2

Standalone, direct-ARC package for categorical Discrete SAC under the frozen
random initial-state distribution. It is independent of DQN, Double DQN,
Dueling Double DQN, Rainbow DQN, SD-SAC and PPO: it has its own source,
configuration, container recipe, Slurm arrays, workspace, checkpoints,
results, validator, tables and figures. Jupyter is neither required nor
included.

## What this package runs

- TurtleBot3 Burger in ROS 2 Foxy and Gazebo 11.
- Mixed arena: four walls, two static boxes and two moving obstacles.
- Seeded random robot pose on every training episode, including episode 1.
- Fixed goal at `(2, 0)` and seeded simulation-time obstacle phases.
- 41-value observation: 36 nearest-rule LiDAR beams plus five scalars.
- Five frozen discrete velocity commands.
- Vanilla categorical Discrete SAC with a fixed `alpha=0.2`.
- Exact 500,000-transition controlled budget for seeds 101, 202, 303, 404, 505.
- Separate stochastic and deterministic evaluation on E1, E2 and E3.

## Start here

1. Read `ALGORITHM.md` and `RANDOM_INIT_SPEC.md`.
2. Build the SIF and execute the zero-skip tests in `ARC_RUNBOOK.md`.
3. Run the 10,000-transition calibration and its evaluation.
4. Run the two-seed pilot and its evaluations.
5. Only after all gates pass unchanged, submit controlled training.

## Important paths

| Path | Purpose |
|---|---|
| `turtlebot3_drl_nav/discretesac.py` | Actor, twin critics, replay, updates and checkpoints |
| `turtlebot3_drl_nav/discretesac_agent_node.py` | ROS agent and canonical learner streams |
| `config/phase1_discretesac.yaml` | SAC-only frozen hyperparameters |
| `config/common_environment.yaml` | Environment, seeds, budgets and evaluation protocol |
| `worlds/phase1_mixed.world` | Static-plus-moving Gazebo arena |
| `arc/discretesac_array.sbatch` | ARC training array |
| `arc/discretesac_eval_array.sbatch` | ARC E2/E3 evaluation array |
| `scripts/validate_run.py` | Fail-closed run validator |
| `scripts/make_tables.py` | Discrete-SAC-only tables |
| `scripts/make_figures.py` | Discrete-SAC-only figures |
| `apptainer/tb3_phase1_foxy.def` | Reproducible ARC runtime |
| `AUDIT_CORRECTIONS.md` | Finding-by-finding v1.0.0 audit response |
| `LOCAL_VERIFICATION_REPORT_v1.0.1.md` | Independent 163-test and live-Gazebo verification report |
| `PROVENANCE.md` | Immutable artifact and evidence lineage |

## Verification status

This source archive is a locally verified ARC source/calibration candidate.
The independently audited v1.0.1 implementation passed 163/163 tests and the
live Foxy/Gazebo probes recorded in `LOCAL_VERIFICATION_REPORT_v1.0.1.md`.
v1.0.2 is a documentation/provenance patch with unchanged learner and runtime
implementation. It becomes frozen/verified only after this exact v1.0.2 ZIP
and exact ARC-built SIF pass the gates in `VERIFICATION_STATUS.md`, including
calibration and the unchanged two-seed pilot.
