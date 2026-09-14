# TurtleBot SD-SAC Random 1.0.2

Standalone, direct-ARC package for SD-SAC under the frozen
random initial-state distribution. It is independent of every other algorithm
package: it has its own source, configuration, container recipe, Slurm arrays,
workspace, checkpoints, results, validator, tables and figures. Jupyter is
neither required nor included.

## What this package runs

- TurtleBot3 Burger in ROS 2 Foxy and Gazebo 11.
- Mixed arena: four walls, two static boxes and two moving obstacles.
- Seeded random robot pose on every training episode, including episode 1.
- Fixed goal at `(2, 0)` and seeded simulation-time obstacle phases.
- 41-value observation: 36 nearest-rule LiDAR beams plus five scalars.
- Five frozen discrete velocity commands.
- SD-SAC with double-average Q, elementwise Q-clip, behavior-entropy replay and
  entropy-change regularization; fixed `alpha=0.2`, `beta=0.5`, `c=0.5`.
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
| `turtlebot3_drl_nav/sdsac.py` | SD-SAC actor, twin critics, replay, updates and checkpoints |
| `turtlebot3_drl_nav/sdsac_agent_node.py` | ROS agent and canonical learner streams |
| `config/phase1_sdsac.yaml` | SD-SAC-only frozen hyperparameters |
| `config/common_environment.yaml` | Environment, seeds, budgets and evaluation protocol |
| `worlds/phase1_mixed.world` | Static-plus-moving Gazebo arena |
| `arc/sdsac_array.sbatch` | ARC training array |
| `arc/sdsac_eval_array.sbatch` | ARC E2/E3 evaluation array |
| `scripts/validate_run.py` | Fail-closed run validator |
| `scripts/make_tables.py` | SD-SAC-only tables |
| `scripts/make_figures.py` | SD-SAC-only figures |
| `apptainer/tb3_phase1_foxy.def` | Reproducible ARC runtime |
| `AUDIT_CORRECTIONS.md` | v1.0.0/v1.0.1 audit findings and v1.0.2 dispositions |
| `PROVENANCE.md` | Parent identity, protected-source hashes and evidence boundary |

## Verification status

This source archive is a release candidate. It becomes frozen/verified only
after the exact built SIF passes all tests with zero skips, a live-Gazebo
calibration passes, and the two-seed ARC pilot passes without changing package
or image bytes. See `VERIFICATION_STATUS.md`.
