# TurtleBot_RainbowDQN_Random 1.0.1

This is a standalone Rainbow-DQN-only package for direct ARC execution of
TurtleBot3 Burger navigation under a reproducible random robot-start
distribution. It uses ROS 2 Foxy, Gazebo 11 Classic and CPU PyTorch. It has no
Jupyter notebook and no source, runtime, checkpoint or analysis dependency on
DQN, Double DQN, Dueling Double DQN or any other algorithm package.

## What this package implements

Rainbow combines six mechanisms in one off-policy learner:

1. Double-Q action selection and target evaluation;
2. a mean-centered dueling value/advantage representation;
3. C51 categorical distributional learning;
4. proportional prioritized experience replay with importance weights;
5. episode-safe three-step returns; and
6. factorized Gaussian NoisyNet exploration.

The frozen learner values and equations are in `ALGORITHM.md`. The random-start
law and complete reset transaction are in `RANDOM_INIT_SPEC.md`.

## Isolation guarantees

- `ALGORITHM` is exactly `RainbowDQN`.
- The only learner is `turtlebot3_drl_nav/rainbowdqn.py`.
- The only learner adapter is
  `turtlebot3_drl_nav/rainbowdqn_agent_node.py`.
- Training and evaluation use package-specific Slurm arrays, workspaces,
  checkpoints and result paths below `rainbowdqn/`.
- The table and figure entry points reject any result whose algorithm identity
  is not `RainbowDQN`; they never create cross-algorithm products.

## Environment and random starts

Episode 1 and every later training episode sample a seeded random robot
position uniformly over the declared clearance-admissible region and an
independent uniform yaw. Gazebo reset, relocation, settling, fresh-sensor
capture and realized-state validation all finish before the first action. The
goal and static map remain fixed. Two obstacles move on seeded simulation-time
schedules. Physics is paused during policy inference and optimization, so each
action receives the declared simulation-time duration independent of network
cost.

## Declared controlled run

- five separate learning seeds: 101, 202, 303, 404 and 505;
- exactly 500,000 training transitions per seed;
- policy checkpoints every 25,000 transitions;
- full-resume checkpoints every 100,000 transitions and at the final budget;
- 20 new random-start E1 episodes at every policy checkpoint;
- 100 frozen E2 scenarios plus 20 E3 anchors at 100k, 200k, 300k, 400k and
  500k; and
- NoisyNet training exploration; noise-free greedy evaluation.

## Important files

| Path | Purpose |
|---|---|
| `ALGORITHM.md` | Rainbow equations, architecture and diagnostics |
| `AUDIT_CORRECTIONS.md` | Independent findings and v1.0.1 dispositions |
| `RANDOM_INIT_SPEC.md` | Random-start law, seeds and reset transaction |
| `DATA_CONTRACT.md` | Canonical streams and standalone products |
| `ARC_RUNBOOK.md` | Direct terminal-based ARC instructions and stop gates |
| `config/common_environment.yaml` | Environment and evaluation protocol |
| `config/phase1_rainbowdqn.yaml` | Rainbow-only hyperparameters |
| `arc/rainbowdqn_array.sbatch` | Training array |
| `arc/rainbowdqn_eval_array.sbatch` | Post-hoc evaluation array |
| `scripts/verify_package.sh` | Host-safe static package gate |
| `scripts/run_tests.sh` | Mandatory zero-skip exact-image test gate |
| `scripts/make_tables.py` | Rainbow-only CSV/Markdown/LaTeX tables |
| `scripts/make_figures.py` | Rainbow-only PDF/PNG figures |

## Verification state

This archive is a release candidate, not a frozen package. It becomes verified
only after the exact SIF executes every test with zero skips, followed by one
live Gazebo calibration and the unchanged two-seed ARC pilot. See
`VERIFICATION_STATUS.md` and `ARC_RUNBOOK.md`.
