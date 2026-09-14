# TurtleBot_DoubleDQN_Random 1.0.1

Standalone, Double-DQN-only, direct-ARC package for TurtleBot3 Burger navigation under the declared random robot-start distribution $\nu_R$. It uses ROS 2 Foxy, Gazebo 11 Classic and CPU PyTorch. No notebook is included or required.

The package is assessed against its own random-initialization, algorithmic and ARC execution contracts. It has no dependency on a fixed-initialization release.

## Principal guarantees

- The archive contains one learner only: `DoubleDQN`. It contains no DQN, Dueling Double DQN, Rainbow DQN, Discrete SAC, PPO or A2C learner implementation.
- The Double DQN target uses the online network for action selection and the target network for evaluation. A discriminator test forces the two networks to rank different actions and rejects the standard-DQN target.
- Episode 1 and every later training episode draw a robot pose from the same seeded, episode-indexed $\nu_R$ sampler.
- Start poses must satisfy the frozen arena, goal and complete dynamic-obstacle-corridor clearance rules.
- Reset, teleportation, simulation-time settling, physics pause, fresh-sensor capture, realized-state verification and obstacle acknowledgement finish before the first policy action.
- Physics remains paused during policy inference and optimization. Each command is applied for one measured simulation-time control interval.
- Dynamic obstacles follow seeded simulation-time trajectories that are checked online and independently reconstructed by the validator.
- Training stops at the exact phase budget, checkpoints at the frozen cadence, and evaluates every policy checkpoint on E1.
- `COMPLETE` is published only after validation and immutable artifact sealing succeed.

These are code-level guarantees. Exact ROS/Gazebo/Apptainer/Slurm behavior must still pass the compute-node container test, calibration and pilot gates in `ARC_RUNBOOK.md`.

## Package map

| Path | Purpose |
|---|---|
| `ALGORITHM.md` | Double DQN description, equation, frozen values and diagnostics |
| `AUDIT_CORRECTIONS.md` | Disposition of the independently reproduced v1.0.0 defects |
| `RANDOM_INIT_SPEC.md` | Exact $\nu_R$, seed roles, reset transaction and E1/E2/E3 |
| `DATA_CONTRACT.md` | Canonical streams, provenance, validation and paper products |
| `ARC_RUNBOOK.md` | Terminal-only ARC image, test, training and evaluation procedure |
| `config/common_environment.yaml` | Shared POMDP, randomization and evaluation authority |
| `config/phase1_doubledqn.yaml` | Double-DQN-only hyperparameters |
| `config/evaluation_scenarios_v1.csv` | Frozen 100-scenario E2 set |
| `turtlebot3_drl_nav/doubledqn.py` | Network, replay, Double target, update and checkpoints |
| `turtlebot3_drl_nav/doubledqn_agent_node.py` | Double DQN ROS adapter |
| `turtlebot3_drl_nav/episode_engine.py` | Transactional reset and fixed-duration control state machine |
| `turtlebot3_drl_nav/validator.py` | Fail-closed reconstruction and validation |
| `turtlebot3_drl_nav/analysis.py` | Paper tables and figures from sealed runs |
| `arc/` | Separate Double DQN training and evaluation Slurm entry points |
| `apptainer/tb3_phase1_foxy.def` | Standalone Double DQN runtime image recipe |

`SHARED_LAYER_FILES.txt` identifies the byte-identical scientific core inherited
from the locally verified DQN v1.1.2 random-start environment. Package-branded
preflight, container documentation, data-contract documentation,
verification-scope documentation and hygiene tests are intentionally excluded
and are authenticated by the release manifest.
This avoids importing DQN package identities into the standalone Double DQN
release. The Double DQN SIF and source tree can be built and retained
independently of every other algorithm package.

## Episode and control sequence

1. Hold both dynamic obstacles and await acknowledgement.
2. Await `/reset_world`; await `/set_entity_state` for the sampled robot pose.
3. Settle for 0.5 simulation seconds, command zero velocity, pause physics and await the pause.
4. Capture bounded-age scan, robot odometry and obstacle odometry while paused.
5. Verify simulator and odometry poses, support membership, clearance, obstacle reset positions and absence of reset contact.
6. Install and acknowledge the episode's seeded obstacle schedule; then publish $o_0$.
7. While paused, Double DQN selects an action or performs an update. The environment applies the command for one simulation-time interval, obtains boundary-fresh sensors, pauses, records $o_{t+1}$ and returns the transition.

## Direct ARC entry points

```bash
sbatch --export=ALL,PHASE_LABEL=calibration --array=0   arc/doubledqn_array.sbatch
sbatch --export=ALL,PHASE_LABEL=pilot       --array=0-1 arc/doubledqn_array.sbatch
sbatch --export=ALL,PHASE_LABEL=controlled  --array=0-4 arc/doubledqn_array.sbatch
```

Post-hoc E2/E3 evaluation uses `arc/doubledqn_eval_array.sbatch`. Controlled tier-2 checkpoints are 100k, 200k, 300k, 400k and 500k transitions.

## Paper data

Every validated training run records `transitions.csv`, `episodes.csv`,
`updates.csv`, `evaluation.csv` and `checkpoints.csv`. These include trajectories,
initialization requests and realizations, reward components, masks, timing,
sensor ages, obstacle errors, Double DQN diagnostics, checkpoint identities and
outcomes. The supplied analysis scripts generate a standalone Double DQN set of
tables T-R1–T-R9 and figures F-R1–F-R14 from Double DQN result directories only.

## Verification boundary

```bash
bash scripts/verify_package.sh
```

This login-node-safe gate checks manifests, inventory, line endings, shell syntax, Python compilation and frozen scenarios. Run `scripts/run_tests.sh` inside the exact ARC Apptainer image on a compute node; any skipped test is a failure. The archive remains a release candidate until exact-image tests and calibration pass.
