# Data contract — canonical run streams (shared layer 3.0.2)

The executable schema is `turtlebot3_drl_nav/recorder.py`; independent checks
are in `turtlebot3_drl_nav/validator.py`. Analysis reads only runs that have a
passing `validation_report.json`, a valid exact `RUN_FILES.sha256`, and a final
`COMPLETE` marker.

## Run directory and commit protocol

```text
<run_dir>/
  run_identity.json
  run_manifest.json
  runtime_manifest.txt
  transitions.csv
  episodes.csv                 training runs only
  updates.csv                  training runs only
  evaluation.csv
  checkpoints.csv              training runs only
  checkpoints/*.pt             training runs only
  checkpoints.sha256           training runs only
  progress.json                training runs
  sim/runtime_world.sha256
  sim/runtime_robot_model.sha256
  logs/*.log
  validation_report.json
  RUN_FILES.sha256
  AGENT_DONE
  COMPLETE | FAILED | INTERRUPTED
```

CSV writers are create-only, write a header before data, serialize booleans as
0/1 and floats with round-trip precision, and periodically flush and `fsync`.
Blank means not applicable.

After every writer and simulator process stops, the wrapper runs the validator,
writes `RUN_FILES.sha256` over the exact output inventory, verifies that
manifest, and only then atomically publishes `COMPLETE`. Symbolic links,
undeclared files, missing files or changed bytes invalidate the run. Status
markers and `RUN_FILES.sha256` itself are excluded from the immutable payload so
the commit marker can be written last.

## Identity prefix on every CSV row

`experiment, run_id, algorithm, algorithm_version, action_space, arm,
learning_seed, world_id, world_seed, initialization_seed,
dynamic_obstacle_seed, evaluation_seed, config_sha256, container_sha256,
shared_layer_sha256, release_sha256, package_version, phase_type, phase_label`

- `config_sha256`: length-prefixed digest of every file in `config/` and
  `worlds/`.
- `shared_layer_sha256`: digest derived from the exact shared-file manifest.
- `release_sha256`: digest of `RELEASE_MANIFEST.sha256`, whose declared paths
  must exactly equal the package inventory.
- `container_sha256`: SHA-256 of the submitted SIF bytes, recomputed by the
  Slurm script rather than trusted from a sidecar.
- `phase_type`: `training` or `evaluation`; `phase_label`: `calibration`,
  `pilot` or `controlled`.

## `transitions.csv`

There is one step-0 state row per episode and then one row per transition.
Training `env_step` values are exactly 1…$B_{env}$; evaluation rows never consume
that budget.

Main fields:

- context: `phase, policy_mode, episode_key, training_episode,
  step_in_episode, env_step, checkpoint_step, condition, scenario_id,
  evaluation_episode`;
- state time and control: `sim_time, state_sim_time, hold_sim_s, hold_odom_s,
  decision_gap_sim_s, decision_latency_wall_s`;
- freshness/dynamics: `scan_age_s, odom_age_s, obs1_age_s, obs2_age_s,
  obstacle_position_error_max`;
- action: `action_index, action_name, linear_cmd, angular_cmd`;
- reward: `reward_total` and all six terms `r_distance, r_step, r_collision,
  r_goal, r_angular, r_near`;
- masks/events: `terminated, episode_end, truncated, collision,
  static_collision, dynamic_collision, safety, goal`;
- physical state: `min_lidar, distance, heading_error, x, y, yaw, obs1_x,
  obs1_y, obs2_x, obs2_y`;
- complete policy observation: `obs_0`…`obs_40`.

The validator reconstructs rewards from adjacent states and the executed action,
checks both masks and event precedence, enforces the action map, links each
`state_sim_time` to the preceding state, and bounds holds, policy-time simulation
drift, sensor ages and obstacle-trajectory error.

Training rows use `policy_mode=epsilon_greedy`; every DQN evaluation row uses
`policy_mode=greedy`. This prevents exploratory training behavior from being
mislabelled as deployment behavior in later plots.

## Shared outcome and initialization blocks

Every `episodes.csv` row and every `evaluation.csv` row includes:

- outcome: length, return, six reward sums, outcome label, event flags,
  truncation, near-band step count, minimum clearance, path length,
  straight-line distance, path efficiency, time to goal, start/end simulation
  time, wall duration, real-time factor, hold statistics, policy latency,
  maximum simulation-time decision gap, maximum sensor age and maximum obstacle
  tracking error;
- initialization: kind and generator seed; requested, simulator-realized and
  odometry poses; position/yaw/odometry errors; scan clearance; support and
  tolerance flags; goal; obstacle phase seed/signs/offsets/reset poses;
  rejection count; reset status; settle duration; reset contact; initial
  simulation time.

Every outcome and diagnostic value is independently recomputed from the raw
transition episode. A changed success label, return, path efficiency or timing
summary therefore fails validation.

## `episodes.csv`

One row per training episode: `episode_key, training_episode, start_env_step,
end_env_step` plus the shared blocks. Episode lengths sum to the exact budget,
episode numbers are contiguous, and starts/phases are recomputed from their
role-specific seeds.

## `updates.csv`

One row per DQN gradient step from the warm-up threshold through the budget:

`gradient_step, env_step, loss, td_error_abs_mean, q_taken_mean, target_mean,
next_max_q_mean, grad_norm, learning_rate, epsilon, target_synced,
batch_terminal_fraction, batch_size, replay_size`.

The shared schema also reserves explicitly empty fields for mechanisms used by
later separate packages. `UPDATE_APPLICABILITY` enforces both directions:
DQN-required fields must be finite and non-DQN fields must be blank.

## `evaluation.csv`

One row per evaluation episode:

`checkpoint_step, checkpoint_index, checkpoint_sha256, condition, scenario_id,
evaluation_episode, policy_mode, episode_key, learner_state_unchanged` plus the
shared blocks and the E2 labels `distance_bin, clearance_bin, heading_bin,
difficulty`.

DQN uses only `policy_mode=greedy`. E1 occurs in the training job at every
checkpoint. Separate post-hoc runs cover all 100 E2 scenarios and 20 E3 anchor
episodes at each preregistered tier-2 checkpoint. The evaluation run manifest
binds the exact selected checkpoint path, step and SHA-256.

## `checkpoints.csv`

`env_step, gradient_step, kind, path, sha256, sim_time, wall_time, validated`.
Policy checkpoints occur at every phase checkpoint; full checkpoints occur at
the full cadence and final budget. Paths must be unique and confined to the run's
`checkpoints/` directory. Both kinds are loaded back and their policy parameters
must match the live network before `validated=1` is recorded.

## Frozen evaluation cadence

The phase block in `config/common_environment.yaml` is the only authority.

| Phase | Training budget | E1 cadence | Tier-2 E2/E3 checkpoints |
|---|---:|---:|---|
| calibration | 10,000 | 5,000 | 10,000 |
| pilot | 10,000 | 5,000 | 10,000 |
| controlled | 500,000 | 25,000 | 100k, 200k, 300k, 400k, 500k |

Each checkpoint has 20 E1 episodes. Each tier-2 run has 100 E2 and 20 E3
episodes. The training budget counts training transitions only.

## Paper products

`scripts/make_tables.py` emits T-R1–T-R9 as CSV, Markdown and LaTeX booktabs.
`scripts/make_figures.py` emits F-R1–F-R14 as vector PDF and 300-dpi PNG. These
cover final held-out performance, learning efficiency, generalization, pairwise
improvement, failure anatomy, per-seed results, provenance, learner diagnostics,
initialization fidelity, learning curves, training windows, spatial performance,
trajectories, reward composition, outcome composition, checkpoint-wise held-out
performance, path/time efficiency and timing. For this standalone package, the
final tables select exactly one largest-checkpoint evaluation per DQN seed;
duplicate finals are an error. F-R12 alone uses all preregistered tier-2
checkpoints. Supplying `--expected-algorithms DQN` activates the DQN completeness
stop gate: missing or duplicate training seeds/tier-2 checkpoints, mixed common
seeds, mixed shared layers or images, and evaluation runs not linked to their
training run are fatal. A passing `campaign_completeness.json` is written with
the table data.
