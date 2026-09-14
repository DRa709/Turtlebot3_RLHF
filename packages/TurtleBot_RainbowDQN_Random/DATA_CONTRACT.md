# Data contract — standalone Rainbow DQN

Executable schemas are in `turtlebot3_drl_nav/recorder.py`; independent checks
are in `turtlebot3_drl_nav/validator.py`. Analysis accepts only sealed runs with
`algorithm=RainbowDQN`.

## Run commit protocol

A training run contains identity/runtime manifests, five canonical CSVs,
checkpoints, simulator digests, logs, validation evidence and one final marker:

```text
run_identity.json
run_manifest.json
runtime_manifest.txt
transitions.csv
episodes.csv
updates.csv
evaluation.csv
checkpoints.csv
checkpoints/*.pt
checkpoints.sha256
progress.json
sim/runtime_world.sha256
sim/runtime_robot_model.sha256
logs/*.log
validation_report.json
RUN_FILES.sha256
AGENT_DONE
COMPLETE | FAILED | INTERRUPTED
```

Evaluation runs contain the applicable subset and no training updates or new
checkpoints. Writers are create-only, periodically flush and `fsync`, and never
append. After all writers stop, the wrapper validates the run, writes and
verifies `RUN_FILES.sha256`, and publishes `COMPLETE` last.

## Identity on every row

Every CSV row begins with experiment/run identity, algorithm/version, action
space, arm, four role-separated simulator/initialization seeds, learning seed,
world identity, configuration/SIF/shared-layer/release digests, package version,
phase type and phase label. The validator binds every row to this package and
the submitted SIF and requires `RainbowDQN`, `discrete`, and `random`.

## `transitions.csv`

One initial-state row precedes each episode, followed by one row per executed
transition. It records phase/policy/checkpoint identities; simulation and
sensor times; measured action hold and decision latency; discrete action and
executed velocity; total reward and six components; both masks and all outcome
flags; robot/obstacle poses; clearance and goal-relative state; and all 41
normalized observation values. The validator reconstructs actions, rewards,
precedence, masks, episode boundaries, fixed-duration control, sensor freshness
and analytic obstacle trajectories.

## `episodes.csv`

One row per training episode records its exact transition interval, return and
reward decomposition; outcome and collision class; length, clearance, path and
time efficiency; timing and tracking diagnostics; and the full requested,
realized and odometric initialization state. It also records start-generator
seed/rejections and obstacle-phase seed/signs/offsets. Summaries and seeded
starts are independently reconstructed from raw rows.

## `updates.csv`

One row is required for every optimizer step from environment transition 5,000
through the exact phase budget. Common finite fields are:

`gradient_step, env_step, loss, td_error_abs_mean, q_taken_mean, target_mean,
next_max_q_mean, grad_norm, learning_rate, target_synced,
batch_terminal_fraction, batch_size, replay_size`.

Rainbow-required finite fields are:

`distributional_loss_mean, beta_is, mean_importance_weight,
max_importance_weight, mean_priority, max_priority, noisy_sigma_mean,
noisy_sigma_min, noisy_sigma_max, n_step, mean_effective_n_step,
value_stream_expected_mean, centered_advantage_abs_mean,
chosen_distribution_entropy_mean, projection_clip_low_fraction,
projection_clip_high_fraction, projection_mass_error_max`.

The canonical schema retains an `epsilon` column for structural compatibility,
but Rainbow must leave it empty on every row. The validator checks the exact
beta schedule, batch/learning rate, replay bounds, normalized importance
weights, positive priorities, n-step bounds, sigma ordering, projection bounds
and probability-mass tolerance.

## `evaluation.csv`

Each row records one frozen-policy episode with checkpoint step/index/digest,
condition, scenario, policy mode, learner-unchanged flag, outcome/timing
summary, initialization block and E2 strata. Only `greedy` is accepted.

- E1: 20 reproducible new random-start episodes at every policy checkpoint.
- E2: all 100 frozen stratified scenarios at each tier-2 checkpoint.
- E3: 20 fixed-anchor episodes in each tier-2 job.

Evaluation transitions never increment the training counter or mutate replay.

## `checkpoints.csv`

Each row records environment/gradient steps, kind, confined path, SHA-256,
simulation/wall time and load-back result. Policy checkpoints occur at the
phase cadence. Full checkpoints occur at the full cadence and exact final
budget and preserve networks, optimizer, PER, n-step tail, counters and all RNG
states. Files are published atomically and recorded only after load validation.

## ARC phase protocol

| Phase | Seeds | Training transitions/seed | Policy cadence | Tier-2 checkpoints |
|---|---:|---:|---:|---|
| Calibration | 1 | 10,000 | 5,000 | 10,000 |
| Pilot | 2 | 10,000 | 5,000 | 10,000 |
| Controlled | 5 | 500,000 | 25,000 | 100k, 200k, 300k, 400k, 500k |

## Standalone tables and figures

`scripts/make_tables.py` and `scripts/make_figures.py` read only sealed Rainbow
runs and require the complete phase-specific seed/checkpoint matrix. They
produce Rainbow-only outcomes, learning efficiency, failure anatomy, per-seed
results, provenance, learner diagnostics, initialization fidelity, learning
curves, trajectories, spatial/stratified success, reward decomposition,
checkpointwise held-out performance, path/time efficiency and timing. Tables
are CSV, Markdown and LaTeX; figures are vector PDF and 300-dpi PNG. No
multi-algorithm table, figure or analysis entry point is included.
