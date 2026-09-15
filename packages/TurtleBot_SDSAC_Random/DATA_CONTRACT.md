# Data contract — standalone SD-SAC

Executable schemas are in `turtlebot3_drl_nav/recorder.py`; independent checks
are in `turtlebot3_drl_nav/validator.py`. Analysis accepts only validated
`algorithm=SDSAC`, `arm=random` runs and refuses foreign algorithms.

## Run commit protocol

A training run contains identity/runtime manifests, five canonical CSVs,
checkpoints, simulator digests, logs, validation evidence and exactly one final
lifecycle marker. Writers are create-only and fsync periodically. `COMPLETE` is
published only after validation plus run-file hashing succeeds.

The five streams are `transitions.csv`, `episodes.csv`, `updates.csv`,
`evaluation.csv` and `checkpoints.csv`. Evaluation runs contain no training
updates or newly written checkpoints.

## Identity

Every CSV row carries run/experiment/algorithm/version, action space, arm,
learning seed, four role-specific world/initialization/obstacle/evaluation
seeds, world ID, package/config/container/shared/release digests, and phase.

## Transition and episode evidence

`transitions.csv` includes a step-0 state and every executed transition:
simulation/sensor time, fixed action hold, inference latency, command, reward
and all six components, both masks, outcome flags, robot/obstacle poses,
navigation state and all 41 normalized observations.

`episodes.csv` stores the exact transition interval, return/decomposition,
outcome class, path/safety/timing diagnostics, and the complete requested,
realized and odometric initialization block. Seeded starts and obstacle phases
are independently reconstructed.

## SD-SAC update evidence

One update row is required at transition 5,000 and after every transition
through the exact budget. Every row contains finite values for:

```text
gradient_step, env_step, target_synced,
batch_terminal_fraction, batch_size, replay_size,
actor_loss, actor_base_loss, entropy_penalty_loss, entropy_penalty_contribution,
critic1_loss, critic2_loss, critic_loss_mean,
td_error_abs_mean, q1_taken_mean, q2_taken_mean, q_gap_abs_mean,
soft_target_mean, next_soft_value_mean, average_q_policy_mean,
policy_entropy_mean, old_policy_entropy_mean, entropy_gap_abs_mean,
next_policy_entropy_mean,
max_action_probability_mean, min_action_probability_mean, alpha,
entropy_penalty_beta, q_clip_range, q1_clipped_mean, q2_clipped_mean,
q1_clip_activation_fraction, q2_clip_activation_fraction,
actor_grad_norm, critic1_grad_norm, critic2_grad_norm,
actor_learning_rate, critic_learning_rate
```

The validator checks the 1:1 update schedule, hard target cadence, uniform
replay size, batch size, fixed alpha/beta/clip range and learning rates, actor
loss decomposition, entropy bounds, Q-clip activation fractions, probability
bounds and non-negative gradient norms. Mathematical tests independently
recompute double-average targets, the entropy penalty and elementwise Q-clip.

## Evaluation evidence

Every checkpoint is evaluated in two separately labeled channels:
`stochastic` and `deterministic`. Their rows are never pooled by the supplied
analysis. The stochastic channel uses episode-and-decision-derived seeds.

- E1: 20 matched new random starts per channel at every training checkpoint.
- E2: all 100 frozen held-out scenarios per channel in each tier-2 job.
- E3: 20 fixed-anchor episodes per channel in each tier-2 job.

For controlled training this is 800 in-run E1 episodes per seed. Across the
five retained tier-2 checkpoints it is another 1,200 episodes per seed, for
2,000 evaluation episodes per seed and 10,000 across all five seeds.

## Checkpoints

Policy checkpoints contain the actor and complete identity/configuration
metadata. Full checkpoints additionally contain both critics, both target
critics, all three optimizers, uniform replay, counters and Python/NumPy/Torch
RNG state. Files are atomic and load-validated before recording.

## Standalone outputs

`scripts/make_tables.py` emits CSV, Markdown and LaTeX. `scripts/make_figures.py`
emits vector PDF and 300-dpi PNG. Both require a complete phase-specific
SD-SAC matrix, reject other algorithms, and preserve policy mode as a
separate grouping variable.
