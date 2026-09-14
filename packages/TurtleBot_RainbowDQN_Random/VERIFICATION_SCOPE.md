# Standalone verification scope — Rainbow DQN random initialization

This package is judged independently against the roadmap and its declared
random-start contract. It neither imports another learner nor accepts another
algorithm's run as evidence. No fixed-versus-random implementation or
performance comparison is part of this package verification.

## Mandatory gates

1. **Inventory and isolation:** safe archive; normalized paths, permissions and
   line endings; no notebook, cache or symlink; exactly one learner/config/ARC
   identity; release, configuration, shared-layer and SIF digests verified.
2. **Double plus Dueling:** online expected-Q argmax selects the next action;
   the target network supplies that action's distribution; categorical
   value/advantage logits use mean-centering over five actions.
3. **C51:** 51 atoms on `[-200,200]`; no-gradient target; terminal-only
   bootstrap mask; clamped projection; exact-atom handling; nonnegative,
   finite, mass-conserving targets; categorical cross-entropy.
4. **Three-step returns:** discount `0.99`; realized length retained; both
   termination and truncation stop accumulation; truncation bootstraps from the
   pre-reset successor; no reward crosses a reset.
5. **PER:** capacity 100,000; proportional exponent 0.6; globally normalized
   importance weights; beta 0.4 to 1.0 over 500,000 transitions; priorities
   from unweighted categorical loss plus `1e-6`; complete checkpoint state.
6. **NoisyNet and update schedule:** factorized noise and exact initialization;
   uniform-random 5,000-transition warm-up; fresh documented noise; no epsilon;
   one update per transition thereafter; Adam `1e-4`; batch 64; gradient clip
   10; hard target sync every 1,000 gradient steps; exact phase budget.
7. **Random initialization:** episode 1 included; role-separated seeds;
   declared uniform support; bounded rejection without fallback; acknowledged
   reset/relocation/settling; realized pose, odometry, clearance, freshness and
   contact validation before any action.
8. **Controlled dynamics:** simulation-time obstacle schedules; physics paused
   during learner work; measured fixed action holds; boundary-fresh sensors;
   analytic obstacle tracking and fatal tolerance enforcement.
9. **Recording and evaluation:** five canonical streams; Rainbow applicability
   enforced both ways; raw-to-summary reconstruction; atomic full/policy
   checkpoints; noise-free isolated evaluation; complete E1/E2/E3 schedule.
10. **ARC lifecycle and outputs:** Foxy/Gazebo/Apptainer consistency; exclusive
    arrays; unique workspaces and leased ROS/Gazebo identities; forwarded Slurm
    provenance; signal-safe interruption; create-only runs; validation and
    immutable inventory before `COMPLETE`; Rainbow-only tables and figures.

## Promotion levels

- Local/static success establishes a release candidate only.
- Exact-SIF success requires every test to execute with zero skips.
- Calibration requires one sealed 10,000-transition training/evaluation pair.
- Pilot requires both preregistered 10,000-transition seeds and evaluations to
  pass with unchanged ZIP and SIF bytes.
- Only then is the package frozen for five 500,000-transition controlled runs.

Every gate is fail-closed. A failed or interrupted output directory is retained
as evidence and never reused.
