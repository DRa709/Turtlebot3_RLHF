# Verification status — 2026-09-04

Package: `TurtleBot_RainbowDQN_Random` 1.0.1.

## Current verdict

**Corrected source release candidate — not frozen/verified.** The reported
source defects have been repaired, but this build host has no PyTorch, ROS 2
Foxy, Gazebo 11, Apptainer or Slurm. It therefore cannot execute the numerical
learner, ROS-adapter, exact-image or live-simulator gates locally.

## Locally established

- Package construction and source inspection are standalone: one algorithm,
  one learner, one config, Rainbow-only ARC paths, and no notebook or imported
  learner package.
- Host-safe tests cover archive hygiene; random-start geometry, seeded index
  purity and transactional reset state machine; two-mask semantics; fixed-time
  control; simulation-time obstacles; schemas, identities, lifecycle scripts;
  standalone analysis; and AST guards for all six Rainbow mechanisms.
- The 161-method suite includes numerical component discriminators for C51,
  NoisyNet, PER and n-step returns; an online-selection/target-evaluation
  discriminator; full diagnostic checks; deterministic mean-weight evaluation;
  and exact checkpoint-resume continuation.
- `scripts/run_tests.sh` rejects any skipped test, so a missing PyTorch/ROS
  dependency cannot be presented as an exact-image pass.

The host-safe `unittest` discovery reports 121 passed and five skip records.
Those skip records represent 40 PyTorch-dependent methods, not acceptable
passes. The static package verifier and deterministic two-million-proposal
initialization characterization pass. Final archive identities and limitations
are recorded in the external v1.0.1 correction report distributed beside the
ZIP.

## Mandatory ARC promotion gates

1. Build the package-specific SIF and preserve its SHA-256.
2. Run `scripts/run_tests.sh` inside that exact SIF with zero skips.
3. Complete and validate the one-seed calibration training and evaluation.
4. Complete and validate the unchanged two-seed pilot and evaluations.
5. Freeze exactly those ZIP/SIF bytes only after all gates pass.

The five controlled 500,000-transition jobs must not begin before promotion.
Commands and stop conditions are in `ARC_RUNBOOK.md`.
