# Changelog

## 1.0.2 — local-audit closure and provenance release

- Incorporates the independent v1.0.1 report recording 163/163 passing tests,
  a clean Foxy build, and successful readiness, relocation, analytic-obstacle
  and physical-contact probes.
- Distinguishes two differently hashed v1.0.0 audit artifacts in an explicit
  provenance ledger instead of claiming they are identical.
- Adds a fail-closed test that authenticates protected implementation and
  bundled-evidence hashes against that ledger.
- Advances the package identity so these documentation/evidence bytes cannot
  be confused with v1.0.1. No learner, environment, simulator, validator, ARC
  wrapper, analysis implementation or scientific setting changes.

## 1.0.1 — audit-corrected release candidate

- Preserves the vanilla Discrete SAC update and ROS agent behavior from 1.0.0;
  only the `batch_size` and `replay_size` diagnostic types change from
  integral float to integer.
- Parses recorded integer identities directly as decimal integers, preserving
  all 63-bit initialization and obstacle generator seeds.
- Makes both commanded dynamic-obstacle links kinematic and gravity-free while
  retaining collision geometry and the analytic trajectory tolerance.
- Replaces the Foxy-incompatible topic CLI readiness probe with a bounded
  Python 3.8 ROS subscriber that requires every topic and an advancing clock.
- Isolates the pinned scientific Python stack in `/opt/tb3-python`, disables
  external pytest plugin auto-loading, and expands the image self-test.
- Repairs the validator harness routing, Python 3.8 source test, reward-message
  assertion, standalone analysis completeness fixture, and related gates.
- Rejects Foxy launch paths containing whitespace, pre-existing Python caches,
  missing executable modes, and unintended alternate learner implementations.

## 1.0.0 — release candidate

- First independent Discrete SAC random-initial-state ARC package.
- Adds vanilla categorical SAC with twin critics, exact five-action
  expectations, fixed alpha, uniform replay and one-step targets.
- Adds stochastic and deterministic matched-start evaluation channels.
- Adds SAC-specific canonical diagnostics, validators, tests, tables and figures.
- Preserves the standalone random-start environment, mixed static/dynamic
  Gazebo world, transactional reset, fixed-duration control and ARC lifecycle.
