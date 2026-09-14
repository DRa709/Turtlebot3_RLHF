# Changelog

## 1.0.1 — 2026-09-04 — audited shared-layer correction

- Replaced the scientific environment core with the locally verified DQN v1.1.2
  random-start implementation without importing the DQN learner, agent,
  configuration or ARC entry points.
- Replaced Foxy-incompatible `ros2 topic echo --once` polling with the bounded
  `rclpy` readiness subscriber and advancing-clock check.
- Made both dynamic obstacles collision-active, kinematic, gravity-free bodies
  while preserving the existing planar controller and 0.05 m validation limit.
- Corrected validator integer parsing so recorded generator seeds never pass
  through IEEE-754 floating point; added exact 63-bit regression coverage.
- Isolated the pinned scientific stack in `/opt/tb3-python` and disabled
  unrelated Pytest plugin auto-loading in the mandatory test command.
- Repaired the manifest-bearing validator fixture and routed obstacle
  acknowledgements through the canonical loop-harness recorder, including one
  step-0 row per represented episode.
- Added whitespace-path rejection for Foxy/Gazebo and changed ARC module
  examples to the documented `apptainer` name with a legacy fallback in batch
  scripts.
- Restricted the runbook and analysis commands to standalone Double DQN result
  directories.
- The Double DQN learner, agent adapter, learner factory, optimizer settings and
  Double-target discriminator tests are unchanged from 1.0.0.

This is a source-corrected release candidate. Exact-SIF tests, a 10+ second live
Gazebo probe, the 10,000-transition calibration and the unchanged two-seed
pilot remain mandatory before controlled training.

## 1.0.0 — 2026-09-03 — standalone Double DQN random-start ARC candidate

- Implements the Double DQN target: online-network argmax followed by target-network evaluation, with the complete target detached from gradient computation.
- Adds an explicit discriminator test in which online and target networks rank different next actions; the test proves the implementation cannot silently reduce to the standard-DQN maximum target.
- Retains uniform replay, 5,000-transition random warm-up, linear ε schedule, one update per transition, Huber loss, gradient clipping, hard target synchronization and terminal-only bootstrap masking.
- Provides strict policy-only and full checkpoints with online/target networks, optimizer, replay ring, counters and Python/NumPy/PyTorch RNG states.
- Ships as a Double-DQN-only archive with its own learner, node, YAML, ARC arrays, wrappers, tests, identifiers and algorithm description.
- Uses shared layer 3.0.1: the random-start distribution, transactional reset, fixed-duration actions, simulation-time dynamic obstacles, canonical streams, validator and analysis pipeline.
- Makes shared preflight algorithm-aware so a standalone package validates its own single algorithm configuration instead of assuming a DQN filename.
- Uses direct ARC only; no Jupyter notebook or fixed-initialization comparison is included.
