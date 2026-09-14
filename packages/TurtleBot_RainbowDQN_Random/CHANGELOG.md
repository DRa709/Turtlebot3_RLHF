# Changelog

## 1.0.1 — 2026-09-04 — audited ARC/runtime correction

- Retains the Rainbow learner and ROS agent logic from v1.0.0; only the
  package/configuration version identity changes in algorithm-specific runtime
  files.
- Replaces Foxy-incompatible `ros2 topic echo --once` polling with a bounded
  `rclpy` subscriber that requires messages from every required topic and an
  advancing simulation clock.
- Makes both moving-obstacle links collision-active, kinematic and gravity-free
  without relaxing the 0.05 m analytic tracking tolerance.
- Parses recorded integer fields directly as decimal integers and adds exact
  63-bit seed boundary tests.
- Routes obstacle acknowledgements through the canonical harness dispatcher so
  every represented episode writes exactly one step-zero transition.
- Repairs the validator package fixture, reward-message assertion and the
  Rainbow Double-Q discriminator's `NoisyLinear.bias_mu` access.
- Isolates the pinned scientific Python stack in `/opt/tb3-python`, disables
  unrelated Pytest plugin auto-loading and validates import origins in
  `apptainer test`.
- Rejects whitespace in Foxy/Gazebo package and run paths and uses ARC's current
  `apptainer` module name first.
- Makes the analysis rendering fixture satisfy a temporary, self-consistent
  one-seed protocol while leaving the production calibration, pilot and
  controlled campaign contracts unchanged.

This is a corrected source release candidate. Exact-SIF tests, live Gazebo
verification, calibration and the two-seed pilot remain mandatory.

## 1.0.0 — 2026-09-03

- First standalone Rainbow DQN random-initialization ARC release candidate.
- Implements Double selection/evaluation, mean-centered dueling categorical
  logits, C51 projection, proportional PER, episode-safe three-step returns and
  factorized Gaussian NoisyNet exploration in one independent learner.
- Adds logarithmic-time sum/min/max replay trees, full n-step/PER/RNG resume
  state, atomic policy/full checkpoints and noise-free isolated evaluation.
- Adds reproducible random robot starts for episode 1 and every later training
  episode, transactional Gazebo realization and simulation-time moving
  obstacles inherited through the sealed standalone environment layer.
- Adds bidirectional Rainbow update applicability, algorithm-specific
  diagnostics, fail-closed validation and standalone tables/figures.
- Adds separate training/evaluation Slurm arrays and package-specific SIF,
  workspace, results and checkpoint paths for direct ARC use.
