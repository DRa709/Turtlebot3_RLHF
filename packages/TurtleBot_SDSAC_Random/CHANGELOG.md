# Changelog

## 1.0.2 — source-lifecycle correction

- Prevents preflight, training, evaluation, analysis, and ROS imports from
  writing bytecode into the authenticated source tree.
- Sets `PYTHONDONTWRITEBYTECODE=1` globally in the container and explicitly
  in direct preflight and both ARC runtime wrappers; both Slurm arrays forward
  it before their first container-side phase query.
- Adds static coverage for every ARC entry path and an executable
  `verify -> preflight -> verify` regression.
- Records the raw and LF-normalized hashes of the first audit and the raw hash
  of the independent v1.0.1 audit.
- Advances the embedded shared execution layer to 3.0.3. The SD-SAC learner,
  initialization law, world geometry, obstacle dynamics, reward, observation,
  action, checkpoint, and evaluation contracts are unchanged.

## 1.0.1 — corrected ARC source candidate

- Replaces the ROS 2 Foxy-incompatible topic-readiness command with a bounded
  `rclpy` subscriber that checks every required topic and advancing `/clock`.
- Parses CSV integers without an IEEE-754 round trip and tests exact 63-bit
  initialization-generator seeds.
- Routes reset step-0 effects through the canonical mock recorder and repairs
  the validator tests so the nominal and tampering cases execute.
- Makes dynamic obstacles kinematic and gravity-free while retaining collision
  geometry, preserving the frozen analytic trajectory tolerance.
- Isolates the container Python stack, disables third-party Pytest auto-loading,
  suppresses test-created bytecode in the authenticated source tree, and checks
  critical import origins.
- Rejects whitespace-bearing package/runtime paths and strengthens cache,
  permission, provenance, and standalone-analysis tests.
- Keeps the SD-SAC objective and agent implementation unchanged except for
  recording integer-valued batch and replay diagnostics as integer CSV fields.

## 1.0.0 — release candidate

- First independent SD-SAC random-initial-state ARC package.
- Adds Zhou et al.'s entropy-change penalty, double-average Q learning and
  elementwise Q-clip to the controlled categorical SAC infrastructure.
- Adds stochastic and deterministic matched-start evaluation channels.
- Adds SAC-specific canonical diagnostics, validators, tests, tables and figures.
- Preserves the standalone random-start environment, mixed static/dynamic
  Gazebo world, transactional reset, fixed-duration control and ARC lifecycle.
