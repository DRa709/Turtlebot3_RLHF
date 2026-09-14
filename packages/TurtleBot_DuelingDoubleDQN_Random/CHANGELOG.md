# Changelog

## 1.0.2 — 2026-09-04 — analysis-fixture and provenance correction

- Repairs the two stale analysis smoke tests without changing production
  campaign completeness. The fixture runs from a temporary package copy whose
  pilot block consistently declares seed 101, budget/checkpoint 40 and the
  matching evaluation-to-training manifest linkage.
- Extends the existing campaign-completeness test to reject both an unexpected
  checkpoint and a missing second seed; the declared test count remains 154.
- Removes the package-specific rendering smoke test from the shared-core list
  and corrects the comparison: 61 declared paths, 58 byte-identical, one
  Dueling recorder specialization and two stronger shared-test files.
- Bumps all package, configuration, container and runbook identities to 1.0.2.
- Does not change the Dueling learner, agent, optimizer, replay, checkpoint,
  random-initialization, environment, validator or production analysis code.

The independent v1.0.1 re-verification confirmed the clean Foxy build and live
Gazebo trajectory correction. Exact v1.0.2 SIF tests and ARC calibration/pilot
remain mandatory before controlled training.

## 1.0.1 — 2026-09-04 — audited ARC/shared-core correction

- Ports the corrected random-start core while retaining the Dueling Double DQN
  learner, agent, factory and algorithm tests. `recorder.py` is intentionally
  specialized to require the value-stream and centered-advantage diagnostics
  written by this learner; v1.0.2 records the corrected comparison counts.
- Replaces Foxy-incompatible `ros2 topic echo --once` polling with a bounded
  `rclpy` readiness subscriber that requires an advancing simulation clock.
- Makes both dynamic obstacles collision-active, kinematic and gravity-free;
  the frozen 0.05 m tracking limit is unchanged and now has static regression
  coverage.
- Parses recorded integer seeds directly without an IEEE-754 floating-point
  round trip and adds exact 63-bit parsing tests.
- Repairs the validator package fixture and routes acknowledgement-generated
  step-zero transitions through the canonical CSV writer.
- Isolates the pinned scientific Python stack in `/opt/tb3-python`, disables
  unrelated Pytest plugin auto-loading, and checks dependency import origins in
  `apptainer test`.
- Rejects whitespace in Foxy/Gazebo package, simulator and evaluation paths.
- Uses the current ARC `apptainer` module name first, with the legacy module
  name retained only as a batch-script fallback.
- Makes both analysis entry points executable Dueling-only gates: they reject
  every other algorithm identity and cannot emit combined tables or figures.

This is a source-corrected release candidate. Exact-SIF tests, a 10+ second
live Gazebo probe, one 10,000-transition calibration and the unchanged two-seed
pilot remain mandatory before controlled training.

## 1.0.0 — 2026-09-03 — standalone Dueling Double DQN ARC candidate

- Adds a 41→256→256 feature trunk with separate 256→1 value and 256→5
  advantage heads and mean-centered dueling aggregation.
- Uses online selection and target evaluation for the one-step Double-Q target.
- Adds independent numerical and structural tests for dueling aggregation,
  target construction, loss, optimizer step, masks, schedules and checkpoints.
- Adds value-stream and centered-advantage diagnostics to `updates.csv`.
- Ships one learner only, with independent configuration, ARC arrays, wrappers,
  manifests, validation, tables and figures.
- Standalone analysis rejects results from every other algorithm identity.
- Preserves the declared random-start reset, fixed-duration control,
  simulation-time obstacle, observation, reward and evaluation contracts.
