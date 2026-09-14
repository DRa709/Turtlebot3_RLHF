# Verification status — 1.0.2 corrected ARC source candidate

## Completed before packaging

- Independent SD-SAC-only tree and ARC namespaces.
- Static package integrity, Python compilation, shell/Slurm syntax and frozen
  scenario regeneration gates pass (`scripts/verify_package.sh`).
- The v1.0.0 functional findings and the v1.0.1 source-cache lifecycle finding
  are addressed in source and covered by regression tests.
- Independent v1.0.1 evidence recorded 168/168 tests, a clean Foxy build, and a
  live 10.25-second Gazebo trajectory/contact probe. Because v1.0.2 changes
  authenticated execution-layer bytes, the exact v1.0.2 gates must be rerun;
  v1.0.1 evidence is not silently promoted to v1.0.2.
- Mathematical/unit test specifications for double-average soft targets,
  entropy-change actor loss, elementwise Q-clip, twin critics,
  termination/truncation, warm-up, uniform replay, hard target sync, checkpoint
  resume and both evaluation modes.
- Canonical SD-SAC diagnostics, two-channel evaluation validation and standalone
  tables/figures.
- Random-start transactional environment, simulation-time obstacle schedules,
  fixed-duration action control and fail-closed ARC lifecycle inherited into
  this self-contained archive.

## Still required on ARC

1. Build the package SIF from `apptainer/tb3_phase1_foxy.def`.
2. Run `apptainer test` and `scripts/run_tests.sh` in that exact SIF with zero skips.
3. Complete and validate the one-seed 10,000-transition live-Gazebo calibration plus E2/E3 evaluation.
4. Complete and validate the unchanged two-seed pilot plus evaluations.

Until all four gates pass without changing the package or SIF, status is a
**corrected ARC source candidate**, not frozen/verified and not authorized for
the five-seed controlled array.
