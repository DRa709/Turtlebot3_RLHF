# Verification status — 1.0.2 ARC source/calibration candidate

## Independently verified local evidence

The bundled `LOCAL_VERIFICATION_REPORT_v1.0.1.md` reports the following
results for the v1.0.1 parent:

- safe archive, authenticated release/shared manifests and frozen-scenario
  regeneration;
- clean ROS 2 Foxy `colcon` build;
- 163/163 tests passed under Python 3.8.10 and the isolated pinned scientific
  environment, with zero failures and zero skips;
- independent 15/15 Discrete SAC numerical/checkpoint probe;
- successful packaged readiness gate with advancing simulation clock;
- robot relocation error 0.0000298306 m against the 0.03 m limit;
- maximum dynamic-obstacle tracking error 0.01034 m against the 0.05 m limit
  across the 5 s and 10 s reversal boundaries;
- named physical contact between the Burger and a dynamic obstacle;
- byte-identical authenticated source/runtime Gazebo world.

The auditor found no remaining local source blocker. The report's original
CRLF bytes and LF-normalized release copy are separately authenticated in
`PROVENANCE.md`.

## v1.0.2 delta and evidence inheritance

v1.0.2 changes only package/algorithm-version markers, documentation,
provenance, one provenance-integrity test and release manifests. It changes no
Discrete SAC learner, environment, simulator, validator, ARC wrapper,
analysis implementation, hyperparameter, seed, budget, evaluation protocol or
data schema. Protected implementation hashes are listed in `PROVENANCE.md`.

Because the package bytes and configuration identity changed, v1.0.2 must
still pass its own package verification and exact ARC SIF gates. The v1.0.1
local results establish evidence for the unchanged implementation, not
permission to reuse a v1.0.1 SIF or mix v1.0.1 results with v1.0.2.

## Required ARC gates

1. Verify the v1.0.2 ZIP after native Linux extraction.
2. Build `discretesac_random_v1.0.2.sif` on an ARC compute/build node and
   record its SHA-256.
3. Pass `apptainer test` and unmodified `scripts/run_tests.sh` in that exact
   SIF with all tests executed and zero skips.
4. Complete and validate the one-seed 10,000-transition calibration plus both
   stochastic and deterministic E2/E3 evaluation channels.
5. Without changing source or SIF bytes, complete and validate the two-seed
   pilot plus both evaluation channels.

Until all five gates pass, status is **ARC source/calibration candidate**, not
frozen/verified and not authorized for the five-seed controlled array.
