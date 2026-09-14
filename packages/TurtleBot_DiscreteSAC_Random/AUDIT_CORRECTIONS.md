# Discrete SAC v1.0.2 — audit closure

Date: 2026-09-04

This document records the finding-by-finding correction and independent local
closure of the v1.0.0 audit. The original source archive SHA-256 is
`34f7b48175ff049d26edc525f6e96f9f43be29452f975e44497c44463a723a3c`.
Two distinct v1.0.0 audit artifacts are known; their identities and roles are
kept separate in `PROVENANCE.md` instead of being described by one ambiguous
“supplied audit report” label.

## Protected algorithm boundary

The production Discrete SAC implementation remains vanilla categorical SAC:
one categorical actor, twin online/target critics, exact five-action
expectations, clipped double Q, fixed alpha 0.2, uniform replay, one-step
terminal-only targets and hard target synchronization. It does not import or
contain Rainbow, SD-SAC, DQN, PER, n-step or distributional components.

`turtlebot3_drl_nav/discretesac_agent_node.py` is byte-identical to the
v1.0.0 parent. In `discretesac.py`, the only production delta converts the
reported `batch_size` and `replay_size` diagnostics from integral floats to
integers so they satisfy the exact CSV schema. This does not alter the SAC
networks, objectives, actions, gradients, optimizer steps, replay or
checkpoints. The remaining corrections affect the shared environment,
execution, validation, packaging and test evidence.

## Incorporated audit findings

1. **63-bit seed precision.** `validator._i` now accepts only signed decimal
   integer text and calls `int` directly. It never passes identity fields
   through IEEE-754 floating point. Boundary and reported-seed regressions are
   in `tests/test_validator_parsing.py`. The two learner diagnostics consumed
   as integers are now emitted as integers, with a regression assertion in
   `tests/test_discretesac.py`.
2. **Dynamic-obstacle tracking.** Both moving links in
   `worlds/phase1_mixed.world` are explicitly kinematic and gravity-free.
   Collision geometry is retained. The 0.05 m analytic tracking threshold is
   unchanged; no tolerance was relaxed.
3. **ROS 2 Foxy readiness.** `scripts/wait_for_sim.sh` no longer uses the
   unsupported `ros2 topic echo --once`. It calls
   `scripts/wait_for_topics.py`, a bounded Python 3.8 `rclpy` subscriber
   that requires every declared topic and strictly advancing simulation time.
4. **Validator fixture routing.** Obstacle acknowledgements are routed back
   through the canonical loop-harness dispatcher, so step-zero transitions are
   written to `transitions.csv`. The fixture copies all manifest-declared
   tests, and the reward-tamper assertion matches the validator's real message.
5. **Python 3.8 source gate.** The Discrete SAC source discriminator uses its
   existing source-segment helper instead of Python 3.9's `ast.unparse`.
6. **Container Python isolation.** The image builds an isolated
   `/opt/tb3-python` virtual environment for pinned Torch, NumPy, PyYAML,
   pandas, Matplotlib and pytest. The image test imports pyplot, 3-D toolkits,
   pytest and ROS modules and rejects any scientific module outside that
   environment.
7. **Foxy pytest compatibility.** `scripts/run_tests.sh` disables third-party
   pytest plugin auto-loading before collection. The ordinary unit suite
   therefore does not load Foxy's incompatible `launch_testing` plugin.
8. **Whitespace paths.** Package, run, evaluation, training-run and simulator
   paths are rejected before ROS/Gazebo startup if they contain whitespace.
9. **Archive hygiene.** Static verification now fails on Python caches and
   missing executable bits for scripts and Slurm entry points instead of
   silently deleting or ignoring them.
10. **ARC module documentation.** The current `module load apptainer` form is
    attempted first in jobs and shown in the runbook, with the legacy module
    name retained only as a fallback in jobs.

## Additional correction found during incorporation

The v1.0.0 analysis-rendering fixture used `--include-incomplete`, omitted
training/evaluation manifest linkage, and tested a one-seed 40-step result
against the production pilot contract. It could render files without proving
the normal completeness gate.

The corrected fixture instead copies the package to a temporary directory, creates
a self-consistent one-seed/40-step test-only pilot there, writes linked
manifests, and invokes `scripts/make_tables.py` and
`scripts/make_figures.py` without the diagnostic bypass. Additional tests
require rejection of missing seeds, missing checkpoint evaluations and
unexpected checkpoint evaluations. Production configuration is not modified.

## Independent v1.0.1 closure evidence

The bundled `LOCAL_VERIFICATION_REPORT_v1.0.1.md` reports that the corrected
v1.0.1 parent passed all 163 tests in the intended isolated Python 3.8 runtime,
built cleanly under ROS 2 Foxy, and passed the packaged readiness and
independent live-Gazebo probes. The largest dynamic-obstacle trajectory error
was 0.01034 m against the unchanged 0.05 m limit; robot relocation error was
0.0000298306 m against 0.03 m; and named physical robot–obstacle contact was
observed. The auditor found no remaining local source blocker.

v1.0.2 changes documentation, evidence lineage, version identity and release
manifests only. The protected learner, environment, simulator and validator
hashes are authenticated in `PROVENANCE.md`.

## Promotion boundary

Local closure is not ARC certification. Before calling v1.0.2 frozen or
starting controlled training:

1. build the exact SIF and record its SHA-256;
2. pass `apptainer test`;
3. pass `bash scripts/run_tests.sh` inside that SIF with every test executed,
   zero failures and zero skips;
4. repeat a 10+ simulation-second obstacle/relocation probe in that SIF;
5. pass the 10,000-transition calibration and both policy-channel evaluation;
6. pass the unchanged two-seed pilot and its evaluations.

Any source or image change after those gates creates a new artifact and
requires repeating the affected gates.
