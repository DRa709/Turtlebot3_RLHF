# ARC/local audit corrections — DQN Random 1.1.7

## Response to the independent v1.1.6 bytecode/build-boundary audit

The 2026-09-04 audit examined
`TurtleBot_DQN_Random_v1.1.6_ARC_Immutable_Source_Corrected.zip`, SHA-256
`e8a4b35a13c392d14164811c1e5163eaf14d62a383eccf51b0857d2e19be61c0`.
It independently reported 158/158 source tests, a clean Foxy build/preflight,
and a passing 10.25-s Gazebo probe. It also correctly demonstrated that the
v1.1.6 verifier ignored cache/bytecode paths while `%files .` copied them into
the SIF build context, and identified that a lease override outside the result
root was not covered by the sole writable bind. Version 1.1.7 was produced as a
separate release from that reviewed archive.

| Audit finding | 1.1.7 disposition | Regression evidence |
|---|---|---|
| A checked-hash `.pyc` under `__pycache__` could be omitted from the release inventory, copied by `%files .`, and imported | Release verification now rejects `.pyc`, `.pyo`, `__pycache__`, `.pytest_cache`, `.git`, `build`, `install`, `log`, `*.egg-info`, and notebooks before invoking package Python. The Python identity verifier independently reports the same forbidden paths, and manifest generation refuses to bless them | A regression constructs a valid checked-hash `identity.*.pyc` containing an executable payload; both trusted-external and package-local verification fail before the release-manifest stage and the payload marker remains absent |
| The documented build operated directly on the working extraction | `scripts/build_sif.sh` creates a fresh temporary source tree through `scripts/materialize_release.py`, which verifies hashes without importing source modules and copies only authenticated manifest entries plus the manifest itself. The staged tree is reverified before the definition is invoked | Materialization tests authenticate an exact clean copy and refuse dirty sources; a functional fake-Apptainer test proves the definition is invoked from the temporary manifest-exact tree rather than the working extraction |
| Python operations could recreate bytecode during verification/build/runtime | The verifier, manifest generator, build wrapper, definition environment/post section, unit-test runner, and final train/evaluation environments set bytecode writing off; compilation uses an external temporary cache root | Contract tests inspect every boundary, while the final archive hygiene gate requires zero cache or bytecode paths |
| `TB3_LEASES` could resolve outside the only writable `TB3_OUTPUT` bind | The only accepted lease path is exactly `$TB3_OUTPUT/.leases`. The submission wrapper creates and re-canonicalizes it; both allocated scripts independently recheck it | Functional wrapper test rejects an external lease directory; static tests require both batch checks |

The DQN learner, random-start sampler, reset/step engine, reward and termination
semantics, observation/action definitions, Gazebo world, obstacle controller,
seed protocol, budgets, evaluation design, recorder, validator, and analysis
semantics remain unchanged from v1.1.6. Changes are limited to provenance,
image materialization, lease-path containment, version metadata, documentation,
tests, and regenerated manifests.

### Version 1.1.7 promotion boundary

This release remains an ARC candidate until one exact manifest-materialized SIF
passes `apptainer test` with the full zero-skip suite, the at-least-10-second
live Foxy/Gazebo probe, and the 10,000-transition calibration train/evaluation
pair. Preserve the SIF SHA-256 and do not begin pilot or controlled runs before
those gates pass.

## Response to the independent v1.1.5 immutable-source audit

The 2026-09-04 audit examined
`TurtleBot_DQN_Random_v1.1.5_ARC_Source_Corrected.zip`, SHA-256
`8d177691a8262c53f12b0075bc714af8afa2306956531a882970e31738fdbd67`.
It reported 153/153 local tests and a passing 10.253-s Foxy/Gazebo probe, but
correctly demonstrated that a manifested external entrypoint could be changed
after authentication and then executed through the writable source bind.
Version 1.1.6 was produced as a separate release from that authenticated
archive.

| Audit finding | 1.1.6 disposition | Regression evidence |
|---|---|---|
| Runtime source was external to the SIF and mounted read/write | The definition's `%files` embeds the complete release at `/opt/turtlebot_dqn_random`, installs it at build time, and makes the embedded tree non-writable. Training/evaluation bind only the result root and execute the in-image entrypoints | Static tests require the embedded path, forbid a source bind, and inspect both final Apptainer calls |
| Rehashing only `RELEASE_MANIFEST.sha256` did not authenticate the current listed file bytes | The trusted in-image verifier runs against the embedded release and against the external extraction through a fixed `:ro` inspection bind; the two release-manifest digests must agree | Fake-Apptainer tests distinguish embedded and external trees and reject either digest/tree mismatch |
| A queued job could later read a changed external batch script | The wrapper copies the selected batch script, checks the snapshot against the expected digest read from the authenticated SIF's manifest, removes its write bits, and submits that snapshot; it rechecks the external release identity before submission, and the allocated script verifies its own spooled bytes | Post-authentication mutation tests change the original entrypoint and batch source; a separate assertion requires the SIF—not the external manifest—to supply the snapshot digest |
| No final release-tree check preceded `COMPLETE` | Both in-image run wrappers call `identity verify-release` after validation and artifact sealing but before setting `STATUS=COMPLETE` | Source contract tests inspect both completion boundaries |

The DQN network/update, replay, exploration, checkpoint state, random-start
sampler, episode engine, reward/termination semantics, Gazebo world, obstacle
schedule, seed values, budgets, evaluation protocol and data schema are
byte-identical to v1.1.5. The algorithm-version marker changes to 1.1.6, so the
configuration digest changes without changing a scientific hyperparameter.

### Version 1.1.6 promotion boundary

This release closes the reported source-level race but remains an ARC candidate.
Build the exact SIF from the package root, pass its built-in `apptainer test`
with zero skips, remove all SIF write bits, record its SHA-256, and repeat the
10+ simulation-second live Foxy/Gazebo probe inside it. Only then submit the
calibration train/evaluation pair. Pilot and controlled arrays remain blocked
until calibration outputs validate and seal as `COMPLETE`.

## Response to the independent v1.1.4 submission-boundary audit

The 2026-09-04 audit examined
`TurtleBot_DQN_Random_v1.1.4_ARC_Permanent_Correction.zip`, SHA-256
`7d23736b8abe2a5f195f6320a69e4d01080edc2852f3df9723e91fad848377df`.
Version 1.1.5 was produced as a separate release from that authenticated
archive. The report was treated as evidence to reproduce, not as executable
authority.

| Audit finding | 1.1.5 disposition | Regression evidence |
|---|---|---|
| `--export=ALL,NAME=value` lets ambient caller values override the wrapper's values and transports the full caller environment | Removed `ALL`. The Slurm client now starts under `env -i` with a reviewed host environment, while `--export` names only canonical package inputs, authenticated digests, controlled host identity/path values, and evaluation selectors when applicable | Functional fake-Slurm test supplies conflicting/symlinked `TB3_*`, guard, Slurm, loader and shell variables; it inspects both the client environment and export list and retargets the original symlinks during the queued interval |
| Apptainer/Singularity controls survive `--cleanenv` and can add environment values or binds | The wrapper removes all four control-prefix families before login-node authentication. Both batch scripts clear them before their first runtime call and assert that only their exact package-owned `APPTAINERENV_*` list exists at the final process boundary | Functional training/evaluation batch test injects all four prefix families; fake Apptainer fails if any reaches authentication/query calls or if any unapproved name reaches the final call |
| A symlink can introduce `%j` after log-root canonicalization | Percent tokens are rejected on input, canonical log root, derived directory and post-creation resolved directory; the general path grammar also excludes `%` | Regression test supplies a percent-free symlink whose resolved target contains `%j` and requires failure before Apptainer or Slurm |
| Source/image symlink retargeting can separate queued execution from submission authentication | The wrapper exports canonical paths plus the authenticated image and release-manifest SHA-256 values; each batch script rejects an identity mismatch before any Apptainer call and rechecks immediately before the scientific call | Post-authentication symlink-retarget test and pre-Apptainer digest-mismatch test |

The DQN learner, random-start sampler, episode engine, reward/termination
semantics, Gazebo world, obstacle schedule, seeds, budgets, checkpoint cadence
and data contract are byte-identical to v1.1.4. The algorithm-version marker
changes to 1.1.5, so the configuration digest changes even though no scientific
hyperparameter changes.

### Version 1.1.5 promotion boundary

This release is a source-corrected ARC candidate, not an ARC-certified result.
Build its exact SIF on ARC, record the SIF SHA-256, pass `apptainer test`,
pass every package test with zero skips, and repeat the 10+ simulation-second
Gazebo relocation/obstacle probe inside that SIF. Then submit calibration and
its dependency-gated E2/E3 evaluation only through `arc/submit_dqn.sh`.
Do not start pilot or controlled arrays until those outputs validate and seal
as `COMPLETE`.

## Response to the independent v1.1.3 wrapper audit

The 2026-09-04 audit examined
`TurtleBot_DQN_Random_v1.1.3_ARC_Permanent_Correction.zip`, SHA-256
`6e00c972a813cfa6ffccc759fecabf2358c6c2b183048f857d7d7a84fde983f6`.
Version 1.1.4 was produced from that authenticated archive; v1.1.3 was not
modified in place.

| Audit finding | 1.1.4 disposition | Regression evidence |
|---|---|---|
| GNU long-option abbreviations, `--`, `:`, short options or an operand can bypass wrapper ownership | Replaced pass-through tokens with an exact attached-value allowlist; approved site arguments precede wrapper-owned arguments; the package script follows the wrapper's own `--`; inherited Slurm option variables are removed from the `sbatch` environment | Every prefix of `--array`, `--export`, `--export-file`, `--chdir`, `--output`, `--error`, `--wrap`, `--dependency` and `--kill-on-invalid-dep`, plus bundled shorts, operands and environment overrides, is rejected by `tests/test_arc_scripts.py` |
| Evaluation may be submitted before training finishes | Evaluation submission adds owned `--dependency=afterok:<training-array-id>` and `--kill-on-invalid-dep=yes` arguments | Functional mocked-`sbatch` test inspects both emitted controls and final script position |
| Only the log root, not the final derived directory, is checked | The canonical derived directory is checked before and after creation; `%` is forbidden in the supplied log root | Symlink-to-release and percent-substitution regression cases |
| Apptainer help contains v1.1.2 image names | `%help` uses `TurtleBot_DQN_Random_Runtime_Foxy.sif` | Version-consistency test now inspects the definition |

The audit independently reported 148/148 tests, a fresh Foxy build/preflight,
and a 10.252-s live Gazebo probe passing the 0.05-m obstacle tolerance for
v1.1.3. Those results support the scientific/runtime layer that remains
unchanged, but they are not evidence for the revised v1.1.4 submission wrapper
or an exact v1.1.4 SIF.

### Version 1.1.4 promotion boundary

Before ARC calibration, build the exact v1.1.4 SIF, record its SHA-256, pass
`apptainer test`, pass the complete package suite with zero skips, and repeat the
10+ simulation-second Gazebo probe inside that SIF. Then submit calibration only
through `arc/submit_dqn.sh` and require both training and its dependency-gated
E2/E3 evaluation to finish with validated, sealed `COMPLETE` outputs.

## Response to ARC job 7361130 task 0

The ARC task built the ROS package successfully but stopped before Gazebo or
training with `PREFLIGHT FAIL: release manifest mismatch`. Its Slurm output and
error files were created in the package directory because the 1.1.2 runbook
instructed submission from `TB3_REPO` and both batch files used relative
`#SBATCH --output` and `#SBATCH --error` paths. Those runtime files were
correctly rejected as undeclared release content. Reproducing the submitted
layout produced these exact diagnostics:

```text
undeclared file tb3-dqn-random-7361130_0.err
undeclared file tb3-dqn-random-7361130_0.out
```

Version 1.1.3 makes `arc/submit_dqn.sh` the only supported submission entry
point. It authenticates the source before submitting, writes Slurm logs below
the separate results/log root, owns the frozen phase array and export fields,
and passes a guard required by the internal batch files. Both the submission
wrapper and allocated job reject path overlap with `TB3_REPO`. The batch job
repeats package authentication before it creates a workspace or run directory.

The failed ARC task collected zero transitions and exercised neither Gazebo nor
the DQN learner. It therefore requires no checkpoint recovery and supplies no
evidence about random initialization or learning. Preserve the failed logs and
run directory outside the package, and use a new Slurm job/run identity.

The parent source archive is
`TurtleBot_DQN_Random_v1.1.2_ARC_Source_Corrected.zip`, SHA-256
`f4a8dadac0a1c836bb7b04d54dd500bbcbbd8aebd8f7e5ca3c968e097cb8bd4a`.
The DQN learner and scientific environment are unchanged in 1.1.3.

### Version 1.1.3 promotion boundary

The static package verifier and all 114 tests that do not require the exact
PyTorch/ROS container stack pass locally. This is source-level evidence, not
ARC certification. Before calibration, the cohort must:

1. Run `apptainer test` on the preserved SIF.
2. Run `bash scripts/run_tests.sh` inside that SIF and require all 148 tests to
   pass with zero failures and zero skips.
3. Repeat the 10+ simulation-second live obstacle/relocation probe inside the
   same SIF.
4. Submit the 10,000-transition calibration only through
   `arc/submit_dqn.sh`, followed by its E2/E3 evaluation.
5. Require validated `COMPLETE` markers and sealed output inventories before
   pilot promotion.

## Response to the v1.1.1 audit

The 2026-09-03 audit examined
`TurtleBot_DQN_Random_v1.1.1_ARC_Audit_Corrected.zip` (SHA-256
`a0894c3542897b2d59d4b65671f7fdcd601516d1335da6104a2c8c6e8c94289f`).
Version 1.1.2 was patched from that exact archive. The audited archive was not
modified in place.

| Audit finding | 1.1.2 correction | Enforced by |
|---|---|---|
| Isolated container testing loads Foxy's incompatible external `launch_testing` plugin | `scripts/run_tests.sh` exports `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before invoking Pytest | `tests/test_arc_scripts.py` and the exact-SIF mandatory command |
| Obstacle acknowledgements route step-0 effects into `FakeSim.transitions` instead of the canonical CSV stream | `FakeSim.apply_obstacle_control` returns the acknowledgement; `LoopHarness` routes the resulting engine effects through `apply_engine_effects` | Nominal validator fixture plus the explicit one-step-0-per-episode-key assertion |
| Reward-tamper test searches for wording the validator does not emit | Assertion now matches `six reward components` | `test_tampered_reward_component_fails_identity` |
| Manual runbook uses the legacy ARC module name | Runbook uses `module load apptainer`; batch scripts try that name first and keep the legacy name as fallback | Shell syntax and ARC script contract tests |

The live results reported by the v1.1.1 audit are accepted as evidence about
that exact source and local Foxy/Gazebo runtime: maximum dynamic-obstacle error
`0.005940 m` over `10.256 s`, including both reversal boundaries, and robot
relocation error `0.0000331216 m`. The scientific runtime code responsible for
those results is unchanged in 1.1.2.

## Current promotion boundary

Version 1.1.2 is source-corrected but is not declared ARC-verified by this
document. Before calibration:

1. Build the SIF from this exact release and record its SHA-256.
2. Pass `apptainer test`.
3. Run `bash scripts/run_tests.sh` unchanged inside that SIF; every collected
   test must pass and the script must report zero skips.
4. Repeat the 10+ simulation-second reset/relocate/obstacle probe inside that
   same SIF and retain its raw samples.
5. Only then submit the 10,000-transition calibration and E2/E3 evaluation.

Do not weaken a threshold or reinterpret a failed test as a pass.

## Response to the v1.1.0 audit

This document records the response to the 2026-09-03 audit of
`TurtleBot_DQN_Random_v1.1.0.zip` (SHA-256
`39237a6b8e584665814838e71a8bf61a1ecabae900930244171212a1b1e73c61`).
The correction was made from that exact archive. No Double DQN, Dueling,
Rainbow, SAC or A2C source is present or used.

### Disposition of every reported finding

| Audit finding | 1.1.1 correction | Regression evidence |
|---|---|---|
| Foxy rejects `ros2 topic echo --once` | `scripts/wait_for_sim.sh` invokes the bounded subscriber in `scripts/wait_for_topics.py` | `tests/test_arc_scripts.py`; Python/shell syntax gates |
| 63-bit seeds lose precision through `float` | `validator._i` accepts only decimal integer text and converts it directly with `int` | `tests/test_validator_parsing.py` uses the exact audited seed |
| Validator fixture omits manifest-declared tests | `package_with_protocol` copies the complete tests directory while excluding only generated caches | Full `tests/test_validator.py` module in the exact SIF |
| Moving obstacles lag the analytic displacement | Dynamic-obstacle links are kinematic, gravity-free, collision-active bodies driven by the existing planar plugin | `tests/test_world.py`; live full-reversal check still required |
| Ubuntu and pip Matplotlib installations mix | Pinned dependencies live in isolated `/opt/tb3-python`; `%test` checks the actual `matplotlib` and `mpl_toolkits.mplot3d` origins | `tests/test_package_hygiene.py`; `apptainer test` |
| Foxy world path with spaces can fall back to `empty.world` | Preflight and simulator startup reject whitespace before launch | `tests/test_package_hygiene.py` |

The frozen `obstacle_position_tolerance: 0.05` and
`validation.max_obstacle_position_error: 0.05` are unchanged. The DQN learner
files and mathematical update are unchanged.

### Historical v1.1.1 promotion boundary

Source/static success is not live Gazebo certification. Before submitting the
10,000-transition calibration array, all of the following must be retained as
evidence:

1. Build the SIF from this archive and record its SHA-256.
2. Run `apptainer test`; the isolated Matplotlib check must pass.
3. Run `bash scripts/run_tests.sh` inside that exact SIF. All 143 tests must
   pass with zero skips.
4. Repeat the live Foxy/Gazebo reset/relocate/settle obstacle probe for at least
   10 simulation seconds. Both obstacle position errors must remain at or below
   0.05 m, including both reversal boundaries.
5. Only then submit calibration and its E2/E3 evaluation as described in
   `ARC_RUNBOOK.md`.

If step 3 or 4 fails, do not relax a threshold and do not start calibration.
Retain the logs and issue a new correction from this release.
