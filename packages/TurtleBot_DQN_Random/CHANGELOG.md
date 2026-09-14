# Changelog

## 1.1.7 — 2026-09-05 — manifest-materialized image boundary

- Changed forbidden caches and generated products from silently excluded
  inventory entries to hard release failures. Verification stops before any
  package Python import when `.pyc`, `.pyo`, `__pycache__`, `.pytest_cache`,
  `.git`, `build`, `install`, `log`, `*.egg-info`, or notebook content exists.
- Added `scripts/materialize_release.py` and `scripts/build_sif.sh`. The SIF is
  now built from a fresh tree containing only authenticated release entries,
  and the staged and embedded copies are both reverified.
- Made manifest materialization and manifest regeneration import-free: both
  inventory and hash file bytes directly before any release module can run.
- Disabled bytecode writes across verification, manifest generation, image
  construction/testing, and training/evaluation runtime boundaries.
- Restricted the transport lease directory to exactly
  `$TB3_OUTPUT/.leases`, keeping it inside the sole writable runtime bind.
- Added a valid checked-hash bytecode exploit regression, exact-materialization
  tests, build-boundary assertions, and lease-containment tests.
- Preserved the DQN learner, random-initialization distribution, environment
  dynamics, reward/termination semantics, seeds, budgets, evaluation protocol,
  recorder, validator, and analysis semantics from 1.1.6.

## 1.1.6 — 2026-09-05 — immutable source execution boundary

- Embedded the complete authenticated DQN release in the Apptainer SIF and
  installed the ROS package during image construction. Runtime tasks no longer
  build from or execute a mutable external source tree.
- Removed every training/evaluation source bind. The only writable runtime bind
  is the standalone DQN result root; entrypoints, configuration, world,
  validator and analysis code are read from the SIF.
- Changed submission authentication to use the trusted in-image verifier. The
  external extraction is inspected through a fixed read-only bind, and its
  release-manifest digest must equal the embedded digest.
- Added a read-only batch-script snapshot checked against the expected digest
  read from the authenticated SIF's manifest. Each allocated task verifies the
  SHA-256 of Slurm's spooled script before any scientific work.
- Required the SIF file to have no write permission bits and rechecked its
  digest immediately before the scientific process.
- Added a final embedded-release verification before either run wrapper can set
  `COMPLETE`.
- Made package Python compilation use an external temporary bytecode directory,
  so the same verifier works against the read-only tree inside a SIF.
- Added regression coverage for post-authentication changes to manifested
  entrypoints and batch scripts.
- DQN equations, network, replay, exploration, checkpoint state, random-start
  distribution, reset/step engine, reward, world, seeds, budgets and scientific
  hyperparameters are byte-identical to 1.1.5.

## 1.1.5 — 2026-09-04 — authenticated environment boundary

- Removed `ALL` from Slurm export. The client now runs under a reviewed
  `env -i` host environment, and the job receives only canonical
  package-owned values, authenticated image/release digests, controlled host
  identity/path values, and explicit evaluation selectors.
- Removed ambient Apptainer and Singularity control-prefix families before the
  wrapper authentication call and at batch entry. Training and evaluation fail
  unless the final runtime environment contains exactly their enumerated
  package-owned `APPTAINERENV_*` variables.
- Bound each queued task to the wrapper-authenticated SIF and release-manifest
  SHA-256 values. Both batch scripts verify those identities before their first
  Apptainer call and again immediately before the scientific process.
- Enforced an absolute literal-safe grammar for every canonical ARC path and
  repeated percent-token rejection after log-root canonicalization, after
  deriving the phase directory, and after its post-creation resolution.
- Added adversarial tests for Slurm `ALL` precedence, ambient shell/loader and
  runtime controls, source/image symlink retargeting, digest substitution,
  unsafe canonical paths, and percent-bearing symlink targets.
- DQN equations, network, replay, exploration, checkpoint state, random-start
  distribution, reset/step engine, reward, world, seeds, budgets and scientific
  hyperparameters are byte-identical to 1.1.4.

## 1.1.4 — 2026-09-04 — strict Slurm submission boundary

- Replaced arbitrary `sbatch` argument forwarding with an exact allowlist for
  attached-value ARC site selectors: `--account=`, `--partition=`, `--qos=`,
  `--reservation=` and `--constraint=`. Empty values, whitespace, duplicates,
  abbreviations, short options, `--`, heterogeneous-job separators and operands
  are rejected before environment access or submission.
- Ordered approved caller site selectors before all wrapper-owned arguments and
  placed the sole package batch script after the wrapper's own `--` boundary.
  The wrapper therefore retains final authority over the array, exports,
  working directory, log paths, dependency and executable even if Slurm accepts
  GNU-style long-option abbreviations.
- Removed inherited `SBATCH_*`, `SLURM_CLUSTERS` and `SLURM_HINT` variables from
  the `sbatch` process environment, closing Slurm's second option-input channel;
  site selection remains explicit through the wrapper allowlist.
- Added `afterok:<training-array-job-id>` and
  `--kill-on-invalid-dep=yes` as wrapper-owned evaluation controls. Evaluation
  does not consume an allocation before the entire referenced training array
  succeeds, and an invalid dependency is cancelled.
- Canonicalized and checked the final derived Slurm log directory before and
  after creation, and rejected `%` in `TB3_SLURM_LOGS` so Slurm substitutions
  cannot redirect a validated path.
- Made Apptainer `%help` use the version-neutral runtime image name and extended
  version/submission regression coverage to include the definition and every
  owned-option prefix and operand bypass.
- DQN equations, network, replay, exploration, checkpoint state, random-start
  distribution, reset/step engine, reward, world, seeds, budgets and scientific
  hyperparameters are byte-identical to 1.1.3.

## 1.1.3 — 2026-09-04 — ARC submission-tree isolation

- Replaced direct `sbatch` use with the mandatory `arc/submit_dqn.sh` entry
  point. It authenticates the release before submission and supplies absolute
  Slurm stdout/stderr paths outside the package tree.
- Removed relative Slurm output/error and default-array directives from both
  batch files. The submission wrapper derives the frozen array range for
  calibration, pilot and controlled phases and refuses overrides of protocol-
  critical `sbatch` options.
- Added a submission guard so direct invocation of either internal batch file
  is refused, rather than silently bypassing the safe wrapper.
- Added canonical-path checks that reject results, leases, images, logs or the
  Slurm submit directory when they overlap the authenticated release tree.
- Repeated package authentication inside the allocated job before creating a
  workspace or run directory. Exact mismatch paths remain visible in Slurm
  evidence; training preflight also emits detailed shared/release diagnostics.
- Added functional regression tests with mocked `apptainer` and `sbatch` to
  prove that the wrapper routes logs externally and owns the array/export/log
  options, plus negative tests for nested output and option overrides.
- Rewrote every training and evaluation command in `ARC_RUNBOOK.md` and
  `README.md` to use the safe wrapper and an algorithm-isolated directory tree.
- DQN equations, network, replay, exploration, checkpoint state, random-start
  sampler, reset/step engine, reward, world, seed values, budgets and scientific
  hyperparameters are unchanged from 1.1.2.

## 1.1.2 — 2026-09-03 — mandatory-test corrections (shared layer 3.0.2)

- Disabled third-party Pytest plugin auto-loading inside the mandatory test
  command. The package tests do not use external plugins, and ROS 2 Foxy's
  legacy `launch_testing` entry point is incompatible with the pinned Pytest
  8.3.3 API in the isolated container environment.
- Corrected the closed-loop test harness so effects emitted by an
  obstacle-control acknowledgement return through the harness dispatcher and
  canonical CSV writer. Step-0 transitions no longer bypass
  `transitions.csv` into the fake simulator's private in-memory list.
- Strengthened the nominal validator test to require exactly one step-0 row for
  every represented `(phase, episode_key)` and corrected the reward-tamper
  assertion to match the validator's actual six-component identity message.
- Changed ARC documentation to use the currently documented `apptainer` module
  name; batch scripts retain `containers/apptainer` as a fallback.
- The DQN learner, random-start sampler, reset/step engine, Gazebo world and
  dynamic-obstacle controller are unchanged from 1.1.1. The only environment
  source edit is the shared-layer version text emitted at startup.

The 1.1.1 audit independently verified the corrected live Gazebo behavior over
10.256 simulation seconds. Version 1.1.2 remains a release candidate until the
exact ARC-built SIF passes `apptainer test`, the mandatory suite with zero
failures/skips, and the repeated live probe inside that same SIF.

## 1.1.1 — 2026-09-03 — ARC/local audit corrections (shared layer 3.0.1)

- Replaced the Foxy-incompatible `ros2 topic echo --once` readiness calls with
  a bounded `rclpy` subscriber. Readiness now requires a message from every
  required simulator topic and two strictly increasing `/clock` stamps.
- Changed validator integer parsing to validate decimal syntax and call `int`
  directly. Recorded 63-bit generator seeds no longer pass through IEEE-754
  floating point; regression coverage includes `6678464068980594013`.
- Corrected the validator test fixture so it retains the test files declared by
  `SHARED_LAYER_FILES.txt` before regenerating its temporary manifests.
- Made both scripted dynamic obstacles collision-active kinematic links with
  gravity disabled. This removes ground/contact-force drag from the prescribed
  planar velocity without weakening the frozen 0.05 m trajectory tolerance.
- Isolated the pinned Python scientific stack in `/opt/tb3-python`. The image
  test now refuses a Matplotlib or `mpl_toolkits.mplot3d` import outside that
  environment, preventing a mixed Ubuntu/pip installation.
- Added fail-closed checks for whitespace in Foxy package/runtime paths, which
  otherwise can cause `gazebo.launch.py` to load `empty.world` silently.
- Added regression tests for readiness, exact seed parsing, manifest-fixture
  completeness, obstacle model properties, container isolation and path safety.
- The DQN equations, network, replay, exploration, budget and checkpoint code
  are unchanged from 1.1.0.

This is a source-corrected ARC release candidate. Promotion still requires the
exact SIF to build and pass every test with zero skips, followed by a live
Foxy/Gazebo obstacle measurement covering at least two half-periods and then the
unchanged ARC calibration gate.

## 1.1.0 — 2026-09-03 — fail-closed random-start ARC release candidate (shared layer 3.0.0)

- Episode 1 and every later episode now use the same deterministic,
  episode-indexed random-start protocol. Robot relocation, reset, settling,
  sensor capture and obstacle scheduling are awaited, acknowledged, checked and
  recorded before the first policy action.
- Policy inference runs while Gazebo is paused. Every action is then applied for
  one simulation-time control interval, removing forward-pass latency from the
  simulated transition duration.
- At a control deadline the robot command is zeroed immediately, even when ROS
  sensor delivery for that boundary is still queued; the environment accepts
  only boundary-fresh scan/odometry/obstacle samples before pausing and recording
  the successor state.
- Dynamic obstacles use simulation time, explicit episode schedules and
  acknowledgements. Measured trajectories are checked against the independent
  analytic schedule on every transition.
- Initialization and transition rows now include requested, realized and
  odometric poses, reset scan clearance, sensor ages, decision timing and
  obstacle tracking errors.
- Training trajectories are explicitly labelled `policy_mode=epsilon_greedy`;
  only frozen DQN evaluation trajectories are labelled `greedy`.
- LiDAR cleaning follows the safety-relevant REP 117 distinction: `+Inf` is
  free range, while `-Inf`, NaN and non-positive invalid values become the
  minimum range and therefore cannot hide a near obstacle.
- The validator independently reconstructs rewards, terminal precedence,
  episode summaries, random starts, obstacle trajectories, checkpoint cadence,
  evaluation membership and exact run inventories from raw artifacts.
- Replay storage is now an O(1)-overwrite list ring, so uniform minibatch
  sampling no longer copies up to 100,000 entries on every one of roughly
  495,000 gradient steps. Full checkpoints preserve the ring write index.
- Training and evaluation wrappers now fail closed on package/image identity,
  launch arguments, checkpoint ownership and digest, output reuse, validation,
  interruption and incomplete final checkpoints. `COMPLETE` is written only
  after the immutable run inventory is sealed and verified.
- The validator now binds `run_manifest.json`, numeric Slurm identities, runtime
  versions, the copied Gazebo world and the contact-instrumented robot model to
  their recorded digests; these files are no longer provenance-by-convention.
- The shared Foxy Apptainer recipe no longer uses an insecure or fail-open APT
  bootstrap. Repository-key acquisition and every package-install step abort
  the build on failure, and one preserved SIF is reused across algorithms.
- ARC arrays request exclusive nodes, preserve Slurm identity through
  `--cleanenv`, use leased ROS/Gazebo identities, and pin each workspace to the
  release-manifest digest.
- The controlled evaluation cadence is frozen at 100k, 200k, 300k, 400k and
  500k transitions so the final tables and checkpoint-wise figures have their
  required inputs.
- Optional campaign-level analysis gating rejects missing/duplicate
  algorithm-seed-checkpoint cells, mixed shared layers or container images, and
  evaluation runs not explicitly linked to their training run.
- This archive contains DQN only, uses a terminal-only direct-ARC workflow and
  is assessed as a standalone random-initialization package. No notebook or
  fixed-initialization comparison is part of its verification scope.

## 1.0.1 — 2026-09-02 — camera-ready analysis outputs (shared layer 2.0.1)

- `analysis.py`: IEEE style (Times-like serif, 8 pt, Type 42 fonts, 3.5 in / 7.16 in column widths), vector PDF plus 300 dpi PNG for every figure, LaTeX booktabs output for every table (`write_table` writes `.csv`, `.md`, `.tex`).
- New renderers so every entry of the table/figure list has a producer: T-R5 failure anatomy, T-R8 learner diagnostics, F-R3 performance profiles, F-R4 probability-of-improvement heatmap, F-R7 stratified held-out success, F-R8 reward composition, F-R12 checkpoint-wise held-out success.
- Readable stratum labels (distance/clearance/heading bins) in T-R3 and F-R7.
- `scripts/make_figures.py --formats pdf,png --style ieee|default`; `tests/test_analysis.py` runs both scripts on a complete harness run.
- No change to the environment, sampler, learner, recorder, validator or ARC scripts. No numerical value changed; the configuration digest moves only because the version strings in the two YAML files changed, and the shared-layer digest moves because `analysis.py` and its test are shared.

## 1.0.0 — 2026-09-02 — first self-contained direct-ARC package (rebuild of the audited submission `47612ce6…`)

Shared layer 2.0.0. Every finding of the Stage I/II record is addressed by code
plus a test; the finding IDs refer to that record.

Random initialization (Gate 4)
- F-01 ν_R declared (`RANDOM_INIT_SPEC.md`, `config/common_environment.yaml`): uniform on the clearance-admissible region derived from the parsed world, yaw uniform; tests `test_geometry.py`, `test_initialization.py`.
- F-02 seeded, episode-indexed draws (`initialization.py`): pure function of (seed, stream, index); reproducibility tests and validator recomputation.
- F-03 episode 1 randomized like every other (`episode_engine.py`); test `test_episode_one_is_randomized_not_spawn_pose`.
- F-04 transactional reset: awaited services, `success` checked, realized pose verified, odometry frame checked, contact during reset fatal, no fallback; world now loads `libgazebo_ros_state.so`; readiness waits for the services. Tests in `test_episode_engine.py` (failure injection, ordering, timeouts).
- F-05 initialization block recorded on every episode and evaluation row.
- F-08 four seed identities with distinct roles, preregistered common values, learning seed separate.
- F-19 clearance rule ≥ d_safe from every surface (centre-to-surface, footprint radius asserted smaller); zero near-band starts.
- F-20 evaluation protocol: E1 in-run per checkpoint from the evaluation seed, E2 frozen stratified held-out list, E3 anchor; `policy_mode` on every row; learner-untouched assertion.
- F-21 settle in simulation time.

Shared MDP (Gate 3)
- F-06 fixed action hold: apply at a tick, close at the next; measured `hold_sim_s`/`hold_odom_s` recorded; validator tolerance gate. Test `test_hold_is_exactly_one_control_period_and_robot_moves`.
- F-07 obstacles on the simulation clock with per-episode seeded phases sent by the environment; hold during resets. `obstacle_schedule.py`, `test_obstacle_schedule.py`.
- F-24 sub-minimum-range behaviour documented as frozen (`state.py` unchanged).

Learner (Gate 5)
- F-14 atomic checkpoints (temp + fsync + rename); policy-only and full kinds.
- F-22 ε logged is the ε used for the action (`ActionChoice.epsilon_used`); validator checks the schedule identity.
- F-25 every checkpoint loaded back and digest-compared before its row is written; final checkpoint at the exact budget.
- Update rule unchanged and re-verified bit-exactly (`test_dqn.py`).

Recording and validation (Gate 7)
- F-12 canonical streams `transitions`, `episodes`, `updates`, `evaluation`, `checkpoints` with the identity block on every row; create-only fsync'ed writers; run validator with adversarial tests (`test_validator.py`).
- Orchestrator (`orchestrator.py`) shared across packages: exact budget via the final-transition flag, checkpoint cadence, tier-1 evaluation, tier-2 post-hoc evaluation.

Launch, runtime, ARC (Gates 2, 9, 10)
- F-09 every launch argument forwarded through `launch_config.build_node_parameters` (tested); phases replace ad-hoc budget arguments.
- F-10 LF line endings everywhere; `bash -n` on every script in the tests.
- F-11 lease-based transport isolation (`arc/task_isolation.py`), `--exclusive`, Slurm identity forwarded through `--cleanenv`.
- F-13 fail-closed lifecycle: preflight, atomic `COMPLETE`/`FAILED`/`INTERRUPTED` written last after validation, `--signal=B:USR1@600` with an emergency checkpoint, non-zero exit on any failure.
- F-15 one distribution (Foxy) across image definition, scripts and metadata; pinned Python dependencies; image digest recorded in every run.
- F-16 documentation rewritten (`README.md`, `ARC_RUNBOOK.md`, `RANDOM_INIT_SPEC.md`, `DATA_CONTRACT.md`, `ALGORITHM.md`); referenced files exist (tested).
- F-17 `RELEASE_MANIFEST.sha256`, `SHARED_LAYER_MANIFEST.sha256`, `VERSION`, `ALGORITHM`; `shared_layer_sha256` in every row.
- F-18 non-empty run directories refused; streams create-only.
- F-23 single authority for the action map and every environment value (nodes declare parameters from the YAML); unused keys removed.
- F-26 behavioural tests replace text-grep tests.
- F-27 duplicate service client removed.

Removed
- Jupyter and Docker files, the legacy `experiments/` reference, the WSL install scripts, the legacy circle-goal script, the X11 recording helpers.
