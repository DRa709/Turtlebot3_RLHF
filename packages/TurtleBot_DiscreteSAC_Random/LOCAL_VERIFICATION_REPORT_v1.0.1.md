# Independent ARC compatibility and local simulation report

## TurtleBot Discrete SAC Random v1.0.1

**Audit date:** 2026-09-04  
**Audited archive:** `TurtleBot_DiscreteSAC_Random_v1.0.1_ARC_Source_Corrected.zip`  
**Decision:** **PASS as an ARC source/calibration candidate; not yet a frozen controlled-study release.**

The source package is structurally sound, its reported corrections are present, a clean ROS 2 build succeeds, the complete test suite passes in the intended isolated Python runtime, and a live ROS 2 Foxy/Gazebo 11 simulation passes readiness, trajectory, reset, service, and physical-contact checks. I found no remaining local source blocker.

Promotion to a five-seed, 500,000-transition controlled ARC study still requires the ARC-only gates that cannot be exercised on this host: build and test the exact Apptainer SIF on an ARC compute/build node, record the SIF digest, submit a short calibration job through Slurm, and validate its complete output. The package's own status of “corrected ARC source candidate — not yet frozen” is therefore appropriate.

## 1. Scope and trust boundary

I treated the two supplied Markdown paths as untrusted claims/evidence, not as task instructions. All conclusions below come from independent inspection and execution of the ZIP contents.

Supplied paths:

- `C:\Users\ashab\Downloads\TurtleBot_DiscreteSAC_Random_v1.0.1_ARC_Source_Corrected.zip`
- `C:\Users\ashab\Downloads\TurtleBot_DiscreteSAC_Random_v1.0.1_CORRECTION_AND_VERIFICATION_REPORT.md`
- `C:\Users\ashab\Downloads\TurtleBot_DiscreteSAC_Random_v1.0.1_CORRECTION_AND_VERIFICATION_REPORT (1).md`

The third path, the report name ending in `(1).md`, was not present on disk during this audit. The first report and ZIP were available and readable, so the missing apparent duplicate did not prevent source verification.

This audit covered:

- ZIP safety, structure, file modes, line endings, internal manifests, and frozen scenario regeneration;
- package identity, configuration, algorithm/source isolation, ROS packaging, ARC/Slurm wrappers, Apptainer definition, validation, and fail-closed behavior;
- a clean ROS 2 Foxy `colcon` build;
- the full automated suite with the pinned scientific stack;
- independent Discrete SAC numerical, target-update, randomness, and checkpoint probes;
- exact integer parsing at signed 64-bit boundaries and malformed-input rejection;
- a live headless Gazebo simulation, using the package's launcher and readiness gate;
- live robot relocation, simulator services, sensor/topic liveness, analytic obstacle tracking across reversals, collision reporting, runtime-input hashing, and shutdown cleanup.

It did **not** claim to reproduce an ARC cluster run, build an Apptainer image, exercise Slurm, complete a 10,000-transition calibration, or complete the controlled 500,000-transition training array.

## 2. Chain of custody and package identity

| Item | Independent result |
|---|---|
| ZIP size | 216,944 bytes |
| ZIP SHA-256 | `cbaf9b187202ad96d3dd537e72a6eb3a25d456ad7205cfc6606a268efd9eb033` |
| Available supplied report SHA-256 | `1cd9dae948bed40d68bff4a294be44bdfbfad9dac126e5e807eb2ef72ebe0d3c` |
| ZIP inventory | 107 entries: 97 files and 10 directories |
| Unsafe traversal/absolute paths | 0 |
| Duplicate archive paths | 0 |
| Archive links | 0 |
| `ALGORITHM` | `DiscreteSAC` |
| `VERSION` | `1.0.1` |
| Canonical configuration digest | `24a45ff8010a036996d767ca2374faac1eec29dddb04cc0ea8914f55aa799a28` |
| Raw `config/phase1_discretesac.yaml` SHA-256 | `b52098ab26be6ec2b0dba06e54d9290d76f8cf22cbf79ee6b90afe6b6b236092` |
| Release-manifest file SHA-256 | `2d98b5c08069f51cf229b17658988703f204ada959092e38c4470bd64d38e3f4` |
| Shared-layer-manifest file SHA-256 | `a17a45dbeee2504288f8818e88f7521c0877f7d196c52f45524a97e591d0fa3b` |

The package verifier passed all of its gates:

- release manifest;
- shared-layer manifest;
- forbidden-file scan;
- LF line endings;
- shell syntax;
- executable entry-point modes;
- Python compilation;
- frozen evaluation-scenario regeneration;
- canonical configuration digest.

The ZIP preserved executable modes for the package shell/Python utilities and ARC entry points when extracted natively in Linux. No cache, bytecode, result, checkpoint, or other prohibited artifact was included.

## 3. Correction closure against v1.0.0

| Prior finding | v1.0.1 implementation | Independent result |
|---|---|---|
| Integer fields were parsed through binary floating point | `_i` now requires the regex `[+-]?[0-9]+` and calls `int(value, 10)` directly | **Closed.** Signed 64-bit extrema round-trip exactly; `1.0`, `1e3`, surrounding whitespace, `nan`, `inf`, `+`, and `--1` are rejected. |
| Dynamic obstacles were displaced by physics/contact and violated the 0.05 m law | Both dynamic links declare `<kinematic>true</kinematic>` and `<gravity>false</gravity>` while retaining collision geometry | **Closed locally.** Worst live tracking error was 0.01034 m; a real robot/obstacle contact was still published. |
| Foxy-incompatible `ros2 topic echo --once` readiness logic | `wait_for_sim.sh` delegates to a bounded `rclpy` subscriber in `wait_for_topics.py` | **Closed.** The packaged gate passed on Foxy and proved both message arrival and clock advancement. |
| Python 3.8-incompatible source test used `ast.unparse` | Tests use `ast.get_source_segment` | **Closed.** Full suite passed under Python 3.8.10. |
| One validator test fixture removed files needed by later tests | Fixture/test setup now keeps required canonical streams and step-zero data consistent | **Closed.** Validator tests pass in the complete suite. |
| Host scientific packages could mix with pinned Matplotlib/mpl_toolkits | Image creates `/opt/tb3-python`, exports it first, sets `PYTHONNOUSERSITE=1`, and verifies module origins including pyplot and mplot3d | **Closed by design and locally reproduced.** Every pinned scientific module loaded from one isolated prefix. |
| Unrelated pytest plugins contaminated collection | `run_tests.sh` sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before pytest | **Closed.** Deterministic collection/execution passed. |
| ROS 2 Foxy silently mishandled workspace/runtime paths containing whitespace | Preflight, training/evaluation wrappers, and simulator launcher reject unsafe paths before startup | **Closed fail-closed.** Valid no-space preflight passed; whitespace input was rejected. |
| Simulator runtime inputs were insufficiently explicit | The launcher copies and compares the authenticated world, generates the contact-enabled model, and records both SHA-256 values | **Closed.** The source and runtime world hashes matched byte-for-byte. |

Additional consistency changes in the release—step-zero harness behavior and analysis-fixture construction—are covered by the now-passing orchestrator, validator, analysis, and source tests.

## 4. Discrete SAC algorithm review

The package contains a vanilla categorical Discrete SAC implementation for five discrete actions. It does not silently include Rainbow, prioritized replay, n-step returns, distributional critics, automatic temperature tuning, or an entropy-penalty critic.

The reviewed design is internally consistent:

- actor: 41 → 256 → 256 → 5 logits;
- two independently initialized critics, each mapping 41 observations to five Q-values;
- target value: exact sum over all five next-action probabilities using the minimum target critic and fixed entropy temperature;
- critic targets detach the next-state computation;
- actor loss is the exact categorical expectation using the minimum detached online critic;
- `alpha=0.2` is fixed;
- replay capacity 100,000; batch size 64; random warm-up 5,000 transitions;
- one actor and two critic updates per eligible environment transition;
- target critics hard-copy every 1,000 gradient updates;
- only true termination masks bootstrap; timeout and exact-budget truncations preserve the final next state and bootstrap;
- deterministic evaluation uses argmax; stochastic evaluation is seeded and separated from training randomness;
- full checkpoints retain actor, critics, targets, optimizers, replay, counters, and RNG state; policy-only checkpoints support evaluation.

An independent executable probe passed all 15 checks, including:

- exact soft-value and actor-objective calculations;
- use of the minimum twin critic;
- termination-only bootstrap masking (`[2.98, 1.0]` in the controlled example);
- independent critic initialization and correct initial targets;
- frozen target gradients;
- random warm-up behavior;
- no update before warm-up;
- first update at the declared boundary;
- no premature target sync and exact sync at the configured period;
- finite diagnostics;
- full checkpoint round-trip;
- reproducible deterministic and seed-controlled stochastic policy actions.

## 5. Environment, randomization, and study protocol

The shared environment and algorithm configuration are separated. The canonical configuration digest covers both. Important frozen values include:

- five discrete motion commands;
- 0.10 s control period;
- 500-step episode limit;
- physical `/bumper_states` collision input, required rather than optional;
- random starts uniformly proposed on `[-2,2] × [-2,2]`, subject to clearance and goal exclusions;
- transactional reset with pose, odometry, contact, sensor-age, and obstacle-position tolerances;
- two dynamic obstacles at 0.22 m/s with 5.0 s half-periods;
- separate world, initialization, dynamic-obstacle, evaluation, and learning seed roles;
- E1 online evaluation, 100 frozen E2 held-out scenarios, and 20 E3 anchor episodes;
- calibration/pilot budgets of 10,000 transitions and controlled budget of 500,000 transitions;
- controlled learning seeds 101, 202, 303, 404, and 505;
- controlled policy checkpoints every 25,000 transitions and full checkpoints every 100,000 transitions.

The validator checks run identity and provenance, stream schemas, step and episode contiguity, exact budgets, terminal/truncation semantics, action mapping, reward reconstruction, initial-state reproducibility, obstacle phases, update cadence, target synchronization, checkpoint schedules/digests, evaluation coverage, diagnostics, and recomputed summaries. Its fail-closed top-level handler writes a durable failure report for malformed run data.

## 6. ARC/Slurm/Apptainer compatibility review

### Slurm wrappers

The training and evaluation entry points are present, executable, and shell-valid. They provide:

- explicit job arrays with no seed-by-modulo identity;
- one node/task, CPU and memory requests, exclusive allocation, wall times, warning signals, and per-task logs;
- explicit phase selection and bounds-checked seed lookup from the frozen protocol;
- clean Apptainer environments with only declared variables forwarded;
- distinct per-task workspaces, run paths, ROS domains, Gazebo master ports, and lease cleanup;
- direct hashing of exact SIF bytes;
- failure/interruption/complete markers and signal forwarding;
- post-run validation and artifact-integrity generation before `COMPLETE`.

Site-specific `--account` and `--partition` are intentionally not frozen in the files and must be supplied according to the ARC allocation. The requested `--exclusive`, 24 GiB/48 h training, and 16 GiB/12 h evaluation resources must also be acceptable on the chosen ARC partition.

### Container definition

The definition targets ROS 2 Foxy and Gazebo 11, installs TurtleBot3 simulation dependencies, creates an isolated Python 3.8 environment, pins the principal scientific package versions, records the resolved environment, and contains a `%test` that imports ROS/Gazebo modules plus the full plotting/scientific stack and verifies module origins.

The local reproduction used:

- Ubuntu 20.04 under WSL;
- ROS 2 Foxy from `/opt/ros/foxy`;
- Gazebo 11.11.0;
- Python 3.8.10;
- PyTorch 2.4.1+cpu;
- NumPy 1.24.4;
- PyYAML 6.0.2;
- pandas 2.0.3;
- Matplotlib 3.7.5;
- pytest 8.3.3.

Every scientific module, including `matplotlib.pyplot` and `mpl_toolkits.mplot3d`, resolved from the same isolated environment. ROS modules remained importable from the Foxy installation.

Apptainer/Singularity and Slurm commands were unavailable on this host. Consequently, the actual `.def` build, `%test`, clean-environment execution, SIF digest, Slurm signal behavior, cgroup/resource behavior, and ARC filesystem/network behavior remain ARC-only gates. Foxy is end-of-life and the definition downloads a current ROS key and packages during build, so the ARC build node also needs compatible outbound access/repositories or a site-supported mirror/cache.

## 7. Build and automated test results

| Check | Result |
|---|---|
| Native Linux package verifier | **PASS** |
| ROS 2 Foxy `colcon build --symlink-install` | **PASS**, 1 package built |
| Initial suite in a reused non-isolated dependency directory | 162 passed, 1 failed due to host Matplotlib namespace mixing |
| Final suite in a clean isolated pinned environment | **PASS, 163/163**, 108.86 s |
| Independent Discrete SAC numerical/checkpoint probe | **PASS, 15/15** |
| Independent exact-integer parser probe | **PASS**, 2 signed-64-bit round trips and 8 malformed forms rejected |
| ARC preflight with valid no-space paths | **PASS** |
| ARC preflight with whitespace in run path | **PASS (expected rejection)** |

The first suite run is retained as diagnostic evidence: it reproduces why the v1.0.1 isolated runtime correction is necessary. It is not counted as a package failure because the final run used the runtime model specified by the release and passed all tests. The supplied correction report had only run the ROS-free subset (124 passed, 5 skipped because PyTorch was absent); this audit closes that local test gap.

## 8. Live local ROS/Gazebo simulation

### Launch and readiness

The source was cleanly built and launched with `scripts/run_sim_headless.sh` in isolated ROS domain 81 and a dedicated Gazebo master. The launcher:

- used world seed 7000;
- copied the package world into a whitespace-free runtime directory;
- generated a Burger SDF with the physical contact sensor;
- spawned the robot successfully;
- started the package dynamic-obstacle controller.

The package's own `scripts/wait_for_sim.sh 60` passed. It observed messages on all required topics:

- `/clock`;
- `/scan`;
- `/odom`;
- `/bumper_states`;
- `/dynamic_obstacle_1/odom`;
- `/dynamic_obstacle_2/odom`;
- `/drl/obstacle_status`.

It also found all required services: `/reset_world`, `/set_entity_state`, `/get_entity_state`, `/pause_physics`, and `/unpause_physics`. The readiness helper observed `/clock` advance from 23,598,000,000 ns to 24,102,000,000 ns.

### Independent live trajectory/reset probe

The independent probe reset the world, held the obstacles, moved the Burger to `(-1.5, -1.5, yaw=0.25)`, allowed the reset to settle, queried realized states, commanded both obstacle schedules, and followed them for 10.262 simulated seconds.

| Measurement | Observed | Required | Result |
|---|---:|---:|---|
| Burger relocation error | 0.0000298306 m | ≤ 0.03 m | **PASS** |
| Dynamic obstacle 1 maximum path error | 0.0103400 m | ≤ 0.05 m | **PASS** |
| Dynamic obstacle 2 maximum path error | 0.0103400 m | ≤ 0.05 m | **PASS** |
| Tracking samples | 514 per obstacle | non-zero, spanning target | **PASS** |
| 5 s reversal boundary | observed for both | required | **PASS** |
| 10 s reversal boundary | observed for both | required | **PASS** |

Representative message counts during the probe were 108 clock, 215 scan, 315 robot odometry, 538 bumper, 538 odometry per obstacle, and 5 obstacle-status messages.

The 0.01034 m maximum is approximately 20.7% of the frozen 0.05 m limit and is dramatically below the v1.0.0 live errors that motivated the correction.

### Physical-contact probe

After resetting and pausing the world, the probe queried `dynamic_obstacle_1`, placed the Burger into its collision volume, unpaused physics, and required a named contact pair. Gazebo published:

`burger::base_link::base_collision` ↔ `dynamic_obstacle_1::link::dynamic_collision`

This passed and demonstrates that making the obstacle link kinematic fixed trajectory fidelity without removing collision geometry or bumper reporting.

### Runtime-input identity and cleanup

| Runtime artifact | SHA-256 |
|---|---|
| Authenticated source world | `ebd5861904d6de326e262367d5cefa2ab2e6172ecbc3e1745a513a4717ba04f4` |
| Runtime world copy | `ebd5861904d6de326e262367d5cefa2ab2e6172ecbc3e1745a513a4717ba04f4` |
| Generated contact-enabled Burger SDF | `d4b1f73f13995680ff6af248fa1fc2114ee6df867d89e3fa2688b7e0db6e6424` |

The source and runtime world were byte-identical. After the probes, the launcher was interrupted and its cleanup removed the tested Gazebo server and dynamic-obstacle controller; the final process check found no matching simulator/controller.

The live probe exercises the shared simulator, reset, timing, obstacle, service, topic, and collision layers. It does not constitute a full real-Gazebo 10,000-transition Discrete SAC training run. Learner update/serialization behavior was instead exercised by the full automated suite and independent numerical probe.

## 9. Findings and discrepancies

### Source blockers

**None found in the tested v1.0.1 source.**

### Documentation/provenance discrepancy

The supplied correction report records a “supplied independent-audit SHA-256” of:

`bf9cf93ac65b0f1f3d4db484b2c68c47e452874b83b05fc5e9f101b6af089ac4`

The locally retained v1.0.0 independent report currently hashes to:

`ff3b1924d7a1d048c50ee7de2a03fe05ea09c78ad134ef560c6a7a1cf5153be0`

Therefore the provenance claim cannot be authenticated against the locally available v1.0.0 audit file. This may simply mean the correction report refers to an earlier/different report artifact, but the exact referenced artifact was not supplied here. It does not alter the independently verified v1.0.1 source behavior, but it should be reconciled before archival/freeze if the audit hash is part of the release provenance record.

### Missing duplicate report path

`TurtleBot_DiscreteSAC_Random_v1.0.1_CORRECTION_AND_VERIFICATION_REPORT (1).md` was absent. Because it appears to be a duplicate-name download and the source ZIP plus primary report were present, this is informational rather than a package blocker.

### Residual operational risks

- The real SIF has not yet been built or tested; external key/package repositories and the unpinned base-image tag can affect future builds even though principal Python versions are pinned.
- ARC site policies may require different partition/account/resource declarations or a local container cache/mirror.
- A WSL Gazebo run is strong compatibility evidence but is not proof of ARC kernel, filesystem, network namespace, or Slurm behavior.
- Long-horizon learning quality, convergence, and performance are not established by a smoke simulation or unit suite; they require calibration/pilot results.

## 10. Release decision and required ARC sequence

### Good to upload/stage on ARC?

**Yes.** This ZIP is suitable to place in a whitespace-free ARC source directory and use for the next validation stage.

### Good to start a calibration job?

**Yes, conditionally**, after the exact SIF builds and its `%test` passes on a compute/build node.

### Good to call frozen or launch the controlled five-seed study now?

**No.** The source is locally verified, but the exact SIF and an ARC calibration run are still missing.

Recommended sequence:

1. Put the ZIP in a whitespace-free ARC project path and independently confirm SHA-256 `cbaf9b187202ad96d3dd537e72a6eb3a25d456ad7205cfc6606a268efd9eb033`.
2. Extract natively on Linux and run `scripts/verify_package.sh`.
3. Build `discretesac_random_v1.0.1.sif` on an allowed compute/build node.
4. Run `apptainer test` and retain its output.
5. Hash the exact SIF and retain the `.sif.sha256` sidecar; do not substitute a sidecar for hashing the actual image at dispatch.
6. Submit the single-seed calibration phase through `arc/discretesac_array.sbatch`, supplying the site account/partition and package image/output paths.
7. Require a `COMPLETE` marker, passing validator report, artifact-integrity verification, runtime manifests, clean logs, and no fatal markers.
8. Review reset acceptance, timing/sensor ages, obstacle errors, collision classifications, throughput, memory, and walltime.
9. Only then freeze the exact source/SIF pair and advance to pilot, followed by the preregistered controlled array.

## 11. Evidence index

Package audit evidence is under `audit_DiscreteSAC_v1.0.1/evidence/`:

- `verify_package.log`
- `executable_modes.log`
- `colcon_build.log`
- `identity_digests.log`
- `isolated_runtime_origins.log`
- `test_suite_exact_console.log` (diagnostic non-isolated run)
- `test_suite_isolated_console.log` and `test_suite_isolated.log`
- `independent_discretesac_probe.json`
- `validator_integer_probe.json`
- `preflight_checks.log`
- `platform_capabilities.log`
- `ros_gazebo_capabilities.log`

Live simulation evidence is under `local_sim_DiscreteSAC_v1.0.1/`:

- `gazebo.log`
- `packaged_readiness.log`
- `live_smoke_result.json`
- `live_collision_result.json`
- `runtime_hashes.log`
- `shutdown_check.log`

## Final conclusion

**TurtleBot_DiscreteSAC_Random v1.0.1 is locally verified and good to proceed to ARC SIF build and calibration.** The v1.0.0 technical blockers tested here are closed, all 163 tests pass in the corrected isolated runtime, and the live simulator meets the frozen 0.05 m obstacle tolerance while retaining physical collision reporting. Do not label the release frozen or begin the controlled five-seed experiment until the exact ARC SIF and a validated ARC calibration run complete successfully.
