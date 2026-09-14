# TurtleBot_DQN_Random 1.1.7

Standalone, DQN-only, direct-ARC package for TurtleBot3 Burger navigation under
the declared random robot-start distribution $\nu_R$. It uses ROS 2 Foxy,
Gazebo 11 Classic and CPU PyTorch. No notebook is included or required.

This package is evaluated on its own specification. It does not depend on, diff
against, or make a performance claim about a fixed-initialization package.

## What this package guarantees

- One learner only: DQN. There is no Double DQN, Dueling, Rainbow, SAC, PPO or
  A2C implementation in this archive.
- Episode 1 and every later training episode draw a robot pose from the same
  seeded, episode-indexed $\nu_R$ sampler.
- Start poses are rejected unless they satisfy the frozen arena, goal and swept
  dynamic-obstacle clearance rules.
- Reset, relocation, settling, physics pause, realized-state verification and
  obstacle-controller acknowledgement finish before the first policy action.
- Physics is paused during policy inference and optimization. Each action is
  applied for a measured simulation-time control interval, so network latency
  cannot shorten the command.
- Dynamic-obstacle motion is a seeded function of simulation time and is checked
  online and again by the offline validator.
- Training stops at the phase's exact transition budget, writes policy and full
  checkpoints on the frozen cadence, and evaluates every checkpoint on E1.
- A run receives `COMPLETE` only after the validator passes and `RUN_FILES.sha256`
  binds every output byte. Failed and interrupted runs are never analysis inputs.

These are code-level guarantees. ROS/Gazebo timing on the target cluster must
still pass the ARC container tests and calibration gate in `ARC_RUNBOOK.md`.

## Package map

| Path | Purpose |
|---|---|
| `ALGORITHM.md` | Short DQN description, equations, fixed values and diagnostics |
| `AUDIT_CORRECTIONS.md` | Traceable disposition of the v1.1.0–v1.1.6 ARC/local findings |
| `VERIFICATION_SCOPE.md` | Standalone scope used for DQN and later separate packages |
| `RANDOM_INIT_SPEC.md` | Exact $\nu_R$, seed roles, reset transaction and E1/E2/E3 |
| `DATA_CONTRACT.md` | Five CSV streams, provenance, validation and analysis products |
| `ARC_RUNBOOK.md` | Terminal-only ARC image, test, calibration, training and evaluation commands |
| `config/common_environment.yaml` | Shared POMDP, randomization, phase and tier-2 cadence authority |
| `config/phase1_dqn.yaml` | DQN-only hyperparameters |
| `config/evaluation_scenarios_v1.csv` | Frozen 100-scenario E2 list |
| `turtlebot3_drl_nav/dqn.py` | DQN network, replay, update, checkpoint and greedy policy |
| `turtlebot3_drl_nav/dqn_agent_node.py` | DQN ROS adapter |
| `turtlebot3_drl_nav/episode_engine.py` | Testable reset/step state machine |
| `turtlebot3_drl_nav/validator.py` | Fail-closed reconstruction and validation of a run |
| `turtlebot3_drl_nav/analysis.py` | Paper tables and figures from sealed runs |
| `arc/` | Separate DQN training and evaluation Slurm entry points |
| `arc/submit_dqn.sh` | Mandatory safe submission wrapper; keeps Slurm output outside the release |
| `scripts/materialize_release.py` | Copies only the authenticated manifest inventory into a clean build tree |
| `scripts/build_sif.sh` | Mandatory compute-node SIF builder using the clean materialized tree |
| `apptainer/tb3_phase1_foxy.def` | Runtime image recipe |

`SHARED_LAYER_FILES.txt` identifies and authenticates this archive's common
environment layer. Algorithm-specific DQN files are not in that list, and the
manifest creates no dependency on another algorithm package.
The Apptainer recipe embeds this complete authenticated DQN release and installs
it inside the SIF together with its pinned dependencies. Training and evaluation
execute only the in-image source at `/opt/turtlebot_dqn_random`; the external
extraction is not a runtime bind. The DQN SIF and result directories can
therefore be built, run and preserved independently of every other algorithm.

## Random-start and action timing sequence

1. Pause/hold dynamic obstacles and await their acknowledgement.
2. Await `/reset_world`, then await `/set_entity_state` for the sampled pose.
3. Unpause for 0.5 simulation seconds of settling, stop, pause and await pause.
4. Capture bounded-age scan/odometry/obstacle samples while paused.
5. Query and verify the realized robot and obstacle poses; reject contact,
   out-of-support starts, odometry disagreement and insufficient scan clearance.
6. Install and acknowledge the seeded obstacle schedule, then publish $o_0$.
7. Keep physics paused while DQN selects/updates; apply the command, unpause,
   stop at the control boundary, require boundary-fresh sensors, pause, capture
   $o_{t+1}$ and record the step.

## Direct ARC entry points

Set `TB3_REPO`, `TB3_IMAGE` and `TB3_OUTPUT` to separate ARC paths
using only the package's documented literal-safe character set. Submit only
through the package wrapper. It verifies the external extraction read-only with
the trusted in-image verifier, requires the embedded and external release
digests to agree, authenticates the read-only SIF, derives the frozen array
range, and submits a batch-script snapshot checked against the manifest inside
the authenticated SIF. The allocated job
uses only the source embedded in the authenticated SIF and writes Slurm logs
below `TB3_OUTPUT`. It receives an explicit environment allowlist—never the
caller's complete environment. The only caller-selectable Slurm options are
the exact attached-value forms
`--account=`, `--partition=`, `--qos=`, `--reservation=` and `--constraint=`:

```bash
bash "$TB3_REPO/arc/submit_dqn.sh" train calibration --account=<acct> --partition=<part>
bash "$TB3_REPO/arc/submit_dqn.sh" train pilot       --account=<acct> --partition=<part>
bash "$TB3_REPO/arc/submit_dqn.sh" train controlled  --account=<acct> --partition=<part>
```

Direct `sbatch arc/dqn_array.sbatch` calls are refused. In particular, never
submit from inside `TB3_REPO`: Slurm creates stdout/stderr before package
preflight and those runtime files invalidate the authenticated release tree.
Ambient `APPTAINER_*`, `APPTAINERENV_*`, `SINGULARITY_*` and
`SINGULARITYENV_*` variables are removed before authentication and cannot
add binds, environment values or runtime options to a queued job.
After building and testing the SIF, remove all of its write bits with
`chmod a-w "$TB3_IMAGE"`; the submission wrapper refuses a writable image.
`TB3_LEASES`, when explicitly set, must resolve exactly to
`$TB3_OUTPUT/.leases`; no second writable container bind is permitted.

Post-hoc E2/E3 evaluation is a separate DQN job. Its wrapper-owned
`afterok:<training-array-id>` dependency prevents evaluation from running until
the entire referenced training array succeeds; an unsatisfiable dependency is
cancelled instead of remaining pending. Controlled evaluation is preregistered
at 100k, 200k, 300k, 400k and 500k transitions; see the runbook.

## Data available for the paper

Every valid training run records `transitions.csv`, `episodes.csv`,
`updates.csv`, `evaluation.csv` and `checkpoints.csv`. Together they include
full robot/obstacle trajectories, initialization request and realization,
reward components, terminal masks, action timing, sensor age, obstacle tracking
error, DQN diagnostics, checkpoint identity and outcomes. The supplied scripts
produce a standalone DQN set of tables T-R1–T-R9 and figures F-R1–F-R14 from
the sealed DQN result directories; no other algorithm package is required.

## Verification boundary

```bash
bash scripts/verify_package.sh
```

This login-node-safe check verifies manifests, inventory, line endings, shell
syntax, Python compilation and the frozen scenario list. It rejects bytecode,
caches, notebooks, repository metadata and build/install/log products before
running package Python. Build the SIF only through `scripts/build_sif.sh`, which
materializes exactly the authenticated manifest inventory into a fresh build
tree. `apptainer test` verifies
the embedded package and runs the complete zero-skip suite; calibration remains
blocked until that succeeds on an ARC compute node.
