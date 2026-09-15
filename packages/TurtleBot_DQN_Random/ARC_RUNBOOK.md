# ARC runbook — TurtleBot_DQN_Random 1.1.7

Training and evaluation run as Slurm batch jobs on ARC. Run tests, Gazebo,
training, and analysis on allocated compute nodes.

## 1. Suggested layout

```text
/projects/<allocation>/tb3/dqn/
  package/TurtleBot_DQN_Random/
  image/TurtleBot_DQN_Random_Runtime_Foxy.sif
  results/
  slurm-logs/
```

The package, image, results and Slurm-log paths must be whitespace-free and
must not be nested inside the verified package directory. The supported
submission wrapper enforces this layout.

## 2. Build and preserve the runtime image

On an allocated compute node:

```bash
module load apptainer
cd /projects/<allocation>/tb3/dqn/package/TurtleBot_DQN_Random
bash scripts/verify_package.sh
bash scripts/build_sif.sh \
  /projects/<allocation>/tb3/dqn/image/TurtleBot_DQN_Random_Runtime_Foxy.sif
apptainer test /projects/<allocation>/tb3/dqn/image/TurtleBot_DQN_Random_Runtime_Foxy.sif
chmod a-w /projects/<allocation>/tb3/dqn/image/TurtleBot_DQN_Random_Runtime_Foxy.sif
sha256sum /projects/<allocation>/tb3/dqn/image/TurtleBot_DQN_Random_Runtime_Foxy.sif \
  > /projects/<allocation>/tb3/dqn/image/TurtleBot_DQN_Random_Runtime_Foxy.sif.sha256
```

Use `scripts/build_sif.sh` to build the image from the files listed in
`RELEASE_MANIFEST.sha256`. The script checks the source inventory, creates a
temporary build tree, and verifies the copy embedded in the image. Keep the
resulting SIF read-only for the campaign. Each task verifies the image checksum
before running its packaged ROS/Gazebo/Python environment and DQN source.

## 3. Package check and mandatory container tests

The following lightweight check may run on the login node:

```bash
cd /projects/<allocation>/tb3/dqn/package/TurtleBot_DQN_Random
bash scripts/verify_package.sh
```

It verifies the exact package inventory and shared layer; rejects `.pyc`,
`.pyo`, `__pycache__`, `.pytest_cache`, `.git`, `build`, `install`, `log`,
`*.egg-info`, and symbolic links; checks LF and shell/Python syntax; and
reproduces the E2 scenario list. Forbidden paths cause failure before any
package module can be imported.

The `apptainer test` command above verifies the embedded release and runs
the complete suite. To capture a separate test log on the compute node, run:

```bash
module load apptainer
apptainer exec --cleanenv \
  /projects/<allocation>/tb3/dqn/image/TurtleBot_DQN_Random_Runtime_Foxy.sif \
  bash /opt/turtlebot_dqn_random/scripts/run_tests.sh \
  | tee /projects/<allocation>/tb3/dqn/slurm-logs/v1.1.7-tests.log
```

`scripts/run_tests.sh` requires every test to run and pass before calibration.

Before calibration, run the live Foxy/Gazebo obstacle probe for at least 10
simulation seconds, covering two 5-second half-periods. Retain the requested
schedule, simulation-time samples, measured
odometry and independent analytic displacements. The maximum error of each
obstacle must remain at or below 0.05 m, including both reversal boundaries.
This live check is mandatory because a static world-file test cannot certify a
Gazebo physics/plugin interaction.

## 4. Submission environment

```bash
export TB3_REPO=/projects/<allocation>/tb3/dqn/package/TurtleBot_DQN_Random
export TB3_IMAGE=/projects/<allocation>/tb3/dqn/image/TurtleBot_DQN_Random_Runtime_Foxy.sif
export TB3_OUTPUT=/projects/<allocation>/tb3/dqn/results
export TB3_SLURM_LOGS=/projects/<allocation>/tb3/dqn/slurm-logs
export TB3_LEASES="$TB3_OUTPUT/.leases"
```

Do not run `sbatch` from `TB3_REPO`. Use `arc/submit_dqn.sh` from any directory
outside the package. Package, image, result, lease and Slurm-log paths must use
only letters, digits, `/`, `.`, `_`, `+`, `@` and `-`; this literal-safe
grammar prevents shell/container evaluation, and percent substitution is
additionally forbidden in every resolved log path.
When supplied, `TB3_LEASES` must resolve to `$TB3_OUTPUT/.leases`.
Both the login-node wrapper and allocated
batch script enforce this after canonicalization.

The wrapper runs the trusted verifier from the SIF against both the embedded
release and the external extraction, which is mounted only at a fixed read-only
inspection path. Their release digests must agree. It records the SIF and
embedded-release checksum values, derives the phase's frozen array range, copies
the selected batch script to a read-only snapshot whose expected digest comes
from the manifest inside the verified SIF, supplies
absolute Slurm-log paths, and accepts only exact attached-value site options:
`--account=`,
`--partition=`, `--qos=`, `--reservation=` and `--constraint=`. It refuses all
other caller tokens, including abbreviations, short options, `--`, `:`, and
alternate batch-script operands. The wrapper owns `--array`, `--export`,
`--chdir`, `--output`, `--error`, `--dependency` and the final script.

The Slurm client is invoked with a small, explicit host environment and the job
uses a named export allowlist without `ALL`. Thus caller values cannot override
the wrapper's canonical `TB3_*` values or inject `BASH_ENV`,
`LD_PRELOAD`, Slurm option variables, or container controls. Before any
Apptainer call, all `APPTAINER_*`, `APPTAINERENV_*`,
`SINGULARITY_*` and `SINGULARITYENV_*` names are removed. Only
the explicitly enumerated package-owned `APPTAINERENV_*` values are created
for the final scientific process. Both batch scripts verify their Slurm-spooled
bytes, the current SIF digest, and the embedded release before scientific work.
They never mount `TB3_REPO`; only `TB3_OUTPUT` is writable inside the
container. They request an exclusive node, forward Slurm identity explicitly,
and lease non-modulo ROS/Gazebo transport identities.

If an earlier submission wrote `.out`, `.err`, results, reports or other files
inside `TB3_REPO`, preserve them outside the package and then run
`bash "$TB3_REPO/scripts/verify_package.sh"`. Do not regenerate the release
manifest to bless runtime files or unreviewed edits.

## 5. Calibration stop gate

```bash
bash "$TB3_REPO/arc/submit_dqn.sh" train calibration \
  --account=<acct> --partition=<part>
```

This performs one 10,000-transition training run with checkpoints at 5k and
10k. Record the returned array job ID as `TRAIN_JOB`, then submit final E2/E3:

```bash
bash "$TB3_REPO/arc/submit_dqn.sh" eval calibration <training-array-id> \
  --account=<acct> --partition=<part>
```

The evaluation submission carries wrapper-owned
`--dependency=afterok:<training-array-id>` and
`--kill-on-invalid-dep=yes`. It remains pending without consuming an evaluation
allocation, becomes eligible only after every training-array task succeeds, and
is cancelled if that prerequisite can never be satisfied. The same rule applies
to pilot and controlled evaluation.

Inspect both directories:

```bash
python3 -m turtlebot3_drl_nav.artifact_integrity verify <run_dir>
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r); raise SystemExit(0 if r["passed"] else 1)' \
  <run_dir>/validation_report.json
```

Required result: `COMPLETE` exists; neither `FAILED`, `INTERRUPTED`,
`ENV_FATAL` nor `AGENT_FATAL` exists; validation passed; recorded inventory
passed; action holds, sensor age, policy-time simulation drift and obstacle error
are within the frozen limits. This is the first real ROS/Gazebo-on-ARC evidence.
Stop and version a correction if it fails.

## 6. Pilot

After calibration passes without changing code, configuration or image:

```bash
bash "$TB3_REPO/arc/submit_dqn.sh" train pilot \
  --account=<acct> --partition=<part>
bash "$TB3_REPO/arc/submit_dqn.sh" eval pilot <training-array-id> \
  --account=<acct> --partition=<part>
```

Both 10k learning seeds must pass the same completion checks. Run directories
are create-only; never rerun into an old job directory.

## 7. Controlled training

```bash
bash "$TB3_REPO/arc/submit_dqn.sh" train controlled \
  --account=<acct> --partition=<part>
```

This launches the configured learning seeds 101, 202, 303, 404 and 505. Each
run has exactly 500,000 training transitions, policy checkpoints every 25k,
full checkpoints every 100k and at the budget, and 20 frozen E1 episodes at
every policy checkpoint. E1 does not consume the training budget.

Monitor without altering the run:

```bash
tail -f "$TB3_OUTPUT/controlled/dqn/seed_101/job_<id>/logs/training_console.log"
python3 -m json.tool "$TB3_OUTPUT/controlled/dqn/seed_101/job_<id>/progress.json"
```

A ten-minute Slurm warning is forwarded to the agent. It writes an emergency
full checkpoint and the run ends `INTERRUPTED`, never `COMPLETE`.

## 8. Configured controlled evaluation

Run all five tier-2 checkpoints; each command launches all five learning seeds
and evaluates 100 E2 scenarios plus 20 E3 anchors:

```bash
for STEP in 100000 200000 300000 400000 500000; do
  bash "$TB3_REPO/arc/submit_dqn.sh" eval controlled <training-array-id> "$STEP" \
    --account=<acct> --partition=<part>
done
```

The dispatcher rejects a step not listed in
`evaluation_protocol.phases.controlled.tier2_checkpoint_steps`. The evaluation
wrapper also rejects an unvalidated training run, changed package/image, ambiguous
checkpoint row, path outside `checkpoints/`, or digest mismatch.

The 500k E2/E3 runs feed final tables. All five checkpoints feed the
checkpoint-wise held-out figure.

## 9. Tables and figures

Run on a compute node inside the preserved DQN SIF. Point `--results` at the
root containing only the validated DQN controlled-run directories. Analysis code
is taken from the same recorded in-image release used for training.

```bash
apptainer exec --cleanenv \
  --bind "$TB3_OUTPUT:$TB3_OUTPUT" \
  "$TB3_IMAGE" bash -c '
    python3 /opt/turtlebot_dqn_random/scripts/make_tables.py \
      --results '"$TB3_OUTPUT"'/controlled \
      --out '"$TB3_OUTPUT"'/analysis/dqn/tables \
      --phase-label controlled \
      --expected-algorithms DQN &&
    python3 /opt/turtlebot_dqn_random/scripts/make_figures.py \
      --results '"$TB3_OUTPUT"'/controlled \
      --out '"$TB3_OUTPUT"'/analysis/dqn/figures \
      --phase-label controlled --formats pdf,png --style ieee \
      --expected-algorithms DQN'
```

Tables are written as CSV, Markdown and LaTeX booktabs. Figures are vector PDF
and 300-dpi PNG at IEEE column sizes. Complete-run discovery rechecks validation
and run-file integrity; `--include-incomplete` is diagnostic only and must not be
used for reported results.

With `--expected-algorithms DQN`, analysis fails unless DQN has exactly one
training run for every configured seed and one evaluation run for every
configured seed/checkpoint. It also requires common randomization seeds, one
shared-layer digest, one container digest, and an explicit evaluation-to-
training link. The passing matrix is preserved as
`campaign_completeness.json` beside the tables.

## 10. Runtime validation

Package tests cover update equations, episode ordering, recording, and validation.
Use the container tests, simulation calibration, and pilot runs to check the
ROS/Gazebo, Apptainer, and Slurm setup. Keep the package, image, logs, and run
directories with the experiment records.

Package and image integrity are checked with SHA-256 manifests.
