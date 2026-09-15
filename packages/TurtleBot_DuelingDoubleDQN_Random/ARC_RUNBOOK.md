# ARC runbook — TurtleBot_DuelingDoubleDQN_Random 1.0.2

Training and evaluation run as Slurm batch jobs on ARC. Run tests, Gazebo,
training, and analysis on allocated compute nodes.

## 1. Suggested layout

```text
/projects/<allocation>/tb3/
  packages/TurtleBot_DuelingDoubleDQN_Random/
  images/TurtleBot_DuelingDoubleDQN_Random_v1.0.2.sif
  results/
```


## 2. Build and preserve the runtime image

On an allocated compute node:

```bash
module load apptainer
cd /projects/<allocation>/tb3/packages/TurtleBot_DuelingDoubleDQN_Random
apptainer build /projects/<allocation>/tb3/images/TurtleBot_DuelingDoubleDQN_Random_v1.0.2.sif \
  apptainer/tb3_phase1_foxy.def
sha256sum /projects/<allocation>/tb3/images/TurtleBot_DuelingDoubleDQN_Random_v1.0.2.sif \
  > /projects/<allocation>/tb3/images/TurtleBot_DuelingDoubleDQN_Random_v1.0.2.sif.sha256
apptainer test /projects/<allocation>/tb3/images/TurtleBot_DuelingDoubleDQN_Random_v1.0.2.sif
```

Preserve the exact SIF used for the campaign. Every task recomputes its bytes'
checksum; a sidecar is informative, not authoritative.
The image contains the runtime for this standalone Dueling Double DQN package. Preserve
its exact bytes and use its recorded digest for every Dueling Double DQN training and
evaluation task. No other algorithm package or result directory is required.

## 3. Package check and mandatory container tests

The following lightweight check may run on the login node:

```bash
cd /projects/<allocation>/tb3/packages/TurtleBot_DuelingDoubleDQN_Random
bash scripts/verify_package.sh
```

It verifies the package inventory and shared files, checks links and line endings,
validates shell/Python syntax, and reproduces the E2 scenario list.

Then allocate a compute node and run every test inside the exact SIF:

```bash
module load apptainer
apptainer exec --cleanenv \
  --bind /projects/<allocation>/tb3:/projects/<allocation>/tb3 \
  /projects/<allocation>/tb3/images/TurtleBot_DuelingDoubleDQN_Random_v1.0.2.sif \
  bash -c 'cd /projects/<allocation>/tb3/packages/TurtleBot_DuelingDoubleDQN_Random && bash scripts/run_tests.sh'
```

`scripts/run_tests.sh` treats any skipped test as failure. For this source
release the required result is **154 passed, zero failed, zero errors and zero
skipped**. Do not submit a calibration job until it is fully green.

Before calibration, repeat the independent Foxy/Gazebo reset, relocation and
dynamic-obstacle probe for at least 10 simulation seconds inside this exact
SIF. Both obstacle errors must remain at or below 0.05 m, including the 5 s and
10 s reversal boundaries. Retain the raw simulation-time samples.

## 4. Submission environment

```bash
export TB3_REPO=/projects/<allocation>/tb3/packages/TurtleBot_DuelingDoubleDQN_Random
export TB3_IMAGE=/projects/<allocation>/tb3/images/TurtleBot_DuelingDoubleDQN_Random_v1.0.2.sif
export TB3_OUTPUT=/projects/<allocation>/tb3/results
cd "$TB3_REPO"
```

Add the ARC-specific account and partition options to every `sbatch` command.
The array scripts request an exclusive node, hash the SIF, explicitly forward
Slurm identity through `--cleanenv`, and lease non-modulo ROS/Gazebo transport
identities.

## 5. Calibration stop gate

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=calibration --array=0 arc/duelingdoubledqn_array.sbatch
```

This performs one 10,000-transition training run with checkpoints at 5k and
10k. Record the returned array job ID as `TRAIN_JOB`, then run final E2/E3:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=calibration,TRAIN_JOB=<training-array-id> \
  --array=0 arc/duelingdoubledqn_eval_array.sbatch
```

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

The calibration evidence must also close the live random-initialization gate:
episode 1 and every subsequent training episode must have one step-zero row;
the requested start and obstacle phase must regenerate from the recorded
role-specific seeds and episode index; reset and relocation must be
acknowledged; realized and odometric poses must satisfy their tolerances; and
the step-zero row must precede the first action. A default-pose fallback or a
missing initialization record is a calibration failure.

## 6. Pilot

After calibration passes without changing code, configuration or image:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=pilot --array=0-1 arc/duelingdoubledqn_array.sbatch
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=pilot,TRAIN_JOB=<training-array-id> \
  --array=0-1 arc/duelingdoubledqn_eval_array.sbatch
```

Both 10k learning seeds must pass the same completion checks. Run directories
are create-only; never rerun into an old job directory.

## 7. Controlled training

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=controlled --array=0-4 arc/duelingdoubledqn_array.sbatch
```

This launches the configured learning seeds 101, 202, 303, 404 and 505. Each
run has exactly 500,000 training transitions, policy checkpoints every 25k,
full checkpoints every 100k and at the budget, and 20 frozen E1 episodes at
every policy checkpoint. E1 does not consume the training budget.

Monitor without altering the run:

```bash
tail -f "$TB3_OUTPUT/controlled/duelingdoubledqn/seed_101/job_<id>/logs/training_console.log"
python3 -m json.tool "$TB3_OUTPUT/controlled/duelingdoubledqn/seed_101/job_<id>/progress.json"
```

A ten-minute Slurm warning is forwarded to the agent. It writes an emergency
full checkpoint and the run ends `INTERRUPTED`, never `COMPLETE`.

## 8. Configured controlled evaluation

Run all five tier-2 checkpoints; each command launches all five learning seeds
and evaluates 100 E2 scenarios plus 20 E3 anchors:

```bash
for STEP in 100000 200000 300000 400000 500000; do
  sbatch --account=<acct> --partition=<part> \
    --export=ALL,PHASE_LABEL=controlled,TRAIN_JOB=<training-array-id>,CHECKPOINT_STEP="$STEP" \
    --array=0-4 arc/duelingdoubledqn_eval_array.sbatch
done
```

The dispatcher rejects a step not listed in
`evaluation_protocol.phases.controlled.tier2_checkpoint_steps`. The evaluation
wrapper also rejects an unvalidated training run, changed package/image, ambiguous
checkpoint row, path outside `checkpoints/`, or digest mismatch.

The 500k E2/E3 runs feed final tables. All five checkpoints feed the
checkpoint-wise held-out figure.

## 9. Tables and figures

Run on a compute node inside the preserved Dueling Double DQN SIF. Point `--results`
only at the standalone Dueling Double DQN controlled-result root.

```bash
apptainer exec --cleanenv \
  --bind /projects/<allocation>/tb3:/projects/<allocation>/tb3 \
  "$TB3_IMAGE" bash -c '
    cd '"$TB3_REPO"' &&
    python3 scripts/make_tables.py \
      --results '"$TB3_OUTPUT"'/controlled/duelingdoubledqn \
      --out '"$TB3_OUTPUT"'/analysis/duelingdoubledqn/tables \
      --phase-label controlled \
      --expected-algorithms DuelingDoubleDQN &&
    python3 scripts/make_figures.py \
      --results '"$TB3_OUTPUT"'/controlled/duelingdoubledqn \
      --out '"$TB3_OUTPUT"'/analysis/duelingdoubledqn/figures \
      --phase-label controlled --formats pdf,png --style ieee \
      --expected-algorithms DuelingDoubleDQN'
```

Tables are written as CSV, Markdown and LaTeX booktabs. Figures are vector PDF
and 300-dpi PNG at IEEE column sizes. Complete-run discovery rechecks validation
and run-file integrity; `--include-incomplete` is diagnostic only and must not be
used for reported results.

With `--expected-algorithms DuelingDoubleDQN`, analysis fails unless Dueling Double DQN has
exactly one training run for every configured seed and one evaluation run for
every configured seed/checkpoint. It also requires common randomization
seeds, one shared-layer digest, one container digest, and an explicit
evaluation-to-training link. The passing matrix is preserved as
`campaign_completeness.json` beside the tables.

## 10. Runtime validation

Package tests cover update equations, episode ordering, recording, and validation.
Use the container tests, simulation calibration, and pilot runs to check the
ROS/Gazebo, Apptainer, and Slurm setup. Keep the package, image, logs, and run
directories with the experiment records.

Package and image integrity are checked with SHA-256 manifests.
