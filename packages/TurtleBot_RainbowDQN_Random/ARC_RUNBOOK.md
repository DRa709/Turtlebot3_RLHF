# ARC runbook — TurtleBot_RainbowDQN_Random 1.0.1

This is a direct-ARC, terminal-only procedure. Jupyter is not used. Run tests,
Gazebo, training, evaluation and analysis on allocated compute nodes rather than
the login node.

## 1. Independent ARC layout

Use a directory dedicated to this algorithm:

```text
/projects/<allocation>/tb3_rainbowdqn/
  package/TurtleBot_RainbowDQN_Random/
  image/rainbowdqn_random_v1.0.1.sif
  results/
```

Do not place another algorithm's source, checkpoints, runs or analysis products
under this directory. Replace `<allocation>` in every command below.

## 2. Build and authenticate this package's image

On an allocated compute node:

```bash
module load apptainer
export RAINBOW_ROOT=/projects/<allocation>/tb3_rainbowdqn
export TB3_REPO="$RAINBOW_ROOT/package/TurtleBot_RainbowDQN_Random"
export TB3_IMAGE="$RAINBOW_ROOT/image/rainbowdqn_random_v1.0.1.sif"
mkdir -p "$RAINBOW_ROOT/image"
cd "$TB3_REPO"
apptainer build "$TB3_IMAGE" apptainer/tb3_phase1_foxy.def
sha256sum "$TB3_IMAGE" > "$TB3_IMAGE.sha256"
apptainer test "$TB3_IMAGE"
```

Preserve the exact SIF bytes after testing. Every Slurm task independently
recomputes the image SHA-256.

## 3. Verify the package and execute every test

The static package check is:

```bash
cd "$TB3_REPO"
bash scripts/verify_package.sh
```

Then execute the complete test suite inside the exact SIF:

```bash
apptainer exec --cleanenv \
  --bind "$RAINBOW_ROOT:$RAINBOW_ROOT" \
  "$TB3_IMAGE" \
  bash -c 'cd '"$TB3_REPO"' && bash scripts/run_tests.sh'
```

`scripts/run_tests.sh` fails if any test is skipped. Do not submit calibration
until both commands succeed.

## 4. Submission environment

```bash
export RAINBOW_ROOT=/projects/<allocation>/tb3_rainbowdqn
export TB3_REPO="$RAINBOW_ROOT/package/TurtleBot_RainbowDQN_Random"
export TB3_IMAGE="$RAINBOW_ROOT/image/rainbowdqn_random_v1.0.1.sif"
export TB3_OUTPUT="$RAINBOW_ROOT/results"
cd "$TB3_REPO"
```

Add the ARC allocation and partition to every `sbatch` command. Training and
evaluation tasks request exclusive nodes. Each task receives a unique workspace,
result directory and leased ROS/Gazebo transport identity.

## 5. Calibration gate

Submit one 10,000-transition training task:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=calibration \
  --array=0 arc/rainbowdqn_array.sbatch
```

Record the returned training array job ID. Then submit its E2/E3 evaluation:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=calibration,TRAIN_JOB=<training-array-id> \
  --array=0 arc/rainbowdqn_eval_array.sbatch
```

Both run directories must contain `COMPLETE`, a passing
`validation_report.json` and a valid `RUN_FILES.sha256`. None may contain
`FAILED`, `INTERRUPTED`, `ENV_FATAL` or `AGENT_FATAL`. Timing, sensor freshness,
initialization fidelity and obstacle tracking must remain inside their declared
limits. A failure stops the process.

## 6. Two-seed pilot

After calibration passes without changing package or image:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=pilot \
  --array=0-1 arc/rainbowdqn_array.sbatch
```

Use the returned array ID to evaluate both pilot seeds:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=pilot,TRAIN_JOB=<training-array-id> \
  --array=0-1 arc/rainbowdqn_eval_array.sbatch
```

Both training and both evaluation tasks must satisfy the calibration checks.
Run directories are create-only; never resubmit into an existing job directory.

## 7. Controlled training

Only after exact-image tests, calibration and pilot pass:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=controlled \
  --array=0-4 arc/rainbowdqn_array.sbatch
```

Array indices select learning seeds 101, 202, 303, 404 and 505 from the frozen
configuration. Each task performs exactly 500,000 training transitions, creates
policy checkpoints every 25,000 transitions, full checkpoints every 100,000
transitions and at the final budget, and evaluates 20 E1 episodes at every
policy checkpoint. E1 transitions do not consume the training budget.

Monitor without editing a run:

```bash
tail -f "$TB3_OUTPUT/controlled/rainbowdqn/seed_101/job_<id>/logs/training_console.log"
python3 -m json.tool \
  "$TB3_OUTPUT/controlled/rainbowdqn/seed_101/job_<id>/progress.json"
```

A Slurm walltime warning requests an atomic emergency checkpoint and produces
`INTERRUPTED`, never `COMPLETE`.

## 8. Controlled E2/E3 evaluation

For the controlled training array ID, submit the five declared checkpoints:

```bash
for STEP in 100000 200000 300000 400000 500000; do
  sbatch --account=<acct> --partition=<part> \
    --export=ALL,PHASE_LABEL=controlled,TRAIN_JOB=<training-array-id>,CHECKPOINT_STEP="$STEP" \
    --array=0-4 arc/rainbowdqn_eval_array.sbatch
done
```

Each task evaluates exactly 100 frozen E2 scenarios and 20 E3 anchors using a
greedy frozen policy. The wrapper rejects an unsealed training run, unexpected
checkpoint step, ambiguous checkpoint, changed checkpoint digest, changed
package identity or changed image identity.

## 9. Standalone tables and figures

Analysis reads only this package's results and rejects any non-
`RainbowDQN` identity:

```bash
apptainer exec --cleanenv \
  --bind "$RAINBOW_ROOT:$RAINBOW_ROOT" \
  "$TB3_IMAGE" bash -c '
    cd '"$TB3_REPO"' &&
    python3 scripts/make_tables.py \
      --results '"$TB3_OUTPUT"'/controlled/rainbowdqn \
      --out '"$TB3_OUTPUT"'/analysis/controlled/tables \
      --phase-label controlled &&
    python3 scripts/make_figures.py \
      --results '"$TB3_OUTPUT"'/controlled/rainbowdqn \
      --out '"$TB3_OUTPUT"'/analysis/controlled/figures \
      --phase-label controlled --formats pdf,png --style ieee'
```

Tables are written as CSV, Markdown and LaTeX; figures are vector PDF and
300-dpi PNG. `standalone_completeness.json` must pass. `--include-incomplete`
is diagnostic-only and cannot support reported results.

## 10. Verification boundary

Local tests cannot certify ROS transport, Gazebo services, Apptainer isolation
or Slurm lifecycle behavior. Under this project's terminology, the package is
not frozen/verified until the exact-SIF test, calibration and two-seed pilot
gates all pass unchanged. Preserve the ZIP, SIF, SIF digest, Slurm logs and
sealed result directories.
