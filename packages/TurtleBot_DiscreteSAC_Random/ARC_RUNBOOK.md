# ARC runbook — TurtleBot_DiscreteSAC_Random 1.0.2

Training and evaluation run as Slurm batch jobs on ARC. Run tests, Gazebo,
training, and analysis on allocated compute nodes.

## 1. Independent ARC layout

Use a directory dedicated to this algorithm:

```text
/projects/<allocation>/tb3_discretesac/
  package/TurtleBot_DiscreteSAC_Random/
  image/discretesac_random_v1.0.2.sif
  results/
```

Replace `<allocation>` with the ARC allocation in every command.

## 2. Build and verify this package's image

On an allocated compute node:

```bash
module load apptainer
export SAC_ROOT=/projects/<allocation>/tb3_discretesac
export TB3_REPO="$SAC_ROOT/package/TurtleBot_DiscreteSAC_Random"
export TB3_IMAGE="$SAC_ROOT/image/discretesac_random_v1.0.2.sif"
mkdir -p "$SAC_ROOT/image"
cd "$TB3_REPO"
apptainer build "$TB3_IMAGE" apptainer/tb3_phase1_foxy.def
sha256sum "$TB3_IMAGE" > "$TB3_IMAGE.sha256"
apptainer test "$TB3_IMAGE"
```

Preserve the exact SIF bytes after testing. Every Slurm task independently
recomputes the image checksum.

## 3. Verify the package and execute every test

The static package check is:

```bash
cd "$TB3_REPO"
bash scripts/verify_package.sh
```

Then execute the complete test suite inside the exact SIF:

```bash
apptainer exec --cleanenv \
  --bind "$SAC_ROOT:$SAC_ROOT" \
  "$TB3_IMAGE" \
  bash -c 'cd '"$TB3_REPO"' && bash scripts/run_tests.sh'
```

`scripts/run_tests.sh` fails if any test is skipped. Do not submit calibration
until both commands succeed.

## 4. Submission environment

```bash
export SAC_ROOT=/projects/<allocation>/tb3_discretesac
export TB3_REPO="$SAC_ROOT/package/TurtleBot_DiscreteSAC_Random"
export TB3_IMAGE="$SAC_ROOT/image/discretesac_random_v1.0.2.sif"
export TB3_OUTPUT="$SAC_ROOT/results"
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
  --array=0 arc/discretesac_array.sbatch
```

Record the returned training array job ID. Then submit its E2/E3 evaluation:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=calibration,TRAIN_JOB=<training-array-id> \
  --array=0 arc/discretesac_eval_array.sbatch
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
  --array=0-1 arc/discretesac_array.sbatch
```

Use the returned array ID to evaluate both pilot seeds:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=pilot,TRAIN_JOB=<training-array-id> \
  --array=0-1 arc/discretesac_eval_array.sbatch
```

Both training and both evaluation tasks must satisfy the calibration checks.
Create a new run directory for each submission.

## 7. Controlled training

Only after exact-image tests, calibration and pilot pass:

```bash
sbatch --account=<acct> --partition=<part> \
  --export=ALL,PHASE_LABEL=controlled \
  --array=0-4 arc/discretesac_array.sbatch
```

Array indices select learning seeds 101, 202, 303, 404 and 505 from the frozen
configuration. Each task performs exactly 500,000 training transitions, creates
policy checkpoints every 25,000 transitions, full checkpoints every 100,000
transitions and at the final budget, and evaluates 20 E1 episodes at every
policy checkpoint. E1 transitions do not consume the training budget.

Monitor without editing a run:

```bash
tail -f "$TB3_OUTPUT/controlled/discretesac/seed_101/job_<id>/logs/training_console.log"
python3 -m json.tool \
  "$TB3_OUTPUT/controlled/discretesac/seed_101/job_<id>/progress.json"
```

A Slurm walltime warning requests an atomic emergency checkpoint and produces
`INTERRUPTED`, never `COMPLETE`.

## 8. Controlled E2/E3 evaluation

For the controlled training array ID, submit the five declared checkpoints:

```bash
for STEP in 100000 200000 300000 400000 500000; do
  sbatch --account=<acct> --partition=<part> \
    --export=ALL,PHASE_LABEL=controlled,TRAIN_JOB=<training-array-id>,CHECKPOINT_STEP="$STEP" \
    --array=0-4 arc/discretesac_eval_array.sbatch
done
```

Each task evaluates exactly 100 frozen E2 scenarios and 20 E3 anchors in each
of two channels: seeded stochastic categorical actions and deterministic
argmax actions (240 episodes total). The wrapper rejects an unvalidated training run, unexpected
checkpoint step, ambiguous checkpoint, changed checkpoint digest, changed
package identity or changed image identity.

Across five tier-2 checkpoints, each controlled seed therefore produces 1,200
post-hoc episodes. Together with 800 in-run E1 episodes, that is 2,000
evaluation episodes per seed.

## 9. Standalone tables and figures

Analysis reads only this package's results and rejects any non-
`DiscreteSAC` identity:

```bash
apptainer exec --cleanenv \
  --bind "$SAC_ROOT:$SAC_ROOT" \
  "$TB3_IMAGE" bash -c '
    cd '"$TB3_REPO"' &&
    python3 scripts/make_tables.py \
      --results '"$TB3_OUTPUT"'/controlled/discretesac \
      --out '"$TB3_OUTPUT"'/analysis/controlled/tables \
      --phase-label controlled &&
    python3 scripts/make_figures.py \
      --results '"$TB3_OUTPUT"'/controlled/discretesac \
      --out '"$TB3_OUTPUT"'/analysis/controlled/figures \
      --phase-label controlled --formats pdf,png --style ieee'
```

Tables are written as CSV, Markdown and LaTeX; figures are vector PDF and
300-dpi PNG. `standalone_completeness.json` must pass. `--include-incomplete`
is diagnostic-only and cannot support reported results.

## 10. Runtime validation

Package tests cover update equations, episode ordering, recording, and validation.
Use the container tests, simulation calibration, and pilot runs to check the
ROS/Gazebo, Apptainer, and Slurm setup. Keep the package, image, logs, and run
directories with the experiment records.

Package and image integrity are checked with SHA-256 manifests.
