# Evaluation data

These CSV files contain the aggregate navigation and human-comparison results
shown in the repository. The figure builder reads this directory.

| File | Unit and role |
| --- | --- |
| `summary.csv` | One row per algorithm: final E1 counts, rates, learner variation, and travel statistics |
| `seed_results.csv` | One row per trained learner, five per algorithm |
| `evaluation_curves.csv` | Primary E1 means at 20 scheduled checkpoints per algorithm |
| `checkpoint_summary.csv` | Continuation outcomes by condition, checkpoint, suite, and action-selection mode |
| `seed_summary.csv` | Continuation outcomes split by learner seed |
| `comparison_budget_metrics.csv` | Reward-prediction measurements at five comparison budgets |
| `policy_diagnostics.csv` | Categorical-policy change on fixed observations |
| `manifest.json` | File hashes and data scope |
| `source_verification.json` | Recorded checks using aggregate, episode, and prediction records |

## Denominators

- Final E1 success uses 20 trials per learner and five learners: 100 trials per method.
  `final_sd_pp` is the sample standard deviation across five learner rates.
- Checkpoint curves use 20 scheduled evaluations. Cases change between checkpoints.
  Stochastic SAC evaluation is recorded separately from the primary channel.
- E2 continuation uses 20 scenarios, two learners, and two modes: 80 trials per checkpoint.
  E3 contains eight trials per checkpoint. `policy_mode=all` already pools modes;
  use either pooled rows or their components. Seed and aggregate tables describe
  the same trials.
- Accuracy is `strict_correct / strict_n` over the same 22 strict validation pairs.
  Cross-entropy averages 37 pairs including 15 ties. Scenario means weight 15
  scenarios equally. `historical_validation_accuracy_pct` uses the validation set
  available at each training snapshot; the plotted curve uses the common set.
- `fitting_pairs + snapshot_validation_pairs + cannot_judge = comparisons`.
- Goal-only path and time statistics summarize successful episodes.

## Reproduce the figures

From the repository root, run `python benchmarks/build_report.py`.
The builder checks hashes, learner/aggregate agreement, exclusive outcomes,
and comparison accounting. `source_verification.json` records checks on 185 saved
prediction rows and 1,144 continuation episodes, including requested E2 scenarios.

## Limitations

The episode and prediction records used for the source checks are not included
in these aggregate tables; those checks require the corresponding records.
Goal-only statistics use different successful subsets, and safety stops change
subsequent contact exposure. Validation pairs were reused for reward-model
selection. These summaries reproduce figures but do not rerun policies or
simulation. The separate `benchmarks/data/mixcol_outcomes_summary.csv` has an
unspecified cohort and is excluded from this analysis.
