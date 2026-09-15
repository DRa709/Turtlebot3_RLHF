# Audited aggregate results

These tables are unchanged exports from the audited ICRA2027 manuscript revision 6
snapshot (15 September 2026). They represent the cohorts reported there; do not
pool them with earlier partial archives, retries, demo output, or unspecified runs.
The reference repository before this correction is commit
`e7eb5a888f54eff136354ecba340cec7d0a26192`.

| File | Unit and role |
| --- | --- |
| `summary.csv` | One row per algorithm; final primary E1 counts/rates, learner variation, checkpoint averages, and descriptive travel statistics |
| `seed_results.csv` | One row per trained learner; five per algorithm |
| `evaluation_curves.csv` | Scheduled primary E1 checkpoint means; 20 per algorithm |
| `checkpoint_summary.csv` | Continuation outcomes and diagnostics by arm/checkpoint/suite/mode; `all` pools the two action-selection modes |
| `seed_summary.csv` | Same continuation summaries split by learning seed |
| `comparison_budget_metrics.csv` | Reward-prediction metrics at five cumulative comparison budgets |
| `policy_diagnostics.csv` | Fixed-observation categorical-policy change during continuation |
| `manifest.json` | SHA-256 hashes and source scope for all seven CSVs |
| `source_verification.json` | The earlier audit record, which also used episode-level and prediction-level records not included here |

## Denominators and interpretation

- Benchmark final primary success: 20 episodes per learner × five learners = 100.
  The Discrete SAC learner goal counts are 20, 20, 17, 20, 20. `final_sd_pp` is
  sample SD over five learner success rates, not a confidence interval.
- Six benchmark methods have 20 scheduled checkpoints. Case difficulty changes
  between checkpoints. Stochastic SAC evaluations are a separate channel and
  are not included in these primary benchmark means.
- E2 continuation: 20 scenarios × two learner seeds × two modes = 80 per checkpoint.
  E3 is a small fixed-requested-pose panel with eight trials per checkpoint.
  Rows with `policy_mode=all` already pool modes: do not sum them again with the
  deterministic and stochastic rows. Likewise, do not sum seed-level and aggregate tables.
- Comparison accuracy: `strict_correct / strict_n` on the same 22 strict pairs.
  `pair_cross_entropy` averages 37 pairs, including 15 ties. Scenario means give
  equal weight to 15 scenarios. `historical_validation_accuracy_pct` uses the
  smaller validation set present at each historical snapshot; it is not the
  common-set curve plotted in the paper.
- `fitting_pairs + snapshot_validation_pairs + cannot_judge = comparisons`.
  Total annotation cost is not the same as the number used to fit the reward.
- Goal-only path/time statistics condition on different successful subsets.
  With one preference success at the final E2 endpoint, they do not establish
  faster, shorter, or smoother navigation.
- Safety stops and recorded physical contacts are separate exclusive outcomes.
  Stops censor later contact exposure; zero contacts does not establish absence
  of unsafe behavior or improvement without the stopping mechanism.

## Reproduction boundary

`python benchmarks/build_audited_report.py` checks the manifest, learner/aggregate
agreement, exclusive-outcome totals, and comparison-cost/accuracy accounting, then
rebuilds five figures plus README tables and the dashboard.

The earlier audit recorded in `source_verification.json` recomputed prediction
losses from 185 saved prediction rows, checked 1,144 continuation episode rows,
and matched requested E2 scenarios. The public aggregate builder cannot repeat
those episode-level checks from these summaries. It does not execute saved
actors, verify missing deployment artifacts, or rerun training/simulation.
Experiment-specific code is available upon request.

No synthetic results or new observations were added. The pre-existing
`../mixcol_outcomes_summary.csv` is an unversioned legacy summary and is not an
input to the audited figures. Its cohort and reporting conventions have not
been reconciled with this snapshot.
