# Evaluation methodology and reproducibility

TurtleBot3 RLHF evaluates discrete-action navigation, goal retention during
policy continuation, and reward prediction from human comparisons. The three
analyses use distinct evaluation cohorts and denominators.

## Evaluation design

| Analysis | Protocol | Reported outcome |
| --- | --- | --- |
| Six-algorithm navigation benchmark (E1) | Five learners per method, 500,000 training actions per learner, 20 final trials per learner | Discrete SAC, Double DQN, and Rainbow each reach 97/100 goals |
| Discrete SAC continuation (E2) | Two selected actors, 20 scenarios, two action-selection modes; 80 trials per checkpoint | Baseline: 78/80 goals; after 10,000 additional actions: 6/80 under original reward and 1/80 under preference reward |
| Human-comparison budgets | 20, 50, 100, 150, and 200 collected comparisons; one reward-model seed and a common validation set | Strict-pair accuracy: 50.00%, 72.73%, 77.27%, 77.27%, and 86.36% |

The final Discrete SAC benchmark combines learner goal counts of
**20 + 20 + 17 + 20 + 20 = 97**. The remaining three trials end in safety stops,
with zero recorded physical contacts and zero timeouts. The sample standard
deviation across its five learner rates is 6.71 percentage points.

Both continuation conditions transfer only the actor and initialize fresh
critics, optimizers, and replay. Both lose goal-reaching ability; the experiment
does not isolate the preference reward as the sole cause. The E2 baseline uses
a different scenario panel from E1.

Prediction accuracy uses the same 22 strict validation pairs. Cross-entropy uses
all 37 pairs, including 15 ties, and is lowest at 150 comparisons. Validation
was reused for checkpoint selection. Navigation was evaluated only at 200
comparisons, so a navigation benefit from increasing the comparison budget
remains a hypothesis for future experiments.

## Outcome definitions

- **Goal reached:** the robot satisfies the goal condition.
- **Safety stop:** the proximity mechanism terminates the episode before further motion.
- **Physical contact:** a recorded contact/collision event.
- **Timeout:** the episode reaches its time or action limit without another terminal outcome.

These outcomes are mutually exclusive. The reward implementation gives contact
precedence over safety stop, and safety stop over goal. Report contact and stop
rates separately: stops change exposure to subsequent contacts. Zero recorded
contacts alone does not establish safe navigation without the stopping mechanism.

Training-episode rolling rates describe behavior during learning. Final benchmark
rates describe the specified evaluation trials. First interpolated crossings of
a training threshold are distinct from sustained-success criteria.

## Data and figure generation

The [data guide](../benchmarks/data/audited/README.md) documents the seven aggregate
tables, their units, SHA-256 manifest, and source verification record. The report
builder uses those tables to generate the README results, dashboard, and five
figures: navigation benchmark, terminal outcomes, goal retention, policy
diagnostics, and reward prediction across comparison budgets.

From the repository root:

```bash
python benchmarks/build_audited_report.py
python benchmarks/build_audited_report.py --check
python -m pytest tests/test_benchmark_reporting.py tests/test_pomdp_contracts.py tests/test_geometry.py -q
```

For exploratory training reports, `benchmarks/generate_benchmark_suite.py`
requires explicit recorded inputs. It rejects missing step axes and preserves
missing reward/length values. Its synthetic demo mode isolates output under
`synthetic_demo`, labels figures and reports, and records provenance. Synthetic
demo output is excluded from the experimental results.

## Validation scope

On 15 September 2026, **22 targeted tests passed**. These tests exercise the
actual observation/action/geometry functions, separate stop/contact reporting,
invalid outcome records, missing step axes, the 97/100 calculation, and demo
isolation and labels.

The report builder and its `--check` mode passed; all five PNG figures were
visually inspected. A synthetic-demo smoke run generated reports for six methods
and five demo seeds per method. The builder checks available aggregate hashes
and accounting consistency. Recomputing the figures does not execute Gazebo or
rerun policy training.

## Reproducibility scope

- The aggregate summaries reproduce the figures. Repeating the full experiments
  requires the corresponding saved actors, configurations, containers, reward
  models, and deployment records; every artifact has not been matched to this source.
- The six-method benchmark includes release/configuration differences. Equal
  action budgets do not imply identical compute or implementation quality.
- Continuation uses two selected actors and development scenarios. Actor-only
  restart, fresh critics, and update scheduling require controlled ablations.
- The included reward ensemble is a software component; equivalence to the
  reward model used in the recorded experiments has not been established.
- The targeted validation does not cover the full PyTorch training, ROS/Gazebo,
  or physical-hardware suites. Physical navigation gains and safety guarantees
  require separate validation.
- The legacy mixed-outcome CSV is outside the source set for these figures.

Experiment-specific code is available upon request. The research objective is
to improve preference alignment while retaining goal competence; stable
continuation and navigation gains across comparison budgets remain open work.
