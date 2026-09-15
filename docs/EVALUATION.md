# Evaluation methodology

TurtleBot3 RLHF measures navigation outcomes, goal retention during policy
continuation, and reward prediction from human comparisons.

## Evaluation design

| Analysis | Protocol | Outcome |
| --- | --- | --- |
| Navigation benchmark (E1) | Six methods, five learners per method, 500,000 training actions and 20 final trials per learner | Discrete SAC, Double DQN, and Rainbow each reach 97/100 goals |
| Policy continuation (E2) | Two selected Discrete SAC actors, 20 scenarios, two action-selection modes; 80 trials per checkpoint | Baseline: 78/80 goals; after 10,000 actions: 6/80 under original reward and 1/80 under preference reward |
| Comparison budgets | 20, 50, 100, 150, and 200 comparisons; one reward-model seed and a common validation set | Strict-pair accuracy: 50.00%, 72.73%, 77.27%, 77.27%, and 86.36% |

The Discrete SAC benchmark combines **20 + 20 + 17 + 20 + 20 = 97** goals,
three safety stops, zero contacts, and zero timeouts. Its sample standard deviation
across five learner rates is 6.71 percentage points.

Both continuation conditions transfer the actor and initialize fresh critics,
optimizers, and replay. E2 uses a different scenario panel from E1.
Reward-prediction accuracy uses 22 strict validation pairs; cross-entropy uses
all 37 pairs including 15 ties. Cross-entropy is lowest at 150 comparisons.

## Outcomes and plots

- **Goal reached:** the goal condition is satisfied.
- **Safety stop:** the proximity mechanism terminates the episode.
- **Physical contact:** a contact/collision event is recorded.
- **Timeout:** the episode reaches its limit without another terminal outcome.

Outcomes are mutually exclusive, with contact, stop, then goal precedence.
Final benchmark plots show five learner rates and their means. Continuation
shading shows the range across two learners. Training reports summarize rolling
rates and first threshold crossings; final evaluation uses specified trial sets.

## Data and figure generation

The [data guide](../results/README.md) describes the seven tables, their
denominators, and the source verification record. From the repository root:

```bash
python benchmarks/build_report.py
python benchmarks/build_report.py --check
python -m pytest tests/test_benchmark_reporting.py tests/test_pomdp_contracts.py tests/test_geometry.py -q
```

The builder checks file hashes and accounting, then generates the README tables,
dashboard, and five figures. Training reports use explicit recorded inputs to
`benchmarks/generate_benchmark_suite.py`. Synthetic previews are labeled and
stored in a separate `synthetic_demo` directory.

## Validation scope

The targeted tests exercise observation/action/geometry functions, stop/contact
reporting, inconsistent records, missing step axes, the 97/100 calculation, and
demo isolation. Each algorithm package also provides inventory, version,
shell-script, and runtime checks described in its ARC runbook.

## Reproducibility scope

Figure reproduction uses the included tables. Repeating the experiments requires
the corresponding actors, configurations, containers, and deployment records.
The [repository README](../README.md#availability-and-reproducibility) describes
material availability.

Reset acceptance changed in public commit `62f5136`. See the
[version-specific reset checks](../README.md#reset-validation-and-experiment-versions)
before applying the current validator to archived runs. The reported navigation
results have not been regenerated or revalidated under that predicate.

## Limitations

The six-method benchmark includes release/configuration differences, and equal
action budgets do not imply equal compute. Safety stops alter exposure to
contacts. Actor-only continuation uses two learners and development scenarios,
so reward effects and restart effects require controlled ablations. Prediction
results use one seed and reused validation pairs; navigation was evaluated only
at 200 comparisons. The current source has not been matched to every experiment
artifact, including the reward model. Tests of reporting and package structure
do not establish ROS/Gazebo timing, physical-robot performance, or safety guarantees.
