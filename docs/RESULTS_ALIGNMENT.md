# Results and documentation alignment

This correction replaces unsupported headline numbers with the aggregate results
audited for manuscript revision 6. It changes reporting and documentation; it does
not change historical measurements or claim a repaired navigation policy.

## Corrected issues

| Earlier repository claim | Correction |
| --- | --- |
| Discrete SAC 94.6 ± 1.4% success, 4.8 ± 1.1% collisions | Audited E1: 97/100 goals, three safety stops, zero contacts; learner SD 6.71 pp |
| Other five benchmark rows with unspecified cohort | Replace all six rows using one specified audited snapshot |
| Safety outcomes counted as collisions | Separate contact and safety-stop counts, rolling rates, plot labels, JSON and comparison CSV |
| Training rolling success presented as converged/final evaluation | Explicitly label training statistics and first interpolated threshold crossings |
| Synthetic and real inputs could be combined | Demo mode ignores real inputs, writes a separate directory, marks figures and reports, and records provenance |
| Missing steps, rewards, and lengths replaced with invented defaults | Reject absent step axes; preserve missing reward/length values instead of inventing observations |
| 0.22/0.18/0.08 m/s action table and angle/pi heading | Document implementation's 0.15/0.12/0.00 m/s commands, sine/cosine heading, and previous-command features |
| Index 18 called 180 degrees forward | Canonical index 18 is forward at zero degrees |
| Automatic entropy temperature and semi-discrete Gaussian SD-SAC | Document fixed-temperature categorical SAC and the source's categorical SD-SAC adaptation |
| Hardware verification, bit-exact reproduction, reward-hacking suppression | Remove conclusions not established by this experiment record |
| Expected test transcript presented without a recorded execution | Replace with reproducible commands and explicitly bounded validation |

The old two headline PNGs are removed from the current tree because their data
provenance was not established. Git history retains them. Their existence does
not establish whether they came from demo mode. New figures use the included
aggregate tables and a dedicated deterministic reporting procedure.

The current README's corrected BibTeX author separator/order is preserved.
`CITATION.cff`, contributor credit, and package author metadata include Asha Barua
while retaining Dhruv Shankar Ray as maintainer. Software credit does not determine
paper author order. No manuscript file is included in this correction.

## Findings that the repository now states explicitly

1. Discrete SAC, Double DQN, and Rainbow tie at 97% final primary E1 success.
2. E2 baseline has 78/80 goals; original-reward and preference-reward continuation
   have 6/80 and 1/80 after 10,000 added actions. Both lose retention.
3. Zero contacts coexist with more stops and lower clearance.
4. Reward prediction uses 20, 50, 100, 150, and 200 comparisons. Strict accuracy
   improves overall; cross-entropy is lowest at 150. These are reused validation
   results with one reward-model seed.
5. Navigation was evaluated only at 200 comparisons. A navigation benefit from
   increasing feedback budgets is a future experiment, not an observed result.

## Validation scope

Run from the repository root:

```bash
python benchmarks/build_audited_report.py
python benchmarks/build_audited_report.py --check
python -m pytest tests/test_benchmark_reporting.py tests/test_pomdp_contracts.py tests/test_geometry.py -q
```

The targeted tests call the actual observation/action/geometry implementation,
exercise separate stop/contact records, reject inconsistent outcomes and invented
step axes, verify the 97/100 calculation, and test demo isolation and figure labels.
The report builder checks all published aggregate hashes and consistency.
These checks do not certify the entire training/deployment repository.

Executed for this correction on 15 September 2026: **22 targeted tests passed**;
the audited builder and its `--check` mode passed; all five PNG figures were
visually inspected. A complete synthetic-demo smoke run generated reports for
six methods and five demo seeds per method, with demo provenance and labels.
The smoke outputs are excluded from this repository change and are not evidence
of navigation performance.

## Remaining reproducibility limits

- Public source has not been matched byte-for-byte to all historical evaluated
  deployments, configurations, containers, reward models, and saved actor files.
- The six-method benchmark includes historical release/configuration differences;
  equal action budgets do not establish identical compute or implementation quality.
- The continuation experiment uses two selected actors and reused development
  scenarios. Actor-only restart, fresh critics, and update scheduling remain
  plausible causes requiring controlled ablations.
- The public reward ensemble is a framework component. Its equivalence to the
  historical reward model is unverified.
- Tests requiring PyTorch, ROS/Gazebo, physical hardware, and historical artifacts
  require a suitable environment and a separate execution audit.
- The existing legacy mixed-outcome CSV is not a source for these audited results.

The correction aligns the reported results with the available audited snapshot;
it does not claim complete executable reproduction or a solved retention problem.
