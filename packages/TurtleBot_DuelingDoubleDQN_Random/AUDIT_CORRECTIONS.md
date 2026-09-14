# ARC/local audit corrections — Dueling Double DQN Random

## v1.0.2 — analysis-fixture and provenance correction

The independent re-verification of v1.0.1 has SHA-256
`ef540971c8f89eac2fab04ddc65a32e74278c19689a2819ca4a1ce44d629d932`.
It authenticated the v1.0.1 ZIP at
`6f689bbc1a7d41f52ded82b32440e77d500dc35ed21369815315a0f04471c0bc`
and independently confirmed the clean Foxy build, readiness path, strict
integer parsing, canonical step-zero recording, robot relocation and live
10.25-second obstacle trajectories. Its maximum obstacle error was 0.009240 m,
below the unchanged 0.05 m limit.

The mandatory v1.0.1 suite nevertheless reported 152 passed and two failed.
Both failures were in `tests/test_analysis.py`: a 40-step, seed-101 rendering
fixture was executed against the production pilot contract, which correctly
requires seeds 101 and 202, checkpoint 10,000 and an evaluation manifest bound
to its training run.

Version 1.0.2 corrects only that test boundary:

- the rendering fixture copies the package into a temporary directory;
- only that temporary copy receives a one-seed, 40-step pilot protocol;
- the real production protocol remains unchanged;
- the evaluation fixture records `training_run_id` and `checkpoint_step` before
  artifact sealing;
- both rendering commands execute from the temporary package; and
- the existing campaign-completeness test now also rejects an unexpected
  checkpoint and a missing second pilot seed.

The rendering smoke test is package-specific and is therefore no longer
declared as byte-identical common core. The v1.0.2 shared list has 61 paths:
58 are byte-identical to the corrected reference, `recorder.py` is the required
Dueling-diagnostics specialization, `tests/test_validator_parsing.py` retains
stronger integer-boundary coverage, and `tests/test_analysis_core.py` adds the
new audit's two negative campaign-completeness cases. Production analysis,
randomization, validation, environment dynamics and the Dueling learner are
unchanged.

## v1.0.1 — original runtime/shared-core correction

This document records the response to the independent 2026-09-03 audit of
`TurtleBot_DuelingDoubleDQN_Random_v1.0.0.zip`, SHA-256
`06e7b72aa57dac1de64e6cbdd314d7cad438b86954a6e0205a4771cd00a51d7c`.
Version 1.0.1 was built from that exact archive; the authenticated parent was
not modified in place. The independent audit report reviewed for this
correction has SHA-256
`23b206ec26bbd980bd7902622359544ca5761d848c6a523b41f2d799b0b33f09`.

## Disposition of the reported defects

| Reported defect | v1.0.1 disposition | Evidence gate |
|---|---|---|
| Pytest 8.3.3 auto-loads Foxy's incompatible `launch_testing` plugin | `scripts/run_tests.sh` exports `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before collection | ARC-script contract test; mandatory exact-SIF test |
| Foxy rejects `ros2 topic echo --once` | `scripts/wait_for_sim.sh` uses the bounded subscriber in `scripts/wait_for_topics.py` and requires advancing `/clock` | ARC readiness contract test; live exact-SIF probe |
| Dynamic obstacles depart from their analytic paths | Both links are kinematic and gravity-free, retain collision geometry, and remain subject to the frozen 0.05 m online/offline limit | `tests/test_world.py`; live 10+ second trajectory probe |
| Validator rounds large seed fields through `float` | Decimal syntax is validated and converted directly with `int` | `tests/test_validator_parsing.py`, including `6678464068980594013` |
| Validator fixture drops manifest-required tests | Temporary package copies retain the complete test inventory before manifest regeneration | All validator tests in the exact SIF |
| Obstacle acknowledgements bypass the canonical transition writer | The fake simulator returns the acknowledgement to `LoopHarness`, which dispatches resulting effects through the canonical streams | Nominal validator fixture requires one step-0 row per represented episode |
| Reward-tamper assertion does not match the validator message | Assertion checks `six reward components`; validator logic is unchanged | Reward-identity tamper test |
| Global pip packages mix with Foxy/Ubuntu Python packages | Pinned dependencies use isolated `/opt/tb3-python`; image test checks Torch, NumPy, YAML, pandas, Matplotlib and `mpl_toolkits` origins | Package hygiene and `apptainer test` |
| Foxy paths containing whitespace can load the wrong world | Preflight, simulator startup and evaluation reject whitespace | Package hygiene and preflight |
| ARC documentation prefers the older module name | Runbook uses `module load apptainer`; job scripts retain the former name only as fallback | Shell and ARC-script tests |

## Shared-layer scope correction

The audit compared v1.0.0 with the corrected 64-file random-start reference and
found 38 identical files, 24 differences and two missing files. Version 1.0.1
ports that scientific core. It records 62 common paths in
`SHARED_LAYER_FILES.txt`; the two analysis command-line entry points are
intentionally package-specific and reject every non-Dueling algorithm identity.

One necessary Dueling-specific boundary was found during correction:
the reference `recorder.py` classified `state_value_mean` and
`centered_advantage_abs_mean` as inapplicable to `DuelingDoubleDQN`, while the
protected learner correctly writes both diagnostics. A literal copy would
therefore make valid training runs fail final validation. The v1.0.1 report
stated that 61 paths were byte-identical. Independent re-verification found 60:
in addition to the intentional recorder specialization,
`tests/test_validator_parsing.py` strengthened the reference test at
$2^{53}-1$, $2^{53}$ and $2^{63}-1$. This was a benign test-only discrepancy,
but v1.0.2 records it explicitly. `tests/test_dueling_recorder_contract.py`
prevents the recorder interface from regressing.

Package-specific files include the preflight, analysis entry points and
rendering smoke test, container, data contract, verification scope and package
hygiene test. They are excluded from the shared-core list and authenticated by
`RELEASE_MANIFEST.sha256`.

This is not a runtime dependency. No other learner, agent, algorithm YAML, ARC
wrapper, SIF or result path is included or required.

## Protected Dueling Double DQN implementation

The following algorithm-specific files are retained from v1.0.0 except for the
required version field in `config/phase1_duelingdoubledqn.yaml`:

- `turtlebot3_drl_nav/duelingdoubledqn.py`;
- `turtlebot3_drl_nav/duelingdoubledqn_agent_node.py`;
- `turtlebot3_drl_nav/learner_factory.py`;
- `tests/test_duelingdoubledqn.py` and `tests/test_duelingdoubledqn_source.py`; and
- `tests/test_orchestrator.py`.

The target remains

$$
y_i=r_i+\gamma(1-m_i^{\mathrm{term}})
Q_{\bar\theta}\!\left(o'_i,
\arg\max_{a'}Q_\theta(o'_i,a')\right),
$$

with the selection/evaluation separation under `no_grad`.

The dueling aggregation remains

$$
Q_\theta(o,a)=V_\theta(o)+A_\theta(o,a)
-\frac{1}{5}\sum_{b=1}^{5}A_\theta(o,b).
$$

No prioritized replay, n-step return, distributional target or noisy layer is
introduced.

## Promotion boundary

Version 1.0.2 must not begin calibration until all of the following pass:

1. Build the exact SIF on an ARC compute node and record its SHA-256.
2. Pass `apptainer test`.
3. Run `bash scripts/run_tests.sh` unchanged inside that SIF and require
   154 passed with zero failures, errors or skips.
4. Repeat the reset/relocate/dynamic-obstacle probe for at least 10 simulation
   seconds inside the same SIF, including both reversal boundaries.
5. Complete and validate a tiny end-to-end run.
6. Complete the 10,000-transition calibration plus E2/E3 evaluation.
7. Complete the unchanged two-seed pilot plus evaluation before submitting the
   five controlled 500,000-transition runs.

Failure of any gate is a stop condition. Do not weaken the scientific or
validation thresholds.
