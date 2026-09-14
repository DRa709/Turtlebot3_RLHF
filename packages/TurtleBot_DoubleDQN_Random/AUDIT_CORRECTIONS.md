# ARC/local audit corrections — Double DQN Random v1.0.1

This document records the response to the independent 2026-09-03 audit of
`TurtleBot_DoubleDQN_Random_v1.0.0.zip`, SHA-256
`0edda04a6c3901ceea04b88af3a31b420a04b9b1c12a5e6babe5d4644484abc9`.
Version 1.0.1 was built from that exact archive; the authenticated parent was
not modified in place.

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
| Global pip packages mix with Foxy/Ubuntu Python packages | Pinned dependencies use isolated `/opt/tb3-python`; image test checks Matplotlib module origins | Package hygiene and `apptainer test` |
| Foxy paths containing whitespace can load the wrong world | Preflight and simulator startup reject whitespace | Package hygiene and preflight |
| ARC documentation prefers the older module name | Runbook uses `module load apptainer`; job scripts retain the former name only as fallback | Shell and ARC-script tests |

## Shared-layer scope correction

The audit requested equality with the complete DQN v1.1.2 shared manifest.
Applying that instruction literally would be incorrect because that manifest
also contains DQN-branded files:

- the DQN preflight requires its DQN-specific algorithm YAML;
- the Apptainer definition labels and names a DQN image;
- `DATA_CONTRACT.md` describes DQN-specific diagnostics and analysis commands;
- `tests/test_package_hygiene.py` checks DQN package structure; and
- `VERIFICATION_SCOPE.md` must reflect the current six-algorithm study scope.

Copying those files byte-for-byte would either break Double DQN preflight or
misidentify this package as DQN or preserve an obsolete study scope. Version 1.0.1 therefore narrows
`SHARED_LAYER_FILES.txt` to the genuinely algorithm-independent scientific
core. Every listed file is byte-identical to DQN v1.1.2. The five
package- or study-scope-specific files are adapted explicitly for Double DQN and authenticated
by `RELEASE_MANIFEST.sha256` rather than the shared manifest.

This is not a learner dependency: no DQN network, DQN agent, DQN YAML, DQN ARC
wrapper or DQN result path is included.

## Protected Double DQN implementation

The following algorithm-specific files are retained from v1.0.0 except for the
required version field in `config/phase1_doubledqn.yaml`:

- `turtlebot3_drl_nav/doubledqn.py`;
- `turtlebot3_drl_nav/doubledqn_agent_node.py`;
- `turtlebot3_drl_nav/learner_factory.py`;
- `tests/test_doubledqn.py` and `tests/test_doubledqn_source.py`; and
- `tests/test_orchestrator.py`.

The target remains

$$
y_i=r_i+\gamma(1-m_i^{\mathrm{term}})
Q_{\bar\theta}\!\left(o'_i,
\arg\max_{a'}Q_\theta(o'_i,a')\right),
$$

with the selection/evaluation separation under `no_grad`.

## Promotion boundary

Version 1.0.1 must not begin calibration until all of the following pass:

1. Build the exact SIF on an ARC compute node and record its SHA-256.
2. Pass `apptainer test`.
3. Run `bash scripts/run_tests.sh` unchanged inside that SIF with every test
   passing and zero skips.
4. Repeat the reset/relocate/dynamic-obstacle probe for at least 10 simulation
   seconds inside the same SIF, including both reversal boundaries.
5. Complete and validate a tiny end-to-end run.
6. Complete the 10,000-transition calibration plus E2/E3 evaluation.
7. Complete the unchanged two-seed pilot plus evaluation before submitting the
   five controlled 500,000-transition runs.

Failure of any gate is a stop condition. Do not weaken the scientific or
validation thresholds.
