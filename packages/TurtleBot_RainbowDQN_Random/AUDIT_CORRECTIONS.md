# Independent-audit corrections — Rainbow DQN Random v1.0.1

Version 1.0.1 is derived from the submitted v1.0.0 ZIP with SHA-256
`d423d13910b5320ad1e5b7469d3f962d638f1caecd32c1c584b54136ee67d3da`.
The parent remains an immutable failed release candidate. The independent audit
incorporated here has SHA-256
`355d41271d786571789ea00f3b7dcf98d52c5afa6e294b835e8e0b782ac1f542`.

## Disposition of the independent findings

| Finding | v1.0.1 correction | Acceptance evidence |
|---|---|---|
| Foxy rejects `ros2 topic echo --once` | `scripts/wait_for_sim.sh` calls the bounded `scripts/wait_for_topics.py` subscriber | Static readiness contract plus mandatory live Foxy test |
| Dynamic obstacles depart from the prescribed trajectory | Both links are kinematic and gravity-free while retaining collision geometry | World regression test plus unchanged 0.05 m live tracking gate |
| Validator rounds 63-bit seeds through binary64 | `_i` validates signed decimal syntax and calls `int(value, 10)` directly | Boundary tests through `2**63-1` and the reported SHA-256-derived seed |
| Step-zero rows bypass `transitions.csv` in the mock loop | Obstacle acknowledgements return to `LoopHarness`, which dispatches all emitted effects | Nominal validator fixture requires one step-zero row per represented episode |
| Validator fixture removes manifest-required tests | Temporary copies preserve the complete authenticated test inventory | Validator suite can reach every test body |
| Reward-tamper test checks the wrong text | Test now checks `six reward components` | Targeted validator test |
| Rainbow Double-Q test uses nonexistent `.bias` | Fixture writes `bias_mu` after zeroing all parameters | Executable selector/evaluator and no-target-gradient test |
| Foxy `launch_testing` conflicts with pinned Pytest | Mandatory runner disables unrelated third-party plugin auto-loading | Runner contract and exact-SIF zero-skip gate |
| Scientific Python packages can mix with Ubuntu toolkits | Pinned stack is installed in `/opt/tb3-python`; `apptainer test` verifies interpreter and module origins | Exact image build/test |
| Foxy world paths containing spaces can load the wrong world | Preflight, simulator and both run wrappers reject whitespace | Package and ARC-script tests |
| Runbook uses the legacy module name first | Documentation and arrays use `module load apptainer` first, retaining the old name only as fallback in arrays | ARC-script contract |

## Additional correction found during integration

The v1.0.0 analysis smoke fixture invoked `--include-incomplete`, omitted the
training/evaluation manifest linkage, and used a 40-transition one-seed result
against the production two-seed pilot contract. Once the reported validator
failures are removed, that fixture can no longer prove the production
completeness gate. Version 1.0.1 gives the fixture a temporary package copy with
an internally consistent seed-101, 40-transition protocol; the production
package still requires both pilot seeds and all declared checkpoints.

## Protected Rainbow implementation

The following files retain their v1.0.0 production logic:

- `turtlebot3_drl_nav/rainbowdqn.py`;
- `turtlebot3_drl_nav/rainbowdqn_agent_node.py`;
- `turtlebot3_drl_nav/learner_factory.py`; and
- `turtlebot3_drl_nav/orchestrator.py` except that its tests now exercise the
  corrected common runtime.

The only algorithm-test source change is the invalid `bias` to `bias_mu`
fixture repair. Double-Q selection, categorical target evaluation, dueling
aggregation, C51 projection, PER, three-step returns, NoisyNet exploration,
checkpointing and evaluation semantics are unchanged.

## Standalone boundary

This archive contains only Rainbow DQN. It has its own learner, configuration,
agent, Slurm arrays, container recipe, checkpoints, validator, tables and
figures. It neither loads nor analyzes another algorithm package. The embedded
runtime files are authenticated inside this package and are not an external
dependency.

## Promotion boundary

Local source checks cannot replace ARC evidence. Before calibration, build the
exact SIF, record its SHA-256, pass `apptainer test`, and run
`bash scripts/run_tests.sh` unchanged with no failures, errors or skips. Then
repeat the 10+ simulation-second relocation/obstacle probe, complete the sealed
10,000-transition calibration with E2/E3 evaluation, and complete the unchanged
two-seed pilot. Any failed gate is a stop condition; do not weaken tolerances or
reuse a failed output directory.
