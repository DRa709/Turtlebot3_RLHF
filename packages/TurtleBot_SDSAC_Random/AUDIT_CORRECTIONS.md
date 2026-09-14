# SD-SAC v1.0.0 audit corrections

This file records the disposition of the independent v1.0.0 audit. It does not
claim ARC, Slurm, Apptainer, ROS, or Gazebo evidence for v1.0.1.

| Audit finding | v1.0.1 correction | Regression evidence |
|---|---|---|
| Foxy Pytest plugin aborts collection | `run_tests.sh` disables external plugin auto-loading | `test_arc_scripts.py` |
| Foxy `ros2 topic echo --once` is unsupported | bounded `rclpy` readiness helper | `test_arc_scripts.py`, package inventory |
| 63-bit seeds lose precision | strict decimal parsing directly to `int` | `test_validator_parsing.py` |
| Validator fixture deletes required tests | temporary package copy preserves declared files | `test_validator.py` |
| Reset step 0 bypasses canonical CSV writer | obstacle acknowledgements return through the harness dispatcher | `test_validator.py` |
| Reward-tamper assertion expects wrong text | assertion matches the validator's six-component diagnostic | `test_validator.py` |
| `ast.unparse` is unavailable in Python 3.8 | source test inspects AST nodes without unparsing | `test_sdsac_source.py` |
| Dynamic obstacles depart from analytic paths | obstacle links are kinematic and gravity-free, with collisions retained | `test_world.py` |
| Runtime paths containing whitespace fail in Foxy launch | preflight and wrappers reject them before launch | `test_package_hygiene.py` |
| Container mixes Python package origins | isolated venv and explicit import-origin checks | Apptainer definition and hygiene tests |
| Analysis smoke test bypasses completion checks | a temporary one-seed protocol includes linked manifests and sealed runs | `test_analysis.py` |

The mandatory test runner also suppresses bytecode generation. Otherwise a
successful test invocation could add cache files to the authenticated package
tree and make the next preflight reject its release inventory.

Strict integer parsing also exposed that SD-SAC wrote `batch_size` and
`replay_size` as floating-point text. v1.0.1 writes those two discrete
diagnostics as integers. This is a schema correction only; it does not alter
network outputs, samples, targets, losses, optimizer steps, target updates,
action selection, replay contents, or checkpoints.

Promotion remains fail-closed. Build and hash the exact SIF, pass
`apptainer test` and the zero-skip suite, repeat the live simulator probe,
then pass calibration and the unchanged two-seed pilot before freezing.

## v1.0.1 independent audit and v1.0.2 correction

The independent v1.0.1 audit passed the complete 168-test suite and the live
Foxy/Gazebo readiness, relocation, analytic obstacle-trajectory, reversal, and
collision-contact probes. It then demonstrated that a valid direct preflight
created three `.pyc` files through the symlink-installed package, causing the
next package verification to fail its cache-free release rule.

Version 1.0.2 exports `PYTHONDONTWRITEBYTECODE=1` from:

- the Apptainer runtime environment;
- `scripts/preflight.sh`;
- `arc/run_sdsac_seed.sh`;
- `arc/run_sdsac_eval.sh`.

Both Slurm arrays also forward
`APPTAINERENV_PYTHONDONTWRITEBYTECODE=1` before their first container-side
phase query.

The regression suite now checks every entry point and executes a valid mocked
preflight between two package-verifier invocations. The second verifier must
pass and the copied release tree must contain no `__pycache__`, `.pyc`, or
`.pyo` artifact.
