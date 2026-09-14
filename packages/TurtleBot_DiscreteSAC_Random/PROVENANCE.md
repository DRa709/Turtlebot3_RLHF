# Provenance ledger — Discrete SAC Random v1.0.2

This ledger distinguishes every known artifact instead of assuming that files
with similar purposes or renamed downloads are byte-identical.

## Release lineage

| Artifact | SHA-256 | Role |
|---|---|---|
| `TurtleBot_DiscreteSAC_Random_v1.0.0.zip` | `34f7b48175ff049d26edc525f6e96f9f43be29452f975e44497c44463a723a3c` | Original standalone source candidate |
| `Pasted markdown(20260904-050939).md` | `bf9cf93ac65b0f1f3d4db484b2c68c47e452874b83b05fc5e9f101b6af089ac4` | v1.0.0 audit attachment received during the v1.0.1 correction round |
| Locally retained v1.0.0 audit reported by the later auditor | `ff3b1924d7a1d048c50ee7de2a03fe05ea09c78ad134ef560c6a7a1cf5153be0` | A distinct predecessor-audit artifact; its bytes were not supplied in the v1.0.2 build round |
| `TurtleBot_DiscreteSAC_Random_v1.0.1_ARC_Source_Corrected.zip` | `cbaf9b187202ad96d3dd537e72a6eb3a25d456ad7205cfc6606a268efd9eb033` | Audit-corrected source candidate |
| `TurtleBot_DiscreteSAC_Random_v1.0.1_CORRECTION_AND_VERIFICATION_REPORT.md` | `1cd9dae948bed40d68bff4a294be44bdfbfad9dac126e5e807eb2ef72ebe0d3c` | v1.0.1 correction report |
| Supplied v1.0.1 independent verification report, original CRLF bytes | `50e0232acc07297a1dcaeff254822ee40f8bdbcbf9aa9ba5b6fb299cce0c143c` | Report incorporated in this round |
| `LOCAL_VERIFICATION_REPORT_v1.0.1.md`, LF-normalized release copy | `d9a6e18571a010ee81eba37a3e0b375c645a5ddf851aba2fc525306b6519927d` | Exact report text normalized only for the package's LF-only policy |

The two v1.0.0 audit hashes are intentionally recorded as different artifacts.
The package does not claim that they are interchangeable. This resolves the
ambiguous phrase “the supplied audit-report SHA-256” in v1.0.1 without
inventing identity for the unavailable `ff3b...` file.

## v1.0.2 scope

v1.0.2 is an immutable documentation/provenance patch over the independently
tested v1.0.1 parent. It adds this ledger and the LF-normalized independent
verification report, updates status/runbook documentation, and advances the
package and algorithm-version markers so the new bytes cannot be confused with
v1.0.1.

No learning or simulator implementation is changed. In particular:

| Protected file | SHA-256 in v1.0.1 and v1.0.2 |
|---|---|
| `turtlebot3_drl_nav/discretesac.py` | `5439f518f73603c4728749cf6d721a92c609fbaae7f0c9fb66a5e1a5898654e9` |
| `turtlebot3_drl_nav/discretesac_agent_node.py` | `b8fe4a7c243a72da17b874dc13d5b59c90afc9754e9443cca0cdd2ab032b55b0` |
| `turtlebot3_drl_nav/initialization.py` | `58423aa582f68e81482746a20b62e13f385b4c5d85aa7dc14466a6f4c681e20c` |
| `turtlebot3_drl_nav/validator.py` | `bd27ba5b1d7a9c35c9bafede2a08fb4ee105ed6fe0994e8b957634faf5daed3f` |
| `worlds/phase1_mixed.world` | `ebd5861904d6de326e262367d5cefa2ab2e6172ecbc3e1745a513a4717ba04f4` |

The configuration digest changes because `algorithm_version` advances to
1.0.2. This prevents training or evaluation outputs from silently mixing the
two release identities even though the scientific settings are unchanged.

## Evidence boundary

The bundled report states that v1.0.1 passed 163/163 tests in the intended
isolated Python 3.8 runtime and passed a live Foxy/Gazebo probe with maximum
dynamic-obstacle error 0.01034 m, robot relocation error 0.0000298306 m and
named physical contact. The underlying raw log directories referenced by that
report were not supplied for inclusion here, so the report is retained as an
independent audit statement rather than represented as raw evidence.

v1.0.2 must still be authenticated, built and tested as an exact SIF on ARC.
It is not frozen until the ARC SIF, calibration and unchanged two-seed pilot
gates pass.
