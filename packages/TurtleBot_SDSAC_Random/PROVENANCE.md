# Provenance and evidence boundary

## Parent artifact

- Parent package: `TurtleBot_SDSAC_Random_v1.0.0.zip`
- Parent ZIP SHA-256:
  `1fafad9dce81cc66008e7d161c698c1ab600d675887310543e020029ec887ef6`
- Independent audit input: `Pasted markdown(20260904-070713).md`
- Raw uploaded file SHA-256 (CRLF bytes):
  `70a2437a96ef87304f6a07d3fda2332027426bc66dde5052abdcd6058d3e3be3`
- LF-normalized content SHA-256:
  `97c02b10db676c7426e3003fbf6794eae010084baeb3f6f69fbf196475aed9a3`

The two hashes identify the same textual audit under different line-ending
representations; neither is substituted for the other.

## Independent v1.0.1 audit

- Input: `Pasted markdown(20260904-123753).md`
- Raw uploaded file SHA-256:
  `2acf13fb62a73e463ff8ac5259dd27699b308de743c35b5a36585a1343b5d510`
- Audited v1.0.1 ZIP SHA-256:
  `2d1890231d016778d32a61ee4e387b8e31c152e2fcf25e261567d77185da3122`

## Protected SD-SAC sources in the parent

- `turtlebot3_drl_nav/sdsac.py`:
  `9ad0318cc74a94614a9ddfd8a1b15a92f59aa1fe7a0f35406b4dad65fc4143fb`
- `turtlebot3_drl_nav/sdsac_agent_node.py`:
  `f624fd351c6f3dc02518c72af3fdba05a6b39daf24d8ff52908966b8afac807f`
- `turtlebot3_drl_nav/initialization.py`:
  `58423aa582f68e81482746a20b62e13f385b4c5d85aa7dc14466a6f4c681e20c`

The agent node and initialization sampler remain byte-identical to the parent.
The learner file changes only the serialization types of `batch_size` and
`replay_size` from float to int so the strict CSV integer contract is
satisfied. The mathematical SD-SAC implementation is unchanged.

## Standalone boundary

This archive contains its own SD-SAC learner, environment, configuration,
world, tests, container recipe, ARC scripts, validator, tables, and figures.
It imports no learner or runtime file from the Discrete SAC, DQN, Double DQN,
Dueling Double DQN, or Rainbow packages. Evidence from another package is not
treated as execution evidence for this one.

## Verification boundary

Source-level and host-available checks do not substitute for exact-image or
live-ARC evidence. The release is not frozen until the exact SIF and unchanged
source bytes pass every gate listed in `VERIFICATION_STATUS.md`.
