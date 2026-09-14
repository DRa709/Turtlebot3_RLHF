# Standalone verification scope — random-initial-state algorithms

## Study boundary

The retained algorithm set is DQN, Double DQN, Dueling Double DQN, Rainbow DQN
and Discrete SAC. PPO, other on-policy methods and the proposed A2C–Rainbow
hybrid are future work and are not execution targets in this study.

Each algorithm must ship as its own archive with its own `ALGORITHM.md`, agent,
configuration, ARC training script, ARC evaluation script and algorithm tests.
No archive may contain a second learner implementation. Common environment,
protocol, recorder, validator and analysis files are authenticated through
`SHARED_LAYER_MANIFEST.sha256`.

Verification is standalone: a package is judged against the roadmap, its own
algorithm equations, this random-initialization contract and its internal ARC
execution/data contract. No fixed-initialization code or performance comparison
is a prerequisite.

## Gates for every package

1. **Archive and provenance:** safe paths, no links/notebooks/caches, LF text,
   one algorithm, exact release inventory, configuration/shared/release/image
   digests in every data row. Forbidden bytecode, cache, VCS and build paths
   are rejected before package code is imported. The SIF is built only from a
   newly materialized tree containing the authenticated manifest inventory.
2. **Algorithm:** equations, masks, exploration, update cadence, target-network
   rule, checkpoint state and evaluation policy match that package's
   `ALGORITHM.md`.
3. **Shared POMDP:** 41-value observation, five frozen actions, reward
   decomposition, collision/safety/goal precedence, and truncation bootstrap.
4. **Random initialization:** episode 1 included; deterministic role-separated
   seeds; declared $\nu_R$ support; bounded rejection; awaited reset/relocation;
   realized pose, odometry, scan clearance and contact verified and recorded.
5. **Controlled dynamics:** policy work occurs while physics is paused; action
   duration is measured in simulation time; obstacles use acknowledged seeded
   simulation-time schedules; deviations fail closed.
6. **Data integrity:** five canonical streams, exact budget/cadence, raw-to-
   summary reconstruction, checkpoint digest selection, evaluation completeness,
   immutable run inventory and atomic completion markers.
7. **ARC:** Foxy/Gazebo/Apptainer consistency, source and trusted entrypoints
   embedded in the authenticated read-only SIF, no mutable source bind,
   manifest-checked Slurm script snapshots, exclusive allocation, leased
   ROS/Gazebo identities, explicit `--cleanenv` forwarding, a lease directory
   contained by the sole writable result bind, live-process monitoring,
   walltime handling and non-reused run directories.
8. **Paper outputs:** final E2/E3 blocks and the preregistered checkpoint-wise E2
   cadence provide the seed-level data, tables and simulation figures.

## Evidence levels and stop rules

- **Static/local pass:** archive, source, pure state machine and non-PyTorch tests
  pass. This is not ARC certification.
- **Container pass:** every test runs without skip inside the exact SIF whose
  SHA-256 will be recorded.
- **Calibration pass:** one direct-ARC 10k run and its E2/E3 evaluation end with
  valid `COMPLETE`, `validation_report.json` and `RUN_FILES.sha256`; timing,
  freshness and obstacle errors pass their frozen limits.
- **Pilot pass:** both preregistered 10k seeds pass without changing code,
  configuration or image.
- **Controlled pass:** all five 500k seeds and all preregistered post-hoc
  checkpoints pass. Only these sealed runs may support paper claims.

A failed gate stops promotion to the next phase. It does not authorize silently
changing a threshold or rerunning under a new identity; any correction produces
a new package version and a fresh run directory.
