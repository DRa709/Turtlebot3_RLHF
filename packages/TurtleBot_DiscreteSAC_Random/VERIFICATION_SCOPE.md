# Verification scope — Discrete SAC random package 1.0.2

This package is judged standalone, without comparison to any fixed-start or
other-algorithm implementation.

1. **Algorithm:** categorical actor; two online and two target critics; exact
   five-action soft value and actor expectation; clipped double Q; fixed
   `alpha=0.2`; one-step terminal-only target mask; uniform replay.
2. **Schedule:** 5,000 uniform-random warm-up transitions with no earlier
   update; one actor plus two critic steps per later transition; hard twin-target
   copy each 1,000 gradient steps; exact phase budget.
3. **State and action:** 41 normalized values, nearest-rule 36-beam LiDAR and
   the frozen five commands.
4. **Random initialization:** seeded episode-indexed draws; episode 1 included;
   awaited reset/teleport/settle; realized pose/odometry/clearance/contact gates;
   fail closed with no default fallback.
5. **World:** four walls, two static boxes, two simulation-time moving
   obstacles; seeded phase and trajectory tracking.
6. **Timing:** policy computation while physics is paused, measured 0.1-second
   odometric holds, bounded sensor age and decision gap.
7. **Checkpointing:** atomic, cross-algorithm-safe policy and full formats;
   complete actor/critic/optimizer/replay/RNG resume state.
8. **Evaluation:** stochastic and deterministic channels on matched starts;
   reproducible stochastic action seeds; no learning or replay mutation; E1/E2/E3 completeness.
9. **Recording:** five typed streams and SAC-specific per-update diagnostics;
   independently recomputed rewards, masks, initialization and outcomes.
10. **ARC lifecycle:** unique transports/workspaces/run directories, exclusive
    Slurm tasks, explicit clean-environment identities, immutable run inventory,
    and validated atomic `COMPLETE`.
11. **Standalone analysis:** only Discrete SAC identities; policy channels never
    pooled; algorithm-specific tables and figures only.
12. **Release gates:** safe archive, syntax/static tests, exact-SIF zero-skip
    tests, live-Gazebo calibration, and unchanged two-seed ARC pilot.
