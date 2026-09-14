"""Closed loop of orchestrator + episode engine + fake simulator, without ROS.
Used by the orchestrator and validator tests to produce complete runs."""

import hashlib
import os
import sys
import time
import unittest
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch  # noqa: E402
except ModuleNotFoundError as exc:  # local hosts need not reproduce the ARC image
    raise unittest.SkipTest("PyTorch is required for the learner integration harness") from exc

from fake_sim import FakeSim  # noqa: E402

from turtlebot3_drl_nav import episode_engine as ee  # noqa: E402
from turtlebot3_drl_nav import orchestrator as orc  # noqa: E402
from turtlebot3_drl_nav.learner_factory import ALGORITHM, make_learner, parameter_digest, policy_loader  # noqa: E402
from turtlebot3_drl_nav.env_config import build_arena, build_environment_config, build_law, load_common_parameters  # noqa: E402
from turtlebot3_drl_nav.identity import build_identity, sha256_file  # noqa: E402
from turtlebot3_drl_nav.initialization import Sampler, generate_scenarios  # noqa: E402
from turtlebot3_drl_nav.protocol import decode_state, decode_step  # noqa: E402
from turtlebot3_drl_nav.recorder import open_stream  # noqa: E402
from turtlebot3_drl_nav.state import OBSERVATION_DIM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config")
WORLD = os.path.join(ROOT, "worlds", "phase1_mixed.world")
with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as _version_stream:
    PACKAGE_VERSION = _version_stream.read().strip()


def make_identity(config_sha: str, seed: int, phase_type: str = "training") -> Dict[str, object]:
    return build_identity(
        experiment="phase1-random-arm", algorithm=ALGORITHM, algorithm_version=PACKAGE_VERSION, arm="random",
        learning_seed=seed, world_id="phase1_mixed", world_seed=7000, initialization_seed=7001,
        dynamic_obstacle_seed=7002, evaluation_seed=9001, config_sha256=config_sha,
        container_sha256="0" * 64, shared_layer_sha256="1" * 64, release_sha256="2" * 64, package_version=PACKAGE_VERSION,
        phase_type=phase_type, phase_label="pilot", job_token="test",
    )


class LoopHarness:
    def __init__(self, run_dir: str, seed: int = 101, budget: int = 60, checkpoint_interval: int = 20,
                 full_interval: int = 40, tier1_episodes: int = 2, episode_max_steps: int = 7,
                 mode: str = "train", eval_checkpoint: Optional[str] = None, scenario_count: int = 12,
                 tier2_e3: int = 2) -> None:
        params = load_common_parameters(CONFIG)
        params.update({"world_seed": 7000, "initialization_seed": 7001, "dynamic_obstacle_seed": 7002,
                       "evaluation_seed": 9001, "episode_max_steps": episode_max_steps})
        self.params = params
        self.cfg = build_environment_config(params)
        arena = build_arena(params, WORLD)
        self.sampler = Sampler(build_law(params), arena)
        self.scenarios = generate_scenarios(self.sampler, 9001, scenario_count, 3, self.cfg.obstacle_half_period)
        self.sim = FakeSim(arena, self.cfg.control_period)
        self.engine = ee.EpisodeEngine(self.cfg, arena, self.sampler, self.scenarios, wall_clock=lambda: self.sim.wall)
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        config_sha = "c" * 64
        self.identity = make_identity(config_sha, seed, "training" if mode == "train" else "evaluation")
        self.streams = {name: open_stream(run_dir, name, self.identity) for name in ("transitions", "episodes", "updates", "evaluation", "checkpoints")}
        self.learner = make_learner(OBSERVATION_DIM, 5, seed, torch.device("cpu"), batch_size=4, replay_capacity=64,
                                    warmup_steps=8, target_update_steps=5, hidden_size=8, torch_threads=1)
        ocfg = orc.OrchestratorConfig(
            algorithm=ALGORITHM, mode=mode, environment_budget=budget, checkpoint_interval=checkpoint_interval,
            full_checkpoint_interval=full_interval, tier1_episodes=tier1_episodes, tier2_e3_episodes=tier2_e3,
            tier2_conditions=("E2", "E3"), checkpoint_dir=os.path.join(run_dir, "checkpoints"), learning_seed=seed,
            action_commands=self.cfg.action_map.commands(), config_sha256=config_sha,
            eval_checkpoint_path=eval_checkpoint or "", eval_checkpoint_step=budget,
            eval_checkpoint_index=budget // checkpoint_interval, evaluation_seed=9001,
        )
        self.orch = orc.Orchestrator(
            ocfg, self.learner, policy_loader(torch.device("cpu")), self.scenarios,
            sha256_file, parameter_digest, wall_clock=lambda: self.sim.wall,
        )
        self.finish: Optional[str] = None
        self.fatal: Optional[str] = None
        self.progress: List[Dict[str, object]] = []
        self.eval_rows: List[Dict[str, object]] = []
        self.checkpoint_rows: List[Dict[str, object]] = []
        self.update_rows: List[Dict[str, object]] = []

    # effects from the orchestrator go to the engine / streams
    def apply_orchestrator_effects(self, effects) -> None:
        for effect in effects:
            if isinstance(effect, orc.SendEpisodeControl):
                self.apply_engine_effects(self.engine.start_episode(effect.spec, self.sim.sim_time))
            elif isinstance(effect, orc.SendAction):
                self.apply_engine_effects(self.engine.receive_action(effect.message))
            elif isinstance(effect, orc.RecordUpdate):
                self.streams["updates"].write(effect.row)
                self.update_rows.append(effect.row)
            elif isinstance(effect, orc.RecordEvaluation):
                for row in effect.rows:
                    self.streams["evaluation"].write(row)
                    self.eval_rows.append(row)
            elif isinstance(effect, orc.RecordCheckpoint):
                self.streams["checkpoints"].write(effect.row)
                self.checkpoint_rows.append(effect.row)
            elif isinstance(effect, orc.Progress):
                self.progress.append(effect.payload)
            elif isinstance(effect, orc.Finish):
                self.finish = effect.reason
            elif isinstance(effect, orc.Fatal):
                self.fatal = effect.reason
            else:
                raise AssertionError(effect)

    # effects from the engine go to the orchestrator / streams / sim
    def apply_engine_effects(self, effects) -> None:
        for effect in effects:
            if isinstance(effect, ee.PublishState):
                self.apply_orchestrator_effects(self.orch.on_state(decode_state(effect.values)))
            elif isinstance(effect, ee.PublishStep):
                self.apply_orchestrator_effects(self.orch.on_step(decode_step(effect.values)))
            elif isinstance(effect, ee.PublishEpisodeSummary):
                self.apply_orchestrator_effects(self.orch.on_episode_summary(effect.payload))
            elif isinstance(effect, ee.RecordTransition):
                self.streams["transitions"].write(effect.row)
            elif isinstance(effect, ee.RecordEpisode):
                self.streams["episodes"].write(effect.row)
            elif isinstance(effect, ee.CallService):
                self.apply_engine_effects(self.sim.answer_service(self.engine, effect))
            elif isinstance(effect, ee.ObstacleControl):
                acknowledgement = self.sim.apply_obstacle_control(effect.payload)
                # A start acknowledgement can emit the step-zero transition.
                # Keep it on this dispatcher so it reaches transitions.csv.
                self.apply_engine_effects(self.engine.obstacle_ack(acknowledgement))
            elif isinstance(effect, ee.Fatal):
                self.fatal = effect.reason
            else:
                self.sim.apply_effects(self.engine, [effect])

    def run(self, max_ticks: int = 200000) -> None:
        self.apply_orchestrator_effects(self.orch.on_env_ready({"config_sha256": "c" * 64}))
        for _ in range(max_ticks):
            if self.finish or self.fatal:
                break
            self.sim.step_physics()
            c, s, d = self.sim.contact()
            self.engine.note_contact(c, s, d)
            self.apply_engine_effects(self.engine.tick(self.sim.sim_time, self.sim.scan(), self.sim.odom(), self.sim.obstacle_samples()))
        for stream in self.streams.values():
            stream.close()
