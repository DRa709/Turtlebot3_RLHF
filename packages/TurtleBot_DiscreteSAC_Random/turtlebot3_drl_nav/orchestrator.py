"""Pure training/evaluation orchestration of an agent node: which episode comes
next, exact-budget accounting, checkpoint cadence, the in-run tier-1 evaluation
of every checkpoint file, and the tier-2 post-hoc evaluation.

Shared layer (identical for every algorithm package); the learner is injected.
No ROS dependency: the ROS agent node feeds messages in and executes effects.

Learner interface (duck-typed):
    act(observation, evaluation=False) -> ActionChoice(action, epsilon_used, random_action)
    observe(Transition with both termination and episode-end masks);
    optimize() -> Optional[dict]; save(path, kind); env_steps; gradient_steps
    online (nn.Module) for the contamination digest; replay (len())
Evaluation policies are loaded from checkpoint files through ``policy_loader(path)``
returning an object with ``act(observation, policy_mode, sampling_seed) -> int``
and ``parameter_digest``.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .initialization import Scenario, derive_seed
from .learner_api import Transition
from .protocol import ActionMessage, EpisodeSpec, StateMessage, StepMessage
from .recorder import EPISODE_OUTCOME, INIT_BLOCK, UPDATE_APPLICABILITY, UPDATES_COLUMNS
from .state import ACTION_NAMES


# ----------------------------------------------------------------- effects

@dataclass(frozen=True)
class SendEpisodeControl:
    spec: EpisodeSpec


@dataclass(frozen=True)
class SendAction:
    message: ActionMessage


@dataclass(frozen=True)
class RecordUpdate:
    row: Dict[str, object]


@dataclass(frozen=True)
class RecordEvaluation:
    rows: List[Dict[str, object]]


@dataclass(frozen=True)
class RecordCheckpoint:
    row: Dict[str, object]


@dataclass(frozen=True)
class Progress:
    payload: Dict[str, object]


@dataclass(frozen=True)
class Finish:
    reason: str


@dataclass(frozen=True)
class Fatal:
    reason: str


@dataclass(frozen=True)
class OrchestratorConfig:
    algorithm: str
    mode: str                        # "train" | "eval"
    environment_budget: int
    checkpoint_interval: int
    full_checkpoint_interval: int
    tier1_episodes: int
    tier2_e3_episodes: int
    tier2_conditions: Sequence[str]
    checkpoint_dir: str
    learning_seed: int
    action_commands: Sequence[Sequence[float]]
    config_sha256: str
    eval_checkpoint_path: str = ""
    eval_checkpoint_step: int = 0
    eval_checkpoint_index: int = 0
    evaluation_seed: int = 0
    policy_modes: Sequence[str] = ("stochastic", "deterministic")

    def validate(self) -> None:
        if self.mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        if self.mode == "train":
            if self.environment_budget <= 0 or self.checkpoint_interval <= 0 or self.full_checkpoint_interval <= 0:
                raise ValueError("budget and checkpoint intervals must be positive")
            if self.environment_budget % self.checkpoint_interval != 0:
                raise ValueError("environment_budget must be a multiple of checkpoint_interval")
            if self.full_checkpoint_interval % self.checkpoint_interval != 0:
                raise ValueError("full_checkpoint_interval must be a multiple of checkpoint_interval")
            if self.tier1_episodes <= 0:
                raise ValueError("tier1_episodes must be positive")
        else:
            if not self.eval_checkpoint_path:
                raise ValueError("eval mode needs eval_checkpoint_path")
            if not self.tier2_conditions:
                raise ValueError("eval mode needs at least one condition")
        if self.algorithm not in UPDATE_APPLICABILITY:
            raise ValueError(f"unknown algorithm {self.algorithm}")
        if self.evaluation_seed < 0:
            raise ValueError("evaluation_seed must be non-negative")
        if tuple(self.policy_modes) != ("stochastic", "deterministic"):
            raise ValueError("Discrete SAC requires separate stochastic and deterministic evaluation channels")


@dataclass
class _PendingEvaluation:
    checkpoint_step: int
    checkpoint_index: int
    checkpoint_path: str
    checkpoint_sha256: str
    condition: str
    episodes: List[EpisodeSpec]


class Orchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        learner,
        policy_loader: Callable[[str], object],
        scenarios: Sequence[Scenario],
        sha256_file: Callable[[str], str],
        parameter_digest: Callable[[object], str],
        wall_clock: Callable[[], float],
    ) -> None:
        config.validate()
        self.cfg = config
        self.learner = learner
        self.policy_loader = policy_loader
        self.scenarios = list(scenarios)
        self.sha256_file = sha256_file
        self.parameter_digest = parameter_digest
        self.wall_clock = wall_clock
        self.finished = False
        self.fatal: Optional[str] = None
        self.env_ready = False
        self.current_spec: Optional[EpisodeSpec] = None
        self.current_observation: Optional[np.ndarray] = None
        self.pending_choice = None
        self.pending_sequence: Optional[int] = None
        self.pending_queue: List[_PendingEvaluation] = []
        self.active_evaluation: Optional[_PendingEvaluation] = None
        self.active_policy = None
        self.eval_rows: List[Dict[str, object]] = []
        self.eval_digest_before: Optional[str] = None
        self.eval_replay_before: Optional[int] = None
        self.checkpoints_written: List[int] = []
        self.applicable = set(UPDATE_APPLICABILITY[config.algorithm])
        self.started_wall = wall_clock()
        self.training_episodes_seen = 0
        self.current_episode_key: Optional[int] = None
        self.pending_summary: Optional[Dict[str, object]] = None

    # ---------------------------------------------------------------- utils

    def _fatal(self, reason: str) -> List[object]:
        self.fatal = reason
        return [Fatal(reason)]

    def _checkpoint_index(self, step: int) -> int:
        return step // self.cfg.checkpoint_interval

    def _checkpoint_path(self, step: int, kind: str) -> str:
        name = f"{self.cfg.algorithm.lower()}_seed{self.cfg.learning_seed}_step{step}_{kind}.pt"
        return os.path.join(self.cfg.checkpoint_dir, name)

    # ------------------------------------------------------------ env ready

    def on_env_ready(self, payload: Dict[str, object]) -> List[object]:
        if self.env_ready:
            return []
        if str(payload.get("config_sha256")) != self.cfg.config_sha256:
            return self._fatal("environment and agent disagree on the configuration digest")
        self.env_ready = True
        if self.cfg.mode == "train":
            return self._start_training_episode()
        return self._start_tier2()

    def _start_training_episode(self) -> List[object]:
        mode = "stochastic"
        spec = EpisodeSpec(phase="training", policy_mode=mode)
        self.current_spec = spec
        self.current_observation = None
        self.current_episode_key = None
        self.pending_summary = None
        return [SendEpisodeControl(spec)]

    # ------------------------------------------------------------ state/step

    def on_state(self, msg: StateMessage) -> List[object]:
        if self.finished or self.fatal or self.current_spec is None:
            return []
        if self.pending_sequence is not None:
            return []  # a decision is already outstanding; duplicate initial state
        if self.current_episode_key is None:
            self.current_episode_key = msg.episode_key
        elif msg.episode_key != self.current_episode_key:
            return self._fatal("state episode_key changed within an episode")
        self.current_observation = msg.observation.copy()
        return self._decide(msg.sequence)

    def _decide(self, sequence: int) -> List[object]:
        spec = self.current_spec
        observation = self.current_observation
        final = False
        if spec.phase == "training":
            choice = self.learner.act(observation, evaluation=False)
            action = choice.action
            final = (self.learner.env_steps + 1) >= self.cfg.environment_budget
        else:
            choice = None
            sampling_seed = derive_seed(
                self.cfg.evaluation_seed,
                "discretesac_evaluation_action",
                spec.checkpoint_index,
                spec.condition,
                spec.scenario_id or "",
                spec.evaluation_episode,
                spec.policy_mode,
                sequence,
            )
            action = int(self.active_policy.act(observation, spec.policy_mode, sampling_seed))
        linear, angular = self.cfg.action_commands[action]
        self.pending_choice = choice
        self.pending_action = action
        self.pending_sequence = sequence
        return [SendAction(ActionMessage(sequence, action, float(linear), float(angular), final))]

    def on_step(self, msg: StepMessage) -> List[object]:
        if self.finished or self.fatal or self.current_spec is None or self.pending_sequence is None:
            return []
        if msg.sequence != self.pending_sequence:
            return []  # stale
        if msg.episode_key != self.current_episode_key:
            return self._fatal("transition episode_key disagrees with the current episode")
        effects: List[object] = []
        spec = self.current_spec
        if spec.phase == "training":
            self.learner.observe(Transition(
                self.current_observation.copy(), self.pending_action,
                msg.reward_total, msg.observation.copy(),
                msg.terminated, msg.episode_end,
            ))
            diagnostics = self.learner.optimize()
            if diagnostics is not None:
                row = {k: None for k in UPDATES_COLUMNS}
                row.update({
                    "gradient_step": self.learner.gradient_steps,
                    "env_step": self.learner.env_steps,
                })
                for key, value in diagnostics.items():
                    if key in row:
                        row[key] = value
                for key in list(row):
                    if key not in self.applicable:
                        row[key] = None
                effects.append(RecordUpdate(row))
            effects.extend(self._checkpoint_if_due())
            if self.learner.env_steps % 1000 == 0:
                effects.append(Progress({"env_steps": self.learner.env_steps, "wall_s": self.wall_clock() - self.started_wall}))
        self.pending_sequence = None
        self.pending_choice = None
        self.current_observation = msg.observation.copy()
        if msg.episode_end:
            if self.pending_summary is not None:
                summary = self.pending_summary
                self.pending_summary = None
                effects.extend(self._consume_episode_summary(summary))
            return effects
        effects.extend(self._decide(msg.sequence + 1))
        return effects

    # ---------------------------------------------------------- checkpoints

    def _checkpoint_if_due(self) -> List[object]:
        step = self.learner.env_steps
        if step == 0 or step % self.cfg.checkpoint_interval != 0 or step in self.checkpoints_written:
            return []
        effects: List[object] = []
        policy_path = self._checkpoint_path(step, "policy")
        self.learner.save(policy_path, kind="policy_only", extra={"action_map": [list(c) for c in self.cfg.action_commands]})
        policy_digest = self.sha256_file(policy_path)
        loaded = self.policy_loader(policy_path)
        validated = loaded.parameter_digest == self.parameter_digest(self.learner.online)
        if not validated:
            return self._fatal(f"policy checkpoint {policy_path} does not load back to the live parameters")
        effects.append(RecordCheckpoint({
            "env_step": step, "gradient_step": self.learner.gradient_steps, "kind": "policy_only",
            "path": policy_path, "sha256": policy_digest, "sim_time": None, "wall_time": self.wall_clock(), "validated": True,
        }))
        if step % self.cfg.full_checkpoint_interval == 0 or step == self.cfg.environment_budget:
            full_path = self._checkpoint_path(step, "full")
            self.learner.save(full_path, kind="full", extra={"action_map": [list(c) for c in self.cfg.action_commands]})
            loaded_full = self.policy_loader(full_path)
            if loaded_full.parameter_digest != self.parameter_digest(self.learner.online):
                return self._fatal(f"full checkpoint {full_path} does not load back to the live parameters")
            effects.append(RecordCheckpoint({
                "env_step": step, "gradient_step": self.learner.gradient_steps, "kind": "full",
                "path": full_path, "sha256": self.sha256_file(full_path), "sim_time": None, "wall_time": self.wall_clock(), "validated": True,
            }))
        self.checkpoints_written.append(step)
        index = self._checkpoint_index(step)
        episodes = [
            EpisodeSpec(phase="evaluation", policy_mode=mode, condition="E1", checkpoint_step=step,
                        checkpoint_index=index, evaluation_episode=i)
            for mode in self.cfg.policy_modes
            for i in range(1, self.cfg.tier1_episodes + 1)
        ]
        self.pending_queue.append(_PendingEvaluation(step, index, policy_path, policy_digest, "E1", episodes))
        return effects

    # ------------------------------------------------------ episode summary

    def on_episode_summary(self, payload: Dict[str, object]) -> List[object]:
        if self.finished or self.fatal or self.current_spec is None:
            return []
        if int(payload.get("episode_key", -1)) != self.current_episode_key:
            return self._fatal("episode summary does not match the current episode")
        if self.pending_sequence is not None:
            if self.pending_summary is not None:
                return self._fatal("duplicate episode summary received before the terminal transition")
            self.pending_summary = dict(payload)
            return []
        return self._consume_episode_summary(payload)

    def _consume_episode_summary(self, payload: Dict[str, object]) -> List[object]:
        spec = self.current_spec
        self.current_spec = None
        self.pending_sequence = None
        self.current_episode_key = None
        if spec.phase == "training":
            self.training_episodes_seen += 1
            return self._after_training_episode()
        # evaluation episode
        block = self.active_evaluation
        row = self._evaluation_row(spec, payload, block)
        self.eval_rows.append(row)
        if block.episodes:
            return self._start_evaluation_episode()
        return self._close_evaluation_block()

    def _after_training_episode(self) -> List[object]:
        if self.pending_queue:
            return self._open_evaluation_block()
        if self.learner.env_steps >= self.cfg.environment_budget:
            self.finished = True
            return [Finish("budget_reached")]
        return self._start_training_episode()

    def _open_evaluation_block(self) -> List[object]:
        block = self.pending_queue.pop(0)
        self.active_evaluation = block
        self.active_policy = self.policy_loader(block.checkpoint_path)
        self.eval_rows = []
        self.eval_digest_before = self.parameter_digest(self.learner.online) if self.cfg.mode == "train" else None
        self.eval_replay_before = len(self.learner.replay) if self.cfg.mode == "train" else None
        return self._start_evaluation_episode()

    def _start_evaluation_episode(self) -> List[object]:
        spec = self.active_evaluation.episodes.pop(0)
        self.current_spec = spec
        self.current_observation = None
        self.current_episode_key = None
        self.pending_summary = None
        return [SendEpisodeControl(spec)]

    def _close_evaluation_block(self) -> List[object]:
        unchanged = True
        if self.cfg.mode == "train":
            unchanged = (
                self.parameter_digest(self.learner.online) == self.eval_digest_before
                and len(self.learner.replay) == self.eval_replay_before
            )
            if not unchanged:
                return self._fatal("learner state changed during evaluation")
        for row in self.eval_rows:
            row["learner_state_unchanged"] = unchanged
        effects: List[object] = [RecordEvaluation(list(self.eval_rows))]
        self.eval_rows = []
        self.active_evaluation = None
        self.active_policy = None
        if self.cfg.mode == "eval":
            if self.pending_queue:
                effects.extend(self._open_evaluation_block())
            else:
                self.finished = True
                effects.append(Finish("evaluation_complete"))
            return effects
        effects.extend(self._after_training_episode())
        return effects

    def _evaluation_row(self, spec: EpisodeSpec, payload: Dict[str, object], block: _PendingEvaluation) -> Dict[str, object]:
        outcome = payload["outcome_block"]
        init = payload["init_block"]
        row: Dict[str, object] = {
            "checkpoint_step": block.checkpoint_step, "checkpoint_index": block.checkpoint_index,
            "checkpoint_sha256": block.checkpoint_sha256, "condition": spec.condition,
            "scenario_id": spec.scenario_id, "evaluation_episode": spec.evaluation_episode,
            "policy_mode": spec.policy_mode, "episode_key": payload["episode_key"], "learner_state_unchanged": None,
        }
        for key in EPISODE_OUTCOME:
            row[key] = outcome.get(key)
        for key in INIT_BLOCK:
            row[key] = init.get(key)
        labels = payload.get("scenario_labels") or {}
        for key in ("distance_bin", "clearance_bin", "heading_bin", "difficulty"):
            row[key] = labels.get(key)
        return row

    # ---------------------------------------------------------------- tier 2

    def _start_tier2(self) -> List[object]:
        cfg = self.cfg
        digest = self.sha256_file(cfg.eval_checkpoint_path)
        for mode in cfg.policy_modes:
            for condition in cfg.tier2_conditions:
                if condition == "E2":
                    episodes = [
                        EpisodeSpec(phase="evaluation", policy_mode=mode, condition="E2", checkpoint_step=cfg.eval_checkpoint_step,
                                    checkpoint_index=cfg.eval_checkpoint_index, scenario_id=s.scenario_id, evaluation_episode=i + 1)
                        for i, s in enumerate(self.scenarios)
                    ]
                elif condition == "E3":
                    episodes = [
                        EpisodeSpec(phase="evaluation", policy_mode=mode, condition="E3", checkpoint_step=cfg.eval_checkpoint_step,
                                    checkpoint_index=cfg.eval_checkpoint_index, evaluation_episode=i)
                        for i in range(1, cfg.tier2_e3_episodes + 1)
                    ]
                elif condition == "E1":
                    episodes = [
                        EpisodeSpec(phase="evaluation", policy_mode=mode, condition="E1", checkpoint_step=cfg.eval_checkpoint_step,
                                    checkpoint_index=cfg.eval_checkpoint_index, evaluation_episode=i)
                        for i in range(1, cfg.tier1_episodes + 1)
                    ]
                else:
                    return self._fatal(f"unknown evaluation condition {condition}")
                self.pending_queue.append(_PendingEvaluation(cfg.eval_checkpoint_step, cfg.eval_checkpoint_index, cfg.eval_checkpoint_path, digest, condition, episodes))
        return self._open_evaluation_block()
