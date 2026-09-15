"""Pure episode logic of the Phase-1 environment: transactional reset,
fixed simulation-time action hold, two-mask transitions, recording rows.

Shared layer. No ROS dependency: the ROS node feeds sensor samples, service
results and actions into this state machine and executes the effects it
returns, so every branch here is unit-testable with a fake simulator.

State machine
    IDLE -> obstacle HOLD/ack -> reset -> relocate -> settle -> pause/ack
      -> verify realized state -> obstacle START/ack -> AWAIT_ACTION
      -> unpause/ack -> HOLD -> stop + pause/ack -> capture -> AWAIT_ACTION
Any failure that the contract forbids to fall back from becomes FATAL.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .geometry import Arena
from .initialization import (
    ObstaclePhases,
    Sampler,
    Scenario,
    StartPose,
    e1_obstacle_seed,
    e1_start_seed,
    e3_obstacle_seed,
    obstacle_phases,
    scenario_by_id,
    training_obstacle_seed,
    training_start_seed,
)
from .protocol import ActionMessage, EpisodeSpec, encode_state, encode_step
from .recorder import EPISODE_OUTCOME, INIT_BLOCK
from .obstacle_schedule import obstacle_displacement
from .state import (
    ACTION_NAMES,
    LIDAR_BINS,
    ActionMap,
    Pose2D,
    RewardConfig,
    build_observation,
    compute_reward,
    goal_features,
    min_valid_range,
    sample_lidar_nearest,
    sample_lidar_sector_min,
)


# ------------------------------------------------------------------ inputs

@dataclass(frozen=True)
class ScanSample:
    ranges: Tuple[float, ...]
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    stamp: float


@dataclass(frozen=True)
class OdomSample:
    x: float
    y: float
    yaw: float
    stamp: float


@dataclass(frozen=True)
class ObstacleSample:
    name: str
    x: float
    y: float
    stamp: float


# ----------------------------------------------------------------- effects

@dataclass(frozen=True)
class PublishVelocity:
    linear: float
    angular: float


@dataclass(frozen=True)
class PublishState:
    values: List[float]


@dataclass(frozen=True)
class PublishStep:
    values: List[float]


@dataclass(frozen=True)
class PublishEpisodeSummary:
    payload: Dict[str, object]


@dataclass(frozen=True)
class ObstacleControl:
    payload: Dict[str, object]


@dataclass(frozen=True)
class CallService:
    kind: str  # reset_world | set_entity_state | get_entity_state | pause_physics | unpause_physics
    entity: Optional[str] = None
    pose: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class RecordTransition:
    row: Dict[str, object]


@dataclass(frozen=True)
class RecordEpisode:
    row: Dict[str, object]


@dataclass(frozen=True)
class Log:
    level: str
    text: str


@dataclass(frozen=True)
class Fatal:
    reason: str


Effect = object


# ------------------------------------------------------------------ config

@dataclass(frozen=True)
class EnvironmentConfig:
    goal_x: float
    goal_y: float
    lidar_max: float
    distance_max: float
    lidar_sampling: str
    control_period: float
    episode_max_steps: int
    action_map: ActionMap
    reward: RewardConfig
    robot_model_name: str
    obstacle_names: Tuple[str, str]
    obstacle_speed: float
    obstacle_half_period: float
    settle_sim_s: float
    position_tolerance: float
    yaw_tolerance: float
    odom_tolerance: float
    odom_yaw_tolerance: float
    hold_tolerance_fraction: float
    decision_gap_tolerance_sim_s: float
    sensor_max_age_sim_s: float
    obstacle_position_tolerance: float
    fixed_start: Tuple[float, float, float]
    initialization_seed: int
    dynamic_obstacle_seed: int
    evaluation_seed: int
    service_timeout_wall_s: float = 30.0
    obstacle_ack_timeout_wall_s: float = 10.0
    sensor_timeout_wall_s: float = 10.0
    action_timeout_wall_s: float = 300.0
    fatal_on_reset_contact: bool = True


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ------------------------------------------------------------------ engine

@dataclass
class _EpisodeState:
    spec: EpisodeSpec
    episode_key: int
    training_episode: Optional[int]
    start: StartPose
    phases: ObstaclePhases
    init_kind: str
    scenario: Optional[Scenario]
    init_row: Dict[str, object] = field(default_factory=dict)
    step_index: int = 0
    sequence: int = 0
    sim_time_start: float = 0.0
    wall_time_start: float = 0.0
    start_env_step: int = 0
    # accumulators
    length: int = 0
    ret: float = 0.0
    sums: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in ("r_distance", "r_step", "r_collision", "r_goal", "r_angular", "r_near")})
    near_penalty_steps: int = 0
    min_clearance: float = math.inf
    path_length: float = 0.0
    holds: List[float] = field(default_factory=list)
    decision_latencies: List[float] = field(default_factory=list)
    decision_gaps: List[float] = field(default_factory=list)
    sensor_ages: List[float] = field(default_factory=list)
    obstacle_errors: List[float] = field(default_factory=list)
    hold_out_of_tol: int = 0
    last_pose: Optional[Tuple[float, float]] = None
    final: Optional[Dict[str, object]] = None


class EpisodeEngine:
    IDLE = "IDLE"
    WAIT_OBSTACLE_HOLD = "WAIT_OBSTACLE_HOLD"
    RESET_WORLD = "RESET_WORLD"
    SET_ROBOT = "SET_ROBOT"
    UNPAUSE_SETTLE = "UNPAUSE_SETTLE"
    SETTLE = "SETTLE"
    PAUSE_RESET = "PAUSE_RESET"
    CAPTURE_RESET = "CAPTURE_RESET"
    VERIFY_ROBOT = "VERIFY_ROBOT"
    VERIFY_OBS = "VERIFY_OBS"
    WAIT_OBSTACLE_START = "WAIT_OBSTACLE_START"
    AWAIT_ACTION = "AWAIT_ACTION"
    UNPAUSE_ACTION = "UNPAUSE_ACTION"
    HOLD = "HOLD"
    PAUSE_ACTION = "PAUSE_ACTION"
    CAPTURE_ACTION = "CAPTURE_ACTION"
    FATAL = "FATAL"

    def __init__(
        self,
        config: EnvironmentConfig,
        arena: Arena,
        sampler: Sampler,
        scenarios: Sequence[Scenario],
        wall_clock: Callable[[], float],
    ) -> None:
        self.cfg = config
        self.arena = arena
        self.sampler = sampler
        self.scenarios = list(scenarios)
        self.wall_clock = wall_clock
        self.state = self.IDLE
        self.episode_key = 0
        self.training_episode = 0
        self.env_step = 0                 # training transitions closed so far
        self.next_sequence = 1
        self.commands = config.action_map.commands()
        self.current: Optional[_EpisodeState] = None
        self.pending_action: Optional[ActionMessage] = None
        self.contact_latched = False
        self.static_latched = False
        self.dynamic_latched = False
        self._service_started_wall: Optional[float] = None
        self._state_entered_wall: float = wall_clock()
        self._settle_until: float = 0.0
        self._set_time: float = 0.0
        self._verify_queue: List[str] = []
        self._realized: Dict[str, Tuple[float, float, float]] = {}
        self._apply_sim: float = 0.0
        self._apply_odom_stamp: float = 0.0
        self._previous_distance: float = 0.0
        self._applied: Optional[ActionMessage] = None
        self._last_state_values: Optional[List[float]] = None
        self._last_observation: Optional[List[float]] = None
        self._await_since_wall: float = 0.0
        self._reset_contact = False
        self._obstacle_started_wall: Optional[float] = None
        self._reset_snapshot = None
        self._transition_snapshot = None
        self._last_state_sim: float = 0.0
        self._last_state_wall: float = 0.0
        self._last_odom_stamp: float = 0.0
        self._last_distance: float = 0.0
        self._decision_gap_sim: float = 0.0
        self._decision_latency_wall: float = 0.0

    # -------------------------------------------------------------- helpers

    def _enter(self, state: str) -> None:
        self.state = state
        self._state_entered_wall = self.wall_clock()

    def note_contact(self, collision: bool, static: bool, dynamic: bool) -> None:
        self.contact_latched = self.contact_latched or collision
        self.static_latched = self.static_latched or static
        self.dynamic_latched = self.dynamic_latched or dynamic

    def _clear_contacts(self) -> None:
        self.contact_latched = self.static_latched = self.dynamic_latched = False

    def _lidar(self, scan: ScanSample) -> List[float]:
        physical_max = min(scan.range_max, self.cfg.lidar_max) if scan.range_max > 0.0 else self.cfg.lidar_max
        if self.cfg.lidar_sampling == "sector_min":
            return sample_lidar_sector_min(scan.ranges, scan.range_min, physical_max, LIDAR_BINS)
        return sample_lidar_nearest(scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, physical_max, LIDAR_BINS)

    def _observe(self, scan: ScanSample, odom: OdomSample, linear: float, angular: float):
        lidar = self._lidar(scan)
        pose = Pose2D(odom.x, odom.y, odom.yaw)
        distance, heading = goal_features(pose, self.cfg.goal_x, self.cfg.goal_y)
        observation = build_observation(
            lidar, distance, heading, linear, angular,
            self.cfg.lidar_max, self.cfg.distance_max, self.cfg.action_map.v_max, self.cfg.action_map.omega_max,
        )
        return observation, distance, heading, min_valid_range(lidar, self.cfg.lidar_max)

    def _fatal(self, reason: str) -> List[Effect]:
        self._enter(self.FATAL)
        return [PublishVelocity(0.0, 0.0), Fatal(reason)]

    # ------------------------------------------------------- start episode

    def start_episode(self, spec: EpisodeSpec, now_sim: float) -> List[Effect]:
        if self.state != self.IDLE:
            return self._fatal(f"start_episode received in state {self.state}")
        spec.validate()
        self.episode_key += 1
        training_episode: Optional[int] = None
        scenario: Optional[Scenario] = None
        if spec.phase == "training":
            self.training_episode += 1
            training_episode = self.training_episode
            start = self.sampler.draw(training_start_seed(self.cfg.initialization_seed, training_episode))
            phases = obstacle_phases(training_obstacle_seed(self.cfg.dynamic_obstacle_seed, training_episode), 2, self.cfg.obstacle_half_period)
            init_kind = "train_nu_R"
        elif spec.condition == "E1":
            start = self.sampler.draw(e1_start_seed(self.cfg.evaluation_seed, spec.checkpoint_index, spec.evaluation_episode))
            phases = obstacle_phases(e1_obstacle_seed(self.cfg.evaluation_seed, spec.checkpoint_index, spec.evaluation_episode), 2, self.cfg.obstacle_half_period)
            init_kind = "E1_nu_R"
        elif spec.condition == "E2":
            scenario = scenario_by_id(self.scenarios, spec.scenario_id)
            start = StartPose(scenario.x, scenario.y, scenario.yaw, 0, scenario.start_generator_seed)
            phases = obstacle_phases(scenario.obstacle_phase_seed, 2, self.cfg.obstacle_half_period)
            init_kind = "E2_scenario"
        else:  # E3
            fx, fy, fyaw = self.cfg.fixed_start
            seed = e3_obstacle_seed(self.cfg.evaluation_seed, spec.checkpoint_index, spec.evaluation_episode)
            start = StartPose(fx, fy, fyaw, 0, seed)
            phases = obstacle_phases(seed, 2, self.cfg.obstacle_half_period)
            init_kind = "E3_fixed"
        self.current = _EpisodeState(
            spec=spec, episode_key=self.episode_key, training_episode=training_episode,
            start=start, phases=phases, init_kind=init_kind, scenario=scenario,
        )
        self.current.start_env_step = self.env_step
        self.pending_action = None
        self._applied = None
        self._realized = {}
        self._reset_contact = False
        self._clear_contacts()
        self._enter(self.WAIT_OBSTACLE_HOLD)
        self._obstacle_started_wall = self.wall_clock()
        return [
            ObstacleControl({"cmd": "hold", "episode_key": self.episode_key}),
            PublishVelocity(0.0, 0.0),
        ]

    # ------------------------------------------------ obstacle transaction

    def obstacle_ack(self, payload: Dict[str, object]) -> List[Effect]:
        """Advance only after an acknowledgement for this episode and command."""
        cur = self.current
        try:
            episode_key = int(payload.get("episode_key", -1))
        except (TypeError, ValueError):
            return self._fatal("dynamic-obstacle acknowledgement has an invalid episode_key")
        if cur is None or episode_key != cur.episode_key:
            return []
        if payload.get("status") != "ok":
            reason = str(payload.get("reason", "unknown error"))
            return self._fatal(f"dynamic-obstacle controller rejected command: {reason}")
        cmd = payload.get("cmd")
        if self.state == self.WAIT_OBSTACLE_HOLD and cmd == "hold":
            self._enter(self.RESET_WORLD)
            self._service_started_wall = self.wall_clock()
            return [CallService("reset_world")]
        if self.state == self.WAIT_OBSTACLE_START and cmd == "start":
            now_sim, wall, scan, odom, obstacles = self._reset_snapshot
            return self._publish_initial(now_sim, wall, scan, odom, obstacles)
        return []

    # ------------------------------------------------------ service results

    def service_result(self, kind: str, ok: bool, payload: Optional[Dict[str, object]], now_sim: float) -> List[Effect]:
        cur = self.current
        if cur is None:
            return self._fatal("service result without an active episode")
        if self.state == self.RESET_WORLD and kind == "reset_world":
            if not ok:
                return self._fatal("/reset_world failed")
            self._enter(self.SET_ROBOT)
            self._service_started_wall = self.wall_clock()
            return [CallService("set_entity_state", self.cfg.robot_model_name, (cur.start.x, cur.start.y, cur.start.yaw))]
        if self.state == self.SET_ROBOT and kind == "set_entity_state":
            if not ok:
                return self._fatal("/set_entity_state reported failure for the robot")
            self._set_time = now_sim
            self._enter(self.UNPAUSE_SETTLE)
            self._service_started_wall = self.wall_clock()
            return [PublishVelocity(0.0, 0.0), CallService("unpause_physics")]
        if self.state == self.UNPAUSE_SETTLE and kind == "unpause_physics":
            if not ok:
                return self._fatal("/unpause_physics failed before reset settling")
            self._settle_until = now_sim + self.cfg.settle_sim_s
            self._enter(self.SETTLE)
            return []
        if self.state == self.PAUSE_RESET and kind == "pause_physics":
            if not ok:
                return self._fatal("/pause_physics failed after reset settling")
            self._enter(self.CAPTURE_RESET)
            self._service_started_wall = None
            return []
        if self.state == self.UNPAUSE_ACTION and kind == "unpause_physics":
            if not ok:
                return self._fatal("/unpause_physics failed before action application")
            self._enter(self.HOLD)
            return []
        if self.state == self.PAUSE_ACTION and kind == "pause_physics":
            if not ok:
                return self._fatal("/pause_physics failed after action hold")
            self._enter(self.CAPTURE_ACTION)
            self._service_started_wall = None
            return []
        if self.state in (self.VERIFY_ROBOT, self.VERIFY_OBS) and kind == "get_entity_state":
            if not ok or payload is None:
                return self._fatal("/get_entity_state failed")
            try:
                name = str(payload["name"])
                pose = (float(payload["x"]), float(payload["y"]), float(payload["yaw"]))
            except (KeyError, TypeError, ValueError):
                return self._fatal("/get_entity_state returned a malformed pose")
            if not all(math.isfinite(value) for value in pose):
                return self._fatal("/get_entity_state returned a non-finite pose")
            self._realized[name] = pose
            if self._verify_queue:
                nxt = self._verify_queue.pop(0)
                self._enter(self.VERIFY_OBS)
                self._service_started_wall = self.wall_clock()
                return [CallService("get_entity_state", nxt)]
            return self._finish_verification()
        return self._fatal(f"unexpected service result {kind} in state {self.state}")

    def _finish_verification(self) -> List[Effect]:
        cur = self.current
        cfg = self.cfg
        rx, ry, ryaw = self._realized[cfg.robot_model_name]
        pos_err = math.hypot(rx - cur.start.x, ry - cur.start.y)
        yaw_err = abs(_wrap(ryaw - cur.start.yaw))
        if pos_err > cfg.position_tolerance or yaw_err > cfg.yaw_tolerance:
            return self._fatal(
                f"realized start pose deviates from the request: pos_err={pos_err:.4f} yaw_err={yaw_err:.4f}"
            )
        for name in cfg.obstacle_names:
            ox, oy, _ = self._realized[name]
            ex, ey = self.arena.dynamic_reset_pose(name)
            if math.hypot(ox - ex, oy - ey) > cfg.position_tolerance:
                return self._fatal(f"obstacle {name} not at its reset pose after /reset_world")
        support_required = cur.init_kind != "E3_fixed"
        support_ok = self.sampler.admissible(rx, ry) if support_required else True
        if not support_ok:
            return self._fatal("realized robot pose lies outside the declared initialization support")
        cur.init_row.update({
            "realized_x": rx, "realized_y": ry, "realized_yaw": ryaw,
            "init_pos_error": pos_err, "init_yaw_error": yaw_err,
            "init_support_ok": support_ok,
        })
        now_sim, _, _, _, _ = self._reset_snapshot
        p = cur.phases
        self._enter(self.WAIT_OBSTACLE_START)
        self._obstacle_started_wall = self.wall_clock()
        return [ObstacleControl({
            "cmd": "start", "episode_key": cur.episode_key, "t0": now_sim,
            "names": list(cfg.obstacle_names), "axes": [d.axis for d in self.arena.dynamic_obstacles],
            "signs": list(p.signs), "offsets": list(p.offsets),
            "speed": cfg.obstacle_speed, "half_period": cfg.obstacle_half_period,
        })]

    # --------------------------------------------------------------- action

    def receive_action(self, msg: ActionMessage) -> List[Effect]:
        cur = self.current
        if cur is None or self.state not in (self.AWAIT_ACTION, self.UNPAUSE_ACTION, self.HOLD):
            return []  # stale or premature; ignored
        if msg.sequence != cur.sequence:
            return []  # duplicate or stale sequence
        if self.state != self.AWAIT_ACTION or self.pending_action is not None:
            return []
        if msg.action_index < 0 or msg.action_index >= len(self.commands):
            return self._fatal(f"invalid action index {msg.action_index}")
        if not all(math.isfinite(value) for value in (msg.linear, msg.angular)):
            return self._fatal("action command contains a non-finite velocity")
        expected_linear, expected_angular = self.commands[msg.action_index]
        if abs(msg.linear - expected_linear) > 1e-5 or abs(msg.angular - expected_angular) > 1e-5:
            return self._fatal("action velocities disagree with the frozen action map")
        self.pending_action = msg
        return []

    def _samples_current(
        self,
        now_sim: float,
        scan: Optional[ScanSample],
        odom: Optional[OdomSample],
        obstacles: Sequence[ObstacleSample],
        after: float,
    ) -> bool:
        if scan is None or odom is None or len(obstacles) != 2:
            return False
        if (
            not scan.ranges
            or not all(math.isfinite(value) for value in (scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max))
            or scan.angle_increment == 0.0
            or scan.range_min < 0.0
            or scan.range_max <= scan.range_min
            or not all(math.isfinite(value) for value in (odom.x, odom.y, odom.yaw))
            or {sample.name for sample in obstacles} != set(self.cfg.obstacle_names)
            or not all(math.isfinite(value) for sample in obstacles for value in (sample.x, sample.y))
        ):
            return False
        samples = [scan.stamp, odom.stamp] + [o.stamp for o in obstacles]
        return all(
            math.isfinite(stamp)
            and stamp > after
            and -1e-6 <= now_sim - stamp <= self.cfg.sensor_max_age_sim_s
            for stamp in samples
        )

    # ----------------------------------------------------------------- tick

    def tick(
        self,
        now_sim: float,
        scan: Optional[ScanSample],
        odom: Optional[OdomSample],
        obstacles: Sequence[ObstacleSample],
    ) -> List[Effect]:
        wall = self.wall_clock()
        if not math.isfinite(now_sim) or not math.isfinite(wall):
            return self._fatal("environment clocks must be finite")
        if self.state in (self.IDLE, self.FATAL):
            return []
        cur = self.current
        if self.state in (
            self.RESET_WORLD, self.SET_ROBOT, self.UNPAUSE_SETTLE, self.PAUSE_RESET,
            self.VERIFY_ROBOT, self.VERIFY_OBS, self.UNPAUSE_ACTION, self.PAUSE_ACTION,
        ):
            if self._service_started_wall is not None and wall - self._service_started_wall > self.cfg.service_timeout_wall_s:
                return self._fatal(f"simulator service timed out in state {self.state}")
            return [PublishVelocity(0.0, 0.0)]
        if self.state in (self.WAIT_OBSTACLE_HOLD, self.WAIT_OBSTACLE_START):
            if self._obstacle_started_wall is not None and wall - self._obstacle_started_wall > self.cfg.obstacle_ack_timeout_wall_s:
                return self._fatal(f"dynamic-obstacle acknowledgement timed out in state {self.state}")
            return [PublishVelocity(0.0, 0.0)]
        if self.state == self.CAPTURE_RESET:
            if not self._samples_current(now_sim, scan, odom, obstacles, self._set_time):
                if wall - self._state_entered_wall > self.cfg.sensor_timeout_wall_s:
                    return self._fatal("bounded-age reset sensors unavailable after physics paused")
                return []
            self._reset_snapshot = (now_sim, wall, scan, odom, list(obstacles))
            self._verify_queue = list(self.cfg.obstacle_names)
            self._enter(self.VERIFY_ROBOT)
            self._service_started_wall = wall
            return [CallService("get_entity_state", self.cfg.robot_model_name)]
        if self.state == self.SETTLE:
            if now_sim < self._settle_until:
                return [PublishVelocity(0.0, 0.0)]
            if not self._samples_current(now_sim, scan, odom, obstacles, self._set_time):
                if wall - self._state_entered_wall > self.cfg.sensor_timeout_wall_s:
                    return self._fatal("fresh, bounded-age sensors unavailable after reset settling")
                return [PublishVelocity(0.0, 0.0)]
            self._reset_snapshot = (now_sim, wall, scan, odom, list(obstacles))
            self._enter(self.PAUSE_RESET)
            self._service_started_wall = wall
            return [PublishVelocity(0.0, 0.0), CallService("pause_physics")]
        if self.state == self.AWAIT_ACTION:
            if self.pending_action is not None:
                return self._apply(now_sim, wall)
            if wall - self._await_since_wall > self.cfg.action_timeout_wall_s:
                return self._fatal("no action received from the agent")
            if cur.step_index == 0 and self._last_state_values is not None:
                return [PublishState(list(self._last_state_values))]
            return []
        if self.state == self.HOLD:
            target = self._apply_sim + self.cfg.control_period
            if now_sim + 1e-9 < target:
                return []
            boundary_after = target - 1e-9
            if not self._samples_current(now_sim, scan, odom, obstacles, boundary_after):
                if now_sim - target > self.cfg.hold_tolerance_fraction * self.cfg.control_period:
                    return self._fatal("action hold exceeded its simulation-time tolerance before closure")
                return [PublishVelocity(0.0, 0.0)]
            self._enter(self.PAUSE_ACTION)
            self._service_started_wall = wall
            return [PublishVelocity(0.0, 0.0), CallService("pause_physics")]
        if self.state == self.CAPTURE_ACTION:
            boundary_after = self._apply_sim + self.cfg.control_period - 1e-9
            if not self._samples_current(now_sim, scan, odom, obstacles, boundary_after):
                if wall - self._state_entered_wall > self.cfg.sensor_timeout_wall_s:
                    return self._fatal("bounded-age transition sensors unavailable after physics paused")
                return []
            return self._close(now_sim, wall, scan, odom, list(obstacles))
        return self._fatal(f"tick in unknown state {self.state}")

    # ------------------------------------------------------ initial state

    def _publish_initial(self, now_sim: float, wall: float, scan: ScanSample, odom: OdomSample, obstacles: Sequence[ObstacleSample]) -> List[Effect]:
        cur = self.current
        cfg = self.cfg
        odom_err = math.hypot(odom.x - cur.init_row["realized_x"], odom.y - cur.init_row["realized_y"])
        odom_yaw_err = abs(_wrap(odom.yaw - cur.init_row["realized_yaw"]))
        if odom_err > cfg.odom_tolerance:
            return self._fatal(
                f"/odom disagrees with the simulator pose after the reset (err={odom_err:.4f}); "
                "check the diff-drive odometry_source"
            )
        if odom_yaw_err > cfg.odom_yaw_tolerance:
            return self._fatal(f"/odom yaw disagrees with the simulator pose after reset (err={odom_yaw_err:.4f})")
        self._reset_contact = self.contact_latched
        if self._reset_contact and cfg.fatal_on_reset_contact:
            return self._fatal("contact reported during the reset transaction")
        observation, distance, heading, min_scan = self._observe(scan, odom, 0.0, 0.0)
        support_required = cur.init_kind != "E3_fixed"
        support_ok = (not support_required) or (
            self.sampler.admissible(odom.x, odom.y)
            and min_scan >= self.sampler.law.start_clearance_min
        )
        if not support_ok:
            return self._fatal("post-settle odometry/LiDAR state lies outside the declared initialization support")
        cur.sequence = self.next_sequence
        self.next_sequence += 1
        cur.step_index = 0
        cur.sim_time_start = now_sim
        cur.wall_time_start = wall
        cur.last_pose = (odom.x, odom.y)
        cur.min_clearance = min(cur.min_clearance, min_scan)
        obs_map = {o.name: o for o in obstacles}
        p = cur.phases
        cur.init_row.update({
            "init_kind": cur.init_kind,
            "init_generator_seed": cur.start.generator_seed,
            "requested_x": cur.start.x, "requested_y": cur.start.y, "requested_yaw": cur.start.yaw,
            "odom_x": odom.x, "odom_y": odom.y, "odom_yaw": odom.yaw,
            "init_odom_error": odom_err,
            "init_odom_yaw_error": odom_yaw_err,
            "init_scan_clearance": min_scan,
            "init_support_ok": support_ok,
            "init_tolerance_ok": True,
            "goal_x": cfg.goal_x, "goal_y": cfg.goal_y,
            "obstacle_phase_seed": p.generator_seed,
            "obs1_sign": p.signs[0], "obs1_offset": p.offsets[0], "obs2_sign": p.signs[1], "obs2_offset": p.offsets[1],
            "obs1_reset_x": self._realized[cfg.obstacle_names[0]][0], "obs1_reset_y": self._realized[cfg.obstacle_names[0]][1],
            "obs2_reset_x": self._realized[cfg.obstacle_names[1]][0], "obs2_reset_y": self._realized[cfg.obstacle_names[1]][1],
            "rejection_count": cur.start.rejection_count,
            "reset_status": "ok",
            "settle_sim_s": cfg.settle_sim_s,
            "contact_during_reset": self._reset_contact,
            "init_sim_time": now_sim,
        })
        self._clear_contacts()
        state_values = encode_state(cur.sequence, cur.episode_key, 0, now_sim, observation)
        self._last_state_values = state_values
        self._last_observation = list(observation)
        self._last_state_sim = now_sim
        self._last_state_wall = wall
        self._last_odom_stamp = odom.stamp
        self._last_distance = distance
        row = self._base_transition_row(now_sim)
        row.update({
            "step_in_episode": 0, "env_step": self.env_step if cur.spec.phase == "training" else None,
            "sim_time": now_sim, "state_sim_time": now_sim,
            "min_lidar": min_scan, "distance": distance, "heading_error": heading,
            "x": odom.x, "y": odom.y, "yaw": odom.yaw,
            "obs1_x": obs_map[cfg.obstacle_names[0]].x, "obs1_y": obs_map[cfg.obstacle_names[0]].y,
            "obs2_x": obs_map[cfg.obstacle_names[1]].x, "obs2_y": obs_map[cfg.obstacle_names[1]].y,
        })
        for i, v in enumerate(observation):
            row[f"obs_{i}"] = float(v)
        self._await_since_wall = wall
        self._enter(self.AWAIT_ACTION)
        return [RecordTransition(row), PublishState(state_values)]

    def _base_transition_row(self, now_sim: float) -> Dict[str, object]:
        cur = self.current
        spec = cur.spec
        return {
            "phase": spec.phase, "policy_mode": spec.policy_mode, "episode_key": cur.episode_key,
            "training_episode": cur.training_episode,
            "checkpoint_step": spec.checkpoint_step, "condition": spec.condition,
            "scenario_id": spec.scenario_id, "evaluation_episode": spec.evaluation_episode,
        }

    # ------------------------------------------------------------- apply

    def _apply(self, now_sim: float, wall: float) -> List[Effect]:
        msg = self.pending_action
        self.pending_action = None
        self._applied = msg
        self._decision_gap_sim = now_sim - self._last_state_sim
        self._decision_latency_wall = wall - self._last_state_wall
        if abs(self._decision_gap_sim) > self.cfg.decision_gap_tolerance_sim_s:
            return self._fatal(f"world advanced {self._decision_gap_sim:.6f}s while the policy selected an action")
        self._apply_sim = self._last_state_sim
        self._apply_odom_stamp = self._last_odom_stamp
        self._previous_distance = self._last_distance
        self._enter(self.UNPAUSE_ACTION)
        self._service_started_wall = wall
        return [PublishVelocity(msg.linear, msg.angular), CallService("unpause_physics")]

    # ------------------------------------------------------------- close

    def _obstacle_error(self, now_sim: float, obstacles: Sequence[ObstacleSample]) -> float:
        cur = self.current
        elapsed = max(0.0, now_sim - float(cur.init_row["init_sim_time"]))
        observed = {o.name: o for o in obstacles}
        errors: List[float] = []
        for index, dynamic in enumerate(self.arena.dynamic_obstacles):
            dx, dy = obstacle_displacement(
                dynamic.axis, cur.phases.signs[index], cur.phases.offsets[index],
                self.cfg.obstacle_speed, self.cfg.obstacle_half_period, elapsed,
            )
            reset_x = float(cur.init_row[f"obs{index + 1}_reset_x"])
            reset_y = float(cur.init_row[f"obs{index + 1}_reset_y"])
            sample = observed[dynamic.name]
            errors.append(math.hypot(sample.x - (reset_x + dx), sample.y - (reset_y + dy)))
        return max(errors)

    def _close(self, now_sim: float, wall: float, scan: Optional[ScanSample], odom: Optional[OdomSample], obstacles: Sequence[ObstacleSample]) -> List[Effect]:
        cur = self.current
        cfg = self.cfg
        if scan is None or odom is None or len(obstacles) != 2:
            return self._fatal("sensor sample missing while closing a transition")
        msg = self._applied
        obstacle_error = self._obstacle_error(now_sim, obstacles)
        if obstacle_error > cfg.obstacle_position_tolerance:
            return self._fatal(
                f"dynamic obstacle departed from its seeded trajectory (max error={obstacle_error:.4f})"
            )
        observation, distance, heading, min_scan = self._observe(scan, odom, msg.linear, msg.angular)
        result = compute_reward(self._previous_distance, distance, min_scan, msg.angular, self.contact_latched, cfg.reward)
        static_hit, dynamic_hit = self.static_latched, self.dynamic_latched
        self._clear_contacts()
        cur.step_index += 1
        cur.length += 1
        truncated = (not result.terminated) and (cur.step_index >= cfg.episode_max_steps or msg.final_transition)
        episode_end = result.terminated or truncated
        hold_sim = now_sim - self._apply_sim
        hold_odom = odom.stamp - self._apply_odom_stamp
        scan_age = now_sim - scan.stamp
        odom_age = now_sim - odom.stamp
        obs_map = {o.name: o for o in obstacles}
        o1, o2 = obs_map[cfg.obstacle_names[0]], obs_map[cfg.obstacle_names[1]]
        obs1_age = now_sim - o1.stamp
        obs2_age = now_sim - o2.stamp
        cur.holds.append(hold_sim)
        cur.decision_latencies.append(self._decision_latency_wall)
        cur.decision_gaps.append(self._decision_gap_sim)
        cur.sensor_ages.extend((scan_age, odom_age, obs1_age, obs2_age))
        cur.obstacle_errors.append(obstacle_error)
        if abs(hold_odom - cfg.control_period) > cfg.hold_tolerance_fraction * cfg.control_period:
            cur.hold_out_of_tol += 1
        cur.ret += result.total
        for key, value in (
            ("r_distance", result.distance_progress), ("r_step", result.step_penalty), ("r_collision", result.collision_penalty),
            ("r_goal", result.goal_bonus), ("r_angular", result.angular_penalty), ("r_near", result.near_penalty),
        ):
            cur.sums[key] += value
        if result.near_penalty < 0.0:
            cur.near_penalty_steps += 1
        cur.min_clearance = min(cur.min_clearance, min_scan)
        if cur.last_pose is not None:
            cur.path_length += math.hypot(odom.x - cur.last_pose[0], odom.y - cur.last_pose[1])
        cur.last_pose = (odom.x, odom.y)
        if cur.spec.phase == "training":
            self.env_step += 1
        row = self._base_transition_row(now_sim)
        row.update({
            "step_in_episode": cur.step_index, "env_step": self.env_step if cur.spec.phase == "training" else None,
            "sim_time": now_sim, "state_sim_time": self._last_state_sim,
            "hold_sim_s": hold_sim, "hold_odom_s": hold_odom,
            "scan_age_s": scan_age, "odom_age_s": odom_age, "obs1_age_s": obs1_age, "obs2_age_s": obs2_age,
            "decision_gap_sim_s": self._decision_gap_sim,
            "decision_latency_wall_s": self._decision_latency_wall,
            "obstacle_position_error_max": obstacle_error,
            "action_index": msg.action_index, "action_name": ACTION_NAMES[msg.action_index],
            "linear_cmd": msg.linear, "angular_cmd": msg.angular,
            "reward_total": result.total, "r_distance": result.distance_progress, "r_step": result.step_penalty,
            "r_collision": result.collision_penalty, "r_goal": result.goal_bonus, "r_angular": result.angular_penalty,
            "r_near": result.near_penalty,
            "terminated": result.terminated, "episode_end": episode_end, "truncated": truncated,
            "collision": result.collision, "static_collision": static_hit, "dynamic_collision": dynamic_hit,
            "safety": result.safety, "goal": result.goal,
            "min_lidar": min_scan, "distance": distance, "heading_error": heading,
            "x": odom.x, "y": odom.y, "yaw": odom.yaw,
            "obs1_x": o1.x, "obs1_y": o1.y, "obs2_x": o2.x, "obs2_y": o2.y,
        })
        for i, v in enumerate(observation):
            row[f"obs_{i}"] = float(v)
        step_values = encode_step(
            cur.sequence, cur.episode_key, cur.step_index, now_sim, hold_sim, hold_odom,
            scan_age, odom_age, obs1_age, obs2_age,
            self._decision_gap_sim, self._decision_latency_wall, obstacle_error, result,
            episode_end, truncated, static_hit, dynamic_hit, min_scan, distance, heading,
            (odom.x, odom.y, odom.yaw), ((o1.x, o1.y), (o2.x, o2.y)), observation,
        )
        effects: List[Effect] = [PublishVelocity(0.0, 0.0), RecordTransition(row), PublishStep(step_values)]
        self._last_observation = list(observation)
        if episode_end:
            effects.extend(self._finish_episode(now_sim, wall, result, truncated, static_hit, dynamic_hit))
        else:
            self._last_state_sim = now_sim
            self._last_state_wall = wall
            self._last_odom_stamp = odom.stamp
            self._last_distance = distance
            cur.sequence = self.next_sequence
            self.next_sequence += 1
            self._await_since_wall = wall
            self._enter(self.AWAIT_ACTION)
        return effects

    # ------------------------------------------------------------ finish

    def _finish_episode(self, now_sim: float, wall: float, result, truncated: bool, static_hit: bool, dynamic_hit: bool) -> List[Effect]:
        cur = self.current
        cfg = self.cfg
        if result.collision:
            outcome = "collision_both" if (static_hit and dynamic_hit) else ("collision_static" if static_hit else ("collision_dynamic" if dynamic_hit else "collision_unknown"))
        elif result.safety:
            outcome = "safety"
        elif result.goal:
            outcome = "goal"
        else:
            outcome = "timeout"
        wall_s = max(wall - cur.wall_time_start, 1e-9)
        straight = math.hypot(cur.start.x - cfg.goal_x, cur.start.y - cfg.goal_y)
        efficiency = (straight / cur.path_length) if (result.goal and cur.path_length > 0.0) else None
        summary: Dict[str, object] = {
            "length": cur.length, "return": cur.ret,
            "sum_r_distance": cur.sums["r_distance"], "sum_r_step": cur.sums["r_step"], "sum_r_collision": cur.sums["r_collision"],
            "sum_r_goal": cur.sums["r_goal"], "sum_r_angular": cur.sums["r_angular"], "sum_r_near": cur.sums["r_near"],
            "outcome": outcome, "collision": result.collision, "static_collision": static_hit, "dynamic_collision": dynamic_hit,
            "safety": result.safety, "goal": result.goal, "truncated": truncated,
            "near_penalty_steps": cur.near_penalty_steps, "min_clearance": cur.min_clearance,
            "path_length": cur.path_length, "straight_line_distance": straight, "path_efficiency": efficiency,
            "time_to_goal_steps": cur.length if result.goal else None,
            "sim_time_start": cur.sim_time_start, "sim_time_end": now_sim, "wall_time_s": wall_s,
            "rtf": (now_sim - cur.sim_time_start) / wall_s,
            "mean_hold_sim_s": (sum(cur.holds) / len(cur.holds)) if cur.holds else None,
            "max_hold_sim_s": max(cur.holds) if cur.holds else None,
            "hold_out_of_tolerance_steps": cur.hold_out_of_tol,
            "mean_decision_latency_wall_s": (sum(cur.decision_latencies) / len(cur.decision_latencies)) if cur.decision_latencies else None,
            "max_decision_latency_wall_s": max(cur.decision_latencies) if cur.decision_latencies else None,
            "max_decision_gap_sim_s": max((abs(v) for v in cur.decision_gaps), default=None),
            "max_sensor_age_s": max(cur.sensor_ages) if cur.sensor_ages else None,
            "max_obstacle_position_error": max(cur.obstacle_errors) if cur.obstacle_errors else None,
        }
        assert set(summary) == set(EPISODE_OUTCOME)
        init = {k: cur.init_row.get(k) for k in INIT_BLOCK}
        effects: List[Effect] = []
        if cur.spec.phase == "training":
            row = {"episode_key": cur.episode_key, "training_episode": cur.training_episode,
                   "start_env_step": cur.start_env_step, "end_env_step": self.env_step}
            row.update(summary)
            row.update(init)
            effects.append(RecordEpisode(row))
        payload: Dict[str, object] = {
            "cmd": "episode_summary", "episode_key": cur.episode_key, "phase": cur.spec.phase,
            "policy_mode": cur.spec.policy_mode, "training_episode": cur.training_episode,
            "condition": cur.spec.condition, "checkpoint_step": cur.spec.checkpoint_step,
            "checkpoint_index": cur.spec.checkpoint_index, "scenario_id": cur.spec.scenario_id,
            "evaluation_episode": cur.spec.evaluation_episode, "env_step": self.env_step,
            "outcome_block": summary, "init_block": init,
        }
        if cur.scenario is not None:
            payload["scenario_labels"] = {
                "distance_bin": cur.scenario.distance_bin, "clearance_bin": cur.scenario.clearance_bin,
                "heading_bin": cur.scenario.heading_bin, "difficulty": cur.scenario.difficulty,
            }
        effects.append(PublishEpisodeSummary(payload))
        self._last_state_values = None
        self.current = None
        self._enter(self.IDLE)
        return effects
