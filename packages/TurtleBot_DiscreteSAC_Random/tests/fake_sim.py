"""A tiny kinematic stand-in for Gazebo used by the ROS-free engine tests.

It integrates unicycle motion for the robot, moves the two planar obstacles on
the same schedule as dynamic_obstacle_node, ray-casts a coarse scan against
the parsed arena, reports contacts from footprint overlap, and answers the three
simulator services the engine uses. It is deliberately simple; it exists to
exercise every branch of the episode engine, not to replace Gazebo evidence.
"""

import math
from typing import Dict, List, Optional, Tuple

from turtlebot3_drl_nav.episode_engine import (
    CallService,
    EpisodeEngine,
    Fatal,
    ObstacleControl,
    ObstacleSample,
    OdomSample,
    PublishEpisodeSummary,
    PublishState,
    PublishStep,
    PublishVelocity,
    RecordEpisode,
    RecordTransition,
    ScanSample,
)
from turtlebot3_drl_nav.geometry import Arena, Box2D, Cylinder2D
from turtlebot3_drl_nav.obstacle_schedule import obstacle_velocity


class FakeSim:
    def __init__(self, arena: Arena, control_period: float, footprint: float = 0.12, beams: int = 72) -> None:
        self.arena = arena
        self.dt = control_period
        self.footprint = footprint
        self.beams = beams
        self.sim_time = 100.0
        self.robot = [0.0, 0.0, 0.0]
        self.cmd = (0.0, 0.0)
        self.obstacles: Dict[str, List[float]] = {
            d.name: [d.shape.cx, d.shape.cy] for d in arena.dynamic_obstacles
        }
        self.axes = {d.name: d.axis for d in arena.dynamic_obstacles}
        self.schedule: Optional[Dict[str, object]] = None
        self.set_state_ok = True
        self.pose_perturbation = (0.0, 0.0, 0.0)
        self.odom_offset = (0.0, 0.0)
        self.services: List[Tuple[str, Optional[str]]] = []
        self.states: List[List[float]] = []
        self.steps: List[List[float]] = []
        self.transitions: List[Dict[str, object]] = []
        self.episodes: List[Dict[str, object]] = []
        self.summaries: List[Dict[str, object]] = []
        self.fatal: Optional[str] = None
        self.wall = 0.0
        self.force_contact = False
        self.paused = False
        self.odom_yaw_offset = 0.0

    # ------------------------------------------------------------ physics

    def step_physics(self) -> None:
        if self.paused:
            self.wall += self.dt / 4.0
            return
        v, w = self.cmd
        x, y, yaw = self.robot
        if abs(w) < 1e-9:
            x += v * self.dt * math.cos(yaw)
            y += v * self.dt * math.sin(yaw)
        else:
            x += (v / w) * (math.sin(yaw + w * self.dt) - math.sin(yaw))
            y -= (v / w) * (math.cos(yaw + w * self.dt) - math.cos(yaw))
            yaw = math.atan2(math.sin(yaw + w * self.dt), math.cos(yaw + w * self.dt))
        self.robot = [x, y, yaw]
        if self.schedule is not None:
            elapsed = self.sim_time - float(self.schedule["t0"])
            for index, name in enumerate(self.obstacles):
                vx, vy = obstacle_velocity(
                    self.axes[name], float(self.schedule["signs"][index]), float(self.schedule["offsets"][index]),
                    float(self.schedule["speed"]), float(self.schedule["half_period"]), elapsed,
                )
                self.obstacles[name][0] += vx * self.dt
                self.obstacles[name][1] += vy * self.dt
        self.sim_time += self.dt
        self.wall += self.dt / 4.0  # pretend RTF 4

    def shapes(self):
        out = list(self.arena.static_shapes())
        for d in self.arena.dynamic_obstacles:
            cx, cy = self.obstacles[d.name]
            if isinstance(d.shape, Cylinder2D):
                out.append(Cylinder2D(d.name, cx, cy, d.shape.radius))
            else:
                out.append(Box2D(d.name, cx, cy, d.shape.half_x, d.shape.half_y, d.shape.yaw))
        return out

    def scan(self) -> ScanSample:
        x, y, yaw = self.robot
        shapes = self.shapes()
        ranges = []
        increment = 2.0 * math.pi / self.beams
        for i in range(self.beams):
            bearing = yaw + i * increment  # Burger LDS: angle_min = 0, beam 0 points forward
            r, hit = 0.12, False
            while r <= 3.5:
                px, py = x + r * math.cos(bearing), y + r * math.sin(bearing)
                if any(s.distance(px, py) == 0.0 for s in shapes):
                    hit = True
                    break
                r += 0.01 if r < 0.6 else 0.04  # fine near the safety band, coarse far away
            ranges.append(r if hit else float("inf"))
        return ScanSample(tuple(ranges), 0.0, increment, 0.12, 3.5, self.sim_time)

    def odom(self) -> OdomSample:
        x, y, yaw = self.robot
        return OdomSample(x + self.odom_offset[0], y + self.odom_offset[1], yaw + self.odom_yaw_offset, self.sim_time)

    def obstacle_samples(self) -> List[ObstacleSample]:
        return [ObstacleSample(name, p[0], p[1], self.sim_time) for name, p in self.obstacles.items()]

    def contact(self) -> Tuple[bool, bool, bool]:
        if self.force_contact:
            return True, True, False
        x, y, _ = self.robot
        static = any(s.distance(x, y) < self.footprint for s in self.arena.static_shapes())
        dynamic = any(s.distance(x, y) < self.footprint for s in self.shapes()[len(self.arena.static_shapes()):])
        return static or dynamic, static, dynamic

    # ------------------------------------------------------------ services

    def answer_service(self, engine: EpisodeEngine, call: CallService):
        self.services.append((call.kind, call.entity))
        if call.kind == "reset_world":
            self.robot = [0.0, 0.0, 0.0]
            self.cmd = (0.0, 0.0)
            self.schedule = None
            for d in self.arena.dynamic_obstacles:
                self.obstacles[d.name] = [d.shape.cx, d.shape.cy]
            return engine.service_result("reset_world", True, None, self.sim_time)
        if call.kind == "pause_physics":
            self.paused = True
            return engine.service_result("pause_physics", True, None, self.sim_time)
        if call.kind == "unpause_physics":
            self.paused = False
            return engine.service_result("unpause_physics", True, None, self.sim_time)
        if call.kind == "set_entity_state":
            if not self.set_state_ok:
                return engine.service_result("set_entity_state", False, None, self.sim_time)
            px, py, pyaw = call.pose
            dx, dy, dyaw = self.pose_perturbation
            self.robot = [px + dx, py + dy, pyaw + dyaw]
            return engine.service_result("set_entity_state", True, None, self.sim_time)
        if call.kind == "get_entity_state":
            if call.entity == "burger":
                x, y, yaw = self.robot
            else:
                x, y = self.obstacles[call.entity]
                yaw = 0.0
            return engine.service_result("get_entity_state", True, {"name": call.entity, "x": x, "y": y, "yaw": yaw}, self.sim_time)
        raise AssertionError(call.kind)

    # ------------------------------------------------------------- driver

    def apply_obstacle_control(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Apply a command and return its acknowledgement to the caller.

        The closed-loop harness must dispatch the effects emitted by
        ``engine.obstacle_ack`` itself, because those effects can include the
        canonical step-zero transition. Direct engine tests still use
        ``apply_effects`` and retain their in-memory recorder path.
        """
        if payload["cmd"] == "start":
            self.schedule = dict(payload)
        else:
            self.schedule = None
        return {"cmd": payload["cmd"], "episode_key": payload["episode_key"], "status": "ok"}

    def apply_effects(self, engine: EpisodeEngine, effects) -> None:
        for effect in effects:
            if isinstance(effect, PublishVelocity):
                self.cmd = (effect.linear, effect.angular)
            elif isinstance(effect, PublishState):
                self.states.append(effect.values)
            elif isinstance(effect, PublishStep):
                self.steps.append(effect.values)
            elif isinstance(effect, RecordTransition):
                self.transitions.append(effect.row)
            elif isinstance(effect, RecordEpisode):
                self.episodes.append(effect.row)
            elif isinstance(effect, PublishEpisodeSummary):
                self.summaries.append(effect.payload)
            elif isinstance(effect, ObstacleControl):
                ack = self.apply_obstacle_control(effect.payload)
                self.apply_effects(engine, engine.obstacle_ack(ack))
            elif isinstance(effect, CallService):
                self.apply_effects(engine, self.answer_service(engine, effect))
            elif isinstance(effect, Fatal):
                self.fatal = effect.reason
            else:
                raise AssertionError(f"unknown effect {effect}")

    def tick(self, engine: EpisodeEngine) -> None:
        self.step_physics()
        c, s, d = self.contact()
        engine.note_contact(c, s, d)
        self.apply_effects(engine, engine.tick(self.sim_time, self.scan(), self.odom(), self.obstacle_samples()))
