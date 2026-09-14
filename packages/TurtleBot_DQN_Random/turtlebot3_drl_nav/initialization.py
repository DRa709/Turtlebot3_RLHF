"""The random initial-state distribution nu_R, its seeded episode-indexed
realization, and the evaluation scenario list.

Shared layer. No ROS dependency. The declared law is RANDOM_INIT_SPEC.md; this
module is its executable form. Every draw is a pure function of
(seed identity, stream tag, episode index): two runs with equal seeds produce
identical sequences, and the draw for episode e never depends on how many
rejections earlier episodes consumed.
"""

import csv
import hashlib
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import Arena, Box2D, clearance, segment_intersects_box

STREAM_TRAIN_START = "train_start_pose"
STREAM_TRAIN_OBSTACLE = "train_obstacle_phase"
STREAM_E1_START = "E1_start_pose"
STREAM_E1_OBSTACLE = "E1_obstacle_phase"
STREAM_E3_OBSTACLE = "E3_obstacle_phase"
STREAM_SCENARIOS = "scenario_list"

PLACEHOLDER_SEED = -1


def derive_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from an ordered tuple of identities."""
    text = ":".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class InitializationLaw:
    """Numerical definition of nu_R over the robot start pose."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    start_clearance_min: float      # centre-to-surface, metres; >= d_safe so no start is in the near band
    goal_exclusion_radius: float
    robot_footprint_radius: float   # must be < start_clearance_min
    obstacle_amplitude: float       # speed * half_period: swept half-length of a dynamic obstacle
    max_rejections: int
    goal_x: float
    goal_y: float

    def validate(self) -> None:
        values = (
            self.x_min, self.x_max, self.y_min, self.y_max, self.start_clearance_min,
            self.goal_exclusion_radius, self.robot_footprint_radius, self.obstacle_amplitude,
            self.goal_x, self.goal_y,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("initialization-law values must be finite")
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("initialization box is empty")
        if self.start_clearance_min <= 0.0 or self.robot_footprint_radius <= 0.0:
            raise ValueError("start clearance and robot footprint radius must be positive")
        if self.robot_footprint_radius >= self.start_clearance_min:
            raise ValueError("robot_footprint_radius must be below start_clearance_min")
        if self.goal_exclusion_radius <= 0.0 or self.max_rejections <= 0:
            raise ValueError("goal_exclusion_radius and max_rejections must be positive")
        if self.obstacle_amplitude < 0.0:
            raise ValueError("obstacle_amplitude must be non-negative")


@dataclass(frozen=True)
class StartPose:
    x: float
    y: float
    yaw: float
    rejection_count: int
    generator_seed: int


@dataclass(frozen=True)
class ObstaclePhases:
    """Per-episode parameters of the two planar-move obstacles."""

    generator_seed: int
    signs: Tuple[float, ...]
    offsets: Tuple[float, ...]   # seconds of simulation time, in [0, half_period)


class Sampler:
    """Rejection sampler realizing nu_R exactly: uniform proposals on the box,
    accepted iff every declared constraint holds, so accepted points are uniform
    on the admissible region and yaw is uniform on (-pi, pi]."""

    def __init__(self, law: InitializationLaw, arena: Arena) -> None:
        law.validate()
        self.law = law
        self.arena = arena
        self.static_shapes = arena.static_shapes()
        self.corridors: List[Box2D] = [
            obstacle.swept_corridor(law.obstacle_amplitude) for obstacle in arena.dynamic_obstacles
        ]

    def admissible(self, x: float, y: float, margin: float = 0.0) -> bool:
        law = self.law
        if not ((law.x_min - margin) <= x <= (law.x_max + margin) and (law.y_min - margin) <= y <= (law.y_max + margin)):
            return False
        if math.hypot(x - law.goal_x, y - law.goal_y) < law.goal_exclusion_radius:
            return False
        if clearance(x, y, self.static_shapes) < (law.start_clearance_min - margin):
            return False
        if clearance(x, y, self.corridors) < (law.start_clearance_min - margin):
            return False
        return True

    def start_clearance(self, x: float, y: float) -> float:
        """Centre-to-surface clearance against static geometry and the obstacle
        footprints at their reset poses (what the first LiDAR scan sees)."""
        shapes = list(self.static_shapes) + [d.shape for d in self.arena.dynamic_obstacles]
        return clearance(x, y, shapes)

    def draw(self, generator_seed: int) -> StartPose:
        rng = random.Random(generator_seed)
        law = self.law
        for rejections in range(law.max_rejections + 1):
            x = rng.uniform(law.x_min, law.x_max)
            y = rng.uniform(law.y_min, law.y_max)
            if self.admissible(x, y):
                yaw = rng.uniform(-math.pi, math.pi)
                return StartPose(x, y, yaw, rejections, generator_seed)
        raise RuntimeError(
            f"start-pose sampler exhausted {law.max_rejections} rejections for seed {generator_seed}"
        )

    def difficulty(self, x: float, y: float, goal_x: float, goal_y: float) -> str:
        """'obstructed' if the straight start-goal segment crosses a static box or a
        dynamic corridor, else 'open'."""
        for box in list(self.arena.static_obstacles) + self.corridors:
            if segment_intersects_box(x, y, goal_x, goal_y, box):
                return "obstructed"
        return "open"


def obstacle_phases(generator_seed: int, count: int, half_period: float) -> ObstaclePhases:
    if count <= 0 or not math.isfinite(half_period) or half_period <= 0.0:
        raise ValueError("obstacle phase count and half-period must be positive")
    rng = random.Random(generator_seed)
    signs = tuple(rng.choice((-1.0, 1.0)) for _ in range(count))
    offsets = tuple(rng.uniform(0.0, half_period) for _ in range(count))
    return ObstaclePhases(generator_seed, signs, offsets)


def training_start_seed(initialization_seed: int, training_episode: int) -> int:
    return derive_seed(initialization_seed, STREAM_TRAIN_START, training_episode)


def training_obstacle_seed(dynamic_obstacle_seed: int, training_episode: int) -> int:
    return derive_seed(dynamic_obstacle_seed, STREAM_TRAIN_OBSTACLE, training_episode)


def e1_start_seed(evaluation_seed: int, checkpoint_index: int, evaluation_episode: int) -> int:
    return derive_seed(evaluation_seed, STREAM_E1_START, checkpoint_index, evaluation_episode)


def e1_obstacle_seed(evaluation_seed: int, checkpoint_index: int, evaluation_episode: int) -> int:
    return derive_seed(evaluation_seed, STREAM_E1_OBSTACLE, checkpoint_index, evaluation_episode)


def e3_obstacle_seed(evaluation_seed: int, checkpoint_index: int, evaluation_episode: int) -> int:
    return derive_seed(evaluation_seed, STREAM_E3_OBSTACLE, checkpoint_index, evaluation_episode)


# ---------------------------------------------------------------- scenarios

SCENARIO_COLUMNS = [
    "scenario_id", "start_generator_seed", "x", "y", "yaw", "goal_x", "goal_y", "obstacle_phase_seed",
    "straight_line_distance", "start_clearance", "heading_error", "difficulty",
    "distance_bin", "clearance_bin", "heading_bin", "physical_subset",
]

DISTANCE_EDGES = (0.5, 1.5, 2.5, 3.5, math.inf)
CLEARANCE_EDGES = (0.30, 0.50, 0.80, math.inf)
HEADING_EDGES = (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0, math.pi + 1e-9)


def _bin(value: float, edges: Sequence[float]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    raise ValueError(f"value {value} outside bins {edges}")


def heading_error(x: float, y: float, yaw: float, goal_x: float, goal_y: float) -> float:
    angle = math.atan2(goal_y - y, goal_x - x) - yaw
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    start_generator_seed: int
    x: float
    y: float
    yaw: float
    goal_x: float
    goal_y: float
    obstacle_phase_seed: int
    straight_line_distance: float
    start_clearance: float
    heading_error: float
    difficulty: str
    distance_bin: int
    clearance_bin: int
    heading_bin: int
    physical_subset: int

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def generate_scenarios(
    sampler: Sampler,
    evaluation_seed: int,
    count: int,
    physical_count: int,
    half_period: float,
) -> List[Scenario]:
    """Stratified held-out list: equal quotas over (distance_bin x heading_bin)
    cells, filled in seed order by rejection, so the list is a pure function of
    (evaluation_seed, count). Clearance bin and difficulty are labels."""
    law = sampler.law
    cells = [(d, h) for d in range(len(DISTANCE_EDGES) - 1) for h in range(len(HEADING_EDGES) - 1)]
    base, extra = divmod(count, len(cells))
    quotas = {cell: base + (1 if index < extra else 0) for index, cell in enumerate(cells)}
    filled = {cell: 0 for cell in cells}
    scenarios: List[Scenario] = []
    draw_index = 0
    attempts = 0
    while len(scenarios) < count:
        attempts += 1
        if attempts > law.max_rejections * max(1, count):
            raise RuntimeError("scenario generation exhausted its attempt budget")
        seed = derive_seed(evaluation_seed, STREAM_SCENARIOS, draw_index)
        draw_index += 1
        pose = sampler.draw(seed)
        distance = math.hypot(pose.x - law.goal_x, pose.y - law.goal_y)
        chi = heading_error(pose.x, pose.y, pose.yaw, law.goal_x, law.goal_y)
        cell = (_bin(distance, DISTANCE_EDGES), _bin(abs(chi), HEADING_EDGES))
        if filled[cell] >= quotas[cell]:
            continue
        filled[cell] += 1
        clear = sampler.start_clearance(pose.x, pose.y)
        scenarios.append(
            Scenario(
                scenario_id=f"S{len(scenarios) + 1:03d}",
                start_generator_seed=pose.generator_seed,
                x=round(pose.x, 6),
                y=round(pose.y, 6),
                yaw=round(pose.yaw, 6),
                goal_x=law.goal_x,
                goal_y=law.goal_y,
                obstacle_phase_seed=derive_seed(evaluation_seed, "E2_obstacle_phase", len(scenarios) + 1),
                straight_line_distance=round(distance, 6),
                start_clearance=round(clear, 6),
                heading_error=round(chi, 6),
                difficulty=sampler.difficulty(pose.x, pose.y, law.goal_x, law.goal_y),
                distance_bin=cell[0],
                clearance_bin=_bin(clear, CLEARANCE_EDGES),
                heading_bin=cell[1],
                physical_subset=0,
            )
        )
    # Physical subset: the first scenario of each cell in cell order, until physical_count.
    marked = 0
    ordered: List[Scenario] = []
    seen_cells = set()
    for scenario in scenarios:
        cell = (scenario.distance_bin, scenario.heading_bin)
        physical = 0
        if cell not in seen_cells and marked < physical_count:
            seen_cells.add(cell)
            marked += 1
            physical = 1
        ordered.append(Scenario(**{**scenario.to_row(), "physical_subset": physical}))
    del half_period  # obstacle timing is fixed by the shared config; kept for signature stability
    return ordered


def write_scenarios(path: str, scenarios: Sequence[Scenario]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCENARIO_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow(scenario.to_row())


def read_scenarios(path: str) -> List[Scenario]:
    out: List[Scenario] = []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SCENARIO_COLUMNS:
            raise ValueError(f"scenario file {path} has unexpected columns {reader.fieldnames}")
        for row in reader:
            try:
                scenario = Scenario(
                    scenario_id=row["scenario_id"],
                    start_generator_seed=int(row["start_generator_seed"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    yaw=float(row["yaw"]),
                    goal_x=float(row["goal_x"]),
                    goal_y=float(row["goal_y"]),
                    obstacle_phase_seed=int(row["obstacle_phase_seed"]),
                    straight_line_distance=float(row["straight_line_distance"]),
                    start_clearance=float(row["start_clearance"]),
                    heading_error=float(row["heading_error"]),
                    difficulty=row["difficulty"],
                    distance_bin=int(row["distance_bin"]),
                    clearance_bin=int(row["clearance_bin"]),
                    heading_bin=int(row["heading_bin"]),
                    physical_subset=int(row["physical_subset"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"scenario file {path} has a malformed row: {error}")
            numeric = (
                scenario.x, scenario.y, scenario.yaw, scenario.goal_x, scenario.goal_y,
                scenario.straight_line_distance, scenario.start_clearance, scenario.heading_error,
            )
            if (
                not all(math.isfinite(value) for value in numeric)
                or scenario.start_generator_seed < 0
                or scenario.obstacle_phase_seed < 0
                or scenario.straight_line_distance <= 0.0
                or scenario.start_clearance < 0.0
                or not -math.pi <= scenario.yaw <= math.pi
                or not -math.pi <= scenario.heading_error <= math.pi
                or scenario.difficulty not in ("open", "obstructed")
                or scenario.distance_bin not in range(len(DISTANCE_EDGES) - 1)
                or scenario.clearance_bin not in range(len(CLEARANCE_EDGES) - 1)
                or scenario.heading_bin not in range(len(HEADING_EDGES) - 1)
                or scenario.physical_subset not in (0, 1)
            ):
                raise ValueError(f"scenario file {path} has an invalid row {scenario.scenario_id!r}")
            out.append(scenario)
    ids = [s.scenario_id for s in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")
    expected_ids = [f"S{index:03d}" for index in range(1, len(out) + 1)]
    if ids != expected_ids:
        raise ValueError("scenario ids must be contiguous and ordered from S001")
    return out


def scenario_by_id(scenarios: Sequence[Scenario], scenario_id: str) -> Scenario:
    for scenario in scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(scenario_id)


def law_from_parameters(params: Dict[str, object], obstacle_speed: float, obstacle_half_period: float) -> InitializationLaw:
    """Build the law from the flat parameter dictionary of common_environment.yaml."""
    return InitializationLaw(
        x_min=float(params["init_x_min"]),
        x_max=float(params["init_x_max"]),
        y_min=float(params["init_y_min"]),
        y_max=float(params["init_y_max"]),
        start_clearance_min=float(params["init_start_clearance_min"]),
        goal_exclusion_radius=float(params["init_goal_exclusion_radius"]),
        robot_footprint_radius=float(params["init_robot_footprint_radius"]),
        obstacle_amplitude=float(obstacle_speed) * float(obstacle_half_period),
        max_rejections=int(params["init_max_rejections"]),
        goal_x=float(params["goal_x"]),
        goal_y=float(params["goal_y"]),
    )


def require_seed(name: str, value: int) -> int:
    value = int(value)
    if value == PLACEHOLDER_SEED or value < 0:
        raise ValueError(f"{name} is unset (placeholder {PLACEHOLDER_SEED}); the dispatcher must forward it")
    return value
