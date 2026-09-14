"""Arena geometry parsed from the frozen Gazebo world, and the clearance rules
of the random initial-state distribution.

Shared layer. No ROS dependency. Everything the start-pose sampler and the
evaluation-scenario generator know about the arena comes from here, and here it
is read from the same ``.world`` file the simulator loads, so the sampler and
the simulated arena cannot drift apart.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Box2D:
    """Axis-aligned-in-body-frame rectangle with a yaw, from a Gazebo box collision."""

    name: str
    cx: float
    cy: float
    half_x: float
    half_y: float
    yaw: float

    def distance(self, px: float, py: float) -> float:
        """Euclidean distance from a point to the rectangle surface (0 inside)."""
        dx, dy = px - self.cx, py - self.cy
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        qx, qy = abs(lx) - self.half_x, abs(ly) - self.half_y
        if qx <= 0.0 and qy <= 0.0:
            return 0.0
        return math.hypot(max(qx, 0.0), max(qy, 0.0))

    def contains(self, px: float, py: float) -> bool:
        return self.distance(px, py) == 0.0

    def corners(self) -> List[Tuple[float, float]]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        out = []
        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            lx, ly = sx * self.half_x, sy * self.half_y
            out.append((self.cx + c * lx - s * ly, self.cy + s * lx + c * ly))
        return out


@dataclass(frozen=True)
class Cylinder2D:
    name: str
    cx: float
    cy: float
    radius: float

    def distance(self, px: float, py: float) -> float:
        return max(math.hypot(px - self.cx, py - self.cy) - self.radius, 0.0)


@dataclass(frozen=True)
class DynamicObstacle:
    """A planar-move obstacle: its reset footprint and the axis it oscillates on."""

    name: str
    shape: object  # Box2D or Cylinder2D at the reset pose
    axis: str      # "x" or "y"

    def swept_corridor(self, amplitude: float) -> Box2D:
        """Axis-aligned rectangle covering every footprint position reachable
        within +-amplitude along the motion axis from the reset pose."""
        if isinstance(self.shape, Cylinder2D):
            hx = hy = self.shape.radius
            cx, cy = self.shape.cx, self.shape.cy
        else:
            box = self.shape
            c, s = abs(math.cos(box.yaw)), abs(math.sin(box.yaw))
            hx = box.half_x * c + box.half_y * s
            hy = box.half_x * s + box.half_y * c
            cx, cy = box.cx, box.cy
        if self.axis == "x":
            return Box2D(self.name + "_corridor", cx, cy, hx + amplitude, hy, 0.0)
        if self.axis == "y":
            return Box2D(self.name + "_corridor", cx, cy, hx, hy + amplitude, 0.0)
        raise ValueError("axis must be 'x' or 'y'")


@dataclass
class Arena:
    walls: List[Box2D] = field(default_factory=list)
    static_obstacles: List[Box2D] = field(default_factory=list)
    dynamic_obstacles: List[DynamicObstacle] = field(default_factory=list)
    physics_max_step: Optional[float] = None
    physics_real_time_update_rate: Optional[float] = None
    has_state_plugin: bool = False

    def static_shapes(self) -> List[object]:
        return list(self.walls) + list(self.static_obstacles)

    def dynamic_reset_pose(self, name: str) -> Tuple[float, float]:
        for obstacle in self.dynamic_obstacles:
            if obstacle.name == name:
                return obstacle.shape.cx, obstacle.shape.cy
        raise KeyError(name)


def _parse_pose(text: Optional[str]) -> Tuple[float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0
    values = [float(v) for v in text.split()]
    while len(values) < 6:
        values.append(0.0)
    return values[0], values[1], values[5]


def _shape_from_collision(name: str, model_pose: Tuple[float, float, float], collision: ET.Element):
    geometry = collision.find("geometry")
    if geometry is None:
        return None
    x, y, yaw = model_pose
    box = geometry.find("box")
    if box is not None:
        size = [float(v) for v in box.findtext("size", "0 0 0").split()]
        return Box2D(name, x, y, size[0] / 2.0, size[1] / 2.0, yaw)
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return Cylinder2D(name, x, y, float(cylinder.findtext("radius", "0")))
    return None


def parse_world(path: str, dynamic_axes: Sequence[Tuple[str, str]] = ()) -> Arena:
    """Read the frozen world. ``dynamic_axes`` maps dynamic model names to their
    motion axis, e.g. ``[("dynamic_obstacle_1", "y"), ("dynamic_obstacle_2", "x")]``.
    """
    axes = dict(dynamic_axes)
    root = ET.parse(path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("world file has no <world> element")
    arena = Arena()
    physics = world.find("physics")
    if physics is not None:
        step = physics.findtext("max_step_size")
        rate = physics.findtext("real_time_update_rate")
        arena.physics_max_step = float(step) if step else None
        arena.physics_real_time_update_rate = float(rate) if rate else None
    for plugin in world.findall("plugin"):
        if plugin.get("filename") == "libgazebo_ros_state.so":
            arena.has_state_plugin = True
    for model in world.findall("model"):
        name = model.get("name", "")
        pose = _parse_pose(model.findtext("pose"))
        link = model.find("link")
        collision = link.find("collision") if link is not None else None
        if collision is None:
            continue
        shape = _shape_from_collision(name, pose, collision)
        if shape is None:
            continue
        if name.startswith("arena_wall"):
            if not isinstance(shape, Box2D):
                raise ValueError("walls must be boxes")
            arena.walls.append(shape)
        elif name.startswith("dynamic_obstacle"):
            if name not in axes:
                raise ValueError(f"no motion axis declared for dynamic model {name}")
            arena.dynamic_obstacles.append(DynamicObstacle(name, shape, axes[name]))
        elif name.startswith("static_"):
            if not isinstance(shape, Box2D):
                raise ValueError("static obstacles must be boxes in this world")
            arena.static_obstacles.append(shape)
    if len(arena.walls) != 4:
        raise ValueError(f"expected four arena walls, found {len(arena.walls)}")
    if set(axes) != {d.name for d in arena.dynamic_obstacles}:
        raise ValueError("declared dynamic obstacles do not match the world")
    return arena


def clearance(px: float, py: float, shapes: Iterable[object]) -> float:
    """Distance from a point to the nearest surface among ``shapes``."""
    best = math.inf
    for shape in shapes:
        best = min(best, shape.distance(px, py))
    return best


def segment_intersects_box(ax: float, ay: float, bx: float, by: float, box: Box2D) -> bool:
    """True if the closed segment A-B touches the rectangle (separating-axis test
    in the rectangle's body frame)."""
    c, s = math.cos(-box.yaw), math.sin(-box.yaw)

    def to_body(x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - box.cx, y - box.cy
        return c * dx - s * dy, s * dx + c * dy

    x0, y0 = to_body(ax, ay)
    x1, y1 = to_body(bx, by)
    # Liang-Barsky clipping against |x| <= half_x, |y| <= half_y
    t0, t1 = 0.0, 1.0
    dx, dy = x1 - x0, y1 - y0
    for p, q in ((-dx, x0 + box.half_x), (dx, box.half_x - x0), (-dy, y0 + box.half_y), (dy, box.half_y - y0)):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def arena_bounds(arena: Arena) -> Tuple[float, float, float, float]:
    """Inner faces of the walls: (x_min, x_max, y_min, y_max)."""
    xs, ys = [], []
    for wall in arena.walls:
        if wall.half_x > wall.half_y:  # wall running along x
            ys.append((wall.cy, wall.half_y))
        else:
            xs.append((wall.cx, wall.half_x))
    if len(xs) != 2 or len(ys) != 2:
        raise ValueError("walls do not form an axis-aligned rectangle")
    (xa, hxa), (xb, hxb) = sorted(xs)
    (ya, hya), (yb, hyb) = sorted(ys)
    return xa + hxa, xb - hxb, ya + hya, yb - hyb
