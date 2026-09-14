import math
import os
import random
import unittest

from turtlebot3_drl_nav.env_config import build_arena, build_law, load_common_parameters
from turtlebot3_drl_nav.geometry import Box2D, arena_bounds, segment_intersects_box
from turtlebot3_drl_nav.initialization import Sampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config")
WORLD = os.path.join(ROOT, "worlds", "phase1_mixed.world")


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.params = load_common_parameters(CONFIG)
        self.arena = build_arena(self.params, WORLD)
        self.sampler = Sampler(build_law(self.params), self.arena)

    def test_world_parses_into_the_expected_arena(self):
        self.assertEqual(arena_bounds(self.arena), (-2.94, 2.94, -2.94, 2.94))
        self.assertEqual([b.name for b in self.arena.static_obstacles], ["static_box_1", "static_box_2"])
        self.assertEqual([(d.name, d.axis) for d in self.arena.dynamic_obstacles], [("dynamic_obstacle_1", "y"), ("dynamic_obstacle_2", "x")])
        self.assertTrue(self.arena.has_state_plugin)

    def test_rotated_box_distance(self):
        box = Box2D("b", 0.0, 0.0, 1.0, 0.5, math.pi / 2.0)
        self.assertEqual(box.distance(0.0, 0.0), 0.0)
        self.assertAlmostEqual(box.distance(1.0, 0.0), 0.5)   # rotated: half_y along x
        self.assertAlmostEqual(box.distance(0.0, 1.5), 0.5)

    def test_segment_intersection(self):
        box = Box2D("b", 1.0, 0.0, 0.5, 0.5, 0.0)
        self.assertTrue(segment_intersects_box(0.0, 0.0, 2.0, 0.0, box))
        self.assertFalse(segment_intersects_box(0.0, 1.0, 2.0, 1.0, box))
        self.assertTrue(segment_intersects_box(0.0, 0.0, 2.0, 1.0, box))

    def test_declared_law_holds_on_every_admissible_point(self):
        law = self.sampler.law
        rng = random.Random(3)
        checked = 0
        for _ in range(20000):
            x, y = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
            if not self.sampler.admissible(x, y):
                continue
            checked += 1
            self.assertGreaterEqual(self.sampler.start_clearance(x, y), law.start_clearance_min - 1e-9)
            self.assertGreaterEqual(math.hypot(x - law.goal_x, y - law.goal_y), law.goal_exclusion_radius)
            self.assertTrue(law.x_min <= x <= law.x_max and law.y_min <= y <= law.y_max)
        self.assertGreater(checked, 5000)

    def test_swept_corridor_excludes_the_obstacle_paths(self):
        # obstacle 1 sweeps y in [-1.45 - 1.1, -1.45 + 1.1] at x = 0: the whole path is inadmissible
        for y in (-2.4, -1.45, -0.5, -0.1):
            self.assertFalse(self.sampler.admissible(0.0, y))
        # but the region north of the corridor plus clearance is admissible
        self.assertTrue(self.sampler.admissible(0.0, 1.5))

    def test_fixed_start_is_outside_the_support(self):
        self.assertFalse(self.sampler.admissible(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
