import os
import unittest
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MixedWorldTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(ROOT, "worlds", "phase1_mixed.world")
        self.root = ET.parse(self.path).getroot()

    def test_contains_static_and_dynamic_collision_models(self):
        names = {model.get("name") for model in self.root.findall(".//model")}
        for name in ("static_box_1", "static_box_2", "dynamic_obstacle_1", "dynamic_obstacle_2",
                     "arena_wall_north", "arena_wall_south", "arena_wall_east", "arena_wall_west"):
            self.assertIn(name, names)

    def test_dynamic_models_are_collision_active_prescribed_motion_bodies(self):
        dynamic = [m for m in self.root.findall(".//model") if m.get("name", "").startswith("dynamic_obstacle")]
        self.assertEqual(len(dynamic), 2)
        for model in dynamic:
            self.assertIsNone(model.find("static"))
            link = model.find("link")
            self.assertEqual(link.findtext("kinematic"), "true")
            self.assertEqual(link.findtext("gravity"), "false")
            self.assertIsNotNone(link.find("collision[@name='dynamic_collision']"))
            plugin = model.find("plugin")
            self.assertEqual(plugin.get("filename"), "libgazebo_ros_planar_move.so")
            self.assertEqual(plugin.findtext("publish_odom"), "true")
            self.assertEqual(plugin.findtext("odometry_frame"), "world")
            self.assertGreaterEqual(float(plugin.findtext("publish_rate")), 20.0)

    def test_world_loads_the_entity_state_services(self):
        plugins = [p.get("filename") for p in self.root.find("world").findall("plugin")]
        self.assertIn("libgazebo_ros_state.so", plugins)

    def test_physics_is_rate_limited_for_reproducible_arc_execution(self):
        physics = self.root.find("world/physics")
        self.assertAlmostEqual(float(physics.findtext("real_time_update_rate")), 1000.0)
        self.assertAlmostEqual(float(physics.findtext("max_step_size")), 0.001)


if __name__ == "__main__":
    unittest.main()
