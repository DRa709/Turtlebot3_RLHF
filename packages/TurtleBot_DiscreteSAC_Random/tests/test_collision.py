import unittest

from turtlebot3_drl_nav.collision import classify_collision_names


class CollisionClassificationTests(unittest.TestCase):
    def test_static_contact(self):
        result = classify_collision_names(("burger::base::collision", "static_box_1::link::static_collision"))
        self.assertEqual(result, (True, True, False))

    def test_dynamic_contact(self):
        result = classify_collision_names(("dynamic_obstacle_1::link::dynamic_collision", "burger::base::collision"))
        self.assertEqual(result, (True, False, True))

    def test_ground_contact_is_not_navigation_collision(self):
        result = classify_collision_names(("burger::wheel::collision", "ground_plane::link::collision"))
        self.assertEqual(result, (False, False, False))

    def test_unknown_contact_is_visible_for_fail_closed_handling(self):
        result = classify_collision_names(("burger::base::collision", "mystery_model::link::collision"))
        self.assertEqual(result, (True, False, False))


if __name__ == "__main__":
    unittest.main()
