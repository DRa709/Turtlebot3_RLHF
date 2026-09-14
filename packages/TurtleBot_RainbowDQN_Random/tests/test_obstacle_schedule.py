import unittest

from turtlebot3_drl_nav.obstacle_schedule import obstacle_displacement, obstacle_velocity


class ObstacleScheduleTests(unittest.TestCase):
    def test_reversal_grid_and_axis(self):
        self.assertEqual(obstacle_velocity("y", 1.0, 0.0, 0.22, 5.0, 0.0), (0.0, 0.22))
        self.assertEqual(obstacle_velocity("y", 1.0, 0.0, 0.22, 5.0, 5.0), (0.0, -0.22))
        self.assertEqual(obstacle_velocity("x", -1.0, 2.0, 0.22, 5.0, 3.5), (0.22, 0.0))  # segment 1 -> flipped sign
        with self.assertRaises(ValueError):
            obstacle_velocity("z", 1.0, 0.0, 0.22, 5.0, 0.0)

    def test_displacement_stays_within_the_declared_amplitude(self):
        for offset in (0.0, 1.3, 4.9):
            position, dt = 0.0, 0.01
            extremes = [0.0]
            t = 0.0
            while t < 60.0:
                _, vy = obstacle_velocity("y", 1.0, offset, 0.22, 5.0, t)
                position += vy * dt
                extremes.append(position)
                t += dt
            self.assertLessEqual(max(extremes) - min(extremes), 1.1 + 0.22 * dt * 5)  # Euler grid slack

    def test_analytic_displacement_matches_fine_integration(self):
        dt = 0.0005
        for offset in (0.0, 1.3, 4.9):
            numerical = 0.0
            t = 0.0
            while t < 12.345:
                numerical += obstacle_velocity("x", -1.0, offset, 0.22, 5.0, t)[0] * dt
                t += dt
            analytic = obstacle_displacement("x", -1.0, offset, 0.22, 5.0, 12.345)[0]
            self.assertAlmostEqual(numerical, analytic, delta=0.22 * dt * 3)


if __name__ == "__main__":
    unittest.main()
