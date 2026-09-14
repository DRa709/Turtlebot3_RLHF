import os
import tempfile
import unittest
from collections import Counter

from turtlebot3_drl_nav.env_config import build_arena, build_law, load_common_parameters
from turtlebot3_drl_nav.initialization import (
    Sampler,
    derive_seed,
    e1_start_seed,
    generate_scenarios,
    obstacle_phases,
    read_scenarios,
    require_seed,
    training_obstacle_seed,
    training_start_seed,
    write_scenarios,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config")
WORLD = os.path.join(ROOT, "worlds", "phase1_mixed.world")


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.params = load_common_parameters(CONFIG)
        self.sampler = Sampler(build_law(self.params), build_arena(self.params, WORLD))

    def test_seed_derivation_is_deterministic_and_stream_separated(self):
        self.assertEqual(derive_seed(7001, "a", 1), derive_seed(7001, "a", 1))
        self.assertNotEqual(derive_seed(7001, "a", 1), derive_seed(7001, "a", 2))
        self.assertNotEqual(training_start_seed(7001, 1), training_obstacle_seed(7001, 1))
        self.assertNotEqual(e1_start_seed(9001, 1, 1), e1_start_seed(9001, 2, 1))

    def test_draw_is_a_pure_function_of_the_generator_seed(self):
        a = self.sampler.draw(training_start_seed(7001, 5))
        b = self.sampler.draw(training_start_seed(7001, 5))
        self.assertEqual(a, b)
        self.assertTrue(self.sampler.admissible(a.x, a.y))
        self.assertNotEqual(a, self.sampler.draw(training_start_seed(7001, 6)))

    def test_episode_draw_does_not_depend_on_earlier_episodes(self):
        first = [self.sampler.draw(training_start_seed(7001, e)) for e in range(1, 6)]
        later = [self.sampler.draw(training_start_seed(7001, e)) for e in (3, 5)]
        self.assertEqual([first[2], first[4]], later)

    def test_rejection_exhaustion_fails_closed(self):
        law = build_law(self.params)
        tight = law.__class__(**{**law.__dict__, "max_rejections": 1, "start_clearance_min": 10.0, "robot_footprint_radius": 0.1})
        with self.assertRaises(RuntimeError):
            Sampler(tight, self.sampler.arena).draw(1)

    def test_obstacle_phases_are_seeded(self):
        p = obstacle_phases(training_obstacle_seed(7002, 1), 2, 5.0)
        self.assertEqual(p, obstacle_phases(training_obstacle_seed(7002, 1), 2, 5.0))
        self.assertTrue(all(0.0 <= o < 5.0 for o in p.offsets))
        self.assertTrue(all(s in (-1.0, 1.0) for s in p.signs))

    def test_placeholder_seed_is_refused(self):
        with self.assertRaises(ValueError):
            require_seed("world_seed", -1)
        self.assertEqual(require_seed("world_seed", 7000), 7000)

    def test_scenario_list_is_stratified_and_round_trips(self):
        scenarios = generate_scenarios(self.sampler, 9001, 100, 10, 5.0)
        self.assertEqual(len(scenarios), 100)
        cells = Counter((s.distance_bin, s.heading_bin) for s in scenarios)
        self.assertEqual(len(cells), 12)
        self.assertTrue(all(8 <= n <= 9 for n in cells.values()))
        self.assertEqual(sum(s.physical_subset for s in scenarios), 10)
        self.assertEqual(len({s.scenario_id for s in scenarios}), 100)
        for s in scenarios:
            self.assertTrue(self.sampler.admissible(s.x, s.y))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            write_scenarios(path, scenarios)
            self.assertEqual([s.to_row() for s in read_scenarios(path)], [s.to_row() for s in scenarios])

    def test_frozen_scenario_file_matches_the_generator(self):
        frozen = read_scenarios(os.path.join(CONFIG, "evaluation_scenarios_v1.csv"))
        regenerated = generate_scenarios(self.sampler, 9001, 100, 10, float(self.params["obstacle_half_period"]))
        self.assertEqual([s.to_row() for s in frozen], [s.to_row() for s in regenerated])


if __name__ == "__main__":
    unittest.main()
