"""Dueling-specific update-schema boundary tests."""

import unittest

from turtlebot3_drl_nav.recorder import (
    UPDATE_APPLICABILITY,
    UPDATE_FIELDS_COMMON,
    UPDATES_COLUMNS,
)


class DuelingRecorderContractTests(unittest.TestCase):
    def test_dueling_diagnostics_are_required_not_rejected(self):
        required = set(UPDATE_APPLICABILITY["DuelingDoubleDQN"])
        self.assertEqual(
            required,
            set(UPDATE_FIELDS_COMMON)
            | {"epsilon", "state_value_mean", "centered_advantage_abs_mean"},
        )
        self.assertIn("state_value_mean", UPDATES_COLUMNS)
        self.assertIn("centered_advantage_abs_mean", UPDATES_COLUMNS)


if __name__ == "__main__":
    unittest.main()
