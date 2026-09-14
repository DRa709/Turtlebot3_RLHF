import os
import tempfile
import unittest

from turtlebot3_drl_nav.identity import IDENTITY_FIELDS
from turtlebot3_drl_nav.recorder import CsvStream, STREAMS, UPDATE_APPLICABILITY, UPDATES_COLUMNS, open_stream, read_stream

IDENTITY = {k: f"v_{k}" for k in IDENTITY_FIELDS}


class RecorderTests(unittest.TestCase):
    def test_create_only_header_first_and_identity_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = open_stream(tmp, "checkpoints", IDENTITY)
            stream.write({"env_step": 25000, "kind": "policy_only", "path": "x.pt", "sha256": "0" * 64, "validated": True})
            stream.close()
            rows = read_stream(os.path.join(tmp, "checkpoints.csv"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "v_run_id")
            self.assertEqual(rows[0]["validated"], "1")
            self.assertEqual(rows[0]["gradient_step"], "")
            with self.assertRaises(FileExistsError):
                open_stream(tmp, "checkpoints", IDENTITY)

    def test_unknown_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = open_stream(tmp, "updates", IDENTITY)
            with self.assertRaises(KeyError):
                stream.write({"not_a_column": 1})
            stream.close()

    def test_floats_round_trip_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = CsvStream(os.path.join(tmp, "s.csv"), ["a"], IDENTITY)
            value = 0.1 + 0.2
            stream.write({"a": value})
            stream.close()
            self.assertEqual(float(read_stream(os.path.join(tmp, "s.csv"))[0]["a"]), value)

    def test_applicability_matrix_is_discrete_sac_only(self):
        for algorithm, fields in UPDATE_APPLICABILITY.items():
            self.assertTrue(set(fields) <= set(UPDATES_COLUMNS), algorithm)
        self.assertEqual(set(UPDATE_APPLICABILITY), {"DiscreteSAC"})
        self.assertEqual(set(UPDATE_APPLICABILITY["DiscreteSAC"]), set(UPDATES_COLUMNS))
        for required in ("actor_loss", "critic1_loss", "critic2_loss", "alpha", "policy_entropy_mean"):
            self.assertIn(required, UPDATE_APPLICABILITY["DiscreteSAC"])
        self.assertEqual(set(STREAMS), {"transitions", "episodes", "updates", "evaluation", "checkpoints"})


if __name__ == "__main__":
    unittest.main()
