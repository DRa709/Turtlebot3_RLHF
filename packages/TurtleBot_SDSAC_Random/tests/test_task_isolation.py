import importlib.util
import os
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "arc", "task_isolation.py")
SPEC = importlib.util.spec_from_file_location("task_isolation", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TaskIsolationTests(unittest.TestCase):
    def test_live_tasks_receive_distinct_transport_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = MODULE.acquire(tmp, hostname="node-a")
            second = MODULE.acquire(tmp, hostname="node-a")
            self.assertNotEqual(first[:2], second[:2])
            self.assertTrue(os.path.isfile(first[2]))
            self.assertTrue(os.path.isfile(second[2]))
            MODULE.release(first[2])
            MODULE.release(second[2])
            self.assertFalse(os.path.exists(first[2]))
            self.assertFalse(os.path.exists(second[2]))


if __name__ == "__main__":
    unittest.main()
