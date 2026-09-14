import os
import tempfile
import unittest

from turtlebot3_drl_nav.artifact_integrity import verify_manifest, write_manifest


class ArtifactIntegrityTests(unittest.TestCase):
    def test_manifest_detects_tampering_and_undeclared_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "result.txt")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("original\n")
            write_manifest(tmp)
            self.assertEqual(verify_manifest(tmp), [])
            with open(path, "a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            self.assertTrue(any("digest mismatch" in problem for problem in verify_manifest(tmp)))
            with open(os.path.join(tmp, "extra.txt"), "w", encoding="utf-8") as stream:
                stream.write("extra\n")
            self.assertTrue(any("undeclared file" in problem for problem in verify_manifest(tmp)))

    def test_complete_and_failure_markers_are_outside_immutable_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "data.csv"), "w", encoding="utf-8") as stream:
                stream.write("x\n")
            write_manifest(tmp)
            for marker in ("COMPLETE", "FAILED", "INTERRUPTED"):
                with open(os.path.join(tmp, marker), "w", encoding="utf-8") as stream:
                    stream.write("status\n")
            self.assertEqual(verify_manifest(tmp), [])

    def test_symbolic_link_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "target.txt")
            with open(target, "w", encoding="utf-8") as stream:
                stream.write("data\n")
            os.symlink(target, os.path.join(tmp, "alias.txt"))
            with self.assertRaisesRegex(ValueError, "symbolic"):
                write_manifest(tmp)

    def test_symbolic_link_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as target:
            os.symlink(target, os.path.join(tmp, "linked_directory"))
            with self.assertRaisesRegex(ValueError, "symbolic"):
                write_manifest(tmp)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO files are not available on this platform")
    def test_non_regular_output_is_refused_without_reading_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkfifo(os.path.join(tmp, "unexpected.fifo"))
            with self.assertRaisesRegex(ValueError, "non-regular"):
                write_manifest(tmp)


if __name__ == "__main__":
    unittest.main()
