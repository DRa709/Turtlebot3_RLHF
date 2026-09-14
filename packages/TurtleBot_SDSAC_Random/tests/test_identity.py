import hashlib
import json
import os
import tempfile
import unittest

from turtlebot3_drl_nav.identity import (
    IDENTITY_FIELDS,
    build_identity,
    config_digest,
    read_identity,
    shared_layer_digest,
    verify_manifest,
    write_identity,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IdentityTests(unittest.TestCase):
    def test_config_digest_changes_with_any_config_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "config"))
            with open(os.path.join(tmp, "config", "a.yaml"), "w") as f:
                f.write("x: 1\n")
            d1 = config_digest(tmp)
            with open(os.path.join(tmp, "config", "a.yaml"), "w") as f:
                f.write("x: 2\n")
            self.assertNotEqual(d1, config_digest(tmp))
            os.rename(os.path.join(tmp, "config", "a.yaml"), os.path.join(tmp, "config", "b.yaml"))
            d3 = config_digest(tmp)
            with open(os.path.join(tmp, "config", "b.yaml"), "w") as f:
                f.write("x: 1\n")
            self.assertNotEqual(d1, config_digest(tmp))  # a rename moves the digest too
            self.assertNotEqual(d3, config_digest(tmp))

    def test_package_manifests_are_satisfied(self):
        self.assertEqual(verify_manifest(ROOT, "SHARED_LAYER_MANIFEST.sha256"), [])
        self.assertEqual(verify_manifest(ROOT, "RELEASE_MANIFEST.sha256"), [])
        self.assertEqual(len(shared_layer_digest(ROOT)), 64)

    def test_identity_round_trip_and_phase_validation(self):
        identity = build_identity("e", "SDSAC", "1.0.2", "random", 101, "phase1_mixed", 7000, 7001, 7002, 9001, "c" * 64, "d" * 64, "a" * 64, "b" * 64, "1.0.2", "training", "controlled", "job")
        self.assertEqual(list(identity), IDENTITY_FIELDS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run_identity.json")
            write_identity(path, identity)
            self.assertEqual(read_identity(path), identity)
        with self.assertRaises(ValueError):
            build_identity("e", "SDSAC", "1.0.0", "random", 101, "w", 1, 1, 1, 1, "c" * 64, "d" * 64, "s" * 64, "r" * 64, "1", "training", "full", "job")
        with self.assertRaises(ValueError):
            build_identity("e", "SDSAC", "1.0.0", "random", 101, "w", 1, 1, 1, 1, "short", "d" * 64, "s" * 64, "r" * 64, "1", "training", "controlled", "job")

    def test_identity_reader_rejects_extra_fields_and_coerced_seeds(self):
        identity = build_identity("e", "SDSAC", "1.0.2", "random", 101, "phase1_mixed", 7000, 7001, 7002, 9001, "c" * 64, "d" * 64, "a" * 64, "b" * 64, "1.0.2", "training", "controlled", "job")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "identity.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(dict(identity, undeclared="value"), stream)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                read_identity(path)
            identity["learning_seed"] = "101"
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(identity, stream)
            with self.assertRaisesRegex(ValueError, "learning_seed"):
                read_identity(path)

    def test_release_manifest_rejects_an_undeclared_extra_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = os.path.join(tmp, "payload.txt")
            with open(payload, "w", encoding="utf-8") as stream:
                stream.write("payload\n")
            digest = hashlib.sha256(b"payload\n").hexdigest()
            with open(os.path.join(tmp, "RELEASE_MANIFEST.sha256"), "w", encoding="utf-8") as stream:
                stream.write(f"{digest}  payload.txt\n")
            self.assertEqual(verify_manifest(tmp, "RELEASE_MANIFEST.sha256"), [])
            with open(os.path.join(tmp, "extra.txt"), "w", encoding="utf-8") as stream:
                stream.write("extra\n")
            self.assertTrue(any("undeclared file extra.txt" in item for item in verify_manifest(tmp, "RELEASE_MANIFEST.sha256")))

    def test_release_manifest_rejects_a_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as target:
            payload = os.path.join(tmp, "payload.txt")
            with open(payload, "w", encoding="utf-8") as stream:
                stream.write("payload\n")
            digest = hashlib.sha256(b"payload\n").hexdigest()
            with open(os.path.join(tmp, "RELEASE_MANIFEST.sha256"), "w", encoding="utf-8") as stream:
                stream.write(f"{digest}  payload.txt\n")
            os.symlink(target, os.path.join(tmp, "linked_directory"))
            self.assertTrue(any("symbolic-link directory" in item for item in verify_manifest(tmp, "RELEASE_MANIFEST.sha256")))


if __name__ == "__main__":
    unittest.main()
