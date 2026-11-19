"""Validate the contained short-timeout run without rediscovering or killing PIDs."""

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class JobEvidenceTest(unittest.TestCase):
    def test_timeout_owns_and_drains_descendants(self) -> None:
        path = ROOT / "_docs/perf131k/tests/timeout-tree-02.json"
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt["status"], "TIMEOUT")
        self.assertTrue(receipt["contained_before_resume"])
        self.assertTrue(receipt["termination_confirmed"])
        self.assertEqual(receipt["job_active_after"], 0)
        self.assertGreaterEqual(receipt["job_total_processes"], 2)
        raw = path.with_suffix(".txt").read_bytes()
        self.assertIn(b"owned fixture descendant created", raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt["raw_sha256"])
        self.assertEqual(receipt["input_sha256"], receipt["input_sha256_after"])
        self.assertIn("utc_started", receipt)
        self.assertEqual(receipt["environment"]["BLACK_CACHE_DIR"], str(ROOT / "_scratch/perf131k/black"))


if __name__ == "__main__":
    unittest.main()
