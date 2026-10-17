"""Offline lifecycle checks for the portable runner; no model or GPU is started."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import run_audit


class FakeEngine:
    calls = []
    closes = 0
    wrong_plan = False

    def __init__(self, cfg, arm, folder, deadline):
        self.p = SimpleNamespace(pid=os.getpid())
        self.calls.append(arm)
        (folder / "stdout.log").write_text(
            "INFO context=" + ("1" if self.wrong_plan else "131072") + "\n"
        )

    def generate(self, prompt, limit, seed, seconds):
        return dict(
            name=prompt["name"],
            output_ids=list(range(limit)),
            produced=limit,
            decode_ms=10,
            finish="length",
            draft_accepted=20,
            draft_offered=30,
            reused=0,
            prompt_read=len(prompt["ids"]),
            decode_file_blobs=0,
        )

    def close(self):
        type(self).closes += 1


class RunnerLifecycle(unittest.TestCase):
    def run_case(self, wrong):
        FakeEngine.calls = []
        FakeEngine.closes = 0
        FakeEngine.wrong_plan = wrong
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "config.json"
            cfg.write_text(
                json.dumps(
                    {
                        "exe": sys.executable,
                        "args": [],
                        "min_available_gib": 0,
                        "reference_plan": {"context": "131072"},
                    }
                )
            )
            with (
                mock.patch.object(run_audit, "Engine", FakeEngine),
                mock.patch.object(run_audit.time, "sleep", lambda _: None),
            ):
                rc = run_audit.main(["--config", str(cfg), "--out", str(root / "run")])
            data = json.loads((root / "run/results.json").read_text())
            return rc, data

    def test_full_schedule_and_aa_modes(self):
        rc, data = self.run_case(False)
        self.assertEqual(rc, 0)
        self.assertTrue(data["completed"])
        self.assertEqual(FakeEngine.closes, 28)
        expected = [
            ("on" if item["kind"] == "aa" else arm)
            for item in data["schedule"]
            for arm in item["order"]
        ]
        self.assertEqual(FakeEngine.calls, expected)
        self.assertEqual(sum(len(a["requests"]) for a in data["arms"]), 84)

    def test_plan_mismatch_closes_owned_engine(self):
        rc, data = self.run_case(True)
        self.assertEqual(rc, 1)
        self.assertFalse(data["completed"])
        self.assertEqual(FakeEngine.closes, 1)
        self.assertIn("differs from reference_plan", data["error"])


if __name__ == "__main__":
    unittest.main()
