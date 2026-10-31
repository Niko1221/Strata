"""serve/test_jev_calib_task.py - task-scoped calibration (cross-task contamination guard).

    python -m unittest serve.test_jev_calib_task -v

The defect this pins down (measured 2026-11-01, 500 banking77 items, single GPU):

  swap() keys fits by READOUT LETTER, so one calib-L.json served every text task.
  A fit carrying a_noul=0.6127, c_noul=+0.6539 (fitted on BoolQ60/ARC60/SST5-60)
  was applied to a balanced binary idea-grouping task that needs c_noul ~ -1.56.
  Effect: yes-rate 67.6% -> 80.4%, specificity 0.584 -> 0.360, accuracy
  0.760 -> 0.664 (McNemar exact p < 0.001).  Doing NOTHING beat what shipped.

Rules pinned here:
  1. no task tag in the environment -> legacy behaviour, byte-identical (no regression);
  2. a fit recorded for the same task -> applied (platt_v1);
  3. a fit recorded for a DIFFERENT task -> refused, identity, probabilities untouched;
  4. an UNTAGGED fit with a task tag set -> refused (fail closed: identity beats a
     stranger's intercept, and an untagged file cannot prove it belongs here);
  5. refusal is reported in the calib id so callers can see it, not silent.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STRATA_JEV_MODES", "chat,decision")
from serve import jev_calib  # noqa: E402


def _fit(task: str, a: float = 0.6127, c: float = 0.6539) -> "jev_calib.Calib":
    return jev_calib.Calib(loaded=True, tuple_str="qfn|jev-port|none|v1|L|text",
                           T_choice=1.0557, T_score=2.263,
                           a_noul=a, c_noul=c, task=task)


class TaskScopedCalibration(unittest.TestCase):
    def setUp(self):
        self._env_task = os.environ.pop("STRATA_JEV_TASK", None)
        jev_calib._set_task_for_test(None)      # fall back to the environment
        jev_calib._set_for_test(None)
        jev_calib._MISMATCH_LOGGED.clear()

    def tearDown(self):
        jev_calib._set_task_for_test(None)
        jev_calib._set_for_test(None)
        jev_calib._MISMATCH_LOGGED.clear()
        os.environ.pop("STRATA_JEV_TASK", None)
        if self._env_task is not None:
            os.environ["STRATA_JEV_TASK"] = self._env_task

    # ── 1. legacy path unchanged ───────────────────────────────────────────
    def test_untagged_runtime_applies_fit_as_before(self):
        fit = _fit("something-else")
        jev_calib._set_for_test(fit, "L")
        pr, cid = jev_calib.apply("noul", [0.5, 0.5], mode="L")
        self.assertEqual(cid, "platt_v1",
                         "no task tag must behave exactly as the last release")
        self.assertAlmostEqual(pr[1], 1.0 / (1.0 + math.exp(-0.6539)), places=6)

    # ── 2. matching task applies ───────────────────────────────────────────
    def test_matching_task_applies_fit(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit("banking77-mf", a=0.4504, c=-1.5607), "L")
        pr, cid = jev_calib.apply("noul", [0.99, 0.01], mode="L")
        self.assertEqual(cid, "platt_v1")
        z = math.log(0.01 / 0.99)
        self.assertAlmostEqual(pr[1], 1.0 / (1.0 + math.exp(-(0.4504 * z - 1.5607))), places=6)

    # ── 3. different task is refused ───────────────────────────────────────
    def test_other_task_fit_is_refused_and_reported(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit("boolq-arc-sst5"), "L")
        before = [0.99, 0.01]
        pr, cid = jev_calib.apply("noul", list(before), mode="L")
        self.assertEqual(cid, "identity", "a fit from another task must not be applied")
        self.assertEqual(pr, before, "probabilities must be untouched under refusal")

    # ── 4. untagged fit fails closed ───────────────────────────────────────
    def test_untagged_fit_is_refused_when_task_tag_set(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit(""), "L")
        pr, cid = jev_calib.apply("noul", [0.99, 0.01], mode="L")
        self.assertEqual(cid, "identity")
        self.assertEqual(pr, [0.99, 0.01])

    # ── 5. refusal also covers choice/score (not just noul) ────────────────
    def test_choice_temperature_also_scoped(self):
        jev_calib._set_task_for_test("banking77-mf")
        jev_calib._set_for_test(_fit("boolq-arc-sst5"), "L")
        pr, cid = jev_calib.apply("choice", [0.5, 0.3, 0.2], mode="L")
        self.assertEqual(cid, "identity")
        self.assertEqual(pr, [0.5, 0.3, 0.2])

    # ── 6. loader round-trips the task field ───────────────────────────────
    def test_loader_reads_task_field(self):
        import json, tempfile
        d = Path(tempfile.mkdtemp())
        (d / "calib-L.json").write_text(json.dumps({
            "tuple": "qfn|jev-port|none|v1|L|banking77-mf",
            "task": "banking77-mf",
            "a_noul": 0.4504, "c_noul": -1.5607,
            "T_choice": 1.0557, "T_score": 2.263,
            "provenance": "unit test"}))
        c = jev_calib.load(d / "calib-L.json")
        self.assertTrue(c.loaded)
        self.assertEqual(c.task, "banking77-mf")
        self.assertAlmostEqual(c.c_noul, -1.5607, places=6)

    # ── 7. STRICT mode does not reject a task-tagged file ──────────────────
    def test_task_key_is_known_to_strict_loader(self):
        self.assertIn("task", jev_calib._KNOWN_KEYS)


if __name__ == "__main__":
    unittest.main()
