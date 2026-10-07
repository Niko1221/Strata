#!/usr/bin/env python3
"""Protocol checks for the staggered exactness regression; no model or GPU required."""
import unittest

from batch_test import ProtocolError, run_batch


class RecordingEngine:
    def __init__(self):
        self.commands = []

    def send(self, command):
        self.commands.append(command)


class BatchProtocolTest(unittest.TestCase):
    def test_stagger_keeps_tokens_during_admission_and_records_overlap(self):
        eng = RecordingEngine()
        lines = iter([
            "T 101", "DONE", "BADM 0 1", "BT 0 102",
            "BT 0 103", "T 201", "DONE", "BADM 1 1", "BT 0 104", "BT 0 105",
            "BT 1 202", "T 301", "DONE", "BADM 2 1",
            "BT 0 106", "BDONE 0", "BT 1 203", "BDONE 1", "BT 2 302", "BDONE 2",
        ])
        got, stats = run_batch(eng, lines, [[1], [2], [3]], 20, "", [2, 5])
        self.assertEqual(got, {0: [101, 102, 103, 104, 105, 106], 1: [201, 202, 203], 2: [301, 302]})
        self.assertEqual([e["anchor_tokens"] for e in stats["admissions"]], [0, 2, 5])
        self.assertEqual([e["active_slots"] for e in stats["admissions"]], [[], [0], [0, 1]])
        self.assertEqual([e["tokens_during_admission"] for e in stats["admissions"]], [0, 1, 1])
        self.assertEqual([c.split()[1] for c in eng.commands], ["0", "1", "2"])

    def test_early_anchor_completion_does_not_pass_as_staggered(self):
        with self.assertRaisesRegex(ProtocolError, "anchor finished"):
            run_batch(RecordingEngine(), iter(["T 1", "BADM 0 1", "BDONE 0"]), [[1], [2]], 20, "", [3])

    def test_anchor_must_remain_active_after_a_staggered_admission(self):
        with self.assertRaisesRegex(ProtocolError, "finished during admission"):
            run_batch(RecordingEngine(), iter(["T 1", "BADM 0 1", "BT 0 2", "BDONE 0", "T 3", "BADM 1 1"]),
                      [[1], [2]], 20, "", [2])

    def test_missing_completion_and_engine_error_fail(self):
        for tail, message in (([], "ended"), (["ERR verify failed"], "ERR verify failed")):
            with self.subTest(tail=tail), self.assertRaisesRegex(ProtocolError, message):
                run_batch(RecordingEngine(), iter(["T 1", "BADM 0 1", *tail]), [[1]], 20, "")

    def test_admission_only_requests_finish_without_waiting_for_bdone(self):
        got, _ = run_batch(RecordingEngine(), iter(["T 1", "BADM 0 0", "T 2", "BADM 1 0"]),
                           [[1], [2]], 1, "")
        self.assertEqual(got, {0: [1], 1: [2]})

    def test_wrong_slot_completion_is_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "unexpected admission"):
            run_batch(RecordingEngine(), iter(["BADM 1 1"]), [[1]], 20, "")

    def test_solo_promotion_drains_tokens_after_stop_and_resumes_exact_prefix(self):
        eng = RecordingEngine()
        lines = iter(["T 10", "T 11", "T 12", "DONE 3 2 1.0 1.0 cancel", "T 13", "BADM 0 1",
                      "BT 0 14", "T 20", "BADM 1 1", "BT 0 15", "BDONE 0", "BT 1 21", "BDONE 1"])
        got, stats = run_batch(eng, lines, [[1, 2], [3, 4]], 20, "", [5], promote_after=2)
        self.assertEqual(got, {0: [10, 11, 12, 13, 14, 15], 1: [20, 21]})
        self.assertEqual(eng.commands, ["GEN 20 1,2", "STOP", "BGEN 0 17 1,2,10,11,12", "BGEN 1 20 3,4"])
        self.assertEqual(stats["promotion"], {"stop_sent_at_tokens": 2, "resumed_after_tokens": 3})


if __name__ == "__main__":
    unittest.main()
