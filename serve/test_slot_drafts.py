"""Backward-compatible per-slot draft accounting, CPU-only."""
import unittest
from serve.server import StrataEngine, carry_leg, add_carry, request_timings

class SlotDrafts(unittest.TestCase):
    def engine(self, **last):
        e = StrataEngine.__new__(StrataEngine)
        e.batch = 2
        e.last = last
        return e

    def test_slot_counts_add_to_admission(self):
        e = self.engine(generated=1, drafts_offered=4, drafts_accepted=2)
        e._apply_bdone('BDONE 1 10 stop 42.5 3 5'.split())
        self.assertEqual((e.last['drafts_accepted'], e.last['drafts_offered']), (5, 9))
        self.assertEqual((e.last['generated'], e.last['decode_ms']), (10, 42.5))

    def test_old_engine_line_preserves_admission(self):
        e = self.engine(drafts_offered=4, drafts_accepted=2)
        e._apply_bdone('BDONE 1 10 length 42.5'.split())
        self.assertEqual((e.last['drafts_accepted'], e.last['drafts_offered']), (2, 4))

    def test_no_counts_stays_unknown_for_old_engine(self):
        e = self.engine()
        e._apply_bdone('BDONE 1 10 length 42.5'.split())
        self.assertNotIn('drafts_offered', e.last)
        e._apply_bdone('BDONE 1 10 length 42.5 0 0'.split())
        self.assertEqual((e.last['drafts_accepted'], e.last['drafts_offered']), (0, 0))

    def test_mixed_legs_keep_optional_drafts_in_request_timings(self):
        # Initial solo, slot handback, final solo DONE: counts must cover all legs.
        # Absence is unknown, while an explicitly reported zero remains a known zero.
        cases = [
            ("all reported", (6, 1100, 28), (4, 910, 21), (1134, 935)),
            ("carried only", (6, None, None), (4, None, None), (6, 4)),
            ("final only", (None, None, 28), (None, None, 21), (28, 21)),
            ("carried zero", (0, 0, None), (0, 0, None), (0, 0)),
            ("final zero", (None, None, 0), (None, None, 0), (0, 0)),
            ("all absent", (None, None, None), (None, None, None), None),
        ]
        for name, offered, accepted, expected in cases:
            with self.subTest(name=name):
                legs = [
                    dict(generated=5, decode_ms=100.0, prompt_ms=1000.0,
                         prompt_read=107, reused=0, finish="cancel"),
                    dict(generated=1464, decode_ms=27000.0, prompt_ms=0.0,
                         prompt_read=1, reused=107, finish="cancel"),
                    dict(generated=31, decode_ms=378.0, prompt_ms=373.0,
                         prompt_read=1, reused=0, finish="length"),
                ]
                for i, leg in enumerate(legs):
                    if offered[i] is not None:
                        leg["drafts_offered"] = offered[i]
                        leg["drafts_accepted"] = accepted[i]
                    elif i == 1:  # Explicit None also means not reported.
                        leg["drafts_offered"] = leg["drafts_accepted"] = None
                carry = {}
                carry_leg(carry, legs[0])
                carry_leg(carry, legs[1])
                total = add_carry(carry, legs[2])
                timings = request_timings(107, 1500, total)
                self.assertEqual((total["generated"], total["decode_ms"]), (1500, 27478.0))
                self.assertEqual((timings["prompt_n"], timings["cache_n"], timings["prompt_ms"],
                                  timings["predicted_n"], timings["predicted_ms"]),
                                 (107, 0, 1000.0, 1500, 27478.0))
                if expected is None:
                    self.assertNotIn("draft_n", timings)
                    self.assertNotIn("draft_n_accepted", timings)
                else:
                    self.assertEqual((timings["draft_n"], timings["draft_n_accepted"]), expected)
