"""--monitor-aggregate: the Monitor's speed and prefill as tokens per second summed over every running stream."""
import unittest
from unittest import mock
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine


def service():
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                  ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    svc.monitor_aggregate = True
    return svc


def note_at(svc, t, n, st):
    with mock.patch("serve.server.time.time", return_value=t):
        svc._note(n, [], st=st, rate=[])


def pp_at(e, t, pos, total, ms, rate):
    with mock.patch("serve.server.time.time", return_value=t):
        e._count_pp(f"PP {pos} {total} {ms} {rate}".split())


class DecodeLastSecond(unittest.TestCase):
    def test_counts_only_the_last_second(self):
        svc = service()
        st = {"generated": 0}
        for i in range(1, 41):                       # 40 tokens over 100.0..103.9 s: 10 a second
            note_at(svc, 100.0 + (i - 1) * 0.1, i, st)
        self.assertEqual(svc._sample_aggregate(now=103.95)[0], 10.0)

    def test_parallel_streams_add_up(self):
        svc = service()
        a, b = {"generated": 0}, {"generated": 0}
        for i in range(1, 21):
            note_at(svc, 100.0 + i * 0.04, i, a)     # 20 in the second
        for i in range(1, 16):
            note_at(svc, 100.0 + i * 0.05, i, b)     # 15 in the second
        self.assertEqual(svc._sample_aggregate(now=101.0)[0], 35.0)

    def test_a_restarted_count_never_goes_negative(self):
        svc = service()
        st = {"generated": 0}
        note_at(svc, 100.5, 10, st)
        note_at(svc, 100.6, 3, st)                   # a retry from token 0
        self.assertEqual(svc._sample_aggregate(now=101.0)[0], 10.0)

    def test_off_records_nothing(self):
        svc = service()
        svc.monitor_aggregate = False
        note_at(svc, 100.5, 10, {"generated": 0})
        self.assertEqual(len(svc._decoded), 0)

    def test_held_until_zero_for_three_seconds(self):
        svc = service()
        note_at(svc, 100.5, 30, {"generated": 0})
        self.assertEqual(svc._sample_aggregate(now=101.0)[0], 30.0)
        self.assertEqual(svc._sample_aggregate(now=102.0)[0], 30.0)   # 0 for 1 s: still shown
        self.assertEqual(svc._sample_aggregate(now=103.0)[0], 30.0)
        self.assertEqual(svc._sample_aggregate(now=104.0)[0], 0.0)    # 0 for 3 s: 0

    def test_telemetry_series_and_live_fields(self):
        svc = service()
        svc._telemetry_extra()
        svc._note(12, [], st={"generated": 0}, rate=[])
        extra = svc._telemetry_extra()
        self.assertIn("tok_s", extra)
        self.assertIn("prefill_tok_s_mean", extra)
        live = svc.metrics()["live"]
        self.assertEqual(live["rates"], "aggregate")
        self.assertEqual((live["agg_tok_s"], live["agg_prefill_tok_s"]), svc._agg_rates)

    def test_off_by_default(self):
        svc = service()
        svc.monitor_aggregate = False
        self.assertNotIn("rates", svc.metrics()["live"])


class PrefillCount(unittest.TestCase):
    def engine(self):
        return StrataEngine("missing-executable", [], lazy=True)

    def pp(self, e, pos, total, ms, rate):
        e._count_pp(f"PP {pos} {total} {ms} {rate}".split())

    def test_counts_only_fresh_tokens(self):
        e = self.engine()
        # 1000 tokens reused, then 512 + 512 read; position includes the reused prefix
        self.pp(e, 1512, 2024, 500, 1024.0)
        self.assertEqual(e.prefill_read_total, 512)
        self.pp(e, 2024, 2024, 1000, 1024.0)
        self.assertEqual(e.prefill_read_total, 1024)

    def test_a_new_prompt_starts_from_zero(self):
        e = self.engine()
        self.pp(e, 800, 800, 1000, 800.0)
        self.pp(e, 300, 600, 250, 1200.0)    # smaller ms: the next prompt
        self.assertEqual(e.prefill_read_total, 1100)

    def test_admission_reset(self):
        e = self.engine()
        self.pp(e, 800, 800, 1000, 800.0)
        e._pp_seen = (0.0, 0)                # what a slot admission or request start does
        self.pp(e, 400, 400, 2000, 200.0)
        self.assertEqual(e.prefill_read_total, 1200)

    def test_short_or_bad_lines_are_ignored(self):
        e = self.engine()
        e._count_pp("PP 10 20".split())
        e._count_pp("PP 10 20 x y".split())
        self.assertEqual(e.prefill_read_total, 0)


class PrefillNow(unittest.TestCase):
    """Prefill is each reading engine's latest chunk speed, never a chunk's tokens dumped into one second."""

    def setUp(self):
        self.svc = service()
        self.e = StrataEngine("missing-executable", [], lazy=True)
        self.e._pp_seen, self.e.prefill_now = (0.0, 0), None
        self.svc.engine = mock.Mock(engines=[self.e])

    def test_a_4_s_chunk_at_1800_reads_1800_not_7200(self):
        pp_at(self.e, 104.0, 7200, 60000, 4000, 1800.0)
        self.assertEqual(self.svc._sample_aggregate(now=104.5)[1], 1800.0)

    def test_the_chunk_speed_not_the_prompt_average(self):
        pp_at(self.e, 104.0, 8000, 60000, 4000, 2000.0)      # 8,000 in 4 s
        pp_at(self.e, 112.0, 16000, 60000, 12000, 16000 / 12)  # the next 8,000 took 8 s: 1,000 tok/s now
        self.assertEqual(self.svc._sample_aggregate(now=112.5)[1], 1000.0)

    def test_stays_up_between_chunks_while_reading(self):
        pp_at(self.e, 104.0, 7200, 60000, 4000, 1800.0)
        for t in (105.0, 106.0, 107.0, 109.9):                 # the next chunk is due by 110 s
            self.assertEqual(self.svc._sample_aggregate(now=t)[1], 1800.0)

    def test_drops_after_the_last_chunk(self):
        pp_at(self.e, 104.0, 7200, 7200, 4000, 1800.0)        # the prompt is fully read
        self.assertEqual(self.svc._sample_aggregate(now=104.5)[1], 1800.0)
        self.assertEqual(self.svc._sample_aggregate(now=106.0)[1], 1800.0)   # held
        self.assertEqual(self.svc._sample_aggregate(now=108.0)[1], 0.0)

    def test_engines_reading_together_add_up(self):
        other = StrataEngine("missing-executable", [], lazy=True)
        other._pp_seen, other.prefill_now = (0.0, 0), None
        self.svc.engine = mock.Mock(engines=[self.e, other])
        pp_at(self.e, 104.0, 3600, 60000, 4000, 900.0)
        pp_at(other, 104.5, 4000, 60000, 4000, 1000.0)
        self.assertEqual(self.svc._sample_aggregate(now=105.0)[1], 1900.0)


if __name__ == "__main__":
    unittest.main()
