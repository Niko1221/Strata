"""--monitor-aggregate: the Monitor's speed and prefill as tokens per second summed over every running stream."""
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine


def service():
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                  ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    svc.monitor_aggregate = True
    return svc


class DecodeSum(unittest.TestCase):
    def test_one_stream(self):
        svc = service()
        st, rate = {"generated": 0}, []
        svc._sample_aggregate(now=100.0)
        for n in range(1, 31):
            svc._note(n, [], st=st, rate=rate)
        self.assertEqual(svc._sample_aggregate(now=101.0), (30.0, 0.0))

    def test_parallel_streams_add_up(self):
        svc = service()
        a, b = ({"generated": 0}, []), ({"generated": 0}, [])
        svc._sample_aggregate(now=100.0)
        for n in range(1, 21):
            svc._note(n, [], st=a[0], rate=a[1])
        for n in range(1, 16):
            svc._note(n, [], st=b[0], rate=b[1])
        self.assertEqual(svc._sample_aggregate(now=101.0)[0], 35.0)
        # the next second only counts what is new
        svc._note(25, [], st=a[0], rate=a[1])
        self.assertEqual(svc._sample_aggregate(now=102.0)[0], 5.0)

    def test_a_restarted_count_never_goes_negative(self):
        svc = service()
        st = {"generated": 0}
        svc._sample_aggregate(now=100.0)
        svc._note(10, [], st=st, rate=[])
        svc._note(3, [], st=st, rate=[])       # a retry from token 0
        self.assertEqual(svc._sample_aggregate(now=101.0)[0], 10.0)

    def test_divides_by_the_real_interval(self):
        svc = service()
        svc._sample_aggregate(now=100.0)
        svc._note(40, [], st={"generated": 0}, rate=[])
        self.assertEqual(svc._sample_aggregate(now=102.0)[0], 20.0)

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

    def test_service_sums_engine_counts(self):
        svc = service()
        svc.engine.prefill_read_total = 0
        svc._sample_aggregate(now=100.0)
        svc.engine.prefill_read_total = 3000
        self.assertEqual(svc._sample_aggregate(now=101.0)[1], 3000.0)


if __name__ == "__main__":
    unittest.main()
