"""Guard the statistics used in the published cache-latency charts."""
import unittest
from cache_latency_report import quantile, summarize


def row(time, *, cached=1024, error=None, phase='measured', output=128):
    return dict(build='main', mode='pinned', prefix_tokens=1024, phase=phase,
                ttft_s=time, total_s=None if time is None else time+1, error=error,
                usage={'input_tokens': 1124, 'output_tokens': output,
                       'input_tokens_details': {'cached_tokens': cached}})


class StatisticsTest(unittest.TestCase):
    def test_linear_quantiles(self):
        self.assertEqual(quantile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(quantile([1, 2, 3, 4], 99), 3.97)
        self.assertEqual(quantile([8], 95), 8)
        self.assertIsNone(quantile([], 95))

    def test_misses_stay_in_latency_population(self):
        got = summarize([row(1), row(9, cached=0)])[0]
        self.assertEqual(got['ttft_s_p50'], 5)
        self.assertEqual(got['cache_hit_rate'], .5)

    def test_failures_count_but_do_not_become_fast_successes(self):
        got = summarize([row(2), row(None, error='timeout')])[0]
        self.assertEqual((got['attempted'], got['n'], got['errors']), (2, 1, 1))
        self.assertEqual(got['ttft_s_p99'], 2)

    def test_warmups_and_alternates_are_not_samples(self):
        got = summarize([row(2), row(50, phase='warmup'), row(90, phase='alternate')])[0]
        self.assertEqual(got['n'], 1)
        self.assertEqual(got['ttft_s_p50'], 2)

    def test_early_stop_is_not_128_token_completion(self):
        got = summarize([row(2), row(1, output=8)])[0]
        self.assertEqual(got['n'], 2)
        self.assertEqual(got['replies_128'], 1)
        self.assertEqual(got['total_128_s_p50'], 3)

    def test_failed_save_does_not_count_as_generation_attempt(self):
        failed = row(None, error='save failed')
        failed['restore'] = {'status': 0}
        got = summarize([failed])[0]
        self.assertEqual((got['scheduled'], got['attempted'], got['blocked']), (1, 0, 1))
        self.assertIsNone(got['ttft_s_p50'])


if __name__ == '__main__':
    unittest.main()
