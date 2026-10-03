"""The benchmark must distinguish queue wait, TTFT and completion latency."""
import unittest
from tools.prefill_preempt_bench import latency_metrics


class BenchTimings(unittest.TestCase):
    def test_queue_wait_is_not_decode_or_total_latency(self):
        metrics = latency_metrics(10, {'started': 12, 'first_token': 13, 'done': 20})
        self.assertEqual(metrics, {'b_queue_wait_s': 2, 'b_ttft_s': 3, 'b_total_s': 10})

    def test_no_output_has_no_ttft(self):
        metrics = latency_metrics(10, {'started': 12, 'done': 20})
        self.assertIsNone(metrics['b_ttft_s'])

    def test_missing_or_invalid_pump_timestamps_fail(self):
        with self.assertRaises(KeyError): latency_metrics(10, {'done': 20})
        for observed in [{'started': 9, 'done': 20}, {'started': 12, 'done': 11},
                         {'started': 12, 'first_token': 11, 'done': 20},
                         {'started': 12, 'first_token': 21, 'done': 20}]:
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                latency_metrics(10, observed)


if __name__ == '__main__': unittest.main()
