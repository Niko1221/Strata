"""Metrics invariants without loading Torch or model shards."""
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from serve import Service


class ServiceMetricsTests(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(api_key_file=None, device='npu:1', context=8192, weight_dtype='fp32')
        with patch.object(Service, '_load'):
            self.service = Service(args)
        self.addCleanup(self.service.worker.shutdown, wait=True)
        self.service.engine = SimpleNamespace(
            layers=48, md={}, w=SimpleNamespace(report=lambda: {}, cache={}, used=0, budget=12 * 1024**3))
        self.service.npu_smi = None

    def generate(self, prompt, count, temperature, callback, progress, cancelled):
        progress(len(prompt), 2)
        for token in range(count):
            callback(token)
        return dict(tokens=list(range(count)), prompt_tokens=prompt, prefill_seconds=2,
                    decode_seconds=4, decode_tokens_per_second=(count - 1) / 4)

    def test_totals_survive_recent_request_eviction_and_failure(self):
        self.service.engine.generate = self.generate
        for _ in range(125):
            self.service.request([785, 1], 3, 0, None, threading.Event()).result(timeout=5)
        metrics = self.service.metrics()
        self.assertEqual(len(metrics['requests']), 120)
        self.assertEqual(metrics['totals'], dict(requests=125, failed=0, prompt_tokens=250, output_tokens=375))
        with patch.object(self.service.engine, 'generate', side_effect=RuntimeError('test failure')):
            with self.assertRaises(RuntimeError):
                self.service.request([785], 1, 0, None, threading.Event()).result(timeout=5)
        self.assertEqual(self.service.metrics()['totals'], dict(requests=125, failed=1, prompt_tokens=250, output_tokens=375))
        self.assertFalse(self.service.lock.locked())

    def test_live_prefill_counts_only_completed_tokens_and_has_no_early_decode_rate(self):
        entered, release = threading.Event(), threading.Event()
        def generate(prompt, count, temperature, callback, progress, cancelled):
            progress(1, 2)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test did not release worker')
            return self.generate(prompt, count, temperature, callback, progress, cancelled)
        self.service.engine.generate = generate
        future = self.service.request([785, 1, 2, 3], 3, 0, None, threading.Event())
        try:
            self.assertTrue(entered.wait(5))
            live = self.service.metrics()['live']
            self.assertEqual(live['prompt_done'], 1)
            self.assertEqual(live['prefill_tok_s'], 0.5)
            self.assertIsNone(live['tok_s'])
            with self.assertRaises(BlockingIOError):
                self.service.request([785], 1, 0, None, threading.Event())
        finally:
            release.set()
        future.result(timeout=5)
        self.assertEqual(self.service.metrics()['live']['state'], 'idle')

    def test_board_capacity_is_not_mixed_with_driver_usable_memory(self):
        gib = 1024**3
        self.service.torch = SimpleNamespace(npu=SimpleNamespace(
            mem_get_info=lambda device: (40*gib, 61*gib),
            memory_allocated=lambda device: 12*gib,
            memory_reserved=lambda device: 16*gib))
        with patch.object(self.service, '_read_npu', return_value=dict(memory_total_bytes=64*gib, memory_used_bytes=20*gib)):
            metrics = self.service.npu_metrics()
        self.assertEqual(metrics['memory_total_bytes'], 64*gib)
        self.assertEqual(metrics['memory_used_bytes'], 20*gib)
        self.assertEqual(metrics['memory_free_bytes'], 44*gib)
        self.assertEqual(metrics['memory_allocated_bytes'], 12*gib)
        with patch.object(self.service, '_read_npu', return_value={}):
            metrics = self.service.npu_metrics()
        self.assertEqual(metrics['memory_total_bytes'], 61*gib)
        self.assertEqual(metrics['memory_used_bytes'], 21*gib)


if __name__ == '__main__':
    unittest.main()
