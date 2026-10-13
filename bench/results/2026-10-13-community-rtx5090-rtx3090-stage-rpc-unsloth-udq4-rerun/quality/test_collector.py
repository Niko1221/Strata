"""Offline regressions for the published collector; never calls an LLM."""
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_quality


class Once:
    def __init__(self):
        self.done = False

    def is_set(self):
        return self.done

    def wait(self, seconds):
        self.done = True


class CollectorTests(unittest.TestCase):
    def check_status(self, code):
        with tempfile.TemporaryDirectory(prefix='strata-collector-regression-') as folder:
            sampler = run_quality.Sampler(SimpleNamespace(out=Path(folder), url='http://unused'))
            sampler.stop = Once()

            def fake_fetch(url, path):
                if path == '/api/requests':
                    error = urllib.error.HTTPError(url, code, 'test', {}, None)
                    error.close()
                    raise error
                return {'live': {'state': 'idle'}, 'totals': {}, 'requests': [], 'nodes': []}

            with patch.object(run_quality, 'fetch', side_effect=fake_fetch), patch.object(
                    run_quality, 'command', return_value={'exit_code': 0, 'stdout': '', 'stderr': ''}):
                sampler.loop()
            return sampler

    def test_disabled_optional_monitor_keeps_metrics_sample(self):
        sampler = self.check_status(404)
        self.assertEqual(sampler.samples, 1)
        self.assertEqual(sampler.errors, [])
        self.assertTrue(sampler.api_monitor_disabled)

    def test_real_http_failure_is_not_silently_ignored(self):
        sampler = self.check_status(500)
        self.assertEqual(sampler.samples, 0)
        self.assertEqual(len(sampler.errors), 1)
        self.assertIn('500', sampler.errors[0]['error'])

    def test_baseline_readiness_is_signalled_after_sample(self):
        sampler = self.check_status(404)
        self.assertTrue(sampler.ready.is_set())

    def test_guard_counts_only_worker_pid(self):
        with tempfile.TemporaryDirectory(prefix='strata-guard-') as folder:
            args = SimpleNamespace(out=Path(folder), url='http://unused', worker_pid=42)
            sampler = run_quality.Sampler(args)
            sampler.stop = Once()
            payload = {'live': {'state': 'idle'}, 'totals': {}, 'requests': [], 'nodes': []}
            gpu = {'exit_code': 0, 'stdout': '99, 24000\n42, 18000\n', 'stderr': ''}
            with patch.object(run_quality, 'fetch', return_value=payload), patch.object(run_quality, 'command', return_value=gpu):
                sampler.loop()
            self.assertIsNone(sampler.guard_error)

    def test_worker_guard_violation_is_recorded(self):
        with tempfile.TemporaryDirectory(prefix='strata-guard-') as folder:
            args = SimpleNamespace(out=Path(folder), url='http://unused', worker_pid=42)
            sampler = run_quality.Sampler(args)
            sampler.stop = Once()
            payload = {'live': {'state': 'idle'}, 'totals': {}, 'requests': [], 'nodes': []}
            gpu = {'exit_code': 0, 'stdout': '42, 21000\n', 'stderr': ''}
            with patch.object(run_quality, 'fetch', return_value=payload), patch.object(run_quality, 'command', return_value=gpu):
                sampler.loop()
            self.assertIn('20480', sampler.guard_error)


if __name__ == '__main__':
    unittest.main()
