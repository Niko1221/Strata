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


if __name__ == '__main__':
    unittest.main()
