"""Four futures must be demonstrated with a real pinned backend and oracle."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


class EvaluatedFuturesTests(unittest.TestCase):
    def test_e04_invariant(self):
        path = Path(__file__).resolve().parents[2]/'examples/branching_demo.py'
        spec = importlib.util.spec_from_file_location('evaluated_branching_demo', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run = getattr(module, 'run_real', None)
        self.assertTrue(callable(run), 'Four-future demonstration must exercise the real backend')
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        self.assertEqual(profile['calibration_status'], 'FROZEN')
        with tempfile.TemporaryDirectory() as directory:
            report = run(profile, directory)
        self.assertEqual(report['winner'], 'C')
        self.assertEqual(set(report['candidates']), {'A','B','C','D'})
        self.assertEqual(len(set(report['candidates'].values())), 4)
        self.assertEqual(report['failed_high_score'], 'B')
        self.assertTrue(report['all_checkpoint_files_evicted'])
        self.assertTrue(report['cold_reopened'])
        self.assertTrue(report['snapshots_unchanged'])
        self.assertEqual(report['replayed_tokens'], report['expected_replay_tokens'])
        self.assertEqual(set(report['cold_checked']), {'A','B','C','D'})
        self.assertEqual(report['checked_boundaries'], report['expected_boundaries'])
        self.assertEqual(report['controller_before'], report['controller_after'])
        self.assertLessEqual(report['max_absolute_error'], profile['atol'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
