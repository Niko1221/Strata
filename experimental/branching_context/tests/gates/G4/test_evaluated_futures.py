"""Registered four-future observations from one real end-to-end execution."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


class EvaluatedFuturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        if profile.get('calibration_status') != 'FROZEN':
            raise AssertionError('Frozen real-backend profile required')
        path = Path(__file__).resolve().parents[3]/'examples/branching_demo.py'
        spec = importlib.util.spec_from_file_location('real_four_future_demo',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            cls.report = module.run_real(profile,directory)

    def test_g4_001_three_future_demonstration(self):
        self.assertEqual(len(set(self.report['candidates'].values())),4)
        self.assertEqual(self.report['checked_boundaries'],self.report['expected_boundaries'])
        for name in 'ABC':
            self.assertIn(name+':0',self.report['checked_boundaries'])

    def test_g4_002_rewind_and_new_fourth_future(self):
        self.assertNotEqual(self.report['candidates']['A'],self.report['candidates']['D'])
        self.assertIn('D:0',self.report['checked_boundaries'])
        self.assertTrue(self.report['snapshots_unchanged'])
        self.assertEqual(self.report['controller_before'],self.report['controller_after'])

    def test_g4_009_winning_branch_cold_replay(self):
        self.assertEqual(self.report['winner'],'C')
        self.assertEqual(self.report['failed_high_score'],'B')
        self.assertTrue(self.report['all_checkpoint_files_evicted'])
        self.assertTrue(self.report['cold_reopened'])
        self.assertIn('cold:C',self.report['checked_boundaries'])
        self.assertEqual(self.report['replayed_tokens'],self.report['expected_replay_tokens'])

    def test_g4_011_retained_loser_accessible(self):
        self.assertEqual(set(self.report['cold_checked']),set('ABCD'))
        self.assertTrue(self.report['snapshots_unchanged'])
        for name in 'ABD':
            self.assertIn('cold:'+name,self.report['checked_boundaries'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
