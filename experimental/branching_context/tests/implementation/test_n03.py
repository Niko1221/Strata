"""Executed current-operator calibration; no synthetic numerical qualification."""
import hashlib
import json
import math
import os
from pathlib import Path
import unittest


class NumericalProfileTests(unittest.TestCase):
    def test_n03_invariant(self):
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        self.assertEqual(profile.get('calibration_status'), 'FROZEN',
                         'Current corrected operator needs its own executed frozen baseline')
        baseline = json.loads(Path(profile['calibration_path']).read_text())
        self.assertEqual(baseline['status'], 'FROZEN')
        self.assertEqual(baseline['calibration_lengths'], [1,31,127,255,256,257,511,1024,4096])
        self.assertEqual(baseline['repetitions'], 3)
        command = profile['execution']['oracle_command']
        self.assertEqual(baseline['command'], command)
        self.assertEqual(baseline['oracle_sha256'], hashlib.sha256(Path(command[0]).read_bytes()).hexdigest())
        self.assertNotEqual(Path(command[0]).resolve(), Path(profile['execution']['sut_command'][0]).resolve())
        expected = {(n,r) for n in baseline['calibration_lengths'] for r in range(3)}
        rows = baseline['commands']
        self.assertEqual(len(rows), len(expected))
        self.assertEqual({(r['length'],r['repeat']) for r in rows}, expected)
        for row in rows:
            self.assertEqual(row['exit_code'], 0)
            counters = row['native_counters']
            self.assertEqual(counters['forward_tokens'], row['length'])
            self.assertEqual(counters['consumed'], row['length'])
            self.assertEqual(counters['position_max'], row['length']-1)
            self.assertEqual(row['bytes'], 248320*4)
        comparisons = baseline['comparisons']
        self.assertEqual(len(comparisons), 18)
        self.assertEqual({(r['length'],r['repeat']) for r in comparisons},
                         {(n,r) for n in baseline['calibration_lengths'] for r in (1,2)})
        worst = max(r['max_absolute_error'] for r in comparisons)
        self.assertTrue(math.isfinite(worst) and worst >= 0)
        self.assertEqual(baseline['baseline_max_absolute_error'], worst)
        self.assertEqual(baseline['atol'], 2*worst)
        self.assertEqual(profile['atol'], baseline['atol'])
        self.assertEqual(profile['rtol'], 0)
        self.assertEqual(baseline['rtol'], 0)
        self.assertEqual(baseline['model_files'], profile['model_files'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
