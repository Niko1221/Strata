"""Capture the first projection input and result alongside consumed state."""
import json, os, unittest
from pathlib import Path
import test_n02_state as state_test

class ProjectionDiagnosticTests(unittest.TestCase):
    def test_n02_projection_is_observed(self):
        state_test.StateDiagnosticTests.test_n02_state_observes_consumed_buffers(self)
        p=Path(os.environ['CTT_SHARING_MEASUREMENTS']).parent/'state-comparison/comparison.json'
        r=json.loads(p.read_text());parts={x['part']['name']:x for x in r['comparisons']}
        self.assertIn('projection.mixed',parts,'Capture actual first-layer projection input')
        self.assertIn('projection.qkv',parts,'Capture actual first-layer projection result')
        for name in ('projection.mixed','projection.qkv'):
            self.assertEqual(parts[name]['part']['dtype'],'f32')
            self.assertGreater(parts[name]['part']['bytes'],0)

if __name__=='__main__':unittest.main(verbosity=2)
