"""Execute the declared draft matrix against a pinned real host profile."""
import json
from pathlib import Path
import platform
import subprocess
import unittest

import test_access_control as access


class SupportedMatrixTests(unittest.TestCase):
    setUpClass = classmethod(access.AccessControlTests.setUpClass.__func__)
    tearDownClass = classmethod(access.AccessControlTests.tearDownClass.__func__)
    setUp = access.AccessControlTests.setUp
    check = access.AccessControlTests.check

    def test_g5_001_supported_matrix(self):
        matrix = json.loads((Path(__file__).resolve().parents[3]/'docs/support-matrix.json').read_bytes())
        self.assertEqual(len(matrix['targets']), 1, 'Every added target needs executed matrix coverage')
        row = matrix['targets'][0]
        self.assertEqual(platform.system(), row['os'])
        filesystem = subprocess.check_output(['findmnt', '-T', self.tmp.name, '-n', '-o', 'FSTYPE'], text=True).strip()
        self.assertEqual(filesystem, row['filesystem'])
        contract = self.profile['model_contract']
        self.assertEqual(contract['weights_sha256'], row['weights_sha256'])
        self.assertEqual(contract['architecture'], row['architecture'])
        self.assertEqual(contract['input_kind'], row['inputs'])
        self.assertEqual(self.profile['execution']['kv_dtype'], row['kv_format'])
        self.assertEqual(self.profile['execution']['capacity'], row['maximum_context'])
        self.assertIn('Verifier.run(T=1)', self.profile['execution']['operator'])
        baseline = json.loads(Path(self.profile['calibration_path']).read_bytes())
        self.assertEqual(baseline['weight_format'], row['weight_format'])
        self.assertEqual(baseline['status'], 'FROZEN')
        self.assertEqual(self.profile['calibration_status'], 'FROZEN')
        self.assertIn('NVIDIA', str(self.profile['hardware']))
        self.assertEqual(row['backend'], 'CUDA')
        self.assertEqual(row['active_execution_leases'], 1)
        original = self.sut.snapshot_bytes(self.root)
        tips = []
        for suffix in (self.A, self.B):
            tip = self.root
            self.check(tip, self.P)
            for index, token in enumerate(suffix):
                tip = self.sut.feed(tip, [token])
                self.check(tip, self.P+suffix[:index+1])
            tips.append((tip, self.P+suffix))
        self.sut.evict_all()
        for tip, history in tips:
            self.check(tip, history)
        self.assertEqual(self.sut.snapshot_bytes(self.root), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
