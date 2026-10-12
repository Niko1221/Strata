"""The advertised prototype matrix must be explicit and evidence scoped."""
import json
from pathlib import Path
import unittest


class MatrixTests(unittest.TestCase):
    def test_q01_invariant(self):
        path = Path(__file__).resolve().parents[2]/'docs/support-matrix.json'
        self.assertTrue(path.is_file(), 'Missing explicit supported-profile matrix')
        matrix = json.loads(path.read_bytes())
        self.assertEqual(matrix['schema'], 1)
        self.assertEqual(matrix['qualification'], 'DRAFT_VALIDATION_PENDING')
        self.assertEqual(len(matrix['targets']), 1)
        row = matrix['targets'][0]
        self.assertEqual(row['weight_format'], 'Q8_0')
        self.assertEqual(row['kv_format'], 'FP16')
        self.assertEqual(row['architecture'], 'qwen4exp')
        self.assertEqual(row['operator'], 'token-at-a-time')
        self.assertEqual(row['os'], 'Linux')
        self.assertEqual(row['filesystem'], 'ext4')
        self.assertEqual(row['backend'], 'CUDA')
        self.assertEqual(row['inputs'], 'token_ids')
        self.assertEqual(row['active_execution_leases'], 1)
        self.assertEqual(row['maximum_context'], 4096)
        self.assertEqual(len(row['weights_sha256']), 64)
        self.assertEqual(matrix['certified_guarantees'], [])
        exclusions = {item['id']:item for item in matrix['outside_matrix']}
        for name in ('batched-prefill', 'Q4', 'quantized-KV', 'Windows', 'HIP', 'SYCL',
                     'multimodal', 'concurrent-GPU-leases', 'power-loss', 'other-models'):
            self.assertIn(name, exclusions)
            self.assertIn(exclusions[name]['status'], ('UNMEASURED', 'UNSUPPORTED', 'FAIL'))
            self.assertTrue(exclusions[name]['reason'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
