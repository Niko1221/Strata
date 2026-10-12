"""A real native-pack oracle must actually consume even a one-token prompt."""
import array
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


class FreshFrontierTest(unittest.TestCase):
    def check_history(self, ids, capacity=None):
        command = json.loads(os.environ['STRATA_ORACLE_COMMAND'])
        if capacity is not None:
            command += ['--max-context', str(capacity)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'frontier.f32'
            env = os.environ.copy()
            env['STRATA_DUMP_FIRST_LOGITS'] = str(output)
            result = subprocess.run(command + ['--tokens', ','.join(map(str, ids))],
                                    env=env, capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists(), result.stdout + result.stderr)
            data = array.array('f')
            data.frombytes(output.read_bytes())
            self.assertEqual(len(data), 248320)
            self.assertTrue(all(map(math.isfinite, data)))
            work = re.findall(r'strata forward: prefill_tokens=(\d+) verifier_tokens=(\d+)', result.stderr)
            self.assertEqual(work, [(str(len(ids)-1), '1')], result.stderr)

    def test_single_token_fresh_forward(self):
        self.check_history([100])

    def test_two_token_fresh_forward(self):
        self.check_history([100, 117])

    def test_exact_context_frontier(self):
        self.check_history([100, 117, 134, 151], capacity=4)


if __name__ == '__main__':
    unittest.main()
