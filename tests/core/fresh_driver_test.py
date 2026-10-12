"""Real Q8 full-history oracle checks; no SUT process or checkpoint input."""
import array
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class FreshDriverTest(unittest.TestCase):
    def forward(self, tokens, capacity=4096):
        command = json.loads(os.environ['STRATA_FRESH_COMMAND'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frontier.f32'
            env = dict(os.environ, STRATA_DUMP_FIRST_LOGITS=str(path))
            result = subprocess.run(command + ['--tokens', ','.join(map(str, tokens)),
                '--max-context', str(capacity)], env=env, capture_output=True, text=True, timeout=1800)
            self.assertEqual(result.returncode, 0, result.stderr[-6000:])
            work = json.loads(result.stdout)
            self.assertEqual(work['status'], 'ok')
            self.assertEqual(work['forward_tokens'], len(tokens))
            self.assertEqual(work['position_max'], len(tokens)-1)
            self.assertEqual(work['consumed'], len(tokens))
            raw = path.read_bytes()
            floats = array.array('f', raw)
            self.assertEqual(len(floats), 248320)
            self.assertTrue(all(map(math.isfinite, floats)))
            self.assertEqual(work['logits_bytes'], len(raw))
            return raw

    def test_single_token_consumed(self):
        self.forward([100])

    def test_full_history_at_capacity(self):
        self.forward([100, 117, 134, 151], capacity=4)

    def test_repeated_fresh_requests_and_wrong_history(self):
        ids = [100 + i*17 % 137 for i in range(31)]
        first = self.forward(ids)
        self.assertEqual(self.forward(ids), first)
        self.assertNotEqual(self.forward([201]+ids[1:]), first)


if __name__ == '__main__':
    unittest.main()
