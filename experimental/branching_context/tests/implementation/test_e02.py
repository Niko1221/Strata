"""Adversarial evaluator isolation; CPU checks are not GPU acceptance."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from context_time_travel import evaluation


class EvaluatorIsolationTests(unittest.TestCase):
    def test_e02_invariant(self):
        run = getattr(evaluation, 'evaluate_read_only', None)
        self.assertTrue(callable(run), 'A measured isolated evaluator is required')
        candidate = {'tokens': [100, 117], 'controller': {'draw_cursor': 19}}
        before = copy.deepcopy(candidate)
        fixtures = {'recorded-tool-result': b'already executed; do not replay'}
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'outside.txt'
            sentinel.write_bytes(b'unchanged')
            source = '''
import os, socket
try:
    candidate['controller']['draw_cursor'] = 0
    raise AssertionError('mutable candidate')
except TypeError:
    pass
try:
    open(SENTINEL, 'wb').write(b'changed')
    raise AssertionError('host filesystem exposed')
except (FileNotFoundError, PermissionError):
    pass
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.1)
    raise AssertionError('external network exposed')
except OSError:
    pass
assert fixtures['recorded-tool-result'] == b'already executed; do not replay'
assert candidate['tokens'] == (100, 117)
result = {'mandatory': {'valid': 'PASS'}, 'score': rng.random(), 'unresolved': []}
'''.replace('SENTINEL', repr(str(sentinel)))
            def evaluate(code=source, timeout=5):
                return run(candidate, code, fixtures=fixtures, seed=17,
                           mandatory=('valid',), timeout_seconds=timeout)
            first, second = evaluate(), evaluate()
            self.assertEqual(first['outcomes'], second['outcomes'])
            self.assertEqual(first['outcomes']['mandatory'], {'valid': 'PASS'})
            self.assertEqual(first['source_sha256'], hashlib.sha256(source.encode()).hexdigest())
            self.assertEqual(first['fixture_sha256'], {
                k: hashlib.sha256(v).hexdigest() for k, v in fixtures.items()})
            self.assertEqual(first['effect_policy']['mode'], 'isolated')
            self.assertGreater(first['resources']['wall_seconds'], 0)
            self.assertGreater(first['resources']['peak_rss_bytes'], 0)
            for code, status in [
                ('raise RuntimeError("failed check")', 'ERROR'),
                ('result = {"mandatory": {}, "score": 99, "unresolved": []}', 'UNKNOWN'),
                ('result = {"mandatory": {"valid": "UNKNOWN"}, "score": 99, "unresolved": ["missing"]}', 'UNKNOWN'),
                ('result = {"mandatory": {"valid": "FAIL"}, "score": 99, "unresolved": []}', 'FAIL'),
                ('import time; time.sleep(30)', 'TIMEOUT'),
            ]:
                with self.subTest(status=status, code=code):
                    report = evaluate(code, timeout=0.5 if status == 'TIMEOUT' else 5)
                    self.assertEqual(report['outcomes']['mandatory']['valid'], status)
                    if status in ('ERROR', 'TIMEOUT'):
                        self.assertIsNone(report['outcomes']['score'])
            self.assertEqual(candidate, before)
            self.assertEqual(sentinel.read_bytes(), b'unchanged')
            self.assertEqual(fixtures, {'recorded-tool-result': b'already executed; do not replay'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
