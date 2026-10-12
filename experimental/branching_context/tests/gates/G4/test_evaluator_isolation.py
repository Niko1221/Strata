"""Registered G4 isolation checks require a frozen real-backend profile."""
from pathlib import Path
import unittest

import test_evaluation_records as records
from context_time_travel.evaluation import evaluate


class EvaluatorIsolationTests(unittest.TestCase):
    setUpClass = classmethod(records.EvaluationRecordTests.setUpClass.__func__)
    tearDownClass = classmethod(records.EvaluationRecordTests.tearDownClass.__func__)
    setUp = records.EvaluationRecordTests.setUp
    check = records.EvaluationRecordTests.check

    def evaluate(self, candidate, source=None):
        source = source or "result = {'mandatory': {'valid': 'PASS'}, 'score': len(candidate['tokens']) + rng.random(), 'unresolved': []}"
        return evaluate(self.sut.store, candidate, dict(task_root=self.root,
            evaluator={'name': 'deterministic-fixture', 'version': '1', 'source': source},
            fixtures={'recorded-result': b'completed tool output'}, seed=17,
            metric={'name': 'score', 'order': 'maximize'}, mandatory=['valid'], timeout_seconds=5))

    def test_g4_003_evaluator_read_only(self):
        a, b = self.sut.feed(self.root, self.A), self.sut.feed(self.root, self.B)
        head = self.sut.store.fork('main', a)
        before = {s: self.sut.snapshot_bytes(s) for s in (self.root, a, b)}
        self.check(a, self.P+self.A)
        report = self.evaluate(a, """
try:
    candidate['controller']['corrupted'] = True
    raise AssertionError('mutable controller')
except TypeError:
    pass
result = {'mandatory': {'valid': 'PASS'}, 'score': 1, 'unresolved': []}
""")
        self.assertEqual(report['outcomes']['mandatory'], {'valid': 'PASS'})
        self.assertEqual(self.sut.store.head('main'), head)
        self.assertEqual({s: self.sut.snapshot_bytes(s) for s in before}, before)
        self.check(a, self.P+self.A)
        self.check(b, self.P+self.B)
        future = self.sut.feed(a, self.B)
        self.check(future, self.P+self.A+self.B)

    def test_g4_004_evaluator_reproducibility(self):
        a, b = self.sut.feed(self.root, self.A), self.sut.feed(self.root, self.B)
        first = self.evaluate(a)
        self.evaluate(b)
        repeated = self.evaluate(a)
        for field in ('outcomes', 'evaluator', 'fixtures', 'seed', 'environment', 'snapshot_id'):
            self.assertEqual(first[field], repeated[field])
        self.check(a, self.P+self.A)
        self.check(b, self.P+self.B)

    def test_g4_005_evaluator_side_effects_sandbox(self):
        marker = Path(self.tmp.name)/'irreversible-tool-sentinel'
        marker.write_bytes(b'not executed again')
        source = """
import socket
assert fixtures['recorded-result'] == b'completed tool output'
try:
    open(MARKER, 'wb').write(b'replayed')
    raise AssertionError('host side effect escaped')
except (FileNotFoundError, PermissionError):
    pass
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.1)
    raise AssertionError('network escaped')
except OSError:
    pass
result = {'mandatory': {'valid': 'PASS'}, 'score': 1, 'unresolved': []}
""".replace('MARKER', repr(str(marker)))
        before = self.sut.snapshot_bytes(self.root)
        for _ in range(2):
            report = self.evaluate(self.root, source)
            self.assertEqual(report['outcomes']['mandatory'], {'valid': 'PASS'})
            self.assertEqual(report['effect_policy']['mode'], 'isolated')
        self.assertEqual(marker.read_bytes(), b'not executed again')
        self.assertEqual(self.sut.snapshot_bytes(self.root), before)
        self.check(self.root, self.P)

    def test_g4_010_evaluation_does_not_consume_rng(self):
        start = self.sut.start_generation(self.root,
            dict(allowed=sorted(set(self.A+self.B)), temperature=1.0, eos_tokens=[]),
            'evaluation-noninterference')
        mid = self.sut.generate(start, 3).snapshot_id
        before = self.sut.snapshot_bytes(mid)
        controller = self.sut.controller(mid)
        self.assertGreater(controller['next_draw'], 0)
        expected = self.sut.generate(mid, 4)
        report = self.evaluate(mid, """
for _ in range(10000):
    rng.random()
result = {'mandatory': {'valid': 'PASS'}, 'score': rng.random(), 'unresolved': []}
""")
        self.assertEqual(report['outcomes']['mandatory'], {'valid': 'PASS'})
        self.assertEqual(self.sut.snapshot_bytes(mid), before)
        self.assertEqual(self.sut.controller(mid), controller)
        actual = self.sut.generate(mid, 4)
        self.assertEqual(actual, expected)
        for boundary in actual.boundaries:
            self.check(boundary, self.sut.history(boundary))


if __name__ == '__main__':
    unittest.main(verbosity=2)
