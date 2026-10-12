"""Composition of isolated scoring and immutable provenance."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from context_time_travel import evaluation
from context_time_travel.store import Store, InvalidOperation


class EvaluationCompositionTests(unittest.TestCase):
    def test_e02_report_composition(self):
        evaluate = getattr(evaluation, 'evaluate', None)
        self.assertTrue(callable(evaluate), 'Scoring must publish canonical evaluation provenance')
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            root = store.create_root('contract', [100], {'draw_cursor': 7})
            candidate = store.append(root, [117], {'draw_cursor': 9})
            head = store.fork('main', candidate)
            foreign = store.create_root('contract', [999], {'draw_cursor': 1})
            before = {p.name: p.read_bytes() for p in store.objects.glob('*.json')}
            source = "result = {'mandatory': {'valid': 'PASS'}, 'score': len(candidate['tokens']) + rng.random(), 'unresolved': []}"
            spec = dict(task_root=root, evaluator={'name': 'length', 'version': '1', 'source': source},
                        fixtures={'expected': b'2'}, metric={'name': 'length', 'order': 'maximize'},
                        seed=17, mandatory=['valid'], timeout_seconds=5)
            first = evaluate(store, candidate, spec)
            second = evaluate(store, candidate, spec)
            self.assertEqual(first['outcomes'], second['outcomes'])
            self.assertEqual(first['snapshot_id'], candidate)
            self.assertEqual(first['task_root'], root)
            self.assertEqual(first['evaluator']['source_sha256'], hashlib.sha256(source.encode()).hexdigest())
            self.assertEqual(first['fixtures']['expected'], hashlib.sha256(b'2').hexdigest())
            self.assertEqual(first['effect_policy']['mode'], 'isolated')
            self.assertIn('worker_sha256', first['environment'])
            self.assertIn('python_sha256', first['environment'])
            self.assertIn('bubblewrap_sha256', first['environment'])
            identity = store.put_evaluation(first)
            self.assertEqual(Store(directory).get_evaluation(identity), first)
            with self.assertRaises(InvalidOperation):
                evaluate(store, candidate, dict(spec, task_root=foreign))
            self.assertEqual({p.name: p.read_bytes() for p in store.objects.glob('*.json')}, before)
            self.assertEqual(store.head('main'), head)
            self.assertEqual(store.get_snapshot(candidate)['controller'], {'draw_cursor': 9})


if __name__ == '__main__':
    unittest.main(verbosity=2)
