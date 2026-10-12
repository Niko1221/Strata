"""Selection uses immutable reports; adoption uses the existing CAS operation."""
import copy
import itertools
import tempfile
import unittest

import test_e01
from context_time_travel import evaluation
from context_time_travel.store import Store, Conflict, InvalidOperation


class SelectionTests(unittest.TestCase):
    def test_e03_invariant(self):
        choose = getattr(evaluation, 'choose_validated', None)
        self.assertTrue(callable(choose), 'Selection must filter mandatory checks before scores')
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            root = store.create_root('contract', [1], {'next_draw': 7})
            candidates = [store.append(root, suffix, {'next_draw': 9}) for suffix in ([2], [3], [4,5])]
            head = store.fork('main', root)
            records = []
            for candidate, score, status in zip(candidates, (7,10,8), ('PASS','FAIL','PASS')):
                record = test_e01.record(candidate, root)
                record['outcomes'] = dict(mandatory={'syntax':status}, score=score, unresolved=[])
                records.append(store.put_evaluation(record))
            before = {p.name:p.read_bytes() for p in store.objects.glob('*.json')}
            for order in itertools.permutations(records):
                winner = choose(store, order, mandatory=('syntax',))
                self.assertEqual(winner['snapshot_id'], candidates[2])
            self.assertEqual(store.get_evaluation(records[1])['outcomes']['mandatory']['syntax'], 'FAIL')
            rejected = []
            for status in ('FAIL','UNKNOWN','ERROR','TIMEOUT'):
                value = store.get_evaluation(records[0])
                value['outcomes']['mandatory']['syntax'] = status
                rejected.append(store.put_evaluation(value))
            for identity in rejected:
                self.assertIsNone(choose(store, [identity], mandatory=('syntax',)))
            self.assertIsNone(choose(store, [], mandatory=('syntax',)))
            self.assertIsNone(choose(store, records, mandatory=('syntax','missing')))
            unresolved = store.get_evaluation(records[0]); unresolved['outcomes']['unresolved']=['unknown test']
            self.assertIsNone(choose(store, [store.put_evaluation(unresolved)], mandatory=('syntax',)))
            tied = []
            for identity in records:
                value = store.get_evaluation(identity)
                value['outcomes'] = dict(mandatory={'syntax':'PASS'},score=8,unresolved=[])
                tied.append(store.put_evaluation(value))
            expected = min(candidates[:2])
            for order in itertools.permutations(tied):
                self.assertEqual(choose(store, order, mandatory=('syntax',))['snapshot_id'], expected)
            incompatible = store.get_evaluation(records[0]); incompatible['evaluator']['version']='2'
            with self.assertRaises(InvalidOperation):
                choose(store, [records[2],store.put_evaluation(incompatible)], mandatory=('syntax',))
            conflicting = store.get_evaluation(records[0]); conflicting['outcomes']['score']=100
            with self.assertRaises(InvalidOperation):
                choose(store, [records[0],store.put_evaluation(conflicting)], mandatory=('syntax',))
            winner = choose(store, records, mandatory=('syntax',))
            moved = store.move('main',head.revision,winner['snapshot_id'],'adopt')
            self.assertEqual(Store(directory).move('main',head.revision,winner['snapshot_id'],'adopt'),moved)
            with self.assertRaises(Conflict):
                store.move('main',head.revision,candidates[0],'stale-adopt')
            with self.assertRaises(InvalidOperation):
                store.move('main',head.revision,candidates[0],'adopt')
            self.assertEqual(store.head('main'),moved)
            self.assertEqual(moved.revision,head.revision+1)
            self.assertEqual({p.name:p.read_bytes() for p in store.objects.glob('*.json')},before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
