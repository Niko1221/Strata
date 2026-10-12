"""CPU-only evaluation persistence; this is not real-backend gate evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from context_time_travel.store import Store, IntegrityError, InvalidOperation


def record(snapshot, task_root):
    return dict(kind='evaluation', schema=1, snapshot_id=snapshot, task_root=task_root,
                evaluator=dict(name='fixture-check', version='1', source_sha256=hashlib.sha256(b'code').hexdigest()),
                fixtures={'fixture.txt': hashlib.sha256(b'fixture').hexdigest()},
                metric=dict(name='accuracy', order='maximize'), seed=17,
                environment={'runtime': 'test-only'}, effect_policy={'mode': 'none'},
                outcomes=dict(mandatory={'syntax': 'PASS'}, score=0.75, unresolved=[]),
                resources=dict(wall_seconds=0.2, cpu_seconds=0.1, peak_rss_bytes=1024),
                artifacts={})


class EvaluationRecordTests(unittest.TestCase):
    def test_e01_invariant(self):
        self.assertTrue(callable(getattr(Store, 'put_evaluation', None)),
                        'Store must durably publish separate immutable evaluation provenance')
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            root = store.create_root('test-contract', [1], {'status': 'idle'})
            task = store.append(root, [2], {'status': 'idle'})
            candidate = store.append(task, [3], {'status': 'idle'})
            store.fork('main', candidate)
            def authoritative_bytes():
                return {str(p.relative_to(store.root)): p.read_bytes()
                        for folder in (store.objects, store.references)
                        for p in folder.glob('*.json')}
            before = authoritative_bytes()
            value = record(candidate, task)
            original = copy.deepcopy(value)
            identity = store.put_evaluation(value)
            self.assertEqual(len(identity), 64)
            self.assertEqual(store.put_evaluation(original), identity)
            value['outcomes']['score'] = 9
            loaded = Store(directory).get_evaluation(identity)
            self.assertEqual(loaded, original)
            loaded['evaluator']['version'] = 'mutated'
            self.assertEqual(store.get_evaluation(identity), original)
            for path, replacement in [('score', 0.5), ('source_sha256', 'a'*64)]:
                changed = copy.deepcopy(original)
                changed['outcomes' if path == 'score' else 'evaluator'][path] = replacement
                self.assertNotEqual(store.put_evaluation(changed), identity)
            self.assertEqual(authoritative_bytes(), before)
            self.assertEqual(store.head('main').tip, candidate)
            self.assertEqual(store.lineage(candidate)[1], [1,2,3])
            foreign = store.create_root('test-contract', [9], {'status': 'idle'})
            bad_root = copy.deepcopy(original); bad_root['task_root'] = foreign
            with self.assertRaises(InvalidOperation):
                store.put_evaluation(bad_root)
            for field in original:
                bad = copy.deepcopy(original); del bad[field]
                with self.subTest(missing=field), self.assertRaises(InvalidOperation):
                    store.put_evaluation(bad)
            for section, key, invalid in [
                    ('evaluator','source_sha256','not-a-hash'),
                    ('metric','order','arbitrary'), ('outcomes','score',float('nan')),
                    ('outcomes','mandatory',{'syntax':'SKIP'}),
                    ('resources','wall_seconds',-1), ('resources','peak_rss_bytes',True),
                    ('effect_policy','mode','unrestricted')]:
                bad = copy.deepcopy(original); bad[section][key] = invalid
                with self.subTest(section=section,key=key), self.assertRaises(InvalidOperation):
                    store.put_evaluation(bad)
            # Authorization metadata is derived, not a second report. Check
            # both locations explicitly instead of conflating equal basenames.
            files = list((Path(directory)/'evaluations').glob(identity+'.json'))
            self.assertEqual(len(files), 1)
            access = Path(directory)/'evaluation-access'/(identity+'.json')
            self.assertEqual(json.loads(access.read_bytes()), {'snapshot': candidate})
            self.assertEqual(set(Path(directory).rglob(identity+'.json')), {files[0], access})
            raw = files[0].read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), identity)
            corrupted = json.loads(raw); corrupted['outcomes']['score'] = 1.0
            files[0].write_text(json.dumps(corrupted))
            with self.assertRaises(IntegrityError):
                store.get_evaluation(identity)
            with self.assertRaises(IntegrityError):
                store.put_evaluation(original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
