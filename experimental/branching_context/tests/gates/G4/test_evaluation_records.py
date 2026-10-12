"""G4-008: immutable provenance around an actual model continuation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'G1'))
import test_continuation as continuation
from context_time_travel.store import IntegrityError, Store


class EvaluationRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        if profile.get('calibration_status') != 'FROZEN':
            raise AssertionError('Real acceptance requires a frozen execution-specific baseline')
        continuation.ContinuationTests.setUpClass.__func__(cls)

    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check

    def test_g4_008_evaluation_records_are_immutable(self):
        candidate = self.sut.feed(self.root, self.A)
        history = self.P + self.A
        store = self.sut.store
        head = store.fork('main', candidate)
        snapshots = {p.name: p.read_bytes() for p in store.objects.glob('*.json')}
        self.check(candidate, history)
        started, cpu_started = time.monotonic(), time.process_time()
        count = self.sut.consumed(candidate)
        self.assertEqual(count, len(history))
        value = dict(kind='evaluation', schema=1, snapshot_id=candidate, task_root=self.root,
                     evaluator=dict(name='consumed-history-length', version='1',
                                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                     fixtures={'history': hashlib.sha256(json.dumps(history).encode()).hexdigest()},
                     metric=dict(name='consumed_tokens', order='maximize'), seed=0,
                     environment={'execution_profile_id': self.profile['execution_profile_id'],
                                  'resource_scope': 'Python evaluator process; RSS lifetime high-water'},
                     effect_policy={'mode': 'none'},
                     outcomes=dict(mandatory={'consumed_boundary': 'PASS'}, score=count, unresolved=[]),
                     resources=dict(wall_seconds=time.monotonic()-started,
                                    cpu_seconds=time.process_time()-cpu_started,
                                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                     artifacts={})
        original = copy.deepcopy(value)
        identity = store.put_evaluation(value)
        value['outcomes']['score'] = -1
        self.assertEqual(Store(self.tmp.name).get_evaluation(identity), original)
        changed = copy.deepcopy(original)
        changed['evaluator']['source_sha256'] = 'a'*64
        changed['outcomes']['score'] = 0
        self.assertNotEqual(store.put_evaluation(changed), identity)
        self.assertEqual(store.get_evaluation(identity), original)
        path = store.root / 'evaluations' / (identity + '.json')
        raw = path.read_bytes()
        try:
            path.write_text(json.dumps(changed))
            with self.assertRaises(IntegrityError):
                store.get_evaluation(identity)
        finally:
            path.write_bytes(raw)
        self.assertEqual(store.head('main'), head)
        self.assertEqual({p.name: p.read_bytes() for p in store.objects.glob('*.json')}, snapshots)
        self.check(candidate, history)
        future = self.sut.feed(candidate, self.B)
        self.check(future, history + self.B)
        self.check(candidate, history)


if __name__ == '__main__':
    unittest.main(verbosity=2)
