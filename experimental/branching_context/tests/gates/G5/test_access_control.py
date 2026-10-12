"""Known IDs cannot bypass principal grants, including a hot GPU image."""
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'G1'))
import test_continuation as continuation
from context_time_travel.evaluation import evaluate


class AccessControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        if profile.get('calibration_status') != 'FROZEN':
            raise AssertionError('Frozen real-backend profile required')
        continuation.ContinuationTests.setUpClass.__func__(cls)

    tearDownClass=classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp=continuation.ContinuationTests.setUp
    check=continuation.ContinuationTests.check

    def test_g5_006_access_control_snapshot_ids(self):
        candidate=self.sut.feed(self.root,self.A)
        before=self.sut.snapshot_bytes(candidate)
        head=self.sut.store.fork('private',candidate)
        frontier=self.sut.next_logits(candidate)
        self.assertIn(candidate,self.sut.cache)
        guest=self.sut.for_principal('guest')
        self.assertIs(guest.cache,self.sut.cache)
        spec=dict(task_root=self.root,evaluator={'name':'count','version':'1',
                  'source':"result={'mandatory':{'valid':'PASS'},'score':len(candidate['tokens']),'unresolved':[]}"},
                  fixtures={},metric={'name':'tokens','order':'maximize'},seed=17,mandatory=['valid'],timeout_seconds=5)
        counters=self.backend._refresh_stats()
        for action in (lambda:guest.snapshot_bytes(candidate),lambda:guest.next_logits(candidate),
                       lambda:guest.feed(candidate,self.B),lambda:guest.checkpoint(candidate),
                       lambda:guest.store.fork('leak',candidate),lambda:evaluate(guest.store,candidate,spec),
                       lambda:guest.store.move('private',head.revision,self.root,'denied')):
            with self.assertRaises(PermissionError):
                action()
        after=self.backend._refresh_stats()
        for field in ('forward_tokens','allocations','copied_bytes','restored_running_bytes','loaded_file_bytes'):
            self.assertEqual(after[field],counters[field],field)
        self.assertEqual(guest.store.work_counters()['snapshot_read_bytes'],0)
        self.sut.store.grant(self.root,'guest',('read',))
        self.assertEqual(guest.snapshot_bytes(candidate),before)
        with self.assertRaises(PermissionError):
            guest.next_logits(candidate)
        self.sut.store.grant(self.root,'guest',('read','materialize','evaluate','fork'))
        self.assertEqual(guest.next_logits(candidate),frontier)
        self.assertEqual(evaluate(guest.store,candidate,spec)['outcomes']['mandatory'],{'valid':'PASS'})
        self.assertEqual(guest.store.fork('authorized',candidate).tip,candidate)
        self.sut.store.grant(self.root,'guest',())
        with self.assertRaises(PermissionError):
            guest.next_logits(candidate)
        self.assertEqual(self.sut.store.head('private'),head)
        self.assertEqual(self.sut.snapshot_bytes(candidate),before)
        self.check(candidate,self.P+self.A)


if __name__ == '__main__':
    unittest.main(verbosity=2)
