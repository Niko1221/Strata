"""Real GPU materialization stays reachable during explicit history collection."""
import multiprocessing
import unittest

import test_access_control as access
from context_time_travel.store import Store, NotFound
from context_time_travel.evaluation import evaluate


def concurrent_operation(directory, root, candidate, operation, barrier, results):
    store = Store(directory)
    barrier.wait(timeout=30)
    if operation == 'collect':
        value = store.collect_history(root)
    elif operation == 'fork':
        value = store.fork('retained', candidate).tip
    elif operation == 'rewind':
        value = store.move('rewinding', 0, root, 'rewind-during-gc').tip
    elif operation == 'evaluate':
        source = "result = {'mandatory': {'valid': 'PASS'}, 'score': len(candidate['tokens']), 'unresolved': []}"
        spec = dict(task_root=root, evaluator={'name': 'length', 'version': '1', 'source': source},
                    fixtures={}, metric={'name': 'length', 'order': 'maximize'},
                    seed=17, mandatory=['valid'], timeout_seconds=5)
        value = store.put_evaluation(evaluate(store, candidate, spec))
    else:
        raise AssertionError('Unknown concurrent operation')
    results.put((operation, value))


class RetentionTests(unittest.TestCase):
    setUpClass=classmethod(access.AccessControlTests.setUpClass.__func__)
    tearDownClass=classmethod(access.AccessControlTests.tearDownClass.__func__)
    setUp=access.AccessControlTests.setUp
    check=access.AccessControlTests.check

    def test_g5_008_retention_and_gc(self):
        candidate=self.sut.feed(self.root,self.A)
        saved=self.sut.snapshot_bytes(candidate)
        self.sut.store.release_history(candidate)
        self.sut.evict_all()
        self.assertEqual(self.sut.snapshot_bytes(candidate),saved)
        self.check(candidate,self.P+self.A)
        self.sut.store.retain('historical',candidate)
        self.assertEqual(self.sut.store.collect_history(self.root)['deleted'],[])
        self.sut.store.release_retained('historical')
        with self.sut.materialize(candidate):
            self.assertEqual(self.sut.store.collect_history(self.root)['deleted'],[])
            self.assertEqual(self.sut.snapshot_bytes(candidate),saved)
        self.assertEqual(self.sut.store.collect_history(self.root)['deleted'],[candidate])
        # A remaining hot image must not resurrect explicitly deleted history.
        self.assertIn(candidate,self.sut.cache)
        with self.assertRaises(NotFound):
            self.sut.next_logits(candidate)
        self.check(self.root,self.P)

    def test_g5_009_concurrent_gc_and_load(self):
        candidate=self.sut.feed(self.root,self.A)
        garbage=self.sut.feed(self.root,self.B)
        self.sut.store.release_history(candidate)
        self.sut.store.release_history(garbage)
        self.sut.store.fork('rewinding',candidate)
        saved=self.sut.snapshot_bytes(candidate)
        context=multiprocessing.get_context('spawn')
        results=context.Queue()
        barrier=context.Barrier(4)
        workers=[context.Process(target=concurrent_operation,
                 args=(self.tmp.name,self.root,candidate,operation,barrier,results))
                 for operation in ('collect','fork','rewind','evaluate')]
        try:
            with self.sut.materialize(candidate):
                for worker in workers:worker.start()
                observed=dict(results.get(timeout=40) for _ in workers)
                for worker in workers:
                    worker.join(10)
                    self.assertEqual(worker.exitcode,0)
                self.assertEqual(observed['collect']['deleted'],[garbage])
                self.assertEqual(observed['fork'],candidate)
                self.assertEqual(observed['rewind'],self.root)
                report=self.sut.store.get_evaluation(observed['evaluate'])
                self.assertEqual(report['snapshot_id'],candidate)
                self.assertEqual(report['outcomes']['mandatory'],{'valid':'PASS'})
                self.assertEqual(report['outcomes']['score'],len(self.P+self.A))
                head=self.sut.store.head('rewinding')
                self.assertEqual((head.tip,head.revision),(self.root,1))
                self.assertEqual(self.sut.store.move('rewinding',0,self.root,'rewind-during-gc'),head)
                self.assertEqual(self.sut.snapshot_bytes(candidate),saved)
            self.check(candidate,self.P+self.A)
            self.assertEqual(self.sut.store.collect_history(self.root)['deleted'],[])
            self.check(candidate,self.P+self.A)
            self.check(self.root,self.P)
        finally:
            for worker in workers:
                if worker.is_alive():worker.terminate();worker.join(10)
            results.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
