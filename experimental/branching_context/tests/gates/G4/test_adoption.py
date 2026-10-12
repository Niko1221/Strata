"""Real candidate adoption, including an actual process CAS race."""
import json
import multiprocessing
import unittest

import test_evaluation_records as records
from context_time_travel.evaluation import evaluate, choose_validated
from context_time_travel.store import Store, Conflict, InvalidOperation


def race_move(directory, revision, target, operation, barrier, results):
    store = Store(directory)
    barrier.wait(timeout=30)
    try:
        head = store.move('main', revision, target, operation)
        results.put(('win', head.tip, head.revision))
    except Conflict:
        results.put(('conflict', target, revision))


class AdoptionTests(unittest.TestCase):
    setUpClass = classmethod(records.EvaluationRecordTests.setUpClass.__func__)
    tearDownClass = classmethod(records.EvaluationRecordTests.tearDownClass.__func__)
    setUp = records.EvaluationRecordTests.setUp
    check = records.EvaluationRecordTests.check

    def candidates(self):
        histories = [self.P+self.A, self.P+self.B, self.P+self.A+self.B]
        candidates = [self.sut.feed(self.root, h[len(self.P):]) for h in histories]
        source = """
import json
rows = json.loads(fixtures['scores'])
row = next(row for row in rows if tuple(row['tokens']) == candidate['tokens'])
result = {'mandatory': {'valid': row['status']}, 'score': row['score'], 'unresolved': []}
"""
        fixtures = {'scores': json.dumps([dict(tokens=h,score=s,status=v)
                    for h,s,v in zip(histories,(7,10,8),('PASS','FAIL','PASS'))]).encode()}
        identities = []
        for candidate in candidates:
            report = evaluate(self.sut.store,candidate,dict(task_root=self.root,
                evaluator={'name':'fixed-fixture','version':'1','source':source},
                fixtures=fixtures,metric={'name':'verified-score','order':'maximize'},
                seed=17,mandatory=['valid'],timeout_seconds=5))
            identities.append(self.sut.store.put_evaluation(report))
        return candidates, histories, identities

    def test_g4_006_winner_adoption_cas(self):
        candidates, histories, identities = self.candidates()
        store = self.sut.store
        head = store.fork('main',self.root)
        before = {s:self.sut.snapshot_bytes(s) for s in candidates}
        winner = choose_validated(store,identities,mandatory=('valid',))
        self.assertEqual(winner['snapshot_id'],candidates[2])
        moved = store.move('main',head.revision,winner['snapshot_id'],'adopt-winner')
        self.assertEqual(Store(self.tmp.name).move('main',head.revision,winner['snapshot_id'],'adopt-winner'),moved)
        self.assertEqual(moved.revision,head.revision+1)
        with self.assertRaises(InvalidOperation):
            store.move('main',head.revision,candidates[0],'adopt-winner')
        self.assertEqual({s:self.sut.snapshot_bytes(s) for s in candidates},before)
        for candidate,history in zip(candidates,histories):
            self.check(candidate,history)

    def test_g4_007_adoption_race_loses_cleanly(self):
        candidates, histories, identities = self.candidates()
        winner = choose_validated(self.sut.store,identities,mandatory=('valid',))
        head = self.sut.store.fork('main',self.root)
        before = {s:self.sut.snapshot_bytes(s) for s in candidates}
        context = multiprocessing.get_context('spawn')
        barrier, results = context.Barrier(2), context.Queue()
        workers = [context.Process(target=race_move,args=(self.tmp.name,head.revision,target,
                   'race-'+str(i),barrier,results))
                   for i,target in enumerate((winner['snapshot_id'],candidates[0]))]
        try:
            for worker in workers:
                worker.start()
            observed = [results.get(timeout=45) for _ in workers]
            for worker in workers:
                worker.join(timeout=10)
                self.assertEqual(worker.exitcode,0)
            self.assertEqual(sorted(row[0] for row in observed),['conflict','win'])
            winning = next(row for row in observed if row[0]=='win')
            self.assertEqual(self.sut.store.head('main').tip,winning[1])
            self.assertEqual(self.sut.store.head('main').revision,head.revision+1)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=10)
            results.close()
        self.assertEqual({s:self.sut.snapshot_bytes(s) for s in candidates},before)
        for candidate,history in zip(candidates,histories):
            self.check(candidate,history)


if __name__ == '__main__':
    unittest.main(verbosity=2)
