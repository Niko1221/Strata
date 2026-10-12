"""Process crash and competing-writer schedules for history collection."""
import multiprocessing
from contextlib import contextmanager
import os
import tempfile
import unittest

from context_time_travel.store import Store, NotFound
from context_time_travel.evaluation import evaluate


def evaluate_while_pinned(directory, root, candidate, ready, release, results):
    class ObservedStore(Store):
        @contextmanager
        def read_lease(self, identity):
            with super().read_lease(identity) as lease:
                ready.set()
                if not release.wait(20):
                    raise RuntimeError('Collection did not release evaluator')
                yield lease
    source = "result = {'mandatory': {'valid': 'PASS'}, 'score': len(candidate['tokens']), 'unresolved': []}"
    spec = dict(task_root=root, evaluator={'name': 'length', 'version': '1', 'source': source},
                fixtures={}, metric={'name': 'length', 'order': 'maximize'},
                seed=17, mandatory=['valid'], timeout_seconds=5)
    store = ObservedStore(directory)
    report = evaluate(store, candidate, spec)
    results.put(store.put_evaluation(report))


def rewind_during_collection(directory, root, barrier, results):
    store = Store(directory)
    barrier.wait(timeout=20)
    head = store.move('main', 0, root, 'rewind')
    results.put(('rewind', 'ok', head.tip))


def crash_collect(directory,root):
    def fault(point):
        if point=='after_history_unlink':
            os._exit(42)
    Store(directory,fault_hook=fault).collect_history(root)
    os._exit(43)


def compete(directory,root,parent,operation,barrier,results):
    store=Store(directory)
    barrier.wait(timeout=20)
    try:
        if operation=='collect':
            value=store.collect_history(root)['deleted']
        elif operation=='fork':
            value=store.fork('racing',parent).tip
        else:
            value=store.append(parent,[7],{})
        results.put((operation,'ok',value))
    except NotFound:
        results.put((operation,'missing',None))


class CollectionProcessTests(unittest.TestCase):
    def test_collection_during_evaluation_and_after_publication(self):
        ctx = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            root = store.create_root('contract', [1], {})
            candidate = store.append(root, [2], {})
            store.release_history(candidate)
            before = (store.objects / (candidate + '.json')).read_bytes()
            ready, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
            worker = ctx.Process(target=evaluate_while_pinned,
                                 args=(directory, root, candidate, ready, release, results))
            worker.start()
            try:
                self.assertTrue(ready.wait(15), 'Public evaluate must acquire a history read lease')
                self.assertEqual(store.collect_history(root)['deleted'], [])
                self.assertEqual((store.objects / (candidate + '.json')).read_bytes(), before)
                release.set()
                identity = results.get(timeout=20)
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)
                report = Store(directory).get_evaluation(identity)
                self.assertEqual(report['outcomes']['mandatory'], {'valid': 'PASS'})
                self.assertEqual(report['outcomes']['score'], 2)
                self.assertEqual(store.collect_history(root)['deleted'], [])
                self.assertEqual(store.lineage(candidate)[1], [1, 2])
            finally:
                if worker.is_alive():
                    worker.terminate(); worker.join(10)
                results.close()

    def test_collection_races_rewind(self):
        ctx = multiprocessing.get_context('spawn')
        for repeat in range(4):
            with self.subTest(repeat=repeat), tempfile.TemporaryDirectory() as directory:
                store = Store(directory)
                root = store.create_root('contract', [1], {})
                candidate = store.append(root, [2], {})
                store.fork('main', candidate)
                store.release_history(candidate)
                barrier, results = ctx.Barrier(2), ctx.Queue()
                workers = [ctx.Process(target=compete, args=(directory, root, candidate, 'collect', barrier, results)),
                           ctx.Process(target=rewind_during_collection, args=(directory, root, barrier, results))]
                try:
                    for worker in workers: worker.start()
                    observed = {row[0]: row[1:] for row in [results.get(timeout=25), results.get(timeout=25)]}
                    for worker in workers:
                        worker.join(10)
                        self.assertEqual(worker.exitcode, 0)
                    self.assertEqual(observed['rewind'], ('ok', root))
                    self.assertEqual(observed['collect'][0], 'ok')
                    self.assertIn(observed['collect'][1], ([], [candidate]))
                    reopened = Store(directory)
                    head = reopened.head('main')
                    self.assertEqual((head.tip, head.revision), (root, 1))
                    self.assertEqual(reopened.move('main', 0, root, 'rewind'), head)
                    for path in reopened.objects.glob('*.json'): reopened.lineage(path.stem)
                finally:
                    for worker in workers:
                        if worker.is_alive(): worker.terminate(); worker.join(10)
                    results.close()

    def test_crash_sweep_preserves_remaining_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(directory)
            root=store.create_root('contract',[1],{})
            store.fork('main',root)
            receipt=store.move('main',0,root,'stable-receipt')
            parent=store.append(root,[2],{})
            child=store.append(parent,[3],{})
            store.release_history(parent)
            store.release_history(child)
            ctx=multiprocessing.get_context('spawn')
            worker=ctx.Process(target=crash_collect,args=(directory,root))
            worker.start();worker.join(20)
            try:
                self.assertEqual(worker.exitcode,42)
            finally:
                if worker.is_alive():worker.terminate();worker.join(10)
            reopened=Store(directory)
            with self.assertRaises(NotFound):
                reopened.get_snapshot(child)
            self.assertEqual(reopened.lineage(parent)[1],[1,2])
            for path in reopened.objects.glob('*.json'):
                reopened.lineage(path.stem)
            self.assertEqual(reopened.move('main',0,root,'stable-receipt'),receipt)
            self.assertEqual(reopened.collect_history(root)['deleted'],[parent])
            self.assertEqual(reopened.head('main'),receipt)
            self.assertEqual(reopened.collect_history(root)['deleted'],[])

    def test_collection_races_fork_and_append(self):
        ctx=multiprocessing.get_context('spawn')
        for operation in ('fork','append'):
            for repeat in range(4):
                with self.subTest(operation=operation,repeat=repeat), tempfile.TemporaryDirectory() as directory:
                    store=Store(directory)
                    root=store.create_root('contract',[1],{})
                    parent=store.append(root,[2],{})
                    store.release_history(parent)
                    barrier,results=ctx.Barrier(2),ctx.Queue()
                    workers=[ctx.Process(target=compete,args=(directory,root,parent,op,barrier,results))
                             for op in ('collect',operation)]
                    try:
                        for worker in workers:worker.start()
                        observed={row[0]:row[1:] for row in [results.get(timeout=25),results.get(timeout=25)]}
                        for worker in workers:
                            worker.join(10)
                            self.assertEqual(worker.exitcode,0)
                        self.assertEqual(observed['collect'][0],'ok')
                        state,value=observed[operation]
                        if state=='ok':
                            self.assertEqual(observed['collect'][1],[])
                            self.assertEqual(store.lineage(value)[1],[1,2] if operation=='fork' else [1,2,7])
                        else:
                            self.assertEqual(state,'missing')
                            self.assertEqual(observed['collect'][1],[parent])
                        for path in store.objects.glob('*.json'):
                            store.lineage(path.stem)
                    finally:
                        for worker in workers:
                            if worker.is_alive():worker.terminate();worker.join(10)
                        results.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
