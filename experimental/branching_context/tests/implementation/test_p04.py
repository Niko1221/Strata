"""Explicit history deletion preserves live reachability and default retention."""
import multiprocessing
import tempfile
import unittest

import test_e01
from test_p03 import hold_read
from context_time_travel.store import Store, NotFound


class CollectionTests(unittest.TestCase):
    def test_p04_invariant(self):
        self.assertTrue(callable(getattr(Store,'collect_history',None)),
                        'History collection must synchronize with references and active leases')
        with tempfile.TemporaryDirectory() as directory:
            store=Store(directory)
            root=store.create_root('contract',[1],{})
            parent=store.append(root,[2],{})
            live=store.append(parent,[3],{})
            orphan=store.append(root,[4],{})
            active=store.append(root,[5],{})
            pinned=store.append(root,[6],{})
            evaluated=store.append(root,[7],{})
            untouched=store.append(root,[8],{})
            store.fork('main',live)
            store.retain('historical',pinned)
            store.put_evaluation(test_e01.record(evaluated,root))
            before={s:store.get_snapshot(s) for s in (root,parent,live,pinned,evaluated,untouched)}
            self.assertEqual(store.collect_history(root)['deleted'],[])
            guest=store.for_principal('guest')
            with self.assertRaises(PermissionError):
                guest.release_history(orphan)
            with self.assertRaises(PermissionError):
                guest.collect_history(root)
            for snapshot in (parent,live,orphan,active,pinned,evaluated):
                store.release_history(snapshot)
            context=multiprocessing.get_context('spawn')
            ready,release=context.Event(),context.Event()
            worker=context.Process(target=hold_read,args=(directory,active,ready,release))
            worker.start()
            try:
                self.assertTrue(ready.wait(15))
                result=Store(directory).collect_history(root)
                self.assertEqual(result['deleted'],[orphan])
                self.assertEqual(store.lineage(live)[1],[1,2,3])
                store.get_snapshot(active)
                release.set()
                worker.join(15)
                self.assertEqual(worker.exitcode,0)
            finally:
                if worker.is_alive():
                    worker.terminate();worker.join(15)
            self.assertEqual(Store(directory).collect_history(root)['deleted'],[active])
            with self.assertRaises(NotFound):
                store.get_snapshot(orphan)
            self.assertEqual({s:store.get_snapshot(s) for s in before},before)
            self.assertEqual(store.collect_history(root)['deleted'],[])
            store.release_retained('historical')
            self.assertEqual(store.collect_history(root)['deleted'],[pinned])
            self.assertEqual(store.head('main').tip,live)


if __name__ == '__main__':
    unittest.main(verbosity=2)
