"""Retention tracks canonical IDs and live OS-backed read leases."""
import multiprocessing
import tempfile
import unittest

import test_e01
from context_time_travel.store import Store


def hold_read(directory, snapshot, ready, release):
    with Store(directory).read_lease(snapshot):
        ready.set()
        if not release.wait(30):
            raise RuntimeError('Parent failed to release lease fixture')


class RetentionTests(unittest.TestCase):
    def test_p03_invariant(self):
        self.assertTrue(callable(getattr(Store,'read_lease',None)),
                        'Active materialization/evaluation must pin canonical reachability')
        self.assertTrue(callable(getattr(Store,'retain',None)))
        with tempfile.TemporaryDirectory() as directory:
            store=Store(directory)
            root=store.create_root('contract',[1],{'next_draw':3})
            head=store.append(root,[2],{'next_draw':5})
            old=store.append(root,[3],{'next_draw':7})
            evaluated=store.append(root,[4],{'next_draw':9})
            active=store.append(root,[5],{'next_draw':11})
            store.fork('main',head)
            store.retain('historical',old)
            store.put_evaluation(test_e01.record(evaluated,root))
            before={p.name:p.read_bytes() for p in store.objects.glob('*.json')}
            retained=Store(directory).retained_snapshots()
            self.assertTrue({root,head,old,evaluated}.issubset(retained))
            self.assertNotIn(active,retained)
            context=multiprocessing.get_context('spawn')
            ready,release=context.Event(),context.Event()
            worker=context.Process(target=hold_read,args=(directory,active,ready,release))
            worker.start()
            try:
                self.assertTrue(ready.wait(15))
                self.assertIn(active,Store(directory).retained_snapshots())
                # Kernel lock lifetime, not a PID file, determines active use.
                worker.terminate()
                worker.join(15)
                self.assertFalse(worker.is_alive())
                self.assertNotIn(active,Store(directory).retained_snapshots())
            finally:
                if worker.is_alive():
                    worker.terminate();worker.join(15)
            guest=store.for_principal('guest')
            with self.assertRaises(PermissionError):
                with guest.read_lease(old):
                    self.fail('Unauthorized lease admitted')
            with self.assertRaises(PermissionError):
                guest.release_retained('historical')
            store.release_retained('historical')
            self.assertNotIn(old,Store(directory).retained_snapshots())
            self.assertEqual({p.name:p.read_bytes() for p in store.objects.glob('*.json')},before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
