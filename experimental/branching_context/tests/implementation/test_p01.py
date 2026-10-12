"""Local principal grants must precede snapshot and reference access."""
import tempfile
import unittest

import test_e01
from context_time_travel.store import Store


class AuthorizationTests(unittest.TestCase):
    def test_p01_invariant(self):
        self.assertTrue(callable(getattr(Store,'for_principal',None)),
                        'Store must bind operations to an explicit local principal')
        self.assertTrue(callable(getattr(Store,'grant',None)),
                        'Root-scoped grants are required; hashes are not capabilities')
        with tempfile.TemporaryDirectory() as directory:
            owner = Store(directory)
            root = owner.create_root('private-model',[1,2],{'next_draw':17})
            secret = owner.append(root,[3],{'next_draw':19})
            other = owner.create_root('private-model',[9],{'next_draw':1})
            head = owner.fork('private',secret)
            report = test_e01.record(secret,root)
            report_id = owner.put_evaluation(report)
            guest = owner.for_principal('guest')
            before = {p.name:p.read_bytes() for p in owner.objects.glob('*.json')}
            denied = [lambda:guest.get_snapshot(secret),lambda:guest.lineage(secret),
                      lambda:guest.fork('leaked',secret),lambda:guest.head('private'),
                      lambda:guest.move('private',head.revision,secret,'denied-move'),
                      lambda:guest.put_evaluation(report),lambda:guest.get_evaluation(report_id),
                      lambda:guest.append(secret,[4],{}),
                      lambda:guest.grant(root,'guest',('read','fork','move','evaluate','append'))]
            for operation in denied:
                with self.assertRaises(PermissionError):
                    operation()
            self.assertEqual(guest.work_counters()['snapshot_read_bytes'],0)
            owner.grant(root,'guest',('read',))
            self.assertEqual(guest.get_snapshot(secret),owner.get_snapshot(secret))
            self.assertEqual(guest.lineage(secret),owner.lineage(secret))
            with self.assertRaises(PermissionError):
                guest.get_snapshot(other)
            with self.assertRaises(PermissionError):
                guest.fork('leaked',secret)
            owner.grant(root,'guest',('read','fork','evaluate'))
            fork = guest.fork('allowed',secret)
            self.assertEqual(fork.tip,secret)
            self.assertEqual(guest.put_evaluation(report),report_id)
            self.assertEqual(guest.get_evaluation(report_id),report)
            with self.assertRaises(PermissionError):
                guest.move('allowed',fork.revision,root,'ungranted-move')
            reopened = Store(directory).for_principal('guest')
            self.assertEqual(reopened.get_snapshot(secret),owner.get_snapshot(secret))
            owner.grant(root,'guest',())
            with self.assertRaises(PermissionError):
                reopened.get_snapshot(secret)
            self.assertEqual(owner.head('private'),head)
            self.assertEqual({p.name:p.read_bytes() for p in owner.objects.glob('*.json')},before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
