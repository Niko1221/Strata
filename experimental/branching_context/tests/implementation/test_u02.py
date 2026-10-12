"""Termination receipts and refusal of unreadable reference formats."""
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.store import Store, IntegrityError, InvalidOperation, Conflict, _json_bytes


def terminate_move(directory, target, point):
    def fault(actual):
        if actual == point: os._exit(73)
    Store(directory, fault_hook=fault).move('main', 0, target, 'attempt')
    os._exit(74)


class RollbackTests(unittest.TestCase):
    def test_u02_invariant(self):
        ctx = multiprocessing.get_context('spawn')
        for point in ('after_refs_stage_fsync', 'after_refs_rename_before_dir_fsync'):
            with self.subTest(point=point), tempfile.TemporaryDirectory() as directory:
                store = Store(directory)
                root = store.create_root('contract', [1], {'draw_cursor': 7})
                tip = store.append(root, [2], {'draw_cursor': 9})
                before = store.fork('main', root)
                snapshots = {p.name:p.read_bytes() for p in store.objects.glob('*.json')}
                worker = ctx.Process(target=terminate_move, args=(directory, tip, point))
                worker.start(); worker.join(15)
                try: self.assertEqual(worker.exitcode, 73)
                finally:
                    if worker.is_alive(): worker.terminate(); worker.join(10)
                reopened = Store(directory)
                head = reopened.head('main')
                self.assertEqual(head.revision, 0 if point == 'after_refs_stage_fsync' else 1)
                if head.revision == 0: self.assertEqual(head, before)
                result = reopened.move('main', 0, tip, 'attempt')
                self.assertEqual((result.tip, result.revision), (tip, 1))
                self.assertEqual(reopened.move('main', 0, tip, 'attempt'), result)
                with self.assertRaises(InvalidOperation): reopened.move('main', 0, root, 'attempt')
                rewind = reopened.move('main', 1, root, 'back')
                self.assertEqual(rewind.revision, 2)
                with self.assertRaises(Conflict): reopened.move('main', 0, tip, 'stale')
                self.assertEqual({p.name:p.read_bytes() for p in store.objects.glob('*.json')}, snapshots)

        for location, version in [('marker', 3), ('marker', 2.0), ('branch', 2),
                                  ('branch', True), ('order', True), ('legacy', True)]:
            with self.subTest(location=location, version=version), tempfile.TemporaryDirectory() as directory:
                store = Store(directory)
                root = store.create_root('contract', [1], {})
                store.fork('main', root)
                path = {'marker':store.refs_path, 'legacy':store.refs_path,
                        'branch':store.references/'main.json', 'order':store.order_path}[location]
                record = store._empty_refs() if location == 'legacy' else json.loads(path.read_bytes())
                record['schema'] = version
                path.write_bytes(_json_bytes(record))
                before = {str(p.relative_to(directory)):p.read_bytes() for p in Path(directory).rglob('*') if p.is_file()}
                with self.assertRaises(IntegrityError):
                    reopened = Store(directory)
                    if location == 'branch': reopened.head('main')
                    if location == 'order': reopened.move('main', 0, root, 'move')
                after = {str(p.relative_to(directory)):p.read_bytes() for p in Path(directory).rglob('*') if p.is_file()}
                self.assertEqual(after, before, 'Refusing rollback must not rewrite authoritative records')


if __name__ == '__main__':
    unittest.main(verbosity=2)
