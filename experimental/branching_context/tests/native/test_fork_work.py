"""Real POSIX reference I/O: fork work must not traverse/copy retained history."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from context_time_travel.store import IntegrityError, NotFound, Store, _json_bytes


def delta(before, after):
    return {key: after[key] - before[key] for key in before}


def observe(store, name, snapshot):
    before = store.work_counters()
    start = time.perf_counter_ns()
    head = store.fork(name, snapshot)
    elapsed = time.perf_counter_ns() - start
    return head, dict(seconds=elapsed / 1e9, work=delta(before, store.work_counters()))


def assert_bounded(test, row):
    work = row['work']
    test.assertEqual(work['ancestry_nodes_visited'], 0, 'Fork must not walk the canonical history')
    test.assertEqual(work['snapshot_read_bytes'], 0, 'Fork must not read the prefix/controller payload')
    test.assertEqual(work['snapshot_written_bytes'], 0, 'Fork creates no logical snapshot copy')
    test.assertEqual(work['reference_entries_scanned'], 0, 'Fork must not scan unrelated references')
    test.assertLessEqual(work['reference_read_bytes'], 1024)
    test.assertLessEqual(work['reference_written_bytes'], 1024)
    test.assertLessEqual(work['root_index_read_bytes'], 1024)
    test.assertEqual(work['root_index_written_bytes'], 0)
    test.assertEqual(work['root_index_repairs'], 0)
    test.assertEqual(work['reference_publications'], 1)


class ForkWorkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def record(self, **row):
        path = os.environ.get('CTT_LOGICAL_WORK_MEASUREMENTS')
        if path:
            with open(path, 'a') as stream:
                stream.write(json.dumps(dict(test=self.id(), **row)) + '\n')

    def test_fork_work_is_bounded_across_history_and_reference_population(self):
        rows = []
        # Vary ancestry depth, token payload and branch population independently.
        for depth, payload, population in [(0, 1, 0), (64, 1, 0), (256, 1, 0),
                                            (0, 65536, 0), (64, 1, 16), (64, 1, 128)]:
            with self.subTest(depth=depth, payload=payload, population=population):
                directory = Path(self.tmp.name) / f'{depth}-{payload}-{population}'
                store = Store(directory)
                root = store.create_root('work-only', [17] * payload, {'draw': 0})
                tip = root
                for step in range(depth):
                    tip = store.append(tip, [23], {'draw': step + 1})
                for branch in range(population):
                    store.fork('b' + format(branch, '05d'), tip)
                # A fresh instance prevents an in-memory root memo from masking
                # a full history walk on the first fork after reopen.
                start = time.perf_counter_ns()
                reopened = Store(directory)
                opened = dict(seconds=(time.perf_counter_ns() - start) / 1e9,
                              work=reopened.work_counters())
                before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in store.objects.glob('*.json')}
                head, measured = observe(reopened, 'b' + format(population, '05d'), tip)
                row = dict(depth=depth, payload=payload, population=population, open=opened, **measured)
                self.record(**row)
                rows.append(row)
                assert_bounded(self, row)
                self.assertEqual((head.tip, head.root, head.revision), (tip, root, 0))
                self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in store.objects.glob('*.json')})
                self.assertEqual(Store(directory).head(head.name), head)
        self.assertEqual(len(rows), 6)

    def test_missing_root_lookup_rebuild_is_visible(self):
        store = Store(self.tmp.name)
        root = store.create_root('work-only', [1, 2], {'draw': 0})
        tip = root
        for step in range(32):
            tip = store.append(tip, [3], {'draw': step + 1})
        path = store.root_index / (tip + '.json')
        path.unlink()
        before = store.work_counters()
        head = store.fork('repaired', tip)
        measured = delta(before, store.work_counters())
        self.record(kind='explicit-index-reconstruction', work=measured)
        self.assertEqual(head.root, root)
        self.assertEqual(measured['root_index_repairs'], 1)
        self.assertEqual(measured['ancestry_nodes_visited'], 33)
        self.assertGreater(measured['snapshot_read_bytes'], 0)
        self.assertTrue(path.exists())
        _, hot = observe(Store(self.tmp.name), 'healthy', tip)
        assert_bounded(self, hot)
        self.assertEqual(store.lineage(tip), (root, [1, 2] + [3] * 32))

    def test_corrupt_root_lookup_is_not_trusted(self):
        store = Store(self.tmp.name)
        root = store.create_root('work-only', [1], {})
        other = store.create_root('work-only', [9], {})
        tip = store.append(root, [2], {})
        path = store.root_index / (tip + '.json')
        raw = path.read_bytes()
        self.assertIn(root.encode(), raw)
        path.write_bytes(raw.replace(root.encode(), other.encode()))
        with self.assertRaises(IntegrityError):
            store.fork('damaged', tip)
        with self.assertRaises(NotFound):
            store.head('damaged')
        self.assertEqual(store.lineage(tip), (root, [1, 2]))

    def test_missing_canonical_root_rejected_even_with_lookup(self):
        store = Store(self.tmp.name)
        root = store.create_root('work-only', [1], {})
        tip = store.append(root, [2], {})
        (store.objects / (root + '.json')).unlink()
        with self.assertRaises(NotFound):
            store.fork('unavailable', tip)
        with self.assertRaises(NotFound):
            store.head('unavailable')

    def test_legacy_reference_migration_preserves_atomic_receipt_and_history(self):
        store = Store(self.tmp.name)
        root = store.create_root('legacy', [1, 2], {'draw': 0})
        child = store.append(root, [3], {'draw': 1})
        request = dict(type='move', name='legacy', expected_revision=0, target=child)
        result = dict(name='legacy', revision=1, tip=child, root=root)
        legacy = dict(schema=1, branches={'legacy': dict(revision=1, tip=child, root=root)},
                      receipts={'legacy_op': dict(digest=hashlib.sha256(_json_bytes(request)).hexdigest(),
                                                  result=result)},
                      events=[dict(type='fork', branch='legacy', tip=root),
                              dict(type='move', branch='legacy', operation_id='legacy_op',
                                   **{'from': root, 'to': child, 'from_revision': 0, 'to_revision': 1})])
        store.refs_path.write_bytes(_json_bytes(legacy))
        objects = {p.name: p.read_bytes() for p in store.objects.glob('*.json')}
        reopened = Store(self.tmp.name)
        head = reopened.head('legacy')
        self.assertEqual(head, reopened.move('legacy', 0, child, 'legacy_op'))
        self.assertEqual((head.revision, head.tip, head.root), (1, child, root))
        self.assertEqual(reopened.events(), legacy['events'])
        self.assertEqual({p.name: p.read_bytes() for p in store.objects.glob('*.json')}, objects)
        _, measured = observe(Store(self.tmp.name), 'new', child)
        self.record(kind='post-migration-fork', **measured)
        assert_bounded(self, measured)


if __name__ == '__main__':
    unittest.main(verbosity=2)
