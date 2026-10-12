"""Actual G3 probes. The owned-copy fallback is expected to fail bounded copying."""
from pathlib import Path
import json
import math
import os
import time
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'G1'))
import test_continuation as continuation
import test_immutability as isolation
import test_partial_cow as partial

class SharingTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged
    pages = staticmethod(partial.PartialTailTests.pages)
    tail_pages = partial.PartialTailTests.tail_pages

    def probe(self, length):
        history = (self.P*200)[:length]
        root = self.sut.create(history)
        before = dict(self.backend.stats)
        start = time.perf_counter()
        child = self.sut.feed(root,self.A[:1])
        elapsed = time.perf_counter()-start
        result = dict(prefix_length=length,accepted_tokens=1,seconds=elapsed,
            prefix_tokens_prefilled=self.backend.stats['consumed_tokens']-before['consumed_tokens']-1,
            imported_bytes=self.backend.stats['checkpoint_import_bytes']-before['checkpoint_import_bytes'],
            exported_bytes=self.backend.stats['checkpoint_export_bytes']-before['checkpoint_export_bytes'],
            parent_state_bytes=len(self.sut.cache[root].blob),
            shared_prefix_pages=None,private_tail_pages=None,kv_unique_bytes=None)
        with open(os.environ.get('CTT_SHARING_MEASUREMENTS','evidence/runs/sharing-measurements.jsonl'),'a') as output:
            output.write(json.dumps(result)+'\n')
        self.check(root,history)
        self.check(child,history+self.A[:1])
        return result

    def test_g3_001_actual_prefix_work_shared(self):
        first = self.probe(256)
        self.assertEqual(first['prefix_tokens_prefilled'],0)

    def copy_probe(self, length, *, require_full_prefix):
        self.sut.evict_all()
        history = tuple(self.P[i % len(self.P)] for i in range(length))
        root = self.sut.create(history)
        saved = self.retain(root, self.fresh(root, history))
        original = self.pages(saved['audit'])
        kv = [row for row in original.values() if row['kind'] in ('k', 'v')]
        self.assertTrue(kv)
        if require_full_prefix:
            self.assertEqual({r['pool'] for r in kv},
                             {r['pool'] for r in kv if r['filled_prefix_bytes'] == r['bytes']},
                             'The long fixture needs a completely consumed physical page in every K/V pool')
        expected = self.tail_pages(saved['audit'], length)
        bound = sum(original[key]['bytes'] for key in expected)
        retained, measurements = [saved], []
        self.assertNotEqual(self.A[0], self.B[0])
        for token in (self.A[0], self.B[0]):
            before = self.backend._refresh_stats()
            child = self.sut.feed(root, (token,))
            after = self.backend._refresh_stats()
            audit = self.backend.audit(self.sut.cache[child].blob)
            current = self.pages(audit)
            self.assertEqual(current.keys(), original.keys())
            changed = {key for key in original
                       if original[key]['allocation_id'] != current[key]['allocation_id']}
            work = dict(prefix_length=length, token=token, before=before, after=after,
                        expected_private_pages=sorted(expected), observed_private_pages=sorted(changed),
                        copied_bytes=after['copied_bytes'] - before['copied_bytes'],
                        physical_tail_bound_bytes=bound,
                        restored_running_bytes=after['restored_running_bytes'] - before['restored_running_bytes'],
                        saved_running_bytes=after['saved_running_bytes'] - before['saved_running_bytes'],
                        child_audit=audit)
            self.record(kind='bounded-tail-work', **work)
            self.assertEqual(changed, expected, 'Appending must not copy completed prefix pages')
            self.assertEqual(work['copied_bytes'], bound)
            self.assertEqual(after['cow_pages'] - before['cow_pages'], len(expected))
            self.assertEqual(after['allocations'] - before['allocations'], len(expected))
            self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
            self.assertEqual(work['restored_running_bytes'], saved['image'].running_bytes)
            self.assertEqual(work['saved_running_bytes'], saved['image'].running_bytes)
            for counter in ('binding_copy_bytes', 'written_file_bytes', 'loaded_file_bytes'):
                self.assertEqual(after[counter], before[counter], counter)
            for key in original.keys() - expected:
                for field in ('allocation_id', 'driver_handle', 'sha256', 'filled_prefix_bytes'):
                    self.assertEqual(current[key][field], original[key][field], (key, field))
            if require_full_prefix:
                self.assertTrue(any(original[key]['kind'] in ('k', 'v') and
                                    original[key]['filled_prefix_bytes'] == original[key]['bytes']
                                    for key in original.keys() - expected))
            self.unchanged(retained)
            retained.append(self.retain(child, self.fresh(child, history + (token,))))
            measurements.append(work)
        self.unchanged(retained)
        # Compare the same offset in a later physical K/V page. This fixes the
        # legitimate writable footprint while increasing retained prefix data.
        periods = [saved['audit']['qsa_page_cells'], saved['audit']['index_block_cells']]
        for row in kv:
            self.assertGreater(row['bytes_per_cell'], 0)
            self.assertEqual(row['bytes'] % row['bytes_per_cell'], 0)
            periods.append(row['bytes'] // row['bytes_per_cell'])
        return dict(next_length=length + math.lcm(*periods), measurements=measurements)

    def test_g3_010_tail_copy_bounded(self):
        small = self.probe(256)
        large = self.probe(1024)
        # Even allowing one complete 256-cell serialized block, activation must
        # not copy the whole 1024-cell prefix to qualify as bounded tail work.
        self.assertLessEqual(large['imported_bytes'],small['parent_state_bytes'])
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'Bounded copying requires direct physical-page and copy instrumentation')
        short = self.copy_probe(256, require_full_prefix=False)
        self.assertLess(short['next_length'], self.profile['execution']['capacity'],
                        'The profile must support a suffix beyond a complete physical K/V page')
        long = self.copy_probe(short['next_length'], require_full_prefix=True)
        for first, later in zip(short['measurements'], long['measurements']):
            self.assertGreater(later['prefix_length'], first['prefix_length'])
            for counter in ('copied_bytes', 'restored_running_bytes', 'saved_running_bytes'):
                self.assertEqual(later[counter], first[counter],
                                 'Increasing the prefix cannot increase this fixed tail footprint: ' + counter)
