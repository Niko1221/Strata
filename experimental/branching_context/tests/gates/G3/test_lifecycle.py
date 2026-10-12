"""T33: actual model cycles must release images, pages and handle metadata."""
import gc
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import weakref

from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError
import test_immutability as isolation
import test_offload as offload


class LifecycleTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged
    assert_ssd_store = offload.OffloadTests.assert_ssd_store

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sut = BranchingModel(self.tmp.name, self.backend, self.profile)
        self.addCleanup(self.release_if_open)

    def release_if_open(self):
        if not self.backend._closed:
            self.sut.evict_all()

    def require_instrumentation(self, counters):
        required = {'handle_allocations', 'handle_releases', 'handle_live_entries', 'handle_live_bytes',
                    'handle_issued', 'snapshot_allocations', 'snapshot_releases', 'snapshot_live_images',
                    'snapshot_payload_capacity_bytes', 'host_heap_allocated_bytes',
                    'integrity_workspace_allocations', 'integrity_workspace_releases',
                    'integrity_workspace_live_bytes', 'live_bytes', 'allocations', 'releases',
                    'mappings', 'unmappings', 'address_reservations', 'address_releases'}
        self.assertFalse(required - counters.keys(), 'Direct native allocation instrumentation is required')
        for name in required:
            self.assertIs(type(counters[name]), int, name + ' must be an executed counter, not null')
            self.assertGreaterEqual(counters[name], 0, name)

    def balanced(self, before, after):
        for name in ('live_bytes', 'handle_live_entries', 'handle_live_bytes', 'snapshot_live_images',
                     'snapshot_payload_capacity_bytes', 'integrity_workspace_live_bytes'):
            self.assertEqual(after[name], before[name], 'Resource did not return to baseline: ' + name)
        for allocated, released in [('allocations', 'releases'), ('handle_allocations', 'handle_releases'),
                                    ('snapshot_allocations', 'snapshot_releases'), ('mappings', 'unmappings'),
                                    ('address_reservations', 'address_releases'),
                                    ('integrity_workspace_allocations', 'integrity_workspace_releases')]:
            self.assertEqual(after[allocated] - before[allocated], after[released] - before[released],
                             'Unbalanced actual resource operations: ' + allocated)

    def empty(self, initial, phase):
        self.sut.evict_all()
        gc.collect()
        self.backend.release_checkpoints()
        self.backend._request('reset')
        after = self.backend._refresh_stats()
        self.record(kind='lifecycle-empty-baseline', phase=phase, initial=initial, after=after)
        self.balanced(initial, after)
        self.assertEqual(after['consumed'], 0)
        self.assertEqual(list(self.sut.cache), [])
        return after

    def test_g3_012_allocation_leak_cycles(self):
        self.assert_ssd_store()
        initial = self.backend._refresh_stats()
        self.require_instrumentation(initial)
        for key in ('handle_live_entries', 'handle_live_bytes', 'snapshot_live_images',
                    'snapshot_payload_capacity_bytes'):
            self.assertEqual(initial[key], 0)
        history = tuple(self.profile['fixtures']['prefix'][:9])
        tokens = [self.profile['fixtures'][name][0] for name in ('suffix_a', 'suffix_b')]
        self.assertNotEqual(*tokens)
        root = self.sut.create(history)
        saved = self.retain(root, self.fresh(root, history))
        baseline = self.backend._refresh_stats()
        self.assertEqual(baseline['snapshot_live_images'], 1)
        self.assertEqual(baseline['handle_live_entries'], 1)
        self.assertGreater(baseline['snapshot_payload_capacity_bytes'], 0)
        warm_heap, heads = [], []
        last_handle = saved['image'].handle
        last_snapshot = None
        # First eight complete cycles establish the allocator's warm high-water
        # baseline. The following 56 cycles use that frozen observed bound.
        # Direct live ownership/call-balance assertions apply to all 64 cycles.
        for cycle in range(64):
            for branch, token in enumerate(tokens):
                self.assertEqual(set(self.sut.cache), {root})
                name = f'cycle-{cycle:03d}-{branch}'
                start = self.sut.start_generation(root, dict(allowed=[token]), name)
                head = self.sut.store.fork(name, start)
                self.fresh(start, history)
                before = self.backend._refresh_stats()
                result = self.sut.generate(start, 1)
                after = self.backend._refresh_stats()
                self.assertEqual(result.accepted_tokens, (token,))
                self.assertEqual(result.boundaries, (result.snapshot_id,))
                self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
                self.assertEqual(self.sut.controller(result.snapshot_id)['next_draw'], 1)
                self.fresh(result.snapshot_id, history + (token,))
                self.unchanged([saved])
                native_handle = self.sut.cache[result.snapshot_id].blob.handle
                self.assertGreater(native_handle, last_handle, 'Native handle IDs must never recycle')
                last_handle = native_handle
                weak_image = weakref.ref(self.sut.cache[result.snapshot_id].blob)
                published = self.sut.store.move(name, head.revision, result.snapshot_id, name + '-publish')
                rewound = self.sut.store.move(name, published.revision, start, name + '-rewind')
                self.assertEqual((rewound.revision, rewound.tip), (2, start))
                self.assertEqual(self.sut.controller(start)['next_draw'], 0)
                heads.append(rewound)
                # Activate the retained root before releasing the child's owners;
                # otherwise the private lease itself legitimately retains pages.
                with self.sut.materialize(root):
                    pass
                del self.sut.cache[result.snapshot_id]
                del self.sut.cache[start]
                gc.collect()
                self.backend.release_checkpoints()
                released = self.backend._refresh_stats()
                self.record(kind='lifecycle-cycle', cycle=cycle, branch=branch, before=before,
                            after=after, released=released, baseline=baseline,
                            released_handle=native_handle, snapshot=result.snapshot_id)
                self.assertIsNone(weak_image(), 'A released child must have no remaining Python image owner')
                self.balanced(baseline, released)
                self.assertEqual(released['consumed'], len(history))
                # A retry cannot free another image; a stale restore cannot see
                # a newer image that happens to occupy the old storage slot.
                self.backend._request('drop ' + str(native_handle))
                with self.assertRaises(IntegrityError):
                    self.backend._request('restore ' + str(native_handle))
                refused = self.backend._refresh_stats()
                self.balanced(released, refused)
                self.assertEqual(refused['forward_tokens'], released['forward_tokens'])
                self.assertEqual(refused['consumed'], released['consumed'])
                if cycle < 8:
                    warm_heap.append(released['host_heap_allocated_bytes'])
                    if cycle == 7 and branch == 1:
                        self.record(kind='lifecycle-frozen-host-heap-bound', samples=warm_heap,
                                    maximum_allocated_bytes=max(warm_heap),
                                    source='mallinfo2 uordblks + hblkhd; excludes file mappings and allocator-free arenas')
                else:
                    self.assertLessEqual(released['host_heap_allocated_bytes'], max(warm_heap),
                                         'Native heap allocation exceeded its fixed warm-cycle baseline')
                last_snapshot = result.snapshot_id
        self.unchanged([saved])
        for head in heads:
            self.assertEqual(self.sut.store.head(head.name), head)
        root_weak = weakref.ref(saved['image'])
        saved = None
        self.empty(initial, 'after-all-64-cycles')
        self.assertIsNone(root_weak())

        # After every optional checkpoint is gone, an old logical child remains
        # usable. Replay must consume fixed history and preserve its draw cursor.
        prior_cursor = self.sut.controller(last_snapshot)['next_draw']
        before = self.backend._refresh_stats()
        last_history = history + (tokens[-1],)
        self.fresh(last_snapshot, last_history)
        after = self.backend._refresh_stats()
        self.assertEqual(after['forward_tokens'] - before['forward_tokens'], len(last_history))
        self.assertEqual(self.sut.controller(last_snapshot)['next_draw'], prior_cursor)
        self.assertGreater(self.sut.cache[last_snapshot].blob.handle, last_handle)
        with self.assertRaises(IntegrityError):
            self.backend._request('restore ' + str(last_handle))
        result = self.sut.generate(last_snapshot, 1)
        self.assertEqual(result.accepted_tokens, (tokens[-1],))
        self.assertEqual(self.sut.controller(result.snapshot_id)['next_draw'], prior_cursor + 1)
        self.fresh(result.snapshot_id, last_history + result.accepted_tokens)
        self.record(kind='lifecycle-replay', before=before, after=after,
                    after_continuation=self.backend._refresh_stats(), preserved_draw_cursor=prior_cursor)
        self.empty(initial, 'after-cold-replay-and-continuation')

        pid = self.backend.process.pid
        self.backend.close()
        self.assertEqual(self.backend.process.returncode, 0, 'Native teardown must exit cleanly')
        self.assertFalse(Path('/proc', str(pid)).exists())
        deadline = time.monotonic() + 5
        while True:
            gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                               '--format=csv,noheader,nounits'], text=True).splitlines()
            if str(pid) not in [value.strip() for value in gpu_pids] or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        self.assertNotIn(str(pid), [value.strip() for value in gpu_pids])
        self.record(kind='lifecycle-process-teardown', pid=pid, returncode=self.backend.process.returncode,
                    proc_entry_exists=False, gpu_process_ids=gpu_pids)
