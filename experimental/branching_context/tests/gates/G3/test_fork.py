"""T33: bounded reference work and separately measured real lease activation."""
import hashlib
import math
from pathlib import Path
import resource
import statistics
import sys
import time
import unittest

from context_time_travel.store import Store
import test_immutability as isolation
import test_offload as offload

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'native'))
from test_fork_work import assert_bounded, delta, observe


def latency(rows):
    ordered = sorted(row['seconds'] for row in rows)
    return dict(samples=len(ordered), minimum=ordered[0], median=statistics.median(ordered),
                p95=ordered[math.ceil(0.95 * len(ordered)) - 1], maximum=ordered[-1])


class ForkTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged
    assert_ssd_store = offload.OffloadTests.assert_ssd_store

    def files(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.sut.store.objects.glob('*.json')}

    def resources(self):
        def resident(pid):
            status = Path('/proc', str(pid), 'status').read_text().splitlines()
            fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in status}
            return {name: fields[name] for name in ('VmRSS', 'VmHWM')}
        return dict(parent=resident('self'), sut=resident(self.backend.process.pid),
                    parent_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def fork_batch(self, root, snapshot, history, depth, population, label):
        # Reopening prevents a process-local root memo from hiding a first-fork
        # history walk. Native statistics bracket only reference operations.
        self.sut.store = Store(self.tmp.name)
        objects = self.files()
        before = self.backend._refresh_stats()
        resident_before = self.resources()
        rows = []
        for repetition in range(20):
            name = f'f{label:03d}_{repetition:03d}'
            head, row = observe(self.sut.store, name, snapshot)
            self.record(kind='logical-fork', prefix_length=len(history), depth=depth,
                        existing_reference_population=population + repetition, name=name, **row)
            assert_bounded(self, row)
            self.assertEqual((head.tip, head.root, head.revision), (snapshot, root, 0))
            rows.append(row)
        after = self.backend._refresh_stats()
        self.record(kind='logical-fork-batch', prefix_length=len(history), depth=depth,
                    before=before, after=after, resources_before=resident_before,
                    resources_after=self.resources(), latency_seconds=latency(rows),
                    latency_scope='Observed fsync-inclusive wall time on this host; no exclusive-host SLA')
        self.assertEqual(after, before, 'Logical fork must do no model, image or physical-page work')
        self.assertEqual(self.files(), objects)
        head = Store(self.tmp.name).head(name)
        self.assertEqual(head.tip, snapshot)
        self.assertEqual(self.sut.store.lineage(head.tip)[1], list(history))
        return head

    def activate(self, snapshot, history, expected, depth):
        self.assertIn(snapshot, self.sut.cache)
        image = self.sut.cache[snapshot].blob
        rows = []
        for repetition in range(5):
            before = self.backend._refresh_stats()
            logical_before = self.sut.store.work_counters()
            start = time.perf_counter_ns()
            with self.sut.materialize(snapshot) as (lease, published):
                actual = self.backend.logits(lease)
                position = self.backend.position(lease)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            after = self.backend._refresh_stats()
            row = dict(seconds=elapsed, before=before, after=after,
                       logical_work=delta(logical_before, self.sut.store.work_counters()))
            self.record(kind='physical-lease-activation', prefix_length=len(history), depth=depth,
                        repetition=repetition, **row)
            self.assertEqual(after['forward_tokens'], before['forward_tokens'])
            self.assertEqual(after['restores'] - before['restores'], 1)
            self.assertEqual(after['restored_running_bytes'] - before['restored_running_bytes'],
                             image.running_bytes)
            for name in ('copied_bytes', 'cow_pages', 'binding_copy_bytes',
                         'loaded_file_bytes', 'written_file_bytes', 'captures'):
                self.assertEqual(after[name], before[name], name)
            self.assertEqual(actual, expected)
            self.assertEqual(published, expected)
            self.assertEqual(position, len(history) - 1)
            rows.append(row)
        self.record(kind='physical-lease-latency', prefix_length=len(history), depth=depth,
                    latency_seconds=latency(rows),
                    scope='History validation, hot native restore, native frontier export and position check; '
                          'includes recurrent copies and any page-integrity scan; no O(1) claim')

    def test_g3_011_fork_latency_and_work(self):
        self.assert_ssd_store()
        self.assertTrue(callable(getattr(self.sut.store, 'work_counters', None)))
        self.assertTrue(callable(getattr(self.backend, 'audit', None)))
        fixture = tuple(self.profile['fixtures']['prefix'])
        suffixes = [tuple(self.profile['fixtures'][name][:2]) for name in ('suffix_a', 'suffix_b')]
        self.assertTrue(fixture)
        self.assertTrue(all(len(suffix) == 2 for suffix in suffixes))
        self.assertNotEqual(suffixes[0][0], suffixes[1][0])
        population = 0
        for fixture_index, length in enumerate((9, 257)):
            with self.subTest(prefix_length=length):
                self.sut.evict_all()
                history = tuple(fixture[i % len(fixture)] for i in range(length))
                root = self.sut.create(history)
                expected = self.fresh(root, history)
                retained = [self.retain(root, expected)]
                tip, current_depth = root, 0
                for batch, (depth, additional_references) in enumerate(((0, 0), (64, 0), (256, 0), (256, 128))):
                    # Real controller-only transitions vary ancestry depth
                    # without changing the consumed model history or frontier.
                    while current_depth < depth:
                        stream = f'fork-{fixture_index}-{current_depth}'
                        if current_depth == 0:
                            tip = self.sut.start_generation(tip, {}, stream)
                        else:
                            tip = self.sut.set_random_stream(tip, stream)
                        current_depth += 1
                    for branch in range(additional_references):
                        self.sut.store.fork(f'p{fixture_index}_{branch:03d}', tip)
                    population += additional_references
                    self.fork_batch(root, tip, history, depth, population, fixture_index * 10 + batch)
                    population += 20
                    self.activate(tip, history, expected, depth)
                # Both identical branch starts receive separate fresh forwards;
                # every subsequent forced token receives another full prefill.
                heads = [self.sut.store.fork(f'suffix{fixture_index}_{branch}', tip) for branch in (0, 1)]
                population += 2
                snapshots = [head.tip for head in heads]
                histories = [history, history]
                for snapshot in snapshots:
                    self.fresh(snapshot, history)
                for step in range(2):
                    for branch in (0, 1):
                        snapshots[branch] = self.sut.feed(snapshots[branch], (suffixes[branch][step],))
                        histories[branch] += (suffixes[branch][step],)
                        self.unchanged(retained)
                        retained.append(self.retain(snapshots[branch],
                                                    self.fresh(snapshots[branch], histories[branch])))
                self.unchanged(retained)
                self.sut.evict_all()
                before = self.backend._refresh_stats()
                self.fresh(root, history)
                after = self.backend._refresh_stats()
                self.record(kind='fork-cold-replay', prefix_length=length, before=before, after=after)
                self.assertEqual(after['forward_tokens'] - before['forward_tokens'], length)
