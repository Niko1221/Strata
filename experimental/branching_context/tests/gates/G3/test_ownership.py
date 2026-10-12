"""T30: inspect physical ownership and counters from actual native work.

These resource tests require exact frontier identity. They do not establish or
replace the independently frozen numerical tolerance required by G1.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import IntegrityError


class OwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        cls.backend = BranchingModel.load_backend(cls.profile)
        cls.addClassCleanup(cls.backend.close)
        cls.oracle = FreshOracle(cls.profile)
        cls.addClassCleanup(cls.oracle.close)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sut = BranchingModel(self.tmp.name, self.backend, self.profile)
        self.addCleanup(self.sut.evict_all)
        self.prefix = tuple(self.profile['fixtures']['prefix'][:9])
        self.a = self.profile['fixtures']['suffix_a'][0]
        self.b = self.profile['fixtures']['suffix_b'][0]

    def record(self, **row):
        path = Path(os.environ.get('CTT_SHARING_MEASUREMENTS',
                                   'evidence/runs/sharing-measurements.jsonl'))
        with path.open('a') as output:
            output.write(json.dumps(dict(test=self.id(), **row)) + '\n')

    def exact_fresh(self, actual, history):
        expected = self.oracle.forward(history)
        self.assertEqual(len(actual), len(expected))
        packed = struct.pack('<' + str(len(actual)) + 'f', *actual)
        self.record(kind='numerics', history=history,
                    max_absolute_error=max(abs(a - b) for a, b in zip(actual, expected)),
                    actual_sha256=hashlib.sha256(packed).hexdigest(),
                    expected_sha256=self.oracle.last['logits_sha256'],
                    oracle_computed_tokens=self.oracle.last['computed_tokens'])
        self.assertEqual(actual, expected)
        self.assertEqual(self.oracle.last['computed_tokens'], len(history))
        self.assertEqual(self.oracle.last['position_max'], len(history) - 1)

    def check_fork_ownership(self, history, *, require_full_pages):
        self.sut.evict_all()
        root = self.sut.create(history)
        # An opaque, fully copied checkpoint cannot satisfy this prerequisite.
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'G3 requires a direct retained physical-page audit')
        # freeze() shares the table itself. Restore makes an independent lease
        # table, so physical-page use_count now measures two direct owners.
        with self.sut.materialize(root) as (lease, frontier):
            self.assertEqual(self.backend.logits(lease), frontier)
        image = self.sut.cache[root].blob
        first = self.backend.audit(image)
        pages = {r['allocation_id']: r for r in first['regions'] if r['pool'] >= 0}
        consumed_pages = {key: row for key, row in pages.items()
                          if row['filled_prefix_bytes'] > 0}
        self.assertTrue(consumed_pages, 'Unused reserved capacity is not prefix sharing')
        kv_regions = [row for row in pages.values() if row['kind'] in ('k', 'v')]
        self.assertTrue(kv_regions)
        if require_full_pages:
            self.assertEqual({row['pool'] for row in kv_regions},
                             {row['pool'] for row in kv_regions
                              if row['filled_prefix_bytes'] == row['bytes']},
                             'Every K/V pool must include a completely consumed physical page')
        self.assertTrue(all(row['driver_handle'] and row['owners'] >= 2
                            for row in consumed_pages.values()))
        total = sum(row['bytes'] for row in pages.values())
        kv_pages = {key: row for key, row in pages.items()
                    if row['kind'] in ('k', 'v', 'k_scale', 'v_scale')}
        kv_bytes = sum(row['bytes'] for row in kv_pages.values())
        index_pages = {key: row for key, row in pages.items() if row['kind'] == 'indexer_pool'}
        self.assertEqual(set(pages), set(kv_pages) | set(index_pages))
        baseline = dict(self.backend.stats)
        root_bytes = self.sut.snapshot_bytes(root)
        rows = []
        for index in range(32):
            start = time.perf_counter()
            head = self.sut.store.fork('n' + str(len(history)) + '-fork-' + str(index), root)
            fork_seconds = time.perf_counter() - start
            self.assertEqual(head.tip, root)
            start = time.perf_counter()
            with self.sut.materialize(head.tip) as (lease, frontier):
                active = self.backend._request('pages')['pages']
                actual = self.backend.logits(lease)
                self.assertEqual(actual, frontier)
                self.assertEqual({row[0]: (row[1], row[3]) for row in active},
                                 {key: (row['driver_handle'], row['bytes'])
                                  for key, row in pages.items()})
            activation_seconds = time.perf_counter() - start
            metrics = self.backend.checkpoint_metrics(self.sut.cache.values())
            self.assertEqual(metrics['checkpoint_unique_physical_pages'], len(pages))
            self.assertEqual(metrics['checkpoint_unique_physical_bytes'], total)
            self.assertEqual(metrics['checkpoint_unique_kv_pages'], len(kv_pages))
            self.assertEqual(metrics['checkpoint_unique_kv_bytes'], kv_bytes)
            self.assertEqual(metrics['checkpoint_unique_index_pages'], len(index_pages))
            self.assertEqual(metrics['checkpoint_unique_index_bytes'], total - kv_bytes)
            self.backend._refresh_stats()
            for counter in ('forward_tokens', 'allocations', 'cow_pages', 'copied_bytes', 'captures'):
                self.assertEqual(self.backend.stats[counter], baseline[counter], counter)
            if index + 1 in (1, 2, 8, 32):
                rows.append(dict(branches=index + 1, unique_kv_bytes=kv_bytes,
                                 unique_physical_bytes=total, unique_index_bytes=total - kv_bytes,
                                 unique_pages=len(pages), consumed_prefix_pages=len(consumed_pages),
                                 fork_seconds=fork_seconds, activation_seconds=activation_seconds,
                                 counters=dict(self.backend.stats)))
        last = self.backend.audit(image)
        self.assertEqual(first['sha256'], last['sha256'])
        self.assertEqual(root_bytes, self.sut.snapshot_bytes(root))
        self.record(kind='unique-prefix-ownership', history=history, observations=rows,
                    before=first, after=last, counters_before=baseline,
                    counters_after=dict(self.backend.stats))
        self.exact_fresh(actual, history)
        # A physical page contains whole logical blocks. Choose a second
        # fixture beyond a completely consumed physical page in every K/V pool.
        block = first['qsa_page_cells']
        cells = max((row['bytes'] + row['bytes_per_cell'] - 1) // row['bytes_per_cell']
                    for row in kv_regions)
        return ((cells + block - 1) // block) * block + 1

    def test_g3_002_unique_kv_page_accounting(self):
        length = self.check_fork_ownership(self.prefix, require_full_pages=False)
        self.assertLessEqual(length, self.profile['execution']['capacity'])
        fixture = tuple(self.profile['fixtures']['prefix'])
        history = (fixture * ((length + len(fixture) - 1) // len(fixture)))[:length]
        self.check_fork_ownership(history, require_full_pages=True)

    def test_g3_015_counters_are_direct(self):
        history = self.prefix[:3]
        root = self.sut.create(history)
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'G3 requires physical instrumentation, not frontend estimates')
        image = self.sut.cache[root].blob
        original = self.backend.audit(image)
        published = {p.name: p.read_bytes() for p in self.sut.store.objects.iterdir()}
        with self.sut.materialize(root) as (lease, frontier):
            before = self.backend._request('stats')
            cached_counter = self.backend.stats['consumed_tokens']
            # Bypass consume(): only the native Verifier can know this work ran.
            # The private lease is deliberately left inconsistent and discarded.
            reply = self.backend._request('feed ' + str(self.a))
            direct = self.backend._request('stats')
            self.assertEqual(self.backend.stats['consumed_tokens'], cached_counter)
            self.assertEqual(reply['consumed'], len(history) + 1)
            self.assertEqual(direct['forward_tokens'] - before['forward_tokens'], 1)
            self.assertEqual(direct['windows'] - before['windows'], 1)
            self.assertGreater(direct['copied_bytes'], before['copied_bytes'])
            path = Path(self.tmp.name) / 'direct-frontier.f32'
            saved = self.backend._request('logits ' + str(path))
            raw = path.read_bytes()
            self.assertEqual(len(raw), saved['bytes'])
            self.assertEqual(len(raw), self.backend.vocab_size * 4)
            actual = struct.unpack('<' + str(self.backend.vocab_size) + 'f', raw)
            path.unlink()
            self.backend._refresh_stats()
            self.assertEqual(self.backend.stats['consumed_tokens'], direct['forward_tokens'])
            with self.assertRaisesRegex(IntegrityError, 'consumed position'):
                self.backend.position(lease)
        self.assertEqual(published, {p.name: p.read_bytes() for p in self.sut.store.objects.iterdir()})
        self.assertEqual(original['sha256'], self.backend.audit(image)['sha256'])
        self.exact_fresh(actual, history + (self.a,))
        with self.sut.materialize(root) as (lease, frontier):
            self.assertEqual(self.backend.logits(lease), frontier)
            restored = self.backend._request('stats')
        self.assertEqual(restored['forward_tokens'], direct['forward_tokens'])
        self.exact_fresh(frontier, history)
        child = self.sut.feed(root, (self.b,))
        self.assertEqual(self.backend.stats['consumed_tokens'] - direct['forward_tokens'], 1)
        self.exact_fresh(self.sut.next_logits(child), history + (self.b,))
        self.record(kind='direct-native-work', before=before, direct=direct,
                    restored=restored, after=dict(self.backend.stats),
                    parent_before=original, parent_after=self.backend.audit(image))


if __name__ == '__main__':
    unittest.main()
