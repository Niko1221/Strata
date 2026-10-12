"""Real Q8 API integration regressions; not substitutes for registered gates.

Exact comparisons here are a deliberately strict regression check. They do not
freeze or modify the acceptance profile's independently calibrated tolerance.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import struct
import unittest

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.store import IntegrityError


class StrataApiTests(unittest.TestCase):
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
        self.addCleanup(self.sut.cache.clear)
        self.P, self.A, self.B = (1, 2, 3), (4, 6), (5, 7)

    def check(self, sid, history):
        expected = self.oracle.forward(history)
        actual = self.sut.next_logits(sid)
        self.observe(kind='numerics', snapshot=sid, history=history,
            max_absolute_error=max(abs(a-b) for a,b in zip(actual,expected)),
            actual_sha256=hashlib.sha256(struct.pack('<'+str(len(actual))+'f',*actual)).hexdigest(),
            expected_sha256=self.oracle.last['logits_sha256'])
        self.assertEqual(actual, expected)
        self.assertEqual(self.sut.history(sid), history)
        self.assertEqual(self.sut.consumed(sid), len(history))
        self.assertEqual(self.oracle.last['computed_tokens'], len(history))
        self.assertEqual(self.oracle.last['position_max'], len(history) - 1)

    def observe(self, **row):
        path = os.environ.get('CTT_API_OBSERVATIONS')
        if path:
            with open(path, 'a') as output:
                output.write(json.dumps(dict(test=self.id(), **row))+'\n')

    def test_hot_two_futures_and_rewind_match_fresh(self):
        before = self.backend.stats['consumed_tokens']
        root = self.sut.create(self.P)
        original = self.sut.snapshot_bytes(root)
        for suffix in (self.A, self.B):
            tip = root
            for consumed in range(len(suffix) + 1):
                self.check(tip, self.P + suffix[:consumed])
                if consumed < len(suffix):
                    tip = self.sut.feed(tip, suffix[consumed:consumed + 1])
        self.check(root, self.P)
        self.assertEqual(self.sut.snapshot_bytes(root), original)
        self.assertEqual(self.backend.stats['consumed_tokens'] - before, len(self.P + self.A + self.B))
        wrong = self.oracle.forward(self.P + self.A + self.B)
        self.assertNotEqual(self.sut.next_logits(tip), wrong)

    def test_cold_checkpoint_and_eviction_keep_history(self):
        root = self.sut.create(self.P)
        child = self.sut.feed(root, self.A[:1])
        history = self.P + self.A[:1]
        before_history = self.sut.snapshot_bytes(child)
        files = self.sut.checkpoint(child)
        payload = files['payload_path'].read_bytes()
        envelope = json.loads(files['manifest_path'].read_text())
        manifest = envelope['manifest']
        self.observe(kind='persisted-checkpoint', payload_bytes=len(payload),
            payload_sha256=hashlib.sha256(payload).hexdigest(), manifest=manifest,
            manifest_bytes=files['manifest_path'].stat().st_size,
            manifest_sha256=hashlib.sha256(files['manifest_path'].read_bytes()).hexdigest())
        self.assertGreater(manifest['state_bytes'], 1024 * 1024)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest['payload_sha256'])
        self.assertEqual(manifest['model_contract_id'], self.profile['model_contract_id'])
        self.assertEqual(manifest['execution_profile_id'], self.profile['execution_profile_id'])
        self.sut.cache.clear()
        before = self.backend.stats['consumed_tokens']
        self.check(child, history)
        self.assertEqual(self.backend.stats['consumed_tokens'], before)
        damaged = bytearray(payload)
        damaged[len(damaged) // 2] ^= 1
        files['payload_path'].write_bytes(damaged)
        self.sut.cache.clear()
        with self.assertRaises(IntegrityError):
            self.sut.next_logits(child)
        self.sut.evict_all()
        self.check(child, history)
        self.assertEqual(self.backend.stats['consumed_tokens'] - before, len(history))
        self.assertEqual(self.sut.snapshot_bytes(child), before_history)

    def test_interrupted_real_feed_keeps_published_root(self):
        root = self.sut.create(self.P)
        before = {p.name: p.read_bytes() for p in self.sut.store.objects.iterdir()}
        work = self.backend.stats['consumed_tokens']
        def interrupt(point):
            if point == 'after_token_consume':
                raise RuntimeError('test interruption after actual GPU consumption')
        self.sut._lease_fault = interrupt
        with self.assertRaisesRegex(RuntimeError, 'actual GPU consumption'):
            self.sut.feed(root, self.A)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.sut.store.objects.iterdir()})
        self.assertEqual(self.backend.stats['consumed_tokens'] - work, 1)
        self.sut._lease_fault = lambda point: None
        self.check(root, self.P)
        self.check(self.sut.feed(root, self.B[:1]), self.P + self.B[:1])

    def test_cold_restored_state_continues_with_real_forward(self):
        root = self.sut.create(self.P)
        child = self.sut.feed(root, self.A)
        self.sut.checkpoint(child)
        self.sut.cache.clear()
        before = self.backend.stats['consumed_tokens']
        tip = child
        for index, token in enumerate(self.B):
            tip = self.sut.feed(tip, (token,))
            self.check(tip, self.P + self.A + self.B[:index + 1])
        self.assertEqual(self.backend.stats['consumed_tokens'] - before, len(self.B))
        self.check(child, self.P + self.A)
        self.check(root, self.P)

    def test_optional_hot_cache_has_bounded_residency(self):
        root = self.sut.create(self.P)
        children = [self.sut.feed(root, (token,)) for token in range(10, 22)]
        original = {sid: self.sut.snapshot_bytes(sid) for sid in [root] + children}
        images = {id(value.blob): value.blob for value in self.sut.cache.values()}
        self.observe(kind='cache-residency', referenced_image_bytes=sum(map(len,images.values())),
                     cached_images=len(images), native_counters=dict(self.backend.stats))
        self.assertLessEqual(sum(map(len, images.values())), 1024 * 1024 * 1024)
        evicted = [sid for sid in children if sid not in self.sut.cache]
        self.assertTrue(evicted, 'Derived cache retained every numerical image')
        work = self.backend.stats['consumed_tokens']
        self.check(evicted[0], self.sut.history(evicted[0]))
        self.assertEqual(self.backend.stats['consumed_tokens'] - work, len(self.sut.history(evicted[0])))
        for sid in (root, children[-1]):
            self.check(sid, self.sut.history(sid))
        self.assertEqual(original, {sid: self.sut.snapshot_bytes(sid) for sid in original})

    def test_eviction_releases_native_handles_before_return(self):
        root = self.sut.create(self.P)
        baseline = self.backend.stats['live_bytes']
        a = self.sut.feed(root, self.A)
        b = self.sut.feed(root, self.B)
        self.assertGreater(self.backend.stats['live_bytes'], baseline)
        before = dict(self.backend.stats)
        self.sut.evict_all()
        self.observe(kind='eviction-releases', baseline_live_bytes=baseline,
                     before=before, after=dict(self.backend.stats))
        self.assertEqual(self.backend.stats['live_bytes'], baseline)
        self.assertEqual(self.backend.stats['consumed_tokens'], before['consumed_tokens'])
        self.assertEqual(self.sut.history(a), self.P + self.A)
        self.assertEqual(self.sut.history(b), self.P + self.B)


if __name__ == '__main__':
    unittest.main()
