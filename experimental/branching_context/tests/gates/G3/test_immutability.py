"""T31: audit retained physical bytes through interleaved real generation.

Exact comparisons make this resource probe usable before calibration as a
supporting check. A gate still requires its separately frozen numerical profile.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle


class SharedPageTests(unittest.TestCase):
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

    def record(self, **row):
        path = Path(os.environ.get('CTT_SHARING_MEASUREMENTS',
                                   'evidence/runs/sharing-measurements.jsonl'))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as output:
            output.write(json.dumps(dict(test=self.id(), **row)) + '\n')

    def fresh(self, snapshot, history):
        expected = self.oracle.forward(history)
        with self.sut.materialize(snapshot) as (lease, published):
            actual = self.backend.logits(lease)
            self.assertEqual(self.backend.position(lease), len(history) - 1)
        self.assertEqual(self.sut.history(snapshot), history)
        self.assertEqual(len(actual), self.backend.vocab_size)
        self.assertEqual(len(actual), len(expected))
        self.record(kind='numerics', snapshot=snapshot, history=history,
                    max_absolute_error=max(abs(a - b) for a, b in zip(actual, expected)),
                    actual_sha256=hashlib.sha256(struct.pack('<' + str(len(actual)) + 'f', *actual)).hexdigest(),
                    expected_sha256=self.oracle.last['logits_sha256'],
                    oracle_computed_tokens=self.oracle.last['computed_tokens'])
        self.assertEqual(actual, expected)
        self.assertEqual(published, expected)
        self.assertEqual(self.oracle.last['computed_tokens'], len(history))
        self.assertEqual(self.oracle.last['position_max'], len(history) - 1)
        return expected

    def retain(self, snapshot, expected):
        image = self.sut.cache[snapshot].blob
        audit = self.backend.audit(image)
        self.assertTrue(any(r['pool'] >= 0 and r['filled_prefix_bytes'] > 0
                            for r in audit['regions']))
        self.record(kind='retained-image', snapshot=snapshot, audit=audit)
        return dict(snapshot=snapshot, image=image, audit=audit, frontier=expected,
                    logical=self.sut.snapshot_bytes(snapshot))

    def unchanged(self, retained):
        for saved in retained:
            # Read the retained image directly. Restoring a different image and
            # hashing that active lease would miss writes to abandoned siblings.
            after = self.backend.audit(saved['image'])
            self.record(kind='retained-image-after-generation',
                        snapshot=saved['snapshot'], audit=after)
            self.assertEqual(after['sha256'], saved['audit']['sha256'],
                             'A child wrote bytes reachable from an immutable checkpoint')
            fields = ('kind', 'pool', 'pool_offset', 'bytes', 'allocation_id',
                      'driver_handle', 'sha256')
            self.assertEqual([tuple(r[k] for k in fields) for r in after['regions']],
                             [tuple(r[k] for k in fields) for r in saved['audit']['regions']])
            self.assertEqual(self.sut.snapshot_bytes(saved['snapshot']), saved['logical'])

    def test_g3_004_shared_page_never_written(self):
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'Real retained-page byte instrumentation is required')
        history = tuple(self.profile['fixtures']['prefix'][:9])
        root = self.sut.create(history)
        expected = self.fresh(root, history)
        retained = [self.retain(root, expected)]
        audit = retained[0]['audit']
        self.assertNotEqual(len(history) % audit['qsa_page_cells'], 0,
                            'Counterexample must exercise a partially filled logical block')
        self.assertTrue(any(0 < r['filled_prefix_bytes'] < r['bytes']
                            for r in audit['regions'] if r['kind'] in ('k', 'v')),
                        'Counterexample must retain a partially filled physical page')
        tokens = [self.profile['fixtures']['suffix_a'][0],
                  self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*tokens)
        tips = [self.sut.start_generation(root, dict(allowed=[token]), 'cow-' + str(index))
                for index, token in enumerate(tokens)]
        histories = [history, history]
        frontiers = [expected, expected]
        scopes = [self.sut.snapshot_bytes(tip) for tip in tips]
        starts = list(tips)
        # Force different legal tokens through the real sampler/stream path.
        # Alternate children across three consumed boundaries, retaining every
        # prior image as well as the root, even if the bounded cache evicts it.
        for step in range(3):
            for branch in (0, 1):
                before = self.backend._refresh_stats()
                result = self.sut.generate(tips[branch], 1)
                self.assertEqual(result.accepted_tokens, (tokens[branch],))
                self.assertEqual(result.boundaries, (result.snapshot_id,))
                tips[branch] = result.snapshot_id
                histories[branch] += result.accepted_tokens
                after = self.backend._refresh_stats()
                self.record(kind='generation-work', step=step, branch=branch,
                            before=before, after=after)
                self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
                self.unchanged(retained)
                frontiers[branch] = self.fresh(tips[branch], histories[branch])
                retained.append(self.retain(tips[branch], frontiers[branch]))
                sibling = 1 - branch
                with self.sut.materialize(tips[sibling]) as (lease, published):
                    self.assertEqual(self.backend.logits(lease), frontiers[sibling])
                    self.assertEqual(published, frontiers[sibling])
                    self.assertEqual(self.backend.position(lease), len(histories[sibling]) - 1)
                self.assertEqual(self.sut.controller(tips[branch])['accepted_in_scope'], step + 1)
                self.assertEqual(self.sut.controller(tips[branch])['next_draw'], step + 1)
        self.unchanged(retained)
        for start, original in zip(starts, scopes):
            self.assertEqual(self.sut.snapshot_bytes(start), original)
        self.fresh(root, history)


if __name__ == '__main__':
    unittest.main()
