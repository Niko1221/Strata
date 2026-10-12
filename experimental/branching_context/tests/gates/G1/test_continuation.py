"""T11: real-model continuation against an independently allocated oracle."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.numerics import assert_logits_close


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
        cls.oracle = FreshOracle(cls.profile)
        cls.backend = BranchingModel.load_backend(cls.profile)

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()
        cls.backend.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sut = BranchingModel(self.tmp.name, self.backend, self.profile)
        self.P = tuple(self.profile['fixtures']['prefix'])
        self.A = tuple(self.profile['fixtures']['suffix_a'])
        self.B = tuple(self.profile['fixtures']['suffix_b'])
        self.root = self.sut.create(self.P)

    def check(self, snapshot, history):
        expected = self.oracle.forward(history)
        actual = self.sut.next_logits(snapshot)
        report = assert_logits_close(actual, expected,
            atol=self.profile['atol'], rtol=self.profile['rtol'])
        self.sut.measure(report, snapshot, history)
        self.assertEqual(self.sut.history(snapshot), history)
        self.assertEqual(self.sut.consumed(snapshot), len(history))
        self.assertEqual(self.sut.position_max(snapshot), len(history)-1)

    def test_g1_001_fresh_suffix_a(self):
        tip = self.root
        for i in range(len(self.A)+1):
            self.check(tip, self.P+self.A[:i])
            if i < len(self.A):
                tip = self.sut.feed(tip, self.A[i:i+1])
        self.check(self.root, self.P)

    def test_g1_002_fresh_suffix_b(self):
        a = self.sut.feed(self.root, self.A)
        b = self.root
        for i in range(len(self.B)+1):
            self.check(b, self.P+self.B[:i])
            if i < len(self.B):
                b = self.sut.feed(b, self.B[i:i+1])
        self.check(a, self.P+self.A)
        wrong = self.oracle.forward(self.P+self.A+self.B)
        with self.assertRaises(AssertionError):
            assert_logits_close(self.sut.next_logits(b), wrong,
                atol=self.profile['atol'], rtol=self.profile['rtol'])

    def test_g1_003_empty_suffix_frontier(self):
        self.check(self.root, self.P)
        saved = self.sut.snapshot_bytes(self.root)
        self.sut.feed(self.root, self.A)
        self.check(self.root, self.P)
        self.assertEqual(saved, self.sut.snapshot_bytes(self.root))

    def test_g1_004_every_accepted_boundary(self):
        tip = self.root
        for token in self.A+self.B:
            parent = tip
            before = self.sut.snapshot_bytes(parent)
            tip = self.sut.feed(parent, (token,))
            self.check(tip, self.sut.history(parent)+(token,))
            self.assertEqual(before, self.sut.snapshot_bytes(parent))

    def test_g1_005_split_prefill_associativity(self):
        whole = self.sut.create(self.P+self.A)
        split = self.sut.create(self.P[:1])
        for token in self.P[1:]+self.A:
            split = self.sut.feed(split, (token,))
        self.check(whole, self.P+self.A)
        self.check(split, self.P+self.A)


if __name__ == '__main__':
    unittest.main()
