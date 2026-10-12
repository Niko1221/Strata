"""T14: checked identities, independent work and consumed stream publication."""
from dataclasses import replace
import os
import unittest
import test_continuation as continuation
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, Store
from context_time_travel.numerics import assert_logits_close


class ProfileTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check

    def test_g1_023_incompatible_model_rejected(self):
        saved = self.sut.cache[self.root]
        self.sut.cache[self.root] = replace(saved,model_contract_id='wrong-weights')
        with self.assertRaises(IntegrityError):
            self.sut.next_logits(self.root)
        self.sut.cache[self.root] = saved
        for key,value in (('weights_sha256','0'*64),('architecture','other'),('adapters',('other',))):
            with self.assertRaises(IntegrityError):
                self.sut.create(self.P,contract=dict(self.sut.contract,**{key:value}))
        self.check(self.root,self.P)

    def test_g1_024_execution_profile_mismatch(self):
        saved = self.sut.cache[self.root]
        self.sut.cache[self.root] = replace(saved,execution_profile_id='wrong-cache-format')
        with self.assertRaises(IntegrityError):
            self.sut.next_logits(self.root)
        self.sut.cache[self.root] = saved
        original = os.environ.get('NVIDIA_TF32_OVERRIDE')
        try:
            os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
            with self.assertRaises(IntegrityError):
                self.backend.validate_execution_profile()
        finally:
            if original is None:
                os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
            else:
                os.environ['NVIDIA_TF32_OVERRIDE'] = original
        self.backend.validate_execution_profile()

    def test_g1_025_true_uncached_oracle(self):
        history = self.P+self.A
        requests,work = self.oracle.requests,self.oracle.tokens_computed
        first = self.oracle.forward(history)
        self.sut.cache.clear()
        second = self.oracle.forward(history)
        self.assertEqual(first,second)
        self.assertEqual(self.oracle.requests-requests,2)
        self.assertEqual(self.oracle.tokens_computed-work,2*len(history))
        self.assertEqual(self.oracle.last['position_max'],len(history)-1)
        self.assertEqual(self.sut.cache,{})

    def test_g1_026_near_tie_numerics(self):
        expected = self.oracle.forward(self.P)
        actual = self.sut.next_logits(self.root)
        atol,rtol = self.profile['atol'],self.profile['rtol']
        assert_logits_close(actual,expected,atol=atol,rtol=rtol)
        ordered = sorted(range(len(expected)),key=expected.__getitem__)
        near = min(zip(ordered,ordered[1:]),key=lambda pair:expected[pair[1]]-expected[pair[0]])
        self.assertLessEqual(expected[near[1]]-expected[near[0]],atol)
        corrupt = list(expected)
        corrupt[near[0]] += 2*atol+1
        with self.assertRaises(AssertionError):
            assert_logits_close(corrupt,expected,atol=atol,rtol=rtol)

    def test_g1_027_stream_publication_boundary(self):
        start = self.sut.start_generation(self.root,dict(allowed=sorted(set(self.A+self.B))), 'stream-publication')
        stream = self.sut.stream(start,3)
        for _ in range(3):
            sid,token = next(stream)
            reopened = Store(self.tmp.name)
            self.assertEqual(reopened.lineage(sid)[1][-1],token)
            self.sut.cache.clear()
            self.check(sid,tuple(reopened.lineage(sid)[1]))
            self.assertEqual(self.sut.consumed(sid),len(reopened.lineage(sid)[1]))
