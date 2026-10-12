"""T13: complete persisted controller at real-model consumed boundaries."""
import unittest
import test_continuation as continuation
from context_time_travel.store import IntegrityError


class ControllerTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check

    def ready(self, **policy):
        defaults = dict(allowed=sorted(set(self.A+self.B)),temperature=1.0,eos_tokens=[])
        defaults.update(policy)
        return self.sut.start_generation(self.root,defaults,'fixed-branch-experiment')

    def test_g1_015_mid_generation_rng_replay(self):
        mid = self.sut.generate(self.ready(),7).snapshot_id
        before = self.sut.snapshot_bytes(mid)
        first = self.sut.generate(mid,20)
        second = self.sut.generate(mid,20)
        self.assertEqual(first,second)
        self.assertEqual(before,self.sut.snapshot_bytes(mid))
        self.assertGreater(self.sut.controller(mid)['next_draw'],0)
        for boundary in first.boundaries:
            self.check(boundary,self.sut.history(boundary))

    def test_g1_016_sampler_multiple_draws(self):
        start = self.ready(rejection_threshold=0.5)
        mid = self.sut.generate(start,7)
        state = self.sut.controller(mid.snapshot_id)
        self.assertGreater(state['next_draw'],7)
        self.assertEqual(self.sut.generate(mid.snapshot_id,20),self.sut.generate(mid.snapshot_id,20))
        self.assertEqual(self.sut.generate(start,27).accepted_tokens,
                         mid.accepted_tokens+self.sut.generate(mid.snapshot_id,20).accepted_tokens)

    def test_g1_017_grammar_mid_production(self):
        a,b = self.A[0],self.B[-1]
        self.assertNotEqual(a,b)
        grammar = {'start':'a','transitions':{'a':{str(a):'b'},'b':{str(b):'a'}}}
        start = self.ready(grammar=grammar)
        mid = self.sut.generate(start,7)
        self.assertEqual(self.sut.controller(mid.snapshot_id)['grammar_state'],'b')
        result = self.sut.generate(mid.snapshot_id,4)
        self.assertEqual(result.accepted_tokens,(b,a,b,a))
        self.assertEqual(result,self.sut.generate(mid.snapshot_id,4))
        with self.assertRaises(IntegrityError):
            self.sut.feed(mid.snapshot_id,(a,))

    def test_g1_018_stop_progress_and_eos(self):
        a,b = self.A[0],self.B[-1]
        grammar = {'start':'a','transitions':{'a':{str(a):'b'},'b':{str(b):'a'}}}
        first = self.sut.generate(self.ready(grammar=grammar,stop_sequences=[[a,b]]),1)
        second = self.sut.generate(first.snapshot_id,10)
        self.assertEqual(second.accepted_tokens,(b,))
        self.assertEqual(self.sut.generate(second.snapshot_id,10).accepted_tokens,())
        eos = self.sut.generate(self.ready(allowed=[a],eos_tokens=[a]),10)
        self.assertEqual(eos.accepted_tokens,(a,))
        self.check(eos.snapshot_id,self.P+(a,))

    def test_g1_019_chunked_generate_equivalence(self):
        start = self.ready(repetition_penalty=1.15)
        whole = self.sut.generate(start,27)
        first = self.sut.generate(start,7)
        second = self.sut.generate(first.snapshot_id,20)
        self.assertEqual(whole.accepted_tokens,first.accepted_tokens+second.accepted_tokens)
        self.assertEqual(self.sut.controller(whole.snapshot_id),self.sut.controller(second.snapshot_id))
        self.assertEqual(whole.snapshot_id,second.snapshot_id)

    def test_g1_020_sibling_rng_independence(self):
        start = self.ready()
        expected = self.sut.generate(start,7)
        sibling = self.sut.set_random_stream(start,'sibling')
        self.sut.generate(sibling,101)
        self.assertEqual(expected,self.sut.generate(start,7))
        self.assertEqual(self.sut.controller(start)['next_draw'],0)

    def test_g1_021_alternate_stream_not_forced_different(self):
        start = self.ready(allowed=[self.A[0]])
        other = self.sut.set_random_stream(start,'different')
        self.assertNotEqual(self.sut.controller(start)['stream_key'],self.sut.controller(other)['stream_key'])
        self.assertEqual(self.sut.generate(start,3).accepted_tokens,self.sut.generate(other,3).accepted_tokens)

    def test_g1_022_unsupported_hybrid_state(self):
        for kind in ('hybrid','recurrent','convolution'):
            contract = dict(self.sut.contract,state_kind=kind)
            with self.assertRaises(IntegrityError):
                self.sut.create(self.P,contract=contract)
        self.check(self.root,self.P)
