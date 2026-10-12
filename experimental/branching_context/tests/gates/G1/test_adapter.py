"""T12: real cache restoration, cache boundaries and model-input identity."""
import unittest
import test_continuation as continuation
from context_time_travel.store import IntegrityError, InvalidOperation


class AdapterTests(unittest.TestCase):
    setUpClass = classmethod(continuation.ContinuationTests.setUpClass.__func__)
    tearDownClass = classmethod(continuation.ContinuationTests.tearDownClass.__func__)
    setUp = continuation.ContinuationTests.setUp
    check = continuation.ContinuationTests.check

    def cache_page_cells(self):
        # Keep the original contiguous backend's inspected padding fixture.
        # Strata reports its actual logical and physical allocation geometry;
        # attention padding is not a substitute for either of those boundaries.
        if self.profile.get('backend', 'llama.cpp') != 'strata':
            return {256}
        audit = self.backend.audit(self.sut.cache[self.root].blob)
        kv = [r for r in audit['regions'] if r['kind'] in ('k', 'v')]
        self.assertTrue(kv, 'Native K/V allocation geometry must be measured')
        sizes = {audit['qsa_page_cells']}
        self.assertGreater(audit['qsa_page_cells'], 0)
        for region in kv:
            self.assertGreater(region['bytes_per_cell'], 0)
            self.assertEqual(region['bytes'], audit['allocation_page_bytes'])
            self.assertEqual(region['bytes'] % region['bytes_per_cell'], 0)
            sizes.add(region['bytes'] // region['bytes_per_cell'])
        return sizes

    def test_g1_006_partial_page_boundary(self):
        lengths = {1, 255, 256, 257, 511, 512}
        for page in self.cache_page_cells():
            lengths.update((page - 1, page, page + 1, 2 * page - 1, 2 * page))
        capacity = self.profile['execution'].get('capacity', 2048)
        for n in sorted(length for length in lengths if 0 < length <= capacity):
            with self.subTest(length=n):
                p = (self.P*((n+len(self.P)-1)//len(self.P)))[:n]
                root = self.sut.create(p)
                original = self.sut.cache[root].blob
                self.check(root, p)
                if n == capacity:
                    # The exact-capacity boundary remains a valid snapshot.
                    # Appending either alternative must fail before ingest or
                    # publication; never grow the profile or truncate history.
                    objects = {file.name: file.read_bytes() for file in self.sut.store.objects.iterdir()}
                    work = self.backend.stats['consumed_tokens']
                    for suffix in (self.A[:1], self.B[-1:]):
                        with self.assertRaises(InvalidOperation):
                            self.sut.feed(root, suffix)
                    self.assertEqual(self.backend.stats['consumed_tokens'], work)
                    self.assertEqual(objects, {file.name: file.read_bytes()
                                             for file in self.sut.store.objects.iterdir()})
                    self.assertEqual(original, self.sut.cache[root].blob)
                    self.check(root, p)
                    continue
                a = self.sut.feed(root,self.A[:1])
                b = self.sut.feed(root,self.B[-1:])
                self.check(a,p+self.A[:1])
                self.check(b,p+self.B[-1:])
                self.assertEqual(original,self.sut.cache[root].blob)
                self.check(root,p)

    def test_g1_007_repeated_token_history(self):
        one = self.P+(self.A[0],)*5
        two = self.P+(self.B[-1],)+(self.A[0],)*4
        self.assertNotEqual(one,two)
        a,b = self.sut.create(one),self.sut.create(two)
        self.assertNotEqual(a,b)
        self.check(a,one)
        self.check(b,two)

    def test_g1_008_same_last_block_different_prefix(self):
        capacity = self.profile['execution'].get('capacity', 2048)
        for size in sorted({256} | self.cache_page_cells()):
            with self.subTest(page_cells=size):
                self.assertLessEqual(size + max(len(self.A), len(self.B)) + 1, capacity,
                                     'Declared layout needs a larger executed identity fixture')
                tail = (self.P * ((size + len(self.P) - 1) // len(self.P)))[:size]
                pa,pb = self.A+tail,self.B+tail
                a,b = self.sut.create(pa),self.sut.create(pb)
                self.check(self.sut.feed(a,self.A[:1]),pa+self.A[:1])
                self.check(self.sut.feed(b,self.A[:1]),pb+self.A[:1])

    def test_g1_009_absolute_positions(self):
        p = (self.P*100)[:1024]
        root = self.sut.create(p)
        for i in range(3):
            root = self.sut.feed(root,self.A[i:i+1])
            self.check(root,p+self.A[:i+1])
            self.assertEqual(self.sut.position_max(root),1024+i)

    def test_g1_010_rope_and_yarn_contract(self):
        for key,value in (('rope_base','10000'),('yarn','changed')):
            bad = dict(self.sut.contract)
            bad[key] = value
            with self.assertRaises(IntegrityError):
                self.sut.create(self.P,contract=bad)
        self.check(self.root,self.P)

    def test_g1_011_tokenizer_and_template_contract(self):
        for key in ('tokenizer_identity','template_identity'):
            bad = dict(self.sut.contract)
            bad[key] = 'different-identity'
            with self.assertRaises(IntegrityError):
                self.sut.create(self.P,contract=bad)
        self.check(self.root,self.P)

    def test_g1_012_attention_mask_visibility(self):
        for key,value in (('attention','bidirectional'),('attention','sliding-window'),('input_kind','multimodal')):
            bad = dict(self.sut.contract)
            bad[key] = value
            with self.assertRaises(IntegrityError):
                self.sut.create(self.P,contract=bad)
        self.check(self.root,self.P)

    def test_g1_013_different_interleave_order(self):
        for order in ((0,1),(1,0)):
            tips = [self.root,self.root]
            suffixes = [self.A,self.B]
            for step in range(max(map(len,suffixes))):
                for branch in order:
                    if step < len(suffixes[branch]):
                        tips[branch] = self.sut.feed(tips[branch],suffixes[branch][step:step+1])
                        self.check(tips[branch],self.P+suffixes[branch][:step+1])

    def test_g1_014_pending_sampled_token(self):
        logits = self.sut.next_logits(self.root)
        selected = max(range(len(logits)),key=logits.__getitem__)
        # Merely choosing a token must leave the public prefix unchanged.
        self.assertEqual(self.sut.consumed(self.root),len(self.P))
        accepted = self.sut.feed(self.root,(selected,))
        self.check(accepted,self.P+(selected,))
        self.check(self.root,self.P)
