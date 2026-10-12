"""Direct real-model byte audits, separate from registered gate qualification."""
from dataclasses import replace
import unittest

import test_strata_api as api
from context_time_travel.store import IntegrityError


class StrataAuditTests(unittest.TestCase):
    setUpClass = classmethod(api.StrataApiTests.setUpClass.__func__)
    setUp = api.StrataApiTests.setUp
    check = api.StrataApiTests.check
    observe = api.StrataApiTests.observe

    def test_retained_gpu_bytes_are_audited_without_restore(self):
        root = self.sut.create(self.P)
        root_image = self.sut.cache[root].blob
        first = self.backend.audit(root_image)
        self.assertGreater(first['bytes'], 100 * 1024 * 1024)
        self.assertEqual(first['consumed'], len(self.P))
        self.assertEqual(first['qsa_page_cells'], 4)
        self.assertEqual(first['index_block_cells'], 4)
        sections = {row['kind'] for row in first['regions']}
        self.assertTrue({'k', 'v', 'indexer_pool', 'gdn', 'ple', 'tails', 'dead', 'block_pos', 'ids', 'logits'} <= sections)
        self.assertFalse({'k_scale', 'v_scale'} & sections, 'Pinned FP16 KV must report its actual layout')
        self.assertTrue(any(row['pool'] >= 0 and row['filled_prefix_bytes'] == 0 for row in first['regions']))
        a = self.sut.feed(root, self.A[:1])
        child_image = self.sut.cache[a].blob
        child = self.backend.audit(child_image)
        before = dict(self.backend.stats)
        second = self.backend.audit(root_image)
        after = dict(self.backend.stats)
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertNotEqual(first['sha256'], child['sha256'])
        self.assertEqual([r['sha256'] for r in first['regions']], [r['sha256'] for r in second['regions']])
        for counter in ('forward_tokens', 'restores', 'captures', 'copied_bytes', 'allocations', 'releases', 'live_bytes'):
            self.assertEqual(before[counter], after[counter], counter)
        self.assertEqual(after['consumed'], len(self.P) + 1)
        gpu_bytes = sum(r['bytes'] for r in second['regions'] if r['pool'] >= 0)
        self.assertEqual(after['audit_read_bytes'] - before['audit_read_bytes'], gpu_bytes)
        with self.assertRaises(IntegrityError):
            self.backend.audit(replace(root_image, handle=99999999))
        self.assertEqual(self.backend.stats['forward_tokens'], before['forward_tokens'])
        self.observe(kind='saved-image-byte-audit', root_before=first, root_after=second, child=child,
                     counters_before=before, counters_after=after)
        self.check(root, self.P)
        self.check(a, self.P + self.A[:1])
        b = self.sut.feed(root, self.B[:1])
        self.check(b, self.P + self.B[:1])
        self.assertEqual(first['sha256'], self.backend.audit(root_image)['sha256'])
        self.assertEqual(child['sha256'], self.backend.audit(child_image)['sha256'])


if __name__ == '__main__':
    unittest.main()
