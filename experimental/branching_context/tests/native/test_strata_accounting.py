"""T30 accounting regressions using actual retained Q8 GPU allocations."""
import unittest

import test_strata_api as api


class StrataAccountingTests(unittest.TestCase):
    setUpClass = classmethod(api.StrataApiTests.setUpClass.__func__)
    setUp = api.StrataApiTests.setUp
    check = api.StrataApiTests.check
    observe = api.StrataApiTests.observe

    def test_kv_totals_exclude_indexer_and_deduplicate_aliases(self):
        root = self.sut.create(self.P)
        checkpoint = self.sut.cache[root]
        audit = self.backend.audit(checkpoint.blob)
        pages = {row['allocation_id']: row for row in audit['regions']
                 if row['pool'] >= 0}
        kv = {key: row for key, row in pages.items()
              if row['kind'] in ('k', 'v', 'k_scale', 'v_scale')}
        index = {key: row for key, row in pages.items()
                 if row['kind'] == 'indexer_pool'}
        self.assertTrue(kv)
        self.assertTrue(index, 'The counterexample requires actual non-KV pages')
        self.assertEqual(set(pages), set(kv) | set(index))
        before = dict(self.backend.stats)
        for aliases in (1, 2, 8, 32):
            measured = self.backend.checkpoint_metrics([checkpoint] * aliases)
            self.observe(kind='separate-physical-accounting', aliases=aliases,
                         measured=measured, audited_pages=list(pages.values()))
            self.assertEqual(measured['checkpoint_unique_kv_bytes'],
                             sum(row['bytes'] for row in kv.values()))
            self.assertEqual(measured['checkpoint_unique_kv_pages'], len(kv))
            self.assertEqual(measured['checkpoint_unique_index_bytes'],
                             sum(row['bytes'] for row in index.values()))
            self.assertEqual(measured['checkpoint_unique_index_pages'], len(index))
            self.assertEqual(measured['checkpoint_unique_physical_bytes'],
                             sum(row['bytes'] for row in pages.values()))
            self.assertEqual(measured['checkpoint_unique_physical_pages'], len(pages))
        self.backend._refresh_stats()
        for counter in ('forward_tokens', 'allocations', 'cow_pages', 'copied_bytes',
                        'audit_read_bytes', 'captures', 'restores'):
            self.assertEqual(self.backend.stats[counter], before[counter], counter)
        self.assertEqual(audit['sha256'], self.backend.audit(checkpoint.blob)['sha256'])
        self.check(root, self.P)


if __name__ == '__main__':
    unittest.main()
