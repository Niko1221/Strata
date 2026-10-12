"""T31: exhaust reachable logical and physical K/V tail offsets on the real model."""
import unittest

import test_immutability as isolation


class PartialTailTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    unchanged = isolation.SharedPageTests.unchanged

    def retain(self, snapshot, expected):
        image = self.sut.cache[snapshot].blob
        audit = self.backend.audit(image)
        self.assertEqual(audit['consumed'], len(self.sut.history(snapshot)))
        self.assertTrue(self.pages(audit))
        # The first partial logical block has no completely filled pool block.
        # Audit its actual physical pages and private running state nonetheless.
        self.record(kind='retained-image', snapshot=snapshot, audit=audit)
        return dict(snapshot=snapshot, image=image, audit=audit, frontier=expected,
                    logical=self.sut.snapshot_bytes(snapshot))

    @staticmethod
    def pages(audit):
        return {(row['pool'], row['pool_offset']): row
                for row in audit['regions'] if row['pool'] >= 0}

    def tail_pages(self, audit, consumed):
        """Page footprint of one new token, including its logical block.

        QSA may re-encode its incomplete logical block. The indexer may write
        both the current pooled row and its new spare row on block completion.
        Earlier complete blocks and later unused pages have no writable role.
        """
        writable = set()
        for key, row in self.pages(audit).items():
            width = row['bytes_per_cell']
            self.assertGreater(width, 0)
            if row['kind'] == 'indexer_pool':
                block = audit['index_block_cells']
                first = consumed // block
                end = (consumed + 1) // block + 1
            else:
                self.assertIn(row['kind'], ('k', 'v', 'k_scale', 'v_scale'))
                block = audit['qsa_page_cells']
                first = consumed // block * block
                end = first + block
            if row['pool_offset'] < end * width and row['pool_offset'] + row['bytes'] > first * width:
                writable.add(key)
        self.assertTrue(writable)
        return writable

    def append_and_check(self, root, saved, token, history, retained):
        before = self.backend._refresh_stats()
        child = self.sut.feed(root, (token,))
        after = self.backend._refresh_stats()
        # Inspect the child before consulting another engine. These are actual
        # allocations/copies, not estimates derived from elapsed inference time.
        audit = self.backend.audit(self.sut.cache[child].blob)
        original = self.pages(saved['audit'])
        current = self.pages(audit)
        self.assertEqual(original.keys(), current.keys())
        changed = {key for key in original
                   if current[key]['allocation_id'] != original[key]['allocation_id']}
        expected = self.tail_pages(saved['audit'], len(history))
        self.record(kind='partial-tail-work', consumed=len(history), token=token,
                    before=before, after=after, expected_private_pages=sorted(expected),
                    observed_private_pages=sorted(changed), child_audit=audit)
        self.assertEqual(changed, expected, 'Only the writable tail may receive private pages')
        self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
        self.assertEqual(after['cow_pages'] - before['cow_pages'], len(expected))
        self.assertEqual(after['allocations'] - before['allocations'], len(expected))
        self.assertEqual(after['copied_bytes'] - before['copied_bytes'],
                         sum(original[key]['bytes'] for key in expected))
        for key in original.keys() - expected:
            for field in ('allocation_id', 'driver_handle', 'sha256', 'filled_prefix_bytes'):
                self.assertEqual(current[key][field], original[key][field], (key, field))
        for key in expected:
            self.assertNotEqual(current[key]['driver_handle'], original[key]['driver_handle'])
        self.unchanged(retained)
        return self.retain(child, self.fresh(child, history + (token,)))

    def test_g3_003_partial_tail_cow(self):
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'Physical ownership and byte instrumentation are mandatory')
        fixture = tuple(self.profile['fixtures']['prefix'])
        capacity = self.profile['execution']['capacity']
        self.assertGreater(capacity, 1)
        history = fixture[:1]
        root = self.sut.create(history)
        geometry = self.backend.audit(self.sut.cache[root].blob)
        sizes = {geometry['qsa_page_cells'], geometry['index_block_cells']}
        kv = [row for row in self.pages(geometry).values()
              if row['kind'] in ('k', 'v', 'k_scale', 'v_scale')]
        self.assertTrue(kv)
        for row in kv:
            self.assertEqual(row['bytes'], geometry['allocation_page_bytes'])
            self.assertGreater(row['bytes_per_cell'], 0)
            self.assertEqual(row['bytes'] % row['bytes_per_cell'], 0)
            sizes.add(row['bytes'] // row['bytes_per_cell'])
        self.assertTrue(all(size > 0 for size in sizes))
        # Every reachable offset, not a representative sample. Also exercise
        # appends after complete physical pages and around the next boundary.
        lengths = set()
        for size in sizes:
            lengths.update(range(1, min(size, capacity)))
            lengths.update(n for n in (size, size + 1, 2 * size - 1, 2 * size)
                           if 0 < n < capacity)
        lengths = sorted(lengths)
        self.record(kind='partial-tail-coverage', page_cells=sorted(sizes),
                    capacity=capacity, required_prefix_lengths=lengths,
                    required_fresh_comparisons=3 * len(lengths))
        suffixes = [self.profile['fixtures']['suffix_a'][0],
                    self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*suffixes)
        completed = []
        for length in lengths:
            if length > len(history):
                extra = tuple(fixture[i % len(fixture)]
                              for i in range(len(history), length))
                root = self.sut.feed(root, extra)
                history += extra
            # Drop only optional old numerical images, retaining the current
            # hot root. Logical histories remain available throughout.
            for snapshot in tuple(self.sut.cache):
                if snapshot != root:
                    del self.sut.cache[snapshot]
            self.backend.release_checkpoints()
            saved = self.retain(root, self.fresh(root, history))
            children = [saved]
            for token in suffixes:
                children.append(self.append_and_check(root, saved, token, history, children))
            self.unchanged(children)
            completed.append(length)
            self.record(kind='partial-tail-position-pass', consumed=length)
            # Retained audit images must not accumulate across the sweep.
            del children, saved
        self.assertEqual(completed, lengths)
        self.record(kind='partial-tail-complete', completed_prefix_lengths=completed)


if __name__ == '__main__':
    unittest.main()
