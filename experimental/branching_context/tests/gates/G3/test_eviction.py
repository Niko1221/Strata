"""T32: evict one hot image without losing its history or retained siblings."""
import gc
import unittest
import weakref

import test_immutability as isolation


class EvictionTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged

    @staticmethod
    def pages(audit):
        return {row['allocation_id']: row for row in audit['regions'] if row['pool'] >= 0}

    def test_g3_007_evict_one_branch_keep_other(self):
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'Eviction requires direct physical ownership and byte instrumentation')
        history = tuple(self.profile['fixtures']['prefix'][:9])
        root = self.sut.create(history)
        root_saved = self.retain(root, self.fresh(root, history))
        tokens = [self.profile['fixtures']['suffix_a'][0],
                  self.profile['fixtures']['suffix_b'][0]]
        self.assertNotEqual(*tokens)
        tips, histories, starts, heads = [], [], [], []
        for branch, token in enumerate(tokens):
            name = 'eviction-' + str(branch)
            start = self.sut.start_generation(root, dict(allowed=[token]), name)
            starts.append(start)
            head = self.sut.store.fork(name, start)
            result = self.sut.generate(start, 1)
            self.assertEqual(result.accepted_tokens, (token,))
            self.assertEqual(result.boundaries, (result.snapshot_id,))
            tips.append(result.snapshot_id)
            histories.append(history + (token,))
            heads.append(self.sut.store.move(name, head.revision, result.snapshot_id, name + '-publish'))
            self.fresh(result.snapshot_id, histories[-1])

        # Keep root/B alive, but retain only A's observed bytes/IDs, never its
        # image object. Otherwise the test itself would prevent A's eviction.
        a, b = tips
        a_audit = self.backend.audit(self.sut.cache[a].blob)
        kept = [root_saved, self.retain(b, self.sut.next_logits(b))]
        a_pages = self.pages(a_audit)
        kept_pages = set().union(*(self.pages(saved['audit']) for saved in kept))
        private = set(a_pages) - kept_pages
        self.assertTrue(private, 'The evicted child must own actual private tail pages')
        self.assertTrue(any(a_pages[key]['filled_prefix_bytes'] > 0 for key in private),
                        'Unused reserved capacity alone cannot prove child-state eviction')
        shared = set(a_pages) & kept_pages
        self.assertTrue(shared, 'The fixture must retain pages with another owner')
        released_bytes = sum(a_pages[key]['bytes'] for key in private)
        logical = {snapshot: self.sut.snapshot_bytes(snapshot) for snapshot in [root, *starts, *tips]}
        controllers = {snapshot: self.sut.controller(snapshot) for snapshot in tips}
        self.assertEqual([controllers[tip]['next_draw'] for tip in tips], [1, 1])
        self.assertEqual(list(self.sut.checkpoint_directory.iterdir()), [])

        # B is the active native lease. No transient lease may keep A's private
        # pages alive while the explicit numerical-cache entry is removed.
        with self.sut.materialize(b):
            pass
        old_image = weakref.ref(self.sut.cache[a].blob)
        before = self.backend._refresh_stats()
        del self.sut.cache[a]
        gc.collect()
        self.backend.release_checkpoints()
        after = self.backend._refresh_stats()
        self.record(kind='single-image-eviction', snapshot=a, before=before, after=after,
                    expected_released_pages=sorted(private), expected_released_bytes=released_bytes,
                    shared_allocations_preserved=sorted(shared), evicted_audit=a_audit)
        self.assertIsNone(old_image(), 'The evicted image must have no Python owner')
        self.assertNotIn(a, self.sut.cache)
        self.assertEqual(after['releases'] - before['releases'], len(private),
                         'Eviction must release exactly the child-only physical pages')
        self.assertEqual(before['live_bytes'] - after['live_bytes'], released_bytes)
        for counter in ('forward_tokens', 'allocations', 'cow_pages', 'copied_bytes',
                        'restored_running_bytes', 'written_file_bytes', 'loaded_file_bytes'):
            self.assertEqual(after[counter], before[counter], counter)
        self.unchanged(kept)
        for snapshot, original in logical.items():
            self.assertEqual(self.sut.snapshot_bytes(snapshot), original)
        for head in heads:
            self.assertEqual(self.sut.store.head(head.name), head)

        # Root/B stay hot and exact. A must replay every retained input without
        # rerolling its prior output or advancing the saved controller cursor.
        before = self.backend._refresh_stats()
        self.fresh(b, histories[1])
        self.fresh(root, history)
        after_hot = self.backend._refresh_stats()
        self.assertEqual(after_hot['forward_tokens'], before['forward_tokens'])
        self.fresh(a, histories[0])
        after_cold = self.backend._refresh_stats()
        self.record(kind='single-image-reconstruction', before=before,
                    after_hot=after_hot, after_cold=after_cold, replayed_history=histories[0])
        self.assertEqual(after_cold['forward_tokens'] - after_hot['forward_tokens'], len(histories[0]))
        self.assertEqual(after_cold['loaded_file_bytes'], after_hot['loaded_file_bytes'])
        for snapshot in tips:
            self.assertEqual(self.sut.controller(snapshot), controllers[snapshot])
        self.unchanged(kept)

        for branch, tip in enumerate(tips):
            before = self.backend._refresh_stats()
            result = self.sut.generate(tip, 1)
            after = self.backend._refresh_stats()
            self.record(kind='post-eviction-continuation', branch=branch, before=before, after=after)
            self.assertEqual(result.accepted_tokens, (tokens[branch],))
            self.assertEqual(result.boundaries, (result.snapshot_id,))
            self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
            self.assertEqual(self.sut.controller(result.snapshot_id)['next_draw'], 2)
            self.fresh(result.snapshot_id, histories[branch] + (tokens[branch],))
            self.unchanged(kept)
        for snapshot, original in logical.items():
            self.assertEqual(self.sut.snapshot_bytes(snapshot), original)
        for head in heads:
            self.assertEqual(self.sut.store.head(head.name), head)


if __name__ == '__main__':
    unittest.main()
