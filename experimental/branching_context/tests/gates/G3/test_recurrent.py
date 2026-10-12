"""T31: a hybrid model must restore its recurrent state for every branch."""
import unittest

import test_immutability as isolation


class RecurrentTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged

    @staticmethod
    def recurrent(audit):
        return {r['kind']: r for r in audit['regions']
                if r['kind'] in ('gdn', 'ple', 'tails', 'dead', 'block_pos')}

    def test_g3_006_recurrent_buffer_isolation(self):
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'Hybrid-state byte instrumentation is required')
        history = tuple(self.profile['fixtures']['prefix'][:5])
        root = self.sut.create(history)
        parent = self.retain(root, self.fresh(root, history))
        running = self.recurrent(parent['audit'])
        self.assertIn('gdn', running, 'This hybrid model must expose its actual recurrent state')
        self.assertGreater(running['gdn']['bytes'], 0)
        self.assertTrue(all(row['pool'] == -1 for row in running.values()),
                        'Recurrent running state is private, not a shared K/V pool')
        retained = [parent]
        tips = [root, root]
        histories = [history, history]
        suffixes = [tuple(self.profile['fixtures']['suffix_a'][:2]),
                    tuple(self.profile['fixtures']['suffix_b'][:2])]
        self.assertNotEqual(suffixes[0][0], suffixes[1][0])
        for step in range(2):
            for branch in (0, 1):
                old = tips[branch]
                checkpoint = self.sut.cache[old]
                before = self.backend._refresh_stats()
                token = suffixes[branch][step]
                tips[branch] = self.sut.feed(old, (token,))
                histories[branch] += (token,)
                after = self.backend._refresh_stats()
                self.record(kind='recurrent-restore-work', branch=branch, step=step,
                            before=before, after=after, running_bytes=checkpoint.blob.running_bytes)
                self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
                self.assertEqual(after['restores'] - before['restores'], 1)
                self.assertEqual(after['restored_running_bytes'] - before['restored_running_bytes'],
                                 checkpoint.blob.running_bytes)
                self.unchanged(retained)
                current = self.retain(tips[branch], self.fresh(tips[branch], histories[branch]))
                self.assertNotEqual(self.recurrent(current['audit'])['gdn']['sha256'],
                                    running['gdn']['sha256'],
                                    'The fixture must actually change recurrent state')
                retained.append(current)
        self.unchanged(retained)
        self.fresh(root, history)


if __name__ == '__main__':
    unittest.main()
