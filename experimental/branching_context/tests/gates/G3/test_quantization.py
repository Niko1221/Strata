"""T31: real INT8 K/V scale pages must belong to each divergent suffix."""
import unittest

import test_immutability as isolation


class QuantizationTests(unittest.TestCase):
    setUpClass = classmethod(isolation.SharedPageTests.setUpClass.__func__)
    setUp = isolation.SharedPageTests.setUp
    record = isolation.SharedPageTests.record
    fresh = isolation.SharedPageTests.fresh
    retain = isolation.SharedPageTests.retain
    unchanged = isolation.SharedPageTests.unchanged

    @staticmethod
    def scales(audit):
        return {(row['pool'], row['pool_offset']): row for row in audit['regions']
                if row['kind'] in ('k_scale', 'v_scale')}

    def test_g3_005_quantization_scale_isolation(self):
        self.assertEqual(self.profile['execution']['kv_dtype'], 'INT8',
                         'This case requires a pinned INT8-KV profile with actual scale pages')
        for name in ('sut_command', 'oracle_command'):
            command = self.profile['execution'][name]
            self.assertIn('--kv', command)
            self.assertEqual(command[command.index('--kv') + 1], 'int8')
        self.assertTrue(callable(getattr(self.backend, 'audit', None)),
                        'Quantizer metadata must be read from retained physical pages')
        history = tuple(self.profile['fixtures']['prefix'][:5])
        root = self.sut.create(history)
        saved = self.retain(root, self.fresh(root, history))
        original = self.scales(saved['audit'])
        self.assertTrue(original, 'Quantized weights are not evidence of quantized K/V state')
        self.assertEqual({row['kind'] for row in original.values()}, {'k_scale', 'v_scale'})
        kv_ordinals = {row['qsa_ordinal'] for row in saved['audit']['regions']
                       if row['kind'] in ('k', 'v')}
        for kind in ('k_scale', 'v_scale'):
            self.assertEqual({row['qsa_ordinal'] for row in original.values() if row['kind'] == kind},
                             kv_ordinals)
        self.assertGreater(len(kv_ordinals), 0)
        self.assertTrue(all(row['pool'] >= 0 and row['driver_handle'] and
                            row['bytes_per_cell'] > 0 and row['bytes'] > 0
                            for row in original.values()))
        self.record(kind='quantized-layout', kv_dtype=self.profile['execution']['kv_dtype'],
                    model_weights_sha256=self.profile['model_weights_sha256'],
                    execution_profile_id=self.profile['execution_profile_id'],
                    scale_pages=list(original.values()),
                    native_header='include/strata/kernels/kv_q8.hpp: FP16 scale per 64 values; no outlier buffer')
        tips = [root, root]
        histories = [history, history]
        suffixes = [tuple(self.profile['fixtures']['suffix_a']),
                    tuple(self.profile['fixtures']['suffix_b'])]
        self.assertNotEqual(suffixes[0][0], suffixes[1][0])
        retained = [saved]
        for step in range(3):
            children = []
            for branch in (0, 1):
                parent = self.backend.audit(self.sut.cache[tips[branch]].blob)
                before = self.backend._refresh_stats()
                token = suffixes[branch][step]
                tips[branch] = self.sut.feed(tips[branch], (token,))
                histories[branch] += (token,)
                after = self.backend._refresh_stats()
                child = self.backend.audit(self.sut.cache[tips[branch]].blob)
                parent_scales, child_scales = self.scales(parent), self.scales(child)
                self.assertEqual(parent_scales.keys(), child_scales.keys())
                logical = parent['qsa_page_cells']
                tail_start = (len(histories[branch]) - 1) // logical * logical
                affected = {key for key, row in parent_scales.items()
                            if row['pool_offset'] < (tail_start + logical) * row['bytes_per_cell']
                            and row['pool_offset'] + row['bytes'] > tail_start * row['bytes_per_cell']}
                self.assertTrue(affected)
                changed = {key for key in parent_scales
                           if child_scales[key]['allocation_id'] != parent_scales[key]['allocation_id']}
                self.record(kind='quantizer-scale-work', branch=branch, step=step,
                            before=before, after=after, parent_scales=list(parent_scales.values()),
                            child_scales=list(child_scales.values()),
                            expected_private_scale_pages=sorted(affected),
                            observed_private_scale_pages=sorted(changed))
                self.assertEqual(changed, affected, 'A sibling must own its writable quantizer scales')
                self.assertEqual(after['forward_tokens'] - before['forward_tokens'], 1)
                self.assertTrue(any(child_scales[key]['sha256'] != parent_scales[key]['sha256']
                                    for key in affected), 'The fixture must actually change scale bytes')
                self.unchanged(retained)
                current = self.retain(tips[branch], self.fresh(tips[branch], histories[branch]))
                retained.append(current)
                children.append(self.scales(current['audit']))
            self.assertTrue(any(children[0][key]['sha256'] != children[1][key]['sha256']
                                for key in children[0]), 'Different suffixes must exercise different scales')
        self.unchanged(retained)
        self.fresh(root, history)


if __name__ == '__main__':
    unittest.main()
