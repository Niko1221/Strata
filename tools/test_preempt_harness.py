"""CPU-only guards against false passes and malformed parity-test configuration."""
import unittest
from tools.prefill_preempt_test import engine_args, state_differences


class HarnessChecks(unittest.TestCase):
    def test_valueless_preempt_flag_preserves_following_model_option(self):
        args = engine_args({'args': ['--prefill-preempt', '--pack', '/model', '--spec', '4']},
                           prefill=2048, preempt=False)
        self.assertEqual(args[:4], ['--pack', '/model', '--spec', '4'])
        self.assertNotIn('--prefill-preempt', args)

    def test_controlled_options_are_not_duplicated(self):
        args = engine_args({'args': ['--adapt-every', '4', '--suffix-draft', '1', '--spec-min-p', '.5']},
                           prefill=2048, preempt=True)
        for flag, value in [('--adapt-every', '100000'), ('--suffix-draft', '0'), ('--spec-min-p', '0')]:
            self.assertEqual(args.count(flag), 1)
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--prefill-preempt', args)

    def test_mtp_and_target_mismatch_fail(self):
        self.assertEqual(state_differences({'mtp': 'a', 'gdn': 'a'}, {'mtp': 'b', 'gdn': 'b'}),
                         ['mtp', 'gdn'])

    def test_uncommitted_stale_cells_are_not_semantic_state(self):
        self.assertEqual(state_differences({'kv': 'a', 'stale': 'a'}, {'kv': 'a', 'stale': 'b'}), [])

    def test_missing_state_cannot_pass(self):
        self.assertEqual(state_differences({'gdn': 'a'}, {}), ['gdn'])


if __name__ == '__main__':
    unittest.main()
