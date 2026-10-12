"""CPU-only guards for the exact-token cache probe."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.conversation_token_parity import engine_args, verify_results


def fixture():
    def records(reused):
        return [
            {'name': 'A-first', 'ids': [1, 2, 3], 'finish': 'length', 'reused': 0},
            {'name': 'B', 'ids': [4], 'finish': 'length', 'reused': 0},
            {'name': 'A-return', 'ids': [1, 2, 3], 'finish': 'length', 'reused': reused},
            {'name': 'A-repeat', 'ids': [1, 2, 3], 'finish': 'length', 'reused': reused},
        ]
    return {'off': records(0), 'ram': records(64)}


class TokenParityGate(unittest.TestCase):
    def test_cache_args_enable_reproducible_policy_and_tier(self):
        cfg = {'args': ['--spec', '4', '--prompt-cache', '2', '--adapt-swaps', '96']}
        args = engine_args(cfg, 'ram-fallback', 8192, Path('/tmp/spill'), 1)
        self.assertIn('--reproducible', args)
        self.assertEqual(args[args.index('--spec') + 1], '2')
        self.assertEqual(args[args.index('--conversation-cache-mib') + 1], '1')
        self.assertEqual(args[args.index('--conversation-cache-disk-mib') + 1], '4096')
        self.assertEqual(cfg['args'], ['--spec', '4', '--prompt-cache', '2', '--adapt-swaps', '96'])

    def test_identical_fresh_and_cached_repeats_pass(self):
        verify_results(fixture(), 'ram')

    def test_token_mismatch_and_cache_miss_fail(self):
        cases = (
            lambda d: d['ram'][2].update(ids=[9, 8, 7]),
            lambda d: d['ram'][3].update(ids=[9, 8, 7]),
            lambda d: d['ram'][2].update(reused=0),
            lambda d: d['off'][2].update(finish='error'),
        )
        for mutate in cases:
            data = copy.deepcopy(fixture())
            mutate(data)
            with self.assertRaises(AssertionError):
                verify_results(data, 'ram')

    def test_dry_run_does_not_open_model_or_create_output(self):
        with tempfile.TemporaryDirectory(prefix='strata-token-parity-') as directory:
            root = Path(directory)
            output = root / 'not-created'
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('conversation_token_parity.py')),
                 '--config', str(root / 'missing.json'), '--engine', str(root / 'missing-engine'),
                 '--output', str(output), '--tier', 'disk'],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('No model loaded', result.stdout)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
