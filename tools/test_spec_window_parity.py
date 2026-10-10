"""Test the gate against protocol failures, missing coverage and token divergence, without a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_window_parity import audit, difference, run, validate


class ParityTests(unittest.TestCase):
    def row(self, ids, case='serial', **extra):
        return dict(case=case, prompt='code', output_ids=ids, finish='length', generated=len(ids),
                    drafts=4, speculative=0, rollbacks=0, reused=0, **extra)

    def test_first_difference_including_length(self):
        self.assertEqual(difference([1, 2], [1, 3]), 1)
        self.assertEqual(difference([1, 2], [1]), 1)
        self.assertEqual(difference([1], [1, 2]), 1)
        self.assertIsNone(difference([1], [1]))

    def test_finish_reason_divergence(self):
        a, b = self.row([1]), self.row([1], 'pipeline')
        b['finish'] = 'eos'
        self.assertFalse(audit([a, b], 2, {})['passed'])

    def test_missing_rows_and_zero_tokens_fail(self):
        self.assertFalse(audit([], 2, {})['passed'])
        self.assertFalse(audit([self.row([])], 1, {})['passed'])

    def test_repeated_reference_must_be_stable(self):
        result = audit([self.row([1]), self.row([2])], 2, {})
        self.assertFalse(result['passed'])
        self.assertEqual(result['failures'], ['serial/code: token difference 0, finish length'])

    def test_different_inputs_cannot_establish_parity(self):
        a, b = self.row([1]), self.row([1], 'pipeline')
        a['input_sha256'], b['input_sha256'] = 'a', 'b'
        self.assertFalse(audit([a, b], 2, {})['passed'])

    def test_missing_rollback_cannot_pass_on_equal_tokens(self):
        result = audit([self.row([1], 'forced')], 1, {'forced': {'rollbacks': 1}})
        self.assertFalse(result['passed'])

    def test_missing_suffix_coverage_fails(self):
        self.assertFalse(audit([self.row([1], 'suffix')], 1,
                               {'suffix': {'suffix_windows': 1}})['passed'])

    def test_coverage_is_counted_across_prompts_and_repeats(self):
        rows = [self.row([1], 'forced'), self.row([1], 'forced')]
        for row in rows:
            row.update(speculative=2, rollbacks=1, suffix_windows=3)
        self.assertTrue(audit(rows, 2, {'forced': dict(rollbacks=2, suffix_windows=6)})['passed'])
        self.assertFalse(audit(rows, 2, {'forced': dict(rollbacks=3)})['passed'])

    def test_reuse_and_count_mismatch_fail(self):
        row = self.row([1])
        row.update(reused=10, generated=2)
        self.assertEqual(len(audit([row], 1, {})['failures']), 2)

    def test_protocol_run_and_real_log_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'engine.py'
            fake.write_text('''import sys
print('READY', flush=True)
for line in sys.stdin:
    if not line.startswith('GEN '): continue
    print('strata pipeline: 10 windows in 20 ms (2 ms/window): 3 speculative, 2 on the path, 1 rolled back, 0 below the gate (theta 0.00)', file=sys.stderr, flush=True)
    print('T 42', flush=True)
    print('DONE 1 2 1.0 2.0 length 3 4 0', flush=True)
''', encoding='utf-8')
            manifest = dict(prompts=[dict(name='code', ids=[1, 2])], repeats=2, max_new=1,
                            arms=[dict(name='mock', exe=sys.executable, prefix_args=[str(fake)], args=[],
                                       cases=[dict(name='serial'), dict(name='forced', require=dict(rollbacks=1))])])
            result = run(manifest, root / 'result', timeout=5)
            self.assertTrue(result['passed'], result)
            self.assertEqual(result['coverage']['forced']['rollbacks'], 2)
            self.assertEqual(result['rows'][0]['output_ids'], [42])
            saved = json.loads((root / 'result/result.json').read_text())
            self.assertTrue(saved['complete'])
            # Plausible pipeline telemetry cannot excuse an ignored requested switch.
            manifest['arms'][0]['cases'][1]['switch'] = dict(pw=2, force_miss=1)
            ignored = run(manifest, root / 'ignored-switch', timeout=5)
            self.assertFalse(ignored['passed'])
            self.assertIn('did not confirm', ignored['error'])
            # A process that exits at startup leaves an explicit failed report.
            fake.write_text('raise SystemExit(2)', encoding='utf-8')
            result = run(manifest, root / 'failure', timeout=5)
            self.assertFalse(result['passed'])
            self.assertFalse(result['complete'])

    def test_reject_sampled_keys_and_duplicate_cases(self):
        manifest = dict(prompts=[dict(name='code', ids=[1])],
                        arms=[dict(cases=[dict(name='serial', keys=dict(temperature=0.7))])])
        with self.assertRaises(ValueError):
            validate(manifest)
        manifest['arms'][0]['cases'] = [dict(name='serial'), dict(name='serial')]
        with self.assertRaises(ValueError):
            validate(manifest)


if __name__ == '__main__':
    unittest.main()
