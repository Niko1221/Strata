"""Model-free provenance checks; these do not qualify any real-backend gate."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from calibration_inputs import execution_settings


class CalibrationInputTests(unittest.TestCase):
    def receipt(self):
        arguments = ['--pack', '/pack', '--native', '/model.gguf', '--mmap-experts',
                     '--max-new', '1', '--max-context', '4096', '--pool-workers', '2',
                     '--spec', '2', '--prefill', '128', '--suffix-draft', '0',
                     '--expert-profile', '/experts.bin', '--expert-cache', '512']
        return dict(status='PASS', expert_cache_slots=512,
                    sut_command=['/sut', '--tokens', '1'] + arguments,
                    oracle_command=['/fresh'] + arguments)

    def test_verified_default_format_is_fp16(self):
        before = self.receipt()
        receipt = copy.deepcopy(before)
        self.assertEqual(execution_settings(receipt),
                         dict(kv_format='FP16', capacity=4096, expert_cache_slots=512))
        self.assertEqual(receipt, before)

    def test_format_comes_from_both_executed_commands(self):
        for flag, label in [('int8', 'INT8'), ('q4_0', 'Q4_0'), ('k8v4', 'K8V4')]:
            with self.subTest(flag=flag):
                receipt = self.receipt()
                for key in ('sut_command', 'oracle_command'):
                    receipt[key] += ['--kv', flag]
                self.assertEqual(execution_settings(receipt)['kv_format'], label)

    def test_different_model_layout_or_work_settings_refused(self):
        for option, value in [('--native', '/other.gguf'), ('--pack', '/other'),
                              ('--max-context', '8192'), ('--expert-cache', '256'),
                              ('--pool-workers', '8'), ('--spec', '4')]:
            with self.subTest(option=option):
                receipt = self.receipt()
                command = receipt['oracle_command']
                command[command.index(option) + 1] = value
                with self.assertRaisesRegex(ValueError, 'differ'):
                    execution_settings(receipt)
        receipt = self.receipt()
        receipt['oracle_command'] += ['--kv', 'int8']
        with self.assertRaisesRegex(ValueError, 'differ'):
            execution_settings(receipt)

    def test_duplicate_option_cannot_hide_the_effective_format(self):
        receipt = self.receipt()
        for key in ('sut_command', 'oracle_command'):
            receipt[key] += ['--kv', 'int8', '--kv', 'fp16']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            execution_settings(receipt)

    def test_unmeasured_admission_and_invented_cache_count_refused(self):
        for status in ('FAIL', 'UNMEASURED', 'SKIPPED'):
            receipt = self.receipt()
            receipt['status'] = status
            with self.assertRaisesRegex(ValueError, 'PASS'):
                execution_settings(receipt)
        receipt = self.receipt()
        receipt['expert_cache_slots'] = 1024
        with self.assertRaisesRegex(ValueError, 'cache'):
            execution_settings(receipt)

    def test_unknown_format_and_missing_arguments_refused(self):
        receipt = self.receipt()
        for key in ('sut_command', 'oracle_command'):
            receipt[key] += ['--kv', 'unknown']
        with self.assertRaisesRegex(ValueError, 'format'):
            execution_settings(receipt)
        receipt = self.receipt()
        receipt['oracle_command'] += ['--kv']
        with self.assertRaisesRegex(ValueError, 'value'):
            execution_settings(receipt)


if __name__ == '__main__':
    unittest.main()
