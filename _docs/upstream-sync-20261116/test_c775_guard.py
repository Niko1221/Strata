"""Synthetic tests of the explicit operation-only RAM exception; no child launch."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('c775_common', ROOT / '_docs/local-053-20261109/native/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
GIB = 1024**3


class OperationRamException(unittest.TestCase):
    def values(self, **changes):
        return dict(available_ram=1*GIB, commit_headroom=8*GIB, H_free=20*GIB, C_free=10*GIB, **changes)

    def test_default_refuses_and_explicit_exception_only_admits_ram(self):
        values = self.values()
        with patch.object(common, 'resources', return_value=values):
            with self.assertRaisesRegex(RuntimeError, 'available_ram below 3221225472'):
                common.admit(ROOT)
            self.assertEqual(common.admit(ROOT, allow_low_ram=True), values)
            for key in ('commit_headroom', 'H_free', 'C_free'):
                with self.subTest(key=key), patch.object(common, 'resources', return_value={**values, key: 0}):
                    with self.assertRaisesRegex(RuntimeError, key+' below '):
                        common.admit(ROOT, allow_low_ram=True)
            with self.assertRaisesRegex(RuntimeError, 'Explicit boolean RAM exception required'):
                common.admit(ROOT, allow_low_ram=1)

    def test_owned_preserves_budget_and_records_nominal_breaches(self):
        def run(command, base, env, log, budget, sample, **options):
            self.assertEqual(budget.min_available_ram, 0)
            self.assertEqual(budget.min_commit_headroom, 4*GIB)
            self.assertEqual(budget.min_H_free, 10*GIB)
            self.assertIsNone(budget.job_commit_bytes)
            self.assertTrue(options['memory_diagnostics'])
            sample(None)
            return dict(status='FINISHED', native_exit_code=0, contained_before_resume=True,
                        termination_confirmed=True, job_active_after=0, observed_dropped_bytes=0)
        with tempfile.TemporaryDirectory() as temp, patch.object(common, 'BASE', Path(temp)), \
             patch.object(common, 'resources', return_value=self.values()), \
             patch.object(common, 'load_helpers', return_value=(None, run)):
            (Path(temp)/'receipts').mkdir()
            result = common.owned(ROOT, 'synthetic', ['fixture'], 30, env={'fixture': 'yes'}, allow_low_ram=True)
        self.assertTrue(result['passed'])
        self.assertEqual(result['ramPolicy']['samples'], 2)
        self.assertEqual(result['ramPolicy']['nominalBreachSamples'], 2)
        self.assertEqual(result['ramPolicy']['minAvailableBytes'], GIB)


if __name__ == '__main__':
    unittest.main()
