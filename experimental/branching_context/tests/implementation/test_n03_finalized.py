"""Finalization binds executed bytes and rejects unknown operator descriptions."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


class CompletedCalibrationTests(unittest.TestCase):
    def test_completed_calibration_profile_binding(self):
        scripts = Path(__file__).resolve().parents[2]/'scripts'
        sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location('profile_builder', scripts/'build_strata_profile.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        profile_path = Path(os.environ['CTT_PROFILE'])
        baseline_path = Path(os.environ['CTT_CALIBRATION_PROGRESS'])
        profile = json.loads(profile_path.read_bytes())
        baseline = json.loads(baseline_path.read_bytes())
        self.assertEqual(baseline['status'], 'FROZEN')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'profile.json'
            result = module.finalize_existing_profile(profile_path, baseline_path, output)
            self.assertEqual(result['execution_profile_id'], profile['execution_profile_id'])
            self.assertEqual(result['model_contract_id'], profile['model_contract_id'])
            self.assertEqual(result['native_build'], profile['native_build'])
            self.assertEqual(result['execution'], profile['execution'])
            self.assertEqual(result['atol'], baseline['atol'])
            self.assertFalse(result['numerical_gate_qualified'])
            invalid = Path(directory)/'unknown.json'
            rejected = Path(directory)/'rejected.json'
            baseline['operator'] = 'batched or unknown operator'
            invalid.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, 'operator mismatch'):
                module.finalize_existing_profile(profile_path, invalid, rejected)
            self.assertFalse(rejected.exists())


if __name__ == '__main__': unittest.main(verbosity=2)
