"""Incomplete real measurements must never be promoted into a frozen profile."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


class FinalizationTests(unittest.TestCase):
    def test_n03_incomplete_cannot_freeze(self):
        scripts = Path(__file__).resolve().parents[2]/'scripts'
        sys.path.insert(0,str(scripts))
        spec = importlib.util.spec_from_file_location('profile_builder',scripts/'build_strata_profile.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        finalize = getattr(module,'finalize_existing_profile',None)
        self.assertTrue(callable(finalize),'Finalization must validate executed raw calibration evidence')
        baseline = json.loads(Path(os.environ['CTT_CALIBRATION_PROGRESS']).read_text())
        self.assertNotEqual(baseline['status'],'FROZEN','This negative fixture requires an incomplete run')
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input.json', Path(directory)/'profile.json'
            source.write_text(json.dumps(baseline))
            with self.assertRaises(ValueError):
                finalize(Path(os.environ['CTT_PROFILE']),source,output)
            self.assertFalse(output.exists())
            baseline.update(status='FROZEN',atol=0,rtol=0,baseline_max_absolute_error=0)
            source.write_text(json.dumps(baseline))
            with self.assertRaises(ValueError):
                finalize(Path(os.environ['CTT_PROFILE']),source,output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
