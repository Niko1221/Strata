"""Executed full-vector diagnostics precede any batched equivalence claim."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PrefillDiagnosticTests(unittest.TestCase):
    def test_n01_invariant(self):
        package=Path(__file__).resolve().parents[2]
        script=package/'scripts/measure_prefill_paths.py'
        self.assertTrue(script.is_file(),'Current-build fresh prefill diagnostic must exist')
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,str(script),'--profile',os.environ['CTT_PROFILE'],
                '--normal',os.environ['CTT_NORMAL_EXECUTABLE'],'--historical',os.environ['CTT_REJECTED_BASELINE'],
                '--destination',directory],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(report['status'],'MEASURED')
            self.assertIsNone(report['atol'])
            self.assertEqual(report['historical']['max_absolute_error'],1.5612177848815918)
            self.assertEqual(len(report['forwards']),8)
            self.assertEqual(len(report['comparisons']),6)
            for row in report['forwards']:
                self.assertEqual(row['exit_code'],0)
                self.assertEqual(row['computed_tokens'],row['length'])
                if row['operator']=='normal':
                    self.assertEqual(row['prefill_tokens'],row['length']-1)
                    self.assertEqual(row['verifier_tokens'],1)
                else:
                    self.assertEqual(row['position_max'],row['length']-1)
                raw=(Path(directory)/row['logits_file']).read_bytes()
                import hashlib
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['logits_sha256'])
                self.assertEqual(len(raw),report['vocab']*4)
            for row in report['comparisons']:
                self.assertEqual(row['compared_logits'],report['vocab'])
                self.assertGreaterEqual(row['max_absolute_error'],0)
            # Keep full vectors durably in the harness run, not only summary numbers.
            import shutil
            evidence=Path(os.environ['CTT_SHARING_MEASUREMENTS']).parent/'prefill-paths'
            shutil.copytree(directory,evidence)
            with Path(os.environ['CTT_SHARING_MEASUREMENTS']).open('a') as stream:
                stream.write(json.dumps({'test':self.id(),'kind':'fresh-prefill-paths',
                    'report':report,'full_vectors':str(evidence)})+'\n')


if __name__=='__main__': unittest.main(verbosity=2)
