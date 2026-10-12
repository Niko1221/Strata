"""Bounded real state-machine check; not the registered 1000-sequence run."""
import importlib.util
import json
import os
from pathlib import Path
import unittest


class RealProgramTests(unittest.TestCase):
    def test_real_operation_interpreter(self):
        path = Path(__file__).resolve().parents[2]/'scripts/state_machine.py'
        spec = importlib.util.spec_from_file_location('state_machine', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        run = getattr(module, 'run_program', None)
        self.assertTrue(callable(run), 'Real state-machine operation interpreter required')
        profile = json.loads(Path(os.environ['CTT_PROFILE']).read_bytes())
        token = profile['fixtures']['suffix_a'][0]
        program = [dict(op=op, source='root', out='s'+str(i), token=token, seed=i)
                   for i, op in enumerate(module.OPERATIONS)]
        directory = Path(os.environ['CTT_STATE_MACHINE_OUTPUT'])
        self.assertFalse(directory.exists(), 'Use a new retained evidence directory for each run')
        report = run(profile, directory, program, seed=17)
        self.assertEqual(report['status'], 'PASS', report.get('failure'))
        self.assertEqual(report['completed_operations'], len(program))
        self.assertGreaterEqual(report['oracle_requests'], len(program)+1)
        self.assertEqual(set(report['executed_operations']), set(module.OPERATIONS))


if __name__ == '__main__': unittest.main(verbosity=2)
