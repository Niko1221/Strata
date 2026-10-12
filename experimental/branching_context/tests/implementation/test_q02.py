"""Deterministic programs and failure-preserving reduction, without a fake GPU."""
import importlib.util
from pathlib import Path
import unittest


class StateMachineHarnessTests(unittest.TestCase):
    def test_q02_invariant(self):
        path = Path(__file__).resolve().parents[2]/'scripts/state_machine.py'
        self.assertTrue(path.is_file(), 'Missing reproducible state-machine runner')
        spec = importlib.util.spec_from_file_location('state_machine', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        programs = [module.make_program(seed, 40, [100, 117, 134]) for seed in range(10)]
        self.assertEqual(programs, [module.make_program(seed, 40, [100, 117, 134]) for seed in range(10)])
        self.assertNotEqual(programs[0], programs[1])
        operations = set()
        for program in programs:
            known = {'root'}
            self.assertEqual(len(program), 40)
            for row in program:
                self.assertIn(row['source'], known)
                self.assertNotIn(row['out'], known)
                self.assertIn(row['token'], [100, 117, 134])
                known.add(row['out'])
                operations.add(row['op'])
        self.assertEqual(operations, {'append', 'fork', 'reseed', 'generate', 'rewind',
                                    'checkpoint', 'evict', 'evaluate', 'retry', 'release', 'gc'})
        for count in (0, 1, 7): self.assertEqual(len(module.make_program(7, count, [100])), count)
        with self.assertRaises(ValueError): module.make_program(7, 2, [])
        with self.assertRaises(ValueError): module.make_program(7, -1, [100])

        program = ['noise', 'create', 'noise', 'corrupt', 'read', 'noise']
        def reproduce(candidate):
            if 'read' in candidate and 'create' not in candidate: return 'invalid-program'
            if all(x in candidate for x in ('create', 'corrupt', 'read')): return 'logit-mismatch'
            return None
        reduced = module.minimize(program, reproduce)
        self.assertEqual(reduced, ['create', 'corrupt', 'read'])
        self.assertEqual(program, ['noise', 'create', 'noise', 'corrupt', 'read', 'noise'])
        for i in range(len(reduced)):
            self.assertNotEqual(reproduce(reduced[:i]+reduced[i+1:]), 'logit-mismatch')
        with self.assertRaises(ValueError): module.minimize(['noise'], reproduce)


if __name__ == '__main__':
    unittest.main(verbosity=2)
