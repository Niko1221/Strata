"""Independent in-memory logical oracle for generated state-machine programs."""
import importlib.util
from pathlib import Path
import unittest


class LogicalOracleTests(unittest.TestCase):
    def test_reference_graph_and_receipts(self):
        path = Path(__file__).resolve().parents[2]/'scripts/state_machine.py'
        spec = importlib.util.spec_from_file_location('state_machine', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        model_type = getattr(module, 'LogicalHistory', None)
        self.assertTrue(callable(model_type), 'Independent logical graph oracle required')
        model = model_type()
        model.add('p', None, [1])
        model.add('a', 'p', [2])
        model.add('b', 'p', [3])
        model.add('c', 'a', [4])
        self.assertEqual(model.history('a'), (1, 2))
        self.assertEqual(model.history('b'), (1, 3))
        self.assertEqual(model.history('c'), (1, 2, 4))
        with self.assertRaises(ValueError): model.add('a', 'b', [2])
        model.fork('main', 'p')
        self.assertEqual(model.move('main', 0, 'a', 'out'), ('a', 1))
        self.assertEqual(model.move('main', 1, 'p', 'back'), ('p', 2))
        self.assertEqual(model.move('main', 0, 'a', 'out'), ('a', 1))
        with self.assertRaises(ValueError): model.move('main', 0, 'b', 'out')
        with self.assertRaises(ValueError): model.move('main', 0, 'b', 'stale')
        model.release('a'); model.release('b'); model.release('c')
        model.protect('evaluation', 'c')
        self.assertEqual(model.collect('p'), {'b'})
        self.assertEqual(model.history('c'), (1, 2, 4))
        with self.assertRaises(KeyError): model.history('b')
        model.unprotect('evaluation')
        self.assertEqual(model.collect('p'), {'c'})
        self.assertEqual(model.history('a'), (1, 2))  # committed retry receipt
        self.assertEqual(model.collect('p'), set())
        model.add('q', None, [8])
        with self.assertRaises(ValueError): model.move('main', 2, 'q', 'cross-root')
        model.release('q')
        self.assertEqual(model.collect('p'), set())
        self.assertEqual(model.history('q'), (8,))


if __name__ == '__main__': unittest.main(verbosity=2)
