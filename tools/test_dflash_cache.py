"""The GPU cache gate must reject changed expert placement before comparing IDs."""
import queue
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash_cache import validate_residency
from dflash_sampling import Engine


class CacheGate(unittest.TestCase):
    def info(self):
        return dict(expert_slots='2000', expert_slots_primary='2000', context='4096',
                    kv='int8', kv_resident='0', spec='2', pcie_frac='0.00', pool_workers='23')

    def test_fixed_and_matching_placement(self):
        validate_residency(self.info(), 2000, self.info())

    def test_clamped_equal_budgets_are_still_rejected(self):
        info = self.info()
        info.update(expert_slots='1999', expert_slots_primary='1999')
        with self.assertRaisesRegex(AssertionError, 'Reduce --expert-cache'):
            validate_residency(info, 2000, info)

    def test_missing_or_different_configuration_rejected(self):
        for key in self.info():
            with self.subTest(key=key):
                info = self.info()
                info.pop(key)
                with self.assertRaises(AssertionError):
                    validate_residency(info, 2000, self.info())

    def test_native_info_is_retained(self):
        engine = Engine.__new__(Engine)
        engine.info = {}
        engine.lines = queue.Queue()
        engine.lines.put('INFO expert_slots=2000 kv=int8')
        engine.line()
        self.assertEqual(engine.info, dict(expert_slots='2000', kv='int8'))

    def test_dead_child_does_not_mask_original_error_during_close(self):
        engine = Engine.__new__(Engine)
        engine.p = Mock()
        engine.p.poll.return_value = 1
        engine.p.stdin.close.side_effect = BrokenPipeError()
        engine.reader = Mock()
        engine.log = Mock()
        engine.close()
        engine.p.stdout.close.assert_called_once()
        engine.log.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
