import importlib.util
from pathlib import Path
import unittest

class SpeedReportTests(unittest.TestCase):
    def test_rejects_missing_reordered_and_different_outputs(self):
        path = Path(__file__).resolve().parents[2] / 'scripts' / 'serial_speed.py'
        spec = importlib.util.spec_from_file_location('serial_speed', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = dict(name='A', prompt=[1,2], ids=[3], seconds=2.0)
        other = dict(name='B', prompt=[1,4], ids=[5], seconds=3.0)
        self.assertTrue(module.compare([row,other],[row,other])['tokens_equal'])
        for bad in ([row], [other,row], [dict(row,ids=[9]),other], [dict(row,prompt=[8]),other]):
            self.assertFalse(module.compare([row,other],bad)['tokens_equal'])
        self.assertIsNone(module.compare([row],[dict(row,ids=[9])])['observed_latency_ratio'])
        self.assertEqual(module.compare([row],[dict(row,seconds=1.0)])['observed_latency_ratio'],2.0)

if __name__ == '__main__': unittest.main()
