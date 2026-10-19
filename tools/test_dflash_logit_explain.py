import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import struct
import tempfile
import unittest
import numpy as np
from dflash_logit_explain import explain,records,philox


class Explain(unittest.TestCase):
    def test_near_tie_requires_actual_swapped_leaders(self):
        self.assertTrue(explain([5,4.999,0],[4.998,5,0],0,1)['documented_near_tie'])
        self.assertFalse(explain([5,3,0],[3,5,0],0,1)['documented_near_tie'])
        self.assertFalse(explain([5,4.999,0],[5,4.998,0],0,1)['documented_near_tie'])
        self.assertFalse(explain([5,4.999,9],[4.998,5,9],0,1)['documented_near_tie'])

    def test_sampled_adjacent_near_tie_requires_replayed_picks(self):
        a,b=[2,1.001,1,0],[2,1,1.001,0]
        r=explain(a,b,2,1,'sampled',49)
        self.assertTrue(r['documented_near_tie'])
        self.assertTrue(r['adjacent_order_swap'])
        self.assertFalse(explain(a,b,1,2,'sampled',49)['documented_near_tie'])
        self.assertFalse(explain([5,1,0],[5,0,1],1,2,'sampled',49)['documented_near_tie'])

    def test_fixture_requires_complete_finite_row(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'row.bin'
            h=struct.pack('<6q',3,10,9,1,2,0)
            f.write_bytes(h+np.array([1,0,-1],dtype='<f4').tobytes())
            self.assertEqual(list(records(f))[0][0]['index'],2)
            for raw in (h[:20],h+b'\0',h+np.array([1,np.nan,0],dtype='<f4').tobytes()):
                f.write_bytes(raw)
                with self.assertRaises(ValueError):list(records(f))



if __name__=='__main__':unittest.main()
