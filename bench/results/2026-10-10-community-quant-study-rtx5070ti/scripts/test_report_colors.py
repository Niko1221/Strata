import unittest
from scorecard_colors import PALETTE,MISSING,cell_style,contrast
class ScorecardColorTests(unittest.TestCase):
 def test_range_endpoints_and_middle_are_distinct(self):
  self.assertEqual(cell_style(94,94,97)[0],PALETTE[0]);self.assertEqual(cell_style(95.5,94,97)[0],PALETTE[2]);self.assertEqual(cell_style(97,94,97)[0],PALETTE[-1])
  self.assertEqual(len(set(PALETTE)),5)
 def test_missing_is_gray_not_weak(self):
  self.assertEqual(cell_style(None,60,100)[0],MISSING);self.assertNotIn(MISSING,PALETTE)
 def test_constant_column_is_neutral(self):
  self.assertEqual(cell_style(100,100,100)[0],PALETTE[2])
 def test_foreground_contrast_at_least_wcag_aa(self):
  for value in [None,0,10,30,50,70,90,100]:
   bg,fg=cell_style(value,0,100);self.assertGreaterEqual(contrast(bg,fg),4.5,(value,bg,fg))
 def test_relative_bins_depend_on_column_not_absolute_score(self):
  self.assertEqual(cell_style(94,94,97)[0],cell_style(60,60,90)[0])
  self.assertEqual(cell_style(97,94,97)[0],cell_style(90,60,90)[0])
  with self.assertRaises(ValueError):cell_style(float('nan'),0,100)
if __name__=='__main__':unittest.main()
