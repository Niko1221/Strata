"""The installer payload carries the calibration helper used by saved tuning. Offline: copies only."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))

import package_terminal as pt  # noqa: E402


class InstallerPayload(unittest.TestCase):
    def test_calibration_helper_is_copied_and_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            staged = Path(d)
            manifest = pt.add_installer(staged, source=ROOT)
            helper = "tools/calibrate.py"
            self.assertTrue((staged / helper).is_file(), helper)
            self.assertEqual((staged / helper).read_bytes(), (ROOT / helper).read_bytes())
            self.assertIn(helper, manifest["files"])


if __name__ == "__main__":
    unittest.main()
