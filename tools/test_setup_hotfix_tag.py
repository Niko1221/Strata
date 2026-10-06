"""A four-part hotfix version (0.1.40.1) cannot break setup: versions are read as their first three numbers, the
engine zips are looked for under the checkout's own v<project version> tag first and the latest release second, and
a binary that says `engine=0.1.40.1` is read as 0.1.40.

    python -m unittest tools.test_setup_hotfix_tag
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class HotfixTag(unittest.TestCase):
    def test_a_four_part_version_is_read_as_its_first_three_numbers(self):
        ver = tuple(int(x) for x in "0.1.40.1".split(".")[:3] if x.isdigit())
        self.assertEqual(ver, (0, 1, 40))
        self.assertGreaterEqual(ver, setup.MIN_ENGINE)

    def test_the_engine_zips_are_found_for_a_hotfix_tag(self):
        # a hotfix that keeps CMakeLists.txt at the engine's version: the v0.1.40 release still holds the zips
        bases = setup.prebuilt_bases(setup.PREBUILT_URL)
        self.assertEqual(bases[0], setup.PREBUILT_TAG_URL.format(version=setup.source_version()))
        self.assertEqual(bases[-1], setup.PREBUILT_URL)
        # and if a four-part project version ever exists, its tag is tried first, the latest release is the fallback
        with mock.patch.object(setup, "source_version", return_value="0.1.40.1"):
            self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL),
                             ["https://github.com/Niko1221/Strata/releases/download/v0.1.40.1/", setup.PREBUILT_URL])

    def test_an_engine_that_says_four_parts_is_read_as_three(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"MZ\x00engine=0.1.40.1\n\x00")
            self.assertEqual(setup.engine_version(exe), (0, 1, 40))
            exe.write_bytes(b"MZ\x00engine=0.1.40\n\x00")
            self.assertEqual(setup.engine_version(exe), (0, 1, 40))


if __name__ == "__main__":
    unittest.main()
