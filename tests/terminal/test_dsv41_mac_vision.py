"""DeepSeek's pictures on a Mac are decided by the installed engine, not by a switch: strata_control/dsv41_engine.mac_vision_state looks for the text
STRATA_DS4_METAL_VISION_V1 in the strata-dsv41 program (it is only read, never started), the way drafting.engine_reads_selector_q4 reads its mark.
setup.py asks after the engine is known; the model lists and the install tool ask the same function."""

import tempfile
import unittest
from pathlib import Path

from strata_control import dsv41_engine as E

ROOT = Path(__file__).resolve().parents[2]


class Marker(unittest.TestCase):
    def prog(self, tmp, body: bytes, name="strata-dsv41") -> Path:
        f = Path(tmp) / name
        f.write_bytes(body)
        return f

    def test_present_absent_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            yes = self.prog(tmp, b"\x7fELF" + b"\x00" * 100 + E.MARK_MAC_VISION + b"\x00" * 100, "with")
            no = self.prog(tmp, b"\x7fELF" + b"\x00" * 100 + b"STRATA_DS4_METAL_VISION_V0" + b"\x00" * 100, "without")
            self.assertTrue(E.reads_mac_vision(yes))
            self.assertFalse(E.reads_mac_vision(no))
            self.assertEqual([E.mac_vision_state(yes), E.mac_vision_state(no)], ["yes", "no"])
            self.assertEqual([E.mac_vision_state(None), E.mac_vision_state(Path(tmp) / "nothing")], ["unknown", "unknown"])
            self.assertFalse(E.reads_mac_vision(Path(tmp) / "nothing"))

    def test_the_answer_follows_the_file_when_it_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.prog(tmp, b"old engine")
            self.assertEqual(E.mac_vision_state(f), "no")
            f.write_bytes(b"new engine " + E.MARK_MAC_VISION)      # (the size changes, so the cache key does)
            self.assertEqual(E.mac_vision_state(f), "yes")

    def test_the_line_is_the_plain_one(self):
        self.assertEqual(E.MAC_TEXT_LINE, "DeepSeek on a Mac reads text only with this engine (its picture reader needs an engine update).")


class SetupFollowsTheEngine(unittest.TestCase):
    def test_setup_settles_the_pictures_after_the_engine_is_known_and_has_no_static_switch(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertNotIn("DSV41_MAC_VISION", src.replace("# DeepSeek's pictures on a Mac are decided by the installed engine", ""))
        self.assertIn('ds_vision = "off" if a.vision in ("no", "off") else "on"', src)
        i = src.index('ok(f"engine: {exe}")\n    if mac and ds_vision == "on":')
        self.assertIn('_dsv41_engine.mac_vision_state(exe) == "no"', src[i:i + 600])
        self.assertIn("warn(_dsv41_engine.MAC_TEXT_LINE)", src[i:i + 600])
        j = src.index("step(5, f\"downloading {fam['title']} {model}\")", i)        # the next step 5 after the engine line: before any model file is fetched
        self.assertLess(i, j)
        self.assertLess(j - i, 1800)        # (the Mac download-total correction of DL-AUDIT sits in between: it fetches nothing)


if __name__ == "__main__":
    unittest.main()
