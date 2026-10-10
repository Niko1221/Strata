"""tools/test_setup_ornith.py - setup's "ornith" family (Ornith-1.5-35B-A3B): its table, its download, its engine
arguments and its size choice.  No GPU, no download: the harness of test_setup_qwen36.py with the family switched.

    python tools/test_setup_ornith.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_setup_qwen36 as q36  # noqa: E402  (the qwen36 harness: setup imported, main() mocked)

setup = q36.setup
IQ4, IQ3 = "Ornith-1.5-IQ4_XS", "Ornith-1.5-IQ3_XXS"


class Tables(unittest.TestCase):
    def test_family(self):
        fam = setup.FAMILIES["ornith"]
        self.assertEqual(list(setup.FAMILIES)[-1], "ornith")              # last: the menu's other numbers stay
        self.assertTrue(setup.small_family("ornith") and setup.small_family("qwen36"))
        self.assertFalse(setup.small_family("qwen") or setup.small_family("unsloth"))
        for k, v in {"tag": "ornith15-", "own_mtp": True, "ple": False, "one_gpu": True, "vision": False,
                     "profile": "expert-profile-qwen36.bin", "shards": 1}.items():
            self.assertEqual(fam[k], v, k)
        self.assertIn("/bartowski/Ornith-1.5-35B-A3B-GGUF/resolve/64b0493d34a5ca4c1b4ad67bb99b41d74b4f07d6/", fam["hf"])

    def test_sizes(self):
        self.assertEqual(setup.model_key("ornith", "IQ4_XS"), IQ4)          # not Flash-Next's IQ3_XXS / Unsloth's
        self.assertEqual(setup.model_key("ornith", "IQ3_XXS"), IQ3)
        self.assertEqual(setup.model_key("qwen", "IQ3_XXS"), "IQ3_XXS")
        self.assertEqual([m for m in setup.MODELS if "ornith" in setup.MODELS[m].get("families", ())], [IQ4, IQ3])
        for m, gb in ((IQ4, 19278554784), (IQ3, 15340447392)):
            name = setup.model_file(setup.FAMILIES["ornith"], m, 1)
            self.assertEqual(setup.ORNITH_FILES[name][0], gb, name)

    def test_size_by_ram(self):
        self.assertEqual(setup.small_size("ornith", q36.RAM32), IQ4)
        self.assertEqual(setup.small_size("ornith", q36.RAM24), IQ3)
        self.assertEqual(setup.small_size("ornith", q36.RAM16), IQ3)      # the smallest, in the low-RAM mode

    def test_qwen36_stays_the_suggestion(self):
        self.assertTrue(setup.qwen36_recommended(q36.RAM16))
        self.assertEqual(setup.qwen36_size(q36.RAM24), "Qwen3.6-UD-IQ3_S")


class Install(q36.Base):
    def test_iq4_xs(self):
        code, out, cfg = self.main(["--family", "ornith", "--model", "IQ4_XS"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-ornith15-iq4_xs.json")
        native = q36.arg(cfg, "--native")
        self.assertTrue(native.endswith("models/ornith15-IQ4_XS/Ornith-1.5-35B-A3B-IQ4_XS.gguf"), native)
        self.assertEqual(q36.arg(cfg, "--mtp"), native)                   # the draft layer is in the model file
        self.assertTrue(q36.arg(cfg, "--pack").endswith("packs/ornith15-iq4_xs"))
        self.assertTrue(q36.arg(cfg, "--expert-profile").endswith("expert-profile-qwen36.bin"))
        self.assertNotIn("--ple-gguf", cfg["args"])
        self.assertEqual(cfg["model_name"], "ornith-1.5-35b-a3b-iq4_xs")
        self.assertTrue(any("bartowski/Ornith-1.5-35B-A3B-GGUF" in u for u in self.downloads), self.downloads)
        self.assertIn(("Ornith-1.5-35B-A3B-IQ4_XS.gguf", 19278554784,
                       "d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70"), self.verified)



if __name__ == "__main__":
    unittest.main()
