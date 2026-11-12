"""Unsloth's UD-Q2_K_XL DeepSeek-V4.1-Flash set: pins, its own Engram (shard 7, Q5_K), the app's file recognition, the fit numbers, the pack call
and the Mac / DGX Spark gate.  No GPU, no downloads.
Run: python -m pytest tools/test_setup_udq2.py"""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("setup_udq2", ROOT / "setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

from strata_control import dsv41_fit, dsv41_sets  # noqa: E402

MODEL = "UD-Q2_K_XL"
# the Hub's sizes and LFS SHA-256 at the pinned commit 8a6ec9cf (the same at main, 2688c0c3)
SHARDS = {1: (5_780_868, "6b6e9ace54298156974cc3e024e5ba86d50c40c06482da8f64fece77d7f73452"),
          2: (49_290_279_808, "fa56e7cb79f88d992170bbc41b192991bd89b9534c53106e328fc03a5d26afed"),
          3: (49_745_070_208, "b279f9bbb331460d182658a9ec6a49eb232ef7d8c69600a2c53d5d797bf037b4"),
          4: (48_680_128_800, "c7fed7b94a11aeafbf07e164940718fab26e99a5f1d9986af0602c6515d18bff"),
          5: (49_970_937_696, "c8d6bd8063607e788db7daed01487f4e87a9529d5756dd561481bed78fea7d6d"),
          6: (1_929_325_664, "2f62bf418b0f722254728bdd88b4b1c88c1b64a4d664c7b696c08fecd49abd76"),
          7: (135_172_021_792, "a653c5309b7c74a6c41580aaa8c2d2bbdae0342253bca2cd3fc9e69d05f3d0eb")}
NAME = "DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-{:05d}-of-00007.gguf"


def md():
    return setup.MULTI_FILE_FAMILIES["deepseek41"]["models"][MODEL]


class Pins(unittest.TestCase):
    def test_every_file_is_pinned_at_the_repositorys_commit(self):
        self.assertEqual(setup.HF_REVISIONS["unsloth/DeepSeek-V4.1-Flash-1017-GGUF"], "8a6ec9cf118e785a362ea94ee123c61c70553be6")
        self.assertEqual(md()["shards"], [NAME.format(i) for i in range(1, 8)])
        for i, pin in SHARDS.items():
            self.assertEqual(setup.DSV41_FILES[NAME.format(i)], pin)
            self.assertEqual(setup.DSV41_ALL_FILES[NAME.format(i)], pin)
        self.assertIs(setup.REPO_PINS["unsloth/DeepSeek-V4.1-Flash-1017-GGUF"], setup.DSV41_FILES)
        self.assertTrue(md()["hf"].endswith("/resolve/8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-Q2_K_XL/"))

    def test_its_engram_is_its_own_last_shard_and_only_the_drafter_is_shared(self):
        self.assertEqual((md()["recipe"], md()["own_engram"]), ("udlow", True))
        self.assertEqual(md()["engram_shard"], md()["shards"][-1])
        self.assertEqual(md()["shared"], [setup.DSV41_DSPARK])
        self.assertNotIn(setup.DSV41_ISTA_ENGRAM, md()["shared"])

    def test_the_other_sets_are_unchanged(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]
        for m in ("UD-IQ1_S", "UD-IQ1_M"):
            self.assertEqual(fam[m]["shared"], [setup.DSV41_ISTA_ENGRAM, setup.DSV41_DSPARK])
            self.assertFalse(fam[m].get("own_engram"))
        self.assertEqual(fam["IST-DAS-2.25bpw"]["shared"], [setup.DSV41_ISTA_ENGRAM, setup.DSV41_SHARD.format(i=1), setup.DSV41_DSPARK])


class PackCall(unittest.TestCase):
    def test_shards_1_to_6_the_seventh_as_engram_the_first_as_metadata(self):
        folder = Path("/m/UD-Q2_K_XL")
        shards = [folder / n for n in md()["shards"]]
        cmd = setup.dsv41_set_pack_cmd(md(), folder, shards, [Path("/s/dspark.gguf")], Path("/p"))
        self.assertEqual(cmd[cmd.index("--shards") + 1:cmd.index("--engram")], [NAME.format(i) for i in range(1, 7)])
        self.assertEqual(cmd[cmd.index("--engram") + 1], str(shards[6]))
        self.assertEqual(cmd[cmd.index("--engram-meta") + 1], str(shards[0]))
        self.assertEqual(cmd[cmd.index("--dspark-gguf") + 1], str(Path("/s/dspark.gguf")))

    def test_the_other_recipes_pack_as_before(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]
        f = Path("/m")
        ud = [f / n for n in fam["UD-IQ1_M"]["shards"]]
        self.assertEqual(setup.dsv41_set_pack_cmd(fam["UD-IQ1_M"], f, ud, [Path("/e"), Path("/d")], Path("/p")),
                         setup.dsv41_pack_cmd(f, [s.name for s in ud], Path("/e"), ud[0], Path("/d"), Path("/p")))
        ist = [f / n for n in fam["IST-DAS-2.25bpw"]["shards"]]
        self.assertEqual(setup.dsv41_set_pack_cmd(fam["IST-DAS-2.25bpw"], f, ist, [Path("/e"), Path("/m1"), Path("/d")], Path("/p"), Path("/v")),
                         setup.dsv41_pack_cmd(f, [s.name for s in ist], Path("/e"), Path("/m1"), Path("/d"), Path("/p"), Path("/v")))


class Recognised(unittest.TestCase):
    def files(self):
        return {NAME.format(i): pin[0] for i, pin in SHARDS.items()}

    def test_the_seven_files_are_the_whole_set_with_their_engram(self):
        row = [r for r in dsv41_sets.match(self.files()) if r["model"] == MODEL][0]
        self.assertTrue(row["complete"])
        self.assertEqual(row["engram"], NAME.format(7))
        self.assertEqual(row["bytes"], sum(p[0] for p in SHARDS.values()))     # (the Engram counted once)

    def test_an_ista_engram_does_not_stand_in_for_shard_7(self):
        files = self.files()
        del files[NAME.format(7)]
        files[setup.DSV41_ISTA_ENGRAM] = setup.DSV41_ISTA_FILES[setup.DSV41_ISTA_ENGRAM][0]
        row = [r for r in dsv41_sets.match(files) if r["model"] == MODEL][0]
        self.assertFalse(row["complete"])
        self.assertEqual(row["missing"], [NAME.format(7)])
        self.assertIsNone(row["engram"])

    def test_the_ud_iq1_sets_still_take_an_ista_engram(self):
        t = dsv41_sets.tables()
        files = dict(t["sets"]["UD-IQ1_M"]["shards"])
        files[setup.DSV41_ISTA_ENGRAM] = setup.DSV41_ISTA_FILES[setup.DSV41_ISTA_ENGRAM][0]
        row = dsv41_sets.match(files)[0]
        self.assertEqual((row["model"], row["complete"], row["engram"]), ("UD-IQ1_M", True, setup.DSV41_ISTA_ENGRAM))


class Fit(unittest.TestCase):
    def test_download_counts_its_own_engram_even_with_an_ista_one_here(self):
        whole = (199_621_523_044 + 135_172_021_792 + dsv41_fit.ISTA_DSPARK_BYTES) / 1e9
        self.assertAlmostEqual(dsv41_fit.ista_download_bytes(MODEL) / 1e9, whole, places=2)
        self.assertAlmostEqual(dsv41_fit.ista_download_bytes(MODEL, have_engram=True) / 1e9, whole, places=2)
        self.assertEqual(sum(p[0] for i, p in SHARDS.items() if i < 7), dsv41_fit.ISTA_SETS[MODEL]["set_bytes"])
        self.assertEqual(SHARDS[7][0], dsv41_fit.engram_bytes(MODEL))
        self.assertEqual(dsv41_fit.engram_bytes("IST-DAS-2.25bpw"), dsv41_fit.ISTA_ENGRAM_BYTES)

    def test_the_plan_names_its_own_table(self):
        p = dsv41_fit.plan([{"name": "R9700", "vram_gb": 32.0, "vendor": "amd"}], 251.0, None, 131072, model=MODEL)
        self.assertEqual(p["engram_gb"], 135.2)
        self.assertIn("135 GB n-gram table", p["lines"][0])
        self.assertIn("has its own table", " ".join(p["lines"]))
        self.assertEqual(p["verdict"], "fits")

    def test_held_back_on_a_mac_and_a_dgx_spark(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertIn("UD-Q2_K_XL set doesn't run on a Mac yet", dsv41_fit.not_here(MODEL))
            self.assertIsNone(dsv41_fit.not_here("UD-IQ1_M"))
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(dsv41_fit.sysinfo, "machine", return_value="aarch64"):
            self.assertIn("UD-Q2_K_XL set was not tested on a DGX Spark", dsv41_fit.not_here(MODEL))
            self.assertIsNone(dsv41_fit.not_here("UD-IQ1_S"))
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(dsv41_fit.sysinfo, "machine", return_value="x86_64"):
            self.assertIsNone(dsv41_fit.not_here(MODEL))


if __name__ == "__main__":
    unittest.main()
