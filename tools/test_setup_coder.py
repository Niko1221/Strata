"""The ISTA-DAS Coder (DeepSeek-V4.1-Flash with half of each layer's experts removed, 2026-11-01): the one file that names its repository, the pins,
the app's file recognition, the fit numbers, the pack call and the engine's profile size gate.
Run: python -m pytest tools/test_setup_coder.py"""
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("setup_coder", ROOT / "setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

from strata_control import dsv41_fit, dsv41_sets  # noqa: E402

JSON = ROOT / "data" / "dsv41" / "coder-set.json"
PIN = json.loads(JSON.read_text(encoding="utf-8"))
NAME = "ISTA-DAS-Coder"
R9700 = {"index": 0, "name": "Radeon AI PRO R9700", "vram_gb": 32, "vendor": "amd", "arch": "gfx1201"}
FOUR = [{**R9700, "index": i} for i in range(4)]


class OnePlace(unittest.TestCase):
    """The repository id, the commit and the files are in data/dsv41/coder-set.json and nowhere else: a new URL is tools/dsv41/pin_coder.py."""

    def test_the_json_is_complete(self):
        self.assertRegex(PIN["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(PIN["repo"], r"^[\w.-]+/[\w.-]+$")
        self.assertEqual(PIN["experts_per_layer"], 192)
        for n in (*PIN["shards"], PIN["engram_shard"], PIN["mmproj"]):
            size, sha = PIN["files"][n]
            self.assertRegex(sha, r"^[0-9a-f]{64}$", n)
            self.assertGreater(size, 0)
        self.assertEqual(len(PIN["shards"]), 3)          # the metadata shard and two model shards; the fourth file is the Engram

    def test_setup_reads_it(self):
        self.assertEqual(setup.DSV41_CODER_REPO, PIN["repo"])
        self.assertEqual(setup.HF_REVISIONS[PIN["repo"]], PIN["revision"])
        self.assertEqual(setup.REPO_PINS.get(PIN["repo"]) or setup.DSV41_CODER_FILES, setup.DSV41_CODER_FILES)
        self.assertEqual({n: list(v) for n, v in setup.DSV41_CODER_FILES.items()}, PIN["files"])
        md = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"][NAME]
        self.assertEqual(md["hf"], f"https://huggingface.co/{PIN['repo']}/resolve/{PIN['revision']}/")
        self.assertEqual((md["shards"], md["engram_shard"]), (PIN["shards"], PIN["engram_shard"]))

    def test_nothing_else_names_the_repository(self):
        """A grep over the sources the packaged Terminal and the app ship: the id appears in the JSON only (tests and notes may quote it)."""
        hits = []
        for p in list(ROOT.glob("*.py")) + list((ROOT / "strata_control").glob("*.py")) + list((ROOT / "strata_cli").rglob("*.py")) \
                + list((ROOT / "tools").glob("*.py")) + list((ROOT / "serve").glob("*.py")) + list((ROOT / "apps" / "host" / "src").rglob("*.ts*")):
            if p.name.startswith("test_") or p.name == "pin_coder.py":
                continue
            if PIN["repo"] in p.read_text(encoding="utf-8", errors="replace"):
                hits.append(str(p.relative_to(ROOT)))
        self.assertEqual(hits, [])

    def test_the_fit_numbers_come_from_the_same_file(self):
        v = dsv41_fit.ISTA_SETS[NAME]
        self.assertEqual(v["set_bytes"], sum(PIN["files"][n][0] for n in PIN["shards"]))
        self.assertEqual(v["experts_per_layer"], PIN["experts_per_layer"])

    def test_the_short_spelling_is_accepted(self):
        self.assertEqual(setup.model_alias("IST-DAS-Coder"), NAME)
        self.assertEqual(setup.model_alias("ista-das-coder"), NAME)
        self.assertEqual(setup.model_alias("ISTA-DAS-2.25bpw"), "IST-DAS-2.25bpw")      # the others as before


class Pins(unittest.TestCase):
    def test_every_file_is_pinned_and_the_model_is_half_the_experts(self):
        md = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"][NAME]
        self.assertEqual(md["recipe"], "coder")
        for n in (*md["shards"], md["engram_shard"]):
            self.assertIn(n, setup.DSV41_ALL_FILES)
        self.assertEqual(setup.DSV41_ALL_FILES[md["engram_shard"]][0], 110_595_290_592)       # the same size as Unsloth's shard 5
        self.assertEqual(md["shared"], [setup.DSV41_ISTA_ENGRAM, setup.DSV41_DSPARK])      # the Engram and the drafter are the shared files; no second metadata file

    def test_the_picture_file_is_the_shared_one_and_the_coders_copy_is_accepted(self):
        """The repository's mmproj has Unsloth's tensors byte for byte (checked on the tensor data) under another header: one shared file serves every set."""
        pin = setup.DSV41_CODER_FILES[setup.DSV41_CODER["mmproj"]]
        self.assertNotEqual(pin, setup.DSV41_ALL_FILES[setup.DSV41_MMPROJ])
        self.assertEqual(setup.DSV41_ALL_FILES[setup.DSV41_MMPROJ][0], 972_208_288)           # Unsloth's, not overwritten by the name clash
        self.assertEqual(setup.DSV41_ALT_FILES[setup.DSV41_MMPROJ], [pin])
        self.assertIn(pin[1], setup.dsv41_also(setup.DSV41_MMPROJ))
        self.assertEqual(setup.DSV41_SHA_ALSO, {setup.DSV41_ISTA_ENGRAM: (setup.DSV41_ENGRAM_CONVERTED[1],)})      # unchanged
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / setup.DSV41_MMPROJ
            f.write_bytes(b"\0" * 16)
            self.assertEqual(setup.dsv41_pin_of(f, setup.DSV41_ALL_FILES[setup.DSV41_MMPROJ]), setup.DSV41_ALL_FILES[setup.DSV41_MMPROJ])

    def test_the_other_sets_are_unchanged(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]
        self.assertEqual(fam["IST-DAS-2.00bpw"]["recipe"], "ista")
        self.assertEqual(len(fam["IST-DAS-2.00bpw"]["shards"]), 4)
        self.assertEqual(list(fam)[:4], ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", NAME])


class Profile(unittest.TestCase):
    """The set's own hot-expert profile (made from the engine's routing over the 264 calibration texts, 519,422 prompt tokens): rows of 192, not the 2.00 set's."""

    def test_the_profile_is_the_sets_own_and_its_hash_is_pinned(self):
        import hashlib
        f = ROOT / "data" / "dsv41" / PIN["profile"]
        data = f.read_bytes()
        self.assertEqual(len(data), (40 + 3) * 192 * 4)            # 40 layers and the drafter's 3 blocks, rows of 192
        self.assertEqual(hashlib.sha256(data).hexdigest(), f.with_name(f.name + ".sha256").read_text().split()[0])
        self.assertEqual(setup.MULTI_FILE_FAMILIES["deepseek41"]["models"][NAME]["profile"], PIN["profile"])
        self.assertEqual(dsv41_fit.ISTA_SETS[NAME]["profile"], PIN["profile"])

    def test_the_engine_reads_it_and_not_the_2_00_one(self):
        """The same size gate as src/dsv41/ds4_experts_rule.hpp profile_size_fits (tests/dsv41/experts_rule_test.cpp runs the C++ one)."""
        def fits(n, layers, n_exp):
            return 4 * n_exp * layers <= n <= 4 * n_exp * (layers + 8)
        mine = (ROOT / "data" / "dsv41" / PIN["profile"]).stat().st_size
        other = (ROOT / "data" / "dsv41" / "profile-2.00.bin").stat().st_size
        self.assertTrue(fits(mine, 40, 192))
        self.assertFalse(fits(other, 40, 192), "a profile counted on 384 experts a layer is not read by the 192-expert pack")
        self.assertTrue(fits(other, 40, 384))
        self.assertFalse(fits(mine, 40, 384))


class Recognised(unittest.TestCase):
    def files(self):
        t = dsv41_sets.tables()["sets"][NAME]
        return {**t["shards"], **t["engram_alt"]}

    def test_a_whole_set_is_recognised_by_names_and_sizes(self):
        rows = [r for r in dsv41_sets.match(self.files()) if r["model"] == NAME]
        self.assertTrue(rows and rows[0]["complete"])
        self.assertEqual(rows[0]["engram"], PIN["engram_shard"])

    def test_another_models_split_files_with_the_same_names_are_not_the_set(self):
        """model-00002-of-00004.gguf is a name any split model has: only the pinned sizes make it this set."""
        other = {n: s + 1000 for n, s in self.files().items()}
        self.assertEqual([r for r in dsv41_sets.match(other) if r["model"] == NAME and r["complete"]], [])
        self.assertEqual([r for r in dsv41_sets.match(other) if r["model"] == NAME and r["present"]], [])

    def test_a_cut_shard_is_refused(self):
        files = self.files()
        files[PIN["shards"][1]] -= 1 << 20
        row = next(r for r in dsv41_sets.match(files) if r["model"] == NAME)
        self.assertFalse(row["complete"])
        self.assertEqual(row["wrong_size"], [PIN["shards"][1]])

    def test_the_launch_claims_a_folder_only_when_the_sizes_are_the_pinned_ones(self):
        from strata_cli.launch.runner import Runner  # noqa: F401  (the staticmethod lives on the runner class)
        with tempfile.TemporaryDirectory() as tmp:
            for n in PIN["shards"]:
                (Path(tmp) / n).write_bytes(b"GGUF")                       # right names, wrong sizes: some other model's split
            self.assertIsNone(self.set_in(tmp))
            sizes = {n: v[0] for n, v in setup.DSV41_ALL_FILES.items()}
            for n in PIN["shards"]:
                with (Path(tmp) / n).open("wb") as f:
                    f.truncate(sizes[n])                                   # sparse files of the pinned sizes
            found = self.set_in(tmp)
            self.assertEqual(found[0], NAME)

    @staticmethod
    def set_in(folder):
        from strata_cli.launch import runner
        cls = next(c for c in vars(runner).values() if isinstance(c, type) and hasattr(c, "deepseek_set_in"))
        return cls.deepseek_set_in(setup, folder)


class Plan(unittest.TestCase):
    def test_the_model_has_half_the_experts(self):
        for cards in ([R9700], FOUR):
            p = dsv41_fit.plan(cards, 125.0, None, 131072, model=NAME)
            self.assertEqual(p["experts_total"], 7680)
            self.assertLessEqual(p["experts_in_vram"], 7680)
            self.assertEqual(p["variant"], NAME)
            self.assertAlmostEqual(p["experts_gb"], 66.0, delta=0.1)
        p = dsv41_fit.plan(FOUR, 251.0, None, 131072, model=NAME)
        self.assertEqual(p["experts_in_vram"], 7680)                       # four 32 GB cards hold every one of them
        self.assertEqual(p["verdict"], "fits")

    def test_the_plan_follows_the_engine_on_r9700_cards(self):
        """The engine's own startup lines (r9700, 128K): one card holds 2,859 of 7,680 with 5.70 GB of other weights; four cards hold all of them with 6.42 GB."""
        p = dsv41_fit.plan([R9700], 251.0, None, 131072, model=NAME)
        self.assertAlmostEqual(p["experts_in_vram"], 2859, delta=2859 * 0.03)
        self.assertAlmostEqual(p["cards"][0]["dense_gb"], 5.70, delta=0.05)
        if __import__("os").name != "nt":                        # (Windows runs one AMD card: the plan counts one there)
            p4 = dsv41_fit.plan(FOUR, 251.0, None, 131072, model=NAME)
            self.assertEqual(p4["experts_in_vram"], 7680)
            self.assertAlmostEqual(sum(c["dense_gb"] for c in p4["cards"]), 6.42, delta=0.1)

    def test_the_ram_words_are_the_sets_own_measured_ones(self):
        """One R9700 with the RAM capped for the whole run (decode median, 3 prompts): 25.6 tok/s uncapped, 20.9 at 40 GB, 18.5 at 32, 15.0 at 24, 8.0 at 16."""
        tight = dsv41_fit.plan([R9700], 45.0, None, 131072, model=NAME)
        self.assertEqual(tight["verdict"], "tight")
        self.assertIn("21 tokens a second with 40 GB", tight["words"])
        self.assertNotIn("4.7 tokens", tight["words"])
        slow = dsv41_fit.plan([R9700], 20.0, None, 131072, model=NAME)
        self.assertEqual(slow["verdict"], "spills")
        self.assertIn("down to 16 GB", " ".join(slow["warnings"]))
        self.assertNotIn("46 GB and 64 GB", " ".join(slow["warnings"]))
        import strata_mcp as M
        s = M.Strata(ROOT)
        s.dsv41_plan = lambda hw, disk_free=None, only=None, model=None: slow
        self.assertTrue(s._dsv41_memory_words({}, NAME).startswith("slow here: this computer has 20 GB of RAM and the set was measured down to 16 GB"))
        # the other sets keep their words
        other = dsv41_fit.plan([R9700], 30.0, None, 131072, model="IST-DAS-2.00bpw")
        self.assertIn("46 GB and 64 GB", " ".join(other["warnings"]))

    def test_the_download_is_the_model_plus_the_shared_files(self):
        gb = dsv41_fit.ista_download_bytes(NAME) / 1e9
        self.assertAlmostEqual(gb, (72_123_566_336 + 110_595_290_816 + 7_976_914_240) / 1e9, delta=0.01)
        self.assertEqual(dsv41_fit.ista_download_bytes(NAME, have_engram=True), 72_123_566_336 + 7_976_914_240)

    def test_the_other_sets_still_have_384_experts_a_layer(self):
        for m in ("IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "UD-IQ1_S", "UD-IQ1_M"):
            self.assertEqual(dsv41_fit.plan([R9700], 125.0, None, 131072, model=m)["experts_total"], 15360, m)

    def test_a_mac_gets_the_plain_not_run_line(self):
        mac = {"index": 0, "name": "Apple M5 Max", "vendor": "apple", "unified_gb": 128.0, "vram_gb": 128.0}
        p = dsv41_fit.plan([mac], 128.0, None, 131072, model=NAME)
        self.assertTrue(p["mac"])
        self.assertIn("Not run on a Mac yet", p["words"])
        self.assertEqual(p["experts_total"], 7680)

    def test_the_menu_row_is_there(self):
        import strata_mcp as M
        s = M.Strata(ROOT)
        fam = next(f for f in s.model_table({"ram_gb": 251.0, "gpus": [{**c, "usable": True} for c in FOUR], "engine_vendor": "amd"})
                   if f["family"] == "deepseek41")
        row = next(x for x in fam["sizes"] if x["model"] == NAME)
        self.assertEqual(row["id"], "deepseek-v4.1-flash-" + NAME.lower())
        self.assertTrue(row["images"] and row["multi_file"])
        self.assertEqual(row["disk_gb"], round(row["download_gb"] + dsv41_fit.DISK_HEADROOM_GB, 1))
        self.assertIn("192", row["about"])
        self.assertTrue(row["on_this_pc"].startswith("fits"))


class Pack(unittest.TestCase):
    def test_the_pack_call_is_the_ista_one_with_the_sets_own_metadata_shard(self):
        md = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"][NAME]
        cmd = setup.dsv41_pack_cmd(Path("/m"), md["shards"], Path("/s/e.gguf"), Path("/m") / md["shards"][0], Path("/s/d.gguf"), Path("/p"), Path("/s/mm.gguf"))
        self.assertEqual(cmd[cmd.index("--shards") + 1:cmd.index("--engram")], md["shards"])
        self.assertEqual(cmd[cmd.index("--engram-meta") + 1], str(Path("/m") / md["shards"][0]))
        self.assertIn("--mmproj", cmd)
        self.assertIn("--dspark-noise-token", cmd)


class PinTool(unittest.TestCase):
    def test_build_reads_the_hub_and_the_headers(self):
        sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
        import pin_coder as P
        names = ["mmproj-BF16.gguf", "model-00001-of-00003.gguf", "model-00002-of-00003.gguf", "model-00003-of-00003.gguf"]
        info = {"sha": "a" * 40, "private": False, "gated": False, "lastModified": "x",
                "siblings": [{"rfilename": n, "size": 100 + i, "lfs": {"sha256": f"{i:064x}"}} for i, n in enumerate(names)] + [{"rfilename": ".gitattributes", "size": 1}]}
        kv_meta = {"general.architecture": "deepseek41", "deepseek41.expert_count": 128, "deepseek41.expert_used_count": 6, "general.license": "mit"}
        heads = {"mmproj-BF16.gguf": {"kv": {"clip.projector_type": "deepseek41v"}, "tensors": {}},
                 "model-00001-of-00003.gguf": {"kv": kv_meta, "tensors": {}},
                 "model-00002-of-00003.gguf": {"kv": {}, "tensors": {"blk.0.ffn_gate_exps.weight": [5120, 2304, 128], "blk.0.ffn_gate_inp.weight": [5120, 128],
                                                                    "blk.0.exp_probs_b.bias": [128]}},
                 "model-00003-of-00003.gguf": {"kv": {}, "tensors": {"per_layer_token_embd.weight": [256, 5]}}}
        with mock.patch.object(P, "get_json", return_value=info), mock.patch.object(P, "read_header", side_effect=lambda url: heads[url.rsplit("/", 1)[1]]):
            out = P.build("owner/new-name", None)
            self.assertEqual((out["repo"], out["revision"], out["experts_per_layer"]), ("owner/new-name", "a" * 40, 128))
            self.assertEqual(out["shards"], ["model-00001-of-00003.gguf", "model-00002-of-00003.gguf"])
            self.assertEqual((out["engram_shard"], out["mmproj"]), ("model-00003-of-00003.gguf", "mmproj-BF16.gguf"))
            heads["model-00002-of-00003.gguf"]["tensors"]["blk.1.ffn_gate_inp.weight"] = [5120, 64]       # a layer that disagrees: refused
            with self.assertRaises(SystemExit):
                P.build("owner/new-name", None)
            self.assertRaises(SystemExit, P.build, "not a repo", None)


if __name__ == "__main__":
    unittest.main()
