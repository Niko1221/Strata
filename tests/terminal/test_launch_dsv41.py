"""`strata launch` for DeepSeek-V4.1-Flash: the model is in the list, `--plan` says where it goes on this computer's cards (the real
numbers of strata_control/dsv41_fit.py), `--model-path` takes the nine shards where they are, and the settings that cannot apply
(parallel requests, pictures, tuning) are said instead of passed to an engine that would refuse them.  Nothing is downloaded or started."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "usable": True, "vendor": "amd"}
HW_4X32 = {**HW_12, "ram_gb": 251.0, "disk_free_gb": 900.0, "engine_vendor": "amd",
           "gpus": [{**R9700, "index": i} for i in range(4)]}


class LaunchDeepSeek(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(HW_4X32)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)
        # several AMD cards share a DeepSeek model on Linux only (dsv41_fit.run_cards asks the host OS); these four R9700 are a Linux PC
        p = mock.patch("strata_control.dsv41_fit._on_windows", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", *args])
        self.assertEqual(code, 0, err + out)
        return json.loads(out)["plan"]

    def test_the_model_is_in_the_list(self):
        rows = [r for r in catalog(self.hw).rows() if r.family == "deepseek41"]
        self.assertEqual([r.model for r in rows], ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "ISTA-DAS-Coder", "UD-IQ1_S", "UD-IQ1_M", "UD-Q8_K_XL"])
        self.assertEqual([r.download_gb for r in rows[:3]], [256.8, 273.8, 239.3])
        self.assertEqual(catalog(self.hw).resolve("ist-das-2.25bpw"), ("deepseek41", "IST-DAS-2.25bpw"))
        self.assertEqual(catalog(self.hw).resolve("ISTA-DAS-2.25bpw"), ("deepseek41", "IST-DAS-2.25bpw"))     # the name on screen is accepted too
        self.assertEqual(catalog(self.hw).resolve("ista-das-coder"), ("deepseek41", "ISTA-DAS-Coder"))
        self.assertEqual(catalog(self.hw).resolve("IST-DAS-Coder"), ("deepseek41", "ISTA-DAS-Coder"))         # (the shorter spelling the other ids have is accepted)
        r = rows[-1]
        self.assertEqual((r.id, r.download_gb, r.verdict, r.images), ("deepseek-v4.1-flash-ud-q8_k_xl", 508.4, "fits", True))   # 0.2.7: pictures through the engine's own tower
        self.assertEqual(catalog(self.hw).resolve("ud-q8_k_xl"), ("deepseek41", "UD-Q8_K_XL"))
        self.assertEqual(catalog(self.hw).max_context("deepseek41", "UD-Q8_K_XL"), 131072)

    def runner_with_own_shards(self, **kw):
        own = self.home.base / "own"
        own.mkdir()
        (own / "x-00001-of-00006.gguf").write_bytes(b"x")
        s = S.LaunchSettings(family="deepseek41", model="ISTA-DAS-Coder", context=8192, vision=kw.pop("vision", "off"), model_path=str(own),
                             data_dir=str(self.home.base / "data"), **kw)
        return Runner(s, catalog=catalog(self.hw))

    def test_the_plan_with_your_own_shards_counts_the_shared_files_setup_still_fetches(self):
        """Strix, 2026-11-02: `--model-path` plan said "Nothing to download" and setup then fetched the 8 GB DSpark drafter (even with --dspark off).  The plan now is the fetch."""
        r = self.runner_with_own_shards(extra={"--dspark": "off"})
        mod = r.catalog.strata.setup_module()
        with mock.patch.object(Runner, "check_own_files"), mock.patch.object(mod, "dsv41_shared_missing", return_value=[(mod.DSV41_DSPARK, 7976914240)]) as m:
            plan = r.download_plan()
        self.assertEqual(plan["gb"], 8.0)
        self.assertIn("DSpark drafter (8.0 GB)", plan["text"])
        self.assertIn("even with --dspark off", plan["text"])
        self.assertNotIn("Nothing to download", plan["text"])
        self.assertEqual(m.call_args.kwargs["vision"], False)             # --vision off: the picture file is not asked for

    def test_the_plan_with_all_shared_files_there_says_nothing_to_download(self):
        r = self.runner_with_own_shards()
        mod = r.catalog.strata.setup_module()
        with mock.patch.object(Runner, "check_own_files"), mock.patch.object(mod, "dsv41_shared_missing", return_value=[]):
            plan = r.download_plan()
        self.assertEqual(plan["gb"], 0)
        self.assertIn("Nothing to download: your model files are used where they are", plan["text"])

    def test_pictures_on_ask_for_the_picture_file_too(self):
        r = self.runner_with_own_shards(vision="on")
        mod = r.catalog.strata.setup_module()
        with mock.patch.object(Runner, "check_own_files"), mock.patch.object(mod, "dsv41_shared_missing", return_value=[]) as m:
            r.download_plan()
        self.assertEqual(m.call_args.kwargs["vision"], True)

    def test_the_plan_places_the_model_on_the_four_cards(self):
        p = self.plan("--model", "ud-q8_k_xl", "--no-tui")
        pl = p["placement"]
        self.assertEqual(p["settings"]["family"], "deepseek41")
        self.assertEqual(p["settings"]["context"], 131072)
        self.assertEqual(p["settings"]["tune"], "skip")
        self.assertEqual(p["settings"]["vision"], "on")              # pictures are on by default (the engine's own tower)
        self.assertEqual(pl["verdict"], "fits")
        self.assertEqual([c["layers"] for c in pl["cards"]], [[0, 9], [10, 19], [20, 29], [30, 39]])
        self.assertEqual(pl["args"], ["--tp", "2"])
        # measured on this very computer: 6,197 experts (116.5 GB) in graphics memory, 9,163 (172.3 GB) on the CPU
        self.assertAlmostEqual(pl["experts_in_vram"], 6197, delta=100)
        self.assertAlmostEqual(pl["cpu_experts_gb"], 172.3, delta=2.5)
        text = " ".join(pl["lines"])
        self.assertIn("n-gram table", text)
        self.assertIn("stays on the SSD", text)
        self.assertIn("508 GB", text)

    def test_the_human_plan_shows_the_placement(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("placement", out)
        self.assertIn("4 cards, one block of layers on each", out)
        self.assertIn("n-gram table", out)
        self.assertIn("Nothing was changed.", out)

    def test_named_cards_are_the_ones_planned(self):
        p = self.plan("--model", "ud-q8_k_xl", "--gpu", "0,1")
        self.assertEqual([c["index"] for c in p["placement"]["cards"]], [0, 1])
        self.assertLess(p["placement"]["experts_in_vram"], 3000)

    def test_a_small_computer_gets_the_warnings_not_a_refusal(self):
        self.hw = {**HW_12, "ram_gb": 64.0, "disk_free_gb": 100.0, "engine_vendor": "nvidia"}
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("read from the SSD", out)
        self.assertIn("free disk space", out)
        self.assertIn("It is still allowed", out)
        row = next(r for r in catalog(self.hw).rows() if r.family == "deepseek41" and r.model == "UD-Q8_K_XL")
        self.assertEqual((row.verdict, row.words), ("tight", "tight: needs more disk space"))      # 100 GB free: the disk is said first; the plan above is printed and nothing is refused

    def test_the_plan_names_dspark_not_mtp(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("drafting  DSpark", out)
        self.assertNotIn("MTP", out)

    def test_the_plan_does_not_promise_a_strata_engine_download_for_deepseek(self):
        p = self.plan("--model", "ud-q8_k_xl", "--no-tui")
        text = json.dumps(p.get("download") or {})
        self.assertNotIn("engine for your graphics card", text)
        self.assertFalse((p.get("download") or {}).get("engine_missing"))

    def test_no_card_is_said(self):
        self.hw = {**HW_12, "gpus": []}
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("No graphics card", out)

    def test_parallel_is_passed_on_and_pictures_and_tuning_stay(self):
        """`--parallel 4` on DeepSeek is no longer dropped (tests/terminal/test_launch_dsv41_parallel.py has the rest)."""
        p = self.plan("--model", "ud-q8_k_xl", "--parallel", "4", "--vision", "on", "--tune", "quick")
        s = p["settings"]
        self.assertEqual(s.get("parallel"), 4)
        self.assertEqual((s["vision"], s["tune"]), ("on", "quick"))           # the DeepSeek tune is real now: the choice stays; pictures stay on
        text = " ".join(p["warnings"])
        self.assertNotIn("one request at a time", text)
        self.assertNotIn("cannot read pictures", text)
        off = self.plan("--model", "ud-q8_k_xl", "--vision", "off")["settings"]
        self.assertEqual(off["vision"], "off")                                 # --vision off still turns them off
        self.assertNotIn("Tuning is not offered", text)

    def test_setup_is_asked_for_the_family_and_the_files(self):
        s = S.LaunchSettings(family="deepseek41", model="UD-Q8_K_XL", context=8192, vision="off", tune="skip",
                             model_path="/data/ds")
        args = Runner(s, catalog=catalog(self.hw)).setup_args()
        self.assertEqual(args[:6], ["--yes", "--no-start", "--family", "deepseek41", "--model", "UD-Q8_K_XL"])
        self.assertIn("--gguf-dir", args)
        self.assertNotIn("--parallel", args)
        self.assertIsNone(Runner(s, catalog=catalog(self.hw)).catalog_parallel())

    def test_the_feed_reaches_setup_so_the_deepseek_engine_comes_from_it(self):
        """`strata launch --feed DIR` must hand the feed to setup: the DeepSeek engine package is fetched by setup, not by `strata update`
        (found on strix, 2026-10-25: the launch read the local feed for its own checks and setup went to GitHub for the engine)."""
        s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", context=8192, vision="off", tune="skip")
        args = Runner(s, catalog=catalog(self.hw), feed="/data/feed").setup_args()
        self.assertIn("--feed", args)
        self.assertEqual(args[args.index("--feed") + 1], "/data/feed")
        self.assertNotIn("--feed", Runner(s, catalog=catalog(self.hw)).setup_args())


class OwnShards(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(HW_4X32)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def folder(self, shards=range(1, 10), nested=False):
        base = Path(os.environ["STRATA_INSTALL_DIR"]) / "ds"
        folder = base / "UD-Q8_K_XL" if nested else base
        folder.mkdir(parents=True, exist_ok=True)
        for i in shards:
            (folder / f"DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-{i:05d}-of-00009.gguf").write_bytes(b"GGUF")
        return base, folder

    def runner(self, path):
        return Runner(S.LaunchSettings(model_path=str(path)), catalog=catalog(self.hw))

    def test_a_folder_of_the_nine_shards_is_the_model(self):
        _base, folder = self.folder()
        r = self.runner(folder)
        r.check_own_files()
        self.assertEqual((r.s.family, r.s.model, r.s.model_path), ("deepseek41", "UD-Q8_K_XL", str(folder)))

    def test_the_repository_folder_with_the_size_inside(self):
        base, folder = self.folder(nested=True)
        r = self.runner(base)
        r.check_own_files()
        self.assertEqual((r.s.family, r.s.model, r.s.model_path), ("deepseek41", "UD-Q8_K_XL", str(folder)))

    def test_one_shard_names_its_folder(self):
        _base, folder = self.folder()
        r = self.runner(folder / "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00003-of-00009.gguf")
        r.check_own_files()
        self.assertEqual(r.s.model_path, str(folder))

    def test_a_missing_shard_is_said_before_anything_is_downloaded(self):
        _base, folder = self.folder(shards=[1, 2, 3, 5, 6, 7, 8, 9])
        with self.assertRaises(LaunchError) as cm:
            self.runner(folder).check_own_files()
        self.assertIn("nine parts", cm.exception.text)
        self.assertIn("00004-of-00009", cm.exception.text)

    def test_the_plan_with_own_shards_counts_the_shared_files_that_are_not_there(self):
        """Only the shards are here: setup still fetches the shared files that are not (the picture file here), so the plan names them (before 0.2.9.1 it said
        "Nothing to download" and the fetch followed)."""
        _base, folder = self.folder()
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(folder), "--no-tui"])
        self.assertEqual(code, 0, err + out)
        self.assertIn("DeepSeek-V4.1-Flash", out)
        self.assertNotIn("Nothing to download", out)
        self.assertIn("to download", out)
        self.assertIn("the picture reader (1.0 GB)", out)           # (UD-Q8_K_XL has no other shared file; the Coder's drafter case is in test_the_plan_with_your_own_shards...)


class CoderHuggingFaceLayout(unittest.TestCase):
    """`--model-path` at the folder `hf download` makes for the official Coder repository: IQ2_XXS/ with the five model files, the Engram and the picture file at the
    top.  The launch plan (runner.deepseek_set_in) and setup's step 5 (dsv41_folder / dsv41_parts) must agree on where the shards are: DF2-STRIX found the plan accepting the
    folder and step 5 then saying shard 1 "is missing" (the Coder's setup.py entry had no "subdir")."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.cat = catalog(HW_4X32)
        self.mod = self.cat.strata.setup_module()
        self.fam = self.mod.MULTI_FILE_FAMILIES["deepseek41"]
        self.md = self.fam["models"]["ISTA-DAS-Coder"]
        # tiny stand-ins of the pinned sizes, so the size-checked recognition and the shared-file search run without 180 GB
        sizes = {n: (3 + i, "a" * 64) for i, n in enumerate(self.md["shards"])}
        sizes[self.mod.DSV41_ISTA_ENGRAM] = (7, "b" * 64)
        sizes[self.mod.DSV41_MMPROJ] = (5, "c" * 64)
        p = mock.patch.dict(self.mod.DSV41_ALL_FILES, sizes)
        p.start()
        self.addCleanup(p.stop)
        self.sizes = sizes
        self.root = self.home.base / "repo"
        (self.root / "IQ2_XXS").mkdir(parents=True)
        for n in self.md["shards"]:
            (self.root / "IQ2_XXS" / n).write_bytes(b"x" * sizes[n][0])
        (self.root / self.mod.DSV41_ISTA_ENGRAM).write_bytes(b"x" * 7)
        (self.root / self.mod.DSV41_MMPROJ).write_bytes(b"x" * 5)

    def test_the_entry_names_the_folder_the_json_names(self):
        self.assertEqual(self.md.get("subdir"), "IQ2_XXS")

    def test_the_plan_finds_the_shards_in_the_repository_folder_and_in_the_folder_itself(self):
        s = S.LaunchSettings(family="deepseek41", model="ISTA-DAS-Coder", context=8192, vision="off", model_path=str(self.root))
        r = Runner(s, catalog=self.cat)
        for given in (self.root, self.root / "IQ2_XXS"):
            found = r.deepseek_set_in(self.mod, str(given), "ISTA-DAS-Coder")
            self.assertEqual((found[0], found[2]), ("ISTA-DAS-Coder", self.root / "IQ2_XXS"), given)

    def test_setups_step_5_looks_in_the_same_folder_and_misses_nothing(self):
        from types import SimpleNamespace
        for given in (self.root, self.root / "IQ2_XXS"):
            a = SimpleNamespace(gguf_dir=str(given), models_dir=str(self.home.base / "models"))
            folder = self.mod.dsv41_folder(a, self.fam, self.fam["tag"] + "ISTA-DAS-Coder", "ISTA-DAS-Coder")
            self.assertEqual(folder, self.root / "IQ2_XXS", given)
            _layout, names, missing = self.mod.dsv41_parts(self.md, lambda n: (folder / n).exists())
            self.assertEqual((len(names), missing), (5, []), given)
            self.assertTrue(all(self.mod.dsv41_pick(folder, n).exists() for n in self.md["shards"]))

    def test_the_engram_and_the_picture_file_at_the_top_are_found_and_not_fetched(self):
        for given in (self.root, self.root / "IQ2_XXS"):
            missing = self.mod.dsv41_shared_missing("ISTA-DAS-Coder", self.home.base / "models", given, vision=True)
            self.assertEqual([n for n, _ in missing], [self.mod.DSV41_DSPARK], given)       # only the drafter is still to fetch

if __name__ == "__main__":
    unittest.main()
