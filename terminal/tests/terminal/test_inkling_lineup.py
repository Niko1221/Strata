"""The Inkling-2-Mini lineup of 0.2.23 (owner decision 2026-11-26).

Offered, in this order everywhere: the Strata packs SQ-20, SQ-15, SQ-10; GSQ-RCO IQ3_S (ISTA-DASLab); Unsloth UD-Q4_K_XL.
The other nine Unsloth quants are not offered any more, but an install that has one keeps working and shows it as installed, "no longer offered".
DFlash 2 is the only drafter of every build: its file is fetched beside every model, no DSpark file is fetched, and without the file there is no drafter.

    python -I terminal/tests/terminal/test_inkling_lineup.py
"""
import argparse
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strata_control import inkling_family as F  # noqa: E402
from strata_control import inkling_fit as T  # noqa: E402

import setup  # noqa: E402

OFFERED = ("SQ-20", "SQ-15", "SQ-10", "GSQ-RCO-IQ3_S", "UD-Q4_K_XL")
LEGACY = ("UD-Q8_K_XL", "UD-Q4_K_M", "UD-IQ4_XS", "UD-Q3_K_XL", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class Pins(unittest.TestCase):
    """The sizes and SHA-256 of every file the lineup downloads (the owner's figures, the Hugging Face API of 2026-11-26)."""

    def test_sq20_v2_and_vision_v2_are_the_new_files(self):
        self.assertEqual(F.SQ20_FILE, "Inkling-2-Mini-SQv2-20.strata")
        self.assertEqual(F.FILES[F.SQ20_FILE], (19_374_768_128, "429671f8eb5b50e593856e6eb82ebb7da9d1f07e99b553438b196f72cc86a6d6"))
        self.assertEqual(F.VISION_FILE, "Inkling-2-Mini-Vision-v2.strata")
        self.assertEqual(F.FILES[F.VISION_FILE], (636_551_168, "34a176505276729b58744fc9ee40a5005bbff9a590d8e1f7dbe82baf33ef07d5"))
        self.assertEqual(F.LEGACY_FILES[F.VISION_FILE_V1], (636_526_592, "6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677"))
        self.assertEqual(F.VISION_FILE_V1, "Inkling-2-Mini-Vision.strata")
        self.assertEqual(F.FILES["Inkling-2-Mini-SQ-15.strata"][1], "2ea32eed533afba257e2cbee7b073dc2c3da80fe55d1d2d794779b5498e30737")     # SQ-15 and SQ-10 unchanged
        self.assertEqual(F.FILES["Inkling-2-Mini-SQ-10.strata"][1], "ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934")
        self.assertNotEqual(F.SQ20_FILE, F.SQ20_FILE_V1)
        self.assertEqual(F.SQ20_FILE, T.BUILDS["SQ-20"]["file"], "inkling_fit.BUILDS and inkling_family name the same SQ-20 file")
        self.assertEqual(F.build_file("SQ-20"), F.SQ20_FILE)
        self.assertEqual(round(F.FILES[F.SQ20_FILE][0] / 1e9, 2), T.BUILDS["SQ-20"]["file_gb"])

    def test_the_0222_files_are_still_recognised_under_their_old_names(self):
        self.assertEqual(F.LEGACY_FILES[F.SQ20_FILE_V1], (19_346_685_952, "0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97"))
        self.assertEqual(F.size_of(F.SQ20_FILE_V1), 19_346_685_952)
        self.assertEqual(F.pins_for(F.SQ20_FILE_V1), [F.LEGACY_FILES[F.SQ20_FILE_V1]])
        self.assertEqual(F.pins_for(F.SQ20_FILE), [F.FILES[F.SQ20_FILE]])         # the new file is checked against the new pin only
        self.assertEqual(F.LEGACY_BUILD_FILES["SQ-20"], (F.SQ20_FILE_V1,))

    def test_gsq_rco_iq3_s(self):
        self.assertEqual(F.FILES[F.GSQ_IQ3S_FILE][0], 11_425_259_648)
        self.assertEqual(F.FILES[F.GSQ_IQ3S_FILE][1], "8a3515374c3cd9b65a1677b17e805bb1e32df8e98756e5f0d2fe1837d62aa4cb")
        self.assertEqual(F.GSQ_IQ3S_FILE, "Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf")
        self.assertEqual(F.GSQ_MMPROJ_PIN[0], 641_806_176)
        self.assertTrue(F.GSQ_MMPROJ_PIN[1].startswith("4f8608d0"))
        self.assertEqual(T.BUILDS["GSQ-RCO-IQ3_S"]["file"], F.GSQ_IQ3S_FILE)
        self.assertEqual(T.BUILDS["GSQ-RCO-IQ3_S"]["format"], "gguf")
        self.assertEqual(round(F.GSQ_IQ3S_PIN[0] / 1e9, 2), T.BUILDS["GSQ-RCO-IQ3_S"]["file_gb"])
        self.assertTrue(HEX40.match(F.GSQ_COMMIT))
        self.assertTrue(F.GSQ_COMMIT.startswith("357051619a88"))

    def test_the_gsq_planner_figures_come_from_the_files_header(self):
        # read from the GGUF header at the pinned commit (704 tensors): routed experts 9,523,298,304 bytes, 25.4e9 expert elements in eight types, no NVFP4
        b = T.BUILDS["GSQ-RCO-IQ3_S"]
        self.assertEqual(b["bodies_gb"], 9.52)
        self.assertEqual(sorted(b["expert_elems_e9"]), ["IQ1_S", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ4_NL", "Q2_0"])
        self.assertAlmostEqual(sum(b["expert_elems_e9"].values()), 24.46, delta=0.05)         # the same 24.46e9 expert weights as every other build
        self.assertAlmostEqual(sum(T.BUILDS["UD-Q4_K_XL"]["expert_elems_e9"].values()), 24.46, delta=0.05)
        self.assertNotIn("NVFP4", b["expert_elems_e9"])

    def test_unsloth_ud_q4_k_xl_is_the_refreshed_file_at_a_fixed_commit(self):
        self.assertEqual(F.FILES[F.UNSLOTH_OFFERED_FILE], (17_629_760_736, "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3"))
        self.assertEqual(F.UNSLOTH_COMMIT, "8078d149286eec7193795413b9e0643da6d11fa1")
        self.assertTrue(HEX40.match(F.UNSLOTH_COMMIT))
        src = F.sources_for(F.UNSLOTH_OFFERED_FILE, env={})[0]
        self.assertEqual(F.urls_for(F.UNSLOTH_OFFERED_FILE, src), [f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/{F.UNSLOTH_COMMIT}/Inkling-2-Mini-UD-Q4_K_XL.gguf"])
        # the file 0.2.22 pinned (and every install that has it) is still accepted, after the new one
        self.assertEqual(F.pins_for(F.UNSLOTH_OFFERED_FILE), [F.FILES[F.UNSLOTH_OFFERED_FILE], (17_629_759_456, "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15")])

    def test_only_the_offered_files_are_downloads(self):
        self.assertEqual(set(F.FILES), {F.SQ20_FILE, "Inkling-2-Mini-SQ-15.strata", "Inkling-2-Mini-SQ-10.strata", F.VISION_FILE, F.GSQ_IQ3S_FILE, F.UNSLOTH_OFFERED_FILE})
        for b in LEGACY:
            self.assertNotIn(T.BUILDS[b]["file"], F.FILES, b)
            self.assertIn(T.BUILDS[b]["file"], F.LEGACY_FILES, b)                    # but a file an install has is recognised
            self.assertIn(T.BUILDS[b]["file"], F.ALL_FILES, b)
            self.assertTrue(F.pins_for(T.BUILDS[b]["file"]), b)

    def test_every_pin_is_a_size_and_a_sha(self):
        for name, (size, sha) in F.ALL_FILES.items():
            self.assertGreater(size, 100_000_000, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)

    def test_every_download_goes_through_a_pinned_commit(self):
        for name in (F.GSQ_IQ3S_FILE, F.UNSLOTH_OFFERED_FILE):
            for src in F.sources_for(name, env={}):
                self.assertRegex(src["base"], r"/resolve/[0-9a-f]{40}$", name)
        # a quant that is no longer offered keeps its own (0.2.22) commit, so a check of a file already there still goes to the bytes it was pinned to
        self.assertTrue(F.sources_for("Inkling-2-Mini-UD-IQ1_S.gguf", env={})[0]["base"].endswith(F.UNSLOTH_OLD_COMMIT))

    def test_every_strata_pack_download_is_pinned_to_the_commit_of_the_sqv2_upload(self):
        self.assertEqual(F.STRATA_PACK_COMMIT, "5623ceb47d7a01db827460ce1c23e7818a75661b")
        self.assertTrue(F.pack_commit_is_set())
        for name in (F.SQ20_FILE, F.VISION_FILE, "Inkling-2-Mini-SQ-15.strata", "Inkling-2-Mini-SQ-10.strata", F.SQ20_FILE_V1, F.TOKENIZER_FILE, F.CHAT_TEMPLATE_FILE):
            self.assertEqual(F.urls_for(name, F.sources_for(name, env={})[0]),
                             [f"https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/5623ceb47d7a01db827460ce1c23e7818a75661b/{name}"], name)
        self.assertEqual(F.urls_for(F.SQ20_FILE, F.sources_for(F.SQ20_FILE, env={})[0])[0].split("/resolve/")[1][:8], "5623ceb4")
        self.assertNotIn("/main/", F.urls_for(F.SQ20_FILE, F.sources_for(F.SQ20_FILE, env={})[0])[0])

    def test_the_old_sq20_is_offered_the_new_file_and_never_deleted(self):
        self.assertTrue(F.has_newer_file("SQ-20", "/m/models/inkling-2-mini/Inkling-2-Mini-SQ-20.strata"))
        self.assertFalse(F.has_newer_file("SQ-20", "/m/Inkling-2-Mini-SQv2-20.strata"))
        self.assertFalse(F.has_newer_file("SQ-15", "/m/Inkling-2-Mini-SQ-15.strata"))
        self.assertTrue(F.has_newer_file("SQ-15", "/m/Inkling-2-Mini-SQ-15.strata", "/m/Inkling-2-Mini-Vision.strata"))     # the old Vision file too
        self.assertFalse(F.has_newer_file("SQ-15", "/m/Inkling-2-Mini-SQ-15.strata", "/m/Inkling-2-Mini-Vision-v2.strata"))
        self.assertFalse(F.has_newer_file("SQ-20", None))
        src = Path(setup.__file__).read_text(encoding="utf-8")
        self.assertNotIn("unlink", src[src.index("def install_inkling2("):src.index("def preview_family(")], "the install never deletes a model file")

    def test_the_shipped_sources_file_adds_nothing(self):
        data = json.loads(F.SOURCES_FILE.read_text(encoding="utf-8"))
        self.assertEqual((data["sources"], data["gguf_sources"], data["gsq_sources"]), ([], [], []))
        self.assertEqual(len(F.sources(env={})), 1)
        self.assertEqual(len(F.gguf_sources(env={})), 1)
        self.assertEqual(len(F.gsq_sources(env={})), 1)

    def test_a_mirror_still_wins_for_every_file(self):
        env = {F.SOURCE_ENV: "https://example.org/mirror"}
        for name in (F.SQ20_FILE, F.GSQ_IQ3S_FILE, F.UNSLOTH_OFFERED_FILE):
            self.assertEqual(F.sources_for(name, env=env)[0]["base"], "https://example.org/mirror", name)
        self.assertEqual(F.sources_for(F.GSQ_IQ3S_FILE, env={F.GSQ_SOURCE_ENV: "https://example.org/g"})[0]["base"], "https://example.org/g")

    def test_credit_and_licences(self):
        self.assertEqual(F.BUILD_CREDIT["GSQ-RCO-IQ3_S"], "GSQ + RCO by IST Austria DASLab, Apache-2.0")
        self.assertIn("Apache-2.0", F.BUILD_CREDIT["UD-Q4_K_XL"])
        self.assertEqual(setup.INKLING_FAMILIES[F.FAMILY]["models"]["GSQ-RCO-IQ3_S"]["credit"], "GSQ + RCO by IST Austria DASLab, Apache-2.0")
        for b in OFFERED:
            self.assertTrue(F.BUILD_ABOUT[b], b)
            self.assertNotIn("DSpark", F.BUILD_ABOUT[b], b)
            self.assertIn("DFlash 2", F.BUILD_ABOUT[b], b)


class Order(unittest.TestCase):
    """Strata packs first (SQ-20, SQ-15, SQ-10), then GSQ-RCO IQ3_S, then Unsloth UD-Q4_K_XL: in setup, the launcher and the app."""

    def test_the_planner_the_menu_and_the_family_table_agree(self):
        self.assertEqual(T.ORDER, OFFERED)
        self.assertEqual(T.STRATA_ORDER, OFFERED[:3])
        self.assertEqual(T.OFFERED_GGUF_ORDER, OFFERED[3:])
        self.assertEqual(tuple(setup.INKLING_FAMILIES[F.FAMILY]["models"]), OFFERED)
        self.assertEqual([r["build"] for r in T.rows(24, 64, False)], list(OFFERED))
        self.assertEqual([b for b, _ in setup.inkling_menu_lines(24.0, 64.0, False)], list(OFFERED))

    def test_the_nine_other_unsloth_quants_are_legacy_and_everything_still_knows_them(self):
        self.assertEqual(T.LEGACY_ORDER, LEGACY)
        self.assertEqual(tuple(setup.INKLING_FAMILIES[F.FAMILY]["legacy_models"]), LEGACY)
        self.assertEqual(set(T.GGUF_ORDER), {"GSQ-RCO-IQ3_S", "UD-Q4_K_XL", *LEGACY})
        for b in LEGACY:
            self.assertIn(b, T.BUILDS)
            self.assertFalse(F.is_offered(b))
            self.assertTrue(F.is_gguf_build(b))
            self.assertTrue(F.about_of(b))
            self.assertTrue(T.place(b, 12.0, 32.0, False, 32768, False)["vram_gb"] >= 0)       # the planner still places one
        for b in OFFERED:
            self.assertTrue(F.is_offered(b))

    def test_nothing_recommends_or_menus_a_legacy_quant(self):
        for vram, ram in ((4, 16), (8, 32), (12, 64), (16, 32), (24, 64), (48, 128)):
            self.assertIn(T.recommend(vram, ram, False)["build"], T.STRATA_ORDER)
        self.assertEqual(len(setup.inkling_menu_lines(12.0, 32.0, False)), 5)

    def test_download_size_counts_the_model_the_vision_file_and_the_dflash2_file(self):
        for b in OFFERED:
            want = round(T.BUILDS[b]["file_gb"] + T.VISION_GB + T.DFLASH2_GB, 2)
            self.assertEqual(T.disk_gb(b, True), want, b)
            self.assertEqual(T.disk_gb(b, False, False), T.BUILDS[b]["file_gb"], b)
        self.assertAlmostEqual(T.DFLASH2_GB, F.size_of(F.DFLASH2_FILE) / 1e9, places=3)
        row = {r["build"]: r for r in T.rows(24, 64, False)}
        self.assertAlmostEqual(row["UD-Q4_K_XL"]["download_gb"], T.BUILDS["UD-Q4_K_XL"]["file_gb"] + T.DFLASH2_GB, places=2)

    def test_files_of(self):
        self.assertEqual(F.files_of("SQ-20"), [F.SQ20_FILE, F.VISION_FILE])
        self.assertEqual(F.files_of("GSQ-RCO-IQ3_S", vision=False), [F.GSQ_IQ3S_FILE])
        self.assertEqual(F.files_of("UD-Q4_K_XL", True, drafter=True), [F.UNSLOTH_OFFERED_FILE, F.VISION_FILE, F.DFLASH2_FILE])
        self.assertEqual(F.files_of("UD-Q4_K_XL", True, dspark=True), [F.UNSLOTH_OFFERED_FILE, F.VISION_FILE])      # the old switch is ignored: no DSpark file

    def _strata(self, installed=()):
        from tools import strata_mcp as M
        s = M.Strata(ROOT)
        s.configs = lambda: [Path(f"strata-{t}.json") for t in installed]
        return s

    def _ink_sizes(self, s, ram=64.0):
        fam = next(f for f in s.model_table({"ram_gb": ram, "gpus": []}) if f["family"] == F.FAMILY)
        return fam["sizes"]

    def test_an_install_of_the_old_sq20_file_is_marked_update_available(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "strata-inkling-2-mini-sq-20.json"
            for name, want in ((F.SQ20_FILE_V1, True), (F.SQ20_FILE, False)):
                cfg.write_text(json.dumps({"args": ["--native", f"/m/{name}", "--max-context", "32768"]}), encoding="utf-8")
                s = self._strata(["inkling-2-mini-sq-20"])
                s.configs = lambda cfg=cfg: [cfg]
                sq20 = next(x for x in self._ink_sizes(s) if x["model"] == "SQ-20")
                self.assertTrue(sq20["installed"])
                self.assertEqual(sq20.get("update_available", False), want, name)
                self.assertTrue(sq20["id"] == "inkling-2-mini-sq-20")
                self.assertTrue(all(not x.get("update_available") for x in self._ink_sizes(s) if x["model"] != "SQ-20"))

    def test_an_install_of_the_old_vision_file_is_marked_update_available(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "strata-inkling-2-mini-sq-15.json"
            for vision, want in ((F.VISION_FILE_V1, True), (F.VISION_FILE, False)):
                cfg.write_text(json.dumps({"args": ["--native", "/m/Inkling-2-Mini-SQ-15.strata", "--vision", f"/m/{vision}"]}), encoding="utf-8")
                s = self._strata(["inkling-2-mini-sq-15"])
                s.configs = lambda cfg=cfg: [cfg]
                sq15 = next(x for x in self._ink_sizes(s) if x["model"] == "SQ-15")
                self.assertEqual(sq15.get("update_available", False), want, vision)

    def test_the_model_table_lists_the_five_in_order_and_hides_the_rest(self):
        sizes = self._ink_sizes(self._strata())
        self.assertEqual([x["model"] for x in sizes], list(OFFERED))
        self.assertTrue(all(x.get("offered", True) for x in sizes))
        by = {x["model"]: x for x in sizes}
        self.assertEqual(by["GSQ-RCO-IQ3_S"]["credit"], "GSQ + RCO by IST Austria DASLab, Apache-2.0")
        self.assertEqual(by["GSQ-RCO-IQ3_S"]["id"], "inkling-2-mini-gsq-rco-iq3_s")
        self.assertEqual(by["SQ-20"]["id"], "inkling-2-mini-sq-20", "the model id of SQ-20 does not change with the new file")
        self.assertEqual(by["UD-Q4_K_XL"]["format"], "gguf")
        for b in OFFERED:                                                      # every download counts the 0.55 GB DFlash 2 file
            self.assertGreater(by[b]["download_gb"], T.BUILDS[b]["file_gb"], b)

    def test_an_installed_legacy_quant_is_listed_after_them_as_no_longer_offered(self):
        sizes = self._ink_sizes(self._strata(["inkling-2-mini-ud-iq3_xxs", "inkling-2-mini-sq-10"]))
        self.assertEqual([x["model"] for x in sizes], [*OFFERED, "UD-IQ3_XXS"])
        legacy = sizes[-1]
        self.assertIs(legacy["offered"], False)
        self.assertTrue(legacy["installed"])
        self.assertIn("no longer offered", legacy["about"])
        self.assertEqual(legacy["id"], "inkling-2-mini-ud-iq3_xxs")
        self.assertTrue(next(x for x in sizes if x["model"] == "SQ-10")["installed"])
        self.assertFalse(next(x for x in sizes if x["model"] == "UD-Q4_K_XL")["installed"])

    def test_a_legacy_quant_that_is_not_installed_never_appears(self):
        for ram in (16.0, 64.0, 256.0):
            self.assertEqual([x["model"] for x in self._ink_sizes(self._strata(), ram)], list(OFFERED))

    def test_the_recommendation_is_never_a_legacy_quant(self):
        s = self._strata(["inkling-2-mini-ud-q8_k_xl"])
        for ram in (16.0, 64.0):
            sizes = self._ink_sizes(s, ram)
            self.assertFalse(any(x.get("inkling", {}).get("recommended") for x in sizes if x["model"] in LEGACY))

    def test_an_installed_choice_is_read_back_for_a_legacy_size(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "strata-inkling-2-mini-ud-iq3_xxs.json"
            cfg.write_text(json.dumps({"arch": "inkling2_loop", "args": ["--native", "m.gguf", "--max-context", "32768", "--vision", "v.strata"]}), encoding="utf-8")
            got = setup.choices_from_config(cfg)
            self.assertEqual((got["family"], got["model"], got["context"]), (F.FAMILY, "UD-IQ3_XXS", 32768))

    def test_the_launcher_rows_carry_the_flags(self):
        from strata_cli.launch import catalog
        row = catalog.ModelRow(family="x", family_title="X", model="m", id="x-m", about="a", download_gb=1.0, need_vram_gb=0, need_ram_gb=0, spill_gb=0,
                               verdict="fits", words="fits", experimental=False, images=True, installed=True)
        self.assertTrue(row.offered)
        self.assertEqual(row.credit, "")
        import dataclasses
        self.assertFalse(dataclasses.replace(row, offered=False).offered)

    def test_the_import_of_a_local_file_finds_the_build_by_name(self):
        from strata_control import models
        f = models._inkling_build_for_bytes
        self.assertEqual(f(19_374_768_128, "Inkling-2-Mini-SQv2-20.strata"), "SQ-20")
        self.assertEqual(f(19_346_685_952, "Inkling-2-Mini-SQ-20.strata"), "SQ-20")
        self.assertEqual(f(11_425_259_648, "Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf"), "GSQ-RCO-IQ3_S")
        self.assertEqual(f(17_629_760_736, "Inkling-2-Mini-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL")
        for b in LEGACY:
            self.assertEqual(f(8_000_000_000, T.BUILDS[b]["file"]), b)


class InstalledFiles(unittest.TestCase):
    """Existing installs keep working: the 0.2.22 SQ-20 and Vision files, and every Unsloth quant already on disk."""

    def test_model_file_in_takes_the_new_name_then_the_old_one(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self.assertIsNone(F.model_file_in(td, "SQ-20"))
            (td / F.SQ20_FILE_V1).write_bytes(b"x")
            self.assertEqual(F.model_file_in(td, "SQ-20").name, F.SQ20_FILE_V1)
            (td / F.SQ20_FILE).write_bytes(b"x")
            self.assertEqual(F.model_file_in(td, "SQ-20").name, F.SQ20_FILE)
            self.assertIsNone(F.model_file_in(td, "SQ-15"))
            (td / "Inkling-2-Mini-UD-IQ3_XXS.gguf").write_bytes(b"x")
            self.assertEqual(F.model_file_in(td, "UD-IQ3_XXS").name, "Inkling-2-Mini-UD-IQ3_XXS.gguf")

    def test_vision_in_takes_either_file_the_new_one_first(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            self.assertIsNone(F.vision_in(a, b))
            (b / F.VISION_FILE_V1).write_bytes(b"x")
            self.assertEqual(F.vision_in(a, b), b / F.VISION_FILE_V1)
            (a / F.VISION_FILE).write_bytes(b"x")
            self.assertEqual(F.vision_in(a, b), a / F.VISION_FILE)
            (a / F.VISION_FILE_V1).write_bytes(b"x")
            self.assertEqual(F.vision_in(a, b), a / F.VISION_FILE, "the new file first")

    def test_setup_finds_a_local_vision_shard_of_either_name(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / F.VISION_FILE_V1).write_bytes(b"x")
            self.assertEqual(setup.inkling_find_local_vision(td, td / "data", True), td / F.VISION_FILE_V1)

    def test_local_source_finds_the_old_sq20_file(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / F.SQ20_FILE_V1).write_bytes(b"x")
            a = argparse.Namespace(gguf_dir=str(td))
            got = setup.inkling_local_source(a, "SQ-20")
            self.assertEqual(got[0].name, F.SQ20_FILE_V1)
            self.assertFalse(got[2])

    def test_local_files_count_the_old_names_too(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / F.SQ20_FILE_V1).write_bytes(b"\0" * 8)
            self.assertEqual(F.local_files(td), {})                                          # not the pinned size: not counted
            with mock.patch.object(F, "LEGACY_FILES", {**F.LEGACY_FILES, F.SQ20_FILE_V1: (8, "0" * 64)}), mock.patch.object(F, "ALL_FILES", {**F.ALL_FILES, F.SQ20_FILE_V1: (8, "0" * 64)}):
                self.assertEqual(list(F.local_files(td, "SQ-20")), [F.SQ20_FILE_V1])
                self.assertEqual(F.local_files(td, "SQ-15"), {})

    def test_a_download_of_a_legacy_quant_is_refused_but_a_local_one_is_used(self):
        # setup's size check: --model UD-IQ3_XXS without the file in the models folder stops with the reason and the offered names
        a = argparse.Namespace(model="UD-IQ3_XXS", gguf_dir=None, models_dir=tempfile.gettempdir(), vision="no", drafter=None, yes=True, backend=None, context=32768,
                               gpu=None, no_start=True, host=None, api_key=None, browser=None, port=8080)
        said = []
        with mock.patch.object(setup, "inkling_folder", return_value=Path(tempfile.mkdtemp())), \
                mock.patch.object(setup, "data_folder_for", return_value=(Path(tempfile.mkdtemp()), "explicit")), \
                mock.patch.object(setup, "inkling_pc", return_value=(12.0, 32.0, False, False, None)), \
                mock.patch.object(setup, "fail", side_effect=lambda m, h=None, **k: (said.append(str(m) + " " + str(h)), (_ for _ in ()).throw(SystemExit(1)))[1]):
            with self.assertRaises(SystemExit):
                setup.install_inkling2(a, F.FAMILY, {"index": 0, "name": "x", "vram_gb": 12.0}, False, False, None, False, 8080, 32.0, None)
        self.assertTrue(any("UD-IQ3_XXS is no longer offered" in m and "UD-Q4_K_XL" in m for m in said), said)


class NoDsparkFetch(unittest.TestCase):
    def test_the_dspark_downloads_are_gone_from_setup(self):
        for name in ("inkling_fetch_drafter", "inkling_find_local_dspark", "inkling_engine_reads_mtp"):
            self.assertFalse(hasattr(setup, name), name)
        src = Path(setup.__file__).read_text(encoding="utf-8")
        body = src[src.index("def install_inkling2("):src.index("def preview_family(")]
        for word in ("MTP_FILE", "DSPARK_GGUF_FILE", "dspark_path", "--dspark", "inkling_fetch_drafter"):
            self.assertNotIn(word, body, word)
        self.assertIn("inkling_fetch_dflash2(", body)

    def test_nothing_offered_lists_a_dspark_file(self):
        for b in OFFERED:
            for n in F.files_of(b, True, dspark=True, drafter=True):
                self.assertNotIn("dspark", n.lower())
                self.assertNotEqual(n, F.MTP_FILE)
        community = setup.INKLING_COMMUNITY_SOURCES
        for key in ("gguf", "gsq"):
            self.assertNotIn("dspark", community[key])
            self.assertEqual(community[key]["dflash2"], F.DFLASH2_FILE)
        self.assertEqual(community["gsq"]["repo"], F.GSQ_REPO)
        self.assertEqual(community["gguf"]["repo"], F.UNSLOTH_REPO)
        self.assertEqual(list(community["gguf"]["models"]), ["UD-Q4_K_XL"])

    def test_the_old_files_are_still_recognised_never_fetched(self):
        self.assertIn(F.DSPARK_GGUF_FILE, F.LEGACY_FILES)
        self.assertIn(F.MTP_FILE, F.DRAFTER_FILES)

    def test_the_drafting_choices_are_dflash2_and_off(self):
        self.assertEqual(F.DEFAULT_DRAFTER, "dflash2")
        self.assertEqual(F.DRAFTERS, ("off", "dflash2"))


def _sidecar_file(folder: Path) -> Path:
    p = folder / F.DFLASH2_FILE
    p.write_bytes(b"x")
    return p


class Dflash2ForEveryModel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _install(self, model, *, sidecar="fetched", drafter=None, own=False):
        """install_inkling2 for `model` with every download faked: returns (cfg, the file names fetched, the DFlash 2 fetch calls)."""
        data_dir = self.root / "data"
        install_dir = self.root / "install"
        folder = data_dir / "models" / F.FAMILY
        for d in (data_dir, install_dir, folder):
            d.mkdir(parents=True, exist_ok=True)
        fetched, df2 = [], []

        def fetch(a, name, fld, own_):
            fetched.append(name)
            (Path(fld)).mkdir(parents=True, exist_ok=True)
            (Path(fld) / name).write_bytes(b"v")
            return Path(fld) / name

        def fetch_df2(a, fld, own_):
            df2.append(model)
            if sidecar == "fetched":
                (Path(fld) / F.DFLASH2_FILE).write_bytes(b"x")
                return Path(fld) / F.DFLASH2_FILE
            return None

        gguf_dir = None
        if own:
            ownd = self.root / "own"
            ownd.mkdir(exist_ok=True)
            (ownd / T.BUILDS[model]["file"]).write_bytes(b"GGUF\x03\x00\x00\x00" if model not in T.STRATA_ORDER else b"STRATA01" + b"\0" * 24)
            (ownd / F.VISION_FILE).write_bytes(b"v")
            gguf_dir = str(ownd / T.BUILDS[model]["file"])
        a = argparse.Namespace(model=model, family=F.FAMILY, context=32768, vision="yes", yes=True, no_start=True, gguf_dir=gguf_dir, models_dir=str(data_dir / "models"),
                               data_dir=str(data_dir), backend=None, gpu=None, host=None, api_key=None, browser=None, port=8080, drafter=drafter)
        eng = self.root / "eng"
        eng.mkdir(exist_ok=True)
        gpu = {"index": 0, "name": "RTX 5070", "vendor": "nvidia", "vram_gb": 12.0, "count": 1}
        with mock.patch.object(setup, "home", return_value=install_dir), \
                mock.patch.object(setup, "data_folder_for", return_value=(data_dir, "explicit")), \
                mock.patch.object(setup, "pip_install"), \
                mock.patch.object(setup, "free_gb", return_value=1000.0), \
                mock.patch.object(setup, "engine_step", return_value=(None, eng, {"component": "strata-inkling"}, [], gpu, "none")), \
                mock.patch.object(setup, "engine_exe", return_value="component:strata-inkling"), \
                mock.patch.object(setup, "inkling_fetch", side_effect=fetch), \
                mock.patch.object(setup, "inkling_fetch_dflash2", side_effect=fetch_df2), \
                mock.patch.object(setup, "inkling_tokenizer_dir", return_value=self.root / "tok"), \
                mock.patch.object(setup, "write_run_script", return_value=install_dir / "run.bat"):
            rc = setup.install_inkling2(a, F.FAMILY, gpu, False, False, None, False, 8080, 32.0, None)
        self.assertEqual(rc, 0)
        cfg = json.loads((install_dir / f"strata-inkling-2-mini-{model.lower()}.json").read_text(encoding="utf-8"))
        return cfg, fetched, df2

    def test_every_offered_build_fetches_the_dflash2_file_and_never_a_dspark_file(self):
        for model in OFFERED:
            cfg, fetched, df2 = self._install(model)
            self.assertEqual(df2, [model], model)
            self.assertEqual(fetched, F.files_of(model, True), model)                      # the model and the Vision shard; the sidecar has its own step
            self.assertNotIn("--dspark", cfg["args"], model)
            self.assertNotIn("--draft", cfg["args"], model)                                  # drafting "auto": the probe decides, nothing is forced off
            self.assertEqual(cfg["drafting"], "auto", model)
            self.assertFalse(any("dspark" in str(x).lower() or "mtp.safetensors" in str(x) for x in cfg["args"]), model)
            self.assertEqual(Path(cfg["args"][cfg["args"].index("--native") + 1]).name, T.BUILDS[model]["file"])
            self.assertEqual(Path(cfg["args"][cfg["args"].index("--vision") + 1]).name, F.VISION_FILE)

    def test_without_the_sidecar_the_model_runs_without_a_drafter(self):
        for model in OFFERED:
            cfg, fetched, df2 = self._install(model, sidecar="missing")
            self.assertEqual(df2, [model])
            self.assertEqual(cfg["drafting"], "off", model)
            self.assertEqual(cfg["args"][cfg["args"].index("--draft") + 1], "off", model)
            self.assertNotIn("--dspark", cfg["args"], model)

    def test_drafter_off_fetches_nothing_extra(self):
        cfg, fetched, df2 = self._install("GSQ-RCO-IQ3_S", drafter="off")
        self.assertEqual(df2, [])
        self.assertEqual(cfg["drafting"], "off")

    def test_dflash2_is_asked_for_by_name_and_the_old_word_still_means_on(self):
        for word in ("dflash2", "dspark"):
            cfg, _f, _d = self._install("UD-Q4_K_XL", drafter=word)
            self.assertEqual(cfg["drafting"], F.DEFAULT_DRAFTER, word)

    def test_the_persons_own_files_use_a_sidecar_beside_them_or_run_without_a_drafter(self):
        for model in ("UD-Q4_K_XL", "GSQ-RCO-IQ3_S", "SQ-15"):
            cfg, fetched, df2 = self._install(model, own=True, sidecar="missing")
            self.assertEqual(cfg["drafting"], "off", model)
            self.assertEqual(df2, [model])

    def test_update_fetches_it_for_every_installed_inkling_model(self):
        for name in (F.SQ20_FILE, F.SQ20_FILE_V1, "Inkling-2-Mini-SQ-15.strata", F.GSQ_IQ3S_FILE, F.UNSLOTH_OFFERED_FILE, "Inkling-2-Mini-UD-IQ3_XXS.gguf"):
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                models = td / "models"
                folder = models / F.FAMILY
                folder.mkdir(parents=True)
                model = folder / name
                model.write_bytes(b"x")
                cfg = {"arch": F.ARCH, "args": ["--native", str(model), "--max-context", "32768"]}
                a = mock.Mock(models_dir=str(models), gguf_dir=None)
                seen = []
                with mock.patch.object(setup, "inkling_fetch_dflash2", side_effect=lambda a_, f, own: seen.append((f.name, own)) or None):
                    setup.inkling_dflash2_for_config(a, cfg)
                self.assertEqual(seen, [(F.FAMILY, False)], name)

    def test_the_sidecar_is_found_beside_the_model_or_by_the_engines_own_variable(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model = td / F.GSQ_IQ3S_FILE
            self.assertIsNone(F.dflash2_sidecar(model, env={}))
            (td / F.DFLASH2_FILE).write_bytes(b"x")
            self.assertIsNone(F.dflash2_sidecar(model, env={}), "a file of another size (a partial download) is not used")
            with mock.patch.object(F, "DRAFTER_FILES", {**F.DRAFTER_FILES, F.DFLASH2_FILE: (1, "0" * 64)}):
                self.assertEqual(F.dflash2_sidecar(model, env={}), td / F.DFLASH2_FILE)
            other = td / "elsewhere.safetensors"
            other.write_bytes(b"y")
            self.assertEqual(F.dflash2_sidecar(model, env={"STRATA_INK_DFLASH2": str(other)}), other)
            self.assertIsNone(F.dflash2_sidecar(None, env={}))


class DrafterFlags(unittest.TestCase):
    """What the server starts the engine with: DFlash 2 when its file is there, `--draft off` when not, never a DSpark file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.td = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.model = self.td / F.GSQ_IQ3S_FILE
        self.model.write_bytes(b"x")
        self.cfg = {"arch": F.ARCH, "drafting": "dflash2", "backend": "cuda", "args": ["--native", str(self.model), "--max-context", "32768"]}
        self.pin = mock.patch.object(F, "DRAFTER_FILES", {**F.DRAFTER_FILES, F.DFLASH2_FILE: (1, "0" * 64)})
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.env = mock.patch.dict(os.environ, {"STRATA_INK_DFLASH2": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_with_the_file_drafting_is_dflash2(self):
        from serve import inkling_kind
        (self.td / F.DFLASH2_FILE).write_bytes(b"x")
        self.assertEqual(inkling_kind.drafter_args(self.cfg, {"drafter": "none"}), ["--draft", "dflash2"])
        self.assertEqual(inkling_kind.drafter_args(self.cfg, {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "dflash2", "--draft-beam", "1", "--draft-adapt"])

    def test_without_the_file_there_is_no_drafter_even_when_the_pack_carries_dspark(self):
        from serve import inkling_kind
        self.assertEqual(inkling_kind.drafter_args(self.cfg, {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "off"])
        self.assertEqual(inkling_kind.drafter_args(dict(self.cfg, drafting="auto"), None), ["--draft", "off"])

    def test_off_stays_off(self):
        from serve import inkling_kind
        (self.td / F.DFLASH2_FILE).write_bytes(b"x")
        self.assertEqual(inkling_kind.drafter_args(dict(self.cfg, drafting="off"), None), ["--draft", "off"])

    def test_engine_args_start_without_a_drafter_and_say_why(self):
        from serve import inkling_kind
        with mock.patch.object(inkling_kind, "probe", return_value={"arch": "x", "drafter": "dspark", "spec_opts": "1"}):
            args = inkling_kind.engine_args(dict(self.cfg), "/e/strata-inkling", announce=False)
        self.assertEqual(args[args.index("--draft") + 1], "off")
        (self.td / F.DFLASH2_FILE).write_bytes(b"x")
        with mock.patch.object(inkling_kind, "probe", return_value={"arch": "x", "drafter": "none", "spec_opts": "1"}):
            args = inkling_kind.engine_args(dict(self.cfg), "/e/strata-inkling", announce=False)
        self.assertEqual(args[args.index("--draft") + 1], "dflash2")
        self.assertNotIn("--dspark", args)

    def test_the_probe_is_not_run_without_a_drafter_to_measure(self):
        from serve import inkling_kind
        auto = dict(self.cfg, drafting="auto")
        self.assertFalse(inkling_kind.needs_drafting_probe(auto))
        (self.td / F.DFLASH2_FILE).write_bytes(b"x")
        self.assertTrue(inkling_kind.needs_drafting_probe(auto))

    def test_a_probe_made_without_the_file_does_not_describe_a_run_with_it(self):
        from strata_control import inkling_probe
        k0 = inkling_probe.placement_key(self.cfg)
        (self.td / F.DFLASH2_FILE).write_bytes(b"x")
        self.assertNotEqual(k0, inkling_probe.placement_key(self.cfg))
        self.assertEqual(inkling_probe.ON_ARGS[:2], ["--draft", "dflash2"])

    def test_the_status_says_dflash2_never_dspark(self):
        from serve import inkling_kind
        self.assertEqual(inkling_kind.drafting_info(["--draft", "dflash2"], {"drafting": "dspark"})["drafting"], "dflash2")
        self.assertEqual(inkling_kind.drafting_info(["--draft", "dflash2"], None)["drafting"], "dflash2")
        self.assertEqual(inkling_kind.drafting_info(["--draft", "off"], {"drafting": "dspark"})["drafting"], "off")
        self.assertEqual(inkling_kind.drafting_info(["--draft", "dflash2"], {"drafting": "off"})["drafting"], "off")


class Credits(unittest.TestCase):
    def test_dflash2_credit_names_tjm8874_only(self):
        notices = ROOT.parent / "notices"
        mit = (notices / "DFLASH2-TJM8874-MIT.txt").read_text(encoding="utf-8")
        self.assertIn("tjm8874", mit)
        text = (ROOT.parent / "docs" / "MODELS.md").read_text(encoding="utf-8")
        self.assertIn("Credit for DFlash 2 goes to tjm8874", text)
        i = text.index("The drafter: DFlash 2")
        paragraph = text[i:text.index("### Inkling-2-Mini: the 0.2.23 pins", i)]
        for other in ("Inco AI", "z-lab"):
            self.assertNotIn(other, paragraph, other)

    def test_the_docs_list_the_five_builds_and_the_gsq_credit(self):
        readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
        i = readme.index("<b>Inkling 2 Mini</b>")
        section = readme[i:readme.index("</details>", i)]
        self.assertIn("SQ-20 (19.37 GB)", section)
        self.assertIn("IQ3\\_S (11.43 GB)", section)
        self.assertIn("UD-Q4\\_K\\_XL (17.63 GB)", section)
        for gone in LEGACY:
            self.assertNotIn(gone.replace("_", "\\_"), section, gone)
        self.assertLess(section.index("**Strata**"), section.index("**GSQ-RCO**"))
        self.assertLess(section.index("**GSQ-RCO**"), section.index("**Unsloth**"))
        models = (ROOT.parent / "docs" / "MODELS.md").read_text(encoding="utf-8")
        self.assertIn("GSQ + RCO by IST Austria DASLab, Apache-2.0", models)


if __name__ == "__main__":
    unittest.main()
