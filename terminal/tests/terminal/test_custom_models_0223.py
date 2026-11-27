"""Custom builds (0.2.23): a person's own .gguf / .strata files of a model Strata's engines run, used in place under a name of their own.

Control side: the neutral names, the GGUF layout match with a file that is not a published build, the .strata manifest read (4 KiB and 16 KiB headers),
the distinct run config, the fit with the real file size, removal that keeps the weights, and the official download that stays strict.
Small synthetic files only; no model, no download, no GPU.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import setup                                                    # noqa: E402
from strata_control import custom_models, hy4_family, hy4_fit, inkling_family, inkling_fit, models, strata_manifest   # noqa: E402
from strata_control.profiles import ProfileError             # noqa: E402
from tests.terminal.strata_fixtures import hy4_strata, inkling_strata, synthetic_strata   # noqa: E402
from tests.terminal.test_inkling_gguf_import import make_inkling_gguf, write_fake_o200k_files   # noqa: E402
from tests.terminal.test_models import gguf_fixture          # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def folder(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True)
        return p


class Names(Base):
    def test_a_label_is_one_clean_line_and_a_slug_is_never_a_path(self):
        self.assertEqual(custom_models.clean_label("  My \x1b[31m fine-tune\n v2\x00 "), "My [31m fine-tune v2")
        self.assertEqual(custom_models.clean_label("x" * 500), "x" * custom_models.MAX_LABEL)
        self.assertEqual(custom_models.clean_label("", "Custom build"), "Custom build")
        for hostile in ("../../etc/passwd", "..\\..\\Windows\\x", "C:\\evil", "a/b/../c", "/abs/path", "...", "\u202eevil", "con.json"):
            s = custom_models.slug(hostile)
            self.assertRegex(s, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
            self.assertLessEqual(len(s), custom_models.MAX_SLUG)
        self.assertEqual(custom_models.slug("../../etc/passwd"), "etc-passwd")
        self.assertEqual(custom_models.slug("???"), "build")
        self.assertEqual(custom_models.slug("Ünïcode Mërge 7B"), "unicode-merge-7b")

    def test_the_run_tag_of_each_family_is_its_own_and_never_an_official_one(self):
        expect = {"qwen": "qwen-custom-mine", "unsloth": "unsloth-custom-mine", "qwen36": "qwen36-custom-mine",
                  "hy4-flash": "hy4-flash-custom-mine", "inkling-2-mini": "inkling-2-mini-custom-mine"}
        for family, tag in expect.items():
            self.assertEqual(custom_models.run_tag(family, "Mine"), tag)
            self.assertTrue(models.MODEL_ID.fullmatch(tag))
        with self.assertRaises(ValueError):
            custom_models.run_tag("llama", "x")                 # a family Strata's engines do not run

    def test_the_same_files_keep_their_tag_and_other_files_with_the_same_name_get_another(self):
        used = {"inkling-2-mini-custom-mine": str(self.root / "a.strata")}
        self.assertEqual(custom_models.run_tag("inkling-2-mini", "Mine", self.root / "a.strata", used), "inkling-2-mini-custom-mine")
        self.assertEqual(custom_models.run_tag("inkling-2-mini", "Mine", self.root / "b.strata", used), "inkling-2-mini-custom-mine-2")

    def test_run_fields_and_summary(self):
        f = custom_models.run_fields("My merge", "qwen36", "UD-Q4_K_XL", 123, "/x/y.gguf")
        self.assertEqual(f, {"custom": True, "display_name": "My merge", "base_family": "qwen36", "base_model": "UD-Q4_K_XL", "custom_bytes": 123, "custom_source": "/x/y.gguf"})
        s = custom_models.summary_fields(f)
        self.assertEqual((s["display_family"], s["badge"]), ("Qwen3.6-35B-A3B", "Custom"))
        self.assertEqual(custom_models.summary_fields({"model_name": "x"}), {})              # an official config has no custom row fields
        self.assertEqual(custom_models.model_alias("hy4-flash", "My Tune"), "hy4-flash-custom-my-tune")

    def test_every_supported_family_has_a_display_family_and_the_four_names_are_there(self):
        titles = {custom_models.DISPLAY_FAMILY[f] for f in custom_models.SUPPORTED_FAMILIES}
        self.assertTrue({"Qwen3.8-Flash-Next", "Qwen3.6-35B-A3B", "HY-4 Flash", "Inkling-2-Mini"} <= titles)


class StrataManifest(Base):
    def write(self, name: str, data: bytes) -> Path:
        p = self.root / name
        p.write_bytes(data)
        return p

    def test_both_header_layouts_are_read_and_name_the_family_whatever_the_file_is_called(self):
        for k16 in (False, True):
            for maker, family in ((inkling_strata, "inkling-2-mini"), (hy4_strata, "hy4-flash")):
                p = self.write(f"my-own-merge-{family}-{int(k16)}.strata".replace("inkling", "x").replace("hy4", "y"), maker(k16=k16))
                info = strata_manifest.inspect(p)
                self.assertEqual(info["layout"], "16k" if k16 else "4k")
                self.assertEqual(info["family"], family, p.name)
                self.assertTrue(info["manifest_read"])
                self.assertFalse(info["guessed"])

    def test_a_manifest_naming_no_supported_family_is_not_taken(self):
        p = self.write("tiny.strata", synthetic_strata({"arch": "llama", "config": {"model_type": "llama"}}))
        self.assertIsNone(strata_manifest.inspect(p)["family"])

    def test_not_a_strata_or_a_cut_short_file_is_refused_in_words(self):
        with self.assertRaises(ValueError):
            strata_manifest.inspect(self.write("a.strata", b"GGUF" + b"\0" * 5000))
        with self.assertRaises(ValueError):
            strata_manifest.inspect(self.write("b.strata", inkling_strata(k16=True)[:5000]))       # the manifest lies beyond the end

    def test_a_manifest_this_python_cannot_unpack_falls_back_to_the_name_then_the_size_and_says_so(self):
        p = self.write("renamed-hy4-tune.strata", hy4_strata())
        with mock.patch.object(strata_manifest, "_decompress", side_effect=strata_manifest.NeedsZstd("no zstd")), \
             mock.patch.object(strata_manifest, "_engine_programs", return_value=[]):                  # no engine installed: only the guess is left
            by_name = strata_manifest.inspect(p)
            self.assertEqual((by_name["family"], by_name["guessed"], by_name["manifest_read"]), ("hy4-flash", True, False))
            plain = strata_manifest.inspect(self.write("plain.strata", inkling_strata()))
            self.assertEqual(plain["family"], "inkling-2-mini")                                   # small: Inkling-2-Mini's size class
            with mock.patch.object(strata_manifest, "INKLING_MAX_BYTES", 10), mock.patch.object(strata_manifest, "HY4_MIN_BYTES", 10 ** 9):
                self.assertIsNone(strata_manifest.inspect(self.write("mid.strata", inkling_strata()))["family"])


class AskTheEngine(Base):
    """No zstd in this Python: the installed engine's `--info FILE` (header only, no weights) names the family before any guess from the name or size."""
    INK = "arch=inkling2_loop layers=24 visits=48 hidden=2304 vocab=200058 experts=128 eos=200006 drafter=dspark images=1 parallel=64\n"
    HY4 = "arch=hy4flash layers=42 hidden=3072 vocab=120832 experts=512 eos=120025 drafter=dfly2 images=1\n"

    def ask(self, name: str, data: bytes, answers: dict, programs=None):
        p = self.root / name
        p.write_bytes(data)
        programs = [("inkling-2-mini", Path("/e/strata-inkling")), ("hy4-flash", Path("/e/strata-hy4"))] if programs is None else programs
        asked = []

        def run_info(exe, path, timeout=15.0):
            asked.append(exe.name)
            return answers.get(exe.name)
        with mock.patch.object(strata_manifest, "_decompress", side_effect=strata_manifest.NeedsZstd("no zstd")), \
             mock.patch.object(strata_manifest, "_engine_programs", return_value=programs), mock.patch.object(strata_manifest, "_run_info", run_info):
            return p, strata_manifest.inspect(p), asked

    def test_the_engine_names_the_family_whatever_the_file_is_called(self):
        _p, info, asked = self.ask("zzz-merge.strata", hy4_strata(), {"strata-hy4": self.HY4})            # the Inkling engine refuses a HY-4 file (no answer)
        self.assertEqual((info["family"], info["guessed"], info["read_by_engine"], info["arch"]), ("hy4-flash", False, True, "hy4flash"))
        self.assertEqual(info["engine_facts"]["experts"], "512")
        self.assertEqual(asked, ["strata-inkling", "strata-hy4"])
        _p, info, _ = self.ask("yyy-tune.strata", inkling_strata(), {"strata-inkling": self.INK})
        self.assertEqual((info["family"], info["guessed"]), ("inkling-2-mini", False))

    def test_the_name_only_chooses_which_engine_is_asked_first(self):
        _p, info, asked = self.ask("my-hy4-tune.strata", hy4_strata(), {"strata-hy4": self.HY4})
        self.assertEqual((info["family"], asked), ("hy4-flash", ["strata-hy4"]))

    def test_an_engine_that_names_nothing_leaves_the_guess_with_its_words(self):
        _p, info, _ = self.ask("renamed-hy4.strata", hy4_strata(), {"strata-hy4": "no arch line here\n", "strata-inkling": "arch=llama layers=2\n"})
        self.assertEqual((info["family"], info["guessed"], info["read_by_engine"] if "read_by_engine" in info else False), ("hy4-flash", True, False))
        _p, info, _ = self.ask("plain-name.strata", inkling_strata(), {}, programs=[])                  # no engine installed at all
        self.assertEqual((info["family"], info["guessed"]), ("inkling-2-mini", True))

    def test_an_answer_counts_only_for_the_family_of_the_engine_that_gave_it(self):
        # a wrong-family engine that printed an arch line of the other family must not decide: only the guess is left (the name here says Inkling)
        _p, info, _ = self.ask("an-inkling-named.strata", hy4_strata(), {"strata-hy4": self.INK}, programs=[("hy4-flash", Path("/e/strata-hy4"))])
        self.assertEqual((info["family"], info["guessed"], info.get("read_by_engine", False)), ("inkling-2-mini", True, False))
        _p, info, _ = self.ask("x.strata", inkling_strata(), {"strata-inkling": self.HY4, "strata-hy4": self.HY4})
        self.assertEqual((info["family"], info["read_by_engine"]), ("hy4-flash", True))

    def test_plan_import_uses_the_engines_answer_and_does_not_say_it_guessed(self):
        f = self.folder("eng")
        p = f / "my-custom-pack.strata"
        p.write_bytes(hy4_strata())
        with mock.patch.object(strata_manifest, "_decompress", side_effect=strata_manifest.NeedsZstd("no zstd")), \
             mock.patch.object(strata_manifest, "_engine_programs", return_value=[("hy4-flash", Path("/e/strata-hy4"))]), \
             mock.patch.object(strata_manifest, "_run_info", return_value=self.HY4):
            plan = models.plan_import(str(p))
        self.assertEqual((plan["file_set_status"], plan["candidate_selection"]["family"]), ("plausible", "hy4-flash"))
        self.assertFalse(any("took the model type from the file" in m for m in plan["messages"]))

    def test_parse_info_and_the_call_to_the_program(self):
        self.assertEqual(strata_manifest.parse_info("noise\n" + self.INK)["vocab"], "200058")
        self.assertEqual(strata_manifest.parse_info(None), {})
        import subprocess
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
            self.assertIsNone(strata_manifest._run_info(Path("/e/strata-hy4"), Path("/f.strata")))
        done = mock.Mock(stdout=self.HY4, returncode=0)
        with mock.patch("subprocess.run", return_value=done) as called:
            self.assertEqual(strata_manifest._run_info(Path("/e/strata-hy4"), Path("/f.strata")), self.HY4)
        self.assertEqual(called.call_args[0][0][1], "--info")
        import subprocess as sp
        self.assertEqual(called.call_args[1]["creationflags"], getattr(sp, "CREATE_NO_WINDOW", 0))     # no console window on Windows
        with mock.patch.object(sp, "CREATE_NO_WINDOW", 0x08000000, create=True), mock.patch("subprocess.run", return_value=done) as called:
            strata_manifest._run_info(Path("/e/strata-hy4"), Path("/f.strata"))
        self.assertEqual(called.call_args[1]["creationflags"], 0x08000000)
        self.assertIs(called.call_args[1]["stdin"], sp.DEVNULL)
        self.assertEqual(called.call_args[1]["timeout"], strata_manifest.INFO_TIMEOUT)


class StrataImportPlan(Base):
    def test_a_custom_named_inkling_strata_is_found_by_its_manifest_in_both_layouts(self):
        for k16 in (False, True):
            f = self.folder(f"ink{int(k16)}")
            p = f / "Thoughtful-Merge-v3.strata"
            p.write_bytes(inkling_strata(k16=k16, pad=4096 * 3))
            plan = models.plan_import(str(p))
            cand = plan["candidate_selection"]
            self.assertEqual(plan["file_set_status"], "plausible", plan["issues"])
            self.assertEqual((cand["family"], cand["basis"], cand["custom"], cand["file_kind"]), ("inkling-2-mini", "inkling2_strata", True, "strata"))
            self.assertEqual((cand["base_family"], cand["family_title"]), ("inkling-2-mini", "Inkling-2-Mini"))
            self.assertEqual(cand["suggested_label"], "Thoughtful-Merge-v3")
            self.assertEqual(plan["metadata"]["strata_layout"], "16k" if k16 else "4k")
            self.assertTrue(plan["custom"])

    def test_a_custom_named_hy4_strata_is_found_by_its_manifest_in_both_layouts(self):
        for k16 in (False, True):
            f = self.folder(f"hy{int(k16)}")
            p = f / "my-tune.strata"
            p.write_bytes(hy4_strata(k16=k16))
            plan = models.plan_import(str(p))
            cand = plan["candidate_selection"]
            self.assertEqual(plan["file_set_status"], "plausible", plan["issues"])
            self.assertEqual((cand["family"], cand["basis"], cand["custom"]), ("hy4-flash", "hy4_strata", True))
            self.assertIn(cand["model"], hy4_fit.BUILDS)
            self.assertEqual(plan["metadata"]["architecture"], "hy4flash")

    def test_a_manifest_name_is_the_suggested_label_and_a_folder_of_one_strata_is_planned(self):
        f = self.folder("named")
        (f / "x.strata").write_bytes(synthetic_strata({"arch": "inkling2_loop", "name": "Ink Special", "config": {}}))
        plan = models.plan_import(str(f))
        self.assertEqual(plan["candidate_selection"]["suggested_label"], "Ink Special")

    def test_the_picture_reader_beside_it_is_named_for_the_right_family(self):
        f = self.folder("withvision")
        (f / "tune.strata").write_bytes(hy4_strata())
        (f / hy4_family.VISION_FILE).write_bytes(synthetic_strata({"arch": "hy4flash"}, profile=3))
        plan = models.plan_import(str(f / "tune.strata"))
        self.assertTrue(plan["reads_pictures"])
        self.assertIn("HY-4 Flash", plan["projectors"][0]["plain"])

    def test_an_unsupported_manifest_and_a_damaged_header_are_refused_with_words(self):
        f = self.folder("bad")
        (f / "llama.strata").write_bytes(synthetic_strata({"arch": "llama"}))
        plan = models.plan_import(str(f / "llama.strata"))
        self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
        self.assertIn("unsupported_or_missing_architecture", plan["issues"])
        self.assertTrue(any("Custom model API" in m for m in plan["messages"]))
        (f / "cut.strata").write_bytes(b"STRATA\x00\x01" + b"\0" * 5000)
        plan = models.plan_import(str(f / "cut.strata"))
        self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
        self.assertTrue(any(".strata" in w for w in plan["plain_issues"]), plan["plain_issues"])

    def test_the_published_file_name_and_size_is_official_not_custom(self):
        f = self.folder("official")
        p = f / "Inkling-2-Mini-SQ-10.strata"
        p.write_bytes(inkling_strata())
        size = p.stat().st_size
        with mock.patch.dict(inkling_family.ALL_FILES, {"Inkling-2-Mini-SQ-10.strata": (size, None)}):
            plan = models.plan_import(str(p))
        self.assertFalse(plan["candidate_selection"]["custom"])
        self.assertEqual(plan["candidate_selection"]["model"], "SQ-10")


class GgufImportPlan(Base):
    def fake_root(self, table: dict) -> Path:
        root = self.root / "fake-root"
        (root / "data").mkdir(parents=True)
        (root / "data" / "gguf_fingerprints.json").write_text(json.dumps(table))
        return root

    def test_a_fine_tuned_qwen38_gguf_with_another_name_size_and_hash_is_accepted_as_the_same_layout(self):
        f = self.folder("q38")
        (f / "My-Flash-Next-Tune-IQ3_S.gguf").write_bytes(gguf_fixture(expert_type=42))
        table = {"qwen/IQ3_S": {"arch": "qwen4exp", "tensors": 1, "bytes": 999_999, "expert_bytes": {"Q2_0": 1, "IQ3_S": 1},
                                "files": ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"], "repo": "x"}}
        with mock.patch.object(models, "ROOT", self.fake_root(table)), mock.patch.object(models, "SUPPORTED_SELECTIONS", frozenset(table)):
            plan = models.plan_import(str(f / "My-Flash-Next-Tune-IQ3_S.gguf"))
        cand = plan["candidate_selection"]
        self.assertEqual(plan["issues"], [])
        self.assertEqual(cand["basis"], "same_layout_other_build")
        self.assertTrue(cand["custom"])
        self.assertEqual((cand["base_family"], cand["base_model"], cand["family_title"]), ("qwen", "IQ3_S", "Qwen3.8-Flash-Next"))
        self.assertEqual(cand["suggested_label"], "My-Flash-Next-Tune-IQ3_S")
        self.assertEqual(plan["file_kind"], "gguf")

    def test_the_makers_general_name_is_the_suggested_label_when_the_file_has_one(self):
        from strata_control import gguf_preflight

        def s(v):
            raw = v.encode()
            return struct.pack("<Q", len(raw)) + raw
        pairs = [("general.architecture", 8, s("inkling2_loop")), ("general.name", 8, s("Ink Remix 7")),
                 ("tokenizer.ggml.model", 8, s("gpt2")), ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + s("a")),
                 ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + s("a b"))]
        head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(pairs)) + b"".join(s(k) + struct.pack("<I", t) + v for k, t, v in pairs)
        head += s("blk.0.ffn_gate_exps.weight") + struct.pack("<IQIQ", 1, 32, 8, 0)
        f = self.folder("named")
        (f / "file-name-is-different.gguf").write_bytes(head + b"\0" * (-len(head) % 32) + b"\0" * 34)
        self.assertEqual(gguf_preflight.general_name(f / "file-name-is-different.gguf"), "Ink Remix 7")
        self.assertIsNone(gguf_preflight.general_name(f / "missing.gguf"))
        self.assertEqual(models.plan_import(str(f / "file-name-is-different.gguf"))["candidate_selection"]["suggested_label"], "Ink Remix 7")

    def test_an_exact_header_match_is_the_official_build_not_custom(self):
        f = self.folder("q38exact")
        p = f / "renamed-copy.gguf"
        p.write_bytes(gguf_fixture(expert_type=42))
        first = models.plan_import(str(p))
        table = {"qwen/IQ3_S": {"arch": "qwen4exp", "tensors": 1, "bytes": first["metadata"]["payload_bytes_required"], "expert_bytes": {"Q2_0": 1},
                                "files": ["x.gguf"], "repo": "x"}}
        with mock.patch.object(models, "ROOT", self.fake_root(table)), mock.patch.object(models, "SUPPORTED_SELECTIONS", frozenset(table)):
            plan = models.plan_import(str(p))
        self.assertEqual(plan["candidate_selection"]["basis"], "header_size_and_tensors")
        self.assertFalse(plan["candidate_selection"]["custom"])

    def test_a_qwen36_gguf_of_another_name_is_custom_and_the_engine_loader_rules_still_apply(self):
        f = self.folder("q36")
        (f / "Qwen3.6-community-merge-Q4_K.gguf").write_bytes(gguf_fixture(architecture="qwen35moe", expert_type=12))
        plan = models.plan_import(str(f / "Qwen3.6-community-merge-Q4_K.gguf"))
        cand = plan["candidate_selection"]
        self.assertEqual((plan["issues"], cand["family"], cand["custom"]), ([], "qwen36", True), plan["issues"])
        self.assertEqual(cand["family_title"], "Qwen3.6-35B-A3B")
        (f / "bad.gguf").write_bytes(gguf_fixture(architecture="qwen35moe", expert_type=39))          # a type the loader does not take
        self.assertIn("unsupported_or_missing_architecture", models.plan_import(str(f / "bad.gguf"))["issues"])

    def test_an_inkling_gguf_of_any_name_or_size_is_custom_and_a_published_name_with_another_size_too(self):
        f = self.folder("ink")
        (f / "my-ink-merge.gguf").write_bytes(make_inkling_gguf(architecture="inkling2_loop", expert_type=8))
        plan = models.plan_import(str(f / "my-ink-merge.gguf"))
        self.assertEqual((plan["issues"], plan["candidate_selection"]["basis"], plan["candidate_selection"]["custom"]), ([], "inkling2_gguf", True))
        (f / "Inkling-2-Mini-UD-Q4_K_XL.gguf").write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=12))
        plan = models.plan_import(str(f / "Inkling-2-Mini-UD-Q4_K_XL.gguf"))
        self.assertTrue(plan["candidate_selection"]["custom"])                                         # the name is Unsloth's, the bytes are not
        size = (f / "Inkling-2-Mini-UD-Q4_K_XL.gguf").stat().st_size
        with mock.patch.dict(inkling_family.ALL_FILES, {"Inkling-2-Mini-UD-Q4_K_XL.gguf": (size, None)}):
            self.assertFalse(models.plan_import(str(f / "Inkling-2-Mini-UD-Q4_K_XL.gguf"))["candidate_selection"]["custom"])

    def test_a_split_inkling_gguf_set_is_accepted_under_its_own_name(self):
        f = self.folder("split")
        (f / "tune-00001-of-00002.gguf").write_bytes(make_inkling_gguf(architecture="inkling2_loop", expert_type=8))
        (f / "tune-00002-of-00002.gguf").write_bytes(make_inkling_gguf(architecture=None, tokenizer=False, expert_type=8))
        plan = models.plan_import(str(f / "tune-00002-of-00002.gguf"))
        self.assertEqual((plan["file_set_status"], plan["file_count"], plan["candidate_selection"]["suggested_label"]), ("plausible", 2, "tune"))

    def test_hy4_has_no_gguf_loader_so_a_hy4_gguf_is_refused_with_the_custom_api_words(self):
        f = self.folder("hy4gguf")
        (f / "hy4.gguf").write_bytes(gguf_fixture(architecture="hy4flash", expert_type=42))
        plan = models.plan_import(str(f / "hy4.gguf"))
        self.assertIn("hy4.gguf: unsupported_architecture", plan["issues"])
        self.assertTrue(any("Custom model API" in m for m in plan["messages"]))

    def test_an_unrelated_architecture_keeps_the_custom_model_api_guidance(self):
        f = self.folder("llama")
        (f / "x.gguf").write_bytes(gguf_fixture(architecture="llama"))
        plan = models.plan_import(str(f / "x.gguf"))
        self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
        self.assertIn("llama-server", models.UNSUPPORTED_MODEL_TYPE)
        self.assertIn(models.UNSUPPORTED_MODEL_TYPE, plan["messages"])


class FitWithTheRealSize(Base):
    @staticmethod
    def total(placed: dict) -> float:
        return sum(float(placed[k]) for k in ("vram_gb", "ram_gb", "ssd_gb"))

    def cfg(self, family, base, nbytes):
        return {"custom": True, "base_family": family, "base_model": base, "custom_bytes": nbytes, "display_name": "x"}

    def test_inkling_fit_uses_the_actual_bytes_on_top_of_the_closest_build(self):
        base_gb = inkling_fit.BUILDS["SQ-15"]["file_gb"]
        same = custom_models.fit_placement(self.cfg("inkling-2-mini", "SQ-15", int(base_gb * 1e9)), 8.0, 16.0, 32768)
        big = custom_models.fit_placement(self.cfg("inkling-2-mini", "SQ-15", int((base_gb + 10) * 1e9)), 8.0, 16.0, 32768)
        plain = inkling_fit.place("SQ-15", 8.0, 16.0, False, 32768, False)
        self.assertEqual(same, plain)                                                   # the same size: the published build's own plan
        self.assertGreater(self.total(big), self.total(plain) + 9.0)                 # ten more GB of weights are placed (memory, RAM or SSD)
        self.assertNotIn(f"custom:SQ-15:{int((base_gb + 10) * 1e9)}", inkling_fit.BUILDS)    # the published list is untouched

    def test_hy4_fit_uses_the_actual_bytes(self):
        base_gb = hy4_fit.BUILDS["SQ-64"]["file_gb"]
        small = custom_models.fit_placement(self.cfg("hy4-flash", "SQ-64", int((base_gb - 20) * 1e9)), 12.0, 64.0, 32768)
        plain = hy4_fit.place("SQ-64", 12.0, 64.0, False, 32768, False)
        self.assertLess(self.total(small), self.total(plain) - 19.0)
        self.assertEqual(sorted(hy4_fit.BUILDS), ["SQ-128", "SQ-64", "SQ-96"])

    def test_the_derived_build_never_goes_below_its_dense_part(self):
        d = custom_models.derived_build(inkling_fit, "SQ-10", 1000)
        self.assertGreaterEqual(d["footprint_gb"], d["dense_gb"])
        self.assertGreater(d["bodies_gb"], 0)

    def test_qwen36_context_rule_uses_the_closest_size_and_a_family_without_a_size_rule_says_none(self):
        got = custom_models.own_path_inputs(self.cfg("qwen36", "UD-Q4_K_XL", 20 * 10 ** 9))
        self.assertIsNotNone(got)
        self.assertIsNone(custom_models.own_path_inputs(self.cfg("inkling-2-mini", "SQ-10", 1)))
        self.assertIsNone(custom_models.fit_placement(self.cfg("qwen36", "UD-Q4_K_XL", 1), 8.0, 16.0))
        self.assertIsNone(custom_models.fit_placement({"model_name": "official"}, 8.0, 16.0))


def namespace(**over):
    base = dict(drafter=None, model="SQ-15", family="inkling-2-mini", context=32768, vision="no", yes=True, no_start=True, gguf_dir=None, models_dir=None,
                data_dir=None, backend=None, gpu=None, host=None, api_key=None, browser=None, port=8080, custom_label=None, custom_source=None,
                rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram=None, feed=None, parallel=None)
    base.update(over)
    return argparse.Namespace(**base)


class SetupRunConfig(Base):
    def setUp(self):
        super().setUp()
        self.install_dir = self.folder("install")
        self.data_dir = self.folder("data")
        self.addCleanup(setup._CUSTOM.clear)
        setup._CUSTOM.clear()

    def install_inkling(self, a):
        eng = self.root / "eng"
        eng.mkdir(exist_ok=True)
        (eng / inkling_family.EXE).write_bytes(b"exe")
        gpu = {"index": 0, "name": "RTX 5070", "vendor": "nvidia", "vram_gb": 12.0, "count": 1}
        with mock.patch.object(setup, "home", return_value=self.install_dir), \
             mock.patch.object(setup, "data_folder_for", return_value=(self.data_dir, "explicit")), \
             mock.patch.object(setup, "pip_install"), \
             mock.patch.object(setup, "engine_step", return_value=(None, eng, {"component": "strata-inkling"}, [], gpu, "none")), \
             mock.patch.object(setup, "engine_exe", return_value="component:strata-inkling"), \
             mock.patch.object(setup, "inkling_fetch", side_effect=lambda a, name, folder, own: ((folder / name).write_bytes(b"v"), folder / name)[1]), \
             mock.patch.object(setup, "write_run_script", return_value=self.install_dir / "run.bat"):
            return setup.install_inkling2(a, "inkling-2-mini", gpu, False, False, None, False, 8080, 32.0, None)

    def test_a_custom_inkling_strata_gets_its_own_run_config_and_the_official_one_is_not_touched(self):
        folder = self.folder("mine")
        weights = folder / "Merged-Inkling-by-me.strata"
        weights.write_bytes(inkling_strata(pad=8192))
        write_fake_o200k_files(folder)
        official = self.install_dir / "strata-inkling-2-mini-sq-15.json"
        official.write_text(json.dumps({"exe": "x", "args": [], "tokenizer": "t", "model_name": "inkling-2-mini-sq-15"}))
        before = official.read_bytes()
        a = namespace(gguf_dir=str(weights), custom_label="Merged Inkling (mine)", custom_source=str(weights), models_dir=str(self.data_dir / "models"))
        self.assertEqual(self.install_inkling(a), 0)
        cfg_path = self.install_dir / "strata-inkling-2-mini-custom-merged-inkling-mine.json"
        self.assertTrue(cfg_path.is_file(), sorted(p.name for p in self.install_dir.iterdir()))
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual((cfg["custom"], cfg["display_name"], cfg["base_family"], cfg["base_model"]), (True, "Merged Inkling (mine)", "inkling-2-mini", "SQ-15"))
        self.assertEqual(cfg["custom_bytes"], weights.stat().st_size)
        self.assertEqual(cfg["custom_source"], str(weights))
        self.assertEqual(cfg["model_name"], "inkling-2-mini-custom-merged-inkling-mine")      # the alias /v1/models lists
        self.assertIn(str(weights), cfg["args"])                                               # used in place
        self.assertEqual(official.read_bytes(), before)                                        # the official install is untouched
        self.assertTrue(weights.is_file())

    def test_drafting_stays_auto_for_a_custom_inkling_and_off_without_the_sidecar(self):
        folder = self.folder("mine2")
        weights = folder / "tune.strata"
        weights.write_bytes(inkling_strata())
        write_fake_o200k_files(folder)
        a = namespace(gguf_dir=str(weights), custom_label="Tune", custom_source=str(weights), models_dir=str(self.data_dir / "models"))
        self.install_inkling(a)
        cfg = json.loads((self.install_dir / "strata-inkling-2-mini-custom-tune.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["drafting"], "off")                                               # no dflash2.safetensors beside it: the existing rule
        (folder / inkling_family.DFLASH2_FILE).write_bytes(b"x")
        with mock.patch.object(inkling_family, "dflash2_sidecar", return_value=folder / inkling_family.DFLASH2_FILE), \
             mock.patch.object(setup, "inkling_fetch_dflash2", return_value=folder / inkling_family.DFLASH2_FILE):
            self.install_inkling(a)
        cfg = json.loads((self.install_dir / "strata-inkling-2-mini-custom-tune.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["drafting"], "auto")

    def test_two_custom_builds_and_the_official_one_coexist(self):
        for name, label in (("one.strata", "Same name"), ("two.strata", "Same name")):
            folder = self.folder(name.split(".")[0])
            (folder / name).write_bytes(inkling_strata())
            write_fake_o200k_files(folder)
            self.install_inkling(namespace(gguf_dir=str(folder / name), custom_label=label, custom_source=str(folder / name), models_dir=str(self.data_dir / "models")))
        files = sorted(p.name for p in self.install_dir.glob("strata-*.json"))
        self.assertEqual(files, ["strata-inkling-2-mini-custom-same-name-2.json", "strata-inkling-2-mini-custom-same-name.json"])
        # the same files again refresh their own run config and add none
        self.install_inkling(namespace(gguf_dir=str(self.root / "one" / "one.strata"), custom_label="Same name", custom_source=str(self.root / "one" / "one.strata"),
                                       models_dir=str(self.data_dir / "models")))
        self.assertEqual(len(list(self.install_dir.glob("strata-*.json"))), 2)

    def test_run_tag_without_a_label_is_the_official_tag(self):
        fam = setup.INKLING_FAMILIES["inkling-2-mini"]
        self.assertEqual(setup.run_tag(namespace(gguf_dir=str(self.root)), "inkling-2-mini", fam, "SQ-20"), "inkling-2-mini-SQ-20")
        self.assertEqual(setup.run_tag(namespace(), "inkling-2-mini", fam, "SQ-20"), "inkling-2-mini-SQ-20")
        self.assertFalse(setup._CUSTOM)

    def test_a_label_without_the_persons_own_files_does_nothing(self):
        fam = setup.INKLING_FAMILIES["inkling-2-mini"]
        self.assertFalse(setup.custom_requested(namespace(custom_label="x")))                 # a download never becomes a custom build
        self.assertEqual(setup.run_tag(namespace(custom_label="x"), "inkling-2-mini", fam, "SQ-20"), "inkling-2-mini-SQ-20")

    def test_hy4_custom_strata_is_used_in_place_with_its_own_config_and_no_pinned_hash_check(self):
        folder = self.folder("hy")
        weights = folder / "My-HY4-fine-tune.strata"
        weights.write_bytes(hy4_strata(pad=12345))
        before = weights.read_bytes()
        engram, vision = folder / "engram-stand-in", folder / "vision-stand-in"
        engram.write_bytes(b"e")
        vision.write_bytes(b"v")
        a = namespace(family="hy4-flash", model="SQ-64", gguf_dir=str(weights), custom_label="HY4 tune", custom_source=str(weights), vision="yes",
                      models_dir=str(self.data_dir / "models"), context=32768)
        lines = []
        gpu = {"vram_gb": 12.0, "count": 1, "index": 0, "name": "RTX"}
        with mock.patch.object(setup, "home", return_value=self.install_dir), mock.patch.object(setup, "pip_install"), \
             mock.patch.object(setup, "runtime_packages", return_value=[]), mock.patch.object(setup, "step"), \
             mock.patch.object(setup, "say", lambda m="": lines.append(str(m))), mock.patch.object(setup, "ok", lambda m: lines.append(str(m))), \
             mock.patch.object(setup, "warn", lambda m: lines.append("WARN " + str(m))), \
             mock.patch.object(setup, "engine_step", return_value=(None, self.root / "engine", {"component": "missing"}, [], gpu, "none")), \
             mock.patch.object(setup, "engine_exe", return_value="component:strata-hy4"), \
             mock.patch.object(setup, "data_folder_for", return_value=(self.data_dir, None)), \
             mock.patch.object(setup, "hy4_tokenizer_dir", return_value=self.root / "tok"), \
             mock.patch.object(setup, "hy4_companion", side_effect=lambda a_, n, f, d: engram if n == hy4_family.ENGRAM_FILE else vision), \
             mock.patch.object(setup, "write_run_script", return_value=Path("run.cmd")), mock.patch.object(setup, "explicit_args", return_value=[]), \
             mock.patch.object(setup, "free_gb", return_value=1000.0), \
             mock.patch.object(setup, "hy4_fetch", side_effect=AssertionError("the person's file is never fetched or hash-checked")):
            setup.install_hy4_flash(a, "hy4-flash", gpu, False, False, None, False, 8095, 64.0, None)
        cfg = json.loads((self.install_dir / "strata-hy4-flash-custom-hy4-tune.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["custom"], cfg["base_family"], cfg["base_model"], cfg["drafting"]), (True, "hy4-flash", "SQ-64", "auto"))
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], str(weights))
        self.assertEqual(cfg["args"][cfg["args"].index("--vision") + 1], str(vision))
        self.assertEqual(cfg["model_name"], "hy4-flash-custom-hy4-tune")
        self.assertEqual(weights.read_bytes(), before)
        self.assertTrue(any("not one of the files Strata publishes" in l for l in lines), lines)

    def test_a_damaged_own_strata_is_refused_and_left_alone(self):
        p = self.root / "cut.strata"
        p.write_bytes(b"STRATA\x00\x01" + b"\0" * 5000)
        with self.assertRaises(SystemExit):
            setup.own_strata_check(p)
        self.assertTrue(p.is_file())


class SetupQwenFamilies(Base):
    """The two GGUF families: a custom Qwen3.6 file through the real own-path steps, and the Flash-Next file set pick."""

    def setUp(self):
        super().setUp()
        setup._CUSTOM.clear()
        self.addCleanup(setup._CUSTOM.clear)

    def test_a_custom_qwen36_gguf_is_used_in_place_its_picture_reader_is_the_official_one_checked_strictly(self):
        install, data, models_dir = self.folder("install"), self.folder("data"), self.folder("models")
        mine = self.folder("mine")
        weights = mine / "Qwen3.6-community-merge.gguf"
        weights.write_bytes(gguf_fixture(architecture="qwen35moe", expert_type=12))
        official_gguf = mine / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"                     # the official file sits in the same folder: it must not be picked
        official_gguf.write_bytes(b"official")
        official_dir = models_dir / "qwen36-UD-Q4_K_XL"
        official_dir.mkdir()
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        reader = official_dir / fam["mmproj"]
        reader.write_bytes(b"reader")
        setup.mark(reader)
        eng = self.folder("eng")
        (eng / setup.VEXE).write_text("x")
        checks, cfg, calls = [], {}, []
        gpu = {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        a = namespace(family="qwen36", model="UD-Q4_K_XL", gguf_dir=str(mine), custom_label="Community merge", custom_source=str(weights), vision="cpu",
                      models_dir=str(models_dir), context=32768, vram_reserve_mib=None, check=False, low_ram="auto")
        patches = [mock.patch.object(setup, "_OWN_FIRST", weights), mock.patch.object(setup, "home", return_value=install),
                   mock.patch.object(setup, "verify_sha256", lambda p, size, sha, own=False, **k: checks.append((p.name, own))),
                   mock.patch.object(setup, "download", lambda *x, **k: calls.append(("download",) + x)),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0), mock.patch.object(setup, "pip_install", lambda *x, **k: None),
                   mock.patch.object(setup, "engine_step", lambda a_, g, hip, amd, ck, v: (self.root / "llama", eng, {"version": "0.2.12"}, [], g, v)),
                   mock.patch.object(setup, "engine_families", lambda e: {"qwen4exp", "qwen35moe"}),
                   mock.patch.object(setup, "run", lambda cmd, **k: calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder_for", return_value=(data, None)),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "recommend_pool_workers", lambda args: args)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), mock.patch.object(setup, "warn"), mock.patch.object(setup, "step"):
            rc = setup.install_own_path(a, "qwen36", gpu, False, [], 13, [], 8080, 64.0, None)
        self.assertEqual(rc, 0)
        written = install / "strata-qwen36-custom-community-merge.json"
        self.assertEqual([p.name for p in install.glob("strata-*.json")], [written.name])
        cfg.update(json.loads(written.read_text(encoding="utf-8")), path=written.name)
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], str(weights))      # the picked file, not the official one beside it
        self.assertIn((weights.name, True), checks)                                          # the person's file: own, no pinned size or hash
        self.assertIn((reader.name, False), checks)                                          # the official picture reader: strict
        self.assertEqual(cfg["args"][cfg["args"].index("--mmproj") + 1] if "--mmproj" in cfg["args"] else str(reader), str(reader))
        self.assertFalse([c for c in calls if c and c[0] == "download"])                      # the reader of an official install is reused, nothing is downloaded
        self.assertEqual(cfg["path"], "strata-qwen36-custom-community-merge.json")
        self.assertEqual((cfg["custom"], cfg["base_family"], cfg["base_model"]), (True, "qwen36", "UD-Q4_K_XL"))
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-custom-community-merge")
        self.assertEqual(cfg["custom_bytes"], weights.stat().st_size)
        self.assertEqual(official_gguf.read_bytes(), b"official")                              # the official file beside it is untouched

    def test_the_flash_next_file_set_is_the_one_the_person_picked_not_the_published_name_beside_it(self):
        fam = setup.FAMILIES["qwen"]
        folder = self.folder("sets")
        official = folder / setup.model_file(fam, "IQ3_S", 1)
        official.write_bytes(b"o")
        mine = folder / "Tuned-Flash-Next-00001-of-00002.gguf"
        mine.write_bytes(b"1")
        (folder / "Tuned-Flash-Next-00002-of-00002.gguf").write_bytes(b"22")
        with mock.patch.object(setup, "_OWN_FIRST", None):
            self.assertEqual(setup.gguf_dir_shards(folder, fam, "IQ3_S")[0], official)
        with mock.patch.object(setup, "_OWN_FIRST", mine):
            got = setup.gguf_dir_shards(folder, fam, "IQ3_S")
        self.assertEqual([p.name for p in got], ["Tuned-Flash-Next-00001-of-00002.gguf", "Tuned-Flash-Next-00002-of-00002.gguf"])
        self.assertEqual(setup.custom_weight_bytes(mine), 3)
        with mock.patch.object(setup, "_CUSTOM", {"label": "x"}):                             # a custom file is not refused by what its name says
            self.assertIsNone(setup.gguf_dir_problem(folder, mine, fam, "IQ3_S"))
        odd = folder / "Tuned-Q4_K_M-00001-of-00002.gguf"
        odd.write_bytes(b"1")
        self.assertIsNotNone(setup.gguf_dir_problem(folder, odd, fam, "IQ3_S"))               # an official-path import still is

    def test_label_flags_are_checked_before_anything_is_done(self):
        import io
        for argv, text in ((["--custom-label", "x"], "--gguf-dir"), (["--custom-label", "x", "--gguf-dir", "/tmp/a.gguf"], "--family and --model")):
            err = io.StringIO()
            with mock.patch.object(sys, "argv", ["setup.py", *argv]), mock.patch("sys.stderr", err), mock.patch.object(setup, "move_over_old_layout"):
                with self.assertRaises(SystemExit) as raised:
                    setup.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(text, err.getvalue())


class OfficialStaysStrict(Base):
    """Only the person's own file skips the pinned size and hash; a download with a wrong hash or size is still refused (and kept as .bad, never trusted)."""

    def make(self, content: bytes) -> Path:
        p = self.root / "official.gguf"
        p.write_bytes(content)
        return p

    def test_a_download_with_the_wrong_hash_is_rejected(self):
        p = self.make(b"x" * 4096)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)) as failed, mock.patch.object(setup, "say"):
            with self.assertRaises(SystemExit):
                setup.verify_sha256(p, 4096, "0" * 64, own=False)
        self.assertIn("does not match the pinned SHA-256", failed.call_args[0][0])
        self.assertFalse(p.exists())                                                     # kept as .bad, not trusted as the official file
        self.assertTrue(p.with_name(p.name + ".bad").exists())

    def test_a_download_with_the_wrong_size_is_rejected(self):
        p = self.make(b"x" * 100)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)), mock.patch.object(setup, "say"):
            with self.assertRaises(SystemExit):
                setup.verify_sha256(p, 4096, "0" * 64, own=False)

    def test_the_persons_own_whole_gguf_with_another_size_and_hash_is_used_and_kept(self):
        p = self.root / "mine.gguf"
        p.write_bytes(make_inkling_gguf(architecture="inkling2_loop", expert_type=8))
        before = p.read_bytes()
        with mock.patch.object(setup, "warn") as warned, mock.patch.object(setup, "fail", side_effect=SystemExit(1)):
            setup.verify_sha256(p, 123456789, "0" * 64, own=True)
        self.assertEqual(p.read_bytes(), before)
        self.assertTrue(warned.called)

    def test_the_hy4_official_fetch_still_checks_size_and_hash_but_a_companion_download_is_not_own(self):
        names = (hy4_family.ENGRAM_FILE,)
        folder = self.folder("models")
        (folder / names[0]).write_bytes(b"wrong size")
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)), mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), \
             mock.patch.object(setup, "warn"), mock.patch.object(setup, "home", return_value=self.root):
            with self.assertRaises(SystemExit):
                setup.hy4_verify(folder / names[0], names[0], False)                    # the published file's byte size is not met


class RemoveKeepsTheWeights(Base):
    def build(self, family: str, weights_name: str, in_strata_folder: bool, tag: str | None = None):
        from tools import strata_mcp
        install = self.folder("strata-" + (tag or family))
        weights_dir = install / "models" if in_strata_folder else self.folder("userfiles-" + (tag or family))
        weights_dir.mkdir(exist_ok=True)
        weights = weights_dir / weights_name
        weights.write_bytes(b"W" * 1000)
        shard = weights_dir / "mmproj-x.gguf"
        shard.write_bytes(b"P")
        cfg = {"exe": "component:strata-engine", "tokenizer": str(install / "tok"), "args": ["--native", str(weights), "--max-context", "32768"],
               "model_name": "alias", **custom_models.run_fields("Mine", family, "X", 1000, str(weights))}
        config = install / f"strata-{custom_models.tag_prefix(family)}custom-mine.json"
        config.write_text(json.dumps(cfg), encoding="utf-8")
        (install / config.name.replace(".json", ".log")).write_text("log")
        return strata_mcp, install, config, weights

    def plan_for(self, mcp, install, config):
        return mcp.Strata(install).removal_plan(config)

    def test_removal_plan_lists_only_the_run_config_and_strata_own_leftovers_for_every_family(self):
        for family, name in (("qwen", "Tune-IQ3_S-00001-of-00002.gguf"), ("qwen36", "Q36-tune.gguf"), ("hy4-flash", "hy4-tune.strata"),
                             ("inkling-2-mini", "ink-tune.strata"), ("inkling-2-mini", "ink-tune.gguf")):
            for inside in (False, True):
                mcp, install, config, weights = self.build(family, name, inside, tag=f"{family}-{int(inside)}-{name[-8:]}")
                plan = self.plan_for(mcp, install, config)
                removed = {Path(i["path"]).name for i in plan["remove"]}
                self.assertEqual(removed, {config.name, config.name.replace(".json", ".log")}, (family, name, inside, plan))
                self.assertTrue(any(k["path"] == str(weights) for k in plan["keep"]), plan["keep"])

    def test_the_cli_remove_deletes_only_the_run_config(self):
        from strata_cli.launch import custom
        mcp, install, config, weights = self.build("hy4-flash", "t.strata", True)
        before = weights.read_bytes()
        custom.remove_reference(config)
        self.assertFalse(config.exists())
        self.assertEqual(weights.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
