"""Tests for the Strata Manager's backend (gui/manager.py).

The Manager only reuses Strata's own logic (setup.py) and its own config format, so the tests cover:
  - reading existing configs (model_summary / discover, on temp copies - never the user's real ones)
  - the surgical edits Save performs (context/rope/KV, Vision, Low-RAM, network, GPU)
  - atomic saves with .bak backups, and that a saved config stays readable
  - the custom-GGUF name detection (shards, family, size)
  - a real HTTP round-trip against a temp Strata folder (GET /, /api/models, /api/save)

Pure stdlib + unittest; no GPU, no network, no downloads.

    python -m unittest gui.test_manager
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup                                          # noqa: E402
import gui.manager as mgr                              # noqa: E402

SHARD1 = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
SHARD2 = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
FAKE_GPU = {"index": 0, "name": "Fake GPU", "vram_gb": 12.0, "arch": "100",
            "driver": "600", "count": 1}


def sample_cfg(**over):
    """A minimal but real-looking strata-q2_0 config (a temp copy - never a user file)."""
    cfg = {
        "exe": "C:/fake/strata.exe",
        "args": ["--pack", "C:/fake/pack/q2_0", "--native", f"C:/fake/{SHARD1}",
                 "--ple-gguf", f"C:/fake/{SHARD2}", "--expert-cache", "auto",
                 "--max-context", "32768", "--kv", "int8"],
        "cwd": "C:/fake",
        "tokenizer": "C:/fake/pack/q2_0/tokenizer",
        "model_name": "qwen3.8-flash-next-q2_0",
        "log": "C:/fake/strata-q2_0.log",
        "port": 8080,
    }
    cfg.update(over)
    return cfg


class ArgsHelpers(unittest.TestCase):
    def test_set_arg_val_in_place(self):
        a = ["--a", "1", "--b", "2"]
        self.assertEqual(mgr.set_arg_val(a, "--b", "9"), ["--a", "1", "--b", "9"])
        self.assertEqual(mgr.set_arg_val(a, "--c", "3"), ["--a", "1", "--b", "2", "--c", "3"])
        self.assertEqual(mgr.arg_val(a, "--b"), "2")
        self.assertIsNone(mgr.arg_val(a, "--nope"))

    def test_drop_arg_pair_and_flag(self):
        self.assertEqual(mgr.drop_arg(["--kv", "int8", "--spec", "4"], "--kv"), ["--spec", "4"])
        self.assertEqual(mgr.drop_arg(["--vision", "--k", "1"], "--vision"), ["--k", "1"])
        self.assertEqual(mgr.drop_arg(["--x", "--y", "2"], "--x"), ["--y", "2"])


class ContextEdits(unittest.TestCase):
    def test_sets_max_context_and_adds_int8_kv_above_8k(self):
        a = mgr.apply_context(["--pack", "p"], 32768)
        self.assertEqual(mgr.arg_val(a, "--max-context"), "32768")
        self.assertEqual(mgr.arg_val(a, "--kv"), "int8")

    def test_below_8k_drops_kv(self):
        a = mgr.apply_context(["--kv", "q4_0", "--max-context", "65536"], 8192)
        self.assertIsNone(mgr.arg_val(a, "--kv"))
        self.assertEqual(mgr.arg_val(a, "--max-context"), "8192")

    def test_past_trained_adds_yarn_with_derived_factor(self):
        a = mgr.apply_context([], 393216)
        self.assertEqual(mgr.arg_val(a, "--rope-scaling"), "yarn")
        self.assertEqual(float(mgr.arg_val(a, "--rope-scale")), 1.5)   # 393216 / 262144
        inside = mgr.apply_context(["--rope-scaling", "yarn", "--rope-scale", "1.5"], 131072)
        self.assertEqual(mgr.arg_val(inside, "--rope-scaling"), "yarn")  # explicit choice is kept

    def test_kv_resident_dropped_below_64k(self):
        a = mgr.apply_context(["--kv-resident", "32768"], 32768)
        self.assertIsNone(mgr.arg_val(a, "--kv-resident"))

    def test_apply_kv_only_above_8k(self):
        self.assertEqual(mgr.apply_kv([], 4096, "q4_0"), [])
        self.assertIn("q4_0", mgr.apply_kv([], 16384, "q4_0"))


class VisionEdits(unittest.TestCase):
    def setUp(self):
        self.cfg = sample_cfg(vision={"exe": "C:/fake/strata-vision.exe", "mmproj": "C:/fake/mmproj.gguf",
                                      "model": f"C:/fake/{SHARD1}", "gpu": True, "max_tokens": 1024},
                              args=["--vision", "--vram-reserve-mib", "700", "--max-context", "32768"])

    def test_off_removes_section_and_flags(self):
        out = mgr.apply_vision(ROOT, self.cfg, "off")
        self.assertNotIn("vision", out)
        self.assertNotIn("--vision", out["args"])
        self.assertNotIn("--vram-reserve-mib", out["args"])
        self.assertEqual(mgr.vision_mode(out), "off")

    def test_gpu_cpu_switch_keeps_paths(self):
        g = mgr.apply_vision(ROOT, self.cfg, "gpu")
        self.assertTrue(g["vision"]["gpu"])
        self.assertEqual(g["vision"]["max_tokens"], setup.VISION["gpu"]["max_tokens"])
        c = mgr.apply_vision(ROOT, self.cfg, "cpu")
        self.assertFalse(c["vision"]["gpu"])
        self.assertEqual(c["vision"]["max_tokens"], setup.VISION["cpu"]["max_tokens"])
        self.assertIn("--vision", c["args"])
        self.assertIn("threads", c["vision"])

    def test_reconstructs_section_from_disk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "engine").mkdir()
            (root / "engine" / "strata-vision.exe").write_bytes(b"")
            shard = root / SHARD1
            shard.write_bytes(b"")
            (root / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"")
            cfg = sample_cfg(args=["--native", str(shard), "--max-context", "32768"])   # no vision section
            out = mgr.apply_vision(root, cfg, "gpu")
            v = out["vision"]
            self.assertTrue(v["gpu"])
            self.assertEqual(v["exe"], str(root / "engine" / "strata-vision.exe"))
            self.assertEqual(v["mmproj"], str(root / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"))
            self.assertEqual(v["model"], str(shard))

    def test_refuses_when_files_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = sample_cfg(args=["--native", str(root / "nope.gguf"), "--max-context", "32768"])
            with self.assertRaises(ValueError):
                mgr.apply_vision(root, cfg, "gpu")


class LowRamNetworkGpu(unittest.TestCase):
    def test_low_ram_flag_round_trip(self):
        cfg = sample_cfg(args=["--mmap-experts", "--max-context", "32768"])
        self.assertEqual(mgr.low_ram_mode(cfg), "mmap")
        off = mgr.apply_low_ram(cfg, False)
        self.assertEqual(mgr.low_ram_mode(off), "off")
        on = mgr.apply_low_ram(off, True, resident=True)
        self.assertEqual(mgr.low_ram_mode(on), "resident")
        on2 = mgr.apply_low_ram(off, True, resident=False)
        self.assertEqual(mgr.low_ram_mode(on2), "mmap")

    def test_network_fields(self):
        cfg = sample_cfg()
        out = mgr.apply_network(cfg, 9090, "0.0.0.0", "secret")
        self.assertEqual(out["port"], 9090)
        self.assertEqual(out["host"], "0.0.0.0")
        self.assertEqual(out["api_key"], "secret")
        back = mgr.apply_network(out, None, "127.0.0.1", "")
        self.assertNotIn("host", back)
        self.assertNotIn("api_key", back)

    def test_gpu(self):
        self.assertEqual(mgr.apply_gpu(sample_cfg(), 2)["gpu"], 2)
        self.assertNotIn("gpu", mgr.apply_gpu(sample_cfg(), "auto"))
        split = sample_cfg(gpu=[0, 1], gpus_asked=True)
        self.assertEqual(mgr.apply_gpu(split, 2)["gpu"], [0, 1])   # a layer split is left alone


class EditRoundTrip(unittest.TestCase):
    def test_full_change_set(self):
        cfg = sample_cfg()
        out = mgr.edit_config(ROOT, cfg, {"context": 131072, "kv": "q4_0", "vision": "off",
                                          "low_ram": True, "port": 8081, "host": "127.0.0.1",
                                          "api_key": "", "gpu": "auto"}, low_ram_resident=False)
        self.assertEqual(mgr.arg_val(out["args"], "--max-context"), "131072")
        self.assertEqual(mgr.arg_val(out["args"], "--kv"), "q4_0")
        self.assertEqual(mgr.low_ram_mode(out), "mmap")
        self.assertEqual(out["port"], 8081)
        # and the result is exactly what setup.py itself can read back
        self.assertEqual(setup.choices_from_config.__name__, "choices_from_config")


class SaveConfig(unittest.TestCase):
    def test_atomic_write_and_backup(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "strata-x.json"
            p.write_text(json.dumps(sample_cfg(), indent=1), encoding="utf-8")
            new = dict(sample_cfg())
            new["args"][new["args"].index("--max-context") + 1] = "65536"
            mgr.save_config(p, new)
            reread = json.loads(p.read_text(encoding="utf-8-sig"))
            self.assertEqual(mgr.arg_val(reread["args"], "--max-context"), "65536")
            backups = list(p.parent.glob("strata-x.json.bak-*"))
            self.assertEqual(len(backups), 1)                    # one .bak of the previous version
            self.assertIn("32768", backups[0].read_text(encoding="utf-8-sig"))

    def test_rewrite_same_content_makes_no_backup(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "strata-x.json"
            cfg = sample_cfg()
            mgr.save_config(p, cfg)
            mgr.save_config(p, cfg)
            self.assertEqual(len(list(p.parent.glob("strata-x.json.bak-*"))), 0)


class GgufDetection(unittest.TestCase):
    def _dir(self, *names):
        d = Path(tempfile.mkdtemp())
        for n in names:
            (d / n).write_bytes(b"x")
        return d

    def test_qwen_q2_0(self):
        d = self._dir(SHARD1, SHARD2)
        r = mgr.detect_gguf_dir(str(d))
        self.assertTrue(r["ok"])
        self.assertEqual(r["family"], "qwen")
        self.assertEqual(r["quant"], "Q2_0")
        self.assertEqual(r["tag"], "q2_0")

    def test_swift_iq3_xxs(self):
        d = self._dir("Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                      "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf")
        r = mgr.detect_gguf_dir(str(d))
        self.assertEqual(r["family"], "swift")
        self.assertEqual(r["quant"], "IQ3_XXS")

    def test_coder(self):
        d = self._dir("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
                      "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf")
        self.assertEqual(mgr.detect_gguf_dir(str(d))["family"], "coder")

    def test_no_shards(self):
        d = self._dir("readme.txt")
        self.assertFalse(mgr.detect_gguf_dir(str(d))["ok"])

    # the real report's pattern: the custom label (abliterated) sits BEFORE the size in the name,
    # and the size is read by setup.py's own GGUF_QUANT rule - not by a lucky substring
    def test_real_abliterated_pattern_uses_setup_quant_rule(self):
        d = self._dir(ABLITERATED_1, ABLITERATED_2)
        r = mgr.detect_gguf_dir(str(d))
        self.assertTrue(r["ok"])
        self.assertEqual(r["quant"], "Q2_0")
        self.assertEqual(r["variant"], "abliterated")
        self.assertEqual(r["tag"], "q2_0")

    def test_unsloth_ud_quant_from_setup_rule(self):
        d = self._dir("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf")
        r = mgr.detect_gguf_dir(str(d))
        self.assertEqual(r["quant"], "UD-Q4_K_XL")
        self.assertEqual(r["family"], "unsloth")

    def test_iq2_xs_is_not_q2_0(self):
        d = self._dir("Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
                      "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf")
        self.assertEqual(mgr.detect_gguf_dir(str(d))["quant"], "IQ2_XS")

    def test_size_inside_another_token_is_not_a_match(self):
        # IQ2_0 is not a Strata size and must never be read as Q2_0 (boundary-guarded scan)
        d = self._dir("Qwen3.8-Flash-Next-GSQ-RCO-IQ2_0-00001-of-00002.gguf",
                      "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_0-00002-of-00002.gguf")
        self.assertFalse(mgr.detect_gguf_dir(str(d))["ok"])

    def test_size_elsewhere_in_the_name_still_detects(self):
        # the size is not adjacent to the shard suffix: the MODELS scan still finds it (copy tag)
        d = self._dir("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-copy-00001-of-00002.gguf",
                      "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-copy-00002-of-00002.gguf")
        r = mgr.detect_gguf_dir(str(d))
        self.assertTrue(r["ok"])
        self.assertEqual(r["quant"], "Q2_0")
        self.assertEqual(r["variant"], "copy")

    def test_pasted_file_path_detects_its_folder(self):
        d = self._dir(SHARD1, SHARD2)
        r = mgr.detect_gguf_dir(str(d / SHARD1))          # the full .gguf path, as pasted
        self.assertTrue(r["ok"])
        self.assertEqual(Path(r["dir"]).resolve(), d.resolve())

    def test_dir_is_canonical_resolved(self):
        # trailing slash / forward slashes: the same folder, the same canonical dir (identity)
        d = self._dir(SHARD1, SHARD2)
        a = mgr.detect_gguf_dir(str(d) + "/")
        b = mgr.detect_gguf_dir(str(d).replace("\\", "/"))
        self.assertTrue(a["ok"] and b["ok"])
        self.assertEqual(a["dir"], b["dir"])
        self.assertEqual(Path(a["dir"]).resolve(), d.resolve())


ABLITERATED_1 = "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00001-of-00002.gguf"
ABLITERATED_2 = "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00002-of-00002.gguf"


def mk_shard_dir(parent: Path, name: str, *shard_names) -> Path:
    """A folder with the shards (empty bytes are enough: detection only reads names)."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    for s in shard_names:
        (d / s).write_bytes(b"x")
    return d


class GgufVariantDetection(unittest.TestCase):
    """The custom-build label that separates same-family+same-size GGUFs (the Manager's variant naming)."""

    def test_published_files_have_no_variant(self):
        d = mk_shard_dir(Path(tempfile.mkdtemp()), "Q2_0", SHARD1, SHARD2)
        det = mgr.detect_gguf_dir(str(d))
        self.assertIsNone(det["variant"])
        self.assertEqual(det["tag"], "q2_0")
        self.assertEqual(det["title"], "Qwen3.8-Flash-Next Q2_0")

    def test_abliterated_build_is_a_variant(self):
        # the report's real example: E:\Model\Q2_0-Abliterated
        d = mk_shard_dir(Path(tempfile.mkdtemp()), "Q2_0-Abliterated", ABLITERATED_1, ABLITERATED_2)
        det = mgr.detect_gguf_dir(str(d))
        self.assertTrue(det["ok"])
        self.assertEqual(det["family"], "qwen")
        self.assertEqual(det["quant"], "Q2_0")
        self.assertEqual(det["variant"], "abliterated")
        self.assertEqual(det["title"], "Qwen3.8-Flash-Next Q2_0 Abliterated")

    def test_variant_slug_is_safe_and_deterministic(self):
        d = mk_shard_dir(Path(tempfile.mkdtemp()), "Q2_0-Abliterated.2!", ABLITERATED_1, ABLITERATED_2)
        v = mgr.detect_gguf_dir(str(d))["variant"]
        self.assertRegex(v, r"^[a-z0-9-]+$")          # filesystem-safe: only letters, digits, dashes
        self.assertEqual(v, mgr.detect_gguf_dir(str(d))["variant"])   # deterministic

    def test_two_variants_are_distinct(self):
        base = Path(tempfile.mkdtemp())
        a = mk_shard_dir(base, "Q2_0-Abliterated", ABLITERATED_1, ABLITERATED_2)
        b = mk_shard_dir(base, "Q2_0-Fine-Tune-A",
                         "Qwen3.8-Flash-Next-GSQ-RCO-fine-tune-a-Q2_0-00001-of-00002.gguf",
                         "Qwen3.8-Flash-Next-GSQ-RCO-fine-tune-a-Q2_0-00002-of-00002.gguf")
        va, vb = mgr.detect_gguf_dir(str(a))["variant"], mgr.detect_gguf_dir(str(b))["variant"]
        self.assertTrue(va and vb and va != vb)

    def test_family_and_variant_in_choices_from_config(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "strata-q2_0-abliterated.json"
            p.write_text(json.dumps({"args": ["--max-context", "32768"]}), encoding="utf-8")
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"]), ("qwen", "Q2_0"))
            self.assertEqual(ch["variant"], "abliterated")
            # canonical reads stay model-only, exactly as before
            q = Path(td) / "strata-q2_0.json"
            q.write_text(json.dumps({"args": ["--max-context", "32768"]}), encoding="utf-8")
            self.assertIsNone(setup.choices_from_config(q)["variant"])
            # the Unsloth size's dash is not mistaken for a variant separator
            u = Path(td) / "strata-unsloth-ud-q4_k_xl.json"
            u.write_text(json.dumps({"args": ["--max-context", "8192"]}), encoding="utf-8")
            ch3 = setup.choices_from_config(u)
            self.assertEqual((ch3["family"], ch3["model"], ch3["variant"]), ("unsloth", "UD-Q4_K_XL", None))
            v = Path(td) / "strata-unsloth-ud-q4_k_xl-fast.json"   # a Unsloth build of its own
            v.write_text(json.dumps({"args": ["--max-context", "8192"]}), encoding="utf-8")
            ch4 = setup.choices_from_config(v)
            self.assertEqual((ch4["model"], ch4["variant"]), ("UD-Q4_K_XL", "fast"))

    def test_variant_summary_gets_a_human_title(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "strata-q2_0-abliterated.json"
            p.write_text(json.dumps(sample_cfg(model_name="qwen3.8-flash-next-q2_0"), indent=1),
                         encoding="utf-8")
            s = mgr.model_summary(p)
            self.assertEqual(s["title"], "Qwen3.8-Flash-Next Q2_0 Abliterated")
            self.assertEqual(s["variant"], "abliterated")
            self.assertEqual(s["quant"], "Q2_0")


class VariantPrepare(unittest.TestCase):
    """prepare_gguf's identity rules, with setup.py's launch mocked (no model is loaded, nothing runs)."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.m = mgr.Manager(self.root)
        self.canonical = mk_shard_dir(self.root, "Q2_0", SHARD1, SHARD2)
        self.abliterated = mk_shard_dir(self.root, "Q2_0-Abliterated", ABLITERATED_1, ABLITERATED_2)
        self.cfg_path = self.root / "strata-q2_0.json"
        self.cfg_path.write_text(json.dumps(sample_cfg(
            args=["--pack", "p", "--native", str(self.canonical / SHARD1),
                  "--ple-gguf", str(self.canonical / SHARD2), "--max-context", "32768"]),
            indent=1), encoding="utf-8")
        self.original = self.cfg_path.read_bytes()
        self.fake_tool = {"pid": 4242, "log": str(self.root / "logs" / "p.log"),
                          "what": "preparing", "started": 1.0}

    def tearDown(self):
        self.td.cleanup()

    def _prepare(self, path, **more):
        with mock.patch.object(mgr, "run_tool", return_value=dict(self.fake_tool)) as rt:
            r = self.m.prepare_gguf({"path": str(path), **more})
            return r, rt

    # 1) the original config exists and the exact same GGUF files are selected -> already installed
    def test_same_files_are_already_installed(self):
        r, rt = self._prepare(self.canonical)
        self.assertTrue(r["ok"])
        self.assertEqual(r["already"], "strata-q2_0.json")
        self.assertEqual(r["config"], "strata-q2_0.json")
        rt.assert_not_called()                        # nothing was launched

    # 2) a DIFFERENT Q2_0 custom GGUF -> its own config; the original file is untouched
    def test_custom_variant_gets_its_own_config(self):
        r, rt = self._prepare(self.abliterated, context=65536)
        self.assertTrue(r["ok"])
        self.assertEqual(r["config"], "strata-q2_0-abliterated.json")
        self.assertEqual(r["title"], "Qwen3.8-Flash-Next Q2_0 Abliterated")
        self.assertIn("--variant", rt.call_args[0][0])
        self.assertEqual(rt.call_args[0][0][rt.call_args[0][0].index("--variant") + 1], "abliterated")
        self.assertIn("q2_0-abliterated", r["log"])
        # the exact reported bug: the original config was NOT treated as "already" and was NOT rewritten
        self.assertEqual(self.cfg_path.read_bytes(), self.original)

    # 3) two different custom Q2_0 variants -> both coexist under their own names
    def test_two_custom_variants_coexist(self):
        tune = mk_shard_dir(self.root, "Q2_0-Fine-Tune-A",
                            "Qwen3.8-Flash-Next-GSQ-RCO-fine-tune-a-Q2_0-00001-of-00002.gguf",
                            "Qwen3.8-Flash-Next-GSQ-RCO-fine-tune-a-Q2_0-00002-of-00002.gguf")
        ra, _ = self._prepare(self.abliterated)
        rb, _ = self._prepare(tune)
        self.assertNotEqual(ra["config"], rb["config"])
        self.assertEqual(ra["config"], "strata-q2_0-abliterated.json")
        self.assertEqual(rb["config"], "strata-q2_0-fine-tune-a.json")
        for name in (ra["config"], rb["config"]):
            self.assertRegex(name, r"^strata-[a-z0-9_-]+\.json$")     # both names stay filesystem-safe
        self.assertEqual(self.cfg_path.read_bytes(), self.original)   # 4) never overwritten

    # 4) the variant, once installed, is recognized by its GGUF paths (not by family+quant alone)
    def test_installed_variant_is_recognized_by_its_files(self):
        cfg = sample_cfg(args=["--pack", "p", "--native", str(self.abliterated / ABLITERATED_1),
                               "--ple-gguf", str(self.abliterated / ABLITERATED_2), "--max-context", "32768"])
        (self.root / "strata-q2_0-abliterated.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        r, rt = self._prepare(self.abliterated)
        self.assertTrue(r["ok"])
        self.assertEqual(r["already"], "strata-q2_0-abliterated.json")
        self.assertEqual(r["title"], "Qwen3.8-Flash-Next Q2_0 Abliterated")
        rt.assert_not_called()

    # 5) the generated names are deterministic, filesystem-safe and collision-resistant
    def test_names_are_deterministic_safe_and_collision_aware(self):
        name1, _ = self._prepare(self.abliterated)
        name2, _ = self._prepare(self.abliterated)
        self.assertEqual(name1["config"], name2["config"])
        self.assertRegex(name1["config"], r"^strata-[a-z0-9_-]+\.json$")
        # a second build whose readable name is already another INSTALLED build's: setup.py must never
        # overwrite - the hash suffix lives in the --variant itself, so the config it writes is unique
        other = mk_shard_dir(self.root, "Q2_0-Copy2", ABLITERATED_1, ABLITERATED_2)   # same files, own folder
        (other / ABLITERATED_1).write_bytes(b"other-tensor")
        (other / ABLITERATED_2).write_bytes(b"other-tensor")
        first_cfg = sample_cfg(args=["--pack", "p", "--native", str(self.abliterated / ABLITERATED_1),
                                     "--ple-gguf", str(self.abliterated / ABLITERATED_2), "--max-context", "32768"])
        (self.root / name1["config"]).write_text(json.dumps(first_cfg, indent=1), encoding="utf-8")
        r3, rt3 = self._prepare(other)
        self.assertNotEqual(r3["config"], name1["config"])           # never reuses the other build's config
        cmd3 = rt3.call_args[0][0]
        v = cmd3[cmd3.index("--variant") + 1]
        self.assertEqual(r3["config"], f"strata-q2_0-{v}.json")      # the name setup.py will write, exactly
        self.assertNotEqual(v, "abliterated")
        self.assertEqual(self.cfg_path.read_bytes(), self.original)

    # the canonical-looking files with the canonical slot taken (by OTHER files) get a folder/hash identity
    def test_canonical_lookalike_uses_folder_or_hash(self):
        clone = mk_shard_dir(self.root, "Q2_0-copy", SHARD1, SHARD2)
        r, kind = self._prepare(clone)
        self.assertTrue(r["ok"])
        self.assertEqual(r["config"], "strata-q2_0-copy.json")   # the folder name says it is a different copy
        r2, _ = self._prepare(clone)
        self.assertEqual(r["config"], r2["config"])              # deterministic
        self.assertEqual(self.cfg_path.read_bytes(), self.original)

    # no canonical config yet -> the custom build still gets its own variant config (never the canonical name)
    def test_first_install_custom_build_keeps_its_variant_name(self):
        self.cfg_path.unlink()
        r, rt = self._prepare(self.abliterated)
        self.assertEqual(r["config"], "strata-q2_0-abliterated.json")
        rt.assert_called_once()
        cmd = rt.call_args[0][0]
        self.assertTrue(any(a.endswith("setup.py") for a in cmd))
        self.assertIn("--gguf-dir", cmd)
        self.assertIn("--no-start", cmd)
        self.assertIn("--yes", cmd)

    # a fresh canonical install behaves exactly as before: the canonical name, no --variant
    def test_fresh_canonical_install_is_unchanged(self):
        self.cfg_path.unlink()
        r, rt = self._prepare(self.canonical)
        self.assertEqual(r["config"], "strata-q2_0.json")
        self.assertNotIn("--variant", rt.call_args[0][0])


class PrepareStatus(unittest.TestCase):
    """The status data behind the UI's preparing/failed/success states (no model is loaded)."""

    def _state(self, root: Path, tool: dict):
        st = {"server": {}, "tool": tool}
        state_path = root / "logs" / "manager.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(st), encoding="utf-8")
        return mgr.Manager(root).status_payload()

    def test_running_tool_is_alive_without_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir()
            with mock.patch.object(mgr.launcher, "pid_alive", return_value=True):
                p = self._state(root, {"pid": 1, "log": str(root / "logs" / "p.log"),
                                       "config": "strata-q2_0-abliterated.json",
                                       "title": "Qwen3.8-Flash-Next Q2_0 Abliterated"})
            self.assertTrue(p["tool"]["alive"])
            self.assertNotIn("failed", p["tool"])

    # 7) a failed prepare exposes the error (and the log path), it never looks like a silent success
    def test_failed_setup_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log = root / "logs" / "p.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("  [X]  the model has no per_layer_token_embd tensor\n\n"
                           "Setup stopped. Fix the item above and run it again.\n", encoding="utf-8")
            with mock.patch.object(mgr.launcher, "pid_alive", return_value=False):
                p = self._state(root, {"pid": 99, "log": str(log), "config": "strata-q2_0-bad.json",
                                       "title": "Qwen3.8-Flash-Next Q2_0 Bad"})
            self.assertFalse(p["tool"]["alive"])
            self.assertTrue(p["tool"]["failed"])
            self.assertFalse(p["tool"]["ready"])   # the config it promised was never written
            self.assertEqual(p["tool"]["log"], str(log))   # the UI shows the log path

    def test_successful_setup_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log = root / "logs" / "p.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("All set.\n", encoding="utf-8")
            (root / "strata-q2_0-abliterated.json").write_text("{}")   # setup actually wrote it
            with mock.patch.object(mgr.launcher, "pid_alive", return_value=False):
                p = self._state(root, {"pid": 99, "log": str(log), "config": "strata-q2_0-abliterated.json",
                                       "title": "Qwen3.8-Flash-Next Q2_0 Abliterated"})
            self.assertFalse(p["tool"]["alive"])
            self.assertFalse(p["tool"]["failed"])
            self.assertTrue(p["tool"]["ready"])

    def test_prepare_http_contract_refreshes_and_selects(self):
        """/api/prepare-gguf returns the config+title+log the UI selects on success; /api/models then lists
        both models (the original and the variant) - the data contract behind "refresh + select" and the
        success toast.  setup.py's launch is mocked: no model is loaded."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = sample_cfg()
            (root / "strata-q2_0.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            srv = mgr.make_server(root, 0)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            abliterated = mk_shard_dir(root, "Q2_0-Abliterated", ABLITERATED_1, ABLITERATED_2)
            with mock.patch.object(mgr, "run_tool",
                                   return_value={"pid": 7, "log": str(root / "logs" / "p.log"),
                                                 "what": "w", "started": 0.0}) as rt:
                with urllib.request.urlopen(urllib.request.Request(
                        base + "/api/prepare-gguf",
                        data=json.dumps({"path": str(abliterated)}).encode(),
                        headers={"Content-Type": "application/json"}), timeout=10) as r:
                    out = json.loads(r.read())
            self.assertTrue(out["ok"])
            self.assertEqual(out["config"], "strata-q2_0-abliterated.json")
            self.assertEqual(out["title"], "Qwen3.8-Flash-Next Q2_0 Abliterated")
            self.assertIn("manager-prepare-q2_0-abliterated.log", out["log"])
            rt.assert_called_once()
            # the variant's config lands (here: written by the test, as setup.py would) and BOTH are listed
            variant_cfg = dict(sample_cfg(
                args=["--pack", "p", "--native", str(abliterated / ABLITERATED_1),
                      "--ple-gguf", str(abliterated / ABLITERATED_2), "--max-context", "32768"]))
            (root / out["config"]).write_text(json.dumps(variant_cfg, indent=1), encoding="utf-8")
            with urllib.request.urlopen(base + "/api/models", timeout=10) as r:
                models = json.loads(r.read())["data"]["models"]
            names = {m["config"] for m in models}
            self.assertEqual(names, {"strata-q2_0.json", "strata-q2_0-abliterated.json"})
            by_name = {m["config"]: m for m in models}
            self.assertEqual(by_name["strata-q2_0-abliterated.json"]["title"],
                             "Qwen3.8-Flash-Next Q2_0 Abliterated")
            srv.shutdown(); srv.server_close()



class Discovery(unittest.TestCase):
    def test_model_summary_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "strata-q2_0.json"
            b = root / "strata-iq3_xxs.json"
            a.write_text(json.dumps(sample_cfg(model_name="qwen3.8-flash-next-q2_0"), indent=1),
                         encoding="utf-8")
            time = 1000000000
            os.utime(a, (time, time))
            b.write_text(json.dumps(sample_cfg(model_name="qwen3.8-flash-next-iq3_xxs",
                                               args=["--max-context", "131072", "--kv", "int8",
                                                     "--resident-experts"]), indent=1), encoding="utf-8")
            os.utime(b, (time + 10, time + 10))
            ms = mgr.discover(root)
            self.assertEqual([m["config"] for m in ms], ["strata-iq3_xxs.json", "strata-q2_0.json"])
            first = ms[0]
            self.assertEqual(first["quant"], "IQ3_XXS")
            self.assertEqual(first["context"], 131072)
            self.assertEqual(first["low_ram"], "resident")
            self.assertEqual(first["vision"], "off")


class HttpRoundTrip(unittest.TestCase):
    """A real Manager HTTP server against a temp Strata folder: the browser's own calls."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name)
        cfg = sample_cfg()
        (cls.root / "strata-q2_0.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        cls.server = mgr.make_server(cls.root, 0)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.td.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, r.read()

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())

    def test_ui_and_models_serve(self):
        status, html = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Strata Manager", html)
        status, _ = self.get("/app.css")
        self.assertEqual(status, 200)
        status, models = self.get("/api/models")
        self.assertEqual(json.loads(models)["data"]["models"][0]["config"], "strata-q2_0.json")

    def test_config_endpoint_edits_nothing(self):
        status, data = self.get("/api/config?config=strata-q2_0.json")
        self.assertEqual(status, 200)
        before = self.root.joinpath("strata-q2_0.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(data)["data"]["context"], 32768)
        self.assertEqual(self.root.joinpath("strata-q2_0.json").read_text(encoding="utf-8"), before)

    def test_save_round_trip_and_backup(self):
        with mock.patch.object(setup, "gpus", return_value=[FAKE_GPU]):
            with mock.patch.object(setup, "ram_gb", return_value=24.0):
                status, data = self.post("/api/save", {"config": "strata-q2_0.json", "context": 131072,
                                                       "kv": "q4_0", "vision": "off", "low_ram": True,
                                                       "port": 8080, "host": "127.0.0.1", "api_key": "", "gpu": 0})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        reread = json.loads(self.root.joinpath("strata-q2_0.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(mgr.arg_val(reread["args"], "--max-context"), "131072")
        self.assertEqual(mgr.arg_val(reread["args"], "--kv"), "q4_0")
        self.assertEqual(mgr.low_ram_mode(reread), "mmap")
        self.assertIn("gpu", reread)
        # the previous version is backed up and untouched
        backups = list(self.root.glob("strata-q2_0.json.bak-*"))
        self.assertEqual(len(backups), 1)
        old = json.loads(backups[0].read_text(encoding="utf-8-sig"))
        self.assertEqual(mgr.arg_val(old["args"], "--max-context"), "32768")
        # saving the same values again creates no second backup
        self.post("/api/save", {"config": "strata-q2_0.json", "context": 131072, "kv": "q4_0",
                                "vision": "off", "low_ram": True, "port": 8080, "host": "127.0.0.1",
                                "api_key": "", "gpu": 0})
        self.assertEqual(len(list(self.root.glob("strata-q2_0.json.bak-*"))), 1)


class LauncherTest(unittest.TestCase):
    """The supervisor is platform-neutral: it never branches on the OS; the platform adapters do.  These
    tests run the supervisor against a fake adapter, so they are valid on every OS."""

    class FakeLauncher:
        name = "fake"
        force_result = {"stopped": True, "forced": True}

        def __init__(self):
            self.spawned = []
            self.terminated = []
            self.force_terminated = []

        def spawn(self, cmd, cwd, log_file):
            self.spawned.append((cmd, cwd, log_file))
            return 4242

        def terminate(self, pid, grace_s):
            self.terminated.append(pid)
            return {"stopped": True, "forced": False}

        def terminate_force(self, pid):
            self.force_terminated.append(pid)
            return self.force_result

        @staticmethod
        def alive(pid):
            return False

    def setUp(self):
        self.fake = self.FakeLauncher()
        self.patch = mock.patch("gui.launcher.get_launcher", return_value=self.fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)

    def _write_state(self, **server):
        mgr.launcher.ServerState(self.root).write({"server": server, "tool": None})

    def test_platform_adapter_chosen_for_this_os(self):
        from gui.platforms import get_launcher
        if os.name == "nt":
            from gui.platforms.windows import WindowsLauncher
            self.assertIsInstance(get_launcher(), WindowsLauncher)
        else:
            from gui.platforms.linux import LinuxLauncher
            self.assertIsInstance(get_launcher(), LinuxLauncher)

    def test_start_command_is_platform_neutral(self):
        import json as _json
        exe = self.root / "strata"
        exe.write_bytes(b"")
        cfg = sample_cfg(exe=str(exe), args=[], port=18080)
        (self.root / "strata-x.json").write_text(_json.dumps(cfg), encoding="utf-8")
        r = mgr.start_server("strata-x.json", 18080, self.root)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["pid"], 4242)
        cmd, cwd, _ = self.fake.spawned[0]
        self.assertEqual(cwd, str(self.root))
        self.assertTrue(any(part.endswith("server.py") for part in cmd))   # run-*.bat/.sh command
        st = mgr.launcher.ServerState(self.root).read()
        self.assertEqual(st["server"]["pid"], 4242)
        self.assertEqual(st["server"]["platform"], "fake")

    def test_duplicate_start_refused(self):
        self._write_state(pid=9999, config="strata-x.json", port=18080, log="x")
        exe = self.root / "strata"
        exe.write_bytes(b"")
        import json as _json
        (self.root / "strata-x.json").write_text(_json.dumps(sample_cfg(exe=str(exe), port=18080)),
                                                 encoding="utf-8")
        with mock.patch("gui.launcher.probe_port", return_value=True):   # port answers = already running
            r = mgr.start_server("strata-x.json", 18080, self.root)
        self.assertFalse(r["ok"])
        self.assertIn("already", r["error"])

    def test_stop_clears_state_through_the_platform_adapter(self):
        self._write_state(pid=9999, config="strata-x.json", port=18080, log="no.log")
        r = mgr.stop_server(self.root)
        self.assertEqual(r["state"], "stopped")
        self.assertEqual(self.fake.terminated, [9999])
        st = mgr.launcher.ServerState(self.root).read()
        self.assertEqual(st["server"], None)
        self.assertEqual(st["lifecycle"], None)
        self.assertEqual(st["last_stop"]["elapsed_s"] >= 0.0, True)   # measured, not faked
        self.assertEqual(st["last_stop"]["forced"], False)

    def test_force_stop_uses_the_hard_path_only(self):
        self._write_state(pid=7777, config="strata-x.json", port=18080, log="no.log")
        r = mgr.force_stop_server(self.root)
        self.assertEqual(r["state"], "stopped")
        self.assertEqual(r["forced"], True)
        self.assertEqual(self.fake.force_terminated, [7777])
        self.assertEqual(self.fake.terminated, [])                    # graceful terminate never ran
        self.assertIsNone(mgr.launcher.ServerState(self.root).read()["server"])

    def test_lifecycle_shows_stopping_and_restarting(self):
        self._write_state(pid=9999, config="strata-x.json", port=18080, log="no.log")
        mgr.launcher.set_lifecycle(self.root, "stopping", "graceful")
        time.sleep(0.05)
        s = mgr.launcher.server_status(self.root)
        self.assertEqual(s["state"], "stopping")
        self.assertEqual(s["phase"], "graceful")
        self.assertGreater(s["elapsed"], 0.0)
        mgr.launcher.set_lifecycle(self.root, "restarting", "starting")
        self.assertEqual(mgr.launcher.server_status(self.root)["state"], "restarting")

    def test_stop_without_a_managed_server_is_a_noop(self):
        r = mgr.stop_server(self.root)
        self.assertEqual(r["state"], "stopped")
        self.assertEqual(self.fake.terminated, [])

    def test_windows_spawn_uses_a_hidden_console(self):
        if os.name != "nt":
            self.skipTest("Windows-launcher behavior checked on Windows")
        from gui.platforms.windows import WindowsLauncher
        calls = {}

        def fake_popen(*a, **kw):
            calls.update(kw)
            return type("P", (), {"pid": 1234})()

        fd, path = tempfile.mkstemp()
        os.close(fd)
        log = Path(path)
        self.addCleanup(lambda: log.unlink(missing_ok=True))
        with mock.patch("gui.platforms.windows.subprocess.Popen", fake_popen):
            pid = WindowsLauncher().spawn(["python", "-m", "x"], str(Path(".").resolve()), log)
        self.assertEqual(pid, 1234)
        flags = calls["creationflags"]
        self.assertTrue(flags & subprocess.CREATE_NEW_CONSOLE)     # a real console to inherit
        self.assertFalse(flags & subprocess.DETACHED_PROCESS)      # ...not console-less
        self.assertEqual(calls["startupinfo"].wShowWindow, 0)      # SW_HIDE: born invisible

    def test_linux_adapter_module_imports_everywhere(self):
        from gui.platforms.linux import LinuxLauncher   # importable on any OS (calls are OS-conditional)
        self.assertEqual(LinuxLauncher.name, "linux")


class NoWindowPolicy(unittest.TestCase):
    """The Manager runs under pythonw (no console): every short utility subprocess it starts (nvidia-smi,
    taskkill) must pass no-window creation flags, or Windows shows a transient black console window."""

    def test_system_info_hides_console_windows(self):
        """The Manager's own nvidia-smi call carries setup's no-window policy (CREATE_NO_WINDOW from a
        console-less parent, 0 from a console): it can never flash a black window under pythonw."""
        calls = []

        def fake_run(cmd, **kw):
            calls.append((cmd, kw))
            return type("R", (), {"stdout": ""})()

        with mock.patch.object(mgr.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(setup, "gpus", return_value=[FAKE_GPU]), \
                mock.patch.object(setup, "cpu_info", return_value=("Fake CPU 1", True, True)):
            mgr.system_info()
        self.assertTrue(calls)
        cmd, kw = calls[0]
        self.assertEqual(cmd[0], "nvidia-smi")
        expected = setup.child_flags()          # CREATE_NO_WINDOW here (pythonw/no console), 0 in a terminal
        self.assertEqual(kw.get("creationflags", 0), expected)

    def test_windows_taskkill_never_flashes_a_console(self):
        if os.name != "nt":
            self.skipTest("Windows launcher behavior checked on Windows")
        import subprocess as _sp
        from gui.platforms.windows import WindowsLauncher
        calls = []

        def fake_run(cmd, **kw):
            calls.append((cmd, kw))
            return type("R", (), {"stdout": ""})()

        with mock.patch("gui.platforms.windows.subprocess.run", side_effect=fake_run), \
                mock.patch.object(WindowsLauncher, "_wait_dead", return_value=True), \
                mock.patch.object(WindowsLauncher, "alive", return_value=False):
            WindowsLauncher().terminate(4242, 5.0)
            WindowsLauncher().terminate_force(4242)
        self.assertTrue(calls)
        for _cmd, kw in calls:
            self.assertTrue(kw["creationflags"] & _sp.CREATE_NO_WINDOW)

    def test_windows_engine_spawn_keeps_its_hidden_console(self):
        """The long-lived server/engine keeps the intentional hidden inherited console: the no-window policy
        is for the SHORT utilities, this spawn must still give the engine a (hidden) console to inherit."""
        if os.name != "nt":
            self.skipTest("Windows launcher behavior checked on Windows")
        import subprocess as _sp
        from gui.platforms.windows import WindowsLauncher
        calls = {}

        def fake_popen(*a, **kw):
            calls.update(kw)
            return type("P", (), {"pid": 1234})()

        fd, path = tempfile.mkstemp()
        os.close(fd)
        log = Path(path)
        self.addCleanup(lambda: log.unlink(missing_ok=True))
        with mock.patch("gui.platforms.windows.subprocess.Popen", side_effect=fake_popen):
            WindowsLauncher().spawn(["python", "-m", "x"], str(Path(".").resolve()), log)
        flags = calls["creationflags"]
        self.assertTrue(flags & _sp.CREATE_NEW_CONSOLE)          # a real console to inherit
        self.assertFalse(flags & _sp.CREATE_NO_WINDOW)           # ...not a console-less engine
        self.assertEqual(calls["startupinfo"].wShowWindow, 0)   # SW_HIDE: born invisible


class CrossPlatformPaths(unittest.TestCase):
    """The Manager's path handling must not assume drive letters or backslashes (Linux: /mnt/Storage/Model)."""

    def test_detect_works_with_forward_slash_paths(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        (d / SHARD1).write_bytes(b"")
        (d / SHARD2).write_bytes(b"")
        posix_style = str(d).replace("\\", "/")          # exactly what a Linux path string looks like
        r = mgr.detect_gguf_dir(posix_style)
        self.assertTrue(r["ok"])
        self.assertEqual(r["quant"], "Q2_0")

    def test_browse_uses_pathlib_only(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        (d / "sub").mkdir()
        b = mgr.Manager(ROOT).browse(str(d))
        self.assertNotIn("error", b)
        self.assertTrue(any(x.endswith("sub") for x in b["dirs"]))


class UnifiedGateway(unittest.TestCase):
    """The unified-page plumbing: the Manager serves the shared Strata UI kit and proxies the Strata
    server's endpoints on the SAME origin, staying alive (clean "offline") when Strata is down."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name)
        (cls.root / "engine").mkdir()
        (cls.root / "engine" / "strata.exe").write_bytes(b"")
        cls.cfg = sample_cfg(exe=str(cls.root / "engine" / "strata.exe"), api_key="top-secret")
        (cls.root / "strata-q2_0.json").write_text(json.dumps(cls.cfg, indent=1), encoding="utf-8")
        # a fake Strata server: answers the real endpoints the gateway forwards
        from http.server import BaseHTTPRequestHandler as BH
        class FakeStrata(BH):
            key = "top-secret"
            def log_message(self, *a): pass
            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def _authorized(self):
                ok = self.headers.get("Authorization") == "Bearer " + self.key
                if not ok:
                    self._json(401, {"error": {"message": "missing or wrong API key"}})
                return ok
            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/metrics":
                    if not self._authorized(): return
                    self._json(200, {"live": {"state": "idle"}, "engine": {"model": "fake"}})
                elif path in ("/api/health", "/health"):
                    if not self._authorized(): return       # proves detection sends the config's key
                    self._json(200, {"status": "ok", "model": "fake", "service": "strata"})
                elif path == "/v1/models":
                    if not self._authorized(): return
                    self._json(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
                else:
                    self._json(404, {"error": {"message": "nope"}})
            def do_POST(self):
                path = self.path.split("?")[0]
                if path == "/v1/chat/completions":
                    if not self._authorized(): return
                    if not self.headers.get("Content-Type", "").startswith("application/json"):
                        self._json(415, {"error": {"message": "send application/json"}}); return
                    n = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(n) or b"{}")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b"data: {" + json.dumps({"choices": [{"delta": {"content": "hello"}}]}).encode() + b"}\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._json(404, {"error": {"message": "nope"}})
        cls.fake = cls.cfg.get("port", 8080)
        # bind the fake Strata on a free port and point the config port at it
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.fake = s.getsockname()[1]
        cls.cfg["port"] = cls.fake
        (cls.root / "strata-q2_0.json").write_text(json.dumps(cls.cfg, indent=1), encoding="utf-8")
        import socketserver
        from http.server import ThreadingHTTPServer as _THS
        class S(_THS):
            daemon_threads = True
        cls.up = S(("127.0.0.1", cls.fake), FakeStrata)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        cls.server = mgr.make_server(cls.root, 0)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()
        cls.up.shutdown(); cls.up.server_close()
        cls.td.cleanup()

    def _get(self, path, key=None):
        req = urllib.request.Request(self.base + path)
        if key:
            req.add_header("Authorization", "Bearer " + key)
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()

    def test_unified_page_is_one_native_app(self):
        status, html = self._get("/")
        self.assertEqual(status, 200)
        text = html.decode("utf-8")
        for tab in ("tab-btn-manager", "tab-btn-chat", "tab-btn-monitor", "tab-btn-about",
                    "view-manager", "view-chat", "view-monitor", "view-about"):
            self.assertIn(tab, text)
        # the shared UI kit is served from serve/web, the Chat/Monitor/About code verbatim
        self.assertIn(b"serve/web/app.js", (self._get("/web/serve-app.js"))[1][:120])
        self.assertIn(b"Manager", (self._get("/web/app.js"))[1][:120])
        self.assertIn(b"--st-accent", (self._get("/web/tokens.css"))[1])
        status, _ = self._get("/fonts/outfit-latin-wght.woff2")
        self.assertEqual(status, 200)

    def test_common_strata_endpoint_is_proxied_with_the_config_key(self):
        status, body = self._get("/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["live"]["state"], "idle")
        self.assertNotIn(b"missing or wrong API key", body)     # the config key was injected, no CORS
        status, body = self._get("/v1/models")
        self.assertEqual(status, 200)
        self.assertIn(b'"object"', body)

    def test_chat_stream_flows_through_the_gateway(self):
        import urllib.request as ur
        req = ur.Request(self.base + "/v1/chat/completions",
                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                         headers={"Content-Type": "application/json"})
        with ur.urlopen(req, timeout=15) as r:
            body = r.read().decode()
        self.assertIn("hello", body)

    def test_offline_answer_when_strata_is_down(self):
        # a fresh Manager pointing at a port where nothing listens
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            busy = 14999
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=busy), indent=1), encoding="utf-8")
            srv = mgr.make_server(root, 0)
            # force the gateway target to the dead port: the fallback to the conventional 8080 would
            # otherwise pick up a real Strata running on this machine and make "offline" non-deterministic
            srv.manager.strata_port = lambda: busy
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                import urllib.error
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(base + "/health", timeout=10)
                self.assertEqual(cm.exception.code, 503)
                err = json.loads(cm.exception.read())
                self.assertTrue(err.get("offline"))
                # and the Manager's own API still answers: the page lives on
                with urllib.request.urlopen(base + "/api/status", timeout=10) as r:
                    self.assertEqual(json.loads(r.read())["data"]["server"]["state"], "stopped")
            finally:
                srv.shutdown(); srv.server_close()

    def test_externally_running_strata_is_detected(self):
        # (the gateway test's fake strata IS external: no Manager-owned server record)
        with urllib.request.urlopen(self.base + "/api/status", timeout=10) as r:
            body = json.loads(r.read())["data"]["server"]
        self.assertEqual(body["state"], "external")
        self.assertEqual(body["port"], self.fake)
        self.assertTrue(body.get("external"))

    def test_start_refused_while_external_runs(self):
        r = mgr.start_server("strata-q2_0.json", self.fake, self.root)
        self.assertFalse(r["ok"])
        self.assertIn("outside the Manager", r["error"])

    def test_random_process_on_the_port_is_not_strata(self):
        # port answers TCP but has no /api/health service=strata identity: not classified as external
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import socket
            from http.server import BaseHTTPRequestHandler as BH, HTTPServer
            class Plain(BH):
                def log_message(self, *a): pass
                def do_GET(self):
                    body = b"<html>not strata</html>"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                p = s.getsockname()[1]
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=p), indent=1), encoding="utf-8")
            srv = HTTPServer(("127.0.0.1", p), Plain)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                self.assertIsNone(mgr.launcher.external_strata(root))
                self.assertEqual(mgr.launcher.server_status(root)["state"], "stopped")
            finally:
                srv.shutdown(); srv.server_close()

    def test_openai_compatible_server_without_strata_identity_is_not_detected(self):
        # LM Studio / llama.cpp style: /v1/models with object+data, but no /api/health service=strata
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import urllib.request as ur
            from http.server import BaseHTTPRequestHandler as BH, HTTPServer
            class OpenAiCompatible(BH):
                def log_message(self, *a): pass
                def _json(self, code, obj):
                    body = json.dumps(obj).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def do_GET(self):
                    path = self.path.split("?")[0]
                    if path == "/v1/models":
                        self._json(200, {"object": "list",
                                         "data": [{"id": "gpt-oss-120b", "object": "model"}]})
                    elif path == "/api/health":
                        self._json(404, {"error": {"message": "no such endpoint"}})
                    else:
                        self._json(404, {"error": {"message": "nope"}})
            import socket
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                p = s.getsockname()[1]
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=p), indent=1), encoding="utf-8")
            srv = HTTPServer(("127.0.0.1", p), OpenAiCompatible)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                self.assertIsNone(mgr.launcher.external_strata(root))
                self.assertEqual(mgr.launcher.server_status(root)["state"], "stopped")
                # and /v1/models alone is fine for readiness, never for identity
                self.assertTrue(mgr.launcher.strata_ready(p))
                self.assertFalse(mgr.launcher.strata_health(p))
            finally:
                srv.shutdown(); srv.server_close()

    def test_key_protected_strata_is_detected_with_the_configured_key(self):
        # the class fake Strata demands the key even on /api/health: detection sends the config's key.
        self.assertTrue(mgr.launcher.strata_health(self.fake, api_key="top-secret"))
        self.assertFalse(mgr.launcher.strata_health(self.fake))            # wrong/no key -> not Strata
        self.assertFalse(mgr.launcher.strata_health(self.fake, api_key="wrong"))
        ext = mgr.launcher.external_strata(self.root)
        self.assertIsNotNone(ext)
        self.assertEqual(ext["port"], self.fake)
        self.assertTrue(ext.get("ready"))

    def test_manager_owned_server_takes_precedence(self):
        # the class fake is RUNNING and matches external_strata, but a live Manager-owned record wins
        mgr.launcher.ServerState(self.root).write({"server": {"pid": 4242,
                                                            "config": "strata-q2_0.json",
                                                            "port": self.fake,
                                                            "log": "x.serve.log"}, "tool": None})
        try:
            with mock.patch.object(mgr.launcher, "pid_alive", return_value=True):
                st = mgr.launcher.server_status(self.root)
            self.assertEqual(st["state"], "running")
            self.assertNotIn("external", st)
        finally:
            mgr.launcher.ServerState(self.root).write({"server": None, "tool": None})

    def test_external_disappears_back_to_stopped(self):
        # a fresh external Strata is detected, then its disappearance flips the status to Stopped
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import urllib.error
            import urllib.request as ur
            import socket
            from http.server import BaseHTTPRequestHandler as BH, HTTPServer
            class Ext(BH):
                def log_message(self, *a): pass
                def _json(self, code, obj):
                    body = json.dumps(obj).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def do_GET(self):
                    path = self.path.split("?")[0]
                    if path == "/api/health":
                        self._json(200, {"status": "ok", "service": "strata"})
                    elif path == "/v1/models":
                        self._json(200, {"object": "list", "data": []})
                    else:
                        self._json(404, {"error": {"message": "nope"}})
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                p = s.getsockname()[1]
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=p), indent=1), encoding="utf-8")
            srv = HTTPServer(("127.0.0.1", p), Ext)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            mgr_srv = mgr.make_server(root, 0)
            threading.Thread(target=mgr_srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{mgr_srv.server_address[1]}"
            try:
                with urllib.request.urlopen(base + "/api/status", timeout=10) as r:
                    body = json.loads(r.read())["data"]["server"]
                self.assertEqual(body["state"], "external")
                srv.shutdown(); srv.server_close()
                time.sleep(0.2)
                with urllib.request.urlopen(base + "/api/status", timeout=10) as r:
                    body = json.loads(r.read())["data"]["server"]
                self.assertEqual(body["state"], "stopped")
            finally:
                mgr_srv.shutdown(); mgr_srv.server_close()
                try:
                    srv.server_close()
                except Exception:
                    pass


class LauncherBat(unittest.TestCase):
    """Structural checks for START-MANAGER.bat: cmd parses it safely (a parenthesized IF with a
    closing parenthesis in its ECHO once produced the real ' . was unexpected at this time.' error)
    and pythonw is launched without a redirect that would lock a file pythonw never writes to."""

    def _bat(self) -> str:
        p = Path(__file__).resolve().parents[1] / "START-MANAGER.bat"
        return p.read_text(encoding="utf-8")

    def test_no_parenthesized_if_with_paren_in_echo(self):
        bat = self._bat()
        self.assertNotIn('if not exist ".venv\\Scripts\\pythonw.exe" (', bat)  # the old hazard
        self.assertNotIn(" (\r\n", bat)
        self.assertNotIn("(\n", bat)
        self.assertIn("goto no_pythonw", bat)          # goto style, not a block, per the fix
        self.assertIn(":no_pythonw", bat)

    def test_launches_pythonw_without_a_redirect_lock(self):
        bat = self._bat()
        self.assertIn("pythonw.exe", bat)              # no console window by design
        self.assertIn('start "" ".venv\\Scripts\\pythonw.exe" gui\\manager.py %*', bat)
        self.assertNotIn(">> \"logs\\manager.log\"", bat)   # the Manager writes its own log
        self.assertNotIn('"logs\\manager.log" 2>&1', bat)

    def test_exit_codes_are_explicit(self):
        bat = self._bat()
        self.assertIn("exit /b 0", bat)
        self.assertIn("exit /b 1", bat)


class SetupVariant(unittest.TestCase):
    """setup.py's --variant: the EXISTING pipeline, one extra flag; the config, pack and run script get the
    variant's own names, and the canonical strata-q2_0.json is neither written nor touched.  setup.main is
    run with every outside effect mocked (no GPU, no downloads, no engine start)."""

    class FakeGGUF:
        """The PLE table lives in shard 2 (the same shape the unsloth harness fakes)."""
        def __init__(self, path):
            self.tensors = [types.SimpleNamespace(name="per_layer_token_embd.weight")]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        self.downloads, self.runs = [], []
        self.shards = mk_shard_dir(self.t, "gguf-abliterated", ABLITERATED_1, ABLITERATED_2)

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, existing_canonical=None):
        eng = self.t / "engine"
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.36", "source": "local"}))
        found = [{"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "arch": "120",
                  "driver": "580.97"}]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_run(cmd, *a, **k):
            self.runs.append([str(x) for x in cmd])

        root_cfg = self.t / "strata-q2_0.json"
        if existing_canonical:
            root_cfg.write_text(json.dumps(sample_cfg(), indent=1), encoding="utf-8")
            canon_before = root_cfg.read_bytes()
        written_scripts = []
        argv = ["setup.py", "--family", "qwen", "--model", "Q2_0", "--gguf-dir", str(self.shards),
                "--context", "32768", "--no-start", "--yes", *argv]
        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: []),
            mock.patch.object(setup, "ram_gb", lambda: 63.7),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "whole_shard", lambda s: False),   # the stub shards are not the model itself
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "write_run_script",
                              lambda tag, cfg, port: written_scripts.append(tag)
                              or self.t / f"run-{tag.lower()}.bat"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=self.FakeGGUF)}),
            mock.patch.object(sys, "argv", argv),
            mock.patch("builtins.input", mock.Mock(side_effect=AssertionError("asked"))),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            try:
                code = setup.main()
            except SystemExit as e:                    # fail() calls sys.exit(1)
                code = e.code
        return code, root_cfg, (json.loads(root_cfg.read_text(encoding="utf-8-sig")) if root_cfg.exists() else None), \
            canon_before if existing_canonical else None, written_scripts

    def test_variant_config_pack_and_script_are_unique(self):
        code, _, _, _, written_scripts = self.main(["--variant", "abliterated"])
        self.assertEqual(code, 0)
        vcfg = self.t / "strata-q2_0-abliterated.json"
        self.assertTrue(vcfg.is_file(), "the variant config was written")
        self.assertFalse((self.t / "strata-q2_0.json").exists(), "no canonical config is fabricated")
        cfg = json.loads(vcfg.read_text(encoding="utf-8-sig"))
        args = cfg["args"]
        # its own pack: the fused experts are rebuilt from THIS build's shards, never the original's
        self.assertTrue(args[args.index("--pack") + 1].endswith(os.path.join("packs", "q2_0-abliterated")))
        packs = [r for r in self.runs if r[1].endswith("strata_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        self.assertTrue(packs[0][packs[0].index("--out") + 1].endswith(os.path.join("packs", "q2_0-abliterated")))
        self.assertTrue(args[args.index("--native") + 1].endswith(ABLITERATED_1))
        self.assertEqual(cfg["log"], str(self.t / "strata-q2_0-abliterated.log"))   # its own log
        self.assertEqual(written_scripts, ["Q2_0-abliterated"])   # its own start script (lowercased on disk)

    def test_variant_never_overwrites_an_existing_canonical_config(self):
        code, root_cfg, _, canon_before, _ = self.main(["--variant", "abliterated"], existing_canonical=True)
        self.assertEqual(code, 0)
        self.assertEqual(root_cfg.read_bytes(), canon_before)   # bytes-identical, untouched
        self.assertTrue((self.t / "strata-q2_0-abliterated.json").is_file())

    def test_bad_variant_is_refused(self):
        code, _, _, _, _ = self.main(["--variant", "not allowed space"])
        self.assertEqual(code, 1)
        self.assertFalse((self.t / "strata-q2_0.json").exists())
        self.assertFalse((self.t / "strata-q2_0-not").exists())


class FrontendStateContract(unittest.TestCase):
    """The Custom GGUF field's state machine (gui/web/gguf_state.js): the pure JS behind "the
    field is the source of truth" - a stale detection is invalidated on change, a late async
    response is ignored, Prepare refuses a detection whose directory is not exactly the field,
    and Browse reaches the same state as manual input.  Runs the node tests when node is
    installed; a machine without node skips (the suite stays stdlib-only)."""

    NODE_TEST = ROOT / "gui" / "web" / "test_gguf_state.mjs"

    def test_gguf_field_state_machine(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        r = subprocess.run([node, str(self.NODE_TEST)], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, f"gui/web/test_gguf_state.mjs failed:\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()