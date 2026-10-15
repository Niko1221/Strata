"""Tests for setup.py's Qwen3.6-35B-A3B install (a family with its own engine path: OWN_PATH_FAMILIES): the pinned and
hashed files, the menu entry, steps 2-7 of install_own_path (size, context, the engine's --families check, the
download, the tokenizer, the config) and reading such a config back.  Mocked engine, downloads and processes -
nothing is built, fetched or started.

    python -m unittest tools.test_setup_qwen36
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")
REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"


def args(**kw):
    base = dict(family="qwen36", model=None, check=False, yes=True, vision=None, models_dir=None, gguf_dir=None,
                context=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto",
                vram_reserve_mib=None, host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None,
                engine_args=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Qwen36Entries(unittest.TestCase):
    def test_pinned_and_hashed(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertIs(setup.PREVIEW_FAMILIES, setup.OWN_PATH_FAMILIES)   # the earlier name still works
        self.assertEqual(setup.HF_REVISIONS[REPO], "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d")
        for key in ("hf", "mmproj_hf"):
            self.assertRegex(fam[key], SHA)
            self.assertIn(REPO, fam[key])
        self.assertEqual(fam["arch"], "qwen35moe")
        self.assertEqual(list(fam["models"]), ["UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"])
        for m in fam["models"]:
            size, sha = fam["sha256"][fam["file"].format(q=m)]
            self.assertGreater(size, 20e9)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertEqual(fam["sha256"]["Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"][0], 22853663008)

    def test_not_a_pack_family(self):
        # FAMILIES are Flash-Next files with packs; Qwen3.6 is installed by install_own_path
        self.assertNotIn("qwen36", setup.FAMILIES)
        for m, d in setup.MODELS.items():
            self.assertNotIn("qwen36", d.get("families", ()))

    def test_size_recommendation(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertEqual(setup.own_path_size(fam, 24.0), "UD-Q4_K_XL")   # none fits: the smallest
        self.assertEqual(setup.own_path_size(fam, 32.0), "UD-Q4_K_XL")
        self.assertEqual(setup.own_path_size(fam, 46.0), "UD-Q6_K_XL")
        self.assertEqual(setup.own_path_size(fam, 64.0), "Q8_0")

    def test_context_recommendation(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertEqual(setup.own_path_context(fam, 12.0), 32768)
        self.assertEqual(setup.own_path_context(fam, 16.0), 32768)
        self.assertEqual(setup.own_path_context(fam, 24.0), 65536)
        self.assertEqual(setup.own_path_context(fam, 32.0), 131072)


class InstallOwnPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.families = {"qwen4exp", "qwen35moe"}
        eng = self.root / "engine"
        self.cfg = {}

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            self.calls.append(("engine", vision))
            return self.root / "llama", eng, {"version": "0.1.44"}, ["/libs"], gpu, vision

        def write_cfg(path, cfg, source=None):
            self.cfg = cfg
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.calls.append(("sha", s.name))),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0),
                   mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                   mock.patch.object(setup, "engine_step", engine_step),
                   mock.patch.object(setup, "engine_families", lambda e: self.families),
                   mock.patch.object(setup, "run", lambda cmd, **k: self.calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder", lambda d: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "recommend_pool_workers", lambda a: a),
                   mock.patch.object(setup, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def install(self, ram=64.0, vram=16.0, **kw):
        gpu = {"index": 0, "count": 1, "vram_gb": vram, "name": "Test GPU"}
        return run(setup.install_own_path, args(models_dir=str(self.root / "models"), **kw), "qwen36", gpu, False, [],
                   13, [], 8080, ram, None)

    def test_installs_the_pinned_file_and_writes_the_config(self):
        rc, out = self.install(model="UD-Q4_K_XL")
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual(len(get), 1)
        gguf = self.root / "models" / "qwen36-UD-Q4_K_XL" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        self.assertEqual(get[0][1], setup.hf(REPO) + gguf.name)
        self.assertEqual(get[0][2], gguf)
        self.assertIn(("sha", gguf.name), self.calls)
        self.assertIn(("engine", "none"), self.calls)            # no image encoder for this family yet
        tok = [c for c in self.calls if c[0] == "run"]
        self.assertTrue(tok and "strata_tokenizer.py" in str(tok[0][1]))
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--native") + 1], str(gguf))
        self.assertEqual(a[a.index("--max-context") + 1], "32768")   # 16 GB card
        self.assertEqual(a[a.index("--mtp") + 1], "2")
        self.assertNotIn("--pack", a)
        self.assertNotIn("--expert-cache", a)
        self.assertEqual(self.cfg["arch"], "qwen35moe")
        self.assertEqual(self.cfg["model_name"], "qwen3.6-35b-a3b-ud-q4_k_xl")
        self.assertEqual(Path(self.cfg["exe"]).parent, self.root / "engine")
        self.assertIn(("config", "strata-qwen36-ud-q4_k_xl.json"), self.calls)

    def test_engine_args_override_setup_flags(self):
        rc, out = self.install(model="UD-Q4_K_XL", engine_args="--prefill 1024 --mtp 3 --ple-io ram")
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        # the user's values replace setup's --prefill/--mtp (one occurrence each); the new flag is appended
        self.assertEqual(a.count("--prefill"), 1)
        self.assertEqual(a[a.index("--prefill") + 1], "1024")
        self.assertEqual(a.count("--mtp"), 1)
        self.assertEqual(a[a.index("--mtp") + 1], "3")
        self.assertEqual(a[a.index("--ple-io") + 1], "ram")
        self.assertGreater(a.index("--ple-io"), a.index("--mtp"))
        # the raw extras are saved in the config, so a later setup run re-applies them without the flag
        self.assertEqual(self.cfg["engine_args_extra"], "--prefill 1024 --mtp 3 --ple-io ram")

    def test_default_size_follows_the_ram(self):
        rc, out = self.install(ram=64.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("Recommended for 64 GB of RAM: Q8_0", out)
        self.assertTrue(self.cfg["args"][self.cfg["args"].index("--native") + 1].endswith("Qwen3.6-35B-A3B-Q8_0.gguf"))
        rc, out = self.install(ram=46.0)
        self.assertTrue(self.cfg["args"][self.cfg["args"].index("--native") + 1].endswith("UD-Q6_K_XL.gguf"))

    def test_a_bigger_size_than_the_ram_is_asked_not_refused(self):
        with mock.patch.object(setup, "confirm_risk", lambda *a, **k: None):
            rc, out = self.install(ram=32.0, model="Q8_0")
        self.assertEqual(rc, 0, out)
        self.assertIn("installing Q8_0 with 32 GB of RAM, as you chose", out)

    def test_an_engine_without_the_family_is_refused_before_the_download(self):
        self.families = {"qwen4exp"}
        rc, out = self.install(model="UD-Q4_K_XL")
        self.assertEqual(rc, 1)
        self.assertIn("does not run Qwen3.6-35B-A3B yet", out)
        self.assertEqual([c for c in self.calls if c[0] == "get"], [])

    def test_a_flash_next_size_is_refused(self):
        rc, out = self.install(model="IQ3_XXS")
        self.assertEqual(rc, 1)
        self.assertIn("has no IQ3_XXS", out)
        self.assertEqual(self.calls, [])

    def test_too_little_disk(self):
        with mock.patch.object(setup, "free_gb", lambda p: 10.0):
            rc, out = self.install(model="Q8_0")
        self.assertEqual(rc, 1)
        self.assertIn("not enough free disk space", out)

    def test_bigger_card_bigger_context_and_vision_said_off(self):
        rc, out = self.install(model="UD-Q4_K_XL", vram=32.0, vision="yes")
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "131072")
        self.assertIn("images are not available", out)

    def test_config_read_back(self):
        self.install(model="UD-Q6_K_XL", context=65536)
        p = self.root / "strata-qwen36-ud-q6_k_xl.json"
        p.write_text(json.dumps(self.cfg), encoding="utf-8")
        ch = setup.choices_from_config(p)
        self.assertEqual((ch["family"], ch["model"], ch["context"]), ("qwen36", "UD-Q6_K_XL", 65536))
        self.assertTrue(setup.own_path_config(self.cfg))
        self.assertEqual(setup.calibration_targets([p], None, None), [])   # the tuning is Flash-Next's


class EngineArgsMerge(unittest.TestCase):
    def test_merge_appends_new_flags(self):
        self.assertEqual(setup.merge_engine_args(["--kv", "q4_0"], ["--spec", "5"]),
                         ["--kv", "q4_0", "--spec", "5"])

    def test_merge_replaces_same_flag(self):
        self.assertEqual(setup.merge_engine_args(["--max-context", "4096", "--prefill", "auto", "--mtp", "2"],
                                                 ["--prefill", "1024", "--ple-io", "ram"]),
                         ["--max-context", "4096", "--mtp", "2", "--prefill", "1024", "--ple-io", "ram"])

    def test_merge_replaces_two_value_flag(self):
        self.assertEqual(setup.merge_engine_args(["--control-vector-layer-range", "4", "44", "--cvec-mode", "project"],
                                                 ["--control-vector-layer-range", "5", "40"]),
                         ["--cvec-mode", "project", "--control-vector-layer-range", "5", "40"])

    def test_merge_no_stray_token(self):
        self.assertEqual(setup.merge_engine_args(["--control-vector-layer-range", "4", "44"],
                                                 ["--control-vector-layer-range", "5", "40"]),
                         ["--control-vector-layer-range", "5", "40"])

    def test_merge_keeps_both_when_arity_differs(self):
        self.assertEqual(setup.merge_engine_args(["--x", "1"], ["--x", "2", "3"]),
                         ["--x", "1", "--x", "2", "3"])

    def test_merge_appends_boolean_flag(self):
        self.assertEqual(setup.merge_engine_args(["--kv", "q4_0"], ["--no-pool"]),
                         ["--kv", "q4_0", "--no-pool"])

    def test_reapply_puts_user_values_last(self):
        # what CAL.apply leaves behind: its --spec-min-p appended last, the user's --pool-workers dropped
        # (calibration's default is None) - re-merge puts the user's values back on top
        cfg = {"args": ["--spec-min-p", "0.7", "--spec-min-p", "0.5"],
               "engine_args_extra": "--spec-min-p 0.7 --pool-workers 8"}
        setup.reapply_engine_args(cfg)
        self.assertEqual(cfg["args"], ["--spec-min-p", "0.7", "--pool-workers", "8"])

    def test_reapply_no_extras(self):
        cfg = {"args": ["--spec-min-p", "0.5"]}
        setup.reapply_engine_args(cfg)
        self.assertEqual(cfg["args"], ["--spec-min-p", "0.5"])


if __name__ == "__main__":
    unittest.main()
