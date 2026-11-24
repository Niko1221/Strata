"""Regression tests for Pass 2 TUI/CLI, setup.py, and strata_control fixes:
- #6469: --prompt-cache-spread alignment in strata launch derive_config & BOOL_FLAGS
- #6691: Intel Arc SYCL setup & adopt for Inkling-2-Mini (.strata/.gguf/.safetensors) & Hy4-Flash (.strata/.safetensors)
- #6672: Inkling-2-Mini GGUF import (inkling2 / inkling2_loop, Unsloth + mainline) & inkling_local_source is_gguf fix
- #6670: Inkling-2-Mini --drafting off explicitly emits --draft off in setup, adopt, apply_choices, derive_config & TUI
- #6681: Inkling-2-Mini KV cache options restricted to fp16/fp8 on HIP/Metal/SYCL (k8v4/k8v8 clamped & hidden)
- #6678: Multi-GPU (2x RTX 3090 + RTX 3060 Ti with GPU roles) install_inkling2 & install_hy4_flash
- #6695: strata_control/drafting.py:gguf_facts() skips string arrays (general.tags, general.languages)
- #6662: setup.py:gpus() records free_gb alongside vram_gb and respects free memory for tier/vision sizing
"""

from __future__ import annotations

import argparse
import asyncio
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

import setup  # noqa: E402
from strata_cli.launch import adopt, modelstate, runner as R, screens, settings as S  # noqa: E402
from strata_cli.launch.app import LaunchApp  # noqa: E402
from strata_control import drafting, inkling_family, models, tune_knobs  # noqa: E402
from tests.terminal.launch_support import Home  # noqa: E402
from tests.terminal.test_drafting_file_fit import _s as gguf_str  # noqa: E402
from tests.terminal.test_inkling_gguf_import import make_inkling_gguf, write_fake_o200k_files  # noqa: E402
from tests.terminal.test_launch_tui import flat, make_ctx  # noqa: E402


class PromptCacheSpread6469Tests(unittest.TestCase):
    BASE = {
        "exe": "component:strata-engine",
        "args": ["--native", "x.gguf", "--max-context", "131072", "--kv", "int8"],
        "port": 8080,
        "model_name": "qwen3.8-flash-next-iq3_s",
    }

    def test_prompt_cache_spread_enabled_by_default_and_toggleable(self):
        self.assertIn("--prompt-cache-spread", tune_knobs.BOOL_FLAGS)
        cfg = R.derive_config(self.BASE, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS)
        self.assertIn("--prompt-cache-spread", cfg["args"])

        # Explicit --set prompt-cache-spread=off removes it via arg_layers
        cfg_off = R.derive_config(
            self.BASE,
            S.LaunchSettings(extra={"--prompt-cache-spread": "off"}),
            bool_flags=tune_knobs.BOOL_FLAGS,
        )
        self.assertNotIn("--prompt-cache-spread", cfg_off["args"])

        # Config flag prompt_cache_spread=False removes it
        cfg_flag_off = R.derive_config(
            {**self.BASE, "prompt_cache_spread": False},
            S.LaunchSettings(),
            bool_flags=tune_knobs.BOOL_FLAGS,
        )
        self.assertNotIn("--prompt-cache-spread", cfg_flag_off["args"])

        # Native .strata engines (Hy4-Flash, Inkling-2-Mini) never receive --prompt-cache-spread
        ink_base = {
            "exe": "component:strata-inkling",
            "arch": "inkling2_loop",
            "args": ["--native", "m.strata", "--max-context", "32768"],
        }
        ink_cfg = R.derive_config(ink_base, S.LaunchSettings(family="inkling-2-mini"), bool_flags=tune_knobs.BOOL_FLAGS)
        self.assertNotIn("--prompt-cache-spread", ink_cfg["args"])


class IntelArcNativeStrata6691Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_adopt_inkling_strata_gguf_and_safetensors_on_sycl(self):
        data_dir = self.root / "data"
        src_dir = ROOT

        # 1. Local .strata with companion tokenizer.json + chat_template.jinja (tokenizer built on the fly!)
        strata_dir = self.root / "ink-strata"
        strata_dir.mkdir()
        strata_file = strata_dir / "Inkling-2-Mini-SQ-20.strata"
        strata_file.write_bytes(b"STRATA\x00\x01" + b"\x00" * 64)
        (strata_dir / inkling_family.VISION_FILE).write_bytes(b"STRATA\x00\x01" + b"\x00" * 64)
        write_fake_o200k_files(strata_dir)

        cfg, problems = adopt.plan(
            family="inkling-2-mini",
            model="SQ-20",
            model_dir=strata_dir,
            data_dir=data_dir,
            src=src_dir,
            context=32768,
            kv="k8v4",  # must clamp to fp8 on sycl (#6681)
            model_name="inkling-2-mini-sq-20",
            family_def={},
            log=self.root / "ink.log",
            backend="sycl",
            notes=(notes := []),
        )
        self.assertEqual(problems, [])
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["exe"], "component:strata-inkling")
        self.assertEqual(cfg["arch"], "inkling2_loop")
        self.assertEqual(cfg["backend"], "sycl")
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], str(strata_file))
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "fp8")
        self.assertIn("--vision", cfg["args"])
        self.assertTrue(any("fp8" in n for n in notes))

        # 2. Local .safetensors directory on SYCL with --drafting off
        st_dir = self.root / "ink-st"
        st_dir.mkdir()
        (st_dir / "config.json").write_text('{"model_type": "inkling2"}', encoding="utf-8")
        (st_dir / "model.safetensors").write_bytes(b"st")
        (st_dir / "mtp.safetensors").write_bytes(b"mtp")
        write_fake_o200k_files(st_dir)

        cfg_st, problems_st = adopt.plan(
            family="inkling-2-mini",
            model="SQ-20",
            model_dir=st_dir,
            data_dir=data_dir,
            src=src_dir,
            context=32768,
            kv="fp16",
            model_name="inkling-2-mini-sq-20",
            family_def={},
            log=self.root / "ink-st.log",
            backend="sycl",
            drafting="off",
        )
        self.assertEqual(problems_st, [])
        self.assertEqual(cfg_st["drafting"], "off")
        self.assertIn("--draft", cfg_st["args"])
        self.assertEqual(cfg_st["args"][cfg_st["args"].index("--draft") + 1], "off")
        self.assertNotIn("--dspark", cfg_st["args"])

    def test_adopt_hy4_strata_and_safetensors_on_sycl(self):
        data_dir = self.root / "data"
        hy4_dir = self.root / "hy4-strata"
        hy4_dir.mkdir()
        model_file = hy4_dir / "Hy4-Flash-SQ-64.strata"
        engram_file = hy4_dir / "Hy4-Flash-Engram-Q8.strata"
        vision_file = hy4_dir / "Hy4-Flash-Vision.strata"
        for f in (model_file, engram_file, vision_file):
            f.write_bytes(b"STRATA\x00\x01" + b"\x00" * 64)
        tok_dir = data_dir / "packs" / "hy4-flash-sq-64" / "tokenizer"
        tok_dir.mkdir(parents=True)
        (tok_dir / "vocab.json").write_text("{}", encoding="utf-8")

        cfg, problems = adopt.plan(
            family="hy4-flash",
            model="SQ-64",
            model_dir=hy4_dir,
            data_dir=data_dir,
            src=ROOT,
            context=131072,
            kv=None,
            model_name="hy4-flash-sq-64",
            family_def={},
            log=self.root / "hy4.log",
            backend="sycl",
        )
        self.assertEqual(problems, [])
        self.assertEqual(cfg["exe"], "component:strata-hy4")
        self.assertEqual(cfg["arch"], "hy4flash")
        self.assertEqual(cfg["backend"], "sycl")
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], str(model_file))
        self.assertEqual(cfg["args"][cfg["args"].index("--engram") + 1], str(engram_file))
        self.assertEqual(cfg["args"][cfg["args"].index("--vision") + 1], str(vision_file))

        # Safetensors folder for Hy4-Flash
        st_dir = self.root / "hy4-st"
        st_dir.mkdir()
        (st_dir / "config.json").write_text("{}", encoding="utf-8")
        (st_dir / "model-00001-of-00002.safetensors").write_bytes(b"st")
        (st_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        cfg_st, probs_st = adopt.plan(
            family="hy4-flash",
            model="SQ-64",
            model_dir=st_dir,
            data_dir=data_dir,
            src=ROOT,
            context=65536,
            kv=None,
            model_name="hy4-flash-sq-64",
            family_def={},
            log=self.root / "hy4-st.log",
            backend="sycl",
        )
        self.assertEqual(probs_st, [])
        self.assertEqual(cfg_st["args"][cfg_st["args"].index("--native") + 1], str(st_dir))
        self.assertNotIn("--engram", cfg_st["args"])

    def test_runner_setup_args_passes_backend_sycl_for_native_strata_on_linux(self):
        for fam, mod in (("inkling-2-mini", "SQ-20"), ("hy4-flash", "SQ-64")):
            r = object.__new__(R.Runner)
            r.s = S.LaunchSettings(family=fam, model=mod, context=32768, vision="on")
            r._catalog = mock.Mock()
            r._catalog.resolve.side_effect = lambda m, f: (f, m)
            r.feed = None
            r.move_data = False
            r.existing_data_dir = lambda: None
            r.installed_card_args = lambda: []
            with mock.patch.object(R, "WINDOWS", False), \
                 mock.patch.object(R.sys, "platform", "linux"), \
                 mock.patch("strata_control.engine_component.detect_backend", return_value=("sycl", "detected")):
                args = r.setup_args()
            self.assertIn("--backend", args, fam)
            self.assertEqual(args[args.index("--backend") + 1], "sycl", fam)


class InklingGguf6672AndDraftOff6670AndKv6681Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_plan_import_accepts_all_inkling_gguf_variants_and_local_source_is_gguf(self):
        folder = self.root / "unsloth-q8"
        folder.mkdir()
        gguf = folder / "Inkling-2-Mini-UD-Q8_K_XL.gguf"
        gguf.write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=8))
        plan = models.plan_import(str(gguf))
        self.assertEqual(plan["file_set_status"], "plausible", plan)
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["family"], "inkling-2-mini")
        self.assertEqual(plan["candidate_selection"]["model"], "UD-Q8_K_XL")

        # Regression: directory pointing at Inkling-2-Mini-UD-Q8_K_XL.gguf must return is_gguf=True
        args_ns = argparse.Namespace(gguf_dir=str(folder))
        local = setup.inkling_local_source(args_ns, "UD-Q8_K_XL")
        self.assertIsNotNone(local)
        self.assertEqual(local[0], gguf)
        self.assertTrue(local[2], "pinned Unsloth GGUF in folder must set is_gguf=True")

    def test_inkling_drafting_off_emits_explicit_draft_off_everywhere(self):
        # 1. derive_config
        base = {
            "exe": "component:strata-inkling",
            "arch": "inkling2_loop",
            "args": ["--native", "m.strata", "--max-context", "65536"],
            "drafting": "dspark",
            "backend": "metal",
        }
        cfg = R.derive_config(
            base,
            S.LaunchSettings(family="inkling-2-mini", drafting="off", kv="k8v4"),
            bool_flags=tune_knobs.BOOL_FLAGS,
        )
        self.assertEqual(cfg["drafting"], "off")
        self.assertIn("--draft", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--draft") + 1], "off")
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "fp8")  # k8v4 clamped on metal (#6681)

        # 2. apply_choices
        prev = R.apply_choices(base, S.LaunchSettings(family="inkling-2-mini", drafting="off", kv="fp8"))
        self.assertEqual(prev["args"][prev["args"].index("--draft") + 1], "off")
        self.assertEqual(prev["args"][prev["args"].index("--kv") + 1], "fp8")

        # 3. modelstate.drafting_for_family preserves "off" for inkling-2-mini
        self.assertEqual(modelstate.drafting_for_family("off", "inkling-2-mini"), "off")
        self.assertIsNone(modelstate.drafting_for_family("dflash2", "inkling-2-mini"))

        # 4. KV mode helpers (#6681)
        for be in ("hip", "metal", "sycl"):
            self.assertEqual(S.inkling_kv_modes(be), ("fp16", "fp8"))
            self.assertEqual(S.safe_inkling_kv("k8v4", be), ("fp8", True))
            self.assertEqual(S.safe_inkling_kv("k8v8", be), ("fp8", True))
            self.assertEqual(S.safe_inkling_kv("fp8", be), ("fp8", False))
        self.assertEqual(S.inkling_kv_modes("cuda"), ("fp16", "fp8", "k8v8", "k8v4"))
        self.assertEqual(S.safe_inkling_kv("k8v4", "cuda"), ("k8v4", False))

    def test_tui_inkling_kv_restricted_on_hip_and_sycl_and_drafting_off_selectable(self):
        hw_sycl = {
            "ram_gb": 64.0,
            "os": "Linux",
            "disk_free_gb": 500.0,
            "cpu": {"avx2": True, "name": "test cpu"},
            "engine_vendor": "intel",
            "gpus": [{"index": 0, "name": "Intel Arc Pro B60", "vram_gb": 24.0, "usable": True, "vendor": "intel"}],
        }

        async def run_case():
            ctx = make_ctx(hw_sycl)
            ctx.settings.family = "inkling-2-mini"
            ctx.settings.model = "SQ-20"
            ctx.settings.context = 32768
            app = LaunchApp(ctx)
            async with app.run_test(size=(110, 36)) as pilot:
                app.goto("context")
                await pilot.pause(0.1)
                txt = flat(app)
                self.assertIn("fp16/fp8", txt)
                self.assertNotIn("k8v4", txt)
                self.assertNotIn("k8v8", txt)
                await pilot.press("m")
                await pilot.pause(0.05)
                self.assertEqual(ctx.settings.kv, "fp8")
                await pilot.press("m")
                await pilot.pause(0.05)
                self.assertEqual(ctx.settings.kv, "fp16")

                # DraftingScreen allows selecting Off for Inkling-2-Mini
                app.goto("drafting")
                await pilot.pause(0.1)
                await pilot.press("down")
                await pilot.press("enter")
                await pilot.pause(0.05)
                self.assertEqual(ctx.settings.drafting, "off")

        with Home():
            asyncio.run(run_case())


class MultiGpu6678RegressionTests(unittest.TestCase):
    def test_2x_3090_plus_3060ti_setup_inkling2_and_hy4_no_crash(self):
        """Issue #6678: 2x RTX 3090 (24 GB + 20 GB) + RTX 3060 Ti (8 GB Off) with int multi=[1, 0]."""
        nv_cards = [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "arch": "86", "driver": "610.88", "count": 3},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 20.0, "arch": "86", "driver": "610.88", "count": 3},
            {"index": 2, "name": "NVIDIA GeForce RTX 3060 Ti", "vram_gb": 8.0, "arch": "86", "driver": "610.88", "count": 3},
        ]
        roles = {0: "main", 1: "helper", 2: "off"}
        roles_cards = [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 20.0, "usable": True, "vendor": "nvidia"},
            {"index": 2, "name": "NVIDIA GeForce RTX 3060 Ti", "vram_gb": 8.0, "usable": True, "vendor": "nvidia"},
        ]

        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            data, home = t / "data", t / "install"
            data.mkdir()
            home.mkdir()
            (data / "models" / "inkling-2-mini").mkdir(parents=True)
            (data / "models" / "inkling-2-mini" / "Inkling-2-Mini-SQ-20.strata").write_bytes(b"STRATA\x00\x01")
            (data / "models" / "inkling-2-mini" / "Inkling-2-Mini-Vision.strata").write_bytes(b"STRATA\x00\x01")
            (data / "models" / "hy4-flash").mkdir(parents=True)
            (data / "models" / "hy4-flash" / "Hy4-Flash-SQ-64.strata").write_bytes(b"STRATA\x00\x01")
            (data / "models" / "hy4-flash" / "Hy4-Flash-Engram-Q8.strata").write_bytes(b"STRATA\x00\x01")
            (data / "models" / "hy4-flash" / "Hy4-Flash-Vision.strata").write_bytes(b"STRATA\x00\x01")

            def make_args(family, model, drafter=None):
                return argparse.Namespace(
                    family=family, model=model, backend=None, context=32768, vision="yes", port=8080,
                    host="127.0.0.1", api_key=None, browser=False, check=False, yes=True, no_start=True,
                    gguf_dir=None, models_dir=str(data / "models"), data_dir=str(data), parallel=None,
                    reasoning_effort=None, rope_scaling=None, rope_scale=None, kv=None,
                    resident_budget_gib=None, low_ram=None, gpu=0, gpus=None, gpu_roles=None,
                    mtp=None, drafter=drafter, experimental_speed_projection=None, layer_split=None,
                    move_data=False, set_default_data_dir=False, prepare_pack=None,
                    _roles=dict(roles), _roles_cards=list(roles_cards),
                )

            with mock.patch.object(setup, "CONTROL", True), \
                 mock.patch.object(setup, "_CONTROL_HOME", home), \
                 mock.patch.object(setup, "gpus", return_value=nv_cards), \
                 mock.patch.object(setup, "data_folder_for", lambda a: (data, [])), \
                 mock.patch.object(setup, "pip_install"), \
                 mock.patch.object(setup, "inkling_have", lambda a, p: True), \
                 mock.patch.object(setup, "inkling_fetch", lambda a, n, folder, own: folder / n), \
                 mock.patch.object(setup, "inkling_tokenizer_dir", lambda *a, **kw: data / "tok"), \
                 mock.patch.object(setup, "hy4_have", lambda a, p: True), \
                 mock.patch.object(setup, "hy4_fetch", lambda a, n, folder, own: folder / n), \
                 mock.patch.object(setup, "hy4_tokenizer_dir", lambda *a, **kw: data / "tok"), \
                 mock.patch.object(setup, "engine_step", lambda *a, **kw: (None, str(t / "eng"), {"component": "ok"}, [], a[1], None)), \
                 mock.patch.object(setup, "write_run_script", lambda *a, **kw: home / "run.bat"), \
                 mock.patch.object(setup, "free_gb", lambda p: 500.0):
                rc_ink = setup.install_inkling2(
                    make_args("inkling-2-mini", "SQ-20", drafter="off"),
                    "inkling-2-mini", nv_cards[0], False, [], None, [1, 0], 8080, 96.0, None,
                )
                self.assertEqual(rc_ink, 0)
                ink_cfg = json.loads((home / "strata-inkling-2-mini-sq-20.json").read_text(encoding="utf-8"))
                self.assertEqual(ink_cfg["gpu"], [1, 0])
                self.assertEqual(ink_cfg["gpu_roles"], {"0": "main", "1": "helper", "2": "off"})
                self.assertIn("--layer-split", ink_cfg["args"])
                self.assertEqual(ink_cfg["drafting"], "off")
                self.assertIn("--draft", ink_cfg["args"])
                self.assertEqual(ink_cfg["args"][ink_cfg["args"].index("--draft") + 1], "off")

                rc_hy4 = setup.install_hy4_flash(
                    make_args("hy4-flash", "SQ-64"),
                    "hy4-flash", nv_cards[0], False, [], None, [1, 0], 8080, 96.0, None,
                )
                self.assertEqual(rc_hy4, 0)
                hy4_cfg = json.loads((home / "strata-hy4-flash-sq-64.json").read_text(encoding="utf-8"))
                self.assertEqual(hy4_cfg["gpu"], [1, 0])
                self.assertEqual(hy4_cfg["gpu_roles"], {"0": "main", "1": "helper", "2": "off"})


class GgufFactsStringArray6695Tests(unittest.TestCase):
    def _gguf_with_tags(self, arch="qwen4exp", layers=48, hidden=2560, vocab=248320) -> bytes:
        tags = ["qwen", "moe", "text-generation", "llama.cpp"]
        langs = ["en", "zh"]
        tags_val = struct.pack("<IIQ", 9, 8, len(tags)) + b"".join(gguf_str(t) for t in tags)
        langs_val = struct.pack("<IIQ", 9, 8, len(langs)) + b"".join(gguf_str(x) for x in langs)
        kvs = [
            (gguf_str("general.architecture"), struct.pack("<I", 8) + gguf_str(arch)),
            (gguf_str("general.tags"), tags_val),
            (gguf_str("general.languages"), langs_val),
            (gguf_str(f"{arch}.block_count"), struct.pack("<II", 4, layers)),
            (gguf_str(f"{arch}.embedding_length"), struct.pack("<II", 4, hidden)),
            (gguf_str("tokenizer.ggml.tokens"), struct.pack("<IIQ", 9, 8, vocab)),
        ]
        return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(k + v for k, v in kvs)

    def test_gguf_facts_reads_past_general_tags_and_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "renamed-flash-next.gguf"
            p.write_bytes(self._gguf_with_tags())
            self.assertEqual(
                drafting.gguf_facts(p),
                {"arch": "qwen4exp", "layers": 48, "hidden": 2560, "vocab": 248320},
            )
            self.assertEqual(drafting.file_fit(["--native", str(p)]), "fits")

            # Corrupt absurd string-array count never hangs or raises
            bad = (
                b"GGUF"
                + struct.pack("<IQQ", 3, 0, 2)
                + gguf_str("general.architecture") + struct.pack("<I", 8) + gguf_str("qwen4exp")
                + gguf_str("general.tags") + struct.pack("<IIQ", 9, 8, 1 << 60)
            )
            p.write_bytes(bad)
            self.assertEqual(drafting.gguf_facts(p), {"arch": "qwen4exp", "layers": None, "hidden": None, "vocab": None})


class GpuFreeVram6662Tests(unittest.TestCase):
    def test_gpus_records_free_gb_and_tolerates_legacy_5_column_output(self):
        csv_6 = "0, NVIDIA GeForce RTX 4090, 24576, 8.9, 565.57, 8192\n"
        with mock.patch.object(setup, "out", return_value=csv_6):
            found = setup.gpus()
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["vram_gb"], 24.0, places=2)
        self.assertAlmostEqual(found[0]["free_gb"], 8.0, places=2)
        self.assertAlmostEqual(setup.effective_vram_gb(found[0]), 8.0, places=2)
        # Busy 24 GB card with only 8 GB free uses CPU vision encoder and 8 GB for low_ram_vram / tier_pick
        self.assertEqual(setup.flash_vision_encoder("yes", found[0]), "cpu")
        self.assertAlmostEqual(setup.low_ram_vram(found[0]), 8.0, places=2)
        self.assertEqual(setup.tier_pick(40.0, setup.low_ram_vram(found[0])), ("swift", "IQ2_XS"))

        # Legacy 5-column mock keeps working cleanly with vram_gb fallback
        csv_5 = "0, NVIDIA GeForce RTX 4090, 24576, 8.9, 565.57\n"
        with mock.patch.object(setup, "out", return_value=csv_5):
            found5 = setup.gpus()
        self.assertEqual(len(found5), 1)
        self.assertNotIn("free_gb", found5[0])
        self.assertAlmostEqual(setup.effective_vram_gb(found5[0]), 24.0, places=2)
        self.assertEqual(setup.flash_vision_encoder("yes", found5[0]), "gpu")


if __name__ == "__main__":
    unittest.main()
