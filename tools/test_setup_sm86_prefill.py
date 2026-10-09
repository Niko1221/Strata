"""SM86 installer opt-in, build caching and lifecycle checks; no GPU/downloads."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup
from test_setup_golden import install, argv_for, card

GPU = card(0, "NVIDIA GeForce RTX 3060", 12.0, "86")


def fake_build(calls):
    def build(gpu, vision, yes, llama, toolkit=None, sm86_prefill=None):
        calls.append((gpu, vision, toolkit, sm86_prefill))
        eng = setup.sm86_engine_dir(toolkit or 13, sm86_prefill) if sm86_prefill else setup.engine_dir(toolkit or 13)
        eng.mkdir(exist_ok=True)
        (eng / setup.EXE).write_bytes(b"engine")
        if vision != "none":
            (eng / setup.VEXE).write_bytes(b"vision")
        (eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.41",
            "archs": [86], "cuda_dirs": ["<cuda>"], "sm86_prefill": sm86_prefill}))
        return eng
    return build


class Choices(unittest.TestCase):
    def test_explicit_install_builds_and_configures_on_windows_and_linux(self):
        for mode, (build_flag, run_flag) in setup.SM86_PREFILL.items():
            for win in (False, True):
                with self.subTest(mode=mode, win=win):
                    calls = []
                    prebuilt = mock.Mock(side_effect=AssertionError("opt-in used a prebuilt"))
                    patches = [mock.patch.object(setup, "build_engine", fake_build(calls)),
                               mock.patch.object(setup, "get_prebuilt", prebuilt), mock.patch.object(setup, "WIN", win)]
                    code, text, cfg, _ = install(63.7, [GPU], argv_for("qwen", "IQ3_XXS") +
                        ["--sm86-prefill", mode, "--vision", "none"], extra=patches)
                    self.assertEqual(code, 0, text[-3000:])
                    self.assertEqual(calls[0][3], mode)
                    self.assertEqual(cfg["sm86_prefill"], mode)
                    self.assertIn("engine-sm86-" + mode, cfg["exe"])
                    self.assertEqual(cfg["env"][run_flag], "1")
                    self.assertEqual(cfg["env"]["STRATA_PF_FUSED"], "1")
                    self.assertIsNone(cfg["sm86_prefill_previous_fused"])

    def test_interactive_choice_is_visible_and_enter_is_off(self):
        for mode in setup.SM86_PREFILL:
            calls = []
            patches = [mock.patch.object(setup, "build_engine", fake_build(calls))]
            args = argv_for("qwen", "IQ3_XXS") + ["--vision", "none"]
            code, text, cfg, asked = install(63.7, [GPU], args, answers={"SM86 prefill variant?": mode}, extra=patches)
            self.assertEqual(code, 0, text[-3000:])
            self.assertEqual(cfg["sm86_prefill"], mode)
            self.assertTrue(any("SM86 prefill variant?" in q for q in asked))
        code, text, cfg, _ = install(63.7, [GPU], argv_for("qwen", "IQ3_XXS"), answers="")
        self.assertEqual(code, 0, text[-3000:])
        self.assertNotIn("sm86_prefill", cfg)
        self.assertNotIn("env", cfg)

    def test_yes_alone_does_not_enable_and_saved_choice_is_kept(self):
        args = argv_for("qwen", "IQ3_XXS") + ["--vision", "none"]
        code, text, cfg, asked = install(63.7, [GPU], args)
        self.assertEqual(code, 0, text[-3000:])
        self.assertNotIn("sm86_prefill", cfg)
        self.assertEqual(asked, [])
        for mode, (_, flag) in setup.SM86_PREFILL.items():
            old = {"exe": "old", "args": [], "sm86_prefill": mode, "sm86_prefill_previous_fused": "0",
                   "env": {"STRATA_PF_FUSED": "1", flag: "1", "USER_TUNING": "keep"}}
            calls = []
            code, text, cfg, _ = install(63.7, [GPU], args,
                configs=[("strata-iq3_xxs.json", old)],
                extra=[mock.patch.object(setup, "build_engine", fake_build(calls))])
            self.assertEqual(code, 0, text[-3000:])
            self.assertEqual(cfg["sm86_prefill"], mode)
            self.assertEqual(cfg["env"]["USER_TUNING"], "keep")
            self.assertEqual(cfg["sm86_prefill_previous_fused"], "0")

    def test_unsupported_gpu_and_backends_fail_before_installing(self):
        mode = next(iter(setup.SM86_PREFILL))
        for backend in ("hip", "sycl"):
            with mock.patch.object(sys, "argv", ["setup.py", "--backend", backend, "--sm86-prefill", mode]), \
                    mock.patch.object(setup, "data_folder", side_effect=AssertionError("touched install")), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as err:
                setup.main()
            self.assertEqual(err.exception.code, 2)
        code, text, _, _ = install(63.7, [card(0, "RTX 4090", 24, "89")],
            argv_for("qwen", "IQ3_XXS") + ["--sm86-prefill", mode],
            extra=[mock.patch.object(setup, "pip_install", side_effect=AssertionError("installed packages"))])
        self.assertNotEqual(code, 0)
        self.assertIn("8.6 GPU", text)

    def test_disable_restores_fused_setting_and_preserves_user_env(self):
        for mode, (_, flag) in setup.SM86_PREFILL.items():
            for previous in (None, "0", "1"):
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d) / "strata-model.json"
                    old = {"exe": "variant", "args": [], "sm86_prefill": mode,
                           "sm86_prefill_previous_fused": previous,
                           "env": {"STRATA_PF_FUSED": "1", flag: "1", "CUSTOM": "keep"}}
                    p.write_text(json.dumps(old))
                    cfg = {"exe": "stock", "args": [], "sm86_prefill": "off"}
                    setup.write_setup_config(p, cfg)
                    new = json.loads(p.read_text())
                    self.assertEqual(new["env"]["CUSTOM"], "keep")
                    self.assertEqual(new["env"][flag], "0")
                    self.assertEqual(new["env"].get("STRATA_PF_FUSED"), previous)
                    self.assertNotIn("sm86_prefill_previous_fused", new)
                    self.assertEqual(json.loads(p.with_name(p.name + ".bak").read_text()), old)

    def test_explicit_off_returns_to_stock_engine(self):
        mode = next(iter(setup.SM86_PREFILL))
        old = {"exe": "variant", "args": [], "sm86_prefill": mode,
               "sm86_prefill_previous_fused": None, "env": {"STRATA_PF_FUSED": "1"}}
        code, text, cfg, _ = install(63.7, [GPU], argv_for("qwen", "IQ3_XXS") + ["--sm86-prefill", "off"],
            configs=[("strata-iq3_xxs.json", old)],
            extra=[mock.patch.object(setup, "build_engine", side_effect=AssertionError("rebuilt on opt-out"))])
        self.assertEqual(code, 0, text[-3000:])
        self.assertEqual(cfg["exe"], "<T>/engine/<EXE>")
        self.assertNotIn("STRATA_PF_FUSED", cfg["env"])


class Builds(unittest.TestCase):
    def test_start_repairs_a_missing_variant_and_keeps_runtime_settings(self):
        for mode in setup.SM86_PREFILL:
            with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
                root = Path(d)
                p = root / "strata-iq3_xxs.json"
                cfg = {"exe": "missing", "args": [], "sm86_prefill": mode, "gpu": 0,
                       "gpus_asked": True, "open_browser": False}
                setup.apply_sm86_prefill(cfg)
                p.write_text(json.dumps(cfg))
                calls = []
                with mock.patch.object(setup, "build_engine", fake_build(calls)), \
                        mock.patch.object(setup, "gpus", return_value=[GPU]), \
                        mock.patch.object(setup, "OLD_GPUS", None), \
                        mock.patch.object(setup.subprocess, "call", return_value=0) as launch:
                    self.assertEqual(setup.start(p, None, yes=True), 0)
                saved = json.loads(p.read_text())
                self.assertIn("engine-sm86-" + mode, saved["exe"])
                self.assertEqual(saved["env"][setup.SM86_PREFILL[mode][1]], "1")
                self.assertEqual(calls[0][3], mode)
                self.assertIn(str(p), launch.call_args.args[0])

    def test_dedicated_cache_flags_and_cache_invalidation(self):
        for mode, (flag, _) in setup.SM86_PREFILL.items():
            for win in (False, True):
                with self.subTest(mode=mode, win=win), tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st:
                    root = Path(d)
                    st.enter_context(mock.patch.object(setup, "ROOT", root))
                    st.enter_context(mock.patch.object(setup, "WIN", win))
                    st.enter_context(mock.patch.object(setup, "cpu_info", return_value=("CPU", True, False)))
                    st.enter_context(mock.patch.object(setup, "source_hash", return_value="source-a"))
                    st.enter_context(mock.patch.object(setup, "source_version", return_value="0.1.41"))
                    st.enter_context(mock.patch.object(setup, "install_build_tools", return_value=("/cuda/bin/nvcc", None)))
                    llama = st.enter_context(mock.patch.object(setup, "get_llama_cpp", return_value=root / "llama"))
                    calls = []
                    def cmake(src, bdir, target, defs, vcvars, script):
                        calls.append((bdir, target, defs, script))
                        bdir.mkdir(parents=True, exist_ok=True)
                        (bdir / setup.EXE).write_bytes(b"engine")
                    st.enter_context(mock.patch.object(setup, "cmake_build", side_effect=cmake))
                    stock = root / "engine"
                    stock.mkdir()
                    (stock / setup.EXE).write_bytes(b"stock-preserved")
                    eng = setup.build_engine(GPU, "none", True, None, sm86_prefill=mode)
                    self.assertEqual(eng.name, "engine-sm86-" + mode)
                    self.assertEqual(calls[0][0].name, "build-sm86-" + mode)
                    self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", calls[0][2])
                    self.assertIn(f"-D{flag}=ON", calls[0][2])
                    for other in setup.SM86_BUILD_FLAGS:
                        if other != flag:
                            self.assertIn(f"-D{other}=OFF", calls[0][2])
                    self.assertIn("-sm86-" + mode, calls[0][3])
                    self.assertEqual((stock / setup.EXE).read_bytes(), b"stock-preserved")
                    meta = json.loads((eng / "BUILD.json").read_text())
                    self.assertEqual(meta["sm86_prefill"], mode)
                    llama.reset_mock()
                    setup.build_engine(GPU, "none", True, None, sm86_prefill=mode)
                    self.assertEqual(len(calls), 1)
                    llama.assert_not_called()
                    meta["sm86_prefill"] = "wrong-build"
                    (eng / "BUILD.json").write_text(json.dumps(meta))
                    setup.build_engine(GPU, "none", True, None, sm86_prefill=mode)
                    self.assertEqual(len(calls), 2)
                    with mock.patch.object(setup, "source_hash", return_value="source-b"):
                        setup.build_engine(GPU, "none", True, None, sm86_prefill=mode)
                    self.assertEqual(len(calls), 3)
                    setup.build_engine(GPU, "none", True, root / "llama", toolkit=12, sm86_prefill=mode)
                    self.assertEqual(calls[-1][0].name, "build-cuda12-sm86-" + mode)
                    self.assertEqual(setup.config_toolkit({"exe": str(root / ("engine-cuda12-sm86-" + mode) / setup.EXE)}), 12)

    def test_start_and_update_use_saved_variant_not_prebuilt(self):
        for mode in setup.SM86_PREFILL:
            with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
                root = Path(d)
                p = root / "strata-iq3_xxs.json"
                cfg = {"exe": "missing", "args": [], "sm86_prefill": mode, "gpu": 0, "lib_dirs": ["<custom>"],
                       "vision": {"gpu": True, "exe": "old-vision"}}
                p.write_text(json.dumps(cfg))
                calls = []
                with mock.patch.object(setup, "build_engine", fake_build(calls)):
                    updated = setup.ensure_engine_for([GPU], p, cfg, True)
                self.assertIn("engine-sm86-" + mode, updated["exe"])
                self.assertIn("engine-sm86-" + mode, updated["vision"]["exe"])
                self.assertEqual(updated["lib_dirs"], ["<cuda>", "<custom>"])
                self.assertEqual(calls[0][3], mode)
                before = p.read_bytes()
                with mock.patch.object(setup, "build_engine", side_effect=RuntimeError("build failed")), \
                        self.assertRaises(RuntimeError):
                    setup.ensure_engine_for([GPU], p, updated, True)
                self.assertEqual(p.read_bytes(), before)
                args = type("Args", (), {"build": False, "prebuilt": "unused"})()
                with mock.patch.object(setup, "build_engine", fake_build(calls)), \
                        mock.patch.object(setup, "gpus", return_value=[GPU]), \
                        mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "update_installed_engine"), \
                        mock.patch.object(setup, "engine_version", return_value=(0, 1, 41)):
                    self.assertEqual(setup.update_install([p], args), 0)
                self.assertEqual(calls[-1][3], mode)


if __name__ == "__main__":
    unittest.main()
