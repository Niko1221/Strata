"""Tests for setup.py --backend sycl, the experimental Intel Arc engine (sycl/, PR #423): it hands the run to
sycl/setup_intel.py without its --backend flag, using native execution on Windows. The CUDA / HIP
paths never see it. No GPU, no network.

    python -m unittest tools.test_setup_sycl
"""
from __future__ import annotations

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
import setup  # noqa: E402


def run(argv, win):
    out = io.StringIO()
    with mock.patch.object(setup, "WIN", win), mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
            contextlib.redirect_stdout(out):
        try:
            rc = setup.sycl_setup(argv)
        except SystemExit as e:
            rc = ("exit", e.code)
    return rc, out.getvalue(), call


class SyclBackend(unittest.TestCase):
    def test_windows_hands_over_to_native_setup(self):
        rc, out, call = run(["--backend", "sycl"], win=True)
        self.assertEqual(rc, 0)
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("docs/INTEL_ARC.md", out)
        self.assertIn("without Docker or WSL", out)
        self.assertEqual(call.call_args[0][0][2:], [])

    def test_linux_hands_over_to_setup_intel_without_the_backend_flag(self):
        rc, out, call = run(["--model", "IQ2_XS", "--backend", "sycl", "--port", "8085"], win=False)
        self.assertEqual(rc, 0)
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("built from source", out)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("sycl") / "setup_intel.py")))
        self.assertEqual(cmd[2:], ["--model", "IQ2_XS", "--port", "8085"])

    def test_backend_equals_form_is_dropped_too(self):
        _, _, call = run(["--backend=sycl", "--check"], win=False)
        self.assertEqual(call.call_args[0][0][2:], ["--check"])

    def test_the_parser_accepts_sycl(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'"--backend", choices=\["cuda", "hip", "sycl"\]')

    def test_setup_intel_finds_the_setup_functions_it_replaces(self):
        """sycl/setup_intel.py swaps these steps; setup.py must still have them (it stops with a message otherwise)."""
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        names = re.search(r'for name in \(([^)]*)\):', src).group(1)
        for name in re.findall(r'"(\w+)"', names):
            self.assertTrue(callable(getattr(setup, name, None)), name)

    def test_setup_intel_replacements_take_every_argument_of_the_function_they_replace(self):
        """#870: setup.py grew write_run_script's fourth argument and the port's replacement still took three, so
        setup died with a TypeError after the download.  Every `S.<name> = <replacement>` in setup_intel.py must accept
        all the parameters of setup.<name> (or take *args / **kwargs)."""
        import ast
        import inspect
        tree = ast.parse((ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8"))
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        checked = 0
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "S"):
                continue
            name = n.targets[0].attr
            fn = n.value if isinstance(n.value, ast.Lambda) else defs.get(getattr(n.value, "id", ""))
            orig = getattr(setup, name, None)
            if fn is None or not callable(orig):
                continue
            a = fn.args
            if a.vararg is not None and a.kwarg is not None:
                checked += 1
                continue
            have = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            for param in inspect.signature(orig).parameters.values():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                self.assertIn(param.name, have, f"setup_intel's {name} lacks setup.{name}'s `{param.name}`")
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_arc_pro_b60_is_known_under_both_pci_ids(self):
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn('"e211": ("Arc Pro B60", 24.0)', src)
        self.assertIn('"e221": ("Arc Pro B60", 24.0)', src)


class NativeStart(unittest.TestCase):
    def test_first_install_main_automatically_selects_intel(self):
        from tools.test_setup_golden import install
        handoff = mock.Mock(return_value=0)
        extra = [mock.patch.object(setup, "WIN", True),
                 mock.patch.object(setup, "_win_display_adapters", return_value=[
                     {"name": "Arc B70", "pnp": r"PCI\VEN_8086&DEV_E223"}]),
                 mock.patch.object(setup, "sycl_setup", handoff)]
        code, _, _, _ = install(64, [], ["--check"], extra=extra)
        self.assertEqual(code, 0)
        self.assertIn("--check", handoff.call_args.args[0])
        self.assertNotIn("--backend", handoff.call_args.args[0])

    def test_explicit_cuda_does_not_auto_select_intel(self):
        from tools.test_setup_golden import install
        handoff = mock.Mock(return_value=0)
        extra = [mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "sycl_setup", handoff)]
        code, text, _, _ = install(64, [], ["--check", "--backend", "cuda"], extra=extra)
        self.assertEqual(code, 1)
        self.assertIn("no NVIDIA GPU found", text)
        handoff.assert_not_called()

    def test_auto_detection_hands_over_without_a_backend_flag(self):
        from sycl import setup_intel
        with mock.patch.object(setup_intel, "intel_gpus_windows", return_value=[{"name": "Arc", "vram_gb": 32}]), \
                mock.patch.object(setup, "sycl_setup", return_value=0) as handoff, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.auto_sycl_setup(["--setup"], [], []), 0)
        handoff.assert_called_once_with(["--setup"])

    def test_mixed_gpu_keeps_original_default_and_allows_intel(self):
        from sycl import setup_intel
        with mock.patch.object(setup_intel, "intel_gpus_windows", return_value=[{"name": "Arc", "vram_gb": 32}]), \
                mock.patch.object(setup, "gpu_problem", return_value=None), \
                mock.patch.object(setup, "ask", side_effect=["1", "2"]) as ask, \
                mock.patch.object(setup, "sycl_setup", return_value=0) as handoff, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(setup.auto_sycl_setup([], [{}], [], yes=True))
            self.assertEqual(setup.auto_sycl_setup([], [{}], []), 0)
        self.assertEqual(ask.call_args_list[0].args[2], "1")
        handoff.assert_called_once_with([])

    def test_no_intel_leaves_the_original_path_alone(self):
        from sycl import setup_intel
        with mock.patch.object(setup_intel, "intel_gpus_windows", return_value=[]), \
                mock.patch.object(setup, "sycl_setup") as handoff:
            self.assertIsNone(setup.auto_sycl_setup([], [], []))
        handoff.assert_not_called()

    def test_windows_engine_step_keeps_native_sycl_build_route(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.44", "source": "local"}), encoding="utf-8")
            gpu = {"arch": "sycl", "index": 0, "count": 1}
            args = SimpleNamespace(build=True, prebuilt=None, gpu=None)
            with mock.patch.object(setup, "WIN", True), \
                    mock.patch.object(setup, "get_llama_cpp", return_value=eng), \
                    mock.patch.object(setup, "get_prebuilt_hip", return_value=eng) as native, \
                    mock.patch.object(setup, "hip_card", return_value=gpu), \
                    mock.patch.object(setup, "hip_lib_dirs", return_value=[]), \
                    mock.patch.object(setup, "build_engine_hip", side_effect=AssertionError("AMD build")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = setup.engine_step(args, gpu, True, [gpu], 13, "none")
            native.assert_called_once_with(None, gpu)
            self.assertEqual(result[1], eng)

    def test_windows_starts_intel_server_without_cuda_checks(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "strata-test.json"
            cfg_path.write_text(json.dumps({"exe": sys.executable, "args": [], "backend": "sycl"}),
                                encoding="utf-8")
            with mock.patch.object(setup, "WIN", True), \
                    mock.patch.object(setup, "upgrade_config", side_effect=lambda path, cfg: cfg), \
                    mock.patch.object(setup, "settings_summary", return_value="native"), \
                    mock.patch.object(setup, "gpus", side_effect=AssertionError("CUDA probe")), \
                    mock.patch.object(setup, "check_gpus", side_effect=AssertionError("CUDA check")), \
                    mock.patch.object(setup, "offer_together", side_effect=AssertionError("CUDA selection")), \
                    mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                    contextlib.redirect_stdout(io.StringIO()):
                rc = setup.start(cfg_path, 8085, gpu=1, open_browser=False,
                                 keep={"host": "127.0.0.1", "api_key": "test-key", "vram_reserve_mib": 1024})
            self.assertEqual(rc, 0)
            cmd = call.call_args.args[0]
            self.assertEqual(cmd[1], str(ROOT / "sycl" / "serve" / "server_intel.py"))
            self.assertEqual(cmd[cmd.index("--port") + 1], "8085")
            self.assertNotIn("--gpu", cmd)
            self.assertNotIn("--open", cmd)
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["host"], "127.0.0.1")
            self.assertEqual(saved["api_key"], "test-key")
            self.assertEqual(saved["gpu"], 1)
            self.assertEqual(saved["env"]["ONEAPI_DEVICE_SELECTOR"], "level_zero:1")
            self.assertIn("--vram-reserve-mib", saved["args"])

    def test_linux_start_keeps_original_gpu_argument_handling(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "strata-test.json"
            cfg_path.write_text(json.dumps({"exe": sys.executable, "args": [], "backend": "sycl"}),
                                encoding="utf-8")
            with mock.patch.object(setup, "WIN", False), \
                    mock.patch.object(setup, "upgrade_config", side_effect=lambda path, cfg: cfg), \
                    mock.patch.object(setup, "settings_summary", return_value="linux"), \
                    mock.patch.object(setup, "OLD_GPUS", None), \
                    mock.patch.object(setup, "old_gpus_opt_in", return_value=None), \
                    mock.patch.object(setup, "gpus", return_value=[]), \
                    mock.patch.object(setup, "check_gpus") as check, \
                    mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                    mock.patch.dict("os.environ", {"STRATA_EXECV": ""}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.start(cfg_path, 8085, gpu=1, open_browser=False), 0)
            check.assert_called_once_with([1], [])
            cmd = call.call_args.args[0]
            self.assertEqual(cmd[1], str(ROOT / "serve" / "server.py"))
            self.assertEqual(cmd[cmd.index("--gpu") + 1], "1")


class CMakeOption(unittest.TestCase):
    def test_sycl_is_off_by_default_and_exclusive(self):
        src = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
        self.assertIn('option(STRATA_ENABLE_SYCL "EXPERIMENTAL', src)
        self.assertRegex(src, r'option\(STRATA_ENABLE_SYCL "[^"]*" OFF\)')
        self.assertIn("add_subdirectory(sycl)", src)


class SyclVersion(unittest.TestCase):
    def setUp(self):
        from sycl import setup_intel
        self.mod = setup_intel

    def test_version_matches_the_source_and_satisfies_native_models(self):
        version = self.mod.sycl_version()
        self.assertNotEqual(version, "0")
        self.assertEqual(version, setup.source_version())
        numbers = tuple(int(x) for x in version.split("."))
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL"):
            with self.subTest(model=model):
                self.assertGreaterEqual(numbers, setup.MODELS[model].get("engine", setup.UNSLOTH_ENGINE))

    def test_version_follows_a_source_hotfix_with_a_variable_sycl_project(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.41.2 LANGUAGES CXX)\n", encoding="utf-8")
            (root / "sycl").mkdir()
            (root / "sycl" / "CMakeLists.txt").write_text(
                "set(STRATA_SYCL_VERSION 0.1.40.2)\n"
                "project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)\n", encoding="utf-8")
            with mock.patch.object(self.mod, "ROOT", root), mock.patch.object(setup, "ROOT", root):
                self.assertEqual(self.mod.sycl_version(), "0.1.41.2")


class IntelIds(unittest.TestCase):
    """sycl/setup_intel.py names a card by its PCI device id. The Arc Pro B60s measured on 2x B60 (`lspci -nn` 8086:e211,
    bmg-g21) must not fall through to "Intel GPU e211 (xe)" with a guessed VRAM size."""

    @staticmethod
    def table():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.INTEL_ARC

    def test_b60_e211_is_named_with_its_vram(self):
        self.assertEqual(self.table()["e211"], ("Arc Pro B60", 24.0))

    def test_the_ids_that_were_listed_before_are_kept(self):
        t = self.table()
        self.assertEqual(t["e223"][0], "Arc Pro B70")
        self.assertEqual(t["e221"][0], "Arc Pro B60")


class ToSycl(unittest.TestCase):
    """The config sycl/setup_intel.py writes: an explicit --vram-reserve-mib is kept, the B-series streams its experts,
    the A-series (i915) cannot (a single pinned host allocation above a few GB fails there) and loads them into RAM."""

    @staticmethod
    def mod():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / "a.gguf"),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", *extra]}

    def test_b_series_streams_and_defaults_the_reserve(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "1024")
        self.assertNotIn("STRATA_VERIFY_NO_HOST", c.get("env", {}))

    def test_xe_ranks_the_experts_with_the_sycl_profile_and_i915_keeps_the_shipped_one(self):
        m = self.mod()
        shipped = str(m.MOUNT / "data" / "expert-profile.bin")
        other = str(m.MOUNT / "mine" / "learned.bin")
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile-sycl.bin"))
        c = m.to_sycl(self.cfg(m, ["--expert-profile", other]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("learned.bin"))   # the user's own file stays
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile.bin"))   # the CPU computes the misses there

    def test_an_explicit_reserve_is_kept_on_both_drivers(self):
        m = self.mod()
        for vram, drv in ((32.0, "xe"), (8.0, "i915")):
            c = m.to_sycl(self.cfg(m, ["--vram-reserve-mib", "500"]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, vram, drv)
            self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
            self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "500")

    def test_a_series_loads_experts_into_ram_and_leaves_no_host_off(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--ple-io") + 1], "ram")
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "300")
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")
        self.assertEqual(c["env"]["STRATA_SYCL_BIN"], "build-sycl/strata")

    def test_split_short_rule(self):
        # #1980: 2x B60 24 GB - IQ3_XXS (42.9 GB) was refused under NO_HOST=1, IQ2_XS (35.5 GB) loads fully
        m = self.mod()
        self.assertTrue(m.split_likely_short(42.9, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(35.5, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(50.3, [32.0, 32.0]))   # IQ3_S on 2x B70: 79%
        self.assertTrue(m.split_likely_short(77.0, [32.0, 32.0]))
        self.assertFalse(m.split_likely_short(0, [24.0]))
        self.assertFalse(m.split_likely_short(40.0, []))

    def test_small_card_rule(self):
        m = self.mod()
        self.assertTrue(m.small_card(8.0, "i915"))
        self.assertTrue(m.small_card(24.0, "i915"))
        self.assertTrue(m.small_card(10.0, "xe"))
        self.assertFalse(m.small_card(32.0, "xe"))

    def test_windows_uses_native_paths_runtime_and_gpu_only(self):
        m = self.mod()
        exe = m.ROOT / "build-sycl-win" / "strata.exe"
        cfg = self.cfg(m)
        cfg["env"] = {"STRATA_STAGER_THREADS": "4"}
        with mock.patch.object(m, "sycl_runtime_dirs", return_value=["runtime"]):
            c = m.to_sycl(cfg, exe, 64.0, {}, 32.0, "level_zero")
        self.assertEqual(c["exe"], str(exe))
        self.assertEqual(c["lib_dirs"], ["runtime"])
        self.assertNotIn("sycl_root", c)
        self.assertNotIn("STRATA_SYCL_BIN", c["env"])
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "1")
        self.assertEqual(c["env"]["STRATA_STAGER_THREADS"], "4")
        self.assertTrue(c["env"]["ONEAPI_DEVICE_SELECTOR"].startswith("level_zero:"))
        self.assertEqual(c["args"][c["args"].index("--native") + 1], str((m.MOUNT / "d" / "a.gguf").resolve()))

    def test_windows_gpu_probe_uses_native_executable(self):
        m = self.mod()
        exe = m.ROOT / "build-sycl-win" / "strata.exe"
        result = mock.Mock(returncode=0, stdout='[{"index":0,"name":"Arc","vram_gb":32,"driver":"level_zero","free_memory_supported":true,"command_graph_supported":true}]', stderr="")
        with mock.patch.object(m.S, "WIN", True), mock.patch.object(m, "sycl_engine", return_value=(exe, None)), \
                mock.patch.object(m.subprocess, "run", return_value=result) as probe:
            cards = m.intel_gpus()
        self.assertEqual(cards[0]["vram_gb"], 32)
        self.assertEqual(probe.call_args.args[0], [str(exe), "--list-devices"])

    def test_windows_runtime_search_includes_tcm_dependencies(self):
        m = self.mod()
        with tempfile.TemporaryDirectory() as d:
            oneapi = Path(d)
            compiler = oneapi / "compiler" / "latest" / "bin"
            tcm = oneapi / "tcm" / "latest" / "bin"
            compiler.mkdir(parents=True)
            tcm.mkdir(parents=True)
            with mock.patch.dict("os.environ", {"ONEAPI_ROOT": d, "LOCALAPPDATA": d}, clear=True), \
                    mock.patch.object(m.shutil, "which", return_value=None):
                dirs = m.sycl_runtime_dirs()
            self.assertEqual(dirs, [str(compiler), str(tcm)])

    def test_windows_config_pins_selected_gpu_and_preserves_explicit_selector(self):
        m = self.mod()
        cfg = self.cfg(m)
        cfg["gpu"] = 1
        exe = m.ROOT / "build-sycl-win" / "strata.exe"
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(m, "sycl_runtime_dirs", return_value=[]):
            c = m.to_sycl(cfg, exe, 64.0, {}, 32.0, "level_zero")
        self.assertEqual(c["env"]["ONEAPI_DEVICE_SELECTOR"], "level_zero:1")
        self.assertNotIn("SYCL_CACHE_PERSISTENT", c["env"])
        cfg["env"] = {"ONEAPI_DEVICE_SELECTOR": "level_zero:2", "SYCL_CACHE_PERSISTENT": "0"}
        with mock.patch.object(m, "sycl_runtime_dirs", return_value=[]):
            c = m.to_sycl(cfg, exe, 64.0, {}, 32.0, "level_zero")
        self.assertEqual(c["env"]["ONEAPI_DEVICE_SELECTOR"], "level_zero:2")
        self.assertEqual(c["env"]["SYCL_CACHE_PERSISTENT"], "0")

    def test_windows_native_probe_retains_a_series_id_for_fp64_emulation(self):
        m = self.mod()
        exe = m.ROOT / "build-sycl-win" / "strata.exe"
        result = mock.Mock(returncode=0, stdout='[{"index":0,"name":"Arc A770","vram_gb":16,"driver":"level_zero","devid":"56a0","free_memory_supported":true,"command_graph_supported":true}]', stderr="")
        with mock.patch.object(m.S, "WIN", True), mock.patch.object(m, "sycl_engine", return_value=(exe, None)), \
                mock.patch.object(m.subprocess, "run", return_value=result), \
                mock.patch.object(m, "sycl_runtime_dirs", return_value=[]):
            card = m.intel_gpus()[0]
            c = m.to_sycl(self.cfg(m), exe, 64.0, {}, card["vram_gb"], card["driver"], card["devid"])
        self.assertEqual(card["devid"], "56a0")
        self.assertEqual(c["env"]["IGC_EnableDPEmulation"], "1")
        self.assertEqual(c["env"]["OverrideDefaultFP64Settings"], "1")
        self.assertEqual(c["env"]["NEOReadDebugKeys"], "1")
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "1")

    def test_windows_gpu_probe_does_not_fall_back_to_cpu(self):
        m = self.mod()
        result = mock.Mock(returncode=1, stdout="[]", stderr="no GPU")
        with mock.patch.object(m.S, "WIN", True), mock.patch.object(m, "sycl_engine", return_value=(Path("strata.exe"), None)), \
                mock.patch.object(m.subprocess, "run", return_value=result), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                m.intel_gpus()

    def test_windows_gpu_probe_rejects_unsupported_backends_and_capabilities(self):
        m = self.mod()
        valid = {"index": 0, "name": "Arc", "vram_gb": 32, "driver": "level_zero",
                 "free_memory_supported": True, "command_graph_supported": True}
        cases = [({"driver": "opencl"}, "requires Level Zero"),
                 ({"driver": "unknown"}, "requires Level Zero"),
                 ({"free_memory_supported": False}, "free VRAM query is unavailable"),
                 ({"command_graph_supported": False}, "command graphs are unavailable"),
                 ({"free_memory_supported": None}, "free VRAM query is unavailable")]
        for fields, message in cases:
            with self.subTest(fields=fields):
                result = mock.Mock(returncode=0, stdout=json.dumps([{**valid, **fields}]), stderr="")
                out = io.StringIO()
                with mock.patch.object(m.S, "WIN", True), \
                        mock.patch.object(m, "sycl_engine", return_value=(Path("strata.exe"), None)), \
                        mock.patch.object(m.subprocess, "run", return_value=result), \
                        mock.patch.dict("os.environ", {"ONEAPI_DEVICE_SELECTOR": "opencl:gpu"}), \
                        contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit):
                        m.intel_gpus()
                self.assertIn(message, out.getvalue())

    def test_windows_gpu_probe_rejects_old_engine_without_capability_fields(self):
        m = self.mod()
        result = mock.Mock(returncode=0, stdout='[{"index":0,"driver":"level_zero"}]', stderr="")
        out = io.StringIO()
        with mock.patch.object(m.S, "WIN", True), \
                mock.patch.object(m, "sycl_engine", return_value=(Path("strata.exe"), None)), \
                mock.patch.object(m.subprocess, "run", return_value=result), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                m.intel_gpus()
        self.assertIn("rebuild the native engine", out.getvalue())

    def test_windows_display_detection_does_not_need_an_engine(self):
        m = self.mod()
        adapters = [{"name": "Arc B70", "pnp": r"PCI\VEN_8086&DEV_E223", "ram": 4 * 2**30},
                    {"name": "Intel integrated", "pnp": r"PCI\VEN_8086&DEV_1234"},
                    {"name": "NVIDIA", "pnp": r"PCI\VEN_10DE&DEV_E223"}]
        registry = [{"MatchingDeviceId": r"PCI\VEN_8086&DEV_E223",
                     "HardwareInformation.qwMemorySize": (32 * 2**30).to_bytes(8, "little")}]
        cards = m.intel_gpus_windows(adapters, registry)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["vram_gb"], 32)
        self.assertEqual(cards[0]["devid"], "e223")
        self.assertEqual(cards[0]["driver"], "unknown")
        with mock.patch.object(m.S, "WIN", True), mock.patch.object(m, "sycl_engine", return_value=(None, "missing")), \
                mock.patch.object(m, "intel_gpus_windows", return_value=cards):
            self.assertEqual(m.intel_gpus(), cards)

    def test_missing_native_engine_is_built_and_probed(self):
        m = self.mod()
        exe = m.ROOT / "build-sycl-win" / "strata.exe"
        with mock.patch.object(m, "sycl_engine", side_effect=[(None, "missing"), (exe, None)]), \
                mock.patch.object(m.subprocess, "call", return_value=0) as build, \
                mock.patch.object(m, "intel_gpus", return_value=[{"name": "Arc"}]) as probe, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.ensure_windows_engine(), exe)
        self.assertEqual(build.call_args.args[0][-1], str(m.ROOT / "sycl" / "tools" / "build_windows.bat"))
        probe.assert_called_once()

    def test_failed_build_does_not_continue_to_model_download(self):
        m = self.mod()
        with mock.patch.object(m, "sycl_engine", return_value=(None, "missing")), \
                mock.patch.object(m.subprocess, "call", return_value=1), \
                mock.patch.object(m, "intel_gpus") as probe, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                m.ensure_windows_engine()
        probe.assert_not_called()
    def test_pictures_asked_for_are_read_on_the_cpu_and_the_vision_section_survives(self):
        """Intel Arc: the CPU reader (strata-vision) on the host when asked for (--vision yes); "gpu" is read as "cpu" (no SYCL GPU reader)."""
        m = self.mod()
        for asked in (None, "yes", "gpu", "cpu", "on"):
            self.assertEqual(m.intel_vision(asked, 8.0), "cpu")
            self.assertEqual(m.intel_vision(asked, 32.0), "cpu")
        for asked in ("no", "none", "off"):
            self.assertEqual(m.intel_vision(asked, 32.0), "none")
        section = {"exe": "/h/engine-sycl/strata-vision", "mmproj": "/h/models/mmproj.gguf", "model": "/h/models/m.gguf", "gpu": False,
                   "max_tokens": 300}
        cfg = self.cfg(m, ["--vision"])
        cfg["vision"] = section
        c = m.to_sycl(cfg, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertEqual(c["vision"], section)                              # host paths: the reader runs on the host, not in the container
        self.assertIn("--vision", c["args"])

    def test_pictures_are_opt_in_on_the_arc_until_the_engine_fault_is_fixed(self):
        """The first picture on the Arc Pro B70 hung the GPU (xe engine reset, 2026-10-19): off unless asked, --vision yes turns it on."""
        m = self.mod()
        self.assertFalse(m.PICTURES_ON_BY_DEFAULT)
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn('if "--vision" not in argv and not PICTURES_ON_BY_DEFAULT:', src)
        self.assertIn("S.hip_vision = intel_vision", src)
        self.assertIn("S.build_vision_cpu(", src)                           # the CPU reader is compiled by setup's own step

    def test_the_container_shares_dev_shm_with_the_server_for_the_embeddings(self):
        sh = (ROOT / "sycl" / "serve" / "strata-sycl.sh").read_text(encoding="utf-8")
        self.assertIn("-v /dev/shm:/dev/shm", sh)
        self.assertIn("STRATA_SYCL_SHM", sh)

    def test_a_picture_that_cannot_reach_the_container_is_said_once(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("server_intel_t", ROOT / "sycl" / "serve" / "server_intel.py")
        src = (ROOT / "sycl" / "serve" / "server_intel.py").read_text(encoding="utf-8")
        self.assertIn("install_shared_embeddings()", src.split('if __name__ == "__main__":')[1])
        sys.path.insert(0, str(ROOT / "sycl" / "serve"))
        try:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        except ImportError as e:                                           # no oneAPI / xe telemetry import here: the source check above stands
            self.skipTest(str(e))
        finally:
            sys.path.pop(0)
        out = io.StringIO()
        with mock.patch.object(m.S, "combined_embeddings_path", lambda d, n: Path("/tmp/x/req.sve")), contextlib.redirect_stdout(out):
            m.install_shared_embeddings()
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
        self.assertEqual(out.getvalue().count("/dev/shm has no room"), 1)

    def test_windows_battlemage_keeps_upstream_prefill_recommendation(self):
        m = self.mod()
        exe = m.ROOT / "build-sycl-win" / "strata.exe"
        for context, expected in ((32768, "4096"), (65536, "8192"), (131072, "8192")):
            with self.subTest(context=context), mock.patch.object(m, "sycl_runtime_dirs", return_value=[]):
                cfg = self.cfg(m)
                m.drop(cfg["args"], "--max-context", True)
                cfg["args"] += ["--max-context", str(context)]
                result = m.to_sycl(cfg, exe, 64.0, {}, 32.0, "level_zero", "e223")
            self.assertEqual(m.flag(result["args"], "--prefill"), expected)
            self.assertNotIn("STRATA_PREFILL_MMQ", result["env"])


if __name__ == "__main__":
    unittest.main()
