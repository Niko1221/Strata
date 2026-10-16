"""Tests for setup.py --backend sycl, the experimental Intel Arc engine (sycl/, PR #423): it warns, stops with a clear
message on Windows, and on Linux hands the run to sycl/setup_intel.py without its --backend flag. The CUDA / HIP
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
    def test_windows_stops_with_the_docs_pointer(self):
        rc, out, call = run(["--backend", "sycl"], win=True)
        self.assertEqual(rc, ("exit", 1))
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("docs/INTEL_ARC.md", out)
        call.assert_not_called()

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

    def test_the_vision_section_is_kept_and_the_cpu_encoder_takes_no_reserve(self):
        """--vision cpu: setup writes the "vision" section (host paths: the server runs the encoder on this PC) and
        --vision with its 700 MiB reserve for the CUDA engine's encoder; the port keeps the section and the flag and
        applies its own reserve rule (the CPU encoder takes no VRAM).  A reserve the user gave is kept."""
        m = self.mod()
        vis = {"exe": "/x/engine-sycl/strata-vision", "mmproj": "/m/mmproj-BF16.gguf", "model": "/m/a.gguf",
               "gpu": False, "max_tokens": 300, "threads": 8}
        cfg = {**self.cfg(m, ["--vision", "--vram-reserve-mib", str(setup.VISION["cpu"]["reserve_mib"])]), "vision": vis}
        c = m.to_sycl(cfg, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertEqual(c["vision"], vis)
        self.assertIn("--vision", c["args"])
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "1024")
        cfg = {**self.cfg(m, ["--vision", "--vram-reserve-mib", "500"]), "vision": vis}
        c = m.to_sycl(cfg, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "500")
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertNotIn("vision", c)                                   # no images: the config as before


class Images(unittest.TestCase):
    """--vision cpu on the SYCL port: the image encoder compiled for the CPU by setup's own build_vision_cpu (the AMD
    path's, #304) into engine-sycl/, once; `yes` / `gpu` leave images off with a pointer.  No compiler runs here:
    cmake_build is replaced."""

    def setUp(self):
        self.m = ToSycl.mod()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "tools" / "vision").mkdir(parents=True)
        (root / "tools" / "vision" / "CMakeLists.txt").write_text("project(strata_vision)\n")
        self.root = root
        self.patches = [mock.patch.object(setup, "ROOT", root), mock.patch.object(self.m, "ROOT", root),
                        mock.patch.object(self.m, "ENGINE_DIR", root / "engine-sycl"),
                        mock.patch.object(self.m, "sycl_version", lambda: "0.1.44.1"),   # read from CMakeLists.txt
                        mock.patch.object(setup, "WIN", False)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def fake_cmake(self, src, bdir, target, defs, vcvars, bat_name):
        self.builds.append((src, bdir, target, defs, vcvars, bat_name))
        (Path(bdir) / "bin").mkdir(parents=True, exist_ok=True)
        (Path(bdir) / "bin" / setup.VEXE).write_bytes(b"encoder")

    def test_vision_choice(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.m.sycl_vision("cpu"), "cpu")
            self.assertEqual(self.m.sycl_vision(None), "none")
            self.assertEqual(self.m.sycl_vision("no"), "none")
            self.assertEqual(out.getvalue(), "")
            self.assertEqual(self.m.sycl_vision("yes"), "none")
            self.assertEqual(self.m.sycl_vision("gpu"), "none")
        self.assertIn("--vision cpu", out.getvalue())

    def test_without_images_the_engine_step_is_the_stub_as_before(self):
        with mock.patch.object(setup, "cmake_build") as cm:
            eng = self.m.build_engine_sycl({"arch": "xe"}, Path("llama"), "none")
        cm.assert_not_called()
        meta = json.loads((eng / "BUILD.json").read_text())
        self.assertEqual(meta, {"version": self.m.sycl_version(), "source": "local", "lib_dirs": []})
        self.assertNotEqual(eng, self.root / "engine-sycl")
        self.assertFalse((self.root / "engine-sycl").exists())

    def test_cpu_encoder_is_built_once_with_the_amd_paths_build(self):
        self.builds = []
        out = io.StringIO()
        with mock.patch.object(setup, "cmake_build", self.fake_cmake), \
                mock.patch.object(self.m.shutil, "which", lambda x: "/usr/bin/g++" if x == "g++" else None), \
                contextlib.redirect_stdout(out):
            eng = self.m.build_engine_sycl({"arch": "xe"}, self.root / "llama", "cpu")
            self.assertEqual(eng, self.root / "engine-sycl")
            self.assertEqual(len(self.builds), 1)
            src, bdir, target, defs, vcvars, bat = self.builds[0]
            self.assertEqual((src, bdir, target), (self.root / "tools" / "vision", self.root / "build-vision", "strata-vision"))
            self.assertEqual(defs, [f"-DLLAMA_DIR={self.root / 'llama'}", "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_PORTABLE=OFF"])
            self.assertIsNone(vcvars)
            self.assertEqual((eng / setup.VEXE).read_bytes(), b"encoder")
            meta = json.loads((eng / "BUILD.json").read_text())
            self.assertEqual(meta["vision"], "cpu")
            self.assertEqual(meta["vision_src"], setup.source_hash(setup.VISION_SOURCES))
            self.assertEqual((meta["version"], meta["source"], meta["lib_dirs"]), (self.m.sycl_version(), "local", []))
            self.m.build_engine_sycl({"arch": "xe"}, self.root / "llama", "cpu")   # a rerun: nothing to compile
            self.assertEqual(len(self.builds), 1)
            (self.root / "tools" / "vision" / "strata_vision.cpp").write_text("// changed\n")
            self.m.build_engine_sycl({"arch": "xe"}, self.root / "llama", "cpu")   # tools/vision changed: again
            self.assertEqual(len(self.builds), 2)
        self.assertIn("Compiling the image encoder (for the CPU)", out.getvalue())

    def test_without_a_compiler_setup_stops_with_the_package_to_install(self):
        out = io.StringIO()
        with mock.patch.object(setup, "cmake_build") as cm, \
                mock.patch.object(self.m.shutil, "which", lambda x: None), contextlib.redirect_stdout(out), \
                self.assertRaises(SystemExit):
            self.m.build_engine_sycl({"arch": "xe"}, self.root / "llama", "cpu")
        cm.assert_not_called()
        self.assertIn("build-essential", out.getvalue())

    def test_install_leaves_images_off_unless_asked(self):
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn('if "--vision" not in argv:\n        sys.argv += ["--vision", "none"]', src)
        self.assertIn("S.hip_vision = sycl_vision", src)
        self.assertIn("S.build_engine_hip = build_engine_sycl", src)


if __name__ == "__main__":
    unittest.main()
