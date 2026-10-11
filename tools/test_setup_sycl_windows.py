"""Tests for the Intel Arc installer and server wrapper on Windows (sycl/setup_intel.py, sycl/serve/server_intel.py,
sycl/serve/xe_telemetry.py): the cards from mocked display adapters and display-class registry values (an Arc Pro
B70 beside the NVIDIA card that drives the desktop and the processor's graphics), the SYCL numbering from
`strata-device --list-devices`, which card setup chooses, the engine check against BUILD.json, the config setup
writes (backend "sycl", native paths, the card's SYCL index) and the engine's environment the server sets.  No GPU,
no oneAPI, no downloads, and no .venv needed (a stand-in replaces jinja2 when it is not installed, see below); runs
on any OS (the Windows switches are patched).

    python tools/test_setup_sycl_windows.py
    python -m unittest tools.test_setup_sycl_windows
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sycl"))
sys.path.insert(0, str(ROOT / "sycl" / "serve"))


# Run with a Python that has no jinja2 (the system one, not the .venv START-HERE.bat creates and setup.py fills from
# requirements.txt), the imports below stop on `import jinja2`.  Upstream's setup.py imports only the standard library
# at module level; the third-party import is serve/frontend.py's, reached through sycl/serve/server_intel.py ->
# serve/server.py: `import jinja2`, `from jinja2.sandbox import ImmutableSandboxedEnvironment`, and an exception class
# derived from `jinja2.exceptions.TemplateError`, all at import time.  Nothing here renders a chat template, so when the
# real package is not installed a stand-in with exactly those names goes into sys.modules; with the real one installed
# nothing is touched.
def _stand_in(name: str, **attrs) -> types.ModuleType:
    """A module under `name` in sys.modules with just these attributes; a dotted name also becomes an attribute of its
    parent, which must already be there."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, mod)
    return mod


if "jinja2" not in sys.modules and importlib.util.find_spec("jinja2") is None:
    _stand_in("jinja2")
    _stand_in("jinja2.exceptions", TemplateError=type("TemplateError", (Exception,), {}))
    _stand_in("jinja2.sandbox", ImmutableSandboxedEnvironment=type("ImmutableSandboxedEnvironment", (), {}))

import setup  # noqa: E402
import setup_intel as SI  # noqa: E402
import server_intel as SRV  # noqa: E402
import xe_telemetry as XE  # noqa: E402

B70 = "Intel(R) Arc(TM) Pro B70 Graphics"
IGPU = "Intel(R) Graphics"
ADAPTERS = [  # Win32_VideoController: name, PNPDeviceID, AdapterRAM (32-bit: at most 4 GB)
    {"name": "NVIDIA GeForce RTX 4070", "pnp": r"PCI\VEN_10DE&DEV_2786&SUBSYS_88D31043&REV_A1\4&1", "ram": 4293918720},
    {"name": B70, "pnp": r"PCI\VEN_8086&DEV_E223&SUBSYS_12348086&REV_01\6&2", "ram": 4293918720},
    {"name": IGPU, "pnp": r"PCI\VEN_8086&DEV_7D55&SUBSYS_88771043&REV_08\3&3", "ram": 134217728},
]
REGISTRY = [  # the display class's driver instances: 64-bit VRAM size (as int, or as bytes)
    {"DriverDesc": "NVIDIA GeForce RTX 4070", "MatchingDeviceId": r"PCI\VEN_10DE&DEV_2786",
     "HardwareInformation.qwMemorySize": 12 << 30},
    {"DriverDesc": "Intel(R) Arc(TM) A770 Graphics", "MatchingDeviceId": r"PCI\VEN_8086&DEV_56A0",   # a removed card
     "HardwareInformation.qwMemorySize": 16 << 30},
    {"DriverDesc": B70, "MatchingDeviceId": r"PCI\VEN_8086&DEV_E223",
     "HardwareInformation.qwMemorySize": (32 << 30).to_bytes(8, "little"), "DriverVersion": "32.0.101.9034"},
    {"DriverDesc": IGPU, "MatchingDeviceId": r"PCI\VEN_8086&DEV_7D55",
     "HardwareInformation.qwMemorySize": 128 << 20},
]
# strata-device.exe --list-devices: the processor's graphics is Level Zero's device 0 here
LIST = ("device 0: Intel(R) Graphics\n  arch intel_gpu_mtl_h, 15.6 GiB\n"
        "  cannot run: GPU 0 (Intel(R) Graphics) has no sub-group size 32\n"
        f"device 1: {B70}\n  arch intel_gpu_bmg_g31, 31.9 GiB\n"
        "device 2: Some Other Device\n  arch unknown, 0.5 GiB\n")


def quiet():
    """setup's output silenced for a test."""
    return mock.patch.multiple(setup, say=lambda *a, **k: None, ok=lambda *a, **k: None, warn=lambda *a, **k: None)


def top_level_version() -> str:
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig"))
    return m.group(1)


class WindowsDetection(unittest.TestCase):
    """The Intel cards from the display adapters, in display order, with the discrete Arc supported and the rest not."""

    def test_adapters(self):
        g = SI.intel_gpus_windows(ADAPTERS, REGISTRY)
        self.assertEqual([x["name"] for x in g], [B70, IGPU])                   # Intel only, present only, in order
        self.assertEqual([x["index"] for x in g], [0, 1])
        self.assertEqual(g[0]["arch"], "intel_gpu_bmg_g31")                     # by PCI device id (E223)
        self.assertAlmostEqual(g[0]["vram_gb"], 32.0)                           # the registry's 64-bit size
        self.assertEqual(g[0]["driver"], "32.0.101.9034")
        self.assertFalse(g[0]["integrated"])
        self.assertTrue(g[1]["arch"].startswith("unknown"))                     # the iGPU: listed, not supported
        self.assertTrue(g[1]["integrated"])
        self.assertIsNone(SI.intel_problem(g[0]))
        self.assertIn("processor's graphics", SI.intel_problem(g[1]))
        self.assertEqual(SI.card_driver(g[0]), "xe")

    def test_table_sizes_before_wmi(self):
        """No registry figure: the card's known size, not WMI's 32-bit one; every discrete Arc the table knows is offered,
        an Alchemist with the i915 rules."""
        ad = [{"name": "", "pnp": r"PCI\VEN_8086&DEV_E20B&SUBSYS_1&REV_00\1", "ram": 4293918720}]
        g = SI.intel_gpus_windows(ad, [])
        self.assertEqual((g[0]["name"], g[0]["arch"]), ("Intel Arc B580", "intel_gpu_bmg_g21"))
        self.assertAlmostEqual(g[0]["vram_gb"], 12.0)
        self.assertIsNone(SI.intel_problem(g[0]))
        b65 = SI.intel_gpus_windows([{"name": "", "pnp": r"PCI\VEN_8086&DEV_E222\1", "ram": 0}], [])
        self.assertEqual((b65[0]["name"], b65[0]["arch"]), ("Intel Arc Pro B65", "intel_gpu_bmg_g31"))
        self.assertIsNone(SI.intel_problem(b65[0]))
        a750 = SI.intel_gpus_windows([{"name": "", "pnp": r"PCI\VEN_8086&DEV_56A1\1", "ram": 0}], [])
        self.assertEqual((a750[0]["name"], a750[0]["arch"], a750[0]["vram_gb"]), ("Intel Arc A750", "intel_gpu_acm_g10", 8.0))
        self.assertIsNone(SI.intel_problem(a750[0]))
        self.assertEqual(SI.card_driver(a750[0]), "i915")

    def test_registry_alone(self):
        """No WMI answer: the registry's own list (which can hold a removed card)."""
        g = SI.intel_gpus_windows([], REGISTRY)
        self.assertEqual([x["arch"] for x in g if not x["arch"].startswith("unknown")],
                         ["intel_gpu_acm_g10", "intel_gpu_bmg_g31"])
        self.assertEqual(len(g), 3)                                             # the removed A770 is listed too
        self.assertEqual(SI._intel_pci_device_id(r"PCI\VEN_1002&DEV_7550"), None)   # AMD is not ours
        self.assertEqual(SI._intel_pci_device_id(r"pci\ven_8086&dev_e223"), 0xE223)


class SyclNumbering(unittest.TestCase):
    """`strata-device --list-devices`: SYCL's numbering, the engine's own refusals, the chosen card."""

    def test_list_devices(self):
        d = SI.sycl_devices(text=LIST)
        self.assertEqual([(x["index"], x["arch"]) for x in d],
                         [(0, "intel_gpu_mtl_h"), (1, "intel_gpu_bmg_g31"), (2, "unknown")])
        self.assertEqual(d[1]["name"], B70)
        self.assertAlmostEqual(d[1]["vram_gb"], 31.9)
        self.assertIn("sub-group size 32", d[0]["cannot_run"])
        self.assertTrue(d[0]["integrated"])
        self.assertEqual(SI.intel_problem(d[0]), d[0]["cannot_run"])          # the engine's own check counts
        self.assertIsNone(SI.intel_problem(d[1]))
        self.assertIn("unknown", SI.intel_problem(d[2]))
        self.assertEqual([x["index"] for x in d if SI.intel_problem(x) is None], [1])   # the B70, never the iGPU
        self.assertEqual(SI.sycl_devices(text="(no device)\n"), [])

    def test_no_sycl_engine_no_probe(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            (eng / "strata-device.exe").write_text("x")
            (eng / "BUILD.json").write_text('{"backend": "hip"}')              # an AMD engine is never asked
            self.assertIsNone(SI.sycl_devices(eng / "strata-device.exe"))
            (eng / "BUILD.json").write_text('{"backend": "sycl"}')
            self.assertIsNone(SI.sycl_devices(eng / "missing.exe"))

    def test_intel_card(self):
        d = SI.sycl_devices(text=LIST)
        with quiet():
            g = SI.intel_card(d[1], d)
            self.assertEqual((g["index"], g["count"], g["arch"]), (1, 3, "intel_gpu_bmg_g31"))
            with self.assertRaises(SystemExit):
                SI.intel_card(d[0], d)                                          # the iGPU: stops before the download


class EngineCheck(unittest.TestCase):
    """engine\\strata.exe with a BUILD.json of backend "sycl" built from these sources."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = Path(self.tmp.name) / "engine"
        self.eng.mkdir()
        self.patches = [mock.patch.object(SI, "ENGINE_DIR", self.eng),
                        mock.patch.object(SI, "ENGINE_EXE", self.eng / "strata.exe"),
                        mock.patch.object(setup, "WIN", True),
                        mock.patch.object(setup, "source_hash", lambda parts: "h" + str(len(parts)))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def build(self, meta):
        (self.eng / "strata.exe").write_text("engine")
        (self.eng / "BUILD.json").write_text(json.dumps(meta))

    def test_states(self):
        exe, why = SI.sycl_engine()
        self.assertIsNone(exe)
        self.assertIn("not built", why)
        self.build({"backend": "cuda", "version": "0.1.41"})
        exe, why = SI.sycl_engine()
        self.assertIsNone(exe)
        self.assertIn("cuda engine", why)
        parts = SI.sycl_sources()                                               # the engine sources plus sycl/'s files
        self.assertEqual(parts[:len(setup.ENGINE_SOURCES)], tuple(setup.ENGINE_SOURCES))
        self.assertIn("sycl/setup_intel.py", parts)
        self.assertTrue(all("__pycache__" not in p for p in parts), "Python's .pyc files must not move the hash")
        src = setup.source_hash(parts)
        self.assertEqual(src, "h" + str(len(parts)))
        self.build({"backend": "sycl", "version": top_level_version(), "src": src})
        self.assertEqual(SI.sycl_engine(), (self.eng / "strata.exe", None))
        self.build({"backend": "sycl", "version": top_level_version(), "src": "old"})
        exe, why = SI.sycl_engine()
        self.assertIsNone(exe)                                                  # a setup: refused
        self.assertIn("build_windows.bat", why)
        warned = []
        with mock.patch.object(setup, "warn", lambda m: warned.append(m)):
            self.assertEqual(SI.sycl_engine(strict=False), (self.eng / "strata.exe", None))   # a start: warned
        self.assertIn("other sources", warned[0])

    def test_version_is_the_top_level_projects(self):
        """sycl/CMakeLists.txt takes its VERSION from ../CMakeLists.txt; the stub BUILD.json and package_windows.py's
        must say that number (upstream's parser then compares it with MIN_ENGINE as for any engine)."""
        self.assertEqual(SI.sycl_version(), top_level_version())
        self.assertRegex(SI.sycl_version(), r"^\d+\.\d+\.\d+")


class ConfigConversion(unittest.TestCase):
    """setup's config for its HIP path -> the SYCL port's on Windows."""

    def cfg(self, ctx, extra=()):
        return {"exe": r"C:\tmp\strata-sycl-x\strata.exe", "backend": "hip", "cwd": str(ROOT), "port": 8085,
                "args": ["--pack", r"D:\Strata-data\packs\coder-iq1_m", "--native", r"D:\Strata-data\models\x-1.gguf",
                         "--ple-gguf", r"D:\Strata-data\models\x-2.gguf", "--expert-profile",
                         str(ROOT / "data" / "expert-profile-coder.bin"), "--expert-cache", "auto", "--prefill", "auto",
                         "--spec", "4", "--spec-min-p", "0.5", "--mtp", r"D:\Strata-data\mtp\rt", "--max-context",
                         str(ctx), "--kv", "int8", "--resident-experts", "--kv-resident", "32768", *extra],
                "tokenizer": r"D:\Strata-data\packs\coder-iq1_m\tokenizer", "model_name": "qwen3.8-flash-next-coder-iq1_m",
                "log": str(ROOT / "strata-coder-iq1_m.log"), "lib_dirs": [], "env": {"STRATA_RESIDENT_PIN": "0"}}

    def test_long_context(self):
        with mock.patch.object(setup, "WIN", True), quiet():
            out = SI.to_sycl(self.cfg(65536), SI.ENGINE_EXE, 64.0, {"sampling": {"temperature": 0.6}}, 32.0, "xe", 1)
        a = out["args"]
        self.assertEqual(out["backend"], "sycl")
        self.assertEqual(out["exe"], str(ROOT / "engine" / "strata.exe"))
        self.assertEqual(out["lib_dirs"], ["engine"])
        self.assertEqual(out["gpu"], 1)                                         # the card's SYCL index
        self.assertNotIn("sycl_root", out)                                      # no container on Windows
        self.assertNotIn("env", out)                                            # the server sets the engine's
        self.assertEqual(a[a.index("--pack") + 1], r"D:\Strata-data\packs\coder-iq1_m")   # native paths
        self.assertIn("--stream-experts", a)
        self.assertNotIn("--resident-experts", a)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "2048")
        self.assertEqual(a[a.index("--prefill") + 1], "4096")
        self.assertEqual(a[a.index("--kv-resident") + 1], "32768")              # 64 GB holds the 0.9 GB of KV
        self.assertEqual(out["sampling"], {"temperature": 0.6})                 # a hand-set block survives a rerun
        self.assertEqual(out["port"], 8085)
        with mock.patch.object(setup, "WIN", True), quiet():                    # a reserve the user gave is kept
            out = SI.to_sycl(self.cfg(65536, ["--vram-reserve-mib", "700"]), SI.ENGINE_EXE, 64.0, {}, 32.0, "xe", 1)
        self.assertEqual(out["args"].count("--vram-reserve-mib"), 1)
        self.assertEqual(out["args"][out["args"].index("--vram-reserve-mib") + 1], "700")

    def test_short_context_and_little_ram(self):
        with mock.patch.object(setup, "WIN", True), quiet():
            out = SI.to_sycl(self.cfg(32768), SI.ENGINE_EXE, 16.0, {}, 32.0, "xe", 0)
        a = out["args"]
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "1024")
        self.assertEqual(a[a.index("--prefill") + 1], "4096")                  # a 24 GB+ card: the Linux rule (INTEL.md)
        self.assertNotIn("--kv-resident", a)
        self.assertEqual(out["gpu"], 0)
        with mock.patch.object(setup, "WIN", True), quiet():
            out = SI.to_sycl(self.cfg(32768), SI.ENGINE_EXE, 16.0, {}, 16.0, "xe", 0)
        self.assertEqual(out["args"][out["args"].index("--prefill") + 1], "auto")   # a smaller card keeps setup's chunks
        with mock.patch.object(setup, "WIN", True), quiet():                    # setup's own choice stays
            out = SI.to_sycl({**self.cfg(32768), "gpu": 2, "gpus_asked": True}, SI.ENGINE_EXE, 16.0, {}, 32.0, "xe", 0)
        self.assertEqual(out["gpu"], 2)
        with mock.patch.object(setup, "WIN", True), quiet():
            out = SI.to_sycl(self.cfg(131072), SI.ENGINE_EXE, 6.0, {}, 32.0, "xe", None)   # 6 GB: no KV streaming, no gpu
        self.assertNotIn("--kv-resident", out["args"])
        self.assertNotIn("gpu", out)

    def test_alchemist_on_windows_takes_the_i915_rules(self):
        with mock.patch.object(setup, "WIN", True), quiet():
            out = SI.to_sycl(self.cfg(32768), SI.ENGINE_EXE, 64.0, {}, 8.0, "i915", 0)
        a = out["args"]
        self.assertNotIn("--stream-experts", a)
        self.assertEqual(a[a.index("--ple-io") + 1], "ram")
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "300")
        self.assertEqual(out["env"]["STRATA_VERIFY_NO_HOST"], "0")             # the server's default is 1: the config wins
        self.assertEqual(out["env"]["IGC_EnableDPEmulation"], "1")
        self.assertNotIn("STRATA_SYCL_BIN", out["env"])                         # a Linux key


class ChildEnv(unittest.TestCase):
    """The engine's environment on Windows, the config's "env" winning."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        (Path(self.tmp.name) / "engine").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_sycl_env(self):
        cfg = {"backend": "sycl", "gpu": 1, "cwd": self.tmp.name, "lib_dirs": ["engine"],
               "env": {"STRATA_SYCL_HEARTBEAT_S": "0", "STRATA_MINE": "1"}}
        with mock.patch.object(SRV, "WIN", True), mock.patch.object(SRV, "physical_cores", lambda: 16):
            env = SRV.sycl_env(cfg, {"PATH": "C:\\x", "SYCL_CACHE_PERSISTENT": "1", "HOME": "h"})
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:1")
        for k, v in (("SYCL_UR_USE_LEVEL_ZERO_V2", "1"), ("SYCL_CACHE_PERSISTENT", "0"),
                     ("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1"), ("ZES_ENABLE_SYSMAN", "1"),
                     ("STRATA_VERIFY_DEVICE_PLAN", "1"), ("STRATA_VERIFY_NO_HOST", "1"), ("STRATA_STAGER_THREADS", "12"),
                     ("STRATA_SYCL_HEARTBEAT_S", "0"), ("STRATA_MINE", "1"), ("HOME", "h")):
            self.assertEqual(env.get(k), v, k)
        self.assertEqual(env["PATH"], str(Path(self.tmp.name) / "engine") + os.pathsep + "C:\\x")
        with mock.patch.object(SRV, "WIN", True), mock.patch.object(SRV, "physical_cores", lambda: 6):
            env = SRV.sycl_env({"backend": "sycl", "cwd": self.tmp.name, "lib_dirs": ["engine", "missing"]}, {})
        self.assertEqual(env["STRATA_STAGER_THREADS"], "6")
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")       # no card named: every Level Zero GPU
        self.assertEqual(env["PATH"], str(Path(self.tmp.name) / "engine"))      # a folder that is not there is skipped
        self.assertEqual(env["STRATA_SYCL_HEARTBEAT_S"], "10")

    def test_installed_wrapper(self):
        orig = SRV.S.child_env
        try:
            with mock.patch.object(SRV, "WIN", True), mock.patch.dict(os.environ, {"PATH": "p"}, clear=True):
                SRV.install_child_env()
                sycl = SRV.S.child_env({"backend": "sycl", "gpu": 0, "cwd": self.tmp.name, "lib_dirs": ["engine"]})
                self.assertEqual(sycl["ONEAPI_DEVICE_SELECTOR"], "level_zero:0")
                cuda = SRV.S.child_env({"backend": "cuda", "gpu": 0})          # upstream's own, untouched
                self.assertNotIn("ONEAPI_DEVICE_SELECTOR", cuda)
                self.assertEqual(cuda["CUDA_VISIBLE_DEVICES"], "0")
        finally:
            SRV.S.child_env = orig


class MonitorReader(unittest.TestCase):
    """The Monitor's GPU reader for a "sycl" config on Windows is the Intel one, never NVML's first card; without a
    Level Zero loader it reads nothing (ok() False), as upstream's Windows HIP state."""

    def test_sysman_without_loader(self):
        with mock.patch.object(XE, "_level_zero", lambda: None):
            g = XE._XeSysman(1)
        self.assertFalse(g.ok())
        self.assertEqual(g.name(), "Intel Arc GPU")
        r = g.read()
        self.assertIsNone(r["mem_used"])
        self.assertIsNone(r["power"])
        with mock.patch.object(XE, "WIN", True), mock.patch.object(XE, "_level_zero", lambda: None):
            self.assertIsInstance(XE.xe_gpu(0), XE._XeSysman)
        with mock.patch.object(XE, "WIN", False):
            self.assertIsInstance(XE.xe_gpu(0), XE._XeGpu)

    def test_struct_layouts(self):
        """The spec's layouts: stype + pNext first, the core device properties inside the Sysman ones."""
        import ctypes
        self.assertEqual(XE._ZesDeviceProperties.core.offset, 16)
        self.assertEqual(XE._ZeDeviceProperties.name.size, 256)
        self.assertEqual(ctypes.sizeof(XE._ZesMemState), 40)                    # 4+4 pad, 8, 4+4 pad, 8, 8
        self.assertEqual(ctypes.sizeof(XE._ZesPowerEnergyCounter), 16)
        self.assertEqual(ctypes.sizeof(XE._ZesEngineStats), 16)
        self.assertEqual(XE._ZesMemProperties.physicalSize.offset, 32)
        self.assertEqual(XE._ZesTempProperties.maxTemperature.offset, 32)

    def test_reader_dispatch(self):
        orig = SRV.T.gpu_reader
        orig_free = SRV.S.Service.free_vram_mib
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg = Path(d) / "strata-coder-iq1_m.json"
                cfg.write_text(json.dumps({"backend": "sycl", "gpu": 1}))
                argv = ["--engine", "strata", "--config", str(cfg)]
                with mock.patch.object(SRV, "WIN", True), mock.patch.object(XE, "_level_zero", lambda: None), \
                        mock.patch.object(XE, "WIN", True):
                    SRV.install_xe_reader(argv)
                    self.assertIsInstance(SRV.T.gpu_reader(1), XE._XeSysman)
                    SRV.install_free_vram(argv)
                    self.assertIsNone(SRV.S.Service.free_vram_mib(object()))      # nothing read: nothing refused
                SRV.T.gpu_reader = orig
                cfg.write_text(json.dumps({"backend": "cuda", "gpu": 0}))
                with mock.patch.object(SRV, "WIN", True), \
                        mock.patch.object(SRV.T, "_Nvml", lambda index=0: mock.Mock(ok=lambda: True, tag="nvml")):
                    SRV.install_xe_reader(["--config", str(cfg)])
                    self.assertEqual(SRV.T.gpu_reader(0).tag, "nvml")           # another backend: NVML as before
        finally:
            SRV.T.gpu_reader = orig
            SRV.S.Service.free_vram_mib = orig_free


class Stubs(unittest.TestCase):
    """The setup.py and serve/server.py entry points the Windows installer and server wrapper replace or call must
    still exist."""

    def test_setup_has_what_install_needs(self):
        for name in ("gpus", "amd_gpus", "amd_problem", "hip_vision", "build_engine_hip", "hipblaslt_table", "ram_gb",
                     "write_run_script", "start", "say", "main", "bench_tips", *SI.WIN_NEEDED):
            self.assertTrue(callable(getattr(setup, name, None)), name)
        for name in ("gpu_list", "child_env", "make_handler", "openai_chunks", "openai_collect", "StrataEngine",
                     "narrate_start"):
            self.assertTrue(hasattr(SRV.S, name), name)
        self.assertTrue(callable(getattr(SRV.S.Service, "free_vram_mib", None)))


if __name__ == "__main__":
    unittest.main()
