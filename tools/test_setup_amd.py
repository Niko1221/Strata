"""Tests for setup.py's AMD card detection on a mocked KFD topology (/sys/class/kfd + /sys/class/drm): the arch
names from gfx_target_version, the CPU node skipped, HIP numbering, product names, which cards are supported and
which TheRock index each family installs from.  No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_amd
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def fake_sysfs(root: Path, nodes: list) -> None:
    """nodes: (gfx_target_version, simd_count, render_minor, product_name or None, vram_bytes)"""
    for i, (ver, simd, minor, name, vram) in enumerate(nodes):
        n = root / "class/kfd/kfd/topology/nodes" / str(i)
        n.mkdir(parents=True)
        (n / "properties").write_text(f"cpu_cores_count {0 if simd else 12}\nsimd_count {simd}\n"
                                      f"gfx_target_version {ver}\ndrm_render_minor {minor}\n")
        if simd:
            d = root / f"class/drm/renderD{minor}/device"
            d.mkdir(parents=True)
            (d / "mem_info_vram_total").write_text(str(vram))
            if name is not None:
                (d / "product_name").write_text(name + "\n")


class KfdDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.win = setup.WIN
        setup.WIN = False

    def tearDown(self):
        setup.WIN = self.win
        self.tmp.cleanup()

    def test_every_family(self):
        fake_sysfs(self.root, [
            (0, 0, 0, None, 0),                                   # the CPU node: skipped
            (110001, 120, 128, "", 16 << 30),                     # gfx1101 without a product name
            (120000, 64, 129, None, 16 << 30),                    # gfx1200, no product_name file
            (120001, 128, 130, "AMD Radeon AI PRO R9700", 32 << 30),
            (110002, 64, 131, None, 8 << 30),                     # gfx1102: listed, not supported
            (100306, 4, 132, None, 512 << 20),                    # an integrated gfx1036: listed, not supported
            (110000, 192, 133, "Radeon RX 7900 XTX", 24 << 30),
        ])
        g = setup.amd_gpus(str(self.root))
        self.assertEqual([x["arch"] for x in g], ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1036", "gfx1100"])
        self.assertEqual([x["index"] for x in g], [0, 1, 2, 3, 4, 5])          # HIP numbers: GPU nodes only
        self.assertEqual(g[0]["name"], setup.AMD_NAMES["gfx1101"])
        self.assertEqual(g[1]["name"], setup.AMD_NAMES["gfx1200"])
        self.assertEqual(g[2]["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(g[3]["name"], "AMD Radeon (gfx1102)")
        self.assertAlmostEqual(g[2]["vram_gb"], 32.0)
        ok = [x["arch"] for x in g if setup.amd_problem(x) is None]
        self.assertEqual(ok, ["gfx1101", "gfx1200", "gfx1201", "gfx1100"])
        self.assertIn("gfx1102", setup.amd_problem(g[3]))
        self.assertIn("gfx1036", setup.amd_problem(g[4]))

    def test_no_kfd(self):
        self.assertEqual(setup.amd_gpus(str(self.root)), [])

    def test_rocm_index_per_family(self):
        for arch in setup.AMD_ARCHS:
            self.assertIn(arch, setup.ROCM_INDEXES)
        self.assertTrue(setup.ROCM_INDEXES["gfx1101"].endswith("/gfx110X-dgpu/"))
        self.assertTrue(setup.ROCM_INDEXES["gfx1200"].endswith("/gfx120X-all/"))
        self.assertEqual(setup.ROCM_INDEXES["gfx1101"], setup.ROCM_INDEXES["gfx1100"])
        self.assertEqual(setup.ROCM_INDEXES["gfx1200"], setup.ROCM_INDEXES["gfx1201"])


class GpuLists(unittest.TestCase):
    """--gpus with AMD cards: every chosen card must be supported; the first is the main one."""
    AMD = [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "arch": "gfx1201"},
           {"index": 1, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"},
           {"index": 2, "name": "AMD Radeon (gfx1036)", "vram_gb": 0.5, "arch": "gfx1036"},
           {"index": 3, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 24.0, "arch": "gfx1100"}]

    def setUp(self):
        self.say = setup.say
        setup.say = lambda *a, **k: None

    def tearDown(self):
        setup.say = self.say

    def test_list_in_order(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("1,0", self.AMD)], [1, 0])
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus(" 0, 3 ", self.AMD)], [0, 3])

    def test_all_is_every_supported_card_most_vram_first(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("all", self.AMD)], [1, 3, 0])

    def test_refusals(self):
        for text in ("1,2", "1,7", "1", "1,1", "x,y"):
            with self.assertRaises(SystemExit, msg=text):
                setup.amd_parse_gpus(text, self.AMD)

    def test_wheels_hold_one_family(self):
        with tempfile.TemporaryDirectory() as d:        # no system ROCm there: the wheels would be needed
            old = setup.os.environ.get("ROCM_PATH")
            setup.os.environ["ROCM_PATH"] = d
            try:
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
            finally:
                if old is None:
                    del setup.os.environ["ROCM_PATH"]
                else:
                    setup.os.environ["ROCM_PATH"] = old

    def test_runtime_only_system_rocm_falls_back_to_the_wheels(self):
        """#446: a system ROCm 7 with hipcc and libhipblas but no HIP development files (no hip-lang CMake package, no
        hip_runtime.h) is not used for the build: a warning says what is missing and the wheels path follows (here
        it stops at the two-family check, which only the wheels path makes); with the files it is used as before."""
        said = []
        setup.say = lambda msg="": said.append(msg)          # tearDown puts the real one back
        with tempfile.TemporaryDirectory() as d:
            sysroot = Path(d)
            for rel, text in (("bin/hipcc", ""), ("lib/libhipblas.so.3", ""),
                              ("include/rocm-core/rocm_version.h",
                               "#define ROCM_VERSION_MAJOR 7\n#define ROCM_VERSION_MINOR 14\n")):
                (sysroot / rel).parent.mkdir(parents=True, exist_ok=True)
                (sysroot / rel).write_text(text)
            with mock.patch.dict(setup.os.environ, {"ROCM_PATH": d}):
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
                self.assertIn(f"the ROCm in {sysroot} has no HIP development files (lib/cmake/hip-lang/hip-lang-"
                              "config.cmake, include/hip/hip_runtime.h): using AMD's wheels", "\n".join(said))
                self.assertIn("two GPU families", "\n".join(said))
                for lib in ("lib64", "lib"):                 # either place CMake looks
                    with self.subTest(lib=lib):
                        cfg = sysroot / lib / "cmake/hip-lang/hip-lang-config.cmake"
                        cfg.parent.mkdir(parents=True, exist_ok=True)
                        cfg.write_text("")
                        said.clear()
                        with self.assertRaises(SystemExit):
                            setup.rocm_root(["gfx1100", "gfx1201"])
                        self.assertIn("(include/hip/hip_runtime.h)", "\n".join(said))
                        (sysroot / "include/hip").mkdir(parents=True, exist_ok=True)
                        (sysroot / "include/hip/hip_runtime.h").write_text("")
                        self.assertEqual(setup.rocm_root(["gfx1100", "gfx1201"]), (sysroot, [str(sysroot / "lib")]))
                        cfg.unlink()
                        (sysroot / "include/hip/hip_runtime.h").unlink()

    def test_build_for_every_arch(self):
        """build_engine_hip compiles for the set of the chosen cards' archs and records it in BUILD.json."""
        calls = {}
        saved = {k: getattr(setup, k) for k in ("ROOT", "rocm_root", "cmake_build", "source_hash", "source_version",
                                                "ok", "shutil")}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def fake_build(src, bdir, target, defs, vcvars, bat):
                calls["defs"] = defs
                (bdir).mkdir(parents=True, exist_ok=True)
                (bdir / setup.EXE).write_text("engine")

            class Sh:
                which = staticmethod(lambda name: "/usr/bin/" + name)
                copy2 = staticmethod(lambda a, b: Path(b).write_text(Path(a).read_text()))
            setup.ROOT, setup.cmake_build, setup.ok, setup.shutil = root, fake_build, lambda *a: None, Sh
            setup.source_hash, setup.source_version = (lambda *a: "h"), (lambda: "0.1.31")
            setup.rocm_root = lambda archs: (calls.setdefault("archs", archs) and root, [str(root / "lib")])
            try:
                setup.build_engine_hip({"arch": "gfx1201", "archs": ["gfx1201", "gfx1100", "gfx1201"]}, root)
                self.assertEqual(calls["archs"], ["gfx1100", "gfx1201"])
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201", calls["defs"])
                import json
                self.assertEqual(json.loads((root / "engine" / "BUILD.json").read_text())["archs"],
                                 ["gfx1100", "gfx1201"])
                calls.clear()                            # one of those cards alone: already built, no compile
                setup.build_engine_hip({"arch": "gfx1100"}, root)
                self.assertNotIn("defs", calls)
                setup.build_engine_hip({"arch": "gfx1101"}, root)   # another arch: compiled again
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1101", calls["defs"])
            finally:
                for k, v in saved.items():
                    setattr(setup, k, v)


class WindowsDetection(unittest.TestCase):
    """Windows: the AMD cards from the display adapters (mocked Win32_VideoController rows and display-class registry
    values), the HIP runtime's numbering from `strata-device --list-devices`, and the card setup chose matched to its
    HIP ordinal (#325: an integrated Radeon is HIP's device 0)."""
    ADAPTERS = [  # Win32_VideoController: name, PNPDeviceID, AdapterRAM (32-bit: at most 4 GB)
        {"name": "NVIDIA GeForce RTX 5070", "pnp": r"PCI\VEN_10DE&DEV_2F04&SUBSYS_1234&REV_A1\4&1", "ram": 4293918720},
        {"name": "AMD Radeon RX 9070 XT", "pnp": r"PCI\VEN_1002&DEV_7550&SUBSYS_0E3B1002&REV_C0\6&2", "ram": 4293918720},
        {"name": "AMD Radeon(TM) Graphics", "pnp": r"PCI\VEN_1002&DEV_164E&SUBSYS_88771043&REV_C1\4&3", "ram": 536870912},
        {"name": "AMD Radeon RX 7800 XT", "pnp": r"PCI\VEN_1002&DEV_7499&SUBSYS_0000&REV_C8\6&4", "ram": 4293918720},
    ]
    REGISTRY = [  # the display class's driver instances: 64-bit VRAM size (as int, or as bytes)
        {"DriverDesc": "NVIDIA GeForce RTX 5070", "MatchingDeviceId": r"PCI\VEN_10DE&DEV_2F04",
         "HardwareInformation.qwMemorySize": 12 << 30},
        {"DriverDesc": "AMD Radeon RX 6600", "MatchingDeviceId": r"PCI\VEN_1002&DEV_73FF",     # a removed card
         "HardwareInformation.qwMemorySize": 8 << 30},
        {"DriverDesc": "AMD Radeon RX 9070 XT", "MatchingDeviceId": r"PCI\VEN_1002&DEV_7550&REV_C0",
         "HardwareInformation.qwMemorySize": (16 << 30).to_bytes(8, "little"), "DriverVersion": "32.0.21013.1000"},
        {"DriverDesc": "AMD Radeon(TM) Graphics", "MatchingDeviceId": r"PCI\VEN_1002&DEV_164E&REV_C1",
         "HardwareInformation.qwMemorySize": 512 << 20},
        {"DriverDesc": "AMD Radeon RX 7800 XT", "MatchingDeviceId": r"PCI\VEN_1002&DEV_7499",
         "HardwareInformation.qwMemorySize": 16 << 30},
    ]
    LIST = ("device 0: AMD Radeon(TM) Graphics\n  arch gfx1036, 28.1 GiB, wave32\n"
            "  cannot run: GPU 0 (AMD Radeon(TM) Graphics, gfx1036) is not an architecture this Strata engine was "
            "compiled for (gfx1100,gfx1101,gfx1102,gfx1200,gfx1201,gfx1030); ...\n"
            "device 1: AMD Radeon RX 9070 XT\n  arch gfx1201, 15.9 GiB, wave32\n"
            "device 2: AMD Radeon RX 7800 XT\n  arch gfx1101, 16.0 GiB, wave32\n")

    def test_adapters(self):
        g = setup.amd_gpus_windows(self.ADAPTERS, self.REGISTRY)
        self.assertEqual([x["name"] for x in g], ["AMD Radeon RX 9070 XT", "AMD Radeon(TM) Graphics",
                                                  "AMD Radeon RX 7800 XT"])     # AMD only, present only, in order
        self.assertEqual([x["index"] for x in g], [0, 1, 2])
        self.assertEqual(g[0]["arch"], "gfx1201")                               # by PCI device id
        self.assertEqual(g[2]["arch"], "gfx1101")                               # an unlisted id: by its name
        self.assertTrue(g[1]["arch"].startswith("unknown"))                     # the iGPU: listed, not supported
        self.assertAlmostEqual(g[0]["vram_gb"], 16.0)                           # the registry's 64-bit size
        self.assertEqual(g[0]["driver"], "32.0.21013.1000")
        self.assertEqual([setup.amd_problem(x) is None for x in g], [True, False, True])

    def test_registry_alone(self):
        """No WMI answer: the registry's own list (which can hold a removed card)."""
        g = setup.amd_gpus_windows([], self.REGISTRY)
        self.assertEqual([x["arch"] for x in g if not x["arch"].startswith("unknown")], ["gfx1201", "gfx1101"])

    def test_arch_names(self):
        for name, arch in (("AMD Radeon RX 9070 GRE", "gfx1201"), ("AMD Radeon AI PRO R9700", "gfx1201"),
                           ("AMD Radeon RX 9060 XT", "gfx1200"), ("AMD Radeon RX 7900 GRE", "gfx1100"),
                           ("AMD Radeon PRO W7800", "gfx1100"), ("AMD Radeon RX 7700 XT", "gfx1101"),
                           ("AMD Radeon RX 7600", "gfx1102"), ("AMD Radeon RX 6950 XT", "gfx1030"),
                           ("AMD Radeon RX 6800M", ""), ("AMD Radeon 780M Graphics", ""), ("AMD Radeon RX 7700S", "")):
            self.assertEqual(setup.win_amd_arch(None, name), arch, name)
        self.assertEqual(setup.win_amd_arch(0x744C, "whatever"), "gfx1100")

    def test_list_devices(self):
        h = setup.hip_devices(text=self.LIST)
        self.assertEqual([(x["index"], x["arch"]) for x in h], [(0, "gfx1036"), (1, "gfx1201"), (2, "gfx1101")])
        self.assertAlmostEqual(h[1]["vram_gb"], 15.9)
        self.assertIn("not an architecture", h[0]["cannot_run"])
        self.assertIsNotNone(setup.amd_problem(h[0]))
        self.assertIsNone(setup.amd_problem(h[1]))
        cannot = setup.hip_devices(text="device 0: AMD Radeon RX 9070 XT\n  arch gfx1201, 15.9 GiB, wave32\n"
                                        "  cannot run: GPU 0 runs wave64\n")
        self.assertEqual(setup.amd_problem(cannot[0]), "GPU 0 runs wave64")   # the engine's own check counts
        self.assertEqual(setup.hip_devices(text="(no GPU device)\n"), [])

    def test_the_card_by_its_hip_ordinal(self):
        """#325: setup's GPU 0 (the 9070 XT, display order) is HIP's device 1 behind the iGPU."""
        listed = setup.amd_gpus_windows(self.ADAPTERS, self.REGISTRY)
        hip = setup.hip_devices(text=self.LIST)
        self.assertEqual(setup.hip_match(listed[0], listed, hip)["index"], 1)
        self.assertEqual(setup.hip_match(listed[2], listed, hip)["index"], 2)
        two = [{"index": 0, "arch": "gfx1201"}, {"index": 1, "arch": "gfx1201"}]   # two of a kind: by rank
        self.assertEqual(setup.hip_match(two[1], two, [{"index": 0, "arch": "gfx1036"}, {"index": 1, "arch": "gfx1201"},
                                                       {"index": 2, "arch": "gfx1201"}])["index"], 2)
        self.assertIsNone(setup.hip_match({"index": 0, "arch": "gfx1100"}, [{"index": 0, "arch": "gfx1100"}], hip))

    def test_hip_card(self):
        listed = setup.amd_gpus_windows(self.ADAPTERS, self.REGISTRY)
        with mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: setup_hip(self.LIST)), \
                mock.patch.object(setup, "ok", lambda *a: None):
            g = setup.hip_card(Path("engine"), listed[0], listed)
        self.assertEqual((g["index"], g["count"], g["arch"], g["vram_gb"]), (1, 3, "gfx1201", 15.9))
        with mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: []), \
                mock.patch.object(setup, "say", lambda *a, **k: None), self.assertRaises(SystemExit):
            setup.hip_card(Path("engine"), listed[0], listed)          # no AMD driver: stops before the download
        with mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: setup_hip(self.LIST)[:1]), \
                mock.patch.object(setup, "say", lambda *a, **k: None), self.assertRaises(SystemExit):
            setup.hip_card(Path("engine"), listed[0], listed)          # HIP does not list the card

    def test_windows_dispatch(self):
        """amd_gpus() on Windows: HIP's numbering once the HIP engine answers, the display adapters before."""
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: None), \
                mock.patch.object(setup, "amd_gpus_windows", lambda adapters=None, registry=None: ["display"]):
            self.assertEqual(setup.amd_gpus(), ["display"])
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: ["hip"]), \
                mock.patch.object(setup, "amd_gpus_windows", lambda adapters=None, registry=None: ["display"]):
            self.assertEqual(setup.amd_gpus(), ["hip"])

    def test_no_hip_engine_no_probe(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            (eng / "strata-device.exe").write_text("x")
            (eng / "BUILD.json").write_text('{"backend": "cuda"}')     # an NVIDIA engine is never asked
            self.assertIsNone(setup.hip_devices(eng / "strata-device.exe"))

    def test_prebuilt_hip_zip(self):
        """get_prebuilt_hip: a published zip (here a local folder) is unpacked into engine/; one without code for the
        card, or older than the first Windows HIP release, is refused."""
        import json
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pub = root / "pub"
            pub.mkdir()

            def publish(meta):
                with zipfile.ZipFile(pub / setup.WIN_HIP_ASSET, "w") as z:
                    z.writestr("strata.exe", "engine")
                    z.writestr("strata-device.exe", "probe")
                    z.writestr("rocm/bin/amdhip64_7.dll", "dll")
                    z.writestr("BUILD.json", json.dumps(meta))
            ver = ".".join(map(str, setup.WIN_HIP_MIN_ENGINE))
            good = {"source": "prebuilt", "backend": "hip", "version": ver, "archs": ["gfx1100", "gfx1201"],
                    "lib_dirs": ["rocm/bin"]}
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "say", lambda *a, **k: None), \
                    mock.patch.object(setup, "ok", lambda *a: None), mock.patch.object(setup, "warn", lambda *a: None):
                publish({**good, "archs": ["gfx1100"]})
                self.assertIsNone(setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"}))
                publish({**good, "version": "0.1.30"})
                self.assertIsNone(setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"}))
                publish(good)
                eng = setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"})
                self.assertEqual(eng, root / "engine")
                self.assertTrue((eng / "rocm" / "bin" / "amdhip64_7.dll").exists())
                self.assertEqual(setup.hip_lib_dirs(eng), [eng / "rocm" / "bin"])
                (pub / setup.WIN_HIP_ASSET).unlink()           # installed: kept, nothing downloaded again
                self.assertEqual(setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"}), eng)


_HIP_DEVICES = setup.hip_devices                      # the real parser, for the tests that mock setup.hip_devices


def setup_hip(text):
    return _HIP_DEVICES(text=text)


class WindowsHipVision(unittest.TestCase):
    def test_no_cpu_encoder_on_windows(self):
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "warn", lambda *a: None):
            self.assertEqual(setup.hip_vision("cpu"), "none")
            self.assertEqual(setup.hip_vision("yes"), "none")
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "warn", lambda *a: None):
            self.assertEqual(setup.hip_vision("cpu"), "cpu")


class HipRuntimeBesideExe(unittest.TestCase):
    """#468 #461: the bundled HIP runtime (and amd_comgr) goes next to strata.exe, so an AMD driver's System32 copy is
    not found first; rocBLAS and the rest stay in rocm/bin."""

    def test_copies_only_the_runtime(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            rb = eng / "rocm" / "bin"
            rb.mkdir(parents=True)
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "lib_dirs": ["rocm/bin"]}))
            for n, data in (("amdhip64_7.dll", b"hip-3686"), ("amd_comgr.dll", b"comgr"), ("rocblas.dll", b"blas")):
                (rb / n).write_bytes(data)
            setup.hip_runtime_beside_exe(eng)
            self.assertEqual((eng / "amdhip64_7.dll").read_bytes(), b"hip-3686")
            self.assertEqual((eng / "amd_comgr.dll").read_bytes(), b"comgr")
            self.assertFalse((eng / "rocblas.dll").exists())          # finds its kernels relative to rocm/bin
            (rb / "amdhip64_7.dll").write_bytes(b"hip-3690-newer")    # a newer zip: replaced on the next start
            setup.hip_runtime_beside_exe(eng)
            self.assertEqual((eng / "amdhip64_7.dll").read_bytes(), b"hip-3690-newer")

    def test_no_rocm_bin_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            setup.hip_runtime_beside_exe(Path(d))                   # no BUILD.json (a CUDA or Linux engine)
            self.assertEqual(list(Path(d).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
