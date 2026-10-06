"""Tests for setup.py's Intel Arc detection on Windows (mocked Win32_VideoController rows and display-class
registry values): the discrete Arc cards from their PCI device ids with the registry's 64-bit VRAM size, the
integrated GPU listed but not supported, and the no-NVIDIA message naming the Intel card instead of only asking
for the NVIDIA driver. No GPU, no downloads.

    python -m unittest tools.test_setup_intel_windows
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


ADAPTERS = [  # Win32_VideoController: name, PNPDeviceID, AdapterRAM (32-bit: at most 4 GB)
    {"name": "Intel(R) UHD Graphics 630", "pnp": r"PCI\VEN_8086&DEV_3E98&SUBSYS_08591028&REV_02\3&11583659&0&10",
     "ram": 1073741824},
    {"name": "Intel(R) Arc(TM) Pro B70 Graphics",
     "pnp": r"PCI\VEN_8086&DEV_E223&SUBSYS_17018086&REV_00\6&33BC15E2&0&00080008", "ram": 2147479552},
    {"name": "NVIDIA GeForce GTX 1080 Ti", "pnp": r"PCI\VEN_10DE&DEV_1B06&SUBSYS_36091462&REV_A1\4&23F45F8A&0&0008",
     "ram": 4293918720},
]
REGISTRY = [  # the display class's driver instances: 64-bit VRAM size
    {"DriverDesc": "Intel(R) UHD Graphics 630", "MatchingDeviceId": r"PCI\VEN_8086&DEV_3E98",
     "DriverVersion": "31.0.101.2115"},
    {"DriverDesc": "NVIDIA GeForce GTX 1080 Ti", "MatchingDeviceId": r"pci\ven_10de&dev_1b06",
     "HardwareInformation.qwMemorySize": 11811160064},
    {"DriverDesc": "Intel(R) Arc(TM) Pro B70 Graphics",
     "MatchingDeviceId": r"PCI\VEN_8086&DEV_E223&SUBSYS_17018086",
     "HardwareInformation.qwMemorySize": 34139537408, "DriverVersion": "32.0.101.8805"},
]


class WindowsDetection(unittest.TestCase):
    def test_discrete_arc_and_integrated(self):
        g = setup.intel_gpus_windows(ADAPTERS, REGISTRY)
        self.assertEqual([x["name"] for x in g],
                         ["Intel(R) UHD Graphics 630", "Intel(R) Arc(TM) Pro B70 Graphics"])  # Intel only, in order
        self.assertEqual([x["index"] for x in g], [0, 1])
        self.assertAlmostEqual(g[1]["vram_gb"], 34139537408 / 2 ** 30, places=1)  # the registry's 64-bit size
        self.assertEqual(g[1]["driver"], "32.0.101.8805")
        self.assertEqual(g[1]["pci_id"], 0xE223)
        self.assertIsNone(setup.intel_problem(g[1]))  # the B70 can be used (experimental, Linux-only)
        self.assertIn("Arc", setup.intel_problem(g[0]) + "discrete Arc cards only")

    def test_table_fallback_when_the_registry_has_no_size(self):
        g = setup.intel_gpus_windows(ADAPTERS[:2], [])
        b70 = [x for x in g if x["pci_id"] == 0xE223][0]
        self.assertAlmostEqual(b70["vram_gb"], 32.0)  # sycl/setup_intel.py INTEL_ARC

    def test_registry_alone(self):
        g = setup.intel_gpus_windows([], REGISTRY)
        self.assertEqual({x["pci_id"] for x in g}, {0x3E98, 0xE223})

    def test_device_ids_match_the_sycl_port(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for did, (name, gb) in mod.INTEL_ARC.items():
            self.assertEqual(setup._WIN_INTEL_DID[int(did, 16)], (name, gb), did)

    def test_no_nvidia_message_names_the_intel_card(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(setup, "gpus", lambda: []), \
                mock.patch.object(setup, "amd_gpus", lambda *a, **k: []), \
                mock.patch.object(setup, "intel_gpus",
                                  lambda: [{"index": 1, "name": "Intel(R) Arc(TM) Pro B70 Graphics",
                                            "vram_gb": 31.8, "arch": "xe", "driver": "xe",
                                            "vendor": "intel", "pci_id": 0xE223}]):
            with self.assertRaises(SystemExit):
                # the no-GPU branch of main(): it fails with the Intel hint
                intel = setup.intel_gpus()
                intel_usable = [g for g in intel if setup.intel_problem(g) is None]
                self.assertEqual(len(intel_usable), 1)
                setup.fail("no NVIDIA GPU found (nvidia-smi did not answer)",
                           "install the NVIDIA driver"
                           + ("; Intel (" + ", ".join(f"{g['name']} {g['vram_gb']:.0f} GB" for g in intel_usable) +
                              "): experimental Linux-only, docs/INTEL_ARC.md (--backend sycl on Linux)"
                              if intel_usable else ""))
        self.assertIn("Intel(R) Arc(TM) Pro B70 Graphics", out.getvalue())
        self.assertIn("docs/INTEL_ARC.md", out.getvalue())

    def test_sycl_windows_hands_over_to_setup_intel(self):
        import contextlib
        out = io.StringIO()
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                contextlib.redirect_stdout(out):
            rc = setup.sycl_setup(["--backend", "sycl"])
        self.assertEqual(rc, 0)
        self.assertIn("EXPERIMENTAL", out.getvalue())
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("sycl") / "setup_intel.py")))
        self.assertEqual(cmd[2:], [])

    def test_sycl_engine_windows_finds_the_exe_without_docker(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "build-sycl-aot").mkdir()
            (root / "build-sycl-aot" / "strata.exe").write_text("x")
            import importlib.util
            spec = importlib.util.spec_from_file_location("setup_intel_win", ROOT / "sycl" / "setup_intel.py")
            mod = importlib.util.module_from_spec(spec)
            with mock.patch.object(setup, "WIN", True):
                spec.loader.exec_module(mod)
                with mock.patch.object(mod, "ROOT", root), mock.patch.object(mod.S, "WIN", True):
                    exe, why = mod.sycl_engine()
                    self.assertIsNotNone(exe)
                    self.assertIsNone(why)


if __name__ == "__main__":
    unittest.main()
