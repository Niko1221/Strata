"""Tests for setup.py's AMD card detection on a mocked KFD topology (/sys/class/kfd + /sys/class/drm): the arch
names from gfx_target_version, the CPU node skipped, HIP numbering, product names, which cards are supported and
which TheRock index each family installs from.  No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_amd
"""
from __future__ import annotations

import sys
import tempfile
import unittest
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
        self.win = setup.WIN
        setup.WIN = False                         # these mock the Linux paths (KFD, /opt/rocm, TheRock wheels)

    def tearDown(self):
        setup.say = self.say
        setup.WIN = self.win

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
    """Windows: hipInfo.exe's own report and the driver name -> architecture map (no GPU, no SDK needed)."""
    HIPINFO = """\
device#                           0
  Name:                           AMD Radeon RX 9070 XT
  totalGlobalMem:                 15.92 GB
  gcnArchName:                    gfx1201:sramecc+:xnack-
  isIntegrated:                   0
device#                           1
  Name:                           AMD Radeon Graphics
  totalGlobalMem:                 4.00 GB
  gcnArchName:                    gfx1103:sramecc+:xnack-
"""

    def test_hipinfo(self):
        cards = setup.hip_info(self.HIPINFO)
        self.assertEqual([c["index"] for c in cards], [0, 1])
        self.assertEqual(cards[0]["name"], "AMD Radeon RX 9070 XT")
        self.assertEqual(cards[0]["arch"], "gfx1201")        # the ":" feature suffix is cut off
        self.assertAlmostEqual(cards[0]["vram_gb"], 15.92)
        self.assertEqual(cards[1]["arch"], "gfx1103")        # listed here, then refused by amd_problem
        self.assertEqual([setup.amd_problem(c) is None for c in cards], [True, False])

    def test_hipinfo_empty(self):
        self.assertEqual(setup.hip_info(""), [])

    def test_arch_from_name(self):
        for name, arch in (("AMD Radeon RX 9070 XT", "gfx1201"), ("AMD Radeon RX 9070", "gfx1201"),
                           ("AMD Radeon AI PRO R9700", "gfx1201"), ("AMD Radeon RX 9060 XT", "gfx1200"),
                           ("AMD Radeon RX 7900 XTX", "gfx1100"), ("AMD Radeon RX 7800 XT", "gfx1101"),
                           ("AMD Radeon RX 7700 XT", "gfx1101"), ("AMD Radeon RX 7600", "gfx1102"),
                           ("AMD Radeon RX 6900 XT", "gfx1030"), ("AMD Radeon RX 6800", "gfx1030"),
                           ("AMD Radeon RX 6700 XT", "gfx1031"), ("AMD Radeon RX 6600", "gfx1032")):
            self.assertEqual(setup.amd_arch_from_name(name), arch, name)
        self.assertIsNone(setup.amd_arch_from_name("AMD Radeon Graphics"))   # integrated: not placed
        self.assertIsNone(setup.amd_arch_from_name("NVIDIA GeForce RTX 5090"))


if __name__ == "__main__":
    unittest.main()
