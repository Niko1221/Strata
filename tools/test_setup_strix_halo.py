"""Tests for setup.py's recognition of AMD Strix Halo (Ryzen AI Max, Radeon 8060S / 8050S / 8040S, gfx1151): the exact
arch check (never a prefix: gfx1150 / gfx1152 are other chips), the PCI ids and marketing names, the unified-memory
sizing, which card setup picks beside a discrete one, and the Windows engine zip (a zip without gfx1151 code is never used).

The Linux fixture is Aurora's real answer (the maintainers' Strix Halo box, read-only):
  KFD node 1: gfx_target_version 110501, simd_count 80, vendor_id 4098, device_id 5510 (0x1586), drm_render_minor 128
  /sys/class/drm/card1/device: vendor 0x1002, device 0x1586, mem_info_vram_total 2147483648 (a 2 GiB BIOS carve-out),
                               mem_info_gtt_total 120259084288 (112 GiB: ttm.pages_limit=29360128), no product_name
  lspci -nn: c6:00.0 Display controller [0380]: AMD/ATI Strix Halo [Radeon Graphics / Radeon 8050S Graphics /
             Radeon 8060S Graphics] [1002:1586] (rev c1);  MemTotal 127098456 kB
The other chips are synthetic (their KFD gfx_target_version and pci.ids ids).  No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_strix_halo
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

AURORA_RAM_GB = 127098456 * 1024 / 2 ** 30          # /proc/meminfo MemTotal on Aurora: 121.2 GiB
GIB = 1 << 30


def fake_gpu(root: Path, node: int, ver: int, simd: int, minor: int, pci_id: int, vram: int, gtt: int,
             product: str | None = None, card: int | None = None) -> None:
    """One KFD GPU node + its drm device, laid out as the kernel does (see the module docstring)."""
    n = root / "class/kfd/kfd/topology/nodes" / str(node)
    n.mkdir(parents=True, exist_ok=True)
    (n / "properties").write_text(f"cpu_cores_count 0\nsimd_count {simd}\nmem_banks_count 1\n"
                                  f"gfx_target_version {ver}\nvendor_id 4098\ndevice_id {pci_id}\n"
                                  f"drm_render_minor {minor}\nlocal_mem_size 0\n")
    d = root / f"class/drm/renderD{minor}/device"
    d.mkdir(parents=True, exist_ok=True)
    for name, val in (("vendor", "0x1002"), ("device", f"0x{pci_id:04x}"), ("mem_info_vram_total", str(vram)),
                      ("mem_info_gtt_total", str(gtt))):
        (d / name).write_text(val + "\n")
    if product:
        (d / "product_name").write_text(product + "\n")
    if card is not None:                              # /sys/class/drm/card<N>/device (what amd_pci_devices reads)
        c = root / f"class/drm/card{card}/device"
        c.mkdir(parents=True, exist_ok=True)
        for name in ("vendor", "device", "mem_info_vram_total", "mem_info_gtt_total"):
            (c / name).write_text((d / name).read_text())


def cpu_node(root: Path, node: int = 0) -> None:
    n = root / "class/kfd/kfd/topology/nodes" / str(node)
    n.mkdir(parents=True, exist_ok=True)
    (n / "properties").write_text("cpu_cores_count 32\nsimd_count 0\ngfx_target_version 0\nvendor_id 0\ndevice_id 0\n")


class LinuxBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.win = setup.WIN
        setup.WIN = False
        p = mock.patch.object(setup, "ram_gb", return_value=AURORA_RAM_GB)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        setup.WIN = self.win
        self.tmp.cleanup()

    def aurora(self, node=1, minor=128, card=1):
        fake_gpu(self.root, node, 110501, 80, minor, 0x1586, 2 * GIB, 120259084288, None, card)


class ExactArch(unittest.TestCase):
    def test_gfx1151_is_exact(self):
        self.assertTrue(setup.gfx_arch_is("gfx1151", "gfx1151"))
        self.assertTrue(setup.gfx_arch_is("gfx1151:sramecc-:xnack-", "gfx1151"))     # gcnArchName
        for other in ("gfx1150", "gfx1152", "gfx115", "gfx11510", "gfx1103", "gfx1100", "gfx1201", "gfx1030", "", None):
            self.assertFalse(setup.gfx_arch_is(other, "gfx1151"), other)
            self.assertFalse(setup.is_strix_halo({"arch": other}), other)
        self.assertFalse(setup.gfx_arch_is("gfx1151", "gfx115"))                     # no prefix match either way
        self.assertFalse(setup.is_strix_halo({}))

    def test_matches_the_engines_header(self):
        """include/strata/kernels/gfx_arch.hpp is the reference: gfx1151 has the WMMA kernels and the gfx1151 defaults;
        gfx1103 and gfx1152 are outside its list."""
        hpp = (ROOT / "include/strata/kernels/gfx_arch.hpp").read_text(encoding="utf-8")
        wmma = re.search(r"gfx_arch_is_gfx11_wmma.*?\n}", hpp, re.S).group(0)
        listed = set(re.findall(r'"(gfx\d+)"', wmma))
        self.assertIn("gfx1151", listed)
        self.assertNotIn("gfx1103", listed)
        self.assertNotIn("gfx1152", listed)
        self.assertIn('gfx_arch_is(gcn, "gfx1151")', hpp)
        self.assertEqual(setup.STRIX_HALO_ARCH, "gfx1151")

    def test_arch_lists(self):
        self.assertIn("gfx1151", setup.AMD_ARCHS)
        self.assertIn("gfx1151", setup.ROCM_INDEXES)
        self.assertTrue(setup.ROCM_INDEXES["gfx1151"].endswith("/gfx1151/"))
        for a in ("gfx1150", "gfx1152", "gfx1103"):                                  # never supported by accident
            self.assertNotIn(a, setup.AMD_ARCHS)
        self.assertIn("docs/STRIX_HALO.md", setup.AMD_CARDS)


class PciIds(unittest.TestCase):
    def test_strix_halo_id(self):
        # pci.ids: one id for every Strix Halo part (8060S, 8050S, 8040S share it; they differ by CU count)
        self.assertEqual(setup.STRIX_HALO_PCI_IDS, {0x1586})
        self.assertEqual(setup.AMD_IGPU_PCI[0x1586], "gfx1151")

    def test_other_igpus_are_not_strix_halo(self):
        for did, arch in ((0x150E, "gfx1150"), (0x1114, "gfx1152"), (0x1902, "gfx1152"), (0x15BF, "gfx1103"),
                          (0x15C8, "gfx1103"), (0x164F, "gfx1103"), (0x1900, "gfx1103"), (0x1901, "gfx1103")):
            self.assertEqual(setup.AMD_IGPU_PCI[did], arch)
            self.assertNotIn(did, setup.STRIX_HALO_PCI_IDS)
        self.assertEqual([a for a in setup.AMD_IGPU_PCI.values() if a == "gfx1151"], ["gfx1151"])

    def test_windows_id_then_name(self):
        f = setup.win_amd_arch
        self.assertEqual(f(0x1586, ""), "gfx1151")
        self.assertEqual(f(0x1586, "AMD Radeon(TM) Graphics"), "gfx1151")           # the generic name Windows may show
        for name in ("AMD Radeon(TM) 8060S Graphics", "AMD Radeon(TM) 8050S Graphics", "AMD Radeon(TM) 8040S Graphics"):
            self.assertEqual(f(None, name), "gfx1151", name)                          # an id we do not know: by name
        self.assertEqual(f(0x150E, "AMD Radeon(TM) 890M Graphics"), "gfx1150")
        self.assertEqual(f(None, "AMD Radeon(TM) 880M Graphics"), "gfx1150")
        self.assertEqual(f(None, "AMD Radeon(TM) 860M Graphics"), "gfx1152")
        self.assertEqual(f(None, "AMD Radeon(TM) 840M Graphics"), "gfx1152")
        self.assertEqual(f(0x15BF, "AMD Radeon(TM) 780M Graphics"), "gfx1103")
        self.assertEqual(f(None, "AMD Radeon(TM) 760M Graphics"), "gfx1103")
        # the discrete cards keep their own arch
        self.assertEqual(f(0x744C, "AMD Radeon RX 7900 XTX"), "gfx1100")
        self.assertEqual(f(None, "AMD Radeon RX 7900 XTX"), "gfx1100")
        self.assertEqual(f(0x7550, "AMD Radeon RX 9070 XT"), "gfx1201")
        self.assertEqual(f(None, "AMD Radeon RX 7600M XT"), "gfx1102")                # not the 780M rule
        self.assertEqual(f(None, "AMD Radeon RX 6800M"), "")


class AuroraFixture(LinuxBase):
    def test_aurora(self):
        cpu_node(self.root, 0)
        self.aurora()
        g = setup.amd_gpus(str(self.root))
        self.assertEqual(len(g), 1)
        g = g[0]
        self.assertEqual((g["index"], g["arch"], g["pci_id"]), (0, "gfx1151", 0x1586))
        self.assertEqual(g["name"], "AMD Radeon 8060S (Ryzen AI Max, Strix Halo, gfx1151)")     # 80 SIMDs = 40 CUs
        self.assertIsNone(setup.amd_problem(g))
        self.assertTrue(g["uma"] and setup.is_strix_halo(g))
        self.assertAlmostEqual(g["dedicated_gb"], 2.0)                                # the BIOS carve-out
        self.assertAlmostEqual(g["shared_gb"], 112.0)                                 # GTT, below RAM - 6
        self.assertAlmostEqual(g["vram_gb"], 114.0)
        self.assertAlmostEqual(setup.low_ram_vram(g), 2.0)                            # the shared pool is the RAM itself
        self.assertIn("unified memory", setup.amd_mem_text(g))

    def test_marketing_name_by_cu_count(self):
        for simd, name in ((80, "8060S"), (64, "8050S"), (32, "8040S")):
            root = Path(tempfile.mkdtemp(dir=self.root))
            fake_gpu(root, 0, 110501, simd, 128, 0x1586, 2 * GIB, 112 * GIB)
            g = setup.amd_gpus(str(root))[0]
            self.assertIn(name, g["name"])
            self.assertEqual(g["arch"], "gfx1151")
        root = Path(tempfile.mkdtemp(dir=self.root))
        fake_gpu(root, 0, 110501, 48, 128, 0x1586, 2 * GIB, 112 * GIB)                  # an unknown CU count
        self.assertEqual(setup.amd_gpus(str(root))[0]["name"], setup.AMD_NAMES["gfx1151"])

    def test_name_from_sysfs_is_kept(self):
        fake_gpu(self.root, 0, 110501, 80, 128, 0x1586, 2 * GIB, 120259084288, "AMD Radeon 8060S")
        self.assertEqual(setup.amd_gpus(str(self.root))[0]["name"], "AMD Radeon 8060S")

    def test_pci_scan(self):
        self.aurora()
        d = setup.amd_pci_devices(str(self.root))
        self.assertEqual([(x["pci_id"], x["arch"]) for x in d], [(0x1586, "gfx1151")])
        self.assertAlmostEqual(d[0]["gtt_gb"], 112.0)
        self.assertIn(d[0]["pci_id"], setup.STRIX_HALO_PCI_IDS)

    def test_gtt_never_exceeds_the_ram(self):
        fake_gpu(self.root, 0, 110501, 80, 128, 0x1586, 512 << 20, 200 * GIB)         # a bogus 200 GiB GTT
        g = setup.amd_gpus(str(self.root))[0]
        self.assertAlmostEqual(g["shared_gb"], AURORA_RAM_GB - setup.UMA_OS_LEFT_GB)

    def test_no_gtt_file_means_half_the_ram(self):
        fake_gpu(self.root, 0, 110501, 80, 128, 0x1586, 2 * GIB, 0)
        g = setup.amd_gpus(str(self.root))[0]
        self.assertAlmostEqual(g["shared_gb"], AURORA_RAM_GB / 2)

    def test_big_carve_out(self):
        fake_gpu(self.root, 0, 110501, 80, 128, 0x1586, 96 * GIB, 24 * GIB)           # 96 GB carve-out, 32 GB left
        with mock.patch.object(setup, "ram_gb", return_value=30.0):
            g = setup.amd_gpus(str(self.root))[0]
        self.assertAlmostEqual(g["dedicated_gb"], 96.0)
        self.assertAlmostEqual(g["vram_gb"], 96.0 + 24.0)
        self.assertAlmostEqual(setup.low_ram_vram(g), 96.0)                           # that memory is extra to the RAM
        notes = setup.strix_halo_notes(g, 30.0)
        self.assertTrue(any(n.startswith("!") and "carve-out" in n for n in notes))

    def test_small_carve_out(self):
        fake_gpu(self.root, 0, 110501, 80, 128, 0x1586, 512 << 20, 60 * GIB)          # 512 MB carve-out
        g = setup.amd_gpus(str(self.root))[0]
        self.assertAlmostEqual(g["dedicated_gb"], 0.5)


class OtherChips(LinuxBase):
    """Synthetic KFD nodes: gfx_target_version as the kernel reports it."""

    def one(self, ver, pci, vram, gtt=0):
        root = Path(tempfile.mkdtemp(dir=self.root))
        fake_gpu(root, 0, ver, 64, 128, pci, vram, gtt)
        return setup.amd_gpus(str(root))[0]

    def test_strix_point(self):
        g = self.one(110500, 0x150E, 512 << 20, 60 * GIB)
        self.assertEqual(g["arch"], "gfx1150")
        self.assertFalse(setup.is_strix_halo(g) or g.get("uma"))
        self.assertIn("gfx1150", setup.amd_problem(g))
        self.assertIn("not Strix Halo", setup.amd_problem(g))
        self.assertIn("890M", g["name"])

    def test_krackan(self):
        g = self.one(110502, 0x1114, 512 << 20, 30 * GIB)
        self.assertEqual(g["arch"], "gfx1152")
        self.assertFalse(g.get("uma"))
        self.assertIn("gfx1152", setup.amd_problem(g))
        self.assertIn("860M", g["name"])

    def test_phoenix_hawk_point(self):
        g = self.one(110003, 0x15BF, 512 << 20, 30 * GIB)
        self.assertEqual(g["arch"], "gfx1103")
        self.assertFalse(g.get("uma"))
        self.assertIn("gfx1103", setup.amd_problem(g))
        self.assertIn("780M", g["name"])

    def test_discrete_cards(self):
        for ver, arch, pci, vram in ((110000, "gfx1100", 0x744C, 24), (110001, "gfx1101", 0x747E, 16),
                                     (110002, "gfx1102", 0x7480, 8), (120001, "gfx1201", 0x7550, 16),
                                     (100300, "gfx1030", 0x73BF, 16)):
            g = self.one(ver, pci, vram * GIB, 32 * GIB)
            self.assertEqual(g["arch"], arch)
            self.assertFalse(g.get("uma"), arch)
            self.assertFalse(setup.is_strix_halo(g), arch)
            self.assertAlmostEqual(g["vram_gb"], vram)                                # a card's VRAM is not changed
            self.assertEqual(setup.low_ram_vram(g), vram)
            self.assertEqual(setup.amd_problem(g) is None, arch != "gfx1102", arch)


class DualGpu(LinuxBase):
    def test_strix_halo_with_a_discrete_card(self):
        cpu_node(self.root, 0)
        fake_gpu(self.root, 1, 110501, 80, 128, 0x1586, 2 * GIB, 112 * GIB, None, 1)
        fake_gpu(self.root, 2, 110000, 192, 129, 0x744C, 24 * GIB, 16 * GIB, "AMD Radeon RX 7900 XTX", 2)
        g = setup.amd_gpus(str(self.root))
        self.assertEqual([(x["index"], x["arch"]) for x in g], [(0, "gfx1151"), (1, "gfx1100")])
        self.assertTrue(g[0]["uma"] and not g[1].get("uma"))
        self.assertTrue(g[0]["vram_gb"] > g[1]["vram_gb"])                           # 114 GB of unified memory vs 24
        self.assertEqual(min(g, key=setup.amd_rank)["arch"], "gfx1100")              # the card with its own memory
        self.assertEqual([x["arch"] for x in setup.amd_parse_gpus("all", g)], ["gfx1100", "gfx1151"])
        self.assertEqual([x["arch"] for x in setup.amd_parse_gpus("0,1", g)], ["gfx1151", "gfx1100"])
        self.assertEqual(sorted(a for a in {x["arch"] for x in g}), ["gfx1100", "gfx1151"])   # compiled for both if split
        # two TheRock families: a split needs a system ROCm (rocm_root says so); one card does not
        self.assertNotEqual(setup.ROCM_INDEXES["gfx1100"], setup.ROCM_INDEXES["gfx1151"])

    def test_strix_halo_with_strix_point_is_impossible_but_not_confused(self):
        cpu_node(self.root, 0)
        fake_gpu(self.root, 1, 110500, 32, 128, 0x150E, 512 << 20, 30 * GIB)
        fake_gpu(self.root, 2, 110501, 80, 129, 0x1586, 2 * GIB, 112 * GIB)
        g = setup.amd_gpus(str(self.root))
        self.assertEqual([x["arch"] for x in g], ["gfx1150", "gfx1151"])
        self.assertEqual([bool(setup.amd_problem(x)) for x in g], [True, False])
        self.assertEqual([bool(x.get("uma")) for x in g], [False, True])
        usable = [x for x in g if setup.amd_problem(x) is None]
        self.assertEqual(min(usable, key=setup.amd_rank)["index"], 1)

    def test_two_strix_halo_ranked_by_index(self):
        fake_gpu(self.root, 0, 110501, 80, 128, 0x1586, 2 * GIB, 112 * GIB)
        fake_gpu(self.root, 1, 110501, 80, 129, 0x1586, 2 * GIB, 112 * GIB)
        g = setup.amd_gpus(str(self.root))
        self.assertEqual(min(g, key=setup.amd_rank)["index"], 0)


class WindowsDetection(unittest.TestCase):
    def setUp(self):
        self.win = setup.WIN
        setup.WIN = True
        p = mock.patch.object(setup, "ram_gb", return_value=96.0)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(lambda: setattr(setup, "WIN", self.win))

    def adapters(self):
        return ([{"name": "AMD Radeon(TM) 8060S Graphics", "pnp": r"PCI\VEN_1002&DEV_1586&SUBSYS_1C3E1D05&REV_C1\4&2",
                  "ram": 4294967295}],
                [{"DriverDesc": "AMD Radeon(TM) 8060S Graphics",
                  "MatchingDeviceId": r"PCI\VEN_1002&DEV_1586&REV_C1", "HardwareInformation.qwMemorySize": 4 * GIB,
                  "DriverVersion": "32.0.21013.1000"}])

    def test_strix_halo(self):
        ad, reg = self.adapters()
        g = setup.amd_gpus_windows(ad, reg)
        self.assertEqual(len(g), 1)
        self.assertEqual((g[0]["arch"], g[0]["pci_id"]), ("gfx1151", 0x1586))
        self.assertIsNone(setup.amd_problem(g[0]))
        self.assertTrue(g[0]["uma"])
        self.assertAlmostEqual(g[0]["dedicated_gb"], 4.0)                              # the registry's carve-out
        self.assertAlmostEqual(g[0]["shared_gb"], 48.0)                                # Windows: half the RAM
        self.assertAlmostEqual(g[0]["vram_gb"], 52.0)
        self.assertEqual(g[0]["name"], "AMD Radeon(TM) 8060S Graphics")

    def test_strix_halo_and_a_7900_xtx(self):
        ad, reg = self.adapters()
        ad.append({"name": "AMD Radeon RX 7900 XTX", "pnp": r"PCI\VEN_1002&DEV_744C&SUBSYS_1&REV_C8\4&3", "ram": 0})
        reg.append({"DriverDesc": "AMD Radeon RX 7900 XTX", "MatchingDeviceId": r"PCI\VEN_1002&DEV_744C&REV_C8",
                    "HardwareInformation.qwMemorySize": 24 * GIB, "DriverVersion": "32.0.21013.1000"})
        g = setup.amd_gpus_windows(ad, reg)
        self.assertEqual([x["arch"] for x in g], ["gfx1151", "gfx1100"])
        self.assertTrue(g[0]["uma"] and not g[1].get("uma"))
        self.assertEqual(min(g, key=setup.amd_rank)["arch"], "gfx1100")

    def test_other_integrated_radeons(self):
        for did, name, arch in ((0x150E, "AMD Radeon(TM) 890M Graphics", "gfx1150"),
                                (0x1114, "AMD Radeon(TM) 860M Graphics", "gfx1152"),
                                (0x15BF, "AMD Radeon(TM) 780M Graphics", "gfx1103")):
            ad = [{"name": name, "pnp": rf"PCI\VEN_1002&DEV_{did:04X}&REV_C1\4&2", "ram": 512 << 20}]
            g = setup.amd_gpus_windows(ad, [])[0]
            self.assertEqual(g["arch"], arch)
            self.assertFalse(g.get("uma"))
            self.assertIsNotNone(setup.amd_problem(g))

    def test_hip_listing_of_strix_halo(self):
        """Once the HIP engine is installed the cards are numbered as HIP lists them; the carve-out still comes from the
        registry, and an APU's HIP figure never lowers the GPU's memory."""
        ad, reg = self.adapters()
        text = "device 0: AMD Radeon(TM) 8060S Graphics\n  arch gfx1151, 64.0 GiB\ndevice 1: AMD Radeon RX 7900 XTX\n  arch gfx1100, 24.0 GiB\n"
        hip = setup.hip_devices(text=text)
        self.assertEqual([x["arch"] for x in hip], ["gfx1151", "gfx1100"])
        with mock.patch.object(setup, "hip_devices", return_value=hip), \
                mock.patch.object(setup, "amd_gpus_windows", return_value=setup.amd_gpus_windows(ad, reg)):
            g = setup.amd_gpus_win()
        self.assertTrue(g[0]["uma"])
        self.assertAlmostEqual(g[0]["dedicated_gb"], 4.0)
        self.assertGreaterEqual(g[0]["vram_gb"], 64.0)
        self.assertFalse(g[1].get("uma"))
        self.assertAlmostEqual(g[1]["vram_gb"], 24.0)


class WindowsZip(unittest.TestCase):
    """get_prebuilt_hip never installs a zip with no code for the card; for a Strix Halo it says why and what to do."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.warnings: list[str] = []
        for p in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "warn", self.warnings.append),
                  mock.patch.object(setup, "say", lambda *a, **k: None), mock.patch.object(setup, "ok", lambda *a, **k: None),
                  mock.patch.object(setup, "download", self.fake_download),
                  mock.patch.object(setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"")),
                  mock.patch.object(setup, "hip_runtime_beside_exe", lambda eng: None)):
            p.start()
            self.addCleanup(p.stop)
        self.archs = ["gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030"]      # the 0.1.39 zip

    def fake_download(self, url, dest, label=""):
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("BUILD.json", json.dumps({"source": "prebuilt", "backend": "hip", "platform": "windows-x64",
                                                  "version": "0.1.40", "archs": self.archs, "lib_dirs": []}))
            z.writestr(setup.EXE, "x")

    def halo(self):
        return {"index": 0, "name": setup.AMD_NAMES["gfx1151"], "vram_gb": 52.0, "arch": "gfx1151", "uma": True,
                "dedicated_gb": 4.0, "shared_gb": 48.0, "vendor": "amd", "driver": "amd"}

    def test_zip_without_gfx1151_is_refused_with_the_reason(self):
        self.assertIsNone(setup.get_prebuilt_hip("https://example.invalid/", self.halo()))
        self.assertEqual(len(self.warnings), 1)
        msg = self.warnings[0]
        self.assertIn("no gfx1151 code", msg)
        self.assertIn("docs/STRIX_HALO.md", msg)
        self.assertIn("build_windows.bat", msg)
        self.assertFalse((self.root / "engine" / "strata-windows-x64-hip.zip").exists())   # dropped
        self.assertFalse((self.root / "engine" / setup.EXE).exists())                      # nothing installed

    def test_zip_with_gfx1151_is_installed(self):
        self.archs = self.archs + ["gfx1151"]
        eng = setup.get_prebuilt_hip("https://example.invalid/", self.halo())
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(self.warnings, [])
        meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
        self.assertIn("gfx1151", meta["archs"])

    def test_a_7900_xtx_still_takes_the_old_zip(self):
        g = {"index": 0, "name": "x", "vram_gb": 24.0, "arch": "gfx1100", "vendor": "amd", "driver": "amd"}
        self.assertIsNotNone(setup.get_prebuilt_hip("https://example.invalid/", g))

    def test_windows_build_script_lists_gfx1151(self):
        bat = (ROOT / "tools/hip/build_windows.bat").read_text(encoding="utf-8")
        m = re.search(r'if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=([^"]+)"', bat)
        self.assertIn("gfx1151", m.group(1).split(";"))
        self.assertTrue({"gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030"} <= set(m.group(1).split(";")))


class Recommendation(unittest.TestCase):
    def halo(self, carve=2.0):
        return {"arch": "gfx1151", "uma": True, "dedicated_gb": carve, "shared_gb": 100.0, "vram_gb": carve + 100.0}

    def test_ud_iq4_xs_with_enough_memory(self):
        self.assertEqual((setup.STRIX_HALO_FAMILY, setup.STRIX_HALO_MODEL), ("unsloth", "UD-IQ4_XS"))
        self.assertIn(setup.STRIX_HALO_MODEL, setup.MODELS)
        self.assertIn(setup.STRIX_HALO_FAMILY, setup.FAMILIES)
        self.assertTrue(setup.strix_halo_recommends(self.halo(), AURORA_RAM_GB))
        self.assertTrue(setup.strix_halo_recommends(self.halo(), 128.0))
        self.assertTrue(setup.strix_halo_recommends(self.halo(32.0), 60.0))             # 32 GB carve-out + 60 GB RAM
        self.assertFalse(setup.strix_halo_recommends(self.halo(), 62.0))                # a 64 GB box: no recommendation
        self.assertFalse(setup.strix_halo_recommends(self.halo(), 30.0))
        self.assertFalse(setup.strix_halo_recommends({"arch": "gfx1100", "vram_gb": 24.0}, 256.0))
        self.assertFalse(setup.strix_halo_recommends({"arch": "gfx1150", "vram_gb": 0.5}, 256.0))

    def test_the_recommended_model_is_a_regular_choice_on_amd(self):
        m = setup.MODELS[setup.STRIX_HALO_MODEL]
        self.assertFalse(m.get("nvidia_only"))
        self.assertFalse(m.get("experimental"))

    def test_notes_are_recommendations(self):
        win = setup.WIN
        try:
            setup.WIN = False
            g = {**self.halo(), "shared_gb": 60.0, "vram_gb": 62.0}
            notes = setup.strix_halo_notes(g, AURORA_RAM_GB)
            joined = "\n".join(notes)
            self.assertIn("docs/STRIX_HALO.md", joined)
            self.assertIn("UD-IQ4_XS", joined)
            self.assertTrue(any(n.startswith("!") and "ttm.pages_limit" in n for n in notes))   # a small GTT: a warning
            self.assertIn("changes no host setting", joined)
            big = {**self.halo(), "shared_gb": 112.0, "vram_gb": 114.0}
            self.assertFalse(any(n.startswith("!") for n in setup.strix_halo_notes(big, AURORA_RAM_GB)))
        finally:
            setup.WIN = win

    def test_hipblaslt_table_for_gfx1151(self):
        t = setup.hipblaslt_table("gfx1151", [], 100401)
        self.assertIsNotNone(t)
        self.assertEqual(t.name, "gfx1151-hipblaslt-100401.txt")


class LowRamUsesTheCarveOutOnly(unittest.TestCase):
    def test_apu_memory_is_not_double_counted(self):
        halo = {"arch": "gfx1151", "uma": True, "dedicated_gb": 2.0, "shared_gb": 112.0, "vram_gb": 114.0, "index": 0}
        card = {"arch": "gfx1100", "vram_gb": 24.0, "index": 0}
        self.assertEqual(setup.low_ram_vram(halo), 2.0)
        self.assertEqual(setup.low_ram_vram(card), 24.0)
        # a 32 GB-RAM Strix Halo: the smallest model is not "made to fit" by counting the GPU's shared memory as extra
        self.assertFalse(setup.low_ram_fits("IQ1_M", 24.0, setup.low_ram_vram(halo)))
        self.assertTrue(setup.low_ram_fits("IQ1_M", 24.0, 40.0))                            # a 40 GB dedicated card does


if __name__ == "__main__":
    unittest.main()
