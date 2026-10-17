"""#1380: the AMD card's dashboard readings on Windows (serve/telemetry.py _AmdWin).  The Windows APIs themselves
(DXGI, PDH, ADL, SetupAPI) cannot run here: these tests cover the parsing, the choices and the fallbacks, with fakes."""
import ctypes
import unittest
from unittest import mock

from serve import telemetry as T

LUID = (0, 0xE3F7)
CARD = {"name": "AMD Radeon RX 9070 XT", "device": 0x7550, "vram": 16 << 30, "luid": LUID}


def inst(pid, low, tail):
    return f"pid_{pid}_luid_0x00000000_0x{low:08X}_{tail}"


class Parsing(unittest.TestCase):
    def test_luid_of(self):
        self.assertEqual(T.luid_of(inst(4, 0xE3F7, "phys_0_eng_0_engtype_3D")), LUID)
        self.assertEqual(T.luid_of("luid_0x00000001_0x0000ABCD_phys_0"), (1, 0xABCD))
        self.assertIsNone(T.luid_of("pid_4_engtype_3D"))
        self.assertIsNone(T.luid_of(None))

    def test_load_is_the_busiest_engine_summed_over_processes(self):
        items = [(inst(1, 0xE3F7, "phys_0_eng_0_engtype_3D"), 10.0),
                 (inst(2, 0xE3F7, "phys_0_eng_0_engtype_3D"), 15.0),          # same engine, another process: 25
                 (inst(2, 0xE3F7, "phys_0_eng_1_engtype_Compute_0"), 60.0),   # the HIP engine's compute queue
                 (inst(3, 0x1111, "phys_0_eng_0_engtype_3D"), 99.0)]          # another adapter: left out
        self.assertEqual(T.pdh_load(items, LUID), 60.0)
        self.assertEqual(T.pdh_load([(inst(1, 0xE3F7, "phys_0_eng_0_engtype_3D"), 80.0)] * 2, LUID), 100.0)   # capped
        self.assertIsNone(T.pdh_load([], LUID))
        self.assertIsNone(T.pdh_load(items[-1:], LUID))

    def test_vram_used(self):
        items = [("luid_0x00000000_0x0000E3F7_phys_0", 5e9), ("luid_0x00000000_0x00001111_phys_0", 9e9)]
        self.assertEqual(T.pdh_vram_used(items, LUID), 5e9)
        self.assertIsNone(T.pdh_vram_used(items[1:], LUID))


class Sensors(unittest.TestCase):
    def test_pick(self):
        s = T.ADL_SENSORS
        r = T.adl_pick({s["temp_edge"]: 54, s["temp_hotspot"]: 70, s["asic_power"]: 180, s["activity_gfx"]: 97})
        self.assertEqual(r, {"temp": 54.0, "power": 180.0, "util": 97.0})

    def test_fallbacks_and_nonsense(self):
        s = T.ADL_SENSORS
        self.assertEqual(T.adl_pick({s["temp_hotspot"]: 71, s["board_power"]: 200})["temp"], 71.0)
        self.assertEqual(T.adl_pick({s["board_power"]: 200})["power"], 200.0)
        self.assertEqual(T.adl_pick({}), {"temp": None, "power": None, "util": None})
        r = T.adl_pick({s["temp_edge"]: 0, s["activity_gfx"]: 400})
        self.assertIsNone(r["temp"])
        self.assertIsNone(r["util"])

    def test_sensor_ids_match_adl_defines(self):                  # adl_defines.h, ADL_PMLOG_SENSORS
        self.assertEqual(T.ADL_SENSORS, {"activity_gfx": 19, "temp_edge": 8, "temp_gfx": 28, "temp_hotspot": 27,
                                         "asic_power": 23, "board_power": 73, "gfx_power": 30})


class AdlVendor(unittest.TestCase):
    def test_both_spellings_of_amds_vendor_id(self):
        self.assertIn(1002, T.ADL_AMD_VENDORS)             # what ADL reports (decimal, as in AMD's samples)
        self.assertIn(0x1002, T.ADL_AMD_VENDORS)           # the PCI id, in case a driver reports it that way
        self.assertNotIn(0x10DE, T.ADL_AMD_VENDORS)


class Layouts(unittest.TestCase):
    def test_sizes_that_do_not_depend_on_the_os(self):
        self.assertEqual(ctypes.sizeof(T._AdlInfo), 1572)         # AdapterInfo on Windows
        self.assertEqual(ctypes.sizeof(T._AdlPmLog), 4 + 256 * 8)
        self.assertEqual(ctypes.sizeof(T._Guid), 16)

    def test_guid(self):
        g = T._guid("770aae78-f26f-4dba-a829-253c83d1b387")        # IID_IDXGIFactory1
        self.assertEqual((g.d1, g.d2, g.d3), (0x770AAE78, 0xF26F, 0x4DBA))
        self.assertEqual(bytes(g.d4), bytes.fromhex("a829253c83d1b387"))


class Choosing(unittest.TestCase):
    def test_integrated_first_like_hip(self):
        igpu = {"name": "AMD Radeon(TM) Graphics", "vram": 512 << 20}
        self.assertEqual(T.amd_order([CARD, igpu]), [igpu, CARD])
        self.assertEqual(T.amd_order([CARD]), [CARD])

    def test_no_dxgi_off_windows(self):
        with mock.patch.object(T, "IS_WIN", False):
            self.assertEqual(T.dxgi_amd_adapters(), [])

    def test_reader_follows_the_os(self):
        with mock.patch.object(T, "IS_WIN", True), mock.patch.object(T, "dxgi_amd_adapters", lambda: [CARD]), \
                mock.patch.object(T, "_Pdh", side_effect=OSError), mock.patch.object(T, "_Adl", side_effect=OSError), \
                mock.patch.object(T, "_PciLink", side_effect=OSError):
            g = T.gpu_reader(0, amd=True)
            self.assertIsInstance(g, T._AmdWin)
            self.assertTrue(g.ok())
            self.assertEqual(g.name(), "AMD Radeon RX 9070 XT")
            r = g.read()                                           # every optional source failed: only what DXGI knows
            self.assertEqual(r["mem_total"], 16 << 30)
            self.assertIsNone(r["util"])
            self.assertIsNone(r["mem_used"])
        with mock.patch.object(T, "IS_WIN", False):
            self.assertIsInstance(T.gpu_reader(0, amd=True), T._Amd)
        self.assertIsInstance(T.gpu_reader(0, amd=False), T._Nvml)

    def test_a_source_that_does_not_start_says_why(self):
        with mock.patch.object(T, "IS_WIN", True), mock.patch.object(T, "dxgi_amd_adapters", lambda: [CARD]), \
                mock.patch.object(T, "_Pdh", side_effect=OSError), mock.patch.object(T, "_Adl", side_effect=OSError("adl broke")), \
                mock.patch.object(T, "_PciLink", side_effect=OSError):
            g = T.gpu_reader(0, amd=True)
            self.assertEqual(g.errors["adl"], "OSError: adl broke")
            self.assertEqual(set(g.errors), {"pdh", "adl", "pci"})

    def test_no_card(self):
        with mock.patch.object(T, "IS_WIN", True), mock.patch.object(T, "dxgi_amd_adapters", lambda: []):
            g = T.gpu_reader(0, amd=True)
            self.assertFalse(g.ok())
            self.assertEqual(g.name(), "AMD Radeon")


class Reading(unittest.TestCase):
    def make(self, pdh=None, adl=None, pci=None):
        with mock.patch.object(T, "IS_WIN", True), mock.patch.object(T, "dxgi_amd_adapters", lambda: [CARD]), \
                mock.patch.object(T, "_Pdh", side_effect=(lambda paths: pdh) if pdh else OSError), \
                mock.patch.object(T, "_Adl", side_effect=(lambda dev: adl) if adl else OSError), \
                mock.patch.object(T, "_PciLink", side_effect=(lambda dev: pci) if pci else OSError):
            return T.gpu_reader(0, amd=True)

    @staticmethod
    def pdh(load, mem):
        p = mock.Mock()
        p.read.return_value = [[(inst(1, 0xE3F7, "phys_0_eng_0_engtype_3D"), load)] if load is not None else [],
                               [("luid_0x00000000_0x0000E3F7_phys_0", mem)] if mem is not None else []]
        return p

    def test_all_sources(self):
        adl = mock.Mock()
        adl.read.return_value = {"temp": 61.0, "power": 190.0, "util": 88.0}
        pci = mock.Mock()
        pci.read.return_value = {"pcie_gen": 4, "pcie_gen_max": 5, "pcie_width": 16, "pcie_width_max": 16}
        r = self.make(self.pdh(40.0, 9e9), adl, pci).read()
        self.assertEqual((r["temp"], r["power"], r["util"]), (61.0, 190.0, 88.0))        # ADL's busy figure wins
        self.assertEqual((r["mem_used"], r["mem_total"]), (9e9, 16 << 30))
        self.assertEqual((r["pcie_gen"], r["pcie_gen_max"], r["pcie_width"]), (4, 5, 16))

    def test_counters_alone_give_load_and_vram(self):
        r = self.make(self.pdh(40.0, 9e9)).read()
        self.assertEqual((r["util"], r["mem_used"]), (40.0, 9e9))
        self.assertNotIn("temp", r)

    def test_a_failing_source_does_not_hide_the_others(self):
        adl = mock.Mock()
        adl.read.side_effect = OSError
        pci = mock.Mock()
        pci.read.return_value = {"pcie_gen": 5, "pcie_gen_max": 5, "pcie_width": 16}
        r = self.make(self.pdh(10.0, 1e9), adl, pci).read()
        self.assertEqual(r["util"], 10.0)
        self.assertEqual(r["pcie_width"], 16)
        self.assertNotIn("temp", r)

    def test_a_skipped_counter_second_keeps_the_last_reading(self):
        p = self.pdh(40.0, 9e9)
        g = self.make(p)
        self.assertEqual(g.read()["util"], 40.0)
        p.read.return_value = [[], []]                             # the query was just reopened: no rates yet
        for _ in range(3):
            self.assertEqual(g.read()["util"], 40.0)
        self.assertIsNone(g.read()["util"])                        # a fourth empty read in a row: gone

    def test_telemetry_wires_it(self):
        with mock.patch.object(T, "IS_WIN", True), mock.patch.object(T, "dxgi_amd_adapters", lambda: [CARD]), \
                mock.patch.object(T, "_Pdh", side_effect=OSError), mock.patch.object(T, "_Adl", side_effect=OSError), \
                mock.patch.object(T, "_PciLink", side_effect=OSError):
            t = T.Telemetry(amd=True)
            try:
                self.assertEqual(t.static["gpu_name"], "AMD Radeon RX 9070 XT")
                self.assertIsNone(t.static["gpu_note"])
                self.assertEqual(t.sample()["gpu_mem_total"], 16 << 30)
            finally:
                t.close()
        with mock.patch.object(T, "IS_WIN", True), mock.patch.object(T, "dxgi_amd_adapters", lambda: []):
            t = T.Telemetry(amd=True)
            try:
                self.assertIn("no readings for this AMD card", t.static["gpu_note"])
            finally:
                t.close()


if __name__ == "__main__":
    unittest.main()
