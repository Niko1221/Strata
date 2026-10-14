"""Windows ADL telemetry contracts, without requiring a GPU or Windows."""
import ctypes as c
import unittest
from unittest import mock

from serve import amd_windows as aw, telemetry


class Function:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class Driver:
    def __init__(self):
        self.destroyed = 0
        self.sensors = {8: 51, 19: 0, 23: 85, 40: 4, 41: 16, 73: 110}
        self.fail = set()
        self.used = 4096
        # ADL enumerates the discrete card first; HIP enumerates the integrated card first.
        self.adapters = [(7, 3, b"Discrete Radeon"), (2, 8, b"Integrated Radeon"),
                         (9, 3, b"Duplicate display")]

    def __getattr__(self, name):
        if name in self.fail:
            raise AttributeError(name)
        fn = getattr(self, name.removeprefix("ADL2_"), None)
        if fn is None:
            raise AttributeError(name)
        return Function(fn)

    def Main_Control_Create(self, allocate, connected, ctx):
        self.assert_allocation = allocate(32)
        ctx._obj.value = 123
        return 0

    def Main_Control_Destroy(self, ctx):
        self.destroyed += 1
        return 0

    def Adapter_NumberOfAdapters_Get(self, ctx, count):
        count._obj.value = len(self.adapters)
        return 0

    def Adapter_AdapterInfo_Get(self, ctx, adapters, size):
        for a, (index, bus, name) in zip(adapters, self.adapters):
            a.index, a.bus, a.vendor, a.present, a.exist, a.name = index, bus, 1002, 1, 1, name
        return 0

    def Adapter_MemoryInfo2_Get(self, ctx, dev, mem):
        mem._obj.size = 16 << 30
        return 0

    def Adapter_DedicatedVRAMUsage_Get(self, ctx, dev, used):
        used._obj.value = self.used
        return 0

    def New_QueryPMLogData_Get(self, ctx, dev, metrics):
        for i, v in self.sensors.items():
            metrics._obj.sensors[i].supported = 1
            metrics._obj.sensors[i].value = v
        return 0

    def Adapter_ChipSetInfo_Get(self, ctx, dev, chip):
        chip._obj.speed_type, chip._obj.width = 6, 16
        return 0


class WindowsTelemetry(unittest.TestCase):
    def setUp(self):
        self.driver = Driver()
        self.loader = mock.patch.object(aw.c, "CDLL", return_value=self.driver)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        pci = mock.patch.object(aw, "_hip_pci", side_effect=lambda index: {0: (8, 0, 0), 1: (3, 0, 0)}.get(index))
        pci.start()
        self.addCleanup(pci.stop)

    def reader(self, index=1):
        g = aw.AmdWindows(index)
        self.addCleanup(g.close)
        return g

    def test_hip_order_and_duplicate_adapters(self):
        self.assertEqual(self.reader(0).name(), "Integrated Radeon")
        self.assertEqual(self.reader(1).dev, 7)
        self.assertFalse(self.reader(2).ok())

    def test_units_idle_zero_and_board_power(self):
        r = self.reader().read()
        self.assertEqual(r, {"mem_total": 16 << 30, "mem_used": 4 << 30, "util": 0, "temp": 51,
                             "power": 110, "pcie_gen": 4, "pcie_width": 16, "pcie_gen_max": 5})
        del self.driver.sensors[73]
        self.assertEqual(self.reader().read()["power"], 85)

    def test_unsupported_sensors_do_not_become_zero(self):
        self.driver.sensors = {}
        r = self.reader().read()
        self.assertIsNone(r["util"])
        self.assertIsNone(r["temp"])
        self.assertIsNone(r["power"])
        self.assertNotIn("pcie_rx_mb", r)
        self.assertEqual(r["pcie_width"], 16)

    def test_missing_optional_api_preserves_other_readings(self):
        self.driver.fail.add("ADL2_New_QueryPMLogData_Get")
        r = self.reader().read()
        self.assertEqual(r["mem_used"], 4 << 30)
        self.assertEqual(r["pcie_gen_max"], 5)
        self.assertNotIn("temp", r)

    def test_bad_memory_value_is_absent(self):
        self.driver.used = 200000
        self.assertNotIn("mem_used", self.reader().read())

    def test_no_hip_only_single_card_is_unambiguous(self):
        with mock.patch.object(aw, "_hip_pci", return_value=None):
            self.assertFalse(self.reader(0).ok())
            self.driver.adapters = self.driver.adapters[:1] + self.driver.adapters[2:]
            self.assertTrue(self.reader(0).ok())
            self.assertFalse(self.reader(1).ok())

    def test_driver_absent_and_initialization_failure(self):
        with mock.patch.object(aw.c, "CDLL", side_effect=OSError("no driver")):
            g = self.reader()
            self.assertFalse(g.ok())
            self.assertEqual(g.read(), {})
        self.driver.fail.add("ADL2_Adapter_AdapterInfo_Get")
        self.assertFalse(self.reader().ok())
        self.assertEqual(self.driver.destroyed, 1)

    def test_close_and_free_vram_release_context(self):
        g = self.reader()
        g.close()
        g.close()
        self.assertEqual(self.driver.destroyed, 1)
        with mock.patch.object(telemetry, "gpu_reader", side_effect=lambda *args: self.reader()):
            self.assertEqual(telemetry.free_vram_mib(1, amd=True), 12288)
        self.assertEqual(self.driver.destroyed, 2)

    def test_sampler_and_routing(self):
        with mock.patch.object(telemetry.os, "name", "nt"):
            self.assertIsInstance(telemetry.gpu_reader(1, amd=True), aw.AmdWindows)
        with mock.patch.object(telemetry, "gpu_reader", side_effect=lambda index, amd: self.reader(index)), \
                mock.patch.object(telemetry.threading.Thread, "start"):
            t = telemetry.Telemetry(gpu_index=1, amd=True)
            s = t.sample()
            self.assertEqual(t.static["gpu_name"], "Discrete Radeon")
            self.assertEqual((s["gpu_util"], s["gpu_mem_used"], s["gpu_temp"], s["gpu_power"]),
                             (0, 4 << 30, 51, 110))


if __name__ == "__main__":
    unittest.main()
