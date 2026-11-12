"""serve/telemetry.py: the Intel Arc (xe) sysfs reader (#5353).

A fake sysfs tree: a hybrid host with an iGPU (vendor 0x8086, no tile0) before the Arc Pro B70 (tile0, dev 0xe223).
Checks: the reader takes the B70 and not the iGPU (#2001), the temperature comes from the hwmon sensor labeled
"pkg", power from the energy counter's delta, the cap from power1_cap, the PCIe link from the first upstream port
wider than x1 (the card's own functions sit behind an internal x1 switch), and a fresh /run/gpustat.json overrides
with load + VRAM while a stale one is ignored. gpu_reader's wiring (NVML wins, then xe, then the NVML stub as
before) is checked against patched reader classes: NVML answers on every machine with the library installed,
whatever else sysfs says.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serve import telemetry as T  # noqa: E402

NEEDS_LINUX = not sys.platform.startswith("linux")


def fake_sysfs(root: Path):
    """root/class/drm/card0 = an iGPU (no tile0); card1 = a B70 on xe behind an internal x1 switch, its parent port
    Gen3 x16 (the slot), the card itself Gen5-capable; hwmon with the xe sensors."""
    igpu = root / "class/drm/card0/device"
    igpu.mkdir(parents=True)
    (igpu / "vendor").write_text("0x8086\n")
    (igpu / "device").write_text("0x3e92\n")

    dev = root / "class/drm/card1/device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x8086\n")
    (dev / "device").write_text("0xe223\n")
    (dev / "tile0").mkdir()
    port = dev.parent                                   # the upstream PCIe port
    (port / "current_link_speed").write_text("8.0 GT/s PCIe\n")
    (port / "current_link_width").write_text("16\n")
    (port / "max_link_speed").write_text("32.0 GT/s PCIe\n")
    (port / "max_link_width").write_text("16\n")
    (dev / "current_link_speed").write_text("8.0 GT/s PCIe\n")
    (dev / "current_link_width").write_text("1\n")      # behind the card's internal switch: not the real link

    hw = dev / "hwmon/hwmon5"
    hw.mkdir(parents=True)
    (hw / "name").write_text("xe\n")
    (hw / "temp1_label").write_text("vram_ch_0\n")      # the pkg sensor is NOT the first one
    (hw / "temp1_input").write_text("50000\n")
    (hw / "temp2_label").write_text("pkg\n")
    (hw / "temp2_input").write_text("52000\n")
    (hw / "energy1_input").write_text("1000000000\n")   # uJ
    (hw / "power1_cap").write_text("230000000\n")       # uW
    return root


class XeReader(unittest.TestCase):
    @unittest.skipIf(NEEDS_LINUX, "the xe reader reads a Linux sysfs layout")
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.sysfs = fake_sysfs(Path(self._t.name))
        self._statfile = Path(self._t.name) / "gpustat.json"
        self.patch = mock.patch.object(T._Xe, "STAT", str(self._statfile))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self._t.cleanup)

    def _gpu(self):
        g = T._Xe(0, sysfs=str(self.sysfs))
        self.assertTrue(g.ok())
        return g

    def test_the_card_is_the_xe_gpu_not_the_igpu(self):
        g = self._gpu()
        assert g.dev is not None
        self.assertEqual(Path(g.dev).parent.name, "card1")
        self.assertEqual(g.name(), "Intel Arc Pro B70")

    def test_no_sampler_temperature_cap_and_link_from_sysfs(self):
        r = self._gpu().read()
        self.assertEqual(r["temp"], 52.0)               # the sensor labeled "pkg", not the first sensor
        self.assertEqual(r["power_limit"], 230.0)
        self.assertEqual((r["pcie_gen"], r["pcie_gen_max"], r["pcie_width"]), (3, 5, 16))
        self.assertIsNone(r.get("util"))                # load needs the sampler
        self.assertNotIn("mem_used", r)                 # ditto VRAM
        self.assertIsNone(r["pcie_rx_mb"])              # xe has no PCIe traffic counters

    def test_power_is_the_energy_counter_delta(self):
        g = self._gpu()
        self.assertIsNone(g.read().get("power"))        # a single sample has no delta yet
        g._e = (time.time() - 10.0, 1_000_000_000)      # the sampler's first tick: 1000 J at t-10
        Path(self.sysfs / "class/drm/card1/device/hwmon/hwmon5/energy1_input").write_text("2000000000\n")
        self.assertAlmostEqual(g.read()["power"], 100.0, delta=2.0)   # 1000 J over ~10 s

    def test_a_fresh_sampler_reading_wins_over_the_slow_sensors(self):
        now = time.time()
        self._statfile.write_text(json.dumps(
            {"name": "Intel(R) Arc(TM) Pro B70 Graphics", "ts": now, "busy_pct": 97.5,
             "vram_used_mb": 31496, "vram_total_mb": 32768, "temp_pkg": 62.0, "power_w": 87.8,
             "power_cap_w": 230, "temp_vram_max": 62.0}))
        g = self._gpu()
        r = g.read()
        self.assertEqual(g.name(), "Intel(R) Arc(TM) Pro B70 Graphics")
        self.assertEqual(r["util"], 97.5)
        self.assertEqual(r["mem_used"], 31496 * 2**20)
        self.assertEqual(r["mem_total"], 32768 * 2**20)
        self.assertEqual(r["temp"], 62.0)               # the sampler's pkg beats hwmon
        self.assertEqual(r["power"], 87.8)

    def test_a_stale_sampler_reading_is_ignored(self):    # the sampler died: sysfs must still answer
        self._statfile.write_text(json.dumps({"name": "x", "ts": time.time() - 60, "busy_pct": 50,
                                              "vram_used_mb": 1, "vram_total_mb": 2}))
        r = self._gpu().read()
        self.assertEqual(r["temp"], 52.0)
        self.assertIsNone(r.get("util"))

    def test_a_host_without_an_arc_has_no_xe_reader(self):
        igpu_only = Path(self._t.name) / "only"
        (igpu_only / "class/drm/card0/device").mkdir(parents=True)
        (igpu_only / "class/drm/card0/device/vendor").write_text("0x8086\n")
        self.assertFalse(T._Xe(0, sysfs=str(igpu_only)).ok())
        self.assertFalse(T._Xe(0, sysfs=str(self.sysfs / "nowhere")).ok())


class GpuReaderWiring(unittest.TestCase):
    """gpu_reader's choice order, with the reader classes patched: on a real machine with NVML installed, NVML must
    keep answering even when an Intel card sits in sysfs (#2001); the xe reader only answers when NVML has no card."""

    def test_nvml_wins_when_it_finds_a_card(self):
        with mock.patch.object(T, "_Nvml") as Nv, mock.patch.object(T, "_Xe") as Xe:
            Nv.return_value.ok.return_value = True
            Xe.return_value.ok.return_value = True
            self.assertIs(T.gpu_reader(0), Nv.return_value)

    def test_xe_answers_when_nvml_finds_nothing(self):
        with mock.patch.object(T, "_Nvml") as Nv, mock.patch.object(T, "_Xe") as Xe:
            Nv.return_value.ok.return_value = False
            Xe.return_value.ok.return_value = True
            self.assertIs(T.gpu_reader(0), Xe.return_value)

    def test_the_amd_backend_still_comes_first(self):
        with mock.patch.object(T, "_Amd") as A, mock.patch.object(T, "_Nvml") as Nv, mock.patch.object(T, "_Xe") as Xe:
            A.return_value.ok.return_value = True
            self.assertIs(T.gpu_reader(0, amd=True), A.return_value)

    def test_no_card_anywhere_keeps_the_nvml_stub(self):
        with mock.patch.object(T, "_Nvml") as Nv, mock.patch.object(T, "_Xe") as Xe:
            Nv.return_value.ok.return_value = False
            Xe.return_value.ok.return_value = False
            self.assertIs(T.gpu_reader(0), Nv.return_value)


if __name__ == "__main__":
    unittest.main()
