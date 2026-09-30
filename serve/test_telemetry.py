"""serve/test_telemetry.py - the Monitor's hardware readings, on whatever machine this runs (no GPU, no pack).

    python -m unittest serve.test_telemetry -v
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import telemetry  # noqa: E402


class AmdBackend(unittest.TestCase):
    """The GPU fields from amdgpu's sysfs, for a machine where NVML does not exist.

    None of this needs a card: with no amdgpu device the backend must report nothing at all rather than raise."""

    def test_pcie_generation_from_a_link_speed_string(self):
        for speed, want in (("2.5 GT/s PCIe", 1), ("5.0 GT/s PCIe", 2), ("8.0 GT/s PCIe", 3),
                            ("16.0 GT/s PCIe", 4), ("32.0 GT/s PCIe", 5)):
            with self.subTest(speed=speed):
                self.assertEqual(telemetry._pcie_gen(speed), want)
        self.assertIsNone(telemetry._pcie_gen("unknown"))
        self.assertIsNone(telemetry._pcie_gen(None))

    def test_only_whole_disks_are_counted(self):
        for name in ("sda", "sdb", "nvme0n1", "vda", "mmcblk0", "hda"):
            with self.subTest(name=name):
                self.assertIsNotNone(telemetry._WHOLE_DISK.match(name))
        for name in ("sda1", "nvme0n1p3", "dm-0", "zram0", "loop0", "sr0"):
            with self.subTest(name=name):
                self.assertIsNone(telemetry._WHOLE_DISK.match(name), "a partition or a virtual device")

    def test_the_backend_reports_nothing_without_a_card(self):
        gpu = telemetry._AmdSysfs(99)                 # an index no scan can reach
        self.assertFalse(gpu.ok())
        self.assertEqual(gpu.read(), {})              # nothing invented
        self.assertEqual(gpu.name(), "AMD GPU")

    def test_a_card_present_reports_sane_values(self):
        gpu = telemetry._AmdSysfs(0)
        if not gpu.ok():                              # a machine with no amdgpu card: the empty case is the contract
            self.assertEqual(gpu.read(), {})
            return
        read = gpu.read()
        self.assertIsInstance(gpu.name(), str)
        self.assertTrue(gpu.name())
        self.assertGreater(read["mem_total"], 0)
        self.assertLessEqual(read["mem_used"], read["mem_total"])
        self.assertTrue(0 <= read["util"] <= 100, read["util"])
        if "temp" in read:                            # the sensor is optional; when present it is a temperature
            self.assertTrue(0 < read["temp"] < 120, read["temp"])
        if "pcie_gen" in read:
            self.assertIn(read["pcie_gen"], (1, 2, 3, 4, 5, 6))


class SampleSeries(unittest.TestCase):
    """`Telemetry.sample()` still reports every series the Monitor plots, with or without psutil."""

    def test_a_sample_carries_the_monitor_series(self):
        tel = telemetry.Telemetry(extra=lambda: {"tok_s": 1.0})
        time.sleep(1.1)                               # a disk rate needs two readings
        snap = tel.snapshot()
        now, static = snap["now"], snap["static"]
        self.assertEqual(now.get("tok_s"), 1.0, "the server's own extra() must still be merged in")
        for key in ("cpu", "ram_used", "disk_read_mb", "gpu_mem_used", "gpu_temp", "gpu_util"):
            with self.subTest(key=key):
                self.assertIn(key, snap["history"], f"{key} must be a recorded series")
        for key in ("cpu", "ram_used", "ram_total", "disk_read_mb"):
            with self.subTest(key=key):
                self.assertIn(key, now)
        self.assertIsInstance(static["cores"], (int, type(None)))
        self.assertIn("gpu_name", static)


class CardSelection(unittest.TestCase):
    """The two rules of a mixed or Ryzen box: the card at the model's HIP device, and AMD numbers where the
    model runs on AMD even if NVML loads.  A fake sysfs and two fake backends stand in for the hardware."""

    def _fake_drm(self, root, cards):
        """cards: [(cardN, pci_slot)] - each with DRIVER=amdgpu in its uevent."""
        for name, slot in cards:
            dev = root / name / "device"
            dev.mkdir(parents=True)
            (dev / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\nPCI_ID=1002:744C\n")

    def test_the_card_is_the_one_at_the_model_s_hip_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_drm(root, [("card0", "0000:00:08.1"),      # a Ryzen iGPU is amdgpu too
                                  ("card1", "0000:03:00.0")])     # the gfx1100 the model runs on
            gpu = telemetry._AmdSysfs(0, {0: "0000:03:00.0"}, root=root)
            self.assertTrue(gpu.ok())
            self.assertEqual(telemetry._card_slot(gpu.dir), "0000:03:00.0",
                             "the iGPU at index 0 of the card scan must not be picked")
            stale = telemetry._AmdSysfs(0, {}, root=root)          # documented fallback: card order
            self.assertEqual(telemetry._card_slot(stale.dir), "0000:00:08.1")

    def test_amd_readings_win_when_the_model_runs_on_amd_even_if_nvml_loads(self):
        class FakeNvml:
            def __init__(self, index=0):
                self.index = index

            def ok(self):
                return True                          # NVML loads on this box

            def name(self):
                return "NVIDIA card"

        class FakeAmd:
            def __init__(self, index=0, hip_map=None, root=None):
                self.index = index

            def ok(self):
                return True

            def name(self):
                return "AMD card"

        saved = telemetry._Nvml, telemetry._AmdSysfs, telemetry._hip_pci_map
        telemetry._Nvml, telemetry._AmdSysfs = FakeNvml, FakeAmd
        telemetry._hip_pci_map = lambda: {0: "0000:03:00.0"}

        def restore():
            telemetry._Nvml, telemetry._AmdSysfs = saved[0], saved[1]
            telemetry._hip_pci_map = saved[2]

        self.addCleanup(restore)
        self.assertIsInstance(telemetry.Telemetry(hip=True, gpu_index=0).gpu, FakeAmd,
                              "the model on the AMD card: its numbers, not NVML's")
        self.assertIsInstance(telemetry.Telemetry(hip=False, gpu_index=0).gpu, FakeNvml,
                              "the model on an NVIDIA card: NVML as before")

    def test_where_the_model_runs_decides_the_backend(self):
        self.assertTrue(telemetry.engine_runs_on_amd({"backend": "hip"}))
        self.assertFalse(telemetry.engine_runs_on_amd({"backend": "cuda"}))
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "strata"
            exe.write_bytes(b"\x7fELF" + b"\0" * 32 + b"libamdhip64.so.1\0")
            self.assertTrue(telemetry.engine_runs_on_amd({"exe": str(exe)}),
                            "no backend key: the binary's own linkage decides")
            exe.write_bytes(b"\x7fELF" + b"\0" * 32 + b"libcudart.so.12\0")
            self.assertFalse(telemetry.engine_runs_on_amd({"exe": str(exe)}))
        self.assertFalse(telemetry.engine_runs_on_amd({}))


if __name__ == "__main__":
    unittest.main()
