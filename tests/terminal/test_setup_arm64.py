"""setup.py on an ARM64 machine (DGX Spark / GB10, #3622): the GPU line, the CPU probe and the older-CPU floor.
x86 machines must keep their old answers: a card whose memory nvidia-smi does not report stays unlisted there."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402

GB10 = "0, NVIDIA GB10, [N/A], 12.1, 580.95.05\n"
MEMINFO = "MemTotal:       130000000 kB\nMemFree:         1000000 kB\n"
ARM_CPUINFO = ("processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics\n"
               "CPU implementer\t: 0x41\n")


class UnifiedMemoryCard(unittest.TestCase):
    def test_a_card_without_a_memory_figure_is_unlisted_on_x86_as_before(self):
        with mock.patch.object(setup, "out", return_value=GB10), mock.patch.object(setup, "is_arm64", return_value=False):
            self.assertEqual(setup.gpus(), [])

    def test_gb10_takes_the_pool_size_from_the_machine_ram(self):
        with mock.patch.object(setup, "out", return_value=GB10), mock.patch.object(setup, "is_arm64", return_value=True), \
                mock.patch("builtins.open", mock.mock_open(read_data=MEMINFO)):
            found = setup.gpus()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["name"], "NVIDIA GB10")
        self.assertEqual(found[0]["arch"], "121")
        self.assertAlmostEqual(found[0]["vram_gb"], 130000000 / 1024.0 / 1024.0, places=3)

    def test_a_normal_card_is_unchanged(self):
        with mock.patch.object(setup, "out", return_value="0, NVIDIA GeForce RTX 5070, 12227, 12.0, 580.1\n"):
            found = setup.gpus()
        self.assertAlmostEqual(found[0]["vram_gb"], 12227 / 1024.0, places=6)


class ArmCpu(unittest.TestCase):
    def test_an_arm_cpuinfo_without_the_x86_flags_line_does_not_crash(self):
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", mock.mock_open(read_data=ARM_CPUINFO)):
            name, avx2, avx512 = setup.cpu_info()
        self.assertFalse(avx2)
        self.assertFalse(avx512)

    def test_an_x86_cpuinfo_is_read_as_before(self):
        text = "flags\t\t: fpu avx avx2 avx512f avx512bw avx512vl avx512_vnni avx512vbmi\nmodel name\t: Test CPU\n"
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", mock.mock_open(read_data=text)):
            name, avx2, avx512 = setup.cpu_info()
        self.assertEqual((name, avx2, avx512), ("Test CPU", True, True))

    def test_no_older_cpu_floor_on_arm(self):
        with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": ""}), mock.patch.object(setup, "is_arm64", return_value=True):
            self.assertEqual(setup.cpu_floor(False), "")

    def test_the_floor_on_x86_without_avx2_is_still_asked_for(self):
        with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": ""}), mock.patch.object(setup, "is_arm64", return_value=False), \
                mock.patch.object(setup, "WIN", False), \
                mock.patch("builtins.open", mock.mock_open(read_data="flags\t: fpu sse4_2 popcnt avx\n")):
            self.assertEqual(setup.cpu_floor(False), "avx")


if __name__ == "__main__":
    unittest.main()
