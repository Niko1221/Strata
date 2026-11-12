"""setup.py and the updater read one AMD table (strata_control/amd_gpus.py).  These tests pin the answer for a
Radeon PRO W6800 - by PCI device id and by name - and keep the two readers answering the same card the same way."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from strata_control import amd_gpus, engine_component as ec, feed as feedmod

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import setup as strata_setup      # noqa: E402

WIN = "windows-x86_64"


def pkg(version, variant, archs):
    return feedmod._engine_package({
        "version": version, "platform": WIN, "backend": "hip", "variant": variant, "archs": list(archs), "min_driver": {},
        "min_runtime": None, "build_id": f"b-{version}-{variant}", "format": "zip", "url": f"e-{version}-{variant}.zip", "size": 10,
        "sha256": "a" * 64, "manifest_sha256": "b" * 64, "installed_bytes": 20, "notes": ""})


FEED = [pkg("0.1.58", "rx9070", ("gfx1200", "gfx1201")),
        pkg("0.1.58", "rdna3", ("gfx1100", "gfx1101", "gfx1102", "gfx1151")),
        pkg("0.1.58", "rdna2", ("gfx1030", "gfx1031"))]


def hw(names=()):
    return {"platform": WIN, "backend": "hip", "runtime": "0.2.12", "nvidia_driver": None, "cuda_archs": [], "gfx": [],
            "gpu_names": list(names), "installed_variant": "", "variant_wanted": ""}


class W6800(unittest.TestCase):
    """A PC with a "AMD Radeon(TM) Graphics" iGPU (PCI 13C0) beside a Radeon PRO W6800 (PCI 73A3)."""

    def test_id_and_name_both_name_the_card(self):
        self.assertEqual(amd_gpus.arch_from_id(0x73A3), "gfx1030")                 # pci.ids: Navi 21 GL-XL [Radeon PRO W6800]
        self.assertEqual(amd_gpus.arch_from_name("AMD Radeon PRO W6800"), "gfx1030")
        self.assertEqual(strata_setup.win_amd_arch(0x73A3, ""), "gfx1030")
        self.assertEqual(strata_setup.win_amd_arch(None, "AMD Radeon PRO W6800"), "gfx1030")

    def test_the_card_is_a_card_and_the_igpu_is_not(self):
        self.assertIn(0x13C0, amd_gpus.IGPU_IDS)        # no gfx known for it, but it is integrated whatever memory the registry says
        cards = ec.windows_amd_cards([
            {"vendor": "1002", "device": 0x13C0, "name": "AMD Radeon(TM) Graphics", "vram_gb": 32.0},
            {"vendor": "1002", "device": 0x73A3, "name": "AMD Radeon PRO W6800", "vram_gb": 32.0}])
        self.assertEqual([c["arch"] for c in cards], ["gfx1030"])

    def test_the_feed_holds_the_rdna2_build(self):
        best, why = feedmod.select_engine(FEED, hw(["AMD Radeon(TM) Graphics", "AMD Radeon PRO W6800"]))
        self.assertIsNotNone(best, "; ".join(why))
        self.assertEqual(best.variant, "rdna2")


class OneTable(unittest.TestCase):
    """setup.win_amd_arch answers with the shared tables; the names below must keep answering the same on both sides."""

    NAMES = ["AMD Radeon RX 9070 XT", "AMD Radeon RX 9060 XT", "AMD Radeon AI PRO R9700",
             "AMD Radeon RX 7900 XTX", "AMD Radeon PRO W7900", "AMD Radeon PRO W7800",
             "AMD Radeon RX 7800 XT", "AMD Radeon RX 7700 XT", "AMD Radeon RX 7700S",
             "AMD Radeon RX 7600 XT", "AMD Radeon RX 7600M XT", "AMD Radeon PRO W7600", "AMD Radeon PRO W7500",
             "AMD Radeon RX 6950 XT", "AMD Radeon RX 6800 XT", "AMD Radeon RX 6800M", "AMD Radeon PRO W6800",
             "AMD Radeon RX 6700 XT", "AMD Radeon RX 6600 XT", "AMD Radeon RX 6500",
             "AMD Radeon(TM) 8060S Graphics", "AMD Radeon(TM) 8040S Graphics", "AMD Ryzen AI Max 390",
             "AMD Radeon 890M Graphics", "AMD Radeon 840M Graphics", "AMD Radeon 780M Graphics",
             "AMD Radeon(TM) Graphics", "AMD Radeon RX Future", "NVIDIA GeForce RTX 4090", ""]

    def test_setup_and_the_updater_answer_the_same_name_by_name(self):
        for name in self.NAMES:
            self.assertEqual(strata_setup.win_amd_arch(None, name), amd_gpus.arch_from_name(name), name)

    def test_an_integrated_radeon_never_decides_the_engine(self):
        for did in amd_gpus.IGPU_ARCH:
            self.assertIn(did, amd_gpus.IGPU_IDS)
        self.assertFalse(set(amd_gpus.UMA_ARCH) & amd_gpus.IGPU_IDS)          # Strix Halo is a card
        self.assertFalse((set(amd_gpus.DEVICE_ARCH) | set(amd_gpus.UMA_ARCH)) & amd_gpus.IGPU_OLD_IDS)


if __name__ == "__main__":
    unittest.main()
