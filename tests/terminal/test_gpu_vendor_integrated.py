"""#4971: an integrated GPU beside a discrete one must not decide this PC's engine family.

`strata components engine` read the PCI vendors in /sys/class/drm in a fixed order (NVIDIA, AMD, Intel) and returned
the first one found, so a 512 MiB Radeon APU iGPU - which Strata cannot run the engine on - made the whole PC an AMD
machine beside the user's Intel Arc Pro B60.  The backend detection now skips a card whose own VRAM is an integrated
GPU's small BIOS carve-out while another card is present.

A real two-vendor sysfs cannot be produced from a test, so each case is a temporary /sys/class/drm tree.

    python -m pytest tests/terminal/test_gpu_vendor_integrated.py
"""
import tempfile
import unittest
from pathlib import Path

from strata_control import engine_component as ec


def card(root: Path, number: int, vendor: str, vram_mib: int | None = None) -> None:
    """One display adapter in a fake /sys/class/drm: `vendor` is the PCI vendor id, `vram_mib` its own VRAM."""
    device = root / f"card{number}" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n", encoding="utf-8")
    if vram_mib is not None:
        (device / "mem_info_vram_total").write_text(str(vram_mib * 1024 * 1024) + "\n", encoding="utf-8")


class IntegratedGpuVendor(unittest.TestCase):
    def backend(self, *cards) -> str | None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (vendor, vram_mib) in enumerate(cards):
                card(root, index, vendor, vram_mib)
            return ec._drm_vendor_backend(str(root))

    def test_an_apu_beside_an_intel_arc_does_not_make_the_pc_amd(self):
        # The report: AMD Raphael iGPU (512 MiB carve-out) + Intel Arc Pro B60 (xe, no mem_info_vram_total file).
        self.assertEqual(self.backend(("0x1002", 512), ("0x8086", None)), "sycl")

    def test_a_discrete_amd_card_still_decides(self):
        self.assertEqual(self.backend(("0x1002", 24576)), "hip")

    def test_an_apu_beside_a_radeon_card_keeps_hip(self):
        self.assertEqual(self.backend(("0x1002", 512), ("0x1002", 16384)), "hip")

    def test_nvidia_keeps_the_old_order(self):
        # NVIDIA reports no mem_info_vram_total: the old first-vendor answer stays.
        self.assertEqual(self.backend(("0x1002", 512), ("0x10de", None)), "cuda")
        self.assertEqual(self.backend(("0x10de", None), ("0x8086", None)), "cuda")

    def test_an_integrated_card_alone_keeps_the_old_answer(self):
        self.assertEqual(self.backend(("0x1002", 512)), "hip")

    def test_no_drm_tree_is_none(self):
        self.assertIsNone(ec._drm_vendor_backend(str(Path(tempfile.gettempdir()) / "no-such-strata-drm")))


if __name__ == "__main__":
    unittest.main()
