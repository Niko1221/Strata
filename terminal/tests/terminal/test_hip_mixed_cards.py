"""#6650: mixed AMD cards. An R9700 (gfx1201) beside a Ryzen's 780M (gfx1103) got the rx7900 build (it lists gfx1103), because "any one card is
covered" counted the iGPU and the first package in feed order won. The engine then rejected the discrete card (Qwen) or served garbage (Inkling).
The discrete cards are what a build has to cover; among builds that do, the one for the card the model runs on wins."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from strata_control import engine_component as ec, feed as feedmod

LINUX = "linux-x86_64"


def pkg(version, variant, archs):
    return feedmod._engine_package({
        "version": version, "platform": LINUX, "backend": "hip", "variant": variant, "archs": list(archs), "min_driver": {},
        "min_runtime": None, "build_id": f"b-{version}-{variant}", "format": "zip", "url": f"e-{version}-{variant}.zip", "size": 10,
        "sha256": "a" * 64, "manifest_sha256": "b" * 64, "installed_bytes": 20, "notes": ""})


# the live feed's order (sequence 131): rx7900 first, then rx9070, and an RX 6000 build
FEED = [
    pkg("0.1.65", "rx7900", ("gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150")),
    pkg("0.1.65", "rx9070", ("gfx1200", "gfx1201")),
    pkg("0.1.65", "rx6000", ("gfx1030", "gfx1031", "gfx1032")),
]


def hw(gfx, main="", **extra):
    out = {"platform": LINUX, "backend": "hip", "runtime": "0.2.20", "nvidia_driver": None, "cuda_archs": [], "gfx": list(gfx),
           "gpu_names": [], "installed_variant": "", "variant_wanted": ""}
    if main:
        out["gfx_main"] = main
    out.update(extra)
    return out


def pick(hardware, feed=FEED):
    best, reasons = feedmod.select_engine(feed, hardware)
    return (best.variant if best else None), reasons


class PackageChoice(unittest.TestCase):
    def test_r9700_beside_a_780m_gets_rx9070_not_rx7900(self):
        # the report: updater.detect_hardware() gave gfx = [gfx1103, gfx1201]
        self.assertEqual(pick(hw(["gfx1103", "gfx1201"]))[0], "rx9070")
        self.assertEqual(pick(hw(["gfx1201", "gfx1103"], main="gfx1201"))[0], "rx9070")

    def test_only_the_igpu_covered_is_not_a_fit_when_the_feed_has_no_build_for_the_card(self):
        got, why = pick(hw(["gfx1103", "gfx1201"]), [FEED[0]])
        self.assertIsNone(got)
        text = " ".join(why)
        self.assertIn("gfx1201", text)
        self.assertIn("integrated gfx1103", text)

    def test_an_igpu_alone_is_still_served(self):
        self.assertEqual(pick(hw(["gfx1103"]))[0], "rx7900")
        self.assertEqual(pick(hw(["gfx1151"]), [pkg("0.1.65", "halo", ("gfx1151",))] + FEED)[0], "halo")

    def test_boxia_7900xt_6700xt_and_an_890m(self):
        # boxia: gfx1100 (24 GB) + gfx1031 (12 GB) + the 890M (gfx1150). Both rx7900 and rx6000 cover a discrete card; the 7900 XT is the main card.
        self.assertEqual(pick(hw(["gfx1031", "gfx1100", "gfx1150"], main="gfx1100"))[0], "rx7900")
        self.assertEqual(pick(hw(["gfx1031", "gfx1100", "gfx1150"]))[0], "rx7900")          # main unknown: the old tie-break (rx6000 never wins)
        self.assertEqual(pick(hw(["gfx1031", "gfx1100", "gfx1150"], main="gfx1031"))[0], "rx6000")   # the person put the 6700 XT first

    def test_two_cards_of_one_family_are_unchanged(self):
        self.assertEqual(pick(hw(["gfx1100", "gfx1100"]))[0], "rx7900")
        self.assertEqual(pick(hw(["gfx1201", "gfx1201"]))[0], "rx9070")

    def test_a_gfx_unknown_is_not_a_refusal(self):
        got, _ = pick(hw([], variant_wanted="rx9070"))
        self.assertEqual(got, "rx9070")

    def test_a_person_who_points_the_model_at_the_igpu_gets_the_build_for_it(self):
        # HIP_VISIBLE_DEVICES=0 (the 780M): the main card is then the iGPU
        self.assertEqual(pick(hw(["gfx1103", "gfx1201"], main="gfx1103"))[0], "rx7900")
        self.assertEqual(pick(hw(["gfx1103", "gfx1201"], main="gfx1103", variant_wanted="rx7900"))[0], "rx7900")

    def test_a_variant_asked_for_never_overrides_what_fits(self):
        got, _ = pick(hw(["gfx1103", "gfx1201"], main="gfx1201", variant_wanted="rx7900"))
        self.assertEqual(got, "rx9070")


class Helpers(unittest.TestCase):
    def test_counted(self):
        self.assertEqual(ec.hip_counted(["gfx1103", "gfx1201"]), ["gfx1201"])
        self.assertEqual(ec.hip_counted(["gfx1103"]), ["gfx1103"])
        self.assertEqual(ec.hip_counted(["GFX1100:xnack-", "gfx1150"]), ["gfx1100"])
        self.assertEqual(ec.hip_counted([]), [])

    def test_igpu_lists_agree(self):
        from serve.server_parts import engines
        self.assertEqual(tuple(engines.HIP_IGPU_ARCHS), tuple(ec.HIP_IGPU_ARCHS))

    def test_main_arch_prefers_the_discrete_card_with_the_most_memory(self):
        cards = [{"index": 0, "arch": "gfx1150", "vram_gb": 0.5}, {"index": 1, "arch": "gfx1031", "vram_gb": 12.0}, {"index": 2, "arch": "gfx1100", "vram_gb": 24.0}]
        self.assertEqual(ec.hip_main_arch(cards, {}), "gfx1100")
        self.assertEqual(ec.hip_main_arch(cards, {"HIP_VISIBLE_DEVICES": "1,2"}), "gfx1031")
        self.assertEqual(ec.hip_main_arch(cards, {"ROCR_VISIBLE_DEVICES": "0"}), "gfx1150")
        self.assertEqual(ec.hip_main_arch(cards, {"HIP_VISIBLE_DEVICES": "9"}), "gfx1100")
        self.assertEqual(ec.hip_main_arch([], {}), "")

    def test_cards_from_the_kfd_topology(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for number, (version, simd, vram) in enumerate([(0, 0, 0), (110003, 12, 536870912), (120001, 64, 34359738368)]):
                node = root / str(number)
                (node / "mem_banks" / "0").mkdir(parents=True)
                (node / "properties").write_text(f"cpu_cores_count 1\nsimd_count {simd}\ngfx_target_version {version}\n", encoding="utf-8")
                (node / "mem_banks" / "0" / "properties").write_text(f"heap_type {1 if simd else 0}\nsize_in_bytes {vram}\n", encoding="utf-8")
            cards = ec.hip_cards_present(str(root))
            self.assertEqual([(c["index"], c["arch"], round(c["vram_gb"], 1)) for c in cards], [(0, "gfx1103", 0.5), (1, "gfx1201", 32.0)])
            self.assertEqual(ec.hip_main_arch(cards, {}), "gfx1201")
        self.assertEqual(ec.hip_cards_present("/nonexistent/kfd"), [])


if __name__ == "__main__":
    unittest.main()
