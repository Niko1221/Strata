"""#7090: a fresh Qwen Q2_0 setup on a CPU with AVX-512 (Zen 4/5, Ice Lake and newer) with a CUDA, HIP or SYCL card stopped at start.

Setup builds the CANONICAL Q2_0 pack there (experts.bin, no native_experts.txt: `elif model == "Q2_0" and avx512 and family == "qwen"`), Auto meant DFlash2, so setup
skipped the MTP layer and Control started the engine with `--draft dflash2 --draft-fallback none`; the engine refuses a DFlash2 on a pack that is not native
("it needs a native pack; there is no MTP to draft instead, so the start ends here").

Auto is the MTP on a canonical pack, in setup (canonical=) and at the start (``--pack`` in the engine arguments), and the MTP is not replaced by DFlash2 as a stand-in
either.  Nothing else moves: Q2_0 on an AVX2 CPU gets a native pack (iq_pack writes native_experts.txt) and keeps DFlash2; DFlash2 asked for by name is the person's choice;
gfx1201 keeps its MTP (AUTO_MTP).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import drafting  # noqa: E402

Q2 = "qwen3.8-flash-next-q2_0"
CARDS = (("cuda", ["sm_120"], 16384), ("cuda", ["sm_86"], 12288), ("hip", ["gfx1100"], 24576), ("sycl", ["arc_a"], 16384), ("hip", ["gfx1151"], None))


class Packs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.canon = self.root / "canonical"          # setup's AVX-512 Q2_0 pack: experts.bin + index.txt
        self.canon.mkdir()
        (self.canon / "experts.bin").write_bytes(b"e")
        (self.canon / "index.txt").write_text("i")
        self.native = self.root / "native"            # setup's AVX2 Q2_0 pack (tools/iq_pack.py): native_experts.txt, no experts.bin needed
        self.native.mkdir()
        (self.native / "native_experts.txt").write_text("x")
        self.native_bin = self.root / "native_bin"    # the low-RAM mode's native pack: experts.bin AND native_experts.txt
        self.native_bin.mkdir()
        (self.native_bin / "experts.bin").write_bytes(b"e")
        (self.native_bin / "native_experts.txt").write_text("x")
        self.empty = self.root / "empty"
        self.empty.mkdir()
        patcher = mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, pack):
        return ["--native", "m.gguf", "--pack", str(pack), "--spec", "4", "--max-context", "65536"]


class CanonicalPack(Packs):
    def test_the_canonical_pack_is_read_from_the_engine_arguments(self):
        self.assertTrue(drafting.canonical_pack(self.args(self.canon)))
        self.assertFalse(drafting.canonical_pack(self.args(self.native)))
        self.assertFalse(drafting.canonical_pack(self.args(self.native_bin)))
        self.assertFalse(drafting.canonical_pack(self.args(self.empty)))                  # not known: neither file
        self.assertFalse(drafting.canonical_pack(self.args(self.root / "missing")))
        self.assertFalse(drafting.canonical_pack(["--native", "m.gguf"]))
        self.assertFalse(drafting.canonical_pack(["--pack"]))
        self.assertFalse(drafting.canonical_pack(None))
        self.assertFalse(drafting.canonical_pack([]))


class Choice(Packs):
    def test_auto_is_the_mtp_on_a_canonical_pack_on_every_backend(self):
        for backend, archs, vram in CARDS:
            got = drafting.choice_of("auto", model_name=Q2, family="qwen", backend=backend, archs=archs, args=self.args(self.canon), vram_mib=vram)
            self.assertEqual(got, "mtp", (backend, archs))

    def test_a_native_pack_keeps_dflash2_where_auto_gave_it(self):
        for pack in (self.native, self.native_bin, self.empty):
            for backend, archs, vram in CARDS:
                before = drafting.choice_of("auto", model_name=Q2, family="qwen", backend=backend, archs=archs, vram_mib=vram)
                got = drafting.choice_of("auto", model_name=Q2, family="qwen", backend=backend, archs=archs, args=self.args(pack), vram_mib=vram)
                self.assertEqual(got, before, (pack.name, backend, archs))
        self.assertEqual(drafting.choice_of("auto", model_name=Q2, family="qwen", backend="cuda", archs=["sm_120"], args=self.args(self.native)), "dflash2")

    def test_gfx1201_is_the_mtp_with_either_pack(self):
        for pack in (self.canon, self.native):
            self.assertEqual(drafting.choice_of("auto", model_name=Q2, family="qwen", backend="hip", archs=["gfx1201"], args=self.args(pack)), "mtp")

    def test_the_mac_is_left_alone(self):
        # the Metal engine has its own DFlash2 rules (SET_DEFAULTS); the canonical-pack refusal is generate.cpp's (CUDA, HIP, SYCL)
        name = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
        for pack in (self.canon, self.native):
            self.assertEqual(drafting.choice_of("auto", model_name=name, family="qwen", backend="metal", archs=[], args=self.args(pack)),
                             drafting.choice_of("auto", model_name=name, family="qwen", backend="metal", archs=[]))
        self.assertEqual(drafting.choice_of("auto", model_name=name, family="qwen", backend="metal", archs=[], args=self.args(self.canon)), "dflash2")

    def test_dflash2_named_is_the_persons_choice(self):
        got = drafting.choice_of("dflash2", model_name=Q2, family="qwen", backend="cuda", archs=["sm_120"], args=self.args(self.canon))
        self.assertEqual(got, "dflash2")

    def test_off_and_mtp_are_as_asked(self):
        self.assertEqual(drafting.choice_of("off", model_name=Q2, family="qwen", backend="cuda", archs=["sm_120"], args=self.args(self.canon)), "off")
        self.assertEqual(drafting.choice_of("mtp", model_name=Q2, family="qwen", backend="cuda", archs=["sm_120"], args=self.args(self.canon)), "mtp")

    def test_setups_flag_says_the_pack_is_not_built_yet(self):
        kw = dict(model_name=Q2, family="qwen", backend="cuda", archs=["sm_120"])
        self.assertEqual(drafting.choice_of("auto", canonical=True, **kw), "mtp")
        self.assertEqual(drafting.choice_of("auto", canonical=False, **kw), "dflash2")
        self.assertEqual(drafting.choice_of("auto", **kw), "dflash2")
        self.assertEqual(drafting.choice_of("auto", canonical=False, args=self.args(self.canon), **kw), "dflash2")   # an explicit False wins

    def test_another_model_is_not_changed_by_the_flag_it_is_the_packs_state_that_counts(self):
        # (the rule reads the pack, not the model: a canonical pack only exists for the AVX-512 Q2_0, so no other set is affected in practice)
        self.assertEqual(drafting.choice_of("auto", model_name="qwen3.8-flash-next-iq3_s", family="qwen", backend="cuda", archs=["sm_86"], args=self.args(self.native)),
                         drafting.choice_of("auto", model_name="qwen3.8-flash-next-iq3_s", family="qwen", backend="cuda", archs=["sm_86"]))


class Decide(Packs):
    def setUp(self):
        super().setUp()
        self.data = self.root / "data"
        (self.data / "dflash2").mkdir(parents=True)
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(b"s" * (2 << 20))
        (self.data / "dflash2" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)

    def decide(self, mode, pack, mtp_here, backend="cuda", archs=("sm_120",)):
        return drafting.decide(mode, model_name=Q2, family="qwen", data_dir=self.data, backend=backend, archs=list(archs), mtp_here=mtp_here, args=self.args(pack), vram_mib=16384)

    def test_the_failing_start_of_7090_now_has_the_mtp_or_no_drafter_never_dflash2(self):
        # the reporter's start: Auto, canonical pack, the MTP layer not on disk, the DFlash2 drafter on disk
        d = self.decide("auto", self.canon, mtp_here=False)
        self.assertNotEqual(d.drafter, "dflash2")
        self.assertEqual(d.drafter, "off")                       # the launch's safety net (fetch_missing_drafter) then fetches the layer
        self.assertEqual(self.decide("auto", self.canon, mtp_here=True).drafter, "mtp")
        self.assertEqual(self.decide("auto", self.canon, mtp_here=None).drafter, "mtp")

    def test_drafting_mtp_without_the_layer_is_not_handed_dflash2_on_a_canonical_pack(self):
        # "--drafting mtp" alone: the layer is not on disk, DFlash2 used to stand in for it and the engine refused the start
        self.assertEqual(self.decide("mtp", self.canon, mtp_here=False).drafter, "off")
        self.assertEqual(self.decide("mtp", self.native, mtp_here=False).drafter, "dflash2")     # a native pack: the stand-in works, as before

    def test_a_native_pack_is_what_it_was(self):
        self.assertEqual(self.decide("auto", self.native, mtp_here=False).drafter, "dflash2")
        self.assertEqual(self.decide("auto", self.native, mtp_here=True).drafter, "dflash2")
        self.assertEqual(self.decide("auto", self.native_bin, mtp_here=True).drafter, "dflash2")

    def test_gfx1201_is_unchanged(self):
        for pack in (self.canon, self.native):
            self.assertEqual(self.decide("auto", pack, mtp_here=True, backend="hip", archs=("gfx1201",)).drafter, "mtp")

    def test_dflash2_named_is_unchanged(self):
        self.assertEqual(self.decide("dflash2", self.canon, mtp_here=False).drafter, "dflash2")
        self.assertEqual(self.decide("off", self.canon, mtp_here=False).drafter, "off")


class SetupRule(unittest.TestCase):
    """setup.auto_drafter: `canonical` is True where step 6 builds the AVX-512 Q2_0 pack (model Q2_0, avx512, family qwen, no downloaded pack)."""

    GPU = {"vram_gb": 16.0, "arch": 120}

    def pick(self, canonical, a=None, hardware=("cuda", ["sm_120"]), hip=False):
        a = a or argparse.Namespace(drafter=None, mtp=None)
        with mock.patch.object(drafting, "hardware_for", return_value=hardware):
            return setup.auto_drafter(a, "qwen", {"name": "qwen3.8-flash-next"}, "Q2_0", hip, self.GPU, False, canonical=canonical)

    def test_the_avx512_cpu_gets_the_mtp_layer_not_dflash2(self):
        for hardware, hip in ((("cuda", ["sm_120"]), False), (("cuda", ["sm_86"]), False), (("hip", ["gfx1100"]), True), (("sycl", ["arc_a"]), False)):
            self.assertIsNone(self.pick(True, hardware=hardware, hip=hip), hardware)

    def test_the_avx2_cpu_keeps_dflash2(self):
        self.assertEqual(self.pick(False), "dflash2")
        self.assertEqual(self.pick(False, hardware=("hip", ["gfx1100"]), hip=True), "dflash2")

    def test_gfx1201_is_the_mtp_either_way(self):
        self.assertIsNone(self.pick(True, hardware=("hip", ["gfx1201"]), hip=True))
        self.assertIsNone(self.pick(False, hardware=("hip", ["gfx1201"]), hip=True))

    def test_the_default_is_not_canonical(self):
        with mock.patch.object(drafting, "hardware_for", return_value=("cuda", ["sm_120"])):
            self.assertEqual(setup.auto_drafter(argparse.Namespace(drafter=None, mtp=None), "qwen", {"name": "qwen3.8-flash-next"}, "Q2_0", False, self.GPU, False), "dflash2")

    def test_step_6_never_hands_dflash2_to_the_canonical_pack(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        i = text.index('q2_avx = model == "Q2_0" and avx512 and family == "qwen"\n    if not ready and (picked := auto_drafter(')
        j = text.index("a.drafter, a.drafter_auto = None, False", i)
        self.assertIn('if q2_avx and not ready and getattr(a, "drafter", None) == "dflash2":', text[i:j])
        self.assertLess(j, text.index("mtp_fetch = 5.2 if"))


if __name__ == "__main__":
    unittest.main()
