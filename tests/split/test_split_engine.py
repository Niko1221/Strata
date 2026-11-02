"""strata_control/split_engine.py (when the all-cards split engine may run) and src/split (no peer access in any path)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from strata_control import split_engine as S

ROOT = Path(__file__).resolve().parents[2]
GIB = 1 << 30

A6000 = dict(name="NVIDIA RTX A6000", vram_mib=49140, cc=86)


def cards(n=2, **kw):
    spec = {**A6000, **kw}
    return [S.Card(index=i, **spec) for i in range(n)]


def udq4(**kw):
    """UD-Q4_K_XL as measured: 71.7 GiB of routed experts, Q8_0 embeddings, a 27.5 GiB embedding table in RAM."""
    f = dict(routed_types={"Q4_K": 97, "Q5_K": 2, "Q5_1": 43, "Q8_0": 5}, routed_bytes=int(71.7 * GIB), embd_type="Q8_0",
             shexp_types={"Q8_0": 144}, embd_ram_bytes=int(27.5 * GIB), layers=48, experts=512)
    f.update(kw)
    return S.ModelFacts(**f)


def decide(**kw):
    a = dict(enabled=True, system="linux", cards=cards(), ram_total_gib=125, ram_available_gib=80, model=udq4())
    a.update(kw)
    return S.decide(**a)


class Rule(unittest.TestCase):
    def test_runs_on_the_tested_setup(self):
        d = decide()
        self.assertTrue(d.use_split, d.reasons)
        self.assertEqual(d.cards, 2)
        self.assertEqual(d.untested, [])
        self.assertIn("2 cards", d.line)

    def test_off_unless_asked(self):
        d = decide(enabled=False)
        self.assertFalse(d.use_split)
        self.assertIn("opt-in", d.line)

    def test_linux_nvidia_only(self):
        for system in ("windows", "darwin"):
            self.assertFalse(decide(system=system).use_split)

    def test_card_count_and_sameness(self):
        self.assertFalse(decide(cards=cards(1)).use_split)
        self.assertFalse(decide(cards=cards(5)).use_split)
        mixed = cards(1) + [S.Card(index=1, name="NVIDIA RTX A5000", vram_mib=24564, cc=86)]
        d = decide(cards=mixed)
        self.assertFalse(d.use_split)
        self.assertIn("same model", d.line)

    def test_untested_counts_archs_and_contexts_stay_off_unless_a_tester_allows_them(self):
        for c in (cards(3), cards(2, cc=89), cards(2, cc=120)):
            d = decide(cards=c)
            self.assertFalse(d.use_split)
            self.assertIn("only been run", d.line)
        d = decide(cards=cards(3, vram_mib=49140), allow_untested=True)
        self.assertTrue(d.use_split, d.reasons)
        self.assertIn("3 cards", d.untested)
        self.assertIn("not tested", d.line)
        self.assertFalse(decide(context=262144).use_split)
        self.assertTrue(decide(context=262144, allow_untested=True).use_split)

    def test_all_the_experts_must_fit_in_the_cards(self):
        d = decide(cards=cards(2, vram_mib=24564, name="NVIDIA RTX A5000"))
        self.assertFalse(d.use_split)
        self.assertIn("do not fit", d.line)
        self.assertFalse(decide(cards=cards(2, vram_mib=40960)).use_split)    # 71.7 / 2 + 9 = 44.9 GiB needed
        self.assertTrue(decide(cards=cards(2, vram_mib=46000)).use_split)

    def test_ram_that_stays_must_be_free(self):
        self.assertTrue(decide(ram_available_gib=52).use_split)          # 27.5 + 3 + 4 + 8 spare = 42.5
        d = decide(ram_available_gib=30)
        self.assertFalse(d.use_split)
        self.assertIn("RAM", d.line)

    def test_only_the_quant_types_it_reads(self):
        for model in (udq4(routed_types={"IQ3_S": 10, "IQ2_S": 20, "IQ4_NL": 39}), udq4(embd_type="IQ4_XS"),
                      udq4(shexp_types={"BF16": 48, "IQ4_NL": 47}), udq4(layers=64), udq4(routed_types={})):
            d = decide(model=model)
            self.assertFalse(d.use_split, model)
            self.assertTrue(d.line.startswith("Using the normal layer split"))

    def test_the_reasons_are_all_reported(self):
        d = decide(system="windows", cards=cards(1), model=udq4(embd_type="Q6_K"))
        self.assertGreaterEqual(len(d.reasons), 3)


class Config(unittest.TestCase):
    def base(self):
        return {"exe": "engine/strata", "args": ["--pack", "p", "--native", "m.gguf", "--layer-split", "auto"], "cwd": "/x",
                "tokenizer": "tok", "port": 8080, "model_name": "m", "log": "l.log", "gpu": [0, 1], "env": {"STRATA_X": "1"},
                "layer_split": "auto"}

    def test_split_config_keeps_the_front_end_and_the_fallback(self):
        d = decide()
        cfg = S.split_config(self.base(), d, exe="/e/strata-split-2", native="/m/shard1.gguf", mtp_dir="/d/mtp/rt",
                             draft_vocab="/d/draft_vocab_en.bin", context=131072, lanes=6, vision=True)
        self.assertEqual(cfg["exe"], "/e/strata-split-2")
        self.assertEqual(cfg["tokenizer"], "tok")
        self.assertEqual(cfg["port"], 8080)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertIn("--vision", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "6")
        self.assertEqual(cfg["env"]["FNX_ARENA"], "0")
        self.assertEqual(cfg["env"]["FNX_EXPERT_RAM_GIB"], "off")
        self.assertEqual(cfg["env"]["STRATA_X"], "1")
        self.assertNotIn("layer_split", cfg)
        fb = cfg["split_fallback"]
        self.assertEqual(fb["exe"], "engine/strata")
        self.assertEqual(fb["args"], self.base()["args"])
        self.assertEqual(fb["layer_split"], "auto")
        self.assertEqual(self.base()["args"], ["--pack", "p", "--native", "m.gguf", "--layer-split", "auto"])   # the base is not changed

    def test_no_config_for_a_refusal(self):
        with self.assertRaises(ValueError):
            S.split_config(self.base(), decide(enabled=False), exe="e", native="n", mtp_dir="m", draft_vocab="v", context=1)


class Apply(unittest.TestCase):
    """apply_to_config: the launcher's side (facts of this PC and model in, a run config out)."""

    def cfg(self, tmp):
        (Path(tmp) / "data").mkdir(exist_ok=True)
        (Path(tmp) / "data" / "draft_vocab_en.bin").write_bytes(b"x")
        return {"exe": "component:strata-engine", "args": ["--pack", "p", "--native", "/m/a-00001-of-00004.gguf", "--mtp", "/d/mtp/rt",
                                                           "--max-context", "131072", "--layer-split", "auto"],
                "cwd": tmp, "tokenizer": "tok", "port": 8080, "gpu": [0, 1], "env": {}, "split_engine": True}

    def run_apply(self, cfg, *, cards_=None, avail=80.0, model=None, exe="/e/strata-split-2", **kw):
        return S.apply_to_config(cfg, exe_for_cards=lambda n: exe, system=kw.pop("system", "linux"), probe=lambda idx: cards_ or cards(),
                                 ram=lambda: (125.0, avail), scan=lambda path: model or udq4(), **kw)

    def test_a_good_pc_gets_the_split_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            d = self.run_apply(cfg)
            self.assertTrue(d.use_split, d.reasons)
            self.assertEqual(cfg["exe"], "/e/strata-split-2")
            self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], "/m/a-00001-of-00004.gguf")
            self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072")
            self.assertEqual(cfg["split_fallback"]["exe"], "component:strata-engine")
            self.assertEqual(cfg["tokenizer"], "tok")

    def test_every_refusal_leaves_the_config_alone(self):
        import copy
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for kw in (dict(cards_=cards(1)), dict(avail=20.0), dict(exe=None), dict(system="windows"),
                       dict(model=udq4(embd_type="Q6_K"))):
                cfg = self.cfg(tmp)
                before = copy.deepcopy(cfg)
                d = self.run_apply(cfg, **kw)
                self.assertFalse(d.use_split, kw)
                self.assertEqual(cfg, before, kw)
                self.assertTrue(d.line.startswith("Using the normal layer split"), d.line)

    def test_a_model_file_it_cannot_read_is_a_reason_not_an_error(self):
        import tempfile

        def boom(path):
            raise OSError("gone")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            d = S.apply_to_config(cfg, exe_for_cards=lambda n: "/e/x", system="linux", probe=lambda i: cards(), ram=lambda: (125.0, 80.0), scan=boom)
            self.assertFalse(d.use_split)
            self.assertIn("could not be read", d.line)

    def test_wanted_is_opt_in(self):
        self.assertFalse(S.wanted({}, {}))
        self.assertFalse(S.wanted({"split_engine": False}, {}))
        self.assertTrue(S.wanted({"split_engine": True}, {}))
        self.assertTrue(S.wanted({}, {"STRATA_SPLIT_ENGINE": "1"}))


class NoPeerAccess(unittest.TestCase):
    """Pictures of the rule 'no peer access in any path' for the NVIDIA split: nothing under src/split may call the CUDA peer APIs, and the HIP peer APIs
    appear in exactly one file, src/split/src/hip/peer.h (AMD only: the owner allowed peer access on the R9700 "if faster, with fallback", 2026-11-02; it runs only
    after a start probe of every directed pair, and FNX_PEER=0 forces the host exchange). The built NVIDIA engine is also checked with nm in
    packaging/build_split_cuda.sh."""

    FORBIDDEN = re.compile(r"(cudaDeviceEnablePeerAccess|cudaDeviceDisablePeerAccess|cuCtxEnablePeerAccess|cudaMemcpyPeer(?:Async)?|"
                           r"cudaMemcpy3DPeer(?:Async)?|cuMemcpyPeer(?:Async)?|cuMemcpy3DPeer(?:Async)?)")
    FORBIDDEN_HIP = re.compile(r"(hipDeviceEnablePeerAccess|hipDeviceDisablePeerAccess|hipMemcpyPeer(?:Async)?|hipMemcpy3DPeer(?:Async)?|hipCtxEnablePeerAccess)")
    HIP_PEER_FILE = "src/split/src/hip/peer.h"

    def _code_lines(self):
        for p in sorted((ROOT / "src" / "split").rglob("*")):
            if p.suffix not in (".cu", ".cuh", ".cpp", ".hpp", ".h"):
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                yield p, n, line, line.split("//", 1)[0]

    def test_no_cuda_peer_call_in_the_engine_sources(self):
        bad = [f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}" for p, n, line, code in self._code_lines() if self.FORBIDDEN.search(code)]
        self.assertEqual(bad, [], "CUDA peer access in the split engine:\n" + "\n".join(bad))

    def test_hip_peer_calls_only_in_the_one_probed_file(self):
        bad = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()[:100]}" for p, n, line, code in self._code_lines()
               if self.FORBIDDEN_HIP.search(code) and p.relative_to(ROOT).as_posix() != self.HIP_PEER_FILE]
        self.assertEqual(bad, [], "HIP peer access outside " + self.HIP_PEER_FILE + ":\n" + "\n".join(bad))

    def test_the_hip_peer_file_enables_access_but_never_uses_the_copy_engine(self):
        text = (ROOT / self.HIP_PEER_FILE).read_text(encoding="utf-8")
        code = "\n".join(l.split("//", 1)[0] for l in text.splitlines())
        self.assertIn("hipDeviceEnablePeerAccess", code)
        self.assertNotRegex(code, r"hipMemcpy(?:3D)?Peer")   # kernel stores only (the owner's condition), no peer copies

    def test_the_peer_file_is_only_compiled_into_the_hip_build_and_the_default_is_pinned(self):
        engine = (ROOT / "src" / "split" / "src" / "engine.cu").read_text(encoding="utf-8")
        i = engine.index('#include "hip/peer.h"')
        self.assertEqual(engine[:i].rstrip().splitlines()[-1].strip(), "#if FNX_HIP")
        policy = (ROOT / "src" / "split" / "src" / "hip" / "peer_policy.hpp").read_text(encoding="utf-8")
        self.assertRegex(policy, r"constexpr bool PEER_DEFAULT_ON = (false|true);")   # flipped to true only on screens that show it wins

    def test_rccl_is_loaded_at_run_time_never_linked_and_its_default_is_pinned(self):
        # the packaging rule (coordinator, 11-02): dlopen only, no link; the split must start without librccl, falling back with one line
        rd = (ROOT / "src" / "split" / "src" / "hip" / "rccl_dl.hpp").read_text(encoding="utf-8")
        self.assertIn("dlopen(", rd)
        self.assertNotIn("hip_runtime", rd)   # CPU only: tools/rccl_policy_test.cpp forces a missing library without a card
        self.assertRegex(rd, r"constexpr bool RCCL_DEFAULT_ON = (false|true);")
        cmake = (ROOT / "src" / "split" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotRegex(cmake.replace("rccl_policy_test", ""), r"(?i)rccl")   # no find_package(rccl), no -lrccl
        for p in (ROOT / "src" / "split" / "src").rglob("*"):
            if p.suffix in (".cu", ".cuh", ".cpp", ".hpp", ".h") and p.name not in ("rccl.h", "rccl_dl.hpp", "prefill.cu", "engine.cu", "engine.hpp"):
                self.assertNotIn("rccl.h>", p.read_text(encoding="utf-8", errors="ignore"), p.name)

    def test_the_peer_probe_and_the_27b_files_are_gone(self):
        names = {p.name for p in (ROOT / "src" / "split" / "src").iterdir()}
        self.assertNotIn("peer_probe.cu", names)
        self.assertFalse([n for n in names if n.startswith("s27_")])

    def test_the_licence_and_the_credit_travel_with_the_code(self):
        lic = (ROOT / "third_party" / "sfnx" / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("John Fink", lic)
        self.assertIn("MIT License", lic)
        notice = (ROOT / "third_party" / "sfnx" / "NOTICE.md").read_text(encoding="utf-8")
        self.assertIn("John Fink", notice)
        self.assertIn("github.com/johnfink8/sfnx", notice)


if __name__ == "__main__":
    unittest.main()
