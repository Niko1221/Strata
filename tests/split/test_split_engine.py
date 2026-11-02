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
        for model in (udq4(layers=64), udq4(routed_types={}),
                      udq4(routed_types={"Q6_K": 48}),                                  # UD-Q6_K_XL-like experts: no kernel
                      udq4(embd_type="IQ4_XS"),                                         # a K-quant set is not converted at load
                      udq4(shexp_types={"BF16": 48, "IQ4_NL": 47}),
                      udq4(output_type="Q6_K"),
                      udq4(ple_type="Q6_K")):
            d = decide(model=model)
            self.assertFalse(d.use_split, model)
            self.assertTrue(d.line.startswith("Using the normal layer split"))

    def test_the_iq3_s_set_and_ud_iq4_xs_run(self):
        iq3s = S.ModelFacts(routed_types={"IQ2_S": 40, "IQ3_XXS": 34, "IQ3_S": 20, "IQ4_XS": 2, "IQ4_NL": 39, "Q2_0": 9},
                            routed_bytes=int(49 * GIB), embd_type="IQ4_XS", shexp_types={"IQ4_XS": 60, "Q4_K": 40, "Q5_K": 20, "IQ4_NL": 24},
                            embd_ram_bytes=int(27.2 * GIB), layers=48, experts=512, ple_type="IQ4_NL", output_type="Q6_K")
        d = decide(model=iq3s)
        self.assertTrue(d.use_split, d.reasons)
        ud_iq4_xs = S.ModelFacts(routed_types={"IQ3_S": 94, "IQ4_XS": 2, "IQ4_NL": 43, "Q8_0": 5}, routed_bytes=int(59.5e9),
                                 embd_type="Q8_0", shexp_types={"Q8_0": 144}, embd_ram_bytes=int(27.5 * GIB), layers=48, experts=512,
                                 ple_type="IQ4_NL", output_type="Q6_K")
        d = decide(model=ud_iq4_xs)
        self.assertTrue(d.use_split, d.reasons)
        self.assertTrue(d.need_vram_gib < 48, d.need_vram_gib)

    def test_a_type_the_loader_cannot_convert_is_a_plain_refusal(self):
        for model, word in ((udq4(routed_types={"IQ3_S": 94}, embd_type="Q3_K"), "cannot convert"),
                            (udq4(routed_types={"IQ3_S": 94}, shexp_types={"Q3_K": 5}), "cannot read"),
                            (udq4(routed_types={"IQ3_S": 94}, ple_type="Q6_K"), "per-layer"),
                            (udq4(routed_types={"IQ3_S": 94, "TQ2_0": 2}), "TQ2_0")):
            d = decide(model=model)
            self.assertFalse(d.use_split)
            self.assertIn(word, d.line)
            self.assertNotIn(chr(10), d.line)          # one plain line

    def test_scan_counts_the_q2_0_repack_twice(self):
        from unittest import mock

        class T:
            def __init__(self, name, type_name, shape, nbytes):
                self.name, self.type_name, self.shape, self._n = name, type_name, shape, nbytes
            def expected_bytes(self):
                return self._n

        class F:
            def __init__(self, path):
                self.tensors = [T("blk.0.ffn_down_exps.weight", "Q2_0", (640, 2560, 512), 1000), T("blk.1.ffn_gate_exps.weight", "IQ3_S", (2560, 640, 512), 5000),
                                T("token_embd.weight", "IQ4_XS", (2560, 248320), 10), T("per_layer_token_embd.weight", "IQ4_NL", (1, 2), 20),
                                T("output.weight", "Q6_K", (2560, 248320), 30)]
        with mock.patch("tools.gguf_reader.GGUFFile", F):
            m = S.scan_model("/x/model-00001-of-00001.gguf")
        self.assertEqual(m.routed_bytes, 2 * 1000 + 5000)
        self.assertEqual((m.embd_type, m.ple_type, m.output_type), ("IQ4_XS", "IQ4_NL", "Q6_K"))
        self.assertEqual(m.embd_ram_bytes, 30)

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
    """Pictures of the rule 'no peer access in any path': nothing under src/split may call the CUDA peer APIs. (The built engine is
    also checked with nm in packaging/build_split_cuda.sh.)"""

    FORBIDDEN = re.compile(r"\b(cudaDeviceEnablePeerAccess|cudaDeviceDisablePeerAccess|cuCtxEnablePeerAccess|cudaMemcpyPeer(?:Async)?|"
                           r"cudaMemcpy3DPeer(?:Async)?|cuMemcpyPeer(?:Async)?|cuMemcpy3DPeer(?:Async)?)\b")

    def test_no_peer_call_in_the_engine_sources(self):
        bad = []
        for p in sorted((ROOT / "src" / "split").rglob("*")):
            if p.suffix not in (".cu", ".cuh", ".cpp", ".hpp", ".h"):
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                code = line.split("//", 1)[0]
                if self.FORBIDDEN.search(code):
                    bad.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
        self.assertEqual(bad, [], "peer access in the split engine:\n" + "\n".join(bad))

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
