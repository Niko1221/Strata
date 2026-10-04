"""Tests for tools/hip/autotune.py without a GPU: the model's shapes, the tuner's output, the end-to-end verdict and
the config it writes.

    python -m unittest tools.hip.test_autotune
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
sys.path.insert(0, str(ROOT / "tools"))
import autotune as AT  # noqa: E402


def T(name, shape, type_id):
    return SimpleNamespace(name=name, shape=shape, type_id=type_id)


class Shapes(unittest.TestCase):
    def test_experts_dense_and_exclusions(self):
        ts = [T(f"blk.{i}.ffn_gate_exps.weight", [2560, 640, 512], 18) for i in range(3)]
        ts += [T(f"blk.{i}.ffn_up_exps.weight", [2560, 640, 512], 18) for i in range(3)]
        ts += [T("blk.0.ffn_down_exps.weight", [640, 2560, 512], 20), T("blk.1.ffn_down_exps.weight", [640, 2560, 512], 20),
               T("blk.2.ffn_down_exps.weight", [640, 2560, 512], 42)]
        ts += [T("blk.0.attn_q.weight", [2560, 4096], 18), T("blk.1.attn_q.weight", [2560, 4096], 18),   # one shape
               T("blk.0.attn_k.weight", [2560, 512], 8),                                                   # Q8_0: not iq_mmvq
               T("token_embd.weight", [2560, 248320], 18),                                                 # an embedding
               T("blk.3.ffn_gate_exps.weight", [2560, 640, 512], 21)]                                     # no down: skipped
        sh = AT.shapes_from_tensors(ts)
        self.assertEqual(sh["experts"], [(18, 20), (18, 42)])
        self.assertEqual((sh["n_embd"], sh["n_ff"]), (2560, 640))
        self.assertEqual(sh["mmvq"], [(18, 2560, 4096)])
        a = AT.tuner_args(sh, Path("/x/t.txt"), quick=True)
        self.assertEqual(a.count("--expert"), 2)
        self.assertIn("18:42", a)
        self.assertIn("18:2560:4096", a)
        self.assertIn("--rounds", a)

    def test_non_native_down_is_skipped(self):
        sh = AT.shapes_from_tensors([T("blk.0.ffn_gate_exps.weight", [2560, 640, 512], 18),
                                     T("blk.0.ffn_down_exps.weight", [640, 2560, 512], 12)])   # Q4_K down
        self.assertEqual(sh["experts"], [])


class TunerOutput(unittest.TestCase):
    def test_parse_results(self):
        out = ("noise\nRESULT experts 18 20 2560 640 default_us=41.20 chosen=16/4 chosen_us=36.10 changed=1\n"
               "RESULT mmvq 18 2560 4096 default_us=9.00 chosen=4 chosen_us=9.00 changed=0\nSUMMARY changed=2 refused=0\n")
        r = AT.parse_results(out)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0]["what"], "experts 18 20 2560 640")
        self.assertEqual(r[0]["chosen"], "16/4")
        self.assertTrue(r[0]["changed"])
        self.assertAlmostEqual(r[0]["chosen_us"], 36.1)
        self.assertFalse(r[1]["changed"])

    def test_table_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.txt"
            p.write_text("# made here\nSTRATA_DECODE_TUNING_V1 gfx1100 1 abc\ngu 18 2560 640 16\n\ndown 20 640 2560 4  # x\n")
            self.assertEqual(AT.table_shapes(p), 2)
            p.write_text("# nothing won\nSTRATA_DECODE_TUNING_V1 gfx1100 1 abc\n")
            self.assertEqual(AT.table_shapes(p), 0)


class FakeEngine:
    def __init__(self, rate, tokens):
        self.rate, self.tokens, self.last, self.proc = rate, tokens, {}, None

    def generate(self, ids, max_new, sampling, cancel):
        toks = self.tokens(ids)
        yield from toks
        self.last = {"decode_ms": len(toks) / self.rate * 1000.0}


class EndToEnd(unittest.TestCase):
    IDS = [[1, 2], [3, 4], [5, 6]]

    def verdict(self, rate_on, tokens_on, tokens_off=None, pairs=2):
        n = {"off": 0}

        def start(table):
            if table:
                return FakeEngine(rate_on, tokens_on)
            n["off"] += 1
            return FakeEngine(50.0, tokens_off(n["off"]) if tokens_off else (lambda ids: list(range(20))))
        return AT.end_to_end(start, "/t.txt", self.IDS, pairs=pairs, say=lambda *_: None, close=lambda e: None)

    def test_faster_and_identical_is_kept(self):
        v = self.verdict(53.0, lambda ids: list(range(20)))
        self.assertTrue(v["keep"])
        self.assertEqual(v["tokens"], "identical tokens")
        self.assertAlmostEqual(v["gain"], 0.06, places=3)

    def test_noise_level_slower_is_still_kept(self):
        self.assertTrue(self.verdict(49.5, lambda ids: list(range(20)))["keep"])   # -1%: within MAX_SLOWDOWN

    def test_slower_is_dropped(self):
        self.assertFalse(self.verdict(45.0, lambda ids: list(range(20)))["keep"])

    def test_different_tokens_are_dropped(self):
        v = self.verdict(60.0, lambda ids: list(range(19)) + [99])
        self.assertFalse(v["keep"])
        self.assertEqual(v["tokens"], "TOKENS DIFFER")

    def test_nondeterministic_default_is_judged_on_speed(self):
        v = self.verdict(55.0, lambda ids: [7] * 20, tokens_off=lambda k: (lambda ids: [k] * 20))
        self.assertTrue(v["keep"])
        self.assertIn("not checkable", v["tokens"])

    def test_quick_cannot_excuse_different_tokens(self):
        v = self.verdict(60.0, lambda ids: [7] * 20, pairs=1)
        self.assertFalse(v["keep"])


class Config(unittest.TestCase):
    def test_save_and_off(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-iq3_xxs.json"
            cfg = {"exe": "x", "args": [], "env": {"STRATA_HIPBLASLT_TUNING": "/h.txt"}, "port": 1}
            p.write_text(json.dumps(cfg))
            AT.save_config(p, json.loads(p.read_text()), "/t.txt")
            w = json.loads(p.read_text())
            self.assertEqual(w["env"], {"STRATA_HIPBLASLT_TUNING": "/h.txt", AT.ENV_KEY: "/t.txt"})
            self.assertTrue((Path(d) / "strata-iq3_xxs.json.bak-autotune").exists())
            self.assertEqual(AT.main([str(p), "--off"]), 0)
            self.assertEqual(json.loads(p.read_text())["env"], {"STRATA_HIPBLASLT_TUNING": "/h.txt"})
            # the backup is the ORIGINAL config, not overwritten by later saves
            self.assertEqual(json.loads((Path(d) / "strata-iq3_xxs.json.bak-autotune").read_text()), cfg)

    def test_env_dropped_when_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"exe": "x", "args": [], "env": {AT.ENV_KEY: "/t.txt"}}))
            AT.save_config(p, json.loads(p.read_text()), None)
            self.assertNotIn("env", json.loads(p.read_text()))


if __name__ == "__main__":
    unittest.main()
