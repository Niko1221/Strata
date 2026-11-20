"""Tests for terminal/tools/mtp_from_gguf.py: a tiny MTP head in HF layout is written the way llama.cpp's Qwen4Exp
converter writes it (renamed tensors, norms as 1 + w in F32, eh_proj and gate_up_proj joined, index_qk_proj split),
and the tool must give back the original BF16 bits.  Also: the newer merged hc_*_down_inject layout is split again,
a head with another number of routed experts than 512 comes through with its own count (the engine reads that count
from experts.bin), and a tensor the tool does not know stops it.  No model files and no network.

    python -m unittest terminal.tools.test_mtp_from_gguf
"""
from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_from_gguf as M  # noqa: E402
import gguf  # noqa: E402  (mtp_from_gguf put gguf-py on the path)

BF16 = gguf.GGMLQuantizationType.BF16
H, HC, E, FF = 8, 4, 6, 4          # hidden, hyper-connection count, routed experts (not 512), expert width


def bf16(shape, seed):
    """Random BF16 bits of values exact in BF16 and in F32 1 + w (multiples of 1/64 in [-1, 1))."""
    v = np.random.default_rng(seed).integers(-64, 64, size=shape).astype(np.float32) / 64
    return M.bf16_bits(v)


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def hf_head():
    """HF name -> BF16 bits of a small head; norms are 1-D."""
    t, s = {}, 0
    for name, norm in M.SIMPLE.values():
        if norm:
            shape = [H]
        elif "experts.down_proj" in name:
            shape = [E, H, FF]
        elif name.endswith("mlp.gate.weight"):
            shape = [E, H]
        elif "inject" in name:
            shape = [HC, H]
        else:
            shape = [H, H]
        s += 1
        t[name] = bf16(shape, s)
    t["mtp.fc_embedding.weight"] = bf16([H, H], 101)
    t["mtp.fc_hidden.weight"] = bf16([H, H], 102)
    t["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"] = bf16([2 * H, H], 103)
    t["mtp.layers.0.mlp.experts.gate_up_proj"] = bf16([E, 2 * FF, H], 104)
    return t


def restore_copy(path):
    """M.restore, copied out of the reader's memory map so Windows can delete the file afterwards."""
    got = {k: np.array(v) for k, v in M.restore(path).items()}
    gc.collect()
    return got


def write_gguf(path, hf, merged=False, extra=None):
    """The converter's side: what llama.cpp writes for this head, in block 3 (any block number works)."""
    w = gguf.GGUFWriter(str(path), "qwen4exp")
    w.add_uint32("qwen4exp.hyper_connection.count", HC)
    back = {v[0]: (k, v[1]) for k, v in M.SIMPLE.items()}
    out = {}
    for name, bits in hf.items():
        if name in back:
            k, norm = back[name]
            out[k] = (f32(bits) + np.float32(1)) if norm else bits
    if merged:
        for part in ("attn", "ffn"):
            out[f"hc_{part}_down_inject.weight"] = np.concatenate(
                [out.pop(f"hc_{part}_down.weight"), out.pop(f"hc_{part}_inject.weight")], axis=0)
    out["nextn.eh_proj.weight"] = np.concatenate([hf["mtp.fc_embedding.weight"], hf["mtp.fc_hidden.weight"]], axis=1)
    qk = hf["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"]
    out["indexer.q_proj.weight"], out["indexer.k_proj.weight"] = qk[:H], qk[H:]
    gu = hf["mtp.layers.0.mlp.experts.gate_up_proj"]
    out["ffn_gate_exps.weight"], out["ffn_up_exps.weight"] = gu[:, :FF], gu[:, FF:]
    out.update(extra or {})
    for k, x in out.items():
        x = np.ascontiguousarray(x)
        if x.dtype == np.uint16:
            w.add_tensor(f"blk.3.{k}", x, raw_dtype=BF16)
        else:
            w.add_tensor(f"blk.3.{k}", x.astype(np.float32))
    w.add_tensor("token_embd.weight", np.zeros([4, H], np.float32))    # shared with the trunk: skipped
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


class RoundTrip(unittest.TestCase):
    def check(self, merged):
        hf = hf_head()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "head.gguf"
            write_gguf(p, hf, merged=merged)
            got = restore_copy(p)
        self.assertEqual(sorted(got), sorted(hf))
        for name, want in hf.items():
            g = np.asarray(got[name])
            self.assertEqual(g.dtype, np.uint16, name)
            self.assertEqual(list(g.shape), list(want.shape), name)
            self.assertTrue(np.array_equal(g, want), name)

    def test_split_layout(self):
        self.check(merged=False)

    def test_merged_down_inject_layout(self):
        self.check(merged=True)

    def test_expert_count_is_the_heads_own(self):
        hf = hf_head()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "head.gguf"
            write_gguf(p, hf)
            got = restore_copy(p)
        self.assertEqual(got["mtp.layers.0.mlp.experts.gate_up_proj"].shape[0], E)
        self.assertEqual(got["mtp.layers.0.mlp.gate.weight"].shape[0], E)

    def test_unknown_tensor_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "head.gguf"
            write_gguf(p, hf_head(), extra={"something_new.weight": bf16([H], 999)})
            with self.assertRaisesRegex(ValueError, "something_new"):
                restore_copy(p)
            gc.collect()


if __name__ == "__main__":
    unittest.main()
