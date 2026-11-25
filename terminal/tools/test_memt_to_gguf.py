"""tools/test_memt_to_gguf.py - synthetic-checkpoint tests for memt_to_gguf.py.

    python tools/test_memt_to_gguf.py [-v]

The fixture writes real BF16 safetensors (small shapes, deterministic values),
converts them, and reads the GGUF back with Strata's own reader.  It proves the
tensor names/shapes, the metadata, and every numeric transform without a GPU or
the 72 GB checkpoint.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memt_to_gguf as C  # noqa: E402
import strata_gguf as SG  # noqa: E402
import strata_quants as SQ  # noqa: E402
import test_memt_audit as T  # noqa: E402


def bf16(value: np.ndarray) -> np.ndarray:
    """Round-trip through BF16 so expected values match what the file holds."""
    words = value.astype(np.float32).view(np.uint32) >> 16
    return (words << 16).view(np.float32)


def bf16_bytes(value: np.ndarray) -> bytes:
    words = (value.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    return words.tobytes()


def write_safetensors(path: Path, tensors) -> None:
    header = {}
    data = bytearray()
    for name, value in tensors.items():
        raw = bf16_bytes(value)
        header[name] = {
            "dtype": "BF16",
            "shape": list(value.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    payload = json.dumps(header, sort_keys=True).encode("utf-8")
    payload += b" " * ((8 - len(payload) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + bytes(data))


def fixture(root: Path) -> dict:
    cfg = T.config()
    specs = T.specs()
    tensors = {}
    for index, (name, (_dtype, shape)) in enumerate(sorted(specs.items())):
        size = int(np.prod(shape))
        # small deterministic values, no zeros in the norm tensors
        values = (np.arange(size, dtype=np.float32) % 17 - 8) / 8.0 + index * 0.001
        tensors[name] = bf16(values.reshape(shape).astype(np.float32))
    memory = {n: v for n, v in tensors.items() if ".self_attn.memt." in n}
    backbone = {n: v for n, v in tensors.items() if ".self_attn.memt." not in n}
    write_safetensors(root / "memt_block.safetensors", memory)
    names = sorted(backbone)
    for shard in range(2):
        write_safetensors(
            root / ("model-%05d.safetensors" % shard),
            {n: backbone[n] for n in names[shard::2]},
        )
    weight_map = {}
    for shard in range(2):
        for name in names[shard::2]:
            weight_map[name] = "model-%05d.safetensors" % shard
    for name in memory:
        weight_map[name] = "memt_block.safetensors"
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return tensors


def reader(path: Path) -> SG.GGUFReader:
    return SG.GGUFReader(str(path))


def tensor(read: SG.GGUFReader, name: str) -> np.ndarray:
    for item in read.tensors:
        if item.name == name:
            return np.asarray(item.data)
    raise AssertionError("missing tensor %s" % name)


def field(read: SG.GGUFReader, name: str):
    return read.fields[name].value


def values(read: SG.GGUFReader, name: str) -> np.ndarray:
    item = next(t for t in read.tensors if t.name == name)
    if item.tensor_type == SG.GGMLQuantizationType.Q8_0:
        return SQ.dequantize(item.data, item.tensor_type)
    return np.asarray(item.data, dtype=np.float32)


def tensor_type(read: SG.GGUFReader, name: str):
    return next(t for t in read.tensors if t.name == name).tensor_type


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "checkpoint"
        self.source.mkdir()
        self.tensors = fixture(self.source)
        self.out = self.root / "out.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self):
        return C.convert(self.source, self.out, None, T.VOCAB, False)

    def test_the_snapshot_identity_is_the_checkpoints(self):
        (self.source / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        (self.source / "tokenizer.json").write_text(json.dumps(
            {"model": {"vocab": {"a": 0, "b": 1, "c": 2, "d": 3}}, "added_tokens": []}), encoding="utf-8")
        import memt_engine
        want = memt_engine.identity_from_model_dir(self.source)
        result = self.convert()
        self.assertEqual(result["identity"], want)
        self.assertEqual(field(reader(self.out), "memt.identity"), want)

    def test_no_identity_without_the_template_and_tokenizer(self):
        result = self.convert()
        self.assertIsNone(result["identity"])
        self.assertNotIn("memt.identity", reader(self.out).fields)

    def test_metadata(self):
        self.convert()
        read = reader(self.out)
        self.assertEqual(field(read, "general.architecture"), "qwen35moe")
        self.assertEqual(field(read, "qwen35moe.block_count"), T.LAYERS)
        self.assertEqual(field(read, "qwen35moe.embedding_length"), T.N)
        self.assertEqual(field(read, "qwen35moe.attention.head_count"), T.H)
        self.assertEqual(field(read, "qwen35moe.attention.head_count_kv"), T.KV)
        self.assertEqual(field(read, "qwen35moe.full_attention_interval"), T.INTERVAL)
        self.assertEqual(field(read, "qwen35moe.ssm.state_size"), T.LKD)
        self.assertEqual(field(read, "qwen35moe.ssm.group_count"), T.LK)
        self.assertEqual(field(read, "qwen35moe.ssm.time_step_rank"), T.LV)
        self.assertEqual(field(read, "memt.chunk_tokens"), 4)
        self.assertEqual(field(read, "qwen35moe.rope.dimension_count"),
                         int(T.D * 0.25))
        self.assertAlmostEqual(field(read, "qwen35moe.rope.freq_base"), 1e7)
        self.assertEqual(list(field(read, "memt.memory_layers")), [1, 3])
        self.assertEqual(len(field(read, "tokenizer.ggml.tokens")), T.VOCAB)

    def test_tensor_set(self):
        self.convert()
        read = reader(self.out)
        names = {t.name for t in read.tensors}
        expected = {
            "token_embd.weight", "output.weight", "output_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
            "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp_shexp.weight",
            "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight",
            "blk.0.ffn_down_shexp.weight", "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
            "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
            "blk.0.ssm_conv1d.weight", "blk.0.ssm_alpha.weight",
            "blk.0.ssm_beta.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
            "blk.0.ssm_norm.weight",
            "blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight",
            "blk.1.attn_output.weight", "blk.1.attn_q_norm.weight",
            "blk.1.attn_k_norm.weight",
            "blk.1.memt.in_proj.weight", "blk.1.memt.f_down.weight",
            "blk.1.memt.g_down.weight", "blk.1.memt.decay_scale_log",
            "blk.1.memt.decay_bias", "blk.1.memt.o_norm.weight",
            "blk.2.attn_qkv.weight", "blk.3.attn_q.weight",
        }
        self.assertTrue(expected <= names, sorted(expected - names))
        self.assertNotIn("blk.0.linear_attn.dt_bias", names)
        self.assertNotIn("blk.0.linear_attn.A_log", names)
        self.assertNotIn("blk.0.mlp.shared_expert_gate.weight", names)

    def test_norm_offset_and_a_log(self):
        self.convert()
        read = reader(self.out)
        original = self.tensors["model.language_model.layers.1.input_layernorm.weight"]
        np.testing.assert_allclose(
            tensor(read, "blk.1.attn_norm.weight"),
            original.astype(np.float32) + 1.0, atol=1e-6,
        )
        a_log = self.tensors["model.language_model.layers.0.linear_attn.A_log"]
        k, per_k, dim = T.LK, T.LV // T.LK, 1
        want = C.reorder_v(-np.exp(a_log.astype(np.float32)), 0, k, per_k, dim)
        np.testing.assert_allclose(tensor(read, "blk.0.ssm_a"), want, rtol=1e-6,
                                   atol=1e-6)

    def test_v_head_reorder_hand_permutation(self):
        values = np.arange(8, dtype=np.float32).reshape(8, 1)
        got = C.reorder_v(values, 0, 2, 2, 2).reshape(-1)
        np.testing.assert_array_equal(
            got, np.array([0, 1, 4, 5, 2, 3, 6, 7], dtype=np.float32)
        )

    def test_linear_attention_transforms(self):
        self.convert()
        read = reader(self.out)
        v_heads, k_heads, per_k, v_dim = T.LV, T.LK, T.LV // T.LK, T.LVD
        qkv = self.tensors["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
        qk = k_heads * v_dim
        want = np.concatenate(
            [qkv[:qk], qkv[qk:2 * qk],
             C.reorder_v(qkv[2 * qk:], 0, k_heads, per_k, v_dim)], axis=0)
        np.testing.assert_allclose(values(read, "blk.0.attn_qkv.weight"), want, atol=0.05)
        conv = self.tensors["model.language_model.layers.0.linear_attn.conv1d.weight"]
        want_conv = np.concatenate(
            [conv[:2 * qk].reshape(-1, conv.shape[-1]),
             C.reorder_v(conv[2 * qk:].reshape(v_heads * v_dim, conv.shape[-1]),
                         0, k_heads, per_k, v_dim)], axis=0)
        self.assertEqual(tensor(read, "blk.0.ssm_conv1d.weight").shape,
                         (2 * qk + v_heads * v_dim, conv.shape[-1]))
        np.testing.assert_allclose(values(read, "blk.0.ssm_conv1d.weight"),
                                   want_conv, atol=1e-6)

    def test_in_proj_z_and_ssm_norm(self):
        self.convert()
        read = reader(self.out)
        k, per_k, v_dim = T.LK, T.LV // T.LK, T.LVD
        z = self.tensors["model.language_model.layers.0.linear_attn.in_proj_z.weight"]
        np.testing.assert_allclose(
            values(read, "blk.0.attn_gate.weight"),
            C.reorder_v(z, 0, k, per_k, v_dim), atol=0.1)
        norm = self.tensors["model.language_model.layers.0.linear_attn.norm.weight"]
        # linear_attn.norm is the one norm without the +1 offset
        np.testing.assert_allclose(values(read, "blk.0.ssm_norm.weight"), norm,
                                   atol=1e-6)

    def test_expert_split_and_shared_gate(self):
        self.convert()
        read = reader(self.out)
        gate_up = self.tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]
        half = gate_up.shape[1] // 2
        self.assertEqual(values(read, "blk.0.ffn_gate_exps.weight").shape,
                         (gate_up.shape[0], half, gate_up.shape[2]))
        self.assertEqual(values(read, "blk.0.ffn_up_exps.weight").shape,
                         (gate_up.shape[0], half, gate_up.shape[2]))
        self.assertEqual(tensor(read, "blk.0.ffn_gate_inp_shexp.weight").shape, (T.N,))

    def test_memory_types(self):
        self.convert()
        read = reader(self.out)
        item = next(t for t in read.tensors if t.name == "blk.1.memt.in_proj.weight")
        # the engine loads in_proj with mat() / native_mmvq: Q8_0, not F32
        self.assertEqual(item.tensor_type, SG.GGMLQuantizationType.Q8_0)
        self.assertEqual(tuple(item.shape), (T.N, 5 * T.H * T.D + 2 * T.R))
        gate = next(t for t in read.tensors if t.name == "blk.1.memt.f_down.weight")
        self.assertEqual(gate.tensor_type, SG.GGMLQuantizationType.F32)
        self.assertEqual(gate.data.dtype, np.float32)
        self.assertEqual(tuple(gate.shape), (T.R, T.H * T.D))

    def test_layout_matches_engine_table(self):
        self.convert()
        read = reader(self.out)
        got = {t.name: tuple(int(x) for x in reversed(t.shape)) for t in read.tensors}
        want = {
            "token_embd.weight": (T.VOCAB, T.N),
            "output.weight": (T.VOCAB, T.N),
            "output_norm.weight": (T.N,),
        }
        channels = 2 * T.LK * T.LKD + T.LV * T.LVD
        value_dim = T.LV * T.LVD
        for layer in range(T.LAYERS):
            p = "blk.%d." % layer
            want.update({
                p + "ffn_gate_inp.weight": (T.EXPERTS, T.N),
                p + "ffn_gate_inp_shexp.weight": (T.N,),
                p + "ffn_gate_shexp.weight": (T.SHARED_FF, T.N),
                p + "ffn_up_shexp.weight": (T.SHARED_FF, T.N),
                p + "ffn_down_shexp.weight": (T.N, T.SHARED_FF),
                p + "ffn_gate_exps.weight": (T.EXPERTS, T.MOE_FF, T.N),
                p + "ffn_up_exps.weight": (T.EXPERTS, T.MOE_FF, T.N),
                p + "ffn_down_exps.weight": (T.EXPERTS, T.N, T.MOE_FF),
                p + "attn_norm.weight": (T.N,),
                p + "post_attention_norm.weight": (T.N,),
            })
            if (layer + 1) % T.INTERVAL == 0:
                want.update({
                    p + "attn_q.weight": (T.H * T.D * 2, T.N),
                    p + "attn_k.weight": (T.KV * T.D, T.N),
                    p + "attn_v.weight": (T.KV * T.D, T.N),
                    p + "attn_output.weight": (T.N, T.H * T.D),
                    p + "attn_q_norm.weight": (T.D,),
                    p + "attn_k_norm.weight": (T.D,),
                })
                for suffix in ("in_proj.weight", "f_down.weight", "g_down.weight",
                               "decay_scale_log", "decay_bias", "o_norm.weight"):
                    shape = {
                        "in_proj.weight": (5 * T.H * T.D + 2 * T.R, T.N),
                        "f_down.weight": (T.H * T.D, T.R),
                        "g_down.weight": (T.H * T.D, T.R),
                        "decay_scale_log": (T.H,),
                        "decay_bias": (T.H * T.D,),
                        "o_norm.weight": (T.D,),
                    }[suffix]
                    want[p + "memt." + suffix] = shape
            else:
                want.update({
                    p + "attn_qkv.weight": (channels, T.N),
                    p + "attn_gate.weight": (value_dim, T.N),
                    p + "ssm_out.weight": (T.N, value_dim),
                    p + "ssm_conv1d.weight": (channels, T.CONV),
                    p + "ssm_alpha.weight": (T.LV, T.N),
                    p + "ssm_beta.weight": (T.LV, T.N),
                    p + "ssm_a": (T.LV,),
                    p + "ssm_dt.bias": (T.LV,),
                    p + "ssm_norm.weight": (T.LVD,),
                })
        self.assertEqual(set(got), set(want))
        for name, shape in want.items():
            self.assertEqual(got[name], shape, name)

    def test_engine_types(self):
        self.convert()
        read = reader(self.out)
        q8 = SG.GGMLQuantizationType.Q8_0
        f32 = SG.GGMLQuantizationType.F32
        for name in ("token_embd.weight", "output.weight", "blk.0.attn_qkv.weight",
                     "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                     "blk.0.ffn_gate_exps.weight", "blk.1.attn_q.weight",
                     "blk.1.attn_output.weight", "blk.0.ffn_gate_shexp.weight",
                     "blk.1.memt.in_proj.weight"):
            self.assertEqual(tensor_type(read, name), q8, name)
        for name in ("output_norm.weight", "blk.0.attn_norm.weight",
                     "blk.0.ffn_gate_inp.weight", "blk.0.ssm_alpha.weight",
                     "blk.0.ssm_beta.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
                     "blk.0.ssm_conv1d.weight", "blk.0.ffn_gate_inp_shexp.weight",
                     "blk.1.memt.f_down.weight", "blk.1.memt.g_down.weight",
                     "blk.1.memt.decay_bias", "blk.1.memt.o_norm.weight"):
            self.assertEqual(tensor_type(read, name), f32, name)

    def test_expert_gate_up_orientation(self):
        self.convert()
        read = reader(self.out)
        gate_up = self.tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]
        half = gate_up.shape[1] // 2
        np.testing.assert_allclose(values(read, "blk.0.ffn_gate_exps.weight"),
                                   gate_up[:, :half, :], atol=0.1)
        np.testing.assert_allclose(values(read, "blk.0.ffn_up_exps.weight"),
                                   gate_up[:, half:, :], atol=0.1)

    def test_out_proj_columns_and_alpha_beta(self):
        self.convert()
        read = reader(self.out)
        k, per_k, v_dim = T.LK, T.LV // T.LK, T.LVD
        out_proj = self.tensors["model.language_model.layers.0.linear_attn.out_proj.weight"]
        np.testing.assert_allclose(
            values(read, "blk.0.ssm_out.weight"),
            C.reorder_v(out_proj, 1, k, per_k, v_dim), atol=0.1)
        for source, target in (
            ("linear_attn.in_proj_a.weight", "ssm_alpha.weight"),
            ("linear_attn.in_proj_b.weight", "ssm_beta.weight"),
            ("linear_attn.dt_bias", "ssm_dt.bias"),
        ):
            value = self.tensors["model.language_model.layers.0." + source]
            np.testing.assert_allclose(
                values(read, "blk.0." + target),
                C.reorder_v(value, 0, k, per_k, 1), atol=1e-6)

    def test_qk_split_uses_the_key_width(self):
        # key_dim 1, value_dim 2: q/k are 2 rows, v is 8; the V tile reorders
        out = C.Out("x", (12, 1), np.dtype(np.float16), "v_qkv", "src", 0,
                    (2, 2, 1, 2), None)
        value = np.arange(12, dtype=np.float32).reshape(12, 1)
        got = C.apply(out, value).reshape(-1)
        np.testing.assert_array_equal(
            got, np.array([0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11], dtype=np.float32))

    def test_tokenizer_mismatch(self):
        tokenizer = self.root / "tok.gguf"
        writer = SG.GGUFWriter(str(tokenizer), "qwen35moe")
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_array("tokenizer.ggml.tokens", ["x"] * (T.VOCAB + 1))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        with self.assertRaises(C.ConvertError):
            C.convert(self.source, self.root / "bad_tok.gguf", tokenizer, 0, False)

    def test_synthetic_vocab_must_match(self):
        with self.assertRaises(C.ConvertError):
            C.convert(self.source, self.root / "bad_vocab.gguf", None,
                      T.VOCAB + 1, False)

    def test_donor_with_the_same_size_but_different_ids_is_refused(self):
        self.convert()
        (self.source / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {"<tok0>": 1, "<tok1>": 0}}}), encoding="utf-8")
        (self.source / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        second = self.root / "second.gguf"
        with self.assertRaisesRegex(C.ConvertError, "differs from the checkpoint at token ID"):
            C.convert(self.source, second, self.out, 0, False)
        self.assertFalse(second.exists())

    def test_partial_write_leaves_no_output(self):
        from unittest import mock

        original = C.SQ.quantize
        calls = {"n": 0}

        def fail(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("boom")
            return original(*args, **kwargs)

        with mock.patch.object(C.SQ, "quantize", side_effect=fail):
            with self.assertRaises(RuntimeError):
                C.convert(self.source, self.out, None, T.VOCAB, False)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob(".memt-gguf-*")), [])

    def test_tokenizer_copy(self):
        self.convert()
        config_path = self.source / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["pad_token_id"] = 3
        config_path.write_text(json.dumps(config), encoding="utf-8")
        template = "{{ messages }}"
        (self.source / "chat_template.jinja").write_text(template, encoding="utf-8")
        (self.source / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {"<tok0>": 0, "<tok1>": 1}}}), encoding="utf-8")
        second = self.root / "second.gguf"
        C.convert(self.source, second, self.out, 0, False)
        read = reader(second)
        self.assertEqual(len(field(read, "tokenizer.ggml.tokens")), T.VOCAB)
        self.assertEqual(field(read, "tokenizer.chat_template"), template)
        self.assertEqual(field(read, "tokenizer.ggml.padding_token_id"), 3)
        self.assertEqual(field(read, "memt.identity"), C.memt_engine.identity_from_model_dir(self.source))

    def test_refuses_overwrite_and_bad_config(self):
        self.convert()
        with self.assertRaises(C.ConvertError):
            C.convert(self.source, self.out, None, T.VOCAB, False)
        bad = self.root / "bad"
        bad.mkdir()
        cfg = T.config()
        cfg["model_type"] = "not_memt"
        (bad / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        with self.assertRaises(C.ConvertError):
            C.convert(bad, self.root / "bad.gguf", None, T.VOCAB, False)

    def test_cli(self):
        code = C.main(["--checkpoint", str(self.source), "--out", str(self.out),
                       "--synthetic-vocab", str(T.VOCAB)])
        self.assertEqual(code, 0)
        self.assertTrue(self.out.is_file())


if __name__ == "__main__":
    unittest.main()
