"""tools/memt_to_gguf.py - convert a MemT-1.0-Nano HF checkpoint to a qwen35moe GGUF.

The output is the file Strata's q36 engine reads: the Qwen3.6-35B-A3B backbone
in `qwen35moe` tensor names, the 60 `blk.<i>.memt.*` tensors, the `memt.*`
metadata, and a tokenizer from an existing GGUF using the checkpoint's chat
template and padding token when available. The release geometry is hidden 2048, 40 layers, 16:2 heads, 256
experts top-8, ff 512, full_attention_interval 4, and the memory layers
3, 7, ... 39.

    python tools/memt_to_gguf.py --checkpoint /models/MemT-1.0-Nano \
        --out /models/MemT-1.0-Nano.memt.gguf \
        --tokenizer-from /models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf

    python tools/memt_to_gguf.py --checkpoint DIR --out F.gguf --synthetic-vocab 16   # tests

Backbone matmul tensors are Q8_0 (the q36 engine's native MMVQ path; F16/F32
would be refused at load), norm/gate/router vectors are F32, and so is the memory
block except its `in_proj` matmul, which is Q8_0 for the same reason.  The file is
written in two passes (headers, then tensor data), so the 36 B-parameter
checkpoint is never held in RAM at once.

`memt.identity` is the checkpoint's `memt-memory-v1` snapshot identity (the
codec, chat template and vocabulary hash `memt_snapshot.snapshot_identity`
computes) when the checkpoint directory carries them; the engine reports it as
`INFO memt_identity=` and the server's POST /v1/memt snapshot route uses it.
Without it the file still converts and runs; the route then refuses save/load,
and the line protocol takes an identity per call.

Transforms (mirroring Strata's converter/conversion/qwen.py):
  * every *.norm.weight except linear_attn.norm.weight gets +1;
  * linear_attn.A_log becomes -exp(A_log);
  * linear_attn.dt_bias is renamed ssm_dt.bias;
  * conv1d is squeezed and its V channels are reordered;
  * linear_attn V heads (16 K heads, 32 V heads) go from grouped to tiled order
    in in_proj_qkv (V rows), in_proj_z (rows), in_proj_a/b (rows), out_proj
    (columns), A_log and dt_bias (entries);
  * mlp.experts.gate_up_proj is split into ffn_gate_exps and ffn_up_exps;
  * mlp.shared_expert_gate.weight loses its leading 1.

Exit status 0 on success.  Never overwrites an existing output unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import memt_audit as A  # noqa: E402
import memt_engine  # noqa: E402
import memt_snapshot  # noqa: E402
import strata_gguf as SG  # noqa: E402
import strata_quants as SQ  # noqa: E402

_NP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

_TOP_MAP = {
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "model.language_model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


class ConvertError(Exception):
    pass


@dataclass
class Out:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    kind: str
    source: str
    part: int = 0
    params: Tuple = ()
    raw_dtype: Optional[int] = None

    @property
    def nbytes(self) -> int:
        if self.raw_dtype is not None:
            block, size = SG.GGML_QUANT_SIZES[SG.GGMLQuantizationType(self.raw_dtype)]
            return int(np.prod(self.shape)) // block * size
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize


@dataclass
class Shards:
    root: Path
    names: List[str] = field(default_factory=list)
    weight_map: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, Tuple[Dict, int]] = field(default_factory=dict)

    @classmethod
    def open(cls, root: Path, index_path: Optional[Path] = None) -> "Shards":
        root = Path(root)
        index = index_path or (root / "model.safetensors.index.json")
        self = cls(root)
        if index.is_file():
            with index.open("r", encoding="utf-8") as stream:
                weight_map = json.load(stream).get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                raise ConvertError("%s: empty weight_map" % index)
            self.weight_map = dict(weight_map)
        else:
            for path in sorted(root.glob("*.safetensors")):
                header, _ = _header(path)
                for name in header:
                    self.weight_map[name] = path.name
        if not self.weight_map:
            raise ConvertError("no safetensors index or shards under %s" % root)
        self.names = sorted(self.weight_map)
        for shard in sorted(set(self.weight_map.values())):
            path = root / shard
            if not path.is_file():
                raise ConvertError("missing shard %s" % path)
            self.headers[shard] = _header(path)
        return self

    def shape(self, name: str) -> Tuple[int, ...]:
        shard = self.weight_map[name]
        header = self.headers[shard][0]
        return tuple(int(x) for x in header[name]["shape"])

    def read(self, name: str) -> np.ndarray:
        shard = self.weight_map[name]
        header, start = self.headers[shard]
        info = header[name]
        shape = tuple(int(x) for x in info["shape"])
        begin, _end = (int(x) for x in info["data_offsets"])
        path = self.root / shard
        dtype = info["dtype"]
        if dtype == "BF16":
            raw = np.memmap(path, mode="r", dtype=np.uint16, offset=start + begin,
                            shape=shape)
            return ((raw.astype(np.uint32) << 16).view(np.float32)).copy()
        if dtype not in _NP:
            raise ConvertError("%s: unsupported safetensors dtype %s" % (name, dtype))
        return np.memmap(path, mode="r", dtype=_NP[dtype], offset=start + begin,
                         shape=shape).copy()


def _header(path: Path) -> Tuple[Dict, int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ConvertError("%s: shorter than a safetensors length" % path)
        (n,) = struct.unpack("<Q", raw)
        if n <= 1 or n > 1 << 30:
            raise ConvertError("%s: implausible header size %d" % (path, n))
        header = json.loads(stream.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + n


def reorder_v(value: np.ndarray, dim: int, num_k_heads: int, num_v_per_k: int,
              head_dim: int) -> np.ndarray:
    """Grouped-by-K-head -> tiled V-head order, along one axis."""
    shape = list(value.shape)
    if dim < 0:
        dim += len(shape)
    new = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    value = value.reshape(new)
    value = np.swapaxes(value, dim, dim + 1)
    return np.ascontiguousarray(value).reshape(shape)


def _layer_kind(geometry: A.Geometry, layer: int) -> str:
    return "full_attention" if (layer + 1) % geometry.interval == 0 else "linear_attention"


def _common_layer(name: str, geometry: A.Geometry) -> Optional[str]:
    for suffix in (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.dt_bias",
        "linear_attn.A_log",
        "linear_attn.conv1d.weight",
        "linear_attn.norm.weight",
    ):
        if name.endswith("." + suffix):
            return suffix
    return None


def _out(name, shape, kind, source, dtype=np.float32, *, raw=None, part=0,
         params=()):
    return Out(name, tuple(shape), np.dtype(dtype), kind, source, part, params, raw)


def plan_tensor(hf_name: str, shape: Tuple[int, ...], geometry: A.Geometry,
                memory: A.Memory) -> List[Out]:
    """The GGUF tensors one HF tensor produces, without reading its values."""
    q8 = int(SG.GGMLQuantizationType.Q8_0)
    v_heads = geometry.linear_value_heads
    k_heads = geometry.linear_key_heads
    v_per_k = v_heads // k_heads
    k_dim = geometry.linear_key_dim
    v_dim = geometry.linear_value_dim

    if hf_name in _TOP_MAP:
        target = _TOP_MAP[hf_name]
        if target == "output_norm.weight":
            return [_out(target, shape, "norm_offset", hf_name, np.float32)]
        # Q8_0: is_iq()/embed_type_supported and native_mmvq_supported both take it
        return [_out(target, shape, "copy", hf_name, np.float16, raw=q8)]

    if ".self_attn.memt." in hf_name:
        layer = int(hf_name.split(".")[3])
        suffix = hf_name.split(".self_attn.memt.", 1)[1]
        # in_proj is a 20992 x 2048 matmul: Q8_0, like every other matmul the
        # engine's native MMVQ reads.  f_down / g_down / the gate vectors are
        # F32, small and read by the FP32 GEMM and the norm.
        dtype = (np.float16, q8) if suffix == "in_proj.weight" else (np.float32, None)
        return [_out("blk.%d.memt.%s" % (layer, suffix), shape, "copy", hf_name,
                     dtype[0], raw=dtype[1])]

    if not hf_name.startswith("model.language_model.layers."):
        raise ConvertError("unmapped tensor %s" % hf_name)
    layer = int(hf_name.split(".")[3])
    suffix = _common_layer(hf_name, geometry)
    if suffix is None:
        raise ConvertError("unmapped tensor %s" % hf_name)
    kind = _layer_kind(geometry, layer)
    prefix = "blk.%d." % layer
    f32 = np.float32
    f16 = np.float16

    if suffix == "input_layernorm.weight":
        return [_out(prefix + "attn_norm.weight", shape, "norm_offset", hf_name, f32)]
    if suffix == "post_attention_layernorm.weight":
        return [_out(prefix + "post_attention_norm.weight", shape, "norm_offset",
                     hf_name, f32)]
    if suffix == "mlp.gate.weight":
        # the router stays F32 (a quantized router can flip top-8 near ties)
        return [_out(prefix + "ffn_gate_inp.weight", shape, "copy", hf_name, f32)]
    if suffix == "mlp.shared_expert.gate_proj.weight":
        return [_out(prefix + "ffn_gate_shexp.weight", shape, "copy", hf_name, f16, raw=q8)]
    if suffix == "mlp.shared_expert.up_proj.weight":
        return [_out(prefix + "ffn_up_shexp.weight", shape, "copy", hf_name, f16, raw=q8)]
    if suffix == "mlp.shared_expert.down_proj.weight":
        return [_out(prefix + "ffn_down_shexp.weight", shape, "copy", hf_name, f16, raw=q8)]
    if suffix == "mlp.shared_expert_gate.weight":
        return [_out(prefix + "ffn_gate_inp_shexp.weight", (shape[-1],), "squeeze",
                     hf_name, f32)]
    if suffix == "mlp.experts.gate_up_proj":
        if len(shape) != 3 or shape[1] % 2:
            raise ConvertError("%s: unexpected gate_up shape %r" % (hf_name, shape))
        half = (shape[0], shape[1] // 2, shape[2])
        return [
            _out(prefix + "ffn_gate_exps.weight", half, "gate_up", hf_name, f16,
                 raw=q8, part=0),
            _out(prefix + "ffn_up_exps.weight", half, "gate_up", hf_name, f16,
                 raw=q8, part=1),
        ]
    if suffix == "mlp.experts.down_proj":
        return [_out(prefix + "ffn_down_exps.weight", shape, "copy", hf_name, f16, raw=q8)]

    if kind == "full_attention":
        full = {
            "self_attn.q_proj.weight": ("attn_q.weight", "copy", f16, q8),
            "self_attn.k_proj.weight": ("attn_k.weight", "copy", f16, q8),
            "self_attn.v_proj.weight": ("attn_v.weight", "copy", f16, q8),
            "self_attn.o_proj.weight": ("attn_output.weight", "copy", f16, q8),
            "self_attn.q_norm.weight": ("attn_q_norm.weight", "norm_offset", f32, None),
            "self_attn.k_norm.weight": ("attn_k_norm.weight", "norm_offset", f32, None),
        }
        target, how, dtype, raw = full.get(suffix, (None, None, None, None))
        if target is None:
            raise ConvertError("%s: %s is not a full-attention tensor"
                               % (hf_name, suffix))
        return [_out(prefix + target, shape, how, hf_name, dtype, raw=raw)]

    linear = {
        "linear_attn.in_proj_qkv.weight": (
            "attn_qkv.weight", "v_qkv", f16, q8,
            (k_heads, v_per_k, k_dim, v_dim)),
        "linear_attn.in_proj_z.weight": (
            "attn_gate.weight", "v_rows", f16, q8, (k_heads, v_per_k, v_dim)),
        "linear_attn.out_proj.weight": (
            "ssm_out.weight", "v_cols", f16, q8, (k_heads, v_per_k, v_dim)),
        "linear_attn.in_proj_a.weight": (
            "ssm_alpha.weight", "v_rows", f32, None, (k_heads, v_per_k, 1)),
        "linear_attn.in_proj_b.weight": (
            "ssm_beta.weight", "v_rows", f32, None, (k_heads, v_per_k, 1)),
        "linear_attn.dt_bias": (
            "ssm_dt.bias", "v_rows", f32, None, (k_heads, v_per_k, 1)),
        "linear_attn.A_log": (
            "ssm_a", "a_log", f32, None, (k_heads, v_per_k, 1)),
        "linear_attn.conv1d.weight": (
            "ssm_conv1d.weight", "conv", f32, None,
            (v_heads, k_heads, v_per_k, v_dim, k_dim)),
        "linear_attn.norm.weight": (
            "ssm_norm.weight", "copy", f32, None, ()),
    }
    target, how, dtype, raw, params = linear.get(
        suffix, (None, None, None, None, None))
    if target is None:
        raise ConvertError("%s: %s is not a linear-attention tensor"
                           % (hf_name, suffix))
    out_shape = tuple(shape)
    if how == "conv":
        out_shape = (shape[0], shape[-1])
    return [_out(prefix + target, out_shape, how, hf_name, dtype, raw=raw,
                 params=params)]


def apply(out: Out, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if out.kind == "copy":
        return value
    if out.kind == "norm_offset":
        return value.astype(np.float32) + 1.0
    if out.kind == "squeeze":
        return value.reshape(-1)
    if out.kind == "gate_up":
        half = value.shape[1] // 2
        return value[:, :half] if out.part == 0 else value[:, half:]
    if out.kind == "a_log":
        k_heads, v_per_k, head_dim = out.params
        return reorder_v(-np.exp(value.astype(np.float32)), 0, k_heads, v_per_k, head_dim)
    if out.kind == "v_qkv":
        k_heads, v_per_k, k_dim, v_dim = out.params
        qk = k_heads * k_dim
        q, k, v = value[:qk], value[qk:2 * qk], value[2 * qk:]
        return np.concatenate(
            [q, k, reorder_v(v, 0, k_heads, v_per_k, v_dim)], axis=0
        )
    if out.kind == "v_rows":
        k_heads, v_per_k, head_dim = out.params
        return reorder_v(value, 0, k_heads, v_per_k, head_dim)
    if out.kind == "v_cols":
        k_heads, v_per_k, head_dim = out.params
        return reorder_v(value, 1, k_heads, v_per_k, head_dim)
    if out.kind == "conv":
        v_heads, k_heads, v_per_k, v_dim, k_dim = out.params
        data = value.reshape(value.shape[0], -1)
        qk_channels = 2 * k_heads * k_dim
        qk, v = data[:qk_channels], data[qk_channels:]
        return np.concatenate(
            [qk, reorder_v(v, 0, k_heads, v_per_k, v_dim)], axis=0
        )
    raise ConvertError("unknown transform %s for %s" % (out.kind, out.name))


def build_plans(shards: Shards, geometry: A.Geometry,
                memory: A.Memory) -> List[Out]:
    plans = []
    seen = set()
    for name in shards.names:
        shape = shards.shape(name)
        for out in plan_tensor(name, shape, geometry, memory):
            if out.name in seen:
                raise ConvertError("duplicate GGUF tensor %s" % out.name)
            seen.add(out.name)
            plans.append(out)
    return plans


def add_metadata(writer: SG.GGUFWriter, geometry: A.Geometry,
                 memory: A.Memory, identity: Optional[str] = None) -> None:
    g = geometry
    values = [
        ("qwen35moe.block_count", g.layers, "u32"),
        ("qwen35moe.embedding_length", g.hidden, "u32"),
        ("qwen35moe.attention.head_count", g.heads, "u32"),
        ("qwen35moe.attention.head_count_kv", g.kv_heads, "u32"),
        ("qwen35moe.attention.key_length", g.head_dim, "u32"),
        ("qwen35moe.attention.value_length", g.head_dim, "u32"),
        ("qwen35moe.attention.layer_norm_rms_epsilon", g.eps, "f32"),
        ("qwen35moe.expert_count", g.experts, "u32"),
        ("qwen35moe.expert_used_count", g.experts_per_tok, "u32"),
        ("qwen35moe.expert_feed_forward_length", g.moe_ff, "u32"),
        ("qwen35moe.expert_shared_feed_forward_length", g.shared_ff, "u32"),
        ("qwen35moe.feed_forward_length", 0, "u32"),
        ("qwen35moe.ssm.conv_kernel", g.conv_kernel, "u32"),
        ("qwen35moe.ssm.state_size", g.linear_key_dim, "u32"),
        ("qwen35moe.ssm.group_count", g.linear_key_heads, "u32"),
        ("qwen35moe.ssm.time_step_rank", g.linear_value_heads, "u32"),
        ("qwen35moe.ssm.inner_size", g.linear_value_heads * g.linear_value_dim, "u32"),
        ("qwen35moe.full_attention_interval", g.interval, "u32"),
        ("qwen35moe.rope.dimension_count", g.rope_dim, "u32"),
        ("qwen35moe.rope.freq_base", g.rope_theta, "f32"),
        ("memt.version", 1, "u32"),
        ("memt.chunk_tokens", memory.chunk_tokens, "u32"),
        ("memt.full_attention_interval", g.interval, "u32"),
        ("memt.head_count", memory.heads, "u32"),
        ("memt.head_dim", memory.head_dim, "u32"),
        ("memt.gate_rank", memory.gate_rank, "u32"),
        ("memt.output_gate_softcap_limit", memory.softcap, "f32"),
        ("memt.rms_norm_epsilon", memory.eps, "f32"),
    ]
    for key, value, kind in values:
        if kind == "u32":
            writer.add_uint32(key, int(value))
        elif kind == "f32":
            writer.add_float32(key, float(value))
        else:
            raise ConvertError("bad metadata kind %s" % kind)
    if not memory.layers:
        raise ConvertError("no memory layers")
    writer.add_array("memt.memory_layers", [int(x) for x in memory.layers])
    if identity:
        writer.add_string("memt.identity", identity)
    writer.add_string("general.name", "MemT-1.0-Nano")


def add_tokenizer(writer: SG.GGUFWriter, tokenizer_from: Optional[Path],
                  synthetic_vocab: int, vocab: int, *,
                  chat_template: Optional[str] = None, pad_token_id: Optional[int] = None,
                  checkpoint_vocab: Optional[Dict[str, int]] = None) -> None:
    if tokenizer_from is not None:
        reader = SG.GGUFReader(os.fspath(tokenizer_from))
        fields = {key: value for key, value in reader.fields.items()
                  if key.startswith("tokenizer.")}
        if "tokenizer.ggml.tokens" not in fields:
            raise ConvertError("%s carries no tokenizer.ggml.tokens" % tokenizer_from)
        tokens = list(fields["tokenizer.ggml.tokens"].value)
        if len(tokens) != vocab:
            raise ConvertError(
                "%s has %d tokens, the checkpoint's vocab is %d"
                % (tokenizer_from, len(tokens), vocab)
            )
        if checkpoint_vocab is not None:
            for token, idx in checkpoint_vocab.items():
                if not 0 <= idx < len(tokens) or tokens[idx] != token:
                    raise ConvertError("%s: tokenizer differs from the checkpoint at token ID %d" % (tokenizer_from, idx))
            for idx, token in enumerate(tokens):
                if checkpoint_vocab.get(token, idx) != idx:
                    raise ConvertError("%s: checkpoint token is duplicated at token ID %d" % (tokenizer_from, idx))
        for name in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
            if name in fields and len(list(fields[name].value)) != len(tokens):
                raise ConvertError("%s: %s does not match the token count"
                                   % (tokenizer_from, name))
        if chat_template:
            fields.pop("tokenizer.chat_template", None)
            writer.add_string("tokenizer.chat_template", chat_template)
        if pad_token_id is not None:
            fields.pop("tokenizer.ggml.padding_token_id", None)
            writer.add_uint32("tokenizer.ggml.padding_token_id", pad_token_id)
        for key, value in fields.items():
            writer.add_key_value(key, value.value, value.types[0],
                                 None if len(value.types) < 2 else value.types[1])
        return
    if synthetic_vocab <= 0:
        raise ConvertError("pass --tokenizer-from, or --synthetic-vocab N for a test file")
    if synthetic_vocab != vocab:
        raise ConvertError("--synthetic-vocab %d does not match the checkpoint's vocab %d"
                           % (synthetic_vocab, vocab))
    tokens = ["<tok%d>" % i for i in range(synthetic_vocab)]
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", [0.0] * synthetic_vocab)
    writer.add_array("tokenizer.ggml.token_type", [1] * synthetic_vocab)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 0)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 0)


def convert(checkpoint: Path, out_path: Path, tokenizer_from: Optional[Path],
            synthetic_vocab: int, force: bool) -> Dict:
    checkpoint = Path(checkpoint)
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise ConvertError("%s exists; pass --force to replace it" % out_path)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ConvertError("%s: no config.json" % checkpoint)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    report = A.Report()
    geometry, memory = A.parse_config(config, report)
    if not report.ok:
        raise ConvertError("checkpoint config rejected:\n  " + "\n  ".join(report.errors))

    shards = Shards.open(checkpoint)
    expected = A.base_tensor_specs(geometry, memory)
    missing = sorted(set(expected) - set(shards.names))
    unknown = sorted(set(shards.names) - set(expected))
    if missing or unknown:
        raise ConvertError(
            "checkpoint tensor set differs: missing %r, unknown %r"
            % (missing[:4], unknown[:4])
        )
    for name, (_dtype, shape) in expected.items():
        if shards.shape(name) != tuple(shape):
            raise ConvertError("%s: shape %r, expected %r"
                               % (name, shards.shape(name), tuple(shape)))
    plans = build_plans(shards, geometry, memory)

    chat_template = memt_engine.checkpoint_chat_template(checkpoint)
    checkpoint_vocab = None
    if (checkpoint / "tokenizer.json").is_file():
        try:
            checkpoint_vocab = memt_engine.checkpoint_vocabulary(checkpoint)
        except (OSError, ValueError, RuntimeError, KeyError, AttributeError, TypeError) as exc:
            raise ConvertError("cannot validate the checkpoint tokenizer: %s" % exc) from exc
        if not checkpoint_vocab:
            raise ConvertError("the checkpoint tokenizer has no vocabulary")

    identity = None
    if chat_template and checkpoint_vocab:
        identity = memt_snapshot.snapshot_identity(config, {"chat_template": chat_template, "vocab": checkpoint_vocab})
    else:
        print("memt_to_gguf: memt.identity not written: the checkpoint needs a chat template and tokenizer.json",
              file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=out_path.parent, prefix=".memt-gguf-")
    os.close(descriptor)
    os.unlink(temporary)
    writer = None
    try:
        writer = SG.GGUFWriter(temporary, "qwen35moe")
        add_metadata(writer, geometry, memory, identity)
        add_tokenizer(writer, tokenizer_from, synthetic_vocab, geometry.vocab,
                      chat_template=chat_template, pad_token_id=config.get("pad_token_id"),
                      checkpoint_vocab=checkpoint_vocab)
        for out in plans:
            if out.raw_dtype is not None:
                quant = SG.GGMLQuantizationType(out.raw_dtype)
                byte_shape = SG.quant_shape_to_byte_shape(out.shape, quant)
                writer.add_tensor_info(out.name, byte_shape, np.uint8, out.nbytes,
                                       raw_dtype=out.raw_dtype)
            else:
                writer.add_tensor_info(out.name, out.shape, out.dtype, out.nbytes)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writer._pad(writer.fout.tell())

        written = 0
        for out in plans:
            value = apply(out, shards.read(out.source))
            if tuple(value.shape) != out.shape:
                raise ConvertError(
                    "%s: transform gave %r, expected %r"
                    % (out.source, value.shape, out.shape)
                )
            if out.raw_dtype is not None:
                quant = SG.GGMLQuantizationType(out.raw_dtype)
                data = SQ.quantize(np.ascontiguousarray(value, dtype=np.float32), quant)
                data = np.ascontiguousarray(data).tobytes(order="C")
            else:
                data = np.ascontiguousarray(value.astype(out.dtype, copy=False)).tobytes(order="C")
            if len(data) != out.nbytes:
                raise ConvertError("%s: %d bytes, expected %d"
                                   % (out.name, len(data), out.nbytes))
            writer.fout.write(data)
            writer._pad(len(data))
            written += 1
        writer.fout.flush()
        os.fsync(writer.fout.fileno())
        writer.fout.close()
        writer = None
        os.replace(temporary, out_path)
    except BaseException:
        if writer is not None and writer.fout is not None:
            try:
                writer.fout.close()
            except OSError:
                pass
        Path(temporary).unlink(missing_ok=True)
        raise
    return {
        "out": os.fspath(out_path),
        "tensors": written,
        "bytes": out_path.stat().st_size,
        "memory_layers": list(memory.layers),
        "backbone_dtype": "Q8_0",
        "memory_dtype": "Q8_0 in_proj + F32 gates",
        "identity": identity,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-from", type=Path,
                        help="an existing qwen35moe GGUF whose tokenizer.* fields are copied")
    parser.add_argument("--synthetic-vocab", type=int, default=0,
                        help="write N placeholder tokens (tests only)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = convert(args.checkpoint, args.out, args.tokenizer_from,
                         args.synthetic_vocab, args.force)
    except (ConvertError, SQ.QuantError, OSError, ValueError, json.JSONDecodeError) as exc:
        print("memt_to_gguf: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
