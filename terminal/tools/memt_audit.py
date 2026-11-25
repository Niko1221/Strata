"""tools/memt_audit.py - read-only audit of a MemT-1.0-Nano HF checkpoint.

It reads `config.json`, the safetensors index and (when the shards are present)
only the safetensors headers, never the weights.  It reports:

  * the memory layers and the exact `blk.<i>.memt.*` GGUF tensors to write;
  * the HF -> `qwen35moe` tensor mapping, missing and unknown tensors;
  * dtype/shape mismatches before a single weight is loaded.

    python tools/memt_audit.py /models/MemT-1.0-Nano [--json] [--out FILE]
    python tools/memt_audit.py --config cfg.json --index index.json      # headers absent
    python tools/memt_audit.py --mapping                               # HF -> GGUF + transforms

Exit status: 0 when the audit has no errors, 1 otherwise.  Warnings do not fail.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MEMT_FIXED = {
    "memory_update_mode": "token",
    "gdn_version": "stable_gdn2",
    "memory_use_short_conv": False,
    "memory_head_mode": "mha",
    "memory_read_scale": "k_inv_sqrt",
    "linear_attention_state_policy": "reset_per_chunk",
    "tail_policy": "commit",
    "output_gate_type": "silu",
    "output_gate_bias": False,
    "output_gate_softcap_enabled": True,
    "stable_gdn2_sqrt_erase_gate": True,
    "backbone_gdn_override": False,
}

TEXT_FIXED = {
    "model_type": "qwen3_5_moe_text",
    "dtype": "bfloat16",
    "quantization_config": None,
}

MEMORY_SUFFIXES = (
    "in_proj.weight",
    "f_down.weight",
    "g_down.weight",
    "decay_scale_log",
    "decay_bias",
    "o_norm.weight",
)


class AuditError(Exception):
    pass


_TOP_MAP = {
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "model.language_model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

_LAYER_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "mlp.experts.gate_up_proj": ("ffn_gate_exps.weight", "ffn_up_exps.weight"),
    "mlp.experts.down_proj": "ffn_down_exps.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
}

@dataclass
class Geometry:
    hidden: int
    heads: int
    head_dim: int
    kv_heads: int
    layers: int
    interval: int
    experts: int
    experts_per_tok: int
    moe_ff: int
    shared_ff: int
    linear_key_heads: int
    linear_value_heads: int
    linear_key_dim: int
    linear_value_dim: int
    conv_kernel: int
    vocab: int
    eps: float
    output_gate: bool
    rope_dim: int
    rope_theta: float

    @property
    def q_rows(self) -> int:
        return self.heads * self.head_dim * (2 if self.output_gate else 1)

    @property
    def kv_rows(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def linear_channels(self) -> int:
        return (
            2 * self.linear_key_heads * self.linear_key_dim
            + self.linear_value_heads * self.linear_value_dim
        )


@dataclass
class Memory:
    heads: int
    head_dim: int
    gate_rank: int
    chunk_tokens: int
    softcap: float
    eps: float
    layers: List[int] = field(default_factory=list)


@dataclass
class Report:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    config: Dict = field(default_factory=dict)
    geometry: Optional[Dict] = None
    memory: Optional[Dict] = None
    gguf_tensors: Dict = field(default_factory=dict)
    counts: Dict = field(default_factory=dict)

    def error(self, text: str) -> None:
        self.ok = False
        self.errors.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)


def _is_int(value) -> bool:
    return type(value) is int


def _get(mapping, key, where, report, default=None):
    if not isinstance(mapping, dict) or key not in mapping:
        report.error("%s is missing" % where)
        return default
    return mapping[key]


def _check_fixed(mapping, fixed, section, report, *, exact=True, variable=()) -> None:
    """Mirror memt_sglang's validate_fixed_fields: exact rejects unknown fields."""
    if not isinstance(mapping, dict):
        report.error("%s must be an object" % section)
        return
    allowed = set(fixed) | set(variable)
    if exact:
        for key in sorted(allowed - set(mapping)):
            report.error("%s.%s is missing" % (section, key))
        for key in sorted(set(mapping) - allowed):
            report.error("%s.%s is not a supported field" % (section, key))
    else:
        for key in sorted(variable):
            if key not in mapping:
                report.error("%s.%s is missing" % (section, key))
    for key, want in fixed.items():
        if key not in mapping:
            if want is not None and not exact:
                report.error("%s.%s is missing" % (section, key))
            continue
        got = mapping[key]
        if got != want or type(got) is not type(want):
            report.error(
                "%s.%s must be %r, got %r" % (section, key, want, got)
            )


def parse_config(config: Dict, report: Report) -> Tuple[Geometry, Memory]:
    if not isinstance(config, dict):
        raise AuditError("config.json must be an object")
    model_type = config.get("model_type")
    if model_type != "memt_qwen3_5_moe":
        report.error("config.model_type must be 'memt_qwen3_5_moe', got %r" % model_type)
    arch = config.get("architectures")
    if arch != ["MemTQwen3_5MoeForCausalLM"]:
        report.error(
            "config.architectures must be ['MemTQwen3_5MoeForCausalLM'], got %r" % arch
        )
    if config.get("auto_map"):
        report.warn("auto_map is present; the engine ignores checkpoint Python code")
    if "memt_checkpoint_id" in config:
        value = config["memt_checkpoint_id"]
        if not isinstance(value, str) or not value.strip():
            report.error("config.memt_checkpoint_id must be a nonempty string")

    backbone = _get(config, "backbone_configs", "config", report, {})
    if not isinstance(backbone, dict):
        report.error("config.backbone_configs must be an object")
        backbone = {}
    if backbone.get("model_type") != "qwen3_5_moe":
        report.error(
            "config.backbone_configs.model_type must be 'qwen3_5_moe', got %r"
            % backbone.get("model_type")
        )
    text = backbone.get("text_config")
    if not isinstance(text, dict):
        report.error("config.backbone_configs.text_config must be an object")
        text = {}
    _check_fixed(
        text, TEXT_FIXED, "backbone_configs.text_config", report, exact=False,
        variable=(
            "hidden_size",
            "num_attention_heads",
            "head_dim",
            "num_hidden_layers",
            "full_attention_interval",
            "rms_norm_eps",
        ),
    )

    def need_int(source, key, where, low=1):
        value = _get(source, key, where, report)
        if value is None:
            return low
        if not _is_int(value) or value < low:
            report.error("%s must be an integer >= %d, got %r" % (where, low, value))
            return low
        return value

    hidden = need_int(text, "hidden_size", "text_config.hidden_size")
    heads = need_int(text, "num_attention_heads", "text_config.num_attention_heads")
    head_dim = need_int(text, "head_dim", "text_config.head_dim")
    kv_heads = need_int(
        text, "num_key_value_heads", "text_config.num_key_value_heads"
    )
    layers = need_int(text, "num_hidden_layers", "text_config.num_hidden_layers")
    interval = need_int(
        text, "full_attention_interval", "text_config.full_attention_interval"
    )
    experts = need_int(text, "num_experts", "text_config.num_experts")
    top_k = need_int(
        text, "num_experts_per_tok", "text_config.num_experts_per_tok"
    )
    moe_ff = need_int(
        text, "moe_intermediate_size", "text_config.moe_intermediate_size"
    )
    shared_ff = need_int(
        text, "shared_expert_intermediate_size",
        "text_config.shared_expert_intermediate_size",
    )
    lk_heads = need_int(
        text, "linear_num_key_heads", "text_config.linear_num_key_heads"
    )
    lv_heads = need_int(
        text, "linear_num_value_heads", "text_config.linear_num_value_heads"
    )
    lk_dim = need_int(
        text, "linear_key_head_dim", "text_config.linear_key_head_dim"
    )
    lv_dim = need_int(
        text, "linear_value_head_dim", "text_config.linear_value_head_dim"
    )
    conv = need_int(
        text, "linear_conv_kernel_dim", "text_config.linear_conv_kernel_dim"
    )
    vocab = need_int(text, "vocab_size", "text_config.vocab_size")
    eps = text.get("rms_norm_eps")
    if not isinstance(eps, (int, float)) or eps <= 0:
        report.error("text_config.rms_norm_eps must be positive, got %r" % (eps,))
        eps = 1e-6
    output_gate = text.get("attn_output_gate") is True
    rope = text.get("rope_parameters", {})
    if not isinstance(rope, dict):
        report.error("text_config.rope_parameters must be an object")
        rope = {}
    partial = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
    if not isinstance(partial, (int, float)) or not 0 < partial <= 1:
        report.error("text_config.partial_rotary_factor must be in (0, 1], got %r"
                     % (partial,))
        partial = 0.25
    rope_theta = rope.get("rope_theta", text.get("rope_theta", 1e7))
    if not isinstance(rope_theta, (int, float)) or rope_theta <= 0:
        report.error("text_config.rope_theta must be positive, got %r" % (rope_theta,))
        rope_theta = 1e7

    layer_types = text.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or len(layer_types) != layers:
            report.error(
                "text_config.layer_types must list %d entries, got %r"
                % (layers, len(layer_types) if isinstance(layer_types, list) else layer_types)
            )
        else:
            full = [
                i for i, kind in enumerate(layer_types)
                if kind == "full_attention"
            ]
            expected = [i for i in range(layers) if (i + 1) % interval == 0]
            if full != expected:
                # MemT itself derives the memory layers from the interval and
                # never compares layer_types; the converter follows the
                # interval too, so a contradiction is a warning, not a refusal.
                report.warn(
                    "text_config.layer_types full_attention layers %r do not match "
                    "(index + 1) %% full_attention_interval: %r; the converter "
                    "follows the interval" % (full, expected)
                )

    memt = _get(config, "memt", "config", report, {})
    _check_fixed(
        memt, MEMT_FIXED, "memt", report, exact=True,
        variable=(
            "num_memory_heads",
            "num_kv_heads",
            "chunk_tokens",
            "output_gate_softcap_limit",
        ),
    )
    mem_heads = need_int(memt, "num_memory_heads", "memt.num_memory_heads")
    mem_kv = need_int(memt, "num_kv_heads", "memt.num_kv_heads")
    if mem_heads != heads:
        report.error(
            "memt.num_memory_heads (%d) must equal text_config.num_attention_heads (%d)"
            % (mem_heads, heads)
        )
    if mem_kv != heads:
        report.error(
            "memt.num_kv_heads (%d) must equal text_config.num_attention_heads (%d)"
            % (mem_kv, heads)
        )
    chunk = need_int(memt, "chunk_tokens", "memt.chunk_tokens")
    softcap = memt.get("output_gate_softcap_limit")
    if not isinstance(softcap, (int, float)) or softcap <= 0:
        report.error(
            "memt.output_gate_softcap_limit must be positive, got %r" % (softcap,)
        )
        softcap = 1.0
    # The released kernels pin gate_rank to head_dim; a config field would be
    # rejected by the exact `memt` validation above.
    gate_rank = head_dim

    memory_layers = [i for i in range(layers) if (i + 1) % interval == 0]
    if not memory_layers:
        report.error("full_attention_interval selects no memory layer")

    mtp = text.get("mtp_num_hidden_layers")
    if mtp not in (0, None):
        report.warn(
            "text_config.mtp_num_hidden_layers=%r; MemT serving forces 0 and this "
            "checkpoint has no MTP tensors -- do not enable MTP for parity" % (mtp,)
        )

    geometry = Geometry(
        hidden=hidden,
        heads=heads,
        head_dim=head_dim,
        kv_heads=kv_heads,
        layers=layers,
        interval=interval,
        experts=experts,
        experts_per_tok=top_k,
        moe_ff=moe_ff,
        shared_ff=shared_ff,
        linear_key_heads=lk_heads,
        linear_value_heads=lv_heads,
        linear_key_dim=lk_dim,
        linear_value_dim=lv_dim,
        conv_kernel=conv,
        vocab=vocab,
        eps=float(eps),
        output_gate=output_gate,
        rope_dim=int(head_dim * partial),
        rope_theta=float(rope_theta),
    )
    memory = Memory(
        heads=mem_heads,
        head_dim=head_dim,
        gate_rank=gate_rank,
        chunk_tokens=chunk,
        softcap=float(softcap),
        eps=float(eps),
        layers=memory_layers,
    )
    return geometry, memory


def memory_tensor_specs_for(
    geometry: Geometry, memory: Memory
) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    h, d, r = memory.heads, memory.head_dim, memory.gate_rank
    n = geometry.hidden
    specs = {}
    for layer in memory.layers:
        prefix = "blk.%d.memt." % layer
        specs[prefix + "in_proj.weight"] = ("BF16", (3 * h * d + 2 * h * d + 2 * r, n))
        specs[prefix + "f_down.weight"] = ("BF16", (h * d, r))
        specs[prefix + "g_down.weight"] = ("BF16", (h * d, r))
        specs[prefix + "decay_scale_log"] = ("BF16", (h,))
        specs[prefix + "decay_bias"] = ("BF16", (h * d,))
        specs[prefix + "o_norm.weight"] = ("BF16", (d,))
    return specs


def base_tensor_specs(
    geometry: Geometry, memory: Memory
) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """HF name -> (dtype, shape) for every tensor the checkpoint must hold."""
    g = geometry
    n = g.hidden
    specs = {}
    for name, _target in _TOP_MAP.items():
        if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
            specs[name] = ("BF16", (g.vocab, n))
        else:
            specs[name] = ("BF16", (n,))
    for layer in range(g.layers):
        p = "model.language_model.layers.%d." % layer
        specs[p + "input_layernorm.weight"] = ("BF16", (n,))
        specs[p + "post_attention_layernorm.weight"] = ("BF16", (n,))
        specs[p + "mlp.gate.weight"] = ("BF16", (g.experts, n))
        specs[p + "mlp.shared_expert.gate_proj.weight"] = ("BF16", (g.shared_ff, n))
        specs[p + "mlp.shared_expert.up_proj.weight"] = ("BF16", (g.shared_ff, n))
        specs[p + "mlp.shared_expert.down_proj.weight"] = ("BF16", (n, g.shared_ff))
        specs[p + "mlp.shared_expert_gate.weight"] = ("BF16", (1, n))
        specs[p + "mlp.experts.gate_up_proj"] = ("BF16", (g.experts, 2 * g.moe_ff, n))
        specs[p + "mlp.experts.down_proj"] = ("BF16", (g.experts, n, g.moe_ff))
        full = (layer + 1) % g.interval == 0
        if full:
            specs[p + "self_attn.q_proj.weight"] = ("BF16", (g.q_rows, n))
            specs[p + "self_attn.k_proj.weight"] = ("BF16", (g.kv_rows, n))
            specs[p + "self_attn.v_proj.weight"] = ("BF16", (g.kv_rows, n))
            specs[p + "self_attn.o_proj.weight"] = ("BF16", (n, g.heads * g.head_dim))
            specs[p + "self_attn.q_norm.weight"] = ("BF16", (g.head_dim,))
            specs[p + "self_attn.k_norm.weight"] = ("BF16", (g.head_dim,))
            for suffix in MEMORY_SUFFIXES:
                specs[p + "self_attn.memt." + suffix] = ("BF16", (1,))
        else:
            specs[p + "linear_attn.in_proj_qkv.weight"] = ("BF16", (g.linear_channels, n))
            specs[p + "linear_attn.in_proj_z.weight"] = (
                "BF16", (g.linear_value_heads * g.linear_value_dim, n)
            )
            specs[p + "linear_attn.out_proj.weight"] = (
                "BF16", (n, g.linear_value_heads * g.linear_value_dim)
            )
            specs[p + "linear_attn.in_proj_a.weight"] = ("BF16", (g.linear_value_heads, n))
            specs[p + "linear_attn.in_proj_b.weight"] = ("BF16", (g.linear_value_heads, n))
            specs[p + "linear_attn.dt_bias"] = ("BF16", (g.linear_value_heads,))
            specs[p + "linear_attn.A_log"] = ("BF16", (g.linear_value_heads,))
            specs[p + "linear_attn.conv1d.weight"] = (
                "BF16", (g.linear_channels, 1, g.conv_kernel)
            )
            specs[p + "linear_attn.norm.weight"] = ("BF16", (g.linear_value_dim,))
    # the memory shapes, replacing the (1,) placeholders above
    memt = memory_tensor_specs_for(geometry, memory)
    for layer in memory.layers:
        p = "model.language_model.layers.%d.self_attn.memt." % layer
        for suffix in MEMORY_SUFFIXES:
            target = "blk.%d.memt.%s" % (layer, suffix)
            specs[p + suffix] = memt[target]
    return specs


def read_safetensors_header(path: Path) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """Read the JSON header only; never maps the tensor data."""
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise AuditError("%s: shorter than a safetensors length" % path)
        (size,) = struct.unpack("<Q", raw)
        if size <= 1 or size > 1 << 30:
            raise AuditError("%s: implausible safetensors header size %d" % (path, size))
        payload = stream.read(size)
    if len(payload) != size:
        raise AuditError("%s: truncated safetensors header" % path)
    try:
        header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AuditError("%s: invalid safetensors header: %s" % (path, exc))
    out = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        try:
            out[name] = (str(value["dtype"]), tuple(int(x) for x in value["shape"]))
        except (KeyError, TypeError, ValueError):
            raise AuditError("%s: bad entry for %r" % (path, name))
    return out


def load_checkpoint(config_path: Optional[Path], index_path: Optional[Path],
                    root: Optional[Path], report: Report):
    if config_path is None and root is not None:
        config_path = root / "config.json"
    if index_path is None and root is not None:
        index_path = root / "model.safetensors.index.json"
    if config_path is None or not config_path.is_file():
        raise AuditError("config.json not found (pass --config or a checkpoint dir)")
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    weight_map = {}
    shards = []
    shard_root = root or (index_path.parent if index_path is not None else config_path.parent)
    if index_path is not None and index_path.is_file():
        with index_path.open("r", encoding="utf-8") as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise AuditError("%s: weight_map is empty" % index_path)
        shards = sorted(set(weight_map.values()))
    else:
        report.warn(
            "no model.safetensors.index.json; falling back to *.safetensors headers"
        )
        for path in sorted(shard_root.glob("*.safetensors")):
            shards.append(path.name)
            for name in read_safetensors_header(path):
                weight_map[name] = path.name
        if not weight_map:
            raise AuditError("no safetensors index or shards found")

    headers = {}
    absent = [shard for shard in shards if not (shard_root / shard).is_file()]
    if absent:
        report.warn(
            "%d of %d shards are not present; checking names only (%s)"
            % (len(absent), len(shards), ", ".join(absent[:3]))
        )
    for shard in shards:
        path = shard_root / shard
        if not path.is_file():
            continue
        headers.update(read_safetensors_header(path))
    if headers and set(headers) != set(weight_map):
        missing = sorted(set(weight_map) - set(headers))[:4]
        extra = sorted(set(headers) - set(weight_map))[:4]
        report.warn(
            "index and shard headers disagree (missing %r, extra %r)" % (missing, extra)
        )
    return config, weight_map, headers, shards


def audit(config: Dict, weight_map: Dict[str, str], headers: Dict,
          report: Report) -> None:
    geometry, memory = parse_config(config, report)
    report.geometry = {
        "hidden_size": geometry.hidden,
        "num_attention_heads": geometry.heads,
        "head_dim": geometry.head_dim,
        "num_key_value_heads": geometry.kv_heads,
        "num_hidden_layers": geometry.layers,
        "full_attention_interval": geometry.interval,
        "num_experts": geometry.experts,
        "num_experts_per_tok": geometry.experts_per_tok,
        "moe_intermediate_size": geometry.moe_ff,
        "shared_expert_intermediate_size": geometry.shared_ff,
        "linear_key_heads": geometry.linear_key_heads,
        "linear_value_heads": geometry.linear_value_heads,
        "linear_key_head_dim": geometry.linear_key_dim,
        "linear_value_head_dim": geometry.linear_value_dim,
        "linear_conv_kernel_dim": geometry.conv_kernel,
        "vocab_size": geometry.vocab,
        "rms_norm_eps": geometry.eps,
        "attn_output_gate": geometry.output_gate,
        "rope_dim": geometry.rope_dim,
        "rope_theta": geometry.rope_theta,
    }
    report.memory = {
        "layers": memory.layers,
        "heads": memory.heads,
        "head_dim": memory.head_dim,
        "gate_rank": memory.gate_rank,
        "chunk_tokens": memory.chunk_tokens,
        "state_shape": [1, memory.heads, memory.head_dim, memory.head_dim],
        "state_bytes_per_layer": memory.heads * memory.head_dim ** 2 * 4,
        "output_gate_softcap_limit": memory.softcap,
        "rms_norm_eps": memory.eps,
    }

    specs = base_tensor_specs(geometry, memory)
    mguf = memory_tensor_specs_for(geometry, memory)
    report.gguf_tensors = {
        target: {
            "dtype": dtype,
            # HF/NumPy order [out, in] and GGUF order [in, out] (dim 0 varies fastest)
            "shape_hf": list(shape),
            "shape_gguf": list(reversed(shape)),
        }
        for target, (dtype, shape) in mguf.items()
    }

    present = set(weight_map)
    for name in sorted(specs):
        if name not in present:
            report.error("missing tensor: %s" % name)
    for name in sorted(present - set(specs)):
        report.error("unknown tensor: %s" % name)

    for name, (want_dtype, want_shape) in sorted(specs.items()):
        if name not in headers:
            continue
        got_dtype, got_shape = headers[name]
        if got_dtype != want_dtype:
            report.error(
                "%s: dtype %s, expected %s" % (name, got_dtype, want_dtype)
            )
        if tuple(got_shape) != tuple(want_shape):
            report.error(
                "%s: shape %r, expected %r" % (name, got_shape, want_shape)
            )

    memory_names = [n for n in present if ".self_attn.memt." in n]
    if len(memory_names) != 6 * len(memory.layers):
        report.error(
            "expected %d memory tensors, found %d"
            % (6 * len(memory.layers), len(memory_names))
        )

    extra_per_layer = sum(len(target) - 1 for target in _LAYER_MAP.values() if isinstance(target, tuple))
    report.counts = {
        "tensors_in_index": len(present),
        "tensors_in_headers": len(headers),
        "shards": len(set(weight_map.values())),
        "memory_tensors": len(memory_names),
        "gguf_backbone_tensors": len(specs) - len(mguf) + geometry.layers * extra_per_layer,
        "gguf_memt_tensors": len(mguf),
    }


# HF-side transforms the converter must apply.  The GGUF names are unchanged;
# writing the HF tensor verbatim gives a wrong (or silently mis-broadcast) GGUF.
TRANSFORMS = {
    "linear_attn.in_proj_qkv.weight":
        "reorder the V rows from grouped-by-K-head to tiled order "
        "(num_k_heads, num_v_heads/num_k_heads, head_v_dim)",
    "linear_attn.in_proj_z.weight":
        "reorder the rows from grouped-by-K-head to tiled order (V heads)",
    "linear_attn.in_proj_a.weight":
        "reorder the rows from grouped-by-K-head to tiled order (V heads, head_dim 1)",
    "linear_attn.in_proj_b.weight":
        "reorder the rows from grouped-by-K-head to tiled order (V heads, head_dim 1)",
    "linear_attn.out_proj.weight":
        "reorder the V columns (input dimension) from grouped to tiled order",
    "linear_attn.A_log":
        "write -exp(A_log) (the kernels expect the negative decay), then reorder "
        "the V entries from grouped to tiled order",
    "linear_attn.dt_bias":
        "reorder the V entries from grouped to tiled order; the GGUF name is "
        "ssm_dt.bias",
    "linear_attn.conv1d.weight":
        "squeeze [channels, 1, kernel] to [channels, kernel]; reorder the V "
        "channel part from grouped to tiled order",
    "mlp.experts.gate_up_proj":
        "split dim -2 into the gate half and the up half; the GGUF names are "
        "ffn_gate_exps.weight and ffn_up_exps.weight",
}

NORM_OFFSET = (
    "add 1 to every *.norm.weight except linear_attn.norm.weight "
    "(Qwen3 RMSNorm stores the offset)"
)


def mapping_table() -> Dict[str, object]:
    return {
        "top": dict(_TOP_MAP),
        "layer": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in _LAYER_MAP.items()
        },
        "memory": {
            "self_attn.memt." + suffix: "blk.N.memt." + suffix
            for suffix in MEMORY_SUFFIXES
        },
        "transforms": dict(TRANSFORMS),
        "norm_offset": NORM_OFFSET,
    }


def run(config_path: Optional[Path], index_path: Optional[Path],
        root: Optional[Path]) -> Report:
    report = Report()
    try:
        config, weight_map, headers, _shards = load_checkpoint(
            config_path, index_path, root, report
        )
        report.config = {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "memt_checkpoint_id": config.get("memt_checkpoint_id"),
        }
        audit(config, weight_map, headers, report)
    except (AuditError, OSError, ValueError, json.JSONDecodeError) as exc:
        report.error(str(exc))
    return report


def _print_human(report: Report) -> None:
    print("MemT-1.0-Nano audit: %s" % ("PASS" if report.ok else "FAIL"))
    if report.geometry:
        g = report.geometry
        print(
            "  backbone: hidden=%d heads=%d head_dim=%d kv=%d layers=%d interval=%d"
            % (g["hidden_size"], g["num_attention_heads"], g["head_dim"],
               g["num_key_value_heads"], g["num_hidden_layers"],
               g["full_attention_interval"])
        )
        print(
            "  MoE: experts=%d top_k=%d moe_ff=%d shared_ff=%d; vocab=%d"
            % (g["num_experts"], g["num_experts_per_tok"], g["moe_intermediate_size"],
               g["shared_expert_intermediate_size"], g["vocab_size"])
        )
    if report.memory:
        m = report.memory
        print(
            "  memory: layers=%r heads=%d head_dim=%d gate_rank=%d chunk=%d"
            % (m["layers"], m["heads"], m["head_dim"], m["gate_rank"],
               m["chunk_tokens"])
        )
        print(
            "  memory state: %r FP32, %d bytes/layer/session"
            % (m["state_shape"], m["state_bytes_per_layer"])
        )
    if report.counts:
        print("  counts: %r" % (report.counts,))
    for warning in report.warnings:
        print("  WARN: %s" % warning)
    for error in report.errors:
        print("  ERROR: %s" % error)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", nargs="?", type=Path,
                        help="checkpoint directory (config.json + index + shards)")
    parser.add_argument("--config", type=Path, help="config.json path")
    parser.add_argument("--index", type=Path, help="model.safetensors.index.json path")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--out", type=Path, help="write JSON to this file")
    parser.add_argument("--mapping", action="store_true",
                        help="print the HF -> GGUF mapping and exit")
    args = parser.parse_args(argv)

    if args.mapping:
        print(json.dumps(mapping_table(), indent=2, sort_keys=True))
        return 0

    if args.checkpoint is None and args.config is None:
        parser.error("pass a checkpoint directory, or --config (and optionally --index)")

    report = run(args.config, args.index, args.checkpoint)
    if args.json or args.out:
        payload = {
            "ok": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "config": report.config,
            "geometry": report.geometry,
            "memory": report.memory,
            "counts": report.counts,
            "gguf_tensors": report.gguf_tensors,
            "mapping": mapping_table(),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        if args.json:
            print(text)
    else:
        _print_human(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
