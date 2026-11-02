"""GGUF -> Qwen4ExpConfig -> meta 骨架 -> 名字规则映射 -> 与 GGUF 张量全量核对。

产出 mapping.json（加载器的依据）。任何 GGUF 张量或 HF 参数落不进映射都会报出来。

用法：python q4_map_check.py <inv1.json> <inv2.json> [mapping.json]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict


def merge(paths):
    kv: dict = {}
    tensors: dict[str, dict] = {}
    for p in paths:
        inv = json.load(open(p, encoding="utf-8"))
        kv.update(inv["kv"])
        for t in inv["tensors"]:
            tensors[t["name"]] = t
    return kv, tensors


def build_config(kv: dict):
    from transformers.models.qwen4_exp import Qwen4ExpTextConfig

    def g(k, d=None):
        return kv.get(f"qwen4exp.{k}", d)

    layer_types = [
        "linear_attention" if r else "qwen_sparse_attention"
        for r in g("attention.recurrent_layers")
    ]
    n_k_heads = g("ssm.group_count")           # 16
    k_head = 128                                # attn_qkv [2560,10240] = q16*128+k16*128+v48*128
    n_v_heads = g("ssm.time_step_rank")        # 48
    assert 2 * n_k_heads * k_head + n_v_heads * k_head == 10240
    return Qwen4ExpTextConfig(
        vocab_size=g("vocab_size", 248320),
        hidden_size=g("embedding_length"),
        num_hidden_layers=g("block_count"),
        num_attention_heads=g("attention.head_count"),
        num_key_value_heads=g("attention.head_count_kv"),
        head_dim=g("attention.key_length"),
        rms_norm_eps=g("attention.layer_norm_rms_epsilon"),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": g("rope.freq_base"),
            "partial_rotary_factor": 0.5,     # rotary 128 = indexer_head_dim（校验要求）
            "mrope_section": [11, 11, 10],    # ×2 交错 = 64 = GGUF rope.dimension_count
        },
        layer_types=layer_types,
        linear_conv_kernel_dim=g("ssm.conv_kernel"),
        linear_key_head_dim=k_head,
        linear_value_head_dim=k_head,
        linear_num_key_heads=n_k_heads,
        linear_num_value_heads=n_v_heads,
        num_experts=g("expert_count"),
        num_experts_per_tok=g("expert_used_count"),
        moe_intermediate_size=g("expert_feed_forward_length"),
        shared_expert_intermediate_size=g("expert_shared_feed_forward_length"),
        eos_token_id=g("ple.eos_token_id"),
        hc_count=g("hyper_connection.count"),
        hc_lowrank=g("hyper_connection.low_rank"),
        # GDN 输出门激活（2026-10-29 dbg_l0 实锤）：引擎 gdn_rec_kernel 用 sigm(z)
        # （kernels.cu:528）。config 默认 output_gate_type=None 会回落到
        # hidden_act="silu"，GDN 输出方向/幅度全错（silu 可为负、无界）。
        output_gate_type="sigmoid",
        ple_layer_ids=[l + 1 for l in g("ple.layers")],
        # 引擎基准事实（ple.hpp）：每 token gather 16 行、每行 160 维 -> 2560。
        # GGUF kv 的 embedding_length_per_layer_input=160 是每头维度，不是总维度。
        ple_embed_dim=2560,
        ple_conv_kernel_size=g("ple.conv_kernel"),
        ngram_size=g("ple.ngram_size"),
        heads_per_ngram=g("ple.heads_per_ngram"),
        indexer_n_heads=g("attention.indexer.head_count"),
        indexer_head_dim=g("attention.indexer.key_length"),
        indexer_kv_heads=1,
        indexer_budget=g("attention.indexer.top_k"),
        indexer_compress_ratio=4,
    )


def rule_map(gname: str) -> str | None:
    """GGUF tensor name -> HF param name. None = 需要特殊处理（拼接/交织/未定）。"""
    if gname == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gname == "output.weight":
        return "lm_head.weight"
    if gname == "output_hc_norm.weight":
        return "model.hyper_connection_mixer.hc_norm.weight"
    if gname == "output_hc_up.weight":
        return "model.hyper_connection_mixer.input_mix_weight_up.weight"
    if gname == "output_hc_down.weight":
        return "model.hyper_connection_mixer.input_mix_weight_down.weight"
    m = re.match(r"blk\.(\d+)\.(.+)", gname)
    if not m:
        return None
    l, rest = m.group(1), m.group(2)
    if rest == "attn_qkv.weight":
        return f"model.layers.{l}.linear_attn.in_proj_qkv.weight"
    if rest == "attn_gate.weight":
        return f"model.layers.{l}.linear_attn.in_proj_z.weight"
    if rest == "ssm_alpha.weight":
        return f"model.layers.{l}.linear_attn.in_proj_a.weight"
    if rest == "ssm_beta.weight":
        return f"model.layers.{l}.linear_attn.in_proj_b.weight"
    if rest == "ssm_a":
        return f"model.layers.{l}.linear_attn.A_log"
    if rest == "ssm_dt.bias":
        return f"model.layers.{l}.linear_attn.dt_bias"
    if rest == "ssm_conv1d.weight":
        return f"model.layers.{l}.linear_attn.conv1d.weight"       # (10240,4)->(10240,1,4)
    if rest == "ssm_norm.weight":
        return f"model.layers.{l}.linear_attn.norm.weight"
    if rest == "ssm_out.weight":
        return f"model.layers.{l}.linear_attn.out_proj.weight"
    if rest == "attn_q.weight":
        return f"model.layers.{l}.self_attn.q_proj.weight"
    if rest == "attn_k.weight":
        return f"model.layers.{l}.self_attn.k_proj.weight"
    if rest == "attn_v.weight":
        return f"model.layers.{l}.self_attn.v_proj.weight"
    if rest == "attn_output.weight":
        return f"model.layers.{l}.self_attn.o_proj.weight"
    if rest == "attn_q_norm.weight":
        return f"model.layers.{l}.self_attn.q_norm.weight"
    if rest == "attn_k_norm.weight":
        return f"model.layers.{l}.self_attn.k_norm.weight"
    if rest == "indexer.q_proj.weight":
        return f"model.layers.{l}.self_attn.indexer.index_qk_proj.weight"  # q 行 + k 行拼接，特判
    if rest == "indexer.k_proj.weight":
        return "INDEXER_K_HALF"                                            # 拼进 index_qk_proj 后半
    if rest == "indexer.k_norm.weight":
        return f"model.layers.{l}.self_attn.indexer.k_layernorm.weight"
    if rest == "indexer.q_norm.weight":
        return f"model.layers.{l}.self_attn.indexer.q_layernorm.weight"
    if rest == "ffn_gate_inp.weight":
        return f"model.layers.{l}.mlp.gate.weight"
    if rest == "ffn_gate_inp_shexp.weight":
        return f"model.layers.{l}.mlp.shared_expert_gate.weight"   # (2560,)->(1,2560)
    if rest == "ffn_down_shexp.weight":
        return f"model.layers.{l}.mlp.shared_expert.down_proj.weight"
    if rest == "ffn_gate_shexp.weight":
        return f"model.layers.{l}.mlp.shared_expert.gate_proj.weight"
    if rest == "ffn_up_shexp.weight":
        return f"model.layers.{l}.mlp.shared_expert.up_proj.weight"
    if rest == "ffn_down_exps.weight":
        return f"model.layers.{l}.mlp.experts.down_proj"
    if rest == "ffn_gate_exps.weight" or rest == "ffn_up_exps.weight":
        return f"model.layers.{l}.mlp.experts.gate_up_proj"        # 两源交织，特判
    hm = re.match(r"hc_(attn|ffn)_(norm|up|down|inject)(?:\.weight)?", rest)
    if hm:
        sub, which = hm.group(1), hm.group(2)
        mod = "attn_hyper_connection" if sub == "attn" else "mlp_hyper_connection"
        which_map = {
            "norm": "hc_norm.weight",
            "up": "input_mix_weight_up.weight",
            "down": "input_mix_weight_down.weight",
            "inject": "block_inject_weight.weight",
        }
        return f"model.layers.{l}.{mod}.{which_map[which]}"
    if rest == "ple_conv1d.weight":
        return f"model.layers.{l}.ple.conv1d.weight"       # (4,10240) 转置 -> (10240,1,4)，引擎 ple.hpp 证实
    if rest == "ple_key.weight":
        return f"model.layers.{l}.ple.key_proj.weight"
    if rest == "ple_value.weight":
        return f"model.layers.{l}.ple.value_proj.weight"
    if rest == "ple_norm_conv.weight":
        return f"model.layers.{l}.ple.norm_conv.weight"
    if rest == "ple_norm_key.weight":
        return f"model.layers.{l}.ple.norm_key.weight"
    if rest == "ple_norm_query.weight":
        return f"model.layers.{l}.ple.norm_query.weight"
    if gname == "per_layer_token_embd.weight":
        return "PLE_NGRAM_EMBEDDING"                               # 维度优先存储，loader 按行 gather
    return None


def main() -> int:
    args = [a for a in sys.argv[1:] if "inventory" in a]
    out_path = next((a for a in sys.argv[1:] if "mapping" in a), None)
    kv, tensors = merge(args)
    from accelerate import init_empty_weights
    from transformers.models.qwen4_exp import Qwen4ExpForCausalLM

    cfg = build_config(kv)
    import torch

    with torch.device("meta"):
        model = Qwen4ExpForCausalLM._from_config(cfg)
    hf = {n: tuple(p.shape) for n, p in model.named_parameters()}
    print(f"config OK: layers {cfg.num_hidden_layers}, ple {cfg.ple_layer_ids} dim {cfg.ple_embed_dim}")
    print(f"HF params {len(hf)}, GGUF tensors {len(tensors)}")

    # HF 侧 ngram 词表推导 vs GGUF 显式值
    gguf_heads = kv.get("qwen4exp.ple.head_vocab_sizes")
    gguf_offsets = kv.get("qwen4exp.ple.head_offsets")
    gguf_mult = kv.get("qwen4exp.ple.layer_multipliers")
    emb = dict(model.named_parameters()).get(
        "model.layers.1.ple.ple_embedding.ngram_embedding") if hasattr(model, "model") else None
    for n in hf:
        if "ngram" in n or "ple_embedding" in n:
            emb = dict(model.named_parameters())[n]
            print("PLE embedding param:", n, tuple(emb.shape))
    if gguf_heads:
        print(f"GGUF ple heads: {len(gguf_heads)} sizes, total {sum(gguf_heads)}, "
              f"offsets[1] {gguf_offsets[1]}, multipliers {gguf_mult}")

    mapping: dict[str, str] = {}
    missing_hf = set(hf)
    specials = defaultdict(list)
    for gname in tensors:
        tgt = rule_map(gname)
        if tgt is None:
            specials["UNMAPPED_GGUF"].append(gname)
        elif tgt == "PLE_NGRAM_EMBEDDING":
            specials["SPECIAL:ple_embedding_transposed"].append(gname)
        elif tgt == "INDEXER_K_HALF":
            specials["SPECIAL:indexer_qk_concat"].append(gname)
        elif "experts.gate_up_proj" in tgt:
            specials["SPECIAL:gate_up_halves"].append(gname)       # gate/up 两源 -> 对半拼
        elif tgt in hf:
            mapping[gname] = tgt
            missing_hf.discard(tgt)
        else:
            specials.setdefault("NO_HF_TARGET", []).append((gname, tgt))

    # gate_up / indexer 的目标由成对张量共同消费，从 missing_hf 里显式核销
    consumed_by_specials = set()
    for gname in tensors:
        tgt = rule_map(gname)
        if tgt and ("experts.gate_up_proj" in tgt or tgt == "INDEXER_K_HALF"
                    or tgt == "PLE_NGRAM_EMBEDDING"):
            hf_name = {"INDEXER_K_HALF": None}.get(tgt)
            if "experts.gate_up_proj" in (tgt or ""):
                hf_name = tgt
            elif tgt == "PLE_NGRAM_EMBEDDING":
                hf_name = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
            if hf_name and hf_name in missing_hf:
                consumed_by_specials.add(hf_name)
    # INDEXER_K_HALF 的 HF 目标与 q_proj 相同
    for gname in tensors:
        if rule_map(gname) == "INDEXER_K_HALF":
            l = re.match(r"blk\.(\d+)\.", gname).group(1)
            t = f"model.layers.{l}.self_attn.indexer.index_qk_proj.weight"
            if t in missing_hf:
                consumed_by_specials.add(t)
    missing_hf -= consumed_by_specials

    print(f"\nmapped by rule: {len(mapping)}  + specials: "
          f"{ {k: len(v) for k, v in specials.items()} }")
    for k, v in specials.items():
        if not k.startswith("SPECIAL"):
            print(f"\n== {k} ({len(v)}) ==")
            for item in v[:12]:
                print("  ", item)
    print(f"\n== HF params still unmatched ({len(missing_hf)}) ==")
    for n in sorted(missing_hf)[:40]:
        print(f"   {tuple(hf[n])} {n}")
    if len(missing_hf) > 40:
        print(f"   ... and {len(missing_hf) - 40} more")
    # PLE 特判对（per_layer_token_embd ↔ ple_embedding.ngram_embedding）由 loader 自己接，
    # 不进 mapping JSON；其余必须全闭合才允许写出。
    ok = (not specials.get("UNMAPPED_GGUF") or
          specials.get("UNMAPPED_GGUF") == ["per_layer_token_embd.weight"]) \
        and all(n == "model.layers.1.ple.ple_embedding.ngram_embedding.weight" for n in missing_hf) \
        and not specials.get("NO_HF_TARGET")
    if out_path and ok:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
