#!/usr/bin/env python3
"""Builds the end-to-end oracle for strata-qwen36 (needs: torch, transformers>=5, gguf, numpy).

  python make_tiny_model.py outdir        ->  outdir/tiny.gguf  and  outdir/expected.txt
  strata-qwen36 --native outdir/tiny.gguf --selftest outdir/expected.txt

1. a tiny Qwen3.5-MoE with random weights is created with HuggingFace and run on a fixed prompt (reference logits);
2. its weights are converted to GGUF conventions with the SAME transformations as llama.cpp's converter
   (conversion/qwen.py): V heads grouped -> tiled, A_log -> -exp(A_log), RMSNorm weight +1 (except the gated norm),
   conv1d squeezed, experts split into gate/up/down;
3. tiny.gguf is F32, so the comparison measures the wiring (GDN, attention, RoPE, router, shared expert, head mapping,
   protocol-free forward pass) without any quantisation error.
"""
import os, sys
import numpy as np
import torch
import gguf
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

H, V, NL = 256, 512, 4
HK, HV, S = 2, 4, 128
NH, NKV, HD = 4, 2, 128
NE, K, FF, FS = 8, 2, 64, 64
PROMPT = [5, 17, 301, 42, 7, 99, 256, 3, 11, 400, 21, 8]


def reorder_v(t: np.ndarray, dim: int, head_dim: int) -> np.ndarray:
    """grouped (by K head) -> tiled, exactly _LinearAttentionVReorderBase._reorder_v_heads."""
    vpk = HV // HK
    shape = list(t.shape)
    new = shape[:dim] + [HK, vpk, head_dim] + shape[dim + 1:]
    t = t.reshape(new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(t.transpose(perm)).reshape(shape)


def selfcheck_reorder():
    idx = np.arange(HV).reshape(HV, 1)
    out = reorder_v(idx, 0, 1).reshape(-1)
    vpk = HV // HK
    for g in range(HK):
        for r in range(vpk):
            assert out[r * HK + g] == g * vpk + r, "reorder does not match the converter"


def main(out):
    selfcheck_reorder()
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=NL, num_attention_heads=NH, num_key_value_heads=NKV,
        head_dim=HD, linear_conv_kernel_dim=4, linear_key_head_dim=S, linear_value_head_dim=S,
        linear_num_key_heads=HK, linear_num_value_heads=HV, moe_intermediate_size=FF,
        shared_expert_intermediate_size=FS, num_experts_per_tok=K, num_experts=NE, max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [6, 5, 5], "mrope_interleaved": True})
    model = Qwen3_5MoeForCausalLM(cfg).float().eval()
    with torch.no_grad():                      # random but well-scaled, and a non-trivial A_log / dt_bias / norms
        for n, p in model.named_parameters():
            if p.dim() >= 2: p.normal_(0, 0.05)
            elif "A_log" in n: p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4.0)))
            elif "dt_bias" in n: p.uniform_(0.2, 1.0)
            else: p.normal_(0, 0.1)
        ids = torch.tensor([PROMPT])
        logits = model(input_ids=ids).logits[0].float().numpy()            # [N, V]

    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    def get(il, suffix): return sd[next(k for k in sd if k.endswith(f"layers.{il}.{suffix}"))]
    top = lambda suffix: sd[next(k for k in sd if k.endswith(suffix) and "layers." not in k)]

    w = gguf.GGUFWriter(os.path.join(out, "tiny.gguf"), "qwen35moe")
    w.add_uint32("qwen35moe.block_count", NL); w.add_uint32("qwen35moe.expert_used_count", K)
    w.add_uint32("qwen35moe.rope.dimension_count", HD // 4); w.add_float32("qwen35moe.rope.freq_base", 1e7)
    w.add_float32("qwen35moe.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    add = lambda name, arr: w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
    add("token_embd.weight", top("embed_tokens.weight")); add("output_norm.weight", top("norm.weight") + 1.0)
    add("output.weight", top("lm_head.weight"))
    key_dim = HK * S
    for il in range(NL):
        p = f"blk.{il}."
        add(p + "attn_norm.weight", get(il, "input_layernorm.weight") + 1.0)
        add(p + "post_attention_norm.weight", get(il, "post_attention_layernorm.weight") + 1.0)
        if cfg.layer_types[il] == "linear_attention":
            qkv = get(il, "linear_attn.in_proj_qkv.weight")
            qkv = np.concatenate([qkv[:2 * key_dim], reorder_v(qkv[2 * key_dim:], 0, S)], 0)
            add(p + "attn_qkv.weight", qkv)
            add(p + "attn_gate.weight", reorder_v(get(il, "linear_attn.in_proj_z.weight"), 0, S))
            add(p + "ssm_alpha.weight", reorder_v(get(il, "linear_attn.in_proj_a.weight"), 0, 1))
            add(p + "ssm_beta.weight", reorder_v(get(il, "linear_attn.in_proj_b.weight"), 0, 1))
            add(p + "ssm_out.weight", reorder_v(get(il, "linear_attn.out_proj.weight"), 1, S))
            conv = get(il, "linear_attn.conv1d.weight").reshape(-1, 4)
            conv = np.concatenate([conv[:2 * key_dim], reorder_v(conv[2 * key_dim:], 0, S)], 0)
            add(p + "ssm_conv1d.weight", conv)
            add(p + "ssm_a", reorder_v(-np.exp(get(il, "linear_attn.A_log")), 0, 1))
            add(p + "ssm_dt.bias", reorder_v(get(il, "linear_attn.dt_bias"), 0, 1))
            add(p + "ssm_norm.weight", get(il, "linear_attn.norm.weight"))          # gated norm: NO +1
        else:
            add(p + "attn_q.weight", get(il, "self_attn.q_proj.weight")); add(p + "attn_k.weight", get(il, "self_attn.k_proj.weight"))
            add(p + "attn_v.weight", get(il, "self_attn.v_proj.weight")); add(p + "attn_output.weight", get(il, "self_attn.o_proj.weight"))
            add(p + "attn_q_norm.weight", get(il, "self_attn.q_norm.weight") + 1.0)
            add(p + "attn_k_norm.weight", get(il, "self_attn.k_norm.weight") + 1.0)
        add(p + "ffn_gate_inp.weight", get(il, "mlp.gate.weight"))
        gu = get(il, "mlp.experts.gate_up_proj")                                      # [E, 2*FF, H]: gate first, then up
        add(p + "ffn_gate_exps.weight", gu[:, :FF]); add(p + "ffn_up_exps.weight", gu[:, FF:])
        add(p + "ffn_down_exps.weight", get(il, "mlp.experts.down_proj"))
        add(p + "ffn_gate_shexp.weight", get(il, "mlp.shared_expert.gate_proj.weight"))
        add(p + "ffn_up_shexp.weight", get(il, "mlp.shared_expert.up_proj.weight"))
        add(p + "ffn_down_shexp.weight", get(il, "mlp.shared_expert.down_proj.weight"))
        add(p + "ffn_gate_inp_shexp.weight", get(il, "mlp.shared_expert_gate.weight").reshape(-1))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    with open(os.path.join(out, "expected.txt"), "w") as f:
        f.write(f"{len(PROMPT)} {V}\n" + " ".join(map(str, PROMPT)) + "\n")
        for row in logits: f.write(" ".join(f"{x:.7e}" for x in row) + "\n")
    print("wrote", out, "- now run:  strata-qwen36 --native", os.path.join(out, "tiny.gguf"), "--selftest", os.path.join(out, "expected.txt"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny_out")
