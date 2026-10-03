"""Writes a GGUF with the tensor NAMES and SHAPES of a (tiny) qwen35moe model and zero/random data, to test the
engine's config derivation on a CPU-only machine:  python write_shape_gguf.py out.gguf"""
import sys, numpy as np
import gguf

H, V, NL = 256, 512, 4                  # hidden, vocab, layers (3 GDN + 1 attention)
HK, HV, S = 2, 4, 128                   # GDN heads / head size
NH, NKV, HD = 4, 2, 128                 # attention heads / kv heads / head dim
NE, K, FF, FS = 8, 2, 64, 64            # experts, used, expert ff, shared ff
QKV, INNER = 2 * HK * S + HV * S, HV * S

w = gguf.GGUFWriter(sys.argv[1], "qwen35moe")
w.add_uint32("qwen35moe.block_count", NL)
w.add_uint32("qwen35moe.expert_used_count", K)
w.add_uint32("qwen35moe.rope.dimension_count", HD // 4)
w.add_float32("qwen35moe.rope.freq_base", 1e7)
w.add_float32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6)
rng = np.random.default_rng(0)
def t(name, *shape): w.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))   # numpy shape = reversed GGUF ne
t("token_embd.weight", V, H); t("output_norm.weight", H); t("output.weight", V, H)
for il in range(NL):
    p = f"blk.{il}."
    t(p + "attn_norm.weight", H); t(p + "post_attention_norm.weight", H)
    if il % 4 != 3:
        t(p + "attn_qkv.weight", QKV, H); t(p + "attn_gate.weight", INNER, H)
        t(p + "ssm_alpha.weight", HV, H); t(p + "ssm_beta.weight", HV, H); t(p + "ssm_out.weight", H, INNER)
        t(p + "ssm_conv1d.weight", QKV, 4); t(p + "ssm_a", HV); t(p + "ssm_dt.bias", HV); t(p + "ssm_norm.weight", S)
    else:
        t(p + "attn_q.weight", NH * HD * 2, H); t(p + "attn_k.weight", NKV * HD, H); t(p + "attn_v.weight", NKV * HD, H)
        t(p + "attn_output.weight", H, NH * HD); t(p + "attn_q_norm.weight", HD); t(p + "attn_k_norm.weight", HD)
    t(p + "ffn_gate_inp.weight", NE, H)
    t(p + "ffn_gate_exps.weight", NE, FF, H); t(p + "ffn_up_exps.weight", NE, FF, H); t(p + "ffn_down_exps.weight", NE, H, FF)
    t(p + "ffn_gate_shexp.weight", FS, H); t(p + "ffn_up_shexp.weight", FS, H); t(p + "ffn_down_shexp.weight", H, FS)
    t(p + "ffn_gate_inp_shexp.weight", H)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", sys.argv[1])
