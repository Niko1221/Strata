#!/usr/bin/env python3
"""NVFP4 checkpoint -> Q8_0 GGUF for GLM-5.3-Flash-UNCENSORED (the recipe's step 2).

mainline llama.cpp's glm5-next runtime does not apply the NVFP4 weight_scale_2 (the
per-expert F32 sidecar tensors load but the graph never multiplies by them), so the
converter's native NVFP4 GGUF carries expert weights ~1/ws2 x too large (measured:
trunk = source x 2/ws2) -> NaN activations.  The fix: dequantize to the true weights,
quantize the experts straight to Q8_0; dense/1D requantized from the checkpoint's
BF16/F32.  Structure (names/dims/KV) mirrors the running reference GGUF except the MTP
head smalls the NVFP4 checkpoint lacks (skipped; load_mtp=false does not need them).
Mapping follows conversion/glm.py's Glm5NextModel: A_log -> -exp(A_log), kv_b_proj ->
v_b (nope-only MLA), experts stacked on dim 0, hc_*/indexer names gain a .weight suffix.
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np
import gguf

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

b256 = np.arange(256, dtype=np.uint8)
_e = (b256 >> 3) & 0xF
_m = (b256 & 7).astype(np.float32)
_v = np.where(_e == 0, _m * 2.0 ** -9,
              np.where(_e == 15, (1 + _m / 8) * 16.0, (1 + _m / 8) * 2.0 ** (_e.astype(np.int32) - 7)))
_s = np.where(b256 >= 128, -1.0, 1.0).astype(np.float32)
E4M3 = (_s * _v).astype(np.float32)

SKIP_MTP_SMALLS = ("nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight")


def q8_0(x):
    x = np.asarray(x).reshape(-1)                      # 3D conv1d 等先摊平;元素序不变
    assert x.size % 32 == 0, x.size
    blocks = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(blocks).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float16)
    codes = np.rint(blocks.astype(np.float32) / scale.astype(np.float32)[:, None]).astype(np.int8)
    return np.concatenate([scale.view(np.uint8).reshape(-1, 2), codes.view(np.uint8)], axis=1).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hc-npz", default=None, help="官方 hc_* 张量 npz(FP8 源中被 header 压毁,用官方值替换)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    ref_kv = {}
    ref_tensors = []
    ref_types = {}
    for p in sorted(glob.glob(a.ref)):
        rr = gguf.GGUFReader(p)
        if not ref_kv:
            for name, f in rr.fields.items():
                if name.startswith("GGUF.") or name.startswith("split."):
                    continue
                t0v = f.types[0]
                if t0v == gguf.GGUFValueType.ARRAY:
                    itype = f.types[1]
                    if itype == gguf.GGUFValueType.STRING:
                        val = [f.parts[i].tobytes().decode("utf-8", "replace") for i in f.data]
                    else:
                        val = [f.parts[i][0] for i in f.data]
                    ref_kv[name] = (t0v, itype, val)
                else:
                    v = f.parts[f.data[0]]
                    val = v.tobytes().decode("utf-8", "replace") if t0v == gguf.GGUFValueType.STRING else v[0]
                    ref_kv[name] = (t0v, None, val)
        for t in rr.tensors:
            ref_tensors.append((t.name, [int(x) for x in t.shape]))
            ref_types[t.name] = t.tensor_type.name
    print(f"reference: {len(ref_tensors)} tensors, {len(ref_kv)} KV", flush=True)

    index = {}
    shards = sorted(glob.glob(a.src + "/model-*.safetensors"))
    for p in shards:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, m in hdr.items():
            if name != "__metadata__":
                index[name] = (p, m["data_offsets"][0], m["data_offsets"][1], m["shape"], m["dtype"])
    print(f"source index: {len(index)} tensors in {len(shards)} shards", flush=True)
    global MLA_DIMS
    with open(a.src + "/config.json") as _f:
        _cfg = json.load(_f).get("text_config", json.load(open(a.src + "/config.json")))
    MLA_DIMS = (_cfg["num_attention_heads"], _cfg["qk_nope_head_dim"],
                _cfg["v_head_dim"], _cfg["kv_lora_rank"])
    print(f"MLA dims: {MLA_DIMS}", flush=True)
    handles = {}

    def src_bytes(key):
        p, o0, o1, shape, dtype = index[key]
        if p not in handles:
            handles[p] = open(p, "rb")
        handles[p].seek(o0)
        return handles[p].read(o1 - o0), shape, dtype

    def bf16_f32(raw, name=""):
        x = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        return sanitize(x, name)

    def sanitize(x, name=""):
        # 消融损伤清理: 非有限值与爆炸值置零(实测 lm_head 有 Inf 和 3.3e38 级元素)
        nf = int((~np.isfinite(x)).sum())
        big = int((np.isfinite(x) & (np.abs(x) > 1e6)).sum())
        if nf or big:
            x = np.where(np.isfinite(x) & (np.abs(x) <= 1e6), x, 0.0).astype(np.float32)
            print(f"  sanitize {name}: non-finite {nf}, |x|>1e6 {big}", flush=True)
        return x

    TABLE = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_a.weight": "self_attn.q_a_proj.weight",
        "attn_q_b.weight": "self_attn.q_b_proj.weight",
        "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
        "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
        "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
        "ssm_conv1d_q.weight": "self_attn.q_conv1d.weight",
        "ssm_conv1d_k.weight": "self_attn.k_conv1d.weight",
        "ssm_conv1d_v.weight": "self_attn.v_conv1d.weight",
        "ssm_a": "self_attn.A_log",
        "ssm_dt.bias": "self_attn.dt_bias",
        "ssm_norm.weight": "self_attn.o_norm.weight",
        "ssm_beta.weight": "self_attn.b_proj.weight",
        "ssm_f_a.weight": "self_attn.f_a_proj.weight",
        "ssm_f_b.weight": "self_attn.f_b_proj.weight",
        "ssm_g_a.weight": "self_attn.g_a_proj.weight",
        "ssm_g_b.weight": "self_attn.g_b_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
        "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
        "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
        "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
        "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
        "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
        "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
        "indexer.attn_k.weight": "self_attn.indexer.wk.weight",
        "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
        "indexer_compressor_ape.weight": "self_attn.indexer.index_kpool_compress_ape",
        "indexer_compressor_gate.weight": "self_attn.indexer.index_kpool_compress_gate",
        "hc_attn_base.weight": "hc_attn_base",
        "hc_attn_fn.weight": "hc_attn_fn",
        "hc_attn_scale.weight": "hc_attn_scale",
        "hc_ffn_base.weight": "hc_ffn_base",
        "hc_ffn_fn.weight": "hc_ffn_fn",
        "hc_ffn_scale.weight": "hc_ffn_scale",
    }

    keys_src_map = {}
    for gk, sk in TABLE.items():
        keys_src_map[gk] = sk
    for gk, sk in [("hc_attn_base.weight", "hc_attn_base"), ("hc_attn_fn.weight", "hc_attn_fn"),
                   ("hc_attn_scale.weight", "hc_attn_scale"), ("hc_ffn_base.weight", "hc_ffn_base"),
                   ("hc_ffn_fn.weight", "hc_ffn_fn"), ("hc_ffn_scale.weight", "hc_ffn_scale")]:
        keys_src_map[gk] = "model.language_model.layers.{L}." + sk

    hc_npz = None
    if a.hc_npz and os.path.exists(a.hc_npz):
        hc_npz = np.load(a.hc_npz)
        print(f"hc npz: {len(hc_npz.files)} 张量", flush=True)

    def source_keys(gguf_name, dims=None):
        if not gguf_name.startswith("blk."):
            table = {"token_embd.weight": ["model.language_model.embed_tokens.weight"],
                     "output.weight": ["lm_head.weight"],
                     "output_norm.weight": ["model.language_model.norm.weight"]}
            return table.get(gguf_name, [])
        parts = gguf_name.split(".")
        layer = int(parts[1])
        sub = ".".join(parts[2:])
        lp = f"model.language_model.layers.{layer}."
        if sub in SKIP_MTP_SMALLS:
            return []
        if sub == "nextn.shared_head_norm.weight":
            return [lp + "shared_head.norm.weight"]
        if sub.endswith("_exps.weight"):
            proj = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[sub.split("_")[1]]
            n_exp = dims[-1] if dims else 288
            return [f"{lp}mlp.experts.{e}.{proj}.weight" for e in range(n_exp)]
        if sub in ("attn_k_b.weight", "attn_v_b.weight"):
            return [lp + "self_attn.kv_b_proj.weight"]
        return [lp + TABLE[sub]] if sub in TABLE else [lp + sub]

    def produce(gguf_name, gguf_dims, out_kind):
        parts = gguf_name.split(".")
        layer = int(parts[1]) if parts[0] == "blk" else -1
        sub = ".".join(parts[2:]) if parts[0] == "blk" else gguf_name
        lp = f"model.language_model.layers.{layer}."
        if sub.endswith("_exps.weight"):
            # 内存有界路径:逐专家 反量化 -> Q8_0 块,只保留 Q8_0 字节(288 x ~8.9 MB 的
            # fp32 全量展开会被 OOM 杀掉,实测 63 GB RSS)
            proj = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[sub.split("_")[1]]
            n_exp = gguf_dims[-1]
            out = []
            for e in range(n_exp):
                base = f"{lp}mlp.experts.{e}.{proj}"
                raw, shape, dtype = src_bytes(base + ".weight")
                assert dtype == "F8_E4M3", (base, dtype)
                K, N = shape[0], shape[1]
                iraw, ishape, idt = src_bytes(base + ".weight_scale_inv")
                inv = np.frombuffer(iraw, dtype=np.float32).reshape(K // 128, N // 128)
                codes = np.frombuffer(raw, dtype=np.uint8)
                deq = (E4M3[codes].reshape(K // 128, 128, N // 128, 128)
                       * inv[:, None, :, None]).reshape(K, N)
                out.append(q8_0(deq))
                del codes, deq
            blob = np.concatenate(out)
            del out
            return blob.tobytes()
        if sub == "ssm_a":
            raw, _, _ = src_bytes(lp + "self_attn.A_log")
            alog = np.frombuffer(raw, dtype=np.float32)
            # 消融损伤: 部分层 A_log 爆炸(>88 → exp 溢出 inf);夹紧到语义合理范围
            alog = np.clip(alog, -10.0, 15.0)
            return (-np.exp(alog.astype(np.float32))).astype(np.float32).tobytes()
        if sub in ("attn_k_b.weight", "attn_v_b.weight"):
            # MLA kv_b 拆分,conversion/glm.py 的数学(参考 dims 已校验: k=[256,512,64], v=[512,256,64])
            n_head, nope, v_head, kv_lora = MLA_DIMS
            raw, shape, _ = src_bytes(lp + "self_attn.kv_b_proj.weight")
            assert shape[0] == n_head * (nope + v_head), (name, shape)
            kv_b = bf16_f32(raw).reshape(n_head, nope + v_head, kv_lora)
            k_b = kv_b[:, :nope, :].transpose(0, 2, 1)   # [n_head, kv_lora, nope]
            v_b = kv_b[:, nope:, :]                      # [n_head, v_head, kv_lora]
            x = k_b if sub == "attn_k_b.weight" else v_b
            ref_dims = dict(REF_DIMS)[gguf_name]
            assert list(x.shape[::-1]) == ref_dims, (gguf_name, list(x.shape[::-1]), ref_dims)
            return q8_0(x).tobytes()
        if sub.startswith("hc_") and sub.endswith(".weight") and hc_npz is not None:
            src_key = keys_src_map.get(sub)
            if src_key is not None:
                npz_key = src_key.replace("{L}", str(layer))
                if npz_key in hc_npz:
                    return hc_npz[npz_key].astype(np.float32).tobytes()
        keys = source_keys(gguf_name, gguf_dims)
        ok = [k for k in keys if k in index]
        assert ok, f"no source for {gguf_name}: tried {keys[:2]}"
        raw, shape, dtype = src_bytes(ok[0])
        if dtype == "F32":
            return sanitize(np.frombuffer(raw, dtype=np.float32), gguf_name).tobytes()
        if dtype == "F8_E4M3":
            assert shape[0] % 128 == 0 and shape[1] % 128 == 0, (ok[0], shape)
            ikey = ok[0].replace(".weight", ".weight_scale_inv")
            iraw, _, _ = src_bytes(ikey)
            inv = np.frombuffer(iraw, dtype=np.float32).reshape(shape[0] // 128, shape[1] // 128)
            K, N = shape[0], shape[1]
            codes = np.frombuffer(raw, dtype=np.uint8)
            x = sanitize((E4M3[codes].reshape(K // 128, 128, N // 128, 128)
                          * inv[:, None, :, None]).reshape(K, N), gguf_name)
            if out_kind == "F32" or len(gguf_dims) <= 1:
                return x.astype(np.float32).tobytes()
            return q8_0(x).tobytes()
        x = sanitize(bf16_f32(raw), gguf_name)
        if out_kind == "F32" or len(gguf_dims) <= 1:
            return x.astype(np.float32).tobytes()
        return q8_0(x).tobytes()

    plan = []
    unmapped = []
    for name, dims in ref_tensors:
        if len(name.split(".")) > 2 and name.split(".", 2)[2] in SKIP_MTP_SMALLS:
            continue
        sub = ".".join(name.split(".")[2:]) if "." in name else ""
        if ref_types.get(name) == "F32" or len(dims) <= 1:
            kind = "F32"
            elems = int(np.prod(dims)) if dims else 1
        else:
            kind = "Q8_0"
            elems = int(np.prod(dims))
        n_bytes = (elems // 32 * 34) if kind == "Q8_0" else elems * 4
        keys = source_keys(name, dims)
        ok = [k for k in keys if k in index]
        if not ok:
            unmapped.append(name)
            continue
        plan.append((name, dims, kind, n_bytes, ok))
    if unmapped:
        print(f"UNMAPPED {len(unmapped)}:", flush=True)
        for n in unmapped[:30]:
            print("  ", n, flush=True)
        return 1
    global REF_DIMS
    REF_DIMS = [(p[0], p[1]) for p in plan]
    total = sum(p[3] for p in plan)
    print(f"plan: {len(plan)} tensors, {total/1e9:.1f} GB payload", flush=True)
    if a.dry_run:
        return 0

    writer = gguf.GGUFWriter(path=None, arch="glm5-next", use_temp_file=False)
    for name, (vtype, sub_type, val) in ref_kv.items():
        writer.add_key_value(name, val, vtype, sub_type=sub_type)
    writer.add_key_value("general.name", "GLM-5.3-Flash-UNCENSORED", gguf.GGUFValueType.STRING)
    for name, dims, kind, n_bytes, ok in plan:
        np_shape = dims[::-1]
        if kind == "Q8_0":
            # the writer expects the BYTE shape for raw quantized tensors: the last dim
            # (elements/32 blocks x 34 B) converts back to the element dims on load
            byte_shape = np_shape[:-1] + [np_shape[-1] // 32 * 34]
            writer.add_tensor_info(name, byte_shape, np.dtype("uint8"), n_bytes,
                                   raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            # F32: no raw_dtype - the writer keeps the element shape as-is
            writer.add_tensor_info(name, np_shape, np.dtype("float32"), n_bytes)
    writer.write_header_to_file(path=a.out)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    done = 0
    for name, dims, kind, n_bytes, ok in plan:
        tb = time.time()
        payload = produce(name, dims, kind)
        assert len(payload) == n_bytes, (name, len(payload), n_bytes)
        # gguf-py 自己的流式写入: 按 add_tensor_info 的顺序弹出并写数据+对齐,
        # 与 TI 表里的 offset 累计天然一致(手写 fout.write 曾与库的累计差一个张量)
        writer.write_tensor_data(np.frombuffer(payload, dtype=np.uint8))
        done += 1
        print(f"  [{done}/{len(plan)}] {name} {time.time()-tb:.0f} s", flush=True)
    writer.close()
    print(f"written: {a.out} in {time.time()-t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
