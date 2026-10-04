#!/usr/bin/env python3
"""Builds a tiny DeepSeek-V4-shaped GGUF (same tensor names as the real UD-IQ1_M file) with random weights and runs a
float64 numpy forward (decode path of the official model.py) to produce reference logits.
  python3 tests/tiny_model.py OUTDIR      ->  OUTDIR/tiny.gguf, OUTDIR/ref_logits.bin, OUTDIR/tokens.txt
QAT activation simulation is OFF in the reference: run dsv4_run with --no-qat-sim to compare."""
import math, os, struct, sys
import numpy as np

rng = np.random.default_rng(7)
# ---- tiny config ----
DIM, NH, HD, RD, QL, OL, G, WIN = 64, 4, 64, 16, 32, 16, 2, 4
HC, ITERS, EPS, HEPS = 4, 5, 1e-6, 1e-6
E, K, FF, NHASH = 8, 2, 32, 1
IH, ID, ITOPK = 4, 32, 2
NL, VOCAB, CTX = 4, 40, 64
RATIOS = [0, 4, 8, 4, 0]
YORIG, YFACT, BFAST, BSLOW, BASE_C, BASE_P = 32, 4.0, 32.0, 1.0, 1000.0, 100.0
ESCALE, CLAMP_E, CLAMP_S = 1.5, 3.0, 4.0
GIN = NH * HD // G
MIX = (2 + HC) * HC

# ---- GGUF writer ----
def s_(b): return struct.pack("<Q", len(b)) + b
class GW:
    def __init__(s): s.kv = []; s.t = []
    def u32(s, k, v): s.kv.append(s_(k.encode()) + struct.pack("<II", 4, v))
    def f32(s, k, v): s.kv.append(s_(k.encode()) + struct.pack("<If", 6, v))
    def b(s, k, v): s.kv.append(s_(k.encode()) + struct.pack("<IB", 7, int(v)))
    def st(s, k, v): s.kv.append(s_(k.encode()) + struct.pack("<I", 8) + s_(v.encode()))
    def ai32(s, k, v): s.kv.append(s_(k.encode()) + struct.pack("<IIQ", 9, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v))
    def af32(s, k, v): s.kv.append(s_(k.encode()) + struct.pack("<IIQ", 9, 6, len(v)) + b"".join(struct.pack("<f", x) for x in v))
    def astr(s, k, v): s.kv.append(s_(k.encode()) + struct.pack("<IIQ", 9, 8, len(v)) + b"".join(s_(x.encode()) for x in v))
    def tensor(s, name, dims, gtype, raw): s.t.append((name, dims, gtype, raw))
    def write(s, path):
        out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(s.t), len(s.kv)))
        for k in s.kv: out += k
        off = 0; infos = bytearray(); datas = []
        for name, dims, gt, raw in s.t:
            infos += s_(name.encode()) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", gt, off)
            pad = (-len(raw)) % 32; datas.append(raw + b"\0" * pad); off += len(raw) + pad
        out += infos; out += b"\0" * ((-len(out)) % 32)
        open(path, "wb").write(bytes(out) + b"".join(datas))

def q8_0(w):  # returns (raw bytes, dequantised float64 array as the C++ will see it)
    flat = w.astype(np.float32).reshape(-1, 32); raw = bytearray(); deq = np.empty_like(flat, dtype=np.float64)
    for i, blk in enumerate(flat):
        amax = np.abs(blk).max(); d = np.float16(amax / 127.0 if amax > 0 else 0.0); dd = float(d)
        q = np.clip(np.rint(blk / dd), -127, 127).astype(np.int8) if dd > 0 else np.zeros(32, np.int8)
        raw += d.tobytes() + q.tobytes(); deq[i] = dd * q
    return bytes(raw), deq.reshape(w.shape)
def bf16(w):
    u = (w.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return u.tobytes(), (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

gw = GW(); W = {}
def add(name, arr, kind="f32"):  # arr in row-major [out][in] (or [E][out][in]); GGUF dims are reversed
    a = np.asarray(arr, dtype=np.float32)
    dims = list(reversed(a.shape))
    if kind == "q8": raw, deq = q8_0(a); gt = 8
    elif kind == "bf16": raw, deq = bf16(a); gt = 30
    elif kind == "i32": raw, deq = np.asarray(arr, dtype="<i4").tobytes(), np.asarray(arr); gt = 26; dims = list(reversed(np.asarray(arr).shape))
    else: raw, deq = a.astype("<f4").tobytes(), a.astype(np.float64); gt = 0
    gw.tensor(name, dims, gt, raw); W[name] = deq
def mat(o, i, s=1.0): return rng.normal(0, s / math.sqrt(i), (o, i))
def nrm(n): return 1.0 + 0.1 * rng.normal(0, 1, n)

gw.st("general.architecture", "deepseek4")
for k, v in [("block_count", NL), ("context_length", CTX), ("embedding_length", DIM), ("attention.head_count", NH), ("attention.head_count_kv", 1),
             ("attention.key_length", HD), ("attention.value_length", HD), ("rope.dimension_count", RD), ("attention.q_lora_rank", QL),
             ("attention.output_lora_rank", OL), ("attention.output_group_count", G), ("attention.sliding_window", WIN), ("expert_count", E),
             ("expert_used_count", K), ("expert_shared_count", 1), ("expert_feed_forward_length", FF), ("expert_gating_func", 4), ("hash_layer_count", NHASH),
             ("attention.indexer.head_count", IH), ("attention.indexer.key_length", ID), ("attention.indexer.top_k", ITOPK),
             ("hyper_connection.count", HC), ("hyper_connection.sinkhorn_iterations", ITERS), ("rope.scaling.original_context_length", YORIG)]:
    gw.u32("deepseek4." + k, v)
for k, v in [("expert_weights_scale", ESCALE), ("hyper_connection.epsilon", HEPS), ("attention.layer_norm_rms_epsilon", EPS), ("rope.freq_base", BASE_P),
             ("attention.compress_rope_freq_base", BASE_C), ("rope.scaling.factor", YFACT), ("rope.scaling.yarn_beta_fast", BFAST), ("rope.scaling.yarn_beta_slow", BSLOW)]:
    gw.f32("deepseek4." + k, v)
gw.b("deepseek4.expert_weights_norm", True)
gw.ai32("deepseek4.attention.compress_ratios", RATIOS)
gw.af32("deepseek4.swiglu_clamp_exp", [CLAMP_E] * NL); gw.af32("deepseek4.swiglu_clamp_shexp", [CLAMP_S] * NL)
gw.astr("tokenizer.ggml.tokens", ["t%d" % i for i in range(VOCAB)]); gw.u32("tokenizer.ggml.bos_token_id", 0); gw.u32("tokenizer.ggml.eos_token_id", VOCAB - 1)

add("token_embd.weight", rng.normal(0, 1, (VOCAB, DIM))); add("output.weight", mat(VOCAB, DIM)); add("output_norm.weight", nrm(DIM))
add("output_hc_fn.weight", rng.normal(0, 0.05, (HC, HC * DIM))); add("output_hc_base.weight", rng.normal(0, 0.5, HC)); add("output_hc_scale.weight", [0.8])
for l in range(NL):
    p = "blk.%d." % l; r = RATIOS[l]; coff = 1 + (r == 4)
    add(p + "attn_norm.weight", nrm(DIM)); add(p + "attn_q_a.weight", mat(QL, DIM)); add(p + "attn_q_a_norm.weight", nrm(QL))
    add(p + "attn_q_b.weight", mat(NH * HD, QL), "q8"); add(p + "attn_kv.weight", mat(HD, DIM), "q8"); add(p + "attn_kv_a_norm.weight", nrm(HD))
    add(p + "attn_output_a.weight", mat(G * OL, GIN)); add(p + "attn_output_b.weight", mat(DIM, G * OL), "q8"); add(p + "attn_sinks.weight", rng.normal(0, 1, NH))
    add(p + "ffn_norm.weight", nrm(DIM)); add(p + "ffn_gate_inp.weight", mat(E, DIM, 2.0), "bf16")
    if l < NHASH: add(p + "ffn_gate_tid2eid.weight", rng.integers(0, E, (VOCAB, K)), "i32")
    else: add(p + "exp_probs_b.bias", rng.normal(0, 0.3, E))
    add(p + "ffn_gate_exps.weight", rng.normal(0, 1 / math.sqrt(DIM), (E, FF, DIM)), "q8"); add(p + "ffn_up_exps.weight", rng.normal(0, 1 / math.sqrt(DIM), (E, FF, DIM)), "q8")
    add(p + "ffn_down_exps.weight", rng.normal(0, 1 / math.sqrt(FF), (E, DIM, FF)))
    add(p + "ffn_gate_shexp.weight", mat(FF, DIM)); add(p + "ffn_up_shexp.weight", mat(FF, DIM)); add(p + "ffn_down_shexp.weight", mat(DIM, FF))
    for nm in ("attn", "ffn"):
        add(p + "hc_%s_fn.weight" % nm, rng.normal(0, 0.05, (MIX, HC * DIM))); add(p + "hc_%s_base.weight" % nm, rng.normal(0, 0.5, MIX)); add(p + "hc_%s_scale.weight" % nm, rng.normal(0.5, 0.2, 3))
    if r:
        add(p + "attn_compressor_kv.weight", mat(coff * HD, DIM)); add(p + "attn_compressor_gate.weight", mat(coff * HD, DIM))
        add(p + "attn_compressor_ape.weight", rng.normal(0, 0.5, (r, coff * HD))); add(p + "attn_compressor_norm.weight", nrm(HD))
        if r == 4:
            add(p + "indexer.attn_q_b.weight", mat(IH * ID, QL)); add(p + "indexer.proj.weight", mat(IH, DIM))
            add(p + "indexer_compressor_kv.weight", mat(2 * ID, DIM)); add(p + "indexer_compressor_gate.weight", mat(2 * ID, DIM))
            add(p + "indexer_compressor_ape.weight", rng.normal(0, 0.5, (4, 2 * ID))); add(p + "indexer_compressor_norm.weight", nrm(ID))

# ---------------- numpy reference (float64) ----------------
def rmsn(x, w=None, eps=EPS): y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps); return y * w if w is not None else y
def yarn(dim, n, orig, base, factor, bf, bs):
    fcd = lambda nr: dim * math.log(orig / (nr * 2 * math.pi)) / (2 * math.log(base))
    fr = (1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))).astype(np.float64)
    if orig > 0:
        lo = max(math.floor(fcd(bf)), 0); hi = min(math.ceil(fcd(bs)), dim - 1); mx = hi + 0.001 if lo == hi else hi
        sm = 1 - np.clip((np.arange(dim // 2) - lo) / (mx - lo), 0, 1); fr = fr / factor * (1 - sm) + fr * sm
    f = np.outer(np.arange(n), fr); return np.cos(f), np.sin(f)
CY, SY = yarn(RD, CTX, YORIG, BASE_C, YFACT, BFAST, BSLOW); CP, SP = yarn(RD, CTX, 0, BASE_P, YFACT, BFAST, BSLOW)
def rot(x, c, s, inv=False):
    y = x.copy(); p = y[..., -RD:]; a = p[..., 0::2].copy(); b = p[..., 1::2].copy(); s = -s if inv else s
    p[..., 0::2] = a * c - b * s; p[..., 1::2] = a * s + b * c; return y
def had(x):
    n = x.shape[-1]; H = np.array([[1.0]])
    while H.shape[0] < n: H = np.kron(np.array([[1, 1], [1, -1]]), H)
    return x @ H * n ** -0.5
sig = lambda v: 1 / (1 + np.exp(-v))
def hc_split(mixes, sc, bs):
    pre = sig(mixes[:HC] * sc[0] + bs[:HC]) + HEPS; post = 2 * sig(mixes[HC:2 * HC] * sc[1] + bs[HC:2 * HC])
    c = (mixes[2 * HC:] * sc[2] + bs[2 * HC:]).reshape(HC, HC); e = np.exp(c - c.max(1, keepdims=True)); c = e / e.sum(1, keepdims=True) + HEPS
    c = c / (c.sum(0, keepdims=True) + HEPS)
    for _ in range(ITERS - 1): c = c / (c.sum(1, keepdims=True) + HEPS); c = c / (c.sum(0, keepdims=True) + HEPS)
    return pre, post, c
def hc_pre(h, fn, sc, bs):
    x = h.reshape(-1); mixes = fn @ x / np.sqrt((x * x).mean() + EPS); pre, post, comb = hc_split(mixes, sc, bs)
    return (pre[:, None] * h).sum(0), post, comb

class Comp:
    def __init__(s, ratio, d, rotate, wkv, wg, ape, norm):
        s.r, s.d, s.rot_, s.wkv, s.wg, s.ape, s.norm = ratio, d, rotate, wkv, wg, ape, norm
        s.ov = ratio == 4; s.coff = 1 + s.ov; s.kvs = np.zeros((s.coff * ratio, s.coff * d)); s.scs = np.full((s.coff * ratio, s.coff * d), -np.inf); s.cache = []
    def step(s, x, pos, C, S):
        r, d = s.r, s.d; kv = s.wkv @ x; sc = s.wg @ x + s.ape[pos % r]
        row = (r if s.ov else 0) + pos % r; s.kvs[row] = kv; s.scs[row] = sc
        if (pos + 1) % r: return
        if s.ov: v = np.concatenate([s.kvs[:r, :d], s.kvs[r:, d:]]); sg = np.concatenate([s.scs[:r, :d], s.scs[r:, d:]])
        else: v, sg = s.kvs, s.scs
        p = np.exp(sg - sg.max(0)); p /= p.sum(0); out = (v * p).sum(0)
        if s.ov: s.kvs[:r] = s.kvs[r:]; s.scs[:r] = s.scs[r:]
        out = rmsn(out, s.norm); fr = pos + 1 - r; out = rot(out, C[fr], S[fr])
        if s.rot_: out = had(out)
        s.cache.append(out)

state = []
for l in range(NL):
    p = "blk.%d." % l; r = RATIOS[l]; st = {"ring": np.zeros((WIN, HD))}
    if r:
        st["ac"] = Comp(r, HD, False, W[p + "attn_compressor_kv.weight"], W[p + "attn_compressor_gate.weight"], W[p + "attn_compressor_ape.weight"], W[p + "attn_compressor_norm.weight"])
        if r == 4: st["ic"] = Comp(4, ID, True, W[p + "indexer_compressor_kv.weight"], W[p + "indexer_compressor_gate.weight"], W[p + "indexer_compressor_ape.weight"], W[p + "indexer_compressor_norm.weight"])
    state.append(st)

def attention(l, x, pos):
    p = "blk.%d." % l; r = RATIOS[l]; st = state[l]; C, S = (CY, SY) if r else (CP, SP); c, s = C[pos], S[pos]
    qr = rmsn(W[p + "attn_q_a.weight"] @ x, W[p + "attn_q_a_norm.weight"]); q = (W[p + "attn_q_b.weight"] @ qr).reshape(NH, HD)
    q = rot(rmsn(q), c, s); kv = rot(rmsn(W[p + "attn_kv.weight"] @ x, W[p + "attn_kv_a_norm.weight"]), c, s)
    st["ring"][pos % WIN] = kv
    keys = [st["ring"][i] for i in (range(WIN) if pos >= WIN - 1 else range(pos + 1))]
    if r:
        st["ac"].step(x, pos, C, S); ncomp = (pos + 1) // r
        if r == 4:
            iq = (W[p + "indexer.attn_q_b.weight"] @ qr).reshape(IH, ID); iq = had(rot(iq, c, s))
            st["ic"].step(x, pos, C, S); wg = (W[p + "indexer.proj.weight"] @ x) * (ID ** -0.5 * IH ** -0.5)
            k = min(ITOPK, ncomp)
            if k > 0:
                kc = np.array(st["ic"].cache[:ncomp]); sc = (np.maximum(iq @ kc.T, 0) * wg[:, None]).sum(0)
                sel = sorted(range(ncomp), key=lambda t: (-sc[t], t))[:k]
            else: sel = []
        else: sel = list(range(ncomp))
        keys += [st["ac"].cache[j] for j in sel]
    Kk = np.array(keys); sc = (q @ Kk.T) * HD ** -0.5; sk = W[p + "attn_sinks.weight"]
    m = np.maximum(sc.max(1), sk); e = np.exp(sc - m[:, None]); den = e.sum(1) + np.exp(sk - m); o = (e / den[:, None]) @ Kk
    o = rot(o, c, s, inv=True).reshape(-1)
    t = np.concatenate([W[p + "attn_output_a.weight"][g * OL:(g + 1) * OL] @ o[g * GIN:(g + 1) * GIN] for g in range(G)])
    return W[p + "attn_output_b.weight"] @ t

def moe(l, x, tok):
    p = "blk.%d." % l; lg = W[p + "ffn_gate_inp.weight"] @ x; sc = np.sqrt(np.logaddexp(0, lg))
    if l < NHASH: idx = W[p + "ffn_gate_tid2eid.weight"][tok]
    else: idx = np.argsort(-(sc + W[p + "exp_probs_b.bias"]), kind="stable")[:K]
    w = sc[idx]; w = w / w.sum() * ESCALE; y = np.zeros(DIM)
    sw = lambda g, u, lim: g.clip(max=lim) / (1 + np.exp(-g.clip(max=lim))) * u.clip(-lim, lim)
    for k, e in enumerate(idx):
        g = W[p + "ffn_gate_exps.weight"][e] @ x; u = W[p + "ffn_up_exps.weight"][e] @ x
        y += W[p + "ffn_down_exps.weight"][e] @ (w[k] * sw(g, u, CLAMP_E))
    g = W[p + "ffn_gate_shexp.weight"] @ x; u = W[p + "ffn_up_shexp.weight"] @ x
    return y + W[p + "ffn_down_shexp.weight"] @ sw(g, u, CLAMP_S)

def forward(tok, pos):
    h = np.tile(W["token_embd.weight"][tok], (HC, 1))
    for l in range(NL):
        p = "blk.%d." % l
        y, post, comb = hc_pre(h, W[p + "hc_attn_fn.weight"], W[p + "hc_attn_scale.weight"], W[p + "hc_attn_base.weight"])
        a = attention(l, rmsn(y, W[p + "attn_norm.weight"]), pos); h = post[:, None] * a[None, :] + (comb[:, :, None] * h[:, None, :]).sum(0)
        y, post, comb = hc_pre(h, W[p + "hc_ffn_fn.weight"], W[p + "hc_ffn_scale.weight"], W[p + "hc_ffn_base.weight"])
        a = moe(l, rmsn(y, W[p + "ffn_norm.weight"]), tok); h = post[:, None] * a[None, :] + (comb[:, :, None] * h[:, None, :]).sum(0)
    x = h.reshape(-1); mixes = W["output_hc_fn.weight"] @ x / np.sqrt((x * x).mean() + EPS)
    pre = sig(mixes * W["output_hc_scale.weight"][0] + W["output_hc_base.weight"]) + HEPS
    return W["output.weight"] @ rmsn((pre[:, None] * h).sum(0), W["output_norm.weight"])

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny"; os.makedirs(out, exist_ok=True)
gw.write(os.path.join(out, "tiny.gguf"))
toks = [int(t) for t in rng.integers(1, VOCAB - 1, 26)]
logits = np.array([forward(t, i) for i, t in enumerate(toks)], dtype=np.float32)
logits.tofile(os.path.join(out, "ref_logits.bin")); open(os.path.join(out, "tokens.txt"), "w").write(",".join(map(str, toks)))
print("tiny model written to %s: %d tokens, logits %s, compress events: ratio4 every 4, ratio8 every 8, window wrap at 4" % (out, len(toks), logits.shape))
