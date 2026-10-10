#!/usr/bin/env python3
"""Golden vectors: a literal numpy port of the reference functions in DeepSeek-V4-Flash inference/model.py
(Gate.forward, precompute_freqs_cis, apply_rotary_emb, get_window_topk_idxs, get_compress_topk_idxs, Expert swiglu).
Usage: python3 tests/gen_golden.py > tests/golden.txt"""
import math
import numpy as np

rng = np.random.default_rng(1234)
out = []

def emit(name, arr, ints=False):
    a = np.asarray(arr).ravel()
    out.append("%s %d %s" % (name, a.size, " ".join(str(int(v)) if ints else "%.9g" % v for v in a)))

def softplus(x): return np.logaddexp(0.0, x)

def gate(logits, topk, func, bias, hash_idx, scale):
    s = {"softmax": lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum(),
         "sigmoid": lambda v: 1 / (1 + np.exp(-v)),
         "sqrtsoftplus": lambda v: np.sqrt(softplus(v))}[func](logits.astype(np.float64))
    orig = s
    if bias is not None: s = s + bias
    idx = np.array(hash_idx) if hash_idx is not None else np.argsort(-s, kind="stable")[:topk]
    w = orig[idx]
    if func != "softmax": w = w / w.sum()
    return idx, w * scale

for name, func, use_bias, use_hash in [("sqrtsoftplus", "sqrtsoftplus", True, False), ("hash", "sqrtsoftplus", False, True),
                                       ("softmax", "softmax", False, False), ("sigmoid", "sigmoid", True, False)]:
    lg = rng.normal(0, 2, 256).astype(np.float32)
    bias = rng.normal(0, 0.5, 256).astype(np.float32) if use_bias else None
    h = [17, 3, 200, 41, 99, 250] if use_hash else None
    idx, w = gate(lg, 6, func, bias, h, 1.5)
    emit("gate_%s_logits" % name, lg)
    if use_bias: emit("gate_%s_bias" % name, bias)
    emit("gate_%s_idx" % name, idx, ints=True)
    emit("gate_%s_w" % name, w)

def yarn(dim, seqlen, orig, base, factor, bf, bs):
    def fcd(nr, dim, base, msl): return dim * math.log(msl / (nr * 2 * math.pi)) / (2 * math.log(base))
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if orig > 0:
        low = max(math.floor(fcd(bf, dim, base, orig)), 0)
        high = min(math.ceil(fcd(bs, dim, base, orig)), dim - 1)
        mx = high + 0.001 if low == high else high
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (mx - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    f = np.outer(np.arange(seqlen), freqs)
    return np.cos(f), np.sin(f)

for tag, orig, base in [("yarn", 65536, 160000.0), ("plain", 0, 10000.0)]:
    c, s = yarn(64, 64, orig, base, 16.0, 32, 1)
    emit("rope_%s_cos" % tag, c); emit("rope_%s_sin" % tag, s)
    x = rng.normal(0, 1, 64).astype(np.float32)
    pos = 37
    z = (x[0::2] + 1j * x[1::2]) * (c[pos] + 1j * s[pos])
    zi = (x[0::2] + 1j * x[1::2]) * (c[pos] - 1j * s[pos])
    emit("rope_%s_x" % tag, x)
    for nm, zz in [("fwd", z), ("inv", zi)]:
        y = np.empty(64); y[0::2] = zz.real; y[1::2] = zz.imag
        emit("rope_%s_%s" % (tag, nm), y)

def window(win, seqlen, start_pos):
    if start_pos >= win - 1:
        sp = start_pos % win
        m = np.concatenate([np.arange(sp + 1, win), np.arange(0, sp + 1)])[None, :]
    elif start_pos > 0:
        m = np.pad(np.arange(start_pos + 1), (0, win - start_pos - 1), constant_values=-1)[None, :]
    else:
        base = np.arange(seqlen)[:, None]
        m = np.clip(base - win + 1, 0, None) + np.arange(min(seqlen, win))
        m = np.where(m > base, -1, m)
    return m

def compress(ratio, seqlen, start_pos, offset):
    if start_pos > 0:
        return (np.arange(0, (start_pos + 1) // ratio) + offset)[None, :]
    m = np.tile(np.arange(seqlen // ratio), (seqlen, 1))
    mask = m >= (np.arange(1, seqlen + 1)[:, None] // ratio)
    return np.where(mask, -1, m + offset)

for nm, (w, s, p) in {"win_prefill": (8, 11, 0), "win_short": (8, 1, 3), "win_wrap": (8, 1, 13), "win_exact": (8, 1, 7)}.items():
    m = window(w, s, p); emit(nm + "_shape", m.shape, ints=True); emit(nm, m, ints=True)
for nm, (r, s, p, o) in {"cmp_prefill": (4, 11, 0, 11), "cmp_decode": (4, 1, 11, 8), "cmp_decode2": (128, 1, 300, 128)}.items():
    m = compress(r, s, p, o); emit(nm + "_shape", m.shape, ints=True); emit(nm, m, ints=True)

g = rng.normal(0, 8, 64).astype(np.float32); u = rng.normal(0, 8, 64).astype(np.float32)
gc = np.minimum(g, 10.0); uc = np.clip(u, -10.0, 10.0)
emit("swiglu_gate", g); emit("swiglu_up", u)
emit("swiglu_out", gc / (1 + np.exp(-gc.astype(np.float64))) * uc)

# ---------- hyper-connections (kernel.py hc_split_sinkhorn + model.py Block.hc_pre/hc_post, ParallelHead.hc_head) ----------
def sigmoid(v): return 1 / (1 + np.exp(-v))

def hc_split(mixes, scale, base, hc, iters, eps):
    pre = sigmoid(mixes[:hc] * scale[0] + base[:hc]) + eps
    post = 2 * sigmoid(mixes[hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[2 * hc:] * scale[2] + base[2 * hc:]).reshape(hc, hc)
    e = np.exp(comb - comb.max(axis=1, keepdims=True))
    comb = e / e.sum(axis=1, keepdims=True) + eps
    comb = comb / (comb.sum(axis=0, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=0, keepdims=True) + eps)
    return pre, post, comb

HC, D, ITERS, EPS, NEPS = 4, 16, 20, 1e-6, 1e-6
x = rng.normal(0, 1, (HC, D)); fn = rng.normal(0, 0.3, ((2 + HC) * HC, HC * D))
sc = rng.normal(0, 1, 3); bs = rng.normal(0, 0.5, (2 + HC) * HC)
xf = x.reshape(-1); rs = 1 / np.sqrt((xf ** 2).mean() + NEPS)
mixes = fn @ xf * rs
pre, post, comb = hc_split(mixes, sc, bs, HC, ITERS, EPS)
y = (pre[:, None] * x).sum(axis=0)
sub = rng.normal(0, 1, D)                      # sublayer output
hp = post[:, None] * sub[None, :] + (comb[:, :, None] * x[:, None, :]).sum(axis=0)   # y[k] = post[k]*sub + sum_j comb[j,k]*res[j]
emit("hc_x", x); emit("hc_fn", fn); emit("hc_scale", sc); emit("hc_base", bs)
emit("hc_pre_out", y); emit("hc_post_w", post); emit("hc_comb", comb); emit("hc_pre_w", pre)
emit("hc_sub", sub); emit("hc_post_out", hp)
hfn = rng.normal(0, 0.3, (HC, HC * D)); hsc = rng.normal(0, 1, 1); hbs = rng.normal(0, 0.5, HC)
hm = hfn @ xf * rs
hpre = sigmoid(hm * hsc[0] + hbs) + EPS
emit("hch_fn", hfn); emit("hch_scale", hsc); emit("hch_base", hbs)
emit("hch_out", (hpre[:, None] * x).sum(axis=0))

# ---------- sparse attention with sink (kernel.py sparse_attn_kernel) ----------
H, DD, N, TK = 4, 16, 10, 7
q = rng.normal(0, 1, (H, DD)); kv = rng.normal(0, 1, (N, DD)); sink = rng.normal(0, 1, H)
idxs = np.array([3, 0, 9, -1, 5, 5 - 5 + 7, -1])
valid = idxs >= 0
sc_ = (q @ kv[idxs[valid]].T) * (DD ** -0.5)                       # [H, nvalid]
m = sc_.max(axis=1, keepdims=True)
e = np.exp(sc_ - m)
den = e.sum(axis=1, keepdims=True) + np.exp(sink[:, None] - m)
o = (e / den) @ kv[idxs[valid]]
emit("sa_q", q); emit("sa_kv", kv); emit("sa_sink", sink); emit("sa_idx", idxs, ints=True); emit("sa_out", o)
print("\n".join(out))
