// metal/tests/qsa_q4_parity.cpp - the q4 decode window of qsa.metal (function constant QSA_KV_Q4 = 25) against the
// PC semantics of src/core/layer.cpp's --kv q4_0 path, with the CPU reference of kv_q4_ref.h (the device-order FWHT
// and q4 packing).  qsa_parity.cpp stays the fc-off gate; this drives the same kernels with fc 25 on:
//
//   qsa_q4_parity [P]   a context of P cells (default 37) built by the fc=25 qsa_prep, then a window of 3 tokens:
//                       prep, score, top-k, attention, merge.  Checked:
//                         * the V pool's 18-byte blocks BIT-LEVEL against q4_block_ref(fwht256_ref(vcur)) - the
//                           fused cross-simdgroup FWHT + q4 pack chain.  The v rows skip norm/RoPE, so the kernel's
//                       * input bits are known exactly.  One cell's v rows are all zero: its blocks pin the
//                           documented all-zero bits (d = -0.0 -> 0x8000, every qs byte 0x88);
//                         * the K pool DEQUANTIZED against the double-precision norm + RoPE + FWHT reference at a
//                           quantization tolerance (q4's code step is max|x|/8; qsa_parity's int8 check at 6e-3 is
//                           the same derivation at max|x|/127).  K passes through norm/RoPE on the GPU in fp32, so
//                           its block bytes are not end-to-end bit-comparable - V above is;
//                         * qcur (the rotated q attention rows) at 1e-5 and qidx at 1e-5 - qidx must come out
//                           UNROTATED (asserted directly).  pooled / dead are NOT directly asserted: they are
//                           covered transitively, through the score check (its reference reads the GPU's
//                           pooled/dead rows) and the tail's +1e9 bias assertion; the scores/top-k are exactly
//                           int8's math on the unrotated indexer chain;
//                         * attention + merge from the GPU's own qcur, ids and pool bytes: the reference dequantizes
//                           both pools, softmaxes in double, rotates the 256-wide output back with fwht256_ref
//                           BEFORE the sigmoid(gate) multiply (the placement layer.cpp uses; the wrong order -
//                           gate first - is printed for contrast and fails by orders of magnitude).
// Fixed RNG (ref::Rng); this Mac is its own reference.
#include "mtl_rt.h"
#include "ref_quant.h"

#include "../../include/strata/kernels/f16_bits.hpp"
#include "kv_q4_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
using namespace kvq4ref;
constexpr int QH = 24, QKV = 2, QD = 256, IH = 4, ID = 128, BUDGET = 2051, CH = 64, NCH = (BUDGET + CH - 1) / CH;
constexpr int ROWB = ROW_BYTES;   // 144 bytes per (cell, kv head) pool row
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
struct Err {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};
}  // namespace

int main(int argc, char** argv) {
    int P = 37;
    for (int i = 1; i < argc; ++i) P = std::atoi(argv[i]);
    const int MAXT = 8, MAXC = std::max(4096, (P + 64) / 4 * 4 + 64), MB = MAXC / 4 + 2, NC = P + MAXT;
    mtl::Context cx;
    // kv_q4.metal first: its shared prelude is qsa.metal's q4 numerics (metal/'s concatenation convention)
    const int lib = cx.compile({"kv_q4.metal", "qsa.metal"}, "#define NO_GGML_COMMON 1\n");
    // fc 25 on (and fc 24, M-RoPE, explicitly off): prep / attn / merge carry the q4 mode; score and top-k do not
    const mtl::Constants cq4 = mtl::Constants().b(24, false).b(25, true);
    mtl::Pipeline pprep = cx.pipeline(lib, "qsa_prep", cq4), pscore = cx.pipeline(lib, "qsa_score"),
                  ptopk = cx.pipeline(lib, "qsa_topk"), pattn = cx.pipeline(lib, "qsa_attn", cq4),
                  pmerge = cx.pipeline(lib, "qsa_merge", cq4);
    mtl::Buffer qfull = cx.alloc(4 * (size_t)NC * QH * 2 * QD), kc = cx.alloc(4 * (size_t)NC * QKV * QD),
                vc = cx.alloc(4 * (size_t)NC * QKV * QD), iq = cx.alloc(4 * (size_t)NC * IH * ID),
                ik = cx.alloc(4 * (size_t)NC * ID), qn = cx.alloc(4 * QD), kn = cx.alloc(4 * QD), iqn = cx.alloc(4 * ID),
                ikn = cx.alloc(4 * ID), cs = cx.alloc(8 * (size_t)(MAXC + 16) * 32), qcur = cx.alloc(4 * MAXT * QH * QD),
                qidx = cx.alloc(4 * MAXT * IH * ID),
                // the q4 pools (bytes): [cell][kv head][144]; ksc/vsc still bound, unused by the fc-25 kernels
                kq = cx.alloc((size_t)MAXC * QKV * ROWB), vq = cx.alloc((size_t)MAXC * QKV * ROWB),
                ksc = cx.alloc(2 * (size_t)MAXC * QKV * 4), vsc = cx.alloc(2 * (size_t)MAXC * QKV * 4),
                tail = cx.alloc(4 * 4 * ID), dead = cx.alloc(4 * ID), pooled = cx.alloc(4 * (size_t)MB * ID),
                scores = cx.alloc(4 * (size_t)MAXT * MB), ids = cx.alloc(4 * MAXT * BUDGET),
                part = cx.alloc(4 * (size_t)MAXT * 2 * NCH * QH * (QD + 2)),
                xp = cx.alloc(4 * (4 * MAXT * QH * QD + MAXT * QH * QD / 16));
    ref::Rng r(23);
    for (size_t i = 0; i < qfull.bytes / 4; ++i) qfull.as<float>()[i] = r.sym() * 2.0f;
    for (size_t i = 0; i < kc.bytes / 4; ++i) kc.as<float>()[i] = r.sym() * 2.0f;
    for (size_t i = 0; i < vc.bytes / 4; ++i) vc.as<float>()[i] = r.sym();
    for (size_t i = 0; i < iq.bytes / 4; ++i) iq.as<float>()[i] = r.sym();
    for (size_t i = 0; i < ik.bytes / 4; ++i) ik.as<float>()[i] = r.sym();
    for (int i = 0; i < QD; ++i) { qn.as<float>()[i] = 1.0f + 0.2f * r.sym(); kn.as<float>()[i] = 1.0f + 0.2f * r.sym(); }
    for (int i = 0; i < ID; ++i) { iqn.as<float>()[i] = 1.0f + 0.2f * r.sym(); ikn.as<float>()[i] = 1.0f + 0.2f * r.sym(); }
    const uint32_t pos_base = 0;
    for (int p = 0; p < MAXC + 16; ++p)
        for (int i = 0; i < 32; ++i) {
            const double inv = std::pow(1e7, -2.0 * i / 64.0), th = (double)p * inv;
            cs.as<float>()[(p * 32 + i) * 2] = (float)std::cos(th);
            cs.as<float>()[(p * 32 + i) * 2 + 1] = (float)std::sin(th);
        }
    // the fixture cell: both v rows all zero (kc stays random) - its blocks pin the all-zero q4 bits
    const int zcell = 5;
    for (int h = 0; h < QKV; ++h)
        for (int i = 0; i < QD; ++i) vc.as<float>()[(size_t)zcell * QKV * QD + (size_t)h * QD + i] = 0.0f;
    auto prep = [&](uint32_t pos0, int nt) {
        QsaArgs a{pos0, pos_base, (uint32_t)MAXC, (uint32_t)nt, 1e-6f};
        mtl::Dispatch d;
        d.p = &pprep;
        d.args = {mtl::Arg::B(qfull, 4 * (size_t)pos0 * QH * 2 * QD), mtl::Arg::B(kc, 4 * (size_t)pos0 * QKV * QD),
                  mtl::Arg::B(vc, 4 * (size_t)pos0 * QKV * QD), mtl::Arg::B(iq, 4 * (size_t)pos0 * IH * ID),
                  mtl::Arg::B(ik, 4 * (size_t)pos0 * ID), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(iqn),
                  mtl::Arg::B(ikn), mtl::Arg::B(cs), mtl::Arg::B(qcur), mtl::Arg::B(qidx), mtl::Arg::B(kq),
                  mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc), mtl::Arg::B(tail), mtl::Arg::B(dead),
                  mtl::Arg::B(pooled), mtl::Arg::V(a), mtl::Arg::B(ksc) /* mtab unused, fc 24 off */};
        d.grid[0] = 33;
        d.grid[1] = nt;
        d.tg[0] = 256;
        return d;
    };
    // the context, one cell at a time
    {
        std::vector<mtl::Dispatch> ds;
        for (int p = 0; p < P; ++p) ds.push_back(prep(p, 1));
        cx.run(ds);
    }
    const int NT = 3;
    std::printf("device %s, context %d cells, QSA_KV_Q4 (fc 25) on\n", cx.device_name().c_str(), P);
    int fails = 0;

    // ---- the norm + RoPE reference (qsa_parity.cpp's, double precision); rope = false gives the indexer forms
    auto normrope = [&](const float* x, const float* g, int w, int absp, bool rope, std::vector<double>& out) {
        double ss = 0;
        for (int i = 0; i < w; ++i) ss += (double)x[i] * x[i];
        const double rs = 1.0 / std::sqrt(ss / w + 1e-6);
        out.assign(w, 0.0);
        for (int i = 0; i < w; ++i) out[i] = rs * x[i] * g[i];
        if (rope)
            for (int i = 0; i < 32; ++i) {
                const double c = cs.as<float>()[(absp * 32 + i) * 2], s = cs.as<float>()[(absp * 32 + i) * 2 + 1];
                const double a = out[i], b = out[i + 32];
                out[i] = a * c - b * s;
                out[i + 32] = a * s + b * c;
            }
    };
    // the PC's per-row rotation applied on the CPU: the device-order fwht of the float-cast row
    auto rotate_row = [&](const std::vector<double>& x, std::vector<float>& o) {
        o.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) o[i] = (float)x[i];
        fwht256_ref(o.data(), o.data(), 1);
    };

    // ---- 1. the V pool bit-level: q4_block_ref(fwht256_ref(vcur)) for every context + window cell
    {
        cx.run({prep(P, NT)});
        const int ncells = P + NT;   // the written cells: the context's P plus the window's NT
        std::vector<float> vrot((size_t)ncells * QKV * QD);
        fwht256_ref(vc.as<float>(), vrot.data(), (int64_t)ncells * QKV);
        long bad = 0, nblocks = 0;
        size_t first = (size_t)-1;
        for (int c = 0; c < ncells; ++c)
            for (int h = 0; h < QKV; ++h) {
                const size_t row = ((size_t)c * QKV + h) * ROWB;
                for (int b = 0; b < BPH; ++b, ++nblocks) {
                    block_q4_0 blk;
                    q4_block_ref(vrot.data() + ((size_t)c * QKV + h) * QD + b * 32, blk);
                    if (std::memcmp((const uint8_t*)vq.ptr + row + (size_t)b * 18, &blk, 18) != 0) {
                        if (first == (size_t)-1) first = row + (size_t)b * 18;
                        ++bad;
                    }
                }
            }
        std::printf("V pool vs q4_block_ref(fwht256_ref(vcur)): %ld of %ld 18-byte blocks differ (bits)  %s\n", bad,
                    nblocks, bad ? "FAIL" : "PASS");
        if (first != (size_t)-1) std::printf("  first differing byte offset %zu\n", first);
        fails += bad != 0;
        // the all-zero fixture cell's documented bits: d = 0 / -8 = -0.0 -> 0x8000, id = 0 -> q = trunc(8.5) = 8
        // -> every qs byte 0x88 (both heads, all 8 blocks)
        int zbad = 0;
        for (int h = 0; h < QKV; ++h)
            for (int b = 0; b < BPH; ++b) {
                const block_q4_0* zb =
                    (const block_q4_0*)((const uint8_t*)vq.ptr + ((size_t)zcell * QKV + h) * ROWB) + b;
                zbad += zb->d != 0x8000;
                for (int t = 0; t < 16; ++t) zbad += zb->qs[t] != 0x88;
            }
        std::printf("all-zero v cell (cell %d) blocks: d = 0x8000, qs = 0x88               %s\n", zcell,
                    zbad ? "FAIL" : "PASS");
        fails += zbad != 0;
    }

    // ---- 2. prep's other outputs: the rotated q, the UNROTATED indexer q, the K pool dequantized
    {
        Err eq, ei, ekv;
        std::vector<double> o;
        std::vector<float> orot;
        for (int t = 0; t < NT; ++t) {
            const int cell = P + t;
            for (int h = 0; h < QH; ++h) {
                normrope(qfull.as<float>() + ((size_t)cell * QH + h) * 2 * QD, qn.as<float>(), QD, cell, true, o);
                rotate_row(o, orot);   // q4: the stored qcur is the rotated row (Hq to pair with the Hk pool)
                for (int i = 0; i < QD; ++i) eq.add(qcur.as<float>()[(t * QH + h) * QD + i], orot[i]);
            }
            for (int h = 0; h < IH; ++h) {   // the indexer q NEVER rotates
                normrope(iq.as<float>() + ((size_t)cell * IH + h) * ID, iqn.as<float>(), ID, cell, true, o);
                for (int i = 0; i < ID; ++i) ei.add(qidx.as<float>()[(t * IH + h) * ID + i], o[i]);
            }
            for (int h = 0; h < QKV; ++h) {
                normrope(kc.as<float>() + ((size_t)cell * QKV + h) * QD, kn.as<float>(), QD, cell, true, o);
                rotate_row(o, orot);
                for (int i = 0; i < QD; ++i) {
                    const size_t blk = ((size_t)cell * QKV + h) * ROWB + (size_t)(i / 32) * 18;
                    const block_q4_0* qb = (const block_q4_0*)((const uint8_t*)kq.ptr + blk);
                    const int j = i & 31, byte = qb->qs[j < 16 ? j : j - 16];
                    const double deq = (j < 16 ? (byte & 0x0F) : (byte >> 4)) - 8.0;
                    ekv.add(deq * strata::kernels::f32_from_f16(qb->d), orot[i]);
                }
            }
        }
        // q4's code step is max|x|/8 -> rms error max|x|/(8 sqrt 12) per block, ~0.08 of a rotated row's rms for
        // this fixture: a quantization tolerance, the int8 check's 6e-3 at its step max|x|/127
        const bool ok = eq.l2() < 1e-5 && ei.l2() < 1e-5 && ekv.l2() < 0.12;
        fails += !ok;
        std::printf("prep: q (rotated) rel L2 %.2e, indexer q (unrotated) %.2e, K deq4 (vs float) %.2e  %s\n", eq.l2(),
                    ei.l2(), ekv.l2(), ok ? "PASS" : "FAIL");
    }

    // ---- 3. score + top-k: unchanged math on the unrotated indexer chain, ids exactly
    QsaArgs a{(uint32_t)P, pos_base, (uint32_t)MAXC, (uint32_t)NT, 1e-6f};
    {
        mtl::Dispatch s;
        s.p = &pscore;
        s.args = {mtl::Arg::B(qidx), mtl::Arg::B(pooled), mtl::Arg::B(dead), mtl::Arg::B(scores), mtl::Arg::V(a)};
        s.grid[0] = ((P + NT) / 4 + 1 + 7) / 8;
        s.grid[1] = NT;
        s.tg[0] = 256;
        cx.run({s});
        Err es;
        for (int t = 0; t < NT; ++t) {
            const int n_kv = P + t + 1, n_bid = n_kv / 4;
            for (int b = 0; b <= n_bid; ++b) {
                const float* key = b == n_bid ? dead.as<float>() : pooled.as<float>() + b * ID;
                double sc = 0;
                for (int h = 0; h < IH; ++h) {
                    double d = 0;
                    for (int i = 0; i < ID; ++i) d += (double)qidx.as<float>()[(t * IH + h) * ID + i] * key[i];
                    sc += std::max(d, 0.0);
                }
                if (b == n_bid && (n_kv & 3)) {
                    if (!(scores.as<float>()[t * MB + b] >= 1e9f)) es.add(1.0, 0.0);
                } else {
                    es.add(scores.as<float>()[t * MB + b], sc);
                }
            }
        }
        const bool ok = es.l2() < 1e-5;
        fails += !ok;
        std::printf("score: rel L2 %.2e (tail +1e9 checked separately)                   %s\n", es.l2(),
                    ok ? "PASS" : "FAIL");
        mtl::Dispatch k;
        k.p = &ptopk;
        k.args = {mtl::Arg::B(scores), mtl::Arg::B(ids), mtl::Arg::V(a)};
        k.grid[0] = NT;
        k.tg[0] = 1024;
        cx.run({k});
        int bad = 0;
        for (int t = 0; t < NT; ++t) {
            const int n_kv = P + t + 1, width = std::min(n_kv, BUDGET);
            std::vector<int> cells(n_kv);
            for (int c = 0; c < n_kv; ++c) cells[c] = c;
            if (n_kv > BUDGET) {   // the selection active: the width largest block scores, lowest cells first
                const float* sc = scores.as<float>() + t * MB;
                std::stable_sort(cells.begin(), cells.end(), [&](int x, int y) { return sc[x / 4] > sc[y / 4]; });
                cells.resize(width);
                std::sort(cells.begin(), cells.end());
            }
            for (int i = 0; i < width; ++i) bad += ids.as<int>()[t * BUDGET + i] != cells[i];
        }
        fails += bad != 0;
        std::printf("top-k: %d of %d ids differ (context %d: selection %s)     %s\n", bad,
                    NT * std::min(P + 1, BUDGET), P, P + 1 > BUDGET ? "active" : "identity", bad ? "FAIL" : "PASS");
    }

    // ---- 4. attention + merge from the GPU's qcur, ids and pool bytes: dequantize, softmax in double, rotate
    // back BEFORE the gate; the gate-first order is printed for contrast (it is the wrong placement)
    {
        mtl::Dispatch at;
        at.p = &pattn;
        at.args = {mtl::Arg::B(qcur), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc),
                   mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)};
        at.grid[0] = NCH;
        at.grid[1] = QKV;
        at.grid[2] = NT;
        at.tg[0] = 256;
        mtl::Dispatch m;
        m.p = &pmerge;
        m.args = {mtl::Arg::B(part), mtl::Arg::B(qfull, 4 * (size_t)P * QH * 2 * QD), mtl::Arg::B(xp), mtl::Arg::V(a)};
        m.grid[0] = QH;
        m.grid[1] = NT;
        m.tg[0] = 256;
        cx.run({at, m});
        auto deq_row = [&](const mtl::Buffer& pool, int cell, int g, std::vector<double>& o) {
            o.assign(QD, 0.0);
            for (int i = 0; i < QD; ++i) {
                const block_q4_0* qb =
                    (const block_q4_0*)((const uint8_t*)pool.ptr + ((size_t)cell * QKV + g) * ROWB) + (i / 32);
                const int j = i & 31, byte = qb->qs[j < 16 ? j : j - 16];
                o[i] = ((j < 16 ? (byte & 0x0F) : (byte >> 4)) - 8.0) * strata::kernels::f32_from_f16(qb->d);
            }
        };
        Err eo, ew;   // eo: rotate-then-gate (the kernel); ew: gate-then-rotate (the wrong order, for contrast)
        std::vector<double> krow, vrow;
        for (int t = 0; t < NT; ++t) {
            const int width = std::min(P + t + 1, BUDGET);
            for (int h = 0; h < QH; ++h) {
                const int g = h / (QH / QKV);
                std::vector<double> s(width);
                double mx = -1e300;
                for (int c = 0; c < width; ++c) {
                    deq_row(kq, ids.as<int>()[t * BUDGET + c], g, krow);
                    double d = 0;
                    for (int i = 0; i < QD; ++i) d += (double)qcur.as<float>()[(t * QH + h) * QD + i] * krow[i];
                    s[c] = d / 16.0;
                    mx = std::max(mx, s[c]);
                }
                double l = 0;
                for (int c = 0; c < width; ++c) l += (s[c] = std::exp(s[c] - mx));
                std::vector<double> o(QD, 0.0);
                for (int c = 0; c < width; ++c) {
                    deq_row(vq, ids.as<int>()[t * BUDGET + c], g, vrow);
                    for (int i = 0; i < QD; ++i) o[i] += s[c] * vrow[i];
                }
                std::vector<float> orot(QD), wgated(QD);
                for (int i = 0; i < QD; ++i) orot[i] = (float)(o[i] / l);
                fwht256_ref(orot.data(), orot.data(), 1);   // the output back, the device-order inverse
                for (int i = 0; i < QD; ++i) {
                    const double gate = qfull.as<float>()[((size_t)(P + t) * QH + h) * 2 * QD + QD + i];
                    const double sig = 1.0 / (1.0 + std::exp(-gate));
                    eo.add(xp.as<float>()[t * QH * QD + h * QD + i], orot[i] * sig);
                    wgated[i] = (float)(o[i] / l * sig);   // the wrong order: gate first, rotate after
                }
                fwht256_ref(wgated.data(), wgated.data(), 1);
                for (int i = 0; i < QD; ++i) ew.add(xp.as<float>()[t * QH * QD + h * QD + i], wgated[i]);
            }
        }
        const bool ok = eo.l2() < 1e-5;
        fails += !ok;
        std::printf("attention + inverse rotation + gate: rel L2 %.2e                %s\n", eo.l2(), ok ? "PASS" : "FAIL");
        std::printf("  (contrast: gate-then-rotate, the wrong placement, is off by %.2e)\n", ew.l2());
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
