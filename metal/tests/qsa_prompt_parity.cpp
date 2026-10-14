// metal/tests/qsa_prompt_parity.cpp - qsa_attn_mm (qsa_prompt.metal, the matrix units, fp16 operands) against qsa_attn
// (fp32, itself checked by qsa_parity), both followed by qsa_merge: the attention output of T query tokens on random
// int8 caches, within the budget (all cells, causal) and past it (2051 selected cells per token); then the speed.
// The same two fast paths on Q4_0 caches (function constant 25, the fixtures host-built with kv_q4_ref.h so the
// pools' bytes are the reference arithmetic) against the fc-25 qsa_attn + qsa_merge chain - the decode qsa_q4_parity
// pinned bit-level - with the flash epilogue's rotation placement discriminated, and the k4-layout pair compiled
// (dispatched only where the hardware's cooperative layout is k4).
//   qsa_prompt_parity [--bench]
#include "coop_probe.h"
#include "kv_q4_ref.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {
constexpr int QH = 24, QKV = 2, QD = 256, QB = 2051, NCH = 33;
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const bool noflash = std::getenv("QSA_NOFLASH") != nullptr;
    mtl::Context cx;
    // kv_q4.metal first: qsa.metal's optional q4 mode (function constant 25) uses its shared prelude
    const int lib = cx.compile({"kv_q4.metal", "qsa.metal", "qsa_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());
    const int T = bench ? 2048 : 96, MAXC = 40000;
    mtl::Buffer kq = cx.alloc((size_t)MAXC * QKV * QD), vq = cx.alloc((size_t)MAXC * QKV * QD), ksc = cx.alloc(2 * (size_t)MAXC * QKV * 4),
                vsc = cx.alloc(2 * (size_t)MAXC * QKV * 4), qcur = cx.alloc(4 * (size_t)T * QH * QD), qh = cx.alloc(2 * (size_t)T * QH * QD),
                qfull = cx.alloc(4 * (size_t)T * QH * 2 * QD), ids = cx.alloc(4 * (size_t)T * QB),
                part = cx.alloc(4 * (size_t)T * NCH * QH * (QD + 2)), xa = cx.alloc(4 * ((size_t)4 * T * QH * QD + (size_t)T * QH * QD / 16)),
                xb = cx.alloc(4 * ((size_t)4 * T * QH * QD + (size_t)T * QH * QD / 16));
    ref::Rng r(5);
    for (size_t i = 0; i < kq.bytes; ++i) { kq.as<int8_t>()[i] = (int8_t)((int)(r.next() % 255) - 127); vq.as<int8_t>()[i] = (int8_t)((int)(r.next() % 255) - 127); }
    for (size_t i = 0; i < ksc.bytes / 2; ++i) { ksc.as<uint16_t>()[i] = ref::f32_to_f16(0.004f + 0.004f * r.uni()); vsc.as<uint16_t>()[i] = ref::f32_to_f16(0.004f + 0.004f * r.uni()); }
    for (size_t i = 0; i < (size_t)T * QH * QD; ++i) qcur.as<float>()[i] = r.sym() * 3.0f;
    for (size_t i = 0; i < (size_t)T * QH * 2 * QD; ++i) qfull.as<float>()[i] = r.sym();
    mtl::Pipeline pa = cx.pipeline(lib, "qsa_attn"), pm = cx.pipeline(lib, "qsa_merge"), pmm = cx.pipeline(lib, "qsa_attn_mm", mtl::Constants().b(7, coop_k4(cx)).i(8, std::getenv("QSA_STOP") ? std::atoi(std::getenv("QSA_STOP")) : 0)),
                  pq = cx.pipeline(lib, "qsa_q_f16"), pf = cx.pipeline(lib, "qsa_flash", mtl::Constants().b(7, coop_k4(cx)));
    mtl::Buffer xf = cx.alloc(2 * (size_t)T * QH * QD);
    std::printf("pipelines: qsa_attn_mm max threads %d, tg mem %d; qsa_flash max threads %d, tg mem %d\n", pmm.max_threads,
                pmm.static_tg_mem, pf.max_threads, pf.static_tg_mem);
    int fails = 0;
    for (uint32_t pos0 : {0u, 30000u}) {
        // ids: within the budget the identity; past it 2051 distinct ascending cells < n_kv
        for (int t = 0; t < T; ++t) {
            const uint32_t n_kv = pos0 + t + 1;
            int* o = ids.as<int>() + (size_t)t * QB;
            if (n_kv <= (uint32_t)QB) { for (uint32_t c = 0; c < n_kv; ++c) o[c] = (int)c; continue; }
            std::set<int> s;
            while ((int)s.size() < QB) s.insert((int)(r.next() % n_kv));
            int j = 0;
            for (int c : s) o[j++] = c;
        }
        QsaArgs a{pos0, 0, (uint32_t)MAXC, (uint32_t)T, 1e-6f};
        auto D = [&](const mtl::Pipeline& p, std::vector<mtl::Arg> args, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tg) {
            mtl::Dispatch d; d.p = &p; d.args = std::move(args); d.grid[0] = gx; d.grid[1] = gy; d.grid[2] = gz; d.tg[0] = tg; return d;
        };
        const uint32_t nq = T * QH * QD;
        std::vector<mtl::Dispatch> ref_ds = {
            D(pa, {mtl::Arg::B(qcur), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc), mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)}, NCH, QKV, T, 256),
            D(pm, {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(xa), mtl::Arg::V(a)}, QH, T, 1, 256)};
        std::vector<mtl::Dispatch> mm_ds = {
            D(pq, {mtl::Arg::B(qcur), mtl::Arg::B(qh), mtl::Arg::V(nq)}, (nq + 255) / 256, 1, 1, 256),
            D(pmm, {mtl::Arg::B(qh), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc), mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)}, NCH, QKV, T, 32),
            D(pm, {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(xb), mtl::Arg::V(a)}, QH, T, 1, 256)};
        const mtl::Dispatch fl = D(pf, {mtl::Arg::B(qh), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc),
                                        mtl::Arg::B(ids), mtl::Arg::B(qfull), mtl::Arg::B(xf), mtl::Arg::V(a)}, QKV, T, 1, 128);
        if (bench) {
            auto timeit = [&](const std::vector<mtl::Dispatch>& ds) {
                cx.warm(ds, 0.3);
                std::vector<double> t;
                for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds));
                return median(t);
            };
            std::printf("pos0 %5u, T %d: qsa_attn %.2f ms, qsa_attn_mm %.2f ms, merge %.2f ms; qsa_flash (no merge) %.2f ms\n", pos0,
                        T, timeit({ref_ds[0]}) * 1e3, timeit({mm_ds[1]}) * 1e3, timeit({ref_ds[1]}) * 1e3, timeit({fl}) * 1e3);
            continue;
        }
        cx.run(ref_ds);
        std::fill(part.as<float>(), part.as<float>() + part.bytes / 4, std::nanf(""));   // a kernel that writes nothing shows
        cx.run(mm_ds);
        if (std::getenv("QSA_DEBUG")) {
            // how much of the partials the matrix-unit kernel wrote, and token 5's first chunk (m, l, acc[0..3])
            size_t written = 0, nonfinite = 0;
            const float* pp = part.as<float>();
            for (size_t i = 0; i < (size_t)T * NCH * QH * (QD + 2); ++i) {
                if (!std::isnan(pp[i])) ++written;
                if (!std::isfinite(pp[i]) && !std::isnan(pp[i])) ++nonfinite;
            }
            const float* q = pp + ((size_t)5 * NCH * QH) * (QD + 2);
            std::printf("debug pos0 %u: %zu of %zu partial floats written (%zu inf); t5 h0: m %g l %g acc %g %g %g %g\n", pos0,
                        written, (size_t)T * NCH * QH * (QD + 2), nonfinite, q[QD], q[QD + 1], q[0], q[1], q[2], q[3]);
        }
        double se = 0, sr = 0;
        for (size_t i = 0; i < (size_t)T * QH * QD; ++i) {
            const double e = xb.as<float>()[i] - xa.as<float>()[i];
            se += e * e;
            sr += (double)xa.as<float>()[i] * xa.as<float>()[i];
        }
        const double rel = std::sqrt(se / sr);
        const bool ok = rel < 3e-3;
        fails += !ok;
        std::printf("pos0 %5u, T %d: qsa_attn_mm vs qsa_attn (after merge) rel L2 %.2e  %s\n", pos0, T, rel, ok ? "PASS" : "FAIL");
        if (noflash) continue;
        cx.run({fl});
        double sf = 0;
        for (size_t i = 0; i < (size_t)T * QH * QD; ++i) {
            const double e = ref::f16_to_f32(xf.as<uint16_t>()[i]) - xa.as<float>()[i];
            sf += e * e;
        }
        const double relf = std::sqrt(sf / sr);
        const bool okf = relf < 3e-3;
        fails += !okf;
        std::printf("pos0 %5u, T %d: qsa_flash (fp16 out) vs qsa_attn + merge rel L2 %.2e  %s\n", pos0, T, relf, okf ? "PASS" : "FAIL");
    }
    // qsa_score_mm vs qsa_score at pos0 30000: the scores of every block <= n_bid
    {
        const uint32_t pos0 = 30000, MB = MAXC / 4 + 2, nb = (pos0 + T) / 4 + 1;
        mtl::Buffer qidx = cx.alloc(4 * (size_t)T * 4 * 128), pooled = cx.alloc(4 * (size_t)MB * 128), dead = cx.alloc(4 * 128),
                    sa = cx.alloc(4 * (size_t)T * MB), sb = cx.alloc(4 * (size_t)T * MB);
        for (size_t i = 0; i < (size_t)T * 4 * 128; ++i) qidx.as<float>()[i] = r.sym();
        for (size_t i = 0; i < (size_t)MB * 128; ++i) pooled.as<float>()[i] = r.sym();
        for (int i = 0; i < 128; ++i) dead.as<float>()[i] = r.sym();
        QsaArgs a{pos0, 0, (uint32_t)MAXC, (uint32_t)T, 1e-6f};
        mtl::Pipeline ps = cx.pipeline(lib, "qsa_score"), psm = cx.pipeline(lib, "qsa_score_mm");
        mtl::Dispatch d1; d1.p = &ps; d1.args = {mtl::Arg::B(qidx), mtl::Arg::B(pooled), mtl::Arg::B(dead), mtl::Arg::B(sa), mtl::Arg::V(a)};
        d1.grid[0] = (nb + 7) / 8; d1.grid[1] = T; d1.tg[0] = 256;
        mtl::Dispatch d2; d2.p = &psm; d2.args = {mtl::Arg::B(qidx), mtl::Arg::B(pooled), mtl::Arg::B(dead), mtl::Arg::B(sb), mtl::Arg::V(a)};
        d2.grid[0] = (nb + 63) / 64; d2.grid[1] = (T + 15) / 16; d2.tg[0] = 128;
        if (bench) {
            auto timeit = [&](const mtl::Dispatch& d) {
                cx.warm({d}, 0.3);
                std::vector<double> t;
                for (int i = 0; i < 5; ++i) t.push_back(cx.run({d}));
                return median(t);
            };
            std::printf("score pos0 %u, T %d: qsa_score %.2f ms, qsa_score_mm %.2f ms\n", pos0, T, timeit(d1) * 1e3, timeit(d2) * 1e3);
        } else {
            cx.run({d1});
            cx.run({d2});
            double worst = 0;
            for (int t = 0; t < T; ++t) {
                const uint32_t nbid = (pos0 + t + 1) / 4;
                for (uint32_t b = 0; b <= nbid; ++b) {
                    const double x = sa.as<float>()[(size_t)t * MB + b], y = sb.as<float>()[(size_t)t * MB + b];
                    worst = std::max(worst, std::fabs(x - y) / (std::fabs(x) + 1.0));   // scores are O(10)
                }
            }
            const bool ok = worst < 1e-5;
            fails += !ok;
            std::printf("score pos0 %u, T %d: qsa_score_mm vs qsa_score worst rel %.2e  %s\n", pos0, T, worst, ok ? "PASS" : "FAIL");
        }
    }
    // ---- q4 (function constant 25): the same two fast paths on Q4_0 pools, against the fc-25 qsa_attn +
    // qsa_merge chain (the decode qsa_q4_parity pinned bit-level).  The fixtures are host-built with kv_q4_ref.h:
    // random fp32 rows through the device-order FWHT, then q4_block_ref into the [cell][QKV][144] pools, so the
    // pools' bytes ARE the reference arithmetic (a qsa_prep fc-25 pool would put the GPU's norm + RoPE in the
    // loop; the host route keeps the fixture exact).  The q rows rotate too - Hq pairs with the Hk pool, the
    // score math unchanged; qsa_q_f16 gives the fp16 operand as on the engine path.
    if (!bench) {
        constexpr int ROWB = kvq4ref::ROW_BYTES;   // 144 B per (cell, kv head)
        const int NC4 = 30000 + T;                 // every cell a pos0 below can name (n_kv <= pos0 + T)
        mtl::Buffer kq4 = cx.alloc((size_t)MAXC * QKV * ROWB), vq4 = cx.alloc((size_t)MAXC * QKV * ROWB),
                    qcur4 = cx.alloc(4 * (size_t)T * QH * QD), sc_ph = cx.alloc(16);   // the scale placeholder
        ref::Rng rq(77);   // a fresh stream: the int8 cases' inputs (so their outputs) stay byte-identical
        std::vector<float> src((size_t)4096 * QKV * QD), rot((size_t)4096 * QKV * QD);
        for (int b0 = 0; b0 < NC4; b0 += 4096) {   // a chunk at a time: the staging stays 8 MB
            const int n = std::min(4096, NC4 - b0);
            for (int rep = 0; rep < 2; ++rep) {   // K then V
                for (int i = 0; i < n * QKV * QD; ++i) src[i] = rq.sym() * 2.0f;
                kvq4ref::fwht256_ref(src.data(), rot.data(), (int64_t)n * QKV);
                for (int c = 0; c < n; ++c)
                    for (int h = 0; h < QKV; ++h)
                        for (int b = 0; b < QD / 32; ++b)
                            kvq4ref::q4_block_ref(rot.data() + ((size_t)c * QKV + h) * QD + b * 32,
                                                  *(kvq4ref::block_q4_0*)((uint8_t*)(rep ? vq4 : kq4).ptr +
                                                      ((size_t)(b0 + c) * QKV + h) * ROWB + (size_t)b * 18));
            }
        }
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < QH; ++h) {
                for (int i = 0; i < QD; ++i) src[i] = rq.sym() * 3.0f;
                kvq4ref::fwht256_ref(src.data(), qcur4.as<float>() + ((size_t)t * QH + h) * QD, 1);
            }
        const bool k4hw = coop_k4(cx);
        const mtl::Constants cq4 = mtl::Constants().b(25, true);
        mtl::Pipeline pa4 = cx.pipeline(lib, "qsa_attn", cq4), pm4 = cx.pipeline(lib, "qsa_merge", cq4),
                      pmm4 = cx.pipeline(lib, "qsa_attn_mm", mtl::Constants().b(7, k4hw).i(8, 0).b(25, true)),
                      pf4 = cx.pipeline(lib, "qsa_flash", mtl::Constants().b(7, k4hw).b(25, true)),
                      // the k4 (M5) layout forced true: compiled for real - the MSL must be valid - and its
                      // numerics dispatched only where the hardware's layout matches
                      pmm4k = cx.pipeline(lib, "qsa_attn_mm", mtl::Constants().b(7, true).i(8, 0).b(25, true)),
                      pf4k = cx.pipeline(lib, "qsa_flash", mtl::Constants().b(7, true).b(25, true));
        std::printf("q4 pipelines: qsa_attn_mm max threads %d, tg mem %d; qsa_flash max threads %d, tg mem %d\n",
                    pmm4.max_threads, pmm4.static_tg_mem, pf4.max_threads, pf4.static_tg_mem);
        auto deq4 = [&](const mtl::Buffer& pool, int cell, int g, std::vector<double>& o) {
            o.assign(QD, 0.0);
            for (int i = 0; i < QD; ++i) {
                const kvq4ref::block_q4_0* qb =
                    (const kvq4ref::block_q4_0*)((const uint8_t*)pool.ptr + ((size_t)cell * QKV + g) * ROWB) + (i / 32);
                const int j = i & 31, byte = qb->qs[j < 16 ? j : j - 16];
                o[i] = ((j < 16 ? (byte & 0x0F) : (byte >> 4)) - 8.0) * strata::kernels::f32_from_f16(qb->d);
            }
        };
        for (uint32_t pos0 : {0u, 30000u}) {
            for (int t = 0; t < T; ++t) {
                const uint32_t n_kv = pos0 + t + 1;
                int* o = ids.as<int>() + (size_t)t * QB;
                if (n_kv <= (uint32_t)QB) { for (uint32_t c = 0; c < n_kv; ++c) o[c] = (int)c; continue; }
                std::set<int> s;
                while ((int)s.size() < QB) s.insert((int)(rq.next() % n_kv));
                int j = 0;
                for (int c : s) o[j++] = c;
            }
            QsaArgs a{pos0, 0, (uint32_t)MAXC, (uint32_t)T, 1e-6f};
            auto D = [&](const mtl::Pipeline& p, std::vector<mtl::Arg> args, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tg) {
                mtl::Dispatch d; d.p = &p; d.args = std::move(args); d.grid[0] = gx; d.grid[1] = gy; d.grid[2] = gz; d.tg[0] = tg; return d;
            };
            const uint32_t nq = T * QH * QD;
            std::vector<mtl::Dispatch> ref_ds = {
                D(pa4, {mtl::Arg::B(qcur4), mtl::Arg::B(kq4), mtl::Arg::B(vq4), mtl::Arg::B(sc_ph), mtl::Arg::B(sc_ph), mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)}, NCH, QKV, T, 256),
                D(pm4, {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(xa), mtl::Arg::V(a)}, QH, T, 1, 256)};
            auto run_case = [&](mtl::Pipeline& pmm, mtl::Pipeline& pf, const char* tag) {
                std::vector<mtl::Dispatch> mm_ds = {
                    D(pq, {mtl::Arg::B(qcur4), mtl::Arg::B(qh), mtl::Arg::V(nq)}, (nq + 255) / 256, 1, 1, 256),
                    D(pmm, {mtl::Arg::B(qh), mtl::Arg::B(kq4), mtl::Arg::B(vq4), mtl::Arg::B(sc_ph), mtl::Arg::B(sc_ph), mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)}, NCH, QKV, T, 32),
                    D(pm4, {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(xb), mtl::Arg::V(a)}, QH, T, 1, 256)};
                cx.run(ref_ds);
                std::fill(part.as<float>(), part.as<float>() + part.bytes / 4, std::nanf(""));   // a kernel that writes nothing shows
                cx.run(mm_ds);
                double se = 0, sr = 0;
                for (size_t i = 0; i < (size_t)T * QH * QD; ++i) {
                    const double e = xb.as<float>()[i] - xa.as<float>()[i];
                    se += e * e;
                    sr += (double)xa.as<float>()[i] * xa.as<float>()[i];
                }
                const double rel = std::sqrt(se / sr);
                bool ok = rel < 3e-3;
                fails += !ok;
                std::printf("pos0 %5u, T %d: %s qsa_attn_mm vs fc-25 qsa_attn (after merge) rel L2 %.2e  %s\n",
                            pos0, T, tag, rel, ok ? "PASS" : "FAIL");
                if (noflash) return;
                const mtl::Dispatch fl = D(pf, {mtl::Arg::B(qh), mtl::Arg::B(kq4), mtl::Arg::B(vq4), mtl::Arg::B(sc_ph),
                                                mtl::Arg::B(sc_ph), mtl::Arg::B(ids), mtl::Arg::B(qfull), mtl::Arg::B(xf), mtl::Arg::V(a)},
                                           QKV, T, 1, 128);
                cx.run({fl});
                double sf = 0;
                for (size_t i = 0; i < (size_t)T * QH * QD; ++i) {
                    const double e = ref::f16_to_f32(xf.as<uint16_t>()[i]) - xa.as<float>()[i];
                    sf += e * e;
                }
                const double relf = std::sqrt(sf / sr);
                ok = relf < 3e-3;
                fails += !ok;
                std::printf("pos0 %5u, T %d: %s qsa_flash (fp16 out) vs fc-25 qsa_attn + merge rel L2 %.2e  %s\n",
                            pos0, T, tag, relf, ok ? "PASS" : "FAIL");
            };
            run_case(pmm4, pf4, "q4");
            if (k4hw) run_case(pmm4k, pf4k, "q4 k4 layout");
            else std::printf("k4 layout (fc 7 = true) q4 pipelines compiled; numerics skipped: this device decodes !k4\n");
            if (pos0 == 0u && !noflash) {
                // the flash epilogue's rotation placement: the double-precision reference of all three placements
                // over the last three tokens' heads - rotate-then-gate (the kernel's) close, gate-then-rotate and
                // no rotation off by far more than the 3e-3 bar (qsa_q4_parity's contrast, here asserted)
                double se = 0, sr = 0, sw = 0, sn = 0;
                std::vector<double> krow, vrow;
                for (int t = T - 3; t < T; ++t)
                    for (int h = 0; h < QH; ++h) {
                        const int g = h / (QH / QKV), width = std::min(t + 1, QB);
                        std::vector<double> s(width);
                        double mx = -1e300;
                        for (int c = 0; c < width; ++c) {
                            deq4(kq4, ids.as<int>()[(size_t)t * QB + c], g, krow);
                            double d = 0;
                            for (int i = 0; i < QD; ++i)
                                d += (double)qcur4.as<float>()[((size_t)t * QH + h) * QD + i] * krow[i];
                            s[c] = d / 16.0;
                            mx = std::max(mx, s[c]);
                        }
                        double l = 0;
                        for (int c = 0; c < width; ++c) l += (s[c] = std::exp(s[c] - mx));
                        std::vector<double> o(QD, 0.0);
                        for (int c = 0; c < width; ++c) {
                            deq4(vq4, ids.as<int>()[(size_t)t * QB + c], g, vrow);
                            for (int i = 0; i < QD; ++i) o[i] += s[c] * vrow[i];
                        }
                        std::vector<float> orot(QD), wg(QD);
                        std::vector<double> sig(QD), out(QD);
                        for (int i = 0; i < QD; ++i) {
                            out[i] = o[i] / l;
                            orot[i] = (float)out[i];
                            sig[i] = 1.0 / (1.0 + std::exp(-(double)qfull.as<float>()[((size_t)t * QH + h) * 2 * QD + QD + i]));
                            wg[i] = (float)(out[i] * sig[i]);
                        }
                        kvq4ref::fwht256_ref(orot.data(), orot.data(), 1);   // the kernel's order: rotate, gate
                        kvq4ref::fwht256_ref(wg.data(), wg.data(), 1);       // the wrong one: gate, rotate
                        for (int i = 0; i < QD; ++i) {
                            const double got = ref::f16_to_f32(xf.as<uint16_t>()[((size_t)t * QH + h) * QD + i]);
                            se += (got - orot[i] * sig[i]) * (got - orot[i] * sig[i]);
                            sr += orot[i] * sig[i] * (orot[i] * sig[i]);
                            sw += (got - wg[i]) * (got - wg[i]);
                            sn += (got - out[i] * sig[i]) * (got - out[i] * sig[i]);
                        }
                    }
                const double rc = std::sqrt(se / sr), rw1 = std::sqrt(sw / sr), rw2 = std::sqrt(sn / sr);
                const bool ok = rc < 3e-3 && rw1 > 3e-3 && rw2 > 3e-3;
                fails += !ok;
                std::printf("pos0 %5u: qsa_flash q4 epilogue: rotate-then-gate %.2e, gate-then-rotate %.2e, no rotation %.2e  %s\n",
                            pos0, rc, rw1, rw2, ok ? "PASS" : "FAIL");
            }
        }
    }
    if (!bench) std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
