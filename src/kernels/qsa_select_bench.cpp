// src/kernels/qsa_select_bench.cpp - the prompt path's QSA selection (qsa_select.hpp) timed per stage, block scores
// (the warp kernel, the sm80 tensor-core one, and the AMD tensor-core one) and top-k, for a batch of consecutive
// queries at a given context, and the scorers compared: score difference and how many selections differ (GPU,
// synthetic, no model).  Timing is a median of 9 samples of `reps` calls each, after 3 warmup samples.
// Usage: qsa_select_bench [context=131072] [queries=256] [reps=10] [capacity_cells]
// capacity_cells (the engine's --max-context): the score buffers and the top-k dispatch follow the CAPACITY
// (max_blocks = capacity / 4 + 2), the work follows the context. Default: capacity = context (max_blocks = ctx / 4 + 1).
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 131072;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : 0;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity > 0 ? capacity / 4 + 2 : ctx / 4 + 1, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // keys with a shared direction plus noise, so the scores have a spread like a real indexer's
    std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
    for (auto& x : dir) x = nd(rng);
    for (int64_t b = 0; b < max_blocks; ++b) {
        const float a = nd(rng);
        for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
    }
    for (auto& x : dead) x = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {   // qsa_step_fill's arithmetic (kept here so the bench links only qsa_select)
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_pooled = up(pooled);
    const float* d_dead = up(dead);
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *sc_old = nullptr, *sc_new = nullptr, *sc_wmma = nullptr;
    int32_t *ids_old = nullptr, *ids_new = nullptr;
    ck(cudaMalloc(&sc_old, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&sc_new, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&sc_wmma, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&ids_old, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_new, (size_t) (nq * cap) * 4), "malloc");
    const int64_t active = steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1;
    auto run_old = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_old, nullptr, active); };
    static bool tc_ok = true;
    auto run_new = [&] {
        if (!k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_new, nullptr, active)) {
            if (tc_ok) { std::printf("tc scorer refused (no tensor-core path on this device) - skipping that arm\n"); tc_ok = false; }
        }
    };
    static bool wmma_ok = true;
    auto run_wmma = [&] {   // the AMD tensor-core arm, staging + main + tail exactly as the prompt path calls it
        if (!k::qsa_block_scores_wmma(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_wmma, nullptr, active)) {
            if (wmma_ok) { std::printf("wmma scorer refused (not this architecture / not compiled in) - skipping that arm\n"); wmma_ok = false; }
        }
    };
    // an opt-in that quietly fell back cannot masquerade as the fast path: with STRATA_SELECT_WMMA=1 the wmma arm
    // MUST run (the same condition the prompt path's dispatch applies), otherwise a refusal is a skip, not a failure
    static const bool wmma_required = [] { const char* v = std::getenv("STRATA_SELECT_WMMA"); return v && std::atoi(v) != 0; }();
    run_old();
    run_new();
    run_wmma();
    if (wmma_required && !wmma_ok) return 2;
    k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr);
    k::qsa_block_topk(sc_new, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
    int32_t* ids_reg = nullptr;   // the register top-k on the OLD scores: must equal the reference exactly
    ck(cudaMalloc(&ids_reg, (size_t) (nq * cap) * 4), "malloc");
    k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr, active);
    ck(cudaDeviceSynchronize(), "warm");
    // compare: the other arms' scores and selections against the warp arm's
    std::vector<float> a((size_t) (nq * max_blocks));
    std::vector<int32_t> ia((size_t) (nq * cap)), ir(ia.size());
    ck(cudaMemcpy(a.data(), sc_old, a.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(ia.data(), ids_old, ia.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(ir.data(), ids_reg, ir.size() * 4, cudaMemcpyDeviceToHost), "down");
    int64_t reg_same = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        reg_same += std::equal(ia.begin() + i * cap, ia.begin() + i * cap + w, ir.begin() + i * cap);
    }
    auto compare = [&](const float* d_sc, int32_t* d_ids, const char* tag) {
        std::vector<float> b(a.size());
        std::vector<int32_t> ib(ia.size());
        ck(cudaMemcpy(b.data(), d_sc, b.size() * 4, cudaMemcpyDeviceToHost), "down");
        k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_ids, nullptr, active);
        ck(cudaMemcpy(ib.data(), d_ids, ib.size() * 4, cudaMemcpyDeviceToHost), "down");
        double max_rel = 0, sum_rel = 0, n_rel = 0;
        int64_t same_sel = 0, cells_diff = 0, cells_all = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* st = steps.data() + i * k::kStepCount;
            for (int64_t j = 0; j <= st[k::kStepNBid]; ++j) {
                const double x = a[(size_t) (i * max_blocks + j)], y = b[(size_t) (i * max_blocks + j)];
                const double r = std::fabs(x - y) / std::max(1e-6, std::fabs(x));
                max_rel = std::max(max_rel, r);
                sum_rel += r;
                n_rel += 1;
            }
            const int64_t w = st[k::kStepWidth];
            std::vector<int32_t> x(ia.begin() + i * cap, ia.begin() + i * cap + w), y(ib.begin() + i * cap, ib.begin() + i * cap + w);
            same_sel += x == y;
            std::vector<int32_t> d;
            std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(d));
            cells_diff += (int64_t) d.size() / 2;
            cells_all += w;
        }
        std::printf("  vs warp (%s): score rel diff mean %.2g max %.2g; selections identical %lld/%lld, cells "
                    "differing %.4f%%\n", tag, sum_rel / std::max(1.0, n_rel), max_rel, (long long) same_sel,
                    (long long) nq, cells_all ? 100.0 * (double) cells_diff / (double) cells_all : 0.0);
    };
    if (tc_ok) compare(sc_new, ids_new, "tc");
    if (wmma_ok) compare(sc_wmma, ids_new, "wmma");
    // accuracy against an FP64 host reference on a sample (blocks below n_bid; the tail block is the warp kernel's own
    // arithmetic in every scorer). The reference is computed once; each arm is gated by ITS OWN declared contract:
    // the sm80/gfx12 tensor-core arms at no worse than 4x the warp kernel's error (floored at 1e-6 of the score
    // scale, the prompt-attention harness's gate), and the gfx1100 FP16 arm at its pre-declared conversion bound
    // B(qi,b) = 2^-9.5 * S(qi,b) + 8 * 2^-24 * |score| (PORTING 18, the same bound qsa_select_wmma_parity
    // enforces with adversarial fixtures and a failing negative control) - 4x-warp would measure the fp16 INPUT
    // contract, not this arm's.
    struct RefSample { int64_t idx; double ref; double s_abs; };
    std::vector<RefSample> ref;
    double scale = 0;
    {
        std::mt19937 srng(11);
        const int64_t nqs = std::min<int64_t>(nq, 32);
        for (int64_t qs = 0; qs < nqs; ++qs) {
            const int64_t i = qs * nq / nqs;
            const int64_t nbid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
            if (nbid <= 0) continue;
            for (int sidx = 0; sidx < 1024; ++sidx) {
                const int64_t j = sidx < 64 ? std::min<int64_t>(nbid - 1, sidx) : (int64_t) (srng() % (uint64_t) nbid);
                double d64 = 0, s_abs = 0;
                for (int h = 0; h < 4; ++h) {
                    double d = 0;
                    for (int c = 0; c < 128; ++c) {
                        const double p = (double) q[(size_t) (i * 512 + h * 128 + c)] * (double) pooled[(size_t) (j * 128 + c)];
                        d += p;
                        s_abs += std::fabs(p);
                    }
                    d64 += d > 0 ? d : 0;
                }
                scale = std::max(scale, std::fabs(d64));
                ref.push_back({i * max_blocks + j, d64, s_abs});
            }
        }
    }
    std::vector<float> b_old, b_tc, b_wmma;
    b_old = a;
    if (tc_ok) { b_tc.resize(a.size()); ck(cudaMemcpy(b_tc.data(), sc_new, b_tc.size() * 4, cudaMemcpyDeviceToHost), "down"); }
    if (wmma_ok) { b_wmma.resize(a.size()); ck(cudaMemcpy(b_wmma.data(), sc_wmma, b_wmma.size() * 4, cudaMemcpyDeviceToHost), "down"); }
    auto fp64_err = [&](const std::vector<float>& b) {
        double e = 0;
        for (const auto& s : ref) e = std::max(e, std::fabs(s.ref - (double) b[(size_t) s.idx]));
        return e;
    };
    const double err_old = fp64_err(b_old), err_tc = tc_ok ? fp64_err(b_tc) : 0.0,
                 err_wmma = wmma_ok ? fp64_err(b_wmma) : 0.0;
    const double bound = std::max(4.0 * err_old, 1e-6 * scale);
    const double kRel = std::exp2(-9.5), kAbs = 8.0 * std::exp2(-24.0);
    double wmma_excess = 0;   // worst |err| / B over the sample; > 1 fails
    for (const auto& s : ref) {
        const double b15 = kRel * s.s_abs + kAbs * std::fabs(s.ref);
        if (b15 > 0) wmma_excess = std::max(wmma_excess, std::fabs(s.ref - (double) b_wmma[(size_t) s.idx]) / b15);
    }
    const bool acc_ok = (!tc_ok || err_tc <= bound) && (!wmma_ok || wmma_excess <= 1.0);
    std::printf("%s accuracy vs FP64 (score scale %.3g): warp max err %.3g", acc_ok ? "PASS" : "FAIL", scale, err_old);
    if (tc_ok) std::printf(", tc %.3g (%.2g of scale)", err_tc, scale > 0 ? err_tc / scale : 0.0);
    if (wmma_ok) std::printf(", wmma %.3g (worst %.2f of its 2^-9.5*S bound)", err_wmma, wmma_excess);
    std::printf("\n");
    // time: median of 9 samples of `reps` calls each, after 3 warmup samples
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto timed = [&](auto f) {
        for (int w = 0; w < 3; ++w) {   // warmup samples
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) f();
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "time");
        }
        std::vector<float> v;
        for (int w = 0; w < 9; ++w) {
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) f();
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "time");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            v.push_back(ms / reps);
        }
        std::sort(v.begin(), v.end());
        return v[4];
    };
    const float t_old = timed(run_old), t_wmma = timed(run_wmma);
    const float t_tc = tc_ok ? timed(run_new) : 0.0f;
    const float t_tk = timed([&] { k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr); });
    const float t_tk2 = timed([&] { k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr, active); });
    std::printf("top-k %.3f -> %.3f ms (%.1fx), register top-k identical to the reference %lld/%lld\n", t_tk, t_tk2,
                t_tk / t_tk2, (long long) reg_same, (long long) nq);
    if (tc_ok) std::printf("tc arm: %.3f ms\n", t_tc);
    std::printf("ctx %lld, %lld queries x %lld blocks: scores warp %.3f ms, wmma %.3f ms (%.2fx); top-k %.3f ms\n",
                (long long) ctx, (long long) nq, (long long) active, t_old, t_wmma, t_old / t_wmma, t_tk);
    return acc_ok && reg_same == nq ? 0 : 1;
}
