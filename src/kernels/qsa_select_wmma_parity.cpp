// src/kernels/qsa_select_wmma_parity.cpp - the AMD tensor-core block scores (qsa_block_scores_wmma, ROCWMMA
// FP16->FP32) against TWO references (GPU, synthetic, no model), per the contract recorded in PORTING.md 18:
//
//   1. a float64 oracle - it quantifies the approximation (FP16 inputs, FP32 accumulation in an unspecified
//      summation order): every score must satisfy the pre-derived bound
//          B(qi,b) = 2^-9.5 * S(qi,b) + 8 * 2^-24 * |score|,  S = the absolute product sum of the entry;
//   2. the exact host model of the warp kernel's arithmetic (per-lane float4 products, the __shfl_xor
//      butterfly o = 16,8,4,2,1 evaluated as the 32-lane array exactly as lane 0 sees it, heads summed in
//      order) - the warp GPU arm must match it BITWISE, which validates the harness itself and isolates a
//      kernel defect from an approximation.
//
// Cases: ragged query tiles (nq % 4 != 0), ragged block tiles (reach % 16 != 0), queries whose n_bid differs
// from their neighbours' (untouched entries keep a canary - the same set the warp kernel leaves), source rows
// past the reach poisoned (must never reach a score), an x8-magnitude headroom case (near FP16 overflow), the
// degenerate step n_kv = 0 (the tail is the warp kernel's, bitwise), near-ties at the selection boundary
// (ids must agree when the gap exceeds 2*B; a gap <= 2*B that flips is a legal near-tie and only reported),
// and a negative control: a deliberately perturbed score array MUST fail the bound check.
//
// Usage: qsa_select_wmma_parity [nq=48] [max_blocks=620] (exit 0 = all cases pass)
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
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

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
}

// ---- the two references ------------------------------------------------------------------------------

// The warp kernel's arithmetic for one (query, block) key, as lane 0 sees it: 32 lane partials, then the
// __shfl_xor butterfly o = 16,8,4,2,1 evaluated as the 32-lane array.  The 4-product sum is compiled on
// gfx11 with FMA contraction while this test's host pass runs on an x86-64 baseline without FMA, so the
// kernel's rounding is one of the three documented patterns of the same expression; the harness pins which
// one empirically and requires that single pattern to match EVERY entry bitwise (PORTING.md 18).
float warp_order_dot(const float* key, const float* q, int pattern) {
    float v[32];
    for (int l = 0; l < 32; ++l) {
        const float* k4 = key + l * 4;
        const float* q4 = q + l * 4;
        float d;
        if (pattern == 0) d = k4[0] * q4[0] + k4[1] * q4[1] + k4[2] * q4[2] + k4[3] * q4[3];
        else if (pattern == 1) {   // ((a*b + c*d) + e*f) + g*h, each mul-add fused left to right
            d = std::fma(k4[0], q4[0], k4[1] * q4[1]);
            d = std::fma(k4[2], q4[2], d);
            d = std::fma(k4[3], q4[3], d);
        } else {                   // a*b + (c*d + (e*f + g*h)), fused right to left
            d = std::fma(k4[0], q4[0], std::fma(k4[1], q4[1], std::fma(k4[2], q4[2], k4[3] * q4[3])));
        }
        v[l] = d;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float t[32];
        for (int l = 0; l < 32; ++l) t[l] = v[l] + v[l ^ o];
        std::memcpy(v, t, sizeof v);
    }
    return v[0];
}

double oracle_score(const double* key, const double* q) {   // heads relu-summed, dots in float64
    double s = 0;
    for (int h = 0; h < 4; ++h) {
        double d = 0;
        for (int i = 0; i < 128; ++i) d += key[i] * q[h * 128 + i];
        s += d > 0 ? d : 0;
    }
    return s;
}

double abs_product_sum(const double* key, const double* q) {   // S(qi,b) of the bound
    double s = 0;
    for (int h = 0; h < 4; ++h)
        for (int i = 0; i < 128; ++i) s += std::fabs(key[i] * q[h * 128 + i]);
    return s;
}

double bound_of(double S, double score) {
    return std::exp2(-9.5) * S + 8 * std::exp2(-24.0) * std::fabs(score);   // PORTING.md 18, verbatim
}

void free_dev(void* p) {   // hipFree is nodiscard in the compat shim
    const cudaError_t e = cudaFree(p);
    (void) e;
}

}  // namespace

int main(int argc, char** argv) {
    const int64_t nq0 = argc > 1 ? std::atoll(argv[1]) : 48;
    const int64_t max_blocks = argc > 2 ? std::atoll(argv[2]) : 620;
    const k::QsaShapes s = k::qsa_real_shapes();

    struct Case { const char* name; int64_t nq; int64_t reach; float scale; int64_t nkv0; };
    const Case cases[] = {
        {"ragged query tiles (nq%4 != 0) + ragged block tiles", nq0, max_blocks - 3, 1.0f, 0},
        {"whole tiles (nq%4 == 0, reach%16 == 0)", (nq0 / 4) * 4, ((max_blocks - 3) / 16) * 16, 1.0f, 0},
        {"x8 magnitude headroom (near FP16 overflow)", nq0, max_blocks - 3, 8.0f, 0},
        {"degenerate step n_kv = 0 on the first query", nq0, max_blocks - 3, 1.0f, 1},
    };

    for (const Case& cs : cases) {
        std::printf("case: %s\n", cs.name);
        const int64_t nq = cs.nq, reach = cs.reach;
        if (nq < 1 || reach < 4) continue;
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> dir(128);
        for (auto& x : dir) x = nd(rng);
        // pooled keys: a shared direction plus noise (a spread like a real indexer's), rows past `reach`
        // POISONED - the arm must never let them reach a score
        std::vector<float> pooled((size_t) max_blocks * 128), dead(128), q((size_t) nq * 512);
        for (int64_t b = 0; b < max_blocks; ++b) {
            const float a = nd(rng);
            float* row = &pooled[(size_t) b * 128];
            for (int d = 0; d < 128; ++d) row[d] = b < reach ? (0.5f * a * dir[d] + nd(rng)) * cs.scale : 1e30f;
        }
        for (auto& x : dead) x = nd(rng) * cs.scale;
        for (int64_t i = 0; i < nq; ++i)
            for (int d = 0; d < 512; ++d) q[(size_t) i * 512 + d] = (0.2f * dir[d % 128] + 0.1f * nd(rng)) * cs.scale;
        // steps: mixed n_bids (neighbours differ), one query at the very first blocks, and optionally a
        // degenerate n_kv = 0 first query
        std::vector<int32_t> steps((size_t) nq * k::kStepCount);
        for (int64_t i = 0; i < nq; ++i) {
            int32_t* st = &steps[(size_t) i * k::kStepCount];
            const int64_t n_bid = (i == 0 && cs.nkv0) ? 0 : 1 + (i * 7919) % (reach - 1);
            const int64_t n_kv = (i == 0 && cs.nkv0) ? 0 : n_bid * 4 + 1 + (int) (i % 3);
            st[k::kStepPos] = (int32_t) (n_kv - 1);
            st[k::kStepNKv] = (int32_t) n_kv;
            st[k::kStepNBid] = (int32_t) n_bid;
            st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
        }
        const float CANARY = -12345.5f;
        std::vector<float> sc_warp((size_t) nq * max_blocks, CANARY), sc_wmma((size_t) nq * max_blocks, CANARY);

        const float* d_pooled = up(pooled);
        const float* d_dead = up(dead);
        const float* d_q = up(q);
        const int32_t* d_steps = up(steps);
        float *d_warp = nullptr, *d_wmma = nullptr;
        ck(cudaMalloc(&d_warp, sc_warp.size() * 4), "malloc");
        ck(cudaMalloc(&d_wmma, sc_wmma.size() * 4), "malloc");
        ck(cudaMemcpy(d_warp, sc_warp.data(), sc_warp.size() * 4, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemcpy(d_wmma, sc_wmma.data(), sc_wmma.size() * 4, cudaMemcpyHostToDevice), "upload");

        k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_warp, nullptr, reach);
        if (!k::qsa_block_scores_wmma(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_wmma, nullptr, reach)) {
            std::fprintf(stderr, "  wmma arm refused (build/arch/geometry) - the case cannot run here\n");
            check(false, "wmma arm accepts the case");
            continue;
        }
        ck(cudaMemcpy(sc_warp.data(), d_warp, sc_warp.size() * 4, cudaMemcpyDeviceToHost), "download");
        ck(cudaMemcpy(sc_wmma.data(), d_wmma, sc_wmma.size() * 4, cudaMemcpyDeviceToHost), "download");

        // 1. the warp arm against its host model: BITWISE, with the single contraction pattern that fits
        // every entry (the harness pins the kernel's rounding - PORTING.md 18)
        int64_t fit[3] = {0, 0, 0}, total = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* st = &steps[(size_t) i * k::kStepCount];
            const int64_t n_kv = st[k::kStepNKv], n_bid = st[k::kStepNBid];
            for (int64_t b = 0; b <= n_bid && b < max_blocks; ++b) {
                const float* key = (b == n_bid) ? dead.data() : &pooled[(size_t) b * 128];
                const float got = sc_warp[(size_t) i * max_blocks + b];
                ++total;
                for (int p = 0; p < 3; ++p) {
                    float score = 0;
                    for (int h = 0; h < 4; ++h)
                        score += std::fmax(warp_order_dot(key, &q[(size_t) i * 512 + h * 128], p), 0.0f);
                    if (b == n_bid && n_kv % 4 != 0) score += 1e9f;
                    if (std::memcmp(&got, &score, 4) == 0) ++fit[p];
                }
            }
        }
        const bool warp_bitexact = total > 0 && (fit[0] == total || fit[1] == total || fit[2] == total);
        check(warp_bitexact, "warp arm == its host model, one pinned contraction pattern (bitwise)");
        std::printf("  contraction pattern fit: plain %lld, fma-left %lld, fma-right %lld of %lld\n",
                    (long long) fit[0], (long long) fit[1], (long long) fit[2], (long long) total);

        // 2. the WMMA arm against the float64 oracle, inside the pre-derived bound; the tail entry must be
        // bitwise the warp arm's (the same tail kernel).  The bound loop is a lambda on purpose: the negative
        // control below runs the SAME code over a perturbed array and must see it fail.
        auto check_bound = [&](const std::vector<float>& scores, double* worst_out) {
            int64_t violations = 0;
            double worst = 0;
            for (int64_t i = 0; i < nq; ++i) {
                const int64_t n_bid = steps[(size_t) i * k::kStepCount + k::kStepNBid];
                for (int64_t b = 0; b < n_bid && b < max_blocks; ++b) {
                    const float* kfp = &pooled[(size_t) b * 128];
                    const float* qfp = &q[(size_t) i * 512];
                    std::vector<double> kd(128), qd(512);
                    for (int d = 0; d < 128; ++d) kd[d] = kfp[d];
                    for (int d = 0; d < 512; ++d) qd[d] = qfp[d];
                    const double o = oracle_score(kd.data(), qd.data());
                    const double B = bound_of(abs_product_sum(kd.data(), qd.data()), o);
                    const double err = std::fabs((double) scores[(size_t) i * max_blocks + b] - o);
                    if (err > B) ++violations;
                    if (B > 0 && err / B > worst) worst = err / B;
                }
            }
            if (worst_out) *worst_out = worst;
            return violations;
        };
        bool tail_same = true, untouched = true;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t n_bid = steps[(size_t) i * k::kStepCount + k::kStepNBid];
            for (int64_t b = 0; b < max_blocks; ++b) {
                const float got = sc_wmma[(size_t) i * max_blocks + b];
                if (b > n_bid) {
                    if (got != CANARY) untouched = false;
                    continue;
                }
                if (b == n_bid) {
                    const float w = sc_warp[(size_t) i * max_blocks + b];
                    if (std::memcmp(&got, &w, 4) != 0) tail_same = false;
                }
            }
        }
        double worst = 0;
        const int64_t violations = check_bound(sc_wmma, &worst);
        check(untouched, "entries past n_bid keep the canary (as the warp kernel leaves them)");
        check(tail_same, "tail block bitwise the warp arm's (shared tail kernel)");
        check(violations == 0, "every score inside the pre-derived bound vs the float64 oracle");
        std::printf("  worst bound utilisation: %.1f%%\n", 100.0 * worst);

        // 3. selection ids: agree wherever the boundary gap exceeds 2*B; a flip inside 2*B is a legal
        // near-tie and only reported
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
        std::vector<int32_t> ids_warp((size_t) nq * cap), ids_wmma((size_t) nq * cap);
        int32_t *d_iw = nullptr, *d_im = nullptr;
        ck(cudaMalloc(&d_iw, ids_warp.size() * 4), "malloc");
        ck(cudaMalloc(&d_im, ids_wmma.size() * 4), "malloc");
        k::qsa_block_topk(d_warp, d_steps, nq, max_blocks, cap, s, d_iw, nullptr);
        k::qsa_block_topk(d_wmma, d_steps, nq, max_blocks, cap, s, d_im, nullptr);
        ck(cudaMemcpy(ids_warp.data(), d_iw, ids_warp.size() * 4, cudaMemcpyDeviceToHost), "download");
        ck(cudaMemcpy(ids_wmma.data(), d_im, ids_wmma.size() * 4, cudaMemcpyDeviceToHost), "download");
        int64_t diff_queries = 0, illegal = 0, near_ties = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int w = steps[(size_t) i * k::kStepCount + k::kStepWidth];
            bool same = std::memcmp(&ids_warp[(size_t) i * cap], &ids_wmma[(size_t) i * cap], (size_t) w * 4) == 0;
            if (same) continue;
            ++diff_queries;
            // the boundary gap of this query in warp-score space: the smallest score drop across the cut
            std::vector<float> sc(&sc_warp[(size_t) i * max_blocks], &sc_warp[(size_t) (i + 1) * max_blocks]);
            std::sort(sc.begin(), sc.end(), std::greater<float>());
            const double gap = w > 0 ? (double) sc[(size_t) w - 1] - (double) sc[(size_t) w] : 0;
            if (gap <= 2 * bound_of(std::fabs((double) sc[0]) * 1024, sc[0])) ++near_ties;   // reported only
            else ++illegal;
        }
        check(illegal == 0, "selection ids agree wherever the gap exceeds 2*B");
        std::printf("  differing queries: %lld (legal near-ties <= 2*B: %lld)\n", (long long) diff_queries,
                    (long long) near_ties);

        // 4. negative control: the SAME bound check over a deliberately perturbed array must FAIL it
        std::vector<float> perturbed = sc_wmma;
        {
            int64_t pi = -1, pb = -1;
            for (int64_t i = 0; i < nq && pi < 0; ++i) {
                const int64_t n_bid = steps[(size_t) i * k::kStepCount + k::kStepNBid];
                for (int64_t b = 0; b < n_bid && b < max_blocks; ++b) { pi = i; pb = b; break; }
            }
            if (pi >= 0) {
                std::vector<double> kd(128), qd(512);
                for (int d = 0; d < 128; ++d) kd[d] = pooled[(size_t) pb * 128 + d];
                for (int d = 0; d < 512; ++d) qd[d] = q[(size_t) pi * 512 + d];
                const double B = bound_of(abs_product_sum(kd.data(), qd.data()), oracle_score(kd.data(), qd.data()));
                perturbed[(size_t) pi * max_blocks + pb] += (float) (2.5 * B);
            }
        }
        check(check_bound(perturbed, nullptr) >= 1, "negative control: a +2.5*B perturbation fails the same check");

        free_dev((void*) d_pooled);
        free_dev((void*) d_dead);
        free_dev((void*) d_q);
        free_dev((void*) d_steps);
        free_dev(d_warp);
        free_dev(d_wmma);
        free_dev(d_iw);
        free_dev(d_im);
    }

    std::printf(g_fail == 0 ? "qsa_select_wmma_parity: all cases PASS\n" : "qsa_select_wmma_parity: %d FAILURES\n",
                g_fail);
    return g_fail == 0 ? 0 : 1;
}
