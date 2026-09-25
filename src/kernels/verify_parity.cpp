// src/kernels/verify_parity.cpp - batched kernels against the per-token kernel sequences they replace.
//
// A verify window's token t must come out bit for bit as it would in a window of any other size (the drafts are
// accepted exactly when greedy decode would have produced them), so every batched kernel is checked BITWISE against
// the per-token kernels it stands in for, on random inputs that include -0.0, denormals and large values.  The
// prompt path's batched PLE arithmetic is held to the same standard.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// values that exercise the rounding edges: ordinary, large, tiny, denormal, -0.0
float edgy(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int kind = (int) (rng() % 16);
    if (kind == 0) return -0.0f;
    if (kind == 1) return 0.0f;
    if (kind == 2) return u(rng) * 1e-39f;   // denormal
    if (kind == 3) return u(rng) * 1e4f;
    return u(rng);
}

int bitwise_diff(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y && !(std::isnan(a[i]) && std::isnan(b[i]))) {
            if (bad < 5) std::fprintf(stderr, "  %s: element %zu: %.9g (0x%08x) vs %.9g (0x%08x)\n", what, i, a[i], x, b[i], y);
            ++bad;
        }
    }
    return bad;
}

// ---- the MoE combine: copy all rows + add the hits + one combine per token  vs  native_moe_gather_combine
int test_gather_combine(std::mt19937& rng, cudaStream_t s) {
    const int64_t N = 2560, K = 10;
    int bad = 0;
    for (int n_tok : {1, 2, 3, 4, 8}) {
        for (int rep = 0; rep < 8; ++rep) {
            const int64_t rows = n_tok * K;
            std::vector<float> hit((size_t) (rows * N)), host((size_t) (rows * N)), w((size_t) rows),
                shared((size_t) (n_tok * N));
            for (auto& v : hit) v = edgy(rng);
            for (auto& v : host) v = edgy(rng);
            for (auto& v : w) v = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
            for (auto& v : shared) v = edgy(rng);
            // the GPU's entries, in a random order and a random share (none, some, all)
            std::vector<int32_t> dst;
            for (int32_t r = 0; r < rows; ++r)
                if (rep == 1 || (rep != 0 && rng() % 3 != 0)) dst.push_back(r);
            std::shuffle(dst.begin(), dst.end(), rng);
            const int32_t count = (int32_t) dst.size();
            // the old path's host rows: the pool zeroed the GPU's
            std::vector<float> host_zeroed = host;
            for (int32_t r : dst) std::fill(host_zeroed.begin() + r * N, host_zeroed.begin() + (r + 1) * N, 0.0f);
            // the new path's mapped rows: the GPU's rows hold garbage it must not read
            std::vector<float> host_poison = host;
            for (int32_t r : dst) std::fill(host_poison.begin() + r * N, host_poison.begin() + (r + 1) * N, NAN);

            float *d_hit = dev<float>(hit.size()), *d_parts = dev<float>(hit.size()), *d_w = dev<float>(w.size()),
                  *d_sh = dev<float>(shared.size()), *d_out_old = dev<float>(shared.size()),
                  *d_out_new = dev<float>(shared.size());
            int32_t *d_dst = dev<int32_t>(std::max<size_t>(1, dst.size()) + 64), *d_count = dev<int32_t>(1);
            up(d_hit, hit);
            up(d_w, w);
            up(d_sh, shared);
            if (!dst.empty()) up(d_dst, dst);
            up(d_count, std::vector<int32_t>{count});
            float* h_map = nullptr;
            float* m_map = nullptr;
            check(cudaHostAlloc((void**) &h_map, host.size() * sizeof(float), cudaHostAllocMapped), "cudaHostAlloc");
            check(cudaHostGetDevicePointer((void**) &m_map, h_map, 0), "cudaHostGetDevicePointer");

            // old: copy all rows, add the hits, combine per token
            std::memcpy(h_map, host_zeroed.data(), host.size() * sizeof(float));
            strata::kernels::copy_from_mapped(d_parts, m_map, rows * N, s);
            strata::kernels::moe_hit_add(d_parts, d_hit, d_dst, d_count, rows, N, s);
            for (int t = 0; t < n_tok; ++t)
                strata::kernels::native_moe_combine(d_parts + t * K * N, d_w + t * K, d_sh + t * N, d_out_old + t * N,
                                                    N, K, s);
            check(cudaStreamSynchronize(s), "old path");
            // new
            std::memcpy(h_map, host_poison.data(), host.size() * sizeof(float));
            strata::kernels::native_moe_gather_combine(d_hit, m_map, d_dst, d_count, d_w, d_sh, d_out_new, N, K, n_tok, s);
            check(cudaStreamSynchronize(s), "new path");
            const int b = bitwise_diff(down(d_out_old, shared.size()), down(d_out_new, shared.size()), "gather_combine");
            if (b) std::fprintf(stderr, "gather_combine: n_tok %d rep %d: %d of %zu differ\n", n_tok, rep, b, shared.size());
            bad += b;
            cudaFreeHost(h_map);
            for (void* p : {(void*) d_hit, (void*) d_parts, (void*) d_w, (void*) d_sh, (void*) d_out_old,
                            (void*) d_out_new, (void*) d_dst, (void*) d_count})
                cudaFree(p);
        }
    }
    std::printf("gather_combine: %s\n", bad ? "MISMATCH" : "bitwise equal (n_tok 1-4, 8; 8 plans each)");
    return bad;
}

// ---- BF16 x FP32 MMVF: one call per row  vs  bf16_gemv_fp32_mmvf_multi (the router, indexer and gate shapes)
int test_mmvf_multi(std::mt19937& rng, cudaStream_t s) {
    int bad = 0;
    for (int64_t n_out : {512, 128, 1}) {
        const int64_t n_in = 2560;
        std::vector<uint16_t> w((size_t) (n_in * n_out));
        for (auto& v : w) {
            const float f = edgy(rng);
            uint32_t b;
            std::memcpy(&b, &f, 4);
            v = (uint16_t) (b >> 16);
        }
        uint16_t* d_w = dev<uint16_t>(w.size());
        up(d_w, w);
        for (int n_tok = 1; n_tok <= 8; ++n_tok) {
            std::vector<float> x((size_t) (n_tok * n_in));
            for (auto& v : x) v = edgy(rng);
            float *d_x = dev<float>(x.size()), *d_a = dev<float>((size_t) (n_tok * n_out)),
                  *d_b = dev<float>((size_t) (n_tok * n_out));
            up(d_x, x);
            for (int t = 0; t < n_tok; ++t)
                strata::kernels::bf16_gemv_fp32_mmvf(d_x + t * n_in, d_w, d_a + t * n_out, n_in, n_out, s);
            strata::kernels::bf16_gemv_fp32_mmvf_multi(d_x, d_w, d_b, n_in, n_out, n_tok, s);
            check(cudaStreamSynchronize(s), "mmvf");
            const int b = bitwise_diff(down(d_a, (size_t) (n_tok * n_out)), down(d_b, (size_t) (n_tok * n_out)), "mmvf_multi");
            if (b) std::fprintf(stderr, "mmvf_multi: n_out %lld n_tok %d: %d differ\n", (long long) n_out, n_tok, b);
            bad += b;
            cudaFree(d_x);
            cudaFree(d_a);
            cudaFree(d_b);
        }
        cudaFree(d_w);
    }
    std::printf("mmvf_multi: %s\n", bad ? "MISMATCH" : "bitwise equal (n_out 512, 128, 1; 1-8 rows)");
    return bad;
}

// ---- the top-10 router: one launch per token  vs  native_router_top10_multi
int test_router_multi(std::mt19937& rng, cudaStream_t s) {
    int bad = 0;
    for (int n_tok = 1; n_tok <= 8; ++n_tok) {
        std::vector<float> logits((size_t) n_tok * 512);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (auto& v : logits) v = nd(rng);
        for (int t = 0; t < n_tok; ++t)   // exact ties, which the lower expert index must win
            for (int i = 0; i < 6; ++i) logits[(size_t) t * 512 + rng() % 512] = logits[(size_t) t * 512 + 7];
        float *d_l = dev<float>(logits.size()), *d_wa = dev<float>((size_t) n_tok * 10),
              *d_wb = dev<float>((size_t) n_tok * 10);
        int32_t *d_ia = dev<int32_t>((size_t) n_tok * 10), *d_ib = dev<int32_t>((size_t) n_tok * 10);
        up(d_l, logits);
        for (int t = 0; t < n_tok; ++t)
            strata::kernels::native_router_top10(d_l + t * 512, d_ia + t * 10, d_wa + t * 10, s);
        strata::kernels::native_router_top10_multi(d_l, d_ib, d_wb, n_tok, s);
        check(cudaStreamSynchronize(s), "router");
        int b = bitwise_diff(down(d_wa, (size_t) n_tok * 10), down(d_wb, (size_t) n_tok * 10), "router weights");
        if (down(d_ia, (size_t) n_tok * 10) != down(d_ib, (size_t) n_tok * 10)) {
            std::fprintf(stderr, "router_multi: n_tok %d: the ids differ\n", n_tok);
            ++b;
        }
        bad += b;
        for (void* p : {(void*) d_l, (void*) d_wa, (void*) d_wb, (void*) d_ia, (void*) d_ib}) cudaFree(p);
    }
    std::printf("router_multi: %s\n", bad ? "MISMATCH" : "bitwise equal (1-8 tokens, with ties)");
    return bad;
}

// ---- the hyper-connection read: fused_gr_read per token  vs  fused_gr_read_multi
int test_gr_read(std::mt19937& rng, cudaStream_t s) {
    const int64_t N = 2560, HC = 4, D = N * HC, LR = 320;
    auto bf16 = [&](size_t n) {
        std::vector<uint16_t> v(n);
        for (auto& x : v) {
            const float f = edgy(rng) * 0.05f;
            uint32_t b;
            std::memcpy(&b, &f, 4);
            x = (uint16_t) (b >> 16);
        }
        return v;
    };
    std::vector<float> wn((size_t) D);
    for (auto& v : wn) v = edgy(rng);
    uint16_t *d_wd = dev<uint16_t>((size_t) (LR * D)), *d_wu = dev<uint16_t>((size_t) (D * LR)),
             *d_wi = dev<uint16_t>((size_t) (HC * D));
    float* d_wn = dev<float>(wn.size());
    up(d_wd, bf16((size_t) (LR * D)));
    up(d_wu, bf16((size_t) (D * LR)));
    up(d_wi, bf16((size_t) (HC * D)));
    up(d_wn, wn);
    int bad = 0;
    for (int n_tok = 1; n_tok <= 8; ++n_tok) {
        for (int variant = 0; variant < 3; ++variant) {   // apply + inject, no apply, no inject
            const bool apply = variant != 1, inject = variant != 2;
            std::vector<float> R((size_t) (n_tok * D)), bo((size_t) (n_tok * N)), inj((size_t) (n_tok * HC));
            for (auto& v : R) v = edgy(rng);
            for (auto& v : bo) v = edgy(rng);
            for (auto& v : inj) v = edgy(rng);
            // one set of outputs per path; R is updated in place when apply, so each path has its own copy
            float* d[2][6];   // R, lo, rs, inject, mixed, xn
            for (int p = 0; p < 2; ++p) {
                d[p][0] = dev<float>(R.size());
                up(d[p][0], R);
                d[p][1] = dev<float>((size_t) (n_tok * LR));
                d[p][2] = dev<float>((size_t) (n_tok * HC));
                d[p][3] = dev<float>((size_t) (n_tok * HC));
                d[p][4] = dev<float>((size_t) (n_tok * N));
                d[p][5] = dev<float>((size_t) (n_tok * D));
                check(cudaMemset(d[p][3], 0, (size_t) (n_tok * HC) * 4), "memset");
            }
            float *d_bo = dev<float>(bo.size()), *d_inj = dev<float>(inj.size());
            up(d_bo, bo);
            up(d_inj, inj);
            std::vector<strata::kernels::FusedGrArgs> args[2];
            for (int p = 0; p < 2; ++p)
                for (int t = 0; t < n_tok; ++t) {
                    strata::kernels::FusedGrArgs a;
                    a.R = d[p][0] + t * D; a.R_out = d[p][0] + t * D; a.apply = apply;
                    a.bo_prev = d_bo + t * N; a.inj_prev = d_inj + t * HC;
                    a.w_norm = d_wn; a.w_down = d_wd; a.w_up = d_wu; a.w_inject = inject ? d_wi : nullptr;
                    a.eps = 1e-6f;
                    a.lo = d[p][1] + t * LR; a.rs = d[p][2] + t * HC; a.inject_out = d[p][3] + t * HC;
                    a.mixed = d[p][4] + t * N;
                    args[p].push_back(a);
                }
            for (int t = 0; t < n_tok; ++t) strata::kernels::fused_gr_read(args[0][(size_t) t], s);
            strata::kernels::fused_gr_read_multi(args[1].data(), n_tok, d[1][5], s);
            check(cudaStreamSynchronize(s), "gr read");
            const char* names[5] = {"R", "lo", "rs", "inject", "mixed"};
            const size_t sizes[5] = {R.size(), (size_t) (n_tok * LR), (size_t) (n_tok * HC), (size_t) (n_tok * HC),
                                     (size_t) (n_tok * N)};
            for (int o = 0; o < 5; ++o) {
                const int b = bitwise_diff(down(d[0][o], sizes[o]), down(d[1][o], sizes[o]), names[o]);
                if (b) std::fprintf(stderr, "gr_read: n_tok %d variant %d: %s: %d differ\n", n_tok, variant, names[o], b);
                bad += b;
            }
            for (int p = 0; p < 2; ++p)
                for (float* q : d[p]) cudaFree(q);
            cudaFree(d_bo);
            cudaFree(d_inj);
        }
    }
    for (void* p : {(void*) d_wd, (void*) d_wu, (void*) d_wi, (void*) d_wn}) cudaFree(p);
    std::printf("gr_read: %s\n", bad ? "MISMATCH" : "bitwise equal to the single-token read (1-8 tokens, 3 variants)");
    return bad;
}

// ---- RoPE over several tokens' heads: native_rope_apply per token  vs  native_rope_apply_tokens
int test_rope_tokens(std::mt19937& rng, cudaStream_t s) {
    const int NH = 24;
    int bad = 0;
    for (int heads : {2, 4, 24}) {
        for (int head_dim : {128, 256}) {
            for (int n_tok = 1; n_tok <= 8; ++n_tok) {
                std::vector<float> x((size_t) n_tok * heads * head_dim);
                for (auto& v : x) v = edgy(rng);
                std::vector<int32_t> pos((size_t) n_tok * NH);   // a window's per-token position vectors
                const int p0 = (int) (rng() % 200000);
                for (int t = 0; t < n_tok; ++t)
                    for (int h = 0; h < NH; ++h) pos[(size_t) t * NH + h] = p0 + t;
                float *d_a = dev<float>(x.size()), *d_b = dev<float>(x.size());
                int32_t* d_pos = dev<int32_t>(pos.size());
                up(d_a, x);
                up(d_b, x);
                up(d_pos, pos);
                for (int t = 0; t < n_tok; ++t)
                    strata::kernels::native_rope_apply(d_a + (size_t) t * heads * head_dim, d_a + (size_t) t * heads * head_dim,
                                                       heads, head_dim, 64, 5000000.0f, d_pos + t * NH, s);
                strata::kernels::native_rope_apply_tokens(d_b, d_b, n_tok * heads, head_dim, 64, 5000000.0f, d_pos, heads,
                                                          NH, s);
                check(cudaStreamSynchronize(s), "rope");
                bad += bitwise_diff(down(d_a, x.size()), down(d_b, x.size()), "rope_tokens");
                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_pos);
            }
        }
    }
    std::printf("rope_tokens: %s\n", bad ? "MISMATCH" : "bitwise equal (2, 4, 24 heads; 1-8 tokens)");
    return bad;
}

// ---- the int8 KV append: kv_append_q8_step per token  vs  kv_append_q8_steps
int test_kv_append(std::mt19937& rng, cudaStream_t s) {
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    const int cells = 4096, pages = cells / (int) sh.page_size, per_tok = (int) (sh.n_head_kv * sh.head_dim);
    const size_t codes = (size_t) cells * per_tok, scales = codes / strata::kernels::KV_Q8_GROUP;
    std::vector<int32_t> table((size_t) pages);
    for (int i = 0; i < pages; ++i) table[(size_t) i] = pages - 1 - i;   // a permuted page table
    int32_t* d_table = dev<int32_t>(table.size());
    up(d_table, table);
    int bad = 0;
    for (int n_tok = 1; n_tok <= 8; ++n_tok) {
        const int p0 = 500 + (int) (rng() % 2000);
        std::vector<float> k((size_t) n_tok * per_tok), v((size_t) n_tok * per_tok);
        for (auto& x : k) x = edgy(rng);
        for (auto& x : v) x = edgy(rng);
        std::vector<int32_t> steps((size_t) n_tok * strata::kernels::kStepCount, 0);
        for (int t = 0; t < n_tok; ++t) steps[(size_t) t * strata::kernels::kStepCount + strata::kernels::kStepPos] = p0 + t;
        float *d_k = dev<float>(k.size()), *d_v = dev<float>(v.size());
        int32_t* d_steps = dev<int32_t>(steps.size());
        up(d_k, k);
        up(d_v, v);
        up(d_steps, steps);
        int8_t* q[2][2];
        uint16_t* sc[2][2];
        for (int p = 0; p < 2; ++p)
            for (int kv = 0; kv < 2; ++kv) {
                q[p][kv] = dev<int8_t>(codes);
                sc[p][kv] = dev<uint16_t>(scales);
                check(cudaMemset(q[p][kv], 0, codes), "memset");
                check(cudaMemset(sc[p][kv], 0, scales * 2), "memset");
            }
        for (int t = 0; t < n_tok; ++t)
            strata::kernels::kv_append_q8_step(q[0][0], q[0][1], sc[0][0], sc[0][1], d_table,
                                               d_steps + t * strata::kernels::kStepCount, d_k + (size_t) t * per_tok,
                                               d_v + (size_t) t * per_tok, sh, s);
        strata::kernels::kv_append_q8_steps(q[1][0], q[1][1], sc[1][0], sc[1][1], d_table, d_steps,
                                            strata::kernels::kStepCount, d_k, d_v, n_tok, sh, s);
        check(cudaStreamSynchronize(s), "kv append");
        for (int kv = 0; kv < 2; ++kv) {
            if (down(q[0][kv], codes) != down(q[1][kv], codes) || down(sc[0][kv], scales) != down(sc[1][kv], scales)) {
                std::fprintf(stderr, "kv_append: n_tok %d: the %s cells differ\n", n_tok, kv ? "V" : "K");
                ++bad;
            }
        }
        for (int p = 0; p < 2; ++p)
            for (int kv = 0; kv < 2; ++kv) {
                cudaFree(q[p][kv]);
                cudaFree(sc[p][kv]);
            }
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_steps);
    }
    cudaFree(d_table);
    std::printf("kv_append_steps: %s\n", bad ? "MISMATCH" : "bitwise equal (1-8 tokens)");
    return bad;
}

// ---- the indexer append: one call per token  vs  native_qsa_indexer_append_multi (blocks completed inside the
// window, the first cell, and rejected tokens' -1 positions)
int test_indexer_append(std::mt19937& rng, cudaStream_t s) {
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    const int D = 128, max_cells = 4096;
    const size_t pooled_n = (size_t) (max_cells / 4 + 1) * D;
    std::vector<float> gamma((size_t) D);
    for (auto& g : gamma) g = 0.5f + std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float* d_gamma = dev<float>(gamma.size());
    up(d_gamma, gamma);
    int bad = 0;
    for (int n_tok = 1; n_tok <= 8; ++n_tok) {
        for (int start : {0, 1, 2, 3, 5, 6, 7, 1022}) {
            for (int rejected = 0; rejected < 2; ++rejected) {
                std::vector<float> raw((size_t) n_tok * D);
                for (auto& x : raw) x = std::normal_distribution<float>(0.0f, 1.0f)(rng);
                std::vector<int32_t> pos((size_t) n_tok);
                for (int t = 0; t < n_tok; ++t) pos[(size_t) t] = rejected && t >= (n_tok + 1) / 2 ? -1 : start + t;
                // the state both paths start from: a random earlier tail, spare and pooled blocks
                std::vector<float> tail((size_t) 3 * D), dead((size_t) D), pooled(pooled_n);
                for (auto& x : tail) x = edgy(rng);
                for (auto& x : dead) x = edgy(rng);
                for (auto& x : pooled) x = edgy(rng);
                float *d_raw = dev<float>(raw.size());
                int32_t* d_pos = dev<int32_t>(pos.size());
                up(d_raw, raw);
                up(d_pos, pos);
                float* bufs[2][3];
                int32_t* bp[2];
                for (int p = 0; p < 2; ++p) {
                    bufs[p][0] = dev<float>(tail.size());
                    bufs[p][1] = dev<float>(dead.size());
                    bufs[p][2] = dev<float>(pooled.size());
                    bp[p] = dev<int32_t>(1);
                    up(bufs[p][0], tail);
                    up(bufs[p][1], dead);
                    up(bufs[p][2], pooled);
                    up(bp[p], std::vector<int32_t>{-7});
                }
                const strata::kernels::QsaIndexerBuffers ib0{bufs[0][0], bufs[0][1], bufs[0][2], bp[0]};
                const strata::kernels::QsaIndexerBuffers ib1{bufs[1][0], bufs[1][1], bufs[1][2], bp[1]};
                for (int t = 0; t < n_tok; ++t)
                    strata::kernels::native_qsa_indexer_append(d_raw + (size_t) t * D, d_pos + t, 0, d_gamma, 1e-6f, ib0, sh,
                                                               max_cells, 5000000.0f, s);
                strata::kernels::native_qsa_indexer_append_multi(d_raw, d_pos, 1, n_tok, 0, d_gamma, 1e-6f, ib1, sh,
                                                                 max_cells, 5000000.0f, s);
                check(cudaStreamSynchronize(s), "indexer");
                const size_t sizes[3] = {tail.size(), dead.size(), pooled.size()};
                const char* names[3] = {"tail", "dead", "pooled"};
                for (int o = 0; o < 3; ++o) {
                    const int b = bitwise_diff(down(bufs[0][o], sizes[o]), down(bufs[1][o], sizes[o]), names[o]);
                    if (b) std::fprintf(stderr, "indexer: n_tok %d start %d rejected %d: %s differ\n", n_tok, start, rejected, names[o]);
                    bad += b;
                }
                if (down(bp[0], 1) != down(bp[1], 1)) {
                    std::fprintf(stderr, "indexer: n_tok %d start %d: block_pos differs\n", n_tok, start);
                    ++bad;
                }
                for (int p = 0; p < 2; ++p) {
                    for (float* q : bufs[p]) cudaFree(q);
                    cudaFree(bp[p]);
                }
                cudaFree(d_raw);
                cudaFree(d_pos);
            }
        }
    }
    cudaFree(d_gamma);
    std::printf("indexer_append_multi: %s\n", bad ? "MISMATCH" : "bitwise equal (1-8 tokens, 8 starts, rejected cells)");
    return bad;
}

// ---- the PLE block after its projections: native_ple_postops + ple_history_advance per token  vs
// native_ple_postops_tokens (chunks shorter and longer than the nine-row history)
int test_ple_tokens(std::mt19937& rng, cudaStream_t s) {
    const int N = 2560, D = 10240, H = 4, HIST = 9;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> nk((size_t) D), nq((size_t) D), nc((size_t) D);
    for (auto* v : {&nk, &nq, &nc})
        for (auto& x : *v) x = 0.5f + std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    std::vector<uint16_t> taps((size_t) 4 * D);
    for (auto& t : taps) {   // F16 in [-1, 1): sign, exponent 0..14, random mantissa
        const uint16_t sign = (uint16_t) ((rng() & 1u) << 15), e = (uint16_t) (rng() % 15), m = (uint16_t) (rng() & 0x3FFu);
        t = (uint16_t) (sign | (e << 10) | m);
    }
    float *d_nk = dev<float>(D), *d_nq = dev<float>(D), *d_nc = dev<float>(D);
    uint16_t* d_taps = dev<uint16_t>(taps.size());
    up(d_nk, nk); up(d_nq, nq); up(d_nc, nc); up(d_taps, taps);
    strata::kernels::PleWeights w;
    w.norm_key = d_nk; w.norm_query = d_nq; w.norm_conv = d_nc; w.conv1d_f16 = d_taps;
    int bad = 0;
    for (int n_tok : {1, 2, 5, 9, 10, 23}) {
        std::vector<float> key((size_t) n_tok * D), hidden((size_t) n_tok * D), value((size_t) n_tok * N),
            hist((size_t) HIST * D);
        for (auto& x : key) x = edgy(rng);
        for (auto& x : hidden) x = edgy(rng);
        for (auto& x : value) x = nd(rng);
        for (auto& x : hist) x = nd(rng);
        float *d_key = dev<float>(key.size()), *d_hidden = dev<float>(hidden.size()), *d_value = dev<float>(value.size());
        up(d_key, key); up(d_hidden, hidden); up(d_value, value);
        // per token: the decode path's calls
        float *h0 = dev<float>(hist.size()), *r0 = dev<float>(hidden.size());
        float *k1 = dev<float>(D), *q1 = dev<float>(D), *g1 = dev<float>(H), *gd1 = dev<float>(D), *c1 = dev<float>(D);
        up(h0, hist);
        for (int t = 0; t < n_tok; ++t) {
            const strata::kernels::NativePlePostopsBuffers b{k1, q1, g1, gd1, q1, c1, r0 + (size_t) t * D};
            strata::kernels::native_ple_postops(d_key + (size_t) t * D, d_hidden + (size_t) t * D,
                                                d_value + (size_t) t * N, h0, w, b, s);
            strata::kernels::ple_history_advance(h0, q1, s);
        }
        // the chunk at once
        float *h2 = dev<float>(hist.size()), *r2 = dev<float>(hidden.size()), *k2 = dev<float>(key.size()),
              *q2 = dev<float>(key.size()), *g2 = dev<float>((size_t) n_tok * H), *gd2 = dev<float>(key.size());
        up(h2, hist);
        strata::kernels::native_ple_postops_tokens(d_key, d_hidden, d_value, h2, w, {k2, q2, g2, gd2, q2, r2}, n_tok, s);
        check(cudaStreamSynchronize(s), "ple tokens");
        const int br = bitwise_diff(down(r0, hidden.size()), down(r2, hidden.size()), "result");
        const int bh = bitwise_diff(down(h0, hist.size()), down(h2, hist.size()), "history");
        if (br || bh) std::fprintf(stderr, "ple tokens: n_tok %d: result or history differ\n", n_tok);
        bad += br + bh;
        for (float* p : {d_key, d_hidden, d_value, h0, r0, k1, q1, g1, gd1, c1, h2, r2, k2, q2, g2, gd2}) cudaFree(p);
    }
    for (float* p : {d_nk, d_nq, d_nc}) cudaFree(p);
    cudaFree(d_taps);
    std::printf("ple_postops_tokens: %s\n", bad ? "MISMATCH" : "bitwise equal (1-23 tokens, results and history)");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: verify_parity [--selftest]\n");
            return 2;
        }
    }
    std::mt19937 rng(20260925);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    bad += test_gather_combine(rng, s);
    bad += test_mmvf_multi(rng, s);
    bad += test_router_multi(rng, s);
    bad += test_gr_read(rng, s);
    bad += test_rope_tokens(rng, s);
    bad += test_kv_append(rng, s);
    bad += test_indexer_append(rng, s);
    bad += test_ple_tokens(rng, s);
    cudaStreamDestroy(s);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
