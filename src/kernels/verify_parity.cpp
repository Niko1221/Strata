// src/kernels/verify_parity.cpp - batched kernels against the per-token kernel sequences they replace.
//
// A verify window's token t must come out bit for bit as it would in a window of any other size (the drafts are
// accepted exactly when greedy decode would have produced them), so every batched kernel is checked BITWISE against
// the per-token kernels it stands in for, on random inputs that include -0.0, denormals and large values.  The
// prompt path's batched PLE and GDN arithmetic is held to the same standard; its attention kernel, which sums in
// another order, is checked against the decode kernel within a tolerance.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/kernels.hpp"

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
            int32_t *d_dst = dev<int32_t>(std::max<size_t>(1, dst.size()) + 64), *d_count = dev<int32_t>(3);
            up(d_hit, hit);
            up(d_w, w);
            up(d_sh, shared);
            if (!dst.empty()) up(d_dst, dst);
            // the new path takes them as two lists (the window's VRAM share and the pool's PCIe share), split at random
            const int32_t split = rep % 2 == 0 ? count : (int32_t) (rng() % (dst.size() + 1));
            up(d_count, std::vector<int32_t>{count, split, count - split});
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
            strata::kernels::native_moe_gather_combine(d_hit, m_map, d_dst, d_count + 1, rep % 2 == 0 ? nullptr : d_dst + split,
                                                       d_count + 2, d_w, d_sh, d_out_new, N, K, n_tok, s);
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
    std::printf("gather_combine: %s\n", bad ? "MISMATCH" : "bitwise equal (n_tok 1-4, 8; 8 plans each, one or two lists)");
    return bad;
}

// ---- the main GPU's hit plan decided on the device  vs  the host's rule: distinct experts in routing order, the
// resident ones as groups of their entries in routing order
int test_hit_plan(std::mt19937& rng, cudaStream_t s) {
    const int NE = 512, K = 10, slots = 300;
    const int64_t cap = 8 * K, ptr_off = ((4 + (cap + 1) + 2 * cap) + 1) & ~1ll, words = ptr_off + 2 * cap;
    std::vector<unsigned long long> slot_ptr(slots);
    for (int i = 0; i < slots; ++i) slot_ptr[(size_t) i] = 0x700000000ull + (unsigned long long) i * 1382400ull;
    unsigned long long* d_sp = dev<unsigned long long>(slots);
    up(d_sp, slot_ptr);
    int32_t *d_ids = dev<int32_t>(cap), *d_res = dev<int32_t>(NE), *d_plan = dev<int32_t>(words);
    int bad = 0;
    for (int n_tok : {1, 2, 3, 4, 8}) {
        for (int rep = 0; rep < 24; ++rep) {
            const int n = n_tok * K;
            // each token's k experts distinct (rep 3: repeats within a token too), drawn from a small set so tokens share
            std::vector<int32_t> set((size_t) (4 + rng() % 40)), ids((size_t) n), res((size_t) NE, -1);
            for (auto& e : set) e = (int32_t) (rng() % NE);
            for (int t = 0; t < n_tok; ++t)
                for (int j = 0; j < K; ++j) {
                    int32_t e;
                    bool again;
                    do {
                        e = rng() % 4 == 0 ? (int32_t) (rng() % NE) : set[rng() % set.size()];
                        again = false;
                        for (int m = 0; m < j && rep != 3; ++m) again = again || ids[(size_t) (t * K + m)] == e;
                    } while (again);
                    ids[(size_t) (t * K + j)] = e;
                }
            for (int e = 0; e < NE; ++e)   // resident: none, all, or at random
                if (rep == 1 || (rep != 0 && rng() % 3 != 0)) res[(size_t) e] = (int32_t) (rng() % slots);
            std::vector<int32_t> first((size_t) n), start, dst, tok;
            std::vector<unsigned long long> ptr;
            for (int i = 0; i < n; ++i) {
                first[(size_t) i] = i;
                for (int j = 0; j < i; ++j)
                    if (ids[(size_t) j] == ids[(size_t) i]) { first[(size_t) i] = j; break; }
            }
            for (int i = 0; i < n; ++i) {
                if (first[(size_t) i] != i || res[(size_t) ids[(size_t) i]] < 0) continue;
                ptr.push_back(slot_ptr[(size_t) res[(size_t) ids[(size_t) i]]]);
                start.push_back((int32_t) dst.size());
                for (int j = i; j < n; ++j)
                    if (first[(size_t) j] == i) { dst.push_back(j); tok.push_back(j / K); }
            }
            start.push_back((int32_t) dst.size());
            up(d_ids, ids);
            up(d_res, res);
            check(cudaMemset(d_plan, 0xff, (size_t) words * 4), "memset");
            strata::kernels::verify_hit_plan(d_ids, n, K, d_res, NE, d_sp, d_plan, cap, ptr_off, s);
            check(cudaStreamSynchronize(s), "hit plan");
            const std::vector<int32_t> pl = down(d_plan, (size_t) words);
            const int groups = (int) ptr.size(), entries = (int) dst.size();
            bool ok = pl[0] == groups && pl[1] == entries;
            for (int q = 0; ok && q <= groups; ++q) ok = pl[(size_t) (4 + q)] == start[(size_t) q];
            for (int q = 0; ok && q < groups; ++q) {
                unsigned long long p;
                std::memcpy(&p, &pl[(size_t) (ptr_off + 2 * q)], 8);
                ok = p == ptr[(size_t) q];
            }
            for (int q = 0; ok && q < entries; ++q)
                ok = pl[(size_t) (4 + cap + 1 + q)] == dst[(size_t) q] && pl[(size_t) (4 + 2 * cap + 1 + q)] == tok[(size_t) q];
            if (!ok) {
                std::fprintf(stderr, "hit_plan: n_tok %d rep %d: groups %d/%d entries %d/%d\n", n_tok, rep, pl[0], groups,
                             pl[1], entries);
                ++bad;
            }
        }
    }
    for (void* p : {(void*) d_sp, (void*) d_ids, (void*) d_res, (void*) d_plan}) cudaFree(p);
    std::printf("hit_plan: %s\n", bad ? "MISMATCH" : "the host's groups (n_tok 1-4, 8; 24 routings each)");
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

// ---- a layer's router: the gemv, the top 10, doorbell_publish and verify_hit_plan  vs  verify_router
int test_verify_router(std::mt19937& rng, cudaStream_t s) {
    const int N = 2560, NE = 512, K = 10, slots = 300;
    const int64_t cap = 8 * K, ptr_off = ((4 + (cap + 1) + 2 * cap) + 1) & ~1ll, words = ptr_off + 4 * cap + 2;
    std::vector<uint16_t> w((size_t) NE * N);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (auto& v : w) {
        const float f = rng() % 64 == 0 ? edgy(rng) : nd(rng);
        uint32_t b;
        std::memcpy(&b, &f, 4);
        v = (uint16_t) (b >> 16);
    }
    std::vector<unsigned long long> slot_ptr(slots);
    for (int i = 0; i < slots; ++i) slot_ptr[(size_t) i] = 0x700000000ull + (unsigned long long) i * 1382400ull;
    uint16_t* d_w = dev<uint16_t>(w.size());
    unsigned long long* d_sp = dev<unsigned long long>(slots);
    int32_t* d_res = dev<int32_t>(NE);
    unsigned* d_counter = dev<unsigned>(1);
    up(d_w, w);
    up(d_sp, slot_ptr);
    check(cudaMemset(d_counter, 0, 4), "memset");
    // per path: logits, ids, weights, plan (device); x, ids, weights, seq (mapped)
    float *d_logits[2], *d_wts[2], *h_x[2], *m_x[2], *h_w[2], *m_w[2];
    int32_t *d_ids[2], *d_plan[2], *h_ids[2], *m_ids[2];
    uint32_t *h_seq[2], *m_seq[2];
    for (int p = 0; p < 2; ++p) {
        d_logits[p] = dev<float>((size_t) 8 * NE);
        d_wts[p] = dev<float>((size_t) 8 * K);
        d_ids[p] = dev<int32_t>((size_t) 8 * K);
        d_plan[p] = dev<int32_t>((size_t) words);
        check(cudaHostAlloc((void**) &h_x[p], (size_t) 8 * N * 4, cudaHostAllocMapped), "cudaHostAlloc");
        check(cudaHostAlloc((void**) &h_ids[p], (size_t) 8 * K * 4, cudaHostAllocMapped), "cudaHostAlloc");
        check(cudaHostAlloc((void**) &h_w[p], (size_t) 8 * K * 4, cudaHostAllocMapped), "cudaHostAlloc");
        check(cudaHostAlloc((void**) &h_seq[p], 64, cudaHostAllocMapped), "cudaHostAlloc");
        check(cudaHostGetDevicePointer((void**) &m_x[p], h_x[p], 0), "mapped");
        check(cudaHostGetDevicePointer((void**) &m_ids[p], h_ids[p], 0), "mapped");
        check(cudaHostGetDevicePointer((void**) &m_w[p], h_w[p], 0), "mapped");
        check(cudaHostGetDevicePointer((void**) &m_seq[p], h_seq[p], 0), "mapped");
    }
    float* d_x = dev<float>((size_t) 8 * N);
    int bad = 0;
    for (int n_tok = 1; n_tok <= 8; ++n_tok) {
        for (int rep = 0; rep < 4; ++rep) {
            std::vector<float> x((size_t) n_tok * N);
            for (auto& v : x) v = rng() % 32 == 0 ? edgy(rng) : std::normal_distribution<float>(0.0f, 1.0f)(rng);
            std::vector<int32_t> res((size_t) NE, -1);
            for (int e = 0; e < NE; ++e)   // resident: none, all, or at random
                if (rep == 1 || (rep != 0 && rng() % 3 != 0)) res[(size_t) e] = (int32_t) (rng() % slots);
            up(d_x, x);
            up(d_res, res);
            for (int p = 0; p < 2; ++p) {
                check(cudaMemset(d_plan[p], 0xff, (size_t) words * 4), "memset");
                std::memset(h_x[p], 0xcd, (size_t) 8 * N * 4);
                *h_seq[p] = 5;
            }
            // the kernels verify_router stands in for
            strata::kernels::bf16_gemv_fp32_mmvf_multi(d_x, d_w, d_logits[0], N, NE, n_tok, s);
            strata::kernels::native_router_top10_multi(d_logits[0], d_ids[0], d_wts[0], n_tok, s);
            strata::kernels::doorbell_publish(d_x, d_ids[0], d_wts[0], (int64_t) n_tok * N, (int64_t) n_tok * K, m_x[0],
                                              m_ids[0], m_w[0], m_seq[0], s);
            strata::kernels::verify_hit_plan(d_ids[0], n_tok * K, K, d_res, NE, d_sp, d_plan[0], cap, ptr_off, s);
            strata::kernels::VerifyRouterArgs a;
            a.x = d_x; a.w = d_w; a.logits = d_logits[1]; a.ids = d_ids[1]; a.weights = d_wts[1];
            a.x_out = m_x[1]; a.ids_out = m_ids[1]; a.w_out = m_w[1]; a.seq = m_seq[1]; a.ring = 6;
            a.res = d_res; a.slot_ptr = d_sp; a.plan = d_plan[1]; a.cap = (int) cap; a.ptr_off = (int) ptr_off;
            a.counter = d_counter; a.n_tok = n_tok; a.n_embd = N; a.n_expert = NE;
            strata::kernels::verify_router(a, s);
            check(cudaStreamSynchronize(s), "router");
            const size_t nk = (size_t) n_tok * K;
            int b = bitwise_diff(down(d_logits[0], (size_t) n_tok * NE), down(d_logits[1], (size_t) n_tok * NE), "logits");
            b += bitwise_diff(down(d_wts[0], nk), down(d_wts[1], nk), "weights");
            b += bitwise_diff(std::vector<float>(h_w[0], h_w[0] + nk), std::vector<float>(h_w[1], h_w[1] + nk), "mapped weights");
            b += bitwise_diff(std::vector<float>(h_x[0], h_x[0] + (size_t) n_tok * N),
                              std::vector<float>(h_x[1], h_x[1] + (size_t) n_tok * N), "mapped x");
            if (down(d_ids[0], nk) != down(d_ids[1], nk) ||
                std::vector<int32_t>(h_ids[0], h_ids[0] + nk) != std::vector<int32_t>(h_ids[1], h_ids[1] + nk)) {
                std::fprintf(stderr, "verify_router: n_tok %d rep %d: the ids differ\n", n_tok, rep);
                ++b;
            }
            if (*h_seq[0] != 6 || *h_seq[1] != 6 || down(d_counter, 1)[0] != 0u) {
                std::fprintf(stderr, "verify_router: n_tok %d rep %d: seq %u / %u, counter %u\n", n_tok, rep, *h_seq[0],
                             *h_seq[1], down(d_counter, 1)[0]);
                ++b;
            }
            if (down(d_plan[0], (size_t) words) != down(d_plan[1], (size_t) words)) {
                std::fprintf(stderr, "verify_router: n_tok %d rep %d: the hit plans differ\n", n_tok, rep);
                ++b;
            }
            if (b) std::fprintf(stderr, "verify_router: n_tok %d rep %d: %d mismatches\n", n_tok, rep, b);
            bad += b;
        }
    }
    for (int p = 0; p < 2; ++p) {
        for (void* q : {(void*) d_logits[p], (void*) d_wts[p], (void*) d_ids[p], (void*) d_plan[p]}) cudaFree(q);
        for (void* q : {(void*) h_x[p], (void*) h_ids[p], (void*) h_w[p], (void*) h_seq[p]}) cudaFreeHost(q);
    }
    for (void* q : {(void*) d_w, (void*) d_sp, (void*) d_res, (void*) d_counter, (void*) d_x}) cudaFree(q);
    std::printf("verify_router: %s\n", bad ? "MISMATCH"
                                           : "bitwise the gemv, top 10, doorbell and hit plan (1-8 tokens, 4 routings each)");
    return bad;
}

// ---- the hyper-connection read: every token of an n-token read bitwise its 1-token read, and within a rounding
// bound of an FP64 reference (|error| <= 1e-4 x the sum of the terms' magnitudes, carried through silu and sigmoid)
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
    auto bf = [](uint16_t v) { const uint32_t b = (uint32_t) v << 16; float f; std::memcpy(&f, &b, 4); return (double) f; };
    std::vector<float> wn((size_t) D);
    for (auto& v : wn) v = edgy(rng);
    const std::vector<uint16_t> wd = bf16((size_t) (LR * D)), wu = bf16((size_t) (D * LR)), wi = bf16((size_t) (HC * D));
    uint16_t *d_wd = dev<uint16_t>(wd.size()), *d_wu = dev<uint16_t>(wu.size()), *d_wi = dev<uint16_t>(wi.size());
    float* d_wn = dev<float>(wn.size());
    up(d_wd, wd);
    up(d_wu, wu);
    up(d_wi, wi);
    up(d_wn, wn);
    const size_t sb = strata::kernels::fused_gr_scratch_bytes();
    float* d_scratch = (float*) dev<uint8_t>(sb);
    check(cudaMemset(d_scratch, 0, sb), "memset");
    int bad = 0, off = 0;
    double worst = 0.0;   // the largest error as a fraction of its bound
    for (int n_tok = 1; n_tok <= 8; ++n_tok) {
        for (int variant = 0; variant < 3; ++variant) {   // apply + inject, no apply, no inject
            const bool apply = variant != 1, inject = variant != 2;
            std::vector<float> R((size_t) (n_tok * D)), bo((size_t) (n_tok * N)), inj((size_t) (n_tok * HC));
            for (auto& v : R) v = edgy(rng);
            for (auto& v : bo) v = edgy(rng);
            for (auto& v : inj) v = edgy(rng);
            // path 0: one read of n_tok tokens; path 1: n_tok reads of one token.  R is updated in place when apply.
            float* d[2][5];   // R, lo, rs, inject, mixed
            for (int p = 0; p < 2; ++p) {
                d[p][0] = dev<float>(R.size());
                up(d[p][0], R);
                d[p][1] = dev<float>((size_t) (n_tok * LR));
                d[p][2] = dev<float>((size_t) (n_tok * HC));
                d[p][3] = dev<float>((size_t) (n_tok * HC));
                d[p][4] = dev<float>((size_t) (n_tok * N));
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
            strata::kernels::fused_gr_read_multi(args[0].data(), n_tok, d_scratch, s);
            for (int t = 0; t < n_tok; ++t) strata::kernels::fused_gr_read_multi(&args[1][(size_t) t], 1, d_scratch, s);
            check(cudaStreamSynchronize(s), "gr read");
            const char* names[5] = {"R", "lo", "rs", "inject", "mixed"};
            const size_t sizes[5] = {R.size(), (size_t) (n_tok * LR), (size_t) (n_tok * HC), (size_t) (n_tok * HC),
                                     (size_t) (n_tok * N)};
            std::vector<float> got[5];
            for (int o = 0; o < 5; ++o) {
                got[o] = down(d[0][o], sizes[o]);
                const int b = bitwise_diff(got[o], down(d[1][o], sizes[o]), names[o]);
                if (b) std::fprintf(stderr, "gr_read: n_tok %d variant %d: %s: %d differ\n", n_tok, variant, names[o], b);
                bad += b;
            }
            // FP64 for the first token of the 1- and 4-token reads
            if (n_tok == 1 || n_tok == 4) {
                const float* Rt = R.data();
                // xa: the magnitude xn's rounding scales with (R' = R + gw bo may cancel)
                std::vector<double> Rp((size_t) D), Ra((size_t) D), xn((size_t) D), xa((size_t) D), rs(HC);
                for (int c = 0; c < HC; ++c) {
                    const double gw = apply ? 2.0 / (1.0 + std::exp(-(double) inj[(size_t) c] / HC)) : 0.0;
                    double ss = 0.0;
                    for (int64_t dd = 0; dd < N; ++dd) {
                        const size_t i = (size_t) (c * N + dd);
                        Rp[i] = (double) Rt[i] + gw * bo[(size_t) dd];
                        Ra[i] = std::fabs((double) Rt[i]) + std::fabs(gw * bo[(size_t) dd]);
                        ss += Rp[i] * Rp[i];
                    }
                    rs[(size_t) c] = 1.0 / std::sqrt(ss / N + 1e-6);
                    for (int64_t dd = 0; dd < N; ++dd) {
                        const size_t i = (size_t) (c * N + dd);
                        xn[i] = Rp[i] * wn[i] * rs[(size_t) c];
                        xa[i] = Ra[i] * std::fabs((double) wn[i]) * rs[(size_t) c];
                    }
                }
                auto near = [&](double gpu, double ref, double bound, const char* what, int64_t i) {
                    if (!std::isfinite(ref) && std::isinf(gpu) && (gpu > 0) == (ref > 0)) return;
                    const double e = std::fabs(gpu - ref);
                    worst = std::max(worst, e / (bound + 1e-300));
                    if (!(e <= bound) && off++ < 5)
                        std::fprintf(stderr, "  gr_read FP64: n_tok %d variant %d: %s[%lld] %.9g vs %.9g (bound %.3g)\n", n_tok,
                                     variant, what, (long long) i, gpu, ref, bound);
                };
                for (int c = 0; c < HC; ++c) near(got[2][(size_t) c], rs[(size_t) c], 1e-5 * rs[(size_t) c], "rs", c);
                if (apply)
                    for (int64_t i = 0; i < D; ++i) near(got[0][(size_t) i], Rp[(size_t) i], 1e-6 * Ra[(size_t) i] + 1e-30, "R", i);
                std::vector<double> lo((size_t) LR), lo_b((size_t) LR);
                for (int64_t r = 0; r < LR + (inject ? HC : 0); ++r) {
                    const uint16_t* w = r < LR ? wd.data() + (size_t) (r * D) : wi.data() + (size_t) ((r - LR) * D);
                    double y = 0.0, mag = 0.0;
                    for (int64_t i = 0; i < D; ++i) {
                        y += bf(w[i]) * xn[(size_t) i];
                        mag += std::fabs(bf(w[i])) * xa[(size_t) i];
                    }
                    if (r < LR) {
                        const double x = y / HC;
                        lo[(size_t) r] = x / (1.0 + std::exp(-x));
                        lo_b[(size_t) r] = 1.1 * 1e-4 * mag / HC + 1e-30;
                        near(got[1][(size_t) r], lo[(size_t) r], lo_b[(size_t) r], "lo", r);
                    } else {
                        near(got[3][(size_t) (r - LR)], y, 1e-4 * mag + 1e-30, "inject", r - LR);
                    }
                }
                for (int64_t dd = 0; dd < N; ++dd) {
                    double mixed = 0.0, bound = 0.0;
                    for (int c = 0; c < HC; ++c) {
                        const size_t i = (size_t) (c * N + dd);
                        double u = 0.0, du = 0.0;
                        for (int64_t k = 0; k < LR; ++k) {
                            const double w = bf(wu[i * LR + (size_t) k]);
                            u += w * lo[(size_t) k];
                            du += std::fabs(w) * (1e-4 * std::fabs(lo[(size_t) k]) + lo_b[(size_t) k]);
                        }
                        const double sg = 1.0 / (1.0 + std::exp(-u));
                        mixed += xn[i] * sg / HC;
                        bound += (1e-4 * xa[i] + 0.25 * std::fabs(xn[i]) * du) / HC;
                    }
                    near(got[4][(size_t) dd], mixed, bound + 1e-30, "mixed", dd);
                }
            }
            for (int p = 0; p < 2; ++p)
                for (float* q : d[p]) cudaFree(q);
            cudaFree(d_bo);
            cudaFree(d_inj);
        }
    }
    for (void* p : {(void*) d_wd, (void*) d_wu, (void*) d_wi, (void*) d_wn, (void*) d_scratch}) cudaFree(p);
    bad += off;
    char msg[160];
    std::snprintf(msg, sizeof msg,
                  "each token bitwise its 1-token read (1-8 tokens, 3 variants); FP64 within bounds (largest error %.2g of its bound)",
                  worst);
    std::printf("gr_read: %s\n", bad ? "MISMATCH" : msg);
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

// ---- the prompt path's attention  vs  the decode kernel (INT8 and FP16 pools; 1 to 2051 selected cells)
int test_prefill_attn(std::mt19937& rng, cudaStream_t s) {
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    const int HD = (int) sh.head_dim, NH = (int) sh.n_head, NKV = (int) sh.n_head_kv;
    const int cells = 4096, pages = cells / (int) sh.page_size;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, sh);
    const size_t n_codes = (size_t) cells * NKV * HD, n_scales = n_codes / strata::kernels::KV_Q8_GROUP;
    std::vector<int32_t> table((size_t) pages);
    for (int i = 0; i < pages; ++i) table[(size_t) i] = (i * 3 + 1) % pages;   // a permuted page table
    int32_t* d_table = dev<int32_t>(table.size());
    up(d_table, table);
    auto half_bits = [&](float lo, float hi) {   // a random positive FP16 in [lo, hi)
        const float f = std::uniform_real_distribution<float>(lo, hi)(rng);
        uint32_t b;
        std::memcpy(&b, &f, 4);
        const int e = (int) ((b >> 23) & 0xFF) - 127 + 15;
        return (uint16_t) ((e << 10) | ((b >> 13) & 0x3FF));
    };
    std::vector<int8_t> kq(n_codes), vq(n_codes);
    for (auto* v : {&kq, &vq})
        for (auto& c : *v) c = (int8_t) ((int) (rng() % 255) - 127);
    std::vector<uint16_t> ks(n_scales), vs(n_scales), kh(n_codes), vh(n_codes);
    for (auto& x : ks) x = half_bits(0.002f, 0.05f);
    for (auto& x : vs) x = half_bits(0.002f, 0.05f);
    for (auto& x : kh) x = (uint16_t) (half_bits(0.01f, 1.0f) | ((rng() & 1u) << 15));
    for (auto& x : vh) x = (uint16_t) (half_bits(0.01f, 1.0f) | ((rng() & 1u) << 15));
    int8_t *d_kq = dev<int8_t>(n_codes), *d_vq = dev<int8_t>(n_codes);
    uint16_t *d_ks = dev<uint16_t>(n_scales), *d_vs = dev<uint16_t>(n_scales), *d_kh = dev<uint16_t>(n_codes),
             *d_vh = dev<uint16_t>(n_codes);
    up(d_kq, kq); up(d_vq, vq); up(d_ks, ks); up(d_vs, vs); up(d_kh, kh); up(d_vh, vh);
    const std::vector<int> widths = {1, 5, 64, 65, 700, 2051};
    const int n_q = (int) widths.size();
    std::vector<int32_t> ids((size_t) n_q * cap, 0), steps((size_t) n_q * strata::kernels::kStepCount, 0);
    for (int i = 0; i < n_q; ++i) {
        std::vector<int32_t> all(cells);
        for (int c = 0; c < cells; ++c) all[(size_t) c] = c;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + widths[(size_t) i]);
        std::copy(all.begin(), all.begin() + widths[(size_t) i], ids.begin() + (size_t) i * cap);
        steps[(size_t) i * strata::kernels::kStepCount + strata::kernels::kStepWidth] = widths[(size_t) i];
    }
    std::vector<float> q((size_t) n_q * NH * HD);
    for (auto& x : q) x = 3.0f * std::normal_distribution<float>(0.0f, 1.0f)(rng);
    int32_t *d_ids = dev<int32_t>(ids.size()), *d_steps = dev<int32_t>(steps.size());
    float *d_q = dev<float>(q.size()), *d_a = dev<float>(q.size()), *d_b = dev<float>(q.size());
    float* d_scratch = dev<float>((size_t) n_q * strata::kernels::qsa_decode_attn_scratch_floats(cap, sh));
    up(d_ids, ids); up(d_steps, steps); up(d_q, q);
    int bad = 0;
    double worst = 0.0;
    for (int int8 = 0; int8 < 2; ++int8) {
        strata::kernels::QsaAttnPools pools;
        pools.page_table = d_table;
        if (int8) { pools.k_q = d_kq; pools.v_q = d_vq; pools.k_scale = d_ks; pools.v_scale = d_vs; }
        else { pools.k_pool = d_kh; pools.v_pool = d_vh; }
        strata::kernels::qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, cap, sh, d_scratch, d_a, n_q, s);
        strata::kernels::qsa_prefill_attn(d_q, pools, d_ids, d_steps, cap, sh, d_b, n_q, s);
        check(cudaStreamSynchronize(s), "prefill attn");
        const std::vector<float> a = down(d_a, q.size()), b = down(d_b, q.size());
        for (int r = 0; r < n_q * NH; ++r) {   // relative to the largest value of the head's output row
            double big = 0.0, diff = 0.0;
            for (int d = 0; d < HD; ++d) {
                big = std::max(big, (double) std::fabs(a[(size_t) r * HD + d]));
                diff = std::max(diff, (double) std::fabs(a[(size_t) r * HD + d] - b[(size_t) r * HD + d]));
            }
            const double rel = diff / std::max(big, 1e-30);
            worst = std::max(worst, rel);
            if (!(rel <= 2e-5)) {
                if (bad < 5) std::fprintf(stderr, "prefill attn: %s pools, query %d head %d: relative difference %.3g\n",
                                          int8 ? "INT8" : "FP16", r / NH, r % NH, rel);
                ++bad;
            }
        }
    }
    for (void* p : {(void*) d_table, (void*) d_kq, (void*) d_vq, (void*) d_ks, (void*) d_vs, (void*) d_kh, (void*) d_vh,
                    (void*) d_ids, (void*) d_steps, (void*) d_q, (void*) d_a, (void*) d_b, (void*) d_scratch})
        cudaFree(p);
    std::printf("prefill_attn: %s (INT8 and FP16 pools, 1-2051 cells; largest relative difference %.2g)\n",
                bad ? "OUTSIDE TOLERANCE" : "within 2e-5 of the decode kernel", worst);
    return bad;
}

// ---- the prompt path's GDN conv and recurrence over a chunk  vs  the verify window's kernels, 8 tokens at a time
int test_prefill_gdn(std::mt19937& rng, cudaStream_t s) {
    const int C = 10240, HK = 16, HV = 48, S = 128, ZV = HV * S;
    const float eps = 1e-6f;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto uni = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    std::vector<float> w((size_t) C * 4), hist((size_t) C * 3), state((size_t) S * HV * S), gamma((size_t) S);
    for (auto& x : w) x = 0.5f * nd(rng);
    for (auto& x : gamma) x = uni(0.5f, 1.5f);
    float *d_w = dev<float>(w.size()), *d_gamma = dev<float>(gamma.size());
    up(d_w, w);
    up(d_gamma, gamma);
    int bad = 0;
    for (int T : {1, 2, 3, 5, 37}) {
        std::vector<float> qkv((size_t) T * C), gate((size_t) T * HV), beta((size_t) T * HV), z((size_t) T * ZV);
        for (auto& x : qkv) x = nd(rng);
        for (auto& x : hist) x = nd(rng);
        for (auto& x : state) x = 0.1f * nd(rng);
        for (auto& x : gate) x = uni(-2.0f, 0.0f);
        for (auto& x : beta) x = uni(0.0f, 1.0f);
        for (auto& x : z) x = nd(rng);
        float *d_qkv = dev<float>(qkv.size()), *d_gate = dev<float>(gate.size()), *d_beta = dev<float>(beta.size()),
              *d_z = dev<float>(z.size());
        up(d_qkv, qkv); up(d_gate, gate); up(d_beta, beta); up(d_z, z);
        float* d_hist[2];
        float* d_state[2];
        float* d_h[2];
        float* d_y[2];
        for (int p = 0; p < 2; ++p) {
            d_hist[p] = dev<float>(hist.size());
            d_state[p] = dev<float>(state.size());
            d_h[p] = dev<float>(qkv.size());
            d_y[p] = dev<float>(z.size());
            up(d_hist[p], hist);
            up(d_state[p], state);
        }
        uint16_t* d_y16 = dev<uint16_t>(z.size());
        // the verify window's kernels, a window of up to 8 tokens after another, each committed
        std::vector<int32_t> counts;
        for (int t0 = 0; t0 < T; t0 += 8) counts.push_back(std::min(8, T - t0));
        int32_t* d_counts = dev<int32_t>(counts.size());
        up(d_counts, counts);
        for (size_t i = 0; i < counts.size(); ++i) {
            const size_t t0 = i * 8;
            const int n = counts[i];
            strata::kernels::gdn_conv_l2_multi(d_hist[0], d_qkv + t0 * C, d_w, d_h[0] + t0 * C, C, 2 * HK, eps, n, s);
            strata::kernels::gdn_conv_commit(d_hist[0], d_qkv + t0 * C, C, d_counts + i, s);
            strata::kernels::gdn_step_norm_multi(d_state[0], d_h[0] + t0 * C, C, d_gate + t0 * HV, d_beta + t0 * HV,
                                                 d_z + t0 * ZV, d_gamma, eps, d_y[0] + t0 * ZV, HK, HV, n, d_counts + i, s);
        }
        // the prompt path, all T at once
        strata::prefill::gdn_conv(d_hist[1], d_qkv, d_w, d_h[1], T, eps, s);
        strata::prefill::gdn_scan(d_state[1], d_h[1], d_gate, d_beta, d_y[1], T, s);
        strata::prefill::gdn_out_norm(d_y[1], d_z, d_gamma, eps, d_y16, T, s);
        check(cudaStreamSynchronize(s), "prefill gdn");
        const int b = bitwise_diff(down(d_h[0], qkv.size()), down(d_h[1], qkv.size()), "conv output") +
                      bitwise_diff(down(d_y[0], z.size()), down(d_y[1], z.size()), "y") +
                      bitwise_diff(down(d_state[0], state.size()), down(d_state[1], state.size()), "state") +
                      bitwise_diff(down(d_hist[0], hist.size()), down(d_hist[1], hist.size()), "conv history");
        if (b) std::fprintf(stderr, "prefill gdn: T %d differs\n", T);
        bad += b;
        for (int p = 0; p < 2; ++p) {
            cudaFree(d_hist[p]);
            cudaFree(d_state[p]);
            cudaFree(d_h[p]);
            cudaFree(d_y[p]);
        }
        for (float* p : {d_qkv, d_gate, d_beta, d_z}) cudaFree(p);
        cudaFree(d_y16);
        cudaFree(d_counts);
    }
    cudaFree(d_w);
    cudaFree(d_gamma);
    std::printf("prefill_gdn: %s\n", bad ? "MISMATCH" : "bitwise equal to the verify window's kernels (1-37 tokens)");
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
    bad += test_hit_plan(rng, s);
    bad += test_mmvf_multi(rng, s);
    bad += test_router_multi(rng, s);
    bad += test_verify_router(rng, s);
    bad += test_gr_read(rng, s);
    bad += test_rope_tokens(rng, s);
    bad += test_kv_append(rng, s);
    bad += test_indexer_append(rng, s);
    bad += test_ple_tokens(rng, s);
    bad += test_prefill_attn(rng, s);
    bad += test_prefill_gdn(rng, s);
    cudaStreamDestroy(s);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
