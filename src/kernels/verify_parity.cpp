// src/kernels/verify_parity.cpp - the verify window's batched kernels against the kernel sequences they replace.
//
// A verify window's token t must come out bit for bit as it would in a window of any other size (the drafts are
// accepted exactly when greedy decode would have produced them), so every batched kernel is checked BITWISE against
// the per-token kernels it stands in for, on random inputs that include -0.0, denormals and large values.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_moe.hpp"
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
    cudaStreamDestroy(s);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
