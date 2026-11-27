// engine/src/inkling2/hip/ink_hip_gguf_types_test.hip.cpp - parity test of the HIP GGUF expert decode (hip_gguf_decode_8_f16 and the fdot2 linear kernel
// k_gguf_linear_fdot2 that all HIP GGUF_RAW paths share) against two independent CPU readings of the formats:
//   (1) a reference written from the GGUF block specs, with the codebooks of strata-quant-common.h (host tables, not the device copies),
//   (2) strata::quant::dequant_row (the engine's CPU path).
// Types: the six GSQ-RCO types (IQ2_XXS 16, IQ2_XS 17, IQ2_S 22, IQ3_S 21, IQ1_S 19, Q2_0 42) plus controls (IQ3_XXS 18, IQ4_NL 20, IQ4_XS 23, Q4_0 2, Q8_0 8,
// Q2_K 10, Q3_K 11, Q4_K 12, Q5_K 13, Q6_K 14; the controls against dequant_row only).
// Decode test: every 8-value group of every row, device fp16 against the reference (tolerance: one fp16 rounding step).
// Dot test: k_gguf_linear_fdot2 (M = 1 and 4, gate/up and down path) against the double-precision dot of the spec weights with the same fp16 activations.
// "bench": time of the linear kernel per type on a weight block larger than the L2.
//
// build (from the repo root, see phaseA-tests/rel0223/amd-rdna32/): hipcc -O2 -std=c++20 --offload-arch=gfx1100 -I engine/include -I engine/src/quant -I engine/src/inkling2/hip
//   engine/src/inkling2/hip/ink_hip_gguf_types_test.hip.cpp engine/src/quant/dequant.cpp engine/src/quant/types.cpp -o ink_hip_gguf_types_test
//
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#define GGML_COMMON_IMPL_CPP
#include "strata/quant/strata-quant-common.h"
#include "strata/quant/dequant.hpp"

#include "ink_hip_codecs.hip.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::inkling2::hip;

#define TCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static float h_f16(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    double v;
    if (e == 0) v = std::ldexp((double)m, -24);
    else if (e == 31) v = 65504.0;
    else v = std::ldexp(1.0 + m / 1024.0, (int)e - 15);
    return (float)(s ? -v : v);
}
static uint16_t f16_bits(float f) {   // round to nearest even, every value (the compiler's _Float16 conversion)
    const half_t h = static_cast<half_t>(f);
    return __builtin_bit_cast(uint16_t, h);
}

struct Tabs {
    std::vector<uint8_t> kmask, ksigns;
    std::vector<uint64_t> g2xxs, g2xs, g2s, g1s;
    std::vector<uint32_t> g3s, g3xxs;
};

static size_t blk_bytes(int t) {
    switch (t) { case 16: return 66; case 17: return 74; case 22: return 82; case 21: return 110; case 19: return 50; case 18: return 98; case 42: return 18; case 20: return 18;
                 case 23: return 136; case 2: return 18; case 8: return 34; case 10: return 84; case 11: return 110; case 12: return 144; case 13: return 176; case 14: return 210; }
    return 0;
}
static int blk_elems(int t) { return t == 42 ? 64 : ((t == 20 || t == 2 || t == 8) ? 32 : 256); }
static bool has_spec(int t) { return t == 16 || t == 17 || t == 22 || t == 21 || t == 19 || t == 18 || t == 42 || t == 20; }
static const char* tname(int t) {
    switch (t) { case 16: return "IQ2_XXS"; case 17: return "IQ2_XS"; case 22: return "IQ2_S"; case 21: return "IQ3_S"; case 19: return "IQ1_S"; case 18: return "IQ3_XXS"; case 42: return "Q2_0";
                 case 20: return "IQ4_NL"; case 23: return "IQ4_XS"; case 2: return "Q4_0"; case 8: return "Q8_0"; case 10: return "Q2_K"; case 11: return "Q3_K"; case 12: return "Q4_K";
                 case 13: return "Q5_K"; case 14: return "Q6_K"; }
    return "?";
}

// reference dequantisation of one row of K elements, from the block specs (same text as the CUDA test)
static void ref_row(int t, const uint8_t* src, int K, const Tabs& T, double* y) {
    const int nb = K / blk_elems(t);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = src + (size_t)i * blk_bytes(t);
        uint16_t dh; std::memcpy(&dh, b, 2);
        const double d = h_f16(dh);
        double* o = y + (size_t)i * blk_elems(t);
        if (t == 42) {
            for (int j = 0; j < 64; ++j) o[j] = (double)(((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
        } else if (t == 16) {
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t a0, a1; std::memcpy(&a0, b + 2 + 8 * ib, 4); std::memcpy(&a1, b + 2 + 8 * ib + 4, 4);
                const double db = d * (0.5 + (a1 >> 28)) * 0.25;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t g = T.g2xxs[(a0 >> (8 * l)) & 255];
                    const uint8_t sg = T.ksigns[(a1 >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 17) {
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t sc = b[66 + ib];
                for (int l = 0; l < 4; ++l) {
                    uint16_t qv; std::memcpy(&qv, b + 2 + 8 * ib + 2 * l, 2);
                    const double db = d * (0.5 + ((l < 2) ? (sc & 15) : (sc >> 4))) * 0.25;
                    const uint64_t g = T.g2xs[qv & 511];
                    const uint8_t sg = T.ksigns[qv >> 9];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 22) {
            const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sgs = qs + 32; const uint8_t* scl = b + 74;
            for (int ib = 0; ib < 8; ++ib) {
                for (int l = 0; l < 4; ++l) {
                    const double db = d * (0.5 + ((l < 2) ? (scl[ib] & 15) : (scl[ib] >> 4))) * 0.25;
                    const uint64_t g = T.g2s[qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
                    const uint8_t sg = sgs[ib * 4 + l];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 21) {
            const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sgs = b + 74; const uint8_t* scl = b + 106;
            for (int ib = 0; ib < 8; ++ib) {
                const double db = d * (1 + 2 * ((ib & 1) ? (scl[ib / 2] >> 4) : (scl[ib / 2] & 15)));
                for (int l = 0; l < 4; ++l) {
                    const uint32_t g1 = T.g3s[qs[ib * 8 + 2 * l] | ((qh[ib] << (8 - 2 * l)) & 256)];
                    const uint32_t g2 = T.g3s[qs[ib * 8 + 2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256)];
                    const uint8_t sg = sgs[ib * 4 + l];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + l * 8 + j]     = db * (double)((g1 >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                        o[ib * 32 + l * 8 + j + 4] = db * (double)((g2 >> (8 * j)) & 255) * ((sg & T.kmask[j + 4]) ? -1.0 : 1.0);
                    }
                }
            }
        } else if (t == 19) {
            for (int ib = 0; ib < 8; ++ib) {
                uint16_t qh; std::memcpy(&qh, b + 34 + 2 * ib, 2);
                const double dl = d * (2 * ((qh >> 12) & 7) + 1);
                const double delta = (qh & 0x8000) ? -0.125 : 0.125;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t g = T.g1s[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = dl * ((double)(int8_t)((g >> (8 * j)) & 255) + delta);
                }
            }
        } else if (t == 18) {
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, b + 66 + 4 * ib, 4);
                const double db = d * (0.5 + (aux >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sg = T.ksigns[(aux >> (7 * l)) & 127];
                    const uint32_t g1 = T.g3xxs[b[2 + ib * 8 + 2 * l]], g2 = T.g3xxs[b[2 + ib * 8 + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + l * 8 + j]     = db * (double)((g1 >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                        o[ib * 32 + l * 8 + j + 4] = db * (double)((g2 >> (8 * j)) & 255) * ((sg & T.kmask[j + 4]) ? -1.0 : 1.0);
                    }
                }
            }
        } else if (t == 20) {
            static const int lut[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            for (int j = 0; j < 16; ++j) { o[j] = d * lut[b[2 + j] & 15]; o[j + 16] = d * lut[b[2 + j] >> 4]; }
        }
    }
}

// one 8-value group per thread, the result as 8 fp16 -> fp32 on the host
__global__ void k_decode_all(uint32_t t, const uint8_t* rows, size_t rb, int N, int K, float ext, uint4* out) {
    const int k8n = K >> 3;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * k8n) return;
    const int r = idx / k8n, k8 = idx - r * k8n;
    out[idx] = hip_gguf_decode_8_f16(t, rows + (size_t)r * rb, k8 << 3, ext);
}

static void make_rows(int t, int K, int N, std::mt19937& rng, std::vector<uint8_t>& w, size_t& rb) {
    rb = hip_gguf_row_bytes((uint32_t)t, K);
    w.resize(rb * N);
    for (auto& b : w) b = (uint8_t)(rng() & 255);
    std::uniform_real_distribution<float> dsc(0.002f, 0.05f);
    const size_t bb = blk_bytes(t);
    for (int r = 0; r < N; ++r)
        for (size_t i = 0; i < rb / bb; ++i) {
            float s = dsc(rng);
            if (t == 42 && (rng() & 1)) s = -s;   // Q2_0 scale is signed
            const uint16_t h = f16_bits(s);
            std::memcpy(&w[(size_t)r * rb + i * bb], &h, 2);
            if (t == 12 || t == 13) { const uint16_t h2 = f16_bits(dsc(rng)); std::memcpy(&w[(size_t)r * rb + i * bb + 2], &h2, 2); }   // dmin
            if (t == 10) { const uint16_t h2 = f16_bits(s); std::memcpy(&w[(size_t)r * rb + i * bb + 80], &h2, 2); const uint16_t h3 = f16_bits(dsc(rng)); std::memcpy(&w[(size_t)r * rb + i * bb + 82], &h3, 2); }
            if (t == 11) { const uint16_t h2 = f16_bits(s); std::memcpy(&w[(size_t)r * rb + i * bb + 108], &h2, 2); }
            if (t == 14) { const uint16_t h2 = f16_bits(s); std::memcpy(&w[(size_t)r * rb + i * bb + 208], &h2, 2); }
        }
    // d is at offset 0 for every type except Q2_K, Q3_K, Q6_K (offsets above); their first 2 bytes are scale/quant data: harmless
}

static int run_type(int t, const Tabs& T, std::mt19937& rng) {
    const int N = 96, M = 4;
    const int K = (t == 42 || t == 20 || t == 8 || t == 2) ? 1152 : 2304;
    std::vector<uint8_t> w; size_t rb;
    make_rows(t, K, N, rng, w, rb);
    // reference rows
    std::vector<double> yspec;   // [N][K]
    std::vector<float> ycpu((size_t)N * K);
    bool cpu_ok = true;
    for (int r = 0; r < N; ++r)
        if (!strata::quant::dequant_row((strata::quant::Type)t, &w[(size_t)r * rb], &ycpu[(size_t)r * K], K)) cpu_ok = false;
    if (has_spec(t)) {
        yspec.resize((size_t)N * K);
        for (int r = 0; r < N; ++r) ref_row(t, &w[(size_t)r * rb], K, T, &yspec[(size_t)r * K]);
    }
    uint8_t* d_w = nullptr; uint4* d_out = nullptr;
    TCK(hipMalloc(&d_w, w.size())); TCK(hipMalloc(&d_out, (size_t)N * (K / 8) * sizeof(uint4)));
    TCK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
    const int ng = N * (K / 8);
    k_decode_all<<<(ng + 255) / 256, 256>>>((uint32_t)t, d_w, rb, N, K, 1.0f, d_out);
    TCK(hipGetLastError()); TCK(hipDeviceSynchronize());
    std::vector<uint16_t> hv((size_t)N * K);
    TCK(hipMemcpy(hv.data(), d_out, (size_t)N * K * 2, hipMemcpyDeviceToHost));
    int fails = 0;
    double worst_spec = 0, worst_cpu = 0, scale = 0;
    long bad_spec = 0, bad_cpu = 0;
    for (size_t i = 0; i < (size_t)N * K; ++i) {
        const double g = h_f16(hv[i]);
        const double refc = ycpu[i];
        scale = std::max(scale, std::fabs(refc));
        const double tolc = std::fabs(refc) * (1.0 / 1024.0) + 7e-8;
        worst_cpu = std::max(worst_cpu, std::fabs(g - refc) / tolc);
        if (std::fabs(g - refc) > tolc) ++bad_cpu;
        if (has_spec(t)) {
            const double refs = yspec[i];
            const double tols = std::fabs(refs) * (1.0 / 1024.0) + 7e-8;
            worst_spec = std::max(worst_spec, std::fabs(g - refs) / tols);
            if (std::fabs(g - refs) > tols) ++bad_spec;
        }
    }
    std::printf("  type %2d %-7s decode K=%4d: %d values, vs strata::quant %ld off (worst %.2f of tolerance)%s", t, tname(t), K, N * K, bad_cpu, worst_cpu, cpu_ok ? "" : " [dequant_row refused]");
    if (has_spec(t)) std::printf(", vs block spec %ld off (worst %.2f)", bad_spec, worst_spec);
    const bool dec_ok = cpu_ok && bad_cpu == 0 && bad_spec == 0;
    std::printf(" %s\n", dec_ok ? "ok" : "FAIL");
    fails += dec_ok ? 0 : 1;
    hipFree(d_out);

    // ---- dot test through k_gguf_linear_fdot2: a synthetic expert body: header, gate/up rows (N), down rows (N)
    const size_t dn_off = (32 + rb * N + 31) & ~(size_t)31;
    std::vector<uint8_t> body(dn_off + rb * N + 64, 0);
    HipGgufExpertHeader hdr{0x47475546u, (uint32_t)t, (uint32_t)t, 1u, 1.0f, 1.0f, 1.0f, 0u};
    std::memcpy(body.data(), &hdr, sizeof(hdr));
    std::memcpy(body.data() + 32, w.data(), rb * N);
    std::memcpy(body.data() + dn_off, w.data(), rb * N);
    uint8_t* d_body = nullptr; TCK(hipMalloc(&d_body, body.size()));
    TCK(hipMemcpy(d_body, body.data(), body.size(), hipMemcpyHostToDevice));
    std::normal_distribution<float> dn(0.0f, 1.0f);
    std::vector<uint16_t> xh((size_t)M * K);
    std::vector<double> xd((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < K; ++i) {
            float v = dn(rng);
            if (i == 2027 % K) v *= 30.0f;
            const uint16_t b = f16_bits(v);
            xh[(size_t)m * K + i] = b;
            xd[(size_t)m * K + i] = h_f16(b);
        }
    half_t* d_x = nullptr; float* d_y = nullptr;
    TCK(hipMalloc(&d_x, xh.size() * 2)); TCK(hipMalloc(&d_y, (size_t)M * N * 4));
    TCK(hipMemcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
    std::vector<double> yc;
    if (!has_spec(t)) { yc.assign(ycpu.begin(), ycpu.end()); }
    const std::vector<double>& wr = has_spec(t) ? yspec : yc;
    for (int path = 0; path < 2; ++path) {          // 0 = gate/up path (is_w13), 1 = down path
        for (int nm : {1, M}) {
            TCK(hipMemset(d_y, 0, (size_t)M * N * 4));
            launch_gguf_linear(d_body, (uint32_t)dn_off, path == 0, d_x, d_y, nm, N, K, N, 0);
            TCK(hipGetLastError()); TCK(hipDeviceSynchronize());
            std::vector<float> y((size_t)M * N);
            TCK(hipMemcpy(y.data(), d_y, y.size() * 4, hipMemcpyDeviceToHost));
            double worst = 0; int bad = 0;
            for (int m = 0; m < nm; ++m)
                for (int r = 0; r < N; ++r) {
                    double s = 0, sa = 0;
                    for (int i = 0; i < K; ++i) { const double p = wr[(size_t)r * K + i] * xd[(size_t)m * K + i]; s += p; sa += std::fabs(p); }
                    const double e = std::fabs((double)y[(size_t)m * N + r] - s) / (sa + 1e-30);
                    worst = std::max(worst, e);
                    if (e > 2e-3) ++bad;
                }
            std::printf("  type %2d %-7s %s linear M=%d: worst error vs sum|w x| %.3g, %d of %d dots above 2e-3 %s\n", t, tname(t), path == 0 ? "gate/up" : "down   ", nm, worst, bad, nm * N, bad ? "FAIL" : "ok");
            fails += bad ? 1 : 0;
        }
    }
    hipFree(d_w); hipFree(d_body); hipFree(d_x); hipFree(d_y);
    return fails;
}

static void bench_all() {
    const int types[] = {16, 17, 22, 21, 19, 18, 42, 20, 23, 12, 14, 8};
    const int N = 147456;
    for (int t : types) {
        const int K = (t == 42 || t == 20 || t == 8) ? 1152 : 2304;
        const size_t rb = hip_gguf_row_bytes((uint32_t)t, K);
        std::vector<uint8_t> w(rb * (size_t)N + 64);
        std::mt19937 rng(1);
        for (auto& b : w) b = (uint8_t)(rng() & 255);
        const size_t dn_off = 32;
        HipGgufExpertHeader hdr{0x47475546u, (uint32_t)t, (uint32_t)t, 1u, 1.0f, 1.0f, 1.0f, 0u};
        std::memcpy(w.data(), &hdr, sizeof(hdr));
        uint8_t* d_w = nullptr; TCK(hipMalloc(&d_w, w.size()));
        TCK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
        const int M = 16;
        std::vector<uint16_t> xh((size_t)M * K, f16_bits(0.5f));
        half_t* d_x = nullptr; float* d_y = nullptr;
        TCK(hipMalloc(&d_x, xh.size() * 2)); TCK(hipMalloc(&d_y, (size_t)M * N * 4));
        TCK(hipMemcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
        hipEvent_t e0, e1; TCK(hipEventCreate(&e0)); TCK(hipEventCreate(&e1));
        for (int nm : {1, 16}) {
            launch_gguf_linear(d_w, (uint32_t)dn_off, true, d_x, d_y, nm, N, K, N, 0);
            TCK(hipDeviceSynchronize());
            TCK(hipEventRecord(e0, 0));
            for (int i = 0; i < 5; ++i) launch_gguf_linear(d_w, (uint32_t)dn_off, true, d_x, d_y, nm, N, K, N, 0);
            TCK(hipEventRecord(e1, 0)); TCK(hipEventSynchronize(e1));
            float ms = 0; TCK(hipEventElapsedTime(&ms, e0, e1)); ms /= 5;
            std::printf("type %2d %-7s K=%4d %2d tokens: %8.3f ms  %7.1f GB/s of weights\n", t, tname(t), K, nm, ms, (double)(rb * (size_t)N) / (ms * 1e-3) / 1e9);
        }
        hipFree(d_w); hipFree(d_x); hipFree(d_y);
    }
}

// ./ink_hip_gguf_types_test gemm: the dense BF16 GEMM of the cards without matrix cores, 64 x 64 tiles (k_bf16_fdot2_tiled64) against the old 16 x 32 kernel: results must be bit-identical; timing in TFLOPs
static int run_gemm_test(bool bench_too) {
    struct Shape { int M, N, K; };
    const Shape shapes[] = {{256, 12288, 2304}, {256, 2304, 4096}, {100, 1000, 2304}, {33, 65, 64}, {512, 4096, 2304}};
    int fails = 0;
    std::mt19937 rng(7);
    std::normal_distribution<float> dn(0.0f, 1.0f);
    for (const Shape& s : shapes) {
        std::vector<uint16_t> w((size_t)s.N * s.K);
        for (auto& v : w) { float f = dn(rng) * 0.03f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
        std::vector<float> x((size_t)s.M * s.K), bias(s.N);
        for (auto& v : x) v = dn(rng);
        for (auto& v : bias) v = dn(rng);
        uint16_t* d_w = nullptr; float *d_x = nullptr, *d_b = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
        TCK(hipMalloc(&d_w, w.size() * 2)); TCK(hipMalloc(&d_x, x.size() * 4)); TCK(hipMalloc(&d_b, s.N * 4));
        TCK(hipMalloc(&d_y1, (size_t)s.M * s.N * 4)); TCK(hipMalloc(&d_y2, (size_t)s.M * s.N * 4));
        TCK(hipMemcpy(d_w, w.data(), w.size() * 2, hipMemcpyHostToDevice)); TCK(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice)); TCK(hipMemcpy(d_b, bias.data(), s.N * 4, hipMemcpyHostToDevice));
        const dim3 g1((s.N + 31) / 32, (s.M + 15) / 16), g2((s.N + 63) / 64, (s.M + 63) / 64);
        k_bf16_fdot2_tiled<16, 32><<<g1, 128>>>(d_w, d_x, d_b, d_y1, s.M, s.N, s.K);
        k_bf16_fdot2_tiled64<<<g2, 256>>>(d_w, d_x, d_b, d_y2, s.M, s.N, s.K);
        TCK(hipGetLastError()); TCK(hipDeviceSynchronize());
        std::vector<float> y1((size_t)s.M * s.N), y2(y1.size());
        TCK(hipMemcpy(y1.data(), d_y1, y1.size() * 4, hipMemcpyDeviceToHost)); TCK(hipMemcpy(y2.data(), d_y2, y2.size() * 4, hipMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < y1.size(); ++i) if (std::memcmp(&y1[i], &y2[i], 4) != 0) ++diff;
        std::printf("  gemm M=%d N=%d K=%d: %zu of %zu outputs differ from the old kernel %s\n", s.M, s.N, s.K, diff, y1.size(), diff ? "FAIL" : "ok");
        fails += diff ? 1 : 0;
        if (bench_too && s.M >= 100 && s.N >= 2304) {
            hipEvent_t e0, e1; TCK(hipEventCreate(&e0)); TCK(hipEventCreate(&e1));
            for (int v = 0; v < 2; ++v) {
                TCK(hipEventRecord(e0, 0));
                for (int i = 0; i < 10; ++i) { if (v == 0) k_bf16_fdot2_tiled<16, 32><<<g1, 128>>>(d_w, d_x, d_b, d_y1, s.M, s.N, s.K); else k_bf16_fdot2_tiled64<<<g2, 256>>>(d_w, d_x, d_b, d_y2, s.M, s.N, s.K); }
                TCK(hipEventRecord(e1, 0)); TCK(hipEventSynchronize(e1));
                float ms = 0; TCK(hipEventElapsedTime(&ms, e0, e1)); ms /= 10;
                std::printf("    %s: %.3f ms, %.2f TFLOPs\n", v == 0 ? "16x32 tile (old)" : "64x64 tile (new)", ms, 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e12);
            }
        }
        hipFree(d_w); hipFree(d_x); hipFree(d_b); hipFree(d_y1); hipFree(d_y2);
    }
    return fails;
}

// ./ink_hip_gguf_types_test moe: the grouped MoE kernels in their tiled form (k_moe_grouped_w13_situ_wmma / k_moe_grouped_w2_wmma: WMMA on RDNA3, fdot2 on RDNA2) against the per-row kernels
// (k_moe_grouped_w13_situ / k_moe_grouped_w2) on one synthetic GGUF expert with 100 tokens (D = 2304, inter = 1152)
static int run_moe_test(int gt, int dt, std::mt19937& rng, bool bench_too) {
    const int D = 2304, inter = 1152, P = 100, NE = 128;
    std::vector<uint8_t> wgu, wdn; size_t rb_gu, rb_dn;
    make_rows(gt, D, 2 * inter, rng, wgu, rb_gu);
    make_rows(dt, inter, D, rng, wdn, rb_dn);
    const size_t dn_off = (32 + wgu.size() + 31) & ~(size_t)31;
    std::vector<uint8_t> body(dn_off + wdn.size() + 64, 0);
    HipGgufExpertHeader hdr{0x47475546u, (uint32_t)gt, (uint32_t)dt, 1u, 1.0f, 1.0f, 1.0f, 0u};
    std::memcpy(body.data(), &hdr, sizeof(hdr));
    std::memcpy(body.data() + 32, wgu.data(), wgu.size());
    std::memcpy(body.data() + dn_off, wdn.data(), wdn.size());
    uint8_t* d_body = nullptr; TCK(hipMalloc(&d_body, body.size())); TCK(hipMemcpy(d_body, body.data(), body.size(), hipMemcpyHostToDevice));
    std::vector<GpuExpertDesc> ex(NE, GpuExpertDesc{nullptr, 0, 0, 0, 0});
    ex[0] = GpuExpertDesc{d_body, (uint32_t)dn_off, (uint8_t)CODE_GGUF_RAW, (uint8_t)CODE_GGUF_RAW, 0};
    GpuExpertDesc* d_ex = nullptr; TCK(hipMalloc(&d_ex, NE * sizeof(GpuExpertDesc))); TCK(hipMemcpy(d_ex, ex.data(), NE * sizeof(GpuExpertDesc), hipMemcpyHostToDevice));
    std::vector<int32_t> off(NE + 1, P); off[0] = 0;
    std::vector<int32_t> btk(P); for (int i = 0; i < P; ++i) btk[i] = i;
    int32_t *d_off = nullptr, *d_btk = nullptr;
    TCK(hipMalloc(&d_off, off.size() * 4)); TCK(hipMemcpy(d_off, off.data(), off.size() * 4, hipMemcpyHostToDevice));
    TCK(hipMalloc(&d_btk, P * 4)); TCK(hipMemcpy(d_btk, btk.data(), P * 4, hipMemcpyHostToDevice));
    std::normal_distribution<float> dn(0.0f, 1.0f);
    std::vector<uint16_t> xh((size_t)P * D);
    for (auto& v : xh) v = f16_bits(dn(rng));
    half_t* d_x = nullptr; half_t *d_a1 = nullptr, *d_a2 = nullptr; float *d_y1 = nullptr, *d_y2 = nullptr;
    TCK(hipMalloc(&d_x, xh.size() * 2)); TCK(hipMemcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
    TCK(hipMalloc(&d_a1, (size_t)P * inter * 2)); TCK(hipMalloc(&d_a2, (size_t)P * inter * 2));
    TCK(hipMalloc(&d_y1, (size_t)P * D * 4)); TCK(hipMalloc(&d_y2, (size_t)P * D * 4));
    TCK(hipMemset(d_a1, 0, (size_t)P * inter * 2)); TCK(hipMemset(d_a2, 0, (size_t)P * inter * 2)); TCK(hipMemset(d_y1, 0, (size_t)P * D * 4)); TCK(hipMemset(d_y2, 0, (size_t)P * D * 4));
    launch_moe_w13_rdna2(d_ex, d_off, d_btk, d_x, d_a1, 64, D, inter, NE, 4.0f, 25.0f, 0);     // B < 96: the per-row kernel
    launch_moe_w13_rdna2(d_ex, d_off, d_btk, d_x, d_a2, 128, D, inter, NE, 4.0f, 25.0f, 0);    // tiles
    launch_moe_w2_rdna2(d_ex, d_off, d_a1, d_y1, 64, D, inter, NE, 0);
    launch_moe_w2_rdna2(d_ex, d_off, d_a1, d_y2, 128, D, inter, NE, 0);
    TCK(hipGetLastError()); TCK(hipDeviceSynchronize());
    std::vector<uint16_t> a1((size_t)P * inter), a2(a1.size());
    std::vector<float> y1((size_t)P * D), y2(y1.size());
    TCK(hipMemcpy(a1.data(), d_a1, a1.size() * 2, hipMemcpyDeviceToHost)); TCK(hipMemcpy(a2.data(), d_a2, a2.size() * 2, hipMemcpyDeviceToHost));
    TCK(hipMemcpy(y1.data(), d_y1, y1.size() * 4, hipMemcpyDeviceToHost)); TCK(hipMemcpy(y2.data(), d_y2, y2.size() * 4, hipMemcpyDeviceToHost));
    double ss = 0, worst = 0, ssy = 0, worsty = 0;
    for (size_t i = 0; i < a1.size(); ++i) ss += (double)h_f16(a1[i]) * h_f16(a1[i]);
    for (size_t i = 0; i < a1.size(); ++i) worst = std::max(worst, std::fabs((double)h_f16(a1[i]) - h_f16(a2[i])));
    for (size_t i = 0; i < y1.size(); ++i) { ssy += (double)y1[i] * y1[i]; worsty = std::max(worsty, std::fabs((double)y1[i] - y2[i])); }
    const double rms = std::sqrt(ss / a1.size()), rmsy = std::sqrt(ssy / y1.size());
    const bool ok = rms > 0 && rmsy > 0 && worst / rms < 5e-3 && worsty / rmsy < 5e-3;
    std::printf("  moe tiles vs per-row kernels, gate/up %s + down %s, %d tokens: w13 max diff %.3g of rms %.3g, w2 max diff %.3g of rms %.3g %s\n", tname(gt), tname(dt), P, worst, rms, worsty, rmsy, ok ? "ok" : "FAIL");
    if (bench_too) {
        hipEvent_t e0, e1; TCK(hipEventCreate(&e0)); TCK(hipEventCreate(&e1));
        for (int v = 0; v < 2; ++v) {
            const int Bsel = v == 0 ? 64 : 128;
            launch_moe_w13_rdna2(d_ex, d_off, d_btk, d_x, d_a2, Bsel, D, inter, NE, 4.0f, 25.0f, 0); TCK(hipDeviceSynchronize());
            TCK(hipEventRecord(e0, 0));
            for (int i = 0; i < 10; ++i) { launch_moe_w13_rdna2(d_ex, d_off, d_btk, d_x, d_a2, Bsel, D, inter, NE, 4.0f, 25.0f, 0); launch_moe_w2_rdna2(d_ex, d_off, d_a1, d_y2, Bsel, D, inter, NE, 0); }
            TCK(hipEventRecord(e1, 0)); TCK(hipEventSynchronize(e1));
            float ms = 0; TCK(hipEventElapsedTime(&ms, e0, e1)); ms /= 10;
            std::printf("    %s: w13 + w2 of one expert with %d tokens %.3f ms\n", v == 0 ? "per-row kernels" : "tiled kernels  ", P, ms);
        }
    }
    hipFree(d_body); hipFree(d_ex); hipFree(d_off); hipFree(d_btk); hipFree(d_x); hipFree(d_a1); hipFree(d_a2); hipFree(d_y1); hipFree(d_y2);
    return ok ? 0 : 1;
}

// ./ink_hip_gguf_types_test pcie: host <-> device copy bandwidth with pinned memory (what the hybrid expert streaming gets on this card)
static void bench_pcie() {
    const size_t n = 256ull << 20;
    void* h = nullptr; void* d = nullptr;
    TCK(hipHostMalloc(&h, n, hipHostMallocDefault)); TCK(hipMalloc(&d, n));
    std::memset(h, 1, n);
    hipEvent_t e0, e1; TCK(hipEventCreate(&e0)); TCK(hipEventCreate(&e1));
    for (int dir = 0; dir < 2; ++dir) {
        TCK(hipMemcpy(dir == 0 ? d : h, dir == 0 ? h : d, n, dir == 0 ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost));
        TCK(hipEventRecord(e0, 0));
        for (int i = 0; i < 8; ++i) TCK(hipMemcpyAsync(dir == 0 ? d : h, dir == 0 ? h : d, n, dir == 0 ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost, 0));
        TCK(hipEventRecord(e1, 0)); TCK(hipEventSynchronize(e1));
        float ms = 0; TCK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("pinned %s: %.2f GB/s (8 x 256 MiB)\n", dir == 0 ? "host -> device" : "device -> host", 8.0 * n / (ms * 1e-3) / 1e9);
    }
    // small copies, the size of one expert body (3.1 MB), 64 in a row
    const size_t eb = 3ull << 20;
    TCK(hipEventRecord(e0, 0));
    for (int i = 0; i < 64; ++i) TCK(hipMemcpyAsync((char*)d + (size_t)i * eb, (char*)h + (size_t)i * eb, eb, hipMemcpyHostToDevice, 0));
    TCK(hipEventRecord(e1, 0)); TCK(hipEventSynchronize(e1));
    float ms = 0; TCK(hipEventElapsedTime(&ms, e0, e1));
    std::printf("pinned host -> device in 3 MiB pieces: %.2f GB/s\n", 64.0 * eb / (ms * 1e-3) / 1e9);
}

int main(int argc, char** argv) {
    hipDeviceProp_t dp{};
    TCK(hipGetDeviceProperties(&dp, 0));
    std::printf("Device: %s (%s)\n", dp.name, dp.gcnArchName);
    if (argc > 1 && std::strcmp(argv[1], "bench") == 0) { bench_all(); return 0; }
    if (argc > 1 && std::strcmp(argv[1], "pcie") == 0) { bench_pcie(); return 0; }
    if (argc > 1 && std::strcmp(argv[1], "moe") == 0) {
        std::mt19937 r(11); int f = 0;
        f += run_moe_test(21, 20, r, true); f += run_moe_test(16, 20, r, false); f += run_moe_test(18, 42, r, false); f += run_moe_test(17, 20, r, false); f += run_moe_test(22, 20, r, false); f += run_moe_test(19, 20, r, false);
        std::printf(f ? "MOE FAILED\n" : "MOE OK\n"); return f ? 1 : 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "gemm") == 0) { const int f = run_gemm_test(true); std::printf(f ? "GEMM FAILED\n" : "GEMM OK\n"); return f ? 1 : 0; }
    Tabs T;
    T.kmask.assign(kmask_iq2xs, kmask_iq2xs + 8);
    T.ksigns.assign(ksigns_iq2xs, ksigns_iq2xs + 128);
    T.g2xxs.assign(iq2xxs_grid, iq2xxs_grid + 256);
    T.g2xs.assign(iq2xs_grid, iq2xs_grid + 512);
    T.g2s.assign(iq2s_grid, iq2s_grid + 1024);
    T.g1s.assign(iq1s_grid, iq1s_grid + 2048);
    T.g3s.assign(iq3s_grid, iq3s_grid + 512);
    T.g3xxs.assign(iq3xxs_grid, iq3xxs_grid + 256);
    std::mt19937 rng(20261126);
    int fails = 0;
    const int types[] = {16, 17, 22, 21, 19, 18, 42, 20, 23, 2, 8, 10, 11, 12, 13, 14};
    for (int t : types) fails += run_type(t, T, rng);
    if (fails == 0) std::printf("ALL HIP GGUF TYPE TESTS PASSED\n");
    else            std::printf("FAILED: %d\n", fails);
    return fails ? 1 : 0;
}
