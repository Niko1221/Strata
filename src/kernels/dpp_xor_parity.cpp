// src/kernels/dpp_xor_parity.cpp - STRATA_DPP_XOR: strata::hip_compat::shfl_xor_dpp against HIP's __shfl_xor, bit for bit.
//
//     build/dpp_xor_parity
//
// Every constant lane mask 1..31 and width 1..64 (the DPP path takes mask 1..16 under a power-of-two width up to 32;
// the rest must fall through to __shfl_xor), on 32-bit integers and floats (random bits: NaNs, infinities, -0 and
// denormals included), with the whole wave active and with lanes switched off (every other lane, lane 0 only off, the
// upper row off, random patterns): every active lane's result compared.  Also a runtime mask (the fallback).
// Exit code 1 on any difference; 77 where the DPP path is not built (not GFX10-12).
#include <cuda_runtime.h>   // (the HIP build's: hip_compat/intrinsics.hpp, shfl_xor_dpp and STRATA_DPP_XOR)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr int kBlocks = 64, kThreads = 128;   // 4 waves a block
constexpr int kPatterns = 8;

__device__ uint32_t pattern(int p, uint32_t salt) {
    switch (p) {
    case 0: return 0xffffffffu;
    case 1: return 0x55555555u;
    case 2: return 0xfffffffeu;
    case 3: return 0x0000ffffu;
    case 4: return 0xffff0000u;
    default: {   // per wave, from the salt
        uint32_t h = salt * 2654435761u + (uint32_t) p * 40503u;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        return h | 1u;
    }
    }
}

template <int M, int W>
__global__ void k_cmp(const uint32_t* __restrict__ in, uint32_t* __restrict__ a, uint32_t* __restrict__ b, int p, int as_float) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % 32;
    const uint32_t on = pattern(p, (uint32_t) (i / 32));
    const uint32_t v = in[i];
    a[i] = 0xdeadbeefu;
    b[i] = 0xdeadbeefu;
    if ((on >> lane) & 1u) {
        if (as_float) {
            const float f = __builtin_bit_cast(float, v);
            a[i] = __builtin_bit_cast(uint32_t, __shfl_xor(f, M, W));
            b[i] = __builtin_bit_cast(uint32_t, strata::hip_compat::shfl_xor_dpp(f, M, W));
        } else {
            a[i] = (uint32_t) __shfl_xor((int) v, M, W);
            b[i] = (uint32_t) strata::hip_compat::shfl_xor_dpp((int) v, M, W);
        }
    }
}
__global__ void k_runtime(const uint32_t* __restrict__ in, uint32_t* __restrict__ a, uint32_t* __restrict__ b, int m, int w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = (uint32_t) __shfl_xor((int) in[i], m, w);
    b[i] = (uint32_t) strata::hip_compat::shfl_xor_dpp((int) in[i], m, w);
}

struct Run {
    const uint32_t* in;
    uint32_t *a, *b;
    std::vector<uint32_t> ha, hb;
    long long cases = 0, bad = 0;
    template <int M, int W>
    void one() {
        for (int p = 0; p < kPatterns; ++p)
            for (int f = 0; f < 2; ++f) {
                k_cmp<M, W><<<kBlocks, kThreads>>>(in, a, b, p, f);
                cudaMemcpy(ha.data(), a, ha.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hb.data(), b, hb.size() * 4, cudaMemcpyDeviceToHost);
                ++cases;
                if (std::memcmp(ha.data(), hb.data(), ha.size() * 4) != 0) {
                    if (bad < 10) {
                        size_t j = 0;
                        while (ha[j] == hb[j]) ++j;
                        std::printf("  DIFF mask %d width %d pattern %d %s: lane %zu: shfl %08x dpp %08x\n", M, W, p,
                                    f ? "float" : "int", j, ha[j], hb[j]);
                    }
                    ++bad;
                }
            }
    }
    template <int W, int... Ms>
    void masks(std::integer_sequence<int, Ms...>) { (one<Ms + 1, W>(), ...); }
};

}  // namespace

int main() {
    int dev = 0;
    cudaDeviceProp pr{};
    cudaGetDeviceProperties(&pr, dev);
    std::printf("dpp_xor_parity on %s (%s), STRATA_DPP_XOR=%d\n", pr.name, pr.gcnArchName, STRATA_DPP_XOR);
    const size_t n = (size_t) kBlocks * kThreads;
    std::vector<uint32_t> h(n);
    std::mt19937 rng(5u);
    for (auto& x : h) x = rng();
    const uint32_t special[] = {0x7fc00000u, 0xffc00001u, 0x7f800001u, 0x7f800000u, 0xff800000u, 0x80000000u, 0x00000001u, 0u};
    for (size_t i = 0; i < n; i += 5) h[i] = special[(i / 5) % 8];
    Run r;
    uint32_t* in = nullptr;
    cudaMalloc(&in, n * 4);
    cudaMalloc(&r.a, n * 4);
    cudaMalloc(&r.b, n * 4);
    cudaMemcpy(in, h.data(), n * 4, cudaMemcpyHostToDevice);
    r.in = in;
    r.ha.resize(n);
    r.hb.resize(n);
    using Seq = std::make_integer_sequence<int, 31>;
    r.masks<1>(Seq{});
    r.masks<2>(Seq{});
    r.masks<4>(Seq{});
    r.masks<8>(Seq{});
    r.masks<16>(Seq{});
    r.masks<32>(Seq{});
    r.masks<64>(Seq{});
    r.masks<12>(Seq{});   // not a power of two: the fallback
    long long rt = 0, rbad = 0;
    for (int w : {1, 2, 4, 8, 16, 32})
        for (int m = 1; m < 32; ++m) {
            k_runtime<<<kBlocks, kThreads>>>(in, r.a, r.b, m, w);
            cudaMemcpy(r.ha.data(), r.a, n * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(r.hb.data(), r.b, n * 4, cudaMemcpyDeviceToHost);
            ++rt;
            if (std::memcmp(r.ha.data(), r.hb.data(), n * 4) != 0) ++rbad;
        }
    const cudaError_t e = cudaDeviceSynchronize();
    std::printf("  constant masks: %lld cases (31 masks x 8 widths x %d lane patterns x int/float), %lld differ\n", r.cases,
                kPatterns, r.bad);
    std::printf("  runtime masks: %lld cases, %lld differ\n", rt, rbad);
    const bool ok = e == cudaSuccess && r.bad == 0 && rbad == 0;
    std::printf("dpp_xor_parity: %s%s\n", ok ? "all bitwise" : "FAILED", e != cudaSuccess ? cudaGetErrorString(e) : "");
    return ok ? 0 : 1;
}
