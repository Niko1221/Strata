// Adapted from native_mmvq.cu and llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "q4_pair_reuse.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int kWarp = 32, kWarps = 4;
struct Q4Block { half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
struct Q8Block { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q4Block) == 144 && alignof(Q4Block) == 4 && offsetof(Q4Block, qs) == 16);
static_assert(sizeof(Q8Block) == 36 && alignof(Q8Block) == 4 && offsetof(Q8Block, qs) == 4);

struct Activation {
    int u[4];
    float d[2];
    int sum[2];
};

// Read only after pdl_wait(). The integer sums cover eight quantized values
// each; Q8_1's half sum covers the original floats and cannot replace them.
__device__ __forceinline__ Activation load_activation(const Q8Block* x, int iqs) {
    Activation a;
    const int off = 2 * ((iqs / 2) / 4);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        a.d[i] = __low2float(x[off + i].ds);
        const int* q = reinterpret_cast<const int*>(x[off + i].qs) + ((iqs / 2) % 4);
        a.u[2 * i] = q[0];
        a.u[2 * i + 1] = q[4];
        a.sum[i] = STRATA_DP4A(0x01010101, a.u[2 * i + 1], STRATA_DP4A(0x01010101, a.u[2 * i], 0));
    }
    return a;
}

__device__ __forceinline__ float dot(const Q4Block* w, const Activation& a, int iqs) {
    const int off = 2 * ((iqs / 2) / 4);
    const int* q = reinterpret_cast<const int*>(w->qs + 16 * off + 4 * ((iqs / 2) % 4));
    const int v[2] = {q[0], q[4]};
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(w->scales);
    const int j = off / 2, jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
    float sd = 0.f, sm = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0 = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1 = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int d = STRATA_DP4A(v1, a.u[2 * i + 1], STRATA_DP4A(v0, a.u[2 * i], 0));
        // Keep the integer products, float conversion and accumulation order
        // of q4_q8_dot_impl, including its final multiply/subtract contraction.
        sd += a.d[i] * (d * sc[i]);
        sm += a.d[i] * (a.sum[i] * m[i]);
    }
    const float2 dm = __half22float2(w->dm);
    return dm.x * sd - dm.y * sm;
}

template<int Rows, bool Pdl>
__launch_bounds__(kWarps * kWarp, 1)
__global__ void q4_pair_reuse_kernel(const Q4Block* __restrict__ gate,
        const Q4Block* __restrict__ up, const Q8Block* xbase, float* yg, float* yu,
        int ni, int no, std::size_t gs, std::size_t us, std::size_t xs,
        const int32_t* ids, unsigned half_grid, unsigned gpt = 0) {
    // Token-major wide form: group g reads token g / gpt's column. Pure pointer
    // arithmetic, safe before the wait. gpt == 0 keeps the flat group stride.
    const Q8Block* STRATA_PDL_RESTRICT x = xbase + (gpt ? blockIdx.y / gpt : blockIdx.y) * xs;
    float* STRATA_PDL_RESTRICT y = yg;
    const Q4Block* w = gate;
    unsigned bx = blockIdx.x;
    if (bx >= half_grid) { bx -= half_grid; w = up; y = yu; gs = us; }
    y += std::size_t(blockIdx.y) * no;
    const int tid = kWarp * int(threadIdx.y) + int(threadIdx.x);
    const int row = Rows * int(bx), nb = ni / 256, iqs = 2 * (tid % 16);
    float acc[Rows] = {};
    if constexpr (Pdl) { pdl_trigger(); pdl_wait(); }
    if (ids != nullptr && ids[blockIdx.y] < 0) return;
    w += (ids ? std::size_t(ids[blockIdx.y]) : std::size_t(blockIdx.y)) * gs;
    for (int kb = tid / 16; kb < nb; kb += 8) {
        const Activation a = load_activation(x + kb * 8, iqs);
#pragma unroll
        for (int r = 0; r < Rows; ++r)
            if (row + r < no) acc[r] += dot(w + std::size_t(row + r) * nb + kb, a, iqs);
    }
    __shared__ float partial[kWarps - 1][Rows][kWarp];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) partial[threadIdx.y - 1][r][threadIdx.x] = acc[r];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int j = 0; j < kWarps - 1; ++j) acc[r] += partial[j][r][threadIdx.x];
#pragma unroll
        for (int d = kWarp / 2; d > 0; d >>= 1)
            acc[r] += __shfl_xor_sync(0xffffffff, acc[r], d, kWarp);
        if (threadIdx.x == r && row + r < no) y[row + r] = acc[r];
    }
}

template<int Rows>
void launch(const void* gate, std::size_t gs, const void* up, std::size_t us,
            const void* x, std::size_t xs, float* yg, float* yu,
            int ni, int no, int groups, cudaStream_t stream, const int32_t* ids, int gpt) {
    const unsigned half = unsigned((std::size_t(no) + Rows - 1) / Rows);
    launch_pdl_pair(q4_pair_reuse_kernel<Rows, true>, q4_pair_reuse_kernel<Rows, false>,
        dim3(2 * half, unsigned(groups)), dim3(kWarp, kWarps), 0, stream,
        static_cast<const Q4Block*>(gate), static_cast<const Q4Block*>(up),
        static_cast<const Q8Block*>(x), yg, yu, ni, no,
        gs / sizeof(Q4Block), us / sizeof(Q4Block), xs / sizeof(Q8Block), ids, half, unsigned(gpt));
}
} // namespace

bool q4_pair_reuse_enabled(int ni, int no, int groups, int rows) {
    // One row has no activation reuse. Larger K keeps the original partition.
    if (ni >= 2048 || rows <= 1) return false;
    static const int mode = [] {
        const char* v = std::getenv("STRATA_Q4_PAIR_REUSE");
        return v == nullptr ? -1 : (std::atoi(v) == 1 ? 1 : 0);
    }();
    if (mode >= 0) return mode == 1;
    // Automatically select only the H20 expert shapes benchmarked for this kernel.
    if (ni != 1536 || no != 1536 || groups < 16 || groups > 64 || groups % 16 != 0 || (rows != 2 && rows != 4)) return false;
    static thread_local int devices[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (devices[dev] == 0) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        devices[dev] = p.major == 9 && p.minor == 0 && std::strcmp(p.name, "NVIDIA H20") == 0 ? 1 : -1;
    }
    return devices[dev] == 1;
}

void q4_pair_reuse_launch(const void* gate, std::size_t gs, const void* up, std::size_t us,
        const void* x, std::size_t xs, float* yg, float* yu, int ni, int no,
        int groups, int rows, void* stream, const int32_t* ids, int groups_per_token) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (rows) {
    case 1: launch<1>(gate, gs, up, us, x, xs, yg, yu, ni, no, groups, s, ids, groups_per_token); break;
    case 2: launch<2>(gate, gs, up, us, x, xs, yg, yu, ni, no, groups, s, ids, groups_per_token); break;
    case 4: launch<4>(gate, gs, up, us, x, xs, yg, yu, ni, no, groups, s, ids, groups_per_token); break;
    case 8: launch<8>(gate, gs, up, us, x, xs, yg, yu, ni, no, groups, s, ids, groups_per_token); break;
    default: throw std::invalid_argument("q4_pair_reuse: rows must be 1, 2, 4 or 8");
    }
}
} // namespace strata::kernels
