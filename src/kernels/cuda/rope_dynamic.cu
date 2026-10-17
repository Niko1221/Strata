// src/kernels/cuda/rope_dynamic.cu - see include/strata/kernels/rope_dynamic.hpp.
#include "strata/kernels/rope_dynamic.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mrope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::kernels {

RopeRates rope_rates(int n_rot, const RopeScaling& s) {
    RopeRates r;
    const int half = n_rot / 2 < 32 ? n_rot / 2 : 32;
    if (s.type == RopeScalingType::None) {
        for (int i = 0; i < half; ++i) r.w[i] = std::pow(s.freq_base, -2.0 * (double) i / (double) n_rot);
        r.mscale = 1.0;
        return r;
    }
    const double fs = s.freq_scale();
    double cd[2];
    s.corr_dims(n_rot, cd);
    const bool correct = s.ext_factor != 0;
    for (int i = 0; i < half; ++i) {
        const double inv = std::pow(s.freq_base, -2.0 * (double) i / (double) n_rot);
        double k = fs;
        if (correct) {
            const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * s.ext_factor;
            k = fs * (1.0 - ramp) + ramp;
        }
        r.w[i] = inv * k;
    }
    r.mscale = s.mscale();
    return r;
}

void build_rope_table_mt(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab, int threads) {
    const int half = n_rot / 2;
    const bool none = sc.type == RopeScalingType::None;
    // the per-pair terms of build_rope_table, hoisted: the same double expressions, so the same values
    std::vector<double> inv((size_t) half), ramp((size_t) half, 0.0);
    double fs = 1.0, ms = 1.0, cd[2] = {0, 0};
    bool correct = false;
    if (!none) {
        fs = sc.freq_scale();
        ms = sc.mscale();
        sc.corr_dims(n_rot, cd);
        correct = sc.ext_factor != 0;
    }
    for (int i = 0; i < half; ++i) {
        inv[(size_t) i] = std::pow(none ? sc.freq_base : sc.freq_base, -2.0 * (double) i / (double) n_rot);
        if (correct) ramp[(size_t) i] = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
    }
    auto rows = [&](int p0, int p1) {
        for (int p = p0; p < p1; ++p) {
            for (int i = 0; i < half; ++i) {
                const size_t at = (size_t) p * half + i;
                if (none) {
                    const double ang = (double) p * inv[(size_t) i];
                    cos_tab[at] = (float) std::cos(ang);
                    sin_tab[at] = (float) std::sin(ang);
                    continue;
                }
                const double extrap = (double) p * inv[(size_t) i];
                const double interp = fs * extrap;
                double ang = interp;
                if (correct) ang = interp * (1.0 - ramp[(size_t) i]) + extrap * ramp[(size_t) i];
                cos_tab[at] = (float) (std::cos(ang) * ms);
                sin_tab[at] = (float) (std::sin(ang) * ms);
            }
        }
    };
    const int n = std::max(1, std::min(threads, 64));
    if (n == 1 || max_pos < 4096) { rows(0, max_pos); return; }
    std::vector<std::thread> pool;
    const int step = (max_pos + n - 1) / n;
    for (int t = 0; t < n; ++t) {
        const int a = t * step, b = std::min(max_pos, a + step);
        if (a < b) pool.emplace_back(rows, a, b);
    }
    for (auto& th : pool) th.join();
}

void rope_rescale_rows_host(float* rows, int64_t n_rows, int idx_dim, double ratio) {
    for (int64_t r = 0; r < n_rows; ++r)
        for (int d = 0; d < 64 && d < idx_dim; ++d) rows[r * idx_dim + d] = (float) ((double) rows[r * idx_dim + d] * ratio);
}

namespace {

struct Delta {
    double dw[32];
    float ratio;
};

Delta delta_of(const RopeRates& from, const RopeRates& to) {
    Delta d{};
    for (int i = 0; i < 32; ++i) d.dw[i] = to.w[i] - from.w[i];
    d.ratio = (float) (to.mscale / from.mscale);
    return d;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "rope_dynamic: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// kv_q4.cu's fwht256 on a warp's registers (element i * 32 + lane in reg[i]), scale 1/16: its own inverse
__device__ __forceinline__ void fwht_regs(float (&reg)[8], int lane) {
#pragma unroll
    for (int i = 0; i < 8; ++i) reg[i] *= 0.0625f;
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, 32);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = 32; h < 256; h *= 2) {
        const int step = h / 32;
#pragma unroll
        for (int j = 0; j < 8; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// one warp per (cell, kv head) row; element i * 32 + lane in reg[i], so pair `lane` is (reg[0], reg[1])
__global__ void rerotate_k_kernel(RopeKPool k, long long rows, Delta d, const int32_t* __restrict__ mtab) {
    const long long r = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    const int lane = threadIdx.x;
    const long long cell = r / k.n_kv_heads;
    const int h = (int) (r % k.n_kv_heads);
    const long long page = k.page_table != nullptr ? (long long) k.page_table[cell / k.page_size] : cell / k.page_size;
    if (page < 0) return;
    const long long row = (page * k.n_kv_heads + h) * k.page_size + cell % k.page_size;
    const int groups = 256 / 64;
    float reg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = i * 32 + lane;
        if (k.k_q != nullptr)
            reg[i] = (float) k.k_q[row * 256 + e] * f32_from_f16(k.k_scale[row * groups + e / 64]);
        else
            reg[i] = f32_from_f16(k.k_pool[row * 256 + e]);
    }
    if (k.hadamard) fwht_regs(reg, lane);
    {
        double s, c;
        sincos((double) mrope_pos(mtab, (int) cell, lane) * d.dw[lane], &s, &c);
        const float a = reg[0], b = reg[1], cf = (float) c, sf = (float) s;
        reg[0] = d.ratio * (a * cf - b * sf);
        reg[1] = d.ratio * (a * sf + b * cf);
    }
    if (k.hadamard) fwht_regs(reg, lane);
    const int g_end = k.hadamard ? groups : 1;   // without H only the rope group changed
    for (int g = 0; g < g_end; ++g) {
        const int i0 = 2 * g, i1 = 2 * g + 1;
        if (k.k_q == nullptr) {
            k.k_pool[row * 256 + i0 * 32 + lane] = f16_from_f32(reg[i0]);
            k.k_pool[row * 256 + i1 * 32 + lane] = f16_from_f32(reg[i1]);
            continue;
        }
        // kv_q8.cu's group code: scale = fp16(max|x| / 127), code = clamp(rint(x / stored scale))
        float a = fmaxf(fabsf(reg[i0]), fabsf(reg[i1]));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        const uint16_t sbits = f16_from_f32(a / 127.0f);
        const float sf = f32_from_f16(sbits);
        const int ii[2] = {i0, i1};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            int q = 0;
            if (sf > 0.0f) {
                q = __float2int_rn(reg[ii[j]] / sf);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
            }
            k.k_q[row * 256 + ii[j] * 32 + lane] = (int8_t) q;
        }
        if (lane == 0) k.k_scale[row * groups + g] = sbits;
    }
}

// one thread per (row, pair): pooled rows [0, n_blocks) at b * idx_block, the spare row n_blocks and `dead` at 0
__global__ void rerotate_pooled_kernel(float* pooled, float* dead, long long n_blocks, int idx_block, int idx_dim,
                                       Delta d, const int32_t* __restrict__ mtab) {
    const long long r = blockIdx.x;
    const int pair = threadIdx.x;
    float* row = r <= n_blocks ? pooled + r * idx_dim : dead;
    const int p = r < n_blocks ? mrope_pos(mtab, (int) (r * idx_block), pair) : 0;
    double s, c;
    sincos((double) p * d.dw[pair], &s, &c);
    const float a = row[pair], b = row[pair + 32], cf = (float) c, sf = (float) s;
    row[pair] = d.ratio * (a * cf - b * sf);
    row[pair + 32] = d.ratio * (a * sf + b * cf);
}

}  // namespace

void rope_rerotate_k(const RopeKPool& k, int64_t cells, const RopeRates& from, const RopeRates& to, void* stream) {
    if (cells <= 0) return;
    if (k.head_dim != 256 || k.n_kv_heads <= 0 || k.page_size <= 0 || (k.k_q == nullptr) == (k.k_pool == nullptr) ||
        (k.k_q != nullptr && k.k_scale == nullptr)) {
        std::fprintf(stderr, "rope_rerotate_k: unsupported K pool (head_dim %d)\n", k.head_dim);
        std::exit(1);
    }
    const Delta d = delta_of(from, to);
    const long long rows = (long long) cells * k.n_kv_heads;
    const dim3 block(32, 8);
    const unsigned grid = (unsigned) ((rows + 7) / 8);
    rerotate_k_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(k, rows, d, mrope_table());
    check("rerotate_k");
}

void rope_rerotate_pooled(float* pooled, float* dead, int64_t n_blocks, int idx_block, int idx_dim,
                          const RopeRates& from, const RopeRates& to, void* stream) {
    if (pooled == nullptr || dead == nullptr || n_blocks < 0 || idx_dim < 64) return;
    const Delta d = delta_of(from, to);
    rerotate_pooled_kernel<<<(unsigned) (n_blocks + 2), 32, 0, (cudaStream_t) stream>>>(pooled, dead, n_blocks, idx_block,
                                                                                        idx_dim, d, mrope_table());
    check("rerotate_pooled");
}

}  // namespace strata::kernels
