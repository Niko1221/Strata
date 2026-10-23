// ---------------------------------------------------------------------------
// Stage 0 acceptance harness for src/dsv4/dsv4_moe_cpu.h
//
//   * builds random NVFP4 expert blobs + activations,
//   * runs the GPU reference kernel (verbatim copy of dsv4_moe.cu:291-328) AND, for the real
//     shapes, the M2 reuse kernel (verbatim copy of dsv4_moe.cu:437-514 with ENT=16, nent=1),
//   * runs the CPU AVX-512 kernel and the CPU scalar reference, and bit-compares the results
//     (float bit patterns, so -0.0 vs +0.0 and NaN payloads count as differences),
//   * benchmarks single-thread and all-core MAC/s on a full 3-role expert.
//
// Build (inside the build container):
//   nvcc -O3 -std=c++17 -arch=sm_89 -ffp-contract=off tools/dsv4/cpu_gemv_s0.cu -o /tmp/cpu_gemv_s0
// Run:
//   /tmp/cpu_gemv_s0 test     -> 200 random cases, bit compare
//   /tmp/cpu_gemv_s0 bench    -> MAC/s single thread + all cores
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#include "../../src/dsv4/dsv4_moe_cpu.h"

// ===========================================================================
// GPU reference: verbatim copies of the kernels under test (dsv4_moe.cu).
// ===========================================================================
__device__ __forceinline__ float dsv4_ue4m3(uint8_t x) {   // dsv4_moe.cu:65-75
    if (x == 0 || x == 0x7f) return 0.0f;
    const int e = (x >> 3) & 0xf, m = x & 7;
    if (e == 0) return ldexpf((float) m, -9);
    return ldexpf(1.0f + (float) m / 8.0f, e - 7);
}
__constant__ float kFP4Lut[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// dsv4_moe.cu:82-87
__device__ __forceinline__ float dsv4_fp4_dec(uint8_t nib) {
    const uint32_t s = ((uint32_t) (nib & 8)) << 28;
    const uint32_t e = (nib >> 1) & 3, mm = nib & 1;
    const uint32_t mag = e ? (((e + 126) << 23) | (mm << 22)) : (mm ? 0x3F000000u : 0u);
    return __int_as_float(mag | s);
}

// dsv4_moe.cu:291-328  (R=1 default path)
__global__ void dsv4_nvfp4_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ xbase,
                                       int64_t xstride, const int32_t* __restrict__ xidx, int cols, int rows,
                                       float s2, float* __restrict__ y) {
    extern __shared__ float sx[];
    const int tpr = cols >> 6;
    const int rpb = 256 / tpr;
    const int rloc = threadIdx.x / tpr;
    const int lane = threadIdx.x % tpr;
    const int row = blockIdx.x * rpb + rloc;
    const int64_t m = blockIdx.y;
    const float* xr = xbase + (xidx ? (int64_t) xidx[m] : m) * xstride;
    for (int i = threadIdx.x; i < cols; i += 256) sx[i] = xr[i];
    __syncthreads();
    if (row >= rows) return;
    const uint8_t* blk = w + ((int64_t) row * tpr + lane) * 36;
    const float d[4] = {dsv4_ue4m3(blk[0]), dsv4_ue4m3(blk[1]), dsv4_ue4m3(blk[2]), dsv4_ue4m3(blk[3])};
    const uint8_t* qs = blk + 4;
    const float* x0 = sx + lane * 64;
    float acc = 0.f;
    for (int s = 0; s < 4; ++s) {
        const float dv = d[s];
        for (int j = 0; j < 8; ++j) {
            const uint8_t byte = qs[s * 8 + j];
            acc += kFP4Lut[byte & 0x0f] * dv * x0[s * 16 + j];
            acc += kFP4Lut[byte >> 4] * dv * x0[s * 16 + 8 + j];
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[8][2];
    if ((lane & 31) == 0) part[rloc][lane >> 5] = acc;
    __syncthreads();
    if (lane == 0) {
        float total = part[rloc][0];
        if (tpr == 64) total += part[rloc][1];
        y[m * rows + row] = total * s2;
    }
}

// dsv4_moe.cu:437-514  (M2 reuse kernel, ENT entries per staged weight block)
template <int ENT>
__global__ void dsv4_nvfp4_gemv_grouped_reuse_kernel(const uint8_t* const* __restrict__ blobs, int64_t role_off,
                                                     int64_t tail_off, int s2i, const float* __restrict__ xbase,
                                                     int64_t xstride, const int32_t* __restrict__ xidx,
                                                     const int32_t* __restrict__ ebase, int cols, int rows,
                                                     float* __restrict__ y) {
    extern __shared__ uint8_t sw[];
    const int z = blockIdx.z;
    const int cnt = (int) (ebase[z + 1] - ebase[z]);
    const int y0 = (int) blockIdx.y * ENT;
    if (y0 >= cnt) return;
    const int nent = (ENT < cnt - y0) ? ENT : (cnt - y0);
    const int64_t m0 = (int64_t) ebase[z] + y0;
    const int tpr = cols >> 6;
    const int rpb = 256 / tpr;
    const int rloc = threadIdx.x / tpr;
    const int lane = threadIdx.x % tpr;
    const int row = blockIdx.x * rpb + rloc;
    const int c31 = lane & 31;
    {
        const uint4* wv = reinterpret_cast<const uint4*>(blobs[z] + role_off +
                                                         (int64_t) (blockIdx.x * rpb) * (tpr * 36));
        uint4* dstv = reinterpret_cast<uint4*>(sw);
        const int v = threadIdx.x;
        const uint4 a = wv[v];
        uint4 b, c;
        if (v + 256 < 576) b = wv[v + 256];
        if (v + 512 < 576) c = wv[v + 512];
        dstv[v] = a;
        if (v + 256 < 576) dstv[v + 256] = b;
        if (v + 512 < 576) dstv[v + 512] = c;
    }
    __syncthreads();
    const uint8_t* blk = sw + ((int64_t) rloc * tpr + lane) * 36;
    const float d[4] = {dsv4_ue4m3(blk[0]), dsv4_ue4m3(blk[1]), dsv4_ue4m3(blk[2]), dsv4_ue4m3(blk[3])};
    const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 4);
    uint32_t qv[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) qv[v] = qw[v];
    __shared__ float part[8][2];
    const bool live = row < rows;
    for (int j = 0; j < nent; ++j) {
        const int64_t m = m0 + j;
        float acc = 0.f;
        if (live) {
            const float4* xv = reinterpret_cast<const float4*>(
                xbase + (xidx ? (int64_t) xidx[m] : m) * xstride + (int64_t) (lane >> 5) * 2048);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float dv = d[s];
                float4 xs[4];
#pragma unroll
                for (int v = 0; v < 4; ++v) xs[v] = xv[(s * 4 + v) * 32 + c31];
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const uint32_t byte = (qv[s * 2 + (jj >> 2)] >> ((jj & 3) * 8)) & 0xffu;
                    const float4 xa = xs[jj >> 2];
                    const float xj = (jj & 2) ? ((jj & 1) ? xa.w : xa.z) : ((jj & 1) ? xa.y : xa.x);
                    const float4 xb = xs[2 + (jj >> 2)];
                    const float xj8 = (jj & 2) ? ((jj & 1) ? xb.w : xb.z) : ((jj & 1) ? xb.y : xb.x);
                    acc += dsv4_fp4_dec(byte & 0x0f) * dv * xj;
                    acc += dsv4_fp4_dec(byte >> 4) * dv * xj8;
                }
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if ((lane & 31) == 0) part[rloc][lane >> 5] = acc;
        }
        __syncthreads();
        if (live && lane == 0) {
            float total = part[rloc][0];
            if (tpr == 64) total += part[rloc][1];
            const float s2 = tail_off ? ((const float*) (blobs[z] + tail_off))[s2i] : 1.0f;
            y[m * rows + row] = total * s2;
        }
        __syncthreads();
    }
}

// ===========================================================================
// helpers
// ===========================================================================
#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);      \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static inline uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t) (g_rng >> 11);
}
static inline float rndf(float lo, float hi) {
    return lo + (hi - lo) * ((float) (rnd() & 0xffffff) / (float) 0x1000000);
}

// Adversarial scale byte: mixes normal UE4M3 exponents with the two decoded special cases
// (e==0 subnormals, 0x7f NaN) so the vector decode is exercised where a shortcut would break.
static inline uint8_t rnd_scale(bool adversarial) {
    const uint32_t k = rnd() % 100;
    if (!adversarial) return (uint8_t) ((rnd() % 3 == 0) ? (rnd() & 0x7f) : (((1 + rnd() % 14) << 3) | (rnd() & 7)));
    if (k < 55) return (uint8_t) (rnd() & 0x7f);
    if (k < 75) return (uint8_t) (((1 + rnd() % 14) << 3) | (rnd() & 7));
    if (k < 90) {
        static const uint8_t z[5] = {0x00, 0x01, 0x07, 0x08, 0x0f};
        return z[rnd() % 5];
    }
    static const uint8_t n[3] = {0x7f, 0x78, 0x77};
    return n[rnd() % 3];
}

// xpose2048 layout consumed by the grouped/reuse kernels (derived from dsv4_moe.cu:365-367,483-497):
//   buf[chunk*2048 + 128*(s*4+u) + 4*(lane&31) + comp] == x[lane*64 + s*16 + 4u + comp]
static void make_xpose(const float* x, int cols, std::vector<float>& out) {
    out.assign((size_t) cols, 0.0f);
    const int tpr = cols >> 6;
    for (int L = 0; L < tpr; ++L) {
        const int chunk = L >> 5, c31 = L & 31;
        for (int s = 0; s < 4; ++s)
            for (int u = 0; u < 4; ++u)
                for (int comp = 0; comp < 4; ++comp)
                    out[(size_t) chunk * 2048 + 128 * (s * 4 + u) + 4 * c31 + comp] =
                        x[(size_t) L * 64 + s * 16 + 4 * u + comp];
    }
}

static bool bit_eq(const std::vector<float>& a, const std::vector<float>& b, size_t* first_bad, size_t* ndiff) {
    size_t nb = 0, fb = (size_t) -1;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        if (x != y) {
            if (nb == 0) fb = i;
            ++nb;
        }
    }
    *first_bad = fb;
    *ndiff = nb;
    return nb == 0;
}

struct Shape {
    int rows, cols;
    const char* name;
};

// ===========================================================================
// test
// ===========================================================================
static int run_test() {
    const int NCASES = 200;
    // real model shapes + one tpr=16 shape to prove the group loop is generic
    // The GPU kernel's warp tree is only well defined for tpr >= 32 (one whole row per warp),
    // i.e. cols 4096 (tpr=64) and cols 2048 (tpr=32) - exactly the model's two shapes.
    const Shape shapes[3] = {{2048, 4096, "gate/up 2048x4096 (tpr=64)"},
                             {4096, 2048, "down    4096x2048 (tpr=32)"},
                             {256, 4096, "gate-sm  256x4096 (tpr=64)"}};
    int fail_fma = 0, fail_mul = 0, fail_ref = 0, fail_reuse_vs_r1 = 0, cases_fma = 0, cases_mul = 0;
    int checked_reuse = 0;
    int checked_tile = 0, fail_tile = 0, fail_tile_gpu = 0;
    double worst_rel = 0.0;

    for (int cs = 0; cs < NCASES; ++cs) {
        // ~48% gate/up, ~48% down, ~4% small-row gate
        const Shape& sh = (cs % 25 == 24) ? shapes[2] : ((cs & 1) ? shapes[1] : shapes[0]);
        const int tpr = sh.cols >> 6;
        const int rpb = 256 / tpr;
        const bool adversarial = (cs % 3 == 0);
        const size_t wbytes = (size_t) sh.rows * tpr * 36;
        const int64_t tail_off = (int64_t) wbytes + 16;  // keep the tail scale inside the allocation

        std::vector<uint8_t> blob((size_t) tail_off + 64);
        for (size_t i = 0; i < wbytes; i += 36) {
            blob[i + 0] = rnd_scale(adversarial);
            blob[i + 1] = rnd_scale(adversarial);
            blob[i + 2] = rnd_scale(adversarial);
            blob[i + 3] = rnd_scale(adversarial);
            for (int b = 0; b < 32; ++b) blob[i + 4 + b] = (uint8_t) rnd();
        }
        const float s2 = ((cs & 1) ? rndf(0.5f, 2.0f) : 1.0f);
        memcpy(blob.data() + tail_off, &s2, 4);

        std::vector<float> x((size_t) sh.cols);
        for (int i = 0; i < sh.cols; ++i) {
            if (cs % 7 == 3 && (i % 97) == 0)
                x[i] = ldexpf(rndf(-1.f, 1.f), (int) (rnd() % 40) - 20);  // wide dynamic range
            else if (cs % 11 == 5 && (i % 13) == 0)
                x[i] = 0.0f;
            else
                x[i] = rndf(-2.0f, 2.0f);
        }

        // ---- GPU reference (R=1) ----
        uint8_t* d_w = nullptr;
        float *d_x = nullptr, *d_y = nullptr;
        CK(cudaMalloc(&d_w, blob.size()));
        CK(cudaMalloc(&d_x, (size_t) sh.cols * 4));
        CK(cudaMalloc(&d_y, (size_t) sh.rows * 4));
        CK(cudaMemcpy(d_w, blob.data(), blob.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_x, x.data(), (size_t) sh.cols * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_y, 0, (size_t) sh.rows * 4));
        std::vector<float> y_gpu((size_t) sh.rows, 0.f);
        dsv4_nvfp4_gemv_kernel<<<dim3((unsigned) (sh.rows / rpb), 1), 256, (size_t) sh.cols * 4>>>(
            d_w, d_x, sh.cols, nullptr, sh.cols, sh.rows, s2, d_y);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y_gpu.data(), d_y, (size_t) sh.rows * 4, cudaMemcpyDeviceToHost));

        // ---- GPU reuse kernel (only where the xpose2048 layout is defined: cols % 2048 == 0) ----
        std::vector<float> y_reuse((size_t) sh.rows, 0.f);
        if ((sh.cols % 2048) == 0) {
            std::vector<float> xp;
            make_xpose(x.data(), sh.cols, xp);
            float* d_xp = nullptr;
            CK(cudaMalloc(&d_xp, (size_t) sh.cols * 4));
            CK(cudaMemcpy(d_xp, xp.data(), (size_t) sh.cols * 4, cudaMemcpyHostToDevice));
            const uint8_t* h_blobs[1] = {d_w};
            uint8_t** d_blobs = nullptr;
            CK(cudaMalloc(&d_blobs, sizeof(void*)));
            CK(cudaMemcpy(d_blobs, h_blobs, sizeof(void*), cudaMemcpyHostToDevice));
            int32_t h_eb[2] = {0, 1};
            int32_t* d_eb = nullptr;
            CK(cudaMalloc(&d_eb, 2 * sizeof(int32_t)));
            CK(cudaMemcpy(d_eb, h_eb, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
            CK(cudaMemset(d_y, 0, (size_t) sh.rows * 4));
            dsv4_nvfp4_gemv_grouped_reuse_kernel<16><<<dim3((unsigned) (sh.rows / rpb), 1, 1), 256, 9216>>>(
                (const uint8_t* const*) d_blobs, 0, tail_off, 0, d_xp, sh.cols, nullptr, d_eb, sh.cols, sh.rows,
                d_y);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(y_reuse.data(), d_y, (size_t) sh.rows * 4, cudaMemcpyDeviceToHost));
            size_t fb = 0, nd = 0;
            if (!bit_eq(y_gpu, y_reuse, &fb, &nd)) {
                ++fail_reuse_vs_r1;
                printf("  [case %3d %s] GPU reuse vs R=1: %zu diffs, first at row %zu (%.9g vs %.9g)\n", cs, sh.name,
                       nd, fb, y_gpu[fb], y_reuse[fb]);
            }
            ++checked_reuse;
            CK(cudaFree(d_xp));
            CK(cudaFree(d_blobs));
            CK(cudaFree(d_eb));
        }
        CK(cudaFree(d_w));
        CK(cudaFree(d_x));
        CK(cudaFree(d_y));

        // ---- CPU ----
        std::vector<float> y_cpu((size_t) sh.rows, 0.f), y_ref((size_t) sh.rows, 0.f);
        float* xT = (float*) aligned_alloc(64, (size_t) ((sh.cols + 15) & ~15) * 4);
        strata::dsv4::CpuGemvJob job;
        job.blob = blob.data();
        job.role_off = 0;
        job.tail_off = tail_off;
        job.s2i = 0;
        job.x = x.data();
        job.xstride = sh.cols;
        job.xidx = nullptr;
        job.nent = 1;
        job.cols = sh.cols;
        job.rows = sh.rows;
        job.y = y_cpu.data();
        job.ystride = sh.rows;

        for (int variant = 0; variant < 2; ++variant) {
            job.variant = variant;
            std::fill(y_cpu.begin(), y_cpu.end(), 0.f);
            strata::dsv4::dsv4_cpu_gemv_rows(job, 0, sh.rows, xT);
            size_t fb = 0, nd = 0;
            const bool ok = bit_eq(y_gpu, y_cpu, &fb, &nd);
            if (variant == strata::dsv4::kCpuVariantFma) {
                ++cases_fma;
                if (!ok) {
                    ++fail_fma;
                    printf("  [case %3d %s] CPU fma vs GPU: %zu/%d diffs, first row %zu: gpu %.9g cpu %.9g\n", cs,
                           sh.name, nd, sh.rows, fb, y_gpu[fb], y_cpu[fb]);
                }
            } else {
                ++cases_mul;
                if (!ok) {
                    ++fail_mul;
                    if (fail_mul <= 3)
                        printf("  [case %3d %s] CPU mul/add vs GPU: %zu/%d diffs, first row %zu: gpu %.9g cpu %.9g\n",
                               cs, sh.name, nd, sh.rows, fb, y_gpu[fb], y_cpu[fb]);
                }
            }
            // relative deviation of the accepted arm (0 when bit-exact)
            if (variant == strata::dsv4::kCpuVariantFma) {
                for (int i = 0; i < sh.rows; ++i) {
                    const double d = fabs((double) y_cpu[i] - (double) y_gpu[i]);
                    const double r = fabs((double) y_gpu[i]) > 1e-30 ? d / fabs((double) y_gpu[i]) : d;
                    if (r > worst_rel) worst_rel = r;
                }
            }
        }

        // ---- CPU scalar reference vs AVX-512 (same order by construction) ----
        job.variant = strata::dsv4::kCpuVariantFma;
        job.y = y_ref.data();
        std::fill(y_ref.begin(), y_ref.end(), 0.f);
        strata::dsv4::dsv4_cpu_gemv_rows_ref(job, 0, sh.rows, job.variant);
        {
            size_t fb = 0, nd = 0;
            std::vector<float> y_avx((size_t) sh.rows, 0.f);
            job.y = y_avx.data();
            std::fill(y_avx.begin(), y_avx.end(), 0.f);
            strata::dsv4::dsv4_cpu_gemv_rows(job, 0, sh.rows, xT);
            if (!bit_eq(y_ref, y_avx, &fb, &nd)) {
                ++fail_ref;
                if (fail_ref <= 3)
                    printf("  [case %3d %s] scalar-ref vs AVX512: %zu diffs, first row %zu: ref %.9g avx %.9g\n", cs,
                           sh.name, nd, fb, y_ref[fb], y_avx[fb]);
            }
        }
        // ---- row-tiled (R=4) vs one-row-at-a-time (R=1) vs GPU: all three must agree bit-exactly ----
        {
            std::vector<float> y_t0((size_t) sh.rows, 0.f), y_t1((size_t) sh.rows, 0.f);
            job.variant = strata::dsv4::kCpuVariantFma;
            job.tile = 0;
            job.y = y_t0.data();
            std::fill(y_t0.begin(), y_t0.end(), 0.f);
            strata::dsv4::dsv4_cpu_gemv_rows(job, 0, sh.rows, xT);
            job.tile = 1;
            job.y = y_t1.data();
            std::fill(y_t1.begin(), y_t1.end(), 0.f);
            strata::dsv4::dsv4_cpu_gemv_rows(job, 0, sh.rows, xT);
            job.tile = 0;
            size_t fb = 0, nd = 0;
            ++checked_tile;
            if (!bit_eq(y_t1, y_t0, &fb, &nd)) {
                ++fail_tile;
                if (fail_tile <= 3)
                    printf("  [case %3d %s] tiled(R=4) vs tiled(R=1): %zu diffs, first row %zu: R1 %.9g R4 %.9g\n", cs,
                           sh.name, nd, fb, y_t1[fb], y_t0[fb]);
            }
            if (!bit_eq(y_gpu, y_t0, &fb, &nd)) {
                ++fail_tile_gpu;
                if (fail_tile_gpu <= 3)
                    printf("  [case %3d %s] tiled(R=4) vs GPU: %zu diffs, first row %zu: gpu %.9g cpu %.9g\n", cs,
                           sh.name, nd, fb, y_gpu[fb], y_t0[fb]);
            }
        }
        free(xT);
    }

    printf("\n[test] %d random cases (adversarial scales every 3rd, wide-dynamic-range x every 7th)\n", NCASES);
    printf("[test] GPU reuse(ENT=16) vs GPU R=1 kernel : %d cases checked, %d failed\n", checked_reuse,
           fail_reuse_vs_r1);
    printf("[test] CPU fma      vs GPU               : %d cases, %d failed  -> %s\n", cases_fma, fail_fma,
           fail_fma ? "MISMATCH" : "BIT-EXACT");
    printf("[test] CPU mul+add  vs GPU               : %d cases, %d failed  -> %s\n", cases_mul, fail_mul,
           fail_mul ? "MISMATCH" : "BIT-EXACT");
    printf("[test] CPU scalar   vs CPU AVX-512       : %d failed\n", fail_ref);
    printf("[test] tiled(R=4)   vs tiled(R=1)        : %d cases, %d failed  -> %s\n", checked_tile, fail_tile,
           fail_tile ? "MISMATCH" : "BIT-EXACT");
    printf("[test] tiled(R=4)   vs GPU               : %d cases, %d failed  -> %s\n", checked_tile, fail_tile_gpu,
           fail_tile_gpu ? "MISMATCH" : "BIT-EXACT");
    printf("[test] max relative deviation (fma arm)  : %.3e\n", worst_rel);
    const bool pass = (fail_fma == 0) && fail_ref == 0 && fail_reuse_vs_r1 == 0 && fail_tile == 0 && fail_tile_gpu == 0;
    printf("[test] RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ===========================================================================
// bench: one full 3-role expert (gate 2048x4096, up 2048x4096, down 4096x2048) = 25.17 MMAC
// ===========================================================================
struct Role {
    const char* name;
    int rows, cols;
    size_t off;
};

static int run_bench(int nthreads, int tile, bool pin) {
    // One 13.5 MiB blob laid out like the arena: gate | up | down | tail scale.
    const int RG = 2048, CG = 4096;
    const int RU = 2048, CU = 4096;
    const int RD = 4096, CD = 2048;
    const Role roles[3] = {{"gate", RG, CG, 0},
                           {"up", RU, CU, (size_t) RG * (CG / 64) * 36},
                           {"down", RD, CD, (size_t) RG * (CG / 64) * 36 + (size_t) RU * (CU / 64) * 36}};
    const size_t total = roles[2].off + (size_t) RD * (CD / 64) * 36;
    const int64_t tail_off = (int64_t) total + 16;
    std::vector<uint8_t> blob((size_t) tail_off + 64);
    for (size_t i = 0; i < total; i += 36) {
        for (int b = 0; b < 4; ++b) blob[i + b] = (uint8_t) (((1 + rnd() % 14) << 3) | (rnd() & 7));
        for (int b = 0; b < 32; ++b) blob[i + 4 + b] = (uint8_t) rnd();
    }
    const float s2 = 1.0f;
    memcpy(blob.data() + tail_off, &s2, 4);

    std::vector<float> xg((size_t) CG), xd((size_t) CD);
    for (auto& v : xg) v = rndf(-2.f, 2.f);
    for (auto& v : xd) v = rndf(-2.f, 2.f);
    std::vector<float> yg((size_t) RG), yu((size_t) RU), yd((size_t) RD);
    float* xT = (float*) aligned_alloc(64, (size_t) (CG + 16) * 4 * (size_t) std::max(1, nthreads));
    std::vector<char> scratch_guard(64, 0);
    (void) scratch_guard;

    strata::dsv4::CpuPool* pool = (nthreads > 1) ? new strata::dsv4::CpuPool(nthreads, pin) : nullptr;

    auto run_all = [&](strata::dsv4::CpuGemvJob& jg, strata::dsv4::CpuGemvJob& ju, strata::dsv4::CpuGemvJob& jd) {
        strata::dsv4::dsv4_cpu_gemv(jg, nthreads, xT, pool);
        strata::dsv4::dsv4_cpu_gemv(ju, nthreads, xT, pool);
        strata::dsv4::dsv4_cpu_gemv(jd, nthreads, xT, pool);
    };

    auto mk = [&](const Role& r, const std::vector<float>& x, std::vector<float>& y) {
        strata::dsv4::CpuGemvJob j;
        j.blob = blob.data();
        j.role_off = (int64_t) r.off;
        j.tail_off = tail_off;
        j.s2i = 0;
        j.x = x.data();
        j.xstride = r.cols;
        j.xidx = nullptr;
        j.nent = 1;
        j.cols = r.cols;
        j.rows = r.rows;
        j.y = y.data();
        j.ystride = r.rows;
        j.variant = strata::dsv4::kCpuVariantFma;
        j.tile = tile;
        return j;
    };
    strata::dsv4::CpuGemvJob jg = mk(roles[0], xg, yg), ju = mk(roles[1], xg, yu), jd = mk(roles[2], xd, yd);

    const double macs = (double) RG * CG + (double) RU * CU + (double) RD * CD;  // 25.17e6
    run_all(jg, ju, jd);  // warm up / bring pages in
    const int REP = 7;
    double best = 1e30;
    for (int i = 0; i < REP; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        run_all(jg, ju, jd);
        auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        if (s < best) best = s;
    }
    printf("[bench] threads=%2d tile=%d  full expert (gate+up+down, %.3f MMAC): %8.3f ms  => %7.3f GMAC/s  (%.3f GMAC/s/core)\n",
           nthreads, tile, macs / 1e6, best * 1e3, macs / best / 1e9, macs / best / 1e9 / nthreads);
    // per-role breakdown (useful for Stage 2 load balancing)
    for (int r = 0; r < 3; ++r) {
        auto& j = (r == 0) ? jg : (r == 1 ? ju : jd);
        double b = 1e30;
        for (int i = 0; i < REP; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            strata::dsv4::dsv4_cpu_gemv(j, nthreads, xT, pool);
            auto t1 = std::chrono::steady_clock::now();
            b = std::min(b, std::chrono::duration<double>(t1 - t0).count());
        }
        const double mm = (double) roles[r].rows * roles[r].cols;
        printf("[bench]   %-4s %5dx%-4d %8.3f ms  %7.3f GMAC/s  %.1f MAC/cycle/core\n", roles[r].name, roles[r].rows,
               roles[r].cols, b * 1e3, mm / b / 1e9, mm / b / 2.6e9 / nthreads);
    }
    delete pool;
    free(xT);
    return 0;
}


// ===========================================================================
// bw: streaming read bandwidth ceiling (the CPU arm's hard limit: 0.5625 B per MAC)
// ===========================================================================

// AVX-512 streaming sum (target attribute keeps the whole TU free of -mavx512f).
STRATA_DSV4_TGT512 static inline size_t dsv4_bw_impl(const uint8_t* p, uint64_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint64_t i = 0;
    for (; i + 64 <= n; i += 64) acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*) (p + i)));
    return (size_t) _mm512_reduce_add_epi64(acc);
}
static size_t dsv4_bw_chunk(const uint8_t* p, uint64_t n) {
#if defined(STRATA_DSV4_CPU_X86)
    return dsv4_bw_impl(p, n);
#else
    (void) p; (void) n; return 0;
#endif
}

static int run_bw(int nthreads, size_t mb) {
    const size_t n = mb << 20;
    uint8_t* buf = (uint8_t*) aligned_alloc(64, n);
    memset(buf, 0xA5, n);
    std::atomic<size_t> sink(0);
    auto measure = [&](size_t bytes, int reps) {
        strata::dsv4::CpuPool* pool = (nthreads > 1) ? new strata::dsv4::CpuPool(nthreads) : nullptr;
        struct S : strata::dsv4::CpuSink {
            const uint8_t* b; size_t n; int nthr; std::atomic<size_t>* sink;
            void run(int tid) override {
                const uint64_t lo = (uint64_t) n * tid / nthr, hi = (uint64_t) n * (tid + 1) / nthr;
                sink->fetch_add(dsv4_bw_chunk(b + (lo & ~(uint64_t) 63), hi - (lo & ~(uint64_t) 63)));
            }
        } s;
        s.b = buf; s.n = bytes; s.nthr = nthreads; s.sink = &sink;
        s.run(0);
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            if (pool) pool->run(&s); else s.run(0);
        }
        auto t1 = std::chrono::steady_clock::now();
        delete pool;
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return (double) bytes * reps / sec / 1e9;
    };
    const double warm = measure(n, 3);
    const double cold = 0.0;
    printf("[bw] threads=%2d  streaming read of %zu MiB (resident): %7.2f GB/s   => ceiling %.1f GMAC/s\n",
           nthreads, mb, warm, warm / 0.5625);
    (void) cold;
    free(buf);
    return 0;
}


// ===========================================================================
// bar: cost of one empty pool barrier (Stage 1's per-entry fan-out pays this)
// ===========================================================================
static int run_bar(int nthreads, int iters) {
    struct Noop : strata::dsv4::CpuSink {
        std::atomic<int> n{0};
        void run(int) override { n.fetch_add(1, std::memory_order_relaxed); }
    } sink;
    strata::dsv4::CpuPool* pool = (nthreads > 1) ? new strata::dsv4::CpuPool(nthreads) : nullptr;
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (pool) pool->run(&sink);
        }
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    printf("[bar] threads=%2d  empty barrier: %8.2f us  (%d iters, ran by %d threads)\n", nthreads,
           best / iters * 1e6, iters, sink.n.load() / std::max(1, iters * 3));
    delete pool;
    return 0;
}

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "test";
    if (mode == "test") return run_test();
    if (mode == "bench") {
        int nt = (argc > 2) ? atoi(argv[2]) : 1;
        int tile = (argc > 3) ? atoi(argv[3]) : 0;
        bool pin = (argc > 4) ? (atoi(argv[4]) != 0) : false;
        return run_bench(nt, tile, pin);
    }
    if (mode == "bw") {
        int nt = (argc > 2) ? atoi(argv[2]) : 1;
        size_t mb = (argc > 3) ? (size_t) atoi(argv[3]) : 512;
        return run_bw(nt, mb);
    }
    if (mode == "bar") {
        int nt = (argc > 2) ? atoi(argv[2]) : 18;
        int it = (argc > 3) ? atoi(argv[3]) : 2000;
        return run_bar(nt, it);
    }
    if (mode == "info") {
        printf("avx512 supported: %d\n", (int) strata::dsv4::dsv4_cpu_gemv_supported());
        return 0;
    }
    printf("usage: %s [test|bench <threads>|bw <threads> <MiB>|info]\n", argv[0]);
    return 2;
}
