// tests/hip/upmix_gemm_bench.cpp - the hc up-projection GEMM on gfx1030: production vs a dot2 candidate.
//
// The hc up-projection is Y[T, 10240] = X[T, 320] . W[10240, 320]^T, W bf16, X the fp16 image, fp32 out - the
// gated round-trip that a gr_upmix-style fusion could absorb (S27: 96 launches per 32k prompt, ~13.1 ms each on
// the production path = 900-1160 ms of prefill, 2.6-3.3% of PP).  This bench A/Bs the two candidates at the
// exact production shape so the fusion decision rests on a measurement:
//
//   Arm A - production: strata::prefill::Gemm with set_f16_io(true) (what STRATA_HIP_PROMPT_F16=1 runs):
//           bf16_to_f16_rows (W bf16 -> fp16 scratch, per launch) -> f16_inplace (rocBLAS's tuned fp16-in
//           fp16-out kernel, or cublasGemmEx) -> widen_rows_f16 (fp16 -> fp32 in Y's own rows).
//           ~2.75 GB of Y traffic per launch (fp16 write + fp16 read + fp32 write); measured 13.1 ms.
//   Arm B - candidate: the same bf16->fp16 W step, then one SIMT dot2 GEMM (v_dot2_f32_f16, fp32 accumulator)
//           that reads fp16 and writes fp32 directly: ~1.48 GB of Y traffic.  pfg-style tiling (BK=32, LDS
//           staging, uint4 LDS loads, 8x8 tile-window swizzle), LDK=40 halves so every uint4 LDS row is
//           16B-aligned.  Gate (S27): needs ~63-80% of the ~26.5 TFLOPS dot2 peak (8.3 ms compute floor) to
//           beat Arm A; also reported kernel-only (a production integration could cache the converted W
//           across the 96 launches of one prompt - W does not change).
//
// Correctness: both arms against a CPU fp32 reference (sampled), arm B against arm A elementwise (device),
// and the partial last m-block of T=33516 (523*64 + 44) against the same reference at its boundary rows.
// Tolerances are fp16's: production rounds its output to fp16 (~5e-4 rel), dot2 accumulates in fp32.
//
// Build (the CMake hip_prefill_rocblas_gemm flags, hand-linked since this test is not in the build):
//   clang++ -x hip -O3 -DNDEBUG -std=c++20 -fPIE -Wall -Wextra \
//     -DSTRATA_HIP_ARCHS='"gfx1030"' -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_PREFILL_FUSED=1 -DSTRATA_PREFILL_MMQ=1 \
//     -DSTRATA_USE_HIP=1 -DSTRATA_VERSION='"0.1.45"' -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 \
//     -I src/prefill -I include -I include/strata/hip_compat \
//     -include include/strata/hip_compat/cuda_runtime.h \
//     tests/hip/upmix_gemm_bench.cpp -o build-hip/upmix_gemm_bench \
//     -x none build-hip/libstrata_prefill.a build-hip/libstrata_hip_runtime.a \
//     -L/opt/rocm/core-10.0/lib -lrocblas -lhipblas -lamdhip64
// Run with the server's env (STRATA_ROCBLAS_TUNING especially - Arm A must pick the same tuned rocBLAS
// kernel production picks) and with the server stopped (the bench needs ~2.8 GB; the loaded lane leaves
// ~1.1 MB free).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(e, what) \
    do { \
        if ((e) != hipSuccess) { \
            std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); \
            std::exit(2); \
        } \
    } while (0)

namespace {

constexpr int kN = 10240;      // hc width = D * HC = 4 * 2560
constexpr int kK = 320;        // low rank
constexpr int kTmax = 33516;   // the production 32k-token prompt

// ---------------------------------------------------------------------------
// Arm B: the dot2 GEMM.  256 threads; thread tile TM x TN at (ty*TM + j, tx + 16*c) with ty = tid/16,
// tx = tid%16 (so the sB LDS reads of a wavefront touch 16 distinct rows and the sA reads 2 - cheap at the
// 20-word row stride).  BK=32 per k-tile, double-buffered LDS, one barrier per k-tile; uint4 LDS loads
// (8 halves = 4 dot2 pairs per load); K=320 = 10 k-tiles, all of it.  fp32 out, written directly (no widen
// round-trip).  Partial m handled (zero-padded X, guarded stores); N and K are exact multiples of the
// tiles (static-asserted).
// ---------------------------------------------------------------------------
template <int BM, int BN, int TM, int TN>
__global__ void __launch_bounds__(256) upmix_dot2_gemm(const __half* __restrict__ X, const __half* __restrict__ W,
                                                       float* __restrict__ Y, int T, int N, int K) {
    constexpr int BK = 32;
    constexpr int LDK = BK + 8;   // 40 halves = 20 words: uint4 rows stay 16B-aligned; the 20-word stride
                                  // spreads the sB reads over 8 distinct banks per 16 lanes (2-way, cheap)
    static_assert(BM == TM * 16 && BN == TN * 16, "16 ty x 16 tx threads");
    static_assert(kN % BN == 0 && kK % BK == 0, "exact tiles");
    constexpr int XL = BM * BK / 8 / 256;   // X uint4 loads per thread per k-tile
    constexpr int WL = BN * BK / 8 / 256;   // W uint4 loads per thread per k-tile
    __shared__ __align__(16) __half sA[2][BM][LDK];
    __shared__ __align__(16) __half sB[2][BN][LDK];

    const int tid = threadIdx.x;
    const int ty = tid / 16, tx = tid % 16;

    // 8x8 tile windows: within a window both the X tiles (BM rows) and the W tiles (BN rows) stay hot in
    // L2; a plain n-fastest sweep streams all 80 W tiles (6.4 MB) past a 4 MB L2 every m-block pass.
    const int mb = (T + BM - 1) / BM, nb = N / BN;
    constexpr int SBM = 8, SBN = 8;
    const int sbx_tot = (nb + SBN - 1) / SBN;
    const int lin = blockIdx.x;
    const int sb = lin / (SBM * SBN), wi = lin % (SBM * SBN);
    const int sw_m = (sb / sbx_tot) * SBM + wi / SBN, sw_n = (sb % sbx_tot) * SBN + wi % SBN;
    if (sw_m >= mb || sw_n >= nb) return;
    const int m0 = sw_m * BM, n0 = sw_n * BN;

    auto stage = [&](int kt, int buf) {
        // X tile [BM, BK]: row = idx/4, part = idx%4 keeps consecutive lanes on consecutive 64 B segments
        // (coalesced); the zero-fill covers the partial last m-block.
        #pragma unroll
        for (int i = 0; i < XL; ++i) {
            const int idx = tid + i * 256;
            const int row = idx / 4, part = idx % 4;
            const int t = m0 + row;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < T) v = *reinterpret_cast<const uint4*>(X + (size_t) t * K + kt * BK + part * 8);
            *reinterpret_cast<uint4*>(&sA[buf][row][part * 8]) = v;
        }
        // W tile [BN, BK]
        #pragma unroll
        for (int i = 0; i < WL; ++i) {
            const int idx = tid + i * 256;
            const int row = idx / 4, part = idx % 4;
            *reinterpret_cast<uint4*>(&sB[buf][row][part * 8]) =
                *reinterpret_cast<const uint4*>(W + (size_t) (n0 + row) * K + kt * BK + part * 8);
        }
    };

    float acc[TM][TN];
    #pragma unroll
    for (int j = 0; j < TM; ++j)
        #pragma unroll
        for (int c = 0; c < TN; ++c) acc[j][c] = 0.f;

    stage(0, 0);
    __syncthreads();
    for (int kt = 0; kt < K; kt += BK) {
        const int ki = kt / BK;
        if (kt + BK < K) stage(ki + 1, (ki + 1) & 1);
        const int buf = ki & 1;
        #pragma unroll
        for (int pg = 0; pg < BK / 8; ++pg) {
            uint4 wb[TN];
            #pragma unroll
            for (int c = 0; c < TN; ++c) wb[c] = *reinterpret_cast<const uint4*>(&sB[buf][tx + 16 * c][pg * 8]);
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                const uint4 xa = *reinterpret_cast<const uint4*>(&sA[buf][ty * TM + j][pg * 8]);
                const __half2* xh = reinterpret_cast<const __half2*>(&xa);
                #pragma unroll
                for (int p = 0; p < 4; ++p)
                    #pragma unroll
                    for (int c = 0; c < TN; ++c) {
                        const __half2* wh = reinterpret_cast<const __half2*>(&wb[c]);
                        acc[j][c] = __builtin_amdgcn_fdot2(xh[p], wh[p], acc[j][c], false);
                    }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        const int row = m0 + ty * TM + j;
        if (row < T) {
            float* yrow = Y + (size_t) row * N + n0;
            #pragma unroll
            for (int c = 0; c < TN; ++c) yrow[tx + 16 * c] = acc[j][c];
        }
    }
}

// Arm B's W step: production's bf16_to_f16_rows semantics, minus the range stats (gemm.cu:96).
__global__ void w_to_f16(const uint16_t* __restrict__ s, __half* __restrict__ d, long long n) {
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: NaN stays NaN
    }
}

// max |a - b| and max |a - b| / max(|b|, 1e-6) over n elements (non-negative floats: unsigned atomicMax).
__global__ void rel_check(const float* __restrict__ a, const float* __restrict__ b, size_t n, float* out) {
    float mrel = 0.f, mabs = 0.f;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float d = fabsf(a[i] - b[i]);
        mabs = fmaxf(mabs, d);
        mrel = fmaxf(mrel, d / fmaxf(fabsf(b[i]), 1e-6f));
    }
    atomicMax((unsigned*) &out[0], __float_as_uint(mrel));
    atomicMax((unsigned*) &out[1], __float_as_uint(mabs));
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t r = u + 0x7fffu + ((u >> 16) & 1u);   // round-to-nearest-even by hand (hip_bfloat16's
    return (uint16_t) (r >> 16);                         // host ctor gave zeros on ROCm 7)
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float f16_bits_to_f32(uint16_t b) { return __half2float(__ushort_as_half(b)); }

template <typename F>
double time_ms(hipStream_t stream, int reps, F&& fn) {
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0), "event");
    CK(hipEventCreate(&e1), "event");
    fn();
    CK(hipGetLastError(), "warmup launch");
    CK(hipStreamSynchronize(stream), "warmup sync");
    std::vector<float> ms;
    for (int i = 0; i < reps; ++i) {
        CK(hipEventRecord(e0, stream), "record");
        fn();
        CK(hipEventRecord(e1, stream), "record");
        CK(hipEventSynchronize(e1), "event sync");
        float t;
        CK(hipEventElapsedTime(&t, e0, e1), "elapsed");
        ms.push_back(t);
    }
    CK(hipEventDestroy(e0), "event");
    CK(hipEventDestroy(e1), "event");
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];   // median of reps (protocol: med-of-N, soak discarded via the warmup)
}

struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { CK(hipMalloc(&p, bytes), "hipMalloc"); }
    ~Buffer() { if (p) (void) hipFree(p); }
};

}  // namespace

// Arm B per launch: the same W bf16->fp16 step production pays inside Gemm::bf16, then the dot2 GEMM.
template <int BM, int BN, int TM, int TN>
void arm_b(hipStream_t stream, const void* x16, const void* wf16, const void* wbf, int T, int grid, float* y) {
    w_to_f16<<<1024, 256, 0, stream>>>((const uint16_t*) wbf, (__half*) wf16, (long long) kN * kK);
    upmix_dot2_gemm<BM, BN, TM, TN><<<grid, 256, 0, stream>>>((const __half*) x16, (const __half*) wf16, y, T, kN, kK);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 12;
    setvbuf(stdout, nullptr, _IONBF, 0);
    assert(kN % 128 == 0 && kK % 32 == 0);

    hipStream_t stream;
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("upmix_gemm_bench on %s (%d CU, clk %d kHz), reps %d\n", prop.name, prop.multiProcessorCount,
                prop.clockRate, reps);
    std::printf("  shapes: T = 2048 / 8192 / %d, N = %d, K = %d; Y fp32 out\n", kTmax, kN, kK);
    if (const char* tun = std::getenv("STRATA_ROCBLAS_TUNING"))
        std::printf("  STRATA_ROCBLAS_TUNING = %s (Arm A must match the server's table)\n", tun);
    else
        std::printf("  STRATA_ROCBLAS_TUNING not set - Arm A uses cublasGemmEx defaults, not the server's tuning\n");

    // host data: X the fp16 image, W bf16 (the production up-projection's inputs)
    std::mt19937 rng(835);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<uint16_t> xh((size_t) kTmax * kK), wb((size_t) kN * kK);
    for (auto& v : xh) {
        const __half h = __float2half_rn(dist(rng));
        std::memcpy(&v, &h, 2);
    }
    for (auto& v : wb) v = to_bf16(dist(rng));

    Buffer dX((size_t) kTmax * kK * 2), dW((size_t) kN * kK * 2), dWf16((size_t) kN * kK * 2),
        dYA((size_t) kTmax * kN * 4), dYB((size_t) kTmax * kN * 4);
    CK(hipMemcpy(dX.p, xh.data(), xh.size() * 2, hipMemcpyHostToDevice), "copy X");
    CK(hipMemcpy(dW.p, wb.data(), wb.size() * 2, hipMemcpyHostToDevice), "copy W");

    // Arm B's W step once here (production pays the same conversion per launch; a real integration could
    // cache it - see the kernel-only rows below)
    w_to_f16<<<1024, 256, 0, stream>>>((const uint16_t*) dW.p, (__half*) dWf16.p, (long long) kN * kK);
    CK(hipGetLastError(), "w_to_f16");
    CK(hipStreamSynchronize(stream), "w_to_f16 sync");

    // Arm A: the production Gemm, f16_io on (the STRATA_HIP_PROMPT_F16=1 path).  Scratch sized to hold the
    // whole W (kN * kK bf16 elements) so one launch covers all 10240 rows - the production launch shape.
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(stream, (int64_t) kN * kK, err)) {
        std::fprintf(stderr, "Gemm::init: %s\n", err.c_str());
        return 2;
    }
    gemm.set_f16_io(true);

    const auto run_a = [&](int T) {
        // beta = 0, x_f16 = false - exactly the production call at prefill.cpp:3010 (the f16_io branch
        // ignores x_f16 and takes X as the fp16 image; bf16x2's X_lo accumulation is off on this path)
        gemm.bf16((const uint16_t*) dX.p, (const uint16_t*) dW.p, (float*) dYA.p, T, kN, kK, 0, 0.0f, 0, false);
    };
    const auto grid_for = [&](int T, int BM) {   // superblocks: ceil(mb/8) x ceil(nb/8) windows of 64 tiles
        const int mb = (T + BM - 1) / BM, nb = kN / 128;
        return ((mb + 7) / 8) * ((nb + 7) / 8) * 64;
    };

    // ---- correctness first (T = 2048, cheap; the timing loop overwrites the buffers afterwards) ----
    const int Tc = 2048;
    run_a(Tc);
    arm_b<64, 128, 4, 8>(stream, dX.p, dWf16.p, dW.p, Tc, grid_for(Tc, 64), (float*) dYB.p);
    CK(hipStreamSynchronize(stream), "correctness sync");
    {
        Buffer rel(8);
        CK(hipMemset(rel.p, 0, 8), "rel zero");
        rel_check<<<2048, 256, 0, stream>>>((const float*) dYA.p, (const float*) dYB.p, (size_t) Tc * kN, (float*) rel.p);
        CK(hipStreamSynchronize(stream), "rel sync");
        float r[2];
        CK(hipMemcpy(r, rel.p, 8, hipMemcpyDeviceToHost), "rel copy");
        std::printf("T=%d: arm B (64x128) vs arm A: max rel %.3e, max abs %.3e\n", Tc, r[0], r[1]);
    }
    // sampled CPU reference: 32 random cells + the four corners
    {
        double worst_a = 0, worst_b = 0;
        int bad_a = 0, bad_b = 0;
        for (int s = 0; s < 36; ++s) {
            const int t = s < 32 ? (int) (rng() % Tc) : (s == 32 ? 0 : s == 33 ? 1 : s == 34 ? Tc - 2 : Tc - 1);
            const int n = s < 32 ? (int) (rng() % kN) : (s & 1 ? kN - 1 : 0);
            float ref = 0.f;
            const uint16_t* xr = &xh[(size_t) t * kK];
            const uint16_t* wr = &wb[(size_t) n * kK];
            for (int k = 0; k < kK; ++k) ref += f16_bits_to_f32(xr[k]) * bf16_to_f32(wr[k]);
            float ya, yb;
            CK(hipMemcpy(&ya, (const float*) dYA.p + (size_t) t * kN + n, 4, hipMemcpyDeviceToHost), "ya");
            CK(hipMemcpy(&yb, (const float*) dYB.p + (size_t) t * kN + n, 4, hipMemcpyDeviceToHost), "yb");
            const double ea = std::fabs((double) ya - ref) / std::max(std::fabs((double) ref), 1e-6);
            const double eb = std::fabs((double) yb - ref) / std::max(std::fabs((double) ref), 1e-6);
            worst_a = std::max(worst_a, ea);
            worst_b = std::max(worst_b, eb);
            if (ea > 2e-3) ++bad_a;
            if (eb > 2e-3) ++bad_b;
        }
        std::printf("T=%d vs CPU fp32 reference (36 samples): arm A max rel %.3e (%d over tol), arm B %.3e (%d over tol)\n",
                    Tc, worst_a, bad_a, worst_b, bad_b);
    }

    // ---- timing: shapes in production order ----
    const int shapes[3] = {2048, 8192, kTmax};
    for (int si = 0; si < 3; ++si) {
        const int T = shapes[si];
        const double flops = 2.0 * (double) T * kN * kK;
        const int g1 = grid_for(T, 64), g2 = grid_for(T, 128);

        const double a = time_ms(stream, reps, [&] { run_a(T); });
        const double b1 = time_ms(stream, reps, [&] { arm_b<64, 128, 4, 8>(stream, dX.p, dWf16.p, dW.p, T, g1, (float*) dYB.p); });
        const double b1k = time_ms(
            stream, reps, [&] { upmix_dot2_gemm<64, 128, 4, 8><<<g1, 256, 0, stream>>>((const __half*) dX.p, (const __half*) dWf16.p, (float*) dYB.p, T, kN, kK); });
        const double b2 = time_ms(stream, reps, [&] { arm_b<128, 128, 8, 8>(stream, dX.p, dWf16.p, dW.p, T, g2, (float*) dYB.p); });
        const double b2k = time_ms(
            stream, reps, [&] { upmix_dot2_gemm<128, 128, 8, 8><<<g2, 256, 0, stream>>>((const __half*) dX.p, (const __half*) dWf16.p, (float*) dYB.p, T, kN, kK); });

        std::printf("\nT=%d (%.1f GFLOP, %d blocks of 64x128 / %d of 128x128):\n", T, flops / 1e9, g1, g2);
        std::printf("  A  production (Gemm f16_io)  %8.3f ms  %6.1f TFLOPS  (B1 kernel-only is %.2fx of it)\n", a,
                    flops / a / 1e9, b1k / a);
        std::printf("  B1 dot2 64x128  w/ convert  %8.3f ms  %6.1f TFLOPS  %s\n", b1, flops / b1 / 1e9,
                    b1 < a ? "FASTER" : "slower");
        std::printf("  B1 dot2 64x128  kernel only %8.3f ms  %6.1f TFLOPS\n", b1k, flops / b1k / 1e9);
        std::printf("  B2 dot2 128x128 w/ convert  %8.3f ms  %6.1f TFLOPS  %s\n", b2, flops / b2 / 1e9,
                    b2 < a ? "FASTER" : "slower");
        std::printf("  B2 dot2 128x128 kernel only %8.3f ms  %6.1f TFLOPS\n", b2k, flops / b2k / 1e9);
    }

    // ---- the partial last m-block of T=33516 (523*64 + 44 rows) against the CPU reference ----
    {
        const int T = kTmax;
        run_a(T);
        arm_b<64, 128, 4, 8>(stream, dX.p, dWf16.p, dW.p, T, grid_for(T, 64), (float*) dYB.p);
        CK(hipStreamSynchronize(stream), "tail sync");
        // copy the partial block's 44 rows + the last full block's rows from both arms and locate bad rows
        const int row0 = 33408, nrows = kTmax - row0;
        std::vector<float> ha((size_t) nrows * kN), hb((size_t) nrows * kN);
        CK(hipMemcpy(ha.data(), (const float*) dYA.p + (size_t) row0 * kN, (size_t) nrows * kN * 4, hipMemcpyDeviceToHost), "tail ya");
        CK(hipMemcpy(hb.data(), (const float*) dYB.p + (size_t) row0 * kN, (size_t) nrows * kN * 4, hipMemcpyDeviceToHost), "tail yb");
        int badrows = 0;
        for (int rr = 0; rr < nrows; ++rr) {
            double mx = 0;
            for (int n = 0; n < kN; ++n) mx = std::max(mx, (double) std::fabs(ha[(size_t) rr * kN + n] - hb[(size_t) rr * kN + n]));
            if (mx > 1e-3 && badrows < 20) {
                std::printf("  row %d (abs %d in partial? maxabs diff %.3e)\n", row0 + rr, row0 + rr >= 33472, mx);
                ++badrows;
            } else if (mx > 1e-3) ++badrows;
        }
        std::printf("tail rows with maxabs diff > 1e-3: %d of %d\n", badrows, nrows);
        double worst_a = 0, worst_b = 0;
        const int ts[4] = {33408, 33471, 33472, kTmax - 1};   // last full block + the partial block's edges
        for (int t : ts) {
            for (int n : {0, 1, kN / 2, kN - 1}) {
                float ref = 0.f;
                const uint16_t* xr = &xh[(size_t) t * kK];
                const uint16_t* wr = &wb[(size_t) n * kK];
                for (int k = 0; k < kK; ++k) ref += f16_bits_to_f32(xr[k]) * bf16_to_f32(wr[k]);
                worst_a = std::max(worst_a, std::fabs((double) ha[(size_t) (t - row0) * kN + n] - ref) / std::max(std::fabs((double) ref), 1e-6));
                worst_b = std::max(worst_b, std::fabs((double) hb[(size_t) (t - row0) * kN + n] - ref) / std::max(std::fabs((double) ref), 1e-6));
            }
        }
        std::printf("\nT=%d boundary rows vs CPU reference: arm A max rel %.3e, arm B max rel %.3e\n", T, worst_a,
                    worst_b);
    }

    std::printf("\ndone\n");
    return 0;
}
