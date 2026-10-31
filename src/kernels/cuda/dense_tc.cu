// src/kernels/cuda/dense_tc.cu - dense short-window INT8 tensor-core projections (BW-DENSE P2.1; sm_120 only).
//
// y[col][row] = sum over the row's blocks of a Q6_K (more formats follow) weight row times a q8_1 activation column,
// for 1..8 columns, as mma.sync.m16n8k16 s8 x s8 -> s32: one weight sub-block (16 values sharing one int8 scale) of
// 16 rows against 8 activation columns per instruction. The weights are decoded straight from the GGUF blocks into
// the instruction's A fragments; the activation columns are staged once per thread block in padded shared memory and
// read as the B fragments.
//
// Arithmetic (the contract in dense_tc.hpp): per (row, column), block b, activation block ab = 8b + a (a = 0..7):
//     S      = sc[2a] * D(2a) + sc[2a+1] * D(2a+1)           D(.) the exact int32 sub-block dots, S exact in int32
//     inner  = fma(d8[col][ab], float(S), inner)             a ascending, inner starts at 0 for every block
//     acc    = fma(d_row[b], inner, acc)                     the blocks a warp owns, ascending
// and the k_split warps of a row tile are summed in warp order ((p0 + p1) + p2) + ... . Nothing in it depends on the
// column count. The weight values are the file's: (ql | qh << 4) - 32 exactly as dequantize_row_q6_K.
//
// Written from the mechanism only (m16n8k16 INT8 MMA fed by decoded quant blocks, q8_1 inputs, FP32 scaled
// accumulation); it shares no code with another engine.
#include "strata/kernels/dense_tc.hpp"

#if !defined(__HIPCC__)

#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::kernels {
namespace {

constexpr int kQ6Bytes = 210;
constexpr int kMaxCols = 8;
// Gate for the tensor-core body: the instruction and the tile live on sm_80+; the kernel is only ever launched on a
// 12.0 card, so older code objects in a multi-arch fat binary get an empty body (never launched).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define STRATA_DTC_BODY 1
#endif

__device__ __forceinline__ int ldg_u16(const void* p) {
    unsigned v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=r"(v) : "l"(p));
    return int(v);
}
// a 32-bit little-endian load from a 2-byte aligned address (Q6_K blocks are 210 bytes)
__device__ __forceinline__ int ldg_b2(const uint8_t* p) { return ldg_u16(p) | (ldg_u16(p + 2) << 16); }
__device__ __forceinline__ int ldg_s8(const void* p) {
    int v;
    asm volatile("ld.global.nc.s8 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
// (x - 32) per byte for bytes of x in [0, 63]: x + 0x60 never carries out of its byte (0x60..0x9f), the xor undoes the 0x80
__device__ __forceinline__ int sub32_bytes(int x) { return (x + 0x60606060) ^ 0x80808080; }

__device__ __forceinline__ void mma_k16(int (&c)[4], int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
}

struct Smem {   // dynamic shared memory: the activation tile, then (reused after the last MMA) the k_split partials
    static __host__ __device__ int pitch(int n_in) { return n_in + 16; }   // bytes a column; +16 spreads the 8 columns over the banks
    static __host__ __device__ size_t q_bytes(int n_in) { return size_t(kMaxCols) * size_t(pitch(n_in)); }
    static __host__ __device__ size_t d_bytes(int n_in) { return size_t(n_in / 32) * kMaxCols * 4; }
    static __host__ __device__ size_t total(int n_in, int rt, int ks) {
        const size_t act = q_bytes(n_in) + d_bytes(n_in);
        const size_t red = size_t(rt) * ks * 16 * 8 * 4;
        return act > red ? act : red;
    }
};

// RT row tiles (16 rows each) x KS k-split warps per thread block. Grid: ceil(ceil(n_out / 16) / RT).
template <int RT, int KS>
__global__ void __launch_bounds__(32 * RT * KS) dense_tc_q6k_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ xq,
                                                                    float* __restrict__ y, int n_in, int n_out, int T) {
#if defined(STRATA_DTC_BODY)
    extern __shared__ __align__(16) uint8_t smem[];
    const int tid = int(threadIdx.x);
    const int nb = n_in / 256, nab = n_in / 32;
    const int pitch = Smem::pitch(n_in);
    uint8_t* sq = smem;
    float* sd = reinterpret_cast<float*>(smem + Smem::q_bytes(n_in));   // [ab][8]
    // stage the activation columns: the int8 codes (4-byte aligned in the 36-byte blocks), and each block's d
    {
        const int per_col = n_in / 4;
        for (int i = tid; i < T * per_col; i += 32 * RT * KS) {
            const int col = i / per_col, k4 = i - col * per_col;
            const int ab = k4 >> 3, p = k4 & 7;
            const int v = *reinterpret_cast<const int*>(xq + (size_t(col) * nab + ab) * 36 + 4 + 4 * p);
            *reinterpret_cast<int*>(sq + size_t(col) * pitch + 4 * k4) = v;
        }
        for (int i = tid; i < nab * kMaxCols; i += 32 * RT * KS) {
            const int ab = i >> 3, col = i & 7;
            sd[i] = col < T ? __half2float(*reinterpret_cast<const __half*>(xq + (size_t(col) * nab + ab) * 36)) : 0.0f;
        }
    }
    __syncthreads();

    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int rt = warp / KS, ks = warp - rt * KS;
    const int row_base = (int(blockIdx.x) * RT + rt) * 16;
    const int r0 = min(row_base + g, n_out - 1), r1 = min(row_base + g + 8, n_out - 1);
    const uint8_t* w0 = w + size_t(r0) * nb * kQ6Bytes;
    const uint8_t* w1 = w + size_t(r1) * nb * kQ6Bytes;
    const int gcol = g < T ? g : 0;   // columns >= T: any valid column (their outputs are not stored)
    const uint8_t* bq = sq + size_t(gcol) * pitch + 4 * t;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};   // (row g, cols 2t, 2t+1), (row g + 8, cols 2t, 2t+1)

    for (int b = ks; b < nb; b += KS) {
        const uint8_t* p0 = w0 + size_t(b) * kQ6Bytes;
        const uint8_t* p1 = w1 + size_t(b) * kQ6Bytes;
        // the 16 int8 scales of both rows: sc?[sj]
        int sc0[16], sc1[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            sc0[j] = ldg_s8(p0 + 192 + j);
            sc1[j] = ldg_s8(p1 + 192 + j);
        }
        const float dw0 = __half2float(__ushort_as_half((unsigned short) ldg_u16(p0 + 208)));
        const float dw1 = __half2float(__ushort_as_half((unsigned short) ldg_u16(p1 + 208)));
        float inner[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            // lh = 0/1: the 16-value half of the 32 a quadrant covers; this lane's 4 values are 4t..4t+3 of it
            int qlA[2][2], qlB[2][2], qh[2][2];   // [row][lh]
#pragma unroll
            for (int lh = 0; lh < 2; ++lh) {
                const int off = 64 * h + 16 * lh + 4 * t;
                qlA[0][lh] = ldg_b2(p0 + off);
                qlB[0][lh] = ldg_b2(p0 + off + 32);
                qh[0][lh] = ldg_b2(p0 + 128 + 32 * h + 16 * lh + 4 * t);
                qlA[1][lh] = ldg_b2(p1 + off);
                qlB[1][lh] = ldg_b2(p1 + off + 32);
                qh[1][lh] = ldg_b2(p1 + 128 + 32 * h + 16 * lh + 4 * t);
            }
#pragma unroll
            for (int qd = 0; qd < 4; ++qd) {   // quadrant: value offset 32 * qd inside the half
                const int ab_l = 4 * h + qd;   // activation block inside the weight block
                int c[2][4];
#pragma unroll
                for (int lh = 0; lh < 2; ++lh) {
                    const int sj = 8 * h + 2 * qd + lh;
                    int a[2];
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int ql = (qd & 1) ? qlB[r][lh] : qlA[r][lh];
                        const int lo = (qd >> 1) ? ((ql >> 4) & 0x0f0f0f0f) : (ql & 0x0f0f0f0f);
                        const int hi = ((qh[r][lh] >> (2 * qd)) & 0x03030303) << 4;
                        a[r] = sub32_bytes(lo | hi);
                    }
                    const int bv = *reinterpret_cast<const int*>(bq + 32 * (8 * b + ab_l) + 16 * lh);
                    mma_k16(c[lh], a[0], a[1], bv);
                }
                const float2 d8 = *reinterpret_cast<const float2*>(sd + (8 * b + ab_l) * kMaxCols + 2 * t);
                const int s2 = 8 * h + 2 * qd;
                // S for (row g | row g + 8) x (col 2t | 2t + 1)
                const int S0 = sc0[s2] * c[0][0] + sc0[s2 + 1] * c[1][0];
                const int S1 = sc0[s2] * c[0][1] + sc0[s2 + 1] * c[1][1];
                const int S2 = sc1[s2] * c[0][2] + sc1[s2 + 1] * c[1][2];
                const int S3 = sc1[s2] * c[0][3] + sc1[s2 + 1] * c[1][3];
                inner[0] = fmaf(d8.x, float(S0), inner[0]);
                inner[1] = fmaf(d8.y, float(S1), inner[1]);
                inner[2] = fmaf(d8.x, float(S2), inner[2]);
                inner[3] = fmaf(d8.y, float(S3), inner[3]);
            }
        }
        acc[0] = fmaf(dw0, inner[0], acc[0]);
        acc[1] = fmaf(dw0, inner[1], acc[1]);
        acc[2] = fmaf(dw1, inner[2], acc[2]);
        acc[3] = fmaf(dw1, inner[3], acc[3]);
    }

    // sum the k_split partials of each row tile in warp order, then store
    __syncthreads();   // every warp is done with the activation tile
    float* red = reinterpret_cast<float*>(smem);   // [rt][ks][4][32]
    if (KS > 1) {
#pragma unroll
        for (int e = 0; e < 4; ++e) red[((rt * KS + ks) * 4 + e) * 32 + lane] = acc[e];
        __syncthreads();
    }
    if (ks == 0) {
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            v[e] = acc[e];
            for (int k = 1; k < KS; ++k) v[e] += red[((rt * KS + k) * 4 + e) * 32 + lane];
        }
        const int col0 = 2 * t;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = row_base + g + (e >> 1) * 8, col = col0 + (e & 1);
            if (row < n_out && col < T) y[size_t(col) * n_out + row] = v[e];
        }
    }
#endif
}

// ---- host side ------------------------------------------------------------------------------------------------
int g_enable = -1;   // dense_tc_enable
int g_tile_rt = 0, g_tile_ks = 0;
int g_variant = 0;

bool env_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_DENSE_TC"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

int device_cc() {   // 120 for a 12.0 card (emulation honoured), else 0
    static int cc[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
    if (cc[dev] == 0) {
        int ma = 0, mi = 0;
        if (cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        cc[dev] = (ma == 12 && mi == 0) ? 120 : -1;
    }
    return cc[dev] > 0 ? cc[dev] : 0;
}

struct Pick { int rt, ks; };
// The (row tiles, k_split) of a shape. The k_split decides the float summation order, so it is a function of the
// shape alone: never of the column count.
Pick pick(int type, int n_in, int n_out) {
    (void) type;
    if (g_tile_rt > 0 && g_tile_ks > 0) return {g_tile_rt, g_tile_ks};
    const int nb = n_in / 256;
    const int tiles = (n_out + 15) / 16;
    int ks = nb % 5 == 0 ? 5 : nb % 4 == 0 ? 4 : nb % 2 == 0 ? 2 : 1;
    if (ks == 5 && tiles >= 640) ks = nb % 2 == 0 && nb >= 20 ? 2 : 1;
    if (tiles >= 1024) ks = nb % 2 == 0 ? 2 : 1;
    int rt = tiles >= 256 ? 2 : 1;
    if (ks == 5 && rt > 2) rt = 2;
    return {rt, ks};
}

template <int RT, int KS>
bool launch_q6k(const uint8_t* w, const uint8_t* xq, float* y, int n_in, int n_out, int T, cudaStream_t s) {
    const size_t smem = Smem::total(n_in, RT, KS);
    static std::once_flag once;
    static bool ok = true;
    std::call_once(once, [&] {
        ok = cudaFuncSetAttribute(dense_tc_q6k_kernel<RT, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(99 * 1024)) == cudaSuccess;
        if (!ok) cudaGetLastError();
    });
    if (!ok) return false;
    const unsigned tiles = unsigned((n_out + 15) / 16);
    const unsigned blocks = (tiles + RT - 1) / RT;
    dense_tc_q6k_kernel<RT, KS><<<blocks, 32 * RT * KS, smem, s>>>(w, xq, y, n_in, n_out, T);
    return true;
}

bool dispatch_q6k(const uint8_t* w, const uint8_t* xq, float* y, int n_in, int n_out, int T, cudaStream_t s, Pick p) {
#define DTC_CASE(R, K) if (p.rt == R && p.ks == K) return launch_q6k<R, K>(w, xq, y, n_in, n_out, T, s);
    DTC_CASE(1, 1) DTC_CASE(1, 2) DTC_CASE(1, 4) DTC_CASE(1, 5) DTC_CASE(1, 10)
    DTC_CASE(2, 1) DTC_CASE(2, 2) DTC_CASE(2, 4) DTC_CASE(2, 5)
    DTC_CASE(4, 1) DTC_CASE(4, 2)
#undef DTC_CASE
    return false;
}

bool tile_known(Pick p) {
    static const int ok[][2] = {{1, 1}, {1, 2}, {1, 4}, {1, 5}, {1, 10}, {2, 1}, {2, 2}, {2, 4}, {2, 5}, {4, 1}, {4, 2}};
    for (const auto& o : ok) if (o[0] == p.rt && o[1] == p.ks) return true;
    return false;
}

}  // namespace

const char* dense_tc_reason(int st) noexcept {
    switch (st) {
    case kDenseTcOk: return "ok";
    case kDenseTcOff: return "off";
    case kDenseTcArch: return "arch";
    case kDenseTcType: return "type";
    case kDenseTcCols: return "cols";
    case kDenseTcShape: return "shape";
    case kDenseTcAlign: return "align";
    case kDenseTcSmem: return "smem";
    case kDenseTcBuild: return "build";
    default: return "unknown";
    }
}

void dense_tc_enable(int on) { g_enable = on; }
bool dense_tc_enabled() { return g_enable >= 0 ? g_enable != 0 : env_on(); }
void dense_tc_tile(int rt, int ks) { g_tile_rt = rt; g_tile_ks = ks; }
int dense_tc_variant(int v) { if (v >= 0) g_variant = v; return g_variant; }
int dense_tc_split_for(int type, int n_in, int n_out) { return pick(type, n_in, n_out).ks; }

int dense_tc_check(int ggml_type, int n_in, int n_out, int ncols, const void* weights, const void* x_q8_1, const void* y) noexcept {
    if (!dense_tc_enabled()) return kDenseTcOff;
    if (device_cc() != 120) return kDenseTcArch;
    if (ggml_type != 14) return kDenseTcType;
    if (ncols < 1 || ncols > kMaxCols) return kDenseTcCols;
    if (n_in <= 0 || n_in % 256 != 0 || n_out <= 0) return kDenseTcShape;
    const Pick p = pick(ggml_type, n_in, n_out);
    if (!tile_known(p)) return kDenseTcShape;
    if (Smem::total(n_in, p.rt, p.ks) > size_t(99) * 1024) return kDenseTcShape;
    if ((weights != nullptr && (reinterpret_cast<uintptr_t>(weights) & 1)) ||
        (x_q8_1 != nullptr && (reinterpret_cast<uintptr_t>(x_q8_1) & 3)) || (y != nullptr && (reinterpret_cast<uintptr_t>(y) & 3)))
        return kDenseTcAlign;
    return kDenseTcOk;
}

bool dense_tc(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (dense_tc_check(ggml_type, n_in, n_out, ncols, weights, x_q8_1, y) != kDenseTcOk) return false;
    const Pick p = pick(ggml_type, n_in, n_out);
    const bool ok = dispatch_q6k(static_cast<const uint8_t*>(weights), static_cast<const uint8_t*>(x_q8_1), y, n_in, n_out, ncols,
                                 static_cast<cudaStream_t>(stream), p);
    return ok;
}

void dense_tc_print_resources() {
    cudaFuncAttributes a;
    auto one = [&](const char* name, const void* f) {
        if (cudaFuncGetAttributes(&a, f) == cudaSuccess)
            std::printf("dense_tc %-24s regs %3d  spill/local %4zu B  static smem %zu B  max threads %d  binary sm_%d\n", name, a.numRegs,
                        a.localSizeBytes, a.sharedSizeBytes, a.maxThreadsPerBlock, a.binaryVersion);
        else {
            cudaGetLastError();
            std::printf("dense_tc %-24s (no attributes)\n", name);
        }
    };
    one("q6k rt1 ks1", reinterpret_cast<const void*>(dense_tc_q6k_kernel<1, 1>));
    one("q6k rt1 ks2", reinterpret_cast<const void*>(dense_tc_q6k_kernel<1, 2>));
    one("q6k rt1 ks4", reinterpret_cast<const void*>(dense_tc_q6k_kernel<1, 4>));
    one("q6k rt1 ks5", reinterpret_cast<const void*>(dense_tc_q6k_kernel<1, 5>));
    one("q6k rt2 ks2", reinterpret_cast<const void*>(dense_tc_q6k_kernel<2, 2>));
    one("q6k rt2 ks4", reinterpret_cast<const void*>(dense_tc_q6k_kernel<2, 4>));
    one("q6k rt4 ks1", reinterpret_cast<const void*>(dense_tc_q6k_kernel<4, 1>));
}

}  // namespace strata::kernels

#else  // HIP: no tensor-core path, every call keeps native_mmvq

namespace strata::kernels {
const char* dense_tc_reason(int) noexcept { return "build"; }
void dense_tc_enable(int) {}
bool dense_tc_enabled() { return false; }
void dense_tc_tile(int, int) {}
int dense_tc_variant(int) { return 0; }
int dense_tc_split_for(int, int, int) { return 1; }
int dense_tc_check(int, int, int, int, const void*, const void*, const void*) noexcept { return kDenseTcBuild; }
bool dense_tc(int, const void*, const void*, float*, int, int, int, void*) { return false; }
void dense_tc_print_resources() {}
}  // namespace strata::kernels

#endif
