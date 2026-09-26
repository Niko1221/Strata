// src/kernels/expert_grouped_parity.cu - `native_expert_grouped` (the verify window's and the second GPU's grouped
// experts) against the kernels it replaced and the i-quant dot products as they were, kept here verbatim: bitwise
// over every GPU gate/up and down format, on synthetic experts (random weights quantized by ggml) and plans of 1-8
// tokens.
//
//     expert_grouped_parity [--bench]
//
// --bench times both inside CUDA graphs (a kernel launched outside one pays WDDM's ~30-40 us) on the model files'
// format pairs, with routings shaped like the verify windows' (tokens share about a quarter of their experts).
#include "strata/kernels/iq_kernels.hpp"
#include "iq_dot.cuh"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t H = 2560, FF = 640, K = 10, MAXT = 8, SLOTS = 48, DISTINCT = 4;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

// ---- the kernels native_expert_grouped launched before, verbatim, and the i-quant dot products as they were
// (sign masks through __vcmpne4 / __vsub4)
namespace strata::kernels::ref {

__device__ __forceinline__ float ref_vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float ref_vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float ref_vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float ref_vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float ref_vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

template<int TY> struct RefFmt : Fmt<TY> {};
template<> struct RefFmt<16> : Fmt<16> {
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return ref_vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct RefFmt<17> : Fmt<17> {
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return ref_vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct RefFmt<18> : Fmt<18> {
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return ref_vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct RefFmt<21> : Fmt<21> {
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return ref_vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct RefFmt<22> : Fmt<22> {
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return ref_vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };

template<int TY>
__device__ __forceinline__ float ref_row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = RefFmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template<int TG>
__global__ void __launch_bounds__(256) native_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                        float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / RefFmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = ref_row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
        if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
    }
}

__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + __expf(-g))) * up[i];
}

template<int TD>
__global__ void __launch_bounds__(256) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / RefFmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = ref_row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane);
        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
    }
}

__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}

size_t scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

void grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
             const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
             int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, cudaStream_t s) {
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const dim3 ggu((unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned) cap_groups);
    switch (L.gu_type) {
#define STRATA_GU(T) case T: native_gu_kernel<T><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        STRATA_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "ref: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    const long long nh = (long long) cap_entries * L.n_ff;
    swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, nh);
    quantize_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh);
    const dim3 gd((unsigned) ((L.n_embd + 7) / 8), (unsigned) cap_groups);
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: native_down_kernel<T><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        STRATA_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "ref: down type %d\n", L.d_type); std::exit(1);
    }
}

}  // namespace strata::kernels::ref

namespace {

// DISTINCT experts of random weights (rows of different magnitude) quantized by ggml, repeated over SLOTS slots of one
// device arena so the kernels stream them from memory as they do the VRAM tier
uint8_t* make_arena(const strata::kernels::NativeExpertLayout& L, size_t stride, int seed) {
    std::vector<uint8_t> blobs((size_t) DISTINCT * L.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        const std::vector<float> imatrix((size_t) cols, 1.0f);
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    for (int b = 0; b < DISTINCT; ++b) {
        uint8_t* blob = blobs.data() + (size_t) b * L.bytes;
        quant(L.gu_type, FF, H, blob);
        quant(L.gu_type, FF, H, blob + L.up_off);
        quant(L.d_type, H, FF, blob + L.down_off);
    }
    uint8_t* arena = nullptr;
    check(cudaMalloc(&arena, (size_t) SLOTS * stride), "arena");
    for (int i = 0; i < SLOTS; ++i)
        check(cudaMemcpy(arena + (size_t) i * stride, blobs.data() + (size_t) (i % DISTINCT) * L.bytes, L.bytes,
                         cudaMemcpyHostToDevice), "arena upload");
    return arena;
}

// A window's routing: n_tok tokens of K distinct experts each, a quarter drawn from the earlier tokens' ones (the
// verify windows' overlap); `hit` of the experts resident.  Grouped as the plan is: distinct resident experts in
// routing order, each with its entries in routing order.
struct Plan {
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst, tok;
};
Plan make_plan(std::mt19937& rng, int n_tok, double hit, const uint8_t* arena, size_t stride) {
    std::vector<int32_t> ids;
    std::vector<uint8_t> resident(SLOTS);
    for (auto& r : resident) r = std::uniform_real_distribution<double>(0, 1)(rng) < hit;
    for (int t = 0; t < n_tok; ++t)
        for (int j = 0; j < K; ++j) {
            int32_t e;
            bool again;
            do {
                const bool reuse = t > 0 && rng() % 4 == 0;
                e = reuse ? ids[rng() % (size_t) (t * K)] : (int32_t) (rng() % SLOTS);
                again = false;
                for (int m = 0; m < j; ++m) again = again || ids[(size_t) (t * K + m)] == e;
            } while (again);
            ids.push_back(e);
        }
    Plan p;
    const int n = n_tok * (int) K;
    for (int i = 0; i < n; ++i) {
        bool first = resident[(size_t) ids[(size_t) i]] != 0;
        for (int j = 0; j < i && first; ++j) first = ids[(size_t) j] != ids[(size_t) i];
        if (!first) continue;
        p.ptr.push_back((unsigned long long) (arena + (size_t) ids[(size_t) i] * stride));
        p.start.push_back((int32_t) p.dst.size());
        for (int j = i; j < n; ++j)
            if (ids[(size_t) j] == ids[(size_t) i]) { p.dst.push_back(j); p.tok.push_back(j / (int) K); }
    }
    p.start.push_back((int32_t) p.dst.size());
    return p;
}

struct Dev {
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *n = nullptr, *dst = nullptr, *tok = nullptr;
    void *xq = nullptr, *scratch_ref = nullptr, *scratch_new = nullptr;
    float *out_ref = nullptr, *out_new = nullptr;
};

void upload(const Plan& p, const Dev& d) {
    const int32_t groups = (int32_t) p.ptr.size();
    if (groups > 0) {
        check(cudaMemcpy(d.ptr, p.ptr.data(), p.ptr.size() * 8, cudaMemcpyHostToDevice), "plan");
        check(cudaMemcpy(d.dst, p.dst.data(), p.dst.size() * 4, cudaMemcpyHostToDevice), "plan");
        check(cudaMemcpy(d.tok, p.tok.data(), p.tok.size() * 4, cudaMemcpyHostToDevice), "plan");
    }
    check(cudaMemcpy(d.start, p.start.data(), p.start.size() * 4, cudaMemcpyHostToDevice), "plan");
    check(cudaMemcpy(d.n, &groups, 4, cudaMemcpyHostToDevice), "plan");
}

void run(bool reference, const strata::kernels::NativeExpertLayout& L, const Dev& d, int64_t cap, cudaStream_t s) {
    if (reference)
        strata::kernels::ref::grouped(L, d.ptr, d.start, d.n, d.dst, d.tok, cap, cap, d.xq, d.scratch_ref, d.out_ref, s);
    else
        strata::kernels::native_expert_grouped(L, d.ptr, d.start, d.n, d.dst, d.tok, cap, cap, d.xq, d.scratch_new,
                                               d.out_new, s);
}

// microseconds per call, inside a graph of `reps` calls
double time_graph(bool reference, const strata::kernels::NativeExpertLayout& L, const Dev& d, int64_t cap,
                  cudaStream_t s) {
    const int reps = 50;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < reps; ++i) run(reference, L, d, cap, s);
    check(cudaStreamEndCapture(s, &graph), "capture");
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    const auto w0 = std::chrono::steady_clock::now();   // until the GPU has clocked up
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 0.3) {
        check(cudaGraphLaunch(exec, s), "launch");
        check(cudaStreamSynchronize(s), "warm");
    }
    double best = 1e30;
    for (int k = 0; k < 10; ++k) {
        const auto t0 = std::chrono::steady_clock::now();
        check(cudaGraphLaunch(exec, s), "launch");
        check(cudaStreamSynchronize(s), "sync");
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    if (argc > 2 || (argc == 2 && !bench)) {
        std::fprintf(stderr, "usage: expert_grouped_parity [--bench]\n");
        return 2;
    }
    cudaStream_t s;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    const int64_t cap = MAXT * K;
    Dev d;
    check(cudaMalloc(&d.ptr, (size_t) cap * 8), "dev");
    check(cudaMalloc(&d.start, (size_t) (cap + 1) * 4), "dev");
    check(cudaMalloc(&d.n, 4), "dev");
    check(cudaMalloc(&d.dst, (size_t) cap * 4), "dev");
    check(cudaMalloc(&d.tok, (size_t) cap * 4), "dev");
    check(cudaMalloc(&d.xq, (size_t) MAXT * (H / 32) * sizeof(block_q8_1)), "dev");
    check(cudaMalloc(&d.scratch_ref, strata::kernels::ref::scratch_bytes(cap, FF)), "dev");
    check(cudaMalloc(&d.scratch_new, strata::kernels::native_expert_scratch_bytes(cap, FF)), "dev");
    check(cudaMalloc(&d.out_ref, (size_t) cap * H * 4), "dev");
    check(cudaMalloc(&d.out_new, (size_t) cap * H * 4), "dev");
    {   // the tokens' activations
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) (MAXT * H));
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        check(cudaMalloc(&dx, x.size() * 4), "dev");
        check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        strata::kernels::quantize_q8_1_rows(dx, MAXT, H, d.xq, s);
        check(cudaStreamSynchronize(s), "x");
        cudaFree(dx);
    }
    // parity: every gate/up format with a down format in turn (down rows are n_ff = 640 long: 32- and 64-blocks);
    // bench: the model files' pairs (IQ2_XXS .. IQ3_S with IQ4_NL / Q2_0: the IQ3_XXS file; Q4_K, Q5_K with Q5_1,
    // Q8_0: UD-Q4_K_XL)
    const int gus[] = {7, 8, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42}, downs[] = {7, 8, 20, 42};
    std::vector<std::pair<int, int>> pairs;
    if (bench) pairs = {{16, 42}, {17, 20}, {18, 20}, {21, 42}, {22, 20}, {12, 7}, {13, 8}};
    else
        for (size_t i = 0; i < sizeof gus / sizeof gus[0]; ++i) pairs.emplace_back(gus[i], downs[i % 4]);
    int bad = 0;
    std::mt19937 rng(11);
    for (const auto& [gu, dn] : pairs) {
        const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(gu, dn, H, FF);
        const size_t stride = (L.bytes + 255) & ~(size_t) 255;
        uint8_t* arena = make_arena(L, stride, gu * 100 + dn);
        const char* names = ggml_type_name((ggml_type) gu);
        if (!bench) {
            int cases = 0, diff = 0;
            for (int n_tok : {1, 2, 3, 4, 8})
                for (int rep = 0; rep < 4; ++rep) {
                    const Plan p = make_plan(rng, n_tok, rep == 0 ? 1.0 : 0.8, arena, stride);
                    upload(p, d);
                    check(cudaMemset(d.out_ref, 0xff, (size_t) cap * H * 4), "memset");
                    check(cudaMemset(d.out_new, 0xff, (size_t) cap * H * 4), "memset");
                    run(true, L, d, n_tok * K, s);
                    run(false, L, d, n_tok * K, s);
                    check(cudaStreamSynchronize(s), "run");
                    std::vector<uint32_t> a((size_t) cap * H), b((size_t) cap * H);
                    check(cudaMemcpy(a.data(), d.out_ref, a.size() * 4, cudaMemcpyDeviceToHost), "down");
                    check(cudaMemcpy(b.data(), d.out_new, b.size() * 4, cudaMemcpyDeviceToHost), "down");
                    diff += a != b;
                    ++cases;
                }
            std::printf("%-8s/%-7s %d plans: %s\n", names, ggml_type_name((ggml_type) dn), cases,
                        diff ? "MISMATCH" : "bitwise equal");
            bad += diff;
        } else {
            std::printf("%-8s/%-7s", names, ggml_type_name((ggml_type) dn));
            for (int n_tok : {1, 2, 4}) {
                const Plan p = make_plan(rng, n_tok, 0.92, arena, stride);
                upload(p, d);
                const double tr = time_graph(true, L, d, n_tok * K, s), tn = time_graph(false, L, d, n_tok * K, s);
                std::printf("  T%d %2zu experts %6.1f -> %5.1f us (%4.0f GB/s)", n_tok, p.ptr.size(), tr, tn,
                            (double) (p.ptr.size() * L.bytes) / tn * 1e-3);
            }
            std::printf("\n");
        }
        cudaFree(arena);
    }
    if (!bench) std::printf("expert_grouped_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
