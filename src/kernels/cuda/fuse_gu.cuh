// src/kernels/cuda/fuse_gu.cuh - STRATA_FUSE_GU (p58/fuse-gu, opt-in): the Q6_K gate/up of the grouped decode experts,
// SwiGLU and the q8_1 blocks of h in ONE launch (UD-Q6_K_XL: Q6_K gate/up, Q8_0 down, n_embd 2560, n_ff 640).
//
// Included once by iq_kernels.cu (after its first anonymous namespace, inside strata::kernels), so it uses that TU's
// Split<14>, warp_sum, q8_1_store and flags (no fast-math), as the kernels it replaces do.
//
// What it replaces (the default for gate/up type 14 on every CUDA card): native_gu_kernel<14> (one warp per weight row,
// each entry re-reads and re-decodes the row: row_dot<14>), the gate/up FP32 round trip through scratch, then
// swiglu_q8_1_entries_kernel.  Here a block owns 32 hidden columns r0 .. r0 + 31 of a group (64 weight rows: the
// gate and up rows of those columns, 134 KB at n_embd 2560); each warp takes CPW = 32 / NW columns, loads a row's
// ten Q6_K block words once into registers (Split<14>::load: ql / qh words, the two int8 scales, d) and applies them
// to every entry of the group (Split<14>::apply), forms h = silu(gate) * up per (entry, column) in the block, and
// warp c then writes entry c's q8_1 block of those 32 columns with q8_1_store.  The group's q8_1 activations are
// staged in shared memory once per block (XS) or read from global memory (!XS).
//
// Bitwise contract (fuse_gu_bench memcmp's out and hq against the default path on real GGUF rows):
//   * per (row, entry): lane L accumulates calls k = L + 32 b, b = 0 .. 9 (kbx = b, iqs = L) in that order from
//     0.0f, s += Split<14>::apply(load(row, b, L), x + 8 b, L) - row_dot<14>'s lane-strided order with Fmt<14>::dot
//     split as native_gu_multi_kernel<14> splits it (iq_multi_parity: bitwise under STRATA_Q6K_SPLIT=1) - then the
//     same warp_sum butterfly;
//   * the default multiplies by the NVFP4 scale 1.0f (exact) - the fused path is taken only without a tail;
//   * h = __fmul_rn(g / (1.0f + __expf(-g)), u): swiglu_q8_1_entries_kernel's expression on the same two floats;
//   * q8_1_store(h, hq, e * n_ff + r): the same function, the same 32-value block (columns r0 .. r0 + 31).
// No FP32 sum changes order, so there is no exact-order variant to add.
#if !defined(__HIPCC__)   // CUDA only: HIP's __fmul_rn is a plain product (see swiglu_q8_1_entries_kernel)
namespace {

constexpr int FG_COLS = 32;   // hidden columns per block: one q8_1 block of h per entry
constexpr int FG_NE = 8;      // entries per pass (a group holds at most one entry per window row)
constexpr int FG_NB = 10;     // Q6_K blocks per row at n_embd 2560
constexpr int FG_XB = 80;     // q8_1 blocks per activation row at n_embd 2560

// Split<14> with the weight side in 3 registers instead of 5 (the two int8 scales and the fp16 d packed in one word,
// unpacked in apply to the same int / float values), so ten calls' words stay in registers without a stack frame
// (Split<14>'s W spilled 104 bytes at 80 registers).  apply runs Split<14>::apply's operations on the same values in
// the same order: per i, vi = __vsubss4(vil | vih, 0x20202020), sumf += d8[i] * (dp4a(vi, u[i], 0) * sc[i]), then
// d * sumf (Split<14> is vec_dot_q6_K_q8_1's order: iq_multi_parity under STRATA_Q6K_SPLIT=1).
struct FgQ6 {
    struct W { int vl, vh, p; };   // p: d's fp16 bits << 16 | (uint8) scales[so + 4] << 8 | (uint8) scales[so]
    __device__ __forceinline__ static W load(const uint8_t* __restrict__ row, int kbx, int iqs) {
        const block_q6_K* bq6_K = (const block_q6_K*) row + kbx;
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        W r;
        r.vl = get_int_b2(bq6_K->ql, iqs);
        r.vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
        static_assert(QR6_K == 2, "two scales per call");
        r.p = (int) (((unsigned) __half_as_ushort(bq6_K->d) << 16) | ((unsigned) (uint8_t) bq6_K->scales[scale_offset + 4] << 8) |
                     (unsigned) (uint8_t) bq6_K->scales[scale_offset]);
        return r;
    }
    __device__ __forceinline__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        int u[QR6_K];
        float d8[QR6_K];
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
            d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
        }
        const int vl = r.vl, vh = r.vh;
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const int sc = (r.p << (24 - 8 * i)) >> 24;   // the int8 scale, sign-extended (= (int) scales[so + 4 i])
            const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = __vsubss4(vil | vih, 0x20202020);   // vi = (vil | vih) - 32
            sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
        }
        const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) r.p >> 16)));
        return d * sumf;
    }
};

// RP 1: a column's gate row, then its up row (10 weight words in registers); RP 2: both rows' words at once (20).
template<int NW, int RP, bool XS, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) fgu_q6k_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                const int32_t* __restrict__ grp_start,
                                                                const int32_t* __restrict__ n_groups,
                                                                const int32_t* __restrict__ ent_tok,
                                                                const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                                block_q8_1* __restrict__ hq) {
    static_assert(FG_COLS % NW == 0, "whole columns per warp");
    constexpr int CPW = FG_COLS / NW;
    using S = FgQ6;
    __shared__ float sh_h[FG_NE][FG_COLS];
    __shared__ block_q8_1 sh_x[XS ? FG_NE * FG_XB : 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ng = *n_groups;
    const int r0 = (int) blockIdx.x * FG_COLS;
    const size_t up_off = L.up_off, gu_row = L.gu_row;
    const long long n_ff = L.n_ff;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int eb = e0; eb < e1; eb += FG_NE) {
            const int n = min(FG_NE, e1 - eb);
            if constexpr (XS) {   // the pass's activation rows into shared memory (36-byte blocks, 9 words each)
                int* dst = (int*) sh_x;
                for (int i = threadIdx.x; i < n * FG_XB * 9; i += NW * 32) {
                    const int c = i / (FG_XB * 9), w = i - c * (FG_XB * 9);
                    dst[i] = ((const int*) (xq + (size_t) ent_tok[eb + c] * FG_XB))[w];
                }
                __syncthreads();
            }
#pragma unroll 1
            for (int j = 0; j < CPW; ++j) {
                const int cl = warp * CPW + j, r = r0 + cl;
                const uint8_t* wg = blob + (size_t) r * gu_row;
                const uint8_t* wu = blob + up_off + (size_t) r * gu_row;
                if constexpr (RP == 2) {
                    typename S::W wgt[FG_NB], wup[FG_NB];
#pragma unroll
                    for (int b = 0; b < FG_NB; ++b) {
                        wgt[b] = S::load(wg, b, lane);
                        wup[b] = S::load(wu, b, lane);
                    }
#pragma unroll 1
                    for (int c = 0; c < n; ++c) {
                        const block_q8_1* x = XS ? sh_x + c * FG_XB : xq + (size_t) ent_tok[eb + c] * FG_XB;
                        float sg = 0.0f, su = 0.0f;
#pragma unroll
                        for (int b = 0; b < FG_NB; ++b) sg += S::apply(wgt[b], x + b * 8, lane);
#pragma unroll
                        for (int b = 0; b < FG_NB; ++b) su += S::apply(wup[b], x + b * 8, lane);
                        sg = warp_sum(sg);
                        su = warp_sum(su);
                        if (lane == 0) sh_h[c][cl] = __fmul_rn(sg / (1.0f + __expf(-sg)), su);
                    }
                } else {
                    typename S::W w[FG_NB];
#pragma unroll
                    for (int b = 0; b < FG_NB; ++b) w[b] = S::load(wg, b, lane);
#pragma unroll 1
                    for (int c = 0; c < n; ++c) {
                        const block_q8_1* x = XS ? sh_x + c * FG_XB : xq + (size_t) ent_tok[eb + c] * FG_XB;
                        float sg = 0.0f;
#pragma unroll
                        for (int b = 0; b < FG_NB; ++b) sg += S::apply(w[b], x + b * 8, lane);
                        sg = warp_sum(sg);
                        if (lane == 0) sh_h[c][cl] = sg;   // the gate value, until its up row is done
                    }
#pragma unroll
                    for (int b = 0; b < FG_NB; ++b) w[b] = S::load(wu, b, lane);
#pragma unroll 1
                    for (int c = 0; c < n; ++c) {
                        const block_q8_1* x = XS ? sh_x + c * FG_XB : xq + (size_t) ent_tok[eb + c] * FG_XB;
                        float su = 0.0f;
#pragma unroll
                        for (int b = 0; b < FG_NB; ++b) su += S::apply(w[b], x + b * 8, lane);
                        su = warp_sum(su);
                        if (lane == 0) {
                            const float gv = sh_h[c][cl];   // written by this lane above
                            sh_h[c][cl] = __fmul_rn(gv / (1.0f + __expf(-gv)), su);
                        }
                    }
                }
            }
            __syncthreads();   // every column's h of the pass is in sh_h
            for (int c = warp; c < n; c += NW) q8_1_store(sh_h[c][lane], hq, (long long) (eb + c) * n_ff + r0 + lane);
            __syncthreads();   // sh_h / sh_x are reused by the next pass or group
        }
    }
}

// STRATA_FUSE_GU: 0 off (default); the variants (NW warps a block, RP rows' words at once, XS staged activations, MINB
// blocks per SM; sm_89 registers / stack from cuobjdump in GU.md - every variant has a zero stack frame):
//   1 = NW 8, RP 1, activations from global memory (L1), MINB 3 - the default variant
//   2 = NW 8, RP 1, staged, MINB 2      3 = NW 8, RP 1, global, MINB 2 (the decode hoisted out of the entry loop)
//   (RP 2 is kept in the template but not launched: 128 registers and a 256-byte stack frame at MINB 2)
//   4 = NW 16, RP 1, staged, MINB 1     5 = NW 4, RP 1, staged, MINB 4  CUDA only (a HIP build never takes it).
// Read at first use (before graph capture); native_expert_set_fuse_gu overrides it (the bench).
std::atomic<int> g_fuse_gu{-1};   // bench override, set before graph capture
int fuse_gu_mode() {
    static const int configured = [] {
        const char* v = std::getenv("STRATA_FUSE_GU");
        const int m = v != nullptr ? std::atoi(v) : 0;
        return (m >= 1 && m <= 5) ? m : 0;
    }();
    const int override = g_fuse_gu.load(std::memory_order_relaxed);
    return override >= 0 ? override : configured;
}

bool fuse_gu_layout_ok(const NativeExpertLayout& L) {
    return L.gu_type == 14 && L.n_embd == 2560 && L.n_ff > 0 && L.n_ff % FG_COLS == 0 && L.tail_off == 0 &&
           L.had_block <= 0 && !L.gu_nib && L.gu_row == (size_t) FG_NB * 210;
}

// false: off, or a layout it does not take (the caller runs the default kernels).  gy: the caller's block rows of
// groups (the kernel strides over the device count, as native_gu_kernel does).
bool fuse_gu_launch(const NativeExpertLayout& L, int64_t gy, cudaStream_t s, const unsigned long long* grp_ptr,
                    const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X,
                    block_q8_1* hq) {
    const int m = fuse_gu_mode();
    if (m == 0 || !fuse_gu_layout_ok(L) || gy <= 0) return false;
    static std::once_flag noted;
    std::call_once(noted, [&] {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void) cudaStreamIsCapturing(s, &cs);
        std::fprintf(stderr, "strata: STRATA_FUSE_GU=%d: Q6_K gate/up + SwiGLU + q8_1 in one launch (n_embd %lld, n_ff %lld, down type %d)%s\n",
                     m, (long long) L.n_embd, (long long) L.n_ff, L.d_type,
                     cs != cudaStreamCaptureStatusNone ? " (captured into a graph)" : "");
    });
    const dim3 grid((unsigned) (L.n_ff / FG_COLS), (unsigned) gy);
    switch (m) {
        case 2: fgu_q6k_kernel<8, 1, true, 2><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        case 3: fgu_q6k_kernel<8, 1, false, 2><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        case 4: fgu_q6k_kernel<16, 1, true, 1><<<grid, 512, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        case 5: fgu_q6k_kernel<4, 1, true, 4><<<grid, 128, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        default: fgu_q6k_kernel<8, 1, false, 3><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
    }
    return true;
}

}  // namespace
#else
namespace {
bool fuse_gu_launch(const NativeExpertLayout&, int64_t, cudaStream_t, const unsigned long long*, const int32_t*,
                    const int32_t*, const int32_t*, const block_q8_1*, block_q8_1*) { return false; }
int g_fuse_gu = 0;
int fuse_gu_mode() { return 0; }
bool fuse_gu_layout_ok(const NativeExpertLayout&) { return false; }
}  // namespace
#endif

void native_expert_set_fuse_gu(int v) { g_fuse_gu = v < 0 ? -1 : ((v >= 1 && v <= 5) ? v : 0); }
int native_expert_fuse_gu() { return fuse_gu_mode(); }
bool native_expert_fuse_gu_takes(const NativeExpertLayout& L) { return fuse_gu_layout_ok(L); }
