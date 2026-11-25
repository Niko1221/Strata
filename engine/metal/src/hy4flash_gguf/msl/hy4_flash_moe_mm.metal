// engine/metal/src/hy4flash_gguf/msl/hy4_flash_moe_mm.metal - the routed experts of a PREFILL chunk as a grouped GEMM on the simdgroup
// matrix units.  Prefill may use MMA and a different summation order; fp16 operands are allowed, with fp32
// accumulation.
//
// WHAT IS AND IS NOT THE SAME AS DECODE.  Decode (hy4_flash_moe.metal) is a GEMV per entry whose reduction order is frozen
// for T invariance.  Prefill may sum in another order and (S = half) round each operand once to fp16:
//   * the weights are dequantized with ref_quant's arithmetic in fp32 - the values decode's units multiply by -
//     IQ2_XXS  db * (+-grid), db = d * (0.5 + s) * 0.25;  Q4_K  d1 * q - m1 (d1 = d * sc, m1 = dmin * m) - then
//     stored as S (float: exact; half: one rounding);
//   * the activations are read as fp32 and stored as S the same way; products and sums are fp32;
//   * the clamped SwiGLU, the routing weight applied after the down, and the combine in SLOT order are decode's
//     (the combine is hy4_flash_moe.metal's hy4_flash_moe_combine_latent itself).
// The plan is hy4_flash_moe.metal's (groups = experts with contiguous entries), plus a host-built TILE LIST: one
// (group, first entry) per TM entries of a group.
//
// REUSE.  A threadgroup takes TM entries of one expert x TN outputs and steps K by TK = 32.  Each K step stages the
// tile's x rows and dequantizes TN x 32 weights ONCE into threadgroup memory; the SGM x SGN simdgroups then split the
// TM x TN tile (each TM / SGM rows x TN / SGN columns, (TM / SGM / 8) x (TN / SGN / 8) fp32 8x8 accumulators).  So one
// dequantized weight serves TM entries: TM 64 / 128 at a 4096 chunk (~128 entries an expert) is what moves the
// kernel off the dequantization bound.
//
// Output: each simdgroup stores a finished 8x8 fragment into its own 256 B of threadgroup memory and its 32 lanes
// write the fragment's entries that exist (r < ne), 2 values a lane - no TM x TN output buffer.
#pragma METAL fp contract(off)
#include <metal_simdgroup_matrix>   // simdgroup_float8x8 and friends: named, not inherited from the file order
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct Hy4FlashMmArgs {
    ulong gate_off;     // bytes from an expert's base to its gate matrix (as Hy4FlashMoeArgs)
    ulong up_off;
    ulong down_off;
    ulong gu_stride;
    ulong d_stride;
    uint gu_row;        // bytes per gate/up row
    uint d_row;         // bytes per down row
    uint n_latent;
    uint n_ff;
    uint n_tile;        // tiles in the tile list
    uint off_gexp;      // ints into the plan buffer (hy4_flash_moe.metal's sections)
    uint off_gstart;
    uint off_etok;
    uint off_ew;
    float limit;
};

constant int HY4_FLASH_MM_TK = 32;

// Function constant 60 = 1: ONE running sum over all of K (the default sums each 32-wide K step in a fresh fragment
// and adds it - summation depth 32 + K / 32 instead of K, so more accurate, at the cost of registers).
constant int HY4_FLASH_MM_FLAT [[function_constant(60)]];
constant bool hy4_flash_mm_flat = is_function_constant_defined(HY4_FLASH_MM_FLAT) && HY4_FLASH_MM_FLAT == 1;
static inline void mm_add(thread simdgroup_float8x8& acc, thread simdgroup_float8x8& step) {
    thread auto& a = acc.thread_elements();
    thread auto& b = step.thread_elements();
    a[0] += b[0];
    a[1] += b[1];
}

// ---- 16 consecutive weights, dequantized in fp32 with ref_quant's arithmetic, stored as S ----------------------------
template <typename S>
static inline void mm_deq16_iq2xxs(device const uchar* row, int k0, threadgroup S* dst, threadgroup const uint* tab) {
    device const uchar* b = row + (k0 >> 8) * 66;
    float q[16];
    const float db = deq<T_IQ2_XXS>(b, (k0 & 255) >> 4, q, tab);
    for (int j = 0; j < 16; ++j) dst[j] = S(db * q[j]);
}
static inline void mm_scale_min_k4(int j, device const uchar* q, thread uint& d, thread uint& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
// Q4_K: 64-chunks of a 256-block are 32 low nibbles then 32 high nibbles of the same 32 bytes
template <typename S>
static inline void mm_deq16_q4k(device const uchar* row, int k0, threadgroup S* dst) {
    device const uchar* b = row + (k0 >> 8) * 144;
    const int k = k0 & 255, c = k >> 6, hi = (k >> 5) & 1, l0 = k & 31;
    const float d = float(*(device const half*)b), mn = float(*(device const half*)(b + 2));
    uint s, m;
    mm_scale_min_k4(2 * c + hi, b + 4, s, m);
    const float d1 = d * float(s), m1 = mn * float(m);
    device const uchar* q = b + 16 + 32 * c + l0;
    for (int j = 0; j < 16; ++j) {
        const uint v = hi ? (q[j] >> 4) : (q[j] & 15);
        dst[j] = S(d1 * float(v) - m1);
    }
}

// x rows of the tile: TM x TK, read as fp32 float4s, stored as S; rows past the tile's entries are zero
template <int TM, typename S, int TK = HY4_FLASH_MM_TK>
static inline void mm_stage_x(device const float* base, device const int* rows, bool by_token, int c, int ne,
                              int width, int k0, threadgroup S* xs, ushort tid, int nt) {
    for (int i = tid; i < TM * TK / 4; i += nt) {
        const int r = i / (TK / 4), q4 = i % (TK / 4);
        float4 v = float4(0.0f);
        if (r < ne) {
            const ulong row = by_token ? (ulong)rows[r] : (ulong)(c + r);
            v = ((device const float4*)(base + row * width + k0))[q4];
        }
        for (int z = 0; z < 4; ++z) xs[4 * i + z] = S(v[z]);
    }
}

// a finished 8x8 fragment (rows r0.., columns n0..) out through the simdgroup's 64 floats: entry rows < ne only
static inline void mm_put8(thread simdgroup_float8x8& f, threadgroup float* scr, ushort lane, int r0, int ne,
                           device float* out, ulong row_stride, int col0, int n_cols, int c,
                           device const int* ew_plan, bool weight) {
    simdgroup_store(f, scr, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int z = 0; z < 2; ++z) {
        const int e = 2 * int(lane) + z, r = r0 + e / 8, col = col0 + e % 8;
        if (r < ne && col < n_cols) {
            float v = scr[e];
            if (weight) v = v * as_type<float>(ew_plan[c + r]);
            out[(ulong)(c + r) * row_stride + col] = v;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// ---- gate + up + the clamped SwiGLU ----------------------------------------------------------------------------------
// grid (ceil(n_ff / TN), n_tile); threads SGM * SGN * 32.  tiles[i] = (group, first entry of the tile in the group).
// The threadgroup budget, checked by the compiler instead of by a comment: a tile that does not fit cannot be
// instantiated.  Without it, an over-budget tile compiles and only fails when its pipeline is created.
#define HY4_FLASH_MM_FITS(BYTES) static_assert((BYTES) <= 32768, "the tile's threadgroup memory is over 32 KB")

template <int TM, int TN, int SGM, int SGN, typename S>
kernel void hy4_flash_mm_gu(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                      device float* h [[buffer(2)]], device const int* plan [[buffer(3)]],
                      constant Hy4FlashMmArgs& a [[buffer(4)]], device const int2* tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TK = HY4_FLASH_MM_TK, NT = SGM * SGN * 32, MS = TM / SGM, NS = TN / SGN, MF = MS / 8, NF = NS / 8;
    threadgroup uint tab[fmt<T_IQ2_XXS>::TAB];
    HY4_FLASH_MM_FITS(sizeof(S) * (TM + 2 * TN) * TK + SGM * SGN * 256 + 4 * fmt<T_IQ2_XXS>::TAB);
    threadgroup S xs[TM * TK];
    threadgroup S wg[TN * TK], wu[TN * TK];
    threadgroup float scr[SGM * SGN * 64];
    if (tg.y >= a.n_tile) return;
    load_tab<T_IQ2_XXS>(tab, tid, NT);
    const int2 tl = tiles[tg.y];
    if (tl.x < 0) return;   // a padding entry of the tile list (the host fills the tail it did not use)
    const int g = tl.x, c = plan[a.off_gstart + g] + tl.y;
    const int ne = min(TM, plan[a.off_gstart + g + 1] - c);
    device const int* etok = plan + a.off_etok + c;
    device const uchar* E = W + (ulong)plan[a.off_gexp + g] * a.gu_stride;
    const int n0 = int(tg.x) * TN, L = int(a.n_latent);
    const int sm = int(sg) / SGN, sn = int(sg) % SGN;   // this simdgroup's block of the tile
    simdgroup_float8x8 ag[MF][NF], au[MF][NF];
    for (int i = 0; i < MF; ++i)
        for (int j = 0; j < NF; ++j) {
            ag[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            au[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    for (int k0 = 0; k0 < L; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous step's reads are done (and the table is in)
        mm_stage_x<TM, S>(x, etok, true, c, ne, L, k0, xs, tid, NT);
        for (int i = tid; i < 2 * TN * (TK / 16); i += NT) {
            const int mat = i / (TN * (TK / 16)), rem = i % (TN * (TK / 16));
            const int r = rem / (TK / 16), part = rem % (TK / 16);
            const int f = min(n0 + r, int(a.n_ff) - 1);
            device const uchar* row = E + (mat ? a.up_off : a.gate_off) + (ulong)f * a.gu_row;
            mm_deq16_iq2xxs(row, k0 + 16 * part, (mat ? wu : wg) + r * TK + 16 * part, tab);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sgk[MF][NF], suk[MF][NF];
        if (!hy4_flash_mm_flat)
            for (int i = 0; i < MF; ++i)
                for (int j = 0; j < NF; ++j) {
                    sgk[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    suk[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
        for (int kk = 0; kk < TK; kk += 8) {
            simdgroup_matrix<S, 8, 8> A[MF];
            for (int i = 0; i < MF; ++i) simdgroup_load(A[i], xs + (sm * MS + 8 * i) * TK + kk, TK);
            for (int j = 0; j < NF; ++j) {
                simdgroup_matrix<S, 8, 8> Bg, Bu;   // B[k][n] = w[n][k]: staged rows are outputs, load transposed
                simdgroup_load(Bg, wg + (sn * NS + 8 * j) * TK + kk, TK, ulong2(0, 0), true);
                simdgroup_load(Bu, wu + (sn * NS + 8 * j) * TK + kk, TK, ulong2(0, 0), true);
                for (int i = 0; i < MF; ++i) {
                    if (hy4_flash_mm_flat) {
                        simdgroup_multiply_accumulate(ag[i][j], A[i], Bg, ag[i][j]);
                        simdgroup_multiply_accumulate(au[i][j], A[i], Bu, au[i][j]);
                    } else {
                        simdgroup_multiply_accumulate(sgk[i][j], A[i], Bg, sgk[i][j]);
                        simdgroup_multiply_accumulate(suk[i][j], A[i], Bu, suk[i][j]);
                    }
                }
            }
        }
        if (!hy4_flash_mm_flat)
            for (int i = 0; i < MF; ++i)
                for (int j = 0; j < NF; ++j) {
                    mm_add(ag[i][j], sgk[i][j]);
                    mm_add(au[i][j], suk[i][j]);
                }
    }
    threadgroup float* my = scr + 64 * int(sg);
    for (int i = 0; i < MF; ++i)
        for (int j = 0; j < NF; ++j) {
            thread auto& eg = ag[i][j].thread_elements();
            thread auto& eu = au[i][j].thread_elements();
            for (int t = 0; t < 2; ++t) {   // gate and up fragments share the element layout
                const float g2 = min(eg[t], a.limit), u2 = clamp(eu[t], -a.limit, a.limit);
                eg[t] = hy4_flash_silu(g2) * u2;
            }
            mm_put8(ag[i][j], my, lane, sm * MS + 8 * i, ne, h, a.n_ff, n0 + sn * NS + 8 * j, int(a.n_ff), c, plan,
                    false);
        }
}

// ---- down, with the routing weight applied after the dot ------------------------------------------------------------
// grid (ceil(n_latent / TN), n_tile); threads SGM * SGN * 32.
template <int TM, int TN, int SGM, int SGN, typename S>
kernel void hy4_flash_mm_down(device const uchar* W [[buffer(0)]], device const float* h [[buffer(1)]],
                        device float* ye [[buffer(2)]], device const int* plan [[buffer(3)]],
                        constant Hy4FlashMmArgs& a [[buffer(4)]], device const int2* tiles [[buffer(5)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TK = HY4_FLASH_MM_TK, NT = SGM * SGN * 32, MS = TM / SGM, NS = TN / SGN, MF = MS / 8, NF = NS / 8;
    HY4_FLASH_MM_FITS(sizeof(S) * (TM + TN) * TK + SGM * SGN * 256);
    threadgroup S xs[TM * TK];
    threadgroup S wd[TN * TK];
    threadgroup float scr[SGM * SGN * 64];
    if (tg.y >= a.n_tile) return;
    const int2 tl = tiles[tg.y];
    if (tl.x < 0) return;   // a padding entry of the tile list (the host fills the tail it did not use)
    const int g = tl.x, c = plan[a.off_gstart + g] + tl.y;
    const int ne = min(TM, plan[a.off_gstart + g + 1] - c);
    device const uchar* D = W + (ulong)plan[a.off_gexp + g] * a.d_stride + a.down_off;
    const int n0 = int(tg.x) * TN, F = int(a.n_ff);
    const int sm = int(sg) / SGN, sn = int(sg) % SGN;
    simdgroup_float8x8 ac[MF][NF];
    for (int i = 0; i < MF; ++i)
        for (int j = 0; j < NF; ++j) ac[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < F; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm_stage_x<TM, S>(h, plan, false, c, ne, F, k0, xs, tid, NT);
        for (int i = tid; i < TN * (TK / 16); i += NT) {
            const int r = i / (TK / 16), part = i % (TK / 16);
            const int d = min(n0 + r, int(a.n_latent) - 1);
            mm_deq16_q4k(D + (ulong)d * a.d_row, k0 + 16 * part, wd + r * TK + 16 * part);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sk[MF][NF];
        if (!hy4_flash_mm_flat)
            for (int i = 0; i < MF; ++i)
                for (int j = 0; j < NF; ++j) sk[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int kk = 0; kk < TK; kk += 8) {
            simdgroup_matrix<S, 8, 8> A[MF];
            for (int i = 0; i < MF; ++i) simdgroup_load(A[i], xs + (sm * MS + 8 * i) * TK + kk, TK);
            for (int j = 0; j < NF; ++j) {
                simdgroup_matrix<S, 8, 8> B;
                simdgroup_load(B, wd + (sn * NS + 8 * j) * TK + kk, TK, ulong2(0, 0), true);
                for (int i = 0; i < MF; ++i) {
                    if (hy4_flash_mm_flat) simdgroup_multiply_accumulate(ac[i][j], A[i], B, ac[i][j]);
                    else simdgroup_multiply_accumulate(sk[i][j], A[i], B, sk[i][j]);
                }
            }
        }
        if (!hy4_flash_mm_flat)
            for (int i = 0; i < MF; ++i)
                for (int j = 0; j < NF; ++j) mm_add(ac[i][j], sk[i][j]);
    }
    threadgroup float* my = scr + 64 * int(sg);
    for (int i = 0; i < MF; ++i)
        for (int j = 0; j < NF; ++j)
            mm_put8(ac[i][j], my, lane, sm * MS + 8 * i, ne, ye, a.n_latent, n0 + sn * NS + 8 * j, int(a.n_latent),
                    c, plan + a.off_ew, true);
}

// ---- instantiations: (TM, TN, SGM, SGN, operand type) ---------------------------------------------------------------
// host name hy4_flash_mm_{gu,down}_TM_TN_SGM_SGN_{float,half}.  Threadgroup memory (gate/up, half): TM*64 + TN*128 + SG*256
// + 2176 B; (float) twice the first two terms.
#define HY4_FLASH_MMGU(M, N, A, B, T) template [[host_name("hy4_flash_mm_gu_" #M "_" #N "_" #A "_" #B "_" #T)]] \
                               kernel decltype(hy4_flash_mm_gu<M, N, A, B, T>) hy4_flash_mm_gu<M, N, A, B, T>;
#define HY4_FLASH_MMDN(M, N, A, B, T) template [[host_name("hy4_flash_mm_down_" #M "_" #N "_" #A "_" #B "_" #T)]] \
                               kernel decltype(hy4_flash_mm_down<M, N, A, B, T>) hy4_flash_mm_down<M, N, A, B, T>;
#define HY4_FLASH_MM(M, N, A, B, T) HY4_FLASH_MMGU(M, N, A, B, T) HY4_FLASH_MMDN(M, N, A, B, T)
// layouts where the simdgroups split N only
HY4_FLASH_MM(32, 64, 1, 4, float) HY4_FLASH_MM(16, 64, 1, 4, half) HY4_FLASH_MM(8, 64, 1, 4, half) HY4_FLASH_MM(32, 64, 1, 4, half)
// simdgroups split M: one dequantized weight serves TM entries
HY4_FLASH_MM(64, 32, 4, 1, half) HY4_FLASH_MM(64, 64, 4, 2, half) HY4_FLASH_MM(128, 32, 8, 1, half) HY4_FLASH_MM(128, 64, 8, 2, half)
HY4_FLASH_MM(64, 32, 2, 2, half) HY4_FLASH_MM(64, 64, 2, 2, half) HY4_FLASH_MM(128, 32, 4, 2, half) HY4_FLASH_MM(32, 32, 2, 2, half)
HY4_FLASH_MM(64, 32, 4, 1, float) HY4_FLASH_MM(64, 64, 4, 2, float) HY4_FLASH_MM(128, 32, 8, 1, float)

// ==== the same two kernels on Metal 4's matmul2d (MetalPerformancePrimitives) ========================================
// The operands are staged exactly as above (fp32-dequantized weights and fp32 x, stored as S); matmul2d multiplies
// the TM x TK and TN x TK threadgroup tiles with fp32 results.  Two-level sum by default (each K step into a fresh
// destination, added to the running one); fc 60 = 1: one running sum (multiply_accumulate).  The finished tile is
// written straight from the cooperative tensor, each element at its (row, column) from get_multidimensional_index,
// only rows < ne.
template <int TM, int TN, int SG, typename S, int TKP>
kernel void hy4_flash_mm2_gu(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                       device float* h [[buffer(2)]], device const int* plan [[buffer(3)]],
                       constant Hy4FlashMmArgs& a [[buffer(4)]], device const int2* tiles [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr int TK = TKP, NT = SG * 32;
    HY4_FLASH_MM_FITS(sizeof(S) * (TM + 2 * TN) * TK + 4 * fmt<T_IQ2_XXS>::TAB);
    threadgroup uint tab[fmt<T_IQ2_XXS>::TAB];
    threadgroup S xs[TM * TK];
    threadgroup S wg[TN * TK], wu[TN * TK];
    if (tg.y >= a.n_tile) return;
    load_tab<T_IQ2_XXS>(tab, tid, NT);
    const int2 tl = tiles[tg.y];
    if (tl.x < 0) return;   // a padding entry of the tile list (the host fills the tail it did not use)
    const int g = tl.x, c = plan[a.off_gstart + g] + tl.y;
    const int ne = min(TM, plan[a.off_gstart + g + 1] - c);
    device const int* etok = plan + a.off_etok + c;
    device const uchar* E = W + (ulong)plan[a.off_gexp + g] * a.gu_stride;
    const int n0 = int(tg.x) * TN, L = int(a.n_latent);
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    constexpr auto desc_acc = matmul2d_descriptor(TM, TN, TK, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    matmul2d<desc_acc, execution_simdgroups<SG>> op_acc;
    tensor<threadgroup S, dextents<int32_t, 2>, tensor_inline> tA(xs, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup S, dextents<int32_t, 2>, tensor_inline> tG(wg, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup S, dextents<int32_t, 2>, tensor_inline> tU(wu, dextents<int32_t, 2>(TK, TN));
    auto accg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    auto accu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int k0 = 0; k0 < L; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm_stage_x<TM, S, TK>(x, etok, true, c, ne, L, k0, xs, tid, NT);
        for (int i = tid; i < 2 * TN * (TK / 16); i += NT) {
            const int mat = i / (TN * (TK / 16)), rem = i % (TN * (TK / 16));
            const int r = rem / (TK / 16), part = rem % (TK / 16);
            const int f = min(n0 + r, int(a.n_ff) - 1);
            device const uchar* row = E + (mat ? a.up_off : a.gate_off) + (ulong)f * a.gu_row;
            mm_deq16_iq2xxs(row, k0 + 16 * part, (mat ? wu : wg) + r * TK + 16 * part, tab);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hy4_flash_mm_flat) {
            op_acc.run(tA, tG, accg);
            op_acc.run(tA, tU, accu);
        } else {
            auto pg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
            auto pu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
            op.run(tA, tG, pg);
            op.run(tA, tU, pu);
            for (uint16_t i = 0; i < pg.get_capacity(); ++i)
                if (pg.is_valid_element(i)) { accg[i] += pg[i]; accu[i] += pu[i]; }
        }
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const auto ix = accg.get_multidimensional_index(i);   // (column, row) in the TM x TN tile
            const int col = int(ix[0]), r = int(ix[1]);
            const float g2 = min(accg[i], a.limit), u2 = clamp(accu[i], -a.limit, a.limit);
            if (r < ne && n0 + col < int(a.n_ff)) h[(ulong)(c + r) * a.n_ff + n0 + col] = hy4_flash_silu(g2) * u2;
        }
}

template <int TM, int TN, int SG, typename S, int TKP>
kernel void hy4_flash_mm2_down(device const uchar* W [[buffer(0)]], device const float* h [[buffer(1)]],
                         device float* ye [[buffer(2)]], device const int* plan [[buffer(3)]],
                         constant Hy4FlashMmArgs& a [[buffer(4)]], device const int2* tiles [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr int TK = TKP, NT = SG * 32;
    HY4_FLASH_MM_FITS(sizeof(S) * (TM + TN) * TK);
    threadgroup S xs[TM * TK];
    threadgroup S wd[TN * TK];
    if (tg.y >= a.n_tile) return;
    const int2 tl = tiles[tg.y];
    if (tl.x < 0) return;   // a padding entry of the tile list (the host fills the tail it did not use)
    const int g = tl.x, c = plan[a.off_gstart + g] + tl.y;
    const int ne = min(TM, plan[a.off_gstart + g + 1] - c);
    device const uchar* D = W + (ulong)plan[a.off_gexp + g] * a.d_stride + a.down_off;
    const int n0 = int(tg.x) * TN, F = int(a.n_ff);
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    constexpr auto desc_acc = matmul2d_descriptor(TM, TN, TK, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    matmul2d<desc_acc, execution_simdgroups<SG>> op_acc;
    tensor<threadgroup S, dextents<int32_t, 2>, tensor_inline> tA(xs, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup S, dextents<int32_t, 2>, tensor_inline> tB(wd, dextents<int32_t, 2>(TK, TN));
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (int k0 = 0; k0 < F; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm_stage_x<TM, S, TK>(h, plan, false, c, ne, F, k0, xs, tid, NT);
        for (int i = tid; i < TN * (TK / 16); i += NT) {
            const int r = i / (TK / 16), part = i % (TK / 16);
            const int d = min(n0 + r, int(a.n_latent) - 1);
            mm_deq16_q4k(D + (ulong)d * a.d_row, k0 + 16 * part, wd + r * TK + 16 * part);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hy4_flash_mm_flat) op_acc.run(tA, tB, acc);
        else {
            auto p = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
            op.run(tA, tB, p);
            for (uint16_t i = 0; i < p.get_capacity(); ++i)
                if (p.is_valid_element(i)) acc[i] += p[i];
        }
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            const int col = int(ix[0]), r = int(ix[1]);
            if (r < ne && n0 + col < int(a.n_latent))
                ye[(ulong)(c + r) * a.n_latent + n0 + col] = acc[i] * as_type<float>(plan[a.off_ew + c + r]);
        }
}

// host name hy4_flash_mm2_{gu,down}_TM_TN_SG_{half,float}[_kTK] (TK 32 has no suffix)
#define HY4_FLASH_MM2GU_(M, N, G, T, K, NM) template [[host_name("hy4_flash_mm2_gu_" #M "_" #N "_" #G "_" #T NM)]] \
                                     kernel decltype(hy4_flash_mm2_gu<M, N, G, T, K>) hy4_flash_mm2_gu<M, N, G, T, K>;
#define HY4_FLASH_MM2DN_(M, N, G, T, K, NM) template [[host_name("hy4_flash_mm2_down_" #M "_" #N "_" #G "_" #T NM)]] \
                                     kernel decltype(hy4_flash_mm2_down<M, N, G, T, K>) hy4_flash_mm2_down<M, N, G, T, K>;
#define HY4_FLASH_MM2(M, N, G, T) HY4_FLASH_MM2GU_(M, N, G, T, 32, "") HY4_FLASH_MM2DN_(M, N, G, T, 32, "")
#define HY4_FLASH_MM2K(M, N, G, T, K) HY4_FLASH_MM2GU_(M, N, G, T, K, "_k" #K) HY4_FLASH_MM2DN_(M, N, G, T, K, "_k" #K)
HY4_FLASH_MM2(64, 32, 4, half) HY4_FLASH_MM2(64, 64, 4, half) HY4_FLASH_MM2(32, 32, 2, half) HY4_FLASH_MM2(16, 32, 1, half)
HY4_FLASH_MM2(32, 64, 2, half) HY4_FLASH_MM2(128, 32, 4, half) HY4_FLASH_MM2(128, 64, 4, half) HY4_FLASH_MM2(64, 32, 2, half)
HY4_FLASH_MM2(64, 32, 4, float) HY4_FLASH_MM2(64, 64, 4, float) HY4_FLASH_MM2(32, 32, 2, float) HY4_FLASH_MM2(128, 32, 4, float)
HY4_FLASH_MM2(16, 32, 1, float) HY4_FLASH_MM2(32, 64, 2, float)
HY4_FLASH_MM2K(64, 32, 4, half, 64) HY4_FLASH_MM2K(64, 64, 4, half, 64) HY4_FLASH_MM2K(32, 32, 2, half, 64)
// TK 128 at TM 64 would declare 34,944 B of threadgroup memory for the gate/up (two weight tiles), over the 32 KB
// limit, so the pipeline cannot be created.
HY4_FLASH_MM2K(32, 32, 2, float, 64) HY4_FLASH_MM2K(32, 64, 2, half, 64)   // f32 gate/up at TM 64, TK 64: > 32 KB
