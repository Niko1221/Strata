// metal/kernels/dense_mm.metal - dense projections for a PROMPT chunk (after quant/units/mma.metal):
//   act_f16    X[t][k] -> half (the matrix units' operand)
//   dq_f16<F>  a quantized matrix -> half rows (ref_quant's values rounded to fp16), once per chunk
//   gemm_f16   Y[t][n] = Xh[t] . Wh[n] (fp32), TM x TN tiles, K dynamic; rows of Y past n_out / t past n_tok are not
//              written. Two bodies: MetalPerformancePrimitives matmul2d (the M5's neural accelerators) and, for the
//              SDKs that do not ship that header at all, the same tiles built from simdgroup_matrix. A host build can
//              define STRATA_MSL_MPP to pick one and A/B it against the other.
#if !defined(STRATA_MSL_MPP)
#  if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#    define STRATA_MSL_MPP 1
#  else
#    define STRATA_MSL_MPP 0
#  endif
#endif
#if STRATA_MSL_MPP
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif

kernel void act_f16(device const float* x [[buffer(0)]], device half* xh [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) xh[gid] = half(x[gid]);
}

struct DqArgs {
    uint n_in;       // values per row
    uint n_out;      // rows
    uint row_bytes;
    uint perm;       // 0: row r -> r; p > 0: row r = c * (n_out / p) + d -> d * p + c (the HC up matrix with the p
                     // streams of an output interleaved, for gemm_hc_up's epilogue)
};

// one thread per unit: grid (units per row / 64, n_out), 64 threads
template <int F>
kernel void dq_f16(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]], constant DqArgs& a [[buffer(2)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, H = UNIT / 2;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u = int(tg.x) * 64 + tid, r = int(tg.y);
    if (u * UNIT >= int(a.n_in)) return;
    // the unit's chunk origin: a 128-wide chunk holds whole units of every format
    const int k0 = (u * UNIT) / 128 * 128;
    float w[UNIT];
    int p0, p1;
    DW<F>::run(W + ulong(r) * a.row_bytes, u, k0, w, p0, p1, tab);
    const int q = a.perm ? int(a.n_out / a.perm) : 1;
    const int ro = a.perm ? (r % q) * int(a.perm) + r / q : r;
    device half* o = Wh + ulong(ro) * a.n_in + k0;
    for (int j = 0; j < H; ++j) {
        o[p0 + j] = half(w[j]);
        o[p1 + j] = half(w[H + j]);
    }
}

struct GemmArgs { int M, N, K; int pad; };   // Y is M rows of N floats

#if STRATA_MSL_MPP
template <int TM, int TN, int SG>
kernel void gemm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* Y [[buffer(2)]],
                     constant GemmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mY);
}
#else
// gemm_f16 without the tensor ops: SG simdgroups, each one a TM/2 x TN/2 sub-tile of 8 x 8 accumulators over a K in
// steps of 8. matmul2d clipped an edge tile with its tensor extents; here a block whose rows run past the matrix is
// loaded from a base pushed back inside it and stored at that same pushed-back base, so the products dropped are
// exactly the rows and columns outside M / N and every value written is the product of the rows read for it. A
// duplicate block may write a row a first block already wrote, with the same number.
template <int TM, int TN, int SG>
kernel void gemm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* Y [[buffer(2)]],
                     constant GemmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]], ushort li [[thread_index_in_simdgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int SM = TM / 2, SN = TN / 2, NI = SM / 8, NJ = SN / 8;
    const int M = a.M, N = a.N, K = a.K;
    const int m0 = int(tg.y) * TM + int(sg >> 1) * SM, n0 = int(tg.x) * TN + int(sg & 1) * SN;
    if (M < 8 || N < 8 || K % 8 != 0) {   // a shape an 8 x 8 block cannot clip: one output per thread, in bounds
        for (int i = int(tid); i < TM * TN; i += 32 * SG) {
            const int r = m0 + i / TN, c = n0 + i % TN;
            if (r >= M || c >= N) continue;
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += float(A[ulong(r) * K + k]) * float(B[ulong(c) * K + k]);
            Y[ulong(r) * N + c] = s;
        }
        return;
    }
    int mbs[NI], nbs[NJ];
    simdgroup_matrix<float, 8, 8> acc[NI][NJ];
    for (int i = 0; i < NI; ++i) {
        mbs[i] = min(m0 + 8 * i, M - 8);
        for (int j = 0; j < NJ; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (int j = 0; j < NJ; ++j) nbs[j] = min(n0 + 8 * j, N - 8);
    for (int k = 0; k < K; k += 8) {
        simdgroup_matrix<half, 8, 8> ma[NI], mb[NJ];
        for (int i = 0; i < NI; ++i) simdgroup_load(ma[i], A + ulong(mbs[i]) * K + k, K);
        for (int j = 0; j < NJ; ++j) simdgroup_load(mb[j], B + ulong(nbs[j]) * K + k, K, ulong2(0, 0), true);
        for (int i = 0; i < NI; ++i)
            for (int j = 0; j < NJ; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }
    threadgroup float st[SG][64];
    for (int i = 0; i < NI; ++i)
        for (int j = 0; j < NJ; ++j) {
            const int r = mbs[i], c = nbs[j];
            if (r == m0 + 8 * i && c == n0 + 8 * j) {
                simdgroup_store(acc[i][j], Y + ulong(r) * N + c, N);
                continue;
            }
            simdgroup_store(acc[i][j], &st[sg][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = li; e < 64; e += 32) {
                const int lr = e >> 3, lc = e & 7;
                if (r + lr < M && c + lc < N) Y[ulong(r + lr) * N + c + lc] = st[sg][lr * 8 + lc];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}
#endif

#define DQ(F) template [[host_name("dq_f16_" #F)]] kernel decltype(dq_f16<F>) dq_f16<F>;
#define DQ_(F) DQ(F)
DQ_(T_F32) DQ_(T_BF16) DQ_(T_Q8_0) DQ_(T_Q4_0) DQ_(T_Q5_0) DQ_(T_IQ4_NL) DQ_(T_IQ4_XS) DQ_(T_Q2_0) DQ_(T_IQ3_S) DQ_(T_Q3_K)
DQ_(T_Q4_K) DQ_(T_Q5_K) DQ_(T_Q6_K)
template [[host_name("gemm_f16_64_64_4")]] kernel decltype(gemm_f16<64, 64, 4>) gemm_f16<64, 64, 4>;
template [[host_name("gemm_f16_128_64_4")]] kernel decltype(gemm_f16<128, 64, 4>) gemm_f16<128, 64, 4>;
template [[host_name("gemm_f16_64_128_4")]] kernel decltype(gemm_f16<64, 128, 4>) gemm_f16<64, 128, 4>;
template [[host_name("gemm_f16_64_32_4")]] kernel decltype(gemm_f16<64, 32, 4>) gemm_f16<64, 32, 4>;
