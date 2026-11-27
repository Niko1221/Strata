// engine/metal/src/hy4flash_gguf/msl/hy4_flash_mm.metal - PREFILL's dense projections on the simdgroup matrix units, all in fp32.
//
//   y[t][n] = sum_k W[n][k] * x[t][k]      t < n_tok (a prefill chunk), n < n_out, k < n_in
//
// WHAT THIS IS ALLOWED TO CHANGE: prefill may use simdgroup matrix multiply, and its sums may be ordered
// differently from decode's GEMV, so prefill and decode do not agree bitwise on a position (--prefill-consistency
// is a tolerance judge).  What it may NOT do is lose precision:
//   * the weights are dequantized to fp32 with ref_quant's arithmetic - BF16 is widened exactly, Q8_0 is
//     float(int8) * d (ref_quant's order; d * q of an fp16 scale and an int8 code is exact in fp32), F32 is read as
//     is - so every tile element is exactly the weight decode multiplies with;
//   * the activations stay fp32;
//   * the products and the accumulation are simdgroup_float8x8, fp32 in and fp32 out.
// Nothing is rounded to fp16 / bf16 (unlike the Qwen prompt path's dense_mm.metal, which rounds both operands to half).
//
// HALF = true (hy4_flash_mmh, HY4_FLASH_PREFILL_DENSE_MM=half) stages both operands as fp16 instead, still accumulating in fp32.
// It is NOT the default and stays an opt-in A/B: this tile is staging / dequantization bound rather than
// matrix-unit bound, so narrowing the operands buys almost nothing and costs orders of magnitude of accuracy.
//
// ONE BODY FOR BOTH: the staging type and the operand matrix type are the template's, the accumulator is always
// simdgroup_float8x8, and the threadgroup pool is carved by hand so neither variant pays for the other's size.
//
// TILING.  A thread group (4 simdgroups, 128 threads) owns BM tokens x BN = 64 output rows; simdgroup s owns the
// 16 rows [16 s, 16 s + 16) of that, as (BM / 8) x 2 8x8 accumulators.  K runs in steps of BK: the step's BN x BK
// weight tile is dequantized into threadgroup memory (BK / 32 Q8_0 blocks a row, BK BF16 / F32 values a row) and the
// BM x BK activation tile copied beside it; then BK / 8 substeps multiply A = x (tokens x k) by B = W^T (k x rows,
// loaded transposed from the n-major tile).  Rows past n_out and tokens past n_tok are clamped on load and never
// stored.  Both staging loops spread 128 threads over the tile: BN * BK / 128 weight values a thread (two threads a
// row, since BN = 64), BM * BK / 128 activation floats (128 / BM threads a token).
// n_in must be a multiple of BK (every hy4-flash dense matrix is a multiple of 64, and the engine's BK is 32).
//
// grid (ceil(n_out / 64), ceil(n_tok / 32), heads), threads 128.  Per-head weights (wk_b / wv_b) use the same head
// strides as gemv.metal (function constants 10-12: W bytes, x floats, y floats a head).

struct Hy4FlashDenseMmArgs {
    uint n_in, n_out, n_tok, row_bytes, x_stride, y_stride;
};

// OPT-IN RANGE PROBE (function constants 44 / 45 - hy4-flash's own range starts at 40, hy4_flash_norm.metal; HY4_FLASH_MM_RANGE on
// the host): every thread group folds max |w| and max |x| over the operands it staged into slot HY4_FLASH_MM_SLOT of
// buffer 4 (two uints: the float bits of non-negative values order like the floats).  Undefined: no buffer 4, no
// code.  It answers whether an operand would fit fp16.
constant bool HY4_FLASH_MM_RANGE_C [[function_constant(44)]];
constant bool hy4_flash_mm_range = is_function_constant_defined(HY4_FLASH_MM_RANGE_C) && HY4_FLASH_MM_RANGE_C;
constant uint HY4_FLASH_MM_SLOT [[function_constant(45)]];

// BM (tokens a thread group) and BK (the K step) are template parameters so the test tree can measure other tiles
// without a second copy of the body (tmp/tests/hy4_flash/metal/hy4_flash_dense_tiles.metal instantiates them); BN and the
// thread count are fixed by the 4 simdgroups x (kTT x 2) accumulator layout.  BK must be a multiple of 32 (MmDec
// decodes 16 values at a time and Q8_0's block is 32 wide) and BM a multiple of 8 that divides 128.
#define HY4_FLASH_MM_BN 64
#define HY4_FLASH_MM_BM 32   // the engine's, and the default below
#define HY4_FLASH_MM_BK 32

// One row's 16 weights starting at k0 (a multiple of 16), as fp32, exactly as ref_quant dequantizes them.
template <int F> struct MmDec;
template <> struct MmDec<T_Q8_0> {
    static inline void run(device const uchar* row, int k0, thread float* w) {
        device const uchar* b = row + (k0 / 32) * 34;
        const float d = rdh(b);
        device const ushort* p = (device const ushort*)(b + 2) + 8 * ((k0 / 16) & 1);
        for (int j = 0; j < 8; ++j) {
            const ushort v = p[j];
            w[2 * j] = float(as_type<char>(uchar(v & 255))) * d;
            w[2 * j + 1] = float(as_type<char>(uchar(v >> 8))) * d;
        }
    }
};
template <> struct MmDec<T_BF16> {
    static inline void run(device const uchar* row, int k0, thread float* w) {
        device const uint4* p = (device const uint4*)(row + 2 * k0);
        const uint4 a = p[0], c = p[1];
        const uint v[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
        for (int j = 0; j < 8; ++j) {
            w[2 * j] = as_type<float>(v[j] << 16);
            w[2 * j + 1] = as_type<float>(v[j] & 0xFFFF0000u);
        }
    }
};
template <> struct MmDec<T_F32> {
    static inline void run(device const uchar* row, int k0, thread float* w) {
        device const float4* p = (device const float4*)(row + 4 * k0);
        for (int j = 0; j < 4; ++j) {
            const float4 v = p[j];
            w[4 * j] = v.x; w[4 * j + 1] = v.y; w[4 * j + 2] = v.z; w[4 * j + 3] = v.w;
        }
    }
};

/// The operand matrix that goes with a staging type.
template <typename ST> struct MmMat;
template <> struct MmMat<float> { typedef simdgroup_float8x8 mat; };
template <> struct MmMat<half> { typedef simdgroup_half8x8 mat; };

template <int F, typename ST, int BM = HY4_FLASH_MM_BM, int BK = HY4_FLASH_MM_BK>
kernel void hy4_flash_mm(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]], constant Hy4FlashDenseMmArgs& a [[buffer(3)]],
                   device atomic_uint* rng [[buffer(4), function_constant(hy4_flash_mm_range)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]]) {
    float mw = 0.0f, mx = 0.0f;   // the range probe
    if (gemv_heads) {
        W += ulong(tg.z) * GEMV_HW;
        x += ulong(tg.z) * GEMV_HX;
        y += ulong(tg.z) * GEMV_HY;
    }
    // the pool: the two staging tiles, and at the end the edge path's fp32 result tile over the same bytes
    constexpr int BN = HY4_FLASH_MM_BN, LD = BK + 4;   // the row stride pads by 4 so a column walk spreads over banks
    constexpr int kStage = (BN + BM) * LD;                                 // elements of ST
    constexpr int kPoolF = (int(sizeof(ST)) * kStage + 3) / 4 > BM * BN ? (int(sizeof(ST)) * kStage + 3) / 4
                                                                       : BM * BN;   // floats
    threadgroup float pool[kPoolF];
    threadgroup ST* Ws = (threadgroup ST*) pool;
    threadgroup ST* Xs = Ws + BN * LD;
    const int n0 = int(tg.x) * BN, t0 = int(tg.y) * BM;
    const int N = int(a.n_out), T = int(a.n_tok), K = int(a.n_in);
    // the loaders: weight row wr (two threads a row, 16 values each), activation token xt (four threads, 8 each)
    constexpr int kWPT = BN * BK / 128, kXTT = 128 / BM, kXPT = BK / kXTT, kTT = BM / 8;
    // The weight loader decodes in 16-wide chunks but stages all kWPT values, so kWPT must be a
    // NON-ZERO multiple of 16: the chunk count is an integer division, and a kWPT under 16 gives zero
    // iterations while the staging still writes uninitialised registers.  Unlike MmDec's 32-wide
    // requirement, that one fails silently, which is why it is an assertion and not a comment.
    static_assert(BN * BK % 2048 == 0, "BN * BK must be a multiple of 2048: kWPT = BN * BK / 128 has "
                                       "to be a non-zero multiple of 16 for the weight staging");
    const int wr = tid >> 1, wh = tid & 1;              // weight row, and which half of its BK values
    device const uchar* wrow = W + ulong(min(n0 + wr, N - 1)) * a.row_bytes;
    const int xt = tid / kXTT, xq = tid % kXTT;         // activation token, and which slice of its BK floats
    device const float* xrow = x + ulong(min(t0 + xt, T - 1)) * a.x_stride;
    simdgroup_float8x8 acc[kTT][2];
    for (int i = 0; i < kTT; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += BK) {
        // the weights first, into registers, so the dequantization overlaps the previous step's multiply
        float w[kWPT];
        for (int c = 0; c < kWPT / 16; ++c) MmDec<F>::run(wrow, k0 + kWPT * wh + 16 * c, w + 16 * c);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous step's tiles are consumed
        for (int j = 0; j < kWPT; ++j) Ws[wr * LD + kWPT * wh + j] = ST(w[j]);
        if (hy4_flash_mm_range)
            for (int j = 0; j < kWPT; ++j) mw = max(mw, fabs(w[j]));
        {
            device const float4* xv = (device const float4*)(xrow + k0 + kXPT * xq);
            threadgroup ST* xd = Xs + xt * LD + kXPT * xq;
            for (int c = 0; c < kXPT / 4; ++c) {
                const float4 v = xv[c];
                xd[4 * c] = ST(v.x); xd[4 * c + 1] = ST(v.y); xd[4 * c + 2] = ST(v.z); xd[4 * c + 3] = ST(v.w);
                if (hy4_flash_mm_range) {
                    const float4 m = fabs(v);
                    mx = max(mx, max(max(m.x, m.y), max(m.z, m.w)));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            typename MmMat<ST>::mat A[kTT], B[2];
            for (int i = 0; i < kTT; ++i) simdgroup_load(A[i], Xs + (8 * i) * LD + kk, LD);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(B[j], Ws + (16 * sg + 8 * j) * LD + kk, LD, ulong2(0, 0), true);
            for (int i = 0; i < kTT; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
        }
    }
    if (hy4_flash_mm_range) {
        mw = simd_max(mw);
        mx = simd_max(mx);
        if ((tid & 31) == 0) {
            atomic_fetch_max_explicit(rng + 2 * HY4_FLASH_MM_SLOT, as_type<uint>(mw), memory_order_relaxed);
            atomic_fetch_max_explicit(rng + 2 * HY4_FLASH_MM_SLOT + 1, as_type<uint>(mx), memory_order_relaxed);
        }
    }
    // store: whole tiles straight to y, edge tiles through threadgroup memory (reusing Ws) with a guarded copy
    const bool full = t0 + BM <= T && n0 + BN <= N;
    if (full) {
        for (int i = 0; i < kTT; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(acc[i][j], y + ulong(t0 + 8 * i) * a.y_stride + n0 + 16 * sg + 8 * j, a.y_stride);
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* Cs = pool;   // BM x BN floats, which kPoolF covers
    for (int i = 0; i < kTT; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Cs + (8 * i) * BN + 16 * sg + 8 * j, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < BM * BN; e += 128) {
        const int t = e / BN, n = e % BN;
        if (t0 + t < T && n0 + n < N) y[ulong(t0 + t) * a.y_stride + n0 + n] = Cs[e];
    }
}

#define HY4_FLASH_MM_INST(T, N) \
    template [[host_name("hy4_flash_mm_" #N)]] kernel decltype(hy4_flash_mm<T, float>) hy4_flash_mm<T, float>; \
    template [[host_name("hy4_flash_mmh_" #N)]] kernel decltype(hy4_flash_mm<T, half>) hy4_flash_mm<T, half>;
HY4_FLASH_MM_INST(T_Q8_0, 8)
HY4_FLASH_MM_INST(T_BF16, 30)
HY4_FLASH_MM_INST(T_F32, 0)
