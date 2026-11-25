// engine/metal/src/hy4flash_gguf/msl/hy4_flash_gemv_q8.metal - the dense GEMV for Q8_0 weights, f32 activations (attn_gate / attn_output on
// the IQ2_AH checkpoint).
//
// WHY A SECOND KERNEL.  gemv.metal's gemv decodes a unit of every row a lane owns into q[NR][32] floats and keeps
// them live across the token loop.  At T = 1 that is fine - the kernel is bandwidth bound - but at T = 2 / 3 (MTP's
// verify windows) the extra tokens' unit_combine calls run with those 64 registers held, and Q8_0 does twice the
// multiply-adds per weight byte BF16 does, so the combine, not the x loads, is what the window waits on.
// hy4_flash_gemv_q8 keeps each row's 16 raw ushorts (the unit's 32 int8) and its scale instead, and re-expands q right
// before every unit_combine.
//
// BITWISE gemv<T_Q8_0, LPR, NR, SG, KS>, by construction:
//   * the expansion is deq<T_Q8_0>'s own arithmetic on the same bytes (float(as_type<char>(byte))) and the scale is
//     rdh() of the same half, so q and sc are the same values gemv's Dec<T_Q8_0> produces;
//   * acc[r][t] += unit_combine<32>(q, sc, xs, xsum) is the same call on the same inputs, once per (row, token,
//     unit), units in gemv's order `u = u0 + li; u += LPR` with gemv's K-split bounds;
//   * lane_sum<LPR>, the K-split merge through `red` and the y write are gemv's, line for line;
//   * the per-head offsets (function constants 10-12, gemv.metal) are applied the same way.
// Only the instruction schedule and the registers differ.  hy4_kernel_parity's gemv_q8 cases compare it with gemv
// byte for byte over the tiles gemv_tile can pick.
//
// STREAM = true (the tall mid tile {16, 2, 4, 1}: attn_gate) goes one step further and never materialises q[32] or
// xs[32]: unit_combine<32>'s four FMA chains are fed straight from the held bytes and float4 loads of x,
//     s0 <- q[j], q[j+2]   s1 <- q[j+1], q[j+3]   s2 <- q[16+j], q[18+j]   s3 <- q[17+j], q[19+j]   (j = 0, 4, .., 12)
// which is unit_combine's chain order (s0 over j = 0, 2, .., 14; s1 over j + 1; s2 / s3 over H + j / H + j + 1),
// and then unit_combine's final expression with its values sc = (d, d, 0, 0), xsum = (0, 0) (X<T_Q8_0>::load zeroes
// xsum).  Same bits again.  The wide tile streams only in the 16-simdgroup shape the engine actually dispatches it
// with (GemvTile::engine_tile), so hy4_flash_gemv_q8_32_2_16_2 streams and the {32, 2, 8, 2} instantiation does not.
//
// Buffers and grid exactly as gemv: W, x, s16 (unused by Q8_0), y, GemvArgs; grid.x = row groups, grid.y = heads.

template <int LPR, int NR, int SG, int KS, bool STREAM>
kernel void hy4_flash_gemv_q8(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                        device const float* s16 [[buffer(2)]], device float* y [[buffer(3)]],
                        constant GemvArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = 32, RS = SG / KS, RG = 32 / LPR;
    if (gemv_heads) {
        W += ulong(tg.y) * GEMV_HW;
        xp += ulong(tg.y) * GEMV_HX;
        s16 += ulong(tg.y) * GEMV_HX;
        y += ulong(tg.y) * GEMV_HY;
    }
    threadgroup float red[KS > 1 ? KS * RS * RG * NR * 8 : 1];
    const int ks = sg / RS, rs = sg % RS;
    const int li = lane % LPR;
    const int lrow0 = (rs * RG + lane / LPR) * NR;
    const int row0 = int(tg.x) * RS * RG * NR + lrow0;
    const int U = int(a.n_in) / UNIT;
    const int u0 = ks * U / KS, u1 = (ks + 1) * U / KS;
    float acc[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
    device const uchar* rp[NR];
    for (int r = 0; r < NR; ++r) rp[r] = W + (ulong)min(row0 + r, int(a.n_out) - 1) * a.row_bytes;
    const int s16n = int(a.n_in) / 16;
    for (int u = u0 + li; u < u1; u += LPR) {
        // Dec<T_Q8_0>'s inputs, held raw: the block of unit u is at u * 34 (QK = UNIT = 32, one unit a block)
        ushort raw[NR][16];
        float s[NR];
        for (int r = 0; r < NR; ++r) {
            device const uchar* b = rp[r] + u * fmt<T_Q8_0>::BB;
            device const ushort* p = (device const ushort*)(b + 2);
            for (int j = 0; j < 16; ++j) raw[r][j] = p[j];
            s[r] = rdh(b);
        }
        if (STREAM && NTOK <= 2) {
            // One or two tokens (decode, and MTP k = 1's verify window): tokens innermost, so a unit's bytes are
            // expanded once per row instead of once per (row, token).  Every chain c[r][t][i] still takes its terms
            // in j order and the per-unit update of acc[r][t] is the same expression, so this is bitwise the
            // per-token loop below (and gemv).  Restricted to NTOK <= 2: beyond that the compiler keeps several
            // token groups live at once and the layout loses more than it saves.
            float c[NR][2][4];
            for (int r = 0; r < NR; ++r)
                for (int t = 0; t < NTOK; ++t) c[r][t][0] = c[r][t][1] = c[r][t][2] = c[r][t][3] = 0.0f;
            for (int j = 0; j < 16; j += 4) {
                float4 xa[2], xb[2];
                for (int t = 0; t < NTOK; ++t) {
                    device const float4* xv = (device const float4*)(xp + t * int(a.x_stride) + u * UNIT);
                    xa[t] = xv[j / 4];
                    xb[t] = xv[4 + j / 4];
                }
                for (int r = 0; r < NR; ++r) {
                    const ushort l0 = raw[r][j / 2], l1 = raw[r][j / 2 + 1];
                    const ushort h0 = raw[r][8 + j / 2], h1 = raw[r][8 + j / 2 + 1];
                    const float q0 = float(as_type<char>(uchar(l0 & 255))), q1 = float(as_type<char>(uchar(l0 >> 8)));
                    const float q2 = float(as_type<char>(uchar(l1 & 255))), q3 = float(as_type<char>(uchar(l1 >> 8)));
                    const float g0 = float(as_type<char>(uchar(h0 & 255))), g1 = float(as_type<char>(uchar(h0 >> 8)));
                    const float g2 = float(as_type<char>(uchar(h1 & 255))), g3 = float(as_type<char>(uchar(h1 >> 8)));
                    for (int t = 0; t < NTOK; ++t) {
                        c[r][t][0] = fma(q0, xa[t].x, c[r][t][0]);
                        c[r][t][1] = fma(q1, xa[t].y, c[r][t][1]);
                        c[r][t][2] = fma(g0, xb[t].x, c[r][t][2]);
                        c[r][t][3] = fma(g1, xb[t].y, c[r][t][3]);
                        c[r][t][0] = fma(q2, xa[t].z, c[r][t][0]);
                        c[r][t][1] = fma(q3, xa[t].w, c[r][t][1]);
                        c[r][t][2] = fma(g2, xb[t].z, c[r][t][2]);
                        c[r][t][3] = fma(g3, xb[t].w, c[r][t][3]);
                    }
                }
            }
            for (int r = 0; r < NR; ++r)
                for (int t = 0; t < NTOK; ++t) {
                    const float4 sc = float4(s[r], s[r], 0.0f, 0.0f);
                    const float xsum[2] = {0.0f, 0.0f};
                    acc[r][t] += sc.x * (c[r][t][0] + c[r][t][1]) + sc.y * (c[r][t][2] + c[r][t][3]) -
                                 (sc.z * xsum[0] + sc.w * xsum[1]);
                }
            continue;
        }
        if (STREAM) {
            for (int t = 0; t < NTOK; ++t) {
                device const float4* xv = (device const float4*)(xp + t * int(a.x_stride) + u * UNIT);
                float c[NR][4];
                for (int r = 0; r < NR; ++r) c[r][0] = c[r][1] = c[r][2] = c[r][3] = 0.0f;
                for (int j = 0; j < 16; j += 4) {
                    const float4 xa = xv[j / 4], xb = xv[4 + j / 4];
                    for (int r = 0; r < NR; ++r) {
                        const ushort l0 = raw[r][j / 2], l1 = raw[r][j / 2 + 1];
                        const ushort h0 = raw[r][8 + j / 2], h1 = raw[r][8 + j / 2 + 1];
                        c[r][0] = fma(float(as_type<char>(uchar(l0 & 255))), xa.x, c[r][0]);
                        c[r][1] = fma(float(as_type<char>(uchar(l0 >> 8))), xa.y, c[r][1]);
                        c[r][2] = fma(float(as_type<char>(uchar(h0 & 255))), xb.x, c[r][2]);
                        c[r][3] = fma(float(as_type<char>(uchar(h0 >> 8))), xb.y, c[r][3]);
                        c[r][0] = fma(float(as_type<char>(uchar(l1 & 255))), xa.z, c[r][0]);
                        c[r][1] = fma(float(as_type<char>(uchar(l1 >> 8))), xa.w, c[r][1]);
                        c[r][2] = fma(float(as_type<char>(uchar(h1 & 255))), xb.z, c[r][2]);
                        c[r][3] = fma(float(as_type<char>(uchar(h1 >> 8))), xb.w, c[r][3]);
                    }
                }
                for (int r = 0; r < NR; ++r) {
                    const float4 sc = float4(s[r], s[r], 0.0f, 0.0f);
                    const float xsum[2] = {0.0f, 0.0f};
                    acc[r][t] += sc.x * (c[r][0] + c[r][1]) + sc.y * (c[r][2] + c[r][3]) -
                                 (sc.z * xsum[0] + sc.w * xsum[1]);
                }
            }
            continue;
        }
        float xs[UNIT], xsum[2];
        for (int t = 0; t < NTOK; ++t) {
            X<T_Q8_0>::load(xp + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
            for (int r = 0; r < NR; ++r) {
                float q[UNIT];
                for (int j = 0; j < 16; ++j) {   // deq<T_Q8_0>, verbatim
                    const ushort v = raw[r][j];
                    q[2 * j] = float(as_type<char>(uchar(v & 255)));
                    q[2 * j + 1] = float(as_type<char>(uchar(v >> 8)));
                }
                acc[r][t] += unit_combine<UNIT>(q, float4(s[r], s[r], 0.0f, 0.0f), xs, xsum);
            }
        }
    }
    constexpr int RPT = RS * RG * NR;
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            if (KS == 1) {
                if (li == 0 && row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
            } else if (li == 0) {
                red[(ks * RPT + lrow0 + r) * 8 + t] = v;
            }
        }
    if (KS > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks == 0 && li == 0)
            for (int r = 0; r < NR; ++r)
                for (int t = 0; t < NTOK; ++t) {
                    float v = 0.0f;
                    for (int k = 0; k < KS; ++k) v += red[(k * RPT + lrow0 + r) * 8 + t];
                    if (row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
                }
    }
}

// Every tile strata_hy4_flash_metal::gemv_tile can return for Q8_0 (hy4_flash_kernels.h: the wide, mid and narrow lists and the
// fallback), so DenseGemvF32 never lacks one.
#define HY4_FLASH_GQ8S(L, R, S, K, ST) template [[host_name("hy4_flash_gemv_q8_" #L "_" #R "_" #S "_" #K)]] kernel decltype(hy4_flash_gemv_q8<L, R, S, K, ST>) hy4_flash_gemv_q8<L, R, S, K, ST>;
#define HY4_FLASH_GQ8(L, R, S, K) HY4_FLASH_GQ8S(L, R, S, K, false)
HY4_FLASH_GQ8(32, 2, 8, 2) HY4_FLASH_GQ8S(32, 2, 16, 2, true) HY4_FLASH_GQ8(32, 1, 8, 2) HY4_FLASH_GQ8(32, 1, 4, 1) HY4_FLASH_GQ8S(16, 2, 4, 1, true) HY4_FLASH_GQ8(16, 1, 4, 1)
HY4_FLASH_GQ8(8, 1, 4, 1) HY4_FLASH_GQ8(4, 1, 4, 1) HY4_FLASH_GQ8(2, 1, 4, 1) HY4_FLASH_GQ8(1, 1, 4, 1) HY4_FLASH_GQ8(1, 1, 2, 1)
