// ink_moe_prefill.cuh: the prefill MoE for sm_80..sm_89 (SF4 and MX8 experts): the same exact FP16 tensor-core arithmetic as
// k_ink_moe_sf4_mx8_fp16tc, with a bigger token tile per expert.
//
// Why: k_ink_moe_sf4_mx8_fp16tc works on 16 tokens of one expert at a time, so at a 2048-token prefill chunk (about 128 tokens per
// expert) every FP4 / FP8 weight tile is decoded to FP16 again for each group of 16 tokens (8 times) and the activation tile of an
// expert is staged from global memory 8 times per weight block.  Here one block takes TR = 8 * NT tokens (64 by default): the weight tile
// is decoded once per 64 tokens, and the m16n8k16 MMAs run NT times per decoded tile.
//
// Arithmetic (so the results do not change): each output element (weight row, token) is accumulated as acc = fma(scale_of_the_16_block,
// mma_m16n8k16(16 weights, 16 activations), acc) in the same k order as the 16-token kernel, and the MMA result of one (row, token)
// pair does not depend on the other tokens in the tile.  The kernel is therefore bit-identical to k_ink_moe_sf4_mx8_fp16tc.
//
// Included from ink_moe_cuda.cuh after the helpers it uses (sf4_decode16_h2, mx8_decode16_h2, e4m3_to_f32, ue8m0_to_f32,
// mma_m16n8k16_f16, situ_f32) and before launch_ink_moe_dispatch_only, which calls it for M >= kInkPrefillTileMinM.
#pragma once

constexpr int kInkPrefillTileMinM = 256;   // below this a chunk has too few tokens per expert for a 64-token tile to pay
constexpr int kInkPrefillTileRows = 64;    // tokens per tile (NT = 8 n-tiles of 8); the bin kernel must be called with tile_rows = this

template <bool GU, int NT>
__global__ void __launch_bounds__(256) k_ink_moe_pf_fp16tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const float* __restrict__ act_f32,        // GU: x_f32 [M, 2304]; DN: h_f32 [P, 1152]
    const float* __restrict__ topk_w,         // DN only
    float* __restrict__ out_buf) {            // GU: h_f32 [P, 1152]; DN: y_pair [P, 2304]
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int kWR    = 128;
    constexpr int TR     = 8 * NT;
    constexpr int K      = GU ? kInkHidden : kInkInter;       // 2304 or 1152
    constexpr int ROWS_N = GU ? (2 * kInkInter) : kInkHidden; // 2304
    constexpr int NFB    = ROWS_N / kWR;                      // 18 blocks of 128 weight rows
    constexpr int N32    = K / 32;                            // 72 (GU) or 36 (DN)

    __shared__ alignas(16) __half s_w[kWR][40];
    __shared__ alignas(16) __half s_a[TR][40];
    __shared__ float              s_wsc[kWR][2];
    __shared__ int32_t            s_row[TR];
    __shared__ float              s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..7: each warp owns 16 weight rows
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int tig  = lane & 3;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        const int fmt = GU ? (int)ed.gu_code : (int)ed.dn_code;
        if (ed.tier > 1 || (fmt != FMT_SF4 && fmt != FMT_MX8)) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;
        const int nnt = (nrows + 7) >> 3;   // n-tiles of 8 tokens that hold a real row (the rest of the tile is skipped)
#ifdef INK_PF_FAULTTEST
        // Test builds only (-DINK_PF_FAULTTEST, never in a release build): the first block of the grid reads one float far outside every
        // allocation, so a compute-sanitizer memcheck that really covers this kernel must report an invalid global read here.  A clean
        // memcheck on the release build only means something if the same filter reports this one on the test build.
        if (blockIdx.x == 0 && tid == 0 && w == 0) {
            volatile const float* bad = act_f32 + ((size_t)1 << 36);
            const float v = *bad;
            if (v == 123456.789f) out_buf[0] = v;
        }
#endif

        const uint8_t* body = sec4_base + ed.body_off;
        float w_outer = 1.0f;
        if (fmt == FMT_SF4) {
            float hdr[4];
            memcpy(hdr, body, 16);
            w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
        }

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
            }
        }
        // rows of the tile past the expert's last token: zero activations (their results are never stored)
        for (int i = tid; i < (TR - nrows) * 16; i += 256) {
            const int r = nrows + (i >> 4);
            reinterpret_cast<uint32_t*>(s_a[r])[i & 15] = 0u;
        }
        __syncthreads();

        float acc[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f; }
        const int wr0 = 16 * warp + g;
        const int wr1 = wr0 + 8;

        auto phys_row_fn = [&](int r) -> int {
            return GU ? (2 * (fb * 64 + 8 * (r >> 4) + (r & 7)) + ((r >> 3) & 1))
                      : (fb * kWR + r);
        };

        // activations of this k32 step -> s_a (every thread helps: nrows * 8 float4 loads)
        auto load_acts = [&](int k32) {
            for (int t = tid; t < nrows * 8; t += 256) {
                const int r = t >> 3;
                const int j = t & 7;
                const float4 fv = *reinterpret_cast<const float4*>(act_f32 + (size_t)s_row[r] * K + k32 * 32 + j * 4);
                __half2* da = reinterpret_cast<__half2*>(s_a[r]);
                da[2 * j + 0] = __floats2half2_rn(fminf(fmaxf(fv.x, -65504.0f), 65504.0f),
                                                  fminf(fmaxf(fv.y, -65504.0f), 65504.0f));
                da[2 * j + 1] = __floats2half2_rn(fminf(fmaxf(fv.z, -65504.0f), 65504.0f),
                                                  fminf(fmaxf(fv.w, -65504.0f), 65504.0f));
            }
        };

        if (fmt == FMT_SF4) {
            constexpr size_t W_P_ROW = K / 2;
            constexpr size_t W_S_ROW = K / 16;
            const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
            const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
            const int pr_t = (tid < kWR) ? phys_row_fn(tid) : 0;
            const uint4* wp16 = reinterpret_cast<const uint4*>(base_p + (size_t)pr_t * W_P_ROW);
            const uint16_t* ws2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)pr_t * W_S_ROW);

            for (int k32 = 0; k32 < N32; ++k32) {
                if (tid < kWR) {
                    const uint4 v = wp16[k32];
                    const uint16_t sp = ws2[k32];
                    s_wsc[tid][0] = e4m3_to_f32((uint8_t)(sp & 0xFFu)) * w_outer;
                    s_wsc[tid][1] = e4m3_to_f32((uint8_t)(sp >> 8))    * w_outer;
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v.x, v.y, w0);
                    sf4_decode16_h2(v.z, v.w, w1);
                    __half2* dw = reinterpret_cast<__half2*>(s_w[tid]);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { dw[j] = w0[j]; dw[8 + j] = w1[j]; }
                }
                load_acts(k32);
                __syncthreads();

#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int ko = sub * 16;
                    const float sc0 = s_wsc[wr0][sub];
                    const float sc1 = s_wsc[wr1][sub];
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 2 * tig);
                    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 2 * tig);
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 8 + 2 * tig);
                    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 8 + 2 * tig);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        if (n < nnt) {
                            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_a[8 * n + g] + ko + 2 * tig);
                            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_a[8 * n + g] + ko + 8 + 2 * tig);
                            float cv[4] = {};
                            mma_m16n8k16_f16(cv, a0, a1, a2, a3, b0, b1);
                            acc[n][0] = fmaf(sc0, cv[0], acc[n][0]);
                            acc[n][1] = fmaf(sc0, cv[1], acc[n][1]);
                            acc[n][2] = fmaf(sc1, cv[2], acc[n][2]);
                            acc[n][3] = fmaf(sc1, cv[3], acc[n][3]);
                        }
                    }
                }
                if (k32 + 1 < N32) __syncthreads();
            }
        } else {
            constexpr size_t W_P_ROW = K;
            constexpr size_t W_S_ROW = K / 32;
            const uint8_t* base_p = GU ? body : (body + ed.dn_off);
            const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
            const int r_w = tid >> 1;
            const int h_w = tid & 1;
            const int pr_t = phys_row_fn(r_w);
            const uint4* wp16 = reinterpret_cast<const uint4*>(base_p + (size_t)pr_t * W_P_ROW);
            const uint8_t* ws = base_s + (size_t)pr_t * W_S_ROW;

            for (int k32 = 0; k32 < N32; ++k32) {
                __half2 wh[8];
                mx8_decode16_h2(wp16[k32 * 2 + h_w], wh);
                __half2* dw = reinterpret_cast<__half2*>(s_w[r_w]) + h_w * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) dw[j] = wh[j];
                if (h_w == 0) s_wsc[r_w][0] = ue8m0_to_f32(ws[k32]);
                load_acts(k32);
                __syncthreads();

                const float sc0 = s_wsc[wr0][0];
                const float sc1 = s_wsc[wr1][0];
                uint32_t a[2][4];
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int ko = sub * 16;
                    a[sub][0] = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 2 * tig);
                    a[sub][1] = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 2 * tig);
                    a[sub][2] = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 8 + 2 * tig);
                    a[sub][3] = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 8 + 2 * tig);
                }
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    if (n < nnt) {
                        float cv[4] = {};
#pragma unroll
                        for (int sub = 0; sub < 2; ++sub) {
                            const int ko = sub * 16;
                            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_a[8 * n + g] + ko + 2 * tig);
                            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_a[8 * n + g] + ko + 8 + 2 * tig);
                            mma_m16n8k16_f16(cv, a[sub][0], a[sub][1], a[sub][2], a[sub][3], b0, b1);
                        }
                        acc[n][0] = fmaf(sc0, cv[0], acc[n][0]);
                        acc[n][1] = fmaf(sc0, cv[1], acc[n][1]);
                        acc[n][2] = fmaf(sc1, cv[2], acc[n][2]);
                        acc[n][3] = fmaf(sc1, cv[3], acc[n][3]);
                    }
                }
                if (k32 + 1 < N32) __syncthreads();
            }
        }

        if constexpr (GU) {
            const int feat = fb * 64 + 8 * warp + g;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if (n < nnt) {
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            out_buf[(size_t)(row0 + r) * kInkInter + feat] = situ_f32(acc[n][c], acc[n][2 + c]);
                        }
                    }
                }
            }
        } else {
            const int rbase = fb * kWR + 16 * warp;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if (n < nnt) {
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            const float sc = s_rw[r];
                            float* dst = out_buf + (size_t)tb.pairs[row0 + r] * kInkHidden;
                            dst[rbase + g + 0] = acc[n][c] * sc;
                            dst[rbase + g + 8] = acc[n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act_f32; (void)topk_w; (void)out_buf;
#endif
}
