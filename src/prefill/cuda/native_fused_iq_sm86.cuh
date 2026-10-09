// CUDA-only SM86 specialization of native_kernel in ../moe_fused_iq.cu.
// Included after that unchanged implementation: reuse its conversion/routing
// helpers and retain its kernel as the fallback. Keep the arithmetic in sync;
// tools/test_cuda_prefill_variant.py compares complete output fingerprints.
#pragma once
#if !defined(__CUDACC__) || defined(__HIPCC__)
#error "The SM86 fused-prefill specialization is CUDA-only"
#endif

namespace strata::prefill::fused {
namespace {

constexpr bool sm86_target(int major, int minor) { return major == 8 && minor == 6; }
static_assert(sm86_target(8, 6) && !sm86_target(8, 0) && !sm86_target(8, 9) && !sm86_target(12, 0));

constexpr size_t sm86_smem(int t, int ww, int stages) {
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) stages * tile_rows(ww) * AB +
           (size_t) tile_rows(ww) * 4 + grid_bytes(t);
}

template <int WT, bool GU, int WW, int AS, int PF>
__global__ void __launch_bounds__(THREADS, 1)
sm86_native_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
                   const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;
    constexpr int NFB = (GU ? 1280 : 2560) / WR;
    constexpr int ACT_LD = NS * AB, BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;
    float* ws = (float*) (smem + 2 * WT_BYTES);
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);
    int* srow = (int*) (stages + AS * ACT_STAGE);
    uint8_t* sgrid = (uint8_t*) (srow + TR);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if ((row0 - tb.off[e]) % TR != 0) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {
            if (GU) return wrow + (s >> 2) * BS;
            return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU ? 2 * (s & 3) + uj : uj; };
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < GU_ROWS_K / 256) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
        };
        if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
            for (size_t o = 128 * (size_t) uj; o < geo.d_row + 127; o += 256) pf_l2(wrow + min(o, geo.d_row - 1));
        }
        __syncthreads();
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % AS) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += THREADS) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[5], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < AS - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[5] = {0, 0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<AS - 2>();
            __syncthreads();
            if (s + AS - 1 < NS) load_act(s + AS - 1);
            cp_commit();
            if (GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % AS) * ACT_STAGE;
                float2 dx[2][2];
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t bq[4];
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if (GU) {
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] = acc[i][n][q];
                    }
                }
        }
    }
#elif defined(__CUDA_ARCH__)
    asm volatile("trap;"); // Host dispatch must use the original kernel on every other SM.
#endif
}

template <int T, bool GU, int WW, int AS, int PF> void sm86_setup(int& occ) {
    ck(cudaFuncSetAttribute(sm86_native_kernel<T, GU, WW, AS, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int) sm86_smem(T, WW, AS)), "SM86 kernel shared memory");
    int blocks = 0;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, sm86_native_kernel<T, GU, WW, AS, PF>,
                                                     THREADS, sm86_smem(T, WW, AS)), "SM86 kernel occupancy");
    if (blocks < 1) { std::fprintf(stderr, "prefill SM86 kernel cannot reside on this device\n"); std::exit(1); }
    occ = std::min(occ, blocks);
}
template <int T, bool GU, int AS, int PF> void sm86_setup_pair(int& occ) {
    sm86_setup<T, GU, 4, AS, PF>(occ);
    sm86_setup<T, GU, 2, AS, PF>(occ);
}
template <int T, bool GU, int AS, int PF>
void sm86_launch(int ww, unsigned grid, const Batch& b, const NativeGeom& g, const Tables& tb,
                  const void* act, const int32_t* src, void* out, float* dm, cudaStream_t stream) {
    if (ww == 4)
        sm86_native_kernel<T, GU, 4, AS, PF><<<grid, THREADS, sm86_smem(T, 4, AS), stream>>>(
            b, g, tb, (const uint8_t*) act, src, (uint8_t*) out, dm);
    else
        sm86_native_kernel<T, GU, 2, AS, PF><<<grid, THREADS, sm86_smem(T, 2, AS), stream>>>(
            b, g, tb, (const uint8_t*) act, src, (uint8_t*) out, dm);
}

} // namespace
} // namespace strata::prefill::fused
