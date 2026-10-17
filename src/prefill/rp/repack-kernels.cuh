// Vendored from mx-llama.cpp (github.com/mxxm-t/mx-llama.cpp, MIT) ggml/src/ggml-cuda/q8_repack/repack-kernels.cuh at c4d12e2c;
// only the include paths are changed (Strata compiles against ggml-cuda of llama.cpp 3cf0325).
// Device kernels and helpers for the Q8_0 repacked-weight path (AMD GCN only). MMV
// and GEMM are templated on HAS_IDS so dense and MoE host paths share identical device
// code. All __global__ kernels and __device__ functions use
// #if defined(GGML_USE_HIP) && defined(__gfx906__) bodies; elsewhere NO_DEVICE_CODE stubs.
#pragma once

#include "common.cuh"
#include "mmq.cuh"
#include "unary.cuh"
#include "repack-common.cuh"

struct repack_tile_meta {
    uint32_t expert;
    uint32_t a_base;
};

template <int BN>
static __global__ void repack_tile_off(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_off,
        repack_tile_meta * __restrict__ tile_meta, const int n_expert) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int acc = 0;
    tile_off[0] = 0;
    for (int e = 0; e < n_expert; e++) {
        const int cnt = expert_bounds[e + 1] - expert_bounds[e];
        const int nt  = (cnt + BN - 1) / BN;
        for (int t = 0; t < nt; t++) {
            tile_meta[acc + t].expert = (uint32_t) e;
            tile_meta[acc + t].a_base = (uint32_t) expert_bounds[e] + (uint32_t) t * BN;
        }
        acc += nt;
        tile_off[e + 1] = acc;
    }
#else
    GGML_UNUSED_VARS(expert_bounds, tile_off, tile_meta, n_expert);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ int mmvq_dp4a_u4(
        const uint4 w, const int x0, const int x1, const int x2, const int x3) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    int a = ggml_cuda_dp4a((int) w.x, x0, 0);
    int b = ggml_cuda_dp4a((int) w.z, x2, 0);
    a = ggml_cuda_dp4a((int) w.y, x1, a);
    b = ggml_cuda_dp4a((int) w.w, x3, b);
    return a + b;
#else
    GGML_UNUSED_VARS(w, x0, x1, x2, x3);
    return 0;
#endif
}

// Fused-FFN epilogue shared by the dense and MoE MMV paths: up result + x_bias,
// gate result + gate_bias, then the GLU op.
static __device__ __forceinline__ float rp_mmv_fusion_epilogue(
        const float up, const float gate, const float * x_bias, const float * gate_bias,
        const uint32_t bias_idx, const ggml_glu_op glu_op, const bool use_gate) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    float result = up;
    if (x_bias != nullptr) {
        result += x_bias[bias_idx];
    }
    if (use_gate) {
        float gate_value = gate;
        if (gate_bias != nullptr) {
            gate_value += gate_bias[bias_idx];
        }
        switch (glu_op) {
            case GGML_GLU_OP_SWIGLU:     result *= ggml_cuda_op_silu_single(gate_value); break;
            case GGML_GLU_OP_GEGLU:      result *= ggml_cuda_op_gelu_single(gate_value); break;
            case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(gate_value, result); break;
            default:                     result *= gate_value; break;
        }
    }
    return result;
#else
    GGML_UNUSED_VARS(gate, x_bias, gate_bias, bias_idx, glu_op, use_gate);
    return up;
#endif
}

template <int ROWS, int NWAVES, bool HAS_IDS, int LANES = 64, bool HAS_FUSION = false>
static __global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const uint32_t nchannels_y, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1,
        const uint8_t * __restrict__ wbase_gate, const float * __restrict__ x_bias,
        const float * __restrict__ gate_bias, const ggml_glu_op glu_op) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int NWARPS = (NWAVES * 64) / LANES;
    constexpr int WPR = NWARPS / ROWS;
    static_assert((NWAVES * 64) % (ROWS * LANES) == 0, "NWAVES*64 must be a multiple of ROWS*LANES");
    static_assert(WPR >= 1 && (WPR & (WPR - 1)) == 0, "WPR must be a power of two");

    if constexpr (!HAS_FUSION) {
        GGML_UNUSED_VARS(wbase_gate, x_bias, gate_bias, glu_op);
    }

    [[maybe_unused]] const bool use_gate = wbase_gate != nullptr;

    [[maybe_unused]] uint32_t expert = 0;
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        expert = e;
        wbase += e * expert_stride;
        if constexpr (HAS_FUSION) {
            // The gate matrix is per expert too - without this every assignment
            // reads expert 0's gate weights.
            if (use_gate) {
                wbase_gate += e * expert_stride;
            }
        }
        xq    += (size_t)(a % nchannels_y) * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, nchannels_y, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t qs_str   = repack_qs_stride(ne0);

    const int warp_id   = threadIdx.x / LANES;
    const int lane      = threadIdx.x % LANES;
    const int row_local = warp_id / WPR;
    const int k_slice   = warp_id % WPR;
    const int row       = blockIdx.x * ROWS + row_local;

    const uint32_t wrow     = min((uint32_t) row, ne1 - 1);
    const uint4    * w_row  = reinterpret_cast<const uint4 *>(wbase + (size_t) wrow * qs_str);
    const uint16_t * d_row  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * qs_str) + (size_t) wrow * n_blocks;
    const uint16_t rmask  = (row < (int) ne1) ? (uint16_t) 0xffffu : (uint16_t) 0x0000u;

    [[maybe_unused]] const uint4 * w_row_gate = nullptr;
    [[maybe_unused]] const uint16_t * d_row_gate = nullptr;
    if constexpr (HAS_FUSION) {
        if (use_gate) {
            w_row_gate = reinterpret_cast<const uint4 *>(wbase_gate + (size_t) wrow * qs_str);
            d_row_gate = reinterpret_cast<const uint16_t *>(
                wbase_gate + (size_t) ne1 * qs_str) + (size_t) wrow * n_blocks;
        }
    }

    [[maybe_unused]] uint32_t bias_row = 0;
    if constexpr (HAS_FUSION) {
        if constexpr (HAS_IDS) {
            bias_row = expert * ne1 + (uint32_t) row;
        } else {
            bias_row = (uint32_t) row;
        }
    }

    float acc = 0.0f;
    [[maybe_unused]] float acc_gate = 0.0f;

    if constexpr (LANES < 64) {
        const uint32_t n_half = n_blocks * 2;
        const uint32_t hb_lo  = (n_half * k_slice) / WPR;
        const uint32_t hb_hi  = (n_half * (k_slice + 1)) / WPR;

        const uint32_t hb0 = hb_lo + lane;
        const uint32_t half = hb0 & 1;
        const uint4 * wp = w_row + hb0;
        const block_q8_1 * xp = xq + (hb0 >> 1);
        const uint16_t * dp = d_row + (hb0 >> 1);

        [[maybe_unused]] const uint4 * wpg = nullptr;
        [[maybe_unused]] const uint16_t * dpg = nullptr;
        if constexpr (HAS_FUSION) {
            if (use_gate) {
                wpg = w_row_gate + hb0;
                dpg = d_row_gate + (hb0 >> 1);
            }
        }

        uint32_t iters = 0;
        for (uint32_t hb = hb0; hb + LANES < hb_hi; hb += 2 * LANES) {
            const float dx0 = __low2float(xp->ds);
            const int * xq32_0 = reinterpret_cast<const int *>(xp->qs) + half * 4;
            const int x00 = xq32_0[0], x01 = xq32_0[1], x02 = xq32_0[2], x03 = xq32_0[3];
            const uint4 wv0 = *wp;
            const uint16_t db0 = *dp & rmask;

            const float dx1 = __low2float((xp + LANES / 2)->ds);
            const int * xq32_1 = reinterpret_cast<const int *>((xp + LANES / 2)->qs) + half * 4;
            const int x10 = xq32_1[0], x11 = xq32_1[1], x12 = xq32_1[2], x13 = xq32_1[3];
            const uint4 wv1 = *(wp + LANES);
            const uint16_t db1 = *(dp + LANES / 2) & rmask;

            const float dw0 = __half2float(*reinterpret_cast<const __half *>(&db0));
            int idot0 = 0;
            idot0 = ggml_cuda_dp4a((int) wv0.x, x00, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.y, x01, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.z, x02, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.w, x03, idot0);
            acc += dw0 * dx0 * (float) idot0;

            const float dw1 = __half2float(*reinterpret_cast<const __half *>(&db1));
            int idot1 = 0;
            idot1 = ggml_cuda_dp4a((int) wv1.x, x10, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.y, x11, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.z, x12, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.w, x13, idot1);
            acc += dw1 * dx1 * (float) idot1;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv0 = *wpg;
                    const uint16_t dgb0 = *dpg & rmask;
                    const float dwg0 = __half2float(*reinterpret_cast<const __half *>(&dgb0));
                    acc_gate += dwg0 * dx0 * (float) mmvq_dp4a_u4(wgv0, x00, x01, x02, x03);

                    const uint4 wgv1 = *(wpg + LANES);
                    const uint16_t dgb1 = *(dpg + LANES / 2) & rmask;
                    const float dwg1 = __half2float(*reinterpret_cast<const __half *>(&dgb1));
                    acc_gate += dwg1 * dx1 * (float) mmvq_dp4a_u4(wgv1, x10, x11, x12, x13);
                }
            }

            wp += 2 * LANES;
            xp += LANES;
            dp += LANES;
            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    wpg += 2 * LANES;
                    dpg += LANES;
                }
            }
            iters += 2;
        }
        if (hb0 + iters * LANES < hb_hi) {
            const float dx = __low2float(xp->ds);
            const int * xq32 = reinterpret_cast<const int *>(xp->qs) + half * 4;
            const int x0 = xq32[0], x1 = xq32[1], x2 = xq32[2], x3 = xq32[3];
            const uint4 wv = *wp;
            const uint16_t db = *dp & rmask;
            const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
            int idot = 0;
            idot = ggml_cuda_dp4a((int) wv.x, x0, idot);
            idot = ggml_cuda_dp4a((int) wv.y, x1, idot);
            idot = ggml_cuda_dp4a((int) wv.z, x2, idot);
            idot = ggml_cuda_dp4a((int) wv.w, x3, idot);
            acc += dw * dx * (float) idot;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv = *wpg;
                    const uint16_t dgb = *dpg & rmask;
                    const float dwg = __half2float(*reinterpret_cast<const __half *>(&dgb));
                    acc_gate += dwg * dx * (float) mmvq_dp4a_u4(wgv, x0, x1, x2, x3);
                }
            }
        }
    } else {
        const uint32_t n_half = n_blocks * 2;
        const uint32_t hb_lo  = (n_half * k_slice) / WPR;
        const uint32_t hb_hi  = (n_half * (k_slice + 1)) / WPR;

        uint32_t base = hb_lo;
        for (; base + 2u * (uint32_t) LANES <= hb_hi; base += 2u * (uint32_t) LANES) {
            const uint32_t hb    = base + (uint32_t) lane;
            const uint32_t sb0   = hb >> 1;
            const uint32_t half0 = hb & 1;
            const uint32_t hb1   = hb + LANES;
            const uint32_t sb1   = hb1 >> 1;
            const uint32_t half1 = hb1 & 1;

            const block_q8_1 * xb0 = xq + sb0;
            const float dx0 = __low2float(xb0->ds);
            const int * xq32_0 = reinterpret_cast<const int *>(xb0->qs) + half0 * 4;
            const int x00 = xq32_0[0], x01 = xq32_0[1], x02 = xq32_0[2], x03 = xq32_0[3];
            const uint4 wv0 = w_row[hb];
            const uint16_t db0 = d_row[sb0] & rmask;

            const block_q8_1 * xb1 = xq + sb1;
            const float dx1 = __low2float(xb1->ds);
            const int * xq32_1 = reinterpret_cast<const int *>(xb1->qs) + half1 * 4;
            const int x10 = xq32_1[0], x11 = xq32_1[1], x12 = xq32_1[2], x13 = xq32_1[3];
            const uint4 wv1 = w_row[hb1];
            const uint16_t db1 = d_row[sb1] & rmask;

            const float dw0 = __half2float(*reinterpret_cast<const __half *>(&db0));
            const int idot0 = mmvq_dp4a_u4(wv0, x00, x01, x02, x03);
            acc += dw0 * dx0 * (float) idot0;

            const float dw1 = __half2float(*reinterpret_cast<const __half *>(&db1));
            const int idot1 = mmvq_dp4a_u4(wv1, x10, x11, x12, x13);
            acc += dw1 * dx1 * (float) idot1;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv0 = w_row_gate[hb];
                    const uint16_t dgb0 = d_row_gate[sb0] & rmask;
                    const float dwg0 = __half2float(*reinterpret_cast<const __half *>(&dgb0));
                    acc_gate += dwg0 * dx0 * (float) mmvq_dp4a_u4(wgv0, x00, x01, x02, x03);

                    const uint4 wgv1 = w_row_gate[hb1];
                    const uint16_t dgb1 = d_row_gate[sb1] & rmask;
                    const float dwg1 = __half2float(*reinterpret_cast<const __half *>(&dgb1));
                    acc_gate += dwg1 * dx1 * (float) mmvq_dp4a_u4(wgv1, x10, x11, x12, x13);
                }
            }
        }

#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const uint32_t hb = base + (uint32_t) (part * LANES + lane);
            if (hb < hb_hi) {
                const uint32_t sb   = hb >> 1;
                const uint32_t half = hb & 1;
                const block_q8_1 * xb = xq + sb;
                const float dx = __low2float(xb->ds);
                const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;
                const int x0 = xq32[0], x1 = xq32[1], x2 = xq32[2], x3 = xq32[3];
                const uint4 wv = w_row[hb];
                const uint16_t db = d_row[sb] & rmask;
                const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
                const int idot = mmvq_dp4a_u4(wv, x0, x1, x2, x3);
                acc += dw * dx * (float) idot;

                if constexpr (HAS_FUSION) {
                    if (use_gate) {
                        const uint4 wgv = w_row_gate[hb];
                        const uint16_t dgb = d_row_gate[sb] & rmask;
                        const float dwg = __half2float(*reinterpret_cast<const __half *>(&dgb));
                        acc_gate += dwg * dx * (float) mmvq_dp4a_u4(wgv, x0, x1, x2, x3);
                    }
                }
            }
        }
    }

    if constexpr (WPR > 1) {
        __shared__ float s_part[ROWS][WPR];
        [[maybe_unused]] __shared__ float s_part_gate[ROWS][WPR];
        const float wsum = rp_warp_reduce_sum<LANES>(acc);
        if (lane == 0) {
            s_part[row_local][k_slice] = wsum;
        }
        if constexpr (HAS_FUSION) {
            const float wsum_gate = rp_warp_reduce_sum<LANES>(acc_gate);
            if (lane == 0) {
                s_part_gate[row_local][k_slice] = wsum_gate;
            }
        }
        __syncthreads();
        if (k_slice == 0 && lane == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < WPR; w++) {
                sum += s_part[row_local][w];
            }
            if (row < (int) ne1) {
                if constexpr (HAS_FUSION) {
                    float gate_sum = 0.0f;
#pragma unroll
                    for (int w = 0; w < WPR; w++) {
                        gate_sum += s_part_gate[row_local][w];
                    }
                    y[row] = rp_mmv_fusion_epilogue(sum, gate_sum, x_bias, gate_bias, bias_row, glu_op, use_gate);
                } else {
                    y[row] = sum;
                }
            }
        }
    } else {
        const float a = rp_warp_reduce_sum<LANES>(acc);
        if constexpr (HAS_FUSION) {
            const float g = rp_warp_reduce_sum<LANES>(acc_gate);
            if (lane == 0 && row < (int) ne1) {
                y[row] = rp_mmv_fusion_epilogue(a, g, x_bias, gate_bias, bias_row, glu_op, use_gate);
            }
        } else {
            if (lane == 0 && row < (int) ne1) {
                y[row] = a;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, nchannels_y, expert_stride, xs_id, dst_s1, wbase_gate, x_bias, gate_bias, glu_op);
    NO_DEVICE_CODE;
#endif
}

// Generic repacked mat-vec: any type with 32-value sub-blocks, fetched through
// rp_traits<WT>::load_w. Simple LANES-strided sub-block loop: both production
// instantiations of the Q8_0 mat-vec run WPR == 1 (dense <16,16,false,64>, MoE
// <64,16,true,16>), so this kernel requires WPR == 1 and skips the k-slice
// machinery. Q8_0 keeps its tuned bespoke kernel; every other type gets this
// one for free, including the fused up/gate variant.
template <ggml_type WT, int ROWS, int NWAVES, bool HAS_IDS, int LANES = 64,
          bool HAS_FUSION = false>
static __global__ void __launch_bounds__(NWAVES * 64) mul_mat_vec_rp(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1,
        const uint32_t nchannels_y, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1,
        const uint8_t * __restrict__ wbase_gate = nullptr,
        const float * __restrict__ x_bias = nullptr,
        const float * __restrict__ gate_bias = nullptr,
        const ggml_glu_op glu_op = GGML_GLU_OP_REGLU) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int NWARPS = (NWAVES * 64) / LANES;
    constexpr int WPR    = NWARPS / ROWS;
    static_assert(WPR == 1, "generic repack mat-vec assumes one lane-group per row");

    if constexpr (!HAS_FUSION) {
        GGML_UNUSED_VARS(wbase_gate, x_bias, gate_bias, glu_op);
    }
    [[maybe_unused]] const bool use_gate = HAS_FUSION && wbase_gate != nullptr;

    [[maybe_unused]] uint32_t expert = 0;
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        expert = e;
        wbase += e * expert_stride;
        if constexpr (HAS_FUSION) {
            if (use_gate) {
                wbase_gate += e * expert_stride;
            }
        }
        xq    += (size_t)(a % nchannels_y) * xs_id;
        y     += (size_t) a * dst_s1;
    } else {
        GGML_UNUSED_VARS(ids_src1, nchannels_y, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_sub = ne0 >> 5;
    const typename rp_traits<WT>::geom wg(ne0, ne1);

    const int warp_id = threadIdx.x / LANES;
    const int lane    = threadIdx.x % LANES;
    const int row     = blockIdx.x * ROWS + warp_id;

    // Clamped rows read a valid row's data and discard the result at the write.
    const uint32_t wrow = min((uint32_t) row, ne1 - 1);

    float acc = 0.0f;
    [[maybe_unused]] float acc_gate = 0.0f;
    for (uint32_t sb = (uint32_t) lane; sb < n_sub; sb += LANES) {
        uint4 lo, hi;
        typename rp_traits<WT>::slot_t ds;
        rp_traits<WT>::load_w(wbase, wg, wrow, sb, lo, hi, ds);
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * x32 = reinterpret_cast<const int *>(xb->qs);
        if constexpr (rp_traits<WT>::split_scales) {
            const int ilo = mmvq_dp4a_u4(lo, x32[0], x32[1], x32[2], x32[3]);
            const int ihi = mmvq_dp4a_u4(hi, x32[4], x32[5], x32[6], x32[7]);
            const float2 dd = rp_traits<WT>::scale2(ds);
            acc += dd.x * dx * (float) ilo + dd.y * dx * (float) ihi;
        } else if constexpr (rp_traits<WT>::affine) {
            // Affine (Q4_K): d*sc times the dot, minus dmin*m times the stored activation sum (block_q8_1.ds.y).
            int idot = mmvq_dp4a_u4(lo, x32[0], x32[1], x32[2], x32[3]);
            idot    += mmvq_dp4a_u4(hi, x32[4], x32[5], x32[6], x32[7]);
            const float2 dd = rp_traits<WT>::scale2(ds);
            acc += dd.x * dx * (float) idot - dd.y * __high2float(xb->ds);
        } else {
            int idot = mmvq_dp4a_u4(lo, x32[0], x32[1], x32[2], x32[3]);
            idot    += mmvq_dp4a_u4(hi, x32[4], x32[5], x32[6], x32[7]);
            acc += rp_traits<WT>::scale(ds) * dx * (float) idot;
        }
        if constexpr (HAS_FUSION) {
            if (use_gate) {
                uint4 glo, ghi;
                typename rp_traits<WT>::slot_t gds;
                rp_traits<WT>::load_w(wbase_gate, wg, wrow, sb, glo, ghi, gds);
                if constexpr (rp_traits<WT>::split_scales) {
                    const int glo_d = mmvq_dp4a_u4(glo, x32[0], x32[1], x32[2], x32[3]);
                    const int ghi_d = mmvq_dp4a_u4(ghi, x32[4], x32[5], x32[6], x32[7]);
                    const float2 gd = rp_traits<WT>::scale2(gds);
                    acc_gate += gd.x * dx * (float) glo_d + gd.y * dx * (float) ghi_d;
                } else if constexpr (rp_traits<WT>::affine) {
                    int gdot = mmvq_dp4a_u4(glo, x32[0], x32[1], x32[2], x32[3]);
                    gdot    += mmvq_dp4a_u4(ghi, x32[4], x32[5], x32[6], x32[7]);
                    const float2 gd = rp_traits<WT>::scale2(gds);
                    acc_gate += gd.x * dx * (float) gdot - gd.y * __high2float(xb->ds);
                } else {
                    int gdot = mmvq_dp4a_u4(glo, x32[0], x32[1], x32[2], x32[3]);
                    gdot    += mmvq_dp4a_u4(ghi, x32[4], x32[5], x32[6], x32[7]);
                    acc_gate += rp_traits<WT>::scale(gds) * dx * (float) gdot;
                }
            }
        }
    }
    const float a = rp_warp_reduce_sum<LANES>(acc);
    [[maybe_unused]] float g = 0.0f;
    if constexpr (HAS_FUSION) {
        g = rp_warp_reduce_sum<LANES>(acc_gate);
    }
    if (lane == 0 && row < (int) ne1) {
        float out = a;
        if constexpr (HAS_FUSION) {
            uint32_t bias_row = (uint32_t) row;
            if constexpr (HAS_IDS) {
                bias_row = expert * ne1 + (uint32_t) row;
            }
            out = rp_mmv_fusion_epilogue(a, g, x_bias, gate_bias,
                bias_row, glu_op, use_gate);
        }
        y[row] = out;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, nchannels_y, expert_stride,
                     xs_id, dst_s1, wbase_gate, x_bias, gate_bias, glu_op);
    NO_DEVICE_CODE;
#endif
}

template <ggml_type WT, bool HAS_IDS, int CW, int TN_, int NRL>
static __device__ void mmq_gemm_repacked_impl(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const int t  = threadIdx.x + threadIdx.y * blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const uint32_t row0 = blockIdx.x * MMQ_RP_Q8_BM;
    constexpr uint32_t BN = CW * TN_;
    uint32_t tok0 = blockIdx.y * BN;
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const repack_tile_meta meta = tile_meta[blockIdx.y];
        const uint32_t e = meta.expert;
        a_base = meta.a_base;
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    // All layout knowledge lives in rp_traits<WT>: geometry constants built
    // once, sub-block fetches through load_w. The LDS scale array stays
    // the trait's slot_t - traits::scale() decodes the raw slot.
    const typename rp_traits<WT>::geom wg(ne0, ne1);
    const block_q8_1_mmq_h * __restrict__ xmmq = reinterpret_cast<const block_q8_1_mmq_h *>(xq);

    const block_q8_1_mmq_h * x_group = xmmq;

    // Raw-register types (MXFP4) prefetch one uint4 of packed nibbles per
    // sub-block - half the live registers of the expanded pair, which spilled
    // 16-22 VGPRs per lane - and expand once per element at the LDS store.
    // LDS stays expanded: expanding at consume instead multiplied the expand
    // by the row*column reuse and measured -40% prefill.
    constexpr bool RAW_REG = rp_traits<WT>::raw_lds;
    __shared__ uint4    sW_lo[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    __shared__ uint4    sW_hi[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    using slot_t = typename rp_traits<WT>::slot_t;
    static_assert(16 % sizeof(slot_t) == 0, "slot must divide a uint4 fetch");
    constexpr int SPW = (int) (16 / sizeof(slot_t));   // slots per uint4 LDS fetch
    __shared__ slot_t   sWdh[MMQ_RP_Q8_BK][MMQ_RP_Q8_BM];
    __shared__ sXq_row_q8 sXq[BN];
    __shared__ float    sXd[BN][MMQ_RP_Q8_BK + 1];
    // Activation sums, affine types only (the [1] alternative keeps the LDS footprint of every other type unchanged).
    __shared__ float    sXs[rp_traits<WT>::affine ? BN : 1][MMQ_RP_Q8_BK + 1];
    __shared__ uint32_t sCol[BN];

    constexpr int NROW = MMQ_RP_Q8_BM / NRL;
    float acc[NROW][TN_] = {};

    constexpr int NTHREADS = CW * NRL;
    constexpr int W_ELM = MMQ_RP_Q8_BM * MMQ_RP_Q8_BK;
    constexpr int X_ELM = BN * MMQ_RP_Q8_BK;
    constexpr int W_EPT = (W_ELM + NTHREADS - 1) / NTHREADS;
    constexpr int X_EPT = (X_ELM + NTHREADS - 1) / NTHREADS;

    const int sxc0 = sX_swizzle<CW>(tx);

    uint4    pw_lo[W_EPT];
    uint4    pw_hi[W_EPT];
    slot_t   pw_d [W_EPT];
    rp_x_sub px   [X_EPT];
    [[maybe_unused]] float px_s[X_EPT];
    if constexpr (HAS_IDS) {
        for (int i = t; i < (int) BN; i += NTHREADS) {
            const uint32_t a = a_base + i;
            sCol[i] = (a < a_end) ? (uint32_t) ids_src1[a] : 0u;
        }
        __syncthreads();
    }
    // Whole BM x BN x BK tile in-bounds -> use the unchecked loads below.
    const bool full_m = row0 + MMQ_RP_Q8_BM <= ne1;
    bool full_n;
    if constexpr (HAS_IDS) {
        full_n = a_base + BN <= a_end;
    } else {
        full_n = tok0 + BN <= n_tok;
    }
    const uint32_t n_main = n_sub & ~(MMQ_RP_Q8_BK - 1u);
    const bool full_k = n_main == n_sub;
    const bool full_tile = full_m && full_n && full_k &&
                           (NTHREADS <= X_ELM) && (NTHREADS <= W_ELM);


    auto prefetch_w_checked = [&](const uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) {
                    break;
                }
            }
            const int      lr   = e / MMQ_RP_Q8_BK;
            const int      lk   = e % MMQ_RP_Q8_BK;
            const uint32_t sb   = sb0 + lk;
            const uint32_t wrow = row0 + lr;
            if (wrow < ne1 && sb < n_sub) {
                if constexpr (RAW_REG) {
                    rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, pw_lo[i], pw_d[i]);
                } else {
                    rp_traits<WT>::load_w(wbase, wg, wrow, sb, pw_lo[i], pw_hi[i], pw_d[i]);
                }
            } else {
                // Zero the payload too, not just the scale. Q8_0 tolerates garbage
                // here because d==0 exactly, but MXFP4's e==0 decodes to 2^-127,
                // not 0, so uninitialised registers would survive into the sum.
                pw_lo[i] = make_uint4(0, 0, 0, 0);
                pw_hi[i] = make_uint4(0, 0, 0, 0);
                pw_d [i] = {};
            }
        }
    };

    auto prefetch_x_checked = [&](const uint32_t sb0, const block_q8_1_mmq_h * x_grp) {
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) {
                    break;
                }
            }
            const int      lr = e / MMQ_RP_Q8_BK;
            const int      lk = e % MMQ_RP_Q8_BK;
            const uint32_t sb = sb0 + lk;
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? sCol[lr] : 0;
            }
            if (xval && sb < n_sub) {
                if constexpr (rp_traits<WT>::affine) {
                    px[i] = rp_x_sub_from_mmq_group_ds(x_grp, xcol, lk, px_s[i]);
                } else {
                    px[i] = rp_x_sub_from_mmq_group(x_grp, xcol, lk);
                }
            } else {
                px[i].d = 0.0f;
                if constexpr (rp_traits<WT>::affine) {
                    // d == 0 does not kill the affine sum term, zero it too
                    px_s[i] = 0.0f;
                }
            }
        }
    };

    auto prefetch_w_full = [&](const uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) break;
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;

            const uint32_t wrow = row0 + lr;
            const uint32_t sb   = sb0 + lk;
            if constexpr (RAW_REG) {
                rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, pw_lo[i], pw_d[i]);
            } else {
                rp_traits<WT>::load_w(wbase, wg, wrow, sb, pw_lo[i], pw_hi[i], pw_d[i]);
            }
        }
    };

    auto prefetch_x_full = [&](const uint32_t sb0, const block_q8_1_mmq_h * x_grp) {
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) break;
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;

            uint32_t xcol = tok0 + lr;
            if constexpr (HAS_IDS) {
                xcol = sCol[lr];
            }

            if constexpr (rp_traits<WT>::affine) {
                px[i] = rp_x_sub_from_mmq_group_ds(x_grp, xcol, lk, px_s[i]);
            } else {
                px[i] = rp_x_sub_from_mmq_group(x_grp, xcol, lk);
            }
        }
    };

    auto store_stage = [&]() {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) {
                    break;
                }
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;
            if constexpr (RAW_REG) {
                uint4 lo, hi;
                rp_traits<WT>::expand_raw(pw_lo[i], lo, hi);
                sW_lo[lr][lk] = lo;
                sW_hi[lr][lk] = hi;
            } else {
                sW_lo[lr][lk] = pw_lo[i];
                sW_hi[lr][lk] = pw_hi[i];
            }
            sWdh [lk][lr] = pw_d[i];
        }
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) {
                    break;
                }
            }
            const int lr  = e / MMQ_RP_Q8_BK;
            const int lk  = e % MMQ_RP_Q8_BK;
            const int sXr = sX_swizzle<CW>(lr);

            sXq[sXr].q[lk][0] = px[i].q0;
            sXq[sXr].q[lk][1] = px[i].q1;
            sXd[sXr][lk]    = px[i].d;
            if constexpr (rp_traits<WT>::affine) {
                sXs[sXr][lk] = px_s[i];
            }
        }
    };

    auto compute_stage = [&]() {
        for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
            float dx  [TN_];
            [[maybe_unused]] float sx[TN_];
            int   xq32[TN_][8];
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const int xcol = sxc0 + n * CW;
                const uint4 q0 = sXq[xcol].q[kk][0];
                const uint4 q1 = sXq[xcol].q[kk][1];
                dx[n] = sXd[xcol][kk];
                if constexpr (rp_traits<WT>::affine) {
                    sx[n] = sXs[xcol][kk];
                }
                xq32[n][0] = (int) q0.x;
                xq32[n][1] = (int) q0.y;
                xq32[n][2] = (int) q0.z;
                xq32[n][3] = (int) q0.w;
                xq32[n][4] = (int) q1.x;
                xq32[n][5] = (int) q1.y;
                xq32[n][6] = (int) q1.z;
                xq32[n][7] = (int) q1.w;
            }
            const int row_base = ty * NROW;
            uint4 wlo = sW_lo[row_base][kk];
            uint4 whi = sW_hi[row_base][kk];
            const uint4 * dp16 = reinterpret_cast<const uint4 *>(&sWdh[kk][row_base]);

#pragma unroll
            for (int g = 0; g < NROW; g += SPW) {
                const uint4 dh4 = dp16[g / SPW];
                const slot_t * dh = reinterpret_cast<const slot_t *>(&dh4);

#pragma unroll
                for (int rr = 0; rr < SPW; rr++) {
                    const int r = g + rr;

                    const uint4 wlo_c = wlo, whi_c = whi;
                    if (r + 1 < NROW) {
                        wlo = sW_lo[row_base + r + 1][kk];
                        whi = sW_hi[row_base + r + 1][kk];
                    }

                    [[maybe_unused]] float d = 0.0f, dlo = 0.0f, dhi = 0.0f;
                    if constexpr (rp_traits<WT>::split_scales || rp_traits<WT>::affine) {
                        const float2 dd = rp_traits<WT>::scale2(dh[rr]);
                        dlo = dd.x;
                        dhi = dd.y;
                    } else {
                        d = rp_traits<WT>::scale(dh[rr]);
                    }

#pragma unroll
                    for (int n = 0; n < TN_; n++) {
                        if constexpr (WT == GGML_TYPE_MXFP4 || rp_traits<WT>::split_scales) {
                            // Stock MXFP4 uses MMQ_DP4A_TXS_Q8_1, i.e. it applies the
                            // scale per 16-value HALF block, while Q8_0 uses TXS_Q8_0
                            // and scales per 32. Split the accumulator the same way -
                            // d*(lo) + d*(hi) instead of d*(lo+hi) - or the result is
                            // mathematically equal but not bit-identical to stock.
                            int lo = 0, hi = 0;
                            lo = ggml_cuda_dp4a((int) wlo_c.x, xq32[n][0], lo);
                            lo = ggml_cuda_dp4a((int) wlo_c.y, xq32[n][1], lo);
                            lo = ggml_cuda_dp4a((int) wlo_c.z, xq32[n][2], lo);
                            lo = ggml_cuda_dp4a((int) wlo_c.w, xq32[n][3], lo);
                            hi = ggml_cuda_dp4a((int) whi_c.x, xq32[n][4], hi);
                            hi = ggml_cuda_dp4a((int) whi_c.y, xq32[n][5], hi);
                            hi = ggml_cuda_dp4a((int) whi_c.z, xq32[n][6], hi);
                            hi = ggml_cuda_dp4a((int) whi_c.w, xq32[n][7], hi);
                            if constexpr (rp_traits<WT>::split_scales) {
                                // Per-16-value scales (Q6_K): each half gets its own.
                                acc[r][n] += dlo * dx[n] * (float) lo + dhi * dx[n] * (float) hi;
                            } else {
                                acc[r][n] += d * dx[n] * (float) lo + d * dx[n] * (float) hi;
                            }
                        } else {
                        int idot = 0;
                        idot = ggml_cuda_dp4a((int) wlo_c.x, xq32[n][0], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.x, xq32[n][4], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.y, xq32[n][1], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.y, xq32[n][5], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.z, xq32[n][2], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.z, xq32[n][6], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.w, xq32[n][3], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.w, xq32[n][7], idot);
                        if constexpr (rp_traits<WT>::affine) {
                            // Affine (Q4_K): d*sc times the dot, minus dmin*m times the sum.
                            acc[r][n] += dlo * dx[n] * (float) idot - dhi * sx[n];
                        } else {
                            acc[r][n] += d * dx[n] * (float) idot;
                        }
                        }
                    }
                }
            }
        }
    };

    // Pipelined K loop: store current slab -> sync -> prefetch next slab -> compute -> sync.
    if (full_tile) {
        prefetch_w_full(0);
        prefetch_x_full(0, x_group);

        for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_Q8_BK) {
            store_stage();
            __syncthreads();

            if (sb0 + MMQ_RP_Q8_BK < n_sub) {
                x_group += n_tok;
                prefetch_w_full(sb0 + MMQ_RP_Q8_BK);
                prefetch_x_full(sb0 + MMQ_RP_Q8_BK, x_group);
            }

            compute_stage();
            __syncthreads();
        }
    } else {
        prefetch_w_checked(0);
        prefetch_x_checked(0, x_group);

        for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_Q8_BK) {
            store_stage();
            __syncthreads();

            if (sb0 + MMQ_RP_Q8_BK < n_sub) {
                x_group += n_tok;
                prefetch_w_checked(sb0 + MMQ_RP_Q8_BK);
                prefetch_x_checked(sb0 + MMQ_RP_Q8_BK, x_group);
            }

            compute_stage();
            __syncthreads();
        }
    }

    // Epilogue: write accumulators to y (float4 when the row stride is aligned;
    // MoE scatters each column through ids_dst to its destination token).
    const uint32_t rbase  = row0 + ty * NROW;
    const int      nvalid = min((int) ne1 - (int) rbase, NROW);
    const bool     vec_ok = (dst_s1 & 3) == 0;
    if (nvalid > 0) {
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            const uint32_t col = tx + n * CW;
            float * dst;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + col;
                if (!(a < a_end && col < BN)) {
                    continue;
                }
                dst = y + (size_t) ids_dst[a] * dst_s1 + rbase;
            } else {
                const uint32_t tok = tok0 + col;
                if (tok >= n_tok) {
                    continue;
                }
                dst = y + (size_t) tok * dst_s1 + rbase;
            }
            static_assert(NROW % 4 == 0, "float4 epilogue needs NROW % 4 == 0");
#pragma unroll
            for (int g = 0; g < NROW; g += 4) {
                if (vec_ok && g + 4 <= nvalid) {
                    const float4 v = { acc[g + 0][n], acc[g + 1][n], acc[g + 2][n], acc[g + 3][n] };
                    *reinterpret_cast<float4 *>(dst + g) = v;
                } else {
#pragma unroll
                    for (int r = g; r < min(g + 4, nvalid); r++) {
                        dst[r] = acc[r][n];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// Launch wrapper: 64-wide waves, tuned for large batches (ne11 >= 128).
template <bool HAS_IDS, int TN_, int NRL, ggml_type WT = GGML_TYPE_Q8_0>
static __global__ void __launch_bounds__(64 * NRL, 2) mmq_gemm_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    mmq_gemm_repacked_impl<WT, HAS_IDS, 64, TN_, NRL>(
        wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst,
        expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// Launch wrapper: 32-wide waves, tuned for small batches.
// Narrow-batch mat-vec: NCOLS tokens per weight pass. A 2-8 token batch (a
// speculative verify step, a tiny ubatch) is too narrow for the tiled GEMM,
// which computes a full 32-wide tile whatever the batch is - measured 274 us
// against 140 us for the canonical mul_mat_vec_q at 4 tokens. Reading the
// weights once and keeping one accumulator per token amortizes the weight
// traffic the same way, on the repacked layout. Dense only: the single-token
// path and the MoE ids path keep their existing kernels.
template <int ROWS, int NWAVES, int NCOLS, int RPL = 1, bool HAS_FUSION = false,
          int LANES = 64, ggml_type WT = GGML_TYPE_Q8_0>
static __device__ __forceinline__ void mul_mat_vec_repacked_nc_impl(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t xs, const uint32_t ys,
        const uint8_t * __restrict__ wbase_gate, const float * __restrict__ x_bias,
        const float * __restrict__ gate_bias, const ggml_glu_op glu_op,
        const uint32_t bias_row_base) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    // LANES is the number of lanes cooperating on ONE row. It is the geometry
    // knob that matters here: the strided loop runs n_blocks/LANES iterations
    // and is then followed by a log2(LANES)-step DPP reduction, so a wide lane
    // group on a short row spends most of its time reducing. At ne0=5120,
    // n_blocks is 160, which is 2.5 iterations at 64 lanes against a 6-step
    // reduction, and 10 iterations at 16 lanes against a 4-step one.
    constexpr int NWARPS = (NWAVES * 64) / LANES;
    static_assert(LANES >= 1 && LANES <= 64 && (LANES & (LANES - 1)) == 0,
                  "LANES must be a power of two, at most one wave");
    static_assert(ROWS == NWARPS * RPL, "row geometry");
    const int warp_id = threadIdx.x / LANES;
    const int lane    = threadIdx.x % LANES;
    const int row     = blockIdx.x * ROWS + warp_id * RPL;

    const uint32_t n_blocks = ne0 >> 5;
    const typename rp_traits<WT>::geom wg(ne0, ne1);

    // RPL rows per lane: the activation block is loaded once into registers and
    // reused against every row the lane owns, so the activation side stops
    // being re-read per row. This is what the canonical mat-vec does with its
    // register blocking, and at NCOLS=4 the activations are otherwise 4x the
    // weight bytes per step.
    uint32_t wrows [RPL];
    bool rvalid[RPL];
#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        const int rr = row + r;
        wrows [r] = min((uint32_t) rr, ne1 - 1);
        // Masking the raw scale slot zeroes dead-row terms for f16; for e8m0 a
        // zero slot decodes to a tiny nonzero, which is finite and discarded at
        // the guarded write below either way.
        rvalid[r] = rr < (int) ne1;
    }

    // Fused FFN: the gate matrix is read in the same pass as up, so the
    // activation blocks, the row bookkeeping and the launch are all shared.
    // Widths 2-8 otherwise run up, gate and the GLU as three launches where
    // the canonical path runs one.
    constexpr int GR = HAS_FUSION ? RPL   : 1;
    constexpr int GC = HAS_FUSION ? NCOLS : 1;

    float acc[RPL][NCOLS] = {};
    [[maybe_unused]] float accg[GR][GC] = {};

    // Whole 32-quant blocks per lane step: one weight-scale read and one
    // activation block per column per 32 quants. The half-block stride paid
    // both twice, and at sharded row counts the loop is a handful of
    // iterations, so per-step overhead is most of the kernel.
    for (uint32_t sb = (uint32_t) lane; sb < n_blocks; sb += (uint32_t) LANES) {
        // Weights for the lane's rows stay in registers; the activation block is
        // re-read per row from L1, which is cheaper than the register pressure
        // of holding NCOLS x 8 ints alongside the accumulators.
        uint4 wv0[RPL], wv1[RPL];
        [[maybe_unused]] float  dw [RPL];
        [[maybe_unused]] float2 dw2[RPL];
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            typename rp_traits<WT>::slot_t db;
            rp_traits<WT>::load_w(wbase, wg, wrows[r], sb, wv0[r], wv1[r], db);
            if constexpr (rp_traits<WT>::split_scales || rp_traits<WT>::affine) {
                dw2[r] = rp_traits<WT>::scale2(rp_slot_keep(db, rvalid[r]));
            } else {
                dw[r] = rp_traits<WT>::scale(rp_slot_keep(db, rvalid[r]));
            }
        }
        [[maybe_unused]] uint4 gv0[GR], gv1[GR];
        [[maybe_unused]] float  dg [GR];
        [[maybe_unused]] float2 dg2[GR];
        if constexpr (HAS_FUSION) {
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                typename rp_traits<WT>::slot_t gb;
                rp_traits<WT>::load_w(wbase_gate, wg, wrows[r], sb, gv0[r], gv1[r], gb);
                if constexpr (rp_traits<WT>::split_scales || rp_traits<WT>::affine) {
                    dg2[r] = rp_traits<WT>::scale2(rp_slot_keep(gb, rvalid[r]));
                } else {
                    dg[r] = rp_traits<WT>::scale(rp_slot_keep(gb, rvalid[r]));
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const block_q8_1 * xb = xq + (size_t) c * xs + sb;
            const float dx = __low2float(xb->ds);
            [[maybe_unused]] const float sx = __high2float(xb->ds);
            const int * x = reinterpret_cast<const int *>(xb->qs);
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                int ilo = 0, ihi = 0;
                ilo = ggml_cuda_dp4a((int) wv0[r].x, x[0], ilo);
                ilo = ggml_cuda_dp4a((int) wv0[r].y, x[1], ilo);
                ilo = ggml_cuda_dp4a((int) wv0[r].z, x[2], ilo);
                ilo = ggml_cuda_dp4a((int) wv0[r].w, x[3], ilo);
                ihi = ggml_cuda_dp4a((int) wv1[r].x, x[4], ihi);
                ihi = ggml_cuda_dp4a((int) wv1[r].y, x[5], ihi);
                ihi = ggml_cuda_dp4a((int) wv1[r].z, x[6], ihi);
                ihi = ggml_cuda_dp4a((int) wv1[r].w, x[7], ihi);
                if constexpr (rp_traits<WT>::split_scales) {
                    acc[r][c] += dw2[r].x * dx * (float) ilo + dw2[r].y * dx * (float) ihi;
                } else if constexpr (rp_traits<WT>::affine) {
                    acc[r][c] += dw2[r].x * dx * (float) (ilo + ihi) - dw2[r].y * sx;
                } else {
                    // one add of the two halves keeps the Q8_0 result bit-identical
                    acc[r][c] += dw[r] * dx * (float) (ilo + ihi);
                }
                if constexpr (HAS_FUSION) {
                    int glo_d = 0, ghi_d = 0;
                    glo_d = ggml_cuda_dp4a((int) gv0[r].x, x[0], glo_d);
                    glo_d = ggml_cuda_dp4a((int) gv0[r].y, x[1], glo_d);
                    glo_d = ggml_cuda_dp4a((int) gv0[r].z, x[2], glo_d);
                    glo_d = ggml_cuda_dp4a((int) gv0[r].w, x[3], glo_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].x, x[4], ghi_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].y, x[5], ghi_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].z, x[6], ghi_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].w, x[7], ghi_d);
                    if constexpr (rp_traits<WT>::split_scales) {
                        accg[r][c] += dg2[r].x * dx * (float) glo_d + dg2[r].y * dx * (float) ghi_d;
                    } else if constexpr (rp_traits<WT>::affine) {
                        accg[r][c] += dg2[r].x * dx * (float) (glo_d + ghi_d) - dg2[r].y * sx;
                    } else {
                        accg[r][c] += dg[r] * dx * (float) (glo_d + ghi_d);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RPL; ++r) {
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const float a = rp_warp_reduce_sum<LANES>(acc[r][c]);
            [[maybe_unused]] float g = 0.0f;
            if constexpr (HAS_FUSION) {
                g = rp_warp_reduce_sum<LANES>(accg[r][c]);
            }
            if (lane == 0 && (row + r) < (int) ne1) {
                float out = a;
                if constexpr (HAS_FUSION) {
                    out = rp_mmv_fusion_epilogue(a, g, x_bias, gate_bias,
                        bias_row_base + (uint32_t)(row + r), glu_op, true);
                }
                y[(size_t) c * ys + row + r] = out;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, xs, ys,
                     wbase_gate, x_bias, gate_bias, glu_op, bias_row_base);
    NO_DEVICE_CODE;
#endif
}

template <int ROWS, int NWAVES, int NCOLS, int RPL = 1, int LANES = 64, ggml_type WT = GGML_TYPE_Q8_0>
static __global__ void mul_mat_vec_repacked_nc(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t xs, const uint32_t ys) {
    mul_mat_vec_repacked_nc_impl<ROWS, NWAVES, NCOLS, RPL, false, LANES, WT>(
        wbase, xq, y, ne0, ne1, xs, ys, nullptr, nullptr, nullptr,
        GGML_GLU_OP_REGLU, 0);
}

// MoE narrow batch: one assignment per block, one launch for all of them.
// Expert token counts are per expert, so at verify widths nearly every active
// expert holds a single assignment and the 32-wide tile computes 32 columns to
// serve it. Each block resolves its own expert from expert_bounds below, and
// ids_src1/ids_dst give the source and destination rows.
// Which expert owns assignment a. expert_bounds is the monotonic per-expert
// start offset, so this is an upper_bound minus one - cheap enough per block
// that building a metadata array up front is not worth a kernel launch.
static __device__ __forceinline__ uint32_t expert_bounds_search(
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const uint32_t a) {
    uint32_t lo = 0, hi = n_expert;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if ((uint32_t) expert_bounds[mid + 1] <= a) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <int ROWS, int NWAVES, int RPL = 1, ggml_type WT = GGML_TYPE_Q8_0>
static __global__ void mul_mat_vec_repacked_id1(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs, const uint32_t ys) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const uint32_t a = blockIdx.y;
    const uint32_t e = expert_bounds_search(expert_bounds, n_expert, a);
    mul_mat_vec_repacked_nc_impl<ROWS, NWAVES, 1, RPL, false, 64, WT>(
        wbase + (size_t) e * expert_stride,
        xq + (size_t) ids_src1[a] * xs,
        y  + (size_t) ids_dst[a]  * ys,
        ne0, ne1, xs, ys, nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU, 0);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds,
                     n_expert, expert_stride, xs, ys);
    NO_DEVICE_CODE;
#endif
}

// Fused MoE narrow batch: one assignment per block, up and gate in one pass.
// The ADD_ID bias is [ne1, n_expert], so the row index carries the expert.
template <int ROWS, int NWAVES, int RPL = 1, ggml_type WT = GGML_TYPE_Q8_0>
static __global__ void mul_mat_vec_repacked_id1_fused(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs, const uint32_t ys,
        const uint8_t * __restrict__ wbase_gate, const float * __restrict__ x_bias,
        const float * __restrict__ gate_bias, const ggml_glu_op glu_op) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const uint32_t a = blockIdx.y;
    const uint32_t e = expert_bounds_search(expert_bounds, n_expert, a);
    mul_mat_vec_repacked_nc_impl<ROWS, NWAVES, 1, RPL, true, 64, WT>(
        wbase + (size_t) e * expert_stride,
        xq + (size_t) ids_src1[a] * xs,
        y  + (size_t) ids_dst[a]  * ys,
        ne0, ne1, xs, ys,
        wbase_gate + (size_t) e * expert_stride,
        x_bias, gate_bias, glu_op, e * ne1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds,
                     n_expert, expert_stride, xs, ys,
                     wbase_gate, x_bias, gate_bias, glu_op);
    NO_DEVICE_CODE;
#endif
}

template <bool HAS_IDS, int TN_, int NRL, ggml_type WT = GGML_TYPE_Q8_0>
static __global__ void __launch_bounds__(32 * NRL, HAS_IDS && (WT == GGML_TYPE_Q4_K || WT == GGML_TYPE_Q5_K) ? 2 : 3) mmq_gemm_repacked_w32(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    mmq_gemm_repacked_impl<WT, HAS_IDS, 32, TN_, NRL>(
        wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst,
        expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}
