// engine/src/inkling2/cuda/ink_moe_fp4_sm120.cuh - Native Blackwell (sm_120a / sm_121a) mxf4nvf4 & mxf8f6f4 MoE interface
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Compiled in its own translation unit (ink_moe_fp4_sm120.cu) with -gencode arch=compute_120a,code=sm_120a
// (and 121a for DGX Spark GB10) so portable builds (sm_70..sm_121) link cleanly and dispatch here at runtime
// whenever compute capability is 12.0 or 12.1.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "strata/core/ink_sm120_table.hpp"

namespace strata::ink {

/// The MoE tile size and kernel variants for a batch of M tokens, from the card's table row (ink_sm120_cfg.cuh: sm120_moe_sel). `active` false means
/// the shipped dispatch (every other field is then the shipped value).
struct Sm120MoeSel {
    bool active = false;
    int tile_rows = 0;
    strata::core::Sm120Tc sf4 = strata::core::Sm120Tc::Base;
    strata::core::Sm120Tc mx8 = strata::core::Sm120Tc::Base;
    int pf_gu = -1;   // L2 prefetch window per weight row, gate/up: -1 = the shipped rule (STRATA_INK_SM120_PF, default 128)
    int pf_dn = -1;   // down: -1 = the shipped rule (STRATA_INK_SM120_PFD, default 0); also the MX8 window
    bool row_quant = false;   // the SF4 variant reads per-row global-scale activation slots (quantize_act_fp4 with per_row = true)
};

struct InkExpertDev {
    uint64_t body_off;  // offset from Section 4 base (4 KiB aligned)
    uint32_t dn_off;    // byte offset of w2 inside expert body
    uint8_t  gu_code;   // InkFmt code id (15=SF4, 16=MX8, 1..8=S_K)
    uint8_t  dn_code;
    uint16_t tier;      // 0 = GPU VRAM, 1 = pinned host, 2 = CPU hybrid (skip on GPU)
};
static_assert(sizeof(InkExpertDev) == 16, "InkExpertDev must be 16 B");

struct InkMoeTables {
    int32_t* cnt;       // [128]
    int32_t* off;       // [129]
    int32_t* active;    // [128]
    int32_t* nact;      // [1]
    int32_t* pairs;     // [P] packed (tok * 8 + slot)
    int32_t* tok_idx;   // [P] source token index for each sorted pair
    int32_t* n_tiles;   // [1] total 64-row tiles across all active experts
    int2*    tiles;     // [max_tiles] {.x = expert_id, .y = pair_row0}
};

namespace sm120 {

/// True when binary includes sm_120a / sm_121a mxf4nvf4 / mxf8f6f4 kernels and active GPU is sm_120 / sm_121.
bool available();

/// -1 = follow STRATA_INK_SM120_PFQ (default: on), 0 = the shipped TR=128 SF4 expert kernels, 1 = the K256 kernels (test hook for the teacher-forced KL harness; not an end-user switch).
int& pfq_override();

/// Quantize FP32 [rows, cols] (cols % 64 == 0) -> packed SF4 activation slots (40 B per 64 values: 32B E2M1x2 + 4B UE4M3 + 4B g_blk).
/// Note: uses 48-byte stride per 64 values (16-byte aligned for cp.async!).
/// per_row = true writes the per-row global-scale slots the BK=128 / direct-MMA / TR=128 variants read (Sm120MoeSel::row_quant).
constexpr int kActFp4SlotBytes = 48;

void quantize_act_fp4(const float* x, int rows, int cols, uint8_t* xa, cudaStream_t stream, bool per_row = false);

/// Per-row global scale FP4 activation quantizer (= quantize_act_fp4 with per_row = true).
void quantize_act_fp4_row(const float* x, int rows, int cols, uint8_t* xa, cudaStream_t stream);

/// Quantize FP32 [rows, cols] (cols % 32 == 0) -> packed MX8 activation slots (32B E4M3 + 1B UE8M0, stored as separate planes).
void quantize_act_mx8(const float* x, int rows, int cols, uint8_t* xq, uint8_t* xsc, cudaStream_t stream);

/// True when the SF4 variant (a strata::core::Sm120Tc value) reads the per-row activation slots.
inline bool sf4_variant_row_quant(int variant) {
    const strata::core::Sm120Tc v = (strata::core::Sm120Tc)variant;
    return v == strata::core::Sm120Tc::Bk128 || v == strata::core::Sm120Tc::Direct || v == strata::core::Sm120Tc::Tr128 || v == strata::core::Sm120Tc::Tr128Bk128;
}

/// Blackwell sm_120a native mxf4nvf4 fused Gate+Up+SITU(4.0, 25.0)+FP4-requant (for SF4 experts, L0..19).
/// tile_rows in {16, 32, 64, 128} matches the k_ink_moe_bin tile size. `sel` (null = the shipped kernels) picks the variant from the card's table row.
void moe_sf4_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xa, uint8_t* ha, int max_tiles, int tile_rows, cudaStream_t stream,
                   const Sm120MoeSel* sel = nullptr);

/// Blackwell sm_120a native mxf4nvf4 Down projection (for SF4 experts, L0..19).
/// When atomic_reduce==true, accumulates directly into y_pair[tok, 2304] (via atomicAdd) instead of [pair_id, 2304]!
void moe_sf4_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* ha, const float* topk_w, float* y_pair, int max_tiles, int tile_rows, cudaStream_t stream,
                   bool atomic_reduce = false, const Sm120MoeSel* sel = nullptr);

/// Blackwell sm_120a native mxf8f6f4 fused Gate+Up+SITU(4.0, 25.0)+MX8-requant (for MX8 experts, L20..23).
void moe_mx8_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xq, const uint8_t* xsc, uint8_t* hq, uint8_t* hsc,
                   int max_tiles, int tile_rows, cudaStream_t stream, const Sm120MoeSel* sel = nullptr);

/// Blackwell sm_120a native mxf8f6f4 Down projection (for MX8 experts, L20..23).
void moe_mx8_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_pair,
                   int max_tiles, int tile_rows, cudaStream_t stream, const Sm120MoeSel* sel = nullptr);

/// Expert-Looped L2-Resident Prefill MoE kernels (reads every expert from GDDR7 ONCE!), 5090 table row.
void moe_sf4_gu_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* xa, uint8_t* ha, cudaStream_t stream);
void moe_sf4_dn_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* ha, const float* topk_w, float* y_pair, cudaStream_t stream);
void moe_mx8_gu_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* xq, const uint8_t* xsc, uint8_t* hq, uint8_t* hsc, cudaStream_t stream);
void moe_mx8_dn_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_pair, cudaStream_t stream);

/// Prefill Direct-L2 Atomic Down Projection (TR=128, BK=128; writes directly to d_moe_raw, ZERO 302 MB y_pair!), 5090 table row.
void moe_sf4_dn_tc_atomic(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                          const uint8_t* ha, const float* topk_w, float* y_out_atomic, int max_tiles, cudaStream_t stream);
void moe_mx8_dn_tc_atomic(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                          const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_out_atomic, int max_tiles, cudaStream_t stream);

/// Standalone single-expert GEMM test entry points (used by ink_kernel_test for FP64 parity & TFLOPS benchmarking).
void test_sf4_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc, float w_outer,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream);

/// Blackwell sm_120a Direct-MMA NVFP4 dense linear GEMM: Y[M, N] = X_fp4[M, K] @ W_sf4[N, K]^T (100% Graph capturable).
/// w_sf4 points to .strata v1.1 FMT_SF4 payload: [float w_outer, float x_scale, uint8_t qs[N*K/2], uint8_t sc[N*K/16]].
/// direct = true: the per-row-scale direct-MMA kernel (needs quantize_act_fp4 with per_row = true).
void dense_sf4_gemm_tc(const uint8_t* w_sf4, const uint8_t* xa_fp4, float* y_f32,
                       int M, int N, int K, cudaStream_t stream, bool direct = false);

void test_mx8_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream);

}  // namespace sm120
}  // namespace strata::ink
