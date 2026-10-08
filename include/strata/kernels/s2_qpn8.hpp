// include/strata/kernels/s2_qpn8.hpp - the V100 (sm_70) expert GEMV path: load-time QPN8 repack + m8n8k4.
//
// **WHAT THIS IS, AND WHAT THE MEASUREMENTS SAID.**  `s2_expert_grouped.cu`'s M=2..8 path unpacks the Q2_0
// codes inside the kernel (`expand_codes`: 2-bit fields -> one byte per element for `dp4a`, 16 ALU ops per
// 32 elements).  `bench/micro/s2_unpack_bench.cu` prices that unpacking on the V100 (Tesla V100-SXM2-16GB,
// 48 blobs cycled past the 6 MB L2; numbers move ~10% run to run, the shape of the tables does not - the
// full record is `bench/results/2026-10-02-skinny-expert/`):
//
//   * The unpack is real money once entries share an expert: at 10 experts x 8 entries the shipped inner
//     loop is 76.1 us and replacing `expand_codes` by zeros is 41.0 us - the unpack is 35 us of the call.
//     At 10 x 1 it is 30.7 vs 15.1 us.
//   * An unpack-free DP4A form at CONSTANT BYTES does not exist.  A `dp4a` operand is one byte per element,
//     so 2-bit codes must expand 4x somewhere.  Three same-bytes forms were built and measured, all slower
//     than the shipped in-kernel unpack: pre-expanded codes (4x the bytes: 96.8 us against 76.1, the DRAM
//     cost eats the win), a within-chunk bit-transposed codes layout with a 256-entry shared LUT (95.9 us,
//     the shared loads cost more than the shifts they replace) and wider dp4a chains (75.6 / 77.0, neutral).
//   * The load-time QPN8 repack (a permutation, byte count unchanged) + `mma.m8n8k4` f16 tensor cores is
//     the one same-bytes form that removes the unpack: the codes arrive as one coalesced 4-byte record per
//     (32-row tile, 16-K group, lane) and a branchless 1024/1025 bias trick turns them straight into B
//     fragments.  Measured on the GEMV inner loop: 43.0 us at 10 experts x 8 entries (1.77x over 76.1),
//     43.0 at x 4, 43.1 at x 2, 50.2 at x 1 - the m8 tile computes 8 rows and unused rows are pure loss, so
//     the crossover is FOUR OR MORE entries per expert.  Without the repack the same kernel gathering from
//     the canonical layout is 156.8 us: the repack IS the coalescing.
//   * **The shipped ENTRY POINT (`moe_grouped_s2_qpn8`, full pipeline with the exact float contract) now
//     CAShes the win at enough entries per expert**: 90.7 us against the DP4A pipeline's 109.9 at 10 x 8
//     (1.21x), 83.1 vs 74.0 at 10 x 4 (0.89x), 76.9 vs 56.3 at 10 x 2 (0.73x), 75.9 vs 50.0 at 10 x 1
//     (0.66x).  Two changes closed the old 0.81x gap (both measured by `s2_qpn8_parity --bench` on the same
//     V100): a per-window prep pass converts every (entry, k16 group) to its 32-byte A fragment ONCE
//     (`prep_a_kernel`), which removed the per-row-tile int8 load + convert that an ablation priced at 16
//     of the gu kernel's 48 us; and the two k16 groups of a chunk now issue into two mma accumulator chains
//     (+4 us on gu).  The crossover stays at FOUR OR MORE entries per expert (the m8 tile wastes rows below
//     it) and the m8n8k4 path is still slower at 1-2 entries - so it stays OFF (`STRATA_QPN8=1` opts in) and
//     the DP4A kernels keep the default.  The prep buffer is one process-lifetime device allocation.
//   * The readout still pays the DP4A contract's per-chunk `dw * dx * s` and the q8_0 block IO; the prep
//     fragment is 2x the activation bytes (f16 vs int8), which the win absorbs because the loads are two
//     uint4 instead of five scalar words.
//
// **THE NUMERICS.**  The m8n8k4 multiplies f16 x f16 into f32; the codes arrive as `code - 1` (the bias
// constant is 1025, not 1024) and the activations as exact int8->f16, so every product and every chunk sum
// is an integer below 2^24 and EXACT in f32.  The per-chunk term `dw * dx * (float)(s - hx)` is therefore
// bit-for-bit the DP4A kernels' (same integer `s - hx`, same float expression); only the order the chunk
// terms are ADDED differs (the DP4A kernels sum lane-strided chunks through a shuffle tree, this one sums
// each output's chunks in k order through the split-K reduce).  `s2_qpn8_parity` measures the resulting
// bound: L1 relative 7e-8..1.6e-7, worst absolute 9.5e-6 at the test's magnitudes - float-order level,
// not the 4.761e-04 of an f16 scale rounding - and the two pipelines' quantized intermediates are
// BYTE-identical on the test's data.
//
// **THE INTEGRATION.**  `s2_qpn8_active()` is the gate: compute capability 7.0 AND `STRATA_QPN8=1` (off by
// default, so no arch and no workload changes behaviour).  When it is on, `ExpertCache::open` sizes every
// canonical-blob slot at `2 * blob` and `ExpertCache::fill_slot*` writes the repacked copy at `slot + blob`
// right after the canonical copy - the repack's one-time cost lives at cache admission.  The M=1 per-hit
// path reads the canonical half untouched (its stride is the slot size); the verify-window grouped path
// calls `moe_grouped_s2_qpn8`, which reads the repacked half at `grp_ptr[g] + blob_bytes` and falls back to
// `moe_grouped_s2` for any other geometry.  The memory cost is the 2x slot - the price of coexistence with
// the M=1 path - halving the auto-sized cache when the knob is on, plus the window's A-fragment buffer
// (`qpn8_window_buf`, ~6 KB per entry; one allocation reused by every call).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The gate: true only on compute capability 7.0 with `STRATA_QPN8=1`.  Read once per process; the env is
/// the switch, the cc check keeps sm_75+ (and the sm_80+ trunk) on the existing kernels no matter what.  The
/// m8n8k4 kernels themselves are compiled only into the experimental Volta build
/// (`-DSTRATA_EXPERIMENTAL_SM60=ON`, PR #600's rule); every other build links the fallbacks below, where this
/// is a constant false and `moe_grouped_s2_qpn8` forwards to `moe_grouped_s2`.
bool s2_qpn8_active();

/// The canonical Q2_0 expert blob this layout knows (1,382,400 bytes; H 2560 / FF 640) - anything else is
/// not repacked and keeps its old slot size.
int64_t s2_qpn8_blob_bytes();

/// Slot bytes for one expert when the QPN8 copy is kept beside the canonical one: `2 * blob` when active
/// AND the blob is the canonical one, `blob` otherwise.  The expert-cache sizing and `ExpertCache::open`
/// agree through this; it is the single policy function for "is this slot dual-form".
int64_t s2_qpn8_slot_bytes(int64_t blob);

/// The repack: canonical Q2_0 expert blob -> QPN8 compute layout, SAME byte count.  `dst != src`; both
/// `blob_bytes` (must be the 1,382,400-byte canonical Q2_0 blob - the only layout this geometry knows).
/// One-time cost at cache admission, per expert: 40 + 80 record groups x 32 rows of one integer bit gather.
void s2_qpn8_repack_blob(uint8_t* dst, const uint8_t* src, int64_t blob_bytes, void* stream);

/// `moe_grouped_s2`'s twin for repacked blobs: per entry (up to 8 of a group) the same gate/up -> swiglu ->
/// quantize -> down pipeline, m8n8k4 mma instead of dp4a.  `grp_ptr[g]` is the CANONICAL blob base of the
/// group's expert (a `ExpertCache` slot); the repacked copy is read from `grp_ptr[g] + blob_bytes`.
/// Per chunk the outputs are the dp4a kernels' bit-exact terms; the chunk sum order differs (see header).
/// `scratch` is `moe_hit_grouped_scratch_bytes(cap_entries, n_embd, n_ff)` as for `moe_grouped_s2`.
void moe_grouped_s2_qpn8(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                         const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                         int64_t blob_bytes, const uint8_t* x_q8_0, const float* x_scales, void* scratch,
                         float* out, void* stream);

}  // namespace strata::kernels
