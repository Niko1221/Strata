// include/strata/kernels/rope_dynamic.hpp - dynamic rope scaling: moving cached keys between two scalings.
//
// K is cached POST-RoPE (rope_scaling.hpp).  Every scaling rotates pair i at position p by p * w_i and scales the
// rotated dims by mscale, so a key stored under scaling A becomes the key of scaling B by one more rotation of
// p * (wB_i - wA_i) and the factor mscaleB / mscaleA - no other input is needed.  Used by --rope-scaling dynamic
// (generate.cpp): no scaling while a dialog fits the trained context, yarn by its length past it.
#pragma once

#include "strata/kernels/rope_scaling.hpp"

#include <cstdint>

namespace strata::kernels {

/// Per pair (32 of a 64-dim rotary slice): the angle per position build_rope_table uses, and the magnitude.
struct RopeRates {
    double w[32] = {};
    double mscale = 1.0;
};
RopeRates rope_rates(int n_rot, const RopeScaling& s);

/// One K pool, kv_q8.hpp's layout: [page][kv_head][page_size][head_dim], int8 codes + fp16 scale per 64, or fp16.
struct RopeKPool {
    uint16_t* k_pool = nullptr;   ///< fp16 values (null when int8)
    int8_t* k_q = nullptr;        ///< int8 codes
    uint16_t* k_scale = nullptr;  ///< fp16 scale per 64 values
    const int32_t* page_table = nullptr;   ///< null: the identity (a streamed layer's host copy)
    int n_kv_heads = 0, page_size = 0, head_dim = 0;
    bool hadamard = false;        ///< stored through kv_q4.hpp's 256-point H (kv_rot): the whole row is re-coded
};

/// Cells [0, cells) of `k` from `from` to `to` (head_dim 256, n_rot 64).  Without H only the rope group (dims 0-63)
/// is re-quantized; with H every group is.  The cell's position is read through the mrope table (identity for text).
void rope_rerotate_k(const RopeKPool& k, int64_t cells, const RopeRates& from, const RopeRates& to, void* stream);

/// The indexer's keys (idx_dim 128, n_rot 64): pooled rows [0, n_blocks) rotated at b * idx_block, the spare row
/// n_blocks and `dead` at position 0 (the magnitude only).
void rope_rerotate_pooled(float* pooled, float* dead, int64_t n_blocks, int idx_block, int idx_dim,
                          const RopeRates& from, const RopeRates& to, void* stream);

/// Scales the rotary dims [0, 64) of `n_rows` host rows of `idx_dim` floats by `ratio` (a checkpoint's `dead`).
void rope_rescale_rows_host(float* rows, int64_t n_rows, int idx_dim, double ratio);

/// build_rope_table(n_rot, s, ...) bit for bit, on `threads` host threads (the same expressions, per-pair terms
/// hoisted out of the position loop).
void build_rope_table_mt(int n_rot, const RopeScaling& s, int max_pos, float* cos_tab, float* sin_tab, int threads);

}  // namespace strata::kernels
