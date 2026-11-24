#pragma once
#include "strata/hy4flash_gguf/vision.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash {

struct MatCtx;

/// The compressed indexer selects visible KV rows for sparse attention.
struct IndexerWeights {
    const void* attn_k = nullptr;   ///< [n_embd, key_dim]
    int attn_k_type = 0;
    const void* attn_q_b = nullptr; ///< [q_lora, key_dim * heads]
    int attn_q_b_type = 0;
    const void* proj = nullptr;     ///< [n_embd, heads]
    int proj_type = 0;
    const void* value = nullptr;    ///< [key_dim, key_dim]
    int value_type = 0;
    const void* gate = nullptr;     ///< [key_dim, key_dim]
    int gate_type = 0;
    const float* k_norm = nullptr;        ///< [key_dim]
    const float* compressor_norm = nullptr;  ///< [key_dim]
    const float* ape = nullptr;            ///< [key_dim, kpool]
};

/// Cache completed pooled keys and the rows of the incomplete group.
struct IndexerCache {
    int n_layers = 0, key_dim = 0, kpool = 0, max_pool = 0;
    int n_pool = 0;   ///< complete pools, all layers in step
    int carry = 0;    ///< rows of the in-flight group in `vg`, 0..kpool-1
    /// Absolute position of the next token expected by the indexer.
    int valid_pos = 0;
    /// Keep carry and pooled-key count on device for CUDA graph replay.
    void* counters = nullptr;   ///< device int2, {carry, n_pool}
    std::vector<void*> pooled;  ///< per layer, device [key_dim, max_pool] f32
    std::vector<void*> vg;      ///< per layer, device [2 * key_dim, kpool] f32, value then gate
    // Last speculative window's unpooled rows, including its incoming carry.
    // Small per-layer buffers make any accepted prefix reversible across pool boundaries.
    std::vector<void*> verify_vg;
    int verify_capacity = 0, verify_base = -1, verify_tokens = 0;
    std::vector<int32_t> ids;   ///< layer index by block id, -1 when the block has no indexer

    ~IndexerCache();
    IndexerCache() = default;
    IndexerCache(const IndexerCache&) = delete;
    IndexerCache& operator=(const IndexerCache&) = delete;
};

bool indexer_cache_init(IndexerCache& c, const std::vector<int32_t>& block_has_indexer, int key_dim,
                        int kpool, int max_tokens, std::string& err);

/// Restore the indexer to the requested sequence position.
bool indexer_cache_rewind(IndexerCache& c, int pos);
// Queue the accepted-prefix carry restore using a device verdict. No readback.
// Call only immediately after verification, before the next window overwrites
// verify_vg; host mirrors are updated separately after collection.
void indexer_commit_device(IndexerCache& c,const struct RoundVerdict* verdict,void* stream);
void indexer_commit_loop(IndexerCache& c,const struct RoundVerdict* verdict,const struct DeviceLoopState* state,int rows,void* stream);

std::size_t indexer_cache_bytes(int n_layers, int key_dim, int kpool, int max_tokens);

/// Describe the sparse selection for the token range starting at base_pos.
struct IndexerPlan {
    bool engaged = false;
    bool select_tail = false;  ///< the forced tail is on, so `nv` excludes the token's own group
    int n_top = 0;   ///< pools RECALLED per token, `min(nv, top_k / kpool - 1)` at the chunk's last position
    int n_sel = 0;   ///< cells per token, `kpool * n_top + kpool` - exactly `top_k` once engaged
};
IndexerPlan indexer_plan(int top_k, int kpool, bool select_tail, int base_pos, int n_tok);

/// Advance the device counters once after all layers have consumed the chunk.
void indexer_advance(IndexerCache& c, int n_tok, void* stream);

std::size_t indexer_scratch_bytes(int heads, int key_dim, int kpool, int n_tok, int max_pool, int n_sel);

/// Pool completed groups, then score and select visible keys for each query.
void indexer_select(IndexerCache& c, int layer, const IndexerWeights& w, const IndexerPlan& plan,
                    const void* cur_q8, const float* cur_f16, const void* q_resid_q8, const float* q_resid_f16,
                    MatCtx* mc, int n_embd, int q_lora, int heads, int n_rot, int n_tok, int base_pos,
                    const int32_t* q_pos, float freq_base, float rms_eps, int32_t* sel_cells, void* scratch,
                    void* stream, ImageRope image = {},
                    /// Save the incomplete group for rollback before verification overwrites it.
                    float* window_save = nullptr);

/// Restore the incomplete pooling group for the accepted prefix.
bool indexer_restore_window(IndexerCache& c, const std::vector<void*>& windows, int win_start, int end_pos,
                            int new_len, std::string& err);

/// Zero the device pair and the host mirror unconditionally. Unlike `indexer_cache_rewind(0)`, never
/// returns early because the host already says 0.
bool indexer_force_reset(IndexerCache& c, std::string& err);

}  // namespace strata::hy4_flash
