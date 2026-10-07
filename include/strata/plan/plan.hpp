// include/strata/plan/plan.hpp - P1.S9 census + memory planner.
//
// This is the component that makes one binary adapt to the GPU it is running on.  The engine has 33.97 GB of
// expert weights (has to live in DRAM and be streamed) and a fixed VRAM budget that has to hold the dense
// weights, the KV cache, the indexer keys, the recurrent state and as many expert-cache slots as remain.
// Every one of those competes, so the plan is an arithmetic result rather than a configuration choice, and
// it must REFUSE rather than silently overcommit.
//
// DERIVATION POLICY.  `phases/00-README.md` Â§2 gives three rows of expected values but no formula, so the
// arithmetic here is derived from the model's geometry and the table is used as a CHECK.  Where the two
// disagree the geometry wins and the disagreement is reported - a table transcribed by hand is not a
// specification, and fitting formulas to it would hide exactly the errors this is supposed to catch.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::plan {

// ---- model geometry ---------------------------------------------------------
// The qwen4exp values are the real Qwen3.8-Flash-Next artifact; the glm-dsa values come from
// derived_glm_dsa().  rch is what the cost functions branch on, so a geometry built for one family
// cannot quietly be charged as the other.
struct Geometry {
    std::string arch = "qwen4exp";
    int n_layers = 48;
    int n_qsa_layers = 12; // layers 3, 7, ... 47 - the full-attention ones
    int n_kv_heads = 2;
    int head_dim = 256;
    int kv_group = 64; // INT8 KV is grouped-64: one fp16 scale per 64 elements
    int qsa_block = 4; // the indexer pools one key per 4-token block
    /// THE INDEXER'S KEY STORE, and the trap that has now been walked into twice in opposite directions.
    ///
    /// `indexer.q_proj.weight` is [2560, 512] = **4 QUERY heads** of 128.  `indexer.k_proj.weight` is
    /// [2560, 128] = **ONE shared KEY head**.  The queries are not cached; only the key is.  So the cache
    /// holds one 128-wide key per TOKEN, pooled to one per `qsa_block`-token block - which
    /// `include/strata/kernels/qsa.hpp` states directly: `pooled (max_cells/idx_block + 1, idx_dim)`, and
    /// its own note that the fp32 store is "50 MiB at 32K" (12 layers x 8192 blocks x 128 x 4 B = 48 MiB).
    ///
    ///   original      indexer_heads(1) * head_dim(256) / 4  = 64 B/token/layer   2x too big
    ///   round 194     indexer_heads(4) * key_dim(128) / 4   = 128 B/token/layer  4x too big
    ///   correct       key_heads(1)     * key_dim(128) / 4   =  32 B/token/layer
    ///
    /// Round 194 "fixed" this by reading `indexer.head_count = 4` out of the metadata and multiplying by it -
    /// right about the metadata, wrong about what was being counted.  **A head count in the metadata does not
    /// say WHICH tensor it counts.**
    int indexer_q_heads = 4;    // query heads: they multiply the SCORES, and cost no cache at all
    int indexer_key_heads = 1;  // k_proj is [n_embd, 128]: ONE shared key head
    int indexer_key_dim = 128;  // the key width, and the width of the thing that IS cached

    /// GDN geometry.  `docs/semantics.md`: `ssm.state_size = 128`, `ssm.time_step_rank = 48` (the v heads),
    /// `ssm.conv_kernel = 4`, and `n_embd_r() = (d_conv-1) * (ssm_d_inner + 2*n_group*d_state) = 3 * 10240`.
    int ssm_state_size = 128;
    int ssm_v_heads = 48;
    int ssm_d_conv = 4;
    int ssm_conv_channels = 10240;

    /// 36 of the 48 layers are GDN; the rest are QSA.  Declared as a derivation rather than a third constant,
    /// because a `n_gdn_layers = 36` alongside `n_layers` and `n_qsa_layers` is a third thing to keep in step.
    int n_gdn_layers() const { return n_layers - n_qsa_layers; }

    /// ---- glm-dsa (GLM-5.3 / GLM-5.2, the 744B family).  Zero for qwen4exp. ----
    ///
    /// Every layer is MLA, so there is no GDN term and no QSA term.  The cache holds the 512-wide latent plus
    /// the 64-wide rope slice, because `kv_b_proj` is absorbed into the two ends - the cache never holds an
    /// expanded key.  `index_kpool` is a GLM-5.3-Flash feature and does not exist in this checkpoint, so the
    /// indexer caches one `index_head_dim`-wide key per token with no pooling block.
    ///
    /// `n_indexer_layers` is 21 for the checkpoint's `indexer_types` (layers 0, 1, 2 then every 4th), and 0
    /// for the container at D:\models\GLM-5.3-colibri-int4-g64, which carries no indexer tensors at all.
    int n_indexer_layers = 0;
    int kv_lora_rank = 0;
    int qk_rope_head_dim = 0;

    int ml_cache_elems() const { return kv_lora_rank + qk_rope_head_dim; }
};

// The glm-dsa geometry, measured on the checkpoint rather than transcribed from a card: 78 layers, hidden
// 6144, 256 experts with 8 active, 64 heads (MLA has no GQA split, so head_count == head_count_kv),
// q_lora_rank 2048, kv_lora_rank 512, qk 192 nope + 64 rope, v_head_dim 256, indexer top_k 2048 /
// key_length 128 / head_count 32.
inline Geometry derived_glm_dsa(bool indexer_present = false) {
    Geometry g;
    g.arch = "glm-dsa";
    g.n_layers = 78;
    g.n_qsa_layers = 0;
    g.n_kv_heads = 64;
    g.head_dim = 256;
    g.kv_group = 64;
    g.qsa_block = 1;              // no pooling block: one key per token
    g.indexer_q_heads = 32;       // index_n_heads: they multiply the scores and cost no cache
    g.indexer_key_heads = 1;
    g.indexer_key_dim = 128;      // index_head_dim
    g.kv_lora_rank = 512;
    g.qk_rope_head_dim = 64;
    g.n_indexer_layers = indexer_present ? 21 : 0;
    return g;
}

// ---- byte costs ------------------------------------------------------------
// int4 group-64 with f32 scales - the format the GLM container is stored in, and the format a GLM pack has to
// carry.  Codes are 4 bits per element, scales 4 B per 64 elements, three tensors per expert.  Measured on the
// checkpoint: 6,291,456 + 786,432 per tensor, so 21,233,664 B per expert.  That is 15.4x the 1,382,400 B
// qwen4exp blob, and `cache_slots` divides by it, so a GLM plan is a different plan, not the same plan with
// different numbers.
inline uint64_t int4_g64_blob(uint64_t elems_per_tensor) {
    return 3ull * (elems_per_tensor / 2 + elems_per_tensor / 64 * 4);
}
inline uint64_t glm_expert_blob() { return int4_g64_blob(6144ull * 2048ull); }

struct Costs {
    uint64_t expert_blob = 1382400; // one expert's gate_up+down, codes and scales (architecture Â§3.2)
    uint64_t dense_bytes = 0;       // from the manifest: every non-expert tensor, canonical form
    uint64_t embd_bytes = 0;        // token embedding
    uint64_t mtp_bytes = 0;         // multi-token-prediction head, when present
    uint64_t workspace_bytes = 0;
    uint64_t state_bytes = 0; // GDN recurrence + conv history + replay log, per sequence
};

// GDN recurrence + conv history, bytes per sequence.  DERIVED, for the same reason everything else here is.
//
// This term was MISSING from the plan until round 198: `Costs::state_bytes` exists and `--state` exists, but
// nothing passed it, so the planner was fed zero and quietly planned without it.  36 GDN layers of fp32 state
// is **117.7 MB, or 85 expert cache slots** - not a rounding error.
//
//   recurrence    (S, h_v, S) fp32  = 128 * 48 * 128 * 4 = 3,145,728 B per layer
//   conv history  (d_conv-1, C) fp32 =     3 * 10240 * 4 =   122,880 B per layer
//
// It is a FIXED cost and not a cache: it is state, it is resident for the life of a sequence, and it cannot be
// evicted to make room for an expert.  A planner that omits it returns a plan it cannot honour, which is the
// failure `DoesNotClose` exists to prevent.
inline uint64_t state_bytes(const Geometry& g) {
    if (g.arch == "glm-dsa") return 0;   // no GDN: nothing recurrent to hold for a sequence
    const uint64_t per_layer = (uint64_t) g.ssm_state_size * (uint64_t) g.ssm_v_heads *
                                   (uint64_t) g.ssm_state_size * sizeof(float) +
                               (uint64_t) (g.ssm_d_conv - 1) * (uint64_t) g.ssm_conv_channels * sizeof(float);
    return (uint64_t) g.n_gdn_layers() * per_layer;
}

// INT8 KV + indexer keys, bytes per token.  Both terms come from the geometry above and nothing else.
inline uint64_t kv_bytes_per_token(const Geometry& g) {
    if (g.arch == "glm-dsa") {
        const uint64_t e = (uint64_t) g.ml_cache_elems();              // 512 latent + 64 rope, not an expanded key
        const uint64_t per_layer = e + (e / (uint64_t) g.kv_group) * 2;
        return (uint64_t) g.n_layers * per_layer +
               (uint64_t) g.n_indexer_layers * (uint64_t) g.indexer_key_heads * (uint64_t) g.indexer_key_dim;
    }
    const uint64_t kv_elems = 2ull * g.n_kv_heads * g.head_dim;       // K and V
    const uint64_t kv_scales = (kv_elems / (uint64_t)g.kv_group) * 2; // fp16 scale per group
    const uint64_t per_layer = kv_elems * 1 + kv_scales;              // INT8: one byte per element
    // indexer: ONE cached key per block, `idx_dim` wide, SHARED across the query heads.  See the note on
    // `indexer_key_heads` above - this term has been wrong in both directions and the derivation is why.
    const uint64_t idx_layer =
        (uint64_t) g.indexer_key_heads * (uint64_t) g.indexer_key_dim / (uint64_t) g.qsa_block;
    return (uint64_t)g.n_qsa_layers * (per_layer + idx_layer);
}

// ---- the plan --------------------------------------------------------------
struct Plan {
    uint64_t max_context = 0;
    uint64_t kv_bytes = 0;    // KV + indexer keys at max_context
    uint64_t state_bytes = 0; // GDN recurrence + conv history - FIXED, not evictable
    uint64_t cache_bytes = 0; // what is left for expert cache slots
    uint64_t cache_slots = 0;
    uint64_t vram_budget = 0;
    uint64_t vram_used = 0;
    uint64_t dram_experts = 0; // the whole expert arena, streamed
    std::vector<std::string> notes;
};

// Thrown when the requested context and the fixed costs cannot both fit.  A planner that returns a plan it
// cannot honour is worse than one that fails, because the failure then happens at token 4000 instead of at
// startup.
struct DoesNotClose : std::runtime_error {
    explicit DoesNotClose(const std::string& w) : std::runtime_error(w) {}
};

// The VRAM budget that the KV cache and the expert cache share.  This is the one number taken from
// `00-README.md` Â§2 rather than derived: the table's three rows all leave `KV + cache` constant, and that
// constant is what the planner is dividing.  Expressed in bytes, decimal, matching how the table rounds.
inline uint64_t vram_pool_bytes() {
    return 5943000000ull;
}

inline Plan make_plan(uint64_t max_context, const Geometry& g, const Costs& c,
                      uint64_t pool = vram_pool_bytes()) {
    Plan p;
    p.max_context = max_context;
    p.vram_budget = pool;
    p.kv_bytes = kv_bytes_per_token(g) * max_context;
    p.state_bytes = c.state_bytes;
    p.dram_experts = c.dense_bytes + c.embd_bytes; // costs that never enter the pool

    const uint64_t fixed = c.dense_bytes + c.embd_bytes + c.mtp_bytes + c.workspace_bytes + c.state_bytes;
    if (p.kv_bytes + fixed > pool) {
        // The glm-dsa dense weights (11,595,965,440 B measured on the checkpoint) are larger than the whole
        // pool, so this is not a context to reduce - it is the wrong pool.  Saying so is what keeps the
        // failure at startup instead of at token 4000.
        if (c.dense_bytes > pool)
            throw DoesNotClose("the dense weights alone are " + std::to_string(c.dense_bytes) + " B, more than the " +
                               std::to_string(pool) + " B VRAM pool - on glm-dsa they are resident in DRAM, so this pool is the wrong pool");
        throw DoesNotClose("max-context " + std::to_string(max_context) + " needs " +
                           std::to_string(p.kv_bytes + fixed) + " B of the " + std::to_string(pool) +
                           " B VRAM pool; reduce --max-context");
    }
    p.cache_bytes = pool - p.kv_bytes - fixed;
    p.cache_slots = p.cache_bytes / c.expert_blob;
    p.vram_used = pool - p.cache_bytes % c.expert_blob; // slot remainder is unusable, not free
    if (p.cache_slots == 0) {
        throw DoesNotClose("no room for even one expert cache slot at max-context " +
                           std::to_string(max_context));
    }
    return p;
}

inline std::string to_string(const Plan& p) {
    auto gb = [](uint64_t b) { return std::to_string(b / 1e9); };
    std::string s;
    s += "memory plan  --max-context " + std::to_string(p.max_context) + "\n";
    s += "  KV + indexer keys   " + gb(p.kv_bytes) + " GB\n";
    s += "  recurrent state     " + gb(p.state_bytes) + " GB  (GDN recurrence + conv history, NOT evictable)\n";
    s += "  expert cache        " + gb(p.cache_bytes) + " GB  (" + std::to_string(p.cache_slots) +
         " slots of 1,382,400 B)\n";
    s += "  VRAM pool           " + gb(p.vram_budget) + " GB\n";
    s += "  expert arena (DRAM) " + gb(p.dram_experts) + " GB\n";
    for (const std::string& n : p.notes) s += "  note: " + n + "\n";
    return s;
}

} // namespace strata::plan
