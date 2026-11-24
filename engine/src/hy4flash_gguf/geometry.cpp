// Read model geometry from GGUF metadata.
#include "strata/hy4flash_gguf/geometry.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::hy4_flash {
namespace {

constexpr const char* kP = "hy_v4_flash.";

std::string key(const char* suffix) { return std::string(kP) + suffix; }

/// A scalar the guard has already proved present and numeric.
uint64_t u(const GgufFile& f, const char* suffix) { return f.get(key(suffix))->u; }
double fl(const GgufFile& f, const char* suffix) {
    const MetaValue* v = f.get(key(suffix));
    return v->num();
}
/// An optional scalar; `missing` when the file does not carry it.
double fl_or(const GgufFile& f, const char* suffix, double missing) {
    const MetaValue* v = f.get(key(suffix));
    return v != nullptr && v->is_num() ? v->num() : missing;
}
uint64_t u_or(const GgufFile& f, const char* suffix, uint64_t missing) {
    const MetaValue* v = f.get(key(suffix));
    return v != nullptr && v->is_num() ? v->u : missing;
}

/// The retained elements of an array the guard has proved complete (count == items.size()).
std::vector<uint64_t> arr(const GgufFile& f, const char* suffix) {
    const MetaValue* v = f.get(key(suffix));
    std::vector<uint64_t> out;
    if (v == nullptr) return out;
    out.reserve(v->items.size());
    for (const MetaValue& e : v->items) out.push_back(e.u);
    return out;
}

}  // namespace

std::string read_geometry(const GgufFile& f, Geometry& g, int32_t max_context,
                          const TensorInfo* engram_tensor) {
    g = Geometry{};

    g.block_count = (int32_t) u(f, "block_count");
    g.n_mtp_blocks = (int32_t) u(f, "nextn_predict_layers");
    g.trunk_blocks = g.block_count - g.n_mtp_blocks;
    g.dense_blocks = (int32_t) u(f, "leading_dense_block_count");

    g.hidden_dim = (int32_t) u(f, "embedding_length");
    g.dense_ff = (int32_t) u_or(f, "feed_forward_length", 0);
    g.vocab_size = (int32_t) u_or(f, "vocab_size", 0);
    g.rms_norm_eps = (float) fl(f, "attention.layer_norm_rms_epsilon");

    // ---- the turn terminators. UNPREFIXED keys: the tokenizer block is not under `hy_v4_flash.`, so these
    // do not go through `key()`. Optional, because a base model without a chat template carries no eot.
    const MetaValue* eos = f.get("tokenizer.ggml.eos_token_id");
    g.eos_token_id = eos && eos->is_num() ? static_cast<int32_t>(eos->u) : -1;
    for (const char* k : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                          "tokenizer.ggml.eom_token_id"}) {
        const MetaValue* v = f.get(k);
        if (v == nullptr || !v->is_num()) continue;
        const int32_t id = (int32_t) v->u;
        if (id < 0) continue;
        bool seen = false;
        for (int32_t have : g.stop_tokens) seen = seen || have == id;
        if (!seen) g.stop_tokens.push_back(id);   // eos and eot are the same id in some files
    }

    // Hyper-connections. The count is what makes the inter-block state hc_mult x hidden wide, so a file that
    // declares none is refused here rather than silently running as an ordinary single-residual model.
    g.hc_mult = (int32_t) u_or(f, "hyper_connection.count", 0);
    g.hc_eps = (float) fl_or(f, "hyper_connection.epsilon", 0);
    g.hc_magnitude = (float) fl_or(f, "hyper_connection.magnitude", 0);
    if (g.hc_mult < 1)
        return "hy_v4_flash.hyper_connection.count is missing or zero; this model keeps that many residual "
               "streams and every buffer is sized from it";

    // MoE
    g.n_experts = (int32_t) u(f, "expert_count");
    g.top_k = (int32_t) u(f, "expert_used_count");
    g.expert_dim = (int32_t) u(f, "expert_latent_length");
    g.expert_ff = (int32_t) u(f, "expert_feed_forward_length");
    g.n_shared_experts = (int32_t) u(f, "expert_shared_count");
    g.gating_func = (int32_t) u(f, "expert_gating_func");
    g.routed_scale = (float) fl_or(f, "expert_weights_scale", 1.0);
    const MetaValue* norm = f.get(key("expert_weights_norm"));
    g.routed_scale_normalized = norm != nullptr && norm->u != 0;

    // RoPE: one base per attention kind, which is why the kind has to be known per block.
    g.rope_dim = (int32_t) u(f, "rope.dimension_count");
    g.rope_theta_sparse = (float) fl_or(f, "rope.freq_base", 0);
    g.rope_theta_sliding = (float) fl_or(f, "rope.freq_base_swa", g.rope_theta_sparse);

    // MLA
    g.q_lora_rank = (int32_t) u_or(f, "attention.q_lora_rank", 0);
    g.kv_lora_rank = (int32_t) u_or(f, "attention.kv_lora_rank", 0);
    g.qk_head_dim = (int32_t) u_or(f, "attention.key_length_mla", 0);
    g.v_head_dim = (int32_t) u_or(f, "attention.value_length_mla", 0);
    g.kv_a_width = (int32_t) u_or(f, "attention.key_length", 0);
    g.sliding_window = (int32_t) u_or(f, "attention.sliding_window", 0);
    if (g.sliding_window <= 0) return "hy_v4_flash.attention.sliding_window is missing or zero";
    // One shared key per token carries the latent and the rope part together (head_count_kv = 1).
    if (g.kv_a_width != g.kv_lora_rank + g.rope_dim)
        return "hy_v4_flash.attention.key_length = " + std::to_string(g.kv_a_width) + " is not kv_lora_rank + " +
               "rope.dimension_count (" + std::to_string(g.kv_lora_rank) + " + " + std::to_string(g.rope_dim) +
               "); this model stores one shared key per token";

    // The compressed indexer's own widths (the sparse blocks').
    g.idx_heads = (int32_t) u_or(f, "attention.indexer.head_count", 0);
    g.idx_key_dim = (int32_t) u_or(f, "attention.indexer.key_length", 0);
    g.idx_kpool = (int32_t) u_or(f, "attention.indexer.kpool", 0);
    g.idx_top_k = (int32_t) u_or(f, "attention.indexer.top_k", 0);
    const MetaValue* tail = f.get(key("attention.indexer.kpool_select_tail"));
    g.idx_select_tail = tail != nullptr && tail->u != 0;

    // The expert activation's clamp. hy4-flash writes it as one value per block; the CPU kernel takes a single
    // clamp, so a file that disagrees between blocks is refused rather than quietly reduced to its first.
    g.swiglu_clamp = (float) fl_or(f, "swiglu_clamp_exp", 0);
    if (const MetaValue* v = f.get(key("swiglu_clamp_exp")); v != nullptr && !v->items.empty()) {
        g.swiglu_clamp = (float) v->items[0].num();   // one per block in the file; one number for this model
        for (const MetaValue& e : v->items)
            if (e.num() != (double) g.swiglu_clamp)
                return "hy_v4_flash.swiglu_clamp_exp is not one value for every block; the expert kernel takes "
                       "one clamp";
    }

    // ---- the per-block description. Two arrays describe it and both are read, because a file where they
    // disagree is describing two different models.
    const std::vector<uint64_t> heads = arr(f, "attention.head_count");
    const std::vector<uint64_t> swa = arr(f, "attention.sliding_window_pattern");
    const std::vector<uint64_t> idx = arr(f, "attention.indexer.types");
    g.layers.resize((size_t) g.block_count);
    int32_t ordinal = 0, n_sliding = 0, n_sparse = 0;
    for (int32_t b = 0; b < g.block_count; ++b) {
        LayerSpec& s = g.layers[(size_t) b];
        s.block_id = b;
        s.is_mtp = b >= g.trunk_blocks;
        s.attention = swa[(size_t) b] != 0 ? Attention::Sliding : Attention::Sparse;
        s.n_heads = (int32_t) heads[(size_t) b];
        (s.attention == Attention::Sliding ? n_sliding : n_sparse)++;
        // The indexer array covers the trunk only; the draft block is shaped like a sparse block and carries
        // one too (its weights have the tensors), so it is derived from its own attention kind.
        s.has_indexer = s.is_mtp ? s.attention == Attention::Sparse
                                 : idx[(size_t) b] != 0;
        if (!s.is_mtp && s.has_indexer != (s.attention == Attention::Sparse))
            return "hy_v4_flash: block " + std::to_string(b) + " has an indexer entry of " +
                   std::to_string(idx[(size_t) b]) + " but the sliding-window pattern makes it " +
                   (s.attention == Attention::Sliding ? "sliding" : "sparse");
        // Block 0..dense_blocks-1 have a plain FFN; every later block, the draft one included, routes experts.
        s.expert_ordinal = (!s.is_mtp && b < g.dense_blocks) ? -1 : ordinal++;
    }
    if (ordinal != g.n_expert_ordinals())
        return "hy_v4_flash: built " + std::to_string(ordinal) + " expert ordinals, expected " +
               std::to_string(g.n_expert_ordinals());
    if (n_sliding == 0 || n_sparse == 0)
        return "hy_v4_flash.attention.sliding_window_pattern makes every block " +
               std::string(n_sparse == 0 ? "sliding" : "sparse") + "; this model mixes the two";

    // ---- the Engram tables, as views of the one merged tensor
    g.engram_ngram_size = (int32_t) u(f, "ple.ngram_size");
    g.engram_heads_per_ngram = (int32_t) u(f, "ple.heads_per_ngram");
    g.engram_head_dim = (int32_t) u(f, "embedding_length_per_layer_input");
    g.engram_pad_token = (int32_t) u_or(f, "engram.pad_token_id", -1);
    g.engram_heads = (g.engram_ngram_size - 1) * g.engram_heads_per_ngram;

    const std::vector<uint64_t> ple_layers = arr(f, "ple.layers");
    const std::vector<uint64_t> offsets = arr(f, "ple.head_offsets");
    const std::vector<uint64_t> vocabs = arr(f, "ple.head_vocab_sizes");
    const std::vector<uint64_t> mults = arr(f, "ple.layer_multipliers");
    const size_t columns = (size_t) g.engram_heads;
    uint64_t base = 0;
    for (size_t t = 0; t < ple_layers.size(); ++t) {
        EngramTable e;
        e.block_id = (int32_t) ple_layers[t];
        e.row_base = base;
        for (size_t c = 0; c < columns; ++c) {
            const size_t i = t * columns + c;
            e.head_offsets.push_back(offsets[i]);
            e.head_bucket_counts.push_back(vocabs[i]);
            // The offsets are a running total across ALL the tables, so a column's offset is this table's base
            // plus the buckets before it. Checking that is what catches a converter that rebased them per
            // table: the hash would then land every table on the first one's rows.
            if (offsets[i] != base)
                return "hy_v4_flash.ple.head_offsets[" + std::to_string(i) + "] = " + std::to_string(offsets[i]) +
                       ", expected " + std::to_string(base) +
                       " (the offsets run on across the tables; they are not rebased per table)";
            base += vocabs[i];
        }
        for (int32_t o = 0; o < g.engram_ngram_size; ++o)
            e.multipliers.push_back(mults[t * (size_t) g.engram_ngram_size + (size_t) o]);
        e.rows = base - e.row_base;
        if (e.block_id >= 0 && e.block_id < g.trunk_blocks) g.layers[(size_t) e.block_id].engram_table = (int32_t) t;
        g.engram_tables.push_back(std::move(e));
    }
    g.engram_total_rows = base;
    // The hash puts a row id in an int32 (llama.cpp's engram_hash), and the converter refuses to write a table
    // past that, so a file whose tables outgrew it cannot be read the way the reference reads it.
    if (g.engram_total_rows > 0x7fffffffULL)
        return "hy_v4_flash: the Engram tables need " + std::to_string(g.engram_total_rows) +
               " rows, past the int32 row id the hash uses";

    // Validate the stored Engram row type and size before addressing the merged table.
    if (const TensorInfo* t = engram_tensor ? engram_tensor : f.find("per_layer_token_embd.weight")) {
        int elems = 0, bytes = 0;
        if (!block_geometry(t->type, elems, bytes))
            return std::string("per_layer_token_embd.weight has type ") + t->type_name() +
                   ", whose block geometry this engine does not know";
        if (elems <= 0 || g.engram_head_dim % elems != 0)
            return std::string("per_layer_token_embd.weight is ") + t->type_name() + ", whose block is " +
                   std::to_string(elems) + " values, but a row of this table is " +
                   std::to_string(g.engram_head_dim) + "; the type must have a block that divides the row "
                   "(block 1 or 32 for a 96-value row - every k-quant and i-quant has 256)";
        const uint64_t row = (uint64_t) (g.engram_head_dim / elems) * (uint64_t) bytes;
        if (row == 0 || row > 4096)
            return "per_layer_token_embd.weight rows are " + std::to_string(row) +
                   " B; the row reader takes 1..4096 (one page)";
        g.engram_table_type = t->type;
        g.engram_row_bytes = (uint32_t) row;
        // The tensor is [head_dim, rows] in GGUF order (dim 0 varies fastest), so its row count is the last
        // dimension. A mismatch against the bucket counts means the metadata and the payload disagree about
        // how many rows there are, and the hash would index past the end of the one that is shorter.
        const uint64_t have = t->shape.empty() ? 0 : t->shape.back();
        if (have != g.engram_total_rows)
            return "per_layer_token_embd.weight has " + std::to_string(have) + " rows, but the per-head bucket "
                   "counts sum to " + std::to_string(g.engram_total_rows);
        if (!t->shape.empty() && t->shape.front() != (uint64_t) g.engram_head_dim)
            return "per_layer_token_embd.weight rows are " + std::to_string(t->shape.front()) +
                   " wide, not the " + std::to_string(g.engram_head_dim) +
                   " that embedding_length_per_layer_input declares";
    }

    g.max_context = max_context > 0 ? max_context : (int32_t) u_or(f, "context_length", 0);
    return {};
}

std::string describe(const Geometry& g) {
    char b[2048];
    int n_sliding = 0, n_dense = 0;
    for (const LayerSpec& s : g.layers) {
        if (s.attention == Attention::Sliding) ++n_sliding;
        if (s.expert_ordinal < 0) ++n_dense;
    }
    std::snprintf(b, sizeof b,
                  "  blocks         %d trunk + %d draft; %d sliding / %d sparse; %d dense FFN of %d\n"
                  "  widths         hidden %d, residual %d (%d hyper-connection streams), vocab %d\n"
                  "  attention      q_lora %d, kv_lora %d, qk %d, v %d, one shared key of %d; window %d\n"
                  "  indexer        %d heads x %d, pool %d, top-k %d%s\n"
                  "  experts        %d x top-%d, subspace %d, ff %d, %d shared, %d ordinals; "
                  "%s scale %.3f, gate %s, SwiGLU clamp %.1f\n"
                  "  engram         %d tables x %d columns x %d; %d predecessors; %" PRIu64 " rows total\n"
                  "                 %s, %u B/row -> %.2f GB\n"
                  "  context        %d",
                  g.trunk_blocks, g.n_mtp_blocks, n_sliding, (int) g.layers.size() - n_sliding, n_dense,
                  g.dense_ff, g.hidden_dim, g.hc_width(), g.hc_mult, g.vocab_size, g.q_lora_rank,
                  g.kv_lora_rank, g.qk_head_dim, g.v_head_dim, g.kv_a_width, g.sliding_window, g.idx_heads,
                  g.idx_key_dim, g.idx_kpool, g.idx_top_k, g.idx_select_tail ? ", tail selected" : "",
                  g.n_experts, g.top_k, g.expert_dim, g.expert_ff, g.n_shared_experts, g.n_expert_ordinals(),
                  g.routed_scale_normalized ? "normalised," : "unnormalised,", (double) g.routed_scale,
                  g.gating_func == 2 ? "sigmoid" : "softmax", (double) g.swiglu_clamp,
                  (int) g.engram_tables.size(), g.engram_heads,
                  g.engram_head_dim, g.engram_predecessors(), g.engram_total_rows,
                  g.engram_row_bytes ? ggml_type_name(g.engram_table_type) : "type not in this shard",
                  g.engram_row_bytes,
                  (double) g.engram_total_rows * (double) g.engram_row_bytes / 1e9, g.max_context);
    return b;
}

}  // namespace strata::hy4_flash
