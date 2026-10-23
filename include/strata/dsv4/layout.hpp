// include/strata/dsv4/layout.hpp - DeepSeek V4-Flash's tensor layout table: every tensor of every layer type,
// by GGUF name, shape (dim 0 contiguous, as tools/gguf_reader.py prints them) and ggml type, plus the root
// tensors, plus the geometry read from a file's "deepseek4.*" metadata.
//
// The dsv4 counterpart of core/family_layout.hpp, family-scoped like core/geometry.hpp's sibling
// dsv4/geometry.hpp.  Header-only and GPU-free: the engine's loader, the probe (src/dsv4/dsv4_probe.cpp)
// and the tests share it.
//
// The authoritative inventory is strata::check_architecture_deepseek4 (artifact/gguf_reader.hpp), frozen
// against the real 164 GB dsv4-flash-nvfp4.gguf on 2026-10-20: the 26 core tensors per layer (20 named +
// the 6 per-expert scale vectors), the 3 NVFP4 expert mergers, the ratio-conditioned compressor and indexer
// sets, and the hash-layer tid2eid / routed-layer exp_probs_b mutual exclusion.  The guard pins shapes and
// types only where the kernels' contract is the tensor itself (the expert mergers, the head/embedding); the
// shapes and types below for the rest were read off that same file's tensor directory and agree with the
// guard's name list exactly (1586 tensors total: 43 layers + 6 root).
//
// Layer types (the split is derived, never read from a second list - geometry.hpp):
//   * Hash        layers 0-1:   26 core + 3 expert mergers + ffn_gate_tid2eid (I32).  No compressor.
//   * HashHybrid  layer 2:      the hash set PLUS the r=4 compressor and indexer stacks.
//   * FullR4      even 2..42:   26 core + 3 mergers + exp_probs_b + compressor(r=4) + indexer (6 tensors).
//   * CompressorR128 odd 3..41: 26 core + 3 mergers + exp_probs_b + compressor(r=128) - a 128-wide pooling
//                               block needs no indexer pass.
// On a hash layer (0-2) tid2eid routes the experts and exp_probs_b must NOT exist; from layer 3 on it is
// the reverse.  Layer 2 carries BOTH the hash routing table and the full r=4 attention stack.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/dsv4/geometry.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace strata::dsv4 {

/// The four layer types of the dsv4 trunk.  HashHybrid is layer 2 exactly: it keeps the hash routing table
/// AND grows the r=4 compressor/indexer stack, so it is its own kind rather than "FullR4 with a quirk".
enum class Dsv4LayerKind { Hash, HashHybrid, FullR4, CompressorR128 };

inline Dsv4LayerKind dsv4_layer_kind(int64_t layer) {
    if (layer < 2) return Dsv4LayerKind::Hash;
    if (layer == 2) return Dsv4LayerKind::HashHybrid;
    return layer % 2 == 0 ? Dsv4LayerKind::FullR4 : Dsv4LayerKind::CompressorR128;
}

inline const char* dsv4_layer_kind_name(Dsv4LayerKind k) {
    switch (k) {
    case Dsv4LayerKind::Hash: return "hash";
    case Dsv4LayerKind::HashHybrid: return "hash+DSA hybrid";
    case Dsv4LayerKind::FullR4: return "r=4 full";
    case Dsv4LayerKind::CompressorR128: return "r=128 compressor-only";
    }
    return "?";
}

/// ggml type ids as the GGUF directory carries them (ggml_type_name() prints them).
inline constexpr uint32_t kGgufF32 = 0;
inline constexpr uint32_t kGgufQ8_0 = 8;
inline constexpr uint32_t kGgufI32 = 26;
inline constexpr uint32_t kGgufBF16 = 30;
inline constexpr uint32_t kGgufNVFP4 = 40;

/// One tensor of the layout table: its blk.N. suffix (a layer's) or full name (a root's), its GGUF shape in
/// dim-0-contiguous order, and its ggml type id.
struct Dsv4TensorWant {
    std::string name;
    std::vector<int64_t> shape;
    uint32_t type;
};

/// The 29 tensors every trunk layer carries: the 26 the guard's core list names (20 weights + the 6
/// per-expert scale vectors) plus the 3 NVFP4 expert mergers it pins by shape.
inline std::vector<Dsv4TensorWant> dsv4_layer_core_tensors(const Dsv4Geometry& g) {
    const int64_t N = g.n_embd, H = g.hc_dim(), NE = g.n_expert, FF = g.n_ff;
    std::vector<Dsv4TensorWant> t;
    auto add = [&](const char* s, std::vector<int64_t> shape, uint32_t ty) {
        t.push_back({s, std::move(shape), ty});
    };
    // MLA attention
    add("attn_norm.weight", {N}, kGgufF32);
    add("attn_q_a.weight", {N, g.q_lora_rank}, kGgufQ8_0);
    add("attn_q_a_norm.weight", {g.q_lora_rank}, kGgufF32);
    add("attn_q_b.weight", {g.q_lora_rank, g.q_rows()}, kGgufQ8_0);
    add("attn_kv.weight", {N, g.kv_lora_rank}, kGgufQ8_0);          // the KV latent projection
    add("attn_kv_a_norm.weight", {g.kv_lora_rank}, kGgufF32);
    add("attn_output_a.weight", {N, g.o_rows()}, kGgufQ8_0);
    add("attn_output_b.weight", {g.o_rows(), N}, kGgufQ8_0);
    add("attn_sinks.weight", {g.n_head}, kGgufF32);
    // hyper-connections around attention and ffn: fn is [hc_dim, 2*hc*... ] as the file carries it
    add("hc_attn_fn.weight", {H, 24}, kGgufF32);
    add("hc_attn_base.weight", {24}, kGgufF32);
    add("hc_attn_scale.weight", {3}, kGgufF32);
    add("hc_ffn_fn.weight", {H, 24}, kGgufF32);
    add("hc_ffn_base.weight", {24}, kGgufF32);
    add("hc_ffn_scale.weight", {3}, kGgufF32);
    // MoE: router, the shared expert, then the routed experts (NVFP4 mergers + per-expert scales)
    add("ffn_norm.weight", {N}, kGgufF32);
    add("ffn_gate_inp.weight", {N, NE}, kGgufBF16);
    add("ffn_gate_shexp.weight", {N, FF}, kGgufQ8_0);
    add("ffn_down_shexp.weight", {FF, N}, kGgufQ8_0);
    add("ffn_up_shexp.weight", {N, FF}, kGgufQ8_0);
    add("ffn_gate_exps.weight", {N, FF, NE}, kGgufNVFP4);
    add("ffn_gate_exps.scale", {NE}, kGgufF32);
    add("ffn_gate_exps.input_scale", {NE}, kGgufF32);
    add("ffn_down_exps.weight", {FF, N, NE}, kGgufNVFP4);
    add("ffn_down_exps.scale", {NE}, kGgufF32);
    add("ffn_down_exps.input_scale", {NE}, kGgufF32);
    add("ffn_up_exps.weight", {N, FF, NE}, kGgufNVFP4);
    add("ffn_up_exps.scale", {NE}, kGgufF32);
    add("ffn_up_exps.input_scale", {NE}, kGgufF32);
    return t;
}

/// The compressor stack of a layer whose compress ratio is `ratio` (4 or 128): kv and gate project n_embd
/// into the compressed width (2*kv_lora at r=4, kv_lora at r=128), ape is the per-position embedding of one
/// pooling block, norm is over the KV latent.  Widths are the real file's; the geometry has no field for the
/// compressor width, so they are pinned here with the ratio.
inline std::vector<Dsv4TensorWant> dsv4_compressor_tensors(const Dsv4Geometry& g, int64_t ratio) {
    const int64_t N = g.n_embd;
    const int64_t W = ratio == 4 ? 2 * g.kv_lora_rank : g.kv_lora_rank;  // 1024 at r=4, 512 at r=128
    return {
        {"attn_compressor_kv.weight", {N, W}, kGgufBF16},
        {"attn_compressor_gate.weight", {N, W}, kGgufBF16},
        {"attn_compressor_ape.weight", {W, ratio}, kGgufF32},
        {"attn_compressor_norm.weight", {g.kv_lora_rank}, kGgufF32},
    };
}

/// The DSA indexer stack of an r=4 layer: the query-side projection pair, then the indexer's own compressor
/// (kv/gate into 2*idx_key_dim, ape over the 4-wide pooling block, norm over idx_key_dim).
inline std::vector<Dsv4TensorWant> dsv4_indexer_tensors(const Dsv4Geometry& g) {
    const int64_t N = g.n_embd;
    return {
        {"indexer.proj.weight", {N, g.idx_heads}, kGgufBF16},
        {"indexer.attn_q_b.weight", {g.q_lora_rank, g.idx_heads * g.idx_key_dim}, kGgufQ8_0},
        {"indexer_compressor_kv.weight", {N, 2 * g.idx_key_dim}, kGgufBF16},
        {"indexer_compressor_gate.weight", {N, 2 * g.idx_key_dim}, kGgufBF16},
        {"indexer_compressor_ape.weight", {2 * g.idx_key_dim, 4}, kGgufF32},
        {"indexer_compressor_norm.weight", {g.idx_key_dim}, kGgufF32},
    };
}

/// Every tensor of trunk layer `layer`, in table order: the core set, then the kind's extras (hash routing
/// table or router bias, compressor, indexer).
inline std::vector<Dsv4TensorWant> dsv4_layer_tensors(const Dsv4Geometry& g, int64_t layer) {
    std::vector<Dsv4TensorWant> t = dsv4_layer_core_tensors(g);
    auto append = [&](std::vector<Dsv4TensorWant> more) {
        t.insert(t.end(), std::make_move_iterator(more.begin()), std::make_move_iterator(more.end()));
    };
    // the hash/routed router split is mutually exclusive, and layer 2 is the hash side of it
    if (dsv4_is_hash_layer(layer))
        // tid2eid: the token-id -> expert-id hash table, one row per routed expert slot over the vocabulary
        append({{"ffn_gate_tid2eid.weight", {g.n_expert_used, g.n_vocab}, kGgufI32}});
    else
        append({{"exp_probs_b.bias", {g.n_expert}, kGgufF32}});
    const int64_t r = dsv4_compress_ratio(layer);
    if (r) append(dsv4_compressor_tensors(g, r));
    if (dsv4_has_indexer(layer)) append(dsv4_indexer_tensors(g));
    return t;
}

/// The tensors outside the layers: embedding and head (Q8_0, shapes pinned by the guard), the final norm and
/// the output hyper-connection's fn/base/scale.
inline std::vector<Dsv4TensorWant> dsv4_root_tensors(const Dsv4Geometry& g) {
    return {
        {"token_embd.weight", {g.n_embd, g.n_vocab}, kGgufQ8_0},
        {"output.weight", {g.n_embd, g.n_vocab}, kGgufQ8_0},
        {"output_norm.weight", {g.n_embd}, kGgufF32},
        {"output_hc_fn.weight", {g.hc_dim(), g.hc}, kGgufF32},
        {"output_hc_base.weight", {g.hc}, kGgufF32},
        {"output_hc_scale.weight", {1}, kGgufF32},
    };
}

/// Fills `g` from the metadata of the GGUF `f` (arch "deepseek4").  False with `err` when the architecture
/// guard refuses the file - the guard pins every integer/float key to the compiled value, so the reads below
/// cannot disagree with kDsv4CompiledGeometry on any file that passes; they exist so the struct is filled
/// from the file rather than assumed.  kv_lora_rank has no metadata key: it is read from blk.0.attn_kv's
/// shape, which IS the KV latent projection.
inline bool dsv4_geometry_from_gguf(const strata::GgufFile& f, Dsv4Geometry& g, std::string& err) {
    err = strata::check_architecture_deepseek4(f);
    if (!err.empty()) return false;
    g = Dsv4Geometry{};
    const std::string p = "deepseek4.";
    auto num = [&](const char* key, int64_t& dst) {
        if (const strata::MetaValue* v = f.get(p + key))
            if (v->is_num()) dst = (int64_t) v->u;
    };
    auto fnum = [&](const char* key, double& dst) {
        if (const strata::MetaValue* v = f.get(p + key))
            if (v->is_num()) dst = v->num();
    };
    num("block_count", g.n_layers);
    num("embedding_length", g.n_embd);
    num("embedding_length_out", g.n_embd_out);
    num("attention.head_count", g.n_head);
    num("attention.key_length", g.head_dim);
    num("attention.q_lora_rank", g.q_lora_rank);
    num("rope.dimension_count", g.n_rot);
    num("attention.sliding_window", g.sliding_window);
    num("attention.output_group_count", g.o_groups);
    num("attention.output_lora_rank", g.o_lora_rank);
    num("attention.indexer.head_count", g.idx_heads);
    num("attention.indexer.key_length", g.idx_key_dim);
    num("attention.indexer.top_k", g.idx_top_k);
    num("hyper_connection.count", g.hc);
    num("hyper_connection.sinkhorn_iters", g.hc_sinkhorn_iters);
    num("expert_count", g.n_expert);
    num("expert_used_count", g.n_expert_used);
    num("expert_shared_count", g.n_expert_shared);
    num("expert_feed_forward_length", g.n_ff);
    num("hash_layer_count", g.n_hash_layers);
    num("rope.scaling.original_context_length", g.yarn_orig_ctx);
    fnum("attention.layer_norm_rms_epsilon", g.rms_eps);
    fnum("hyper_connection.epsilon", g.hc_eps);
    fnum("rope.freq_base", g.rope_theta);
    fnum("attention.compress_rope_freq_base", g.compress_rope_theta);
    fnum("rope.scaling.factor", g.yarn_factor);
    fnum("rope.scaling.yarn_beta_fast", g.yarn_beta_fast);
    fnum("rope.scaling.yarn_beta_slow", g.yarn_beta_slow);
    fnum("expert_weights_scale", g.route_scale);
    // the KV latent width: from the shape of any layer's attn_kv (the guard required it on every layer)
    if (const strata::TensorInfo* t = f.find("blk.0.attn_kv.weight")) {
        if (t->shape.size() != 2 || t->shape[0] != (uint64_t) g.n_embd) {
            err = "blk.0.attn_kv.weight is not [embedding_length, kv_lora_rank]";
            return false;
        }
        g.kv_lora_rank = (int64_t) t->shape[1];
    }
    // swiglu clamps: per-layer f32 arrays, the file carries 10.0 on every layer of both kinds
    for (const char* key : {"swiglu_clamp_exp", "swiglu_clamp_shexp"}) {
        const strata::MetaValue* v = f.get(p + key);
        if (!v || v->type != strata::MetaType::ARRAY || v->count != (uint64_t) g.n_layers) {
            err = p + std::string(key) + " must list one entry per layer";
            return false;
        }
        for (const strata::MetaValue& e : v->items)
            if (e.num() != 10.0) {
                err = p + std::string(key) + " carries a value != 10.0, the dsv4 kernels clamp at 10";
                return false;
            }
        g.swiglu_clamp = 10.0;
    }
    if (const strata::MetaValue* v = f.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY && v->count > 0) g.n_vocab = (int64_t) v->count;
    return true;
}

/// Checks the GGUF `f` (metadata shard) against the layout table of `g`: every table entry present with its
/// shape and type, and no tensor in the file the table does not name (the dsv4 table is complete - a main
/// model file holds exactly the trunk and the roots).  `checked` counts the tensors verified.
inline bool dsv4_check_gguf_layout(const strata::GgufFile& f, const Dsv4Geometry& g, std::string& err,
                                   size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const std::string& name, const std::vector<int64_t>& shape, uint32_t type) {
        named.insert(name);
        const strata::TensorInfo* t = f.find(name);
        if (!t) {
            err = "missing " + name;
            return false;
        }
        bool same = t->shape.size() == shape.size();
        for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == (uint64_t) shape[i];
        if (!same || t->type != type) {
            std::string got, want;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
            for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
            err = name + " is [" + got + "] " + t->type_name() + ", the dsv4 layout wants [" + want + "] " +
                  strata::ggml_type_name(type);
            return false;
        }
        ++n;
        return true;
    };
    for (const Dsv4TensorWant& w : dsv4_root_tensors(g))
        if (!check(w.name, w.shape, w.type)) return false;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const Dsv4TensorWant& w : dsv4_layer_tensors(g, l))
            if (!check(b + w.name, w.shape, w.type)) return false;
    }
    for (const strata::TensorInfo& t : f.tensors())
        if (!named.count(t.name)) {
            err = t.name + " is not in the dsv4 layout";
            return false;
        }
    if (checked) *checked = n;
    return true;
}

}  // namespace strata::dsv4
