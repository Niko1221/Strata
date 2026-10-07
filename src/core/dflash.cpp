// src/core/dflash.cpp - the DFlash drafter: the artifact (above) and the standalone block
// forward.  Semantics: docs/DFLASH.md.  The first implementation is eager (no captured graphs),
// BF16 activations through the bf16 GEMV path, one GPU, greedy; correctness before speed.
#include "strata/core/dflash.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/rope.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace strata::core {

using strata::GgufFile;
using strata::MetaValue;
using strata::TensorInfo;

namespace {

const MetaValue* meta(const GgufFile& f, const std::vector<const char*>& keys) {
    for (const char* k : keys)
        if (const MetaValue* v = f.get(k)) return v;
    return nullptr;
}

bool meta_i64(const GgufFile& f, const std::vector<const char*>& keys, int64_t* out) {
    if (const MetaValue* v = meta(f, keys)) {
        if (!v->is_num()) return false;
        *out = (int64_t) v->num();
        return true;
    }
    return false;
}

bool meta_f64(const GgufFile& f, const std::vector<const char*>& keys, double* out) {
    if (const MetaValue* v = meta(f, keys)) {
        if (!v->is_num()) return false;
        *out = v->num();
        return true;
    }
    return false;
}

bool meta_flag(const GgufFile& f, const std::vector<const char*>& keys, bool* out) {
    if (const MetaValue* v = meta(f, keys)) {
        if (v->type == strata::MetaType::STRING) {
            *out = v->s == "true" || v->s == "1";
            return true;
        }
        if (v->is_num()) {
            *out = v->u != 0;
            return true;
        }
    }
    return false;
}

std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r + "]";
}

bool shape_is(const TensorInfo& t, std::vector<uint64_t> want) {
    return t.shape == want;
}

}  // namespace

bool DFlashArtifact::require(const char* canon, std::vector<const char*> aliases,
                             std::vector<uint64_t> shape, DFlashTensor* out, std::string& err) {
    const TensorInfo* found = nullptr;
    for (const char* name : aliases) {
        if (const TensorInfo* t = file_->find(name)) {
            if (found) {
                err = std::string("dflash: tensor '") + canon + "' is present under two names ('" + found->name +
                      "' and '" + name + "'); the artifact must pick one family";
                return false;
            }
            found = t;
        }
    }
    if (!found) {
        err = std::string("dflash: missing required tensor '") + canon + "' (looked for";
        for (const char* name : aliases) err += std::string(" '") + name + "',";
        err += " in " + path_ + ")";
        return false;
    }
    if (found->type != 30) {   // GGML_TYPE_BF16
        err = std::string("dflash: tensor '") + found->name + "' is " + found->type_name() +
              "; this implementation reads BF16 DFlash GGUFs only";
        return false;
    }
    if (!shape_is(*found, shape)) {
        err = std::string("dflash: tensor '") + found->name + "' has shape " + shape_str(found->shape) +
              ", expected " + shape_str(shape) + " for '" + canon + "'";
        return false;
    }
    const uint64_t payload = file_->file_size() - file_->data_start();
    const uint64_t bytes = strata::tensor_payload_bytes(*found);
    if (bytes == 0 || found->offset > payload || bytes > payload - found->offset) {
        err = std::string("dflash: tensor '") + found->name + "' lies outside its file's data section";
        return false;
    }
    out->name = canon;
    out->file_name = found->name;
    // 1-D norms ride as one row of `cols` values; matrices are [cols, rows] (ne0 fastest)
    out->rows = (int64_t) found->shape.size() > 1 ? (int64_t) found->shape[1] : 1;
    out->cols = (int64_t) found->shape[0];
    out->d = nullptr;
    tensors_.push_back(*out);
    return true;
}

const DFlashTensor* DFlashArtifact::tensor(const std::string& name) const {
    for (const auto& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

const uint8_t* DFlashArtifact::host_data(const DFlashTensor& t) const {
    for (const auto& info : file_->tensors())
        if (info.name == t.file_name) return file_->tensor_data(info);
    return nullptr;
}

bool DFlashArtifact::open(const std::string& path, std::string& err) {
    path_ = path;
    try {
        file_ = std::make_unique<GgufFile>(path);
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
    const GgufFile& f = *file_;

    if (const MetaValue* arch = f.get("general.architecture")) {
        if (arch->s != "dflash") {
            err = "dflash: general.architecture is '" + arch->s + "', a DFlash artifact declares 'dflash'";
            return false;
        }
    } else {
        err = "dflash: missing general.architecture";
        return false;
    }

    DFlashGeometry& g = geom_;
    struct Num {
        int64_t* out;
        std::vector<const char*> keys;
        const char* what;
        bool required;
    };
    const Num nums[] = {
        {&g.hidden, {"dflash.embedding_length", "qwen3.embedding_length", "qwen3_dflash.embedding_length"}, "embedding_length", true},
        {&g.layers, {"dflash.block_count", "qwen3.block_count", "qwen3_dflash.block_count"}, "block_count", true},
        {&g.n_head, {"dflash.attention.head_count", "qwen3.attention.head_count"}, "attention.head_count", true},
        {&g.n_head_kv, {"dflash.attention.head_count_kv", "qwen3.attention.head_count_kv"}, "attention.head_count_kv", true},
        {&g.intermediate, {"dflash.feed_forward_length", "qwen3.feed_forward_length"}, "feed_forward_length", true},
        {&g.vocab, {"dflash.vocab_size", "qwen3.vocab_size"}, "vocab_size", true},
        {&g.block_size, {"dflash.block_size"}, "block_size", true},
        {&g.head_dim, {"dflash.attention.key_length", "qwen3.attention.key_length"}, "attention.key_length", false},
        {&g.mask_token_id, {"dflash.mask_token_id"}, "mask_token_id", false},
        {&g.markov_rank, {"dflash.markov_rank"}, "markov_rank", false},
        {&g.selector_top_k, {"dflash.selector_top_k"}, "selector_top_k", false},
    };
    for (const auto& n : nums) {
        if (!meta_i64(f, n.keys, n.out)) {
            if (n.required) {
                err = std::string("dflash: missing metadata key") + (n.keys.size() > 1 ? " (any of" : "") + " " +
                      n.keys[0] + (n.keys.size() > 1 ? "...)" : "") + " in " + path_;
                return false;
            }
            if (n.out == &g.head_dim) g.head_dim = 0;   // derived below from hidden / n_head
        }
    }
    if (g.head_dim <= 0) {
        if (g.hidden <= 0 || g.n_head <= 0 || g.hidden % g.n_head != 0) {
            err = "dflash: cannot derive the head dim from embedding_length / attention.head_count";
            return false;
        }
        g.head_dim = g.hidden / g.n_head;
    }
    (void) meta_f64(f, {"dflash.rope.frequency_base", "qwen3.rope.frequency_base"}, &g.rope_theta);
    meta_flag(f, {"dflash.sample_from_anchor"}, &g.sample_from_anchor);
    meta_flag(f, {"dflash.attention.causal", "dflash.causal"}, &g.causal);
    meta_flag(f, {"dflash.has_confidence_head"}, &g.confidence_head);

    if (const MetaValue* tl = meta(f, {"dflash.target_layers", "target_layers"})) {
        if (tl->type != strata::MetaType::ARRAY || tl->elem == strata::MetaType::STRING) {
            err = "dflash: target_layers must be an integer array";
            return false;
        }
        for (const auto& e : tl->items) g.target_layers.push_back((int32_t) e.num());
        // The reader keeps a 64-item sample; a tap list that long is not a tap list.
        if ((int64_t) g.target_layers.size() != (int64_t) tl->count) {
            err = "dflash: implausibly long target_layers array";
            return false;
        }
    } else {
        err = "dflash: missing dflash.target_layers (the taps this drafter was trained against)";
        return false;
    }

    // Structural sanity before any tensor is resolved.
    if (g.hidden <= 0 || g.layers <= 0 || g.layers > 64 || g.n_head <= 0 || g.n_head_kv <= 0 ||
        g.head_dim <= 0 || g.intermediate <= 0 || g.vocab <= 0 || g.block_size <= 0 || g.block_size > 128) {
        err = "dflash: implausible geometry (hidden=" + std::to_string(g.hidden) +
              " layers=" + std::to_string(g.layers) + " heads=" + std::to_string(g.n_head) + "/" +
              std::to_string(g.n_head_kv) + " head_dim=" + std::to_string(g.head_dim) +
              " intermediate=" + std::to_string(g.intermediate) + " vocab=" + std::to_string(g.vocab) +
              " block_size=" + std::to_string(g.block_size) + ")";
        return false;
    }
    for (size_t i = 0; i < g.target_layers.size(); ++i) {
        if (g.target_layers[i] < 0 || (i && g.target_layers[i] <= g.target_layers[i - 1])) {
            err = "dflash: target_layers must be strictly increasing and non-negative, got [" +
                  std::to_string(g.target_layers.front()) + ", ..., " + std::to_string(g.target_layers.back()) + "]";
            return false;
        }
    }
    if (!g.sample_from_anchor) {
        err = "dflash: this artifact declares the generic 1+N fill-in layout "
              "(dflash.sample_from_anchor=false); this implementation supports only the DeepSpec "
              "anchor layout (query_zero_predicts_next)";
        return false;
    }
    if (g.causal) {
        err = "dflash: this artifact declares causal block attention; the DeepSpec drafter is trained "
              "non-causal (dflash.attention.causal=false)";
        return false;
    }
    if (g.markov_rank > 0 || g.confidence_head) {
        err = "dflash: this artifact carries DSpark extras (Markov head / confidence gate); "
              "this implementation supports plain DeepSpec DFlash only";
        return false;
    }
    if (g.selector_top_k > 0) {
        err = "dflash: this artifact carries a DFlash2 selector (dflash.selector_top_k=" +
              std::to_string(g.selector_top_k) + "); DFlash2 is out of scope";
        return false;
    }

    // The embedding and the LM head must come from the target; an artifact that ships either is
    // not the stripped DeepSpec layout this implementation binds.
    for (const auto& t : f.tensors()) {
        if (t.name == "token_embd.weight" || t.name == "output.weight") {
            err = "dflash: the artifact ships its own '" + t.name +
                  "'; the DeepSpec drafter strips both and binds them from the target at load";
            return false;
        }
        for (const char* bad : {"selector", "conv_base", "conv_proj", "markov", "conf_proj", "d2t"}) {
            if (t.name.find(bad) != std::string::npos) {
                err = "dflash: unexpected tensor '" + t.name + "' (DFlash2/DSpark extension); " +
                      "this implementation reads plain DeepSpec DFlash artifacts";
                return false;
            }
        }
    }

    // The 58-tensor inventory, each resolved under either naming family and shape-checked.
    tensors_.clear();
    weight_bytes_ = 0;
    auto add = [&](uint64_t bytes) { weight_bytes_ += bytes; };
    DFlashTensor t;
    const int64_t H = g.hidden, D = g.head_dim, I = g.intermediate;
    const int64_t Q = g.n_head * D, KV = g.n_head_kv * D, F = g.fusion_in();
    if (!require("fc", {"fc.weight"}, {(uint64_t) F, (uint64_t) H}, &t, err)) return false;
    add((uint64_t) F * H * 2);
    if (!require("hidden_norm", {"enc.output_norm.weight", "hidden_norm.weight"}, {(uint64_t) H}, &t, err)) return false;
    add((uint64_t) H * 2);
    if (!require("norm", {"output_norm.weight", "norm.weight"}, {(uint64_t) H}, &t, err)) return false;
    add((uint64_t) H * 2);
    for (int64_t l = 0; l < g.layers; ++l) {
        const uint64_t u = (uint64_t) l;
        struct M {
            const char* canon;
            std::vector<const char*> llama, hf;
            std::vector<uint64_t> shape;
            int64_t elems;
        };
        const M ms[] = {
            {"input_layernorm", {"blk.%d.attn_norm.weight"}, {"layers.%d.input_layernorm.weight"}, {(uint64_t) H}, H},
            {"post_attention_layernorm", {"blk.%d.ffn_norm.weight"}, {"layers.%d.post_attention_layernorm.weight"}, {(uint64_t) H}, H},
            {"self_attn.q_proj", {"blk.%d.attn_q.weight"}, {"layers.%d.self_attn.q_proj.weight"}, {(uint64_t) H, (uint64_t) Q}, H * Q},
            {"self_attn.k_proj", {"blk.%d.attn_k.weight"}, {"layers.%d.self_attn.k_proj.weight"}, {(uint64_t) H, (uint64_t) KV}, H * KV},
            {"self_attn.v_proj", {"blk.%d.attn_v.weight"}, {"layers.%d.self_attn.v_proj.weight"}, {(uint64_t) H, (uint64_t) KV}, H * KV},
            {"self_attn.o_proj", {"blk.%d.attn_output.weight"}, {"layers.%d.self_attn.o_proj.weight"}, {(uint64_t) Q, (uint64_t) H}, Q * H},
            {"self_attn.q_norm", {"blk.%d.attn_q_norm.weight"}, {"layers.%d.self_attn.q_norm.weight"}, {(uint64_t) D}, D},
            {"self_attn.k_norm", {"blk.%d.attn_k_norm.weight"}, {"layers.%d.self_attn.k_norm.weight"}, {(uint64_t) D}, D},
            {"mlp.gate_proj", {"blk.%d.ffn_gate.weight"}, {"layers.%d.mlp.gate_proj.weight"}, {(uint64_t) H, (uint64_t) I}, H * I},
            {"mlp.up_proj", {"blk.%d.ffn_up.weight"}, {"layers.%d.mlp.up_proj.weight"}, {(uint64_t) H, (uint64_t) I}, H * I},
            {"mlp.down_proj", {"blk.%d.ffn_down.weight"}, {"layers.%d.mlp.down_proj.weight"}, {(uint64_t) I, (uint64_t) H}, I * H},
        };
        for (const auto& m : ms) {
            std::vector<const char*> aliases;
            std::string n1, n2;
            for (const char* a : m.llama) {
                char buf[128];
                std::snprintf(buf, sizeof buf, a, (int) l);
                n1 = buf;
                aliases.push_back(n1.c_str());
            }
            for (const char* a : m.hf) {
                char buf[128];
                std::snprintf(buf, sizeof buf, a, (int) l);
                n2 = buf;
                aliases.push_back(n2.c_str());
            }
            char canon[128];
            std::snprintf(canon, sizeof canon, "layers.%d.%s", (int) l, m.canon);
            if (!require(canon, aliases, m.shape, &t, err)) return false;
            add((uint64_t) m.elems * 2);
        }
    }
    // Anything else in the directory is a tensor this implementation does not know: refuse rather
    // than guess whether it matters.  (An unchosen alias for a resolved tensor was already refused
    // as a two-name conflict inside require().)
    for (const auto& info : f.tensors()) {
        bool known = false;
        for (const auto& t2 : tensors_)
            if (info.name == t2.file_name) known = true;
        if (!known) {
            err = "dflash: unknown tensor '" + info.name + "' in the artifact";
            return false;
        }
    }
    return true;
}

bool DFlashArtifact::validate_supported(const DFlashGeometry& g, std::string& err) {
    struct Fixed {
        int64_t got, want;
        const char* what;
    };
    const Fixed fixed[] = {
        {g.hidden, 2560, "embedding_length"},
        {g.layers, 5, "block_count"},
        {g.n_head, 24, "attention.head_count"},
        {g.n_head_kv, 2, "attention.head_count_kv"},
        {g.head_dim, 256, "attention.key_length"},
        {g.intermediate, 7680, "feed_forward_length"},
    };
    for (const auto& f : fixed) {
        if (f.got != f.want) {
            err = std::string("dflash: ") + f.what + " = " + std::to_string(f.got) + "; the runtime's " +
                  "geometry-specific fast path implements the published Flash-Next drafter (" +
                  std::to_string(f.want) + ") only";
            return false;
        }
    }
    if ((int64_t) g.target_layers.size() != 5) {
        err = "dflash: target_layers holds " + std::to_string(g.target_layers.size()) +
              " taps; the runtime implements five-tap drafters only";
        return false;
    }
    if (g.rope_theta <= 0) {
        err = "dflash: rope.frequency_base missing or non-positive";
        return false;
    }
    return true;
}


namespace {

constexpr float kEps = 1e-6f;   // the architecture's rms_norm eps (config.json), not stored

/// A tiny bump allocator over one device block, the MTP carve's shape.
struct Bump {
    Bump(uint8_t* base) : base_(base) {}
    template <class T> T* take(int64_t n) {
        return reinterpret_cast<T*>(base_ + (at_ += sizeof(T) * (size_t) n) - sizeof(T) * (size_t) n);
    }
    uint8_t* base_;
    size_t at_ = 0;
};

}  // namespace

void dflash_identity_fill(int32_t* host, int rows, int64_t cap) {
    for (int r = 0; r < rows; ++r)
        for (int64_t i = 0; i < cap; ++i) host[(size_t) r * (size_t) cap + (size_t) i] = (int32_t) i;
}

// ============================================================================================
// The runtime: weights, the drafter's own K/V pools, the fusion (context cells) and the block
// forward.  Eager, one stream, greedy.  All semantics: docs/DFLASH.md.
// ============================================================================================

namespace {

uint64_t mapped_bytes(int64_t n) { return ((uint64_t) n + 63) & ~uint64_t(63); }

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}


/// Mapped pinned host memory + its device alias (the MTP staging's shape, local here).
bool dflash_mapped(int64_t n, void** host, void** dev) {
    void* h = nullptr;
    if (cudaHostAlloc(&h, mapped_bytes(n), cudaHostAllocMapped) != cudaSuccess) return false;
    if (cudaHostGetDevicePointer(dev, h, 0) != cudaSuccess) { cudaFreeHost(h); return false; }
    *host = h;
    return true;
}

}  // namespace

uint64_t DFlashDrafter::bind_bytes(int64_t n_vocab, int max_t) const {
    // The draft logits (max_t rows over the full vocabulary) plus the block's scratch (a few MiB
    // of q/k/v, attention and MLP intermediates for at most 8 rows).
    return (uint64_t) max_t * (uint64_t) n_vocab * 4 + ((uint64_t) 24 << 20);
}

bool DFlashDrafter::upload(const ModelGeometry& target_g, SessionState& ss, int device, int64_t window,
                           std::string& err) {
    const DFlashGeometry& dg = artifact_.geom();
    device_ = device;
    mask_ = dg.mask_token_id;
    if (cudaSetDevice(device_) != cudaSuccess) { err = "dflash: no such device"; return false; }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "dflash: cannot create its stream";
        return false;
    }

    // ---- the weights: one device block, the GGUF layout preserved (row-major rows of bf16).
    // Earlier phases may have left a sticky error in the per-thread state: start clean.
    cudaGetLastError();
    if (cudaMalloc(&w_, artifact_.weight_bytes()) != cudaSuccess) {
        err = "dflash: the BF16 weights do not fit in VRAM";
        return false;
    }
    vram_ += artifact_.weight_bytes();
    size_t at = 0;
    for (const auto& t : artifact_.tensors()) {
        wt_.push_back({t.name, w_ + at / 2});
        if (cudaMemcpyAsync(w_ + at / 2, artifact_.host_data(t), (size_t) t.rows * t.cols * 2,
                            cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = std::string("dflash: the weight upload of '") + t.name + "' failed: " +
                  cudaGetErrorString(cudaGetLastError());
            return false;
        }
        at += (size_t) t.rows * t.cols * 2;
    }
    // The rms_norm gammas run as F32 device vectors (the kernel's contract); widen the small ones
    // at upload.  Qwen3 norms apply `y * w` with NO Gemma +1.
    auto is_norm = [](const std::string& n) {
        return n == "hidden_norm" || n == "norm" || n.find("layernorm") != std::string::npos ||
               n.find("q_norm") != std::string::npos || n.find("k_norm") != std::string::npos;
    };
    for (const auto& t : artifact_.tensors()) {
        if (!is_norm(t.name)) continue;
        float* f32d = nullptr;
        if (cudaMalloc(&f32d, (size_t) t.rows * t.cols * 4) != cudaSuccess) {
            err = "dflash: the norm weights do not fit";
            return false;
        }
        vram_ += (size_t) t.rows * t.cols * 4;
        const uint16_t* host = (const uint16_t*) artifact_.host_data(t);
        std::vector<float> wide((size_t) t.rows * t.cols);
        for (int64_t i = 0; i < t.rows * t.cols; ++i) {
            uint32_t bits = (uint32_t) host[i] << 16;
            std::memcpy(&wide[(size_t) i], &bits, 4);
        }
        if (cudaMemcpyAsync(f32d, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the norm weight upload failed";
            return false;
        }
        wf_.push_back({t.name, f32d});
    }

    // ---- the drafter's own K/V: the target's QSA pool shapes, but only `layers` pools.  The pool
    // geometry is the target's with n_layers shrunk so is_qsa_layer counts exactly 5 pools
    // (layers 3, 7, 11, ... under the interval-4 layout).
    pool_g_ = target_g;
    if (target_g.qsa_interval <= 0 || dg.layers * target_g.qsa_interval > target_g.n_layers) {
        err = "dflash: cannot size the draft K/V pools from the target geometry";
        return false;
    }
    pool_g_.n_layers = dg.layers * target_g.qsa_interval;
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * dg.layers + 64 : 0;
    const bool kv_int8_was = strata::core::qsa_kv_int8();
    const bool kv_hybrid_was = strata::core::qsa_kv_hybrid();
    strata::core::qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) strata::core::qsa_set_kv_int8(true);   // K8V4 targets run their drafter INT8
    const uint64_t sb = strata::core::qsa_state_bytes(pool_g_, max_cells, false, ring);
    // ONE STATE PER DRAFT LAYER: a QsaState is a single layer's pools, and the five draft layers
    // must not share them (a shared pool made every layer attend layer 0's K/V).
    if (pool_g_.n_qsa_layers() != dg.layers) {
        err = "dflash: the pool geometry does not give one state per draft layer";
        return false;
    }
    st_.assign((size_t) dg.layers, QsaState{});
    arenas_.assign((size_t) dg.layers, nullptr);
    for (int l = 0; l < (int) dg.layers; ++l) {
        if (cudaMalloc(&arenas_[(size_t) l], sb) != cudaSuccess) {
            strata::core::qsa_set_kv_int8(kv_int8_was);
            strata::core::qsa_set_kv_hybrid(kv_hybrid_was);
            err = "dflash: the draft K/V states do not fit in VRAM";
            return false;
        }
        if (strata::core::qsa_state_init(pool_g_, max_cells, arenas_[(size_t) l], st_[(size_t) l],
                                         &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
            strata::core::qsa_set_kv_int8(kv_int8_was);
            strata::core::qsa_set_kv_hybrid(kv_hybrid_was);
            err = "dflash: the draft K/V state init failed";
            return false;
        }
        strata::core::qsa_state_zero(st_[(size_t) l], pool_g_, nullptr);
        vram_ += sb;
    }
    cudaDeviceSynchronize();

    // ---- buffers: at most 8 rows ride through the forward at once
    max_rows_ = 8;
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    shapes_ = strata::core::shapes_of(pool_g_);
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, shapes_);
    const int64_t R = max_rows_, N = dg.hidden, F = dg.fusion_in(), I = dg.intermediate;
    const int64_t Q = dg.n_head * dg.head_dim, KVW = dg.n_head_kv * dg.head_dim;
    if (!dflash_mapped(R * 4 + 64, (void**) &h_out_, (void**) &out_) ||
        cudaHostAlloc(&h_tok_, (size_t)(R + 4) * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_step_, (size_t) R * 16, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_pos_, (size_t) R * dg.n_head * 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "dflash: mapped staging failed";
        return false;
    }
    auto take = [&](size_t n, void** p) -> bool {
        if (cudaMalloc(p, n) != cudaSuccess) {
            err = "dflash: the drafter buffers do not fit in VRAM";
            return false;
        }
        vram_ += n;
        return true;
    };
    const int64_t W16 = std::max(N, std::max(I, Q));   // the widest bf16 activation (MLP down's input)
    bool ok = take((R + 4) * 4, (void**) &tok_) && take(R * 4 * 4, (void**) &step_) &&
              take(R * (int64_t) dg.n_head * 4, (void**) &pos_) && take(R * cap_ * 4, (void**) &ident_) &&
              take(R * F * 2, (void**) &tapin_) && take(R * W16 * 2, (void**) &xn16_) &&
              take(R * W16 * 2, (void**) &attn16_) && take(R * F * 4, (void**) &tapf_) &&
              take(R * N * 4, (void**) &emb_) && take(R * N * 4, (void**) &h_) &&
              take(R * N * 4, (void**) &xn_) && take(R * N * 4, (void**) &ctx_) &&
              take(R * Q * 4, (void**) &q_) &&
              take(R * KVW * 4, (void**) &kc_) && take(R * KVW * 4, (void**) &vc_) &&
              take(R * Q * 4, (void**) &attn_) && take(R * N * 4, (void**) &bo_) &&
              take(R * I * 4, (void**) &gate_) && take(R * I * 4, (void**) &up_) &&
              take(R * dg.vocab * 4, (void**) &logits_) && take(N * 4, (void**) &mask_row_) &&
              take(strata::kernels::argmax_rows_scratch_bytes((int) R), (void**) &arg_scratch_) &&
              take((size_t) strata::kernels::native_q8_1_bytes((int) N, (int) R), (void**) &xq_) &&
              take((size_t) max_rows_ * (size_t) attn_scratch_floats_ * 4, (void**) &attn_scratch_);
    if (!ok) return false;
    // the identity cell selection, once, for EVERY query row: the batch attention offsets the
    // table by row * cap (ids += blockIdx.z * cap), so rows 1..K-1 read garbage when only row 0
    // is initialized - the constant-mask-row symptom.  [r][i] = i, duplicated per row on purpose
    // (no optimization before correctness).
    {
        std::vector<int32_t> id_host((size_t) max_rows_ * (size_t) cap_);
        dflash_identity_fill(id_host.data(), (int) max_rows_, cap_);
        if (cudaMemcpy(ident_, id_host.data(), id_host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dflash: the identity selection upload failed";
            return false;
        }
    }
    if (cudaMemset(arg_scratch_, 0, strata::kernels::argmax_rows_scratch_bytes((int) R)) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "dflash: the upload did not land";
        return false;
    }
    return true;
}

bool DFlashDrafter::bind(const WeightTable& wt, const NativeHead* head, std::string& err) {
    head_ = head;
    if (head_ == nullptr || !head_->loaded()) {
        err = "dflash: the target's native head is required (the full-vocabulary draft head)";
        return false;
    }
    if (mask_ < 0) { err = "dflash: no mask token id (metadata or --dflash-mask-token)"; return false; }
    if (const strata::core::NativeEmbed* ne = strata::core::native_embed()) {
        ne->gather_one(mask_, mask_row_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash: the mask row gather failed"; return false; }
        return true;
    }
    emb_ref_ = wt.find("token_embd.weight");
    if (emb_ref_ == nullptr) { err = "dflash: the target's token_embd.weight is missing"; return false; }
    int32_t* id_dev = nullptr;
    if (cudaMalloc(&id_dev, 4) != cudaSuccess) { err = "dflash: the mask row staging failed"; return false; }
    const int32_t one = (int32_t) mask_;
    const auto* codes = (const uint8_t*) emb_ref_->data;
    const auto* scales = (const float*) (codes + emb_ref_->codes_bytes);
    const auto* offsets = emb_ref_->has_offset ? (const float*) (codes + emb_ref_->codes_bytes + emb_ref_->scales_bytes)
                                               : nullptr;
    strata::kernels::embedding_gather_dev(codes, scales, offsets, id_dev, 1, emb_ref_->ne0, emb_ref_->code_bits,
                                          emb_ref_->code_bias, emb_ref_->group_elems,
                                          (uint64_t) (emb_ref_->ne0 / (8 / emb_ref_->code_bits)),
                                          (uint64_t) (emb_ref_->ne0 / emb_ref_->group_elems), mask_row_, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        cudaFree(id_dev);
        err = "dflash: the mask row gather failed";
        return false;
    }
    cudaFree(id_dev);
    return true;
}

/// One batch of up to 8 rows: ctx = hidden_norm(fc(taps)) into `ctx_`, then each draft layer's
/// K/V projected, k-normed, roped and appended at [pos0, pos0+rows).
bool DFlashDrafter::add_context(const uint16_t* taps, int n_taps, int64_t stride_rows, int64_t pos0, int64_t rows,
                                std::string& err) {
    // The prompt path's capture: [n_taps][stride_rows][n_embd] bf16.  The fusion wants [rows][F]
    // (tap-major inside a row): transpose each 8-row batch into tapin_, then fuse + append.
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, F = dg.fusion_in();
    if (n_taps * N != F) { err = "dflash: the tap count does not match the fusion input"; return false; }
    for (int64_t r0 = 0; r0 < rows; r0 += 8) {
        const int nr = (int) std::min<int64_t>(8, rows - r0);
        for (int t = 0; t < n_taps; ++t)
            strata::kernels::bf16_gather_strided(taps + (size_t) ((int64_t) t * stride_rows + r0) * N, N,
                                                 tapin_ + (size_t) t * N, F, (int) N, nr, cs_);
        if (!fusion_rows(pos0 + r0, nr, err)) return false;
    }
    return true;
}

bool DFlashDrafter::add_context_f32(const float* taps, int n_taps, int64_t stride_floats, int64_t pos0, int64_t rows,
                                    std::string& err) {
    // The verify window's capture: [n_taps][stride_floats] f32 (stride_floats = max_t*n_embd), the
    // rows [0, rows) of each tap valid.  (The prompt path's add_context takes ROWS instead: its
    // tap stride is the chunk capacity in rows.)
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, F = dg.fusion_in();
    if (n_taps * N != F) { err = "dflash: the tap count does not match the fusion input"; return false; }
    if (rows > max_rows_) { err = "dflash: more context rows than the forward's width"; return false; }
    // f32 source: each tap's rows go into tapf_ ([rows][F]) row by row, one conversion.  (Rows of
    // one tap are consecutive; a 2-D copy here trips the driver's pitch rules for no gain.)
    for (int t = 0; t < n_taps; ++t) {
        for (int r = 0; r < rows; ++r) {
            if (cudaMemcpyAsync(tapf_ + (size_t) ((int64_t) t * rows + r) * N,
                                taps + (size_t) ((int64_t) t * stride_floats + r * N), (size_t) N * 4,
                                cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
                err = std::string("dflash: the tap gather failed: ") + cudaGetErrorString(cudaGetLastError()) +
                      " (t=" + std::to_string(t) + " r=" + std::to_string(r) + " stride=" +
                      std::to_string(stride_floats) + " rows=" + std::to_string(rows) + ")";
                return false;
            }
        }
    }
    strata::kernels::f32_to_bf16_bulk(tapf_, tapin_, (int64_t) rows * F, cs_);
    if (!fusion_rows(pos0, (int) rows, err)) return false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("dflash: the context update failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaGetLastError();
    return true;
}

bool DFlashDrafter::fusion_rows(int64_t pos0, int rows, std::string& err) {
    using namespace strata::kernels;
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, F = dg.fusion_in(), KVW = dg.n_head_kv * dg.head_dim;
    auto wp = [&](const char* name) -> const uint16_t* {
        for (auto& [n, p] : wt_)
            if (n == name) return p;
        return nullptr;
    };
    auto wf = [&](const char* name) -> const float* {
        for (auto& [n, p] : wf_)
            if (n == name) return p;
        return nullptr;
    };
    // ctx = hidden_norm(fc(taps)); the projections run one row per launch (the bf16 path has no
    // multi-row variant yet - the prompt batches loop, the cycle needs at most 8)
    for (int r = 0; r < rows; ++r)
        bf16_gemv(tapin_ + (size_t) r * F, wp("fc"), ctx_ + (size_t) r * N, F, N, cs_);
    native_qsa_rms_norm_weighted(ctx_, wf("hidden_norm"), ctx_, (int) N, rows, kEps, cs_);
    f32_to_bf16_bulk(ctx_, xn16_, (int64_t) rows * N, cs_);
    for (int64_t l = 0; l < dg.layers; ++l) {
        const std::string pre = "layers." + std::to_string(l);
        for (int r = 0; r < rows; ++r) {
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.k_proj").c_str()), kc_ + (size_t) r * KVW, N, KVW, cs_);
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.v_proj").c_str()), vc_ + (size_t) r * KVW, N, KVW, cs_);
        }
        native_qsa_rms_norm_weighted(kc_, wf((pre + ".self_attn.k_norm").c_str()), kc_, (int) dg.head_dim,
                                     (int) (rows * dg.n_head_kv), kEps, cs_);
        // rope at each row's own position (k rows of one row sit NKV apart: [row r][head][hd]);
        // the rope reads DEVICE positions
        for (int r = 0; r < rows; ++r)
            for (int64_t hh = 0; hh < dg.n_head_kv; ++hh) h_pos_[(size_t) r * dg.n_head_kv + hh] = (int32_t)(pos0 + r);
        if (cudaMemcpyAsync(pos_, h_pos_, (size_t) rows * dg.n_head_kv * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the position staging failed";
            return false;
        }
        dflash_rope_neox_apply(kc_, kc_, (int) (rows * dg.n_head_kv), (int) dg.head_dim, dg.rope_theta, pos_, cs_);
        // append at the true cells
        for (int r = 0; r < rows; ++r) {
            const int64_t cell = pos0 + r;
            h_step_[(size_t) r * 4 + 0] = (int32_t) cell;
            h_step_[(size_t) r * 4 + 1] = (int32_t)(cell + 1);
            h_step_[(size_t) r * 4 + 2] = (int32_t)((cell + 1) / 4);
            h_step_[(size_t) r * 4 + 3] = (int32_t)(cell + 1);
        }
        if (cudaMemcpyAsync(step_, h_step_, (size_t) rows * 16, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the step staging failed";
            return false;
        }
        const QsaState& stl = st_[(size_t) l];
        const QsaAttnPools pools = qsa_attn_pools(stl);
        if (stl.kv_q4)
            kv_append_q4_steps(stl.k_q4, stl.v_q4, stl.page_table, step_, 4, rows, kc_, vc_, shapes_, cs_, &stl.host);
        else if (stl.kv_int8)
            kv_append_q8_steps(stl.k_q, stl.v_q, stl.k_scale, stl.v_scale, stl.page_table, step_, 4, kc_, vc_,
                               (int) KVW, rows, shapes_, cs_, &stl.host);
        else
            for (int r = 0; r < rows; ++r)
                kv_append_step(stl.k_pool, stl.v_pool, stl.page_table, step_ + r * 4, kc_ + (size_t) r * KVW,
                               vc_ + (size_t) r * KVW, shapes_, cs_, &stl.host);
    }
    return true;
}

bool DFlashDrafter::propose(int32_t x, int64_t pos, int block, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, Q = dg.n_head * dg.head_dim, KVW = dg.n_head_kv * dg.head_dim, I = dg.intermediate;
    auto wp = [&](const char* name) -> const uint16_t* {
        for (auto& [n, p] : wt_)
            if (n == name) return p;
        return nullptr;
    };
    auto wf = [&](const char* name) -> const float* {
        for (auto& [n, p] : wf_)
            if (n == name) return p;
        return nullptr;
    };
    if (block < 1 || block > max_rows_) { err = "dflash: bad block size"; return false; }
    const int K = block;

    // ---- the query rows: [x, mask x (K-1)]; row 0 = the anchor's own embedding
    h_tok_[0] = x;
    for (int r = 1; r < K; ++r) h_tok_[r] = (int32_t) mask_;
    if (cudaMemcpyAsync(tok_, h_tok_, (size_t) K * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "dflash: the token staging failed";
        return false;
    }
    if (const strata::core::NativeEmbed* ne = strata::core::native_embed()) {
        ne->gather_dev(tok_, K, emb_, cs_);
    } else {
        const auto* codes = (const uint8_t*) emb_ref_->data;
        const auto* scales = (const float*) (codes + emb_ref_->codes_bytes);
        const auto* offsets = emb_ref_->has_offset ? (const float*) (codes + emb_ref_->codes_bytes + emb_ref_->scales_bytes)
                                                   : nullptr;
        embedding_gather_dev(codes, scales, offsets, tok_, K, emb_ref_->ne0, emb_ref_->code_bits, emb_ref_->code_bias,
                             emb_ref_->group_elems, (uint64_t) (emb_ref_->ne0 / (8 / emb_ref_->code_bits)),
                             (uint64_t) (emb_ref_->ne0 / emb_ref_->group_elems), emb_, cs_);
    }
    if (cudaMemcpyAsync(h_, emb_, (size_t) K * N * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "dflash: the residual init failed";
        return false;
    }
    // per-head rope positions of the query rows: row r at pos+r
    for (int r = 0; r < K; ++r)
        for (int64_t hh = 0; hh < dg.n_head; ++hh) h_pos_[(size_t) r * dg.n_head + hh] = (int32_t)(pos + r);

    for (int64_t l = 0; l < dg.layers; ++l) {
        const std::string pre = "layers." + std::to_string(l);
        // ---- attention half
        native_qsa_rms_norm_weighted(h_, wf((pre + ".input_layernorm").c_str()), xn_, (int) N, K, kEps, cs_);
        f32_to_bf16_bulk(xn_, xn16_, (int64_t) K * N, cs_);
        for (int r = 0; r < K; ++r) {
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.q_proj").c_str()), q_ + (size_t) r * Q, N, Q, cs_);
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.k_proj").c_str()), kc_ + (size_t) r * KVW, N, KVW, cs_);
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.v_proj").c_str()), vc_ + (size_t) r * KVW, N, KVW, cs_);
        }
        // per-head q/k norms, then rope (q rows: NH heads at [pos..pos+K); append uses true cells)
        native_qsa_rms_norm_weighted(q_, wf((pre + ".self_attn.q_norm").c_str()), q_, (int) dg.head_dim,
                                     (int) (K * dg.n_head), kEps, cs_);
        native_qsa_rms_norm_weighted(kc_, wf((pre + ".self_attn.k_norm").c_str()), kc_, (int) dg.head_dim,
                                     (int) (K * dg.n_head_kv), kEps, cs_);
        if (cudaMemcpyAsync(pos_, h_pos_, (size_t) K * dg.n_head * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the position staging failed";
            return false;
        }
        dflash_rope_neox_apply(q_, q_, (int) (K * dg.n_head), (int) dg.head_dim, dg.rope_theta, pos_, cs_);
        for (int r = 0; r < K; ++r)
            for (int64_t hh = 0; hh < dg.n_head_kv; ++hh) h_pos_[(size_t) r * dg.n_head_kv + hh] = (int32_t)(pos + r);
        if (cudaMemcpyAsync(pos_, h_pos_, (size_t) K * dg.n_head_kv * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the position staging failed";
            return false;
        }
        dflash_rope_neox_apply(kc_, kc_, (int) (K * dg.n_head_kv), (int) dg.head_dim, dg.rope_theta, pos_, cs_);
        // append the queries' own cells at their true positions, then every query reads [0, pos+K)
        for (int r = 0; r < K; ++r) {
            const int64_t cell = pos + r;
            h_step_[(size_t) r * 4 + 0] = (int32_t) cell;
            h_step_[(size_t) r * 4 + 1] = (int32_t)(cell + 1);
            h_step_[(size_t) r * 4 + 2] = (int32_t)((cell + 1) / 4);
            h_step_[(size_t) r * 4 + 3] = (int32_t)(cell + 1);
        }
        if (cudaMemcpyAsync(step_, h_step_, (size_t) K * 16, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the step staging failed";
            return false;
        }
        const QsaState& stl = st_[(size_t) l];
        const QsaAttnPools pools = qsa_attn_pools(stl);
        if (stl.kv_q4)
            kv_append_q4_steps(stl.k_q4, stl.v_q4, stl.page_table, step_, 4, K, kc_, vc_, shapes_, cs_, &stl.host);
        else if (stl.kv_int8)
            kv_append_q8_steps(stl.k_q, stl.v_q, stl.k_scale, stl.v_scale, stl.page_table, step_, 4, kc_, vc_,
                               (int) KVW, K, shapes_, cs_, &stl.host);
        else
            for (int r = 0; r < K; ++r)
                kv_append_step(stl.k_pool, stl.v_pool, stl.page_table, step_ + r * 4, kc_ + (size_t) r * KVW,
                               vc_ + (size_t) r * KVW, shapes_, cs_, &stl.host);
        // non-causal over every cell: each row's record reads [0, pos+K)
        {
            std::vector<int32_t> attn_steps((size_t) K * 4);
            for (int r = 0; r < K; ++r) {
                attn_steps[(size_t) r * 4 + 0] = (int32_t)(pos + K - 1);
                attn_steps[(size_t) r * 4 + 1] = (int32_t)(pos + K);
                attn_steps[(size_t) r * 4 + 2] = (int32_t)((pos + K) / 4);
                attn_steps[(size_t) r * 4 + 3] = (int32_t)(pos + K);
            }
            if (cudaMemcpyAsync(step_, attn_steps.data(), (size_t) K * 16, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
                err = "dflash: the attention staging failed";
                return false;
            }
        }
        if ((int64_t)(pos + K) > cap_) {
            err = "dflash: the window cap is exceeded (raise --dflash-window)";
            return false;
        }
        static const bool df_dbg = std::getenv("STRATA_DF_DBG") != nullptr;
        if (l == 0 && df_dbg) {
            const QsaAttnPools pl = qsa_attn_pools(st_[0]);
            std::fprintf(stderr,
                         "dflash dbg: attn cap=%lld K=%d step0=[%d %d %d %d] pools k=%p kq=%p kq4=%p pt=%p "
                         "kv_mode=%d rot=%d int8=%d q4=%d shapes hd=%lld nh=%lld nkv=%lld ps=%lld rot_bits=%lld\n",
                         (long long) cap_, K, h_step_[0], h_step_[1], h_step_[2], h_step_[3], (const void*) pl.k_pool,
                         (const void*) pl.k_q, (const void*) pl.k_q4, (const void*) pl.page_table, (int) st_[0].kv_mode,
                         (int) st_[0].kv_rot, (int) st_[0].kv_int8, (int) st_[0].kv_q4, (long long) shapes_.head_dim,
                         (long long) shapes_.n_head, (long long) shapes_.n_head_kv, (long long) shapes_.page_size,
                         (long long) shapes_.n_rot);
        }
        qsa_decode_attn_batch(q_, pools, ident_, step_, cap_, shapes_, (float*) attn_scratch_, attn_, K, cs_);
        f32_to_bf16_bulk(attn_, attn16_, (int64_t) K * Q, cs_);
        for (int r = 0; r < K; ++r)
            bf16_gemv(attn16_ + (size_t) r * Q, wp((pre + ".self_attn.o_proj").c_str()), bo_ + (size_t) r * N, Q, N, cs_);
        if (l == 0 && df_dbg) {
            std::vector<float> hb(4), ab(4);
            cudaMemcpyAsync(hb.data(), h_, 16, cudaMemcpyDeviceToHost, cs_);
            cudaMemcpyAsync(ab.data(), bo_, 16, cudaMemcpyDeviceToHost, cs_);
            cudaStreamSynchronize(cs_);
            std::fprintf(stderr, "df dbg: after attn h0=%.3e bo0=%.3e\n", hb[0], ab[0]);
        }
        add_inplace(h_, bo_, K * N, cs_);
        // ---- MLP half
        native_qsa_rms_norm_weighted(h_, wf((pre + ".post_attention_layernorm").c_str()), xn_, (int) N, K, kEps, cs_);
        f32_to_bf16_bulk(xn_, xn16_, (int64_t) K * N, cs_);
        for (int r = 0; r < K; ++r) {
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".mlp.gate_proj").c_str()), gate_ + (size_t) r * I, N, I, cs_);
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".mlp.up_proj").c_str()), up_ + (size_t) r * I, N, I, cs_);
        }
        swiglu_inplace(gate_, up_, K * I, cs_);
        f32_to_bf16_bulk(gate_, xn16_, (int64_t) K * I, cs_);
        for (int r = 0; r < K; ++r)
            bf16_gemv(xn16_ + (size_t) r * I, wp((pre + ".mlp.down_proj").c_str()), bo_ + (size_t) r * N, I, N, cs_);
        add_inplace(h_, bo_, K * N, cs_);
    }
    // ---- final norm, the target's head, the row argmaxes
    static const bool df_dbg2 = std::getenv("STRATA_DF_DBG") != nullptr;
    if (df_dbg2) {
        std::vector<float> hb(4), eb(4);
        cudaMemcpyAsync(hb.data(), h_, 16, cudaMemcpyDeviceToHost, cs_);
        cudaMemcpyAsync(eb.data(), emb_, 16, cudaMemcpyDeviceToHost, cs_);
        cudaStreamSynchronize(cs_);
        std::fprintf(stderr, "df dbg: final h0=%.3e emb0=%.3e\n", hb[0], eb[0]);
    }
    native_qsa_rms_norm_weighted(h_, wf("norm"), xn_, (int) N, K, kEps, cs_);
    native_quantize_q8_1(xn_, xq_, (int) N, K, cs_);
    native_mmvq(head_->type(), head_->weights(), xq_, logits_, (int) N, (int) dg.vocab, K, cs_);
    argmax_rows(logits_, K, (int) dg.vocab, arg_scratch_, out_, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("dflash: its stream failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    std::memcpy(out, h_out_, (size_t) K * 4);
    return true;
}

}  // namespace strata::core
