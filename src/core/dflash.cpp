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
    if (found->type != 30 && !(shape.size() == 2 && shape[0] % 32 == 0 &&
                              (found->type == 2 || found->type == 6 || found->type == 8))) {
        err = std::string("dflash: tensor '") + found->name + "' is " + found->type_name() +
              "; expected BF16 norms and BF16, Q8_0, Q5_0 or Q4_0 matrices";
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
    out->type = found->type;
    out->bytes = bytes;
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
    struct Rollback {
        DFlashArtifact* self;
        std::string path;
        std::unique_ptr<GgufFile> file;
        DFlashGeometry geom;
        std::vector<DFlashTensor> tensors;
        uint64_t weight_bytes;
        bool keep = false;
        ~Rollback() {
            if (keep || self == nullptr) return;
            self->path_ = path;
            self->file_ = std::move(file);
            self->geom_ = geom;
            self->tensors_ = std::move(tensors);
            self->weight_bytes_ = weight_bytes;
        }
    } rb{this, path_, std::move(file_), geom_, std::move(tensors_), weight_bytes_};
    // TRANSACTIONAL: the parse fills this artifact's members exactly as before, and the rollback
    // guard holds a snapshot of everything it touches - any failure (a bad artifact, or a bad
    // re-open over a good one) restores them, so loaded() and every accessor stay exactly as they
    // were.  The good state commits by keeping the guard.
    try {
        file_ = std::make_unique<GgufFile>(path);
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
    path_ = path;
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
    auto add = [&](uint64_t) { weight_bytes_ += (tensors_.back().bytes + 31) & ~uint64_t(31); };
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
    // every check passed: the guard keeps the parsed state
    rb.keep = true;
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


void dflash_identity_fill(int32_t* host, int rows, int64_t cap) {
    for (int r = 0; r < rows; ++r)
        for (int64_t i = 0; i < cap; ++i) host[(size_t) r * (size_t) cap + (size_t) i] = (int32_t) i;
}
}  // namespace strata::core
