#include "strata/muse/model.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#ifdef STRATA_MUSE_CUDA
#include "cuda_matvec.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <iostream>

namespace strata::muse {
namespace {
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("Muse: " + s); }
using Vec = std::vector<float>;
constexpr size_t max_header_bytes = 256 * 1024 * 1024;
const std::string prefix = "muse-spark.";

// Preflight before GgufModel: bound hostile header counts and alignment arithmetic,
// and retain complete geometry arrays (the shared artifact reader samples 64).
// Tokenizer strings are skipped, never copied into a whole-vocabulary table here.
class Header {
    std::ifstream in;
    size_t size, pos = 0;
    void need(size_t n) {
        if (n > size - pos || n > max_header_bytes - std::min(pos, max_header_bytes)) fail("truncated/oversized GGUF header");
    }
    template<class T> T read() {
        need(sizeof(T)); T v{};
        if (!in.read(reinterpret_cast<char*>(&v), sizeof(T))) fail("cannot read GGUF header");
        pos += sizeof(T); return v;
    }
    std::string string(bool keep) {
        const auto n64 = read<uint64_t>();
        if (n64 > max_header_bytes) fail("oversized GGUF string");
        const size_t n = size_t(n64); need(n);
        std::string s;
        if (keep) { s.resize(n); if (!in.read(s.data(), std::streamsize(n))) fail("truncated GGUF string"); }
        else { in.seekg(std::streamoff(n), std::ios::cur); if (!in) fail("truncated GGUF string"); }
        pos += n; return s;
    }
    MetaValue value(MetaType t, bool keep, int depth = 0) {
        MetaValue v; v.type = t;
        switch (t) {
        case MetaType::U8: v.u = read<uint8_t>(); break;
        case MetaType::I8: v.u = uint64_t(read<int8_t>()); break;
        case MetaType::U16: v.u = read<uint16_t>(); break;
        case MetaType::I16: v.u = uint64_t(read<int16_t>()); break;
        case MetaType::U32: v.u = read<uint32_t>(); break;
        case MetaType::I32: v.u = uint64_t(read<int32_t>()); break;
        case MetaType::U64: v.u = read<uint64_t>(); break;
        case MetaType::I64: v.u = uint64_t(read<int64_t>()); break;
        case MetaType::BOOL: v.u = read<uint8_t>(); if (v.u > 1) fail("invalid GGUF boolean"); break;
        case MetaType::F32: v.f = read<float>(); break;
        case MetaType::F64: v.f = read<double>(); break;
        case MetaType::STRING: v.s = string(keep); break;
        case MetaType::ARRAY:
            if (depth >= 2) fail("nested GGUF array too deep");
            v.elem = MetaType(read<uint32_t>()); v.count = read<uint64_t>();
            if (v.count > (1u << 24) || (keep && v.count > 100000)) fail("oversized GGUF array");
            for (uint64_t i = 0; i < v.count; ++i) {
                auto item = value(v.elem, keep, depth + 1);
                if (keep) v.items.push_back(std::move(item));
            }
            break;
        default: fail("unknown GGUF metadata type");
        }
        return v;
    }
public:
    std::map<std::string, MetaValue> meta;
    explicit Header(const std::string& path) : in(path, std::ios::binary), size(0) {
        if (!in) fail("cannot open model shard");
        in.seekg(0, std::ios::end); const auto end = in.tellg();
        if (end < 0 || uint64_t(end) > std::numeric_limits<size_t>::max()) fail("invalid file size");
        size = size_t(end); in.seekg(0);
        if (read<uint32_t>() != 0x46554747 || read<uint32_t>() != 3) fail("expected GGUF v3");
        const auto nt = read<uint64_t>(), nk = read<uint64_t>();
        if (nt > 100000 || nk > 100000 || nt > size / 24 || nk > size / 12) fail("implausible GGUF header counts");
        std::set<std::string> keys;
        for (uint64_t i = 0; i < nk; ++i) {
            auto key = string(true);
            if (!keys.insert(key).second) fail("duplicate GGUF metadata key");
            const auto type = MetaType(read<uint32_t>());
            const bool keep = key.compare(0, prefix.size(), prefix) == 0 || key == "general.architecture" ||
                              key == "general.alignment" || key == "tokenizer.ggml.eos_token_id" ||
                              key == "tokenizer.ggml.eot_token_id";
            auto v = value(type, keep);
            if (keep) meta.emplace(std::move(key), std::move(v));
        }
        size_t align = 32;
        if (auto it = meta.find("general.alignment"); it != meta.end()) {
            if (it->second.type != MetaType::U32) fail("alignment must be uint32");
            align = size_t(it->second.u);
        }
        if (!align || align > 1024 * 1024 || (align & (align - 1))) fail("invalid GGUF alignment");
        std::vector<std::pair<size_t, size_t>> ranges;
        std::set<std::string> names;
        for (uint64_t i = 0; i < nt; ++i) {
            const auto name = string(true);
            if (!names.insert(name).second) fail("duplicate GGUF tensor name");
            const auto nd = read<uint32_t>();
            if (!nd || nd > 4) fail("invalid tensor rank: " + name);
            size_t cols = 0, rows = 1;
            for (uint32_t d = 0; d < nd; ++d) {
                const auto n = read<uint64_t>();
                if (!n || n > size_t(std::numeric_limits<int32_t>::max())) fail("invalid tensor dimension: " + name);
                if (!d) cols = size_t(n); else rows = checked_mul(rows, size_t(n));
            }
            const auto type = read<uint32_t>();
            if (type >= GGML_TYPE_COUNT) fail("unknown tensor encoding: " + name);
            const auto* traits = ggml_get_type_traits(ggml_type(type));
            if (!traits || traits->blck_size <= 0 || !traits->type_size ||
                (type != GGML_TYPE_F32 && !traits->to_float) || traits->blck_size_interleave > 1)
                fail("unsupported GGML row encoding: " + name + " (type " + std::to_string(type) + ")");
            if (cols % size_t(traits->blck_size)) fail("invalid quantized row block size: " + name);
            const auto bytes = checked_mul(checked_mul(cols / size_t(traits->blck_size), traits->type_size), rows);
            const auto off64 = read<uint64_t>();
            if (off64 > size || off64 % align) fail("invalid tensor offset: " + name);
            const size_t off = size_t(off64);
            ranges.emplace_back(off, checked_add(off, bytes));
        }
        const size_t data = checked_add(pos, align - 1) / align * align;
        if (nt && data > size) fail("tensor data starts past EOF");
        const size_t payload = size - std::min(data, size);
        std::sort(ranges.begin(), ranges.end());
        size_t last = 0;
        for (const auto& r : ranges) {
            if (r.first < last || r.second > payload) fail("overlapping/truncated tensor payload");
            last = r.second;
        }
    }
};
struct Prepared {
    std::vector<std::string> paths;
    std::map<std::string, MetaValue> meta;
    explicit Prepared(const std::string& path) : paths(gguf_split_paths(path)) {
        for (size_t i = 0; i < paths.size(); ++i) {
            Header h(paths[i]);
            if (!i) meta = std::move(h.meta);
        }
    }
};
const MetaValue* required(const std::map<std::string, MetaValue>& m, const std::string& k) {
    auto it = m.find(k); if (it == m.end()) fail("missing metadata " + k); return &it->second;
}
size_t integer(const MetaValue& v, const std::string& k, bool zero = false) {
    if (v.type != MetaType::U32 && v.type != MetaType::U64 && v.type != MetaType::U16 &&
        v.type != MetaType::U8 && v.type != MetaType::I32 && v.type != MetaType::I64 &&
        v.type != MetaType::I16 && v.type != MetaType::I8) fail("expected integer metadata " + k);
    if ((!zero && !v.u) || v.u > uint64_t(std::numeric_limits<int32_t>::max())) fail("integer metadata out of range " + k);
    return size_t(v.u);
}
float real(const MetaValue& v, const std::string& k, bool positive = true) {
    if (v.type != MetaType::F32 && v.type != MetaType::F64) fail("expected floating metadata " + k);
    const float f = float(v.f);
    if (!std::isfinite(f) || (positive ? f <= 0 : f < 0)) fail("floating metadata out of range " + k);
    return f;
}
Geometry geometry(const Prepared& p, const GgufModel& model) {
    const auto& m = p.meta;
    const auto& arch = *required(m, "general.architecture");
    if (arch.type != MetaType::STRING || arch.s != "muse-spark") fail("architecture must be muse-spark");
    auto n = [&](const char* key) { return integer(*required(m, prefix + key), key); };
    auto f = [&](const char* key, bool positive = true) { return real(*required(m, prefix + key), key, positive); };
    Geometry g;
    g.hidden = n("embedding_length"); g.layers = n("block_count");
    g.heads = n("attention.head_count"); g.kv_heads = n("attention.head_count_kv");
    g.head_dim = n("attention.key_length");
    if (n("attention.value_length") != g.head_dim || g.heads % g.kv_heads) fail("incompatible GQA/head dimensions");
    g.rotary_dim = g.head_dim;
    if (auto it = m.find(prefix + "rope.dimension_count"); it != m.end()) g.rotary_dim = integer(it->second, it->first);
    if (g.rotary_dim > g.head_dim || g.rotary_dim % 2) fail("invalid NeoX rotary dimensions");
    g.latent = n("moe_latent_size"); g.experts = n("expert_count"); g.top_k = n("expert_used_count");
    if (g.top_k > g.experts) fail("top-k exceeds expert count");
    g.train_context = n("context_length"); g.window = n("attention.sliding_window");
    g.epsilon = f("attention.layer_norm_rms_epsilon"); g.post_epsilon = f("post_norm_epsilon");
    g.rope_base = f("rope.freq_base_swa"); g.logit_scale = f("logit_scale");
    if (auto it = m.find(prefix + "rope.scaling.type"); it != m.end())
        if (it->second.type != MetaType::STRING || it->second.s != "none") fail("scaled RoPE is not supported by this reference");
    if (auto it = m.find(prefix + "rope.scaling.factor"); it != m.end())
        if (real(it->second, it->first) != 1.0f) fail("scaled RoPE is not supported by this reference");
    g.softcap = f("final_logit_softcapping", false); g.expert_scale = f("expert_weights_scale");
    if (n("expert_gating_func") != 2) fail("only sigmoid expert routing is supported");
    const auto& norm = *required(m, prefix + "expert_weights_norm");
    if (norm.type != MetaType::BOOL || norm.u != 1) fail("Muse requires normalized expert weights");
    const auto& pattern = *required(m, prefix + "attention.sliding_window_pattern");
    if (pattern.type != MetaType::ARRAY || pattern.elem != MetaType::BOOL || pattern.count != g.layers)
        fail("sliding window pattern length/type mismatch");
    for (const auto& x : pattern.items) g.local.push_back(x.u != 0);
    const auto& ff = *required(m, prefix + "expert_feed_forward_length");
    if (ff.type == MetaType::ARRAY) {
        if (ff.count != g.layers) fail("expert FF array length mismatch");
        for (const auto& x : ff.items) g.ff.push_back(integer(x, "expert_feed_forward_length"));
    } else g.ff.assign(g.layers, integer(ff, "expert_feed_forward_length"));
    const auto* emb = model.find("token_embd.weight");
    if (!emb || emb->shape.size() != 2) fail("missing/malformed token embedding");
    g.vocab = size_t(emb->shape[1]);
    if (const auto* tokens = model.meta().get("tokenizer.ggml.tokens"))
        if (tokens->type != MetaType::ARRAY || tokens->elem != MetaType::STRING || tokens->count != g.vocab)
            fail("tokenizer vocabulary length/type mismatch");
    // This also validates every dynamic dimension against scratch arithmetic.
    g.memory_bytes(1);
    return g;
}
void finite(const Vec& x) { for (float v : x) if (!std::isfinite(v)) fail("nonfinite numerical result/weight"); }
void rms(float* x, size_t n, float eps) {
    float ss = 0;
    for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    if (!std::isfinite(ss)) fail("nonfinite RMS accumulator");
    const float scale = 1.0f / std::sqrt(ss / float(n) + eps);
    for (size_t i = 0; i < n; ++i) x[i] *= scale;
}
float sigmoid(float x) { return x >= 0 ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x)); }
} // namespace

size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) fail("size arithmetic overflow");
    return a + b;
}
size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) fail("size arithmetic overflow");
    return a * b;
}
uint32_t parse_uint(const std::string& s, const char* name) {
    if (s.empty()) fail(std::string("empty ") + name);
    uint32_t n = 0;
    for (char c : s) {
        if (c < '0' || c > '9' || n > (UINT32_MAX - uint32_t(c - '0')) / 10)
            fail(std::string("invalid/out-of-range ") + name);
        n = n * 10 + uint32_t(c - '0');
    }
    return n;
}
std::vector<uint32_t> parse_tokens(const std::string& s) {
    if (s.size() > 16 * 1024 * 1024) fail("token list too large");
    std::vector<uint32_t> out;
    size_t start = 0;
    do {
        const size_t end = s.find(',', start);
        out.push_back(parse_uint(s.substr(start, end == std::string::npos ? end : end - start), "token"));
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    return out;
}
Options parse_options(int argc, const char* const* argv) {
    Options o; std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (!seen.insert(key).second) fail("duplicate argument " + key);
        if (key != "--model" && key != "--tokens" && key != "--ctx" && key != "--steps" &&
            key != "--backend" && key != "--gpu" && key != "--logits-file") fail("unknown argument " + key);
        if (++i == argc) fail("missing value for " + key);
        const std::string v(argv[i]);
        if (key == "--model") o.model = v;
        else if (key == "--tokens") o.tokens = parse_tokens(v);
        else if (key == "--ctx") o.context = parse_uint(v, "context");
        else if (key == "--steps") o.steps = parse_uint(v, "steps");
        else if (key == "--gpu") o.gpu = parse_uint(v, "GPU");
        else if (key == "--backend") o.backend = v;
        else o.logits_file = v;
    }
    if (o.model.empty() || o.logits_file.empty() || o.tokens.empty()) fail("--model, --tokens and --logits-file are required");
    if (!o.context || o.tokens.size() > o.context || o.steps > o.context - o.tokens.size())
        fail("prompt plus generation exceeds context");
    if (o.backend != "cpu" && o.backend != "cuda") fail("backend must be cpu or cuda");
    if (o.gpu > uint32_t(std::numeric_limits<int>::max())) fail("GPU index out of range");
    return o;
}
size_t Geometry::memory_bytes(size_t context) const {
    if (!context || context > train_context || !layers || !hidden || !vocab || !heads || !kv_heads ||
        !head_dim || !latent || !experts || !top_k || ff.size() != layers || local.size() != layers)
        fail("invalid context/geometry");
    const size_t qwidth = checked_mul(heads, head_dim), kvwidth = checked_mul(kv_heads, head_dim);
    size_t width = std::max({hidden, vocab, qwidth, kvwidth, latent, experts});
    for (auto n : ff) { if (!n) fail("zero expert FF dimension"); width = std::max(width, n); }
    if (width > tile_bytes / sizeof(float)) fail("geometry exceeds bounded row/activation size");
    size_t kv = 0;
    for (size_t i = 0; i < layers; ++i)
        kv = checked_add(kv, checked_mul(checked_mul(local[i] ? std::min(context, window) : context, kvwidth), 2));
    // Conservative bound: live vector temporaries, indices, attention scores,
    // full logits, and one decoded tile. Output is streamed, not stored per token.
    const size_t scratch = checked_add(tile_bytes, checked_mul(checked_add(checked_mul(width, 64), context), sizeof(float)));
    const size_t bytes = checked_add(checked_mul(kv, sizeof(float)), scratch);
    if (bytes > host_budget_bytes) fail("KV plus scratch exceeds 1 GiB reference budget");
    return bytes;
}

struct Model::Impl {
    Prepared prepared;
    GgufModel weights;
    Geometry g;
    size_t context, position = 0;
    Vec tile;
    struct KV { Vec key, value; size_t slots; };
    std::vector<KV> cache;
    std::set<uint32_t> end_tokens;
#ifdef STRATA_MUSE_CUDA
    std::unique_ptr<CudaMatvec> cuda;
#endif
    Impl(const std::string& path, size_t ctx, const std::string& backend, int gpu)
        : prepared(path), weights(prepared.paths), g(muse::geometry(prepared, weights)), context(ctx) {
        if (backend != "cpu" && backend != "cuda") fail("unknown backend");
        if (gpu < 0) fail("invalid GPU index");
#ifndef STRATA_MUSE_CUDA
        if (backend == "cuda") fail("CUDA adapter was not compiled");
#endif
        g.memory_bytes(context); // Before any tile, activation, KV or device allocation.
        validate_tensors();
        for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"}) {
            auto it = prepared.meta.find(key);
            if (it != prepared.meta.end()) {
                const size_t id = integer(it->second, key, true);
                if (id >= g.vocab) fail("EOS/EOT outside vocabulary");
                end_tokens.insert(uint32_t(id));
            }
        }
        tile.resize(tile_bytes / sizeof(float));
        cache.reserve(g.layers);
        const size_t kvwidth = checked_mul(g.kv_heads, g.head_dim);
        for (size_t l = 0; l < g.layers; ++l) {
            const size_t slots = g.local[l] ? std::min(context, g.window) : context;
            const size_t count = checked_mul(slots, kvwidth);
            cache.push_back({Vec(count), Vec(count), slots});
        }
#ifdef STRATA_MUSE_CUDA
        if (backend == "cuda") cuda = std::make_unique<CudaMatvec>(gpu);
#endif
    }
    const TensorInfo& tensor(const std::string& name) const {
        const auto* t = weights.find(name);
        if (!t) fail("missing tensor " + name);
        return *t;
    }
    void shape(const std::string& name, std::initializer_list<size_t> dims) const {
        const auto& t = tensor(name);
        if (t.shape.size() != dims.size() || !std::equal(dims.begin(), dims.end(), t.shape.begin()))
            fail("tensor shape mismatch: " + name);
    }
    void validate_tensors() const {
        // Recheck the mapped directory with GGML's byte geometry, not the shared
        // reader's independent type-size table. No tensor payload is copied.
        for (size_t s = 0; s < weights.size(); ++s) {
            const auto& shard = weights.shard(s);
            for (const auto& t : shard.tensors()) {
                if (t.type >= GGML_TYPE_COUNT || t.shape.empty()) fail("invalid mapped tensor directory");
                const auto* tr = ggml_get_type_traits(ggml_type(t.type));
                if (!tr || tr->blck_size <= 0 || !tr->type_size ||
                    (t.type != GGML_TYPE_F32 && !tr->to_float) || tr->blck_size_interleave > 1)
                    fail("unsupported mapped row encoding");
                size_t rows = 1;
                for (auto d : t.shape) if (!d || d > size_t(INT32_MAX)) fail("invalid mapped tensor dimension");
                for (size_t i = 1; i < t.shape.size(); ++i) rows = checked_mul(rows, size_t(t.shape[i]));
                if (t.shape[0] % uint64_t(tr->blck_size)) fail("invalid mapped row block size");
                const auto bytes = checked_mul(checked_mul(size_t(t.shape[0]) / size_t(tr->blck_size), tr->type_size), rows);
                const auto payload = shard.file_size() - shard.data_start();
                if (t.offset > payload || bytes > payload - t.offset) fail("mapped tensor exceeds shard payload");
            }
        }
        shape("token_embd.weight", {g.hidden, g.vocab}); shape("output_norm.weight", {g.hidden});
        shape("output.weight", {g.hidden, g.vocab});
        const size_t q = checked_mul(g.heads, g.head_dim), kv = checked_mul(g.kv_heads, g.head_dim);
        for (size_t l = 0; l < g.layers; ++l) {
            const auto b = "blk." + std::to_string(l) + ".";
            for (const char* n : {"attn_norm", "attn_resid_alpha", "attn_resid_beta", "ffn_norm",
                                  "post_ffw_norm", "ffn_resid_alpha", "ffn_resid_beta"}) shape(b + n + ".weight", {g.hidden});
            for (const char* n : {"attn_q_norm", "attn_k_norm"}) shape(b + n + ".weight", {g.head_dim});
            shape(b + "attn_q.weight", {g.hidden, q}); shape(b + "attn_k.weight", {g.hidden, kv});
            shape(b + "attn_v.weight", {g.hidden, kv}); shape(b + "attn_gate.weight", {g.hidden, q});
            shape(b + "attn_output.weight", {q, g.hidden});
            shape(b + "ffn_gate_inp.weight", {g.hidden, g.experts}); shape(b + "exp_probs_b.bias", {g.experts});
            shape(b + "ffn_latent_down.weight", {g.hidden, g.latent}); shape(b + "ffn_latent_up.weight", {g.latent, g.hidden});
            for (const char* n : {"ffn_pre_exp_norm", "ffn_post_exp_norm"}) shape(b + n + ".weight", {g.latent});
            for (const char* n : {"ffn_gate_exps", "ffn_up_exps"}) shape(b + n + ".weight", {g.latent, g.ff[l], g.experts});
            shape(b + "ffn_down_exps.weight", {g.ff[l], g.latent, g.experts});
        }
    }
    void row(const TensorInfo& t, size_t index, float* out) const {
        size_t shard = 0; weights.find(t.name, &shard);
        const auto* traits = ggml_get_type_traits(ggml_type(t.type));
        const size_t cols = size_t(t.shape[0]);
        size_t rows = 1; for (size_t i = 1; i < t.shape.size(); ++i) rows = checked_mul(rows, size_t(t.shape[i]));
        if (index >= rows) fail("tensor row index out of bounds");
        const size_t bytes = checked_mul(cols / size_t(traits->blck_size), traits->type_size);
        const auto* src = weights.shard(shard).tensor_data(t) + checked_mul(index, bytes);
        if (t.type == GGML_TYPE_F32) std::memcpy(out, src, cols * sizeof(float));
        else traits->to_float(src, out, int64_t(cols));
        for (size_t i = 0; i < cols; ++i) if (!std::isfinite(out[i])) fail("nonfinite tensor data");
    }
    Vec vector(const std::string& name) const {
        const auto& t = tensor(name); Vec out(size_t(t.shape[0])); row(t, 0, out.data()); return out;
    }
    Vec mm(const std::string& name, const Vec& x, size_t expert = 0) {
        const auto& t = tensor(name);
        if (x.size() != t.shape[0] || t.shape.size() < 2 ||
            (t.shape.size() == 3 ? expert >= t.shape[2] : expert != 0)) fail("invalid matvec geometry");
        const size_t rows = size_t(t.shape[1]), cols = x.size();
        const size_t batch = std::min(size_t(64), tile.size() / cols);
        Vec y(rows);
        for (size_t start = 0; start < rows; start += batch) {
            const size_t n = std::min(batch, rows - start);
            for (size_t r = 0; r < n; ++r) row(t, expert * rows + start + r, tile.data() + r * cols);
#ifdef STRATA_MUSE_CUDA
            if (cuda) cuda->multiply(tile.data(), x.data(), y.data() + start, cols, n);
            else
#endif
            for (size_t r = 0; r < n; ++r) {
                float sum = 0;
                for (size_t c = 0; c < cols; ++c) sum += tile[r * cols + c] * x[c];
                y[start + r] = sum;
            }
        }
        finite(y); return y;
    }
    void norm(Vec& x, const std::string& name, float eps, bool input_scaled = false) const {
        const auto w = vector(name);
        if (x.size() != w.size()) fail("invalid norm geometry");
        if (input_scaled) for (size_t i = 0; i < x.size(); ++i) x[i] *= w[i];
        rms(x.data(), x.size(), eps);
        if (!input_scaled) for (size_t i = 0; i < x.size(); ++i) x[i] *= w[i];
    }
    void head_norm(Vec& x, const std::string& name) const {
        const auto w = vector(name);
        for (size_t h = 0; h < x.size() / g.head_dim; ++h) {
            float* p = x.data() + h * g.head_dim; rms(p, g.head_dim, g.epsilon);
            for (size_t d = 0; d < g.head_dim; ++d) p[d] *= w[d];
        }
    }
    void rope(Vec& x) const {
        for (size_t h = 0; h < x.size() / g.head_dim; ++h)
            for (size_t d = 0; d < g.rotary_dim / 2; ++d) {
                const float theta = float(position) / std::pow(g.rope_base, 2.0f * float(d) / float(g.rotary_dim));
                const float c = std::cos(theta), s = std::sin(theta);
                const size_t i = h * g.head_dim + d, j = i + g.rotary_dim / 2;
                const float a = x[i], b = x[j]; x[i] = a * c - b * s; x[j] = a * s + b * c;
            }
    }
    Vec attention(size_t l, Vec& q, Vec& k, const Vec& v) {
        if (g.local[l]) { rope(q); rope(k); }
        auto& kv = cache[l]; const size_t width = k.size();
        const size_t write = (position % kv.slots) * width;
        std::copy(k.begin(), k.end(), kv.key.begin() + write);
        std::copy(v.begin(), v.end(), kv.value.begin() + write);
        const size_t first = g.local[l] && position + 1 > g.window ? position + 1 - g.window : 0;
        Vec out(q.size(), 0), scores(position + 1 - first);
        const float scale = 1.0f / std::sqrt(float(g.head_dim));
        for (size_t h = 0; h < g.heads; ++h) {
            const size_t kh = h / (g.heads / g.kv_heads);
            for (size_t t = first; t <= position; ++t) {
                const size_t offset = (t % kv.slots) * width + kh * g.head_dim;
                float dot = 0;
                for (size_t d = 0; d < g.head_dim; ++d) dot += q[h * g.head_dim + d] * kv.key[offset + d];
                scores[t - first] = dot * scale;
            }
            const float max = *std::max_element(scores.begin(), scores.end());
            float sum = 0;
            for (auto& s : scores) { s = std::exp(s - max); sum += s; }
            for (size_t t = first; t <= position; ++t) {
                const size_t offset = (t % kv.slots) * width + kh * g.head_dim;
                const float p = scores[t - first] / sum;
                for (size_t d = 0; d < g.head_dim; ++d) out[h * g.head_dim + d] += p * kv.value[offset + d];
            }
            rms(out.data() + h * g.head_dim, g.head_dim, g.epsilon);
        }
        return out;
    }
    void residual(Vec& stream, const Vec& branch, const std::string& b, const std::string& kind) const {
        const auto alpha = vector(b + kind + "_resid_alpha.weight"), beta = vector(b + kind + "_resid_beta.weight");
        for (size_t i = 0; i < stream.size(); ++i) stream[i] = stream[i] * alpha[i] + branch[i] * beta[i];
    }
    Vec forward(uint32_t token) {
        if (token >= g.vocab) fail("token outside vocabulary");
        if (position >= context) fail("context exhausted");
        Vec x(g.hidden); row(tensor("token_embd.weight"), token, x.data());
        rms(x.data(), x.size(), g.epsilon);
        for (size_t l = 0; l < g.layers; ++l) {
            const auto b = "blk." + std::to_string(l) + ".";
            Vec cur = x; norm(cur, b + "attn_norm.weight", g.epsilon);
            auto q = mm(b + "attn_q.weight", cur), k = mm(b + "attn_k.weight", cur), v = mm(b + "attn_v.weight", cur);
            auto gate = mm(b + "attn_gate.weight", cur);
            head_norm(q, b + "attn_q_norm.weight"); head_norm(k, b + "attn_k_norm.weight");
            auto attn = attention(l, q, k, v);
            for (size_t i = 0; i < attn.size(); ++i) attn[i] *= sigmoid(gate[i]);
            cur = mm(b + "attn_output.weight", attn); rms(cur.data(), cur.size(), g.post_epsilon);
            residual(x, cur, b, "attn");
            cur = x; norm(cur, b + "ffn_norm.weight", g.epsilon);
            auto latent = mm(b + "ffn_latent_down.weight", cur);
            norm(latent, b + "ffn_pre_exp_norm.weight", g.epsilon);
            auto probs = mm(b + "ffn_gate_inp.weight", cur);
            for (auto& p : probs) p = sigmoid(p);
            const auto bias = vector(b + "exp_probs_b.bias");
            std::vector<size_t> selected(g.experts); std::iota(selected.begin(), selected.end(), size_t(0));
            std::partial_sort(selected.begin(), selected.begin() + g.top_k, selected.end(), [&](size_t a, size_t c) {
                const float sa = probs[a] + bias[a], sc = probs[c] + bias[c];
                return sa == sc ? a < c : sa > sc;
            });
            float sum = 1e-15f;
            for (size_t j = 0; j < g.top_k; ++j) sum += probs[selected[j]];
            Vec merged(g.latent, 0);
            for (size_t j = 0; j < g.top_k; ++j) {
                const size_t e = selected[j];
                auto eg = mm(b + "ffn_gate_exps.weight", latent, e), up = mm(b + "ffn_up_exps.weight", latent, e);
                for (size_t i = 0; i < eg.size(); ++i) eg[i] = eg[i] * sigmoid(eg[i]) * up[i];
                auto down = mm(b + "ffn_down_exps.weight", eg, e);
                // Input-scaled norm is per expert, BEFORE probability weighting.
                norm(down, b + "ffn_post_exp_norm.weight", g.post_epsilon, true);
                const float weight = probs[e] / sum * g.expert_scale;
                for (size_t i = 0; i < merged.size(); ++i) merged[i] += weight * down[i];
            }
            cur = mm(b + "ffn_latent_up.weight", merged);
            norm(cur, b + "post_ffw_norm.weight", g.post_epsilon, true);
            residual(x, cur, b, "ffn"); finite(x);
        }
        norm(x, "output_norm.weight", g.epsilon);
        auto logits = mm("output.weight", x);
        for (auto& f : logits) { f *= g.logit_scale; if (g.softcap) f = g.softcap * std::tanh(f / g.softcap); }
        finite(logits); ++position; return logits;
    }
};
Model::Model(const std::string& path, size_t context, const std::string& backend, int gpu)
    : impl_(new Impl(path, context, backend, gpu)) {}
Model::~Model() = default;
const Geometry& Model::geometry() const { return impl_->g; }
std::vector<float> Model::forward(uint32_t token) { return impl_->forward(token); }
bool Model::is_end(uint32_t token) const { return impl_->end_tokens.count(token) != 0; }

void run(const Options& o) {
    if (o.model.empty() || o.logits_file.empty() || o.tokens.empty() || !o.context ||
        o.tokens.size() > o.context || o.steps > o.context - o.tokens.size() || o.gpu > uint32_t(INT32_MAX))
        fail("invalid execution arguments/context");
    Model model(o.model, o.context, o.backend, int(o.gpu));
    for (auto t : o.tokens) if (t >= model.geometry().vocab) fail("token outside vocabulary");
    // Never overwrite an input shard with output, including equivalent/symlink paths.
    const auto paths = gguf_split_paths(o.model);
    for (const auto& p : paths) {
        std::error_code ec;
        if (std::filesystem::equivalent(p, o.logits_file, ec)) fail("output aliases a model shard");
    }
    std::ofstream out(o.logits_file, std::ios::trunc);
    if (!out) fail("cannot open logits file");
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\"tokens\":[";
    for (size_t i = 0; i < o.tokens.size(); ++i) { if (i) out << ','; out << o.tokens[i]; }
    out << "],\"logits\":[";
    Vec logits;
    for (size_t i = 0; i < o.tokens.size(); ++i) {
        logits = model.forward(o.tokens[i]);
        if (i) out << ',';
        out << '[';
        for (size_t j = 0; j < logits.size(); ++j) { if (j) out << ','; out << logits[j]; }
        out << ']';
        if (!out) fail("cannot write logits file");
    }
    out << "],\"generated_tokens\":[";
    for (size_t i = 0; i < o.steps; ++i) {
        const auto token = uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (i) out << ',';
        out << token;
        if (model.is_end(token)) break;
        if (i + 1 < o.steps) logits = model.forward(token);
    }
    out << "]}\n"; out.close();
    if (!out) fail("cannot finish logits file");
}
} // namespace strata::muse
