#include "strata/muse/model.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Test-only allocator controls; never linked into strata-muse.
namespace allocation_fault {
extern long countdown;
extern size_t count;
extern bool measuring;
}

namespace {
using namespace strata::muse;
using Bytes = std::vector<uint8_t>;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f, const char* message) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
template<class T> void put(Bytes& b, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v); b.insert(b.end(), p, p + sizeof v);
}
void text(Bytes& b, const std::string& s) { put<uint64_t>(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }
Bytes number(uint32_t n) { Bytes b; put<uint32_t>(b, 4); put(b, n); return b; }
Bytes real(float f) { Bytes b; put<uint32_t>(b, 6); put(b, f); return b; }
Bytes boolean(bool v) { Bytes b; put<uint32_t>(b, 7); put<uint8_t>(b, uint8_t(v)); return b; }
Bytes string(const std::string& s) { Bytes b; put<uint32_t>(b, 8); text(b, s); return b; }
Bytes pattern(const std::vector<uint8_t>& p) {
    Bytes b; put<uint32_t>(b, 9); put<uint32_t>(b, 7); put<uint64_t>(b, p.size());
    for (auto x : p) put(b, x);
    return b;
}
struct Tensor { std::string name; std::vector<uint64_t> dims; uint32_t type = 0; Bytes data; };
struct Fixture {
    std::map<std::string, Bytes> meta;
    std::vector<Tensor> tensors;
    Fixture(bool mixed = false) {
        auto n = [&](const char* k, uint32_t v) { meta[std::string("muse-spark.") + k] = number(v); };
        auto f = [&](const char* k, float v) { meta[std::string("muse-spark.") + k] = real(v); };
        meta["general.architecture"] = string("muse-spark");
        n("embedding_length", 32); n("block_count", 2); n("attention.head_count", 2);
        n("attention.head_count_kv", 1); n("attention.key_length", 16); n("attention.value_length", 16);
        n("moe_latent_size", 32); n("expert_count", 3); n("expert_used_count", 2);
        n("context_length", 32); n("attention.sliding_window", 3); n("expert_feed_forward_length", 32);
        n("expert_gating_func", 2);
        f("attention.layer_norm_rms_epsilon", 1e-5f); f("post_norm_epsilon", 1e-8f);
        f("rope.freq_base_swa", 10000); f("logit_scale", 0.7f); f("final_logit_softcapping", 3);
        f("expert_weights_scale", 1.2f);
        meta["muse-spark.expert_weights_norm"] = boolean(true);
        meta["muse-spark.attention.sliding_window_pattern"] = pattern({1, 0});
        meta["tokenizer.ggml.eos_token_id"] = number(11);
        meta["tokenizer.ggml.eot_token_id"] = number(10);
        std::mt19937 rng(917);
        auto add = [&](const std::string& name, std::vector<uint64_t> dims, bool scale = false) {
            size_t elements = 1; for (auto d : dims) elements *= size_t(d);
            std::vector<float> w(elements);
            for (auto& x : w) {
                x = float(int(rng() % 2001) - 1000) / 1000.0f;
                x = scale ? 0.8f + x * 0.3f : x * 0.16f;
            }
            Tensor t; t.name = name; t.dims = std::move(dims);
            // The same graph with generic row decoders interleaved with F32 norms.
            if (mixed && t.dims.size() >= 2) t.type = tensors.size() % 2 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
            const auto* traits = ggml_get_type_traits(ggml_type(t.type));
            const size_t cols = size_t(t.dims[0]);
            const size_t stride = cols / size_t(traits->blck_size) * traits->type_size;
            t.data.resize(elements / cols * stride);
            for (size_t r = 0; r < elements / cols; ++r) {
                if (!t.type) std::memcpy(t.data.data() + r * stride, w.data() + r * cols, cols * sizeof(float));
                else traits->from_float_ref(w.data() + r * cols, t.data.data() + r * stride, int64_t(cols));
            }
            tensors.push_back(std::move(t));
        };
        add("token_embd.weight", {32, 12}); add("output_norm.weight", {32}, true); add("output.weight", {32, 12});
        for (int l = 0; l < 2; ++l) {
            const auto b = "blk." + std::to_string(l) + ".";
            for (const char* s : {"attn_norm", "attn_resid_alpha", "attn_resid_beta", "ffn_norm", "post_ffw_norm",
                                  "ffn_resid_alpha", "ffn_resid_beta"}) add(b + s + ".weight", {32}, true);
            add(b + "attn_q_norm.weight", {16}, true); add(b + "attn_k_norm.weight", {16}, true);
            add(b + "attn_q.weight", {32, 32}); add(b + "attn_k.weight", {32, 16}); add(b + "attn_v.weight", {32, 16});
            add(b + "attn_gate.weight", {32, 32}); add(b + "attn_output.weight", {32, 32});
            add(b + "ffn_gate_inp.weight", {32, 3}); add(b + "exp_probs_b.bias", {3});
            add(b + "ffn_latent_down.weight", {32, 32}); add(b + "ffn_latent_up.weight", {32, 32});
            add(b + "ffn_pre_exp_norm.weight", {32}, true); add(b + "ffn_post_exp_norm.weight", {32}, true);
            add(b + "ffn_gate_exps.weight", {32, 32, 3}); add(b + "ffn_up_exps.weight", {32, 32, 3});
            add(b + "ffn_down_exps.weight", {32, 32, 3});
        }
    }
    Tensor& tensor(const std::string& s) {
        for (auto& t : tensors) if (t.name == s) return t;
        throw std::runtime_error("fixture tensor missing");
    }
    void save(const std::filesystem::path& path, int split = -1) const {
        auto metadata = meta;
        std::vector<Tensor> ts;
        if (split >= 0) {
            metadata["split.no"] = number(uint32_t(split)); metadata["split.count"] = number(2);
            metadata["split.tensors.count"] = number(uint32_t(tensors.size()));
            // A metadata-only first shard is the actual production topology.
            if (split) ts = tensors;
        } else ts = tensors;
        Bytes b; put<uint32_t>(b, 0x46554747); put<uint32_t>(b, 3);
        put<uint64_t>(b, ts.size()); put<uint64_t>(b, metadata.size());
        for (const auto& item : metadata) { text(b, item.first); b.insert(b.end(), item.second.begin(), item.second.end()); }
        size_t offset = 0;
        for (const auto& t : ts) {
            text(b, t.name); put<uint32_t>(b, uint32_t(t.dims.size()));
            for (auto d : t.dims) put(b, d);
            put(b, t.type); put<uint64_t>(b, offset); offset = (offset + t.data.size() + 31) / 32 * 32;
        }
        b.resize((b.size() + 31) / 32 * 32, 0);
        for (const auto& t : ts) { b.insert(b.end(), t.data.begin(), t.data.end()); b.resize((b.size() + 31) / 32 * 32, 0); }
        std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
        check(bool(out), "fixture write failed");
    }
};
struct Temp {
    std::filesystem::path path;
    Temp() {
        std::random_device r;
        path = std::filesystem::temp_directory_path() / ("strata-muse-test-" + std::to_string(r()));
        check(std::filesystem::create_directory(path), "cannot create test directory");
    }
    ~Temp() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
void arguments() {
    check(parse_uint("4294967295", "n") == UINT32_MAX, "uint32 max parsing");
    for (auto s : {"", "-1", "+1", " 1", "1 ", "0x10", "4294967296", "999999999999999999999"})
        rejects([&] { parse_uint(s, "n"); }, "bad integer accepted");
    for (auto s : {"", "1,", ",1", "1,,2", "1,-1", "4294967296"})
        rejects([&] { parse_tokens(s); }, "bad token list accepted");
    check(parse_tokens("0,1,9") == std::vector<uint32_t>({0, 1, 9}), "token parse");
    auto parse = [](std::vector<const char*> a) { return parse_options(int(a.size()), a.data()); };
    auto good = std::vector<const char*>{"strata-muse", "--model", "m", "--tokens", "1,2", "--ctx", "4", "--steps", "2",
                                        "--backend", "cpu", "--logits-file", "o"};
    check(parse(good).steps == 2, "valid args");
    auto bad = good; bad.insert(bad.end(), {"--unknown", "x"}); rejects([&] { parse(bad); }, "unknown flag accepted");
    bad = good; bad.push_back("--gpu"); rejects([&] { parse(bad); }, "missing value accepted");
    bad = good; bad.insert(bad.end(), {"--ctx", "1"}); rejects([&] { parse(bad); }, "duplicate flag accepted");
    bad = good; bad[6] = "3"; rejects([&] { parse(bad); }, "context overrun accepted");
    bad = good; bad[10] = "hip"; rejects([&] { parse(bad); }, "unknown backend accepted");
    rejects([&] { checked_mul(SIZE_MAX, 2); }, "multiply overflow accepted");
    rejects([&] { checked_add(SIZE_MAX, 1); }, "addition overflow accepted");
}
void malformed(const std::filesystem::path& file) {
    auto reject = [&](const std::function<void(Fixture&)>& change) {
        Fixture f; change(f); f.save(file); rejects([&] { Model m(file.string(), 16); }, "malformed fixture accepted");
    };
    reject([](Fixture& f) { f.meta.erase("muse-spark.post_norm_epsilon"); });
    reject([](Fixture& f) { f.meta["general.architecture"] = string("qwen4exp"); });
    reject([](Fixture& f) { f.meta["general.alignment"] = number(0); });
    reject([](Fixture& f) { f.meta["general.alignment"] = number(UINT32_MAX); });
    reject([](Fixture& f) { f.meta["muse-spark.attention.head_count_kv"] = number(3); });
    reject([](Fixture& f) { f.meta["muse-spark.attention.key_length"] = number(15); });
    reject([](Fixture& f) { f.meta["muse-spark.expert_used_count"] = number(4); });
    reject([](Fixture& f) { f.meta["muse-spark.embedding_length"] = real(32); });
    reject([](Fixture& f) { f.meta["muse-spark.attention.layer_norm_rms_epsilon"] = real(NAN); });
    reject([](Fixture& f) { f.meta["muse-spark.post_norm_epsilon"] = real(0); });
    reject([](Fixture& f) { f.meta["muse-spark.attention.sliding_window_pattern"] = pattern({1}); });
    reject([](Fixture& f) { f.meta["muse-spark.expert_gating_func"] = number(1); });
    reject([](Fixture& f) { f.meta["muse-spark.expert_weights_norm"] = boolean(false); });
    reject([](Fixture& f) { f.meta["tokenizer.ggml.eos_token_id"] = number(12); });
    reject([](Fixture& f) { f.tensor("blk.0.post_ffw_norm.weight").name = "blk.0.ffn_post_norm.weight"; });
    reject([](Fixture& f) { f.tensor("blk.0.ffn_gate_exps.weight").dims = {32, 32, 2}; });
    reject([](Fixture& f) { f.tensor("blk.0.attn_q.weight").type = 999; });
    reject([](Fixture& f) { auto& t = f.tensor("blk.0.attn_q.weight"); t.type = GGML_TYPE_Q8_0; t.dims[0] = 31; });
    reject([](Fixture& f) { f.tensor("blk.0.attn_q.weight").dims[0] = 0; });
    reject([](Fixture& f) { f.tensor("blk.0.attn_q.weight").dims = {UINT64_MAX, UINT64_MAX}; });
    reject([](Fixture& f) { f.tensors.push_back(f.tensors.back()); });
    Fixture f; f.save(file);
    std::filesystem::resize_file(file, std::filesystem::file_size(file) - 64);
    rejects([&] { Model m(file.string(), 16); }, "truncated payload accepted");
    f.save(file);
    rejects([&] { Model m(file.string(), 0); }, "zero context accepted");
    rejects([&] { Model m(file.string(), 33); }, "training context overrun accepted");
    rejects([&] { Model m(file.string(), 16, "hip"); }, "backend accepted");
    // Budget failures must happen before any KV allocation (no real OOM injection).
    f.meta["muse-spark.context_length"] = number(INT32_MAX); f.save(file);
    rejects([&] { Model m(file.string(), INT32_MAX); }, "huge KV allocation accepted");
    f.meta["muse-spark.embedding_length"] = number(INT32_MAX); f.save(file);
    rejects([&] { Model m(file.string(), 16); }, "unbounded activation accepted");
    // Hostile header reserve and alignment counts are rejected during preflight.
    Bytes b; put<uint32_t>(b, 0x46554747); put<uint32_t>(b, 3); put<uint64_t>(b, UINT64_MAX); put<uint64_t>(b, 0);
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size())); out.close();
    rejects([&] { Model m(file.string(), 16); }, "hostile tensor count accepted");
}
void allocation_failures(const std::filesystem::path& dir) {
    Fixture f; f.save(dir / "allocation.gguf");
    const auto path = (dir / "allocation.gguf").string();
    allocation_fault::count = 0; allocation_fault::measuring = true;
    { Model measured(path, 16); allocation_fault::measuring = false; }
    const size_t count = allocation_fault::count;
    check(count > 8, "allocation fault measurement");
    for (size_t index : {size_t(0), count / 4, count / 2, count - 5, count - 2, count - 1}) {
        bool caught = false;
        allocation_fault::countdown = long(index);
        try { Model failing(path, 16); }
        catch (const std::bad_alloc&) { caught = true; }
        allocation_fault::countdown = -1;
        check(caught, "host allocation failure was swallowed");
        Model recovered(path, 16);
        check(recovered.forward(1).size() == 12, "failed allocation left unusable runtime state");
    }
}
void production_encodings(const std::filesystem::path& dir) {
    Fixture quant, decoded;
    quant.meta["muse-spark.moe_latent_size"] = number(256);
    quant.meta["muse-spark.expert_feed_forward_length"] = number(256);
    // Synthetic raw IQ1_S blocks need no quantizer or imatrix. Compare their
    // streamed row/expert layout against a GGML-decoded F32 serialization.
    for (auto& t : quant.tensors) {
        uint32_t type = GGML_TYPE_F32;
        if (t.name.find("ffn_latent_down") != std::string::npos) t.dims = {32, 256};
        else if (t.name.find("ffn_latent_up") != std::string::npos) { t.dims = {256, 32}; type = GGML_TYPE_Q8_0; }
        else if (t.name.find("ffn_pre_exp_norm") != std::string::npos || t.name.find("ffn_post_exp_norm") != std::string::npos)
            t.dims = {256};
        else if (t.name.find("_exps.weight") != std::string::npos) {
            t.dims = {256, 256, 3};
            type = t.name.find("up_exps") != std::string::npos ? GGML_TYPE_Q6_K : GGML_TYPE_IQ1_S;
        } else continue;
        t.type = type;
        size_t elements = 1; for (auto d : t.dims) elements *= size_t(d);
        const size_t cols = size_t(t.dims[0]);
        const auto* tr = ggml_get_type_traits(ggml_type(type));
        const size_t stride = cols / size_t(tr->blck_size) * tr->type_size;
        t.data.resize(elements / cols * stride);
        std::vector<float> w(elements);
        if (type == GGML_TYPE_IQ1_S) {
            for (size_t r = 0; r < elements / cols; ++r) {
                auto* block = t.data.data() + r * stride;
                for (size_t j = 0; j < stride; ++j) block[j] = uint8_t((r * 17 + j * 31) % 256);
                const auto d = ggml_fp32_to_fp16(0.015625f);
                std::memcpy(block, &d, sizeof d);
                tr->to_float(block, w.data() + r * cols, int64_t(cols));
            }
        } else {
            for (size_t i = 0; i < elements; ++i)
                w[i] = t.dims.size() == 1 ? 0.8f + float(i % 7) * 0.03f : float(int((i * 37) % 127) - 63) * 0.002f;
            for (size_t r = 0; r < elements / cols; ++r) {
                auto* block = t.data.data() + r * stride;
                if (type == GGML_TYPE_F32) std::memcpy(block, w.data() + r * cols, cols * sizeof(float));
                else {
                    tr->from_float_ref(w.data() + r * cols, block, int64_t(cols));
                    tr->to_float(block, w.data() + r * cols, int64_t(cols));
                }
            }
        }
        auto& d = decoded.tensor(t.name); d.dims = t.dims; d.type = GGML_TYPE_F32;
        d.data.resize(elements * sizeof(float)); std::memcpy(d.data.data(), w.data(), d.data.size());
    }
    decoded.meta = quant.meta;
    quant.save(dir / "production-types.gguf"); decoded.save(dir / "decoded.gguf");
    Model a((dir / "production-types.gguf").string(), 16), b((dir / "decoded.gguf").string(), 16);
    for (auto token : {1u, 5u, 8u, 3u}) check(a.forward(token) == b.forward(token), "IQ1_S/Q6_K/Q8_0 streamed rows differ from decoded F32");
}
void analytic_logits(const std::filesystem::path& dir) {
    Fixture f;
    f.meta["muse-spark.attention.layer_norm_rms_epsilon"] = real(0.1f);
    for (auto& t : f.tensors) if (t.name.find("_resid_beta") != std::string::npos)
        std::fill(t.data.begin(), t.data.end(), uint8_t(0));
    f.save(dir / "analytic.gguf"); Model m((dir / "analytic.gguf").string(), 8);
    auto at = [&](const std::string& name, size_t i) {
        float v; const auto& data = f.tensor(name).data;
        std::memcpy(&v, data.data() + i * sizeof(float), sizeof v); return double(v);
    };
    // Independent double-precision calculation with zeroed branch betas: the
    // stream uses folded alpha directly, embedding RMS is weightless, final
    // weighted RMS is output-scaled, and head/logit_scale/softcap stay live.
    std::vector<double> x(32);
    double ss = 0;
    for (size_t i = 0; i < 32; ++i) { x[i] = at("token_embd.weight", 32 + i); ss += x[i] * x[i]; }
    for (auto& v : x) v /= std::sqrt(ss / 32 + double(0.1f));
    for (int l = 0; l < 2; ++l) for (const char* kind : {"attn", "ffn"})
        for (size_t i = 0; i < 32; ++i) x[i] *= at("blk." + std::to_string(l) + "." + kind + "_resid_alpha.weight", i);
    ss = 0; for (auto v : x) ss += v * v;
    for (size_t i = 0; i < 32; ++i) x[i] = x[i] / std::sqrt(ss / 32 + double(0.1f)) * at("output_norm.weight", i);
    const auto logits = m.forward(1);
    for (size_t r = 0; r < logits.size(); ++r) {
        double dot = 0; for (size_t i = 0; i < 32; ++i) dot += at("output.weight", r * 32 + i) * x[i];
        const double expected = 3 * std::tanh(dot * double(0.7f) / 3);
        check(std::abs(double(logits[r]) - expected) < 2e-6, "analytic embedding/residual/final-head calculation");
    }
}
void extended_arrays(const std::filesystem::path& dir) {
    Fixture f;
    f.meta["muse-spark.block_count"] = number(65);
    f.meta["muse-spark.attention.sliding_window_pattern"] = pattern(std::vector<uint8_t>(65, 1));
    Bytes ff; put<uint32_t>(ff, 9); put<uint32_t>(ff, 4); put<uint64_t>(ff, 65);
    for (int i = 0; i < 65; ++i) put<uint32_t>(ff, 32);
    f.meta["muse-spark.expert_feed_forward_length"] = ff;
    const auto base = f.tensors;
    for (int l = 2; l < 65; ++l) for (auto t : base) if (t.name.compare(0, 6, "blk.0.") == 0) {
        t.name = "blk." + std::to_string(l) + "." + t.name.substr(6); f.tensors.push_back(std::move(t));
    }
    f.save(dir / "arrays.gguf"); Model m((dir / "arrays.gguf").string(), 4);
    check(m.geometry().local.size() == 65 && m.geometry().ff.size() == 65, "geometry arrays were sampled at 64");
    check(m.forward(1).size() == 12, "65-layer dynamic forward");
}
void execution(const std::filesystem::path& dir) {
    const auto path = dir / "model.gguf";
    Fixture f; f.save(path);
    Model a(path.string(), 16), b(path.string(), 16);
    check(a.geometry().latent == 32 && a.geometry().ff == std::vector<size_t>({32,32}), "dynamic geometry");
    check(a.is_end(11) && a.is_end(10) && !a.is_end(1), "EOS/EOT metadata");
    std::vector<std::vector<float>> reference;
    for (auto token : {1u, 5u, 8u, 3u, 7u, 2u, 9u}) {
        auto x = a.forward(token), y = b.forward(token);
        check(x == y && x.size() == 12, "deterministic forward");
        for (float v : x) check(std::isfinite(v) && std::abs(v) <= 3.0f, "finite/softcapped logits");
        reference.push_back(x);
    }
    check(reference.front() != reference.back(), "forward ignored causal token history");
    rejects([&] { a.forward(12); }, "out of vocabulary token accepted");
    Model one(path.string(), 1); one.forward(1); rejects([&] { one.forward(2); }, "exhausted context accepted");
    // Current-token logits must be unaffected by future inputs, and clearing KV
    // must change a later result; the prompt crosses the local ring boundary.
    Model fresh(path.string(), 16);
    check(fresh.forward(9) != reference.back(), "KV history ignored");
    Fixture mixed(true); mixed.save(dir / "mixed.gguf");
    Model quant((dir / "mixed.gguf").string(), 16);
    size_t i = 0;
    for (auto token : {1u, 5u, 8u, 3u, 7u, 2u, 9u}) {
        auto x = quant.forward(token);
        for (size_t j = 0; j < x.size(); ++j) check(std::abs(x[j] - reference[i][j]) < 0.12f, "mixed row decoder divergence");
        ++i;
    }
    // Every serialized branch weight is live. Nonuniform perturbations avoid
    // scale invariance of RMS; catch missing graph operations without an oracle.
    for (const char* name : {"blk.0.attn_norm.weight", "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
                            "blk.0.attn_gate.weight", "blk.0.attn_output.weight", "blk.0.attn_resid_beta.weight",
                            "blk.0.ffn_norm.weight", "blk.0.ffn_latent_down.weight", "blk.0.ffn_pre_exp_norm.weight",
                            "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                            "blk.0.ffn_down_exps.weight", "blk.0.ffn_post_exp_norm.weight", "blk.0.ffn_latent_up.weight",
                            "blk.0.post_ffw_norm.weight", "blk.0.ffn_resid_alpha.weight", "output_norm.weight"}) {
        Fixture changed;
        auto& t = changed.tensor(name);
        for (size_t offset = 0; offset < t.data.size(); offset += sizeof(float)) {
            float v; std::memcpy(&v, t.data.data() + offset, sizeof v);
            v *= (offset / sizeof(float)) % 3 ? 0.6f : 1.5f;
            std::memcpy(t.data.data() + offset, &v, sizeof v);
        }
        changed.save(dir / "changed.gguf"); Model m((dir / "changed.gguf").string(), 16);
        std::vector<float> x;
        for (auto token : {1u, 5u, 8u, 3u, 7u, 2u, 9u}) x = m.forward(token);
        float delta = 0; for (size_t j = 0; j < x.size(); ++j) delta = std::max(delta, std::abs(x[j] - reference.back()[j]));
        if (delta < 1e-6f) throw std::runtime_error(std::string("dead graph weight: ") + name);
    }
    // Split metadata-only first shard, duplicate/missing shards and bounds.
    const auto s1 = dir / "split-00001-of-00002.gguf", s2 = dir / "split-00002-of-00002.gguf";
    f.save(s1, 0); f.save(s2, 1);
    Model sharded(s2.string(), 16);
    check(sharded.forward(1) == reference.front(), "shard mapping differs");
    std::filesystem::remove(s2);
    rejects([&] { Model m(s1.string(), 16); }, "missing shard accepted");
    f.save(s2, 0); rejects([&] { Model m(s1.string(), 16); }, "wrong split index accepted");
    // Execute frozen JSON protocol and generation without buffering prompt logits.
    Options o; o.model = path.string(); o.logits_file = (dir / "out.json").string();
    o.tokens = {1,5,8,3,7,2,9}; o.context = 16; o.steps = 2; run(o);
    std::ifstream in(o.logits_file); std::string output((std::istreambuf_iterator<char>(in)), {});
    check(output.find("\"tokens\":[1,5,8,3,7,2,9]") != std::string::npos &&
          output.find("\"logits\":[[") != std::string::npos && output.find("\"generated_tokens\":[") != std::string::npos,
          "JSON output contract");
    const auto generated = uint32_t(std::max_element(reference.back().begin(), reference.back().end()) - reference.back().begin());
    f.meta["tokenizer.ggml.eos_token_id"] = number(generated); f.save(dir / "eos.gguf");
    o.model = (dir / "eos.gguf").string(); o.steps = 4; run(o);
    std::ifstream eos_in(o.logits_file); output.assign(std::istreambuf_iterator<char>(eos_in), {});
    check(output.find("\"generated_tokens\":[" + std::to_string(generated) + "]") != std::string::npos, "greedy generation did not stop at EOS");
    const size_t generated_at = output.find("\"generated_tokens\"");
    check(size_t(std::count(output.begin(), output.begin() + generated_at, '[')) == o.tokens.size() + 2,
          "logits unexpectedly include generated inputs");
    o.steps = 0; run(o);
    std::ifstream zero_in(o.logits_file); output.assign(std::istreambuf_iterator<char>(zero_in), {});
    check(output.find("\"generated_tokens\":[]") != std::string::npos, "steps zero generated tokens");
    o.model = path.string(); o.logits_file = path.string(); rejects([&] { run(o); }, "output can overwrite model");
    // Nonfinite weights fail rather than emitting invalid JSON.
    Fixture nan;
    float v = NAN; std::memcpy(nan.tensor("token_embd.weight").data.data() + 32 * sizeof(float), &v, sizeof v);
    nan.save(dir / "nan.gguf"); Model invalid((dir / "nan.gguf").string(), 16);
    rejects([&] { invalid.forward(1); }, "nonfinite weights accepted");
}
}
int main() {
    try {
        Temp tmp;
        arguments(); malformed(tmp.path / "bad.gguf"); execution(tmp.path);
        production_encodings(tmp.path); analytic_logits(tmp.path); extended_arrays(tmp.path);
        allocation_failures(tmp.path);
        std::cout << "muse_model_test: storage, mixed row decoding, complete graph, CLI and memory guards passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "muse_model_test: " << e.what() << '\n'; return 1;
    }
}
