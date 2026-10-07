// src/glm/container.cpp - config.json and the safetensors shard index.  See the header for the format.
#include "strata/glm/container.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace strata::glm {
namespace {

// UTF-8 std::string <-> path, without C++20's deprecated u8path or a char8_t string leaking out
std::filesystem::path to_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }
std::string from_path(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

bool slurp(const std::string& path, std::string& out, uint64_t cap, std::string& err) {
    std::ifstream f(to_path(path), std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n < 0 || (uint64_t) n > cap) { err = path + ": size out of range"; return false; }
    f.seekg(0);
    out.resize((size_t) n);
    if (n > 0 && !f.read(out.data(), n)) { err = path + ": short read"; return false; }
    return true;
}

bool get_int(const Json& r, const char* k, int& v, int lo, int hi, std::string& err, bool required = true) {
    const Json* j = r.get(k);
    if (!j || !j->is_num()) {
        if (!required) return true;
        err = std::string("config.json: '") + k + "' is missing";
        return false;
    }
    const double d = j->num;
    if (d < lo || d > hi || d != (double) (int64_t) d) {
        err = std::string("config.json: '") + k + "' = " + std::to_string(d) + " is outside [" + std::to_string(lo) +
              ", " + std::to_string(hi) + "]";
        return false;
    }
    v = (int) d;
    return true;
}

void add_eos(const Json* j, std::vector<int>& eos) {
    if (!j) return;
    auto add = [&](const Json& x) {
        if (!x.is_num()) return;
        const int id = (int) x.num;
        if (std::find(eos.begin(), eos.end(), id) == eos.end()) eos.push_back(id);
    };
    if (j->is_arr()) for (const Json& x : j->arr) add(x);
    else add(*j);
}

}  // namespace

bool load_config(const std::string& dir, GlmConfig& c, std::string& err) {
    std::string text;
    if (!slurp(dir + "/config.json", text, 64ull << 20, err)) return false;
    Json r;
    if (!json_parse(text, r, err)) { err = "config.json: " + err; return false; }
    // The family check: GlmMoeDsaForCausalLM is the only architecture this engine computes.
    const Json* mt = r.get("model_type");
    if (!mt || !mt->is_str() || mt->str != "glm_moe_dsa") {
        err = "config.json: model_type is '" + (mt && mt->is_str() ? mt->str : std::string("?")) +
              "', this engine runs 'glm_moe_dsa' (GLM-5.2 / GLM-5.3)";
        return false;
    }
    c = GlmConfig{};
    if (!get_int(r, "hidden_size", c.hidden, 64, 1 << 16, err)) return false;
    if (!get_int(r, "num_hidden_layers", c.n_layers, 1, 256, err)) return false;
    if (!get_int(r, "num_attention_heads", c.n_heads, 1, 1024, err)) return false;
    if (!get_int(r, "n_routed_experts", c.n_experts, 1, 4096, err)) return false;
    if (!get_int(r, "num_experts_per_tok", c.topk, 1, 64, err)) return false;
    if (!get_int(r, "moe_intermediate_size", c.moe_inter, 64, 1 << 16, err)) return false;
    if (!get_int(r, "intermediate_size", c.dense_inter, 64, 1 << 18, err)) return false;
    if (!get_int(r, "first_k_dense_replace", c.first_dense, 0, c.n_layers, err)) return false;
    if (!get_int(r, "n_shared_experts", c.n_shared, 0, 8, err)) return false;
    if (!get_int(r, "q_lora_rank", c.q_lora, 64, 1 << 16, err)) return false;
    if (!get_int(r, "kv_lora_rank", c.kv_lora, 64, 1024, err)) return false;
    if (!get_int(r, "qk_nope_head_dim", c.qk_nope, 1, 1024, err)) return false;
    if (!get_int(r, "qk_rope_head_dim", c.qk_rope, 2, 256, err)) return false;
    if (!get_int(r, "v_head_dim", c.v_head, 1, 1024, err)) return false;
    if (!get_int(r, "vocab_size", c.vocab, 1, 1 << 22, err)) return false;
    if (!get_int(r, "index_topk", c.index_topk, 0, 1 << 20, err, false)) return false;
    if (!get_int(r, "index_n_heads", c.index_n_heads, 0, 1024, err, false)) return false;
    if (!get_int(r, "index_head_dim", c.index_head_dim, 0, 1024, err, false)) return false;
    if (c.topk > c.n_experts) { err = "config.json: num_experts_per_tok exceeds n_routed_experts"; return false; }
    if (c.qk_rope % 2) { err = "config.json: qk_rope_head_dim must be even"; return false; }
    int n_group = 1;
    if (!get_int(r, "n_group", n_group, 1, 64, err, false)) return false;
    if (n_group != 1) { err = "config.json: n_group must be 1 (group-limited routing is not implemented)"; return false; }
    const Json* sf = r.get("scoring_func");
    if (sf && sf->is_str() && sf->str != "sigmoid") { err = "config.json: scoring_func must be 'sigmoid'"; return false; }
    if (const Json* j = r.get("norm_topk_prob"); j && j->type == Json::Type::Bool) c.norm_topk = j->b;
    if (const Json* j = r.get("routed_scaling_factor"); j && j->is_num()) c.routed_scale = (float) j->num;
    if (const Json* j = r.get("rms_norm_eps"); j && j->is_num()) c.eps = (float) j->num;
    if (const Json* rp = r.get("rope_parameters")) {
        if (const Json* th = rp->get("rope_theta"); th && th->is_num()) c.rope_theta = th->num;
        if (const Json* ty = rp->get("rope_type"); ty && ty->is_str() && ty->str != "default") {
            err = "config.json: rope_type '" + ty->str + "' is not implemented (only 'default')";
            return false;
        }
    } else if (const Json* th = r.get("rope_theta"); th && th->is_num()) {
        c.rope_theta = th->num;
    }
    add_eos(r.get("eos_token_id"), c.eos);
    std::string gtext, gerr;
    if (slurp(dir + "/generation_config.json", gtext, 1ull << 20, gerr)) {   // optional; HF's authority on stops
        Json g;
        if (json_parse(gtext, g, gerr)) add_eos(g.get("eos_token_id"), c.eos);
    }
    return true;
}

bool Container::open(const std::string& dir, std::string& err) {
    namespace fs = std::filesystem;
    dir_ = dir;
    shards_.clear();
    index_.clear();
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(to_path(dir), ec)) {
        if (!e.is_regular_file()) continue;
        const std::string name = from_path(e.path().filename());
        if (name.size() > 12 && name.compare(name.size() - 12, 12, ".safetensors") == 0) shards_.push_back(from_path(e.path()));
    }
    if (ec) { err = "cannot list " + dir + ": " + ec.message(); return false; }
    if (shards_.empty()) { err = "no .safetensors shards in " + dir; return false; }
    std::sort(shards_.begin(), shards_.end());
    for (int s = 0; s < (int) shards_.size(); ++s) {
        std::ifstream f(to_path(shards_[s]), std::ios::binary);
        if (!f) { err = "cannot open " + shards_[s]; return false; }
        uint64_t n = 0;
        if (!f.read((char*) &n, 8) || n == 0 || n > (256ull << 20)) { err = shards_[s] + ": bad header length"; return false; }
        std::string hdr((size_t) n, '\0');
        if (!f.read(hdr.data(), (std::streamsize) n)) { err = shards_[s] + ": short header"; return false; }
        f.seekg(0, std::ios::end);
        const uint64_t fsize = (uint64_t) f.tellg();
        Json h;
        if (!json_parse(hdr, h, err)) { err = shards_[s] + ": " + err; return false; }
        if (!h.is_obj()) { err = shards_[s] + ": header is not an object"; return false; }
        const uint64_t base = 8 + n;
        for (const auto& [name, v] : h.obj) {
            if (name == "__metadata__") continue;
            const Json* dt = v.get("dtype");
            const Json* off = v.get("data_offsets");
            if (!dt || !dt->is_str() || !off || !off->is_arr() || off->arr.size() != 2) {
                err = shards_[s] + ": malformed entry '" + name + "'";
                return false;
            }
            TensorSpan t;
            t.shard = s;
            t.dtype = dt->str;
            const uint64_t a = (uint64_t) off->arr[0].num, b = (uint64_t) off->arr[1].num;
            if (b < a || base + b > fsize) { err = shards_[s] + ": '" + name + "' is out of the file's bounds"; return false; }
            t.offset = base + a;
            t.bytes = b - a;
            if (const Json* sh = v.get("shape"); sh && sh->is_arr())
                for (const Json& d : sh->arr) t.shape.push_back((int64_t) d.num);
            if (!index_.emplace(name, std::move(t)).second) { err = "tensor '" + name + "' appears in two shards"; return false; }
        }
    }
    return true;
}

const TensorSpan* Container::find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &it->second;
}

bool expert_span(const Container& ct, const GlmConfig& c, int layer, int e, ExpertSpan& out, std::string& err) {
    const std::string p = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
    const uint64_t code_plane = (uint64_t) c.hidden * c.moe_inter / 2;
    const uint64_t scale_plane = (uint64_t) c.hidden * c.moe_inter / 64 * 4;
    const char* order[3] = {"down_proj", "gate_proj", "up_proj"};
    const TensorSpan* pl[6];
    for (int i = 0; i < 3; ++i) {
        pl[i] = ct.find(p + order[i] + ".weight");
        pl[3 + i] = ct.find(p + order[i] + ".weight.qs");
        if (!pl[i] || !pl[3 + i]) { err = "missing tensor " + p + order[i] + ".weight(.qs)"; return false; }
        if (pl[i]->dtype != "U8" || pl[3 + i]->dtype != "F32" || pl[i]->bytes != code_plane || pl[3 + i]->bytes != scale_plane) {
            err = p + order[i] + ": not int4-g64 of the configured shape";
            return false;
        }
    }
    out = ExpertSpan{};
    for (int i = 0; i < 6; ++i) {
        ExpertSpan::Run* r = out.n_runs ? &out.runs[out.n_runs - 1] : nullptr;
        // a plane joins the previous run when it is the next plane of the same kind and starts where the run ends
        if (r && i != 3 && r->shard == pl[i]->shard && r->off + r->bytes == pl[i]->offset) {
            r->bytes += pl[i]->bytes;
            ++r->count;
            continue;
        }
        out.runs[out.n_runs++] = ExpertSpan::Run{pl[i]->shard, pl[i]->offset, pl[i]->bytes, i, 1};
    }
    return true;
}

}  // namespace strata::glm
