// src/core/qwen35_layout_test.cpp - the Qwen35MoE (Ornith) architecture guard, on synthetic GGUF headers.
//
// The point of the guard is to REFUSE a malformed artifact before a kernel reads it, with a message naming
// the tensor and the two shapes.  So the test builds a valid Qwen35MoE header, proves it passes, then
// breaks exactly one thing at a time and proves each break is caught.  No model, no GPU, no ggml: the
// header parser is the only dependency, and the tensor data section is never read.
//
//   qwen35_layout_test
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/qwen35.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

void put(std::vector<uint8_t>& b, const void* p, size_t n) {
    const uint8_t* c = (const uint8_t*) p;
    b.insert(b.end(), c, c + n);
}
template <class T> void putv(std::vector<uint8_t>& b, T v) { put(b, &v, sizeof(T)); }
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    putv<uint64_t>(b, s.size());
    put(b, s.data(), s.size());
}

std::vector<uint8_t> enc_str(const std::string& s) {
    std::vector<uint8_t> b; putv<uint32_t>(b, 8); put_str(b, s); return b;
}
std::vector<uint8_t> enc_u32(uint32_t v) {
    std::vector<uint8_t> b; putv<uint32_t>(b, 4); putv<uint32_t>(b, v); return b;
}
std::vector<uint8_t> enc_f32(float v) {
    std::vector<uint8_t> b; putv<uint32_t>(b, 6); putv<float>(b, v); return b;
}
std::vector<uint8_t> enc_u32arr(const std::vector<uint32_t>& v) {
    std::vector<uint8_t> b; putv<uint32_t>(b, 9); putv<uint32_t>(b, 4); putv<uint64_t>(b, v.size());
    for (uint32_t x : v) putv<uint32_t>(b, x);
    return b;
}

struct KV { std::string key; std::vector<uint8_t> value; };
struct Ten { std::string name; std::vector<int64_t> shape; uint32_t type; };

// A minimal, VALID Qwen35MoE header.  The geometry is small (64-wide, 4 layers) but every cross-field rule
// the real artifact obeys is obeyed here: 3:1 recurrent:attention, inner = dt_rank * head_v, key_dim =
// state * groups, query = 2 * heads * head_dim.
struct Fixture {
    int64_t n_layers = 4, n_embd = 64, n_expert = 8, n_used = 2, n_ff = 64, n_ffs = 64;
    int64_t head = 4, head_kv = 2, hd = 16, rope_dim = 8;
    int64_t conv = 4, inner = 64, state = 16, dt_rank = 4, groups = 2, interval = 4;
    int64_t vocab = 32;
    std::string arch = "qwen35moe";
    std::vector<KV> kvs;
    std::vector<Ten> tensors;

    int64_t key_dim() const { return state * groups; }
    int64_t qkv_dim() const { return 2 * key_dim() + inner; }
    bool recr(int64_t l) const { return (l + 1) % interval != 0; }

    void meta() {
        kvs.push_back({"general.architecture", enc_str(arch)});
        kvs.push_back({"qwen35moe.block_count", enc_u32((uint32_t) n_layers)});
        kvs.push_back({"qwen35moe.embedding_length", enc_u32((uint32_t) n_embd)});
        kvs.push_back({"qwen35moe.expert_count", enc_u32((uint32_t) n_expert)});
        kvs.push_back({"qwen35moe.expert_used_count", enc_u32((uint32_t) n_used)});
        kvs.push_back({"qwen35moe.expert_feed_forward_length", enc_u32((uint32_t) n_ff)});
        kvs.push_back({"qwen35moe.expert_shared_feed_forward_length", enc_u32((uint32_t) n_ffs)});
        kvs.push_back({"qwen35moe.attention.head_count", enc_u32((uint32_t) head)});
        kvs.push_back({"qwen35moe.attention.head_count_kv", enc_u32((uint32_t) head_kv)});
        kvs.push_back({"qwen35moe.attention.key_length", enc_u32((uint32_t) hd)});
        kvs.push_back({"qwen35moe.attention.value_length", enc_u32((uint32_t) hd)});
        kvs.push_back({"qwen35moe.attention.layer_norm_rms_epsilon", enc_f32(1e-6f)});
        kvs.push_back({"qwen35moe.rope.dimension_count", enc_u32((uint32_t) rope_dim)});
        kvs.push_back({"qwen35moe.rope.dimension_sections", enc_u32arr({(uint32_t) rope_dim, 0, 0, 0})});
        kvs.push_back({"qwen35moe.rope.freq_base", enc_f32(1e7f)});
        kvs.push_back({"qwen35moe.context_length", enc_u32(262144)});
        kvs.push_back({"qwen35moe.ssm.conv_kernel", enc_u32((uint32_t) conv)});
        kvs.push_back({"qwen35moe.ssm.inner_size", enc_u32((uint32_t) inner)});
        kvs.push_back({"qwen35moe.ssm.state_size", enc_u32((uint32_t) state)});
        kvs.push_back({"qwen35moe.ssm.time_step_rank", enc_u32((uint32_t) dt_rank)});
        kvs.push_back({"qwen35moe.ssm.group_count", enc_u32((uint32_t) groups)});
        kvs.push_back({"qwen35moe.full_attention_interval", enc_u32((uint32_t) interval)});
    }

    void add(const std::string& name, std::vector<int64_t> shape, uint32_t type = 0) {
        tensors.push_back({name, std::move(shape), type});
    }
    void tensors_all() {
        add("token_embd.weight", {n_embd, vocab});
        add("output_norm.weight", {n_embd});
        add("output.weight", {n_embd, vocab});
        for (int64_t l = 0; l < n_layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            add(p + "attn_norm.weight", {n_embd});
            add(p + "post_attention_norm.weight", {n_embd});
            add(p + "ffn_gate_inp.weight", {n_embd, n_expert});
            add(p + "ffn_gate_inp_shexp.weight", {n_embd});
            add(p + "ffn_gate_shexp.weight", {n_embd, n_ffs});
            add(p + "ffn_up_shexp.weight", {n_embd, n_ffs});
            add(p + "ffn_down_shexp.weight", {n_ffs, n_embd});
            add(p + "ffn_gate_exps.weight", {n_embd, n_ff, n_expert});
            add(p + "ffn_up_exps.weight", {n_embd, n_ff, n_expert});
            add(p + "ffn_down_exps.weight", {n_ff, n_embd, n_expert});
            if (recr(l)) {
                add(p + "attn_qkv.weight", {n_embd, qkv_dim()});
                add(p + "attn_gate.weight", {n_embd, inner});
                add(p + "ssm_conv1d.weight", {conv, qkv_dim()});
                add(p + "ssm_dt.bias", {dt_rank});
                add(p + "ssm_a", {dt_rank});
                add(p + "ssm_beta.weight", {n_embd, dt_rank});
                add(p + "ssm_alpha.weight", {n_embd, dt_rank});
                add(p + "ssm_norm.weight", {state});
                add(p + "ssm_out.weight", {inner, n_embd});
            } else {
                add(p + "attn_q.weight", {n_embd, 2 * head * hd});
                add(p + "attn_k.weight", {n_embd, head_kv * hd});
                add(p + "attn_v.weight", {n_embd, head_kv * hd});
                add(p + "attn_output.weight", {head * hd, n_embd});
                add(p + "attn_q_norm.weight", {hd});
                add(p + "attn_k_norm.weight", {hd});
            }
        }
    }

    std::filesystem::path write(const std::string& tag) const {
        std::vector<uint8_t> b;
        putv<uint32_t>(b, 0x46554747u);
        putv<uint32_t>(b, 3);
        putv<uint64_t>(b, tensors.size());
        putv<uint64_t>(b, kvs.size());
        for (const KV& kv : kvs) { put_str(b, kv.key); put(b, kv.value.data(), kv.value.size()); }
        for (const Ten& t : tensors) {
            put_str(b, t.name);
            putv<uint32_t>(b, (uint32_t) t.shape.size());
            for (int64_t d : t.shape) putv<uint64_t>(b, (uint64_t) d);
            putv<uint32_t>(b, t.type);
            putv<uint64_t>(b, 0);
        }
        b.resize((b.size() + 31) / 32 * 32 + 64, 0);   // data_start <= file size
        const auto path = std::filesystem::temp_directory_path() / ("strata_qwen35_" + tag + ".gguf");
        std::ofstream(path, std::ios::binary).write((const char*) b.data(), (std::streamsize) b.size());
        return path;
    }
};

// Runs check_qwen35_all on a fixture and returns whether it PASSED (the caller asserts pass/fail).
bool validate(const Fixture& f, const std::string& tag, std::string& err, strata::core::Qwen35Geometry* out = nullptr) {
    err.clear();
    const auto path = f.write(tag);
    bool ok = false;
    try {
        strata::GgufModel model({path.string()});
        strata::core::Qwen35Geometry g;
        ok = strata::core::check_qwen35_all(model, g, err);
        if (ok && out) *out = g;
    } catch (const std::exception& e) {
        err = std::string("exception: ") + e.what();
    }
    std::filesystem::remove(path);
    return ok;
}

bool mentions(const std::string& err, const char* needle) { return err.find(needle) != std::string::npos; }

}  // namespace

int main(int argc, char** argv) {
    std::printf("qwen35_layout_test\n");
    using namespace strata::core;

    // With a path, validate a REAL artifact header instead of the synthetic cases below.  The header alone is
    // enough (tensor data is never read), so this runs against an ~11 MB prefix of the 20 GB file:
    //   python tools/ornith_inspect.py --repo ... --file ... --out layout.txt --header-out /tmp/ornith-header.gguf
    if (argc > 1) {
        std::string err;
        Qwen35Geometry g;
        try {
            strata::GgufModel model({argv[1]});
            if (!check_qwen35_all(model, g, err)) {
                std::printf("  %s: FAIL %s\n", argv[1], err.c_str());
                return 1;
            }
        } catch (const std::exception& e) {
            std::printf("  %s: exception %s\n", argv[1], e.what());
            return 1;
        }
        std::printf("  %s: qwen35moe ok\n", argv[1]);
        std::printf("    layers=%lld embd=%lld experts=%lld top=%lld ff_exp=%lld ff_shexp=%lld\n",
                    (long long) g.n_layers, (long long) g.n_embd, (long long) g.n_expert,
                    (long long) g.n_expert_used, (long long) g.n_ff_exp, (long long) g.n_ff_shexp);
        std::printf("    heads=%lld kv=%lld head_dim=%lld rope=%lld sections=[%lld,%lld,%lld,%lld]\n",
                    (long long) g.n_head, (long long) g.n_head_kv, (long long) g.head_dim,
                    (long long) g.rope_dim, (long long) g.rope_sections[0], (long long) g.rope_sections[1],
                    (long long) g.rope_sections[2], (long long) g.rope_sections[3]);
        std::printf("    gdn conv=%lld inner=%lld state=%lld groups=%lld v_heads=%lld key_dim=%lld value_dim=%lld\n",
                    (long long) g.ssm_conv_kernel, (long long) g.ssm_inner, (long long) g.ssm_state,
                    (long long) g.ssm_groups, (long long) g.ssm_dt_rank, (long long) g.key_dim(),
                    (long long) g.value_dim());
        std::printf("    recurrent=%lld attention=%lld interval=%lld vocab=%lld rms_eps=%.3g\n",
                    (long long) g.n_recurrent_layers(), (long long) g.n_attention_layers(),
                    (long long) g.full_attention_interval, (long long) g.n_vocab, (double) g.rms_eps);
        return 0;
    }

    // ---- 1. the valid header passes and the geometry is read back
    {
        Fixture f; f.meta(); f.tensors_all();
        Qwen35Geometry g; std::string err;
        const bool ok = validate(f, "valid", err, &g);
        check(ok, "a valid qwen35moe header is accepted");
        if (!ok) std::printf("      %s\n", err.c_str());
        check(g.n_layers == 4 && g.n_embd == 64 && g.n_expert == 8, "geometry reads back");
        check(g.key_dim() == 32 && g.value_dim() == 64 && g.qkv_dim() == 128, "GDN dimensions derive");
        check(g.n_recurrent_layers() == 3 && g.n_attention_layers() == 1, "3:1 recurrent:attention split");
        check(!g.output_tied, "output.weight present -> not tied");
    }

    // ---- 2. wrong architecture is refused
    {
        Fixture f; f.arch = "qwen4exp"; f.meta(); f.tensors_all();
        std::string err;
        check(!validate(f, "arch", err) && mentions(err, "not 'qwen35moe'") , "architecture 'qwen4exp' is refused");
    }

    // ---- 3. a wrong attention shape names the tensor and both shapes
    {
        Fixture f; f.meta(); f.tensors_all();
        for (auto& t : f.tensors) if (t.name == "blk.3.attn_q.weight") t.shape[1] = 999;
        std::string err;
        const bool ok = validate(f, "shape", err);
        check(!ok && mentions(err, "blk.3.attn_q.weight") && mentions(err, "999") && mentions(err, "128"),
              "attn_q with a bad output width is refused, naming tensor and shapes");
    }

    // ---- 4. a missing recurrent tensor is refused by name
    {
        Fixture f; f.meta(); f.tensors_all();
        for (size_t i = 0; i < f.tensors.size(); ++i)
            if (f.tensors[i].name == "blk.1.ssm_a") { f.tensors.erase(f.tensors.begin() + (long) i); break; }
        std::string err;
        check(!validate(f, "missing", err) && mentions(err, "blk.1.ssm_a"), "a missing ssm_a is refused by name");
    }

    // ---- 5. a Qwen4Exp-only tensor under a Qwen35 block is refused
    {
        Fixture f; f.meta(); f.tensors_all();
        f.add("blk.2.hc_attn_down.weight", {f.n_embd, 320}, 0);
        std::string err;
        check(!validate(f, "hc", err) && mentions(err, "Qwen4Exp tensor"),
              "a hyper-connection tensor in a qwen35moe block is refused");
    }
    {
        Fixture f; f.meta(); f.tensors_all();
        f.add("blk.2.indexer.k_proj.weight", {f.n_embd, 128}, 0);
        std::string err;
        check(!validate(f, "indexer", err) && mentions(err, "Qwen4Exp tensor"),
              "a QSA indexer tensor in a qwen35moe block is refused");
    }

    // ---- 6. layer-family tensors cannot cross: a dense projection on a recurrent layer, GDN on attention
    {
        Fixture f; f.meta(); f.tensors_all();
        f.add("blk.1.attn_q.weight", {f.n_embd, 2 * f.head * f.hd}, 0);
        std::string err;
        check(!validate(f, "cross1", err) && mentions(err, "belongs to a full-attention layer"),
              "a dense attn_q on a recurrent layer is refused");
    }
    {
        Fixture f; f.meta(); f.tensors_all();
        f.add("blk.3.ssm_out.weight", {f.inner, f.n_embd}, 0);
        std::string err;
        check(!validate(f, "cross2", err) && mentions(err, "belongs to a recurrent layer"),
              "a GDN ssm_out on a full-attention layer is refused");
    }

    // ---- 7. a norm in the wrong type is refused (the kernels read F32)
    {
        Fixture f; f.meta(); f.tensors_all();
        for (auto& t : f.tensors) if (t.name == "blk.0.attn_norm.weight") t.type = 8;   // Q8_0
        std::string err;
        check(!validate(f, "type", err) && mentions(err, "F32"),
              "a quantized norm is refused: the kernel reads F32");
    }

    // ---- 8. missing metadata is refused, naming the key
    {
        Fixture f; f.meta(); f.tensors_all();
        for (size_t i = 0; i < f.kvs.size(); ++i)
            if (f.kvs[i].key == "qwen35moe.ssm.state_size") { f.kvs.erase(f.kvs.begin() + (long) i); break; }
        std::string err;
        check(!validate(f, "meta", err) && mentions(err, "qwen35moe.ssm.state_size"),
              "missing ssm.state_size is refused by key");
    }

    // ---- 9. an inconsistent value_length is refused
    {
        Fixture f; f.meta(); f.tensors_all();
        for (auto& kv : f.kvs) if (kv.key == "qwen35moe.attention.value_length") kv.value = enc_u32(32);
        std::string err;
        check(!validate(f, "vlen", err) && mentions(err, "value_length"),
              "value_length != key_length is refused");
    }

    // ---- 10. a missing output.weight means a tied head, and still passes
    {
        Fixture f; f.meta(); f.tensors_all();
        for (size_t i = 0; i < f.tensors.size(); ++i)
            if (f.tensors[i].name == "output.weight") { f.tensors.erase(f.tensors.begin() + (long) i); break; }
        Qwen35Geometry g; std::string err;
        const bool ok = validate(f, "tied", err, &g);
        check(ok && g.output_tied, "a missing output.weight is accepted as a tied head");
    }

    std::printf("qwen35_layout_test: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
