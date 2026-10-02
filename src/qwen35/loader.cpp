// src/qwen35/loader.cpp - build `TrunkWeights` from a real Ornith/Qwen35MoE GGUF.
//
// The model is mmapped through the existing `strata::GgufModel`; every `Mat` points straight at the GGUF
// blocks, so nothing is copied or dequantized.  `row_bytes` comes from the reader's own `block_geometry`,
// which is the same (elements, bytes) ggml's `ggml_row_size` derives from - no second opinion about a block.
#include "strata/qwen35/qwen35.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <stdexcept>

namespace strata::qwen35 {
namespace {

Mat make_mat(const strata::GgufModel& m, const std::string& name, std::string& err) {
    size_t sh = 0;
    const strata::TensorInfo* t = m.find(name, &sh);
    if (!t) { err = "missing tensor " + name; return {}; }
    if (t->shape.size() != 2) { err = name + " is not 2-D"; return {}; }
    int be = 0, bb = 0;
    if (!strata::block_geometry(t->type, be, bb)) { err = name + " is " + t->type_name() + " (no kernel)"; return {}; }
    Mat r;
    r.data = m.shard(sh).tensor_data(*t);
    r.type = (int) t->type;
    r.n_in = (int64_t) t->shape[0];
    r.n_out = (int64_t) t->shape[1];
    r.row_bytes = (size_t) (r.n_in / be) * (size_t) bb;
    return r;
}

/// One expert's slice of a 3-D expert tensor [n_in, n_out, n_expert].
Mat make_expert(const strata::GgufModel& m, const std::string& name, int64_t expert, std::string& err) {
    size_t sh = 0;
    const strata::TensorInfo* t = m.find(name, &sh);
    if (!t) { err = "missing tensor " + name; return {}; }
    if (t->shape.size() != 3) { err = name + " is not 3-D"; return {}; }
    int be = 0, bb = 0;
    if (!strata::block_geometry(t->type, be, bb)) { err = name + " is " + t->type_name() + " (no kernel)"; return {}; }
    Mat r;
    r.type = (int) t->type;
    r.n_in = (int64_t) t->shape[0];
    r.n_out = (int64_t) t->shape[1];
    r.row_bytes = (size_t) (r.n_in / be) * (size_t) bb;
    r.data = (const uint8_t*) m.shard(sh).tensor_data(*t) +
             (size_t) expert * r.row_bytes * (size_t) r.n_out;
    return r;
}

}  // namespace

bool load_trunk(const std::string& path, Qwen35Geometry& g, TrunkWeights& w, std::string& err) {
    try {
        // The Mats point INTO this model's mmap, so it must outlive them: keep it in `w.backing`.
        auto holder = std::make_shared<strata::GgufModel>(std::vector<std::string>{path});
        strata::GgufModel& model = *holder;
        w.backing = holder;
        if (!core::check_qwen35_all(model, g, err)) return false;
        const int64_t L = g.n_layers, H = g.n_embd;

        w.token_embd = make_mat(model, "token_embd.weight", err);
        w.output = make_mat(model, "output.weight", err);
        if (!err.empty()) return false;
        {
            const strata::TensorInfo* t = model.find("output_norm.weight");
            size_t sh = 0;
            model.find("output_norm.weight", &sh);
            w.output_norm = t ? (const float*) model.shard(sh).tensor_data(*t) : nullptr;
            if (!w.output_norm) { err = "missing output_norm.weight"; return false; }
        }

        w.attn_norm.assign((size_t) L, nullptr);
        w.post_attn_norm.assign((size_t) L, nullptr);
        w.gdn.assign((size_t) L, {});
        w.attn.assign((size_t) L, {});
        w.moe.assign((size_t) L, {});
        w.expert_store.assign((size_t) L, {});

        for (int64_t l = 0; l < L; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            auto f32 = [&](const char* suffix) -> const float* {
                size_t sh = 0;
                const strata::TensorInfo* t = model.find(p + suffix, &sh);
                return t ? (const float*) model.shard(sh).tensor_data(*t) : nullptr;
            };
            w.attn_norm[(size_t) l] = f32("attn_norm.weight");
            w.post_attn_norm[(size_t) l] = f32("post_attention_norm.weight");
            if (!w.attn_norm[(size_t) l] || !w.post_attn_norm[(size_t) l]) { err = "layer " + std::to_string(l) + ": missing norm"; return false; }

            if (g.is_recurrent(l)) {
                GdnLayerWeights& d = w.gdn[(size_t) l];
                d.attn_norm = w.attn_norm[(size_t) l];
                d.wqkv = make_mat(model, p + "attn_qkv.weight", err);
                d.wgate = make_mat(model, p + "attn_gate.weight", err);
                d.ssm_beta = make_mat(model, p + "ssm_beta.weight", err);
                d.ssm_alpha = make_mat(model, p + "ssm_alpha.weight", err);
                d.ssm_out = make_mat(model, p + "ssm_out.weight", err);
                d.ssm_conv = f32("ssm_conv1d.weight");
                d.ssm_dt = f32("ssm_dt.bias");
                d.ssm_a = f32("ssm_a");
                d.ssm_norm = f32("ssm_norm.weight");
                if (!err.empty()) return false;
                if (!d.ssm_conv || !d.ssm_dt || !d.ssm_a || !d.ssm_norm) { err = "layer " + std::to_string(l) + ": missing GDN F32 tensor"; return false; }
            } else {
                AttnLayerWeights& d = w.attn[(size_t) l];
                d.attn_norm = w.attn_norm[(size_t) l];
                d.wq = make_mat(model, p + "attn_q.weight", err);
                d.wk = make_mat(model, p + "attn_k.weight", err);
                d.wv = make_mat(model, p + "attn_v.weight", err);
                d.wo = make_mat(model, p + "attn_output.weight", err);
                d.q_norm = f32("attn_q_norm.weight");
                d.k_norm = f32("attn_k_norm.weight");
                if (!err.empty()) return false;
                if (!d.q_norm || !d.k_norm) { err = "layer " + std::to_string(l) + ": missing attention norm"; return false; }
            }

            MoeLayerWeights& m = w.moe[(size_t) l];
            m.gate_inp = f32("ffn_gate_inp.weight");
            m.gate_inp_shexp = f32("ffn_gate_inp_shexp.weight");
            m.gate_shexp = make_mat(model, p + "ffn_gate_shexp.weight", err);
            m.up_shexp = make_mat(model, p + "ffn_up_shexp.weight", err);
            m.down_shexp = make_mat(model, p + "ffn_down_shexp.weight", err);
            if (!err.empty()) return false;
            if (!m.gate_inp || !m.gate_inp_shexp) { err = "layer " + std::to_string(l) + ": missing router"; return false; }

            std::vector<ExpertWeights>& ex = w.expert_store[(size_t) l];
            ex.resize((size_t) g.n_expert);
            for (int64_t e = 0; e < g.n_expert; ++e) {
                ex[(size_t) e].gate = make_expert(model, p + "ffn_gate_exps.weight", e, err);
                ex[(size_t) e].up = make_expert(model, p + "ffn_up_exps.weight", e, err);
                ex[(size_t) e].down = make_expert(model, p + "ffn_down_exps.weight", e, err);
                if (!err.empty()) return false;
            }
            m.experts = ex.data();
        }
        (void) H;
        return true;
    } catch (const std::exception& e) {
        err = std::string("loading ") + path + ": " + e.what();
        return false;
    }
}

}  // namespace strata::qwen35
