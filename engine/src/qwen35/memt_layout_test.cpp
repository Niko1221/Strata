// src/qwen35/memt_layout_test.cpp - the MemT layout table (header-only, no GPU).
//
// The six blk.<i>.memt.* tensors must be required on every full-attention
// layer when memt.version is present, and absent otherwise.  The shapes are
// GGUF order (dim 0 contiguous), matching the Qwen3.6 MemT loader.
#include "strata/qwen35/memt_model.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_fail = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_fail;
    }
}

const strata::core::TensorWant* find(const std::vector<strata::core::TensorWant>& t,
                                     const std::string& name) {
    for (const auto& item : t)
        if (item.name == name) return &item;
    return nullptr;
}

bool has_shape(const std::vector<strata::core::TensorWant>& t, const std::string& name,
               std::vector<int64_t> shape) {
    const strata::core::TensorWant* item = find(t, name);
    return item != nullptr && item->shape == shape;
}

strata::core::ModelGeometry nano() {
    strata::core::ModelGeometry g;
    g.family = strata::ModelFamily::Qwen35Moe;
    g.n_embd = 2048;
    g.n_layers = 40;
    g.qsa_interval = 4;
    g.n_head = 16;
    g.n_head_kv = 2;
    g.head_dim = 256;
    g.ssm_state_size = 128;
    g.ssm_k_heads = 16;
    g.ssm_v_heads = 32;
    g.ssm_d_conv = 4;
    g.ssm_value_dim = 128 * 32;
    g.ssm_conv_channels = 2 * 128 * 16 + 128 * 32;
    g.n_expert = 256;
    g.n_expert_used = 8;
    g.n_ff = 512;
    g.n_ff_shexp = 512;
    g.hc = 0;
    g.hc_lr = 0;
    g.idx_q_heads = 0;
    g.idx_key_dim = 0;
    return g;
}

}  // namespace

int main() {
    strata::core::ModelGeometry g = nano();
    const strata::qwen35::MemtConfig config{16, 256, 256, 2.0, 1e-6, {}};

    const int64_t HD = 16 * 256;
    const auto tensors = strata::qwen35::memt_tensors(g, config);
    check(has_shape(tensors, "blk.3.memt.in_proj.weight", {g.n_embd, 5 * HD + 2 * 256}),
          "in_proj shape");
    check(has_shape(tensors, "blk.3.memt.f_down.weight", {256, HD}), "f_down shape");
    check(has_shape(tensors, "blk.3.memt.g_down.weight", {256, HD}), "g_down shape");
    check(has_shape(tensors, "blk.3.memt.decay_scale_log", {16}), "decay_scale_log shape");
    check(has_shape(tensors, "blk.3.memt.decay_bias", {HD}), "decay_bias shape");
    check(has_shape(tensors, "blk.3.memt.o_norm.weight", {256}), "o_norm shape");

    for (int64_t layer : {7, 11, 39}) {
        const std::string name = "blk." + std::to_string(layer) + ".memt.in_proj.weight";
        check(find(tensors, name) != nullptr,
              "memory on full-attention layer " + std::to_string(layer));
    }
    for (int64_t layer : {0, 1, 2, 4, 38}) {
        const std::string name = "blk." + std::to_string(layer) + ".memt.in_proj.weight";
        check(find(tensors, name) == nullptr,
              "no memory on GDN layer " + std::to_string(layer));
    }

    strata::core::ModelGeometry stock = nano();
    const auto plain = strata::core::layer_tensors(stock, 3);
    check(find(plain, "memt.in_proj.weight") == nullptr,
          "a stock model has no memt tensors");

    // memory is on the trunk's full-attention layers only, never the MTP block
    g.n_mtp_layers = 1;
    const auto with_mtp = strata::qwen35::memt_tensors(g, config);
    check(find(with_mtp, "blk." + std::to_string(g.n_layers) + ".memt.in_proj.weight") == nullptr,
          "the MTP block carries no memt tensors");

    if (g_fail == 0) std::printf("memt_layout_test: %d checks passed\n", g_checks);
    return g_fail == 0 ? 0 : 1;
}
