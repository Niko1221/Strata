#pragma once
// Diagnostic-only readback. No restore path or canonical state representation.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/step_projection.hpp"
#include <fstream>
#include <vector>

namespace strata::core {
inline bool dump_first_consumed_state(const char* prefix, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    if (!prefix) return true;
    if (ss.ple_prev[0] != -1 || ss.ple_prev[1] < 0 || cudaDeviceSynchronize() != cudaSuccess) {
        error = "step state: not a completed one-token boundary"; return false;
    }
    ConversationStateSizes sizes;
    if (!conversation_session_sizes(g, ss, sizes, error)) return false;
    std::ofstream manifest(std::string(prefix) + ".json", std::ios::trunc);
    manifest << "{\"consumed\":1,\"ple_prev\":[" << ss.ple_prev[0] << "," << ss.ple_prev[1] << "],\"parts\":[";
    bool first = true;
    auto save = [&](const std::string& name, const void* data, size_t bytes, const char* dtype) {
        if (!data || !bytes) { error = "step state: missing populated buffer"; return false; }
        std::vector<char> host(bytes);
        if (cudaMemcpy(host.data(), data, bytes, cudaMemcpyDefault) != cudaSuccess) {
            error = "step state: readback failed"; return false;
        }
        std::ofstream out(std::string(prefix) + "." + name, std::ios::binary | std::ios::trunc);
        out.write(host.data(), (std::streamsize) bytes); out.close();
        if (!out) { error = "step state: payload write failed"; return false; }
        if (!first) manifest << ",";
        first = false;
        manifest << "{\"name\":\"" << name << "\",\"bytes\":" << bytes << ",\"dtype\":\"" << dtype << "\"}";
        return (bool) manifest;
    };
    const size_t recurrence = (size_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
    const size_t convolution = (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    for (int64_t j = 0; j < ss.gdn_alloc; ++j) {
        const float* row = ss.gdn_state + (size_t) j * (recurrence + convolution);
        const std::string name = "gdn" + std::to_string(ss.gdn_ord0 + j);
        if (!save(name + ".recurrence", row, recurrence * 4, "f32") ||
            !save(name + ".convolution", row + recurrence, convolution * 4, "f32")) return false;
    }
    if (ss.ple_hist && !save("ple", ss.ple_hist, sizes.ple, "f32")) return false;
    const int64_t page_size = strata::kernels::qsa_real_shapes().page_size;
    for (int64_t j = ss.qsa_ord0; j < ss.qsa_ord0 + ss.qsa_alloc; ++j) {
        const auto& q = ss.qsa_states[j];
        if (q.kv_mode || q.kv_int8 || q.kv_q4 || q.kv_hybrid) {
            error = "step state: diagnostic supports resident FP16 KV only"; return false;
        }
        int32_t page = -1;
        if (!q.page_table || cudaMemcpy(&page, q.page_table, sizeof(page), cudaMemcpyDefault) != cudaSuccess ||
            page < 0 || page >= q.n_slots) { error = "step state: missing first physical page"; return false; }
        const std::string name = "qsa" + std::to_string(j);
        for (int64_t head = 0; head < g.n_head_kv; ++head) {
            const size_t offset = ((size_t) page * g.n_head_kv + head) * page_size * g.head_dim;
            if (!save(name + ".k" + std::to_string(head), q.k_pool + offset, (size_t) g.head_dim * 2, "f16") ||
                !save(name + ".v" + std::to_string(head), q.v_pool + offset, (size_t) g.head_dim * 2, "f16")) return false;
        }
        // Only the first token's tail row is populated. The spare pooled row is live state.
        if (!save(name + ".tail", q.idx_tail, (size_t) g.idx_key_dim * 4, "f32") ||
            !save(name + ".dead", q.idx_dead, sizes.dead, "f32") ||
            !save(name + ".pooled", q.idx_pooled, (size_t) g.idx_key_dim * 4, "f32") ||
            !save(name + ".block_pos", q.idx_block_pos, sizeof(int32_t), "i32")) return false;
    }
    if (step_projection &&
        (!save("projection.mixed", step_projection->mixed, step_projection->input_count * 4, "f32") ||
         !save("projection.qkv", step_projection->qkv, step_projection->output_count * 4, "f32"))) return false;
    manifest << "]}"; manifest.close();
    if (!manifest) { error = "step state: manifest write failed"; return false; }
    return true;
}
} // namespace strata::core
