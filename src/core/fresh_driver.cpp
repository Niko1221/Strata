#include "strata/core/fresh_driver.hpp"
#include "strata/core/expert_source.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace strata::core {
int fresh_driver(const WeightTable& weights, const ModelGeometry& geometry, SessionState& state,
                 const VerifyHits& hits, const NativeHead* head, PoolMultiFn pool, void* user,
                 ExpertDispatch& dispatch, int pcie_mode, const std::vector<int64_t>& tokens, const std::string& logits_path) {
    std::string error;
    auto fail = [&](const char* message) {
        std::fprintf(stderr, "strata fresh: %s: %s\n", message, error.c_str());
        return 1;
    };
    if (tokens.empty() || tokens.size() > (size_t)state.max_cells || logits_path.empty())
        return fail("missing full history/output path or exceeded context");
    Verifier forward;
    if (!forward.init(weights, geometry, state, hits, head, 2, error)) return fail("initialize");
    // The verifier consumes the GPU plan produced by this exact dispatch.
    // Without this binding, resident experts are omitted from both CPU and GPU work.
    forward.set_pcie_mode(pcie_mode);
    dispatch.plan = forward.plan_sink();
    struct PlanReset {
        ExpertDispatch& dispatch;
        ~PlanReset() { dispatch.plan = nullptr; }
    } plan_reset{dispatch};
    bool& pool_failed = dispatch.failed;
    forward.set_head_sampling(false);
    for (auto token : tokens)
        if (token < 0 || token >= forward.vocab()) return fail("invalid token ID");
    int64_t consumed = 0;
    for (auto token : tokens) {
        const int32_t input = (int32_t)token;
        int32_t unused_sample = 0;
        pool_failed = false;
        if (!forward.run(1, &input, consumed, pool, user, &unused_sample, error) || pool_failed ||
            !forward.commit(1, error)) return fail("full-history forward/commit");
        ++consumed;
    }
    if (forward.completed_forward_tokens != tokens.size()) return fail("incomplete measured work");
    std::vector<float> logits((size_t)forward.vocab());
    if (!forward.copy_logits(0, logits.data())) return fail("frontier copy");
    for (float x : logits) if (!std::isfinite(x)) return fail("nonfinite frontier");
    std::ofstream out(logits_path, std::ios::binary | std::ios::trunc);
    out.write((const char*)logits.data(), (std::streamsize)logits.size()*sizeof(float));
    out.close();
    if (!out) return fail("frontier write");
    std::cout << "{\"status\":\"ok\",\"consumed\":" << consumed
              << ",\"position_max\":" << consumed-1
              << ",\"forward_tokens\":" << forward.completed_forward_tokens
              << ",\"windows\":" << forward.windows
              << ",\"resident_expert_entries\":" << dispatch.cache_hits
              << ",\"cpu_expert_entries\":" << dispatch.cache_refused
              << ",\"pcie_expert_groups\":" << dispatch.pcie_experts
              << ",\"logits_bytes\":" << logits.size()*sizeof(float) << "}" << std::endl;
    return 0;
}
} // namespace strata::core
