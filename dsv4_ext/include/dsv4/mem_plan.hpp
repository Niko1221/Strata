// dsv4/mem_plan.hpp - how many experts live in VRAM.  Pure arithmetic (no CUDA), so it is testable anywhere.
//
//   free          = vram_free - reserve
//   budget_expert = max(0, free - fixed)            fixed = weights + kv + act + prefill + mtp + scratch
//   expert_budget = P/100 * budget_expert           P is a share of what is LEFT, not of physical VRAM
//
// Slots are handed out per layer (never a global counter: that concentrates them in the first layers).
// Layers may have different bytes-per-expert (dynamic quants), so the plan works on real per-layer sizes.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

struct MemPlanInput {
    int64_t vram_total = 0, vram_free = 0;
    int64_t reserve = 1024LL << 20;                      // --expert-vram-reserve-mib (default 1024)
    int64_t weights = 0, kv = 0, act = 0, prefill = 0, mtp = 0, scratch = 0;  // fixed costs, bytes
    int n_expert = 0;                                    // experts per layer
    std::vector<uint64_t> bytes_per_expert;              // per layer (0 = layer has no routed experts)
    double pct = -1.0;                                   // 0..100; < 0 = auto (all if they fit, else 90)
    int64_t abs_slots = -1;                              // --expert-cache N; >= 0 overrides pct
    uint64_t ram_expert_bytes = 0;                       // host copy of all experts (report only)
};

struct MemPlan {
    bool ok = false;
    std::string err;
    int64_t libera = 0, fixed = 0, budget_expert = 0, expert_budget = 0;
    double pct_used = 0;
    std::vector<int> slots_layer;      // per layer, <= n_expert
    int64_t slots_total = 0;
    uint64_t slot_bytes_total = 0;     // sum of slots_layer[l] * bytes_per_expert[l]
    bool all_resident = false;         // every layer holds all of its experts
    uint64_t all_expert_bytes = 0;     // sum over layers of n_expert * bytes_per_expert
};

MemPlan plan_expert_memory(const MemPlanInput& in);
std::string plan_report(const MemPlanInput& in, const MemPlan& p);

}  // namespace dsv4
