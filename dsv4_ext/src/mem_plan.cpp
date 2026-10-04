#include "dsv4/mem_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace dsv4 {

MemPlan plan_expert_memory(const MemPlanInput& in) {
    MemPlan p;
    const int L = (int) in.bytes_per_expert.size();
    if (L == 0 || in.n_expert <= 0) { p.err = "no layers / n_expert"; return p; }
    p.slots_layer.assign((size_t) L, 0);
    uint64_t sumb = 0;
    int active = 0;
    for (int l = 0; l < L; ++l) {
        if (in.bytes_per_expert[(size_t) l] == 0) continue;
        sumb += in.bytes_per_expert[(size_t) l];
        ++active;
        p.all_expert_bytes += (uint64_t) in.n_expert * in.bytes_per_expert[(size_t) l];
    }
    if (active == 0) { p.err = "no layer has routed experts"; return p; }

    p.libera = in.vram_free - in.reserve;
    p.fixed = in.weights + in.kv + in.act + in.prefill + in.mtp + in.scratch;
    if (p.libera <= 0 || p.fixed > p.libera) {
        std::ostringstream e;
        e << "fixed costs (" << (p.fixed >> 20) << " MiB) exceed the usable VRAM (" << (p.libera >> 20)
          << " MiB after the reserve): lower --max-context, use a KV int8 cache, fewer MTP/checkpoint slots, or a smaller reserve";
        p.err = e.str();
        return p;
    }
    p.budget_expert = p.libera - p.fixed;

    std::vector<int> active_idx;
    for (int l = 0; l < L; ++l) if (in.bytes_per_expert[(size_t) l] != 0) active_idx.push_back(l);

    if (in.abs_slots >= 0) {  // --expert-cache N wins over --expert-vram-pct
        int64_t S = std::min<int64_t>(in.abs_slots, (int64_t) active * in.n_expert);
        const int64_t q = S / active, r = S % active;
        for (int k = 0; k < active; ++k) {
            int64_t s = q + (k >= active - (int) r ? 1 : 0);  // the remainder goes to the LAST layers, one each
            p.slots_layer[(size_t) active_idx[(size_t) k]] = (int) std::min<int64_t>(s, in.n_expert);
        }
        uint64_t bytes = 0;
        for (int l = 0; l < L; ++l) bytes += (uint64_t) p.slots_layer[(size_t) l] * in.bytes_per_expert[(size_t) l];
        if ((int64_t) bytes > p.budget_expert) {
            std::ostringstream e;
            e << "--expert-cache " << in.abs_slots << " needs " << (bytes >> 20) << " MiB but only " << (p.budget_expert >> 20)
              << " MiB are available for experts";
            p.err = e.str();
            return p;
        }
        p.pct_used = p.budget_expert > 0 ? 100.0 * (double) bytes / (double) p.budget_expert : 0.0;
        p.expert_budget = (int64_t) bytes;
    } else {
        double pct = in.pct;
        if (pct < 0) pct = ((int64_t) p.all_expert_bytes <= p.budget_expert) ? 100.0 : 90.0;  // auto
        if (pct > 100.0) { p.err = "--expert-vram-pct must be in 0..100 (or auto)"; return p; }
        p.pct_used = pct;
        p.expert_budget = (int64_t) (pct / 100.0 * (double) p.budget_expert);
        uint64_t B = (uint64_t) p.expert_budget;
        uint64_t q = std::min<uint64_t>((uint64_t) in.n_expert, B / sumb);
        for (int l : active_idx) p.slots_layer[(size_t) l] = (int) q;
        uint64_t used = 0;
        for (int l = 0; l < L; ++l) used += (uint64_t) p.slots_layer[(size_t) l] * in.bytes_per_expert[(size_t) l];
        uint64_t left = B - used;
        bool grew = true;
        while (grew) {  // spend the remainder one slot at a time, last layers first
            grew = false;
            for (int k = active - 1; k >= 0; --k) {
                const int l = active_idx[(size_t) k];
                const uint64_t b = in.bytes_per_expert[(size_t) l];
                if (p.slots_layer[(size_t) l] < in.n_expert && left >= b) {
                    ++p.slots_layer[(size_t) l];
                    left -= b;
                    grew = true;
                }
            }
        }
    }

    p.slots_total = 0;
    p.slot_bytes_total = 0;
    p.all_resident = true;
    for (int l = 0; l < L; ++l) {
        if (in.bytes_per_expert[(size_t) l] == 0) continue;
        p.slots_total += p.slots_layer[(size_t) l];
        p.slot_bytes_total += (uint64_t) p.slots_layer[(size_t) l] * in.bytes_per_expert[(size_t) l];
        if (p.slots_layer[(size_t) l] != in.n_expert) p.all_resident = false;
    }
    p.ok = true;
    return p;
}

std::string plan_report(const MemPlanInput& in, const MemPlan& p) {
    auto mib = [](int64_t b) { return (long long) (b >> 20); };
    char buf[2048];
    int lo = 1 << 30, hi = 0;
    for (size_t l = 0; l < p.slots_layer.size(); ++l) {
        if (in.bytes_per_expert[l] == 0) continue;
        lo = std::min(lo, p.slots_layer[l]);
        hi = std::max(hi, p.slots_layer[l]);
    }
    std::snprintf(buf, sizeof buf,
                  "VRAM total:      %8lld MiB\n"
                  "VRAM free:       %8lld MiB\n"
                  "reserve:         %8lld MiB\n"
                  "fixed weights:   %8lld MiB\n"
                  "KV estimate:     %8lld MiB\n"
                  "activation:      %8lld MiB\n"
                  "prefill buffers: %8lld MiB\n"
                  "MTP/checkpoint:  %8lld MiB\n"
                  "scratch:         %8lld MiB\n"
                  "expert budget:   %8lld MiB (%.1f%% of the %lld MiB left after fixed costs)\n"
                  "expert slots:    %8lld of %lld (%s)\n"
                  "slots/layer:     %d..%d\n"
                  "slot bytes:      %8lld MiB\n"
                  "RAM expert copy: %8lld MiB\n",
                  mib(in.vram_total), mib(in.vram_free), mib(in.reserve), mib(in.weights), mib(in.kv), mib(in.act),
                  mib(in.prefill), mib(in.mtp), mib(in.scratch), mib(p.expert_budget), p.pct_used, mib(p.budget_expert),
                  (long long) p.slots_total, (long long) (p.all_expert_bytes ? (int64_t) in.n_expert * std::count_if(in.bytes_per_expert.begin(), in.bytes_per_expert.end(), [](uint64_t b) { return b != 0; }) : 0),
                  p.all_resident ? "ALL RESIDENT: fast path" : "tiered: hit/miss", lo == (1 << 30) ? 0 : lo, hi,
                  mib((int64_t) p.slot_bytes_total), mib((int64_t) in.ram_expert_bytes));
    return buf;
}

}  // namespace dsv4
