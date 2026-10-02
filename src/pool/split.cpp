// src/pool/split.cpp - the pool's placement search (see split.hpp for the cost model).
#include "strata/pool/split.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::pool {

SplitPlan evaluate_split(const SplitModel& m, const std::vector<NodeCap>& nodes, const std::vector<int64_t>& at) {
    SplitPlan p;
    p.at = at;
    const int ns = (int) nodes.size();
    if (ns < 1 || (int) at.size() != ns - 1) { p.why = "the split needs one point per worker"; return p; }
    auto lb_of = [&](int i) { return i == 0 ? (int64_t) 0 : at[(size_t) i - 1]; };
    auto le_of = [&](int i) { return i + 1 < ns ? at[(size_t) i] : m.n_layers; };
    for (int i = 0; i < ns; ++i) {
        const int64_t lb = lb_of(i), le = le_of(i);
        if (le <= lb || (i == 0 && le < m.first_min) || le > m.n_layers) {
            p.why = "node " + nodes[(size_t) i].name + " would get no layers (" + std::to_string(lb) + "-" +
                    std::to_string(le - 1) + ")";
            return p;
        }
    }
    p.node_slots.assign((size_t) ns, 0);
    p.node_arena.assign((size_t) ns, 0);
    std::vector<int64_t> room((size_t) ns), used((size_t) ns, 0);
    for (int i = 0; i < ns; ++i) {
        const int64_t lb = lb_of(i), le = le_of(i);
        int64_t arena = 0;
        for (int64_t l = lb; l < le; ++l)
            arena += (size_t) l < m.layer_arena_bytes.size() ? m.layer_arena_bytes[(size_t) l] : 0;
        p.node_arena[(size_t) i] = arena;
        const NodeCap& n = nodes[(size_t) i];
        if (n.ram_room > 0 && arena > n.ram_room) {
            char b[256];
            std::snprintf(b, sizeof b, "%s's RAM cannot hold the experts of layers %lld-%lld (%.1f GiB, %.1f GiB free)",
                          n.name.c_str(), (long long) lb, (long long) (le - 1), (double) arena / 1073741824.0,
                          (double) n.ram_room / 1073741824.0);
            p.why = b;
            return p;
        }
        room[(size_t) i] = n.vram_room - (m.session_bytes ? m.session_bytes(lb, le) : 0);
    }
    double total = 0, held_mass = 0;
    std::vector<bool> full((size_t) ns, false);
    for (size_t r = 0; r < m.profile.size(); ++r) {
        const double w = std::pow((double) r + 1.0, -1.2);
        total += w;
        const int64_t l = m.profile[r].first;
        int st = 0;
        while (st + 1 < ns && l >= at[(size_t) st]) ++st;
        if (full[(size_t) st]) continue;
        const int64_t c = m.slot_bytes ? m.slot_bytes(l) : 1;
        if (used[(size_t) st] + c > room[(size_t) st]) { full[(size_t) st] = true; continue; }
        used[(size_t) st] += c;
        ++p.node_slots[(size_t) st];
        ++p.held;
        held_mass += w;
    }
    p.mass = total > 0 ? held_mass / total : 0.0;
    p.ms = m.miss_ms * (1.0 - p.mass) + m.hop_ms * (double) (ns - 1) * 2.0;
    for (int i = 0; i < ns; ++i) p.ms += (double) (le_of(i) - lb_of(i)) * nodes[(size_t) i].layer_ms;
    p.ok = true;
    return p;
}

SplitPlan auto_split(const SplitModel& m, const std::vector<NodeCap>& nodes) {
    const int ns = (int) nodes.size();
    SplitPlan best;
    best.why = "no placement fits";
    if (ns < 1) return best;
    if (ns == 1) return evaluate_split(m, nodes, {});
    const int64_t L = m.n_layers;
    std::string last_why;
    auto consider = [&](const std::vector<int64_t>& at) {
        SplitPlan p = evaluate_split(m, nodes, at);
        if (!p.ok) { last_why = p.why; return; }
        // ties (within 0.1%): the placement whose busiest node holds the fewest layers above its share
        if (!best.ok || p.ms < best.ms * 0.999) best = p;
    };
    std::vector<int64_t> at((size_t) ns - 1);
    if (ns == 2) {
        for (int64_t k = m.first_min; k < L; ++k) { at[0] = k; consider(at); }
    } else if (ns == 3) {
        for (int64_t k1 = m.first_min; k1 + 1 < L; ++k1)
            for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(at); }
    } else {
        double tot = 0;
        for (const NodeCap& n : nodes) tot += 1.0 / std::max(n.layer_ms, 1e-3);
        double acc = 0;
        for (int i = 0; i + 1 < ns; ++i) {
            acc += 1.0 / std::max(nodes[(size_t) i].layer_ms, 1e-3);
            at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / tot * (double) L),
                                                 i == 0 ? m.first_min : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
        }
        consider(at);
        // nudge each point either way while it improves (and so that RAM fits)
        for (int pass = 0; pass < 4; ++pass)
            for (int i = 0; i + 1 < ns; ++i)
                for (const int d : {-1, 1}) {
                    std::vector<int64_t> t = best.ok ? best.at : at;
                    t[(size_t) i] += d;
                    consider(t);
                }
    }
    if (!best.ok) best.why = last_why.empty() ? "no placement fits" : last_why;
    return best;
}

bool parse_split(const std::string& s, std::vector<int64_t>& at) {
    at.clear();
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        const std::string t = s.substr(a, b - a);
        if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
        at.push_back(std::atoll(t.c_str()));
        a = b + 1;
    }
    for (size_t i = 1; i < at.size(); ++i)
        if (at[i] <= at[i - 1]) return false;
    return !at.empty();
}

}  // namespace strata::pool
