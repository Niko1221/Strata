// strata-inkling CUDA: visit- and speed-aware automatic layer split (#7030).
//
// Host-only, no GPU dependency (unit test: engine/tests/core/ink_layer_split_test.cpp).
//
// Inkling-2-Mini runs its 24 physical layers over 48 visits per token: prelude L0..5 (1x),
// looped core L6..17 (3x = 36 visits), coda L18..23 (1x). Splitting strictly by free VRAM
// puts part of the looped core on a slower peer GPU and adds cross-GPU hops on every pass.
// This planner evaluates all valid contiguous partitions across 2..4 stages, minimizing
// (host_overflow_bytes, estimated_decode_cost). Stage 0's KV/ShortConv reserve is sized for
// the visits Stage 0 owns in each candidate partition (matching InkCudaRunner::init).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace strata::core {

inline constexpr int kInkLayerSplitMaxStages = 4;

struct InkLayerSplitKvBudget {
    size_t swa_kv_per_visit = 0;              // SWA KV bytes per SWA visit owned by stage 0
    size_t glob_kv_per_visit = 0;             // full-context global KV bytes per global visit owned by stage 0
    size_t sconv_per_visit = 0;               // ShortConv state bytes per visit owned by stage 0
    size_t ws_reserve_bytes = 0;              // workspace + drafter + vision reserve on stage 0
};

struct InkLayerSplitIn {
    int n_phys = 24;
    int core_start = 6;
    int core_end = 18;
    int num_passes = 3;
    std::vector<int> is_swa;                  // size n_phys: 1 = SWA, 0 = global
    std::vector<size_t> layer_exp_bytes;      // size n_phys: expert bytes per physical layer
    std::vector<size_t> layer_trunk_bytes;    // size n_phys: packed trunk bytes per physical layer

    int n_stages = 2;                         // 2..kInkLayerSplitMaxStages
    std::vector<double> stage_speed;          // size n_stages: positive relative throughput (e.g. BW * arch_mult)
    bool stage_p2p[kInkLayerSplitMaxStages][kInkLayerSplitMaxStages] = {};

    size_t stage0_free_bytes = 0;             // free VRAM on stage 0 at partition time (after trunk upload)
    size_t stage0_base_reserve_bytes = 0;     // baseline card reserve floor
    size_t stage0_wddm_floor_bytes = 0;       // WDDM / OS safety floor on stage 0
    size_t stage0_extra_reserve_bytes = 0;    // min_reserve_bytes from plan_vram_reserve (+ vision/override)
    size_t stage0_ws_reserve_bytes = 0;       // workspace + drafter + vision reserve for full-KV budget
    size_t stage0_swa_kv_per_visit = 0;       // SWA KV bytes per SWA visit owned by stage 0
    size_t stage0_glob_kv_per_visit = 0;      // full-context global KV bytes per global visit owned by stage 0
    size_t stage0_sconv_per_visit = 0;        // ShortConv state bytes per visit owned by stage 0

    std::vector<size_t> peer_room_bytes;      // size n_stages (index 0 unused): usable VRAM on peer stage s >= 1
    size_t per_layer_peer_overhead = 48ull << 20;
    size_t user_expert_cap_bytes = std::numeric_limits<size_t>::max();
};

struct InkLayerSplitOut {
    std::vector<int> counts;                  // layers assigned to each stage (sum == n_phys)
    std::vector<int> visits;                  // visits per token executed on each stage
    int hops = 0;                             // cross-GPU handoffs per token along the visit schedule
    size_t stage0_budget_bytes = 0;           // stage 0 expert budget for the chosen partition
    size_t overflow_bytes = 0;                // stage 0 expert bytes overflowing to host RAM
    double cost = 0.0;                        // decode cost objective of the chosen partition
};

inline double ink_stage_rel_speed(int mem_clk_khz, int bus_bits, int sm_cc) noexcept {
    const double bw = (mem_clk_khz > 0 && bus_bits > 0)
        ? (2.0 * static_cast<double>(mem_clk_khz) * 1e-6 * (static_cast<double>(bus_bits) / 8.0))
        : 300.0;
    const double arch_mult = (sm_cc >= 120) ? 1.15 : ((sm_cc >= 89) ? 1.05 : 1.0);
    return std::max(1.0, bw * arch_mult);
}

inline InkLayerSplitOut ink_plan_layer_split(const InkLayerSplitIn& in) {
    InkLayerSplitOut out;
    const int N = in.n_phys;
    const int S = std::max(1, std::min(in.n_stages, kInkLayerSplitMaxStages));
    out.counts.assign(S, 0);
    out.visits.assign(S, 0);
    if (N <= 0) return out;
    out.counts[0] = N;

    const int passes = std::max(1, in.num_passes);
    std::vector<int> layer_visits(N, 1);
    for (int il = std::max(0, in.core_start); il < std::min(in.core_end, N); ++il) {
        layer_visits[il] = passes;
    }

    std::vector<int> sched;
    sched.reserve(static_cast<size_t>(N) * passes);
    for (int il = 0; il < std::min(in.core_start, N); ++il) sched.push_back(il);
    for (int p = 0; p < passes; ++p) {
        for (int il = std::max(0, in.core_start); il < std::min(in.core_end, N); ++il) sched.push_back(il);
    }
    for (int il = std::max(0, in.core_end); il < N; ++il) sched.push_back(il);

    std::vector<double> spd(S, 1.0);
    double max_spd = 1.0;
    for (int s = 0; s < S; ++s) {
        const double v = (s < static_cast<int>(in.stage_speed.size()) && in.stage_speed[s] > 0.0)
            ? in.stage_speed[s] : 1.0;
        spd[s] = v;
        if (v > max_spd) max_spd = v;
    }
    for (int s = 0; s < S; ++s) spd[s] /= max_spd;

    std::vector<size_t> pref_exp(N + 1, 0), pref_trk(N + 1, 0);
    std::vector<int> pref_swa_v(N + 1, 0), pref_glob_v(N + 1, 0);
    for (int il = 0; il < N; ++il) {
        pref_exp[il + 1] = pref_exp[il] + (il < static_cast<int>(in.layer_exp_bytes.size()) ? in.layer_exp_bytes[il] : 0ull);
        pref_trk[il + 1] = pref_trk[il] + (il < static_cast<int>(in.layer_trunk_bytes.size()) ? in.layer_trunk_bytes[il] : 0ull);
        const bool swa = (il < static_cast<int>(in.is_swa.size())) ? (in.is_swa[il] != 0) : true;
        pref_swa_v[il + 1]  = pref_swa_v[il]  + (swa ? layer_visits[il] : 0);
        pref_glob_v[il + 1] = pref_glob_v[il] + (swa ? 0 : layer_visits[il]);
    }

    auto stage0_budget_for = [&](int n0) -> size_t {
        const size_t n_swa  = static_cast<size_t>(pref_swa_v[n0]);
        const size_t n_glob = static_cast<size_t>(pref_glob_v[n0]);
        const size_t kv_g0  = n_swa * in.stage0_swa_kv_per_visit
                            + n_glob * in.stage0_glob_kv_per_visit
                            + (n_swa + n_glob) * in.stage0_sconv_per_visit;
        const size_t kv_res0 = std::max(in.stage0_extra_reserve_bytes, kv_g0 + in.stage0_ws_reserve_bytes);
        const size_t need_res0 = std::max(in.stage0_base_reserve_bytes, kv_res0 + in.stage0_wddm_floor_bytes);
        const size_t avail = (in.stage0_free_bytes > need_res0) ? (in.stage0_free_bytes - need_res0) : 0ull;
        return std::min(in.user_expert_cap_bytes, avail);
    };

    auto peer_fits_range = [&](int s, int l0, int l1) -> bool {
        if (l1 <= l0) return true;
        const size_t room = (s < static_cast<int>(in.peer_room_bytes.size())) ? in.peer_room_bytes[s] : 0ull;
        const size_t exp = pref_exp[l1] - pref_exp[l0];
        const size_t trk = pref_trk[l1] - pref_trk[l0];
        const size_t cost = exp + trk + static_cast<size_t>(l1 - l0) * in.per_layer_peer_overhead;
        return cost <= room && exp <= in.user_expert_cap_bytes;
    };

    const size_t bud_all = stage0_budget_for(N);
    out.stage0_budget_bytes = bud_all;
    out.overflow_bytes = (pref_exp[N] > bud_all) ? (pref_exp[N] - bud_all) : 0ull;
    out.cost = std::numeric_limits<double>::infinity();
    out.visits[0] = pref_swa_v[N] + pref_glob_v[N];

    if (S == 1) {
        out.cost = static_cast<double>(out.visits[0]) / spd[0];
        return out;
    }

    int bnd[kInkLayerSplitMaxStages + 1] = {};
    bnd[0] = 0;
    bnd[S] = N;
    std::vector<int> owner(N, 0);

    auto eval_partition = [&]() {
        int s_vis[kInkLayerSplitMaxStages] = {};
        int active_peers = 0;
        int peer_layers = 0;
        int stage_index_weight = 0;
        for (int s = 0; s < S; ++s) {
            const int cnt = bnd[s + 1] - bnd[s];
            for (int il = bnd[s]; il < bnd[s + 1]; ++il) {
                owner[il] = s;
                s_vis[s] += layer_visits[il];
            }
            if (s > 0 && cnt > 0) {
                ++active_peers;
                peer_layers += cnt;
                stage_index_weight += s * cnt;
            }
        }
        const size_t bud0 = stage0_budget_for(bnd[1]);
        const size_t exp0 = pref_exp[bnd[1]];
        const size_t ovf = (exp0 > bud0) ? (exp0 - bud0) : 0ull;
        double cost = 0.0;
        for (int s = 0; s < S; ++s) {
            cost += static_cast<double>(s_vis[s]) / spd[s];
        }
        int hops = 0;
        for (size_t i = 1; i < sched.size(); ++i) {
            const int sa = owner[sched[i - 1]];
            const int sb = owner[sched[i]];
            if (sa != sb) {
                ++hops;
                cost += in.stage_p2p[sa][sb] ? 0.15 : 0.60;
            }
        }
        // Tie-breaker: prefer fewer active peers and keeping layers on earlier stages when equal.
        cost += 1e-4 * active_peers + 1e-5 * peer_layers + 1e-6 * stage_index_weight;
        if (ovf < out.overflow_bytes || (ovf == out.overflow_bytes && cost < out.cost)) {
            out.overflow_bytes = ovf;
            out.stage0_budget_bytes = bud0;
            out.cost = cost;
            out.hops = hops;
            for (int s = 0; s < S; ++s) {
                out.counts[s] = bnd[s + 1] - bnd[s];
                out.visits[s] = s_vis[s];
            }
        }
    };

    std::function<void(int, int)> search_splits = [&](int s, int l0) {
        if (s == S - 1) {
            if (peer_fits_range(s, l0, N)) {
                bnd[s] = l0;
                eval_partition();
            }
            return;
        }
        const int min_end = (s == 0) ? 1 : l0;
        for (int l1 = min_end; l1 <= N; ++l1) {
            if (s > 0 && !peer_fits_range(s, l0, l1)) break;
            bnd[s + 1] = l1;
            search_splits(s + 1, l1);
        }
    };
    search_splits(0, 0);
    return out;
}

}  // namespace strata::core
