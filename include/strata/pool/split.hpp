// include/strata/pool/split.hpp - where the pool cuts the model between its PCs.
//
// The same cost model as the multi-GPU "auto" split (generate.cpp, measured on the 5080 + 3090 rig): a decode
// window costs every layer its GPU's per-layer time, plus ~190 ms per unit of routed mass that no GPU cache holds
// (the CPU pool computes those experts).  A placement gives each node a contiguous range; that node's cache takes
// its range's profiled pairs, hottest first, until its free VRAM (less the session of its range) is used.
//
// What the pool adds: each node's RAM must hold the experts of its own range (the arena is per range here, which is
// what lets two 32-64 GB PCs run a model neither could alone), and every stage boundary costs a network hop.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::pool {

struct NodeCap {
    std::string name;
    int64_t vram_room = 0;   ///< bytes free for the session of its range + its expert cache
    int64_t ram_room = 0;    ///< bytes its RAM can give the expert arena of its range (<= 0: no limit)
    double layer_ms = 0.5;   ///< GPU time per layer per window (from SMs x clock)
};

struct SplitModel {
    int64_t n_layers = 48;
    std::vector<std::pair<int32_t, int32_t>> profile;     ///< (layer, expert), hottest first
    std::function<int64_t(int64_t layer)> slot_bytes;     ///< VRAM per cached expert of a layer
    std::vector<int64_t> layer_arena_bytes;               ///< RAM for all experts of a layer
    std::function<int64_t(int64_t lb, int64_t le)> session_bytes;
    double miss_ms = 190.0;   ///< one unit of routed mass on the CPU pool, per window
    double hop_ms = 1.5;      ///< one network crossing per window (constant per extra node)
    int64_t first_min = 2;    ///< the coordinator keeps at least layers 0 and 1 (the PLE runs at layer 1)
};

struct SplitPlan {
    bool ok = false;
    std::string why;                 ///< when !ok: what does not fit
    std::vector<int64_t> at;         ///< the first layer of each later node (size = nodes - 1)
    double ms = 0;                   ///< predicted ms per decode window
    double mass = 0;                 ///< the routed mass the caches hold (0..1)
    int64_t held = 0;                ///< profiled pairs cached across the pool
    std::vector<int64_t> node_slots; ///< predicted cached experts per node
    std::vector<int64_t> node_arena; ///< RAM bytes of each node's range
};

/// The best placement over `nodes` (coordinator first).  Every placement for 2 or 3 nodes; beyond that, layers in
/// proportion to speed, then nudged to fit RAM.
SplitPlan auto_split(const SplitModel& m, const std::vector<NodeCap>& nodes);

/// The prediction for a given placement (`at` as above); `ok` false with `why` when a node's RAM cannot hold its
/// range or a range is empty.
SplitPlan evaluate_split(const SplitModel& m, const std::vector<NodeCap>& nodes, const std::vector<int64_t>& at);

/// "18" / "16,32" -> the split points; false on anything else.
bool parse_split(const std::string& s, std::vector<int64_t>& at);

}  // namespace strata::pool
