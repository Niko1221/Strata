// include/strata/platform/numa.hpp - NUMA placement of a device's staging (STRATA_NUMA_GPU_STAGING).
//
// Pure decisions, no OS calls and no globals, so a test can feed them a made-up machine - the same split as
// kernels/cpu/numa_pool.hpp: that header places the CPU expert pool's memory, this one places a GPU's staging
// ring and its copy threads.  The caller queries the OS once per device (how many NUMA nodes exist, which one
// the device hangs off) and hands both answers in; the decision is the node the staging should live on, or -1
// for the legacy "wherever the allocator puts it" path.
//
// STRATA_NUMA_GPU_STAGING is unset by default and the unset/off path is byte-identical to before the switch
// existed.
//   auto     each device's staging lives on the node the device is attached to (the multi-socket case the
//            allocator has no idea about: WDDM put a whole ring on the wrong socket's node)
//   <N>      force every device's staging onto node N, 0 included (debug/verification arm)
//   off word (off false no, any case) or anything not understood -> legacy path
// ("0" is a real node number here, not an off word: on a multi-node box there is no other way to name the
//  first node.  The pool's numa_switch_off still treats "0" as off - it has no nodes to name.)
//
// On a machine with a single NUMA node there is nothing to choose: auto degrades to the legacy path, and an
// OS that cannot name the device's node (old driver, VM without NUMA exposure, non-PCI device) does the same.
// Nothing here is ever worse than the pre-switch behaviour: every failure ends at -1.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::platform {

/// The off-word spelling of the NUMA switches (0/off/false/no, any case) - same convention as the pool's
/// numa_switch_off (kernels/cpu/numa_pool.hpp); kept separate so this header stands alone.
inline bool numa_word_off(const char* v) {
    if (v == nullptr) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "0" || s == "off" || s == "false" || s == "no";
}

struct NumaGpuPick {
    int node = -1;      ///< the NUMA node this device's ring and copy threads should live on; -1 = legacy path
    std::string log;    ///< a one-line account of the decision to print ("" = say nothing)
};

/// `env` is STRATA_NUMA_GPU_STAGING's value (nullptr or "" = unset).  `device_node` is the node the device is
/// attached to (-1 = the OS could not say).  `node_count`
/// is the machine's NUMA nodes with CPUs (<=1 = nothing to choose between).
inline NumaGpuPick decide_numa_gpu(const char* env, int device_node, int node_count) {
    NumaGpuPick p;
    if (env == nullptr || *env == '\0') return p;
    std::string s(env);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    // digits first: "0" names node 0, it is not an off word on this switch
    const bool digits = std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
    if (digits) {
        if (node_count > 0 && std::atoi(s.c_str()) >= node_count) {
            p.log = "numa-gpu: node " + s + " does not exist (" + std::to_string(node_count) +
                    " nodes) - staging stays as before";
            return p;
        }
        p.node = std::atoi(s.c_str());
        p.log = "numa-gpu: staging forced to node " + s;
        return p;
    }
    if (s == "off" || s == "false" || s == "no") return p;
    if (s == "auto") {
        if (node_count <= 1) {
            p.log = "numa-gpu: auto, but the machine has a single NUMA node - staging stays where the allocator puts it";
            return p;
        }
        if (device_node < 0) {
            p.log = "numa-gpu: auto, but the OS does not name this device's NUMA node - staging stays as before";
            return p;
        }
        p.node = device_node;
        p.log = "numa-gpu: staging on the device's node " + std::to_string(device_node) + " (auto)";
        return p;
    }
    p.log = "numa-gpu: STRATA_NUMA_GPU_STAGING=" + s + " not understood (auto|<node>|off) - staging stays as before";
    return p;
}

struct NumaArenaSplit {
    std::vector<int64_t> bounds;   ///< the first layer of each domain after domain 0 (domain 0 starts at layer 0)
    std::string log;               ///< a one-line account of the decision to print ("" = say nothing)
};

/// `env` is STRATA_NUMA_ARENA_SPLIT's value: a comma list of the layer at which each later arena domain
/// begins - the same numbering `--layer-split` uses, so domain r is the layers pipeline stage r (CUDA
/// device r) streams.  The caller turns bounds into byte boundaries through the expert layout and into
/// nodes through device_numa_node(domain index).  Empty bounds = the legacy single-domain path: unset or
/// an off word, a machine with one node (nothing to choose), a token that is not a layer number, or bounds
/// not strictly inside (0, n_layers) and increasing.
inline NumaArenaSplit decide_numa_arena_split(const char* env, int64_t n_layers, int node_count) {
    NumaArenaSplit p;
    if (env == nullptr || *env == '\0' || numa_word_off(env)) return p;
    if (node_count <= 1) {
        p.log = "numa-arena: split asked, but the machine has a single NUMA node - one domain as before";
        return p;
    }
    std::string s(env);
    size_t at = 0;
    while (at <= s.size()) {
        const size_t comma = s.find(',', at);
        const std::string tok = s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        const bool digits = !tok.empty() && std::all_of(tok.begin(), tok.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
        const int64_t b = digits ? std::atoll(tok.c_str()) : -1;
        if (b <= 0 || b >= n_layers || (!p.bounds.empty() && b <= p.bounds.back())) {
            p.bounds.clear();
            p.log = "numa-arena: STRATA_NUMA_ARENA_SPLIT=" + s + " refused (" + tok +
                    " is not a layer boundary inside (0," + std::to_string(n_layers) + ") or not increasing) - one domain as before";
            return p;
        }
        p.bounds.push_back(b);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    p.log = "numa-arena: one domain per stage, boundaries at layers " + s;
    return p;
}

}  // namespace strata::platform
