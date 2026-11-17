// include/strata/prefill/split_relay_rule.hpp - which staged expert copies of a layer split's prompt go through the
// other card (prefill.cpp SplitRelay, STRATA_PREFILL_SPLIT_RELAY).
//
// A one-chunk prompt on a layer split runs the stages one after the other, so while one stage reads, the other card's
// PCIe link is idle.  Below stream_all_min() a chunk takes the staged walk (only the routed experts are copied, one by
// one) and the layer-split prompt help does not apply, so that walk is bound by one card's link.  With the relay on, a
// share of the staged copies goes host -> the other card (its own link) -> this card (NVLink or another peer path):
// the same bytes into the same ring slot, so the results are bit-identical to the direct copies.
// Pure functions: no GPU, no environment read here (the caller passes getenv's answer).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace strata::prefill {

/// STRATA_PREFILL_SPLIT_RELAY: on for a number other than 0; unset or "0": off.
inline bool split_relay_on(const char* env) { return env != nullptr && std::atoi(env) != 0; }

/// STRATA_PREFILL_SPLIT_RELAY_FRAC: the share of the staged bytes that goes through the other card, clamped to [0, 1];
/// unset or not a number: 0.5 (two links of the same width).
inline double split_relay_frac(const char* env) {
    if (env == nullptr || env[0] == '\0') return 0.5;
    char* end = nullptr;
    const double v = std::strtod(env, &end);
    if (end == env) return 0.5;
    return std::clamp(v, 0.0, 1.0);
}

/// Does this chunk's staged walk use the relay?  `armed`: the stage has a relay (peer access works, its buffers exist);
/// `single_chunk`: the prompt is one chunk (the other stage idles while this one reads); `helped`: the layer-split
/// prompt help or a --peer-device peer owns this run's other card; `streamed`: the streamed walk (not the staged one).
inline bool split_relay_chunk(bool armed, bool single_chunk, bool helped, bool streamed) {
    return armed && single_chunk && !helped && !streamed;
}

/// Byte-balancing: a copy goes through the relay while that keeps the relayed bytes at or below `frac` of all bytes
/// so far.  With equal blobs and frac 0.5 the copies alternate, the first one direct.
struct RelayShare {
    double frac = 0.5;
    uint64_t total = 0, relayed = 0;
    bool take(uint64_t bytes) {
        total += bytes;
        if ((double) (relayed + bytes) <= frac * (double) total) {
            relayed += bytes;
            return true;
        }
        return false;
    }
};

}  // namespace strata::prefill
