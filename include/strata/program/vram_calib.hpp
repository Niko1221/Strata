// include/strata/program/vram_calib.hpp - STRATA_VRAM_CALIB (#5571): the host-side arithmetic of the measured VRAM reserve.
//
// generate.cpp measures, per card, what is allocated after the expert cache was sized (`need`: the free VRAM when the cache
// was sized - the free VRAM with everything loaded - the cache) and what is lost later (`grow`: the further drop to the lowest
// free figure after a request, the window graphs captured on first use). The next start of the same binary with the same
// arguments and STRATA_* variables keeps need + grow + a margin free instead of --vram-reserve-mib / --vram-reserve-later-mib
// and the fixed late-allocation terms. This header holds the parts that touch no GPU, so tests/program/vram_calib_test.cpp
// can pin them: the file key, the file format, the reserve and the growth update.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace strata::program::vram_calib {

constexpr long long kDefaultMarginMib = 192;   ///< STRATA_VRAM_CALIB_MARGIN_MIB; 192 kept two 16 GB cards stall-free (#5571)

/// STRATA_VRAM_CALIB_MARGIN_MIB: a number of MiB, or a tier - "tight" (96: the most cache, for a config shown not to stall),
/// "normal" (192, the default) or "safe" (384: a card shared with a desktop or other programs, or WDDM, where a full card
/// pages instead of failing, #2192). Unset, empty or not understood: the default.
inline long long margin_mib(const char* v) {
    if (v == nullptr || *v == '\0') return kDefaultMarginMib;
    if (std::strcmp(v, "tight") == 0) return 96;
    if (std::strcmp(v, "normal") == 0) return kDefaultMarginMib;
    if (std::strcmp(v, "safe") == 0) return 384;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    return end != v && *end == '\0' && n >= 0 ? n : kDefaultMarginMib;
}

struct Card {
    int dev = -1;
    long long need = -1;   ///< bytes allocated on the card after its cache was sized (-1: not measured)
    long long grow = 0;    ///< bytes lost after the first requests (window graphs captured on first use)
};

/// Switches that only measure or report (timers, traces, node lists, dispatch counts, empty probe nodes): a run that differs
/// from another only in them is the same configuration for the reserve, so an A/B arm with a trace keeps the calibration.
inline bool measure_only(const char* kv) {
    static const char* const names[] = {"STRATA_TRACE=", "STRATA_DECODE_TIMING=", "STRATA_PREFILL_TIMING=", "STRATA_DISPATCH_REPORT=",
                                        "STRATA_VERIFY_NODES=", "STRATA_DUMMY_NODES="};
    for (const char* n : names)
        if (std::strncmp(kv, n, std::strlen(n)) == 0) return true;
    return false;
}

/// The STRATA_* variables that select the engine's behaviour, sorted. The calibration's own switches are left out (turning the
/// margin up must not throw the measurement away), and so are the measurement-only ones (measure_only).
inline std::vector<std::string> sizing_env(char** envp) {
    std::vector<std::string> env;
    for (char** e = envp; e != nullptr && *e != nullptr; ++e)
        if (std::strncmp(*e, "STRATA_", 7) == 0 && std::strncmp(*e, "STRATA_VRAM_CALIB", 17) != 0 && !measure_only(*e))
            env.emplace_back(*e);
    std::sort(env.begin(), env.end());
    return env;
}

/// 16 hex digits naming the calibration file: FNV-1a over the binary's identity, every argument and the sizing variables.
inline std::string file_key(const std::string& exe_id, const std::vector<std::string>& args, const std::vector<std::string>& env) {
    std::string key = exe_id + ";";
    for (const auto& a : args) { key += a; key += '\x1f'; }
    for (const auto& e : env) { key += e; key += '\x1e'; }
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : key) { h ^= c; h *= 1099511628211ull; }
    char hex[24];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) h);
    return hex;
}

/// The file of the last measurement made with this binary in any configuration ("binary-<16 hex>"): the need after the cache
/// and the growth hardly depend on the configuration, so a first start of a new configuration sizes from it instead of from
/// the fixed terms, and only a new binary's very first start does not.
inline std::string binary_key(const std::string& exe_id) {
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : exe_id) { h ^= c; h *= 1099511628211ull; }
    char hex[32];
    std::snprintf(hex, sizeof hex, "binary-%016llx", (unsigned long long) h);
    return hex;
}

/// One line per card: "<dev> <need bytes> <grow bytes>".
inline std::string format(const std::vector<Card>& cards) {
    std::string out;
    char line[96];
    for (const Card& c : cards) {
        std::snprintf(line, sizeof line, "%d %lld %lld\n", c.dev, c.need, c.grow);
        out += line;
    }
    return out;
}
inline std::vector<Card> parse(const std::string& text) {
    std::vector<Card> cards;
    std::istringstream in(text);
    Card c;
    while (in >> c.dev >> c.need >> c.grow) cards.push_back(c);
    return cards;
}

/// MiB to keep free on `dev` after its cache: need + grow rounded up to a MiB, plus the margin; -1 without a measurement.
inline long long reserve_mib(const std::vector<Card>& cards, int dev, long long margin_mib) {
    for (const Card& c : cards)
        if (c.dev == dev && c.need >= 0)
            return (c.need + std::max(0ll, c.grow) + (1ll << 20) - 1) / (1ll << 20) + std::max(0ll, margin_mib);
    return -1;
}

/// `need` from the free figure when the cache was sized, the free figure with everything loaded and the cache's size.
inline long long need_bytes(long long free_at_mark, long long free_now, long long cache_bytes) {
    if (free_at_mark < 0 || free_now < 0) return -1;
    return std::max(0ll, free_at_mark - free_now - cache_bytes);
}

/// After a request: the growth since everything was loaded, never smaller than what is already kept. True when it grew.
inline bool update_grow(long long& grow, long long seal_free, long long& low_free, long long free_now) {
    if (free_now < 0 || free_now >= low_free) return false;
    low_free = free_now;
    const long long g = seal_free - low_free;
    if (g <= grow) return false;
    grow = g;
    return true;
}

}  // namespace strata::program::vram_calib
