// L2 RAM PARKED control-plane adapter.
// This file holds all cache-control command parsing and formatting. Upstream
// generate.cpp only gets one call site; no KV ownership moves out of the engine.
#pragma once
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/conversation_cache.hpp"

namespace strata::core::l2control {

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

inline long long number_after(const std::string& s, const char* key) {
    const std::string k = std::string(key) + "=";
    const size_t at = s.find(k);
    if (at == std::string::npos) return -1;
    return std::strtoll(s.c_str() + at + k.size(), nullptr, 10);
}

inline std::string safe_source(std::string s) {
    for (char& c : s)
        if (!(std::isalnum((unsigned char)c) || c == '/' || c == '.' || c == '_' || c == '-'))
            c = '_';
    return s;
}

inline bool cache_command(const std::string& line, ConversationCache& conversations) {
    if (line.rfind("CACHE ", 0) != 0) return false;
    const std::string rest = trim(line.substr(6));
    if (rest == "STATS") {
        const auto st = conversations.l2_stats();
        std::printf("CACHESTAT enabled=%d budget_bytes=%zu slots=%zu entries=%zu bytes=%zu "
                    "evictions=%zu retained_bytes=%zu pinned=%zu guarded=%zu\n",
                    st.enabled ? 1 : 0, st.budget_bytes, st.slots, st.entries, st.bytes,
                    st.evictions, st.retained_bytes, st.pinned, st.guarded);
        std::fflush(stdout);
        return true;
    }
    if (rest == "LIST") {
        const auto rows = conversations.l2_list();
        std::printf("CACHELIST %zu\n", rows.size());
        for (const auto& e : rows) {
            std::printf("CACHEENTRY serial=%llu tokens=%zu bytes=%zu ckpts=%zu pinned=%d guarded=%d "
                        "cvec=%d source=%s\n",
                        (unsigned long long)e.serial, e.tokens, e.bytes, e.checkpoints,
                        e.pinned ? 1 : 0, e.guarded ? 1 : 0, e.cvec ? 1 : 0,
                        safe_source(e.source).c_str());
        }
        std::printf("CACHEEND\n");
        std::fflush(stdout);
        return true;
    }
    long long serial = number_after(rest, "serial");
    if (serial < 0) {
        const size_t sp = rest.find(' ');
        if (sp != std::string::npos) serial = std::strtoll(rest.c_str() + sp + 1, nullptr, 10);
    }
    const bool pin = rest.rfind("PIN ", 0) == 0;
    const bool unpin = rest.rfind("UNPIN ", 0) == 0;
    const bool drop = rest.rfind("DROP ", 0) == 0;
    const bool touch = rest.rfind("TOUCH ", 0) == 0;
    bool ok = false;
    const char* op = "unknown";
    if (serial >= 0) {
        if (pin) { ok = conversations.l2_set_pinned((uint64_t)serial, true); op = "pin"; }
        else if (unpin) { ok = conversations.l2_set_pinned((uint64_t)serial, false); op = "unpin"; }
        else if (drop) { ok = conversations.l2_drop((uint64_t)serial); op = "drop"; }
        else if (touch) { ok = conversations.l2_touch((uint64_t)serial); op = "touch"; }
    }
    if (pin || unpin || drop || touch) {
        if (ok) std::printf("CACHEOK op=%s serial=%lld\n", op, serial);
        else std::printf("CACHEERR op=%s serial=%lld\n", op, serial);
    } else {
        std::printf("CACHEERR op=unknown\n");
    }
    std::fflush(stdout);
    return true;
}

inline std::string probe_extra(const ConversationCache::Match& best) {
    return " parked_best=" + std::to_string(best.tokens) +
           " parked_serial=" + std::to_string(best.serial) +
           " parked_live=" + (best.live ? "1" : "0");
}

inline uint64_t mark_source(ConversationCache& conversations, const std::string& path) {
    const uint64_t serial = conversations.l2_last_serial();
    conversations.l2_set_source(serial, path);
    return serial;
}

} // namespace strata::core::l2control
