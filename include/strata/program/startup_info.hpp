// include/strata/program/startup_info.hpp - the conversation-cache fields of the engine's startup INFO line.
//
// The INFO line printf in generate.cpp ends its fixed head at spec_min_p and continues with the prompt-path
// fields; everything between them - the conversation-cache thresholds, the disk-tier fields when the spill
// directory is serving, and tail_role_token/vram_elastic - is rendered here, as one string the printf emits
// through a single %s.  That keeps ONE argument list per printf: a format string and its arguments can never
// drift apart in field count (C varargs have no names, so an extra argument in the middle of a list silently
// shifts every field after it - which is exactly what a READ=0/READ>0 ternary over one shared list did).
//
// A READ of 0 renders the bytes a build without --conversation-cache-min-tokens' READ dimension printed; a READ
// above 0 adds conversation_cache_min_read_tokens right after conversation_cache_min_tokens. The disk-tier
// fields appear only when the tier is on. tail_role_token and vram_elastic always follow the optional cache fields.
#pragma once

#include <cinttypes>
#include <cstdio>
#include <string>

namespace strata::program {

struct ConversationCacheInfoFields {
    int64_t mib = 0;              // conversation_cache_mib: the host-RAM budget (MiB)
    int slots = 0;                // conversation_cache_slots
    int64_t min_free_mib = 0;     // conversation_cache_min_free_mib
    int64_t min_tokens = 0;       // conversation_cache_min_tokens: the WRITE threshold
    int64_t min_read_tokens = 0;  // the READ threshold; 0 = off, and then the field is not printed at all
    bool disk_tier_on = false;    // the spill directory is serving: the disk-tier fields belong in the line
    int64_t disk_mib = 0;         // conversation_cache_disk_mib
    bool disk_only = false;       // conversation_cache_disk_only
    bool disk_delta = false;      // conversation_cache_disk_delta
    double similarity = 0.0;      // conversation_cache_similarity
    int64_t n_min = 0;            // conversation_cache_n_min
    int64_t tail_role_token = -1; // tail_role_token
    bool vram_elastic = false;    // vram_elastic
};

/// The fields between spec_min_p and prefill_cap of the startup INFO line, space-separated, without a leading
/// or trailing space.  Field order and rendering (decimal integers, the similarity at three decimals) match
/// the printf that printed them before this helper existed.
inline std::string conversation_cache_info_fields(const ConversationCacheInfoFields& f) {
    std::string s = "conversation_cache_mib=" + std::to_string(f.mib) +
                    " conversation_cache_slots=" + std::to_string(f.slots) +
                    " conversation_cache_min_free_mib=" + std::to_string(f.min_free_mib) +
                    " conversation_cache_min_tokens=" + std::to_string(f.min_tokens);
    if (f.min_read_tokens > 0)
        s += " conversation_cache_min_read_tokens=" + std::to_string(f.min_read_tokens);
    if (f.disk_tier_on) {
        char similarity[32];
        std::snprintf(similarity, sizeof similarity, "%.3f", f.similarity);
        s += " conversation_cache_disk_mib=" + std::to_string(f.disk_mib) +
             " conversation_cache_disk_only=" + std::to_string(f.disk_only ? 1 : 0) +
             " conversation_cache_disk_delta=" + std::to_string(f.disk_delta ? 1 : 0) +
             " conversation_cache_similarity=" + similarity +
             " conversation_cache_n_min=" + std::to_string(f.n_min);
    }
    // These fields follow the optional cache thresholds and disk fields in every configuration.
    s += " tail_role_token=" + std::to_string(f.tail_role_token) +
         " vram_elastic=" + std::to_string(f.vram_elastic ? 1 : 0);
    return s;
}

}  // namespace strata::program
