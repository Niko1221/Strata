// The system-prompt prefill cache (--system-prompt-cache): the checkpoint root that ends the system prompt, which
// today lives only in RAM (--prompt-cache-root), persisted to disk as an ORDINARY SESSION FILE (conversation_file.cpp
// - the same format, model/config identity and slot save/restore rules; NOT a second format) and reloaded at start.
//
// The artifact holds the running state and the K/V of EVERYTHING UP TO AND INCLUDING the end of the system prompt -
// never a conversation tail. It is a resume point for a NEW chat whose system prompt is byte-for-byte the stored one:
// a prompt that shares it reads only the tokens after it.
//
//   CAUSALITY (read this before changing the load path). A transformer's attention is causal: the K/V of a token is a
//   function of that token and everything BEFORE it, and of the system prompt that was in front of it. A stored tail
//   K/V therefore means nothing when it is placed behind a DIFFERENT system prompt - it would attend to a prefix the
//   model never saw it attend to and silently corrupt every later token. So a variant is keyed by the EXACT hash of
//   its system-prompt token prefix (plus the optional --system-prompt-cache-key and the model/config identity), and a
//   loaded artifact is only ever attached as the root of a prompt that begins with exactly those tokens. If the
//   system prompt changes, its hash changes: the old variant is not reused, the request is reprocessed from the start
//   (only the system prompt is re-read), and the new prefix is written as its own variant. A variant is never removed
//   because the system prompt changed - only the GC (space or age, oldest first) removes one.
//
// Pure host code: it calls the session-file writer/reader, which are host code too. Inert unless open() succeeds.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::core {

// A stored variant the caller may load (from the sidecar alone: no K/V read).
struct ConversationPromptMatch {
    std::string path;        // the session file to read back
    uint64_t file_bytes = 0;
    int64_t tokens = 0;      // the system-prompt prefix length the variant ends at
    explicit operator bool() const { return !path.empty() && tokens > 0; }
};

class ConversationPromptCache {
public:
    // Opens (creating if needed) the cache directory and indexes the variants already in it. As with the spill
    // directory the SCAN NEVER DELETES: a foreign-identity file, an unreadable sidecar, a variant whose session file
    // is gone and an orphan session file are all ignored and counted. Removal happens only in the GC: the variant
    // count/byte budget (--system-prompt-cache-slots / -mib) and the optional age lever (oldest first, with
    // counters). `slots` <= 0 means no variant cap (only the byte budget); `max_age_days` <= 0 means no age pruning.
    bool open(const std::filesystem::path& directory, SessionFileIdentity identity, uint64_t budget_bytes,
              int64_t slots, int64_t max_age_days, std::string& error);
    bool enabled() const { return enabled_; }

    // The key of a system-prompt token prefix: its token bytes + the optional declared key + this model/config
    // identity, each length-delimited so two different field lists never collide (SessionIdentityBuilder).
    uint64_t key_for(const int64_t* tokens, size_t count, const std::string& declared) const;

    // A variant stored under this key, from its sidecar only. No K/V is read; the caller validates the prefix.
    bool lookup(uint64_t key, ConversationPromptMatch& out) const;

    // Counters the caller drives: a hit when a stored variant is reused (with the tokens it saved), a miss when the
    // incoming prefix is not stored (a system-prompt change, or the first chat).
    void note_hit(uint64_t key, int64_t tokens);
    void note_miss(uint64_t key);

    // Writes a variant. The two forms write the same file; `store_streamed` avoids a host K/V copy when the engine
    // can stream the K/V (SessionKvSource), `store` takes an already captured image (tests, and the no-draft path).
    bool store_streamed(uint64_t key, const SavedConversation& meta,
                        const std::vector<SessionKvSource>& sources, std::string& error);
    bool store(uint64_t key, const SavedConversation& image, std::string& error);
    // Reads a variant back with the engine's read limits and identity (the session-file reader).
    bool load(const std::string& path, SavedConversation& image,
              const std::vector<SessionReadLimits>& limits, std::string& error) const;
    bool erase(uint64_t key, std::string& error);

    // Metrics.
    size_t variants() const { return entries_.size(); }        // live variants
    uint64_t bytes() const { return bytes_; }                  // bytes on disk
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t tokens_saved() const { return tokens_saved_; }
    uint64_t hash_changes() const { return hash_changes_; }    // new variants written beside an existing one
    size_t evicted_by_space() const { return evicted_by_space_; }
    size_t evicted_by_age() const { return evicted_by_age_; }
    size_t foreign_files_kept() const { return foreign_files_kept_; }
    size_t stale_files_kept() const { return stale_files_kept_; }
    size_t orphan_files_kept() const { return orphan_files_kept_; }
    size_t oversized_files_kept() const { return oversized_files_kept_; }

private:
    struct Entry {
        uint64_t key = 0;
        int64_t tokens = 0;
        uint64_t file_bytes = 0;
        std::filesystem::path session;
        std::filesystem::path meta;
        std::filesystem::file_time_type stamp{};
        uint64_t hits = 0;
    };
    Entry* find(uint64_t key);
    const Entry* find(uint64_t key) const;
    void enforce();
    bool remove_entry(size_t index);
    std::filesystem::path variant_file(uint64_t key) const;
    std::filesystem::path meta_file(uint64_t key) const;
    bool read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity, std::string& error) const;
    bool write_sidecar(const Entry& entry, std::string& error) const;
    bool index_written(uint64_t key, int64_t tokens, uint64_t file_bytes, std::string& error);

    bool enabled_ = false;
    SessionFileIdentity identity_{};
    uint64_t budget_ = 0, bytes_ = 0;
    int64_t slots_ = 2, max_age_days_ = 0;
    uint64_t hits_ = 0, misses_ = 0, tokens_saved_ = 0, hash_changes_ = 0;
    size_t evicted_by_space_ = 0, evicted_by_age_ = 0, foreign_files_kept_ = 0, stale_files_kept_ = 0;
    size_t orphan_files_kept_ = 0, oversized_files_kept_ = 0;
    std::filesystem::path directory_;
    std::vector<Entry> entries_;   // oldest variant first
};

} // namespace strata::core
