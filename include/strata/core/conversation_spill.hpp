// A durable disk tier for --serve's parked conversations: when the RAM cache evicts a conversation, it is written
// as an ordinary session file (conversation_file.hpp, the same format and model/config identity as the slot save/
// restore API) and can be restored after a restart. A small sidecar beside each file holds only the token/image
// metadata, so a new request finds the best disk match without reading the conversation's K/V.
//
// A layer-split conversation (SavedConversation::stage_images) is written as ONE FILE PER STAGE plus one joint
// sidecar: <stem>.sess holds the first stage (the one the draft layer's K/V travels with when there is a draft),
// <stem>.stage<k>.sess the k-th later stage - the same carve each stage's own snapshot holds, so a stage file is a
// byte-for-byte session file of that stage. The sidecar records the stage count, so load() reads them all back.
//
// This is host code only (no CUDA): it calls the session-file writer and reader, which are host code too.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::core {

// What the disk tier does when its budget is reached (--conversation-cache-spill-when-full).
//   evict_oldest  the default, and what upstream always did: the GC drops the oldest conversation to make room.
//   reject        never evicts: a spill that would not fit the budget is refused and reported, and nothing is removed.
enum class SpillWhenFull { evict_oldest, reject };

struct ConversationSpillMatch {
    std::string path;        // the session file to read back
    uint64_t file_bytes = 0;
    int64_t tokens = 0;
    bool live = false;
    size_t stages = 0;       // extra stage files beyond the first (the sidecar's stage count)
    explicit operator bool() const { return !path.empty() && tokens > 0; }
};

class ConversationSpillCache {
public:
    // Opens (creating if needed) the spill directory and indexes the conversations already in it. The SCAN NEVER
    // DELETES: a file of another model/config identity, an unreadable sidecar, a session file whose sidecar is gone,
    // an orphan session file and a leftover temporary are all ignored and counted, never removed - so a variant or a
    // newer format beside this one is left alone. Removal happens only in the GC (enforce_budget and enforce_age):
    // budget plus age, oldest first, with counters. The GC also leaves a single conversation larger than the whole
    // budget alone (removing it could not bring the directory under the budget), and reports it.
    //   `when_full`  evict_oldest (default) = the GC drops the oldest to make room; reject = the GC removes nothing
    //                and a spill that would not fit is refused (nothing is ever removed for a full directory).
    //   `max_age_days`  0 = off: no deletion by time at all. A positive value prunes conversations older than that,
    //                oldest first, counted in age_evictions(). This age lever is independent of `when_full`.
    bool open(const std::filesystem::path& directory, SessionFileIdentity identity, uint64_t budget_bytes,
              std::string& error, SpillWhenFull when_full = SpillWhenFull::evict_oldest, int64_t max_age_days = 0);
    bool enabled() const { return enabled_; }
    SpillWhenFull when_full() const { return when_full_; }
    int64_t max_age_days() const { return max_age_days_; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    // What the last scan and the budget kept instead of wiping, by reason (reporting only; nothing is deleted here).
    size_t stale_files_kept() const { return stale_files_kept_; }        // unreadable/broken sidecar or a lost session file
    size_t foreign_files_kept() const { return foreign_files_kept_; }    // another model/config identity
    size_t orphan_files_kept() const { return orphan_files_kept_; }      // a session file with no sidecar
    size_t oversized_files_kept() const { return oversized_files_kept_; }// a conversation larger than the budget
    size_t disk_evictions() const { return disk_evictions_; }            // removed by the budget GC (oldest first)
    size_t age_evictions() const { return age_evictions_; }              // removed by --conversation-cache-spill-max-age-days

    // The best resume this directory offers for the prompt, from the sidecars only (no K/V read). similarity and
    // n_min filter weak hits exactly as the RAM cache's best() does.
    template<class Token>
    ConversationSpillMatch best(const std::vector<Token>& prompt,
                                const std::vector<ConversationImageKey>& images, bool cvec,
                                double similarity, int64_t n_min) const {
        ConversationSpillMatch best_match;
        int64_t best_tokens = 0;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& entry = entries_[i];
            const ConversationMatch match = conversation_metadata_match(entry.live_meta,
                    entry.checkpoint_lengths, prompt, images, cvec, entry.cvec, similarity, n_min);
            if (match.tokens > best_tokens)
                best_tokens = match.tokens,
                best_match = {entry.session_path(), entry.file_bytes, match.tokens, match.live, entry.stages};
        }
        return best_match;
    }

    // Reads a spilled conversation back into `image` (its K/V included), with the engine's read limits and identity.
    // limits[0] bounds the first stage's file, limits[k] stage k's; a missing entry falls back to the last one given,
    // and an empty list reads every file with open limits. The stage files named in the sidecar are read into
    // image.stage_images in stage order. A single-image conversation (the common case) uses limits[0] alone.
    bool load(const std::string& path, SavedConversation& image,
              const std::vector<SessionReadLimits>& limits, std::string& error) const;
    // Writes a parked conversation (its K/V in RAM) as one session file per stage plus one joint sidecar, and indexes
    // it. A conversation larger than the budget is written and indexed anyway; the GC decides about removal.
    bool spill(const SavedConversation& image, std::string& error);
    bool erase(const std::string& path, std::string& error);
    // A conversation being restored is held out of eviction until it is put back in RAM or dropped.
    void pin(const std::string& path);
    void unpin(const std::string& path);
    // Removes the disk copies of this conversation a turn back, the same rule the RAM cache applies before evicting.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec);
    // The stage file beside a conversation's first file: stage k (1-based) is "<stem>.stage<k>.sess".
    static std::string stage_path(const std::string& session_path, size_t stage);

private:
    struct Entry {
        std::filesystem::path stem;          // "<dir>/strata-conv-<serial>" (every file shares it)
        uint64_t file_bytes = 0;             // every stage file, summed
        size_t stages = 0;                   // stage files beyond the first
        bool cvec = true;
        ConversationCheckpoint live_meta;    // ids + imgs only; no running state
        std::vector<size_t> checkpoint_lengths;
        std::filesystem::file_time_type stamp{};   // last write, for the age lever and oldest-first ordering
        std::string session_path() const { return stem.string() + ".sess"; }
        std::string stage_path(size_t k) const { return ConversationSpillCache::stage_path(session_path(), k); }
        std::string meta_path() const { return stem.string() + ".meta"; }
    };
    const Entry* find(const std::string& path) const;
    void enforce_budget();
    void enforce_age();
    bool remove_entry(size_t index);
    bool read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity, std::string& error) const;
    bool write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error) const;

    bool enabled_ = false;
    SpillWhenFull when_full_ = SpillWhenFull::evict_oldest;
    int64_t max_age_days_ = 0;
    SessionFileIdentity identity_{};
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0;
    size_t stale_files_kept_ = 0, foreign_files_kept_ = 0, orphan_files_kept_ = 0, oversized_files_kept_ = 0;
    size_t disk_evictions_ = 0, age_evictions_ = 0;
    std::filesystem::path directory_;
    std::vector<Entry> entries_;   // oldest spill first
    std::string pinned_path_;
};

} // namespace strata::core
