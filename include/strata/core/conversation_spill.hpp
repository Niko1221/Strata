// A durable disk tier for --serve's parked conversations: when the RAM cache evicts a conversation, it is written
// as an ordinary session file (conversation_file.hpp, the same format and model/config identity as the slot save/
// restore API) and can be restored after a restart. A small sidecar beside each file holds only the token/image
// metadata, so a new request finds the best disk match without reading the conversation's K/V.
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

struct ConversationSpillMatch {
    std::string path;        // the session file to read back
    uint64_t file_bytes = 0;
    int64_t tokens = 0;
    bool live = false;
    explicit operator bool() const { return !path.empty() && tokens > 0; }
};

class ConversationSpillCache {
public:
    // Opens (creating if needed) the spill directory and indexes the conversations already in it. Files whose
    // sidecar names another model/config identity, or whose session file is missing or unreadable, are removed.
    bool open(const std::filesystem::path& directory, SessionFileIdentity identity, uint64_t budget_bytes,
              std::string& error);
    bool enabled() const { return enabled_; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    size_t stale_files_wiped() const { return stale_files_wiped_; }
    size_t disk_evictions() const { return disk_evictions_; }

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
                best_tokens = match.tokens, best_match = {entry.session_path(), entry.file_bytes, match.tokens, match.live};
        }
        return best_match;
    }

    // Reads a spilled conversation back into image (its K/V included), with the engine's read limits and identity.
    bool load(const std::string& path, SavedConversation& image, const SessionReadLimits& limits,
              std::string& error) const;
    // Writes a parked conversation (its K/V in RAM) as a session file plus its sidecar, and indexes it.
    bool spill(const SavedConversation& image, std::string& error);
    // --conversation-cache-disk-only: the same file and sidecar, written straight from the live session - `meta`
    // carries everything but the K/V (meta.kv empty), `kv` streams each layer from its pool into the file - so no
    // host image of the conversation is ever held.  `bytes` is the file's size.  durable=false skips the flushes: a
    // file torn by a crash fails the payload hash when it is read and is discarded, so a cache need not pay for them.
    // prefix=true writes a PREFIX entry (strata-conv-prefix-N): a conversation's root - its system prompt and tools -
    // that any new conversation opening the same way resumes from.  Prefix entries persist: drop_superseded never
    // removes them and the disk budget evicts every ordinary conversation before one of them.
    bool spill_streamed(const SavedConversation& meta, const std::vector<SessionKvSource>& kv, size_t& bytes,
                        std::string& error, bool durable = false, bool prefix = false);
    // A prefix entry holding exactly these tokens (and images) is on disk already.
    bool has_prefix(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images, bool cvec) const;
    size_t prefix_count() const;
    bool erase(const std::string& path, std::string& error);
    // A conversation being restored is held out of eviction until it is put back in RAM or dropped.
    void pin(const std::string& path);
    void unpin(const std::string& path);
    // Removes the disk copies of this conversation a turn back, the same rule the RAM cache applies before evicting.
    // `keep` (a session path) is never dropped: a disk-only save writes the new copy first and then drops the old ones.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                           const std::string& keep = {});
    // The session path of the newest file (the one the last spill wrote), or empty.
    std::string newest_path() const { return entries_.empty() ? std::string() : entries_.back().session_path(); }

private:
    struct Entry {
        std::filesystem::path stem;          // "<dir>/strata-conv-<serial>" (both files share it)
        uint64_t file_bytes = 0;
        bool cvec = true;
        ConversationCheckpoint live_meta;    // ids + imgs only; no running state
        std::vector<size_t> checkpoint_lengths;
        bool prefix = false;                 // a persistent root entry (named strata-conv-prefix-N)
        std::string session_path() const { return (stem.string() + ".sess"); }
        std::string meta_path() const { return (stem.string() + ".meta"); }
    };
    void enforce_budget();
    std::filesystem::path next_stem(bool prefix = false);
    bool index_written(const std::filesystem::path& stem, uint64_t file_bytes, const SavedConversation& meta,
                       std::string& error, bool prefix = false);
    bool read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity, std::string& error) const;
    bool write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error) const;

    bool enabled_ = false;
    SessionFileIdentity identity_{};
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0;
    size_t stale_files_wiped_ = 0, disk_evictions_ = 0;
    std::filesystem::path directory_;
    std::vector<Entry> entries_;   // oldest spill first
    std::string pinned_path_;
};

} // namespace strata::core
