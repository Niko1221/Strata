// The conversation cache's disk tier (--conversation-cache-dir): parked conversations kept as files on a local drive
// (an NVMe SSD), so that RAM holds only the conversations in use while tens of them stay restorable - and survive an
// engine restart.  docs/CONVERSATION_CACHE_DISK.md has the design; in short:
//
//   * one conversation = one file per layer-split stage (session file format v1, conversation_file.hpp: aligned
//     16 MiB blocks, O_DIRECT where the filesystem takes it, payload hash, atomic rename) plus a small index file
//     (`<id>.idx`) written LAST: it commits the set.  Stage files without an index are a write that did not finish
//     and are removed when the tier opens.
//   * the files live in a subfolder named after the session identity (model and config fingerprints) and the stage
//     layout (each stage's layer range), so another model, another engine version or another --layer-split carve
//     never sees them.  Other subfolders are left alone and only reported.
//   * the index file carries what a request is matched by - every resume point's length, the hash of its tokens and
//     its pictures - so the tier keeps a few hundred bytes per conversation in RAM and matches a prompt without
//     reading a stage file.  A hash match is confirmed after the load (the loaded tokens are compared again by the
//     caller's ordinary validation), so a collision costs one wasted read, never a wrong state.
//   * least recently used conversations leave when the byte budget or the entry cap is reached.
//
// Pure host code (no CUDA), single-threaded: the engine calls it from its request loop only.  The hashes detect
// accidental corruption; they do not authenticate a file: keep the folder private to the engine's user.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The engine's stage layout: [layer_lo, layer_hi) of stage 0 (the main session), then of each later stage.  A
/// conversation's files are usable only by an engine with the same layout (each stage file holds one carve).
struct ConversationDiskLayout {
    std::vector<std::pair<int64_t, int64_t>> stages;
    uint64_t key() const;
};

struct ConversationDiskOptions {
    std::string dir;                 ///< --conversation-cache-dir (the tier works in a subfolder of it)
    uint64_t budget_bytes = 0;       ///< --conversation-cache-disk-gib: the most bytes of files kept
    size_t max_entries = 256;        ///< the most conversations kept, whatever their size
    uint64_t min_free_bytes = 0;     ///< free-space preflight of every stage file (--session-min-free-mib)
    bool durable = true;             ///< flush each file and the folder (crash safety); off only in tests
};

/// One point a request can resume from: the conversation's live end or one of its checkpoints.
struct ConversationDiskPoint {
    int64_t tokens = 0;
    uint64_t hash = 0;                          ///< conversation_disk_ids_hash of the first `tokens` ids
    std::vector<ConversationImageKey> imgs;     ///< the pictures below `tokens`, as the checkpoint keeps them
    bool live = false;
    bool operator==(const ConversationDiskPoint&) const = default;
};

uint64_t conversation_disk_ids_hash(const int32_t* ids, size_t n);
/// The live point first, then every checkpoint's, in the image's order.
std::vector<ConversationDiskPoint> conversation_disk_points(const SavedConversation& image);

class ConversationDiskTier {
public:
    struct Match {
        uint64_t id = 0;       ///< 0: none
        int64_t tokens = 0;
        bool live = false;
    };
    struct Entry {
        uint64_t id = 0, used = 0, bytes = 0;
        bool cvec = true;
        std::vector<ConversationDiskPoint> points;
        std::vector<uint64_t> files;   ///< each stage file's size, stage 0 first
        bool failed = false;           ///< this run could not restore it: not offered again until a restart (kept)
    };
    struct OpenReport {
        size_t kept = 0, dropped = 0, orphans = 0;   ///< index files taken, refused and stray files removed
        size_t other_folders = 0;                    ///< subfolders of other identities or layouts, left alone
        uint64_t other_bytes = 0;
    };

    /// Creates (or reopens) the tier's subfolder and indexes the conversations already there.  False: the tier stays
    /// off and `error` says why (the engine runs on without it).
    bool open(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
              const ConversationDiskLayout& layout, std::string& error, OpenReport* report = nullptr);
    bool enabled() const { return open_; }
    const std::string& folder() const { return folder_; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    size_t writes() const { return writes_; }
    size_t reads() const { return reads_; }
    size_t evictions() const { return evictions_; }
    bool contains(uint64_t id) const { return find(id) != nullptr; }
    const Entry* find(uint64_t id) const;

    /// The deepest point of any kept conversation that `prompt` starts with (same rule as conversation_prefix: the
    /// prompt's last token is never covered, pictures must match in order).  Ties prefer the most recently used.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::vector<int32_t> ids(prompt.size());
        for (size_t i = 0; i < prompt.size(); ++i) ids[i] = (int32_t) prompt[i];
        return best_ids(ids, images, cvec);
    }
    Match best_ids(const std::vector<int32_t>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const;

    /// A kept conversation with exactly this image's resume points and control-vector mode (nothing to write).
    uint64_t holding(const SavedConversation& image) const;
    /// Writes `image` (its stage images included) as one conversation: the conversations it supersedes go first
    /// (drop_superseded), then the least recently used until it fits.  `id` is the new (or the identical kept)
    /// conversation.  On failure nothing new is left on the disk and `status->error` says what kind of failure.
    bool spill(const SavedConversation& image, uint64_t& id, std::string& error, SessionStatus* status = nullptr,
               const SessionProgress& progress = {}, const SessionPhase& phase = {});
    /// Reads conversation `id` into `out` (stage images included; out.disk_id = id).  `limits[k]` bounds stage k's
    /// file (conversation_session_read_limits; a stage past the list reads with defaults).  A file that is corrupt or
    /// not this runtime's drops the conversation; a RAM or I/O refusal keeps it.  `out` is unchanged on failure.
    bool load(uint64_t id, const std::vector<SessionReadLimits>& limits, SavedConversation& out, std::string& error,
              SessionStatus* status = nullptr);
    /// Removes conversation `id` (its index file first, then its stage files).
    bool drop(uint64_t id);
    /// The kept conversations `image` supersedes (ConversationCache::drop_superseded's rule: their deepest checkpoint
    /// is one of the image's points), removed.  Returns how many.
    size_t drop_superseded(const SavedConversation& image, uint64_t keep = 0);
    /// This run cannot restore `id` (read limits, snapshot validation): it is not offered again, its files stay
    /// (another start-up with other settings may restore it; the budget's LRU removes it in time).
    void mark_failed(uint64_t id);
    /// Every kept conversation removed (#879: after non-finite logits any of them may carry the poison).
    size_t clear();

private:
    bool open_impl(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                   const ConversationDiskLayout& layout, std::string& error, OpenReport* report);
    bool parse_index(const std::string& path, uint64_t id, Entry& e) const;
    bool write_index(const Entry& e, std::string& error) const;
    bool make_room(uint64_t incoming, std::string& error, uint64_t keep = 0);
    void remove_files(uint64_t id, size_t stages) const;
    std::string path_of(uint64_t id, const char* suffix) const;
    std::string stage_path(uint64_t id, size_t stage) const;
    Entry* find_mut(uint64_t id);

    bool open_ = false;
    ConversationDiskOptions options_;
    SessionFileIdentity identity_;
    ConversationDiskLayout layout_;
    std::string folder_;
    std::vector<Entry> entries_;
    uint64_t bytes_ = 0, clock_ = 0, loading_ = 0;
    size_t writes_ = 0, reads_ = 0, evictions_ = 0;
};

} // namespace strata::core
