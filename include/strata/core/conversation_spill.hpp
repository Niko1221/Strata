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

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

// What the disk tier does when its budget is reached (--conversation-cache-spill-when-full).
//   evict_oldest  the default, and what upstream always did: the GC drops the oldest conversation to make room.
//   reject        never evicts: a spill that would not fit the budget is refused and reported, and nothing is removed.
enum class SpillWhenFull { evict_oldest, reject };

// What --conversation-cache-spill-on selects. evict = the delta-1 behaviour: a conversation reaches disk only when
// the RAM cache evicts it (the tier is overflow). park = the mirror: the conversation is also written the moment it
// is parked at the end of a request, so an abrupt close loses at most the request in flight. Without a spill
// directory both are inert: nothing is created and no byte is written.
enum class SpillOn { evict, park };

// Delta 5a (opt-in): what the disk tier does with a copy it would otherwise DISCARD - the compacted conversation
// (--conversation-cache-spill-divergence-tokens), the copy a re-park supersedes, and the provisional state of a
// cancelled request. off (the default) is the delta-4 behaviour, byte for byte: the copy is removed. ids MOVES (or,
// for the cancellation, WRITES) only the sidecar - the token and image ids, no K/V - into the archive; state moves
// the whole session file (the K/V included) together with the sidecar. The archive is NEVER a reuse candidate: it
// is evidence and recovery, not a cache (the attention is causal, so an archived tail can never be a hit), so its
// entries never enter the tier's index and best() never returns one.
enum class SpillArchiveMode { off, ids, state };

// The archive's own settings: its own folder, its own budget and its own GC, none of them shared with the tier's
// bytes_/enforce_budget. Only used when `mode` is not off (the tier must be on: the archive lives beside it).
struct SpillArchiveConfig {
    std::filesystem::path directory;              // where the discarded copies are kept (default <spill-dir>/archive)
    SpillArchiveMode mode = SpillArchiveMode::off;
    uint64_t budget_bytes = 0;                    // the archive's own byte budget (0 = no byte budget)
    size_t keep = 0;                              // most recent copies kept per conversation (0 = no per-conversation cap)
    int64_t max_age_days = 0;                     // optional age retention: copies older than N days go (0 = off)
    int64_t turn_token = -1;                      // groups archived copies into conversations (the tier's --turn-token)
};

// The header of a conversation: the tokens through its first exchange (system prompt + first user turn, ending at
// the first assistant turn). It is the fixed part of a chat: it does not change as turns are added, and a different
// conversation has a different one. --turn-token < 0 (no turn token) yields 0, so nothing can be called a rewrite.
inline size_t conversation_header_length(const std::vector<int32_t>& ids, int64_t turn_token) {
    if (turn_token < 0 || ids.empty()) return 0;
    int seen = 0;
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == (int32_t) turn_token && ++seen == 3) return i;   // start of the first assistant turn
    return ids.size();   // fewer than three turns: the whole thing is the header (nothing to call a tail rewrite)
}

// The writer's key for a conversation: a stable fingerprint of its header (or of its first tokens when there is no
// turn token), unchanged as the conversation grows. Two conversations of one client share the system prompt but not
// the header, so they are distinct cells.
inline uint64_t conversation_key(const std::vector<int32_t>& ids, int64_t turn_token) {
    if (ids.empty()) return 0;
    size_t n = conversation_header_length(ids, turn_token);
    // No turn token (or a chat shorter than three turns): a fixed prefix of the first tokens, still unchanged as the
    // conversation grows, so a park of a growing chat stays one cell.
    if (n == 0 || n >= ids.size()) n = ids.size() < 4096 ? ids.size() : 4096;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= (uint64_t) (uint32_t) ids[i]; h *= 1099511628211ull; }
    return h;
}

// R4: a request cancelled before it finished leaves a provisional session. Its live ids are reverted to the last
// turn boundary the read actually reached, so the state describes exactly the closed conversation the client will
// resend; the caller then suppresses the durable copy (the previous good one is left alone). Returns true when the
// ids were reverted. `reached` is the exclusive end of what was read: the boundary is never taken past it.
inline bool revert_to_turn_boundary(std::vector<int32_t>& ids, int64_t turn_token, size_t reached) {
    if (turn_token < 0) return false;
    const size_t limit = reached < ids.size() ? reached : ids.size();
    for (size_t i = limit; i-- > 0;)
        if (ids[i] == (int32_t) turn_token) {
            if (i + 1 == ids.size()) return false;   // already ends on the boundary: nothing to revert
            ids.resize(i + 1);
            return true;
        }
    return false;
}

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
    // Locked: the park writer drains the same directory from its own thread.
    bool enabled() const { std::lock_guard<std::mutex> lk(mu_); return enabled_; }
    SpillWhenFull when_full() const { std::lock_guard<std::mutex> lk(mu_); return when_full_; }
    int64_t max_age_days() const { std::lock_guard<std::mutex> lk(mu_); return max_age_days_; }
    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return entries_.size(); }
    uint64_t bytes() const { std::lock_guard<std::mutex> lk(mu_); return bytes_; }
    // What the last scan and the budget kept instead of wiping, by reason (reporting only; nothing is deleted here).
    size_t stale_files_kept() const { std::lock_guard<std::mutex> lk(mu_); return stale_files_kept_; }        // unreadable/broken sidecar or a lost session file
    size_t foreign_files_kept() const { std::lock_guard<std::mutex> lk(mu_); return foreign_files_kept_; }    // another model/config identity
    size_t orphan_files_kept() const { std::lock_guard<std::mutex> lk(mu_); return orphan_files_kept_; }      // a session file with no sidecar
    size_t oversized_files_kept() const { std::lock_guard<std::mutex> lk(mu_); return oversized_files_kept_; }// a conversation larger than the budget
    size_t disk_evictions() const { std::lock_guard<std::mutex> lk(mu_); return disk_evictions_; }            // removed by the budget GC (oldest first)
    size_t age_evictions() const { std::lock_guard<std::mutex> lk(mu_); return age_evictions_; }              // removed by --conversation-cache-spill-max-age-days
    // R2: copies discarded because the client rewrote the conversation's tail (a compaction or an edited history).
    size_t compacted() const { std::lock_guard<std::mutex> lk(mu_); return compacted_; }

    // Delta 5a: the archive the tier MOVES its discarded copies into instead of removing them. Off by default: with
    // mode off no directory is created and no byte is written (the tier's behaviour is byte-for-byte delta 4).
    // `archived` is how many conversations the archive holds now, `archived_bytes` what they take (its OWN budget,
    // never the tier's bytes_). The archive's GC is its own (enforce_archive: per-conversation keep, then budget,
    // oldest first) and never touches the live tier; the tier's enforce_budget/enforce_age never touch the archive.
    bool open_archive(const SpillArchiveConfig& config, std::string& error);
    bool archive_enabled() const { std::lock_guard<std::mutex> lk(mu_); return archive_enabled_; }
    SpillArchiveMode archive_mode() const { std::lock_guard<std::mutex> lk(mu_); return archive_mode_; }
    size_t archived() const { std::lock_guard<std::mutex> lk(mu_); return archive_entries_.size(); }
    uint64_t archived_bytes() const { std::lock_guard<std::mutex> lk(mu_); return archive_bytes_; }
    size_t archive_evictions() const { std::lock_guard<std::mutex> lk(mu_); return archive_evictions_; }
    size_t archive_age_evictions() const { std::lock_guard<std::mutex> lk(mu_); return archive_age_evictions_; }
    // Delta 5a, gate refinement (a): the PRE-revert state of a cancelled request. Only the ids and images are
    // archived (the K/V past the last turn boundary is the discarded answer's, not a clean prefix), so the entry is
    // a sidecar with no session file. Never a reuse candidate; best() never returns it.
    bool archive_provisional(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                             const std::vector<size_t>& checkpoint_lengths, bool cvec, std::string& error);

    // The best resume this directory offers for the prompt, from the sidecars only (no K/V read). similarity and
    // n_min filter weak hits exactly as the RAM cache's best() does.
    template<class Token>
    ConversationSpillMatch best(const std::vector<Token>& prompt,
                                const std::vector<ConversationImageKey>& images, bool cvec,
                                double similarity, int64_t n_min) const {
        std::lock_guard<std::mutex> lk(mu_);
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
    // `stored_path`, when given, receives the session file just written (the park writer keeps it out of the
    // supersede pass that follows, so the new copy is never the one removed).
    bool spill(const SavedConversation& image, std::string& error, std::string* stored_path = nullptr);
    bool erase(const std::string& path, std::string& error);
    // A conversation being restored is held out of eviction until it is put back in RAM or dropped.
    void pin(const std::string& path);
    void unpin(const std::string& path);
    // Removes the disk copies of this conversation a turn back, the same rule the RAM cache applies before evicting.
    // `keep` (a session path) is never removed: the copy just written supersedes the older ones.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                           const std::string& keep = std::string());
    // R2: the client rewrote the conversation's tail (a compaction or an edited history), so the stored copy
    // describes nothing it will send. A stored copy the prompt EXTENDS is the normal turn and is kept; a copy whose
    // header does not match the prompt is another conversation and is left alone. A copy whose common prefix is
    // shorter than `divergence_tokens` while its header still matches is discarded and counted in compacted().
    // Returns how many were discarded. The prompt's token type is the caller's (the serve loop reads int64).
    template<class Token>
    size_t discard_diverged(const std::vector<Token>& prompt, bool cvec, size_t divergence_tokens, int64_t turn_token) {
        std::vector<int32_t> ids;
        ids.reserve(prompt.size());
        for (const Token& token : prompt) ids.push_back((int32_t) token);   // token ids fit int32, the sidecar's width
        return discard_diverged_impl(ids, cvec, divergence_tokens, turn_token);
    }
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
    size_t discard_diverged_impl(const std::vector<int32_t>& prompt, bool cvec, size_t divergence_tokens, int64_t turn_token);
    void enforce_budget();
    void enforce_age();
    bool remove_entry(size_t index);
    // Delta 5a: the discard point shared by the compaction and the supersede pass. It MOVES the entry into the
    // archive when the archive is on, and removes it otherwise; false only when nothing left the index (a failed
    // move/removal, exactly as a failed removal left the entry before).
    bool discard_entry(size_t index);
    bool archive_entry(size_t index);
    // How many bytes a tier entry would occupy in the archive (its sidecar, plus the session files in state mode):
    // used to refuse archiving a copy that could not fit the archive's own budget.
    uint64_t archive_entry_bytes(const Entry& entry) const;
    std::filesystem::path next_archive_stem();
    bool remove_archive_entry(size_t index);
    void enforce_archive();
    bool read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity, std::string& error) const;
    bool write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error) const;

    bool enabled_ = false;
    SpillWhenFull when_full_ = SpillWhenFull::evict_oldest;
    int64_t max_age_days_ = 0;
    SessionFileIdentity identity_{};
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0;
    size_t stale_files_kept_ = 0, foreign_files_kept_ = 0, orphan_files_kept_ = 0, oversized_files_kept_ = 0;
    size_t disk_evictions_ = 0, age_evictions_ = 0, compacted_ = 0;
    std::filesystem::path directory_;
    std::vector<Entry> entries_;   // oldest spill first
    std::string pinned_path_;
    // Delta 5a: the archive (its own index, bytes and serial, none shared with the tier). The archive entries are
    // NOT in entries_, so best() can never return one - the archive is evidence, never a reuse candidate.
    bool archive_enabled_ = false;
    SpillArchiveMode archive_mode_ = SpillArchiveMode::off;
    std::filesystem::path archive_dir_;
    uint64_t archive_budget_ = 0, archive_bytes_ = 0;
    size_t archive_keep_ = 0, archive_evictions_ = 0, archive_age_evictions_ = 0;
    uint64_t archive_serial_ = 0;
    int64_t archive_turn_token_ = -1;
    int64_t archive_max_age_days_ = 0;
    std::vector<Entry> archive_entries_;   // oldest first
    mutable std::mutex mu_;        // the park writer drains the same directory from its own thread
};

// The park writer (--conversation-cache-spill-on park). The disk tier stops being overflow and becomes a mirror:
// the state parked at the end of a request is written to disk. One cell per conversation holds only the NEWEST
// state to write; a background thread drains it. Parks that arrive while a write is in flight REPLACE the cell
// (collapsed, counted) instead of queueing, so a turn with five tool calls writes once, not five times. Each
// drained state is spilled and the copy it supersedes (the previous copy of the same conversation, and the copies
// a supersede pass removes) is dropped AFTER the new one is on disk, so the folder holds one copy per conversation
// and the previous good copy is only replaced once its successor exists.
class ConversationSpillWriter {
public:
    ConversationSpillWriter() = default;
    ~ConversationSpillWriter() { stop(); }
    ConversationSpillWriter(const ConversationSpillWriter&) = delete;
    ConversationSpillWriter& operator=(const ConversationSpillWriter&) = delete;

    // `cache` must outlive the writer (otherwise stop() first). throttle_s: 0 = write on every post; N > 0 = do not
    // rewrite the same conversation inside an N-second window (the park is postponed to a later one, counted).
    void start(ConversationSpillCache* cache, int64_t throttle_s) {
        std::lock_guard<std::mutex> lk(mu_);
        cache_ = cache;
        throttle_s_ = throttle_s < 0 ? 0 : throttle_s;
        if (running_) return;
        running_ = true;
        quit_ = false;
        worker_ = std::thread([this] { loop(); });
    }
    // Stops the thread (after draining what it holds) and writes anything left on the calling thread.
    void stop() {
        std::thread worker;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (running_) {
                quit_ = true;
                running_ = false;
                worker = std::move(worker_);
            }
        }
        cv_.notify_all();
        if (worker.joinable()) worker.join();
        drain();
    }
    bool running() const { std::lock_guard<std::mutex> lk(mu_); return running_; }

    // Cheap pre-check so the caller does not build the copy post() would only discard: would a state of this size
    // for this key be written right now (grown past the last write, and outside the throttle window)?
    bool wants(uint64_t key, size_t tokens) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto last = last_written_.find(key);
        if (last != last_written_.end() && tokens <= last->second) return false;
        if (throttle_s_ > 0) {
            auto throttled = throttle_at_.find(key);
            if (throttled != throttle_at_.end() &&
                std::chrono::steady_clock::now() - throttled->second < std::chrono::seconds(throttle_s_)) return false;
        }
        return true;
    }

    // Hands a conversation to the writer. A state whose live ids did not grow past the last one written for this
    // key is skipped (nothing changed); a park inside the throttle window is postponed. Returns true when the state
    // was queued (or replaced a queued one).
    bool post(uint64_t key, SavedConversation&& image) {
        std::lock_guard<std::mutex> lk(mu_);
        if (image.live.ids.empty()) return false;
        const size_t size = image.live.ids.size();
        auto last = last_written_.find(key);
        if (last != last_written_.end() && size <= last->second) { ++skipped_; return false; }
        const auto now = std::chrono::steady_clock::now();
        auto throttled = throttle_at_.find(key);
        if (throttle_s_ > 0 && throttled != throttle_at_.end() &&
            now - throttled->second < std::chrono::seconds(throttle_s_)) { ++throttled_; return false; }
        Cell& cell = cells_[key];
        if (cell.occupied) ++collapsed_;
        cell.key = key;
        cell.image = std::move(image);
        cell.occupied = true;
        cv_.notify_one();
        return true;
    }

    // Writes every pending state, on the calling thread. Returns how many were written.
    size_t drain() {
        size_t written = 0;
        for (;;) {
            Cell cell;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!take_locked(cell)) break;
            }
            if (write_cell(cell)) ++written;
        }
        return written;
    }

    size_t writes() const { std::lock_guard<std::mutex> lk(mu_); return writes_; }        // states that reached disk
    size_t collapsed() const { std::lock_guard<std::mutex> lk(mu_); return collapsed_; }  // parks a newer state replaced
    size_t throttled() const { std::lock_guard<std::mutex> lk(mu_); return throttled_; }  // parks postponed by the throttle
    size_t skipped() const { std::lock_guard<std::mutex> lk(mu_); return skipped_; }      // parks that changed nothing
    size_t refused() const { std::lock_guard<std::mutex> lk(mu_); return refused_; }      // cache refused a write

private:
    struct Cell {
        uint64_t key = 0;
        SavedConversation image;
        bool occupied = false;
    };
    // A cell leaves the writer's map while its write runs, so the NEXT park of the same conversation becomes a
    // fresh cell (written after this one and superseding it). Anything posted while the write runs is a collapse
    // candidate only once it is queued; the collapse counter counts replaced queued states.
    bool take_locked(Cell& out) {
        for (auto& kv : cells_) {
            if (!kv.second.occupied) continue;
            out = std::move(kv.second);
            kv.second.occupied = false;
            kv.second.image = SavedConversation{};
            return true;
        }
        return false;
    }
    bool write_cell(Cell& cell) {
        ConversationSpillCache* cache = cache_;
        bool ok = false;
        if (cache != nullptr && cache->enabled()) {
            std::string path, error;
            ok = cache->spill(cell.image, error, &path);
            if (ok) cache->drop_superseded(cell.image.live.ids, cell.image.live.imgs, cell.image.checkpoints,
                                           cell.image.cvec, path);
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (ok) {
                ++writes_;
                last_written_[cell.key] = cell.image.live.ids.size();
                throttle_at_[cell.key] = std::chrono::steady_clock::now();
            } else {
                ++refused_;
            }
        }
        return ok;
    }
    void loop() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [this] { return quit_ || any_pending_locked(); });
            if (quit_) break;
            Cell cell;
            if (!take_locked(cell)) continue;
            lk.unlock();
            write_cell(cell);
            lk.lock();
        }
    }
    bool any_pending_locked() const {
        for (const auto& kv : cells_) if (kv.second.occupied) return true;
        return false;
    }

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread worker_;
    ConversationSpillCache* cache_ = nullptr;
    bool running_ = false, quit_ = false;
    int64_t throttle_s_ = 0;
    size_t writes_ = 0, collapsed_ = 0, throttled_ = 0, skipped_ = 0, refused_ = 0;
    std::map<uint64_t, Cell> cells_;
    std::map<uint64_t, size_t> last_written_;   // per key: the live ids of the last state written
    std::map<uint64_t, std::chrono::steady_clock::time_point> throttle_at_;
};

} // namespace strata::core
