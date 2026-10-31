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
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct ConversationSpillMatch {
    std::string path;        // the session file to read back
    uint64_t file_bytes = 0;
    int64_t tokens = 0;
    bool live = false;
    // --conversation-cache-disk-delta: with a chain, the delta log's committed length (`path` is its base) and the
    // token count the chain's state holds (the last record's live ids).
    uint64_t delta_committed = 0;
    int64_t live_tokens = 0;
    explicit operator bool() const { return !path.empty() && tokens > 0; }
};

// A file target reserved on the request thread. The writer uses these owned paths and identity, never the cache,
// while it runs in the background. Publication checks the same reservation before changing the index.
struct ConversationSpillWrite {
    std::filesystem::path stem;
    SessionFileIdentity identity;
    bool prefix = false;
    bool valid = false;
    std::string session_path() const { return stem.string() + ".sess"; }
};
struct ConversationSpillDeltaWrite {
    size_t index = 0;
    std::filesystem::path stem;
    SessionFileIdentity identity;
    uint64_t file_bytes = 0, delta_bytes = 0, delta_accounted_bytes = 0;
    bool had_chain = false;
    bool valid = false;
    std::string session_path() const { return stem.string() + ".sess"; }
    std::string delta_path() const { return stem.string() + ".dlog"; }
};

// ---- The opt-in RECOMPUTE GATE and its directory ledger (.strata-cache-gate).
//
// A sidecar written under another engine CONFIGURATION is normally kept, never indexed.  With the gate on, the
// directory also reads its ledger: a text file the recompute tool writes after it has re-run a sample of a cached
// conversation's tokens on the current engine and found the result identical.  A foreign-config entry is indexed,
// and its stored config accepted when it is read, only when the ledger names it AND its batch passed for exactly
// (source model+config -> this model+config+this engine build).  A foreign MODEL is never approved.  The ledger is
// a rebuildable cache, never authority: the sidecar and the session file decide what an entry is, and a missing,
// truncated, edited or checksum-mismatched ledger approves nothing while the same-identity cache works as before.
//
// Format: UTF-8 text, one record per line, fields separated by single spaces, in this order (version 1):
//   strata-cache-gate <version>
//   batch <id> <src_model> <src_config> <dst_model> <dst_config> <engine_build> <sample> <pass|fail> <time_ms> <reason>
//   entry <stem> <batch_id> <file_bytes> <tokens> <last_used_ms>
//   sum <hash>
// The tokens <id>, <engine_build>, <sample>, <stem>, <batch_id> and <reason> carry no whitespace (a writer replaces
// it with '_'; an empty reason is '-').  Fingerprints are 0x followed by 16 hex digits; byte counts, token counts
// and times are decimal.  <stem> is a file name without its extension ("strata-conv-prefix-34").  <hash> is the
// 64-bit session hash (session_hash64, seed 0) of every byte before the "sum " line, as 16 hex digits, so a
// half-written or edited ledger is refused whole.  A writer must create a temporary file beside the ledger and
// rename it into place; a reader never sees a partially written one.  The recompute tool writes this file; the
// engine only reads it.
struct ConversationSpillGateBatch {
    std::string id;
    SessionFileIdentity source{};   // the identity the cached files carry
    SessionFileIdentity target{};   // the identity this cache serves
    std::string engine_build;       // the build that recomputed the sample
    std::string sample;             // the stem of the file the recompute used
    bool passed = false;            // result: pass (true) or fail (false)
    uint64_t verified_ms = 0;       // wall-clock time of the recompute, milliseconds since the epoch
    std::string reason;             // why a fail failed; empty for a pass
};
struct ConversationSpillGateEntry {
    std::string stem;               // the file name without its extension
    std::string batch_id;
    uint64_t file_bytes = 0;        // the base .sess size the batch approved
    int64_t tokens = 0;             // the sidecar's live token count at approval time
    uint64_t last_used = 0;         // wall-clock milliseconds since the epoch, informational
};
// Reads <directory>/.strata-cache-gate.  False (outputs cleared) for a missing, unreadable, unknown-version,
// malformed or checksum-mismatched file: a caller then approves nothing.
bool session_spill_gate_read(const std::filesystem::path& directory, std::vector<ConversationSpillGateBatch>& batches,
                             std::vector<ConversationSpillGateEntry>& entries, std::string& error);
// Writes the ledger atomically: an exclusive temporary beside it, flushed, then renamed over it.  False when a
// field would break the format (empty or whitespace-bearing token field) or the write fails.
bool session_spill_gate_write(const std::filesystem::path& directory,
                              const std::vector<ConversationSpillGateBatch>& batches,
                              const std::vector<ConversationSpillGateEntry>& entries, std::string& error);
// True when an entry row names this stem with this base size and token count, and its batch passed for exactly
// the given source identity, target identity and engine build.
bool session_spill_gate_approves(const std::vector<ConversationSpillGateBatch>& batches,
                                 const std::vector<ConversationSpillGateEntry>& entries, const std::string& stem,
                                 const SessionFileIdentity& source, const SessionFileIdentity& target,
                                 const std::string& engine_build, uint64_t file_bytes, int64_t tokens);

// Opt-in knobs of ConversationSpillCache::open.  The defaults are today's behaviour: the ledger is neither read nor
// written and a sidecar of another identity stays retained and unindexed.
struct ConversationSpillGateOptions {
    bool enabled = false;
    std::string engine_build;   // the current engine build (e.g. "0.1.52.20"); part of a batch's target tuple
};

class ConversationSpillCache {
public:
    // Opens (creating if needed) the spill directory and indexes conversations already in it. Left-over write
    // temporaries and incomplete owned groups are cleaned when possible; a partial cleanup is retained only as an
    // ineligible accounting entry. Files whose sidecar names another model/config identity are KEPT: never deleted,
    // not indexed, outside the disk budget; open() logs a warning with their count and bytes - remove them manually.
    // delta_chains (--conversation-cache-disk-delta) selects what NEW files look like: with it, saves stamp their
    // sidecar v2 (last-use stamp) and the budget evicts least-recently-used; without it (the default) new sidecars
    // stay baseline v1 and first-spilled eviction runs. Both modes read v1/v2 sidecars; a valid chain is retained and
    // accounted in base-only mode, but only delta mode offers it to the streaming restore.
    // gate.enabled (opt-in, default off): read the directory's recompute ledger and index a same-model file that
    // was saved under another engine configuration when the ledger approves it for this identity and engine build
    // (see ConversationSpillGateOptions); such an entry is read with its own config accepted.  A failed ledger read
    // approves nothing and leaves the rest of the cache unaffected; its reason is kept for gate_error().  A
    // gate-approved entry is READ-ONLY: reserve_delta (and so spill_delta) refuses it, since an appended record would
    // hold this cache's bytes after a file of the other configuration - such a conversation is written as a new file.
    bool open(const std::filesystem::path& directory, SessionFileIdentity identity, uint64_t budget_bytes,
              std::string& error, bool delta_chains = false, const ConversationSpillGateOptions& gate = {});
    bool enabled() const { return enabled_; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    size_t stale_files_wiped() const { return stale_files_wiped_; }
    size_t disk_evictions() const { return disk_evictions_; }
    // The recompute gate's startup facts, for the caller's one log line (open only reads the ledger when the gate is
    // on).  `gate_error` is the ledger read's refusal, empty when there was none or the gate is off.
    size_t gate_batch_count() const { return gate_batches_.size(); }
    size_t gate_approved_count() const;
    const std::string& gate_error() const { return gate_error_; }

    // The best resume this directory offers for the prompt, from the sidecars only (no K/V read). similarity and
    // n_min filter weak hits exactly as the RAM cache's best() does.
    template<class Token, class TokenAllocator, class ImageAllocator = std::allocator<ConversationImageKey>>
    ConversationSpillMatch best(const std::vector<Token, TokenAllocator>& prompt,
                                const std::vector<ConversationImageKey, ImageAllocator>& images, bool cvec,
                                double similarity, int64_t n_min) const {
        ConversationSpillMatch best_match;
        int64_t best_tokens = 0;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& entry = entries_[i];
            // A partial removal, a missing dependency, or a chain opened without the streaming restore mode is
            // never an eligible hit. Check regular-file types as well as path presence: a directory or symlink at a
            // required path is not a session/manifest/log.
            if (!entry.complete || (entry.has_chain && !delta_) || !entry_files_valid(entry)) continue;
            const ConversationMatch match = conversation_metadata_match(entry.live_meta,
                    entry.checkpoint_lengths, prompt, images, cvec, entry.cvec, similarity, n_min);
            if (match.tokens > best_tokens)
                best_tokens = match.tokens,
                best_match = {entry.session_path(), entry.file_bytes, match.tokens, match.live, entry.delta_bytes,
                              (int64_t) entry.live_meta.ids.size()};
        }
        return best_match;
    }

    // Reads a spilled conversation back into image (its K/V included), with the engine's read limits and identity.
    // `status` (optional) reports the verified file format: on a RAM refusal of a verified descriptor-format (v2/v3)
    // file the caller keeps the file for a later attempt (a format-v1 refusal, as before, discards it).
    bool load(const std::string& path, SavedConversation& image, const SessionReadLimits& limits,
              std::string& error, SessionStatus* status = nullptr) const;
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
                        std::string& error, bool durable = false, bool prefix = false, bool state_ref = false,
                        bool state_codec_rht = false, SessionWriteTimings* timings = nullptr,
                        bool layout_v3 = false);
    // --conversation-cache-disk-delta: appends one delta record for an indexed entry instead of rewriting its file.
    // `meta` + `kv` describe the conversation's current state; kv[i].sizes are full part sizes and first_bytes the
    // offset each part's data starts at.  The record is written to the entry's `<stem>.dlog`, its committed length is
    // published in `<stem>.dchain` (atomically) and then in the sidecar, so a crash between any two leaves the chain
    // at its last committed record.  `bytes` is the record's length.  On success the entry's sidecar describes the
    // new state. A prefix, disabled directory or incomplete/unusable entry refuses.
    bool spill_delta(size_t index, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                     int64_t first_token, size_t& bytes, std::string& error, bool durable = false,
                     bool state_codec_rht = false, SessionWriteTimings* timings = nullptr,
                     bool layout_v3 = false);
    // Two-phase disk-only saves. reserve_* runs before the task starts; the task writes target.session_path() with
    // session_file_write, or target.delta_path() at target.delta_bytes with session_delta_write, using target.identity.
    // It owns all source/state inputs and never calls this cache. At a safe boundary, publish_* runs on the request
    // thread, after joining the task. Base publication writes the sidecar and enforces the budget. Delta publication
    // also accounts a failed or uncommitted append, preserving the previous committed length. A delta reservation is
    // rejected if its entry's index, path, base size or committed/accounted log generation changed before publication.
    bool reserve_streamed(const SavedConversation& meta, ConversationSpillWrite& target, std::string& error,
                          bool prefix = false);
    bool publish_streamed(const ConversationSpillWrite& target, const SavedConversation& meta, size_t bytes,
                          std::string& error);
    bool reserve_delta(size_t index, ConversationSpillDeltaWrite& target, std::string& error);
    bool publish_delta(const ConversationSpillDeltaWrite& target, const SavedConversation& meta, size_t written,
                       bool write_succeeded, std::string& error);
    // The index of the entry whose session file is `path`, or size() when none.
    size_t find(const std::string& path) const;
    // The identity the entry's file and sidecar carry: this cache's identity, except for a gate-approved entry saved
    // under another engine configuration.  False for an unknown index.
    bool entry_source_identity(size_t index, SessionFileIdentity& out) const;
    // What a delta save needs to know about an entry (see generate.cpp): its token count, its deepest checkpoint,
    // its file sizes and whether its live ids are a prefix of this conversation's tokens and images.  False for a
    // prefix entry or an unknown index.
    bool entry_info(size_t index, int64_t& tokens, int64_t& deepest, uint64_t& base_bytes, uint64_t& delta_bytes,
                    bool& prefix, std::string& path) const;
    // For a delta save: the entry's saved token ids (and images) match this state's first `n` tokens - the prefix
    // the chain's K/V covers.  Only that prefix matters: a restore rewrites the tail, so the state may be shorter
    // than the entry and still be a valid append target.
    bool entry_matches_prefix(size_t index, const std::vector<int32_t>& ids, size_t n,
                              const std::vector<ConversationImageKey>& images) const;
    // The publication-gap check (the sidecar is written after the chain manifest, so a crash can leave it one save
    // behind): the chain's last committed record must describe exactly what the sidecar advertises - the same live
    // token ids and images and the same checkpoint lengths - or restoring would mix two saves' state.  False for a
    // prefix entry or an unknown index.
    bool entry_chain_tail_matches(size_t index, const std::vector<int32_t>& tail_ids,
                                  const std::vector<ConversationImageKey>& tail_imgs,
                                  const std::vector<size_t>& tail_checkpoint_lengths) const;
    // A successful restore is an access: stamps the entry as the most recently used (persisted in the sidecar, so
    // the budget's eviction order - least recently used first - survives a restart).  Failures are not.
    void touch(const std::string& path);
    // The entry's persisted last-use stamp in milliseconds since the epoch (0: a v1 sidecar, unknown age).
    uint64_t entry_last_used(size_t index) const;
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
        std::filesystem::path stem;          // "<dir>/strata-conv-<serial>" (entry artifacts share it)
        uint64_t file_bytes = 0;             // accounted bytes of the regular .sess base
        uint64_t delta_bytes = 0;            // committed length of "<stem>.dlog" (restore/append boundary)
        uint64_t delta_accounted_bytes = 0;   // bytes of the log actually charged to this entry
        uint64_t meta_bytes = 0;              // zero for a legacy v1 sidecar; otherwise its charged bytes
        uint64_t chain_file_bytes = 0;        // charged bytes of "<stem>.dchain"
        uint64_t last_used = 0;               // milliseconds since the epoch of the last save or successful restore
        SessionFileIdentity source_identity{};   // the identity the sidecar/file carry (its config may be a gate-approved legacy one)
        bool cvec = true;
        ConversationCheckpoint live_meta;    // ids + imgs only; no running state
        std::vector<size_t> checkpoint_lengths;
        bool prefix = false;                 // a persistent root entry (named strata-conv-prefix-N)
        bool has_chain = false;              // a log/manifest exists or this entry has published a chain
        bool complete = true;                // false while owned files are partial/unusable; retained for accounting
        std::string session_path() const { return (stem.string() + ".sess"); }
        std::string meta_path() const { return (stem.string() + ".meta"); }
        std::string delta_path() const { return (stem.string() + ".dlog"); }
        std::string chain_path() const { return (stem.string() + ".dchain"); }
        uint64_t total_bytes() const { return file_bytes + delta_accounted_bytes + meta_bytes + chain_file_bytes; }
    };
    bool entry_files_valid(const Entry& entry) const;
    void enforce_budget();
    void adjust_bytes(uint64_t before, uint64_t after);
    // Removes only this entry's owned artifacts. `removed` is the bytes debited from this entry's existing ledger;
    // a false result retains an incomplete, ineligible entry with exact remaining ownership charges for retry.
    bool remove_group(Entry& entry, uint64_t& removed, std::vector<std::string>& left);
    std::filesystem::path next_stem(bool prefix = false);
    // The gate's identity-level approval: a passed batch for exactly this source identity -> this cache's identity
    // and the configured engine build.  False when the gate is off or the ledger approved nothing.
    bool gate_identity_approved(const SessionFileIdentity& source) const;
    // The gate's entry-level approval: a ledger row for this stem whose batch also passes (identity + build), and
    // the row's size and token count match the file and sidecar (which stay authoritative).
    bool gate_entry_approved(const std::string& stem, const SessionFileIdentity& source, uint64_t file_bytes,
                             int64_t tokens) const;
    bool index_written(const std::filesystem::path& stem, uint64_t file_bytes, const SavedConversation& meta,
                       std::string& error, bool prefix = false);
    bool read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity,
                      bool& stamped /* out: the v2 sidecar carries a persisted last-use stamp */, std::string& error) const;
    // with_last_used (--conversation-cache-disk-delta) writes the v2 sidecar carrying Entry::last_used; without it
    // the baseline v1 bytes are written, so a directory a pre-delta build reads is never migrated by a delta-off run
    bool write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error,
                       bool with_last_used) const;
    // The chain manifest beside <stem>.dlog: base size (must match the session file) and the committed length.
    bool read_chain(const std::filesystem::path& chain, uint64_t base_size, const std::filesystem::path& dlog,
                    uint64_t& committed, std::string& error) const;
    bool write_chain(const std::filesystem::path& chain, uint64_t base_size, uint64_t committed,
                     std::string& error) const;

    bool enabled_ = false;
    SessionFileIdentity identity_{};
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0;
    size_t stale_files_wiped_ = 0, disk_evictions_ = 0;
    std::filesystem::path directory_;
    std::vector<Entry> entries_;   // index order only; the budget evicts by Entry::last_used (least recent first)
                                   // when delta_chains is on, else first-spilled first (the baseline policy)
    bool delta_ = false;           // --conversation-cache-disk-delta: v2 sidecars, LRU eviction, chain appends
    bool gate_ = false;            // the opt-in recompute gate: the ledger below may approve foreign-config entries
    std::string gate_engine_build_;
    std::vector<ConversationSpillGateBatch> gate_batches_;
    std::vector<ConversationSpillGateEntry> gate_entries_;
    std::string gate_error_;       // the ledger read's refusal (empty: none, or the gate is off); see gate_error()
    std::string pinned_path_;
};

} // namespace strata::core
