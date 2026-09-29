// include/strata/platform/kv_nvme.hpp - NVMe cold tier for warm sessions (docs/nvme-kv-cache-design.md).
//
// A whole-session snapshot: the full attention KV (host copy, all formats) + the indexer positional state
// (idx_pooled / idx_block_pos / idx_dead / idx_tail) + the running state (gdn / ple) + the MTP drafter's host
// KV copy, keyed by the exact token prefix.  Dumped synchronously from the pinned host arena on DONE; promoted
// by reading back into that arena + kv_stream_reset + the drafter ring refill.  No new CUDA kernels: plain file
// I/O on mapped pinned memory for the KV host copies, cudaMemcpy for the device-side indexer/gdn/ple arrays.
//
// `KvNvmeStore` is the automatic layer (Steps 1-3): a directory of snapshots scanned at startup, an idempotent
// dump (exact-prefix skip; the previous dump of a growing conversation is superseded), promote-by-prefix-match
// in the serve loop's resume selection, and an LRU byte cap.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/conversation_cache.hpp"    // ConversationImageKey, ConversationCheckpoint
#include "strata/core/conversation_snapshot.hpp"  // conversation_geometry_key: the ONE geometry identity
#include "strata/core/layer.hpp"   // QsaState
#include "strata/core/session.hpp" // SessionState
#include "strata/core/weights.hpp" // ModelGeometry

namespace strata::platform {

/// THE FORMAT VERSION this build writes, and THE HEADER SIZE a reader must skip to reach the ids.  Both are
/// named here, once, because two things besides this file depend on them: the `static_assert`s under the struct
/// (which pin them against the layout the file depends on), and `tools/nvme_p0_test.sh` /
/// `tools/nvme_steps123_test.sh`, which read a snapshot's ids at a fixed offset and were previously told the
/// number by hand (`HDR=208`, before that a hardcoded `104`).  A shell script cannot evaluate a C++
/// `sizeof`, so the scripts read these two lines (`tools/nvme_header_layout.sh`) and refuse to run if either is
/// missing - and they also check the version field of the file they are reading against `$NVME_VERSION`, so a
/// snapshot written by a DIFFERENT build is caught instead of being parsed at the wrong offset.
inline constexpr uint32_t kNvmeFormatVersion = 3;
inline constexpr uint64_t kNvmeHeaderBytes = 208;

/// On-disk header.  Carries the geometry + format tag so a restore refuses (never converts) a mismatch.
///
/// THIS IS A RAW C++ STRUCT COPIED INTO THE ENVELOPE, which is exactly what the shared core's boundary forbids for
/// a disk adapter (no C++ structs, no native layout, an implicit ABI).  It stays that way in this step on purpose
/// - the segment table and the field-by-field encode are C9's remaining work.  What step 3 DID settle is the key:
/// the tag below IS the shared core's `conversation_geometry_key`, so the two tiers refuse the same mismatch, and
/// the static_asserts under it pin the layout the file depends on (every field fixed-width, the struct
/// padding-free) because the integrity footer cannot see the header that describes the layout.
/// See docs/nvme-kv-cache-design.md ("Where step 2 leaves NvmeHeader").
struct NvmeHeader {
    uint32_t magic = 0x5E564D45;   // "^VME"
    // v3: the running-state `dead` / `block_pos` segments hold the TURN-BOUNDARY checkpoint's copies, not the
    // live device arrays (collision C5).  A v2 file's `block_pos` is the state at the point the dump was taken,
    // which for a turn-boundary snapshot is a position the snapshot does not describe, so the two files are not
    // interchangeable even though the segments are the same size in the same order: a reader cannot tell them
    // apart from the bytes.  Version 2 is therefore REFUSED, not reinterpreted.
    uint32_t version = kNvmeFormatVersion;
    int64_t L = 0;                 // ids consumed (the prefix length)
    int64_t n_imgs = 0;
    int32_t cvec = 0;
    int32_t kv_format = 0;         // strata::kernels::KvFormat (kKvF16 / kKvInt8 / kKvQ4)
    // ONE geometry identity (collision C9 / step 3): the SHARED CORE's 18-field key, verbatim and in its order -
    // `strata::core::conversation_geometry_key(g)`, the same array `SavedConversation::geometry` holds.  The old
    // tag was a DERIVED PROJECTION of it (n_qsa / n_gdn from n_layers + qsa_interval, idx_dim from idx_key_dim),
    // so the two tiers keyed the same conversation differently and could refuse different mismatches.  The three
    // fields after it are NOT part of that key and are not model identity: they are the runtime shapes the segment
    // walk cannot re-derive from a model, recorded so a reader VALIDATES them instead of re-deriving them from a
    // live engine (a v2 restore did re-derive them, which is why a sizing change surfaced as "layout mismatch").
    std::array<int64_t, 18> geometry{};
    int64_t page_size = 0, idx_block = 0, max_cells = 0;
    int64_t mtp_host = 0;          // how many drafter host-KV arrays the file holds (0 if the drafter is resident)
};

// The file IS this struct's bytes, so the layout is the format.  Every field is fixed-width and the struct is
// padding-free at 208 bytes, which is what lets the ids start at a fixed offset; a field added, widened or
// REORDERED fails the build rather than silently re-mapping every segment after it.  The integrity footer covers
// only the payload, so the header - the one thing that can move the whole layout - has to be pinned here.
static_assert(sizeof(NvmeHeader) == kNvmeHeaderBytes && alignof(NvmeHeader) == 8,
              "the NVMe envelope's header no longer matches kNvmeHeaderBytes: the ids no longer start where a "
              "reader expects (tools/nvme_header_layout.sh reads that constant for the shell oracles)");
static_assert(offsetof(NvmeHeader, L) == 8 && offsetof(NvmeHeader, geometry) == 32 &&
                  offsetof(NvmeHeader, page_size) == 176 && offsetof(NvmeHeader, mtp_host) == 200,
              "the NVMe header gained padding or reordered a field: the file is no longer a described record");
static_assert(sizeof(strata::core::conversation_geometry_key(strata::core::ModelGeometry{})) ==
                  sizeof(NvmeHeader{}.geometry),
              "the shared core's geometry key and the envelope's copy of it must be the same 18 int64 fields");

/// Dump the live session's full state to `path`.  The caller has synchronized the device (the checkpoint_save
/// contract).  Requires streamed KV (kv_mode != 0): the host copy is the source of truth.  The file is fsynced.
///
/// `at` takes the snapshot at an EARLIER position T = at->ids.size() <= L - the chat-turn boundary - and `ids` is
/// then that boundary (the key the next request replays).
/// The running state then comes from the checkpoint's blobs (a ConversationCheckpoint's gdn/ple/tails, which are
/// the state AT T), the KV cells / pooled rows / dead key are truncated to T (their contents below T are untouched
/// by the generation that followed), and the drafter copy covers [0, min(T, max_cells)).  This is the snapshot the
/// NEXT REQUEST can match: a chat client re-sends the prompt but not the model's hidden reasoning tokens, so a
/// full-L snapshot (which includes them) can never full-prefix-match the next turn.
/// `at` is the SHARED core's checkpoint type (docs/nvme-kv-cache-design.md step 2): its `ids` are the
/// boundary, and its blobs are validated against `conversation_state_sizes` before a byte of them is written.
/// `imgs` are the images BELOW that boundary - the checkpoint's own list, not the live conversation's pictures.
/// An image at or past `L` describes a token the snapshot does not hold, and the dump refuses one: the resume
/// match compares the next request's images below `L` against this segment.
/// Its `dead` / `block_pos` blobs ARE what the file carries (collision C5, settled here): every running-state
/// segment - gdn / ple / tails / dead / block_pos - comes from ONE source, the boundary checkpoint when one is
/// given and the live device arrays otherwise (where the live arrays ARE the state at `L`).  The alternative -
/// reading `dead` / `block_pos` off the live device at a turn boundary - writes a value from past `L`: the
/// pooling kernel rewrites `block_pos` with each completed block's first-cell position (qsa.cu:216), so the live
/// copy describes a position the snapshot does not hold.
///
/// THE SPILL PRIMITIVE (collision C1): this is the ONLY way a session reaches disk.  A spill re-keys at the chat
/// turn boundary; it never copies a parked image verbatim.  A parked (RAM-evicted) conversation's consumed ids
/// include the model's hidden reasoning tokens, which a client re-sending history will not reproduce, so its
/// checkpoint - not its live arrays - is what a spill may write.
bool nvme_dump_at(const char* path, const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
                  const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
                  const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec,
                  const strata::core::ConversationCheckpoint* at, std::string& err);

/// The full consumed state at DONE (the Step 0 spike's dump; the automatic store uses nvme_dump_at at the
/// turn boundary - see the comment above).
bool nvme_dump(const char* path, const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
               const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
               const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec, std::string& err);

/// Read a snapshot back into `ss` (and the drafter's state), and say WHICH KIND of failure happened if it did.
///
/// THE FAILURE CONTRACT (collision C7, docs/nvme-kv-cache-design.md §5).  The return type is the SHARED CORE's
/// `ConversationRestore`, not a `bool` and not a second enum: the RAM tier and the disk tier report a failed
/// restore with one vocabulary, so one serve-loop rule can cover both.  What the two values mean HERE:
///
///   `restored`        - every segment applied and the final `cudaDeviceSynchronize()` succeeded.
///   `invalid`         - RECOVERABLE.  Either the tier refused before the apply pass began (magic, format version,
///                       geometry, header sizes, layout walk, payload digest, a live array too small, a null target
///                       buffer) - so no session byte was written - or the snapshot applied cleanly and something
///                       OTHER than the session disagreed (see `KvNvmeStore::restore`).  In both cases the caller
///                       may drop the snapshot and re-read the prompt from token 0.
///   `transfer_failed` - FATAL.  A `cudaMemcpy` inside the apply pass, or a `cudaDeviceSynchronize` before or after
///                       it, failed.  The apply pass is a loop, so a failure in its middle leaves the session
///                       HALF-WRITTEN by construction (the shared core's own fixture asserts exactly this shape for
///                       its restore: `conversation_validation_test.cpp:167-169`), and nothing has yet shown the
///                       CUDA context still answers.  A clean reset from here is the recovery the core's boundary
///                       says needs "its own proof that the CUDA context remains usable"; this tier does not have
///                       that proof, so it reports the class and lets the caller stop.
///
/// The atomicity claim is therefore about the VALIDATION pass, not about the whole function: the file is read,
/// walked and digest-checked before any `Apply` runs, so every pre-apply refusal provably touches nothing.  Once
/// the apply pass has begun, only the final sync can prove anything.
///
/// On success `ids`/`imgs`/`cvec`/`L` are the stored prefix; the caller sets the live session from them so the
/// existing `starts_with` resume path takes over.
strata::core::ConversationRestore nvme_restore(const char* path, strata::core::SessionState& ss,
                                               strata::core::QsaState& mtp_state,
                                               const strata::core::ModelGeometry& g, std::vector<int32_t>& ids,
                                               std::vector<strata::core::ConversationImageKey>& imgs, bool& cvec,
                                               int64_t& L, std::string& err);

/// One stored session: its snapshot file and the token prefix AND pictures it was keyed by (both read at scan
/// time, so the resume match never trusts a filename - and never compares a request's pictures against an entry
/// that did not read its own).
struct NvmeEntry {
    std::string path;
    std::vector<int32_t> ids;                                ///< the consumed tokens (the key)
    std::vector<strata::core::ConversationImageKey> imgs;    ///< the images among them, below the key's length
    int64_t L = 0;                                           ///< ids.size(), kept for the match loops
    bool cvec = false;                                       ///< the control-vector state it was read with
    int64_t mtime = 0;                                       ///< seconds, for the LRU cap
    uint64_t bytes = 0;                                      ///< file size, for the cap
};

/// The NVMe cold tier: a directory of whole-session snapshots with automatic dump (on DONE, synchronous,
/// idempotent), promote (the serve loop matches entries by exact token prefix) and an LRU byte cap.
/// THE RESUME MATCH, stated once (the serve loop's promote rule and the host fixture's oracle).
///
/// An entry is a candidate when: it was stored with this control-vector state; it is LONGER than what the session
/// already holds (`resume`) and SHORTER than the request - the last prompt token always starts the next verify
/// window, so an entry as long as the request cannot be resumed from; its ids start the request; and the request's
/// pictures BELOW the entry's length are exactly the entry's pictures.  The longest such entry wins, not the last
/// one scanned.
///
/// The image rule is the same filter `checkpoint_at` uses when it keys a checkpoint, and it is what makes a
/// turn-boundary snapshot reachable: a snapshot that stored a picture at or past its own `L` matches no request,
/// because no request can present a picture its token prefix does not contain.
///
/// Templated on the token type as the shared core's `conversation_prefix` is, because the serve loop's prompt is
/// `int64_t` and a snapshot's ids are `int32_t`.  It lives here rather than inline in the serve loop so that the
/// promote decision - not just the file a promote reads - is testable without a GPU
/// (`src/platform/kv_nvme_host_test.cpp`).
template <class Token>
inline const NvmeEntry* kv_nvme_match(const std::vector<NvmeEntry>& entries, const std::vector<Token>& ids,
                                      const std::vector<strata::core::ConversationImageKey>& req_imgs, bool cvec,
                                      int64_t resume) {
    const int64_t n = (int64_t) ids.size();
    const NvmeEntry* best = nullptr;
    for (const NvmeEntry& e : entries) {
        const int64_t EL = e.L;
        if (e.cvec != cvec || EL <= resume || EL < 1 || EL > n - 1) continue;
        if (EL > (int64_t) e.ids.size() ||
            !std::equal(e.ids.begin(), e.ids.begin() + EL, ids.begin())) continue;
        std::vector<strata::core::ConversationImageKey> below;   // the request's pictures below the ENTRY's length
        for (const strata::core::ConversationImageKey& k : req_imgs) if (k.start < EL) below.push_back(k);
        if (!(below == e.imgs)) continue;
        if (best == nullptr || EL > best->L) best = &e;   // the LONGEST prefix wins, not the last scanned
    }
    return best;
}

class KvNvmeStore {
public:
    /// Creates `dir` if needed and scans the snapshots already in it, dropping any whose geometry/format tag
    /// does not match the live engine (they are left on disk, never converted).
    bool open(const std::string& dir, const strata::core::ModelGeometry& g, int kv_format, std::string& err);
    void set_cap_bytes(int64_t bytes) { cap_ = bytes; }      ///< 0 = unlimited (the default)
    /// Dumps the live session.  Idempotent: an exact match is skipped (its recency is refreshed); the previous
    /// dump of the same growing conversation is superseded (its file deleted) so a conversation stays one file.
    /// `at` takes the snapshot at a turn boundary (see nvme_dump_at): its `ids` are the key, its `imgs` the
    /// pictures below that boundary, and its blobs the running state there.  Without it the snapshot is the full
    /// consumed state at DONE, and `imgs` are the pictures below the whole consumed prefix.
    bool dump(const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
              const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
              const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec,
              const strata::core::ConversationCheckpoint* at = nullptr, std::string& err = dummy_err());
    /// Reads `e` back into the live arena (see nvme_restore for the failure contract).  The caller then sets
    /// `live` from `e`.
    /// The one failure this class adds is a TOCTOU: the file disagreed with its own digest and applied cleanly,
    /// but its header no longer matches the entry the scan built.  That is `invalid`, not `transfer_failed`, on
    /// the tier's own evidence - the restore's final `cudaDeviceSynchronize()` succeeded, which is the proof a
    /// recovery needs that the device answered after the last write - and the session now holds a complete,
    /// self-consistent snapshot rather than a half-applied one.
    strata::core::ConversationRestore restore(const NvmeEntry& e, strata::core::SessionState& ss,
                                              strata::core::QsaState& mtp_state,
                                              const strata::core::ModelGeometry& g, std::string& err);
    /// Forgets an entry (its file is deleted): a snapshot the tier refused is not worth keeping.  A transfer
    /// failure is NOT a reason to drop one - it says nothing about the file - and the caller must not reach this
    /// on that path anyway, because the process is stopping.
    void drop(const NvmeEntry& e);
    const std::vector<NvmeEntry>& entries() const { return entries_; }
    uint64_t total_bytes() const { return total_; }
    size_t size() const { return entries_.size(); }

private:
    static std::string& dummy_err() { static std::string s; return s; }
    void enforce_cap();                                      ///< evict by oldest mtime until under the cap
    std::string dir_;
    std::vector<NvmeEntry> entries_;
    int64_t cap_ = 0;
    uint64_t total_ = 0;
    std::vector<int32_t> last_ids_;                          ///< this process's previous dump (the supersede hint)
    std::string last_path_;
    int fmt_ = 0;
    long seq_ = 0;
};

}  // namespace strata::platform
