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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/conversation_cache.hpp"  // ConversationImageKey, ConversationCheckpoint
#include "strata/core/layer.hpp"   // QsaState
#include "strata/core/session.hpp" // SessionState
#include "strata/core/weights.hpp" // ModelGeometry

namespace strata::platform {

/// On-disk header.  Carries the geometry + format tag so a restore refuses (never converts) a mismatch.
///
/// THIS IS A RAW C++ STRUCT COPIED INTO THE ENVELOPE, which is exactly what the shared core's boundary forbids for
/// a disk adapter (no C++ structs, no native layout, an implicit ABI).  It stays that way in this step on purpose
/// - the format is not changed here.  See docs/nvme-kv-cache-convergence.md ("Where step 2 leaves NvmeHeader")
/// for the full list of violations and for the shared core's own key (`SavedConversation::geometry`), which is
/// what this tag has to become in step 3.  No second geometry key is introduced here.
struct NvmeHeader {
    uint32_t magic = 0x5E564D45;   // "^VME"
    uint32_t version = 2;          // v2 adds `mtp_host` (the drafter arrays written)
    int64_t L = 0;                 // ids consumed (the prefix length)
    int64_t n_imgs = 0;
    int32_t cvec = 0;
    int32_t kv_format = 0;         // strata::kernels::KvFormat (kKvF16 / kKvInt8 / kKvQ4)
    // geometry tag (restore must match the live engine exactly): a DERIVED PROJECTION of the model geometry -
    // n_qsa / n_gdn come from `n_layers` + `qsa_interval`, `idx_dim` is `idx_key_dim`, and page_size / idx_block
    // are `qsa_real_shapes()` granules rather than geometry at all.  The shared core keys on all 18 geometry
    // fields instead; reconciling the two is step 3, not step 2.
    int64_t n_qsa = 0, n_gdn = 0, n_head_kv = 0, head_dim = 0, idx_dim = 0, page_size = 0, idx_block = 0,
            max_cells = 0;
    int64_t mtp_host = 0;          // how many drafter host-KV arrays the file holds (0 if the drafter is resident)
};

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
/// `at` is the SHARED core's checkpoint type (docs/nvme-kv-cache-convergence.md step 2): its `ids` are the
/// boundary, and its blobs are validated against `conversation_state_sizes` before a byte of them is written.
/// `imgs` are the images BELOW that boundary - the checkpoint's own list, not the live conversation's pictures.
/// An image at or past `L` describes a token the snapshot does not hold, and the dump refuses one: the resume
/// match compares the next request's images below `L` against this segment.
/// Its `dead` / `block_pos` blobs are deliberately NOT written: the file still carries the LIVE device arrays,
/// which is what our v2 envelope has always held.  Which of the two the disk format owns is collision C5 - step 3.
bool nvme_dump_at(const char* path, const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
                  const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
                  const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec,
                  const strata::core::ConversationCheckpoint* at, std::string& err);

/// The full consumed state at DONE (the Step 0 spike's dump; the automatic store uses nvme_dump_at at the
/// turn boundary - see the comment above).
bool nvme_dump(const char* path, const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
               const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
               const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec, std::string& err);

/// Read a snapshot back into `ss` (and the drafter's state).  Atomic: the whole file is read and validated
/// before anything is applied, so a truncated snapshot fails without touching the session.  On success
/// `ids`/`imgs`/`cvec`/`L` are the stored prefix; the caller sets the live session from them so the existing
/// `starts_with` resume path takes over.
bool nvme_restore(const char* path, strata::core::SessionState& ss, strata::core::QsaState& mtp_state,
                  const strata::core::ModelGeometry& g, std::vector<int32_t>& ids,
                  std::vector<strata::core::ConversationImageKey>& imgs, bool& cvec, int64_t& L, std::string& err);

/// One stored session: its snapshot file and the token prefix it was keyed by (read at scan time, so the
/// resume match never trusts a filename).
struct NvmeEntry {
    std::string path;
    std::vector<int32_t> ids;                                ///< the consumed tokens (the key)
    std::vector<strata::core::ConversationImageKey> imgs;    ///< the images among them
    int64_t L = 0;                                           ///< ids.size(), kept for the match loops
    bool cvec = false;                                       ///< the control-vector state it was read with
    int64_t mtime = 0;                                       ///< seconds, for the LRU cap
    uint64_t bytes = 0;                                      ///< file size, for the cap
};

/// The NVMe cold tier: a directory of whole-session snapshots with automatic dump (on DONE, synchronous,
/// idempotent), promote (the serve loop matches entries by exact token prefix) and an LRU byte cap.
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
    /// Reads `e` back into the live arena (see nvme_restore).  The caller then sets `live` from `e`.
    bool restore(const NvmeEntry& e, strata::core::SessionState& ss, strata::core::QsaState& mtp_state,
                 const strata::core::ModelGeometry& g, std::string& err);
    /// Forgets an entry (its file is deleted): a snapshot that failed to restore is not worth keeping.
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
