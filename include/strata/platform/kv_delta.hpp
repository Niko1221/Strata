// include/strata/platform/kv_delta.hpp - the NVMe delta tier's record family (docs/nvme-delta-cache-handoff.md §5).
//
// The delta tier stores a conversation as SEALED, CONTENT-ADDRESSED chunks (whole pages of the v3 snapshot's KV
// segments) plus one small per-turn State record (the ragged tail and the running state) plus a manifest that
// names the chunk list.  Reassembling a manifest's chunks + its State record must produce a byte stream IDENTICAL
// to what `nvme_dump_at` writes for the same session and boundary (§5.2, the whole design hangs on this), which
// is why every payload here is laid out as exact contiguous slices of the v3 segments - the chunk and State
// payloads carry no format of their own, only cuts of the v3 walk.
//
// The v3 snapshot format stays frozen (kNvmeFormatVersion = 3, NvmeHeader untouched): "v3" is the SNAPSHOT
// format, and this family is a new record family living in `delta/` under the same store directory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include <vector>

#include "strata/core/conversation_snapshot.hpp"  // ConversationStateSizes
#include "strata/core/layer.hpp"                  // QsaState
#include "strata/core/session.hpp"                // SessionState
#include "strata/core/weights.hpp"                // ModelGeometry
#include "strata/platform/kv_nvme.hpp"            // nvme_fnv1a / kNvmeFnvBasis / the KV host-array walk

namespace strata::platform {

// the shared core's types, named the way kv_nvme.hpp names them
using strata::core::ConversationCheckpoint;
using strata::core::ConversationImageKey;
using strata::core::ConversationStateSizes;
using strata::core::ModelGeometry;
using strata::core::QsaState;
using strata::core::SessionState;

// The three magics are ASCII ("CHNK" / "STAT" / "DLOG") so a hexdump of the wrong file in the store identifies
// itself; the value choice is otherwise free, and the version is the family's own - the SNAPSHOT format's
// version stays 3 and is never bumped for a delta-record change.
inline constexpr uint32_t kDeltaChunkMagic = 0x4B4E4843;     // "CHNK"
inline constexpr uint32_t kDeltaStateMagic = 0x54415453;     // "STAT"
inline constexpr uint32_t kDeltaManifestMagic = 0x474F4C44;  // "DLOG"
inline constexpr uint32_t kDeltaFormatVersion = 1;

inline constexpr uint64_t kDeltaChunkHeaderBytes = 48;
inline constexpr uint64_t kDeltaStateHeaderBytes = 16;
// The header the §5.6 field list adds up to.  (The handoff's prose says 256; its own field list - which includes
// `reserved` - sums to 264, and the FIELD LIST is the format: a reader pins offsets, not a round number.)
inline constexpr uint64_t kDeltaManifestHeaderBytes = 264;

/// A sealed chunk's on-disk header.  48 bytes, padding-free, pinned below: the file IS this struct's bytes plus
/// payload plus an 8-byte FNV-1a footer over the payload (the footer covers the payload only, never the header -
/// the header is what describes where the payload's slices go, so a corrupt header must surface as a LAYOUT
/// refusal, not an integrity one).
struct DeltaChunkHeader {
    uint32_t magic = kDeltaChunkMagic;
    uint32_t version = kDeltaFormatVersion;
    uint64_t key = 0;              ///< the content key (§5.7); also the file name - a chunk is reachable only
                                   ///  through the prefix it commits to, so stale chunks are unreachable
    int64_t a = 0;                 ///< first token position; a % BLOCK == 0 (sealed chunks start on a boundary)
    int64_t b = 0;                 ///< a + BLOCK (sealed chunks only - the ragged tail lives in the State record)
    int64_t layers = 0;            ///< n_qsa_layers, recorded so a reader can sanity-check the walk it is about
    int64_t payload_bytes = 0;     ///< everything between this header and the footer
};

/// A State record's header: one per turn, content-addressed.  The record's IDENTITY key is
/// `delta_state_key(tag, payload)` (§5.7 - it commits to the conversation's geometry AND its token prefix
/// through the tag), while the footer is the plain payload FNV-1a: integrity and identity are different
/// questions and get different checks.
struct DeltaStateHeader {
    uint32_t magic = kDeltaStateMagic;
    uint32_t version = kDeltaFormatVersion;
    int64_t payload_bytes = 0;
};

/// A manifest's header (§5.6).  One per conversation head; the body that follows is
/// `ids (int32 * L) || imgs (ConversationImageKey * n_imgs) || per chunk {u64 key; int64_t a}`, and the footer
/// is an 8-byte FNV-1a over the BODY (the ids and imgs the resume match reads live here, so they are covered).
struct DeltaManifestHeader {
    uint32_t magic = kDeltaManifestMagic;
    uint32_t version = kDeltaFormatVersion;
    int64_t L = 0;                 ///< the boundary length T - the resume key
    int64_t block = 0;             ///< BLOCK, validated against the live lcm at open/restore (never re-derived)
    int64_t n_chunks = 0;          ///< sealed(T) / BLOCK
    int64_t n_imgs = 0;
    int32_t cvec = 0;
    int32_t kv_format = 0;
    int64_t page_size = 0, idx_block = 0, max_cells = 0, mtp_host = 0;
    std::array<int64_t, 18> geometry{};  ///< conversation_geometry_key(g), verbatim (the same 18 fields NvmeHeader holds)
    uint64_t weights_fp = 0;       ///< the weight-set fingerprint (§5.8); match-time, like cvec
    uint64_t state_key = 0;        ///< the head State record's key
    int64_t pid = 0, seq = 0;      ///< the file name's numbers, for debugging a store by eye
    /// How many lcm-blocks one of this manifest's chunk refs covers.  This is the header's old `reserved`
    /// field at the same offset: existing manifests hold 0 there, which reads as 1 (one block per chunk - the
    /// pre-grouping layout), so nothing is refused and nothing converts; new manifests write
    /// kDeltaBlocksPerChunk.  A chunk ref at `a` covers [a, a + blocks_per_chunk * block).
    int64_t blocks_per_chunk = 0;
};

// The file IS these structs' bytes, so the layout is the format - the same discipline NvmeHeader's static_asserts
// pin: every field fixed-width, padding-free, in declaration order.  A field added, widened or REORDERED fails
// the build rather than silently re-mapping every record after it.
static_assert(sizeof(DeltaChunkHeader) == kDeltaChunkHeaderBytes && alignof(DeltaChunkHeader) == 8 &&
                  offsetof(DeltaChunkHeader, key) == 8 && offsetof(DeltaChunkHeader, a) == 16 &&
                  offsetof(DeltaChunkHeader, payload_bytes) == 40,
              "the delta chunk header gained padding or reordered a field");
static_assert(sizeof(DeltaStateHeader) == kDeltaStateHeaderBytes &&
                  offsetof(DeltaStateHeader, payload_bytes) == 8,
              "the delta state header gained padding or reordered a field");
static_assert(sizeof(DeltaManifestHeader) == kDeltaManifestHeaderBytes && alignof(DeltaManifestHeader) == 8 &&
                  offsetof(DeltaManifestHeader, L) == 8 && offsetof(DeltaManifestHeader, cvec) == 40 &&
                  offsetof(DeltaManifestHeader, geometry) == 80 &&
                  offsetof(DeltaManifestHeader, weights_fp) == 224 &&
                  offsetof(DeltaManifestHeader, state_key) == 232 && offsetof(DeltaManifestHeader, pid) == 240 &&
                  offsetof(DeltaManifestHeader, seq) == 248 &&
                  offsetof(DeltaManifestHeader, blocks_per_chunk) == 256,
              "the delta manifest header gained padding or reordered a field");

/// The runtime shapes the delta tier cuts on (§5.3).  A chunk must not straddle a KV page or an indexer block,
/// so BLOCK = lcm(page_size, idx_block), computed once from the SAME `qsa_real_shapes()` the v3 walk reads -
/// never a tuning constant - and recorded in every manifest so an engine whose shapes changed refuses old
/// manifests instead of re-deriving them wrong.
struct DeltaShapes {
    strata::kernels::QsaShapes shapes;  // page_size / idx_block (and the rest, for qsa_pooled_rows)
    int64_t block = 0;                  ///< BLOCK: the lcm granule (a chunk is a whole number of these)
    int64_t span = 0;                   ///< the CHUNK SPAN in tokens: kDeltaBlocksPerChunk * block
    int64_t rows_per_chunk = 0;         ///< sealed pooled rows per BLOCK = BLOCK / idx_block
};

/// How many lcm-blocks ONE sealed chunk covers.  The §5.3 rule (a chunk must not straddle a KV page nor an
/// indexer block) requires the span to be a MULTIPLE of lcm(page_size, idx_block) - it does NOT pin it to ONE
/// block, and the live store proved one-block chunks are operationally wrong: a big turn's append became
/// ~2,300 individual 61 KB files, each with its own fsync + rename (a journal op), and the dump drained for
/// tens of seconds while the client's generation sat frozen.  64 blocks = 256 tokens = the handoff's own
/// worked example (~4 MB files); the key chain is per BLOCK either way, so a chunk's key is the chain value
/// through its LAST token and the prefix property, the fork sharing and the byte-identity invariant are
/// unchanged - only the file count drops 64x.  Recorded per manifest (`blocks_per_chunk`); readers treat 0
/// (the field's old reserved value) as 1, so existing manifests keep reading, nothing converts.
inline constexpr int64_t kDeltaBlocksPerChunk = 64;

DeltaShapes delta_shapes();

/// The token length covered by sealed chunks: the largest multiple of the CHUNK SPAN (kDeltaBlocksPerChunk
/// blocks) not exceeding T.  Everything past it is the ragged tail, and the ragged tail lives in the State
/// record - only sealed, span-aligned chunks are content-addressed, because a partial chunk's content changes
/// every turn and re-writing an "immutable" file is the contradiction the sealed/tail split exists to prevent.
inline int64_t delta_sealed(int64_t t, const DeltaShapes& sh) { return (t / sh.span) * sh.span; }

/// The payload bytes one sealed chunk covering [a, a+span) holds (span = kDeltaBlocksPerChunk blocks), given
/// the live arrays' formats and the drafter's residency - i.e. the exact cut of the v3 segments (§5.4): per
/// QSA layer, every KV host array's pages [a/page, (a+span)/page) then span/idx_block pooled rows; then the
/// drafter's pages for the same range while `a < max_cells` (the delta path's T <= max_cells gate keeps every
/// sealed chunk inside the ring, so a chunk never straddles it).
/// Pure byte math, no I/O: the writer sizes its buffer with it and the reader sizes its expectation with it,
/// which is what makes a size disagreement a REFUSAL rather than a short read.
int64_t delta_chunk_payload_bytes(const SessionState& ss, const QsaState& mtp, const ModelGeometry& g,
                                  const DeltaShapes& sh, int64_t a);

/// The payload bytes the State record for boundary T holds (§5.5): gdn, ple (when the session has PLE history),
/// then per QSA layer the tail KV pages [sealed(T)/page, ceil(T/page)), the tail pooled rows
/// [sealed(T)/idx_block, qsa_pooled_rows(T)) (the in-progress block and the spare row included), the
/// checkpoint's tail/dead/block_pos, then the drafter's tail pages.  Pure byte math, no I/O.
int64_t delta_state_payload_bytes(const SessionState& ss, const QsaState& mtp, const ModelGeometry& g,
                                  const ConversationStateSizes& z, const DeltaShapes& sh, int64_t t);

/// 16 lowercase hex digits - the on-disk name of a content-addressed record (key or state digest).
std::string delta_key_name(uint64_t key);

/// Write one sealed chunk to `path` (header + payload + footer).  Crash-safe by construction: the bytes land in
/// `<path>.tmp`, are fsynced, and are renamed into place, so a chunk file under its content-addressed name is
/// never partial - a half-written chunk can only exist under a `.tmp` name the scan ignores (§5.15).
/// `h.payload_bytes` must equal the payload's size; the write refuses rather than fixing it, because a header
/// that disagrees with its own payload is a caller bug the digest would only hide.
bool delta_write_chunk(const std::string& path, const DeltaChunkHeader& h, const void* payload,
                       size_t payload_bytes, std::string& err);

/// Read one sealed chunk back, verifying EVERYTHING a reader can check before trusting a byte of payload:
/// magic, version, the expected key/range (the caller's manifest says which chunk belongs where), the file size
/// against the header's own payload_bytes, and only then the footer digest.  Layout and size facts are
/// reported BEFORE integrity verdicts - a layout-drifted file must not masquerade as a corrupt one.
/// On success `payload` holds exactly `h.payload_bytes` bytes.
bool delta_read_chunk(const std::string& path, uint64_t expected_key, int64_t expected_a, int64_t expected_b,
                      std::vector<uint8_t>& payload, std::string& err);

/// A State record's identity key: FNV-1a of the payload seeded with the conversation's tag (§5.7) - not a bare
/// payload digest, so a state record is reachable only by the conversation (and prefix) it belongs to.
inline uint64_t delta_state_key(uint64_t tag, const void* payload, size_t n) {
    return nvme_fnv1a(tag, payload, n);
}

/// Write one State record into `dir` under its content key.  Same temp+fsync+rename discipline as chunks.
/// Returns the key (also the file name, and the manifest's `state_key`).
bool delta_write_state(const std::string& dir, uint64_t tag, const void* payload, size_t payload_bytes,
                       uint64_t& key_out, std::string& err);

/// Read one State record back, verifying magic, version, file size, the footer digest and - when `tag` and
/// `expected_key` are both non-zero - that the payload really hashes to the key the manifest named (the identity
/// check the footer cannot do, because the footer is the bare payload hash).
bool delta_read_state(const std::string& path, uint64_t tag, uint64_t expected_key, std::vector<uint8_t>& payload,
                      std::string& err);

/// The conversation tag the whole key space hangs on (§5.7): FNV-1a over the geometry key's bytes, the KV
/// format, the control-vector state, the weight-set fingerprint and BLOCK.  EVERY derived key goes through it,
/// so a conversation stored with a different geometry, format, cvec or weight set simply never shares a chunk -
/// and a mid-conversation cvec toggle behaves as a fork, which is the correct reading of it.
uint64_t delta_tag(const ModelGeometry& g, int kv_format, bool cvec, uint64_t weights_fp, int64_t block);

/// The chain value after hashing `blocks` BLOCK-sized id runs: c_j = FNV1a(c_{j-1}, ids[(j-1)*BLOCK, j*BLOCK)).
/// A chunk's key is the chain value THROUGH ITS OWN LAST TOKEN, so it is reachable only by prompts sharing that
/// prefix - forks share chunks for free, and stale chunks are unreachable (64-bit keys: a birthday collision at
/// 10^6 chunks is ~1e-8, and one is caught by the footer check - a refused restore, never corruption).
uint64_t delta_chunk_key(uint64_t tag, const int32_t* ids, int64_t blocks);

/// One manifest body's chunk reference: the sealed chunk covering [a, a + blocks_per_chunk*block) - `b` is
/// not stored, it comes from the header's span (16 bytes on disk, the layout §5.6 pins).
struct DeltaChunkRef {
    uint64_t key = 0;
    int64_t a = 0;
};

/// Write a manifest (header + ids + imgs + chunk refs + a footer over the body).  Same temp+fsync+rename
/// discipline: a manifest under its real name is complete and digest-checked, which is what lets the writer's
/// commit point be the RENAME (§5.9 step 6).
bool delta_write_manifest(const std::string& path, const DeltaManifestHeader& h, const std::vector<int32_t>& ids,
                          const std::vector<ConversationImageKey>& imgs, const std::vector<DeltaChunkRef>& chunks,
                          std::string& err);

/// Read a manifest back, verifying magic, version, the header's own counts against the file size, that
/// `n_chunks` matches `L / (blocks_per_chunk * block)` (a manifest whose chunk count disagrees with its own
/// boundary is not scanned at all), and only then the footer over the body.
bool delta_read_manifest(const std::string& path, DeltaManifestHeader& h, std::vector<int32_t>& ids,
                         std::vector<ConversationImageKey>& imgs, std::vector<DeltaChunkRef>& chunks,
                         std::string& err);

/// The previous head a delta dump appends to: the manifest this process wrote last for this conversation, whose
/// ids must be a strict (or equal) prefix of the new ones.  `path` is what the writer unlinks AFTER the new
/// manifest is durable - the supersede (§5.9 step 6, §5.12).
struct DeltaHead {
    int64_t L = 0;                    ///< the previous head's boundary length (0 = none)
    std::vector<int32_t> ids;         ///< its ids, for the strict-prefix check
    std::string path;                 ///< its manifest file
};

/// THE WRITER (§5.9): at a turn boundary, append the sealed chunks the previous head does not already cover,
/// write the ragged tail + running state as one State record, then move the manifest head.
///
/// `dir` is the delta directory (containing chunks/, states/, the manifests).  `at` is REQUIRED - the delta tier
/// writes turn boundaries only; a dump without a boundary checkpoint is the v3 path's job.  Everything the v3
/// dump refuses, this refuses too (kv_mode 0, a split session, images outside the prefix, a checkpoint that
/// does not fit, a pooled array too small) - and `T > mtp_state.max_cells` refuses with the whole-snapshot
/// fallback message, because the drafter ring wraps and wrap-aware chunking is out of scope (§5.15).
/// `weights_fp` is the weight-set fingerprint (§5.8) recorded in the manifest; `pid`/`seq` name the manifest
/// file, exactly as the v3 store names its snapshots.
///
/// On failure: no manifest is written and no head moves; the only residue is content-addressed files the sweep
/// reclaims - a dump-side failure is not a correctness event (§5.13).  `STRATA_DELTA_FAIL_AT=C1..C5` aborts at
/// a named step of the write protocol (the crash matrix's C1..C6, minus the in-memory C6 which is the store's);
/// debug-only, documented like [STRATA_TEST_FAIL_CUDA].
bool delta_dump_at(const DeltaHead* prev, const std::string& dir, const SessionState& ss, const QsaState& mtp_state,
                   const ModelGeometry& g, const std::vector<int32_t>& ids,
                   const std::vector<ConversationImageKey>& imgs, bool cvec, const ConversationCheckpoint* at,
                   uint64_t weights_fp, int64_t pid, int64_t seq, std::string& err);

/// THE READER (§5.10): validate EVERYTHING with no CUDA call, assemble the exact v3 image, then run the
/// existing validation+apply pass (`nvme_restore_image`) on it.  Every manifest/chunk/state problem is
/// `invalid` - recoverable, provably before any CUDA call (zero copies, sentinel buffers untouched); only the
/// apply pass inside nvme_restore_image can report `transfer_failed`, with its unchanged meaning: fatal.
///
/// `e` is a delta entry (`kind == 1`); `e.path` is the manifest, and the chunks/states hang off the manifest's
/// own directory.  `weights_fp` is the live process's fingerprint - a mismatch is the match-time refusal (§5.8)
/// re-checked here as a backstop behind the scan's filtering.  On success the caller sets the live session from
/// the restored prefix, exactly as it does after a v3 restore.
strata::core::ConversationRestore delta_restore(const NvmeEntry& e, strata::core::SessionState& ss,
                                                strata::core::QsaState& mtp_state, const ModelGeometry& g,
                                                uint64_t weights_fp, std::vector<int32_t>& ids,
                                                std::vector<ConversationImageKey>& imgs, bool& cvec, int64_t& L,
                                                std::string& err);

/// THE WEIGHT-SET FINGERPRINT (§5.8): FNV-1a over each model file's resolved path bytes, its size (int64), and
/// its first and last 64 KiB - two sequential reads per file at startup, no false positives across
/// same-geometry-different-weights models.  Recorded in every manifest and checked at MATCH time (a delta entry
/// whose fingerprint differs is not a candidate - the same treatment as cvec), never per chunk.  `files` is the
/// resolved model shard list; a single-file model matches the §5.8 formula exactly.
uint64_t kv_delta_weights_fp(const std::vector<std::string>& model_files);

/// The delta tier's automatic layer: a `delta/` directory beside the v3 store's snapshots, scanned at startup
/// into the SAME NvmeEntry vocabulary (kind = 1), appended to at every DONE by the §5.9 writer, promoted through
/// the same kv_nvme_match, and garbage-collected by mark-and-sweep (§5.12) - no refcounts on the write path.
class KvDeltaStore {
public:
    /// Creates `v3_dir + "/delta"` and its subdirectories on demand; a missing delta dir is not an error (the
    /// tier simply starts empty).  Scans the manifests, skipping (never converting) any whose geometry, format
    /// or weight-set fingerprint does not match the live engine.  `model_files` is the resolved shard list for
    /// the §5.8 fingerprint, computed ONCE per process here.
    bool open(const std::string& v3_dir, const ModelGeometry& g, int kv_format,
              const std::vector<std::string>& model_files, std::string& err);
    /// Dumps the live session at a turn boundary (§5.9).  Idempotent (an exact match refreshes recency); the
    /// previous head of THIS process is superseded (its manifest unlinked after the new one is durable).  The
    /// caller pre-checks the delta path's own conditions (not split, T <= mtp max_cells) and routes everything
    /// else to the v3 store; a false return here means NOT CACHED THIS TURN (§5.13), never a correctness event.
    /// `act`, when given, reports what THIS call did (docs/nvme-kv-cache-web-design.md §3): `skipped` on the
    /// exact-match head refresh, `written` = the same turn's write volume the `appended N chunks` line prints
    /// (manifest + State + only the chunks THIS turn sealed), and `dropped`/`dropped_bytes` ONLY on a real
    /// supersede - a fork left the previous head on disk, so it reports no drop it did not make.
    bool dump(const SessionState& ss, const QsaState& mtp_state, const ModelGeometry& g,
              const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec,
              const ConversationCheckpoint* at, std::string& err, TierActivity* act = nullptr);
    /// Reads `e` back (see delta_restore for the failure contract), plus the store's own TOCTOU check: the
    /// reassembled image disagreeing with the entry the scan built is `invalid` - same rule as the v3 store's.
    strata::core::ConversationRestore restore(const NvmeEntry& e, SessionState& ss, QsaState& mtp_state,
                                              const ModelGeometry& g, std::string& err);
    /// Forgets an entry: unlinks its manifest (its exclusive chunks are reclaimed by the next sweep).  A
    /// transfer failure is NOT a reason to drop one - the caller stops the engine on that path anyway.
    void drop(const NvmeEntry& e);
    /// Mark-and-sweep (§5.12): unlink every chunks/states file no live manifest references.  Run at open and at
    /// cap pressure, after eviction - the union of references is computed BEFORE any unlink, so a chunk shared
    /// with a live manifest is never reclaimed.  Returns what it reclaimed (and still prints the operator's line).
    TierActivity sweep();
    const std::vector<NvmeEntry>& entries() const { return entries_; }
    uint64_t total_bytes() const { return total_; }   ///< chunks + states + manifests, this tier
    size_t size() const { return entries_.size(); }

private:
    std::string dir_;                                  ///< the delta directory (the v3 store's + "/delta")
    std::vector<NvmeEntry> entries_;
    std::vector<std::vector<uint64_t>> entry_chunks_;  ///< parallel: each entry's chunk keys (the sweep's marks)
    std::vector<uint64_t> entry_states_;               ///< parallel: each entry's State key
    uint64_t weights_fp_ = 0;
    uint64_t total_ = 0;
    std::vector<int32_t> last_ids_;                    ///< this process's previous dump (the supersede hint)
    std::string last_path_;
    long seq_base_ = 0;                                ///< this instance's manifest-number block: two opens in
    long seq_ = 0;                                     ///< ONE process (a test, a restart-less re-open) must not
                                                       ///< overwrite another's manifest by renaming over its name
};

/// ONE byte cap across BOTH tiers (§5.12): evict by oldest mtime across the combined entry lists until the
/// summed bytes fit (never emptying the store), then sweep the delta tier once - eviction of a delta
/// conversation is just its manifest's unlink, and the sweep is what reclaims its exclusive chunks.  A free
/// function so the serve loop's rule is host-testable; with the delta tier off it is simply never called.
/// Returns the combined activity: the evictions from BOTH tiers' entry lists plus the sweep's numbers.
TierActivity kv_delta_enforce_cap(KvNvmeStore& v3, KvDeltaStore& delta, int64_t cap_bytes);

}  // namespace strata::platform
