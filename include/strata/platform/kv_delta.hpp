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
#include <string>
#include <vector>

#include "strata/core/conversation_snapshot.hpp"  // ConversationStateSizes
#include "strata/core/layer.hpp"                  // QsaState
#include "strata/core/session.hpp"                // SessionState
#include "strata/core/weights.hpp"                // ModelGeometry
#include "strata/platform/kv_nvme.hpp"            // nvme_fnv1a / kNvmeFnvBasis / the KV host-array walk

namespace strata::platform {

// the shared core's types, named the way kv_nvme.hpp names them
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
    int64_t geometry[18] = {};     ///< conversation_geometry_key(g), verbatim (the same 18 fields NvmeHeader holds)
    uint64_t weights_fp = 0;       ///< the weight-set fingerprint (§5.8); match-time, like cvec
    uint64_t state_key = 0;        ///< the head State record's key
    int64_t pid = 0, seq = 0;      ///< the file name's numbers, for debugging a store by eye
    int64_t reserved = 0;
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
                  offsetof(DeltaManifestHeader, seq) == 248 && offsetof(DeltaManifestHeader, reserved) == 256,
              "the delta manifest header gained padding or reordered a field");

/// The runtime shapes the delta tier cuts on (§5.3).  A chunk must not straddle a KV page or an indexer block,
/// so BLOCK = lcm(page_size, idx_block), computed once from the SAME `qsa_real_shapes()` the v3 walk reads -
/// never a tuning constant - and recorded in every manifest so an engine whose shapes changed refuses old
/// manifests instead of re-deriving them wrong.
struct DeltaShapes {
    strata::kernels::QsaShapes shapes;  // page_size / idx_block (and the rest, for qsa_pooled_rows)
    int64_t block = 0;                  ///< BLOCK: tokens per sealed chunk
    int64_t rows_per_chunk = 0;         ///< sealed pooled rows per chunk = BLOCK / idx_block
};

DeltaShapes delta_shapes();

/// The token length covered by sealed chunks: the largest multiple of BLOCK not exceeding T.  Everything past
/// it is the ragged tail, and the ragged tail lives in the State record - only sealed, block-aligned chunks are
/// content-addressed, because a partial chunk's content changes every turn and re-writing an "immutable" file
/// is the contradiction the sealed/tail split exists to prevent (§5.4).
inline int64_t delta_sealed(int64_t t, const DeltaShapes& sh) { return (t / sh.block) * sh.block; }

/// The payload bytes one sealed chunk covering [a, a+BLOCK) holds, given the live arrays' formats and the
/// drafter's residency - i.e. the exact cut of the v3 segments (§5.4): per QSA layer, every KV host array's
/// pages [a/page, (a+BLOCK)/page) then rows_per_chunk pooled rows; then the drafter's pages for the same range
/// while `a < max_cells` (the delta path's T <= max_cells gate keeps every sealed chunk inside the ring).
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

}  // namespace strata::platform
