// engine/metal/src/hy4flash_gguf/prefill_snap.h - save the state a prompt leaves behind, so a decode measurement can skip the
// prompt.
//
// WHY: at a long context the prefill dominates a run's wall time while the decode is the part being measured,
// and the prompt does not change between two A/B runs.  The state a decode window actually reads is the MLA
// latent cache, the indexer cache, how many pooled keys each sparse block has, and the last token - about
// 0.5 GB at 8K, which is quick to write and read back.
//
// WHAT MAKES IT SAFE: a snapshot is only valid for the exact model, prompt, context and geometry it was taken
// with, and `tmp/scratch` can be deleted or half-written at any time.  So every one of those is recorded and
// checked on load, and the payload carries a SHA-256 of itself: a wrong, stale or truncated snapshot is
// REFUSED with a message naming the field that differs, never silently used.  A decode that continues from the
// wrong KV cache produces plausible tokens, which is the worst possible failure - it would look like a
// regression in whatever was being A/B'd.
//
// WHAT IT DOES NOT PRESERVE: the expert arena's contents.  After a real prefill the arena holds what the
// prompt touched; after a load it holds what the profile prefilled.  Both are legitimate decode states and the
// TOKENS are identical either way (the arena is a cache), but the two can differ in SPEED, which is exactly
// what a snapshot-based measurement reports.  tmp/tests/hy4_flash/engine/prefill_snap.sh measures that difference
// rather than assuming it away.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash {

/// Everything that has to match for a snapshot to describe the same state.  All of it is compared on load, and
/// all of it is inside the payload's digest - a field outside the digest could be edited with the digest still
/// matching, and `last_tok` is the FIRST token a continuation emits.
struct SnapBinding {
    /// THE WEIGHTS, by size + mtime (to the nanosecond) + a digest of the file's first and last 1 MB.  Not a
    /// digest of the whole file: hashing 8 GB costs more than the prefill this saves.  The edges are hashed
    /// because size and mtime alone accept a same-size edit with a preserved timestamp, and a GGUF's header and
    /// its last tensor are what such an edit almost always moves.  A same-size edit confined to the middle of
    /// the file is the residual risk, and it is recorded as one.
    uint64_t model_bytes = 0;
    int64_t model_mtime = 0, model_mtime_ns = 0;
    std::string model_name;        ///< basename; compared, not decoration
    std::string model_edges_sha;   ///< SHA-256 of min(1 MB, size) at each end of the model file
    /// The Engram shard and the pack's token map.  They are NOT in the model file and they decide which rows
    /// the prompt's forward pass read, so a snapshot taken with a different shard or map describes different
    /// state while every other field matches.
    uint64_t engram_bytes = 0;
    int64_t engram_mtime = 0, engram_mtime_ns = 0;
    std::string token_map_sha;     ///< SHA-256 of pack/engram_token_map.bin (0.5 MB - cheap enough to hash whole)
    /// DELIBERATE WRONGNESS, as a string of the switches that were on: HY4_FLASH_MUT (the mutation probe), the
    /// prefill probe, the MTP probe.  Each changes the prompt's arithmetic, so a snapshot taken under one
    /// must not load into a normal run - which would print no warning at all, the switches being unset there.
    std::string mode;
    std::string tokens_sha;        ///< SHA-256 of the prompt token ids, as 4-byte little-endian int32s
    int32_t n_tokens = 0;          ///< how many of them
    int32_t context = 0;           ///< --context: the cache stride, so a mismatch makes every offset wrong
    int32_t chunk = 0;             ///< the prefill chunk size (it changes the sub-batch split, not the result)
    int32_t blocks = 0, kv_w = 0, idx_blocks = 0, idx_dim = 0, vocab = 0;
    int32_t used = 0;              ///< positions in the cache = where the next window starts
    /// Rows of the MTP block's `h` history in the payload, 0 when the snapshot was taken without --mtp.  Part
    /// of the binding, not just the region list: a snapshot without them loaded into an MTP run would leave
    /// hidden(q) zeroed, and MTP would then draft from a zero hidden state - plausible tokens, wrong state.
    int32_t mtp_rows = 0;
    int32_t last_tok = -1;         ///< what the head produced for the last of them
    uint32_t bad_reported = 0;     ///< the non-finite count already reported
    /// The binding's canonical bytes, which the payload digest covers.  Exposed so save and load hash exactly
    /// the same thing, and so a test can show that changing any field changes the digest.
    std::string blob() const;
    /// "" when the two describe the same state, else the first field that differs.
    std::string differs_from(const SnapBinding& other) const;
};

/// One region to write or read.  `name` is compared on load: the region list is part of the format, so a
/// snapshot taken by an engine that saved different state is refused rather than read into the wrong buffers.
struct SnapSlice {
    std::string name;
    void* ptr = nullptr;
    uint64_t bytes = 0;
};

/// Writes the snapshot to a temporary file and renames it into place, so an interrupted save cannot leave a
/// half-written snapshot that passes the header check.  Returns "" or an error.
std::string snap_save(const std::string& path, const SnapBinding& b, const std::vector<SnapSlice>& slices);

/// Reads it back into `slices`.  Returns "" only when the binding matches `want` field for field, the region
/// names and sizes match, and the payload's SHA-256 matches the stored one; otherwise nothing is written to
/// the slices and the message says which check failed.  `got` receives what the file claimed, for the message.
std::string snap_load(const std::string& path, const SnapBinding& want, SnapBinding& got,
                      const std::vector<SnapSlice>& slices);

/// SHA-256 of `n` bytes, lowercase hex.  Self-contained: the engine has no crypto dependency, and this is used
/// for the payload digest as well as the token list, so a single flipped byte in a 0.5 GB snapshot is caught.
std::string sha256_hex(const void* p, size_t n);

}  // namespace strata::hy4_flash
