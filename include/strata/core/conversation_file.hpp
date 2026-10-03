// On-disk persistence of a conversation's state (SavedConversation): one file holding the running state, the
// checkpoints the caller passes (the engine passes only the deepest one), every QSA layer's authoritative K/V and
// the draft layer's K/V.  Pure host code: no CUDA.  Format v1 and the fail-closed read order are in
// docs/DETAILS.md (Session files).
//
// Format v1 is little-endian, with fixed-width integers and IEEE-754 floats; a big-endian build is refused at
// compile time.  The hashes detect accidental corruption; they do not authenticate a file: restore trusted files only.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// What the file is bound to: the model files it was computed with, and the engine settings that change what the
// saved bytes mean.  A read with any other identity is refused before the payload is parsed.
struct SessionFileIdentity {
    uint64_t model = 0, config = 0;
};

// Non-cryptographic 64-bit hash with xxHash64-style rounds (NOT the standard XXH64 stream): corruption and identity
// checks, not adversarial input.  Streaming and one-shot agree.
class SessionHasher {
public:
    explicit SessionHasher(uint64_t seed = 0);
    void update(const void* data, size_t n);
    uint64_t digest() const;
private:
    void block(const uint8_t* p);
    uint64_t lane_[4];
    uint8_t pending_[32];
    size_t npending_ = 0;
    uint64_t total_ = 0;
};
uint64_t session_hash64(const void* data, size_t n, uint64_t seed);

// Typed, length-delimited fields for an identity hash: each field is (tag, type, length, bytes), so two different
// field lists never give the same byte stream, and a double is hashed by its exact bits (no decimal rounding).
class SessionIdentityBuilder {
public:
    explicit SessionIdentityBuilder(uint64_t seed) : hash_(seed) {}
    void str(const std::string& tag, const std::string& value);
    void i64(const std::string& tag, int64_t value);
    void u64(const std::string& tag, uint64_t value);
    void f64(const std::string& tag, double value);
    void bytes(const std::string& tag, const void* data, size_t n);
    uint64_t digest() const { return hash_.digest(); }
private:
    void field(const std::string& tag, uint8_t type, const void* data, uint64_t n);
    SessionHasher hash_;
};

// The RESOLVED rope configuration (rope_scaling.hpp, after the CLI flags and the model file's keys met).  The cached
// K is post-RoPE, so a different rotation makes the saved K mean something else.
struct SessionRope {
    int64_t type = 0;
    double freq_base = 0, factor = 0, freq_scale = 0, orig_ctx = 0, ext_factor = 0, attn_factor = 0, beta_fast = 0,
           beta_slow = 0;
};

// Every engine setting that changes what the saved state means.  Sampling, seeds, draft tuning, the expert tier and
// the prompt-reading chunk are not here: they change what is computed next, not what the saved cells hold.
struct SessionConfig {
    std::string engine_version;     // a file is bound to one engine version
    std::string backend;            // "cuda" or "hip": no cross-backend restore
    std::string kv;                 // --kv
    int64_t max_context = 0, kv_resident = 0, mtp_window = 0;
    bool kv_rot = false;            // STRATA_KV_ROT
    SessionRope rope;
    // the control vectors as loaded: a digest of the tables the engine uploaded (every file's content x its exact
    // scale, summed), the mode, the layer range and the direction; 0 when none is loaded
    uint64_t cvec = 0;
    // the arithmetic switches that change the computed state (name, value), in a fixed order
    std::vector<std::pair<std::string, int64_t>> switches;
};
uint64_t session_config_fingerprint(const SessionConfig& config);

// One model input the engine loaded and the role it plays ("native shard 1", "pack dense.bin", ...).  The role, not
// the path, enters the fingerprint, so a moved model folder still matches.  `optional`: a missing file is recorded
// as absent instead of failing (a pack without experts.bin).
struct SessionModelFile {
    std::string role, path;
    bool optional = false;
};
// Sampled fingerprint of the model inputs: role, size, first and last MiB of each file, in the given order.  Paths
// are UTF-8 (wide APIs on Windows; invalid UTF-8 is refused).  An edit in the middle of a file that keeps its size is
// NOT detected: model files must not change while sessions saved with them are kept.
bool session_model_fingerprint(const std::vector<SessionModelFile>& files, uint64_t& fingerprint, std::string& error);

// Optional knobs of a write.
struct SessionWriteOptions {
    uint64_t min_free_bytes = 0;    // refuse when the disk would keep less than this free after the file
    bool durable = true;            // flush the file before the rename and the directory after it
    std::function<void(uint64_t done, uint64_t total)> progress;   // about every 256 MiB
};

// Writes a temporary file beside `path` (a unique name, created exclusively: never an existing file or link; mode
// 0600 on POSIX), flushes it to the disk, renames it over `path` and flushes the directory.  On failure an existing
// file at `path` is kept and only this call's own temporary file is removed.
bool session_file_write(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                        size_t& bytes, std::string& error, const SessionWriteOptions& options = {});

// One K/V layer read straight from where it lives (device or pinned host pool) into the file's staging buffer,
// without a full host copy first.  `read(part, offset, dst, n)` copies bytes [offset, offset+n) of part
// 0..4 = k, v, k_scale, v_scale, pooled.  The file bytes are identical to writing a captured ConversationKv.
struct SessionKvSource {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<size_t, 5> sizes{};
    std::function<bool(size_t part, size_t offset, void* dst, size_t n)> read;
};
// `meta` carries everything but the K/V (meta.kv must be empty); `kv` supplies the layers in order.
bool session_file_write(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                        const SessionFileIdentity& id, size_t& bytes, std::string& error,
                        const SessionWriteOptions& options = {});

// The checkpoints worth a disk write: the deepest one (the next turn's resume point when the live tail was
// rewritten).  Earlier checkpoints only serve edits further back and cost ~118 MB each at this geometry.
std::vector<ConversationCheckpoint> session_checkpoints_to_save(const std::vector<ConversationCheckpoint>& chain);

// What a read checks before it allocates.  `admit(file_bytes, why)` is asked once the header and identity match and
// before any payload allocation (the parsed image is about as large as the file); false refuses the file.
struct SessionReadLimits {
    uint64_t max_file_bytes = UINT64_MAX;
    uint64_t max_tokens = UINT64_MAX;       // token and image counts of the live state and of each checkpoint
    uint64_t max_checkpoints = UINT64_MAX;
    uint64_t max_kv_layers = UINT64_MAX;
    std::function<bool(uint64_t file_bytes, std::string& why)> admit;
    std::function<void(uint64_t done, uint64_t total)> progress;   // about every 256 MiB
};
// Full validation (a regular file with one link, size, header, identity, bounded parse, payload hash) before `image`
// is replaced.  The file is opened without following a symbolic link (or a reparse point) at its name, and the checks
// are made on the opened handle.  On failure `image` is unchanged.
bool session_file_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       size_t& bytes, std::string& error, const SessionReadLimits& limits = {});

} // namespace strata::core
