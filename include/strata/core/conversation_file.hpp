// On-disk persistence of a parked conversation (SavedConversation): one file that holds the running state,
// the checkpoint chain, every QSA layer's authoritative K/V and the draft layer's K/V.  Pure host code: no CUDA.
// Format and the fail-closed read order are in docs/DETAILS.md (Session files).
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

// What the file is bound to: the model files it was computed with, and the engine options that change what the
// saved bytes mean.  A read with any other identity is refused.
struct SessionFileIdentity {
    uint64_t model = 0, config = 0;
};

// Non-cryptographic 64-bit hash (corruption and identity, not adversarial input).  Streaming and one-shot agree.
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

// Sampled fingerprint of model files: base name, size, first and last MiB of each, in the given order.
bool session_model_fingerprint(const std::vector<std::string>& paths, uint64_t& fingerprint, std::string& error);

// Writes `<path>.tmp` then renames it over `path`; on failure no file is left at either name.
bool session_file_write(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                        size_t& bytes, std::string& error);

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
                        const SessionFileIdentity& id, size_t& bytes, std::string& error);

// The checkpoints worth a disk write: the deepest one (the next turn's resume point when the live tail was
// rewritten).  Earlier checkpoints only serve edits further back and cost ~118 MB each at this geometry.
std::vector<ConversationCheckpoint> session_checkpoints_to_save(const std::vector<ConversationCheckpoint>& chain);
// Full validation (size, header, identity, bounded parse, payload hash) before `image` is replaced.
// On failure `image` is unchanged.
bool session_file_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       size_t& bytes, std::string& error);

} // namespace strata::core
