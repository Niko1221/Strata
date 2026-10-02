#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <array>
#include <string>
#include <vector>

namespace strata::core {

// Each stage is drained on its own device before its running state or K/V is read (saving) or written
// (restoring); a copy on one card cannot wait for kernels still running on another.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);

// What a save of one layer up to `upto` produces: its K/V geometry and the byte size of each of the five buffers
// (K, V, K scales, V scales, pooled indexer rows). A disk image is checked against it before anything is written.
struct ConversationKvExtent {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<uint64_t, 5> sizes{};
    bool operator==(const ConversationKvExtent&) const = default;
};
bool conversation_kv_extent(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                            ConversationKvExtent& extent, std::string& error);
// The leading bytes of each buffer a save with `unchanged_tokens` keeps instead of copying (conversation_kv_save's rule).
std::array<uint64_t, 5> conversation_kv_kept(const ConversationKvExtent& extent, const ModelGeometry& g,
                                             int64_t unchanged_tokens, bool include_index);
// The five authoritative buffers in extent order: the host pool for streamed and ring layers, VRAM otherwise.
std::array<void*, 5> conversation_kv_buffers(const QsaState& state);
// After the authoritative buffers were written: refill the VRAM slots or the draft ring from them.
bool conversation_kv_restored(const QsaState& state, const ModelGeometry& g, int64_t upto, std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
// One stage of a layer split: its session's carve plus the device its running state lives on.
// Stage 0 is first; `dev` -1 means the caller's current device, so single-GPU parking needs no scope.
// Checkpoints store one part per stage, and the flat K/V vectors run in this same stage order with the
// drafter last — an entry is pinned to its stage by position, so host storage stays device-agnostic.
struct ConversationStage {
    SessionState* ss;   // read through this in validation, written through it in restore
    int dev = -1;
};
using ConversationStages = std::vector<ConversationStage>;
bool conversation_snapshot_bytes(const ConversationView& view, const ConversationStages& stages,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const ConversationStages& stages, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const ConversationStages& stages, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const ConversationStages& stages,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
// conversation_snapshot_validate without the K/V payloads: geometry, carve, live parts and checkpoints.
bool conversation_snapshot_validate_state(const SavedConversation& image, const ConversationStages& stages,
                                          const ModelGeometry& g, std::string& error);
// The K/V entries in a snapshot's flat order (each stage's owned layers, then the drafter), with their device.
struct ConversationKvEntry {
    const QsaState* state;
    int dev;
    bool index;   // the drafter's attention has no indexer
};
std::vector<ConversationKvEntry> conversation_kv_entries(const ConversationStages& stages, const QsaState& draft);
// The live running state of every stage (image.live and its stage parts) without K/V or checkpoints.
bool conversation_live_save(SavedConversation& image, const ConversationView& view, const ConversationStages& stages,
                            const ModelGeometry& g, std::string& error);
bool conversation_live_restore(const ConversationCheckpoint& live, const ConversationStages& stages,
                               const ModelGeometry& g, std::string& error);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, const ConversationStages& stages,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);

} // namespace strata::core
