// The SHARED CONVERSATION-CACHE CORE's capture/restore primitives (issue #57).  Imported verbatim by the NVMe
// convergence branch (docs/nvme-kv-cache-design.md, step 2): these are the primitives our NVMe tier is now
// built on top of, and the only home for the running-state byte counts (`ConversationStateSizes`).
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <array>
#include <string>

namespace strata::core {

/// THE ONE GEOMETRY IDENTITY the two tiers share (docs/nvme-kv-cache-design.md, C9 and step 3): the 18 raw
/// `ModelGeometry` fields, in this order.  The RAM core stores it in `SavedConversation::geometry` and refuses a
/// mismatch; the NVMe adapter writes it into its header and refuses a mismatch.  Neither keeps its own derived
/// projection any more.  It is model identity, not a runtime shape: the shapes a disk reader also needs
/// (`page_size`, `idx_block`, `max_cells`) are recorded beside it, not folded into it.
std::array<int64_t, 18> conversation_geometry_key(const ModelGeometry& g);

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
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
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);

} // namespace strata::core
