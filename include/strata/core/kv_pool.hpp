// include/strata/core/kv_pool.hpp - one pinned host KV pool shared by every session (--kv-pool-tokens).
//
// With KV streaming (--kv-resident) every QSA layer of every session keeps the authoritative K/V of all its cells
// in pinned host memory. Sized per session at the full context, N batch slots plus the main session pin
// (N + 1) x the context's K/V although the conversations together rarely fill it. The pool pins `tokens` cells per
// QSA layer once, and a session holds only the chunks its conversation has reached.
//
// A chunk holds --kv-chunk-tokens cells (4096 by default). Chunk c of the pool is the same block range in every
// QSA layer's arrays, so one table per session (logical chunk -> pool chunk, kv_stream.hpp's KvHostPools::chunk)
// serves all its layers. Entries the session has not reserved point at a spare TRASH chunk: a write past the
// reservation lands there instead of in another session's cells, and nothing reads it back.
//
// A layer split (one session per stage, each on its own GPU, each with its own QSA layers) shares the same chunk
// numbers: a LANE is the main session or one batch slot across every stage. Its chunk table is one list; each
// stage's session reads a copy of it in its own GPU's memory. Reserving, shrinking and swapping act on the lane,
// named by any of its sessions.
//
// The pool only hands out and takes back chunks (kv_chunks.hpp decides which). Who gives way when it runs out is
// the server loop's call (generate.cpp): an idle slot's cached conversation first, then the request that needs
// the room.
#pragma once

#include "strata/core/kv_chunks.hpp"
#include "strata/core/session.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class KvPool {
public:
    static constexpr int64_t kDefaultChunkTokens = 4096;

    /// A stage of the layer split: a session carved for the stage's layers and the GPU it runs on (-1: the current
    /// one). A single GPU is one stage.
    struct StageRef {
        const SessionState* ss = nullptr;
        int device = -1;
    };

    /// Pins the pool for the streamed QSA layers of the stages' reference sessions (built with
    /// qsa_set_kv_shared_host): `tokens` cells per layer, rounded up to whole chunks, plus the trash chunk.
    bool init(const ModelGeometry& g, const std::vector<StageRef>& stages, int64_t tokens, std::string& error,
              int64_t chunk_tokens = kDefaultChunkTokens);
    bool active() const { return map_.active(); }
    void set_share_prefix(bool enabled) { share_prefix_ = enabled; }
    bool sharing() const { return share_prefix_; }

    /// Opens a lane with `ss` as its stage-0 session: gives the session's streamed QSA states the pool's arrays and
    /// the lane's chunk table, everything pointed at the trash chunk. Before any graph that reads the states is
    /// captured (they bake the pointers in).
    bool attach(SessionState& ss, std::string& error);
    /// The same for the lane's session on stage `stage` (after its stage-0 session `lane` was attached).
    bool attach_stage(const SessionState& lane, SessionState& ss, size_t stage, std::string& error);

    /// Closes the lane `ss` belongs to, giving its chunks back: before its sessions are destroyed.
    void detach(const SessionState& ss);

    /// Backs cells [0, cells) of the lane with pool chunks. False, with nothing changed, when the free chunks
    /// do not cover it.
    bool reserve(const SessionState& ss, int64_t cells);
    /// Every chunk of the lane back to the pool.
    void release(const SessionState& ss) { shrink(ss, 0); }
    /// The lane's chunks past the one holding cell `cells - 1` back to the pool.
    void shrink(const SessionState& ss, int64_t cells);
    /// Exchanges the chunks of two lanes (the same context length): a conversation's K/V changes session without a
    /// byte copied. Their VRAM residency maps still name the old blocks - reset both (kv_stream_reset).
    bool swap(const SessionState& a, const SessionState& b);

    // Caller is at a commit/prefill boundary. Changes synchronize every stage before touching host tables.
    bool prepare_write(const SessionState& ss, int64_t first, int64_t end);
    bool prepare_write(const QsaState& st, int64_t first, int64_t end);
    int64_t write_need_cells(const SessionState& ss, int64_t first, int64_t end,
                             const std::vector<const SessionState*>& releasing = {},
                             const std::vector<const KvChunkMap::Prefix*>& prefixes = {}) const;
    int64_t reclaimable_cells(const std::vector<const SessionState*>& sessions,
                              const std::vector<const KvChunkMap::Prefix*>& prefixes = {}) const;
    // Snapshot primitives use the layer's lane; indexer state is never carried by these calls.
    std::shared_ptr<const KvChunkMap::Prefix> retain(const QsaState& st, int64_t cells);
    bool restore(const QsaState& st, const KvChunkMap::Prefix& prefix, int64_t cells);
    // Pure capacity admission using the same transaction planner as restore; no CUDA or ownership changes.
    bool can_restore(const SessionState& ss, const KvChunkMap::Prefix& prefix, int64_t shared_cells,
                      int64_t reserve_cells, const std::vector<const SessionState*>& releasing = {},
                      const std::vector<const KvChunkMap::Prefix*>& prefixes = {}) const;
    bool share(const SessionState& from, const SessionState& to, int64_t cells);
    bool compatible(const KvChunkMap::Prefix& prefix) const { return map_.compatible(prefix); }
    int64_t reference_cells() const { return map_.references() * chunk_cells(); }
    int64_t unique_cells() const { return total_cells() - free_cells(); }
    int64_t reserved_cells(const SessionState& ss) const;
    int64_t free_cells() const { return map_.free_chunks() * chunk_cells(); }
    int64_t total_cells() const { return map_.n_chunks() * chunk_cells(); }
    int64_t chunk_cells() const { return page_size_ << chunk_shift_; }
    uint64_t pinned_bytes() const { return pinned_bytes_; }
    /// Counts the changes to the chunk tables (the server reports the pool when it changes).
    uint64_t version() const { return map_.version(); }

private:
    /// One session of a lane: its copy of the lane's table, on its GPU.
    struct Part {
        const SessionState* ss = nullptr;
        int device = -1;
        int32_t* dev = nullptr;          ///< the kernels' copy (fixed address)
    };
    struct Lane {
        KvChunkMap::Lane id = 0;
        std::vector<Part> parts;         ///< by stage
    };
    Lane* find(const SessionState& ss);
    const Lane* find(const SessionState& ss) const;
    Lane* find(const QsaState& st);
    void synchronize_or_die() const;
    void copy_chunks(const std::vector<KvChunkMap::Copy>& copies);
    bool upload(const Lane& lane);
    bool bind(Lane& lane, SessionState& ss, size_t stage, std::string& error);
    void uploaded_or_die(const Lane& lane);

    KvChunkMap map_;
    int64_t page_size_ = 0;
    int chunk_shift_ = 0;
    uint64_t pinned_bytes_ = 0;
    bool share_prefix_ = false;
    int64_t heads_ = 0, head_dim_ = 0;
    struct StageLayers {
        int64_t qsa_ord0 = 0;
        int device = -1;
        std::vector<strata::kernels::KvHostPools> layers;   ///< by QSA ordinal - qsa_ord0
    };
    std::vector<StageLayers> stages_;
    std::vector<std::unique_ptr<Lane>> lanes_;
};

}  // namespace strata::core
