// include/strata/core/kv_pool.hpp - one pinned host KV pool shared by every session (--kv-pool-tokens).
//
// With KV streaming (--kv-resident) every QSA layer of every session keeps the authoritative K/V of all its cells
// in pinned host memory. Sized per session at the full context, N batch slots plus the main session pin
// (N + 1) x the context's K/V although the conversations together rarely fill it. The pool pins `tokens` cells per
// QSA layer once, and a session holds only the chunks its conversation has reached.
//
// A chunk is 1024 blocks (4096 cells). Chunk c of the pool is the same block range in every QSA layer's arrays,
// so one table per session (logical chunk -> pool chunk, kv_stream.hpp's KvHostPools::chunk) serves all its
// layers. Entries the session has not reserved point at a spare TRASH chunk: a write past the reservation lands
// there instead of in another session's cells, and nothing reads it back.
//
// The pool only hands out and takes back chunks. Who gives way when it runs out is the server loop's call
// (generate.cpp): an idle slot's cached conversation first, then the request that needs the room.
#pragma once

#include "strata/core/session.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class KvPool {
public:
    static constexpr int kChunkShift = 10;   ///< 1024 blocks per chunk

    /// Pins the pool for the streamed QSA layers of `ref` (a session built with qsa_set_kv_shared_host): `tokens`
    /// cells per layer, rounded up to whole chunks, plus the trash chunk.
    bool init(const ModelGeometry& g, const SessionState& ref, int64_t tokens, std::string& error);
    bool active() const { return n_chunks_ > 0; }

    /// Gives the session's streamed QSA states the pool's arrays and the session's own chunk table, everything
    /// pointed at the trash chunk. Before any graph that reads the states is captured (they bake the pointers in).
    bool attach(SessionState& ss, std::string& error);

    /// Backs cells [0, cells) of the session with pool chunks. False, with nothing changed, when the free chunks
    /// do not cover it.
    bool reserve(const SessionState& ss, int64_t cells);
    /// Every chunk of the session back to the pool.
    void release(const SessionState& ss) { shrink(ss, 0); }
    /// The session's chunks past the one holding cell `cells - 1` back to the pool.
    void shrink(const SessionState& ss, int64_t cells);
    /// Exchanges the chunks of two sessions (the same context length): a conversation's K/V changes session without a
    /// byte copied. Their VRAM residency maps still name the old blocks - reset both (kv_stream_reset).
    bool swap(const SessionState& a, const SessionState& b);

    int64_t reserved_cells(const SessionState& ss) const;
    int64_t free_cells() const { return (int64_t) free_.size() * chunk_cells(); }
    int64_t total_cells() const { return n_chunks_ * chunk_cells(); }
    int64_t chunk_cells() const { return page_size_ << kChunkShift; }
    uint64_t pinned_bytes() const { return pinned_bytes_; }
    /// Counts the changes to the chunk tables (the server reports the pool when it changes).
    uint64_t version() const { return version_; }

private:
    struct Table {
        const SessionState* ss = nullptr;
        int32_t* dev = nullptr;          ///< the kernels' copy (fixed address)
        std::vector<int32_t> host;       ///< the DMA movers' copy, kept equal to `dev`
        int64_t held = 0;                ///< reserved logical chunks: [0, held)
    };
    Table* find(const SessionState& ss);
    const Table* find(const SessionState& ss) const;
    bool upload(const Table& t);

    int64_t n_chunks_ = 0;               ///< usable chunks; chunk n_chunks_ is the trash
    int64_t page_size_ = 0;
    uint64_t pinned_bytes_ = 0;
    uint64_t version_ = 0;
    std::vector<strata::kernels::KvHostPools> layers_;   ///< by QSA ordinal - qsa_ord0_
    int64_t qsa_ord0_ = 0;
    std::vector<int32_t> free_;
    std::vector<std::unique_ptr<Table>> tables_;
};

}  // namespace strata::core
