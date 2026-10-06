// include/strata/core/kv_chunks.hpp - the shared KV pool's bookkeeping, without a byte of memory (--kv-pool-tokens).
//
// The pool (kv_pool.hpp) is `n_chunks` chunks of pinned K/V plus one spare TRASH chunk. A lane (a session: the main
// one or a batch slot, on every stage of a layer split) owns a table, logical chunk -> pool chunk, and holds a prefix
// of it: the first `held` entries are pool chunks handed out by `reserve`, the rest all name the trash chunk, so a
// write past the reservation lands there instead of in another lane's cells and nothing reads it back.
//
// This class only decides which chunk is whose. It has no CUDA in it, so its rules (lowest chunk first, a failed
// reserve changes nothing, shrink gives the tail back, swap exchanges two lanes' tables) are tested without a GPU.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace strata::core {

class KvChunkMap {
public:
    using Lane = int;

    /// `n_chunks` usable chunks, all free; the trash chunk is number `n_chunks`. Forgets every lane.
    void init(int64_t n_chunks);
    bool active() const { return n_chunks_ > 0; }

    /// A lane whose table has `logical_chunks` entries, all on the trash chunk, nothing held.
    Lane add_lane(int64_t logical_chunks);
    int64_t lanes() const { return (int64_t) lanes_.size(); }

    /// Backs the lane's chunks [0, need) (`need` is clamped to the table). False, with nothing changed, when the
    /// free chunks do not cover the ones it lacks. True when it changed the table, `changed` says so.
    bool reserve(Lane lane, int64_t need, bool* changed = nullptr);
    /// The lane's chunks [keep, held) back to the pool (a fresh pool hands out chunk 0, 1, 2 ..; after that a chunk
    /// freed is the next one handed out). True when it changed the table.
    bool shrink(Lane lane, int64_t keep);
    /// Exchanges two lanes' tables in place (same logical size): their buffers stay where they are, since
    /// the kernels and the DMA movers hold pointers to them.
    bool swap(Lane a, Lane b);

    int64_t held(Lane lane) const { return lanes_[(size_t) lane]->held; }
    int64_t logical(Lane lane) const { return (int64_t) lanes_[(size_t) lane]->table.size(); }
    const int32_t* table(Lane lane) const { return lanes_[(size_t) lane]->table.data(); }
    int64_t free_chunks() const { return (int64_t) free_.size(); }
    int64_t n_chunks() const { return n_chunks_; }
    int32_t trash() const { return (int32_t) n_chunks_; }
    /// Counts the changes to the tables (the server reports the pool when it changes).
    uint64_t version() const { return version_; }

private:
    struct LaneData {
        std::vector<int32_t> table;   ///< fixed size: its address is part of the kernels' arguments
        int64_t held = 0;
    };
    int64_t n_chunks_ = 0;
    uint64_t version_ = 0;
    std::vector<int32_t> free_;       ///< a stack: the next chunk to hand out is on top (a fresh pool: chunk 0)
    std::vector<std::unique_ptr<LaneData>> lanes_;
};

}  // namespace strata::core
