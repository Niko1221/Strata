// CPU-only ownership of the pinned KV pool. Tables keep their addresses for captured graphs.
#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace strata::core {

class KvChunkMap {
    struct Storage;
public:
    using Lane = int;
    // A cache image is an owner too. Its bookkeeping outlives the pool object: destruction never calls a pool.
    class Prefix {
    public:
        Prefix() = default;
        Prefix(const Prefix&) = delete;
        Prefix& operator=(const Prefix&) = delete;
        ~Prefix();
        const std::vector<int32_t>& chunks() const { return chunks_; }
        size_t bytes() const { return sizeof(Prefix) + chunks_.capacity() * sizeof(int32_t); }
    private:
        friend class KvChunkMap;
        std::shared_ptr<Storage> storage_;
        std::vector<int32_t> chunks_;
    };
    using Copy = std::pair<int32_t, int32_t>; // old physical chunk -> new physical chunk

    ~KvChunkMap();
    void init(int64_t n_chunks);
    bool active() const;
    Lane add_lane(int64_t logical_chunks);
    int64_t lanes() const { return (int64_t) lanes_.size(); }
    bool reserve(Lane lane, int64_t need, bool* changed = nullptr);
    bool shrink(Lane lane, int64_t keep);
    bool swap(Lane a, Lane b);

    // Only completed chunks belong in a prefix. The caller stores a partial final chunk privately.
    std::shared_ptr<const Prefix> retain(Lane lane, int64_t whole_chunks);
    bool compatible(const Prefix& prefix) const;
    // Replace [0, whole_chunks) by shared references and privately back the rest through `need`.
    // Like prepare_write, exhaustion is transactional. Copies must land before publishing the device table.
    bool share(Lane lane, const Prefix& prefix, int64_t whole_chunks, int64_t need,
               std::vector<Copy>& copies, bool* changed = nullptr);
    bool can_share(Lane lane, const Prefix& prefix, int64_t whole_chunks, int64_t need,
                    const std::vector<Lane>& releasing = {},
                    const std::vector<const Prefix*>& prefixes = {}) const;
    // Exclusively back every chunk in [first, end); includes reservation through end.
    bool prepare_write(Lane lane, int64_t first, int64_t end, std::vector<Copy>& copies,
                       bool* changed = nullptr);
    int64_t write_need(Lane lane, int64_t first, int64_t end, const std::vector<Lane>& releasing = {},
                        const std::vector<const Prefix*>& prefixes = {}) const;
    // Actual unique physical chunks released by the whole set, not the sum of each lane's held count.
    int64_t reclaimable(const std::vector<Lane>& lanes, const std::vector<const Prefix*>& prefixes = {}) const;
    int64_t references() const;
    int64_t owners(int32_t physical) const;

    int64_t held(Lane lane) const { return lanes_[(size_t) lane]->held; }
    int64_t logical(Lane lane) const { return (int64_t) lanes_[(size_t) lane]->table.size(); }
    const int32_t* table(Lane lane) const { return lanes_[(size_t) lane]->table.data(); }
    int64_t free_chunks() const;
    int64_t n_chunks() const;
    int32_t trash() const { return (int32_t) n_chunks(); }
    uint64_t version() const;

private:
    struct LaneData {
        std::vector<int32_t> table;
        int64_t held = 0;
    };
    struct Storage {
        std::vector<int64_t> refs;
        std::vector<int32_t> free;
        uint64_t version = 0;
        void drop(int32_t physical);
    };
    struct SharePlan {
        std::vector<int64_t> refs;
        std::vector<int32_t> free, table;
        std::vector<Copy> copies;
    };
    bool plan_share(Lane lane, const Prefix& prefix, int64_t count, int64_t need,
                     const std::vector<Lane>& releasing, const std::vector<const Prefix*>& prefixes,
                     SharePlan& plan) const;
    std::vector<int64_t> released_refs(const std::vector<Lane>& lanes, const std::vector<const Prefix*>& prefixes,
                                      Lane exclude = -1) const;
    std::shared_ptr<Storage> storage_;
    std::vector<std::unique_ptr<LaneData>> lanes_;
};

} // namespace strata::core
