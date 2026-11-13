#include "strata/core/kv_chunks.hpp"

#include <algorithm>
#include <numeric>

namespace strata::core {

KvChunkMap::~KvChunkMap() {
    for (Lane l = 0; l < (Lane) lanes_.size(); ++l) shrink(l, 0);
}
void KvChunkMap::Storage::drop(int32_t c) {
    if (--refs[(size_t) c] == 0) free.push_back(c);
}
KvChunkMap::Prefix::~Prefix() {
    if (!storage_) return;
    for (auto c : chunks_) storage_->drop(c);
    if (!chunks_.empty()) ++storage_->version;
}
void KvChunkMap::init(int64_t n) {
    for (Lane l = 0; l < (Lane) lanes_.size(); ++l) shrink(l, 0);
    lanes_.clear();
    storage_ = std::make_shared<Storage>();
    storage_->refs.assign((size_t) std::max<int64_t>(n, 0), 0);
    for (int64_t c = n_chunks() - 1; c >= 0; --c) storage_->free.push_back((int32_t) c);
}
bool KvChunkMap::active() const { return n_chunks() > 0; }
int64_t KvChunkMap::n_chunks() const { return storage_ ? (int64_t) storage_->refs.size() : 0; }
int64_t KvChunkMap::free_chunks() const { return storage_ ? (int64_t) storage_->free.size() : 0; }
uint64_t KvChunkMap::version() const { return storage_ ? storage_->version : 0; }
int64_t KvChunkMap::references() const {
    return storage_ ? std::accumulate(storage_->refs.begin(), storage_->refs.end(), int64_t(0)) : 0;
}
int64_t KvChunkMap::owners(int32_t c) const { return storage_->refs[(size_t) c]; }

KvChunkMap::Lane KvChunkMap::add_lane(int64_t logical_chunks) {
    auto l = std::make_unique<LaneData>();
    l->table.assign((size_t) std::max<int64_t>(logical_chunks, 0), trash());
    lanes_.push_back(std::move(l));
    return (Lane) lanes_.size() - 1;
}
bool KvChunkMap::reserve(Lane lane, int64_t need, bool* changed) {
    if (changed) *changed = false;
    LaneData& l = *lanes_[(size_t) lane];
    need = std::clamp<int64_t>(need, 0, (int64_t) l.table.size());
    if (need <= l.held) return true;
    if (need - l.held > free_chunks()) return false;
    for (; l.held < need; ++l.held) {
        const int32_t c = storage_->free.back();
        storage_->free.pop_back();
        l.table[(size_t) l.held] = c;
        ++storage_->refs[(size_t) c];
    }
    ++storage_->version;
    if (changed) *changed = true;
    return true;
}
bool KvChunkMap::shrink(Lane lane, int64_t keep) {
    LaneData& l = *lanes_[(size_t) lane];
    keep = std::max<int64_t>(keep, 0);
    if (l.held <= keep) return false;
    for (int64_t c = l.held - 1; c >= keep; --c) {
        storage_->drop(l.table[(size_t) c]);
        l.table[(size_t) c] = trash();
    }
    l.held = keep;
    ++storage_->version;
    return true;
}
bool KvChunkMap::swap(Lane a, Lane b) {
    if (a == b) return false;
    LaneData& la = *lanes_[(size_t) a];
    LaneData& lb = *lanes_[(size_t) b];
    if (la.table.size() != lb.table.size()) return false;
    std::swap_ranges(la.table.begin(), la.table.end(), lb.table.begin());
    std::swap(la.held, lb.held);
    ++storage_->version;
    return true;
}
std::shared_ptr<const KvChunkMap::Prefix> KvChunkMap::retain(Lane lane, int64_t count) {
    const LaneData& l = *lanes_[(size_t) lane];
    if (count < 0 || count > l.held) return {};
    auto out = std::make_shared<Prefix>();
    out->chunks_.assign(l.table.begin(), l.table.begin() + count);
    out->storage_ = storage_;
    for (auto c : out->chunks_) ++storage_->refs[(size_t) c];
    if (count) ++storage_->version;
    return out;
}
bool KvChunkMap::compatible(const Prefix& p) const { return p.storage_ == storage_; }
int64_t KvChunkMap::write_need(Lane lane, int64_t first, int64_t end, const std::vector<Lane>& releasing,
                              const std::vector<const Prefix*>& prefixes) const {
    const LaneData& l = *lanes_[(size_t) lane];
    end = std::clamp<int64_t>(end, 0, (int64_t) l.table.size());
    first = std::clamp<int64_t>(first, 0, end);
    int64_t n = std::max<int64_t>(0, end - l.held);
    const auto removed = releasing.empty() && prefixes.empty() ? std::vector<int64_t>{}
                                                               : released_refs(releasing, prefixes, lane);
    for (int64_t c = first; c < std::min(end, l.held); ++c) {
        const int32_t physical = l.table[(size_t) c];
        int64_t refs = owners(physical);
        if (!removed.empty()) refs -= removed[(size_t) physical];
        n += refs > 1;
    }
    return n;
}
bool KvChunkMap::prepare_write(Lane lane, int64_t first, int64_t end, std::vector<Copy>& copies, bool* changed) {
    copies.clear();
    if (changed) *changed = false;
    LaneData& l = *lanes_[(size_t) lane];
    end = std::clamp<int64_t>(end, 0, (int64_t) l.table.size());
    first = std::clamp<int64_t>(first, 0, end);
    if (write_need(lane, first, end) > free_chunks()) return false;
    copies.reserve((size_t) (end - first)); // allocate before changing references
    bool grew = false;
    reserve(lane, end, &grew);
    for (int64_t c = first; c < end; ++c) {
        const int32_t old = l.table[(size_t) c];
        if (owners(old) == 1) continue;
        const int32_t fresh = storage_->free.back();
        storage_->free.pop_back();
        ++storage_->refs[(size_t) fresh];
        storage_->drop(old);
        l.table[(size_t) c] = fresh;
        copies.emplace_back(old, fresh);
    }
    if (!copies.empty()) ++storage_->version;
    if (changed) *changed = grew || !copies.empty();
    return true;
}
bool KvChunkMap::plan_share(Lane lane, const Prefix& p, int64_t count, int64_t need,
                            const std::vector<Lane>& releasing, const std::vector<const Prefix*>& prefixes,
                            SharePlan& plan) const {
    const LaneData& l = *lanes_[(size_t) lane];
    if (!compatible(p) || count < 0 || count > (int64_t) p.chunks_.size() ||
        count > (int64_t) l.table.size() || need < count || need > (int64_t) l.table.size() ||
        std::find(prefixes.begin(), prefixes.end(), &p) != prefixes.end()) return false;
    plan.refs = storage_->refs;
    plan.free = storage_->free;
    plan.free.reserve((size_t) n_chunks());
    plan.table = l.table;
    const auto removed = released_refs(releasing, prefixes, lane);
    for (size_t c = 0; c < removed.size(); ++c) {
        plan.refs[c] -= removed[c];
        if (removed[c] && plan.refs[c] == 0) plan.free.push_back((int32_t) c);
    }
    auto& refs = plan.refs;
    auto& free = plan.free;
    auto& table = plan.table;
    auto& copies = plan.copies;
    copies.reserve((size_t) (need - count));
    for (int64_t c = l.held - 1; c >= need; --c) {
        const int32_t old = table[(size_t) c];
        if (--refs[(size_t) old] == 0) free.push_back(old);
        table[(size_t) c] = trash();
    }
    for (int64_t c = 0; c < count; ++c) {
        const int32_t old = table[(size_t) c], src = p.chunks_[(size_t) c];
        if (old == src) continue;
        ++refs[(size_t) src];
        if (c < l.held && --refs[(size_t) old] == 0) free.push_back(old);
        table[(size_t) c] = src;
    }
    for (int64_t c = count; c < need; ++c) {
        const int32_t old = table[(size_t) c];
        if (c < l.held && refs[(size_t) old] == 1) continue;
        if (free.empty()) return false;
        const int32_t fresh = free.back(); free.pop_back();
        ++refs[(size_t) fresh];
        if (c < l.held) { --refs[(size_t) old]; copies.emplace_back(old, fresh); }
        table[(size_t) c] = fresh;
    }
    return true;
}
bool KvChunkMap::can_share(Lane lane, const Prefix& p, int64_t count, int64_t need,
                           const std::vector<Lane>& releasing, const std::vector<const Prefix*>& prefixes) const {
    SharePlan plan;
    return plan_share(lane, p, count, need, releasing, prefixes, plan);
}
bool KvChunkMap::share(Lane lane, const Prefix& p, int64_t count, int64_t need,
                       std::vector<Copy>& copies, bool* changed) {
    copies.clear();
    if (changed) *changed = false;
    SharePlan plan;
    if (!plan_share(lane, p, count, need, {}, {}, plan)) return false;
    LaneData& l = *lanes_[(size_t) lane];
    if (plan.table == l.table && need == l.held) return true;
    storage_->refs.swap(plan.refs); storage_->free.swap(plan.free);
    std::copy(plan.table.begin(), plan.table.end(), l.table.begin());
    l.held = need;
    copies = std::move(plan.copies);
    ++storage_->version;
    if (changed) *changed = true;
    return true;
}
std::vector<int64_t> KvChunkMap::released_refs(const std::vector<Lane>& lanes,
                                             const std::vector<const Prefix*>& prefixes, Lane exclude) const {
    std::vector<int64_t> removed((size_t) n_chunks(), 0);
    std::vector<bool> seen(lanes_.size(), false);
    for (Lane id : lanes) {
        if (id == exclude || seen[(size_t) id]) continue;
        seen[(size_t) id] = true;
        const LaneData& l = *lanes_[(size_t) id];
        for (int64_t c = 0; c < l.held; ++c) ++removed[(size_t) l.table[(size_t) c]];
    }
    for (size_t i = 0; i < prefixes.size(); ++i) {
        const Prefix* p = prefixes[i];
        if (!compatible(*p) || std::find(prefixes.begin(), prefixes.begin() + i, p) != prefixes.begin() + i) continue;
        for (int32_t physical : p->chunks_) ++removed[(size_t) physical];
    }
    return removed;
}
int64_t KvChunkMap::reclaimable(const std::vector<Lane>& lanes, const std::vector<const Prefix*>& prefixes) const {
    const auto removed = released_refs(lanes, prefixes);
    int64_t n = 0;
    for (size_t c = 0; c < removed.size(); ++c) n += removed[c] && removed[c] == storage_->refs[c];
    return n;
}

} // namespace strata::core
