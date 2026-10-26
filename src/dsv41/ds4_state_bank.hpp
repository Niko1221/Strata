// Strata. The parked conversation states of the burst-interleaving path (docs/DSV41_BATCHING.md, Path B).
//
// Two requests take turns on the one model: each runs a burst of decoded tokens, then Model::slot_save()
// writes its state (tokens, window rings, compressed KV, index keys) into a byte vector here, and the next
// request slot_load()s its own.  The bank owns those vectors, bounded by bytes and count, oldest first.
//
// Pure host memory: the tests below need no GPU.  The serve loop integration is what uses it.
#pragma once

#include <cstdint>
#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata {
namespace ds4 {

class StateBank {
public:
    struct Options {
        size_t max_bytes = 2ull << 30;   // 2 GiB of parked states
        size_t max_count = 8;            // requests parked at once
    };

    StateBank() = default;
    explicit StateBank(Options o) : o_(o) {}

    /// Park `state` for `id`, replacing what it held.  Returns false when the state alone exceeds the budget.
    bool put(int64_t id, std::vector<uint8_t> state) {
        if (state.size() > o_.max_bytes) return false;
        drop(id);
        const size_t n = state.size();
        order_.push_back(id);
        items_[id] = Item{std::move(state), std::prev(order_.end())};
        bytes_ += n;
        evict();
        return true;
    }

    /// The state parked for `id` (nullptr: none).  A hit moves the entry to the newest end.
    const std::vector<uint8_t>* get(int64_t id) {
        auto it = items_.find(id);
        if (it == items_.end()) return nullptr;
        order_.splice(order_.end(), order_, it->second.at);
        return &it->second.bytes;
    }

    bool drop(int64_t id) {
        auto it = items_.find(id);
        if (it == items_.end()) return false;
        bytes_ -= it->second.bytes.size();
        order_.erase(it->second.at);
        items_.erase(it);
        return true;
    }

    size_t bytes() const { return bytes_; }
    size_t count() const { return items_.size(); }
    bool has(int64_t id) const { return items_.find(id) != items_.end(); }
    const Options& options() const { return o_; }

private:
    struct Item {
        std::vector<uint8_t> bytes;
        std::list<int64_t>::iterator at;
    };

    void evict() {
        while (!order_.empty() && (bytes_ > o_.max_bytes || items_.size() > o_.max_count)) {
            const int64_t victim = order_.front();
            amount_dropped_ += items_[victim].bytes.size();
            ++evictions_;
            drop(victim);
        }
    }

    Options o_{};
    size_t bytes_ = 0;
    size_t amount_dropped_ = 0;
    size_t evictions_ = 0;
    std::list<int64_t> order_;                                  // oldest first
    std::unordered_map<int64_t, Item> items_;

public:
    size_t evictions() const { return evictions_; }
    size_t evicted_bytes() const { return amount_dropped_; }
};

}  // namespace ds4
}  // namespace strata
