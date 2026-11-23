// MIT License - Copyright (c) 2026 Strata contributors
#pragma once
#include "strata/core/prefix_radix_cache.hpp"

#include <cstdlib>
#include <cstdio>
#include <string>
#include <limits>

namespace strata::core {

// A backend describes a device region as packed rows with an optional device
// pitch. Host checkpoints always pack the useful bytes, including native scales.
struct PrefixRows {
    void* ptr;
    size_t width, count = 1, pitch = 0;
};
using PrefixLayout = std::vector<PrefixRows>;

inline bool parse_radix_mib(const std::string& value, size_t& bytes) {
    char* end = nullptr;
    const long mib = std::strtol(value.c_str(), &end, 10);
    if (value.empty() || *end || mib < 0 || mib > 8192 ||
        static_cast<unsigned long>(mib) > (std::numeric_limits<size_t>::max() >> 20)) return false;
    bytes = static_cast<size_t>(mib) << 20;
    return true;
}

inline int radix_chunk(int pos, int remaining, int cap) {
    int take = std::min(remaining, std::min(cap, 512 - pos % 512));
    if (take == remaining && remaining > 1) --take;
    return take;
}

class RadixStateCache {
    struct Segment {
        Segment(size_t& total, std::shared_ptr<const Segment> prev, int pos, size_t n)
            : total(total), prev(std::move(prev)), pos(pos), data(n),
              chain_bytes(sizeof(*this) + data.capacity() + (this->prev ? this->prev->chain_bytes : 0)) {
            total += sizeof(*this) + data.capacity();
        }
        ~Segment() { total -= sizeof(*this) + data.capacity(); }
        size_t& total;
        std::shared_ptr<const Segment> prev;
        int pos;
        std::vector<uint8_t> data;
        size_t chain_bytes;
    };
public:
    struct Checkpoint {
        int length;
        std::shared_ptr<const Segment> tail;
        std::vector<uint8_t> state;
        size_t bytes() const { return sizeof(*this) + state.capacity(); }
    };

    // A request may share append-only rows only with a checkpoint it restored
    // or saved on its own trajectory. The weak cursor never prevents eviction.
    using Cursor = std::weak_ptr<const Segment>;
    explicit RadixStateCache(size_t bytes) : index_(bytes), enabled_(bytes > 0) {}
    RadixStateCache(const RadixStateCache&) = delete;
    bool enabled() const { return enabled_; }
    void clear() { index_.clear(); }
    const Checkpoint* find(const std::vector<int32_t>& ids) {
        return ids.empty() ? nullptr : index_.find(ids, ids.size() - 1, any, true);
    }

    // Layout(begin,end) describes immutable append-only rows; State(end)
    // describes recurrent state or the current ring window. Finish waits
    // for all submitted copies before eviction can reclaim their host buffers.
    template<class Layout, class State, class Copy, class Finish>
    void save(const std::vector<int32_t>& ids, int pos, Cursor& cursor, Layout layout, State state, Copy copy, Finish finish) {
        if (pos < 16 || (pos % 512 && static_cast<size_t>(pos + 1) != ids.size())) return;
        auto* existing = index_.find(ids, pos, any, false);
        if (existing && existing->length == pos) return;
        try {
            auto parent = cursor.lock();
            const auto ranges = layout(parent ? parent->pos : 0, pos);
            const auto private_ranges = state(pos);
            const size_t shared = size(ranges), own = size(private_ranges);
            const size_t charge = shared + own + sizeof(Segment) + sizeof(Checkpoint) + index_.index_bytes(pos);
            if (!index_.reserve(charge, parent ? parent->chain_bytes : 0, [&] { return shared_bytes_; })) return;
            auto payload = std::make_unique<Checkpoint>();
            payload->length = pos;
            payload->state.resize(own);
            auto segment = std::make_shared<Segment>(shared_bytes_, std::move(parent), pos, shared);
            transfer(ranges, segment->data.data(), false, copy);
            transfer(private_ranges, payload->state.data(), false, copy);
            finish();
            payload->tail = segment;
            index_.insert(ids, std::move(payload));
            cursor = segment;
        } catch (const std::bad_alloc&) {
            clear();
            enabled_ = false;
            std::fprintf(stderr, "Disabling radix cache after a host allocation failure\n");
        }
    }

    template<class Layout, class State, class Copy, class Finish>
    bool restore(const Checkpoint& cp, Cursor& cursor, Layout layout, State state, Copy copy, Finish finish) {
        // Resolve pitches and device pointers before the first write. If a host
        // allocation fails, the resident checkpoint is still safe to select.
        struct Part { PrefixLayout rows; const uint8_t* bytes; };
        std::vector<Part> parts;
        try {
            for (auto* s = cp.tail.get(); s; s = s->prev.get()) {
                parts.push_back({layout(s->prev ? s->prev->pos : 0, s->pos), s->data.data()});
            }
            parts.push_back({state(cp.length), cp.state.data()});
        } catch (const std::bad_alloc&) {
            return false;
        }
        for (const auto& part : parts) transfer(part.rows, part.bytes, true, copy);
        finish();
        cursor = cp.tail;
        return true;
    }

private:
    static bool any(const Checkpoint&) { return true; }
    static size_t size(const PrefixLayout& ranges) {
        size_t total = 0;
        for (const auto& r : ranges) total += r.width * r.count;
        return total;
    }
    template<class Byte, class Copy>
    static void transfer(const PrefixLayout& ranges, Byte* host, bool restore, Copy copy) {
        for (const auto& r : ranges) {
            if (!r.width || !r.count) continue;
            copy(r, const_cast<uint8_t*>(host), restore);
            host += r.width * r.count;
        }
    }

    size_t shared_bytes_ = 0;
    PrefixRadixCache<Checkpoint> index_;
    bool enabled_;
};

} // namespace strata::core
