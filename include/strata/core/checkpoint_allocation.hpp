#pragma once
#include "strata/core/conversation_cache.hpp"
#include <array>

namespace strata::core {
// Keep successful capture's moves recoverable until the cache accepts the image.
// Used only by selected-context server planning; disabled lifecycle is unchanged.
struct CapturedCheckpointMoveRollback {
    SavedConversation& image;
    std::vector<ConversationCheckpoint>& moved;
    size_t first;
    bool on;
    ~CapturedCheckpointMoveRollback() {
        if (on) for (size_t i = 0; i < moved.size(); ++i)
            moved[i] = std::move(image.checkpoints[first + i]);
    }
};
inline size_t park_capture_item_bound(const std::vector<ConversationKv>& kv, size_t bytes, size_t layers) {
    size_t count = memory_bytes_sum({bytes / ConversationBuffer::segment_bytes,
                                    layers > SIZE_MAX / 5 ? SIZE_MAX : 5 * layers});
    for (const auto& k : kv)
        for (const auto* p : {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled})
            p->visit(0, p->size(), [&](const uint8_t*, size_t, size_t) {
                count = memory_bytes_sum({count, 1}); return true;
            });
    return count;
}
// Payload capacities after clear/reserve/copy. Additional is a conservative peak:
// charge every replacement in full while ALL old backing is still held. Never
// subtract a moved allocation as though it were a newly allocated copy.
struct CheckpointAllocationBound {
    size_t resident = 0, additional = 0;
    void add(CheckpointAllocationBound b) {
        resident = memory_bytes_sum({resident, b.resident});
        additional = memory_bytes_sum({additional, b.additional});
    }
};
inline size_t allocation_product(size_t a, size_t b) { return b && a > SIZE_MAX / b ? SIZE_MAX : a * b; }
inline CheckpointAllocationBound checkpoint_array_bound(size_t capacity, size_t count, size_t element, bool context) {
    const auto charge = [&](size_t n) {
        const size_t bytes = allocation_product(n, element);
        return context ? context_allocation_bytes(bytes) : bytes;
    };
    return {charge(std::max(capacity, count)), count > capacity ? charge(count) : 0};
}
// The baseline grows directories with push_back, not reserve. Supported standard
// libraries grow by at most 2x: final capacity < 2*count, and old+new directory
// backing < 4*count at any replacement. This estimates that existing behavior;
// it does not impose an exact reservation on disabled context.
inline CheckpointAllocationBound checkpoint_directory_bound(size_t capacity, size_t count, size_t element, bool geometric) {
    if (!geometric || count <= capacity) return checkpoint_array_bound(capacity, count, element, false);
    return {allocation_product(allocation_product(count, 2), element),
            allocation_product(allocation_product(count, 4), element)};
}
using CheckpointShape = std::array<size_t, 7>;
inline CheckpointShape checkpoint_shape(const ConversationCheckpoint& c, bool capacity = false) {
    if (capacity) return {c.ids.capacity(), c.imgs.capacity(), c.gdn.capacity(), c.ple.capacity(),
                          c.tails.capacity(), c.dead.capacity(), c.block_pos.capacity()};
    return {c.ids.size(), c.imgs.size(), c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()};
}
inline CheckpointAllocationBound checkpoint_payload_bound(const ConversationCheckpoint& dst, const CheckpointShape& shape,
                                                          bool live_resize = false) {
    const auto caps = checkpoint_shape(dst, true);
    CheckpointAllocationBound b;
    for (size_t i = 0; i < shape.size(); ++i) {
        // A single baseline resize may retain geometric spare capacity. A new
        // allocation is bounded by old capacity + requested count (no smaller
        // than either libstdc++'s size-based or libc++'s capacity-based growth).
        const size_t count = live_resize && !ContextLockPolicy::enabled() && i >= 2 && shape[i] > caps[i] ?
            memory_bytes_sum({caps[i], shape[i]}) : shape[i];
        b.add(checkpoint_array_bound(caps[i], count, i == 0 ? sizeof(int32_t) : i == 1 ? sizeof(ConversationImageKey) : 1, i >= 2));
    }
    return b;
}
inline CheckpointAllocationBound checkpoint_copy_bound(const ConversationCheckpoint& dst, const ConversationCheckpoint& src,
                                                       bool into_storage = true) {
    auto b = checkpoint_payload_bound(dst, checkpoint_shape(src));
    b.add(checkpoint_directory_bound(dst.stage_parts.capacity(), src.stage_parts.size(), sizeof(ConversationCheckpoint),
                                    into_storage && !ContextLockPolicy::enabled()));
    for (const auto& part : src.stage_parts) b.add(checkpoint_copy_bound({}, part, false));
    return b;
}
// Mirrors live-pop then copy_checkpoints_beating's back-to-front pool consumption.
inline CheckpointAllocationBound checkpoint_capture_bound(const CheckpointShape& live,
        const std::vector<ConversationCheckpoint>& copies, const std::vector<ConversationCheckpoint>& storage,
        const std::vector<ConversationCheckpoint>& moved) {
    size_t left = storage.size();
    const ConversationCheckpoint empty;
    const auto& dst = left ? storage[--left] : empty;
    auto b = checkpoint_payload_bound(dst, live, true);
    b.add({allocation_product(dst.stage_parts.capacity(), sizeof(ConversationCheckpoint)), 0});
    if (!ContextLockPolicy::enabled() && moved.size() == 1) {
        // Exactly one push beyond the baseline reserve(copies.size()).
        const size_t capacity = copies.empty() ? 1 : allocation_product(copies.size(), 2);
        b.add({allocation_product(capacity, sizeof(ConversationCheckpoint)),
               allocation_product(memory_bytes_sum({capacity, copies.size()}), sizeof(ConversationCheckpoint))});
    } else {
        b.add(checkpoint_directory_bound(0, memory_bytes_sum({copies.size(), moved.size()}), sizeof(ConversationCheckpoint),
                                        !ContextLockPolicy::enabled() && !moved.empty()));
    }
    for (const auto& c : copies) {
        const bool pooled = left != 0;
        b.add(checkpoint_copy_bound(pooled ? storage[--left] : empty, c, pooled));
    }
    for (const auto& c : moved) b.add({c.bytes(), 0});
    return b;
}
} // namespace strata::core
