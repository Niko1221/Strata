// The indexer's score buffer of a Hy4-Flash prefill chunk (#6744). The runner's work arena holds, besides its other scratch, `chunk x (max_ctx / 4 + 8)` floats
// (the IndexPool scores of every chunk row against every pooled key). It grows with the context: 8.59 GB at a 1M context and a chunk of 8192 (the reporter's
// "HIP error 2 (out of memory)" at the first dalloc after the experts), 4.29 GB at 512K, 1.07 GB at 128K. plan_memory did not count it, so the reserve for the
// buffers was too small at 256K and above and the experts took the memory this buffer needed. Plain arithmetic: no GPU, no HIP header.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace hy4 {

inline uint64_t score_buffer_bytes(uint64_t chunk, uint64_t max_ctx) { return chunk * (max_ctx / 4 + 8) * 4ull; }

// The work arena's size in bytes, as Runner::init allocates it: the largest of the activation scratch, the score buffer and the MoE workspace.
inline uint64_t work_arena_bytes(uint64_t activation_bytes, uint64_t chunk, uint64_t max_ctx, uint64_t moe_ws_bytes) {
    return std::max({activation_bytes, score_buffer_bytes(chunk, max_ctx), moe_ws_bytes});
}

}  // namespace hy4
