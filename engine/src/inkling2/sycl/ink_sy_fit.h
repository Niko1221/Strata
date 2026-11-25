// ink_sy_fit.h - what the SYCL runner needs per slot and in total, and how many slots fit a card (no SYCL, unit-tested)
//
// The SYCL runner allocates every slot's K/V (fp16, the whole context), the ShortConv state, the DSpark context and its batch buffers at start.
// With `--parallel 32 --max-context 4096` that is about 5.8 GiB of KV alone: on an 8 GB Arc A750 the start died in the middle of the load with
// "dalloc (268435456 bytes): out of device memory" (0221-INKOTHER, 32 clients). The same numbers drive the expert planner's reserve, so they live here;
// the backend now checks them against the card's free memory BEFORE the runner allocates and refuses with a plain message that names the largest
// --parallel that fits.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ink_sycl {

struct InkSyFitGeom {
    std::vector<uint8_t> visit_is_swa;  // [n_visits], 1 = sliding window layer, 0 = global
    int swa_win = 1024;
    int n_kv = 8;
    int head_dim = 128;
    int hidden = 2304;
    bool dspark = false;
    bool vision = false;
};

// The bytes the runner, its batch buffers, DSpark, Vision, the hybrid cache ring and the XMX scratch need on the device for `par` slots at `ctx` tokens
// (the planner keeps this much free of experts). Same formulas as the backend used before this header existed.
inline size_t ink_sy_runner_reserve(const InkSyFitGeom& g, int par, int ctx, int chunk) {
    const size_t p = (size_t)std::max(1, par);
    const int batch_cap = std::max(256, std::max(chunk, par));
    size_t kv = 0;
    for (uint8_t swa : g.visit_is_swa) {
        const size_t stride = swa ? (size_t)g.swa_win : (size_t)ctx;
        kv += 2ull * p * stride * (size_t)g.n_kv * (size_t)g.head_dim * 2ull;
    }
    const size_t dsp = g.dspark ? (3ull * 2ull * p * (size_t)ctx * 1024ull * 2ull) : 0ull;
    const size_t sc = g.visit_is_swa.size() * p * 3ull * (2048ull + 2ull * (size_t)g.hidden) * 8ull;
    const size_t act = (size_t)batch_cap * 512ull * 1024ull + (128ull << 20);
    const size_t vis = g.vision ? (700ull << 20) : 0ull;
    const size_t hyb_xmx = 320ull << 20;  // elastic VRAM cache ring (<= 256 MiB) + XMX scratch + ping-pong
    return kv + dsp + sc + act + vis + hyb_xmx;
}

// The largest --parallel (1..64) whose reserve fits `avail` bytes; 0 when even one slot does not.
inline int ink_sy_max_parallel(const InkSyFitGeom& g, int ctx, int chunk, size_t avail) {
    int best = 0;
    for (int p = 1; p <= 64; ++p) {
        if (ink_sy_runner_reserve(g, p, ctx, chunk) <= avail) best = p;
        else break;
    }
    return best;
}

// The refusal text; empty when the request fits `avail`.
inline std::string ink_sy_fit_message(const InkSyFitGeom& g, int par, int ctx, int chunk, size_t avail) {
    const size_t need = ink_sy_runner_reserve(g, par, ctx, chunk);
    if (need <= avail) return std::string();
    const double gib = 1024.0 * 1024.0 * 1024.0;
    char buf[480];
    const int most = ink_sy_max_parallel(g, ctx, chunk, avail);
    if (most >= 1) {
        std::snprintf(buf, sizeof buf,
                      "sycl: --parallel %d at --max-context %d needs about %.1f GiB of this card for the K/V caches and buffers, but only %.1f GiB are free after the weights; "
                      "the most that fits is --parallel %d (or a shorter --max-context)",
                      par, ctx, need / gib, avail / gib, most);
    } else {
        std::snprintf(buf, sizeof buf,
                      "sycl: --max-context %d needs about %.1f GiB of this card for the K/V caches and buffers even for one slot, but only %.1f GiB are free after the weights; "
                      "use a shorter --max-context",
                      ctx, need / gib, avail / gib);
    }
    return std::string(buf);
}

}  // namespace ink_sycl
