// strata-inkling CUDA: the prefill chunk the engine picks when --chunk is not given.
//
// The chunk is how many prompt tokens one forward step takes (ink_serve token_budget, runner max_batch). A bigger chunk reads a prompt faster
// (RTX 5090, bench, one stream, 4K prompt: SQ-20 512 -> 8.1K tok/s, 2048 -> 10.5K, 4096 -> 10.7K; UD-Q4_K_XL 4.07K -> 4.47K at 2048, flat after) and costs
// working memory: the runner and its K/V took 1460 / 1822 / 2710 / 4558 MiB at chunk 512 / 1024 / 2048 / 4096 on an RTX 4090 (SQ-20 all resident, 5255
// MiB free after the experts; the engine caps max_batch at 4096), 1440 / 1804 MiB at 512 / 1024 on an RTX 3060 Ti 8 GB (SQ-10, 2955 MiB free), where
// 2048 did not fit and the engine's halve-on-fail retry brought it back to 1024.
//
// The experts are placed first, with a reserve for chunk 512, so the memory left after they are in (free_after_load) is what a bigger chunk may use:
// a card whose experts are all resident has plenty, a hybrid card keeps 512. Many slots share each step (decode tokens wait for the prefill tokens
// of the others), so the chunk is capped by the slot count.
#pragma once

#include <initializer_list>

namespace strata::core {

constexpr int kInkChunkDefault = 512;
constexpr double kInkChunkBaseMiB = 1460.0;      // runner + K/V at chunk 512 (RTX 4090 SQ-20: 1460, RTX 3060 Ti SQ-10: 1440; 1 slot)
constexpr double kInkChunkMiBPerToken = 0.87;    // working memory per chunk token above 512 (4090: 1822 / 2710 / 4558 MiB at 1024 / 2048 / 4096, 0.86 MiB per token)
constexpr double kInkChunkMarginMiB = 1500.0;    // left free after the runner: K/V growth, the drafter, the desktop

// Memory (MiB) that must be free after the experts are placed for `chunk` to be chosen. kv_grow_mib: the global K/V that still has to be
// allocated for every slot to reach the context (slots x (max_ctx - the window allocated at start) x the bytes of a position): working memory the
// chunk must not take, or the K/V cannot grow on the card and its tail goes to host RAM (RTX 4090, UD-Q4_K_XL, 2 slots at 32K: decode at 30K tokens
// fell from 91 to 36 tok/s, 3 slots 14.7, with the K/V on the card 87).
inline double ink_chunk_need_mib(int chunk, double kv_grow_mib = 0.0) {
    return kInkChunkBaseMiB + kInkChunkMiBPerToken * (chunk > kInkChunkDefault ? chunk - kInkChunkDefault : 0) + kInkChunkMarginMiB
           + (kv_grow_mib > 0.0 ? kv_grow_mib : 0.0);
}

// Slots that share one step: the more of them, the smaller the prefill share one step may carry (a decoding slot waits for the whole step).
// 32 slots, a 16K prompt arriving among 31 decoders, longest wait of a decoding slot (ms) at chunk 512 / 1024 / 2048: RTX 3060 Ti 8 GB (SQ-10 hybrid)
// 2322 / 2210 / 43,424 (the long prompt read in 91 / 56 / 218 s); RTX 4090 (SQ-20 resident) 2096 / 3974 / -, the long prompt 108 / 110 s: at 32 slots a
// bigger chunk costs the decoders and does not read the long prompt faster. So: 2048 for 1-3 slots, 1024 for 4-15 slots, the old 512 from 16 slots up;
// 4096 is left to --chunk (RTX 5090 bench: 10.7K tok/s against 10.5K at 2048, flat).
inline int ink_chunk_cap_for_slots(int slots) {
    if (slots >= 16) return kInkChunkDefault;
    return slots >= 4 ? 1024 : 2048;
}

// The largest of 2048 / 1024 / 512 that fits `free_after_load_mib` and the slot cap; 512 when the memory is not known (< 0).
inline int ink_auto_chunk(double free_after_load_mib, int slots, double kv_grow_mib = 0.0) {
    if (free_after_load_mib < 0.0) return kInkChunkDefault;
    const int cap = ink_chunk_cap_for_slots(slots);
    for (int c : {2048, 1024}) {
        if (c <= cap && free_after_load_mib >= ink_chunk_need_mib(c, kv_grow_mib)) return c;
    }
    return kInkChunkDefault;
}

}  // namespace strata::core
