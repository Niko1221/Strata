// engine/src/inkling2/ink_ckpt_lru.h - how many slots may hold a prompt-end checkpoint, and which one gives its checkpoint up.
// No GPU: the CUDA backend keeps the checkpoints themselves (host RAM, ~80 MiB fp8 / ~160 MiB fp16 K/V each); this is only the policy.
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace ink2 {

// Up to 8 slots every slot keeps its prompt-end checkpoint (as before).  Above 8 there were none at all, so a follow-up turn
// resumed only when the new prompt continued the slot's whole held stream token for token (no thinking dropped, no reply
// re-tokenized).  STRATA_INK_PROMPT_CKPT_SLOTS=N (> 0) keeps checkpoints for up to N slots at once above 8, least recently
// used first out; unset or 0 keeps the old behaviour (none above 8).
inline int ink_prompt_ckpt_slots(int max_slots, const char* env) {
    if (max_slots <= 8) return max_slots;
    if (!env || !*env) return 0;
    const int n = std::atoi(env);
    if (n <= 0) return 0;
    return n < max_slots ? n : max_slots;
}

// Before `self` saves a checkpoint: the slot whose checkpoint must go so at most `cap` stay valid (the least recently saved or
// restored one, never `self`), or -1 when nothing has to go.  Call it until it returns -1.
inline int ink_prompt_ckpt_victim(const std::vector<bool>& valid, const std::vector<uint64_t>& last_use, int self, int cap) {
    int held = 0, victim = -1;
    for (int s = 0; s < static_cast<int>(valid.size()); ++s) {
        if (s == self || !valid[s]) continue;
        ++held;
        if (victim < 0 || last_use[s] < last_use[victim]) victim = s;
    }
    return held >= cap ? victim : -1;
}

}  // namespace ink2
