#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

// STRATA_SLOT_CHECKPOINTS: which of the main session's checkpoints a batch slot keeps with the conversation it takes
// (BADM) or keeps (park=B): the deepest `keep` that are a strict start of `held`, shallowest first, one per length
// (the first of equal ones, as the single deepest one was picked before).
template <class Checkpoint>
std::vector<const Checkpoint*> deepest_prefix_checkpoints(const std::vector<Checkpoint>& checks,
                                                         const std::vector<int32_t>& held, std::size_t keep) {
    std::vector<const Checkpoint*> out;
    for (const Checkpoint& c : checks)
        if (c.ids.size() < held.size() && std::equal(c.ids.begin(), c.ids.end(), held.begin())) out.push_back(&c);
    std::stable_sort(out.begin(), out.end(),
                     [](const Checkpoint* a, const Checkpoint* b) { return a->ids.size() < b->ids.size(); });
    out.erase(std::unique(out.begin(), out.end(),
                          [](const Checkpoint* a, const Checkpoint* b) { return a->ids.size() == b->ids.size(); }),
              out.end());
    if (out.size() > keep) out.erase(out.begin(), out.end() - (std::ptrdiff_t) keep);
    return out;
}

// How many leading tokens a prompt and a held conversation share.
inline int64_t shared_prefix(const std::vector<int64_t>& prompt, const std::vector<int32_t>& held) {
    const std::size_t n = std::min(prompt.size(), held.size());
    std::size_t i = 0;
    while (i < n && prompt[i] == (int64_t) held[i]) ++i;
    return (int64_t) i;
}

// STRATA_BRANCH_CHECKPOINT=<margin>: the read stops for a checkpoint at `shared`, where the prompt leaves a
// conversation the engine holds (the client rewrote what came after it), so the next prompt that branches from this
// one later still finds a state to continue from.  -1 for none: only `margin` tokens or more past where the read
// starts, and before the turn boundary (whose checkpoint is taken anyway) or, without one, the last prompt token.
inline int64_t branch_checkpoint_boundary(int64_t shared, int64_t read_from, int64_t turn_at, int64_t n,
                                          int64_t margin) {
    const int64_t end = turn_at > 0 ? turn_at : n - 1;
    if (margin <= 0 || shared <= 0 || shared - read_from < margin || shared >= end) return -1;
    return shared;
}

} // namespace strata::program
