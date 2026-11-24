// checkpoint_budget.hpp: the two budgets a checkpoint chain lives under (synthetic: no model, no GPU).
//
// A checkpoint is a running-state image (the running state of every layer plus the token ids and every stage's
// copy of them), so a chain of a thousand of them is tens of gigabytes.  Two budgets bound it:
//
//   - the count budget (--prompt-cache): how many checkpoints the chain may hold;
//   - the byte budget: how much host memory the chain may hold.
//
// The count budget alone does not bound the memory: how big one checkpoint is depends on the model and on the
// layer split, and it changes with the conversation (a checkpoint's running state grows with the tokens it
// covers).  The byte budget is the one that bounds the memory.
//
// Two places need them:
//   - the admission (the copy of the chain into a batch slot): only some of the chain is worth copying;
//   - the hand-back (the slot's chain merged into the main chain): the merge does not go through the path that
//     applies the budget, so the chain can grow past it.
//
// Both are plain functions over the chain's metadata, so they can be reasoned about (and tested) without a GPU.

#ifndef STRATA_PROGRAM_CHECKPOINT_BUDGET_HPP
#define STRATA_PROGRAM_CHECKPOINT_BUDGET_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata {
namespace program {

// One checkpoint of a chain, as far as the budgets are concerned.
struct CheckpointBudgetItem {
    size_t bytes = 0;      // the checkpoint's own size (its running state and its ids)
    uint64_t used = 0;     // the LRU stamp: a larger stamp is a more recently used checkpoint
    bool pinned = false;   // a pinned prefix is never the victim
};

// The checkpoints to carry into a batch slot: the most recently used ones first, a pinned one first of all, and
// as many as the byte budget holds.  The chain's own order is kept in the result (the caller copies in chain
// order; the selection is a set, not a sequence).
//
// The chain's order is kept by the caller: the returned indices are in chain order, the choice is by recency.
inline std::vector<size_t> checkpoint_budget_select(const std::vector<CheckpointBudgetItem>& items, size_t byte_budget) {
    std::vector<size_t> chosen;
    if (items.empty()) return chosen;
    if (byte_budget == 0) {                 // 0: no byte budget, every checkpoint is carried
        chosen.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) chosen.push_back(i);
        return chosen;
    }
    // the pinned ones first, then the most recently used: a rewritten history diverges at a recent turn, and a
    // pinned prefix is the one many requests branch from
    std::vector<size_t> order(items.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (items[a].pinned != items[b].pinned) return items[a].pinned;
        return items[a].used > items[b].used;
    });
    size_t bytes = 0;
    for (size_t i : order) {
        if (bytes + items[i].bytes > byte_budget && !chosen.empty()) break;   // always carry at least one
        bytes += items[i].bytes;
        chosen.push_back(i);
    }
    std::sort(chosen.begin(), chosen.end());   // the chain's own order
    return chosen;
}

// The checkpoints to drop from a chain: over the count budget first, then over the byte budget, least recently
// used first, a pinned one never (a pinned prefix is the one many suffix queries branch from).  A budget of 0 means
// that budget does not apply.
inline std::vector<size_t> checkpoint_budget_victims(const std::vector<CheckpointBudgetItem>& items,
                                                      size_t count_budget, size_t byte_budget) {
    std::vector<size_t> victims;
    if (items.size() < 2) return victims;
    std::vector<bool> alive(items.size(), true);
    size_t count = items.size();
    size_t bytes = 0;
    for (const CheckpointBudgetItem& it : items) bytes += it.bytes;
    // the least recently used first; a pinned one is skipped (the count can stay over the budget because of it)
    std::vector<size_t> order(items.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return items[a].used < items[b].used;
    });
    auto drop = [&](size_t idx) {
        if (!alive[idx]) return;
        alive[idx] = false;
        bytes -= items[idx].bytes;
        --count;
        victims.push_back(idx);
    };
    if (count_budget != 0) {          // 0: the count is not bounded (the same sense as the byte budget's 0)
        for (size_t i : order) {
            if (count <= count_budget) break;
            if (items[i].pinned) continue;
            drop(i);
        }
    }
    if (byte_budget != 0) {
        for (size_t i : order) {
            if (bytes <= byte_budget) break;
            if (items[i].pinned) continue;
            drop(i);
        }
    }
    std::sort(victims.begin(), victims.end());
    return victims;
}

}  // namespace program
}  // namespace strata

#endif  // STRATA_PROGRAM_CHECKPOINT_BUDGET_HPP
