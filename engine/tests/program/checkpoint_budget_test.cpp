// checkpoint_budget.hpp: the budgets a checkpoint chain lives under (synthetic: no model, no GPU).
#include "strata/program/checkpoint_budget.hpp"

#include <cstdio>

using strata::program::CheckpointBudgetItem;
using strata::program::checkpoint_budget_select;
using strata::program::checkpoint_budget_victims;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static CheckpointBudgetItem item(size_t bytes, uint64_t used, bool pinned = false) {
    CheckpointBudgetItem it;
    it.bytes = bytes;
    it.used = used;
    it.pinned = pinned;
    return it;
}

int main() {
    // ---- the selection: the byte budget bounds what an admission copies ----
    {
        // no budget: every checkpoint is carried (the #6533 behaviour)
        std::vector<CheckpointBudgetItem> items = {item(100, 1), item(100, 2), item(100, 3)};
        const std::vector<size_t> all = checkpoint_budget_select(items, 0);
        CHECK(all.size() == 3);
    }
    {
        // a budget that holds two of the three: the most recently used two are carried, in the chain's order
        std::vector<CheckpointBudgetItem> items = {item(100, 1), item(100, 3), item(100, 2)};
        const std::vector<size_t> sel = checkpoint_budget_select(items, 200);
        CHECK(sel.size() == 2);
        CHECK(sel[0] == 1);   // the most recently used one (used=3) is at index 1
        CHECK(sel[1] == 2);   // then the next (used=2)
    }
    {
        // a pinned prefix is carried first of all, even when it is the oldest
        std::vector<CheckpointBudgetItem> items = {item(100, 1, true), item(100, 9), item(100, 8)};
        const std::vector<size_t> sel = checkpoint_budget_select(items, 100);
        CHECK(sel.size() == 1);
        CHECK(sel[0] == 0);   // the pinned one
    }
    {
        // a budget too small for even one checkpoint still carries one (a slot with no checkpoint cannot resume)
        std::vector<CheckpointBudgetItem> items = {item(1000, 1), item(1000, 2)};
        const std::vector<size_t> sel = checkpoint_budget_select(items, 10);
        CHECK(sel.size() == 1);
        CHECK(sel[0] == 1);   // the most recently used one
    }
    {
        // an empty chain carries nothing
        const std::vector<CheckpointBudgetItem> items;
        CHECK(checkpoint_budget_select(items, 4096).empty());
    }

    // ---- the victims: the chain is trimmed to the count budget, then to the byte budget ----
    {
        // over the count budget: the least recently used goes first
        std::vector<CheckpointBudgetItem> items = {item(10, 1), item(10, 2), item(10, 3)};
        const std::vector<size_t> v = checkpoint_budget_victims(items, 2, 0);
        CHECK(v.size() == 1);
        CHECK(v[0] == 0);
    }
    {
        // within the count budget but over the byte budget: the byte budget applies too
        std::vector<CheckpointBudgetItem> items = {item(100, 1), item(100, 2), item(100, 3)};
        const std::vector<size_t> v = checkpoint_budget_victims(items, 10, 250);
        CHECK(v.size() == 1);
        CHECK(v[0] == 0);   // the least recently used
    }
    {
        // a pinned one is never the victim: the count stays over the budget because of it
        std::vector<CheckpointBudgetItem> items = {item(10, 1, true), item(10, 2)};
        const std::vector<size_t> v = checkpoint_budget_victims(items, 1, 0);
        CHECK(v.size() == 1);
        CHECK(v[0] == 1);
    }
    {
        // the byte budget alone (a count budget of 0 does not apply) trims the chain
        std::vector<CheckpointBudgetItem> items = {item(100, 1), item(100, 2)};
        const std::vector<size_t> v = checkpoint_budget_victims(items, 0, 150);
        CHECK(v.size() == 1);
        CHECK(v[0] == 0);
    }
    {
        // nothing to do: within both budgets
        std::vector<CheckpointBudgetItem> items = {item(100, 1), item(100, 2)};
        CHECK(checkpoint_budget_victims(items, 2, 200).empty());
    }
    {
        // one checkpoint is never dropped (a chain of one is the conversation's own turn)
        std::vector<CheckpointBudgetItem> items = {item(1000, 1)};
        CHECK(checkpoint_budget_victims(items, 0, 1).empty());
    }
    {
        // the victims come back in the chain's order, so the caller can drop them from the back
        std::vector<CheckpointBudgetItem> items = {item(10, 1), item(10, 2), item(10, 3), item(10, 4)};
        const std::vector<size_t> v = checkpoint_budget_victims(items, 1, 0);
        CHECK(v.size() == 3);
        CHECK(v[0] == 0 && v[1] == 1 && v[2] == 2);
    }

    if (fails == 0) std::printf("checkpoint_budget_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
