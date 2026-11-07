// Checkpoint admission policy with generated token IDs and payload bytes; no model or device is needed.
#include "strata/program/batch_checkpoints.hpp"

#include <cstdio>
#include <initializer_list>
#include <numeric>

using Checkpoint = strata::core::ConversationCheckpoint;
using strata::program::batch_checkpoint_prefix;
using strata::program::select_batch_checkpoints;

static int checks = 0, failures = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); ++failures; } } while (0)

static std::vector<int32_t> tokens(size_t count) {
    std::vector<int32_t> result(count);
    std::iota(result.begin(), result.end(), int32_t{101});
    return result;
}

static Checkpoint checkpoint(size_t count, size_t stages = 0, uint64_t stamp = 1) {
    Checkpoint c;
    c.ids = tokens(count);
    c.gdn = {1, 3, 5};
    c.ple = {7, 9};
    c.tails = {11, 13, 15, 17};
    c.dead = {19};
    c.block_pos = {21, 23, 25};
    c.used = stamp;
    c.dev_tag = 100 + stamp;
    c.plan = {8, 0, static_cast<int64_t>(count), 3, 0.5, true, true};
    for (size_t k = 0; k < stages; ++k) {
        auto part = checkpoint(count, 0, stamp + k + 1);
        part.gdn[0] = static_cast<uint8_t>(31 + k);
        c.stage_parts.push_back(std::move(part));
    }
    return c;
}

static bool same(const Checkpoint& a, const Checkpoint& b) {
    if (a.ids != b.ids || a.imgs != b.imgs || a.gdn != b.gdn || a.ple != b.ple || a.tails != b.tails ||
        a.dead != b.dead || a.block_pos != b.block_pos || a.used != b.used || a.dev_tag != b.dev_tag ||
        a.pinned != b.pinned || a.plan.chunk != b.plan.chunk || a.plan.from != b.plan.from ||
        a.plan.to != b.plan.to || a.plan.lend != b.plan.lend || a.plan.pipe != b.plan.pipe ||
        a.plan.single != b.plan.single || a.plan.chunked != b.plan.chunked ||
        a.stage_parts.size() != b.stage_parts.size()) return false;
    for (size_t k = 0; k < a.stage_parts.size(); ++k)
        if (!same(a.stage_parts[k], b.stage_parts[k])) return false;
    return true;
}

static bool lengths(const std::vector<const Checkpoint*>& points, std::initializer_list<size_t> expected) {
    if (points.size() != expected.size()) return false;
    size_t i = 0;
    for (size_t n : expected)
        if (points[i++]->ids.size() != n) return false;
    return true;
}

static void prefix_checks() {
    const auto held = tokens(80);
    for (size_t stages = 0; stages <= 3; ++stages) {
        const auto good = checkpoint(24, stages);
        CHECK(batch_checkpoint_prefix(good, held, 24, stages));
        CHECK(batch_checkpoint_prefix(good, held, 80, stages));
        CHECK(!batch_checkpoint_prefix(good, held, 23, stages));
        CHECK(!batch_checkpoint_prefix(good, tokens(23), 80, stages));
        CHECK(!batch_checkpoint_prefix(good, {}, 80, stages));
        CHECK(!batch_checkpoint_prefix(good, held, 80, stages + 1));
        CHECK(!batch_checkpoint_prefix(checkpoint(0, stages), held, 80, stages));
        auto changed = good;
        changed.ids[12] = -1;
        CHECK(!batch_checkpoint_prefix(changed, held, 80, stages));
        changed = good;
        changed.imgs.push_back({0, 1});
        CHECK(!batch_checkpoint_prefix(changed, held, 80, stages));
        for (size_t k = 0; k < stages; ++k) {
            changed = good;
            changed.stage_parts[k].ids[12] = -1;
            CHECK(!batch_checkpoint_prefix(changed, held, 80, stages));
            changed = good;
            changed.stage_parts[k].ids.pop_back();
            CHECK(!batch_checkpoint_prefix(changed, held, 80, stages));
            changed = good;
            changed.stage_parts[k].imgs.push_back({0, 2});
            CHECK(!batch_checkpoint_prefix(changed, held, 80, stages));
        }
        // Device-state tags are preserved metadata, not stage/device identity checks.
        changed = good;
        changed.dev_tag = 0;
        for (auto& part : changed.stage_parts) part.dev_tag = 0;
        CHECK(batch_checkpoint_prefix(changed, held, 80, stages));
    }
}

static void budget_and_copy_checks() {
    const auto held = tokens(80);
    for (size_t stages = 0; stages <= 3; ++stages) {
        // Unsorted on purpose. Selection must order references without rewriting the source.
        std::vector<Checkpoint> source;
        for (size_t n : {size_t{56}, size_t{8}, size_t{64}, size_t{24}, size_t{40}})
            source.push_back(checkpoint(n, stages, n));
        source[3].pinned = true;
        const auto before = source;
        CHECK(select_batch_checkpoints(source, held, 64, stages, 0).empty());
        CHECK(lengths(select_batch_checkpoints(source, held, 64, stages, 1), {64}));
        CHECK(lengths(select_batch_checkpoints(source, held, 64, stages, 5), {8, 24, 40, 56, 64}));
        CHECK(lengths(select_batch_checkpoints(source, held, 64, stages, 6), {8, 24, 40, 56, 64}));
        CHECK(lengths(select_batch_checkpoints(source, held, 40, stages, 6), {8, 24, 40}));
        CHECK(lengths(select_batch_checkpoints(source, tokens(25), 64, stages, 6), {8, 24}));
        CHECK(select_batch_checkpoints(source, held, 7, stages, 6).empty());
        CHECK(select_batch_checkpoints({}, held, 80, stages, 6).empty());

        const auto selected = select_batch_checkpoints(source, held, 64, stages, 6);
        std::vector<Checkpoint> copies;
        for (const auto* point : selected) copies.push_back(*point);
        for (size_t i = 0; i < selected.size(); ++i) {
            CHECK(same(copies[i], *selected[i]));
            CHECK(copies[i].ids.data() != selected[i]->ids.data());
            CHECK(copies[i].gdn.data() != selected[i]->gdn.data());
            for (size_t k = 0; k < stages; ++k) {
                CHECK(copies[i].stage_parts[k].ids.data() != selected[i]->stage_parts[k].ids.data());
                CHECK(copies[i].stage_parts[k].gdn.data() != selected[i]->stage_parts[k].gdn.data());
                copies[i].stage_parts[k].gdn[0] = 255;
            }
            copies[i].ids[0] = -1;
            copies[i].ple[0] = 255;
            copies[i].pinned = !copies[i].pinned;
        }
        for (size_t i = 0; i < source.size(); ++i) CHECK(same(source[i], before[i]));
    }
}

static void duplicate_and_borrow_checks() {
    const auto held = tokens(80);
    std::vector<Checkpoint> source = {checkpoint(40), checkpoint(8), checkpoint(24), checkpoint(24, 0, 99)};
    source.back().pinned = true;
    auto borrowed = checkpoint(56, 0, 7);
    const auto donor_before = borrowed;
    auto selected = select_batch_checkpoints(source, held, 64, 0, 6, false, -1, &borrowed);
    CHECK(lengths(selected, {8, 24, 40, 56}));
    CHECK(selected[1] == &source[2]);  // stable, first valid source point wins a duplicate length
    CHECK(selected.back() == &borrowed);
    CHECK(same(borrowed, donor_before));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 1, false, -1, &borrowed), {56}));
    borrowed = checkpoint(24);
    selected = select_batch_checkpoints(source, held, 64, 0, 6, false, -1, &borrowed);
    CHECK(lengths(selected, {8, 24, 40}));
    CHECK(selected[1] == &source[2]);  // a borrowed duplicate must not displace the main chain's point
    source[2].ids[0] = -1;
    selected = select_batch_checkpoints(source, held, 64, 0, 6, false, -1, &borrowed);
    CHECK(selected[1] == &source[3]);  // rejection precedes deduplication
    borrowed.ids[0] = -1;
    CHECK(select_batch_checkpoints({}, held, 64, 0, 6, false, -1, &borrowed).empty());
    borrowed = checkpoint(56);
    CHECK(lengths(select_batch_checkpoints({}, held, 64, 0, 6, false, -1, &borrowed), {56}));
    CHECK(select_batch_checkpoints({}, held, 40, 0, 6, false, -1, &borrowed).empty());
    CHECK(select_batch_checkpoints(source, held, 64, 0, 0, false, -1, &borrowed).empty());
}

static void policy_checks() {
    const auto held = tokens(80);
    std::vector<Checkpoint> source = {checkpoint(8, 0, 10), checkpoint(24, 0, 40), checkpoint(40, 0, 20),
                                      checkpoint(56, 0, 50), checkpoint(64, 0, 30)};
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 3), {8, 24, 56}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 2), {8, 56}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 4, false, 24), {8, 40, 56, 64}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 4, false, 56), {8, 24, 56, 64}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 4, false, 8), {8, 24, 56, 64}));
    source[2].pinned = true;
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 3), {8, 40, 56}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 3, false, 40), {8, 40, 56}));
    for (auto& c : source) c.pinned = true;
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 2), {8, 56}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 1), {64}));
    for (auto& c : source) c.pinned = false;
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 3, true), {8, 56, 64}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 2, true), {8, 64}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 4, true, 56), {8, 24, 40, 64}));
    source[1].pinned = true;
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 3, true), {8, 24, 64}));
    CHECK(lengths(select_batch_checkpoints(source, held, 64, 0, 4, true, 24), {8, 24, 56, 64}));
}

int main() {
    prefix_checks();
    budget_and_copy_checks();
    duplicate_and_borrow_checks();
    policy_checks();
    std::printf("batch_checkpoints_test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
