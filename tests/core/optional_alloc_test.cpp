// NativeDense::load's allocation order under a VRAM budget, through the sequencer load itself uses
// (strata::core::RequiredThenOptional, strata/core/optional_alloc.hpp).  A fake device, no GPU, no model.  `load_like`
// makes the calls in load's order: for each matrix `required` (its upload) and then `want` (its STRATA_Q8_PACKED copy),
// then `required` for the q8_1 scratch, then `finish`.  Its steps classify a failed allocation as load's do: out of memory
// for a copy is AllocTry::out_of_memory, any other copy failure (an upload) AllocTry::failed.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "strata/core/optional_alloc.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {
constexpr uint64_t MiB = 1ull << 20;

struct FakeDevice {
    uint64_t budget, used = 0;
    int fail_upload_of_copy = -1;   // >= 0: the upload of that copy fails (allocation fits, the copy to it does not)
    int copies_tried = 0;           // how many optional steps ran
    explicit FakeDevice(uint64_t b) : budget(b) {}
    bool alloc(uint64_t bytes) {
        if (bytes > budget - used) return false;
        used += bytes;
        return true;
    }
};

struct Item {                 // load's Pending, reduced to what the order decides
    uint64_t bytes;
    bool wants_copy;
    bool has_copy = false;    // load registers the packed copy (native_q8_0_packed_register) only when it was made
};

struct Loaded {
    bool ok = false;
    OptionalPlaced copies;
};

Loaded load_like(FakeDevice& dev, std::vector<Item>& items, uint64_t scratch) {
    RequiredThenOptional plan;
    Loaded out;
    for (size_t k = 0; k < items.size(); ++k) {
        if (!plan.required([&] { return dev.alloc(items[k].bytes); })) return out;
        if (items[k].wants_copy) {
            const int copy = (int) plan.wanted();
            plan.want([&, k, copy] {
                ++dev.copies_tried;
                if (!dev.alloc(items[k].bytes)) return AllocTry::out_of_memory;
                if (copy == dev.fail_upload_of_copy) return AllocTry::failed;
                items[k].has_copy = true;
                return AllocTry::ok;
            });
        }
    }
    if (!plan.required([&] { return dev.alloc(scratch); })) return out;
    out.copies = plan.finish();
    out.ok = !out.copies.failed;
    return out;
}

std::vector<Item> matrices(size_t n, uint64_t bytes, bool copies = true) {
    return std::vector<Item>(n, Item{bytes, copies});
}

// The order before the fix, kept as the counterexample: each matrix's packed copy right after it, the first copy that
// does not fit dropped and packing stopped, the copies already made kept.
bool interleaved_order(FakeDevice& dev, const std::vector<Item>& items, uint64_t scratch) {
    bool pack_off = false;
    for (const Item& m : items) {
        if (!dev.alloc(m.bytes)) return false;
        if (!pack_off && !dev.alloc(m.bytes)) pack_off = true;
    }
    return dev.alloc(scratch);
}
}  // namespace

int main() {
    const uint64_t scratch = MiB / 2;
    {   // the round-2 review's case: ~13 MiB free, three 4 MiB matrices.  Interleaved: A, packed A, B fit, packed B does
        // not, and then C does not fit either, although the three matrices alone do.
        auto items = matrices(3, 4 * MiB);
        FakeDevice before(13 * MiB);
        CHECK(!interleaved_order(before, items, scratch));
        FakeDevice dev(13 * MiB);
        const Loaded r = load_like(dev, items, scratch);
        CHECK(r.ok && r.copies.placed == 0 && r.copies.out_of_memory);
        CHECK(dev.used == 12 * MiB + scratch);
        CHECK(dev.copies_tried == 1);   // the first copy did not fit: the other two were not tried
        for (const Item& m : items) CHECK(!m.has_copy);
    }
    {   // room for one copy: the first is placed and registered, the second stops the rest (they keep the GGUF layout)
        auto items = matrices(3, 4 * MiB);
        FakeDevice dev(17 * MiB);
        const Loaded r = load_like(dev, items, scratch);
        CHECK(r.ok && r.copies.placed == 1 && r.copies.out_of_memory && !r.copies.failed);
        CHECK(dev.used == 16 * MiB + scratch && dev.copies_tried == 2);
        CHECK(items[0].has_copy && !items[1].has_copy && !items[2].has_copy);
    }
    {   // room for all: every copy, no out-of-memory
        auto items = matrices(3, 4 * MiB);
        FakeDevice dev(64 * MiB);
        const Loaded r = load_like(dev, items, scratch);
        CHECK(r.ok && r.copies.placed == 3 && !r.copies.out_of_memory);
        CHECK(dev.used == 24 * MiB + scratch);
        for (const Item& m : items) CHECK(m.has_copy);
    }
    {   // only some matrices want a copy: copy indices follow the wanting matrices
        std::vector<Item> items{{4 * MiB, false}, {2 * MiB, true}, {4 * MiB, false}, {2 * MiB, true}};
        FakeDevice dev(15 * MiB);   // 12 MiB + scratch required, then room for one 2 MiB copy
        const Loaded r = load_like(dev, items, scratch);
        CHECK(r.ok && r.copies.placed == 1 && r.copies.out_of_memory);
        CHECK(items[1].has_copy && !items[3].has_copy && !items[0].has_copy && !items[2].has_copy);
    }
    {   // the required allocations alone do not fit: the load fails before any copy is tried
        auto items = matrices(3, 4 * MiB);
        FakeDevice dev(12 * MiB);   // the scratch does not fit
        CHECK(!load_like(dev, items, scratch).ok);
        CHECK(dev.copies_tried == 0);
        auto items2 = matrices(3, 4 * MiB);
        FakeDevice dev2(9 * MiB);   // the third matrix does not fit
        CHECK(!load_like(dev2, items2, scratch).ok);
        CHECK(dev2.copies_tried == 0);
    }
    {   // an upload failure of a copy is an error (load fails), not taken for out of memory; later copies are not tried
        auto items = matrices(4, 4 * MiB);
        FakeDevice dev(64 * MiB);
        dev.fail_upload_of_copy = 1;
        const Loaded r = load_like(dev, items, scratch);
        CHECK(!r.ok && r.copies.failed && !r.copies.out_of_memory && r.copies.placed == 1);
        CHECK(dev.copies_tried == 2);
    }
    {   // no copies wanted
        auto items = matrices(2, 4 * MiB, false);
        FakeDevice dev(9 * MiB);
        const Loaded r = load_like(dev, items, scratch);
        CHECK(r.ok && r.copies.placed == 0 && !r.copies.out_of_memory && dev.copies_tried == 0);
    }
    {   // a required allocation after finish() is refused and not run; finish() runs the copies once
        RequiredThenOptional plan;
        int runs = 0, copies = 0;
        CHECK(plan.required([&] { ++runs; return true; }));
        plan.want([&] { ++copies; return AllocTry::ok; });
        CHECK(copies == 0);   // queued, not run
        CHECK(plan.finish().placed == 1 && copies == 1 && plan.placed(0));
        CHECK(!plan.required([&] { ++runs; return true; }) && runs == 1 && plan.late_required());
        CHECK(plan.finish().placed == 1 && copies == 1);
    }
    {   // place_optional: out of memory stops at that copy, the ones after it are not tried
        int tried = 0;
        const OptionalPlaced r = place_optional(5, [&](size_t i) { ++tried; return i == 2 ? AllocTry::out_of_memory : AllocTry::ok; });
        CHECK(r.placed == 2 && r.out_of_memory && !r.failed && tried == 3);
    }
    std::printf(failed ? "optional_alloc_test: %d FAILED\n" : "optional_alloc_test: OK\n", failed);
    return failed ? 1 : 0;
}
