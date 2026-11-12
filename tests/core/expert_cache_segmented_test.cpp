// #533: the segmented expert cache (--vram-elastic).  A shrink gives the last segments' VRAM back to the driver and
// keeps every slot left exactly as it was (same address, same bytes); a grow maps them again at the same addresses.
// Uniform and sized slots.  Needs a CUDA device with virtual memory management (exits 77 without one).
// The `VRAM` command's residency bookkeeping on such a cache: a refill or a sync that fails part-way leaves a table
// that names only slots holding their expert, uploaded to the device as it is, and the next grow puts the rest back.
#include "strata/core/expert_cache.hpp"
#include "strata/program/vram_rules.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int fail(const char* what, const std::string& err = "") {
    std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    return 1;
}

size_t free_vram() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

// slot i holds the byte (i * 7 + 1) everywhere
bool fill(strata::core::ExpertCache& c, int64_t from, int64_t to, int64_t blob) {
    std::vector<uint8_t> b((size_t) blob);
    for (int64_t i = from; i < to; ++i) {
        std::fill(b.begin(), b.end(), (uint8_t) (i * 7 + 1));
        if (cudaMemcpy(c.device_slot((int32_t) i), b.data(), (size_t) blob, cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
    }
    return true;
}

bool check(const strata::core::ExpertCache& c, int64_t to, int64_t blob) {
    std::vector<uint8_t> b((size_t) blob);
    for (int64_t i = 0; i < to; ++i) {
        if (cudaMemcpy(b.data(), c.device_slot((int32_t) i), (size_t) blob, cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        for (const uint8_t x : b)
            if (x != (uint8_t) (i * 7 + 1)) return false;
    }
    return true;
}

// The `VRAM` command's error paths: a cache of 2 layers x 48 experts, row r in slot r, its blob all (r * 5 + 3).
// After every step the residency table is uploaded as the command's exit does (res_upload), and the device's copy
// must equal the host table, and every row it names resident must sit in a live slot of its own holding its blob.
int vram_error_paths() {
    namespace vr = strata::program::vram_rules;
    constexpr int64_t kBlob = 1 << 20, kExperts = 48, kRows = 2 * kExperts, kSeg = 8ll << 20;
    strata::core::ExpertCache c;
    c.set_segment_bytes(kSeg);
    std::string err;
    if (!c.open(kRows, 2, kExperts, kBlob, err)) return fail("vram: open", err);
    std::vector<std::vector<uint8_t>> blobs((size_t) kRows);
    std::vector<int32_t> host_res((size_t) kRows);
    for (int64_t r = 0; r < kRows; ++r) {
        blobs[(size_t) r].assign((size_t) kBlob, (uint8_t) (r * 5 + 3));
        if (!c.fill_slot_queued((int32_t) r, blobs[(size_t) r].data(), err)) return fail("vram: fill", err);
        host_res[(size_t) r] = (int32_t) r;
    }
    if (!c.sync_queued(err)) return fail("vram: sync", err);
    int32_t* d_res = nullptr;
    if (cudaMalloc((void**) &d_res, (size_t) kRows * sizeof(int32_t)) != cudaSuccess) return fail("vram: cudaMalloc");
    auto settle = [&]() {   // what the command's exit uploads
        return cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess;
    };
    auto consistent = [&](const char* step) -> bool {
        std::vector<int32_t> dev((size_t) kRows);
        if (cudaMemcpy(dev.data(), d_res, dev.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail(step, "read back") == 0;
        if (dev != host_res) return fail(step, "the device's residency differs from the host table") == 0;
        std::vector<int> owner((size_t) c.full_slots(), -1);
        std::vector<uint8_t> b((size_t) kBlob);
        for (int64_t r = 0; r < kRows; ++r) {
            const int32_t s = host_res[(size_t) r];
            if (s < 0) continue;
            if (s >= c.slots()) return fail(step, "a resident row names a slot that is gone") == 0;
            if (owner[(size_t) s] >= 0) return fail(step, "two resident rows share a slot") == 0;
            owner[(size_t) s] = (int) r;
            if (cudaMemcpy(b.data(), c.device_slot(s), (size_t) kBlob, cudaMemcpyDeviceToHost) != cudaSuccess ||
                std::any_of(b.begin(), b.end(), [&](uint8_t x) { return x != (uint8_t) (r * 5 + 3); }))
                return fail(step, "a resident row's slot does not hold its expert") == 0;
        }
        return true;
    };
    auto all_resident = [&]() {
        return std::all_of(host_res.begin(), host_res.end(), [](int32_t s) { return s >= 0; });
    };
    auto pick_miss = [&](int64_t row) -> int64_t {   // the layer's first miss (the engine: its most-routed one)
        const int64_t l = row / kExperts;
        for (int64_t j = l * kExperts; j < (l + 1) * kExperts; ++j)
            if (host_res[(size_t) j] < 0) return j;
        return -1;
    };
    int fills = 0, fail_at = -1;
    bool sync_fails = false;
    auto fill_row = [&](int64_t row, int32_t slot, std::string& e) {
        if (fills++ == fail_at) { e = "injected fill failure"; return false; }
        return c.fill_slot_queued(slot, blobs[(size_t) row].data(), e);
    };
    auto sync = [&](std::string& e) {
        if (!c.sync_queued(e)) return false;
        if (sync_fails) { e = "injected sync failure"; return false; }
        return true;
    };
    auto owns = [](int64_t) { return true; };
    vr::Evicted evicted;
    int rc = 0;
    // a shrink to 32 slots: rows 32.. leave the table
    if (!c.shrink(32ll << 20, err) || c.slots() != 32) rc = fail("vram: shrink", err);
    vr::evict_past(host_res, c.slots(), owns, evicted);
    if (rc == 0 && (!settle() || !consistent("vram: after the shrink") || evicted.size() != (size_t) (kRows - 32)))
        rc = fail("vram: the shrink's evictions");
    // a grow whose fifth copy fails: four rows are back, the failed one and the rest stay owed
    if (rc == 0 && !c.grow(1ll << 40, err)) rc = fail("vram: grow", err);
    fills = 0, fail_at = 4;
    if (rc == 0 && vr::refill(host_res, c.slots(), evicted, pick_miss, fill_row, sync, err))
        rc = fail("vram: the injected fill failure was not reported");
    if (rc == 0 && (!settle() || !consistent("vram: after a failed fill"))) rc = 1;
    if (rc == 0 && evicted.size() != (size_t) (kRows - 32 - 4))
        rc = fail("vram: after a failed fill the rows refilled are owed again, or the rest are not",
                  std::to_string(evicted.size()) + " owed");
    // the next grow puts every row back, each in a slot of its own
    fail_at = -1;
    if (rc == 0 && !vr::refill(host_res, c.slots(), evicted, pick_miss, fill_row, sync, err)) rc = fail("vram: refill", err);
    if (rc == 0 && (!settle() || !consistent("vram: the refill after a failed fill"))) rc = 1;
    if (rc == 0 && (!all_resident() || !evicted.empty())) rc = fail("vram: the refill after a failed fill left rows out");
    // a sync that fails: no row whose copy was in the queue is resident, all of them are owed
    if (rc == 0 && (!c.shrink(32ll << 20, err) || c.slots() != 32)) rc = fail("vram: shrink 2", err);
    if (rc == 0) vr::evict_past(host_res, c.slots(), owns, evicted);
    if (rc == 0 && !c.grow(1ll << 40, err)) rc = fail("vram: grow 2", err);
    sync_fails = true;
    if (rc == 0 && vr::refill(host_res, c.slots(), evicted, pick_miss, fill_row, sync, err))
        rc = fail("vram: the injected sync failure was not reported");
    if (rc == 0 && (!settle() || !consistent("vram: after a failed sync"))) rc = 1;
    if (rc == 0 && (evicted.size() != (size_t) (kRows - 32) ||
                    std::count(host_res.begin(), host_res.end(), strata::core::kNotResident) != kRows - 32))
        rc = fail("vram: after a failed sync a row whose copy was queued is resident or not owed");
    sync_fails = false;
    if (rc == 0 && !vr::refill(host_res, c.slots(), evicted, pick_miss, fill_row, sync, err)) rc = fail("vram: refill 2", err);
    if (rc == 0 && (!settle() || !consistent("vram: the refill after a failed sync"))) rc = 1;
    if (rc == 0 && (!all_resident() || !evicted.empty())) rc = fail("vram: the refill after a failed sync left rows out");
    cudaFree(d_res);
    return rc;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    constexpr int64_t kBlob = 1 << 20, kSlots = 96, kSeg = 8ll << 20;
    for (int sized = 0; sized < 2; ++sized) {
        strata::core::ExpertCache c;
        c.set_segment_bytes(kSeg);
        std::string err;
        const bool ok = sized ? c.open_sized(std::vector<int64_t>(kSlots, kBlob), 4, 64, err)
                              : c.open(kSlots, 4, 64, kBlob, err);
        if (!ok) {
            // #1475: a HIP build has no segmented cache ("--vram-elastic ... is CUDA-only for now"): a skip, not a failure
            if (err.find("virtual memory management") != std::string::npos ||
                err.find("CUDA-only") != std::string::npos) {
                std::printf("%s: skipped\n", err.c_str());
                return 77;
            }
            return fail("open", err);
        }
        if (!c.segmented() || c.slots() != kSlots || c.mapped_bytes() < kSlots * kBlob) return fail("open sizes");
        uint8_t* const first = c.device_slot(0);
        if (!fill(c, 0, kSlots, kBlob)) return fail("fill");
        const size_t before = free_vram();
        // keep 30 MiB: rounded up to whole 8 MiB segments = 32 MiB = 32 slots
        if (!c.shrink(30ll << 20, err)) return fail("shrink", err);
        if (c.slots() != 32 || c.mapped_bytes() != 32ll << 20 || c.bytes() != 32ll << 20)
            return fail("shrunk sizes");
        if (c.full_slots() != kSlots || c.full_bytes() != kSlots * kBlob) return fail("full sizes");
        const size_t freed = free_vram() - before;
        if (freed + (8u << 20) < (size_t) (64ll << 20)) {   // ~64 MiB back to the driver (WDDM rounds a little)
            std::fprintf(stderr, "freed %zu bytes\n", freed);
            return fail("VRAM not given back");
        }
        if (c.device_slot(0) != first || !check(c, 32, kBlob)) return fail("slots kept after the shrink");
        // grow to 70 MiB: whole segments only, 64 MiB = 64 slots
        if (!c.grow(70ll << 20, err) || c.slots() != 64) return fail("grow part", err);
        if (!c.grow(1ll << 40, err) || c.slots() != kSlots || c.mapped_bytes() < kSlots * kBlob)
            return fail("grow all", err);
        if (c.device_slot(0) != first || !check(c, 32, kBlob)) return fail("slots kept after the grow");
        if (!fill(c, 32, kSlots, kBlob) || !check(c, kSlots, kBlob)) return fail("the grown slots");
        if (c.slots_within(0) != 0 || c.slots_within(kBlob * 3 + 5) != 3 || c.bytes_of(5) != 5 * kBlob)
            return fail("slots_within / bytes_of");
        c.close();
        if (c.valid() || c.slots() != 0) return fail("close");
    }
    // the default (no segment size): one allocation, shrink refused
    strata::core::ExpertCache d;
    std::string err;
    if (!d.open(8, 2, 4, kBlob, err) || d.segmented() || d.shrink(0, err)) return fail("unsegmented", err);
    if (vram_error_paths() != 0) return 1;
    std::puts("expert cache segments: shrink gives VRAM back and keeps the slots, grow maps them again; a failed "
              "refill or sync leaves the residency table naming only slots that hold their expert");
    return 0;
}
