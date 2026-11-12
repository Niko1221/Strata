// CPU-only checks. Linux also exercises the real store's reads, mapping lifecycle and budget.
#include "ds4_store_pages.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

#if defined(__linux__)
#include "ds4_store.hpp"
#endif

using namespace strata::ds4::store_pages;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)

static void layout_test() {
    Layout x;
    CHECK(!layout(0, 8, kPage, x));
    CHECK(!layout(4096, 0, kPage, x));
    CHECK(!layout(kMaxRegistration + 1, 8, kMaxRegistration, x));
    CHECK(!layout(4096, 8, kPage - 1, x));
    CHECK(layout(4096, 8, kPage, x) && x.slots == 8 && x.arena_bytes == kPage);
    CHECK(layout((1u << 20) + 4096, 8, 14u << 20, x));
    CHECK(x.chunk_slots == 8 && x.chunk_bytes == (10u << 20) && x.slots == 11 && x.arena_bytes == (14u << 20));
    CHECK(layout(kMaxRegistration, 4096, kMaxRegistration, x));
    CHECK(x.chunk_slots == 1 && x.slots == 1 && x.arena_bytes == kMaxRegistration);
    CHECK(!layout(kPage, 8, std::numeric_limits<size_t>::max(), x));

    // Independent slot-by-slot packing oracle: padding, the short tail, and the
    // largest registration all count against the budget. No memory is allocated.
    std::mt19937 rng(41);
    for (int trial = 0; trial < 10000; ++trial) {
        const size_t slot = (1 + rng() % 5120) * 4096ull;
        const size_t requested = 8 + rng() % 4090;
        const size_t budget = rng() % (128u << 20);
        size_t chunk_slots = 0;
        while (chunk_slots < requested && (chunk_slots + 1) * slot <= kMaxRegistration) ++chunk_slots;
        const size_t chunk = ((chunk_slots * slot + kPage - 1) / kPage) * kPage;
        size_t count = 0, used = 0;
        for (;;) {
            const size_t n = count + 1;
            const size_t full = n / chunk_slots, tail = n % chunk_slots;
            const size_t bytes = full * chunk + ((tail * slot + kPage - 1) / kPage) * kPage;
            if (bytes > budget) break;
            count = n;
            used = bytes;
        }
        CHECK(layout(slot, requested, budget, x) == (count != 0));
        if (count) {
            CHECK(x.slots == count && x.arena_bytes == used && x.arena_bytes <= budget);
            CHECK(x.chunk_slots == chunk_slots && x.chunk_bytes == chunk && chunk <= kMaxRegistration);
        }
    }
    std::puts("PASS: layout, budget, registration bound and 10000 packing cases");
}

#if defined(__linux__)
using strata::ds4::ExpertStore;
namespace os = strata::ds4::os;

static void store_test(bool huge, bool fail_map) {
    CHECK(setenv("DS4_STORE_CHUNK_SLOTS", "8", 1) == 0);
    CHECK(setenv("DS4_STORE_PREGROW", "1", 1) == 0);
    ExpertStore::LayerGeo geo;
    geo.n_exp = 16;
    geo.blob = (1u << 20) - 12288;   // slot size: 1 MiB + 4 KiB, forces chunk padding
    for (int r = 0; r < 3; ++r) {
        geo.file[r] = 0;
        geo.off[r] = 37 + r * 65536;
        geo.stride[r] = (2u << 20) + 17;
        geo.tr[r] = r * 65536;
        geo.bytes[r] = r < 2 ? 12289 + r * 4098 : geo.blob - geo.tr[r];
    }
    std::vector<uint8_t> source(geo.off[2] + (geo.n_exp - 1) * geo.stride[2] + geo.bytes[2]);
    for (size_t i = 0; i < source.size(); ++i) source[i] = (uint8_t) (i * 37 + (i >> 12));
    FILE* f = std::tmpfile();
    CHECK(f != nullptr);
    CHECK(std::fwrite(source.data(), 1, source.size(), f) == source.size());
    CHECK(std::fflush(f) == 0);

    struct Mapping { void* host; size_t bytes; std::unique_ptr<uint8_t[]> alias; bool live; };
    std::vector<Mapping> mappings;
    size_t mapped = 0, unmapped = 0;
    ExpertStore::Cfg cfg;
    cfg.threads = 2;
    cfg.budget_bytes = 14u << 20;
    cfg.floor_bytes = 0;
    cfg.huge_pages = huge;
    cfg.map_chunk = [&](void* p, size_t bytes) -> void* {
        if (huge) {
            CHECK((uintptr_t) p % kPage == 0 && bytes % kPage == 0 && bytes <= kMaxRegistration);
            // Every byte has been faulted and cleared before registration.
            for (size_t i = 0; i < bytes; ++i) CHECK(((uint8_t*) p)[i] == 0);
        }
        if (fail_map) return nullptr;
        mappings.push_back({p, bytes, std::unique_ptr<uint8_t[]>(new uint8_t[bytes]), true});
        ++mapped;
        return mappings.back().alias.get();   // deliberately not the host address
    };
    cfg.unmap_chunk = [&](void* p) {
        for (auto& m : mappings) if (m.host == p && m.live) { m.live = false; ++unmapped; return; }
        CHECK(false);
    };
    {
        ExpertStore store;
        CHECK(store.init(cfg, {geo}, {os::kBadFd}, {(os::fd_t) fileno(f)}));
        const size_t expected_slots = huge ? 11 : cfg.budget_bytes / store.slot_bytes();
        CHECK(store.slots_max() == expected_slots && store.slots_allocated() == expected_slots);
        for (int cycle = 0; cycle < 2; ++cycle) {
            std::vector<int> experts(expected_slots);
            for (size_t e = 0; e < experts.size(); ++e) experts[e] = (int) e;
            std::vector<char> ok;
            CHECK(store.ensure(0, experts.data(), (int) experts.size(), ok));
            for (int e : experts) {
                const int s = store.find(0, e);
                CHECK(s >= 0 && ok[e]);
                for (int r = 0; r < 3; ++r)
                    CHECK(std::memcmp(store.mat(s, r), source.data() + geo.off[r] + e * geo.stride[r], geo.bytes[r]) == 0);
                if (fail_map) CHECK(store.dev_blob(s) == 0);
                else {
                    bool found = false;
                    for (const auto& m : mappings) {
                        const uintptr_t offset = (uintptr_t) store.blob(s) - (uintptr_t) m.host;
                        if (m.live && offset < m.bytes) {
                            CHECK(store.dev_blob(s) == (uintptr_t) m.alias.get() + offset);
                            found = true;
                        }
                    }
                    CHECK(found);
                }
            }
            CHECK(store.shrink_chunks(99) == 0);   // pinned readers keep their chunks
            store.release(0, experts.data(), (int) experts.size(), ok);
            if (cycle == 0) {
                CHECK(store.shrink_chunks(99) == 2);   // unregister before decommit; next cycle grows again
                CHECK(store.slots_allocated() == 0 && mapped == unmapped);
            }   // the second set of mappings is released by the destructor
        }
    }
    CHECK(mapped == unmapped);
    CHECK(std::fclose(f) == 0);
    std::printf("PASS: store huge=%d, map-failure=%d, exact matrix bytes, aliases, shrink/regrow and cleanup\n", huge, fail_map);
}
#endif

int main() {
    layout_test();
#if defined(__linux__)
    store_test(false, false);
    store_test(true, false);
    store_test(true, true);
#else
    std::puts("SKIP: Linux store mmap, prefault and I/O lifecycle checks");
#endif
}
