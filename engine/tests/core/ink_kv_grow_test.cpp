// strata-inkling CUDA: when the global K/V may grow in VRAM (strata/core/ink_kv_grow.hpp, #6770 / #6755).
// A model of the growth loop: the card has room for the K/V of the whole planned context after the experts are in; every
// 8K step is one request for more. The old check (free memory against the WHOLE new K/V) stopped at about half of the context
// with a gigabyte free; the new one (net growth plus one held old visit) reaches the full context.
#include "strata/core/ink_kv_grow.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr size_t kPerPos = 4096;            // K+V of one visit for one position (8 heads x 128 dim x 2 x 2 B)
constexpr size_t kFloor = 256ull << 20;

// the grow rule as it was before the fix (the whole new K/V against the free memory)
bool old_fits(size_t vfree, int n_glob, int slots, int new_cap) {
    return vfree > kFloor + (size_t)n_glob * slots * (size_t)new_cap * kPerPos;
}

// returns the context the K/V reached when requests grow it by 1 position beyond the capacity each time
template <typename Fits>
int run(size_t free_after_alloc_start, int n_glob, int slots, int start_cap, int max_ctx, Fits fits) {
    size_t vfree = free_after_alloc_start;
    int cap = start_cap;
    while (cap < max_ctx) {
        const int next = ink_kv_next_cap(cap, cap + 1, max_ctx);
        if (!fits(vfree, n_glob, slots, cap, next)) return cap;
        vfree -= (size_t)n_glob * slots * (size_t)(next - cap) * kPerPos;     // the net growth stays in the card
        cap = next;
    }
    return cap;
}
}  // namespace

int main() {
    // next capacity: whole 8K steps, covers the request, never past the context, 64-aligned
    CHECK(ink_kv_next_cap(8192, 8193, 131072) == 16384);
    CHECK(ink_kv_next_cap(8192, 20000, 131072) == 20032);
    CHECK(ink_kv_next_cap(120000, 120001, 131072) == 128064 || ink_kv_next_cap(120000, 120001, 131072) % 64 == 0);
    CHECK(ink_kv_next_cap(131000, 131001, 131072) == 131072);
    CHECK(ink_kv_next_cap(8192, 8193, 8192 + 64) == 8256);

    // peak bytes: net growth of all visits plus one held old visit; zero when nothing grows
    CHECK(ink_kv_grow_peak_bytes(8, 1, 8192, 16384, kPerPos) == 8ull * 8192 * kPerPos + 8192ull * kPerPos);
    CHECK(ink_kv_grow_peak_bytes(8, 2, 8192, 16384, kPerPos) == 2 * (8ull * 8192 * kPerPos + 8192ull * kPerPos));
    CHECK(ink_kv_grow_peak_bytes(0, 1, 8192, 16384, kPerPos) == 0);
    CHECK(ink_kv_grow_peak_bytes(8, 1, 16384, 16384, kPerPos) == 0);

    // the plan left room for the whole 131072-token K/V (8 visits), the one visit that is held twice while it grows
    // (the planner's growth margin), the floor and 64 MiB spare
    const int max_ctx = 131072;
    const int n_glob = 8;
    const size_t one_visit = (size_t)max_ctx * kPerPos;
    const size_t room = (size_t)n_glob * max_ctx * kPerPos + one_visit + kFloor + (64ull << 20);
    const int start = 8192;
    const size_t free_start = room - (size_t)n_glob * start * kPerPos;       // after the first 8K are in the card
    const int reached_old = run(free_start, n_glob, 1, start, max_ctx, [](size_t f, int g, int s, int, int nx) { return old_fits(f, g, s, nx); });
    const int reached_new = run(free_start, n_glob, 1, start, max_ctx, [](size_t f, int g, int s, int oc, int nx) {
        return ink_kv_grow_fits(f, kFloor, g, s, oc, nx, kPerPos); });
    std::printf("old rule reached %d of %d tokens, new rule %d\n", reached_old, max_ctx, reached_new);
    CHECK(reached_old < max_ctx * 6 / 10);        // the bug: stops at about half
    CHECK(reached_old > max_ctx * 4 / 10);
    CHECK(reached_new == max_ctx);                // the fix: the whole planned context

    // a card with room for only 3/4 of the context: the new rule gets as far as the room allows (not less than the old one)
    const size_t room34 = (size_t)n_glob * (max_ctx * 3 / 4) * kPerPos + kFloor + (64ull << 20);
    const size_t free34 = room34 - (size_t)n_glob * start * kPerPos;
    const int old34 = run(free34, n_glob, 1, start, max_ctx, [](size_t f, int g, int s, int, int nx) { return old_fits(f, g, s, nx); });
    const int new34 = run(free34, n_glob, 1, start, max_ctx, [](size_t f, int g, int s, int oc, int nx) {
        return ink_kv_grow_fits(f, kFloor, g, s, oc, nx, kPerPos); });
    CHECK(new34 > old34);
    CHECK(new34 > max_ctx * 6 / 10 && new34 <= max_ctx * 3 / 4 + 8192);

    // target: the full step when it fits, the tight one when only that fits, 0 when nothing fits
    const size_t plenty = (size_t)n_glob * 40000 * kPerPos + kFloor;
    CHECK(ink_kv_grow_target(plenty, kFloor, n_glob, 1, 8192, 8193, max_ctx, kPerPos) == 16384);
    const size_t tight_only = kFloor + ink_kv_grow_peak_bytes(n_glob, 1, 8192, 8448, kPerPos) + 1;
    CHECK(ink_kv_grow_target(tight_only, kFloor, n_glob, 1, 8192, 8193, max_ctx, kPerPos) == 8448);
    CHECK(ink_kv_grow_target(kFloor, kFloor, n_glob, 1, 8192, 8193, max_ctx, kPerPos) == 0);
    CHECK(ink_kv_grow_target(0, kFloor, n_glob, 1, 8192, 8193, max_ctx, kPerPos) == 0);

    // #6905: when the card cannot grow the K/V and the rest goes to host RAM, the host tail is the WHOLE rest of the context (so no later, longer request
    // needs to grow it again), and only if that does not fit in host memory just what the request needs
    CHECK(ink_kv_host_tail_full(40960, 65536) == 24576);       // the reporter: the card stopped at 40960 of a 64K context
    CHECK(ink_kv_host_tail_need(40960, 49152) == 8192);
    CHECK(ink_kv_host_tail_full(40960, 65536) > ink_kv_host_tail_need(40960, 41984));
    CHECK(ink_kv_host_tail_full(65536, 65536) == 0);           // nothing left to map
    CHECK(ink_kv_host_tail_full(70000, 65536) == 0);           // never negative
    CHECK(ink_kv_host_tail_need(8192, 8192) == 0);
    // a K/V that grew to its host tail serves every position up to max_ctx: dev + full tail == max_ctx
    for (int dev : {256, 8192, 32768, 40960, 65471}) CHECK(dev + ink_kv_host_tail_full(dev, 65536) == 65536);

    if (failed) {
        std::fprintf(stderr, "ink_kv_grow_test: %d failed\n", failed);
        return 1;
    }
    std::printf("ink_kv_grow_test: ok\n");
    return 0;
}
