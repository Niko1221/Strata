// #6648: the one-pool arithmetic of the Hy4-Flash expert placement on unified memory (no GPU, no model).
#include <cstdint>
#include <cstdio>

#include "hy4_uma_pool.h"

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static const uint64_t GB = 1000000000ull;

int main() {
    using namespace hy4;
    // the reporter's 128 GiB Strix Halo, SQ-128 at 128K: free GPU pool 97.45 GB, all experts 105.76 GB, safe RAM budget about 104 GB
    const uint64_t vram = 97450000000ull, all = 105760000000ull, ram = 104 * GB;
    // discrete card: nothing changes
    CHECK(uma_vram_budget(false, vram, 10 * GB) == vram);
    CHECK(uma_dyn_frac(false, 1.0) == 1.0);
    CHECK(uma_host_room(false, ram, vram, all) == ram);
    // unified memory: the arena is capped by the RAM budget, no adaptive slots, the host gets what is left
    CHECK(uma_vram_budget(true, vram, ram) == vram);
    CHECK(uma_vram_budget(true, 110 * GB, ram) == ram);
    CHECK(uma_dyn_frac(true, 1.0) == 0.0);
    CHECK(uma_dyn_frac(true, 0.35) == 0.0);
    const uint64_t room = uma_host_room(true, ram, vram, all);
    CHECK(room == ram - vram);
    // arena + host-resident experts never exceed the pool (before: 97.45 + 105.76 = 203 GB on a 124 GiB machine)
    CHECK(vram + room <= ram);
    // all experts fit the arena: no host-resident experts are needed, the room is what remains of the pool
    CHECK(uma_host_room(true, ram, 120 * GB, 60 * GB) == ram - 60 * GB);
    // an arena as big as the pool leaves nothing for the host (the cold tail goes to the SSD tier), never a negative size
    CHECK(uma_host_room(true, 50 * GB, 80 * GB, 100 * GB) == 0);
    // a 64 GB machine with a 780M: 40 GB arena, 50 GB budget
    CHECK(uma_host_room(true, 50 * GB, 40 * GB, 62 * GB) == 10 * GB);
    if (failed) return 1;
    std::printf("hy4_uma_pool_test: ok\n");
    return 0;
}
