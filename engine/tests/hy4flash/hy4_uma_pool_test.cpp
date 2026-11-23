// #6648: the one-pool arithmetic of the Hy4-Flash expert placement on unified memory (no GPU, no model).
#include <cstdint>
#include <cstdio>

#include "hy4_uma_pool.h"

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static const uint64_t GB = 1000000000ull;

int main() {
    using namespace hy4;
    // the reporter's 128 GiB Strix Halo on Linux, SQ-128 at 128K: free GPU pool 97.45 GB (GTT), all experts 105.76 GB, safe RAM budget about 104 GB,
    // a 0.5 GB BIOS carve-out
    const uint64_t vram = 97450000000ull, all = 105760000000ull, ram = 104 * GB, ded = GB / 2;
    // discrete card: nothing changes
    CHECK(uma_vram_budget(false, vram, 10 * GB, 0) == vram);
    CHECK(uma_dyn_frac(false, 1.0, vram, 0) == 1.0);
    CHECK(uma_host_room(false, ram, vram, 0, all) == ram);
    // unified memory (GTT): the arena is capped by dedicated + the RAM budget, no adaptive slots beyond the carve-out, the host gets what is left
    CHECK(uma_vram_budget(true, vram, ram, ded) == vram);
    CHECK(uma_vram_budget(true, 110 * GB, ram, ded) == ram + ded);
    CHECK(uma_dyn_frac(true, 1.0, vram, ded) == 0.0);
    CHECK(uma_dyn_frac(true, 0.35, vram, ded) == 0.0);
    const uint64_t room = uma_host_room(true, ram, vram, ded, all);
    CHECK(room == ram - (vram - ded));
    // the RAM-backed arena and the host-resident experts never exceed the pool (before: 97.45 + 105.76 = 203 GB on a 124 GiB machine)
    CHECK((vram - ded) + room <= ram);
    // all experts fit the arena: the host-resident experts get the rest of the pool
    CHECK(uma_host_room(true, ram, 120 * GB, ded, 60 * GB) == ram - (60 * GB - ded));
    // an arena as big as the pool leaves nothing for the host (the cold tail goes to the SSD tier), never a negative size
    CHECK(uma_host_room(true, 50 * GB, 80 * GB, 0, 100 * GB) == 0);
    // a 64 GB machine with a 780M: 0.5 GB carve-out, 40 GB arena, 50 GB budget
    CHECK(uma_host_room(true, 50 * GB, 40 * GB, ded, 62 * GB) == 50 * GB - (40 * GB - ded));
    // a Strix Halo whose BIOS gives the GPU 112 GB of dedicated memory (Windows, 31.6 GB left to the system): the arena is inside the carve-out,
    // so nothing changes: no cap, the adaptive slots stay, the host-resident experts keep the whole RAM budget
    const uint64_t ded_win = 111800000000ull, ram_win = 20 * GB;
    CHECK(uma_vram_budget(true, 97 * GB, ram_win, ded_win) == 97 * GB);
    CHECK(uma_dyn_frac(true, 1.0, 97 * GB, ded_win) == 1.0);
    CHECK(uma_host_room(true, ram_win, 97 * GB, ded_win, all) == ram_win);
    if (failed) return 1;
    std::printf("hy4_uma_pool_test: ok\n");
    return 0;
}
