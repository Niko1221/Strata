// pin_budget_test: the pinned-memory headroom (include/strata/core/pin_budget.hpp) with the simulated ceiling.
// Needs a GPU for the pinned calls; without one it prints SKIP and passes.
#include "strata/core/pin_budget.hpp"
#include "strata/core/pinned.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

#ifdef _WIN32
#define SETENV(k, v) _putenv_s(k, v)
#else
#define SETENV(k, v) setenv(k, v, 1)
#endif

int main(int argc, char** argv) {
    // the singleton reads its environment on first use: set everything first
    SETENV("STRATA_TEST_PIN_BUDGET_MIB", "100");
    SETENV("STRATA_PIN_RESERVE_MIB", "32");   // soft 32 MiB, hard 64 MiB (a quarter of 32 is below the 64 MiB floor)
    using namespace strata::core;
    if (cudaFree(0) != cudaSuccess) {
        std::printf("pin_budget_test: SKIP (no GPU)\n");
        return 0;
    }
    (void) cudaGetLastError();

    // HIP, KFD ceiling: the arena's sliced pin stops when the simulated budget refuses a slice; the HIP build must give
    // ~1 GiB back for code objects.  Here 100 MiB allows six 16 MiB slices, so the 1 GiB back-off drops all six.
    if (argc > 1 && std::strcmp(argv[1], "arena-backoff") == 0) {
        constexpr uint64_t MiB = 1ull << 20;
        std::vector<uint64_t> bounds;
        for (int i = 0; i <= 8; ++i) bounds.push_back((uint64_t) i * 16 * MiB);
        PinnedArena arena(128 * MiB, bounds);
        if (kfd_system_pin_limit_bytes() != 0) {
            CHECK(arena.note.find("KFD system pin limit") != std::string::npos);
            CHECK(arena.note.find("backed off") != std::string::npos);
            CHECK(arena.registered_bytes == 0);
            CHECK(arena.registered_slices == 0);
        } else {
            CHECK(arena.registered_bytes == 96 * MiB);   // not a HIP/Linux build: no back-off
            CHECK(arena.registered_slices == 6);
        }
        std::printf("arena note: %s\n", arena.note.c_str());
        std::printf(g_fail == 0 ? "pin_budget_test arena-backoff: OK\n" : "pin_budget_test arena-backoff: %d FAILED\n",
                    g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    // 1. a registration beyond the ceiling is refused, one inside it works and is charged
    {
        void* big = std::malloc(160u << 20);
        void* small = std::malloc(8u << 20);
        CHECK(pin_register(big, 160u << 20, cudaHostRegisterPortable) == cudaErrorMemoryAllocation);
        CHECK(pin_register(small, 8u << 20, cudaHostRegisterPortable) == cudaSuccess);
        CHECK(pin_detail::st().sim_used == (8u << 20));
        CHECK(pin_unregister(small) == cudaSuccess);
        CHECK(pin_detail::st().sim_used == 0);
        std::free(big);
        std::free(small);
    }
    // 2. without a reserve a refused guarded allocation stays refused
    {
        void* p = nullptr;
        CHECK(host_alloc_guarded(&p, 200u << 20, cudaHostAllocMapped) != cudaSuccess);
        (void) cudaGetLastError();
    }
    // 3. the reserves are held, the arena (here: a registration) takes what is left, a required buffer is then
    //    refused, the held headroom is handed back and the retry works
    {
        pin_reserve_take();
        CHECK(pin_detail::st().hard != nullptr && pin_detail::st().hard_bytes == (64u << 20));
        CHECK(pin_detail::st().soft != nullptr && pin_detail::st().soft_bytes == (32u << 20));
        CHECK(pin_detail::st().sim_used == (96u << 20));
        void* arena = std::malloc(4u << 20);
        CHECK(pin_register(arena, 4u << 20, cudaHostRegisterPortable) == cudaSuccess);   // 100 MiB used: the ceiling
        pin_reserve_release_soft();
        CHECK(pin_detail::st().soft == nullptr);
        CHECK(pin_detail::st().sim_used == (68u << 20));
        void* a = nullptr;
        CHECK(host_alloc_guarded(&a, 16u << 20, cudaHostAllocMapped) == cudaSuccess);   // fits beside the hard reserve
        void* b = nullptr;
        CHECK(host_alloc_guarded(&b, 40u << 20, cudaHostAllocMapped) == cudaSuccess);   // refused first, then the hard reserve goes
        CHECK(pin_detail::st().hard == nullptr);
        CHECK(pin_detail::st().hard_retries == 1);
        void* c = nullptr;
        CHECK(host_alloc_guarded(&c, 60u << 20, cudaHostAllocMapped) != cudaSuccess);   // truly out: reported, not hidden
        (void) cudaGetLastError();
        CHECK(pin_exhausted_hint().find("[pinned-memory-exhausted]") != std::string::npos);
        (void) pin_unregister(arena);
        std::free(arena);
    }
    std::printf(g_fail == 0 ? "pin_budget_test: OK\n" : "pin_budget_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
