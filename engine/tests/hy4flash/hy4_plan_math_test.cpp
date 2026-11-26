// #6744: the Hy4-Flash work arena's score buffer (no GPU, no model). The numbers are the reporter's / Boston's allocation trace.
#include <cstdint>
#include <cstdio>

#include "hy4_plan_math.h"

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    using namespace hy4;
    // the trace on an RTX A6000 at --max-context 1048576, chunk 8192: dalloc of 8,590,196,736 bytes right after the experts
    CHECK(score_buffer_bytes(8192, 1048576) == 8590196736ull);
    CHECK(score_buffer_bytes(8192, 524288) == 4295229440ull);                  // 512K: 4,295,229,440 (also in the trace)
    CHECK(score_buffer_bytes(4096, 1048576) == 4295098368ull);                 // the R9700's chunk of 4096 at 1M
    // at 128K it is small, so the old plan was right there: the activation scratch is the larger term
    const uint64_t act = 2ull * 8192 * 16384 * 4, moe = 8192ull * 16 * 1536 * 6;
    CHECK(score_buffer_bytes(8192, 131072) < moe);
    CHECK(work_arena_bytes(act, 8192, 131072, moe) == moe);
    // from 256K up the score buffer is the arena and the plan has to reserve it: 1M at chunk 8192 adds 7.4 GB over the old estimate
    CHECK(work_arena_bytes(act, 8192, 1048576, moe) == 8590196736ull);
    CHECK(work_arena_bytes(act, 8192, 1048576, moe) > moe + 7000000000ull);
    CHECK(work_arena_bytes(act, 8192, 262144, moe) == 2147745792ull);
    if (failed) return 1;
    std::printf("hy4_plan_math_test: ok\n");
    return 0;
}
