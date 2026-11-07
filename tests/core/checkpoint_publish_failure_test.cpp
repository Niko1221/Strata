// Allocation-failure regression for selected-context cache publication. No GPU.
#include "strata/core/checkpoint_allocation.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>

static long fail_after = -1;
void* operator new(std::size_t n) {
    if (fail_after == 0) throw std::bad_alloc();
    if (fail_after > 0) --fail_after;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace strata::core;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
int main() {
    try {
        CHECK(ContextLockPolicy::enabled());
        int faults = 0; bool completed = false;
        for (int n = 0; n < 32 && !completed; ++n) {
            ConversationCache cache(SIZE_MAX, 128);
            // Cross the uint64_t serial deque's allocation boundary on Linux.
            // Also exercises the entry directory allocation and its rollback.
            for (int i = 0; i < 63; ++i) {
                SavedConversation seed; seed.live.ids = {100 + i};
                CHECK(cache.put(std::move(seed)));
            }
            const size_t before = cache.bytes();
            SavedConversation image; image.live.ids = {42}; image.checkpoints.resize(1);
            image.checkpoints[0].ids = {1}; image.checkpoints[0].gdn = {7, 8, 9};
            const auto* backing = image.checkpoints[0].gdn.data();
            std::vector<ConversationCheckpoint> moved(1);
            bool failed = false;
            {
                CapturedCheckpointMoveRollback restore{image, moved, 0, true};
                fail_after = n;
                try {
                    CHECK(cache.put(std::move(image)));
                    restore.on = false; completed = true;
                } catch (const std::bad_alloc&) {
                    fail_after = -1; failed = true; ++faults;
                    CHECK(cache.size() == 63 && cache.bytes() == before);
                    CHECK(image.checkpoints.size() == 1); // publication did not consume ownership
                }
                fail_after = -1;
            }
            if (failed) {
                CHECK(moved[0].ids == std::vector<int32_t>{1});
                CHECK(moved[0].gdn.data() == backing && moved[0].gdn[0] == 7);
                image.checkpoints[0] = std::move(moved[0]);
                CHECK(cache.put(std::move(image))); // directories remain consistent and reusable
            }
            CHECK(cache.size() == 64);
        }
        CHECK(completed && faults >= 2);
        std::printf("cache publication: %d allocation failures preserved checkpoint ownership and cache state\n", faults);
        return 0;
    } catch (const std::exception& e) {
        fail_after = -1; std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
