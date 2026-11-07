#include <cstdlib>
#include <cstdio>
#include <new>
#include <utility>
#include "strata/core/conversation_cache.hpp"

// Standalone, single-threaded CPU ownership probe. No production instrumentation.
struct Record { void* ptr; size_t size; };
static Record records[128]{};
static bool recording = false;
static size_t live_bytes = 0, frees = 0;
void* operator new(size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    if (recording && n >= 65536) {
        bool saved = false;
        for (auto& r : records) if (!r.ptr) { r = {p,n}; live_bytes += n; saved = true; break; }
        if (!saved) std::abort();
    }
    return p;
}
void operator delete(void* p) noexcept {
    for (auto& r : records) if (r.ptr && r.ptr == p) { live_bytes -= r.size; r = {}; ++frees; break; }
    std::free(p);
}
void operator delete(void* p, size_t) noexcept { ::operator delete(p); }
void* operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }

using namespace strata::core;
static void require(bool x) { if (!x) std::abort(); }
static SavedConversation snapshot() {
    SavedConversation s;
    s.live.ids = {1,2,3};
    s.live.gdn.resize(1<<20, 7);
    s.checkpoints.resize(1);
    s.checkpoints[0].gdn.resize(1<<20, 8);
    s.stage_images.resize(1);
    s.stage_images[0].live.gdn.resize(1<<20, 9);
    s.kv.resize(1);
    s.kv[0].k.resize(1<<20, 10);
    s.kv[0].v.resize(1<<20, 11);
    return s;
}
int main() {
    for (int mode = 0; mode < 3; ++mode) {
        require(live_bytes == 0);
        ConversationCache cache(8<<20, 4);
        recording = true;
        auto s = snapshot();
        require(cache.put(std::move(s)));
        recording = false;
        const size_t before = live_bytes, free_before = frees;
        require(before == 5*(1<<20));
        if (mode == 0) {
            require(cache.evict_oldest());
            require(cache.bytes() == 0 && cache.size() == 0 && live_bytes == 0);
        } else if (mode == 1) {
            cache.clear();
            require(cache.bytes() == 0 && live_bytes == 0);
        } else {
            auto held = cache.take(0);
            require(cache.bytes() == 0 && live_bytes == before);
            held = {};
            require(live_bytes == 0);
        }
        std::printf("mode=%d tracked_before=%zu tracked_after=%zu free_calls=%zu PASS\n",
                    mode,before,live_bytes,frees-free_before);
    }
}
