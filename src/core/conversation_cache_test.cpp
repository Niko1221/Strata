#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
}

int main() {
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        struct Evictions { size_t calls = 0, bytes = 0; int32_t first = 0; } evicted;
        const auto spill = [](void* user, const SavedConversation& image) noexcept {
            auto& record = *static_cast<Evictions*>(user);
            ++record.calls; record.bytes = image.bytes(); record.first = image.live.ids.front();
        };
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one * 2, 2, spill, &evicted);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(evicted.calls == 0, "parking within budget does not spill");
        auto held = cache.take(0);
        check(evicted.calls == 0, "promotion does not spill");
        check(cache.make_room(one, held.bytes()), "reserve while holding incoming image");
        check(evicted.calls == 1 && evicted.first == 9 && evicted.bytes == one, "eviction borrows complete image before release");
        cache.put(image({5,6,7}), held.bytes());
        check(evicted.calls == 1, "non-evicting insert after reservation does not spill again");
        check(!cache.make_room(one * 3), "oversized reservation rejected before callback");
        check(evicted.calls == 1, "oversized reservation does not spill");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one * 3, 2);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.make_staging_room(one), "disk staging fits remaining byte budget");
        check(cache.size() == 2 && cache.evictions() == 0, "staging does not consume a parked slot");
        check(cache.make_staging_room(one * 2), "larger disk staging evicts for bytes");
        check(cache.size() == 1 && cache.bytes() + one * 2 <= one * 3, "disk staging shares total RAM bound");
        check(!cache.make_staging_room(one * 3 + 1) && cache.size() == 1, "oversize staging leaves retained images alone");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
