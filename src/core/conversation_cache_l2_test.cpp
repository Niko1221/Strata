// CPU-only contracts for the PARKED control plane.
// It never allocates or touches K/V; SavedConversation images here are tiny.
#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

using strata::core::ConversationCache;
using strata::core::SavedConversation;

static int checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "conversation_cache_l2_test: FAIL: %s\n", what); std::exit(1); }
}

static SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids.assign(ids.begin(), ids.end());
    s.cvec = cvec;
    return s;
}

int main() {
    ConversationCache cache(1 << 20, 2);
    check(cache.put(image({1, 2, 3})), "put first");
    const uint64_t a = cache.l2_last_serial();
    check(a != 0, "serial assigned");
    cache.l2_set_source(a, "/tmp/l2-slots/a");
    check(cache.l2_set_pinned(a, true), "pin whole entry");
    check(cache.l2_stats().pinned == 1, "stats sees pin");

    auto rows = cache.l2_list();
    check(rows.size() == 1 && rows[0].serial == a, "list serial");
    check(rows[0].tokens == 3 && rows[0].source == "/tmp/l2-slots/a", "list metadata");
    check(rows[0].pinned, "list pin");

    // A pinned whole entry cannot be evicted by a new image in a one-slot cache.
    ConversationCache one(1 << 20, 1);
    check(one.put(image({1, 2})), "one-slot first");
    const uint64_t pinned = one.l2_last_serial();
    check(one.l2_set_pinned(pinned, true), "one-slot pin");
    check(!one.put(image({1, 2, 3})), "pinned entry blocks eviction");
    check(one.l2_set_pinned(pinned, false), "unpin");
    check(one.put(image({1, 2, 3})), "unpinned entry can be evicted");

    check(cache.l2_touch(a), "touch");
    check(cache.l2_drop(a), "drop");
    check(cache.l2_list().empty(), "list empty after drop");
    check(!cache.l2_set_pinned(a, true), "stale serial rejected");

    std::printf("conversation_cache_l2_test: %d checks passed\n", checks);
    return 0;
}
