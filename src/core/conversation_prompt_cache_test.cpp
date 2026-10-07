// CPU-only tests for the system-prompt prefill cache (include/strata/core/conversation_prompt_cache.hpp): a variant
// is a session file keyed by the hash of a system-prompt token prefix, matched from its sidecar, reloaded whole, and
// never removed because the prefix changed - only the GC (space or age, oldest first) removes one.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_prompt_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}
std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}
ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.gdn = bytes_of(4099, seed); c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2); c.dead = bytes_of(64, seed + 3); c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}
SavedConversation sample(size_t tokens) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48; s.cvec = false;
    s.live = checkpoint(tokens, 1);
    s.checkpoints.push_back(checkpoint(tokens / 4, 2));
    s.checkpoints.push_back(checkpoint(tokens / 2, 3));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = 9; kv.idx_dim = 128;
        kv.k = pattern(4096, uint8_t(layer)); kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(512, uint8_t(layer + 20)); kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}
size_t count_sess(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".sess") ++n;
    return n;
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-prompt-cache-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    std::string error;

    // ---- inert: a closed cache writes nothing and answers nothing ----
    ConversationPromptCache off;
    ConversationPromptMatch match;
    check(!off.enabled(), "closed cache disabled");
    check(!off.lookup(123, match), "closed cache has no lookup");
    SavedConversation first = sample(1000);
    check(!off.store(123, first, error), "closed cache refuses a store");
    check(!fs::exists(dir), "closed cache wrote nothing to disk");

    // ---- open, store a variant, hit it ----
    ConversationPromptCache cache;
    check(cache.open(dir, id, 1ull << 30, 2, 0, error), "open");
    check(cache.enabled(), "enabled");
    check(cache.variants() == 0, "empty index");
    std::vector<int64_t> prefix1(first.live.ids.begin(), first.live.ids.end());
    const uint64_t k1 = cache.key_for(prefix1.data(), prefix1.size(), "");
    check(!cache.lookup(k1, match), "no variant yet");
    cache.note_miss(k1);
    check(cache.store(k1, first, error), "store the first variant");
    check(cache.variants() == 1, "one variant");
    check(cache.lookup(k1, match) && match.tokens == (int64_t) first.live.ids.size(), "hit own variant");
    cache.note_hit(k1, match.tokens);
    check(cache.hits() == 1 && cache.misses() == 1, "hit and miss counters");
    check(cache.tokens_saved() == (uint64_t) first.live.ids.size(), "tokens saved counted");
    check(cache.bytes() > 0, "bytes on disk counted");
    check(cache.hash_changes() == 0, "the first store is not a hash change");

    // read it back whole
    SavedConversation back;
    check(cache.load(match.path, back, {}, error), "load the variant");
    check(back.live.ids == first.live.ids, "ids round-trip");
    check(back.kv.size() == first.kv.size(), "kv layer count round-trip");

    // ---- a restart: the variant is a hit again ----
    ConversationPromptCache again;
    check(again.open(dir, id, 1ull << 30, 2, 0, error), "reopen");
    check(again.variants() == 1, "variant survives a reopen");
    check(again.lookup(k1, match) && match.tokens == (int64_t) first.live.ids.size(), "hit after reopen");
    check(again.load(match.path, back, {}, error) && back.live.ids == first.live.ids, "load after reopen");

    // ---- a changed system-prompt prefix: miss + rewrite; the old variant stays ----
    std::vector<int64_t> prefix2 = prefix1;
    prefix2[999] += 1;   // the prefix diverges inside the system prompt
    const uint64_t k2 = again.key_for(prefix2.data(), prefix2.size(), "");
    check(k2 != k1, "a different prefix gives a different key");
    check(!again.lookup(k2, match), "the changed prefix is a miss");
    again.note_miss(k2);
    SavedConversation second = sample(1200);
    check(again.store(k2, second, error), "rewrite as a second variant");
    check(again.variants() == 2, "two variants coexist");
    check(again.hash_changes() == 1, "the hash change is counted");
    check(again.lookup(k1, match), "the old variant is still a hit");
    check(again.lookup(k2, match), "the new variant is a hit");
    // the declared --system-prompt-cache-key enters the key too
    check(again.key_for(prefix1.data(), prefix1.size(), "declared") != k1, "the declared key changes the key");

    // ---- a foreign identity is ignored and never removed ----
    ConversationPromptCache foreign;
    check(foreign.open(dir, SessionFileIdentity{1, 2}, 1ull << 30, 2, 0, error), "open with a foreign identity");
    check(foreign.variants() == 0, "foreign identity indexes nothing");
    check(foreign.foreign_files_kept() >= 1, "foreign identity reported, not removed");
    check(count_sess(dir) >= 2, "another identity's variants are left alone");

    // ---- slots: a third variant evicts the oldest ----
    std::vector<int64_t> prefix3 = prefix1;
    prefix3[998] += 7;
    const uint64_t k3 = again.key_for(prefix3.data(), prefix3.size(), "");
    check(again.store(k3, sample(800), error), "store a third variant");
    check(again.variants() == 2, "slots=2 keeps two variants");
    check(again.evicted_by_space() >= 1, "the oldest variant is evicted by the slot GC");

    // ---- max-age-days 0: no deletion by time ----
    ConversationPromptCache age0;
    check(age0.open(dir, id, 1ull << 30, 2, 0, error), "open with max-age 0");
    const size_t before_age = age0.variants();
    check(before_age >= 1, "variants present before the age test");
    check(age0.evicted_by_age() == 0, "max-age 0 deletes nothing by time");
    check(age0.variants() == before_age, "max-age 0 keeps every variant");

    // ---- max-age-days positive: prune the old ones, oldest first ----
    const auto old = fs::file_time_type::clock::now() - std::chrono::hours(24 * 10);
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".sess") fs::last_write_time(e.path(), old);
    ConversationPromptCache age7;
    check(age7.open(dir, id, 1ull << 30, 2, 7, error), "open with max-age 7 days");
    check(age7.evicted_by_age() >= 1, "variants older than the limit are pruned");
    check(age7.variants() == 0, "every old variant went");

    // ---- a variant larger than the whole budget is KEPT and reported ----
    const fs::path tiny_dir = dir.string() + "-tiny";
    fs::remove_all(tiny_dir);
    ConversationPromptCache tiny;
    check(tiny.open(tiny_dir, id, 4096, 2, 0, error), "open with a tiny budget");
    SavedConversation big = sample(1000);
    const uint64_t tk = tiny.key_for(prefix1.data(), prefix1.size(), "");
    check(tiny.store(tk, big, error), "store a variant over the budget");
    check(tiny.variants() == 1, "the oversized variant is indexed, not dropped");
    check(tiny.bytes() > 4096, "it is larger than the budget");
    check(tiny.oversized_files_kept() >= 1, "reported as oversized");
    ConversationPromptCache tiny_again;
    check(tiny_again.open(tiny_dir, id, 4096, 2, 0, error), "reopen the tiny budget");
    check(tiny_again.variants() == 1, "the oversized variant survives a reopen");
    check(tiny_again.oversized_files_kept() >= 1, "reported oversized on reopen");

    fs::remove_all(dir);
    fs::remove_all(tiny_dir);
    std::printf("conversation_prompt_cache_test: %d checks passed\n", checks);
    return 0;
}
