// CPU-only tests for the conversation cache's disk tier (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_disk.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
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

std::vector<int32_t> tokens(size_t n, int32_t base) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + int32_t(i % 1000);
    return v;
}

ConversationCheckpoint checkpoint(const std::vector<int32_t>& ids, size_t n, uint8_t seed) {
    ConversationCheckpoint c;
    c.ids.assign(ids.begin(), ids.begin() + (std::ptrdiff_t) n);
    c.gdn.assign(4099, seed);
    c.ple.assign(77, uint8_t(seed + 1));
    c.tails.assign(301, uint8_t(seed + 2));
    c.dead.assign(64, uint8_t(seed + 3));
    c.block_pos.assign(8, uint8_t(seed + 4));
    c.used = seed;
    return c;
}

// one stage's image of a conversation of `ids` with a checkpoint at `at`
SavedConversation stage_image(const std::vector<int32_t>& ids, size_t at, int64_t lo, int64_t hi, uint8_t seed) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo; s.layer_hi = hi;
    s.live = checkpoint(ids, ids.size(), seed);
    s.checkpoints.push_back(checkpoint(ids, at, uint8_t(seed + 10)));
    ConversationKv kv;
    kv.format = 2; kv.cells = (int64_t) ids.size(); kv.heads = 2; kv.head_dim = 256; kv.page_size = 64; kv.idx_dim = 128;
    kv.k = pattern(65536 + seed, seed);
    kv.v = pattern(4096, uint8_t(seed + 1));
    s.kv.push_back(std::move(kv));
    return s;
}

// a two-stage (layer split) conversation, as park_current builds it
SavedConversation split_image(const std::vector<int32_t>& ids, size_t at, uint8_t seed) {
    SavedConversation s = stage_image(ids, at, 0, 19, seed);
    s.stage_images.push_back(stage_image(ids, at, 19, 48, uint8_t(seed + 50)));
    return s;
}

ConversationDiskLayout split_layout() {
    ConversationDiskLayout l;
    l.stages = {{0, 19}, {19, 48}};
    return l;
}

std::vector<int32_t> prompt_of(const std::vector<int32_t>& ids, size_t extra) {
    std::vector<int32_t> p = ids;
    for (size_t i = 0; i < extra; ++i) p.push_back(7);
    return p;
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-convdisk-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    ConversationDiskOptions opt;
    opt.dir = dir.string();
    opt.budget_bytes = 64ull << 20;
    opt.durable = false;
    std::string error;

    const auto a_ids = tokens(5000, 1000), b_ids = tokens(3000, 5000);
    const SavedConversation a = split_image(a_ids, 4000, 1), b = split_image(b_ids, 2000, 2);
    uint64_t ida = 0, idb = 0;
    {
        ConversationDiskTier t;
        check(t.open(opt, id, split_layout(), error), "open an empty folder");
        check(t.spill(a, ida, error) && ida != 0, "spill a two-stage conversation");
        check(t.spill(b, idb, error) && idb != 0 && idb != ida, "spill a second one");
        check(t.size() == 2 && t.writes() == 2, "two kept, two written");
        uint64_t again = 0;
        check(t.spill(a, again, error) && again == ida && t.writes() == 2, "an identical image is not written again");

        // matching: the live end (the prompt goes on past it), a checkpoint (the prompt diverges after it), none
        auto m = t.best(prompt_of(a_ids, 3), {}, true);
        check(m.id == ida && m.tokens == 5000 && m.live, "the live end matches a continuing prompt");
        auto diverged = std::vector<int32_t>(a_ids.begin(), a_ids.begin() + 4500);
        diverged.push_back(-1);
        m = t.best(diverged, {}, true);
        check(m.id == ida && m.tokens == 4000 && !m.live, "a checkpoint matches a prompt that rewrote the tail");
        check(t.best(prompt_of(a_ids, 3), {}, false).id == 0, "the control-vector mode must match");
        check(t.best(a_ids, {}, true).tokens != 5000, "the prompt's last token is never covered");
        check(t.best(tokens(10, 99999), {}, true).id == 0, "an unrelated prompt matches nothing");

        // load: stage images back in stage order, the same bytes
        SavedConversation back;
        check(t.load(ida, {}, back, error), "load");
        check(back.disk_id == ida && back.stage_images.size() == 1, "loaded with its stage image and its disk id");
        check(back.live.ids == a.live.ids && back.layer_hi == 19 && back.stage_images[0].layer_lo == 19,
              "each stage's own layer range");
        check(back.kv[0].k == a.kv[0].k && back.stage_images[0].kv[0].k == a.stage_images[0].kv[0].k,
              "K/V bytes of both stages");
        check(back.checkpoints.size() == 1 && back.checkpoints[0].ids.size() == 4000, "checkpoints");
    }
    {   // restart: the index files bring both back; the LRU order survives (a was read last)
        ConversationDiskTier t;
        ConversationDiskTier::OpenReport rep;
        check(t.open(opt, id, split_layout(), error, &rep) && rep.kept == 2 && t.size() == 2, "reopen keeps both");
        check(t.best(prompt_of(b_ids, 1), {}, true).id == idb, "a match after a restart");
        // the same conversation a turn later supersedes its older copy (its deepest checkpoint is a point of the new one)
        auto longer = a_ids;
        for (int i = 0; i < 700; ++i) longer.push_back(42);
        SavedConversation a2 = split_image(longer, 5000, 3);
        a2.checkpoints.insert(a2.checkpoints.begin(), checkpoint(longer, 4000, 9));
        a2.stage_images[0].checkpoints.insert(a2.stage_images[0].checkpoints.begin(), checkpoint(longer, 4000, 9));
        uint64_t id2 = 0;
        check(t.spill(a2, id2, error) && id2 != ida && !t.contains(ida) && t.contains(idb), "the older copy goes");
    }
    {   // another stage layout (another --layer-split carve) does not see these files, and leaves them alone
        ConversationDiskTier t;
        ConversationDiskLayout one;
        one.stages = {{0, 48}};
        ConversationDiskTier::OpenReport rep;
        check(t.open(opt, id, one, error, &rep) && t.size() == 0 && rep.other_folders == 1, "another layout: own folder");
        uint64_t x = 0;
        check(!t.spill(a, x, error), "an image of another stage count is refused");
    }
    {   // crash leftovers: a stage file without an index and a temporary file go; a damaged index drops its entry
        ConversationDiskTier t;
        check(t.open(opt, id, split_layout(), error), "reopen");
        const std::string folder = t.folder();
        { std::ofstream(fs::path(folder) / "00000000000000aa.s0.strses") << "x"; }
        { std::ofstream(fs::path(folder) / ".00000000000000aa.s1.strses.0123456789abcdef.tmp") << "x"; }
        fs::path victim;
        for (const auto& d : fs::directory_iterator(folder))
            if (d.path().extension() == ".idx") { victim = d.path(); break; }
        { std::fstream f(victim, std::ios::in | std::ios::out | std::ios::binary); f.seekp(40); f.put('\x7f'); }
        ConversationDiskTier u;
        ConversationDiskTier::OpenReport rep;
        check(u.open(opt, id, split_layout(), error, &rep), "reopen after a crash");
        check(rep.orphans >= 2 && rep.dropped == 1 && u.size() == 1, "stray files removed, the damaged entry dropped");
    }
    {   // the budget: the least recently used conversation leaves
        ConversationDiskOptions small = opt;
        small.dir = (dir / "small").string();
        small.budget_bytes = a.bytes() + b.bytes() / 2;
        ConversationDiskTier t;
        check(t.open(small, id, split_layout(), error), "open with a small budget");
        uint64_t x = 0, y = 0;
        check(t.spill(a, x, error) && t.spill(b, y, error), "two spills");
        check(t.size() == 1 && t.contains(y) && t.evictions() == 1, "the older one was evicted for room");
        check(t.clear() == 1 && t.size() == 0 && t.bytes() == 0, "clear");
    }
    {   // a conversation this run cannot restore is not offered again, but its files stay for the next start-up
        ConversationDiskOptions o2 = opt;
        o2.dir = (dir / "failed").string();
        uint64_t x = 0;
        {
            ConversationDiskTier t;
            check(t.open(o2, id, split_layout(), error) && t.spill(a, x, error), "spill for the failed case");
            t.mark_failed(x);
            check(t.best(prompt_of(a_ids, 3), {}, true).id == 0 && t.contains(x), "a failed one is not offered, kept");
        }
        ConversationDiskTier t;
        check(t.open(o2, id, split_layout(), error) && t.best(prompt_of(a_ids, 3), {}, true).id == x,
              "after a restart it is offered again");
    }
    {   // the LRU victim is the least recently USED one: an identical spill (holding) refreshes it
        const auto c_ids = tokens(4000, 20000);
        const SavedConversation c = split_image(c_ids, 3000, 4);
        ConversationDiskOptions o3 = opt;
        o3.dir = (dir / "lru").string();
        o3.budget_bytes = a.bytes() + b.bytes() + c.bytes() / 2;
        ConversationDiskTier t;
        uint64_t x = 0, y = 0, z = 0, again = 0;
        check(t.open(o3, id, split_layout(), error) && t.spill(a, x, error) && t.spill(b, y, error), "a, then b");
        check(t.spill(a, again, error) && again == x, "a again: kept, now the most recent");
        check(t.spill(c, z, error) && t.contains(x) && !t.contains(y) && t.contains(z), "b (least recent) goes, not a");
    }
    {   // the RAM cache hands evicted entries to the hook, and the idle writer's marks
        ConversationCache cache(1ull << 30, 2);
        size_t seen = 0;
        cache.on_evict([&](const SavedConversation& s) { seen += s.live.ids.size(); });
        check(cache.put(stage_image(a_ids, 10, 0, 48, 1)) && cache.put(stage_image(b_ids, 10, 0, 48, 2)), "two parked");
        check(cache.unpersisted() == 0, "both unwritten, the oldest first");
        cache.mark_persisted(0, 77);
        check(cache.unpersisted() == 1 && cache.at(0).disk_id == 77, "marked");
        check(cache.put(stage_image(tokens(100, 9000), 10, 0, 48, 3)), "a third evicts the oldest");
        check(seen == a_ids.size(), "the hook saw the evicted conversation");
        check(cache.take(cache.best(prompt_of(b_ids, 1), {}, true).index).live.ids == b_ids && seen == a_ids.size(),
              "take() is not an eviction");
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
