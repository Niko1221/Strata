// CPU-only tests for the spill directory (include/strata/core/conversation_spill.hpp): a parked conversation is
// written as a session file plus a metadata sidecar, matched from the sidecar alone, and read back whole.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_spill.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>

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
bool buffers_equal(const ConversationBuffer& a, const ConversationBuffer& b) {
    if (a.size() != b.size()) return false;
    bool same = true;
    size_t at = 0;
    a.visit(0, a.size(), [&](const uint8_t* p, size_t c, size_t off) {
        b.visit(off, c, [&](const uint8_t* q, size_t c2, size_t) {
            if (c2 != c) { same = false; return false; }
            for (size_t i = 0; i < c; ++i) if (p[i] != q[i]) { same = false; return false; }
            return true;
        });
        at += c;
        return same;
    });
    return same && at == a.size();
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};

    ConversationSpillCache spill;
    std::string error;
    check(spill.open(dir, id, 1ull << 30, error), "open empty dir");
    check(spill.enabled(), "enabled");
    check(spill.size() == 0, "empty index");

    SavedConversation a = sample(1000);
    check(spill.spill(a, error), "spill first");
    check(spill.size() == 1, "one entry");
    // the prompt continues the conversation (one token past its end): the whole live state resumes.
    std::vector<int32_t> continuing = a.live.ids;
    continuing.push_back(int32_t(7777));
    const auto first = spill.best(continuing, a.live.imgs, false, 0.0, 0);
    check(bool(first), "match own conversation");
    check(first.tokens == (int64_t) a.live.ids.size(), "full prefix match");
    check(first.live, "live match");

    // read it back whole and compare the K/V byte for byte
    SavedConversation back;
    check(spill.load(first.path, back, {}, error), "load back");
    check(back.live.ids == a.live.ids, "ids round-trip");
    check(back.checkpoints.size() == a.checkpoints.size(), "checkpoint count round-trip");
    check(back.kv.size() == a.kv.size(), "kv layer count");
    bool kv_same = true;
    for (size_t i = 0; i < back.kv.size(); ++i)
        kv_same = kv_same && buffers_equal(back.kv[i].k, a.kv[i].k) && buffers_equal(back.kv[i].v, a.kv[i].v) &&
                  buffers_equal(back.kv[i].pooled, a.kv[i].pooled);
    check(kv_same, "kv bytes round-trip");

    // a different prompt sharing a prefix resumes at the common prefix, not the whole thing
    std::vector<int32_t> other = a.live.ids;
    other.resize(400);
    for (size_t i = 400; i < a.live.ids.size(); ++i) other.push_back(int32_t(9000 + i));
    check(other.size() == a.live.ids.size(), "other same length, diverges at 400");
    const auto partial = spill.best(other, {}, false, 0.0, 0);
    // the resume lands on the deepest checkpoint still a prefix (250); the 500 checkpoint crosses the divergence.
    check(bool(partial) && partial.tokens == 250, "resume at the deepest still-valid checkpoint");
    check(!partial.live, "checkpoint resume, not live");

    // similarity filters a weak hit: 400/1000 = 0.4 LCP is refused at similarity 0.5
    const auto weak = spill.best(other, {}, false, 0.5, 0);
    check(!bool(weak), "similarity filters weak hit");
    // n_min filters a short hit
    const auto short_hit = spill.best(std::vector<int32_t>(a.live.ids.begin(), a.live.ids.begin() + 10), {}, false, 0.0, 100);
    check(!bool(short_hit), "n_min filters short hit");

    // cvec mismatch never matches
    const auto wrong_cvec = spill.best(a.live.ids, a.live.imgs, true, 0.0, 0);
    check(!bool(wrong_cvec), "cvec mismatch refused");

    // reopen the directory: the sidecar is reindexed without reading the session file
    ConversationSpillCache again;
    check(again.open(dir, id, 1ull << 30, error), "reopen dir");
    check(again.size() == 1, "reindexed one entry");
    const auto rematched = again.best(continuing, a.live.imgs, false, 0.0, 0);
    check(bool(rematched) && rematched.tokens == (int64_t) a.live.ids.size(), "rematch after reopen");

    // a foreign identity is refused and its files removed
    ConversationSpillCache foreign;
    check(foreign.open(dir, SessionFileIdentity{1, 2}, 1ull << 30, error), "open with foreign identity");
    check(foreign.size() == 0, "foreign identity: nothing indexed");
    check(fs::exists(dir / "strata-conv-1.sess"), "another identity's files are left alone");

    // budget eviction drops the oldest
    ConversationSpillCache tight;
    check(tight.open(dir, id, 1ull << 30, error), "open for budget test");
    const uint64_t before = tight.bytes();
    SavedConversation b = sample(2000);
    check(tight.spill(b, error), "spill second");
    const uint64_t newest = tight.bytes() - before;   // the newest session file's bytes
    // a budget that fits the newest alone: reopening keeps it and evicts the older one.
    ConversationSpillCache small;
    check(small.open(dir, id, newest, error), "open with a budget for one");
    check(small.size() == 1, "budget keeps one");
    check(small.disk_evictions() >= 1, "budget evicted the oldest");

    // the RAM cache handing its evictions to the disk tier: make_room's spill callback writes what it drops.
    fs::remove_all(dir);
    ConversationSpillCache tier;
    check(tier.open(dir, id, 1ull << 30, error), "open for the RAM-cache tier test");
    SavedConversation c1 = sample(1000), c2 = sample(1500);
    const size_t need = c1.bytes() + c2.bytes();
    ConversationCache ram(need, 1);   // one slot: putting the second evicts the first
    check(ram.enabled(), "ram cache enabled");
    check(ram.put(std::move(c1)), "put first in RAM");
    size_t spilled = 0;
    auto spill_cb = [&](const SavedConversation& evicted) { ++spilled; tier.spill(evicted, error); };
    // make_room for the second conversation evicts the first and hands it to the callback.
    check(ram.make_room(c2.bytes(), 0, spill_cb), "make_room evicts to make space");
    check(spilled == 1, "one conversation handed to the spill callback");
    check(tier.size() == 1, "the evicted conversation reached disk");
    // the disk copy resumes the same conversation the RAM cache dropped.
    std::vector<int32_t> cont1 = sample(1000).live.ids;
    cont1.push_back(int32_t(7777));
    const auto from_disk = tier.best(cont1, sample(1000).live.imgs, false, 0.0, 0);
    check(bool(from_disk) && from_disk.tokens == 1000, "disk resumes the evicted conversation");
    // spill_all empties the RAM cache onto disk (the shutdown path).
    check(ram.put(std::move(c2)), "put second in RAM");
    const size_t drained = ram.spill_all(spill_cb);
    check(drained == 1, "spill_all drained the parked conversation");
    check(ram.size() == 0, "ram cache empty after spill_all");
    check(tier.size() == 2, "both conversations now on disk");

    fs::remove_all(dir);
    // --conversation-cache-disk-only: a conversation streamed from its sources (no host K/V image) is written as the
    // same file the image spill writes, indexed the same way, and reads back with the same K/V bytes.
    {
        const fs::path sdir = fs::temp_directory_path() / "strata_spill_streamed_test";
        fs::remove_all(sdir);
        auto make_sources = [](const SavedConversation& img) {
            std::vector<SessionKvSource> sources;
            for (const auto& k : img.kv) {
                SessionKvSource src;
                src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
                src.page_size = k.page_size; src.pooled_rows = k.pooled_rows; src.idx_dim = k.idx_dim;
                const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                for (size_t i = 0; i < 5; ++i) src.sizes[i] = parts[i]->size();
                src.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
                sources.push_back(std::move(src));
            }
            return sources;
        };
        auto slurp = [](const std::string& path) {
            FILE* f = std::fopen(path.c_str(), "rb");
            std::vector<uint8_t> v;
            if (!f) return v;
            uint8_t buf[65536];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
            std::fclose(f);
            return v;
        };
        ConversationSpillCache img_dir, str_dir;
        check(img_dir.open(sdir / "image", id, 1ull << 30, error), "streamed: open image dir");
        check(str_dir.open(sdir / "streamed", id, 1ull << 30, error), "streamed: open streamed dir");
        const SavedConversation c = sample(1500);
        check(img_dir.spill(c, error), "streamed: image spill");
        SavedConversation meta = c;
        meta.kv.clear();
        size_t written = 0;
        check(str_dir.spill_streamed(meta, make_sources(c), written, error), "streamed: streamed spill");
        check(written > 0 && str_dir.bytes() == written, "streamed: size accounted");
        check(slurp(img_dir.newest_path()) == slurp(str_dir.newest_path()), "streamed: file equals the image spill's");
        // it matches and reads back like any spilled conversation
        std::vector<int32_t> next_prompt = c.live.ids;
        next_prompt.push_back(7);
        const auto hit = str_dir.best(next_prompt, {}, c.cvec, 0.0, 0);
        check(bool(hit) && hit.tokens == (int64_t) c.live.ids.size() && hit.live, "streamed: full live match");
        SavedConversation back;
        check(str_dir.load(hit.path, back, {}, error), "streamed: load back");
        bool same = back.kv.size() == c.kv.size();
        for (size_t i = 0; same && i < c.kv.size(); ++i)
            same = buffers_equal(back.kv[i].k, c.kv[i].k) && buffers_equal(back.kv[i].v, c.kv[i].v) &&
                   buffers_equal(back.kv[i].pooled, c.kv[i].pooled);
        check(same, "streamed: K/V bytes round-trip");
        // a host K/V image passed to the streamed spill is refused (ambiguous), and so is an empty conversation
        check(!str_dir.spill_streamed(c, make_sources(c), written, error), "streamed: image K/V refused");
        SavedConversation empty = meta;
        empty.live.ids.clear();
        check(!str_dir.spill_streamed(empty, {}, written, error), "streamed: empty conversation refused");
        // a failing source leaves no file and no index entry
        auto broken = make_sources(c);
        broken[0].read = [](size_t, size_t, void*, size_t) { return false; };
        const size_t before = str_dir.size();
        check(!str_dir.spill_streamed(meta, broken, written, error), "streamed: failing source fails");
        check(str_dir.size() == before, "streamed: failed write not indexed");
        size_t sess_files = 0;
        for (const auto& e : fs::directory_iterator(sdir / "streamed"))
            if (e.path().extension() == ".sess") ++sess_files;
        check(sess_files == before, "streamed: failed write leaves no session file");
        // the new copy is written before the older ones are dropped: with keep, drop_superseded spares it
        SavedConversation longer = c;
        for (int i = 0; i < 200; ++i) longer.live.ids.push_back(int32_t(5000 + i));
        longer.checkpoints.push_back(checkpoint(c.live.ids.size(), 9));   // the turn checkpoint: the old conversation's end
        SavedConversation longer_meta = longer;
        longer_meta.kv.clear();
        check(str_dir.spill_streamed(longer_meta, make_sources(longer), written, error), "keep: spill the newer copy");
        const std::string kept = str_dir.newest_path();
        check(str_dir.size() == 2, "keep: old and new copy on disk");
        const size_t dropped = str_dir.drop_superseded(longer.live.ids, {}, longer.checkpoints, longer.cvec, kept);
        check(dropped == 1 && str_dir.size() == 1 && str_dir.newest_path() == kept, "keep: old copy dropped, new kept");
        // without keep the same rule would drop the new copy too (its deepest checkpoint is on the live path)
        check(str_dir.drop_superseded(longer.live.ids, {}, longer.checkpoints, longer.cvec) == 1 && str_dir.size() == 0,
              "keep: the rule alone drops the new copy");

        // Real disk_save_live saves only the deepest checkpoint, not the full RAM chain above.
        // Turn-back cleanup must still remove turn 0 even though its checkpoint is absent from turn 1's file.
        SavedConversation turn0 = meta;
        turn0.checkpoints = session_checkpoints_to_save(c.checkpoints);
        check(turn0.checkpoints.size() == 1 && turn0.checkpoints[0].ids.size() == 750,
              "streamed turns: turn 0 keeps only its deepest checkpoint");
        check(str_dir.spill_streamed(turn0, make_sources(c), written, error), "streamed turns: save turn 0");
        const std::string old_turn = str_dir.newest_path();
        SavedConversation turn1 = longer_meta;
        turn1.checkpoints = session_checkpoints_to_save(longer.checkpoints);
        check(turn1.checkpoints.size() == 1 && turn1.checkpoints[0].ids.size() == 1500,
              "streamed turns: turn 1 omits turn 0's checkpoint");
        check(str_dir.spill_streamed(turn1, make_sources(longer), written, error), "streamed turns: save turn 1");
        const std::string new_turn = str_dir.newest_path();
        check(str_dir.drop_superseded(longer.live.ids, {}, longer.checkpoints, longer.cvec, new_turn) == 1 &&
              str_dir.size() == 1 && str_dir.newest_path() == new_turn,
              "streamed turns: stale turn-back removed, replacement retained");
        check(!fs::exists(old_turn) && !fs::exists(fs::path(old_turn).replace_extension(".meta")),
              "streamed turns: stale payload and sidecar removed");
        fs::remove_all(sdir);
    }

    // prefix entries (a conversation's root): named strata-conv-prefix-N, kept by drop_superseded, evicted last, deduped,
    // and still prefix entries after a reopen (restart).
    {
        const fs::path pdir = fs::temp_directory_path() / "strata_spill_prefix_test";
        fs::remove_all(pdir);
        auto sources_of = [](const SavedConversation& img) {
            std::vector<SessionKvSource> sources;
            for (const auto& k : img.kv) {
                SessionKvSource src;
                src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
                src.page_size = k.page_size; src.pooled_rows = k.pooled_rows; src.idx_dim = k.idx_dim;
                const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                for (size_t i = 0; i < 5; ++i) src.sizes[i] = parts[i]->size();
                src.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
                sources.push_back(std::move(src));
            }
            return sources;
        };
        auto meta_of = [](SavedConversation img) { img.kv.clear(); return img; };
        ConversationSpillCache sp;
        check(sp.open(pdir, id, 1ull << 30, error), "prefix: open");
        // the root: 600 tokens, its own (only) checkpoint
        SavedConversation root = sample(600);
        root.checkpoints.clear();
        root.checkpoints.push_back(checkpoint(600, 7));
        size_t written = 0;
        check(!sp.has_prefix(root.live.ids, {}, root.cvec), "prefix: none yet");
        check(sp.spill_streamed(meta_of(root), sources_of(root), written, error, true, true), "prefix: save root");
        check(sp.prefix_count() == 1 && sp.has_prefix(root.live.ids, {}, root.cvec), "prefix: indexed as prefix");
        check(fs::path(sp.newest_path()).filename().string().rfind("strata-conv-prefix-", 0) == 0, "prefix: file name");
        // a conversation that continues the root: the root entry is a valid resume for its prompt
        SavedConversation conv = sample(1000);   // ids 1000.. : the root's 600 ids are its first 600
        conv.checkpoints.clear();
        conv.checkpoints.push_back(checkpoint(600, 8));
        check(sp.spill_streamed(meta_of(conv), sources_of(conv), written, error), "prefix: save a conversation");
        std::vector<int32_t> new_chat(root.live.ids);
        new_chat.push_back(99999);                // a new conversation: same root, different first message
        const auto hit = sp.best(new_chat, {}, root.cvec, 0.0, 0);
        check(bool(hit) && hit.tokens == 600, "prefix: a new conversation resumes at the root");
        // drop_superseded on the conversation's path removes the old conversation copy, never the prefix
        const size_t dropped = sp.drop_superseded(conv.live.ids, {}, conv.checkpoints, conv.cvec);
        check(dropped == 1 && sp.size() == 1 && sp.prefix_count() == 1, "prefix: drop_superseded keeps the prefix");
        // the budget evicts ordinary conversations first: add two, shrink the budget to fit only the prefix + one
        check(sp.spill_streamed(meta_of(conv), sources_of(conv), written, error), "prefix: save conversation 2");
        const uint64_t one_conv = written;
        check(sp.spill_streamed(meta_of(sample(900)), sources_of(sample(900)), written, error), "prefix: save conversation 3");
        ConversationSpillCache tight;
        const uint64_t prefix_bytes = sp.bytes() - one_conv - written;
        check(tight.open(pdir, id, prefix_bytes + std::max<uint64_t>(one_conv, written) + 16, error), "prefix: reopen with a tight budget");
        check(tight.prefix_count() == 1, "prefix: still a prefix after reopen");
        check(tight.size() == 2, "prefix: budget evicted an ordinary conversation, not the prefix");
        check(tight.has_prefix(root.live.ids, {}, root.cvec), "prefix: dedupe sees the reopened prefix");
        // Both cleanup call sites must retain the root: disk-only passes keep; RAM parking does not.
        // Reopening also has to preserve that distinction before any new prefix is written.
        const std::string replacement = tight.newest_path();
        tight.drop_superseded(conv.live.ids, {}, conv.checkpoints, conv.cvec, replacement);
        check(tight.has_prefix(root.live.ids, {}, root.cvec), "prefix: keep cleanup after reopen retains root");
        tight.drop_superseded(conv.live.ids, {}, conv.checkpoints, conv.cvec);
        check(tight.has_prefix(root.live.ids, {}, root.cvec), "prefix: RAM cleanup after reopen retains root");
        const auto root_again = tight.best(new_chat, {}, root.cvec, 0.0, 0);
        check(bool(root_again) && root_again.tokens == 600, "prefix: root still resumes a sibling after both cleanups");
        fs::remove_all(pdir);
    }

    // the checkpoints a spill keeps: the shallowest (the shared root) and the deepest (the next turn's resume point)
    {
        SavedConversation t = sample(1000);   // checkpoints at 250 and 500
        t.checkpoints.push_back(checkpoint(750, 5));
        t.checkpoints.insert(t.checkpoints.begin(), checkpoint(100, 6));
        conversation_spill_trim_checkpoints(t);
        check(t.checkpoints.size() == 2 && t.checkpoints[0].ids.size() == 100 && t.checkpoints[1].ids.size() == 750,
              "trim: root and deepest kept, in order");
        check(t.checkpoints[0].gdn == checkpoint(100, 6).gdn && t.checkpoints[1].gdn == checkpoint(750, 5).gdn,
              "trim: the kept checkpoints' state intact");
        SavedConversation two = sample(1000);
        conversation_spill_trim_checkpoints(two);
        check(two.checkpoints.size() == 2, "trim: two checkpoints stay two");
    }

    // the background write: the image is taken, the file is matched only once wait() has indexed it
    {
        const fs::path adir = fs::temp_directory_path() / ("strata-spill-async-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::remove_all(adir);
        ConversationSpillCache tier2;
        check(tier2.open(adir, id, 1ull << 30, error), "async: open");
        check(!tier2.wait(), "async: nothing to wait for");
        SavedConversation x = sample(1000);
        x.checkpoints.push_back(checkpoint(750, 5));
        x.checkpoints.insert(x.checkpoints.begin(), checkpoint(100, 6));
        const SavedConversation expect = x;
        check(tier2.spill_async(std::move(x), error), "async: started");
        check(x.live.ids.empty() && x.kv.empty(), "async: the image was taken");
        check(tier2.size() == 0, "async: not indexed before wait");
        std::vector<int32_t> cont = expect.live.ids;
        cont.push_back(int32_t(7777));
        check(!tier2.best(cont, expect.live.imgs, false, 0.0, 0), "async: not matched before wait");
        const auto w = tier2.wait();
        check(w && w->ok && w->tokens == 1000 && w->bytes > 0, "async: the write is reported");
        check(!tier2.writing() && tier2.size() == 1, "async: indexed after wait");
        const auto m = tier2.best(cont, expect.live.imgs, false, 0.0, 0);
        check(bool(m) && m.tokens == 1000 && m.live, "async: matched after wait");
        SavedConversation back2;
        check(tier2.load(m.path, back2, {}, error), "async: load back");
        check(back2.live.ids == expect.live.ids && back2.live.gdn == expect.live.gdn, "async: live state round-trip");
        check(back2.checkpoints.size() == 2 && back2.checkpoints[0].ids.size() == 100 &&
              back2.checkpoints[1].ids.size() == 750, "async: the file holds the root and the deepest checkpoint");
        bool same = back2.kv.size() == expect.kv.size();
        for (size_t i = 0; same && i < back2.kv.size(); ++i)
            same = buffers_equal(back2.kv[i].k, expect.kv[i].k) && buffers_equal(back2.kv[i].v, expect.kv[i].v);
        check(same, "async: K/V bytes round-trip");
        // a prompt that leaves the conversation past the root resumes at the root from disk
        std::vector<int32_t> branch(expect.live.ids.begin(), expect.live.ids.begin() + 120);
        branch.push_back(int32_t(4242));
        const auto root = tier2.best(branch, {}, false, 0.0, 0);
        check(bool(root) && root.tokens == 100 && !root.live, "async: a new branch resumes at the root");

        // a second write waits for the first (one at a time), and both end up indexed
        check(tier2.spill_async(sample(1500), error), "async: second write");
        check(tier2.spill_async(sample(2000), error), "async: third write waits for the second");
        check(tier2.size() == 2, "async: the second is indexed when the third starts");
        check(tier2.wait() && tier2.size() == 3, "async: the third indexed after wait");

        // the RAM cache hands its evictions over as rvalues: the tier takes them without a copy and RAM accounting holds
        ConversationCache ram2(sample(1000).bytes() * 3, 1);
        check(ram2.put(sample(1000)), "async ram: put");
        const size_t ram_bytes = ram2.bytes();
        size_t handed = 0;
        auto take = [&](SavedConversation&& evicted) { ++handed; tier2.spill_async(std::move(evicted), error); };
        check(ram2.make_room(ram_bytes, 0, take) && handed == 1 && ram2.bytes() == 0 && ram2.size() == 0,
              "async ram: the eviction is moved out, the cache's bytes drop by its size");
        // put() hands what it evicts to the callback too
        check(ram2.put(sample(1500), 0, take) && handed == 1, "async ram: put into an empty cache evicts nothing");
        check(ram2.put(sample(1000), 0, take) && handed == 2 && ram2.size() == 1, "async ram: put evicts through the callback");
        check(tier2.wait() && tier2.size() == 5, "async ram: both evicted conversations reached disk");

        // a destroyed tier waits for its write: the file is complete on disk
        {
            ConversationSpillCache scoped;
            check(scoped.open(adir, id, 1ull << 30, error), "async: open scoped");
            check(scoped.spill_async(sample(800), error), "async: scoped write");
        }
        ConversationSpillCache reopened;
        check(reopened.open(adir, id, 1ull << 30, error) && reopened.size() == 6,
              "async: a write finished by the destructor is indexed on reopen");

        // the free-space preflight: a write that would leave less than min_free on the disk is refused, no file kept
        ConversationSpillCache full;
        check(full.open(adir, id, 1ull << 30, error, UINT64_MAX / 2), "async: open with an impossible free-space floor");
        const size_t files_before = full.size();
        check(full.spill_async(sample(600), error), "async: full disk write started");
        const auto refused = full.wait();
        check(refused && !refused->ok && !refused->error.empty(), "async: refused by the free-space preflight");
        check(full.size() == files_before, "async: a refused write is not indexed");
        size_t sess = 0;
        for (const auto& e : fs::directory_iterator(adir))
            if (e.path().extension() == ".sess") ++sess;
        check(sess == files_before, "async: a refused write leaves no session file");
        fs::remove_all(adir);
    }

    std::printf("conversation_spill_test: %d checks passed\n", checks);
    return 0;
}
