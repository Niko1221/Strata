// CPU-only tests for the spill directory (include/strata/core/conversation_spill.hpp): a parked conversation is
// written as a session file plus a metadata sidecar, matched from the sidecar alone, and read back whole.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_spill.hpp"
#include "strata/core/rht_state_codec.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <system_error>

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
    const std::vector<uint8_t> gdn = bytes_of(4099, seed);
    c.gdn.assign(gdn.begin(), gdn.end());
    const std::vector<uint8_t> ple = bytes_of(77, seed + 1);
    c.ple.assign(ple.begin(), ple.end());
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
SavedConversation rht_sample(size_t tokens) {
    SavedConversation s = sample(tokens);
    s.geometry = {2560, 2, 4, 128, 16, 1, 2, 1, 6144, 24, 2, 256, 4, 128, 4, 320, 512, 640};
    s.layer_lo = 0;
    s.layer_hi = 2;
    rht::Layout layout{};
    if (!rht::layout_from(s.geometry, s.layer_lo, s.layer_hi, layout)) std::abort();
    const auto finite_gdn = [&](ConversationCheckpoint& checkpoint, uint8_t seed) {
        checkpoint.gdn.resize((size_t) layout.logical);
        for (size_t at = 0; at < checkpoint.gdn.size(); at += sizeof(float)) {
            const float value = (float) ((int) (at / sizeof(float) % 127) - 63) * 0.001f + (float) seed * 0.01f;
            std::memcpy(checkpoint.gdn.data() + at, &value, sizeof(value));
        }
    };
    finite_gdn(s.live, 1);
    for (size_t i = 0; i < s.checkpoints.size(); ++i) finite_gdn(s.checkpoints[i], (uint8_t) (i + 2));
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
    std::vector<int32_t> continuing(a.live.ids.begin(), a.live.ids.end());
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
    std::vector<int32_t> other(a.live.ids.begin(), a.live.ids.end());
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
    const auto wrong_cvec = spill.best(std::vector<int32_t>(a.live.ids.begin(), a.live.ids.end()),
                                       a.live.imgs, true, 0.0, 0);
    check(!bool(wrong_cvec), "cvec mismatch refused");

    // reopen the directory: the sidecar is reindexed without reading the session file
    ConversationSpillCache again;
    check(again.open(dir, id, 1ull << 30, error), "reopen dir");
    check(again.size() == 1, "reindexed one entry");
    const auto rematched = again.best(continuing, a.live.imgs, false, 0.0, 0);
    check(bool(rematched) && rematched.tokens == (int64_t) a.live.ids.size(), "rematch after reopen");

    // a foreign identity is refused: its files are KEPT on disk (never auto-deleted), not indexed and outside the
    // disk budget; open() logs a warning with their count and bytes - remove them manually.
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

    // a process killed mid-write leaves the writer's hidden temporary beside the spill file (a 6 GB one here, on
    // the user's PC): the scan at open is the only thing that removes it, and it must not touch anyone else's.
    const fs::path abandoned = dir / ".strata-conv-99.sess.0123456789abcdef.tmp";
    { std::ofstream(abandoned) << "half a conversation"; }
    const fs::path foreign_temp = dir / ".strata-notes.0123456789abcdef.tmp";   // not a spill file's temp
    { std::ofstream(foreign_temp) << "not ours"; }
    // a killed sidecar or chain-manifest write leaves a <stem>.meta.tmp / <stem>.dchain.tmp behind: same wipe
    const fs::path meta_temp = dir / "strata-conv-98.meta.tmp";
    { std::ofstream(meta_temp) << "junk"; }
    const fs::path chain_temp = dir / "strata-conv-97.dchain.tmp";
    { std::ofstream(chain_temp) << "junk"; }
    ConversationSpillCache wiped;
    check(wiped.open(dir, id, 1ull << 30, error), "open with an abandoned temp");
    check(!fs::exists(abandoned), "abandoned spill temp removed at open");
    check(!fs::exists(meta_temp), "abandoned sidecar temp removed at open");
    check(!fs::exists(chain_temp), "abandoned chain-manifest temp removed at open");
    check(fs::exists(foreign_temp), "a foreign dot file is left alone");
    check(wiped.stale_files_wiped() >= 3, "the abandoned temps count as stale files wiped");
    check(wiped.size() == 1, "the temps are not indexed as a conversation");
    fs::remove(foreign_temp);


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
    std::vector<int32_t> cont1;
    {
        const SavedConversation probe = sample(1000);
        cont1.assign(probe.live.ids.begin(), probe.live.ids.end());
    }
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
        std::vector<int32_t> next_prompt(c.live.ids.begin(), c.live.ids.end());
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
        std::vector<int32_t> new_chat(root.live.ids.begin(), root.live.ids.end());
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
        fs::remove_all(pdir);
    }

    // --conversation-cache-disk-delta: a base session file plus an append log and its manifest are one entry.
    // spill_delta appends the changed K/V and republishes the sidecar; the chain survives a reopen; erase, the
    // budget and drop_superseded remove the chain files together; a chain that cannot be proven (short log, edited
    // manifest) makes open() drop the whole entry rather than restore a state that does not match its sidecar.
    auto sources_of = [](const SavedConversation& img, size_t first = 0) {
        std::vector<SessionKvSource> sources;
        for (const auto& k : img.kv) {
            SessionKvSource src;
            src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
            src.page_size = k.page_size; src.pooled_rows = k.pooled_rows; src.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) {
                src.sizes[i] = parts[i]->size();
                src.first_bytes[i] = parts[i]->size() > first ? first : 0;
            }
            src.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
            sources.push_back(std::move(src));
        }
        return sources;
    };
    auto meta_of = [](SavedConversation img) { img.kv.clear(); return img; };
    auto read_file = [](const fs::path& p) {
        std::vector<char> v;
        if (FILE* f = std::fopen(p.string().c_str(), "rb")) {
            char buf[65536];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
            std::fclose(f);
        }
        return v;
    };
    auto write_file = [](const fs::path& p, const std::vector<char>& v) {
        if (FILE* f = std::fopen(p.string().c_str(), "wb")) {
            if (!v.empty()) std::fwrite(v.data(), 1, v.size(), f);
            std::fclose(f);
        }
    };
    const fs::path cdir = fs::temp_directory_path() / "strata_spill_delta_test";
    fs::remove_all(cdir);
    {
        SavedConversation base = sample(1500);
        SavedConversation grown = base;
        // the ids continue the sample's (1000 + i): a checkpoint generator keeps matching them
        for (int i = 0; i < 300; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        grown.checkpoints.push_back(checkpoint(1800, 8));

        ConversationSpillCache ch;
        size_t written = 0;
        check(ch.open(cdir, id, 1ull << 30, error, true), "delta spill: open");
        check(ch.spill_streamed(meta_of(base), sources_of(base, 0), written, error), "delta spill: base");
        const uint64_t base_file = written;
        const size_t index = ch.find(ch.newest_path());
        check(index == 0, "delta spill: find by path");
        int64_t tokens = 0, deepest = 0;
        uint64_t info_base = 0, info_delta = 0;
        bool is_prefix = true;
        std::string path;
        check(ch.entry_info(index, tokens, deepest, info_base, info_delta, is_prefix, path) && tokens == 1500 &&
              info_base == base_file && info_delta == 0 && !is_prefix, "delta spill: entry info before the delta");
        size_t delta_written = 0;
        check(ch.spill_delta(index, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(),
                             delta_written, error), "delta spill: append");
        check(delta_written > 0, "delta spill: the record was written");
        check(ch.entry_matches_prefix(index, grown.live.ids, grown.live.ids.size(), {}),
              "delta spill: the entry covers the state's tokens");
        // a state the restore rewrote shorter than the entry still appends: only the covered prefix counts
        std::vector<int32_t> shorter;
        shorter.assign(grown.live.ids.begin(), grown.live.ids.begin() + 1700);
        check(ch.entry_matches_prefix(index, shorter, 1600, {}), "delta spill: a shorter rewritten state matches");
        std::vector<int32_t> diverging = grown.live.ids;
        diverging[700] = 12345;
        check(!ch.entry_matches_prefix(index, diverging, grown.live.ids.size(), {}),
              "delta spill: a diverged prompt is not a match");
        // a prefix entry refuses the delta append
        {
            ConversationSpillCache pc;
            check(pc.open(cdir / "never", id, 1ull << 30, error, true), "delta spill: prefix dir");
            SavedConversation root = sample(600);
            root.checkpoints.clear();
            root.checkpoints.push_back(checkpoint(600, 7));
            size_t w = 0;
            check(pc.spill_streamed(meta_of(root), sources_of(root, 0), w, error, true, true), "delta spill: prefix saved");
            size_t wd = 0;
            check(!pc.spill_delta(0, meta_of(root), sources_of(root, 0), 600, wd, error),
                  "delta spill: a prefix entry refuses the append");
            fs::remove_all(cdir / "never");
        }
        check(ch.bytes() == base_file + delta_written + fs::file_size(fs::path(path).replace_extension(".meta")) +
                          fs::file_size(fs::path(path).replace_extension(".dchain")),
              "delta spill: the budget counts the chain and its sidecars");
        fs::path dlog(path), dchain(path);
        dlog.replace_extension(".dlog");
        dchain.replace_extension(".dchain");
        check(fs::file_size(dlog) == delta_written, "delta spill: the log holds the record");
        std::vector<int32_t> cont(grown.live.ids.begin(), grown.live.ids.end());
        cont.push_back(7);
        const auto hit = ch.best(cont, {}, grown.cvec, 0.0, 0);
        const bool hit_matches = bool(hit);
        const bool hit_committed_matches = hit.delta_committed == delta_written;
        const bool hit_path_matches = hit.path == ch.newest_path();
        const bool hit_tokens_match = hit.tokens == (int64_t) grown.live.ids.size() &&
                                      hit.live_tokens == (int64_t) grown.live.ids.size();
        if (!hit_matches || !hit_committed_matches || !hit_path_matches || !hit_tokens_match)
            std::fprintf(stderr, "  delta spill match: mode=delta-on match=%d committed=%llu expected=%llu path='%s' expected_path='%s' tokens=%lld live_tokens=%lld\n",
                         (int) hit_matches, (unsigned long long) hit.delta_committed,
                         (unsigned long long) delta_written, hit.path.c_str(), ch.newest_path().c_str(),
                         (long long) hit.tokens, (long long) hit.live_tokens);
        check(hit_matches && hit_committed_matches && hit_path_matches && hit_tokens_match,
              "delta spill: best() offers the chain");
        // The chain is reindexed on a reopen (an engine restart); pass the enabled mode explicitly, as generate.cpp
        // does when --conversation-cache-disk-delta is active. The default-off safe miss is covered by the real-chain
        // cross-mode fixture below.
        ConversationSpillCache reopened;
        check(reopened.open(cdir, id, 1ull << 30, error, true) && reopened.size() == 1,
              "delta spill: reopen keeps the entry");
        const auto again = reopened.best(cont, {}, grown.cvec, 0.0, 0);
        const bool again_matches = bool(again);
        const bool committed_matches = again.delta_committed == delta_written;
        const bool path_matches = again.path == hit.path;
        if (!again_matches || !committed_matches || !path_matches)
            std::fprintf(stderr, "  delta spill reopen: mode=delta-on match=%d committed=%llu expected=%llu path='%s' expected_path='%s'\n",
                         (int) again_matches, (unsigned long long) again.delta_committed,
                         (unsigned long long) delta_written, again.path.c_str(), hit.path.c_str());
        check(again_matches && committed_matches && path_matches, "delta spill: the chain survives a reopen");
        // erase removes the base, the sidecar, the log and the manifest together
        check(reopened.erase(again.path, error) && reopened.size() == 0, "delta spill: erase removes the chain");
        check(!fs::exists(path) && !fs::exists(fs::path(path).replace_extension(".meta")) && !fs::exists(dlog) &&
              !fs::exists(dchain), "delta spill: no chain file left");

        // a committed length past the log's bytes: the chain cannot be proven and the entry goes
        const fs::path broken = fs::temp_directory_path() / "strata_spill_delta_broken";
        fs::remove_all(broken);
        {
            ConversationSpillCache c1;
            check(c1.open(broken, id, 1ull << 30, error, true), "delta spill: open broken-dir");
            check(c1.spill_streamed(meta_of(base), sources_of(base, 0), written, error), "delta spill: broken base");
            size_t w = 0;
            check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), w, error),
                  "delta spill: broken append");
            fs::path d2 = c1.newest_path();
            d2.replace_extension(".dlog");
            fs::resize_file(d2, fs::file_size(d2) - 4);
        }
        ConversationSpillCache c2;
        check(c2.open(broken, id, 1ull << 30, error, true) && c2.size() == 0,
              "delta spill: a short log drops the entry");
        {
            ConversationSpillCache c3;
            check(c3.open(broken, id, 1ull << 30, error, true), "delta spill: reopen broken-dir");
            check(c3.spill_streamed(meta_of(base), sources_of(base, 0), written, error), "delta spill: broken base 2");
            size_t w = 0;
            check(c3.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), w, error),
                  "delta spill: broken append 2");
            fs::path d3 = c3.newest_path();
            d3.replace_extension(".dchain");
            std::fstream f(d3, std::ios::binary | std::ios::in | std::ios::out);
            char c = 0;
            f.read(&c, 1);
            c = char(c ^ 0x40);
            f.seekp(0);
            f.write(&c, 1);
            f.close();
        }
        ConversationSpillCache c4;
        check(c4.open(broken, id, 1ull << 30, error, true) && c4.size() == 0,
              "delta spill: an edited manifest drops the entry");
        for (const auto& e : fs::directory_iterator(broken))
            check(e.path().extension() != ".sess" && e.path().extension() != ".meta" &&
                  e.path().extension() != ".dlog" && e.path().extension() != ".dchain",
                  "delta spill: a dropped chain leaves no file behind");
        fs::remove_all(broken);

        // a full write that supersedes the chain removes the chain files with it
        {
            ConversationSpillCache c5;
            check(c5.open(cdir, id, 1ull << 30, error, true), "delta spill: open for the supersede test");
            check(c5.spill_streamed(meta_of(base), sources_of(base, 0), written, error), "delta spill: base 2");
            size_t w = 0;
            check(c5.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), w, error),
                  "delta spill: append 2");
            fs::path d5 = c5.newest_path();
            d5.replace_extension(".dlog");
            check(fs::exists(d5), "delta spill: the chain is on disk");
            SavedConversation longer = grown;
            for (int i = 0; i < 200; ++i) longer.live.ids.push_back(int32_t(2800 + i));
            longer.checkpoints.push_back(checkpoint((size_t) grown.live.ids.size(), 9));
            size_t w2 = 0;
            check(c5.spill_streamed(meta_of(longer), sources_of(longer, 0), w2, error), "delta spill: full write");
            const std::string kept = c5.newest_path();
            const size_t dropped = c5.drop_superseded(longer.live.ids, {}, longer.checkpoints, longer.cvec, kept);
            check(dropped == 1, "delta spill: the old chain is superseded");
            check(!fs::exists(d5) && !fs::exists(fs::path(d5.string().substr(0, d5.string().size() - 5) + ".dchain")),
                  "delta spill: the superseded chain's files are gone");
            fs::remove_all(cdir);
        }
    }

    // A hard link outside the spill directory must not let a rejected append truncate shared-inode bytes.
    {
        const fs::path hard_link_root = fs::temp_directory_path() / ("strata_spill_hardlink_test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const fs::path hard_link_dir = hard_link_root / "spill";
        fs::create_directories(hard_link_root);
        SavedConversation base = sample(1000);
        SavedConversation grown = base;
        for (int i = 0; i < 200; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        grown.checkpoints.push_back(checkpoint(1200, 8));
        ConversationSpillCache hard_link_cache;
        check(hard_link_cache.open(hard_link_dir, id, 1ull << 30, error, true), "hardlink: open delta directory");
        size_t base_bytes = 0;
        check(hard_link_cache.spill_streamed(meta_of(base), sources_of(base, 0), base_bytes, error),
              "hardlink: save base");
        size_t committed = 0;
        check(hard_link_cache.spill_delta(0, meta_of(grown), sources_of(grown, 1024),
                                          (int64_t) grown.live.ids.size(), committed, error),
              "hardlink: create valid appendable chain");
        const std::string session = hard_link_cache.newest_path();
        const fs::path log = fs::path(session).replace_extension(".dlog");
        const fs::path manifest = fs::path(session).replace_extension(".dchain");
        const fs::path sidecar = fs::path(session).replace_extension(".meta");
        const fs::path sentinel = hard_link_root / "external-sentinel.dlog";
        check(hard_link_cache.size() == 1 && fs::is_regular_file(log) && fs::is_regular_file(manifest) &&
              fs::is_regular_file(sidecar), "hardlink: chain, log and sidecar are valid before the guard");

        std::error_code link_error;
        fs::create_hard_link(log, sentinel, link_error);
        const bool hard_links_unsupported = link_error == std::errc::operation_not_supported ||
                                            link_error == std::errc::function_not_supported ||
                                            link_error == std::errc::not_supported;
        if (hard_links_unsupported) {
            std::fprintf(stderr, "SKIP: hard-link sentinel fixture unsupported: %s\n", link_error.message().c_str());
        } else {
            check(!link_error, "hardlink: create external sentinel hard link");
            const std::vector<char> suffix = {'u', 'n', 'c', 'o', 'm', 'm', 'i', 't', 't', 'e', 'd'};
            {
                std::ofstream out(log, std::ios::binary | std::ios::app);
                out.write(suffix.data(), (std::streamsize) suffix.size());
                out.close();
                check(bool(out), "hardlink: append the permitted uncommitted suffix");
            }
            const std::vector<char> log_before = read_file(log);
            const std::vector<char> sentinel_before = read_file(sentinel);
            const std::vector<char> manifest_before = read_file(manifest);
            const std::vector<char> sidecar_before = read_file(sidecar);
            check(sentinel_before == log_before && log_before.size() == committed + suffix.size() &&
                  manifest_before.size() == 40 && !sidecar_before.empty(),
                  "hardlink: snapshots contain the shared log and published metadata");
            size_t refused_bytes = 0;
            check(!hard_link_cache.spill_delta(0, meta_of(grown), sources_of(grown, 1024),
                                               (int64_t) grown.live.ids.size(), refused_bytes, error),
                  "hardlink: append is refused");
            check(error.find("hard link") != std::string::npos,
                  "hardlink: refusal came from the opened-inode link-count guard");
            check(read_file(log) == log_before && read_file(sentinel) == sentinel_before,
                  "hardlink: refusal leaves exact shared sentinel bytes unchanged");
            check(read_file(manifest) == manifest_before && read_file(sidecar) == sidecar_before,
                  "hardlink: refusal leaves the published manifest and sidecar unchanged");
            check(fs::remove(sentinel), "hardlink: remove only the test-owned external hard link");
        }
        fs::remove_all(hard_link_root);
    }

    // an interrupted append: bytes past the manifest's committed length are dropped before the next record, and the
    // committed chain reads back exactly (base + records, in order, rebuild the newest state)
    {
        const fs::path tdir = fs::temp_directory_path() / "strata_spill_torn_test";
        fs::remove_all(tdir);
        SavedConversation base = sample(1200);
        SavedConversation grown = base;
        for (int i = 0; i < 300; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        grown.checkpoints.push_back(checkpoint(1500, 8));
        ConversationSpillCache c1;
        size_t w = 0;
        check(c1.open(tdir, id, 1ull << 30, error, true), "torn: open");
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "torn: base");
        size_t w1 = 0;
        check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), w1, error),
              "torn: first record");
        const fs::path dlog = fs::path(c1.newest_path()).replace_extension(".dlog");
        const uint64_t committed = fs::file_size(dlog);
        // a crash mid-write: bytes past the committed length
        {
            std::ofstream out(dlog, std::ios::binary | std::ios::app);
            const std::string junk(997, 'x');
            out.write(junk.data(), (std::streamsize) junk.size());
        }
        check(fs::file_size(dlog) == committed + 997, "torn: the log is longer than the committed length");
        // the next append starts at the committed length: the tail is cut, the record lands after the first
        SavedConversation longer = sample(1700);
        size_t w2 = 0;
        check(c1.spill_delta(0, meta_of(longer), sources_of(longer, 1024), (int64_t) longer.live.ids.size(), w2, error),
              "torn: second append over the tail");
        check(fs::file_size(dlog) == committed + w2, "torn: the uncommitted tail is cut before the append");
        // ordinary reconstruction: base + both records rebuild `longer` byte for byte
        ConversationSpillCache c2;
        check(c2.open(tdir, id, 1ull << 30, error, true) && c2.size() == 1, "torn: reopen keeps the entry");
        const auto m = c2.best(longer.live.ids, {}, longer.cvec, 0.0, 0);
        const uint64_t expected_committed = committed + w2;
        const std::string expected_path = c1.newest_path();
        const bool matches = bool(m);
        const bool committed_matches = m.delta_committed == expected_committed;
        const bool path_matches = m.path == expected_path;
        if (!matches || !committed_matches || !path_matches)
            std::fprintf(stderr, "  torn reopen: mode=delta-on match=%d committed=%llu expected=%llu path='%s' expected_path='%s'\n",
                         (int) matches, (unsigned long long) m.delta_committed,
                         (unsigned long long) expected_committed, m.path.c_str(), expected_path.c_str());
        check(matches && committed_matches && path_matches, "torn: the committed chain matches");
        SessionReadLimits lim;
        SavedConversation base_meta;
        std::vector<std::array<uint64_t, 5>> base_sizes;
        size_t base_bytes = 0;
        check(session_file_read_streamed(m.path, id, base_meta, base_bytes, error, lim, &base_sizes),
              "torn: the base reads streamed");
        std::vector<std::array<std::vector<uint8_t>, 5>> rebuilt(longer.kv.size());
        for (size_t l = 0; l < longer.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&longer.kv[l].k, &longer.kv[l].v,
                                                                    &longer.kv[l].k_scale, &longer.kv[l].v_scale,
                                                                    &longer.kv[l].pooled};
            for (size_t part = 0; part < 5; ++part) rebuilt[l][part].resize(parts[part]->size());
        }
        auto sink = [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
            if (layer >= rebuilt.size() || offset > rebuilt[layer][part].size() ||
                n > rebuilt[layer][part].size() - offset) return false;
            std::memcpy(rebuilt[layer][part].data() + offset, data, n);
            return true;
        };
        check(session_file_apply_streamed(m.path, id, sink, base_sizes, error), "torn: the base applies");
        uint64_t at = 0;
        int applied = 0;
        while (at < m.delta_committed) {
            SessionDeltaRecord rec;
            uint64_t next = 0;
            check(session_delta_read(dlog.string(), at, m.delta_committed, id, lim, rec, next, error),
                  "torn: a committed record reads");
            check(session_delta_apply(dlog.string(), at, id, rec, sink, error), "torn: a committed record applies");
            at = next;
            ++applied;
        }
        check(applied == 2, "torn: exactly the two committed records apply");
        bool rebuilt_same = true;
        for (size_t l = 0; rebuilt_same && l < longer.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&longer.kv[l].k, &longer.kv[l].v,
                                                                    &longer.kv[l].k_scale, &longer.kv[l].v_scale,
                                                                    &longer.kv[l].pooled};
            for (size_t part = 0; rebuilt_same && part < 5; ++part) {
                std::vector<uint8_t> want(parts[part]->size());
                if (!want.empty())
                    parts[part]->visit(0, want.size(), [&](const uint8_t* p, size_t c, size_t at2) {
                        std::memcpy(want.data() + at2, p, c);
                        return true;
                    });
                rebuilt_same = rebuilt[l][part] == want;
            }
        }
        check(rebuilt_same, "torn: base + records rebuild the newest state exactly");
        fs::remove_all(tdir);
    }

    // a chain whose base file is gone: a clean miss at reopen, and the owned chain files go with the sidecar
    {
        const fs::path mdir = fs::temp_directory_path() / "strata_spill_missing_base_test";
        fs::remove_all(mdir);
        SavedConversation base = sample(1200);
        SavedConversation grown = base;
        for (int i = 0; i < 200; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        ConversationSpillCache c1;
        size_t w = 0;
        check(c1.open(mdir, id, 1ull << 30, error, true), "missing-base: open");
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "missing-base: base");
        size_t wd = 0;
        check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
              "missing-base: append");
        const fs::path sess = c1.newest_path();
        const fs::path meta = fs::path(sess).replace_extension(".meta");
        const fs::path dlog = fs::path(sess).replace_extension(".dlog");
        const fs::path dchain = fs::path(sess).replace_extension(".dchain");
        check(fs::remove(sess), "missing-base: the base file goes");
        ConversationSpillCache c2;
        check(c2.open(mdir, id, 1ull << 30, error, true) && c2.size() == 0, "missing-base: a clean miss");
        check(!fs::exists(meta) && !fs::exists(dlog) && !fs::exists(dchain),
              "missing-base: the owned chain files go with the sidecar");
        fs::remove_all(mdir);
    }

    // A sidecar missing after a crash cannot make a surviving base/log a usable entry; the owned orphan group is swept.
    {
        const fs::path sdir = fs::temp_directory_path() / "strata_spill_missing_sidecar_test";
        fs::remove_all(sdir);
        SavedConversation base = sample(1000), grown = sample(1200);
        ConversationSpillCache c1;
        size_t w = 0, wd = 0;
        check(c1.open(sdir, id, 1ull << 30, error, true), "missing-sidecar: open");
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "missing-sidecar: base");
        check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
              "missing-sidecar: append");
        const fs::path sess = c1.newest_path();
        const fs::path meta = fs::path(sess).replace_extension(".meta");
        const fs::path dlog = fs::path(sess).replace_extension(".dlog");
        const fs::path dchain = fs::path(sess).replace_extension(".dchain");
        check(fs::remove(meta), "missing-sidecar: remove metadata to simulate interruption");
        ConversationSpillCache reopened;
        check(reopened.open(sdir, id, 1ull << 30, error, true), "missing-sidecar: reopen");
        check(reopened.size() == 0 && reopened.bytes() == 0,
              "missing-sidecar: an entry without ownership metadata is a clean miss");
        check(!fs::exists(sess) && !fs::exists(dlog) && !fs::exists(dchain),
              "missing-sidecar: the orphan group is removed without following external paths");
        fs::remove_all(sdir);
    }

    // A surviving .dlog without its manifest is an incomplete chain, never a plain snapshot carrying newer IDs.
    {
        const fs::path ldir = fs::temp_directory_path() / "strata_spill_missing_manifest_test";
        fs::remove_all(ldir);
        SavedConversation base = sample(1000), grown = sample(1200);
        ConversationSpillCache c1;
        size_t w = 0, wd = 0;
        check(c1.open(ldir, id, 1ull << 30, error, true), "missing-manifest: open");
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "missing-manifest: base");
        check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
              "missing-manifest: append");
        const fs::path sess = c1.newest_path();
        const fs::path meta = fs::path(sess).replace_extension(".meta");
        const fs::path dlog = fs::path(sess).replace_extension(".dlog");
        const fs::path dchain = fs::path(sess).replace_extension(".dchain");
        check(fs::remove(dchain), "missing-manifest: remove publication marker");
        ConversationSpillCache reopened;
        check(reopened.open(ldir, id, 1ull << 30, error), "missing-manifest: reopen in base-only mode");
        check(reopened.size() == 0 && reopened.bytes() == 0,
              "missing-manifest: the log prevents fallback to a plain base entry");
        check(!fs::exists(sess) && !fs::exists(meta) && !fs::exists(dlog),
              "missing-manifest: incomplete owned artifacts are cleaned together");
        fs::remove_all(ldir);
    }

    // The base-only restore mode safely misses a real chain without deleting or loading the valid disk-only entry.
    {
        const fs::path xdir = fs::temp_directory_path() / "strata_spill_cross_mode_chain_test";
        fs::remove_all(xdir);
        SavedConversation base = sample(1000), grown = sample(1200);
        ConversationSpillCache disk_only;
        size_t w = 0, wd = 0;
        check(disk_only.open(xdir, id, 1ull << 30, error, true), "cross-mode: disk-only open");
        check(disk_only.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "cross-mode: base");
        check(disk_only.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
              "cross-mode: real chain append");
        const std::string sess = disk_only.newest_path();
        const uint64_t charged = disk_only.bytes();
        const fs::path meta = fs::path(sess).replace_extension(".meta");
        const fs::path dlog = fs::path(sess).replace_extension(".dlog");
        const fs::path dchain = fs::path(sess).replace_extension(".dchain");
        ConversationSpillCache base_only;
        check(base_only.open(xdir, id, 1ull << 30, error), "cross-mode: base-only open");
        check(base_only.size() == 1 && base_only.bytes() == charged,
              "cross-mode: the valid chain remains indexed and accounted");
        check(!bool(base_only.best(grown.live.ids, {}, grown.cvec, 0.0, 0)),
              "cross-mode: base-only matching skips chain metadata");
        check(base_only.find(sess) == base_only.size(), "cross-mode: append lookup skips unsupported chain");
        SavedConversation loaded;
        check(!base_only.load(sess, loaded, {}, error), "cross-mode: plain load refuses chain-backed state");
        check(fs::exists(sess) && fs::exists(meta) && fs::exists(dlog) && fs::exists(dchain),
              "cross-mode: unsupported mode does not delete a valid chain");
        fs::remove_all(xdir);
    }

    // Deleting a symlink removes only the link and debits the entry's prior charge, never the external target size.
#ifndef _WIN32
    {
        const fs::path sdir = fs::temp_directory_path() / "strata_spill_symlink_accounting_test";
        fs::remove_all(sdir);
        ConversationSpillCache c1;
        check(c1.open(sdir, id, 1ull << 30, error), "symlink-accounting: open");
        SavedConversation image = sample(1000);
        size_t written = 0;
        check(c1.spill_streamed(meta_of(image), sources_of(image, 0), written, error), "symlink-accounting: save");
        const std::string sess = c1.newest_path();
        const fs::path target = sdir / "outside-target.bin";
        const uint64_t target_size = 2ull << 20;
        { std::ofstream out(target, std::ios::binary); std::vector<char> payload((size_t) target_size, 'X'); out.write(payload.data(), payload.size()); }
        check(fs::remove(sess), "symlink-accounting: replace the indexed base");
        fs::create_symlink(target.filename(), sess);
        const uint64_t before = c1.bytes();
        check(before == written, "symlink-accounting: legacy v1 charge excludes its sidecar");
        check(!bool(c1.best(image.live.ids, {}, image.cvec, 0.0, 0)),
              "symlink-accounting: a symlink is not a regular session hit");
        std::string remove_error;
        check(c1.erase(sess, remove_error), "symlink-accounting: unlink the owned path");
        check(c1.bytes() == 0, "symlink-accounting: only previously charged entry bytes are debited");
        check(fs::file_size(target) == target_size, "symlink-accounting: the external target remains untouched");
        fs::remove_all(sdir);
    }
#endif

    // the disk budget evicts a delta chain as one entry: the base, the sidecar, the log and the manifest go together
    // and the byte accounting is exact (the sidecar and manifest bytes count too)
    {
        const fs::path edir = fs::temp_directory_path() / "strata_spill_evict_chain_test";
        fs::remove_all(edir);
        SavedConversation base = sample(1000);
        SavedConversation grown = base;
        for (int i = 0; i < 200; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        ConversationSpillCache c1;
        size_t w = 0;
        check(c1.open(edir, id, 1ull << 30, error, true), "evict-chain: open");
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "evict-chain: base");
        const uint64_t meta_sz = fs::file_size(fs::path(c1.newest_path()).replace_extension(".meta"));
        // a budget that fits the base and its sidecar: the append pushes the chain over it and the whole entry goes
        ConversationSpillCache c2;
        check(c2.open(edir, id, w + meta_sz, error, true), "evict-chain: open at the base's bytes plus its sidecar");
        check(c2.size() == 1 && c2.bytes() == w + meta_sz, "evict-chain: the base and its sidecar fit");
        const fs::path sess = c2.newest_path();
        const fs::path meta = fs::path(sess).replace_extension(".meta");
        const fs::path dlog = fs::path(sess).replace_extension(".dlog");
        const fs::path dchain = fs::path(sess).replace_extension(".dchain");
        size_t wd = 0;
        check(c2.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
              "evict-chain: the append over the budget");
        check(c2.size() == 0 && c2.bytes() == 0 && c2.disk_evictions() == 1,
              "evict-chain: the chain is evicted as one entry, the accounting exact");
        check(!fs::exists(sess) && !fs::exists(meta) && !fs::exists(dlog) && !fs::exists(dchain),
              "evict-chain: all four files go");
        fs::remove_all(edir);
    }

    // the publication gap: the sidecar is written AFTER the chain manifest, so a crash can leave it describing the
    // state before the last record (same token count, different ids).  The restore-time gate compares the chain's
    // tail with the sidecar in full - ids, images and checkpoint lengths - and misses cleanly instead of applying a
    // mix of two saves' state.  Here: a stale sidecar that a token-COUNT check alone cannot catch.
    {
        const fs::path gdir = fs::temp_directory_path() / "strata_spill_pubgap_test";
        fs::remove_all(gdir);
        // the restore-time gate, factored the way generate.cpp's read pass runs it: every committed record reads,
        // then the last record's tail must equal the sidecar in full.
        auto gate_ok = [&](ConversationSpillCache& cache, const std::string& entry_path, uint64_t committed) {
            if (committed == 0) return true;
            const fs::path dlog = fs::path(entry_path).replace_extension(".dlog");
            uint64_t at = 0;
            SessionDeltaRecord rec;
            SessionReadLimits lim;
            while (at < committed) {
                uint64_t next = 0;
                std::string e;
                if (!session_delta_read(dlog.string(), at, committed, id, lim, rec, next, e)) return false;
                if (next <= at || next > committed) return false;
                at = next;
            }
            std::vector<size_t> lengths;
            lengths.reserve(rec.meta.checkpoints.size());
            for (const auto& c : rec.meta.checkpoints) lengths.push_back(c.ids.size());
            return cache.entry_chain_tail_matches(cache.find(entry_path), rec.meta.live.ids, rec.meta.live.imgs,
                                                  lengths);
        };
        ConversationSpillCache c0;
        check(c0.open(gdir, id, 1ull << 30, error, true), "gap: open");
        SavedConversation a = sample(1500);
        size_t w = 0;
        check(c0.spill_streamed(meta_of(a), sources_of(a, 0), w, error), "gap: base saved");
        const fs::path meta_path = fs::path(c0.newest_path()).replace_extension(".meta");
        const std::vector<char> sidecar_a = read_file(meta_path);
        // b: the same token count, but the last 400 ids differ - what a client regenerating the tail would send
        SavedConversation b = a;
        for (size_t i = 1100; i < b.live.ids.size(); ++i) b.live.ids[i] = int32_t(77000 + i);
        size_t wb = 0;
        check(c0.spill_delta(0, meta_of(b), sources_of(b, 1024), (int64_t) b.live.ids.size(), wb, error),
              "gap: the record commits");
        // published together, the chain and the sidecar agree: the gate passes
        {
            ConversationSpillCache ok_cache;
            check(ok_cache.open(gdir, id, 1ull << 30, error, true) && ok_cache.size() == 1,
                  "gap: reopen keeps the chain");
            const auto m = ok_cache.best(b.live.ids, {}, b.cvec, 0.0, 0);
            const std::string expected_path = c0.newest_path();
            const bool matches = bool(m);
            const bool committed_matches = m.delta_committed == wb;
            const bool path_matches = m.path == expected_path;
            if (!matches || !committed_matches || !path_matches)
                std::fprintf(stderr, "  gap reopen: mode=delta-on match=%d committed=%llu expected=%llu path='%s' expected_path='%s'\n",
                             (int) matches, (unsigned long long) m.delta_committed,
                             (unsigned long long) wb, m.path.c_str(), expected_path.c_str());
            check(matches && committed_matches && path_matches, "gap: the committed chain matches");
            check(gate_ok(ok_cache, m.path, m.delta_committed), "gap: a consistent chain passes the tail check");
        }
        // simulate the interruption: roll the sidecar back to a's (stale: same count, different ids)
        write_file(meta_path, sidecar_a);
        ConversationSpillCache gap_cache;
        check(gap_cache.open(gdir, id, 1ull << 30, error, true) && gap_cache.size() == 1,
              "gap: the stale sidecar still indexes (a clean miss is the reader's call, not open's)");
        const auto stale = gap_cache.best(b.live.ids, {}, b.cvec, 0.0, 0);
        const std::string expected_path = c0.newest_path();
        const bool stale_matches = bool(stale);
        const bool committed_matches = stale.delta_committed == wb;
        const bool path_matches = stale.path == expected_path;
        if (!stale_matches || !committed_matches || !path_matches)
            std::fprintf(stderr, "  gap stale reopen: mode=delta-on match=%d committed=%llu expected=%llu path='%s' expected_path='%s'\n",
                         (int) stale_matches, (unsigned long long) stale.delta_committed,
                         (unsigned long long) wb, stale.path.c_str(), expected_path.c_str());
        check(stale_matches && committed_matches && path_matches, "gap: the sidecar offers the chain");
        check(stale.live_tokens == (int64_t) b.live.ids.size(),
              "gap: the count alone cannot tell the two states apart");
        check(!gate_ok(gap_cache, stale.path, stale.delta_committed),
              "gap: the full tail comparison catches what the count cannot");
        fs::remove_all(gdir);
    }

    // a foreign identity's chain is kept whole, never deleted: not indexed, not matched, outside the budget; a later
    // own-identity pass still finds everything (the foreign pass claims nothing)
    {
        const fs::path fdir = fs::temp_directory_path() / "strata_spill_foreign_chain_test";
        fs::remove_all(fdir);
        SavedConversation base = sample(1000);
        SavedConversation grown = base;
        for (int i = 0; i < 200; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        ConversationSpillCache own;
        size_t w = 0;
        check(own.open(fdir, id, 1ull << 30, error, true), "foreign-chain: open");
        check(own.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "foreign-chain: base");
        size_t wd = 0;
        check(own.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
              "foreign-chain: append");
        const fs::path sess = own.newest_path();
        const fs::path meta = fs::path(sess).replace_extension(".meta");
        const fs::path dlog = fs::path(sess).replace_extension(".dlog");
        const fs::path dchain = fs::path(sess).replace_extension(".dchain");
        ConversationSpillCache foreign;
        check(foreign.open(fdir, SessionFileIdentity{1, 2}, 1ull << 30, error), "foreign-chain: foreign open");
        check(foreign.size() == 0 && foreign.bytes() == 0, "foreign-chain: nothing indexed or counted");
        check(fs::exists(sess) && fs::exists(meta) && fs::exists(dlog) && fs::exists(dchain),
              "foreign-chain: all four files kept, none deleted");
        ConversationSpillCache back;
        check(back.open(fdir, id, 1ull << 30, error, true) && back.size() == 1,
              "foreign-chain: the own identity finds it again");
        const auto m = back.best(grown.live.ids, {}, grown.cvec, 0.0, 0);
        const bool matches = bool(m);
        const bool committed_matches = m.delta_committed == wd;
        const bool path_matches = m.path == sess.string();
        if (!matches || !committed_matches || !path_matches)
            std::fprintf(stderr, "  foreign-chain reopen: mode=delta-on match=%d committed=%llu expected=%llu path='%s' expected_path='%s'\n",
                         (int) matches, (unsigned long long) m.delta_committed,
                         (unsigned long long) wd, m.path.c_str(), sess.string().c_str());
        check(matches && committed_matches && path_matches, "foreign-chain: the chain matches again");
        fs::remove_all(fdir);
    }

    // the budget evicts the least recently used, not first-written: a restore's touch makes an old entry the newest,
    // and the stamps survive a reopen (a v1 sidecar's unknown age sorts as the oldest)
    {
        const fs::path ldir = fs::temp_directory_path() / "strata_spill_lru_test";
        fs::remove_all(ldir);
        ConversationSpillCache lru;
        check(lru.open(ldir, id, 1ull << 30, error, true), "lru: open");
        SavedConversation c1 = sample(1000);
        SavedConversation c2 = sample(1500);   // the largest: its eviction must free enough for the budget below
        SavedConversation c3 = sample(900);
        size_t w = 0, w1 = 0, w2 = 0, w3 = 0;
        check(lru.spill_streamed(meta_of(c1), sources_of(c1, 0), w, error), "lru: c1 saved");
        const std::string p1 = lru.newest_path();
        w1 = w;
        const uint64_t m1 = fs::file_size(fs::path(p1).replace_extension(".meta"));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        check(lru.spill_streamed(meta_of(c2), sources_of(c2, 0), w, error), "lru: c2 saved");
        const std::string p2 = lru.newest_path();
        w2 = w;
        const uint64_t m2 = fs::file_size(fs::path(p2).replace_extension(".meta"));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        check(lru.spill_streamed(meta_of(c3), sources_of(c3, 0), w, error), "lru: c3 saved");
        const std::string p3 = lru.newest_path();
        w3 = w;
        const uint64_t m3 = fs::file_size(fs::path(p3).replace_extension(".meta"));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        lru.touch(p1);   // c1 was used again - it is the newest now, though it was saved first
        check(lru.entry_last_used(lru.find(p1)) > lru.entry_last_used(lru.find(p3)), "lru: a touch stamps the newest");
        check(lru.bytes() == w1 + w2 + w3 + m1 + m2 + m3, "lru: the three and their sidecars are accounted");
        // a budget that fits two of the three: reopening evicts c2, the least recently used
        ConversationSpillCache reopened;
        check(reopened.open(ldir, id, w1 + w3 + m1 + m3 + 16, error, true), "lru: reopen with a budget for two");
        check(reopened.size() == 2, "lru: one entry evicted");
        check(reopened.disk_evictions() >= 1, "lru: the eviction is counted");
        check(reopened.find(p2) == reopened.size(), "lru: the least recently used one is gone");
        check(reopened.find(p1) != reopened.size() && reopened.find(p3) != reopened.size(),
              "lru: the touched and the newer one stay");
        check(reopened.entry_last_used(reopened.find(p1)) > reopened.entry_last_used(reopened.find(p3)),
              "lru: the last-use stamps are persisted");
        fs::remove_all(ldir);
    }

    // a v1 sidecar (written before the last-use stamp existed) is still indexed and restorable; its unknown age
    // sorts as the oldest, so an upgrade never drops the cache
    {
        const fs::path vdir = fs::temp_directory_path() / "strata_spill_v1_test";
        fs::remove_all(vdir);
        fs::create_directories(vdir);
        SavedConversation v = sample(1000);
        size_t w = 0;
        check(session_file_write((vdir / "strata-conv-1.sess").string(), v, id, w, error), "v1: the session file");
        {
            std::ofstream out(vdir / "strata-conv-1.meta", std::ios::binary | std::ios::trunc);
            SessionHasher h(0);
            auto raw = [&](const void* p, size_t n) {
                out.write(static_cast<const char*>(p), (std::streamsize) n);
                h.update(p, n);
            };
            auto u64 = [&](uint64_t x) { raw(&x, 8); };
            auto i64 = [&](int64_t x) { raw(&x, 8); };
            raw("SCSM", 4);
            u64(1);                        // the v1 version: no last-use stamp follows the checkpoint lengths
            u64(id.model);
            u64(id.config);
            u64(v.cvec ? 1 : 0);
            u64(v.live.ids.size());
            for (int32_t x : v.live.ids) i64(x);
            u64(v.live.imgs.size());
            for (const auto& k : v.live.imgs) { i64(k.start); u64(k.hash); }
            u64(v.checkpoints.size());
            for (const auto& c : v.checkpoints) u64(c.ids.size());
            u64(h.digest());
        }
        ConversationSpillCache v1;
        check(v1.open(vdir, id, 1ull << 30, error), "v1: open keeps it");
        check(v1.size() == 1, "v1: indexed");
        check(v1.bytes() == w, "v1: the budget counts the session file alone");
        check(v1.entry_last_used(0) == 0, "v1: the unknown age is 0");
        const auto hit = v1.best(v.live.ids, v.live.imgs, v.cvec, 0.0, 0);
        check(bool(hit), "v1: it matches");
        SavedConversation back;
        check(v1.load(hit.path, back, {}, error) && back.live.ids == v.live.ids, "v1: it loads back");
        fs::remove_all(vdir);
    }

    // the flag off keeps the baseline on-disk format: new sidecars are v1 (no last-use stamp), the append is
    // refused, and a directory mixing v1 and v2 sidecars is indexed, matched and restorable by either setting -
    // nothing is deleted, so a pre-delta engine can read the same directory unchanged
    {
        const fs::path odir = fs::temp_directory_path() / "strata_spill_offpath_test";
        fs::remove_all(odir);
        SavedConversation a = sample(1000), b = sample(1500);
        // Keep the two candidates distinct: best() walks newest-first and retains the first equal-length match.
        // These original fixtures shared tokens 1000..1999, so a 1000-token prompt tied and selected b even though
        // this assertion deliberately loads a. The test is about cross-version file compatibility, not tie policy.
        for (auto& token : b.live.ids) token += 40000;
        for (auto& saved : b.checkpoints) for (auto& token : saved.ids) token += 40000;
        size_t w = 0;
        ConversationSpillCache off;
        check(off.open(odir, id, 1ull << 30, error), "off-path: open");
        check(off.spill_streamed(meta_of(a), sources_of(a, 0), w, error), "off-path: save v1");
        const fs::path meta_a = fs::path(off.newest_path()).replace_extension(".meta");
        const std::vector<char> v1bytes = read_file(meta_a);
        uint64_t version = 0;
        std::memcpy(&version, v1bytes.data() + 4, 8);
        check(version == 1, "off-path: the delta-off sidecar is v1");
        size_t wd = 0;
        check(!off.spill_delta(off.find(off.newest_path()), meta_of(a), sources_of(a, 0), 1000, wd, error),
              "off-path: the append refuses with delta off");
        ConversationSpillCache on;
        check(on.open(odir, id, 1ull << 30, error, true), "off-path: reopen with delta on");
        check(on.spill_streamed(meta_of(b), sources_of(b, 0), w, error), "off-path: save v2");
        const fs::path meta_b = fs::path(on.newest_path()).replace_extension(".meta");
        uint64_t version2 = 0;
        const std::vector<char> v2bytes = read_file(meta_b);
        std::memcpy(&version2, v2bytes.data() + 4, 8);
        check(version2 == 2, "off-path: the delta-on sidecar is v2");
        // mixed v1/v2 directory: either setting indexes both, deletes nothing, matches and loads both
        ConversationSpillCache mixed_off;
        check(mixed_off.open(odir, id, 1ull << 30, error), "off-path: mixed reopen, delta off");
        check(mixed_off.size() == 2, "off-path: both entries indexed with delta off");
        ConversationSpillCache mixed_on;
        check(mixed_on.open(odir, id, 1ull << 30, error, true), "off-path: mixed reopen, delta on");
        check(mixed_on.size() == 2, "off-path: both entries indexed with delta on");
        check(fs::exists(meta_a) && fs::exists(meta_b), "off-path: no sidecar deleted by either setting");
        const auto hit_a = mixed_off.best(a.live.ids, {}, a.cvec, 0.0, 0);
        const auto hit_b = mixed_on.best(b.live.ids, {}, b.cvec, 0.0, 0);
        check(bool(hit_a) && bool(hit_b), "off-path: both match in a mixed directory");
        check(hit_a.path == fs::path(meta_a).replace_extension(".sess").string(),
              "off-path: the prompt selects the intended v1 entry");
        SavedConversation back;
        const bool loaded_v1 = mixed_off.load(hit_a.path, back, {}, error);
        if (!loaded_v1)
            std::fprintf(stderr, "  off-path v1 load error: %s\n", error.c_str());
        else if (back.live.ids != a.live.ids) {
            size_t mismatch = 0;
            while (mismatch < back.live.ids.size() && mismatch < a.live.ids.size() &&
                   back.live.ids[mismatch] == a.live.ids[mismatch]) ++mismatch;
            std::fprintf(stderr, "  off-path v1 id mismatch at %zu: got %d, expected %d\n", mismatch,
                         mismatch < back.live.ids.size() ? back.live.ids[mismatch] : -1,
                         mismatch < a.live.ids.size() ? a.live.ids[mismatch] : -1);
        }
        check(loaded_v1, "off-path: the v1 entry loads");
        check(back.live.ids == a.live.ids, "off-path: the v1 entry loads back");
        fs::remove_all(odir);
    }

    // the default-off boundary, exactly: a pre-existing plain v1 directory is budgeted on the session file alone -
    // an exact-fit budget keeps the entry and a below-file budget evicts it, the way the pre-feature engine did it
    {
        const fs::path bdir = fs::temp_directory_path() / "strata_spill_v1_budget_test";
        fs::remove_all(bdir);
        ConversationSpillCache c1;
        check(c1.open(bdir, id, 1ull << 30, error), "v1-budget: open");
        SavedConversation v = sample(1000);
        size_t vw = 0;
        check(c1.spill_streamed(meta_of(v), sources_of(v, 0), vw, error), "v1-budget: save");
        check(c1.bytes() == vw, "v1-budget: the session file alone is counted");
        ConversationSpillCache exact;
        check(exact.open(bdir, id, vw, error), "v1-budget: reopen at the exact file budget");
        check(exact.size() == 1, "v1-budget: an exact-fit budget keeps a legacy entry (the baseline boundary)");
        check(exact.bytes() == vw, "v1-budget: exact-fit accounting remains the session file alone");
        ConversationSpillCache tight;
        check(tight.open(bdir, id, vw - 16, error), "v1-budget: reopen just under the file budget");
        check(tight.size() == 0 && tight.disk_evictions() == 1, "v1-budget: a below-file budget evicts it");
        check(tight.bytes() == 0, "v1-budget: below-fit eviction debits only this entry's session bytes");
        fs::remove_all(bdir);
    }

    // a long chain is validated and applied one record at a time: 40 records, a per-record bound refuses the walk
    // mid-chain, and base + every record rebuilds the newest state byte for byte
    {
        const fs::path ndir = fs::temp_directory_path() / "strata_spill_many_records_test";
        fs::remove_all(ndir);
        ConversationSpillCache c1;
        check(c1.open(ndir, id, 1ull << 30, error, true), "many-records: open");
        SavedConversation base = sample(1000);
        size_t w = 0;
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "many-records: base");
        SavedConversation grown = base;
        uint64_t expected_committed = 0;
        for (int i = 1; i <= 40; ++i) {
            grown.live.ids.push_back(int32_t(9000 + i));
            size_t wd = 0;
            check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 64), (int64_t) grown.live.ids.size(), wd,
                                 error), "many-records: append");
            expected_committed += wd;
        }
        ConversationSpillCache c2;
        check(c2.open(ndir, id, 1ull << 30, error, true) && c2.size() == 1, "many-records: reopen keeps the chain");
        const auto m = c2.best(grown.live.ids, {}, grown.cvec, 0.0, 0);
        const bool matches = bool(m) && m.live_tokens == (int64_t) grown.live.ids.size();
        const bool committed_matches = m.delta_committed == expected_committed;
        const bool path_matches = m.path == c1.newest_path();
        if (!matches || !committed_matches || !path_matches)
            std::fprintf(stderr, "  many-records reopen: mode=delta-on match=%d committed=%llu expected=%llu path='%s' expected_path='%s'\n",
                         (int) matches, (unsigned long long) m.delta_committed,
                         (unsigned long long) expected_committed, m.path.c_str(), c1.newest_path().c_str());
        check(matches && committed_matches && path_matches, "many-records: the chain matches the newest state");
        // per-record read bounds refuse the walk mid-chain (each record is validated one at a time, before the tail)
        SessionReadLimits tight;
        tight.max_tokens = base.live.ids.size() + 20;
        std::string e2;
        const std::string dlog = fs::path(m.path).replace_extension(".dlog").string();
        uint64_t at = 0;
        bool refused = false;
        while (at < m.delta_committed) {
            SessionDeltaRecord rec;
            uint64_t next = 0;
            if (!session_delta_read(dlog, at, m.delta_committed, id, tight, rec, next, e2)) { refused = true; break; }
            at = next;
        }
        check(refused && at > 0 && at < m.delta_committed,
              "many-records: a per-record bound refuses the chain mid-walk");
        // exact reconstruction: the base, then every record in order, one at a time
        SessionReadLimits lim;
        SavedConversation base_meta;
        std::vector<std::array<uint64_t, 5>> base_sizes;
        size_t base_bytes = 0;
        check(session_file_read_streamed(m.path, id, base_meta, base_bytes, e2, lim, &base_sizes),
              "many-records: the base reads streamed");
        std::vector<std::array<std::vector<uint8_t>, 5>> rebuilt(grown.kv.size());
        for (size_t l = 0; l < grown.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&grown.kv[l].k, &grown.kv[l].v,
                                                                    &grown.kv[l].k_scale, &grown.kv[l].v_scale,
                                                                    &grown.kv[l].pooled};
            for (size_t part = 0; part < 5; ++part) rebuilt[l][part].resize(parts[part]->size());
        }
        auto sink = [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
            if (layer >= rebuilt.size() || offset > rebuilt[layer][part].size() ||
                n > rebuilt[layer][part].size() - offset) return false;
            std::memcpy(rebuilt[layer][part].data() + offset, data, n);
            return true;
        };
        check(session_file_apply_streamed(m.path, id, sink, base_sizes, e2), "many-records: the base applies");
        at = 0;
        int applied = 0;
        while (at < m.delta_committed) {
            SessionDeltaRecord rec;
            uint64_t next = 0;
            check(session_delta_read(dlog, at, m.delta_committed, id, lim, rec, next, e2),
                  "many-records: a committed record reads");
            check(session_delta_apply(dlog, at, id, rec, sink, e2), "many-records: a committed record applies");
            at = next;
            ++applied;
        }
        check(applied == 40, "many-records: exactly the 40 committed records apply");
        bool rebuilt_same = true;
        for (size_t l = 0; rebuilt_same && l < grown.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&grown.kv[l].k, &grown.kv[l].v,
                                                                    &grown.kv[l].k_scale, &grown.kv[l].v_scale,
                                                                    &grown.kv[l].pooled};
            for (size_t part = 0; rebuilt_same && part < 5; ++part) {
                std::vector<uint8_t> want(parts[part]->size());
                if (!want.empty())
                    parts[part]->visit(0, want.size(), [&](const uint8_t* p, size_t c, size_t at2) {
                        std::memcpy(want.data() + at2, p, c);
                        return true;
                    });
                rebuilt_same = rebuilt[l][part] == want;
            }
        }
        check(rebuilt_same, "many-records: base + 40 records rebuild the newest state exactly");
        // The walk preflight accounts for actual object/vector storage and the read/apply overlap, never log length.
        // The conversation-file control-path fixture also exercises admission refusal and a streamed payload larger
        // than its resident allowance; these assertions only check the bounded formula's shape.
        {
            SessionReadLimits bounded;
            bounded.max_tokens = 64;
            bounded.max_checkpoints = 1;
            bounded.max_kv_layers = 1;
            bounded.max_state_bytes = {4096, 4096, 4096, 4096, 4096};
            const uint64_t old_payload_only_estimate =
                64ull * (sizeof(int32_t) + sizeof(ConversationImageKey)) + 5ull * 4096 + 64;
            const uint64_t metadata = session_read_meta_peak_bytes(bounded);
            const uint64_t walk = session_delta_walk_peak_bytes(bounded);
            check(metadata > 2 * old_payload_only_estimate && metadata < 4 * old_payload_only_estimate,
                  "walk preflight: one metadata state includes object headers and page-rounded vector capacity");
            check(walk > metadata + (16ull << 20) && walk < (17ull << 20),
                  "walk preflight: overlapping record/parser metadata and one bounded I/O block");
            check(session_delta_walk_peak_bytes(lim) >= session_read_meta_peak_bytes(lim),
                  "walk preflight: an open-limit walk saturates safely");
            SessionReadLimits generous;
            generous.max_tokens = uint64_t(grown.live.ids.size()) * 4;
            generous.max_checkpoints = 64;
            generous.max_kv_layers = grown.kv.size();
            generous.max_state_bytes.fill(2ull << 20);
            check(session_delta_walk_peak_bytes(generous) > m.delta_committed,
                  "walk preflight: generous finite limits can exceed this log without pricing its length");
        }
        fs::remove_all(ndir);
    }

    // the manifest's fixed temp name is created exclusively with no link following, the way the session files and
    // the sidecar's temp are (conversation_file.cpp): a pre-existing ordinary file or a symlink is refused, and a
    // symlink's target is never truncated
    {
        const fs::path cdir = fs::temp_directory_path() / "strata_spill_chain_tmp_test";
        fs::remove_all(cdir);
        ConversationSpillCache c1;
        check(c1.open(cdir, id, 1ull << 30, error, true), "chain-tmp: open");
        SavedConversation base = sample(1000);
        size_t w = 0;
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "chain-tmp: base");
        SavedConversation grown = base;
        for (int i = 0; i < 200; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        const std::string sess = c1.newest_path();
        const fs::path chain = fs::path(sess).replace_extension(".dchain");
        const fs::path tmp = fs::path(chain.string() + ".tmp");

        {   // an ordinary pre-existing file at the temp name: refused, left as it was
            { std::ofstream(tmp.string().c_str(), std::ios::binary | std::ios::trunc) << "sentinel-bytes"; }
            size_t wd = 0;
            check(!c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
                  "chain-tmp: an existing temp file refuses the append");
            check(read_file(tmp) == std::vector<char>({'s', 'e', 'n', 't', 'i', 'n', 'e', 'l', '-', 'b', 'y', 't', 'e', 's'}),
                  "chain-tmp: the pre-existing temp file is untouched");
            check(!fs::exists(chain), "chain-tmp: no manifest is published over the refusal");
            check(fs::remove(tmp), "chain-tmp: pre-existing temp removed");
        }
        {   // A symlink at the temp name must not be followed or removed as an unowned temp. Windows runs this when
            // the host permits symlink creation; the ordinary sentinel case above remains portable and mandatory.
            const fs::path linkdir = cdir.string() + "_link";
            fs::remove_all(linkdir);
            ConversationSpillCache linked;
            check(linked.open(linkdir, id, 1ull << 30, error, true), "chain-tmp: link fixture open");
            SavedConversation link_base = sample(1000);
            check(linked.spill_streamed(meta_of(link_base), sources_of(link_base, 0), w, error),
                  "chain-tmp: link fixture base");
            SavedConversation link_grown = link_base;
            for (int i = 0; i < 200; ++i) link_grown.live.ids.push_back(int32_t(2500 + i));
            const fs::path link_chain = fs::path(linked.newest_path()).replace_extension(".dchain");
            const fs::path link_tmp = link_chain.string() + ".tmp";
            const fs::path target = linkdir / "sentinel-owner.bin";
            { std::ofstream(target.string().c_str(), std::ios::binary | std::ios::trunc) << "precious"; }
            std::error_code link_error;
            fs::create_symlink(target.filename(), link_tmp, link_error);
#ifdef _WIN32
            if (link_error)
                std::fprintf(stderr, "SKIP: chain-tmp Windows symlink fixture unavailable: %s\n", link_error.message().c_str());
#else
            check(!link_error, "chain-tmp: create symlink sentinel");
#endif
            if (!link_error) {
                size_t wd = 0;
                check(!linked.spill_delta(0, meta_of(link_grown), sources_of(link_grown, 1024),
                                          (int64_t) link_grown.live.ids.size(), wd, error),
                      "chain-tmp: a symlinked temp refuses the append");
                check(read_file(target) == std::vector<char>({'p', 'r', 'e', 'c', 'i', 'o', 'u', 's'}),
                      "chain-tmp: the link's target is never written through");
                check(fs::is_symlink(link_tmp), "chain-tmp: the symlink itself is left for the operator");
                check(!fs::exists(link_chain), "chain-tmp: no manifest is published over the symlink");
                fs::remove(link_tmp);
            }
            fs::remove_all(linkdir);
        }
        fs::remove_all(cdir);
    }

    // Partial removal debits only bytes already counted for this entry. The ownership sidecar and manifest remain
    // beside a survivor, so a restart can retain an ineligible ledger and retry without double-debiting.
    {
        const fs::path ddir = fs::temp_directory_path() / "strata_spill_delete_fail_test";
        fs::remove_all(ddir);
        SavedConversation base = sample(1000);
        SavedConversation grown = base;
        for (int i = 0; i < 200; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        ConversationSpillCache c1;
        check(c1.open(ddir, id, 1ull << 30, error, true), "delete-fail: open");
        size_t w = 0;
        check(c1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "delete-fail: chain base");
        size_t wd = 0;
        check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wd, error),
              "delete-fail: append");
        const uint64_t total_a = c1.bytes();
        const std::string sp = c1.newest_path();
        SavedConversation other = sample(800);
        check(c1.spill_streamed(meta_of(other), sources_of(other, 0), w, error), "delete-fail: second entry");
        const uint64_t total_b = c1.bytes() - total_a;

        ConversationSpillCache c2;
        check(c2.open(ddir, id, total_a + total_b, error, true), "delete-fail: reopen at the exact budget");
        const fs::path dlog = fs::path(sp).replace_extension(".dlog");
        const uint64_t s_sz = fs::file_size(sp);
        {
            std::error_code ec;
            fs::remove(dlog, ec);
            fs::create_directory(dlog);
            { std::ofstream((dlog / "pinned").string().c_str()) << "busy"; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        SavedConversation third = sample(700);
        for (auto& t : third.live.ids) t += 40000;
        size_t w3 = 0;
        check(c2.spill_streamed(meta_of(third), sources_of(third, 0), w3, error), "delete-fail: third entry");
        const uint64_t total_c = w3 + fs::file_size(fs::path(c2.newest_path()).replace_extension(".meta"));
        const uint64_t owned_remainder = total_a - s_sz;
        check(c2.size() == 2, "delete-fail: the incomplete chain stays indexed, the complete entry went");
        check(c2.disk_evictions() == 1, "delete-fail: exactly one complete eviction is counted");
        check(c2.bytes() == owned_remainder + total_c,
              "delete-fail: partial eviction debits the removed base but retains log, manifest and sidecar charges");
        const fs::path meta = fs::path(sp).replace_extension(".meta");
        const fs::path dchain = fs::path(sp).replace_extension(".dchain");
        check(!fs::exists(sp) && fs::is_directory(dlog) && fs::exists(meta) && fs::exists(dchain),
              "delete-fail: preserve ownership metadata beside the blocking log path");
        check(c2.find(sp) == c2.size(), "delete-fail: an incomplete entry is not found for append");
        check(!bool(c2.best(grown.live.ids, {}, grown.cvec, 0.0, 0)),
              "delete-fail: an incomplete chain is not offered as a hit");
        SavedConversation unavailable;
        check(!c2.load(sp, unavailable, {}, error), "delete-fail: an incomplete base cannot be loaded");
        size_t rejected_append = 0;
        check(!c2.spill_delta(0, meta_of(grown), sources_of(grown, 1024),
                              (int64_t) grown.live.ids.size(), rejected_append, error),
              "delete-fail: an incomplete entry refuses append");

        // Reopen retains the same remaining debit using the manifest's committed length, without making the entry
        // eligible. A retry against the still-blocked path changes no accounting.
        ConversationSpillCache c3;
        check(c3.open(ddir, id, total_a + total_b, error, true), "delete-fail: reopen partial ownership");
        check(c3.size() == 2 && c3.find(sp) == c3.size(), "delete-fail: reopen retains but hides the incomplete entry");
        check(c3.bytes() == owned_remainder + total_c, "delete-fail: reopen accounts the exact remaining owned bytes");
        check(!c3.load(sp, unavailable, {}, error), "delete-fail: reopened incomplete state still cannot load");
        check(!c3.spill_delta(0, meta_of(grown), sources_of(grown, 1024),
                              (int64_t) grown.live.ids.size(), rejected_append, error),
              "delete-fail: reopened incomplete entry still refuses append");
        SavedConversation longer = grown;
        for (int i = 0; i < 100; ++i) longer.live.ids.push_back(int32_t(3000 + i));
        longer.checkpoints.push_back(checkpoint(grown.live.ids.size(), 9));
        size_t wl = 0;
        check(c3.spill_streamed(meta_of(longer), sources_of(longer, 0), wl, error), "delete-fail: full write");
        const std::string kept = c3.newest_path();
        const uint64_t before_drop = c3.bytes();
        const size_t dropped = c3.drop_superseded(longer.live.ids, {}, longer.checkpoints, longer.cvec, kept);
        check(dropped == 0, "delete-fail: a partial group is not claimed dropped");
        check(c3.find(sp) == c3.size(), "delete-fail: partial entry remains ineligible after drop_superseded");
        check(c3.bytes() == before_drop, "delete-fail: a failed drop does not debit a second time");
        check(fs::exists(dlog) && fs::exists(meta) && fs::exists(dchain),
              "delete-fail: the survivor and its ownership metadata are untouched");

        const uint64_t before_erase = c3.bytes();
        std::string ee;
        check(!c3.erase(sp, ee), "delete-fail: erase reports the survivor");
        check(ee.find(dlog.string()) != std::string::npos, "delete-fail: erase names the surviving path");
        check(c3.find(sp) == c3.size(), "delete-fail: erase does not expose the incomplete entry");
        check(c3.bytes() == before_erase, "delete-fail: retrying the same partial removal does not double-debit");
        fs::remove_all(dlog);
        check(c3.erase(sp, ee), "delete-fail: retry succeeds after the blocking path is gone");
        check(!fs::exists(sp) && !fs::exists(meta) && !fs::exists(dlog) && !fs::exists(dchain),
              "delete-fail: successful retry removes the remaining owned group");
        check(before_erase - c3.bytes() == owned_remainder,
              "delete-fail: retry debits exactly the remaining per-entry ledger");
        fs::remove_all(ddir);
    }

    // --conversation-cache-disk-state-ref (prefix saves only, PLAN 6.5): a prefix whose checkpoint's five
    // running-state arrays equal the live state's is written as a format-v2 file, accounted by its physical bytes,
    // matched and read back with no read-side flag; a RAM refusal keeps the file and the index entry
    {
        const fs::path rdir = fs::temp_directory_path() / "strata_spill_state_ref_test";
        fs::remove_all(rdir);
        auto homo_root = [](SavedConversation root, size_t tokens, int64_t used, uint64_t imghash) {
            root.checkpoints.clear();
            ConversationCheckpoint cp = root.live;   // the five arrays copy: the REF hit; ids/imgs/used stay its own
            cp.ids.assign(root.live.ids.begin(), root.live.ids.begin() + (std::ptrdiff_t) tokens);
            cp.imgs = {{5, imghash}};
            cp.used = used;
            root.checkpoints.push_back(cp);
            return root;
        };
        const SavedConversation root = homo_root(sample(600), 300, 77, 0x777);
        const uint64_t state_bytes = root.live.gdn.size() + root.live.ple.size() + root.live.tails.size() +
                                     root.live.dead.size() + root.live.block_pos.size();
        ConversationSpillCache sp;
        check(sp.open(rdir, id, 1ull << 30, error), "state-ref: open");
        size_t written = 0;
        check(sp.spill_streamed(meta_of(root), sources_of(root, 0), written, error, true, true, true),
              "state-ref: prefix save with the flag");
        const std::string ref_path = sp.newest_path();
        check(fs::path(ref_path).filename().string().rfind("strata-conv-prefix-", 0) == 0, "state-ref: prefix name");
        const std::vector<char> ref_bytes = read_file(ref_path);
        uint32_t version = 0; std::memcpy(&version, ref_bytes.data() + 8, 4);
        check(version == 2, "state-ref: the prefix file is format v2");
        // exact whole-file savings against the same prefix written without the flag (a miss stays byte-identical v1)
        {
            ConversationSpillCache plain;
            const fs::path plain_dir = fs::temp_directory_path() / "strata_spill_state_ref_plain_test";
            check(plain.open(plain_dir, id, 1ull << 30, error), "state-ref: plain open");
            size_t plain_written = 0;
            check(plain.spill_streamed(meta_of(root), sources_of(root, 0), plain_written, error, true, true, false),
                  "state-ref: the same prefix without the flag");
            uint32_t plain_version = 0;
            const std::vector<char> plain_bytes = read_file(plain.newest_path());
            std::memcpy(&plain_version, plain_bytes.data() + 8, 4);
            check(plain_version == 1, "state-ref: without the flag the prefix file is v1");
            check(plain_written - written == state_bytes - 160, "state-ref: exact whole-file savings state_bytes - 160");
            check(written == fs::file_size(ref_path), "state-ref: the accounting size is the physical file size");
            fs::remove_all(plain_dir);
        }
        // delta-off accounting charges the session file alone (the baseline v1 sidecar is unpriced)
        check(sp.bytes() == written, "state-ref: delta-off accounting is the file alone");
        check(sp.prefix_count() == 1 && sp.has_prefix(root.live.ids, {}, root.cvec), "state-ref: indexed as a prefix");
        // no read-side flag: the v2 prefix matches a new conversation and loads back whole
        std::vector<int32_t> new_chat(root.live.ids.begin(), root.live.ids.end());
        new_chat.push_back(424242);
        const auto hit = sp.best(new_chat, {}, root.cvec, 0.0, 0);
        check(bool(hit) && hit.tokens == (int64_t) root.live.ids.size() && hit.live && hit.file_bytes == written,
              "state-ref: the v2 prefix matches a new conversation, the physical size carried");
        SavedConversation back;
        check(sp.load(ref_path, back, {}, error), "state-ref: a v2 prefix loads with no flag");
        check(back.live.ids == root.live.ids && back.checkpoints.size() == 1, "state-ref: round trip");
        check(back.checkpoints[0].ids == root.checkpoints[0].ids &&
              back.checkpoints[0].imgs == root.checkpoints[0].imgs && back.checkpoints[0].used == root.checkpoints[0].used,
              "state-ref: checkpoint ids/imgs/used preserved, not taken from live");
        check(back.checkpoints[0].gdn == back.live.gdn && back.checkpoints[0].block_pos == back.live.block_pos,
              "state-ref: the referenced checkpoint's arrays equal the live state's");
        back.checkpoints[0].gdn[0] ^= 0xff;
        check(back.checkpoints[0].gdn != back.live.gdn && back.live.gdn == root.live.gdn,
              "state-ref: the restored checkpoint owns its copy (no aliasing)");
        // a prefix entry refuses the delta append, the v2 file just as the v1 one
        size_t wd = 0;
        check(!sp.spill_delta(sp.find(ref_path), meta_of(root), sources_of(root, 0), 600, wd, error),
              "state-ref: a prefix entry refuses the delta append");
        // v2 + memory: a refused RAM load keeps the file, the index entry and the match
        {
            SavedConversation wont;
            wont.live.ids = {42};
            SessionReadLimits low;
            size_t calls = 0;
            low.admit = [&](uint64_t, std::string& w) { ++calls; w = "controlled v2 RAM refusal"; return false; };
            check(!sp.load(ref_path, wont, low, error) && calls > 0 &&
                  error.find("controlled v2 RAM refusal") != std::string::npos,
                  "state-ref: a v2 RAM refusal fails the load");
            check(wont.live.ids.size() == 1 && wont.live.ids[0] == 42,
                  "state-ref: the refused load leaves the output alone");
            check(fs::exists(ref_path) && sp.size() == 1 && bool(sp.best(new_chat, {}, root.cvec, 0.0, 0)),
                  "state-ref: v2+memory keeps the file and the index entry");
        }
        // a foreign identity keeps the v2 file whole, indexes nothing
        {
            ConversationSpillCache foreign;
            check(foreign.open(rdir, SessionFileIdentity{3, 4}, 1ull << 30, error), "state-ref: foreign open");
            check(foreign.size() == 0 && fs::exists(ref_path),
                  "state-ref: a foreign identity keeps the v2 file, indexes nothing");
        }
        // a torn v2 file: open() reads the sidecar only, so the entry stays indexed and matched; load() rejects the
        // payload and does not delete the file - the caller decides
        fs::resize_file(ref_path, fs::file_size(ref_path) - 16);
        {
            ConversationSpillCache reopened;
            check(reopened.open(rdir, id, 1ull << 30, error), "state-ref: reopen with a truncated v2 file");
            check(reopened.size() == 1, "state-ref: open() does not read the payload, the entry stays indexed");
            check(bool(reopened.best(new_chat, {}, root.cvec, 0.0, 0)), "state-ref: the sidecar still offers the entry");
            SavedConversation wont;
            check(!reopened.load(ref_path, wont, {}, error), "state-ref: load rejects the truncated v2 file");
            check(fs::exists(ref_path), "state-ref: a rejected load does not delete the file");
        }
        fs::remove_all(rdir);
    }

    // the delta-on accounting around a v2 prefix: the budget counts the v2 file AND its v2 sidecar (the .sess
    // version says nothing about the .meta sidecar); the reopen keeps the match; the budget still evicts ordinary
    // conversations before a prefix; the delta append path itself is untouched by the flag
    {
        const fs::path ddir = fs::temp_directory_path() / "strata_spill_state_ref_delta_test";
        fs::remove_all(ddir);
        SavedConversation root = sample(600);
        root.checkpoints.clear();
        {
            ConversationCheckpoint cp = root.live;
            cp.ids.assign(root.live.ids.begin(), root.live.ids.begin() + 300);
            cp.imgs = {{5, 0x777}};
            cp.used = 77;
            root.checkpoints.push_back(cp);
        }
        ConversationSpillCache d;
        check(d.open(ddir, id, 1ull << 30, error, true), "state-ref delta-on: open");
        size_t written = 0;
        check(d.spill_streamed(meta_of(root), sources_of(root, 0), written, error, true, true, true),
              "state-ref delta-on: v2 prefix save");
        const std::string ref_path = d.newest_path();
        const uint64_t sidecar = fs::file_size(fs::path(ref_path).replace_extension(".meta"));
        check(d.bytes() == written + sidecar, "state-ref delta-on: the budget counts the v2 file and its sidecar");
        check(d.has_prefix(root.live.ids, {}, root.cvec) && d.prefix_count() == 1,
              "state-ref delta-on: indexed as a prefix");
        std::vector<int32_t> new_chat(root.live.ids.begin(), root.live.ids.end());
        new_chat.push_back(424242);
        // two ordinary conversations, then a tight budget: an ordinary one goes, the v2 prefix stays
        SavedConversation c1s = sample(1000);
        size_t w1 = 0;
        check(d.spill_streamed(meta_of(c1s), sources_of(c1s, 0), w1, error), "state-ref delta-on: conversation 1");
        const std::string c1 = d.newest_path();
        SavedConversation c2s = sample(900);
        size_t w2 = 0;
        check(d.spill_streamed(meta_of(c2s), sources_of(c2s, 0), w2, error), "state-ref delta-on: conversation 2");
        const uint64_t m1 = fs::file_size(fs::path(c1).replace_extension(".meta"));
        const uint64_t m2 = fs::file_size(fs::path(d.newest_path()).replace_extension(".meta"));
        ConversationSpillCache tight;
        check(tight.open(ddir, id, written + sidecar + std::max(w1 + m1, w2 + m2) + 16, error, true),
              "state-ref delta-on: reopen with a tight budget");
        check(tight.prefix_count() == 1 && tight.has_prefix(root.live.ids, {}, root.cvec),
              "state-ref delta-on: the v2 prefix survives the budget, still a prefix after the reopen");
        check(tight.size() == 2, "state-ref delta-on: an ordinary conversation was evicted, not the prefix");
        const auto hit = tight.best(new_chat, {}, root.cvec, 0.0, 0);
        check(bool(hit) && hit.path == ref_path, "state-ref delta-on: the reopened v2 prefix still matches");
        // the delta append path is untouched: a REF-eligible conversation still appends a v1 record
        ConversationSpillCache ch;
        const fs::path cdir = ddir / "chain";
        check(ch.open(cdir, id, 1ull << 30, error, true), "state-ref delta-on: chain open");
        SavedConversation base = sample(800);
        size_t wb = 0;
        check(ch.spill_streamed(meta_of(base), sources_of(base, 0), wb, error), "state-ref delta-on: chain base");
        SavedConversation grown = base;
        for (int i = 0; i < 100; ++i) grown.live.ids.push_back(int32_t(6000 + i));
        grown.checkpoints.push_back(checkpoint(grown.live.ids.size(), 9));
        size_t wr = 0;
        check(ch.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wr, error),
              "state-ref delta-on: append over a REF-eligible state still works");
        SessionDeltaRecord rec;
        uint64_t next = 0;
        check(session_delta_read(fs::path(ch.newest_path()).replace_extension(".dlog").string(), 0, wr, id, {}, rec,
                                 next, error) && next == wr && rec.meta.live.ids == grown.live.ids,
              "state-ref delta-on: the record is still a v1 record and reads back");
        fs::remove_all(ddir);
    }

    // RHT is opt-in for ordinary disk spills and delta records; prefix files and slot/RAM saves remain RAW.
    {
        auto session_version = [&](const fs::path& path) {
            const std::vector<char> data = read_file(path);
            uint32_t version = 0;
            if (data.size() >= 12) std::memcpy(&version, data.data() + 8, 4);
            return version;
        };
        auto word64 = [](const std::vector<char>& data, size_t at) {
            uint64_t value = 0;
            if (at <= data.size() && data.size() - at >= sizeof(value)) std::memcpy(&value, data.data() + at, sizeof(value));
            return value;
        };
        const fs::path pdir = dir / "rht-prefix-on";
        const fs::path plain_dir = dir / "rht-prefix-off";
        fs::remove_all(pdir); fs::remove_all(plain_dir);
        const SavedConversation root = rht_sample(600);
        ConversationSpillCache prefix_on;
        check(prefix_on.open(pdir, id, 1ull << 30, error), "rht spill: prefix-on open");
        size_t prefix_on_bytes = 0;
        check(prefix_on.spill_streamed(meta_of(root), sources_of(root, 0), prefix_on_bytes, error,
                                       true, true, false, true),
              "rht spill: prefix true accepts the requested codec flag");
        const std::vector<char> prefix_on_data = read_file(prefix_on.newest_path());
        ConversationSpillCache prefix_off;
        check(prefix_off.open(plain_dir, id, 1ull << 30, error), "rht spill: prefix-off open");
        size_t prefix_off_bytes = 0;
        check(prefix_off.spill_streamed(meta_of(root), sources_of(root, 0), prefix_off_bytes, error,
                                        true, true, false, false),
              "rht spill: prefix baseline save");
        const std::vector<char> prefix_off_data = read_file(prefix_off.newest_path());
        check(session_version(prefix_on.newest_path()) == 1 && prefix_on_data == prefix_off_data,
              "rht spill: prefix forces the codec off and preserves its f32 RAW bytes");

        const fs::path slot_path = dir / "rht-slot-default.sess";
        size_t slot_bytes = 0;
        check(session_file_write(slot_path.string(), root, id, slot_bytes, error) &&
              session_version(slot_path) == 1 && read_file(slot_path) == prefix_off_data,
              "rht spill: default slot SAVE remains byte-identical RAW");
        const fs::path ram_dir = dir / "rht-ram-default";
        ConversationSpillCache ram_spill;
        check(ram_spill.open(ram_dir, id, 1ull << 30, error) && ram_spill.spill(root, error) &&
              session_version(ram_spill.newest_path()) == 1 && read_file(ram_spill.newest_path()) == prefix_off_data,
              "rht spill: the RAM-eviction spill remains byte-identical RAW");

        const fs::path ordinary_dir = dir / "rht-ordinary";
        ConversationSpillCache ordinary;
        check(ordinary.open(ordinary_dir, id, 1ull << 30, error, true), "rht spill: ordinary open");
        size_t base_bytes = 0;
        check(ordinary.spill_streamed(meta_of(root), sources_of(root, 0), base_bytes, error,
                                      false, false, false, true),
              "rht spill: ordinary base accepts codec opt-in");
        const std::string base_path = ordinary.newest_path();
        check(session_version(base_path) == 2, "rht spill: ordinary base uses v2 for actual codec-2 descriptors");
        SavedConversation grown = root;
        for (int i = 0; i < 100; ++i) grown.live.ids.push_back(int32_t(8000 + i));
        size_t delta_bytes = 0;
        check(ordinary.spill_delta(ordinary.find(base_path), meta_of(grown), sources_of(grown, 1024),
                                   (int64_t) grown.live.ids.size(), delta_bytes, error, false, true),
              "rht spill: ordinary delta accepts codec opt-in");
        const fs::path ordinary_log = fs::path(base_path).replace_extension(".dlog");
        const std::vector<char> ordinary_log_bytes = read_file(ordinary_log);
        SessionDeltaRecord ordinary_record;
        uint64_t ordinary_next = 0;
        check(word64(ordinary_log_bytes, 8) == 2 &&
              session_delta_read(ordinary_log.string(), 0, ordinary_log_bytes.size(), id, {}, ordinary_record,
                                 ordinary_next, error) && ordinary_next == ordinary_log_bytes.size(),
              "rht spill: ordinary delta uses v2 and reads back");

        // A base v1 file with a v1 record followed by a v2 record is a supported mixed chain.
        const fs::path mixed_dir = dir / "rht-mixed-chain";
        ConversationSpillCache mixed;
        check(mixed.open(mixed_dir, id, 1ull << 30, error, true), "rht spill: mixed chain open");
        const SavedConversation mixed_base = rht_sample(600);
        size_t mixed_base_bytes = 0;
        check(mixed.spill_streamed(meta_of(mixed_base), sources_of(mixed_base, 0), mixed_base_bytes, error),
              "rht spill: mixed chain v1 base");
        const std::string mixed_base_path = mixed.newest_path();
        SavedConversation mixed_one = mixed_base;
        mixed_one.live.ids.push_back(9001);
        size_t delta_one = 0;
        check(mixed.spill_delta(0, meta_of(mixed_one), sources_of(mixed_one, 1024),
                                (int64_t) mixed_one.live.ids.size(), delta_one, error),
              "rht spill: mixed chain v1 delta");
        SavedConversation mixed_two = mixed_one;
        mixed_two.live.ids.push_back(9002);
        size_t delta_two = 0;
        check(mixed.spill_delta(0, meta_of(mixed_two), sources_of(mixed_two, 1024),
                                (int64_t) mixed_two.live.ids.size(), delta_two, error, false, true),
              "rht spill: mixed chain v2 delta");
        const fs::path mixed_log = fs::path(mixed_base_path).replace_extension(".dlog");
        const std::vector<char> mixed_log_bytes = read_file(mixed_log);
        const uint64_t mixed_limit = mixed_log_bytes.size();
        SessionDeltaRecord mixed_v1, mixed_v2;
        uint64_t mixed_at = 0, mixed_end = 0;
        const bool first_ok = session_delta_read(mixed_log.string(), 0, mixed_limit, id, {}, mixed_v1, mixed_at, error);
        const bool second_ok = first_ok && session_delta_read(mixed_log.string(), mixed_at, mixed_limit, id, {},
                                                               mixed_v2, mixed_end, error);
        const SessionKvBlock discard_kv = [](size_t, size_t, uint64_t, const void*, size_t) { return true; };
        const bool mixed_apply = first_ok && second_ok &&
            session_delta_apply(mixed_log.string(), 0, id, mixed_v1, discard_kv, error) &&
            session_delta_apply(mixed_log.string(), mixed_at, id, mixed_v2, discard_kv, error);
        check(session_version(mixed_base_path) == 1 && word64(mixed_log_bytes, 8) == 1 &&
              first_ok && word64(mixed_log_bytes, (size_t) mixed_at + 8) == 2 && second_ok &&
              mixed_end == mixed_limit && mixed_apply,
              "rht spill: readers and apply accept a v1 base with a mixed v1/v2 delta chain");

        // The opt-in flag does not force v2 when every GDN array falls back to RAW.
        const fs::path fallback_dir = dir / "rht-raw-fallback";
        ConversationSpillCache fallback;
        check(fallback.open(fallback_dir, id, 1ull << 30, error, true), "rht spill: fallback open");
        SavedConversation nonfinite = rht_sample(600);
        const float infinity = std::numeric_limits<float>::infinity();
        auto make_nonfinite = [&](SavedConversation& image) {
            const auto set = [&](ConversationCheckpoint& cp) { std::memcpy(cp.gdn.data(), &infinity, sizeof(infinity)); };
            set(image.live);
            for (auto& cp : image.checkpoints) set(cp);
        };
        make_nonfinite(nonfinite);
        size_t fallback_base_bytes = 0;
        check(fallback.spill_streamed(meta_of(nonfinite), sources_of(nonfinite, 0), fallback_base_bytes, error,
                                      false, false, false, true) && session_version(fallback.newest_path()) == 1,
              "rht spill: all-RAW base fallback stays v1");
        SavedConversation nonfinite_grown = nonfinite;
        nonfinite_grown.live.ids.push_back(9100);
        size_t fallback_delta_bytes = 0;
        check(fallback.spill_delta(0, meta_of(nonfinite_grown), sources_of(nonfinite_grown, 1024),
                                   (int64_t) nonfinite_grown.live.ids.size(), fallback_delta_bytes, error,
                                   false, true),
              "rht spill: all-RAW delta fallback writes");
        const fs::path fallback_log = fs::path(fallback.newest_path()).replace_extension(".dlog");
        const std::vector<char> fallback_log_bytes = read_file(fallback_log);
        check(word64(fallback_log_bytes, 8) == 1,
              "rht spill: all-RAW delta fallback remains v1 even with codec opt-in");
        fs::remove_all(pdir); fs::remove_all(plain_dir); fs::remove_all(ram_dir);
        fs::remove_all(ordinary_dir); fs::remove_all(mixed_dir); fs::remove_all(fallback_dir);
    }

    // format v3 pass-through: spill_streamed/spill_delta forward the K/V-first layout; a v3 base restores whole, a
    // v3 chain restores base + records, and a v1 base accepts a v3 record (the versions are independent).
    {
        const fs::path v3dir = fs::temp_directory_path() / "strata_spill_v3_test";
        fs::remove_all(v3dir);
        auto word64 = [](const std::vector<char>& data, size_t at) {
            uint64_t v = 0;
            std::memcpy(&v, data.data() + at, 8);
            return v;
        };
        auto file_version = [&word64](const std::vector<char>& data) { return uint32_t(word64(data, 8) & 0xffffffffu); };
        SavedConversation base = sample(1200);
        SavedConversation grown = base;
        for (int i = 0; i < 200; ++i) grown.live.ids.push_back(int32_t(2500 + i));
        grown.checkpoints.back() = checkpoint(1400, 8);
        for (size_t i = 0; i < grown.kv.size(); ++i) {
            const std::array<ConversationBuffer*, 5> parts = {&grown.kv[i].k, &grown.kv[i].v, &grown.kv[i].k_scale,
                                                              &grown.kv[i].v_scale, &grown.kv[i].pooled};
            for (size_t part = 0; part < 5; ++part) {
                if (parts[part]->size() <= 1024) continue;
                ConversationBuffer& b = *parts[part];
                b.visit(1024, b.size() - 1024, [&](uint8_t* p, size_t c, size_t at) {
                    for (size_t j = 0; j < c; ++j) p[j] = uint8_t((at + j) * 41u + i + part);
                    return true;
                });
            }
        }
        // a v3 base: the file's header says 3 and a plain load round-trips
        ConversationSpillCache v3;
        size_t w = 0;
        check(v3.open(v3dir, id, 1ull << 30, error, true), "v3 spill: open");
        check(v3.spill_streamed(meta_of(base), sources_of(base, 0), w, error, false, false, false, false,
                                nullptr, true),
              "v3 spill: the streamed save forwards the layout");
        const std::vector<char> v3_file = read_file(v3.newest_path());
        check(file_version(v3_file) == 3, "v3 spill: the base file is version 3");
        SavedConversation back;
        check(v3.load(v3.newest_path(), back, {}, error) && back.live.ids == base.live.ids &&
              back.kv.size() == base.kv.size(), "v3 spill: the base loads whole");
        // a v3 record on it: the chain survives a reopen and restores base + record exactly
        ConversationSpillCache c1;
        check(c1.open(v3dir, id, 1ull << 30, error, true), "v3 chain: open");
        size_t w1 = 0;
        check(c1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), w1, error,
                             false, false, nullptr, true),
              "v3 chain: the append forwards the layout");
        const fs::path dlog = fs::path(c1.newest_path()).replace_extension(".dlog");
        check(word64(read_file(dlog), 8) == 3, "v3 chain: the record is version 3");
        ConversationSpillCache c2;
        check(c2.open(v3dir, id, 1ull << 30, error, true) && c2.size() == 1, "v3 chain: reopen keeps the entry");
        const auto m = c2.best(grown.live.ids, {}, grown.cvec, 0.0, 0);
        check(bool(m) && m.delta_committed == w1, "v3 chain: the committed chain matches");
        SavedConversation base_meta;
        std::vector<std::array<uint64_t, 5>> base_sizes;
        check(session_file_read_streamed(m.path, id, base_meta, w, error, {}, &base_sizes),
              "v3 chain: the base reads streamed");
        std::vector<std::array<std::vector<uint8_t>, 5>> rebuilt(grown.kv.size());
        for (size_t l = 0; l < grown.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&grown.kv[l].k, &grown.kv[l].v,
                                                                    &grown.kv[l].k_scale, &grown.kv[l].v_scale,
                                                                    &grown.kv[l].pooled};
            for (size_t part = 0; part < 5; ++part) rebuilt[l][part].resize(parts[part]->size());
        }
        auto sink = [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
            if (layer >= rebuilt.size() || offset > rebuilt[layer][part].size() ||
                n > rebuilt[layer][part].size() - offset) return false;
            std::memcpy(rebuilt[layer][part].data() + offset, data, n);
            return true;
        };
        check(session_file_apply_streamed(m.path, id, sink, base_sizes, error), "v3 chain: the base applies");
        uint64_t at = 0;
        while (at < m.delta_committed) {
            SessionDeltaRecord rec;
            uint64_t next = 0;
            check(session_delta_read(dlog.string(), at, m.delta_committed, id, {}, rec, next, error),
                  "v3 chain: a committed record reads");
            check(session_delta_apply(dlog.string(), at, id, rec, sink, error),
                  "v3 chain: a committed record applies");
            at = next;
        }
        bool same = true;
        for (size_t l = 0; same && l < grown.kv.size(); ++l) {
            const std::array<const ConversationBuffer*, 5> parts = {&grown.kv[l].k, &grown.kv[l].v,
                                                                    &grown.kv[l].k_scale, &grown.kv[l].v_scale,
                                                                    &grown.kv[l].pooled};
            for (size_t part = 0; same && part < 5; ++part) {
                std::vector<uint8_t> want(parts[part]->size());
                parts[part]->visit(0, want.size(), [&](const uint8_t* p, size_t c, size_t at2) {
                    std::memcpy(want.data() + at2, p, c); return true;
                });
                same = rebuilt[l][part] == want;
            }
        }
        check(same, "v3 chain: base + v3 record rebuild the newest K/V exactly");
        fs::remove_all(v3dir);

        // a v1 base with a v3 record: the mixed chain restores too (the file and record versions are independent)
        const fs::path mxdir = fs::temp_directory_path() / "strata_spill_v3_mixed_test";
        fs::remove_all(mxdir);
        ConversationSpillCache mx1;
        check(mx1.open(mxdir, id, 1ull << 30, error, true), "v3 mix: open");
        check(mx1.spill_streamed(meta_of(base), sources_of(base, 0), w, error), "v3 mix: a v1 base");
        check(file_version(read_file(mx1.newest_path())) == 1, "v3 mix: the base stays version 1");
        size_t wm = 0;
        check(mx1.spill_delta(0, meta_of(grown), sources_of(grown, 1024), (int64_t) grown.live.ids.size(), wm, error,
                              false, false, nullptr, true),
              "v3 mix: a v3 record on a v1 base");
        ConversationSpillCache mx2;
        check(mx2.open(mxdir, id, 1ull << 30, error, true) && mx2.size() == 1, "v3 mix: reopen keeps the entry");
        const auto mm = mx2.best(grown.live.ids, {}, grown.cvec, 0.0, 0);
        check(bool(mm) && mm.delta_committed == wm, "v3 mix: the committed chain matches");
        check(word64(read_file(fs::path(mm.path).replace_extension(".dlog")), 8) == 3,
              "v3 mix: the record is version 3");
        fs::remove_all(mxdir);
    }

    // Two-phase background saves: only the reservation and publication touch the cache. The file task owns the
    // captured identity and paths; an uncommitted record or an old reservation never replaces the committed chain.
    {
        const fs::path bdir = fs::temp_directory_path() / "strata_spill_background_test";
        fs::remove_all(bdir);
        ConversationSpillCache cache;
        check(cache.open(bdir, id, 1ull << 30, error, true), "background spill: open");
        SavedConversation base = sample(1200);
        SavedConversation meta = meta_of(base);
        ConversationSpillWrite base_write;
        check(cache.reserve_streamed(meta, base_write, error) && base_write.valid && cache.size() == 0,
              "background spill: reserve a base without publishing it");
        SessionWriteOptions opt;
        opt.layout_v3 = true;
        opt.background_state = true;
        size_t base_bytes = 0;
        bool released = false;
        opt.on_sources_released = [&] { released = true; };
        check(session_file_write(base_write.session_path(), meta, sources_of(base), base_write.identity, base_bytes,
                                 error, opt) && released && cache.size() == 0,
              "background spill: the file task writes v3 without touching the cache");
        check(cache.publish_streamed(base_write, meta, base_bytes, error) && cache.size() == 1 &&
              cache.newest_path() == base_write.session_path(),
              "background spill: publish the completed base at the cache boundary");
        const size_t index = cache.find(base_write.session_path());
        ConversationSpillDeltaWrite delta_write;
        check(cache.reserve_delta(index, delta_write, error) && delta_write.valid &&
              delta_write.session_path() == base_write.session_path() && delta_write.delta_bytes == 0,
              "background spill: reserve the first delta append");
        SavedConversation first = base;
        first.live.ids.push_back(8001);
        meta = meta_of(first);
        const uint64_t accounted_before_record = cache.bytes();
        uint64_t first_record = 0;
        check(session_delta_write(delta_write.delta_path(), delta_write.delta_bytes, meta, sources_of(first),
                                  (int64_t) first.live.ids.size(), delta_write.identity, first_record, error, opt),
              "background spill: write the first delta record");
        check(cache.size() == 1 && cache.bytes() == accounted_before_record,
              "background spill: the uncommitted record has not changed the cache or its accounting");
        check(cache.publish_delta(delta_write, meta, (size_t) first_record, true, error),
              "background spill: publish the first delta record");
        int64_t saved_tokens = 0, deepest = 0;
        uint64_t saved_base = 0, committed = 0;
        bool prefix = true;
        std::string saved_path;
        check(cache.entry_info(index, saved_tokens, deepest, saved_base, committed, prefix, saved_path) &&
              committed == first_record && saved_tokens == (int64_t) first.live.ids.size(),
              "background spill: publication advances the committed length once");

        SavedConversation failed = first;
        failed.live.ids.push_back(8002);
        meta = meta_of(failed);
        ConversationSpillDeltaWrite failed_write;
        check(cache.reserve_delta(index, failed_write, error), "background spill: reserve a later delta");
        opt.fault = [](const char* step) { return std::strcmp(step, "write") == 0 ? EIO : 0; };
        released = false;
        opt.on_sources_released = [&] { released = true; };
        uint64_t failed_record = 0;
        check(!session_delta_write(failed_write.delta_path(), failed_write.delta_bytes, meta, sources_of(failed),
                                   (int64_t) failed.live.ids.size(), failed_write.identity, failed_record, error, opt) &&
              released && !cache.publish_delta(failed_write, meta, 0, false, error),
              "background spill: a failed task releases sources and does not publish");
        check(cache.entry_info(index, saved_tokens, deepest, saved_base, committed, prefix, saved_path) &&
              committed == first_record && cache.best(failed.live.ids, {}, failed.cvec, 0.0, 0).live_tokens ==
              (int64_t) first.live.ids.size(),
              "background spill: failure leaves the previous committed chain visible");
        ConversationSpillCache reopened;
        check(reopened.open(bdir, id, 1ull << 30, error, true) && reopened.size() == 1 &&
              reopened.best(failed.live.ids, {}, failed.cvec, 0.0, 0).delta_committed == first_record,
              "background spill: the old chain also survives a reopen");

        ConversationSpillDeltaWrite stale;
        check(reopened.reserve_delta(0, stale, error), "background spill: reserve a generation guard");
        SavedConversation newest = first;
        newest.live.ids.push_back(8003);
        size_t newest_bytes = 0;
        check(reopened.spill_delta(0, meta_of(newest), sources_of(newest), (int64_t) newest.live.ids.size(),
                                   newest_bytes, error),
              "background spill: a newer generation commits");
        uint64_t new_committed = 0;
        check(reopened.entry_info(0, saved_tokens, deepest, saved_base, new_committed, prefix, saved_path) &&
              new_committed > first_record, "background spill: read the newer committed length");
        check(!reopened.publish_delta(stale, meta_of(failed), 1, true, error) &&
              error.find("reservation") != std::string::npos,
              "background spill: an old task cannot publish into the newer generation");
        check(reopened.entry_info(0, saved_tokens, deepest, saved_base, committed, prefix, saved_path) &&
              committed == new_committed && reopened.best(newest.live.ids, {}, newest.cvec, 0.0, 0).live_tokens ==
              (int64_t) newest.live.ids.size(),
              "background spill: rejecting the old task leaves the newer chain intact");
        fs::remove_all(bdir);
    }

    // ---- the opt-in recompute gate: a same-model/other-config sidecar is normally kept but never indexed; the
    // directory ledger (.strata-cache-gate) can approve exactly those entries for this identity and engine build.
    {
        const fs::path gdir = fs::temp_directory_path() / "strata_spill_gate_test";
        fs::remove_all(gdir);
        const SessionFileIdentity legacy{id.model, id.config ^ 0x0f0f0f0f0f0f0f0full};
        const SessionFileIdentity other_model{id.model ^ 0x10, id.config};
        SavedConversation foreign1 = sample(600);
        SavedConversation foreign2 = sample(640);
        SavedConversation foreign_other_model = sample(680);
        SavedConversation same = sample(720);
        for (auto& c : same.checkpoints) for (auto& v : c.ids) v += 4000;
        for (auto& v : same.live.ids) v += 4000;
        {
            ConversationSpillCache w1;
            check(w1.open(gdir, legacy, 1ull << 30, error) && w1.spill(foreign1, error) && w1.spill(foreign2, error),
                  "gate fixture: two legacy-config entries");
            ConversationSpillCache w2;
            check(w2.open(gdir, other_model, 1ull << 30, error) && w2.spill(foreign_other_model, error),
                  "gate fixture: an other-model entry");
            ConversationSpillCache w3;
            check(w3.open(gdir, id, 1ull << 30, error) && w3.spill(same, error),
                  "gate fixture: a same-identity entry");
        }
        struct GateRow {
            std::string stem, path;
            uint64_t bytes = 0;
            int64_t tokens = 0;
        };
        auto entry_at = [&](const SessionFileIdentity& who, size_t index) {
            ConversationSpillCache c;
            GateRow row;
            check(c.open(gdir, who, 1ull << 30, error) && index < c.size(), "gate fixture: reopen for entry info");
            int64_t deepest = 0, tokens = 0;
            uint64_t base = 0, delta_bytes = 0;
            bool prefix = false;
            check(c.entry_info(index, tokens, deepest, base, delta_bytes, prefix, row.path),
                  "gate fixture: entry info");
            row.stem = fs::path(row.path).stem().string();
            row.bytes = base;
            row.tokens = tokens;
            return row;
        };
        const GateRow legacy1 = entry_at(legacy, 0);
        const GateRow legacy2 = entry_at(legacy, 1);
        const GateRow foreign_model_row = entry_at(other_model, 0);
        const GateRow same_row = entry_at(id, 0);
        check(legacy1.tokens == 600 && same_row.tokens == 720, "gate fixture: token counts");

        ConversationSpillCache off;
        check(off.open(gdir, id, 1ull << 30, error) && off.size() == 1,
              "gate off: only the same-identity entry indexes");
        const fs::path legacy1_meta = fs::path(legacy1.path).replace_extension(".meta");
        check(fs::exists(legacy1.path) && fs::exists(legacy1_meta) && fs::exists(legacy2.path),
              "gate off: the foreign files are kept");

        ConversationSpillGateOptions gate;
        gate.enabled = true;
        gate.engine_build = "test-build";
        auto reopen = [&](size_t& size, std::string& e) {
            ConversationSpillCache c;
            const bool ok = c.open(gdir, id, 1ull << 30, e, false, gate);
            size = c.size();
            return ok;
        };
        size_t indexed = 0;
        check(reopen(indexed, error) && indexed == 1, "gate on without a ledger: nothing foreign is approved");
        {
            ConversationSpillCache c;
            check(c.open(gdir, id, 1ull << 30, error, false, gate), "gate on: open for a same-identity load");
            SavedConversation back;
            SessionStatus gst;
            check(c.load(same_row.path, back, {}, error, &gst) && back.live.ids == same.live.ids,
                  "gate on without a ledger: same-identity loads as before");
        }

        ConversationSpillGateBatch batch;
        batch.id = "b1";
        batch.source = legacy;
        batch.target = id;
        batch.engine_build = "test-build";
        batch.sample = legacy1.stem;
        batch.passed = false;
        batch.verified_ms = 1234;
        batch.reason = "route counts differ";
        std::vector<ConversationSpillGateBatch> batches{batch};
        std::vector<ConversationSpillGateEntry> rows{{legacy1.stem, "b1", legacy1.bytes, legacy1.tokens, 99}};
        check(session_spill_gate_write(gdir, batches, rows, error), "gate: write the ledger");
        check(reopen(indexed, error) && indexed == 1, "gate: a failed batch approves nothing");

        batch.passed = true;
        batch.reason.clear();
        batch.engine_build = "other-build";
        batches = {batch};
        check(session_spill_gate_write(gdir, batches, rows, error) && reopen(indexed, error) && indexed == 1,
              "gate: another engine build approves nothing");
        batch.engine_build = "test-build";
        batch.target.config = id.config ^ 1;
        batches = {batch};
        check(session_spill_gate_write(gdir, batches, rows, error) && reopen(indexed, error) && indexed == 1,
              "gate: another target configuration approves nothing");
        batch.target = id;
        batches = {batch};
        const std::vector<ConversationSpillGateEntry> no_rows;
        check(session_spill_gate_write(gdir, batches, no_rows, error) && reopen(indexed, error) && indexed == 1,
              "gate: a batch without an entry row approves nothing");
        rows[0].file_bytes = legacy1.bytes + 1;
        check(session_spill_gate_write(gdir, batches, rows, error) && reopen(indexed, error) && indexed == 1,
              "gate: a stale file size approves nothing");
        rows[0].file_bytes = legacy1.bytes;
        rows[0].tokens = legacy1.tokens + 1;
        check(session_spill_gate_write(gdir, batches, rows, error) && reopen(indexed, error) && indexed == 1,
              "gate: a stale token count approves nothing");
        rows[0].tokens = legacy1.tokens;
        batch.source = other_model;
        rows[0].stem = foreign_model_row.stem;
        rows[0].file_bytes = foreign_model_row.bytes;
        rows[0].tokens = foreign_model_row.tokens;
        batches = {batch};
        check(session_spill_gate_write(gdir, batches, rows, error) && reopen(indexed, error) && indexed == 1,
              "gate: another model is never approved");

        batch.source = legacy;
        rows[0] = {legacy1.stem, "b1", legacy1.bytes, legacy1.tokens, 1700000000000ull};
        batches = {batch};
        check(session_spill_gate_write(gdir, batches, rows, error), "gate: write the approving ledger");
        ConversationSpillCache approved;
        check(approved.open(gdir, id, 1ull << 30, error, false, gate) && approved.size() == 2,
              "gate: a passed batch and its row index the approved entry");
        check(approved.find(legacy2.path) == approved.size(),
              "gate: an entry without a row stays unindexed");
        // the next turn's prompt extends the cached conversation: its live state is a strict prefix of the prompt
        std::vector<int32_t> prompt(foreign1.live.ids.begin(), foreign1.live.ids.end());
        prompt.push_back(7777);
        const ConversationSpillMatch hit = approved.best(prompt, {}, foreign1.cvec, 0.0, 0);
        check(bool(hit) && hit.tokens == 600 && hit.path == legacy1.path,
              "gate: the approved entry is the match");
        SavedConversation restored;
        SessionStatus load_st;
        check(approved.load(hit.path, restored, {}, error, &load_st) && restored.live.ids == foreign1.live.ids &&
                  restored.checkpoints.size() == foreign1.checkpoints.size(),
              "gate: the approved entry loads with its own config");
        SessionFileIdentity kept;
        check(approved.entry_source_identity(approved.find(hit.path), kept) && kept.model == legacy.model &&
                  kept.config == legacy.config,
              "gate: the entry keeps its original identity");

        // the ledger is a rebuildable cache: corrupt, half-written or missing approves nothing, and the
        // same-identity cache keeps working.
        const fs::path ledger = gdir / ".strata-cache-gate";
        std::string text;
        {
            std::ifstream in(ledger, std::ios::binary);
            text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        auto spit_text = [&](const std::string& bytes) {
            std::ofstream out(ledger, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), (std::streamsize) bytes.size());
        };
        std::string edited = text;
        edited[edited.size() / 3] = char(edited[edited.size() / 3] ^ 0x01);
        spit_text(edited);
        check(reopen(indexed, error) && indexed == 1, "gate: a corrupt ledger approves nothing");
        {
            ConversationSpillCache c;
            check(c.open(gdir, id, 1ull << 30, error, false, gate), "gate: open with a corrupt ledger");
            SavedConversation back;
            SessionStatus gst;
            check(c.load(same_row.path, back, {}, error, &gst) && back.live.ids == same.live.ids,
                  "gate: a corrupt ledger does not stop same-identity loads");
        }
        std::vector<ConversationSpillGateBatch> corrupt_batches;
        std::vector<ConversationSpillGateEntry> corrupt_rows;
        check(!session_spill_gate_read(gdir, corrupt_batches, corrupt_rows, error) && !error.empty(),
              "gate: the reader refuses a corrupt ledger");
        spit_text(text.substr(0, text.size() / 2));
        check(reopen(indexed, error) && indexed == 1, "gate: a half-written ledger approves nothing");
        fs::remove(ledger);
        check(reopen(indexed, error) && indexed == 1, "gate: a missing ledger approves nothing");

        // the format round-trips and refuses fields it could not keep on one line
        std::vector<ConversationSpillGateBatch> read_batches;
        std::vector<ConversationSpillGateEntry> read_rows;
        check(!session_spill_gate_read(gdir, read_batches, read_rows, error), "gate: reading a missing ledger fails");
        check(session_spill_gate_write(gdir, batches, rows, error), "gate: rewrite the approving ledger");
        check(session_spill_gate_read(gdir, read_batches, read_rows, error) && read_batches.size() == 1 &&
                  read_rows.size() == 1 && read_batches[0].passed && read_batches[0].source.config == legacy.config &&
                  read_batches[0].target.config == id.config && read_rows[0].stem == legacy1.stem &&
                  read_rows[0].tokens == legacy1.tokens && error.empty(),
              "gate: the ledger round-trips");
        {
            std::vector<ConversationSpillGateBatch> bad = batches;
            bad[0].sample = "has space";
            check(!session_spill_gate_write(gdir, bad, rows, error), "gate: the writer refuses whitespace");
            std::vector<ConversationSpillGateEntry> bad_rows = rows;
            bad_rows[0].batch_id.clear();
            check(!session_spill_gate_write(gdir, batches, bad_rows, error), "gate: the writer refuses an empty field");
            check(session_spill_gate_read(gdir, read_batches, read_rows, error) && read_rows.size() == 1,
                  "gate: a refused write leaves the previous ledger");
        }
        // A gate-approved entry is read-only: appending a record would put this engine configuration's bytes behind
        // a file of the other one, so the chain would no longer be what the ledger approved.
        {
            check(session_spill_gate_write(gdir, batches, rows, error), "gate write: restore the approving ledger");
            ConversationSpillCache dm;
            check(dm.open(gdir, id, 1ull << 30, error, true, gate) && dm.size() == 2,
                  "gate write: delta mode indexes the approved entry");
            ConversationSpillDeltaWrite target;
            const size_t legacy_index = dm.find(legacy1.path);
            check(legacy_index < dm.size() && !dm.reserve_delta(legacy_index, target, error) &&
                      error.find("read-only") != std::string::npos && !target.valid,
                  "gate write: reserve_delta refuses an entry of another configuration");
            size_t written = 0;
            SavedConversation appended = foreign1;
            appended.live.ids.push_back(8888);
            check(!dm.spill_delta(legacy_index, appended, {}, 600, written, error) && written == 0,
                  "gate write: spill_delta refuses an entry of another configuration");
            const size_t same_index = dm.find(same_row.path);
            check(same_index < dm.size() && dm.reserve_delta(same_index, target, error) && target.valid,
                  "gate write: reserve_delta still accepts this cache's own entries");
            // A hit stamps the entry (touch rewrites the sidecar): the rewritten sidecar must keep the entry's OWN
            // identity, or the file would be forged as this cache's and read after a restart without the ledger.
            {
                SavedConversation back;
                SessionStatus tst;
                check(dm.load(legacy1.path, back, {}, error, &tst) && back.live.ids == foreign1.live.ids,
                      "gate touch: the approved entry loads with its own configuration");
                dm.touch(legacy1.path);
                ConversationSpillCache without_gate;
                check(without_gate.open(gdir, id, 1ull << 30, error) && without_gate.size() == 1 &&
                          without_gate.find(legacy1.path) == without_gate.size(),
                      "gate touch: the rewritten sidecar still needs the gate (unindexed without it)");
                ConversationSpillCache with_gate;
                check(with_gate.open(gdir, id, 1ull << 30, error, true, gate) &&
                          with_gate.find(legacy1.path) < with_gate.size(),
                      "gate touch: with the ledger the rewritten entry is still indexed");
                SessionFileIdentity kept_after;
                check(with_gate.entry_source_identity(with_gate.find(legacy1.path), kept_after) &&
                          kept_after.model == legacy.model && kept_after.config == legacy.config,
                      "gate touch: the rewritten sidecar keeps the entry's own identity");
                SavedConversation again;
                SessionStatus ast;
                check(with_gate.load(legacy1.path, again, {}, error, &ast) &&
                          again.live.ids == foreign1.live.ids,
                      "gate touch: the rewritten entry still loads under the ledger");
            }
            fs::remove(ledger);
        }
        fs::remove_all(gdir);
    }

    std::printf("conversation_spill_test: %d checks passed\n", checks);
    return 0;
}
