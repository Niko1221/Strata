// CPU-only tests for the spill directory (include/strata/core/conversation_spill.hpp): a parked conversation is
// written as a session file plus a metadata sidecar, matched from the sidecar alone, and read back whole.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_spill.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#ifdef _MSC_VER
#include <fcntl.h>
#include <io.h>
#include <share.h>
#endif

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
// A conversation whose live ids are exactly `ids` (the sidecar keeps only the checkpoint lengths, so their own ids
// do not have to match - the divergence rule reads live ids alone). `cp` names the one checkpoint's length; 0 = half.
SavedConversation conversation_with_ids(const std::vector<int32_t>& ids, size_t cp = 0) {
    SavedConversation s = sample(ids.size());
    s.live.ids = ids;
    s.checkpoints.clear();
    const size_t len = cp != 0 ? cp : ids.size() / 2;
    ConversationCheckpoint c = checkpoint(len, 9);
    c.ids.assign(ids.begin(), ids.begin() + (std::ptrdiff_t) len);
    s.checkpoints.push_back(std::move(c));
    return s;
}
// A conversation template: a header of twelve tokens (three turn markers, `seed` in it, so every seed is its own
// conversation) followed by `n` tokens of tail. Two calls with the same seed share the header and are one chat.
std::vector<int32_t> conv_ids(int32_t seed, size_t n) {
    std::vector<int32_t> ids = {seed, seed + 1, seed + 2, 500, seed + 3, seed + 4, seed + 5, 500,
                                seed + 6, seed + 7, seed + 8, 500};
    for (size_t i = 0; i < n; ++i) ids.push_back((int32_t) (seed + 100 + (int32_t) i));
    return ids;
}
std::vector<uint8_t> read_file_bytes(const fs::path& path) {
    std::vector<uint8_t> bytes;
    std::ifstream in(path, std::ios::binary);
    if (!in) return bytes;
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return bytes;
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

// ---- Delta 5a helpers: an archive under test ----
// A conversation whose header ends at the first assistant turn (turn token 500, 52 tokens), with a long tail; the
// `compacted` member is its rewritten-tail variant (same header, tail rewritten) that discard_diverged archives.
struct D5aConv {
    std::vector<int32_t> ids;
    std::vector<int32_t> compacted;
    int64_t turn = 500;
    size_t header = 0;
};
D5aConv d5a_conv(size_t tail = 5000) {
    D5aConv c;
    for (int32_t i = 1; i <= 10; ++i) c.ids.push_back(i);
    c.ids.push_back((int32_t) c.turn);
    for (int32_t i = 100; i < 120; ++i) c.ids.push_back(i);
    c.ids.push_back((int32_t) c.turn);
    for (int32_t i = 200; i < 220; ++i) c.ids.push_back(i);
    c.ids.push_back((int32_t) c.turn);
    c.header = conversation_header_length(c.ids, c.turn);
    for (int32_t i = 300; i < (int32_t) (300 + tail); ++i) c.ids.push_back(i);
    c.compacted.assign(c.ids.begin(), c.ids.begin() + (std::ptrdiff_t) c.header + 100);
    for (int32_t i = 0; i < 100; ++i) c.compacted[(size_t) c.header + (size_t) i] = int32_t(70000 + i);
    return c;
}
size_t d5a_count_ext(const fs::path& archive, const std::string& ext) {
    size_t n = 0;
    if (!fs::exists(archive)) return 0;
    for (const auto& e : fs::directory_iterator(archive))
        if (fs::is_regular_file(e.path()) && e.path().extension() == ext) ++n;
    return n;
}
uint64_t d5a_total_bytes(const fs::path& archive) {
    uint64_t total = 0;
    if (!fs::exists(archive)) return 0;
    for (const auto& e : fs::directory_iterator(archive))
        if (fs::is_regular_file(e.path())) total += fs::file_size(e.path());
    return total;
}
// The numeric suffix of every archive sidecar, ascending: "strata-conv-4.meta" -> 4.
std::vector<uint64_t> d5a_serials(const fs::path& archive) {
    std::vector<uint64_t> serials;
    if (!fs::exists(archive)) return serials;
    for (const auto& e : fs::directory_iterator(archive)) {
        if (e.path().extension() != ".meta") continue;
        const std::string name = e.path().stem().string();
        const size_t at = name.rfind('-');
        if (at != std::string::npos) serials.push_back(std::stoull(name.substr(at + 1)));
    }
    std::sort(serials.begin(), serials.end());
    return serials;
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

    // a foreign identity is refused, nothing is indexed, and its files are left alone (the scan never deletes)
    ConversationSpillCache foreign;
    check(foreign.open(dir, SessionFileIdentity{1, 2}, 1ull << 30, error), "open with foreign identity");
    check(foreign.size() == 0, "foreign identity: nothing indexed");
    check(foreign.foreign_files_kept() >= 1, "foreign identity reported, not removed");
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

    // ---- Delta 1: a layer split is one file per stage plus one joint sidecar ----
    fs::remove_all(dir);
    ConversationSpillCache split;
    check(split.open(dir, id, 1ull << 30, error), "open for the layer-split test");
    SavedConversation stage0 = sample(1200);        // the first stage: its own carve
    SavedConversation stage1 = sample(600);         // a later stage: its own carve, no draft layer
    stage1.layer_lo = 48; stage1.layer_hi = 96;
    stage0.stage_images.push_back(stage1);
    check(split.spill(stage0, error), "spill a two-stage conversation");
    check(fs::exists(dir / "strata-conv-1.sess"), "first stage file written");
    check(fs::exists(dir / "strata-conv-1.stage1.sess"), "later stage file written");
    check(fs::exists(dir / "strata-conv-1.meta"), "one joint sidecar");
    std::vector<int32_t> cont = stage0.live.ids;
    cont.push_back(int32_t(4242));
    const auto split_match = split.best(cont, stage0.live.imgs, false, 0.0, 0);
    check(bool(split_match), "match a two-stage conversation");
    check(split_match.stages == 1, "the sidecar reports one extra stage");
    SavedConversation back_split;
    check(split.load(split_match.path, back_split, {}, error), "load a two-stage conversation back");
    check(back_split.stage_images.size() == 1, "one stage image read back");
    check(back_split.stage_images[0].live.ids == stage1.live.ids, "stage ids round-trip");
    check(back_split.stage_images[0].layer_lo == 48 && back_split.stage_images[0].layer_hi == 96,
          "stage layer range round-trip");
    bool stage_kv_same = back_split.stage_images[0].kv.size() == stage1.kv.size();
    for (size_t i = 0; stage_kv_same && i < back_split.stage_images[0].kv.size(); ++i)
        stage_kv_same = buffers_equal(back_split.stage_images[0].kv[i].k, stage1.kv[i].k) &&
                        buffers_equal(back_split.stage_images[0].kv[i].v, stage1.kv[i].v);
    check(stage_kv_same, "stage kv bytes round-trip");
    // the stage count comes from the sidecar, so a reopen re-reads both files
    ConversationSpillCache split_again;
    check(split_again.open(dir, id, 1ull << 30, error), "reopen the split dir");
    const auto split_rematch = split_again.best(cont, stage0.live.imgs, false, 0.0, 0);
    check(bool(split_rematch) && split_rematch.stages == 1, "stage count survives a reopen");
    SavedConversation back_split2;
    check(split_again.load(split_rematch.path, back_split2, {}, error), "reopen load");
    check(back_split2.stage_images.size() == 1 && back_split2.stage_images[0].live.ids == stage1.live.ids,
          "stage read back after reopen");
    // every file of the conversation counts against the budget and goes together
    check(split.bytes() == fs::file_size(dir / "strata-conv-1.sess") + fs::file_size(dir / "strata-conv-1.stage1.sess"),
          "both stage files counted against the budget");
    check(split.erase(split_match.path, error), "erase a two-stage conversation");
    check(!fs::exists(dir / "strata-conv-1.sess") && !fs::exists(dir / "strata-conv-1.stage1.sess") &&
          !fs::exists(dir / "strata-conv-1.meta"), "erase removes every stage file and the sidecar");

    // ---- a conversation larger than the whole budget is KEPT and reported ----
    fs::remove_all(dir);
    ConversationSpillCache tiny;
    check(tiny.open(dir, id, 4096, error), "open with a tiny budget");
    SavedConversation big = sample(800);
    check(tiny.spill(big, error), "spill a conversation over the budget");
    check(tiny.size() == 1, "oversized conversation indexed, not dropped");
    check(tiny.bytes() > 4096, "it is larger than the budget");
    check(tiny.oversized_files_kept() >= 1, "reported as oversized");
    check(tiny.disk_evictions() == 0, "the GC did not evict it");
    check(fs::exists(dir / "strata-conv-1.sess"), "its file is on disk");
    ConversationSpillCache tiny_again;
    check(tiny_again.open(dir, id, 4096, error), "reopen with the tiny budget");
    check(tiny_again.size() == 1, "the oversized conversation survives a reopen");
    check(tiny_again.oversized_files_kept() >= 1, "reported oversized on reopen");

    // ---- the scan never deletes: an orphan session file is reported and left in place ----
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        std::ofstream out(dir / "strata-conv-99.sess", std::ios::binary);
        out << "not a session file";
    }
    check(fs::exists(dir / "strata-conv-99.sess"), "orphan file created");
    ConversationSpillCache orphan;
    check(orphan.open(dir, id, 1ull << 30, error), "scan with an orphan session file");
    check(orphan.size() == 0, "an orphan is not indexed");
    check(orphan.orphan_files_kept() >= 1, "orphan reported");
    check(fs::exists(dir / "strata-conv-99.sess"), "orphan left in place");

    // ---- --conversation-cache-spill-when-full: reject never evicts, evict-oldest (default) drops the oldest ----
    fs::remove_all(dir);
    ConversationSpillCache seed;
    check(seed.open(dir, id, 1ull << 30, error), "seed open for the when-full test");
    SavedConversation w1 = sample(1000), w2 = sample(1200);
    check(seed.spill(w1, error), "seed one conversation");
    const uint64_t one_file = seed.bytes();   // exactly one conversation's bytes: the budget below is "one file"
    check(one_file > 0, "one conversation's bytes");
    std::string keep_sess;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".sess") keep_sess = e.path().string();
    check(!keep_sess.empty(), "the stored session file is on disk");

    ConversationSpillCache reject;
    check(reject.open(dir, id, one_file, error, SpillWhenFull::reject, 0), "open with when-full=reject");
    check(reject.when_full() == SpillWhenFull::reject, "reject mode recorded");
    check(reject.size() == 1, "reject indexes the existing conversation");
    std::string reject_error;
    check(!reject.spill(w2, reject_error), "reject refuses a spill over the budget");
    check(!reject_error.empty(), "reject says why");
    check(reject.size() == 1 && reject.disk_evictions() == 0, "reject evicts nothing");
    check(fs::exists(keep_sess), "reject leaves the stored conversation alone");

    ConversationSpillCache evict;
    check(evict.open(dir, id, one_file, error, SpillWhenFull::evict_oldest, 0), "open with when-full=evict-oldest");
    check(evict.spill(w2, error), "evict-oldest accepts the spill");
    check(evict.size() == 1, "one conversation after the eviction");
    check(evict.disk_evictions() >= 1, "evict-oldest dropped the oldest");

    // ---- --conversation-cache-spill-max-age-days: 0 = no deletion by time, positive = oldest first ----
    fs::remove_all(dir);
    ConversationSpillCache age_seed;
    check(age_seed.open(dir, id, 1ull << 30, error), "age seed open");
    check(age_seed.spill(sample(1000), error), "age seed spill one");
    check(age_seed.spill(sample(1200), error), "age seed spill two");
    check(age_seed.max_age_days() == 0, "max-age defaults to 0");
    check(age_seed.age_evictions() == 0, "max-age 0 deletes nothing by time");
    const auto ten_days_ago = fs::file_time_type::clock::now() - std::chrono::hours(24 * 10);
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".sess") fs::last_write_time(e.path(), ten_days_ago);
    ConversationSpillCache keep0;
    check(keep0.open(dir, id, 1ull << 30, error, SpillWhenFull::evict_oldest, 0), "open with max-age 0 on old files");
    check(keep0.age_evictions() == 0 && keep0.size() == 2, "max-age 0 keeps the old conversations");
    ConversationSpillCache aged;
    check(aged.open(dir, id, 1ull << 30, error, SpillWhenFull::evict_oldest, 7), "open with max-age 7 days");
    check(aged.max_age_days() == 7, "max-age recorded");
    check(aged.age_evictions() >= 1, "old conversations pruned by age");
    check(aged.size() == 0, "every conversation older than the limit went");

    // ---- Delta 2 / R3: the park writer writes a conversation with no eviction, collapses, throttles ----
    fs::remove_all(dir);
    {
        ConversationSpillCache park;
        check(park.open(dir, id, 1ull << 30, error), "park: open the spill directory");
        ConversationSpillWriter writer;
        writer.start(&park, 0);
        writer.stop();   // thread joined: post() + drain() below run deterministically on this thread
        SavedConversation p1 = conversation_with_ids(conv_ids(1000, 800));
        const uint64_t key = conversation_key(conv_ids(1000, 800), 248045);
        check(writer.post(key, std::move(p1)), "park: post the conversation");
        check(writer.writes() == 0, "park: nothing written before the drain");
        check(writer.drain() == 1, "park: drain writes once");
        check(writer.writes() == 1 && writer.refused() == 0, "park: one write, none refused");
        // The file is on disk although nothing was evicted from any RAM cache: the acceptance criterion.
        check(park.size() == 1, "park: one conversation on disk without an eviction");
        check(park.disk_evictions() == 0, "park: no eviction happened");
        std::vector<int32_t> cont = conv_ids(1000, 800);
        cont.push_back(int32_t(7777));
        const auto park_hit = park.best(cont, {}, false, 0.0, 0);
        check(bool(park_hit) && park_hit.tokens == (int64_t) cont.size() - 1, "park: the written copy matches");

        // R3 collapse: two parks of one conversation before the drain write once, replacing the queued state.
        SavedConversation q1 = conversation_with_ids(conv_ids(2000, 800));
        SavedConversation q2 = conversation_with_ids(conv_ids(2000, 1200));   // same chat, one turn longer
        const uint64_t qkey = conversation_key(conv_ids(2000, 800), 248045);
        check(writer.post(qkey, std::move(q1)), "park: post a growing conversation");
        check(writer.post(qkey, std::move(q2)), "park: post its next state (replaces the queued one)");
        check(writer.collapsed() == 1, "park: the replaced state is counted as collapsed");
        const size_t before_q = park.size();
        check(writer.drain() == 1, "park: the drain writes only the newest state");
        check(writer.writes() == 2, "park: the second conversation added exactly one write");
        check(park.size() == before_q + 1, "park: one copy for the conversation, not two");

        // Two parks of the SAME conversation where the second supersedes the first on disk: one copy, new content.
        SavedConversation r1 = conversation_with_ids(conv_ids(3000, 800), 500);
        SavedConversation r2 = conversation_with_ids(conv_ids(3000, 1100), 500);
        const uint64_t rkey = conversation_key(conv_ids(3000, 800), 248045);
        const size_t before_r = park.size();
        check(writer.post(rkey, std::move(r1)), "park: post state A");
        check(writer.drain() == 1, "park: write A");
        check(writer.post(rkey, std::move(r2)), "park: post the longer state B");
        check(writer.drain() == 1, "park: write B supersedes A");
        check(park.size() == before_r + 1, "park: one copy after the replacement");
        std::vector<int32_t> rc = conv_ids(3000, 1100);
        rc.push_back(int32_t(8888));
        const auto rematch = park.best(rc, {}, false, 0.0, 0);
        check(bool(rematch) && rematch.tokens == (int64_t) rc.size() - 1, "park: the replacement holds the newer state");

        // The skip lever: a state that did not grow past what was written for that key is not posted again.
        SavedConversation s1 = conversation_with_ids(conv_ids(4000, 500));
        const uint64_t skey = conversation_key(conv_ids(4000, 500), 248045);
        check(writer.post(skey, std::move(s1)), "park: post a state");
        check(writer.drain() == 1, "park: write it");
        SavedConversation s2 = conversation_with_ids(conv_ids(4000, 500));   // same size: nothing changed
        check(!writer.post(skey, std::move(s2)), "park: an unchanged state is skipped");
        check(writer.skipped() == 1, "park: the skip is counted");
    }

    // ---- Delta 2 / R3: park + reject keeps the older copy and counts the refusal ----
    fs::remove_all(dir);
    {
        ConversationSpillCache seedp;
        check(seedp.open(dir, id, 1ull << 30, error), "park/reject: open");
        SavedConversation w1 = conversation_with_ids(conv_ids(5000, 800));
        check(seedp.spill(w1, error), "park/reject: seed one conversation");
        const uint64_t one_file = seedp.bytes();
        std::string keep;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".sess") keep = e.path().string();
        const std::vector<uint8_t> keep_bytes = read_file_bytes(keep);
        check(!keep_bytes.empty(), "park/reject: the stored copy has bytes");

        ConversationSpillCache reject;
        check(reject.open(dir, id, one_file, error, SpillWhenFull::reject, 0), "park/reject: open with reject");
        ConversationSpillWriter writer;
        writer.start(&reject, 0);
        writer.stop();
        SavedConversation w2 = conversation_with_ids(conv_ids(6000, 800));   // another conversation
        const uint64_t wkey = conversation_key(conv_ids(6000, 800), 248045);
        check(writer.post(wkey, std::move(w2)), "park/reject: post the conversation");
        check(writer.drain() == 0, "park/reject: the drain writes nothing");
        check(writer.refused() == 1, "park/reject: the refusal is counted");
        check(reject.disk_evictions() == 0 && reject.size() == 1, "park/reject: nothing was evicted");
        check(read_file_bytes(keep) == keep_bytes, "park/reject: the older copy is byte-for-byte intact");
    }

    // ---- Delta 2 / R3: the throttle postpones a rewrite inside its window ----
    fs::remove_all(dir);
    {
        ConversationSpillCache park;
        check(park.open(dir, id, 1ull << 30, error), "throttle: open");
        ConversationSpillWriter writer;
        writer.start(&park, 60);   // a 60 s window
        writer.stop();             // join the worker; post()/drain() stay deterministic below
        SavedConversation t1 = conversation_with_ids(conv_ids(7000, 500));
        const uint64_t tkey = conversation_key(conv_ids(7000, 500), 248045);
        check(writer.post(tkey, std::move(t1)), "throttle: the first state is queued");
        check(writer.drain() == 1, "throttle: written");
        SavedConversation t2 = conversation_with_ids(conv_ids(7000, 700));   // grew
        check(!writer.post(tkey, std::move(t2)), "throttle: the rewrite inside the window is postponed");
        check(writer.throttled() == 1, "throttle: the postponement is counted");
        check(writer.writes() == 1, "throttle: still one write");
    }
    // Same, with the throttle off (0): the rewrite goes through.
    fs::remove_all(dir);
    {
        ConversationSpillCache park;
        check(park.open(dir, id, 1ull << 30, error), "throttle0: open");
        ConversationSpillWriter writer;
        writer.start(&park, 0);
        writer.stop();
        SavedConversation t1 = conversation_with_ids(conv_ids(8000, 500));
        const uint64_t tkey = conversation_key(conv_ids(8000, 500), 248045);
        check(writer.post(tkey, std::move(t1)), "throttle0: post");
        check(writer.drain() == 1, "throttle0: write");
        SavedConversation t2 = conversation_with_ids(conv_ids(8000, 700));
        check(writer.post(tkey, std::move(t2)), "throttle0: the rewrite is not postponed");
        check(writer.drain() == 1, "throttle0: rewritten");
    }

    // ---- Delta 2 / R2: a compacted conversation's copy is discarded; another conversation's is not ----
    fs::remove_all(dir);
    {
        const int64_t turn = 500;
        std::vector<int32_t> ids;
        for (int32_t i = 1; i <= 10; ++i) ids.push_back(i);
        ids.push_back((int32_t) turn);                       // 1st turn: start of the system message
        for (int32_t i = 100; i < 120; ++i) ids.push_back(i);
        ids.push_back((int32_t) turn);                       // 2nd turn: end of the system prompt
        for (int32_t i = 200; i < 220; ++i) ids.push_back(i);
        ids.push_back((int32_t) turn);                       // 3rd turn: end of the first user turn
        const size_t header = conversation_header_length(ids, turn);
        check(header == 52, "R2: the header ends at the first assistant turn");
        for (int32_t i = 300; i < 5300; ++i) ids.push_back(i);   // a long tail

        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "R2: open");
        SavedConversation stored = conversation_with_ids(ids);
        check(c.spill(stored, error), "R2: store the long conversation");
        std::string stored_path;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".sess") stored_path = e.path().string();
        const std::vector<uint8_t> stored_bytes = read_file_bytes(stored_path);

        // an extension (the normal next turn) is kept
        std::vector<int32_t> extension = ids;
        extension.push_back(int32_t(99999));
        check(c.discard_diverged(extension, false, 4096, turn) == 0, "R2: an extension discards nothing");
        check(c.size() == 1, "R2: the extension kept the copy");

        // another conversation: the divergence is inside the header, so nothing is touched
        std::vector<int32_t> other = ids;
        other[40] = 123456;                                  // different first user turn
        const std::vector<uint8_t> before_other = read_file_bytes(stored_path);
        check(c.discard_diverged(other, false, 4096, turn) == 0, "R2: another conversation discards nothing");
        check(c.size() == 1 && read_file_bytes(stored_path) == before_other, "R2: the other conversation's copy is untouched");

        // a compaction: same header, rewritten short tail
        std::vector<int32_t> compacted(ids.begin(), ids.begin() + (std::ptrdiff_t) header + 100);
        for (int32_t i = 0; i < 100; ++i) compacted[(size_t) header + (size_t) i] = int32_t(70000 + i);
        check(c.discard_diverged(compacted, false, 4096, turn) == 1, "R2: the compacted copy is discarded");
        check(c.compacted() == 1, "R2: the compaction is counted");
        check(c.size() == 0, "R2: the discarded copy left the index");
        check(!fs::exists(stored_path), "R2: its file is gone, not left orphaned");
        (void) stored_bytes;

        // after the discard, the next park writes the compacted history (the conversation is new again)
        check(c.spill(conversation_with_ids(compacted), error), "R2: the compacted conversation is repar ked");
        check(c.size() == 1, "R2: one copy again");
    }

    // ---- Delta 2 / R4: a cancelled request is provisional; the previous copy is untouched ----
    fs::remove_all(dir);
    {
        const int64_t turn = 500;
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "R4: open");
        std::vector<int32_t> closed = {1, 2, turn, 3, 4, turn, 5, 6, turn, 7, 8, 9};   // the closed conversation
        const size_t closed_len = closed.size();
        check(c.spill(conversation_with_ids(closed), error), "R4: the closed conversation is on disk");
        std::string path;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".sess") path = e.path().string();
        const std::vector<uint8_t> good = read_file_bytes(path);
        check(!good.empty(), "R4: the good copy has bytes");

        // The cancelled request: the prompt grew with a new assistant turn whose answer was cut short.
        std::vector<int32_t> consumed = closed;
        consumed.push_back((int32_t) turn);          // <|im_start|>assistant
        for (int32_t i = 0; i < 40; ++i) consumed.push_back(int32_t(9000 + i));   // the partial answer
        const size_t reached = consumed.size();
        check(revert_to_turn_boundary(consumed, turn, reached), "R4: the live ids are reverted");
        check(consumed.size() == closed_len + 1, "R4: the partial answer is dropped");
        check(consumed.back() == (int32_t) turn, "R4: the live state ends on the last turn token");
        // the boundary is never taken past what the read reached
        std::vector<int32_t> no_turns = {1, 2, 3, 4, 5};
        std::vector<int32_t> short_read = no_turns;
        short_read.push_back((int32_t) turn);
        short_read.push_back(int32_t(4242));
        check(!revert_to_turn_boundary(short_read, turn, no_turns.size()), "R4: a boundary past the read is refused");

        // Provisional: no post is made (the engine sets park_provisional and skips the writer), so the disk copy
        // is byte-for-byte the same one.
        check(read_file_bytes(path) == good, "R4: the previous copy is intact after the cancel");
        check(c.size() == 1, "R4: no new copy was published");

        // The retry succeeds: the stored (closed) copy is the prefix the reverted state starts with.
        std::vector<int32_t> retry = consumed;
        retry.push_back(int32_t(919191));            // the client's retried answer
        const auto retry_hit = c.best(retry, {}, false, 0.0, 0);
        check(bool(retry_hit) && retry_hit.tokens == (int64_t) closed_len,
              "R4: the retry resumes from the closed conversation");
    }

    // ================= Delta 5a: the archive (MOVE instead of remove; opt-in, off by default) =================

    // V4 / invariant 1: mode off changes nothing - no directory, no byte, counters at 0.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/off: open the tier");
        const fs::path archive_dir = dir / "archive";
        strata::core::SpillArchiveConfig cfg;
        cfg.directory = archive_dir;
        cfg.mode = SpillArchiveMode::off;
        cfg.budget_bytes = 64ull << 20;
        cfg.keep = 4;
        cfg.turn_token = 500;
        check(c.open_archive(cfg, error), "D5a/off: open_archive with mode off succeeds");
        check(!c.archive_enabled(), "D5a/off: the archive is not enabled");
        check(c.archived() == 0 && c.archived_bytes() == 0, "D5a/off: counters at 0");
        check(!fs::exists(archive_dir), "D5a/off: no archive directory is created");
        D5aConv conv = d5a_conv();
        check(c.spill(conversation_with_ids(conv.ids), error), "D5a/off: spill a conversation");
        check(c.discard_diverged(conv.compacted, false, 4096, conv.turn) == 1, "D5a/off: the compaction is discarded");
        check(c.archived() == 0 && c.archived_bytes() == 0, "D5a/off: nothing was archived");
        check(!fs::exists(archive_dir), "D5a/off: still no archive directory");
        check(c.size() == 0, "D5a/off: the copy was removed, as in delta 4");
    }

    // V10 / gate refinement (b): the archive subdirectory never counts as a stale spill file.
    fs::remove_all(dir);
    {
        fs::create_directories(dir);
        fs::create_directories(dir / "archive");
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/stale: open with an archive subdirectory present");
        check(c.stale_files_kept() == 0, "D5a/stale: the archive subdirectory is not counted as stale");
        check(c.size() == 0, "D5a/stale: nothing indexed");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::ids, 64ull << 20, 4, 0, 500};
        check(c.open_archive(cfg, error), "D5a/stale: open the archive over the existing subdirectory");
        check(c.stale_files_kept() == 0, "D5a/stale: still not counted after the archive is opened");
    }

    // V2 + V6 + V8 + V11: a compaction in state mode MOVES the .sess and .meta into the archive; the tier loses the
    // copy; counters match the filesystem; the archive is never a reuse candidate; the tier's GC is untouched.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/state: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::state, 64ull << 20, 4, 0, 500};
        check(c.open_archive(cfg, error), "D5a/state: open the archive");
        check(c.archive_enabled(), "D5a/state: archive enabled");
        D5aConv conv = d5a_conv();
        check(c.spill(conversation_with_ids(conv.ids), error), "D5a/state: spill the conversation");
        check(c.size() == 1 && c.bytes() > 0, "D5a/state: one tier copy with bytes");
        const size_t tier_evictions = c.disk_evictions();
        check(c.discard_diverged(conv.compacted, false, 4096, conv.turn) == 1, "D5a/state: the compaction is detected");
        check(c.size() == 0 && c.bytes() == 0, "D5a/state: the tier copy left the index");
        check(c.disk_evictions() == tier_evictions, "D5a/state: the tier's GC did not evict");
        check(c.archived() == 1, "D5a/state: archived=1");
        check(c.archived_bytes() > 0, "D5a/state: archived_bytes>0");
        check(d5a_count_ext(dir, ".sess") == 0, "D5a/state: no session file left in the tier");
        check(d5a_count_ext(dir / "archive", ".sess") == 1 && d5a_count_ext(dir / "archive", ".meta") == 1,
              "D5a/state: the session file and the sidecar moved to the archive");
        // V6: counters consistent with the filesystem
        check(c.archived() == d5a_count_ext(dir / "archive", ".meta"), "D5a/state: archived = sidecars in the archive");
        check(c.archived_bytes() == d5a_total_bytes(dir / "archive"), "D5a/state: archived_bytes = archive bytes");
        // V8: the archive is never offered by best()
        std::vector<int32_t> cont = conv.ids;
        cont.push_back(7777);
        check(!bool(c.best(cont, {}, false, 0.0, 0)), "D5a/state: best() never returns an archived copy");
    }

    // V3: ids mode archives ONLY the sidecar (no .sess, no K/V) and the copy is far smaller than the state one.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/ids: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::ids, 64ull << 20, 4, 0, 500};
        check(c.open_archive(cfg, error), "D5a/ids: open the archive");
        D5aConv conv = d5a_conv(2000);
        check(c.spill(conversation_with_ids(conv.ids), error), "D5a/ids: spill the conversation");
        check(c.discard_diverged(conv.compacted, false, 4096, conv.turn) == 1, "D5a/ids: the compaction is detected");
        check(d5a_count_ext(dir / "archive", ".sess") == 0, "D5a/ids: no session file is archived");
        check(d5a_count_ext(dir / "archive", ".meta") == 1 && c.archived() == 1, "D5a/ids: one sidecar-only entry");
        // ids-sized, not K/V-sized: the sidecar writes each id as an int64, so ~8 B/token; allow generous slack
        check(c.archived_bytes() > 0 && c.archived_bytes() < conv.ids.size() * 16 + 4096,
              "D5a/ids: archived_bytes is ids-sized, not K/V-sized");
        std::vector<int32_t> cont = conv.ids;
        cont.push_back(1);
        check(!bool(c.best(cont, {}, false, 0.0, 0)), "D5a/ids: best() never returns the archived ids");
    }

    // V5: the archive's OWN budget (-mib) trims the OLDEST and keeps the NEWEST.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/budget: open the tier");
        strata::core::SpillArchiveConfig big{dir / "archive", SpillArchiveMode::ids, 256ull << 20, 0, 0, 500};
        check(c.open_archive(big, error), "D5a/budget: open the archive with no cap");
        for (int k = 0; k < 4; ++k)
            check(c.archive_provisional(conv_ids(20000 + k * 10, 200), {}, {}, false, error),
                  "D5a/budget: archive a conversation");
        check(c.archived() == 4, "D5a/budget: four archived");
        const uint64_t budget = c.archived_bytes() / 2 + 32;   // room for about two entries
        strata::core::SpillArchiveConfig tight{dir / "archive", SpillArchiveMode::ids, budget, 0, 0, 500};
        check(c.open_archive(tight, error), "D5a/budget: reopen with a budget for about two");
        check(c.archived() <= 2, "D5a/budget: the archive keeps at most two");
        check(c.archived_bytes() <= budget, "D5a/budget: archived_bytes respects the archive's own budget");
        check(c.archive_evictions() >= 1, "D5a/budget: the archive's GC counted the evictions");
        check(c.disk_evictions() == 0, "D5a/budget: the tier's GC is untouched");
        const std::vector<uint64_t> serials = d5a_serials(dir / "archive");
        check(!serials.empty() && serials.back() == 4, "D5a/budget: the NEWEST archived entry survives");
    }

    // V7: the archive keeps at most -keep copies per conversation, and never enters the live tier.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/keep: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::ids, 256ull << 20, 1, 0, 500};
        check(c.open_archive(cfg, error), "D5a/keep: open the archive with keep=1");
        const std::vector<int32_t> a = conv_ids(31000, 200);
        check(c.archive_provisional(a, {}, {}, false, error), "D5a/keep: archive copy A1");
        check(c.archive_provisional(a, {}, {}, false, error), "D5a/keep: archive copy A2 (same conversation)");
        check(c.archived() == 1, "D5a/keep: only the newest copy of one conversation is kept");
        check(c.archive_evictions() >= 1, "D5a/keep: the keep GC counted the removal");
        check(c.archive_provisional(conv_ids(32000, 200), {}, {}, false, error), "D5a/keep: archive another conversation");
        check(c.archived() == 2, "D5a/keep: two conversations coexist");
        check(c.size() == 0 && c.bytes() == 0, "D5a/keep: the archive never enters the live tier");
    }

    // V9 / gate refinement (a): the cancellation archives the PRE-revert state (the generated tokens), and that
    // archived state is never offered by best() - even after a reopen.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/cancel: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::ids, 256ull << 20, 4, 0, 500};
        check(c.open_archive(cfg, error), "D5a/cancel: open the archive");
        std::vector<int32_t> closed = {1, 2, 500, 3, 4, 500, 5, 6, 500, 7, 8, 9};
        std::vector<int32_t> pre_revert = closed;                 // the state the request produced, before the revert
        pre_revert.push_back(500);                                // <|im_start|>assistant
        for (int32_t i = 0; i < 40; ++i) pre_revert.push_back(int32_t(9000 + i));   // the partial answer
        check(c.archive_provisional(pre_revert, {}, {}, false, error), "D5a/cancel: the pre-revert state is archived");
        check(c.archived() == 1, "D5a/cancel: one archived entry");
        check(c.size() == 0, "D5a/cancel: no tier copy is published");
        std::vector<int32_t> retry = pre_revert;
        retry.push_back(424242);
        check(!bool(c.best(retry, {}, false, 0.0, 0)), "D5a/cancel: best() never returns the archived pre-revert state");
        ConversationSpillCache again;
        check(again.open(dir, id, 1ull << 30, error), "D5a/cancel: reopen the tier");
        strata::core::SpillArchiveConfig cfg2{dir / "archive", SpillArchiveMode::ids, 256ull << 20, 4, 0, 500};
        check(again.open_archive(cfg2, error), "D5a/cancel: reopen the archive");
        check(again.archived() == 1, "D5a/cancel: the archived pre-revert state survives a reopen");
        check(!bool(again.best(retry, {}, false, 0.0, 0)), "D5a/cancel: still no match after reopen");
    }

    // V11 / gate refinement (c): archiving moves the copy out of the tier's bytes_ but is NOT a tier eviction, and
    // the archive never contaminates the tier's diagnostics.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/tiercounters: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::state, 256ull << 20, 0, 0, 500};
        check(c.open_archive(cfg, error), "D5a/tiercounters: open the archive");
        check(c.spill(conversation_with_ids(conv_ids(41000, 200)), error), "D5a/tiercounters: spill the other conversation");
        const uint64_t other_only = c.bytes();
        D5aConv conv = d5a_conv(1000);
        check(c.spill(conversation_with_ids(conv.ids), error), "D5a/tiercounters: spill the long conversation");
        check(c.bytes() > other_only, "D5a/tiercounters: two copies on the tier");
        const size_t tier_evictions = c.disk_evictions();
        check(c.discard_diverged(conv.compacted, false, 4096, conv.turn) == 1, "D5a/tiercounters: archive the long one");
        check(c.archived() == 1, "D5a/tiercounters: it reached the archive");
        check(c.bytes() == other_only, "D5a/tiercounters: the tier's bytes_ excludes the archive");
        check(c.disk_evictions() == tier_evictions, "D5a/tiercounters: archiving is not a tier eviction");
        check(c.stale_files_kept() == 0, "D5a/tiercounters: the archive does not count as stale");
    }

    // C8 / strict budget: a copy that ALONE exceeds the archive budget is NOT archived (it is discarded as before,
    // so the archive never holds more than --conversation-cache-archive-mib).
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/oversized: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::state, 1024, 0, 0, 500};
        check(c.open_archive(cfg, error), "D5a/oversized: open a tiny archive budget");
        D5aConv conv = d5a_conv(1000);
        check(c.spill(conversation_with_ids(conv.ids), error), "D5a/oversized: spill a conversation larger than the budget");
        check(c.discard_diverged(conv.compacted, false, 4096, conv.turn) == 1, "D5a/oversized: the compaction is detected");
        check(c.archived() == 0 && c.archived_bytes() == 0, "D5a/oversized: nothing was archived");
        check(c.size() == 0, "D5a/oversized: the tier copy was discarded as before");
        check(d5a_count_ext(dir / "archive", ".meta") == 0, "D5a/oversized: no stray sidecar in the archive");
    }

    // C8 / retention by age: --conversation-cache-archive-max-age-days is the archive's own time retention.
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/age: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::ids, 256ull << 20, 0, 0, 500};
        check(c.open_archive(cfg, error), "D5a/age: open the archive with no age retention");
        check(c.archive_provisional(conv_ids(51000, 200), {}, {}, false, error), "D5a/age: archive one");
        check(c.archive_provisional(conv_ids(52000, 200), {}, {}, false, error), "D5a/age: archive two");
        check(c.archived() == 2, "D5a/age: two archived");
        const auto old = fs::file_time_type::clock::now() - std::chrono::hours(24 * 10);
        for (const auto& e : fs::directory_iterator(dir / "archive"))
            if (e.path().extension() == ".meta") fs::last_write_time(e.path(), old);
        ConversationSpillCache keep0;
        check(keep0.open(dir, id, 1ull << 30, error), "D5a/age: reopen the tier");
        strata::core::SpillArchiveConfig cfg0{dir / "archive", SpillArchiveMode::ids, 256ull << 20, 0, 0, 500};
        check(keep0.open_archive(cfg0, error), "D5a/age: reopen with max-age 0");
        check(keep0.archived() == 2 && keep0.archive_age_evictions() == 0, "D5a/age: max-age 0 keeps the old copies");
        strata::core::SpillArchiveConfig cfg7{dir / "archive", SpillArchiveMode::ids, 256ull << 20, 0, 7, 500};
        check(keep0.open_archive(cfg7, error), "D5a/age: reopen with max-age 7 days");
        check(keep0.archive_age_evictions() >= 1, "D5a/age: the age GC counted the removals");
        check(keep0.archived() == 0, "D5a/age: every copy older than the limit went");
    }

    // O2 (audit): a failed MOVE in state mode leaves the tier entry INTACT - the discard does not happen, so the
    // tier is never left worse than before. The failure is forced by holding the .sess open with no sharing (MSVC:
    // _SH_DENYRW), so the rename AND the copy-then-remove fallback both fail. (On a platform without _sopen_s the
    // case is skipped.)
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/relocate: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::state, 256ull << 20, 0, 0, 500};
        check(c.open_archive(cfg, error), "D5a/relocate: open the archive");
        D5aConv conv = d5a_conv(500);
        check(c.spill(conversation_with_ids(conv.ids), error), "D5a/relocate: spill a conversation");
        std::string sess;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".sess") sess = e.path().string();
        check(!sess.empty(), "D5a/relocate: the session file is on disk");
#ifdef _MSC_VER
        int hold = -1;
        const errno_t held = _sopen_s(&hold, sess.c_str(), _O_RDONLY | _O_BINARY, _SH_DENYRW, 0);
        check(held == 0 && hold != -1, "D5a/relocate: the session file is held open exclusively");
#endif
        check(c.discard_diverged(conv.compacted, false, 4096, conv.turn) == 0, "D5a/relocate: the discard is not reported");
        check(c.size() == 1 && c.bytes() > 0, "D5a/relocate: the tier entry is INTACT after the failed archive");
        check(fs::exists(sess), "D5a/relocate: the session file is still there");
        check(c.archived() == 0 && c.archived_bytes() == 0, "D5a/relocate: nothing was archived");
        check(d5a_count_ext(dir / "archive", ".meta") == 0, "D5a/relocate: the archive holds no stray sidecar");
#ifdef _MSC_VER
        if (hold != -1) _close(hold);
#endif
    }

    // F3 (audit addendum): force the failure AFTER a successful MOVE, so the multi-file rollback really runs. A
    // two-stage conversation is archived in state mode: the FIRST move (.sess) succeeds, the SECOND (.stage1.sess)
    // is held open, so archive_entry must put the .sess back and leave the tier entry intact (both files present).
    fs::remove_all(dir);
    {
        ConversationSpillCache c;
        check(c.open(dir, id, 1ull << 30, error), "D5a/relocate-multi: open the tier");
        strata::core::SpillArchiveConfig cfg{dir / "archive", SpillArchiveMode::state, 256ull << 20, 0, 0, 500};
        check(c.open_archive(cfg, error), "D5a/relocate-multi: open the archive");
        D5aConv conv = d5a_conv(500);
        SavedConversation two = conversation_with_ids(conv.ids);
        SavedConversation stage1 = sample(600);
        stage1.layer_lo = 48; stage1.layer_hi = 96;
        two.stage_images.push_back(stage1);
        check(c.spill(two, error), "D5a/relocate-multi: spill a two-stage conversation");
        std::string sess, stage;
        for (const auto& e : fs::directory_iterator(dir)) {
            const std::string p = e.path().string();
            if (p.find(".stage1.sess") != std::string::npos) stage = p;
            else if (e.path().extension() == ".sess") sess = p;
        }
        check(!sess.empty() && !stage.empty(), "D5a/relocate-multi: both stage files are on disk");
        const uint64_t bytes_before = c.bytes();
#ifdef _MSC_VER
        int hold = -1;
        const errno_t held = _sopen_s(&hold, stage.c_str(), _O_RDONLY | _O_BINARY, _SH_DENYRW, 0);
        check(held == 0 && hold != -1, "D5a/relocate-multi: the stage file is held open exclusively");
#endif
        check(c.discard_diverged(conv.compacted, false, 4096, conv.turn) == 0, "D5a/relocate-multi: the discard is not reported");
        check(c.size() == 1 && c.bytes() == bytes_before, "D5a/relocate-multi: the tier entry is INTACT (bytes unchanged)");
        check(fs::exists(sess) && fs::exists(stage), "D5a/relocate-multi: the rolled-back .sess is back in the tier");
        check(c.archived() == 0 && c.archived_bytes() == 0, "D5a/relocate-multi: nothing was archived");
        check(d5a_count_ext(dir / "archive", ".meta") == 0, "D5a/relocate-multi: no stray sidecar after the rollback");
#ifdef _MSC_VER
        if (hold != -1) _close(hold);
#endif
    }

    // ============================ Delta 6: the cascade (RAM -> warm -> cold) ============================
    // A level wired to a next one (set_next) does not remove what its two CAPACITY levers take: it MOVES the copy
    // down.  One test per frontier.  T11 (a real cross-volume move) is NOT covered here: the harness has no second
    // volume to mount, so the copy-then-remove half of relocate_file is left to the report's non-claims.  What IS
    // covered is its consequence: a move that fails leaves nothing half-done on either side.

    // The bytes of one conversation as it lands on disk: the budgets below are built from this, so "room for two"
    // and "room for one" are exact rather than guessed.
    auto measure_bytes = [&](const SavedConversation& c) -> uint64_t {
        ConversationSpillCache p;
        std::string e;
        const fs::path pd = dir / ("probe-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        if (!p.open(pd, id, 1ull << 30, e) || !p.spill(c, e)) return 0;
        const uint64_t b = p.bytes();
        fs::remove_all(pd);
        return b;
    };

    // T1 + T4: the warm level degrades its OLDEST to the cold by BUDGET, in write order, and a read does not
    // reorder that order.  `warm.disk_evictions()` stays 0: a degradation is not an eviction.
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm;
        check(cold.open(cold_dir, id, 1ull << 30, error), "T1: open the cold level");
        warm.set_next(&cold);
        check(warm.has_next(), "T1: the warm level is wired to the cold one");
        const uint64_t one = measure_bytes(conversation_with_ids(conv_ids(91000, 400)));
        check(one > 0, "T1: one conversation has bytes");
        check(warm.open(warm_dir, id, 2 * one, error), "T1: open the warm level with room for two");

        const std::vector<int32_t> idsA = conv_ids(91000, 400), idsB = conv_ids(92000, 400), idsC = conv_ids(93000, 400);
        check(warm.spill(conversation_with_ids(idsA), error), "T1: spill A");
        check(warm.spill(conversation_with_ids(idsB), error), "T1: spill B");
        check(warm.size() == 2 && cold.size() == 0, "T1: two in the warm, none in the cold");
        check(warm.spill(conversation_with_ids(idsC), error), "T1: spill C (the warm overflows)");
        check(warm.size() == 2 && warm.degraded() == 1, "T1: the oldest left the warm, degraded=1");
        check(cold.size() == 1, "T1: it arrived in the cold level");
        check(warm.disk_evictions() == 0 && warm.age_evictions() == 0, "T1: the warm level evicted nothing");

        // the oldest is the one that went, and it is found in the cold by the same match rule
        std::vector<int32_t> contA = idsA; contA.push_back(7777);
        std::vector<int32_t> contB = idsB; contB.push_back(7777);
        const auto coldA = cold.best(contA, {}, false, 0.0, 0);
        check(bool(coldA) && coldA.tokens == (int64_t) idsA.size(), "T1: A resumes from the cold level");
        check(!bool(warm.best(contA, {}, false, 0.0, 0)), "T1: A is no longer in the warm level");
        const auto warmB = warm.best(contB, {}, false, 0.0, 0);
        check(bool(warmB), "T1: B is still in the warm level");

        // T4: a READ does not reorder the FIFO (no MRU anywhere).  best() and load() on B, then the next overflow
        // must take B - the next one by INSERTION order, not the one that was just read (and not C, the newest).
        SavedConversation read_back;
        check(warm.load(warmB.path, read_back, {}, error), "T4: load B back");
        check(read_back.live.ids == idsB, "T4: the loaded copy is B");
        const std::vector<int32_t> idsD = conv_ids(94000, 400);
        check(warm.spill(conversation_with_ids(idsD), error), "T4: spill D (a second overflow)");
        std::vector<int32_t> contC = idsC; contC.push_back(7777);
        std::vector<int32_t> contD = idsD; contD.push_back(7777);
        check(bool(cold.best(contB, {}, false, 0.0, 0)), "T4: B went next: FIFO is write order, not last read");
        check(bool(warm.best(contC, {}, false, 0.0, 0)), "T4: C is still in the warm level");
        check(bool(warm.best(contD, {}, false, 0.0, 0)), "T4: D is in the warm level");
        check(cold.size() == 2 && cold.degraded() == 0, "T4: the cold level received two and degraded nothing");
    }

    // T3: the COLD level is the last one: its GC REMOVES, oldest first, and counts it as a disk eviction.
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm;
        const uint64_t one = measure_bytes(conversation_with_ids(conv_ids(95000, 400)));
        check(cold.open(cold_dir, id, 2 * one, error), "T3: open the cold level with room for two");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, 2 * one, error), "T3: open the warm level with room for two");
        const std::vector<int32_t> ids[5] = {conv_ids(95000, 400), conv_ids(96000, 400), conv_ids(97000, 400),
                                             conv_ids(98000, 400), conv_ids(99000, 400)};
        for (int k = 0; k < 5; ++k)
            check(warm.spill(conversation_with_ids(ids[k]), error), "T3: spill through the cascade");
        check(cold.size() == 2, "T3: the cold level holds its budget's worth");
        check(cold.disk_evictions() >= 1, "T3: the cold level REMOVED the overflow (it is the last level)");
        check(cold.degraded() == 0, "T3: the cold level has nothing below, so it degrades nothing");
        std::vector<int32_t> cont0 = ids[0]; cont0.push_back(1);
        check(!bool(cold.best(cont0, {}, false, 0.0, 0)), "T3: the OLDEST of the cold level is the one removed");
        std::vector<int32_t> cont4 = ids[4]; cont4.push_back(1);
        check(bool(cold.best(cont4, {}, false, 0.0, 0)) || bool(warm.best(cont4, {}, false, 0.0, 0)),
              "T3: the newest conversation still resumes somewhere");
    }

    // T2: the AGE lever of the warm level degrades too, and the cold level RE-SEALS the arrival's stamp, so its own
    // age retention does not prune it at once.
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm;
        check(cold.open(cold_dir, id, 1ull << 30, error, SpillWhenFull::evict_oldest, 7), "T2: open the cold with max-age=7");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, 1ull << 30, error, SpillWhenFull::evict_oldest, 7), "T2: open the warm with max-age=7");
        const std::vector<int32_t> idsA = conv_ids(20000, 400);
        check(warm.spill(conversation_with_ids(idsA), error), "T2: spill one");
        check(warm.size() == 1 && warm.age_evictions() == 0, "T2: young: nothing by age yet");
        // age the copy in the WARM: the age lever now fires and must DEGRADE, not delete
        const auto ten_days_ago = fs::file_time_type::clock::now() - std::chrono::hours(24 * 10);
        for (const auto& e : fs::directory_iterator(warm_dir))
            if (e.path().extension() == ".sess") fs::last_write_time(e.path(), ten_days_ago);
        ConversationSpillCache warm2;
        warm2.set_next(&cold);
        check(warm2.open(warm_dir, id, 1ull << 30, error, SpillWhenFull::evict_oldest, 7), "T2: reopen the warm");
        check(warm2.size() == 0 && warm2.degraded() >= 1, "T2: the aged copy DEGRADED, it was not deleted");
        check(warm2.age_evictions() == 0, "T2: the warm level deleted nothing by age");
        check(cold.size() == 1, "T2: it is in the cold level");
        // re-sealed: a SECOND pass of the cold's own age GC does not take it (it arrived now, whatever its age above)
        ConversationSpillCache cold2;
        check(cold2.open(cold_dir, id, 1ull << 30, error, SpillWhenFull::evict_oldest, 7), "T2: reopen the cold");
        check(cold2.size() == 1 && cold2.age_evictions() == 0, "T2: the arrival's stamp was re-sealed to now");
    }

    // T5: ALL-OR-NOTHING - the most dangerous path of the change.  The small conversation A fits the warm's
    // budget, the big one B does not (so B is left alone by the GC's oversized rule and A is the candidate).  (a)
    // the cold level in `reject` cannot take A: A STAYS in the warm with its bytes, and the GC finishes.  (b) the
    // move itself fails: the same, and the cold level keeps no half copy.
    const std::vector<int32_t> ids_small = conv_ids(30000, 200), ids_big = conv_ids(31000, 2000);
    const uint64_t small_bytes = measure_bytes(conversation_with_ids(ids_small));
    const uint64_t big_bytes = measure_bytes(conversation_with_ids(ids_big));
    check(small_bytes > 1 && small_bytes < big_bytes, "T5: a small copy and a big one, both measurable");
    const uint64_t warm_budget = big_bytes - 1;   // A fits, B is oversized and left alone: A is the degradation
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm;
        check(cold.open(cold_dir, id, 1, error, SpillWhenFull::reject), "T5a: open the cold with a 1-byte budget, reject");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, warm_budget, error), "T5a: open the warm with room for the small one");
        check(warm.spill(conversation_with_ids(ids_small), error), "T5a: spill A (it fits)");
        check(warm.spill(conversation_with_ids(ids_big), error), "T5a: spill B (the warm overflows)");
        check(warm.size() == 2 && warm.bytes() > 0, "T5a: A STAYED in the warm level");
        check(warm.degraded() == 0 && warm.disk_evictions() == 0, "T5a: it was neither degraded nor removed");
        check(cold.size() == 0 && cold.bytes() == 0, "T5a: the cold level took nothing");
        check(d5a_count_ext(cold_dir, ".meta") == 0, "T5a: the cold level left no half copy");
        std::vector<int32_t> contA = ids_small; contA.push_back(1);
        check(bool(warm.best(contA, {}, false, 0.0, 0)), "T5a: A still resumes from the warm level");
    }
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm;
        check(cold.open(cold_dir, id, 1ull << 30, error), "T5b: open the cold level");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, warm_budget, error), "T5b: open the warm with room for the small one");
        check(warm.spill(conversation_with_ids(ids_small), error), "T5b: spill A");
        std::string held_path;
        for (const auto& e : fs::directory_iterator(warm_dir))
            if (e.path().extension() == ".sess") held_path = e.path().string();
        check(!held_path.empty(), "T5b: the warm copy is on disk");
#ifdef _MSC_VER
        int hold = -1;
        const errno_t held = _sopen_s(&hold, held_path.c_str(), _O_RDONLY | _O_BINARY, _SH_DENYRW, 0);
        check(held == 0 && hold != -1, "T5b: the warm session file is held open exclusively");
#endif
        // the second spill overflows the warm: the degradation of A is attempted and the MOVE fails
        check(warm.spill(conversation_with_ids(ids_big), error), "T5b: spill B (the warm overflows)");
        check(warm.degraded() == 0 && warm.disk_evictions() == 0, "T5b: the refused move degraded nothing");
        check(warm.size() == 2, "T5b: the warm level was not left worse off");
        check(fs::exists(held_path), "T5b: the copy is still in the warm level");
        check(d5a_count_ext(cold_dir, ".meta") == 0, "T5b: the cold level rolled its sidecar back");
        check(d5a_count_ext(cold_dir, ".sess") == 0, "T5b: nothing half-moved reached the cold level");
        std::vector<int32_t> contA = ids_small; contA.push_back(1);
        check(bool(warm.best(contA, {}, false, 0.0, 0)), "T5b: A still resumes from the warm level");
#ifdef _MSC_VER
        if (hold != -1) _close(hold);
#endif
    }

    // T6: a CORRECTION still discards.  A compaction and a supersede pass on the warm level must NOT push the copy
    // down: a copy that can never be a hit is not promoted into a level that IS a reuse candidate.
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm;
        check(cold.open(cold_dir, id, 1ull << 30, error), "T6: open the cold level");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, 1ull << 30, error), "T6: open the warm level");
        D5aConv conv = d5a_conv();
        check(warm.spill(conversation_with_ids(conv.ids), error), "T6: spill a long conversation");
        check(warm.discard_diverged(conv.compacted, false, 4096, conv.turn) == 1, "T6: the compaction is discarded");
        check(warm.size() == 0, "T6: the copy left the warm level");
        check(cold.size() == 0 && cold.degraded() == 0, "T6: the compaction did NOT degrade into the cold");
        // the supersede pass, the other correction path
        const std::vector<int32_t> ids = conv_ids(33000, 2000);
        SavedConversation img = conversation_with_ids(ids);
        check(warm.spill(img, error), "T6: spill a conversation to supersede");
        check(warm.drop_superseded(ids, {}, img.checkpoints, img.cvec) == 1, "T6: its own copy is superseded");
        check(warm.size() == 0 && cold.size() == 0, "T6: the supersede discarded in place, not into the cold");
        check(warm.degraded() == 0 && cold.degraded() == 0, "T6: no correction degraded anything");
    }

    // T7: the degraded copy keeps the identity.  The cold level re-opens over what the degradation wrote and
    // indexes it: nothing is foreign, stale or orphan.
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm, probe;
        check(probe.open(dir / "probe", id, 1ull << 30, error), "T7: open the probe");
        check(probe.spill(conversation_with_ids(conv_ids(34000, 400)), error), "T7: probe spill");
        const uint64_t one = probe.bytes();
        check(cold.open(cold_dir, id, 1ull << 30, error), "T7: open the cold level");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, one, error), "T7: open the warm with room for one");
        const std::vector<int32_t> idsA = conv_ids(34000, 400), idsB = conv_ids(35000, 400);
        check(warm.spill(conversation_with_ids(idsA), error), "T7: spill A");
        check(warm.spill(conversation_with_ids(idsB), error), "T7: spill B (A degrades)");
        check(warm.degraded() == 1 && cold.size() == 1, "T7: A reached the cold level");
        ConversationSpillCache cold_reopened;
        check(cold_reopened.open(cold_dir, id, 1ull << 30, error), "T7: REOPEN the cold level");
        check(cold_reopened.size() == 1, "T7: the degraded copy is reindexed");
        check(cold_reopened.foreign_files_kept() == 0, "T7: nothing is foreign (same identity)");
        check(cold_reopened.stale_files_kept() == 0 && cold_reopened.orphan_files_kept() == 0,
              "T7: no stale and no orphan file");
        std::vector<int32_t> contA = idsA; contA.push_back(1);
        const auto again = cold_reopened.best(contA, {}, false, 0.0, 0);
        check(bool(again) && again.tokens == (int64_t) idsA.size(), "T7: the reopened cold resumes A");
    }

    // T8: a MULTISTAGE conversation degrades whole: every stage file moves and the cold level reads them back with
    // their own stage images.
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm, probe;
        check(probe.open(dir / "probe", id, 1ull << 30, error), "T8: open the probe");
        SavedConversation two_stage = sample(1200);
        SavedConversation stage1 = sample(600);
        stage1.layer_lo = 48; stage1.layer_hi = 96;
        two_stage.stage_images.push_back(stage1);
        check(probe.spill(two_stage, error), "T8: probe spill a two-stage conversation");
        const uint64_t one = probe.bytes();
        check(cold.open(cold_dir, id, 1ull << 30, error), "T8: open the cold level");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, one, error), "T8: open the warm with room for one");
        check(warm.spill(two_stage, error), "T8: spill the two-stage conversation");
        check(warm.spill(conversation_with_ids(conv_ids(36000, 400)), error), "T8: spill another (the first degrades)");
        check(warm.degraded() == 1 && cold.size() == 1, "T8: the two-stage copy reached the cold level");
        check(d5a_count_ext(warm_dir, ".sess") <= 1, "T8: its stage files left the warm level");
        std::vector<int32_t> cont = two_stage.live.ids; cont.push_back(4242);
        const auto match = cold.best(cont, two_stage.live.imgs, false, 0.0, 0);
        check(bool(match) && match.stages == 1, "T8: the cold level reports the extra stage");
        SavedConversation back;
        check(cold.load(match.path, back, {}, error), "T8: the cold level reads every stage back");
        check(back.stage_images.size() == 1, "T8: one stage image read back");
        check(back.stage_images[0].live.ids == stage1.live.ids, "T8: the stage ids round-trip through the degradation");
        check(back.stage_images[0].layer_lo == 48 && back.stage_images[0].layer_hi == 96,
              "T8: the stage layer range round-trips");
    }

    // T9: when-full=reject on the WARM level: it refuses the new copy and degrades NOTHING (reject is "this level
    // removes nothing", and a degradation is an eviction by another name).
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm;
        check(cold.open(cold_dir, id, 1ull << 30, error), "T9: open the cold level");
        warm.set_next(&cold);
        // `spill`'s reject pre-check sizes the incoming copy as image.bytes() + 1 MiB, so the warm's budget is
        // built from that exact figure: the first copy fits, the second one cannot.
        SavedConversation imgA = conversation_with_ids(conv_ids(37000, 400));
        const uint64_t reject_budget = imgA.bytes() + (1ull << 20) + 4096;
        check(warm.open(warm_dir, id, reject_budget, error, SpillWhenFull::reject), "T9: open the warm with room for one, reject");
        check(warm.spill(imgA, error), "T9: seed one conversation");
        std::string keep;
        for (const auto& e : fs::directory_iterator(warm_dir))
            if (e.path().extension() == ".sess") keep = e.path().string();
        const std::vector<uint8_t> keep_bytes = read_file_bytes(keep);
        std::string reject_error;
        check(!warm.spill(conversation_with_ids(conv_ids(38000, 4000)), reject_error), "T9: the warm refuses the new copy");
        check(!reject_error.empty(), "T9: the refusal says why");
        check(warm.size() == 1 && warm.degraded() == 0, "T9: nothing was degraded for it");
        check(warm.degraded() == 0 && warm.disk_evictions() == 0, "T9: reject evicts and degrades nothing");
        check(cold.size() == 0, "T9: the cold level is untouched");
        check(read_file_bytes(keep) == keep_bytes, "T9: the stored copy is byte-for-byte intact");
    }

    // T10: with NO next level the tier is byte for byte what it always was: the budget GC REMOVES the oldest.
    fs::remove_all(dir);
    {
        ConversationSpillCache cold_less, probe;
        check(probe.open(dir / "probe", id, 1ull << 30, error), "T10: open the probe");
        check(probe.spill(conversation_with_ids(conv_ids(39000, 400)), error), "T10: probe spill");
        const uint64_t one = probe.bytes();
        check(cold_less.open(dir / "flat", id, 2 * one, error), "T10: open a FLAT tier (no next level)");
        check(!cold_less.has_next(), "T10: it has no next level");
        for (int k = 0; k < 3; ++k)
            check(cold_less.spill(conversation_with_ids(conv_ids(39000 + k * 100, 400)), error), "T10: spill");
        check(cold_less.size() == 2, "T10: the budget kept two");
        check(cold_less.disk_evictions() == 1, "T10: the oldest was REMOVED, as before");
        check(cold_less.degraded() == 0, "T10: nothing was degraded");
    }

    // T11: the ENTRY CAP is NOT a removal trigger.  A cold level holding exactly kMaxSidecarEntries copies, with
    // MiB to spare, must NOT delete one when a degradation lands: the byte budget is what a level removes for, and
    // a count the operator never set must not.  This is the cascade bug of 2026-10-15 made a regression test - at
    // 256 the cold lost exactly one copy per degradation of the warm (45 for 45) with 460800 MiB configured and
    // 265848 MiB in use, because enforce_budget() removed on `entries_.size() > kMaxSidecarEntries` too.
    fs::remove_all(dir);
    {
        const fs::path warm_dir = dir / "warm", cold_dir = dir / "cold";
        ConversationSpillCache cold, warm, probe;
        check(probe.open(dir / "probe", id, 1ull << 30, error), "T11: open the probe");
        check(probe.spill(conversation_with_ids(conv_ids(60000, 400)), error), "T11: probe spill");
        const uint64_t one = probe.bytes();
        check(one > 0, "T11: one conversation has bytes");
        const size_t cap = strata::core::ConversationSpillCache::entries_cap();
        check(cap == 256, "T11: the entry cap is the 256 the bug hid");
        // The cold: room in MiB for a thousand copies, so ONLY the entry count could ever be the lever that fires.
        check(cold.open(cold_dir, id, 1024 * one, error), "T11: open the cold with MiB headroom far past the cap");
        warm.set_next(&cold);
        check(warm.open(warm_dir, id, one, error), "T11: open the warm with room for one");
        // Fill the cold to EXACTLY its entry cap, directly (the cold level's own writer).  The byte budget is never
        // reached, so this alone removes nothing - the level is simply AT its cap.
        bool filled = true;
        for (size_t k = 0; k < cap; ++k)
            filled = cold.spill(conversation_with_ids(conv_ids(61000 + (int32_t) k, 400)), error) && filled;
        check(filled, "T11: fill the cold to its entry cap");
        check(cold.size() == cap, "T11: the cold holds exactly its entry cap");
        check(cold.disk_evictions() == 0, "T11: filling to the cap removed nothing");
        // The OLDEST cold copy - the exact victim the bug would have deleted on the degradation below.
        std::vector<int32_t> oldest = conv_ids(61000, 400);
        oldest.push_back(5);
        check(bool(cold.best(oldest, {}, false, 0.0, 0)), "T11: the oldest cold copy is present before the degradation");
        // Now a real degradation: the warm overflows and its oldest copy moves DOWN into the full cold level.
        check(warm.spill(conversation_with_ids(conv_ids(70000, 400)), error), "T11: warm spill A");
        check(warm.spill(conversation_with_ids(conv_ids(71000, 400)), error), "T11: warm spill B (A degrades)");
        check(warm.degraded() == 1, "T11: the warm degraded one copy into the cold");
        check(cold.size() == cap + 1, "T11: the cold ACCEPTED the copy: the entry cap removed nothing");
        check(cold.disk_evictions() == 0, "T11: a degradation at the entry cap deleted NOTHING (the fix)");
        check(bool(cold.best(oldest, {}, false, 0.0, 0)), "T11: the oldest cold copy SURVIVED the degradation");
        std::vector<int32_t> arrived = conv_ids(70000, 400);
        arrived.push_back(5);
        check(bool(cold.best(arrived, {}, false, 0.0, 0)), "T11: the degraded copy is in the cold level");
    }

    fs::remove_all(dir);
    std::printf("conversation_spill_test: %d checks passed\n", checks);
    return 0;
}
