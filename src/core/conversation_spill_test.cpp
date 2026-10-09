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

    fs::remove_all(dir);
    std::printf("conversation_spill_test: %d checks passed\n", checks);
    return 0;
}
