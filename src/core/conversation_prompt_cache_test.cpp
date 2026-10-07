// CPU-only tests for the system-prompt prefill cache (include/strata/core/conversation_prompt_cache.hpp): a variant
// is a session file keyed by the hash of a system-prompt token prefix, matched from its sidecar, reloaded whole, and
// never removed because the prefix changed - only the GC (space or age, oldest first) removes one.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_prompt_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
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
// The same image without its K/V (the streamed form: the K/V comes from per-stage sources).
SavedConversation sample_meta(size_t tokens) {
    SavedConversation s = sample(tokens);
    s.kv.clear();
    return s;
}
// A layer-split image: stage 0 (layers [0,24)) plus one later stage (layers [24,48)) with visibly different K/V.
SavedConversation two_stage(size_t tokens) {
    SavedConversation s = sample(tokens);
    s.layer_lo = 0; s.layer_hi = 24;
    SavedConversation stage = sample(tokens);
    stage.layer_lo = 24; stage.layer_hi = 48;
    stage.kv.clear();
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 5 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = 9; kv.idx_dim = 128;
        kv.k = pattern(2048, uint8_t(layer + 100)); kv.v = pattern(2048 + layer, uint8_t(layer + 110));
        kv.k_scale = pattern(256, uint8_t(layer + 120)); kv.v_scale = pattern(256, uint8_t(layer + 130));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(layer + 140));
        stage.kv.push_back(std::move(kv));
    }
    s.stage_images.push_back(std::move(stage));
    return s;
}
bool kv_equal(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads && a.head_dim == b.head_dim &&
           a.page_size == b.page_size && a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim &&
           a.k == b.k && a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}
bool image_equal(const SavedConversation& a, const SavedConversation& b) {
    if (a.live.ids != b.live.ids || a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi ||
        a.cvec != b.cvec || a.kv.size() != b.kv.size() || a.stage_images.size() != b.stage_images.size()) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) if (!kv_equal(a.kv[i], b.kv[i])) return false;
    for (size_t i = 0; i < a.stage_images.size(); ++i) if (!image_equal(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}
// A streamed K/V source over a whole in-RAM part table: what a --kv-resident layer's authoritative host pool exposes.
SessionKvSource stream_source(int format, const std::vector<std::vector<uint8_t>>& parts) {
    SessionKvSource s;
    s.format = format; s.cells = 1024; s.heads = 2; s.head_dim = 256; s.page_size = 64; s.pooled_rows = 9;
    s.idx_dim = 128;
    for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i].size();
    auto table = std::make_shared<std::vector<std::vector<uint8_t>>>(parts);
    s.read = [table](size_t part, size_t offset, void* dst, size_t n) -> bool {
        if (part >= 5 || offset > (*table)[part].size() || n > (*table)[part].size() - offset) return false;
        std::memcpy(dst, (*table)[part].data() + offset, n);
        return true;
    };
    s.error = [] { return std::string(); };
    return s;
}
size_t count_sess(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".sess") ++n;
    return n;
}
// The stage-1 file beside stage 0's ("<stem>.sess" -> "<stem>.stage1.sess").
fs::path stage1_of(const std::string& stage0) {
    std::string s = stage0;
    if (s.size() > 5 && s.compare(s.size() - 5, 5, ".sess") == 0) s.resize(s.size() - 5);
    return fs::path(s + ".stage1.sess");
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
    check(!off.store_streamed(123, first, {}, error), "closed cache refuses a streamed store");
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

    // ================= a layer split: one session file per stage + one joint sidecar =================
    // (a) a two-stage variant round-trips byte for byte through store() (host image)
    {
        const fs::path mdir = dir.string() + "-stages";
        fs::remove_all(mdir);
        ConversationPromptCache mc;
        check(mc.open(mdir, id, 1ull << 30, 8, 0, error), "stages: open");
        SavedConversation ts = two_stage(1000);
        std::vector<int64_t> pre(ts.live.ids.begin(), ts.live.ids.end());
        const uint64_t mk = mc.key_for(pre.data(), pre.size(), "");
        check(mc.store(mk, ts, error), "stages: store a two-stage variant");
        check(mc.variants() == 1, "stages: one variant");
        check(count_sess(mdir) == 2, "stages: two session files on disk");
        ConversationPromptMatch smatch;
        check(mc.lookup(mk, smatch) && smatch.stages == 2, "stages: the sidecar says two stages");
        check(fs::exists(stage1_of(smatch.path)), "stages: stage 1 file exists");
        SavedConversation back;
        check(mc.load(smatch.path, back, {}, error), "stages: load the two-stage variant");
        check(back.stage_images.size() == 1, "stages: one later-stage image read back");
        check(image_equal(back, ts), "stages: two stages round-trip byte for byte");
        // a restart still reads both files
        ConversationPromptCache mc2;
        check(mc2.open(mdir, id, 1ull << 30, 8, 0, error), "stages: reopen");
        check(mc2.lookup(mk, smatch) && smatch.stages == 2, "stages: two stages after a reopen");
        SavedConversation back2;
        check(mc2.load(smatch.path, back2, {}, error) && image_equal(back2, ts), "stages: byte-for-byte after a reopen");
        fs::remove_all(mdir);
    }

    // (b) a stage mismatch is refused clean, before anything is applied to the caller's image
    {
        const fs::path mdir = dir.string() + "-mismatch";
        fs::remove_all(mdir);
        ConversationPromptCache mc;
        check(mc.open(mdir, id, 1ull << 30, 8, 0, error), "mismatch: open");
        SavedConversation ts = two_stage(1000);
        std::vector<int64_t> pre(ts.live.ids.begin(), ts.live.ids.end());
        const uint64_t mk = mc.key_for(pre.data(), pre.size(), "");
        check(mc.store(mk, ts, error), "mismatch: store a two-stage variant");
        ConversationPromptMatch mmatch;
        check(mc.lookup(mk, mmatch) && mmatch.stages == 2, "mismatch: two stages");
        // a stage file of ANOTHER identity: the session reader's model/config check refuses it
        const fs::path fdir = dir.string() + "-foreignstage";
        fs::remove_all(fdir);
        ConversationPromptCache fc;
        check(fc.open(fdir, SessionFileIdentity{9, 9}, 1ull << 30, 8, 0, error), "mismatch: open the foreign cache");
        SavedConversation fs1 = sample(400);
        std::vector<int64_t> fpre(fs1.live.ids.begin(), fs1.live.ids.end());
        const uint64_t fk = fc.key_for(fpre.data(), fpre.size(), "");
        check(fc.store(fk, fs1, error), "mismatch: store a foreign-identity session");
        ConversationPromptMatch fmatch;
        check(fc.lookup(fk, fmatch), "mismatch: the foreign variant is present");
        fs::copy_file(fmatch.path, stage1_of(mmatch.path), fs::copy_options::overwrite_existing);
        SavedConversation out_img;
        out_img.live.ids = {7, 7, 7};
        std::string load_error;
        check(!mc.load(mmatch.path, out_img, {}, load_error), "mismatch: a foreign-identity stage is refused");
        check(out_img.live.ids == std::vector<int32_t>{7, 7, 7}, "mismatch: nothing was applied to the output image");
        check(mc.lookup(mk, mmatch), "mismatch: the variant is kept (never deleted) until the GC");
        // restore the real stage 1, then truncate it: a torn stage is refused too, still cleanly
        check(mc.store(mk, ts, error), "mismatch: re-store the valid two-stage variant");
        {
            std::ofstream out(stage1_of(mmatch.path), std::ios::binary | std::ios::trunc);
            out << "broken";
        }
        out_img.live.ids = {8, 8, 8};
        check(!mc.load(mmatch.path, out_img, {}, load_error), "mismatch: a truncated stage is refused");
        check(out_img.live.ids == std::vector<int32_t>{8, 8, 8}, "mismatch: still nothing applied");
        fs::remove_all(mdir);
        fs::remove_all(fdir);
    }

    // (c) a single-stage and a two-stage variant coexist, each loads with its own stage count
    {
        const fs::path cdir = dir.string() + "-coexist";
        fs::remove_all(cdir);
        ConversationPromptCache cc;
        check(cc.open(cdir, id, 1ull << 30, 8, 0, error), "coexist: open");
        SavedConversation s1 = sample(1000);
        SavedConversation s2 = two_stage(1200);
        std::vector<int64_t> p1(s1.live.ids.begin(), s1.live.ids.end());
        std::vector<int64_t> p2(s2.live.ids.begin(), s2.live.ids.end());
        const uint64_t kA = cc.key_for(p1.data(), p1.size(), "");
        const uint64_t kB = cc.key_for(p2.data(), p2.size(), "");
        check(cc.store(kA, s1, error), "coexist: store the single-stage variant");
        check(cc.store(kB, s2, error), "coexist: store the two-stage variant beside it");
        check(cc.variants() == 2, "coexist: both variants live");
        ConversationPromptMatch ma, mb;
        check(cc.lookup(kA, ma) && ma.stages == 1, "coexist: the single-stage variant has one stage");
        check(cc.lookup(kB, mb) && mb.stages == 2, "coexist: the two-stage variant has two");
        SavedConversation ba, bb;
        check(cc.load(ma.path, ba, {}, error) && ba.stage_images.empty(), "coexist: the single-stage variant loads single");
        check(cc.load(mb.path, bb, {}, error) && bb.stage_images.size() == 1, "coexist: the two-stage variant loads both");
        check(image_equal(bb, s2), "coexist: the two-stage variant round-trips whole");
        check(cc.hash_changes() == 1, "coexist: the second prefix counts as a hash change");
        fs::remove_all(cdir);
    }

    // (d) the streamed form: K/V read from a per-stage host pool (--kv-resident), no host image capture
    {
        const fs::path sdir = dir.string() + "-streamed";
        fs::remove_all(sdir);
        ConversationPromptCache sc;
        check(sc.open(sdir, id, 1ull << 30, 8, 0, error), "streamed: open");
        const std::vector<std::vector<uint8_t>> lb0 = {bytes_of(4096, 1), bytes_of(4096, 2), bytes_of(512, 3),
                                                       bytes_of(512, 4), bytes_of(9 * 128 * 4, 5)};
        const std::vector<std::vector<uint8_t>> lb1 = {bytes_of(2048, 11), bytes_of(2048, 12), bytes_of(256, 13),
                                                       bytes_of(256, 14), bytes_of(9 * 128 * 4, 15)};
        SavedConversation m0 = sample_meta(1000);
        SavedConversation m1 = sample_meta(1000);
        m1.layer_lo = 24; m1.layer_hi = 48;
        std::vector<std::vector<SessionKvSource>> sources(2);
        sources[0].push_back(stream_source(2, lb0));
        sources[1].push_back(stream_source(5, lb1));
        std::vector<int64_t> pre(m0.live.ids.begin(), m0.live.ids.end());
        const uint64_t sk = sc.key_for(pre.data(), pre.size(), "");
        check(sc.store_streamed(sk, {m0, m1}, sources, error), "streamed: store two stages from their sources");
        check(sc.variants() == 1, "streamed: one variant");
        check(count_sess(sdir) == 2, "streamed: two session files on disk");
        ConversationPromptMatch smatch;
        check(sc.lookup(sk, smatch) && smatch.stages == 2, "streamed: two stages");
        SavedConversation back;
        check(sc.load(smatch.path, back, {}, error), "streamed: load");
        check(back.stage_images.size() == 1, "streamed: one later-stage image");
        check(back.live.ids == m0.live.ids, "streamed: ids round-trip");
        check(back.kv.size() == 1 && back.stage_images[0].kv.size() == 1, "streamed: one K/V layer per stage");
        std::vector<uint8_t> k0(4096);
        check(back.kv[0].k.read(k0.data(), 0, k0.size()), "streamed: read stage 0 K back");
        check(k0 == lb0[0], "streamed: stage 0's K round-trips byte for byte");
        std::vector<uint8_t> p1(9 * 128 * 4);
        check(back.stage_images[0].kv[0].pooled.read(p1.data(), 0, p1.size()), "streamed: read stage 1 pooled back");
        check(p1 == lb1[4], "streamed: stage 1's pooled row round-trips byte for byte");
        fs::remove_all(sdir);
    }

    fs::remove_all(dir);
    fs::remove_all(tiny_dir);
    std::printf("conversation_prompt_cache_test: %d checks passed\n", checks);
    return 0;
}
