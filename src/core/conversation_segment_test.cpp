// CPU-only tests for the per-segment tier (include/strata/core/conversation_segments.hpp): each phase of a turn is
// its own content-addressed file, the conversation is the ordered list of references (the manifest), identical
// segments deduplicate with a refcount, restore reads only the segments up to the chosen frontier, and the tier's GC
// never removes a shared segment. Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_segments.hpp"
#include "strata/core/conversation_spill.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
// A conversation image whose live ids are exactly `ids`: the slice a segment file carries.
SavedConversation conversation_with_ids(const std::vector<int32_t>& ids) {
    SavedConversation s = sample(ids.empty() ? 8 : ids.size());
    s.live.ids = ids;
    s.checkpoints.clear();
    ConversationCheckpoint c = checkpoint(ids.size(), 9);
    c.ids = ids;
    s.checkpoints.push_back(std::move(c));
    return s;
}
std::vector<int32_t> tokens(int32_t base, size_t n) {
    std::vector<int32_t> v;
    v.reserve(n);
    for (size_t i = 0; i < n; ++i) v.push_back(base + (int32_t) i);
    return v;
}
std::vector<int32_t> joined(std::initializer_list<std::vector<int32_t>> parts) {
    std::vector<int32_t> out;
    for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}
SegmentInput make_input(SegmentKind kind, std::vector<int32_t> ids, int64_t role = -1, uint16_t domain = 0) {
    SegmentInput in;
    in.kind = kind;
    in.tokens = std::move(ids);
    in.role_token = role;
    in.domain = domain;
    in.image = conversation_with_ids(in.tokens);
    return in;
}
size_t count_ext(const fs::path& dir, const std::string& ext) {
    size_t n = 0;
    if (!fs::exists(dir)) return 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (fs::is_regular_file(e.path()) && e.path().extension() == ext) ++n;
    return n;
}

// A whole-conversation image whose K/V part sizes are exact multiples of `cells` (the page-aligned row count), the
// relation the real engine has (conversation_snapshot.cpp: part = cells * heads * per). So a token-range slice
// `size * t / cells` is an integer row boundary and the slices tile the buffer exactly. `page_size` is a stand-in 64.
SavedConversation full_image(const std::vector<int32_t>& ids) {
    const int64_t cells = (int64_t) ((ids.size() + 63) / 64) * 64;
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48; s.cvec = false;
    s.live = checkpoint(ids.empty() ? 1 : ids.size(), 1);
    s.live.ids = ids;
    s.checkpoints.push_back(checkpoint((ids.empty() ? 1 : ids.size()) / 4, 2));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = cells; kv.heads = 2; kv.head_dim = 32;
        kv.page_size = 64; kv.pooled_rows = (int64_t) ids.size(); kv.idx_dim = 8;
        const size_t size = (size_t) cells * 64;
        kv.k = pattern(size, uint8_t(layer));
        kv.v = pattern(size, uint8_t(layer + 10));
        kv.k_scale = pattern(size, uint8_t(layer + 20));
        kv.v_scale = pattern(size, uint8_t(layer + 30));
        kv.pooled = pattern(size, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}
bool kv_equal(const SavedConversation& a, const SavedConversation& b) {
    if (a.kv.size() != b.kv.size()) return false;
    for (size_t l = 0; l < a.kv.size(); ++l) {
        const ConversationKv& x = a.kv[l]; const ConversationKv& y = b.kv[l];
        if (!(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-seg-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    const uint64_t model = id.model, config = id.config;
    const uint64_t head = segment_head_frontier(model, config);
    SegmentStoreConfig cfg;
    cfg.directory = dir;
    std::string error;

    // ---- 1. Content address: stable, sensitive to the precedent, unaffected by a later segment ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "1: open");
        const std::vector<int32_t> a = tokens(100, 20);
        const uint64_t h1 = segment_hash(SegmentKind::Sys, 0, a, -1, head, model, config);
        const uint64_t h2 = segment_hash(SegmentKind::Sys, 0, a, -1, head, model, config);
        check(h1 == h2, "1: the address is stable");
        check(h1 == segment_hash(SegmentKind::Sys, 0, a, -1, head, model, config), "1: and deterministic");
        check(h1 != segment_hash(SegmentKind::Sys, 0, a, -1, head + 1, model, config),
              "1: same tokens, different precedent -> different address");
        check(h1 != segment_hash(SegmentKind::Read, 0, a, -1, head, model, config),
              "1: different kind -> different address");
        check(h1 != segment_hash(SegmentKind::Sys, 1, a, -1, head, model, config),
              "1: different domain -> different address");
        check(h1 != segment_hash(SegmentKind::Sys, 0, a, 7, head, model, config),
              "1: different role token -> different address");
        // A later segment does not touch an earlier address: sys's address is the same in two chats with different
        // tails (checked at the store level in 3/4).
        const uint64_t later = segment_hash(SegmentKind::Answer, 0, tokens(200, 5), -1, h1, model, config);
        check(later != h1, "1: a later segment is its own address");
    }

    // ---- 2. Write/read a segment: the manifest exists, the segment file is a session file of its slice ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "2: open");
        const std::vector<int32_t> sys_ids = tokens(1000, 32);
        const std::vector<int32_t> read_ids = tokens(2000, 24);
        std::string manifest_path;
        check(store.store(501, -1, false, {make_input(SegmentKind::Sys, sys_ids),
                                           make_input(SegmentKind::Read, read_ids)}, error, &manifest_path),
              "2: store sys+read");
        check(fs::exists(manifest_path), "2: the manifest exists");
        check(store.size() == 1, "2: one manifest");
        check(store.segment_count() == 2, "2: two segment files");
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        const std::string sys_file = SegmentStore::segment_path(dir.string(), sys_id, 0);
        check(fs::exists(sys_file), "2: <seg_id>.seg exists");
        SavedConversation back;
        size_t bytes = 0;
        check(session_file_read(sys_file, id, back, bytes, error), "2: the segment file reads as a session file");
        check(back.live.ids == sys_ids, "2: the segment file holds its own slice");
        check(back.kv.size() == 2, "2: its K/V layers round-trip");
    }

    // ---- 3. Dedup: two conversations sharing the systemprompt keep ONE file, refcount 2 ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "3: open");
        const std::vector<int32_t> sys_ids = tokens(3000, 40);
        const std::vector<int32_t> tail_a = tokens(4000, 30);
        const std::vector<int32_t> tail_b = tokens(5000, 30);
        check(store.store(1, -1, false, {make_input(SegmentKind::Sys, sys_ids),
                                         make_input(SegmentKind::Answer, tail_a)}, error), "3: store chat A");
        check(store.store(2, -1, false, {make_input(SegmentKind::Sys, sys_ids),
                                         make_input(SegmentKind::Answer, tail_b)}, error), "3: store chat B");
        check(store.size() == 2, "3: two manifests");
        check(count_ext(dir, ".seg") == 3, "3: three segment files (sys shared, two tails)");
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        check(store.refcount(sys_id) == 2, "3: the shared systemprompt has refcount 2");
        const uint64_t tail_a_id = segment_hash(SegmentKind::Answer, 0, tail_a, -1, sys_id, model, config);
        check(store.refcount(tail_a_id) == 1, "3: a tail has refcount 1");
        check(fs::exists(SegmentStore::segment_path(dir.string(), sys_id, 0)), "3: one file for the shared sys");
    }

    // ---- 4. No dedup when the precedent differs: same tokens after a different sys -> two files ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "4: open");
        const std::vector<int32_t> sys_x = tokens(6000, 20);
        const std::vector<int32_t> sys_y = tokens(7000, 20);
        const std::vector<int32_t> block = tokens(8000, 15);   // identical tokens in both chats
        check(store.store(1, -1, false, {make_input(SegmentKind::Sys, sys_x),
                                         make_input(SegmentKind::Read, block)}, error), "4: store chat X");
        check(store.store(2, -1, false, {make_input(SegmentKind::Sys, sys_y),
                                         make_input(SegmentKind::Read, block)}, error), "4: store chat Y");
        check(count_ext(dir, ".seg") == 4, "4: four segment files, the block is not shared");
        const uint64_t sys_x_id = segment_hash(SegmentKind::Sys, 0, sys_x, -1, head, model, config);
        const uint64_t sys_y_id = segment_hash(SegmentKind::Sys, 0, sys_y, -1, head, model, config);
        const uint64_t block_x = segment_hash(SegmentKind::Read, 0, block, -1, sys_x_id, model, config);
        const uint64_t block_y = segment_hash(SegmentKind::Read, 0, block, -1, sys_y_id, model, config);
        check(block_x != block_y, "4: same tokens, different precedent -> different address");
        check(store.refcount(block_x) == 1 && store.refcount(block_y) == 1, "4: both blocks refcount 1");
    }

    // ---- 5. Restore a single segment: a deleted tail does not stop the earlier frontier, and is never opened ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "5: open");
        const std::vector<int32_t> sys_ids = tokens(10000, 30);
        const std::vector<int32_t> pre_ids = tokens(11000, 20);
        const std::vector<int32_t> answer_ids = tokens(12000, 25);
        check(store.store(77, -1, false, {make_input(SegmentKind::Sys, sys_ids),
                                          make_input(SegmentKind::Other, pre_ids),
                                          make_input(SegmentKind::Answer, answer_ids)}, error), "5: store");
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        const uint64_t pre_id = segment_hash(SegmentKind::Other, 0, pre_ids, -1, sys_id, model, config);
        const uint64_t answer_id = segment_hash(SegmentKind::Answer, 0, answer_ids, -1, pre_id, model, config);
        check(fs::remove(SegmentStore::segment_path(dir.string(), answer_id, 0)) == 1, "5: the answer file is gone");
        std::vector<int32_t> extend_sys = sys_ids;
        extend_sys.push_back(9999);
        const SegmentMatch m = store.best(extend_sys, false, 0.0, 0);
        check(bool(m), "5: a prompt extending sys matches");
        check(m.tokens == (int64_t) sys_ids.size() && m.index == 0, "5: the frontier is sys");
        SavedConversation back;
        check(store.load(m, back, {}, error), "5: load reads only sys and succeeds");
        check(back.live.ids == sys_ids, "5: the sum is exactly sys");
        // The full prompt: the missing answer file truncates the run at pre, not at the deleted segment.
        const std::vector<int32_t> full = joined({sys_ids, pre_ids, answer_ids, {1234}});
        const SegmentMatch m2 = store.best(full, false, 0.0, 0);
        check(m2.tokens == (int64_t) (sys_ids.size() + pre_ids.size()),
              "5: the run stops before the missing segment");
    }

    // ---- 6. A hole in the middle: the run degrades to the longest valid run, the manifest is kept ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "6: open");
        const std::vector<int32_t> sys_ids = tokens(13000, 30);
        const std::vector<int32_t> pre_ids = tokens(14000, 20);
        const std::vector<int32_t> think_ids = tokens(15000, 15);
        const std::vector<int32_t> answer_ids = tokens(16000, 18);
        check(store.store(88, -1, false, {make_input(SegmentKind::Sys, sys_ids),
                                          make_input(SegmentKind::Other, pre_ids),
                                          make_input(SegmentKind::Think, think_ids),
                                          make_input(SegmentKind::Answer, answer_ids)}, error), "6: store");
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        const uint64_t pre_id = segment_hash(SegmentKind::Other, 0, pre_ids, -1, sys_id, model, config);
        check(fs::remove(SegmentStore::segment_path(dir.string(), pre_id, 0)) == 1, "6: the middle file is gone");
        const std::vector<int32_t> full = joined({sys_ids, pre_ids, think_ids, answer_ids});
        const SegmentMatch m = store.best(full, false, 0.0, 0);
        check(bool(m) && m.tokens == (int64_t) sys_ids.size(), "6: the run degrades to the longest valid run (sys)");
        SavedConversation back;
        check(store.load(m, back, {}, error), "6: load of the degraded frontier succeeds");
        check(back.live.ids == sys_ids, "6: and reads only sys");
        // Reopen: the manifest is still there and nothing was deleted by the scan.
        SegmentStore again;
        check(again.open(cfg, id, error), "6: reopen");
        check(again.size() == 1, "6: the manifest survives a reopen");
        check(again.stale_files_kept() >= 1, "6: a manifest with a missing segment is reported stale");
        const SegmentMatch m3 = again.best(full, false, 0.0, 0);
        check(bool(m3) && m3.tokens == (int64_t) sys_ids.size(), "6: still degrades after a reopen");
    }

    // ---- 7. Compatibility: a .sess + .meta v2 with no manifest is left alone and read as before ----
    fs::remove_all(dir);
    {
        ConversationSpillCache tier;
        check(tier.open(dir, id, 1ull << 30, error), "7: open the .sess tier");
        check(tier.spill(sample(500), error), "7: spill a conversation as .sess + .meta");
        check(fs::exists(dir / "strata-conv-1.sess") && fs::exists(dir / "strata-conv-1.meta"),
              "7: the compat files are on disk");
        SegmentStore store;
        check(store.open(cfg, id, error), "7: open the segment store beside them");
        check(store.size() == 0, "7: no manifests indexed");
        check(store.segment_count() == 0, "7: no segments indexed");
        check(store.stale_files_kept() == 0, "7: the .sess/.meta files are not counted stale");
        check(fs::exists(dir / "strata-conv-1.sess") && fs::exists(dir / "strata-conv-1.meta"),
              "7: the compat files are untouched");
        ConversationSpillCache again;
        check(again.open(dir, id, 1ull << 30, error), "7: the .sess tier reopens");
        check(again.size() == 1, "7: the .sess compat path still reads it");
    }

    // ---- 8. Budget/GC by refcount: refcount-1 segments go, the shared sys stays, erase decrements ----
    fs::remove_all(dir);
    {
        const std::vector<int32_t> sys_ids = tokens(17000, 40);
        const std::vector<int32_t> tail_a = tokens(18000, 30);
        const std::vector<int32_t> tail_b = tokens(19000, 30);
        uint64_t two_chats = 0;
        {
            SegmentStore seed;
            check(seed.open(cfg, id, error), "8: open without a budget");
            check(seed.store(1, -1, false, {make_input(SegmentKind::Sys, sys_ids),
                                            make_input(SegmentKind::Answer, tail_a)}, error), "8: seed chat A");
            check(seed.store(2, -1, false, {make_input(SegmentKind::Sys, sys_ids),
                                            make_input(SegmentKind::Answer, tail_b)}, error), "8: seed chat B");
            two_chats = seed.bytes();
            check(two_chats > 0, "8: two chats have bytes");
        }
        SegmentStoreConfig tight = cfg;
        tight.budget_bytes = two_chats - 1;   // room for everything but one segment
        SegmentStore store;
        check(store.open(tight, id, error), "8: open just under two chats");
        check(store.size() == 2, "8: both manifests indexed");
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        check(store.refcount(sys_id) == 2, "8: the shared sys has refcount 2");
        check(store.segment_evictions() >= 1, "8: the GC evicted a refcount-1 segment");
        check(store.bytes() <= tight.budget_bytes, "8: the segment bytes are under the budget");
        check(fs::exists(SegmentStore::segment_path(dir.string(), sys_id, 0)), "8: the shared sys is not evicted");
        const std::string manifest_a = (dir / "strata-conv-1.segs").string();
        const std::string manifest_b = (dir / "strata-conv-2.segs").string();
        check(fs::exists(manifest_a) && fs::exists(manifest_b), "8: both manifests are on disk");
        check(store.erase(manifest_a, error), "8: erase chat A");
        check(store.refcount(sys_id) == 1, "8: refcount drops to 1");
        check(fs::exists(SegmentStore::segment_path(dir.string(), sys_id, 0)),
              "8: the shared file survives one erase");
        check(store.erase(manifest_b, error), "8: erase chat B");
        check(store.refcount(sys_id) == 0, "8: refcount reaches 0");
        check(!fs::exists(SegmentStore::segment_path(dir.string(), sys_id, 0)),
              "8: the file goes when the last reference goes");
    }

    // ---- 9. A corrupt manifest and a foreign identity are counted and never deleted ----
    fs::remove_all(dir);
    {
        fs::create_directories(dir);
        { std::ofstream out(dir / "strata-conv-99.segs", std::ios::binary); out << "not a manifest"; }
        SegmentStore store;
        check(store.open(cfg, id, error), "9: open with a corrupt manifest");
        check(store.size() == 0, "9: the corrupt manifest is ignored");
        check(store.stale_files_kept() >= 1, "9: it is counted stale");
        check(fs::exists(dir / "strata-conv-99.segs"), "9: and left in place");
        // a valid manifest of ANOTHER identity
        const fs::path other_dir = dir / "other";
        SegmentStore writer;
        check(writer.open({other_dir}, SessionFileIdentity{0xAAAA, 0xBBBB}, error), "9: open a foreign writer");
        check(writer.store(1, -1, false, {make_input(SegmentKind::Sys, tokens(20000, 12))}, error),
              "9: write a foreign manifest");
        std::string foreign_manifest;
        for (const auto& e : fs::directory_iterator(other_dir))
            if (e.path().extension() == ".segs") foreign_manifest = e.path().string();
        check(!foreign_manifest.empty(), "9: the foreign manifest exists");
        fs::copy_file(foreign_manifest, dir / "strata-conv-100.segs");
        SegmentStore reopened;
        check(reopened.open(cfg, id, error), "9: reopen with the foreign manifest beside it");
        check(reopened.foreign_files_kept() >= 1, "9: it is counted foreign");
        check(fs::exists(dir / "strata-conv-100.segs"), "9: and left in place");
    }

    // ---- 10. Layer-split: a segment is one file per stage and the manifest records the stage count ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "10: open");
        const std::vector<int32_t> sys_ids = tokens(21000, 30);
        SegmentInput input = make_input(SegmentKind::Sys, sys_ids);
        SavedConversation stage = conversation_with_ids(tokens(22000, 10));
        stage.layer_lo = 48; stage.layer_hi = 96;
        input.stage_images.push_back(stage);
        check(store.store(1, -1, false, {input}, error), "10: store a two-stage segment");
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        check(fs::exists(SegmentStore::segment_path(dir.string(), sys_id, 0)), "10: stage 0 file written");
        check(fs::exists(SegmentStore::segment_path(dir.string(), sys_id, 1)), "10: stage 1 file written");
        std::vector<int32_t> extend = sys_ids;
        extend.push_back(7);
        const SegmentMatch m = store.best(extend, false, 0.0, 0);
        check(bool(m), "10: the split segment matches");
        SavedConversation back;
        check(store.load(m, back, {}, error), "10: load reads both stages");
        check(back.stage_images.size() == 1, "10: the stage image is read back");
        check(back.stage_images[0].layer_lo == 48 && back.stage_images[0].layer_hi == 96,
              "10: the stage layer range round-trips");
        SegmentStore again;
        check(again.open(cfg, id, error), "10: reopen the split directory");
        check(again.segment_count() == 1, "10: the stages are one segment");
        const SegmentMatch rematch = again.best(extend, false, 0.0, 0);
        check(bool(rematch) && rematch.live, "10: the stage count survives a reopen");
    }

    // ---- 11. The thinking-effort case: rojo-verde. The frontier is len(sys)+len(pre); read_tokens is the effort
    // and the answer alone and does not grow with the prefix. ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "11: open");
        const std::vector<int32_t> sys_ids = tokens(30000, 64);
        const std::vector<int32_t> pre_small = tokens(31000, 40);
        const std::vector<int32_t> pre_big = tokens(33000, 400);
        const std::vector<int32_t> effort_high = tokens(35000, 12);   // the non-default effort turn (xhigh)
        const std::vector<int32_t> effort_low = tokens(36000, 12);    // low: the same length, other tokens
        const std::vector<int32_t> answer_high = tokens(37000, 20);
        const std::vector<int32_t> answer_low = tokens(38000, 20);
        const int64_t tail_role = 32000;                               // --tail-role-token
        check(store.store(1, tail_role, false,
                          {make_input(SegmentKind::Sys, sys_ids),
                           make_input(SegmentKind::Other, pre_small),
                           make_input(SegmentKind::Think, effort_high, tail_role),
                           make_input(SegmentKind::Answer, answer_high)}, error), "11: store the small-prefix chat");
        check(store.store(2, tail_role, false,
                          {make_input(SegmentKind::Sys, sys_ids),
                           make_input(SegmentKind::Other, pre_big),
                           make_input(SegmentKind::Think, effort_high, tail_role),
                           make_input(SegmentKind::Answer, answer_high)}, error), "11: store the big-prefix chat");
        const std::vector<int32_t> high_small = joined({sys_ids, pre_small, effort_high, answer_high});
        const std::vector<int32_t> low_small = joined({sys_ids, pre_small, effort_low, answer_low});
        const std::vector<int32_t> low_big = joined({sys_ids, pre_big, effort_low, answer_low});
        // Control (verde): the stored prompt still matches its whole live.
        const SegmentMatch full = store.best(high_small, false, 0.0, 0);
        check(bool(full) && full.live && full.tokens == (int64_t) high_small.size(),
              "11: the stored prompt matches its whole live");
        // The effort changed: the frontier is the prefix before it, and read_tokens excludes the prefix.
        const SegmentMatch small = store.best(low_small, false, 0.0, 0);
        check(bool(small), "11: P_low matches");
        check(small.tokens == (int64_t) (sys_ids.size() + pre_small.size()),
              "11: the frontier is len(sys)+len(pre)");
        check(!small.live, "11: the frontier is NOT the whole live");
        const int64_t read_small = (int64_t) low_small.size() - small.tokens;
        check(read_small == (int64_t) (effort_high.size() + answer_high.size()),
              "11: read_tokens == len(effort) + len(answer)");
        const SegmentMatch big = store.best(low_big, false, 0.0, 0);
        check(bool(big) && big.tokens == (int64_t) (sys_ids.size() + pre_big.size()),
              "11: with a big prefix the frontier is still len(sys)+len(pre)");
        const int64_t read_big = (int64_t) low_big.size() - big.tokens;
        check(read_big == read_small, "11: read_tokens does NOT grow with the prefix (rojo-verde)");
        // Rojo: if best matched the whole live (the design reverted), read_tokens would be 0 or the whole prompt.
        check(read_small > 0 && read_small < (int64_t) low_small.size(),
              "11: read_tokens is strictly between 0 and the whole prompt");
        // The systemprompt is shared: two chats, one sys file.
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        check(store.refcount(sys_id) == 2, "11: the systemprompt is shared (refcount 2)");
    }

    // ---- 12. store_image derives the segments from ONE captured image: the sum reconstructs it exactly, a rewrite
    // is a no-op, and the manifest is replaced in place (one per conversation) ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "12: open");
        const std::vector<int32_t> sys_ids = tokens(40000, 40);
        const std::vector<int32_t> pre_ids = tokens(41000, 30);
        const std::vector<int32_t> ans_ids = tokens(42000, 20);
        const int32_t turn = 90001, tail = 90002;
        const std::vector<int32_t> full = joined({sys_ids, {turn}, pre_ids, {tail}, ans_ids});
        const SavedConversation image = full_image(full);
        const std::vector<SegmentPlan> plan = segment_plan_split(full, turn, tail);
        check(plan.size() == 3, "12: the plan splits sys/pre/answer");
        check(plan[0].kind == SegmentKind::Sys && plan[1].kind == SegmentKind::Other &&
              plan[2].kind == SegmentKind::Answer, "12: the plan kinds");
        check(plan[0].tokens + plan[1].tokens + plan[2].tokens == full.size(), "12: the plan covers the prompt");
        std::string manifest;
        check(store.store_image(77, turn, false, image, plan, error, &manifest), "12: store_image");
        check(fs::exists(manifest), "12: the manifest exists");
        check(count_ext(dir, ".seg") == 3, "12: three segment files (one per phase)");
        const SegmentMatch m = store.best(full, false, 0.0, 0);
        check(bool(m) && m.live && m.tokens == (int64_t) full.size(), "12: the stored prompt matches the whole live");
        SavedConversation back;
        check(store.load(m, back, {}, error), "12: load the whole chain");
        check(back.live.ids == full, "12: the reconstructed ids equal the prompt");
        check(kv_equal(back, image), "12: the reconstructed K/V equals the original image, byte for byte");
        check(back.checkpoints.size() == image.checkpoints.size(), "12: the running state survives the sum");
        // Idempotent + one manifest per conversation: writing the same chain again adds no file and reuses the path.
        const size_t files_before = count_ext(dir, ".seg");
        std::string manifest2;
        check(store.store_image(77, turn, false, image, plan, error, &manifest2), "12: store_image again");
        check(manifest2 == manifest, "12: the manifest is replaced in place (one per conversation)");
        check(count_ext(dir, ".seg") == files_before, "12: no new segment file (idempotent)");
        check(store.size() == 1, "12: still one manifest");
    }

    // ---- 13. Invariant A (thinking effort): changing the effort rereads ONLY the effort+answer suffix, and read
    // does not grow with the prefix. Red: without the cut at the effort turn the criterion discriminates. ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "13: open");
        const std::vector<int32_t> sys_ids = tokens(50000, 64);
        const std::vector<int32_t> pre_small = tokens(51000, 40);
        const std::vector<int32_t> pre_big = tokens(53000, 400);
        const std::vector<int32_t> effort_high = tokens(55000, 12);
        const std::vector<int32_t> effort_low = tokens(56000, 12);   // same length, other tokens
        const std::vector<int32_t> ans_high = tokens(57000, 20);
        const std::vector<int32_t> ans_low = tokens(58000, 20);
        const int32_t turn = 90003, tail = 90004;
        auto prompt = [&](const std::vector<int32_t>& pre, std::vector<int32_t> effort,
                          const std::vector<int32_t>& ans) {
            effort[0] = tail;   // --tail-role-token opens the effort turn: it is the effort turn's first token
            return joined({sys_ids, {turn}, pre, effort, ans});
        };
        const std::vector<int32_t> p_high = prompt(pre_small, effort_high, ans_high);
        const std::vector<int32_t> p_low_small = prompt(pre_small, effort_low, ans_low);
        const std::vector<int32_t> p_high_big = prompt(pre_big, effort_high, ans_high);
        const std::vector<int32_t> p_low_big = prompt(pre_big, effort_low, ans_low);
        check(store.store_image(1, turn, false, full_image(p_high), segment_plan_split(p_high, turn, tail), error),
              "13: store the small-prefix chat (effort xhigh)");
        check(store.store_image(2, turn, false, full_image(p_high_big),
                                segment_plan_split(p_high_big, turn, tail), error), "13: store the big-prefix chat");
        const std::vector<SegmentPlan> plan_high = segment_plan_split(p_high, turn, tail);
        const int64_t frontier = (int64_t) (plan_high[0].tokens + plan_high[1].tokens);
        const SegmentMatch full = store.best(p_high, false, 0.0, 0);
        check(bool(full) && full.live, "13: the stored prompt matches its whole live (control, verde)");
        const SegmentMatch small = store.best(p_low_small, false, 0.0, 0);
        check(bool(small), "13: P_low (small prefix) matches");
        check(small.tokens == frontier, "13: the frontier is len(sys)+len(pre)");
        check(!small.live, "13: the frontier is NOT the whole live");
        const int64_t read_small = (int64_t) p_low_small.size() - small.tokens;
        check(read_small == (int64_t) (effort_low.size() + ans_low.size()),
              "13: read_tokens == len(effort) + len(answer)");
        // Only the segments 0..frontier are read: the answer segment is never opened.
        SavedConversation back;
        check(store.load(small, back, {}, error), "13: load the effort frontier");
        check(back.live.ids == joined({sys_ids, {turn}, pre_small}), "13: the sum is exactly sys+pre (no effort)");
        const SegmentMatch big = store.best(p_low_big, false, 0.0, 0);
        check(bool(big) && big.tokens == (int64_t) (sys_ids.size() + 1 + pre_big.size()),
              "13: with a big prefix the frontier is still len(sys)+len(pre)");
        const int64_t read_big = (int64_t) p_low_big.size() - big.tokens;
        check(read_big == read_small, "13: read_tokens does NOT grow with the prefix (rojo-verde)");
        check(read_small > 0 && read_small < (int64_t) p_low_small.size(), "13: read_tokens is strictly partial");
        // RED: a design without the effort cut (one whole-live segment) cannot offer a frontier for the changed tail.
        SegmentStoreConfig redcfg = cfg;
        redcfg.directory = dir / "red";
        SegmentStore red;
        check(red.open(redcfg, id, error), "13: open the reverted store (no effort cut)");
        const std::vector<SegmentPlan> whole{{SegmentKind::Other, 0, (uint64_t) p_high.size(), -1, true}};
        check(red.store_image(1, turn, false, full_image(p_high), whole, error), "13: store one whole segment");
        const SegmentMatch red_match = red.best(p_low_small, false, 0.0, 0);
        const int64_t red_read = red_match ? (int64_t) p_low_small.size() - red_match.tokens
                                           : (int64_t) p_low_small.size();
        check(red_read != read_small, "13: RED - without the effort cut read_tokens differs (the criterion discriminates)");
    }

    // ---- 14. Invariant B (systemprompt): two variants coexist; a revisiting is a HIT (no re-prefill) and neither
    // variant is deleted by the other ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "14: open");
        const std::vector<int32_t> sys_a = tokens(60000, 50);
        const std::vector<int32_t> sys_b = tokens(61000, 50);   // variant B: same length, other tokens
        const std::vector<int32_t> tail = tokens(62000, 25);    // the SAME tail in both
        const int32_t turn = 90005, tailrole = 90006;
        const std::vector<int32_t> pa = joined({sys_a, {turn}, tail, {tailrole}, tokens(63000, 10)});
        const std::vector<int32_t> pb = joined({sys_b, {turn}, tail, {tailrole}, tokens(64000, 10)});
        check(store.store_image(11, turn, false, full_image(pa), segment_plan_split(pa, turn, tailrole), error),
              "14: store variant A");
        check(store.store_image(12, turn, false, full_image(pb), segment_plan_split(pb, turn, tailrole), error),
              "14: store variant B");
        check(store.size() == 2, "14: two manifests coexist (one per variant)");
        const SegmentMatch ma = store.best(pa, false, 0.0, 0);
        const SegmentMatch mb = store.best(pb, false, 0.0, 0);
        check(bool(ma) && ma.live && ma.tokens == (int64_t) pa.size(), "14: variant A is a HIT (whole live)");
        check(bool(mb) && mb.live && mb.tokens == (int64_t) pb.size(), "14: variant B is a HIT (whole live)");
        check(ma.manifest_path != mb.manifest_path, "14: the variants live in different manifests");
        const uint64_t sys_a_id = segment_hash(SegmentKind::Sys, 0, sys_a, -1, head, model, config);
        const uint64_t sys_b_id = segment_hash(SegmentKind::Sys, 0, sys_b, -1, head, model, config);
        check(sys_a_id != sys_b_id, "14: the two systemprompts have different addresses (no false sharing)");
        // Re-storing variant A does not delete variant B, and adds no file.
        const size_t files = count_ext(dir, ".seg");
        check(store.store_image(11, turn, false, full_image(pa), segment_plan_split(pa, turn, tailrole), error),
              "14: re-store variant A (revisiting)");
        check(store.size() == 2, "14: re-storing A keeps B (neither is deleted)");
        check(count_ext(dir, ".seg") == files, "14: re-storing A adds no file (idempotent)");
        check(fs::exists(SegmentStore::segment_path(dir.string(), sys_a_id, 0)),
              "14: variant A's systemprompt is still on disk");
        check(fs::exists(SegmentStore::segment_path(dir.string(), sys_b_id, 0)),
              "14: variant B's systemprompt is still on disk");
        const SegmentMatch ma2 = store.best(pa, false, 0.0, 0);
        check(bool(ma2) && ma2.live, "14: variant A is STILL a HIT after B was stored (no re-prefill)");
        // RED: the shared tail is NOT reused across the two systemprompts (the precedent differs).
        check(segment_hash(SegmentKind::Answer, 0, tail, tailrole, sys_a_id, model, config) !=
              segment_hash(SegmentKind::Answer, 0, tail, tailrole, sys_b_id, model, config),
              "14: RED - the same tail under a different systemprompt is a different segment");
    }

    // ---- 15. Incremental: a later park of the same chat does not rewrite the prefix segments ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "15: open");
        const std::vector<int32_t> sys_ids = tokens(70000, 50);
        const std::vector<int32_t> pre = tokens(71000, 20);
        const int32_t turn = 90007, tail = 90008;
        const std::vector<int32_t> p1 = joined({sys_ids, {turn}, pre, {tail}, tokens(72000, 10)});
        check(store.store_image(21, turn, false, full_image(p1), segment_plan_split(p1, turn, tail), error),
              "15: store p1");
        check(count_ext(dir, ".seg") == 3, "15: p1 has three segments");
        const uint64_t sys_id = segment_hash(SegmentKind::Sys, 0, sys_ids, -1, head, model, config);
        const std::string sys_path = SegmentStore::segment_path(dir.string(), sys_id, 0);
        const auto sys_stamp = fs::last_write_time(sys_path);
        // p2 extends the answer (the same sys and pre): only the answer segment changes.
        const std::vector<int32_t> p2 = joined({sys_ids, {turn}, pre, {tail}, tokens(72000, 10), tokens(73000, 15)});
        check(store.store_image(21, turn, false, full_image(p2), segment_plan_split(p2, turn, tail), error),
              "15: store p2 (the same chat, a longer answer)");
        check(fs::last_write_time(sys_path) == sys_stamp, "15: the systemprompt segment is NOT rewritten");
        check(count_ext(dir, ".seg") == 3, "15: still three segment files (the superseded answer went)");
        check(store.size() == 1, "15: still one manifest for the conversation");
        const SegmentMatch m2 = store.best(p2, false, 0.0, 0);
        check(bool(m2) && m2.live && m2.tokens == (int64_t) p2.size(), "15: p2 matches its whole live");
    }

    // ---- 16. --conversation-cache-segment-min-tokens fuses a short non-system run in store_image too ----
    fs::remove_all(dir);
    {
        SegmentStoreConfig fuse_cfg = cfg;
        fuse_cfg.min_tokens = 100;   // a non-system run shorter than 100 tokens fuses into the one before it
        SegmentStore store;
        check(store.open(fuse_cfg, id, error), "16: open with min-tokens 100");
        const std::vector<int32_t> sys_ids = tokens(80000, 50);
        const int32_t turn = 90009, tail = 90010;
        const std::vector<int32_t> pre = tokens(81000, 20);
        std::vector<int32_t> effort = tokens(82000, 12); effort[0] = tail;
        const std::vector<int32_t> ans = tokens(83000, 15);
        const std::vector<int32_t> full = joined({sys_ids, {turn}, pre, effort, ans});
        check(store.store_image(31, turn, false, full_image(full), segment_plan_split(full, turn, tail), error),
              "16: store_image with min-tokens");
        // The systemprompt is never a fusion partner: [sys] stays, and the short prefix and answer fuse -> 2 files.
        check(count_ext(dir, ".seg") == 2, "16: the short runs fused (two files, not three)");
        const SegmentMatch m = store.best(full, false, 0.0, 0);
        check(bool(m) && m.live && m.tokens == (int64_t) full.size(), "16: the fused chain still matches its whole live");
    }

    // ---- 17. The frontier segment carries the recurrent snapshot at its own boundary (the effort checkpoint) ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "17: open");
        const std::vector<int32_t> sys_ids = tokens(84000, 40);
        const std::vector<int32_t> pre = tokens(85000, 20);
        const int32_t turn = 90011, tail = 90012;
        std::vector<int32_t> effort = tokens(86000, 12); effort[0] = tail;
        const std::vector<int32_t> ans = tokens(87000, 15);
        const std::vector<int32_t> full = joined({sys_ids, {turn}, pre, effort, ans});
        SavedConversation img = full_image(full);
        const size_t frontier = sys_ids.size() + 1 + pre.size();
        // the engine checkpoints in front of the effort turn: a snapshot exactly at the frontier
        ConversationCheckpoint ck = checkpoint(frontier, 77);
        ck.ids.assign(full.begin(), full.begin() + (std::ptrdiff_t) frontier);
        img.checkpoints.push_back(ck);
        check(store.store_image(41, turn, false, img, segment_plan_split(full, turn, tail), error), "17: store_image");
        // A changed effort: the frontier is the pre segment, whose state must be the checkpoint's, not empty.
        std::vector<int32_t> low = effort; low[1] ^= 7;
        const std::vector<int32_t> p_low = joined({sys_ids, {turn}, pre, low, ans});
        const SegmentMatch m = store.best(p_low, false, 0.0, 0);
        check(bool(m) && m.tokens == (int64_t) frontier, "17: the frontier is the pre segment");
        SavedConversation back;
        check(store.load(m, back, {}, error), "17: load the effort frontier");
        check(back.live.ids.size() == frontier, "17: the frontier ids are sys+pre");
        check(back.live.gdn == ck.gdn && back.live.tails == ck.tails,
              "17: the recurrent state at the frontier is the checkpoint's (not empty)");
    }

    // ---- 18. layer/ladder-t3: the effort frontier is anchored at the <|im_start|> that OPENS the turn (where the
    // engine checkpoints), not at the role token one past it.  MEASURED on the pack's real tokenizer: <|im_start|>
    // 248045 opens the turn, its role "system" (--tail-role-token 8678) sits at opener + 1.  The OLD frontier (the
    // role token) is not an engine checkpoint, so that frontier's has_state never travelled and the segment chain
    // broke exactly at the effort turn. ----
    fs::remove_all(dir);
    {
        SegmentStore store;
        check(store.open(cfg, id, error), "18: open");
        const int32_t im_start = 248045, im_end = 248046, role_sys = 8678;   // the measured pack ids
        const int32_t role_user = 248052, role_ast = 248053;
        const std::vector<int32_t> sys_body = tokens(100000, 40);
        const std::vector<int32_t> user_body = tokens(101000, 12);
        const std::vector<int32_t> ast_body = tokens(102000, 10);
        const std::vector<int32_t> effort_body = tokens(103000, 8);
        // The template as the server renders it: <|im_start|>, then the role token, then the body, then <|im_end|>.
        const std::vector<int32_t> prompt = joined({
            {im_start}, {role_sys}, sys_body, {im_end},
            {im_start}, {role_user}, user_body, {im_end},
            {im_start}, {role_ast}, ast_body, {im_end},
            {im_start}, {role_sys}, effort_body, {im_end},
            {im_start}, {role_ast}});
        // The engine's own positions (generate.cpp:12704-12737): turn_at is the last <|im_start|>, pulled back (#458)
        // to the one whose NEXT token is --tail-role-token; root_at is the first <|im_start|> at index >= 1 (the end
        // of the system prompt).  This prompt starts with a <|im_start|> at 0 (index 0 is skipped), so root_at is the
        // first non-system turn's opener.
        int64_t turn_at = -1;
        for (int64_t i = (int64_t) prompt.size() - 1; i > 0; --i)
            if (prompt[(size_t) i] == im_start) { turn_at = i; break; }
        if (turn_at > 0)
            for (int64_t i = turn_at - 1; i > 0; --i)
                if (prompt[(size_t) i] == im_start) {
                    if (prompt[(size_t) i + 1] == role_sys) turn_at = i;
                    break;
                }
        int64_t root_at = -1;
        for (size_t i = 1; i < prompt.size(); ++i)
            if (prompt[i] == im_start) { root_at = (int64_t) i; break; }
        int64_t role_pos = -1;
        for (int64_t i = (int64_t) prompt.size() - 1; i >= 0; --i)
            if (prompt[(size_t) i] == role_sys) { role_pos = i; break; }
        const std::vector<int64_t> engine = {turn_at, root_at, (int64_t) prompt.size() - 1};
        const auto is_checkpoint = [&](int64_t p) {
            for (int64_t c : engine) if (c == p) return true;
            return false;
        };
        check(role_pos == turn_at + 1, "18: the MEASURED off-by-one (the role token is one past the opener)");

        const std::vector<SegmentPlan> plan = segment_plan_split(prompt, im_start, role_sys);
        std::vector<int64_t> bounds;
        int64_t at = 0;
        for (const SegmentPlan& s : plan) { at += (int64_t) s.tokens; bounds.push_back(at); }
        check(bounds.back() == (int64_t) prompt.size(), "18: the plan covers the prompt");
        check(bounds.size() == 2 && bounds[0] == turn_at,
              "18: GREEN - the effort frontier is the <|im_start|> that opens the turn (= the engine's turn_at)");
        // RED: the OLD rule (cut at the role token) produced a frontier that is NOT a checkpoint, so its has_state
        // could not travel - the defect measured live.
        check(role_pos == turn_at + 1 && !is_checkpoint(role_pos),
              "18: RED - the old frontier (the role token) is not an engine checkpoint");
        check(bounds[0] == turn_at && is_checkpoint(bounds[0]), "18: GREEN - the anchored frontier IS that checkpoint");
        // The full invariant: EVERY non-final plan boundary lands on an engine checkpoint, not only the effort one.
        for (size_t k = 0; k + 1 < bounds.size(); ++k)
            check(is_checkpoint(bounds[k]), "18: every non-final plan boundary is an engine checkpoint");

        // has_state proof: with a checkpoint exactly at turn_at, the frontier segment carries its running state.
        SavedConversation img = full_image(prompt);
        img.checkpoints.clear();
        ConversationCheckpoint ck = checkpoint((size_t) turn_at, 77);
        ck.ids.assign(prompt.begin(), prompt.begin() + (std::ptrdiff_t) turn_at);
        img.checkpoints.push_back(ck);
        check(store.store_image(51, im_start, false, img, plan, error), "18: store_image with the anchored plan");
        std::vector<int32_t> changed = prompt;
        changed[(size_t) turn_at + 2] ^= 7;   // change the effort body: the tail rewrites, the frontier must not
        const SegmentMatch m = store.best(changed, false, 0.0, 0);
        check(bool(m) && m.tokens == turn_at, "18: a changed effort finds the frontier exactly at turn_at");
        SavedConversation back;
        check(store.load(m, back, {}, error), "18: load the effort frontier");
        check(back.live.ids.size() == (size_t) turn_at, "18: the frontier ids are the prompt's prefix up to turn_at");
        check(back.live.gdn == ck.gdn && back.live.tails == ck.tails,
              "18: the frontier carries the checkpoint's recurrent state (has_state travelled)");

        // DECLARED, not fixed: a frontier where the tail role token does NOT immediately follow a <|im_start|>
        // keeps the OLD boundary (the role token), which the engine never checkpoints.  The real template always
        // emits the role right after the opener (the measured pair), so this is the non-template case.
        const std::vector<int32_t> loose = joined({tokens(200000, 20), {90021}, tokens(201000, 4), {8678},
                                                   tokens(202000, 6)});
        const std::vector<SegmentPlan> loose_plan = segment_plan_split(loose, 90021, 8678);
        int64_t loose_frontier = 0;
        for (size_t k = 0; k + 1 < loose_plan.size(); ++k) loose_frontier += (int64_t) loose_plan[k].tokens;
        check(loose_frontier < (int64_t) loose.size() && loose[(size_t) loose_frontier] == 8678,
              "18: DECLARED - a non-adjacent role token keeps the old frontier (the role token)");
        check(loose[(size_t) loose_frontier] != 90021,
              "18: DECLARED - that frontier is not a <|im_start|>, so the engine never checkpoints there");
    }

    fs::remove_all(dir);
    std::printf("conversation_segment_test: %d checks passed\n", checks);
    return 0;
}
