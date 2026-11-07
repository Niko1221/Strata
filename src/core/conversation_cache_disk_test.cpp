// Round-trip test for the conversation-cache disk tier (conversation_cache_disk.cpp).
// A synthetic parked conversation is written, its heavy parts freed to a shell, and read back: every field must
// come back byte-identical, a truncated file must be refused, and an empty image must survive the trip too.
#include "strata/core/conversation_cache.hpp"
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

using namespace strata::core;
static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static std::mt19937 rng(1234);
static std::vector<uint8_t> blob(size_t n) {
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = (uint8_t) rng();
    return v;
}
static void fill_buf(ConversationBuffer& b, size_t n) {
    b.resize_for_overwrite(n);
    b.visit(0, n, [](uint8_t* p, size_t c, size_t) { for (size_t i = 0; i < c; ++i) p[i] = (uint8_t) rng(); return true; });
}
static void fill_ckpt(ConversationCheckpoint& c, size_t nids, size_t blobn) {
    c.ids.resize(nids);
    for (size_t i = 0; i < nids; ++i) c.ids[i] = (int32_t) rng();
    c.imgs = {{5, 0xdead}, {70000, 0xbeef}, {999999, 1}};
    c.gdn = blob(blobn); c.ple = blob(blobn / 2); c.tails = blob(17); c.dead = blob(3); c.block_pos = blob(blobn / 3);
}
static void fill_kv(ConversationKv& k, int i) {
    k.format = 2; k.cells = 64 + i; k.heads = 8; k.head_dim = 128; k.page_size = 16; k.pooled_rows = 4096; k.idx_dim = 32 + i;
    fill_buf(k.k, (1u << 20) + 5 + i); fill_buf(k.v, 16u << 20); fill_buf(k.k_scale, 0); fill_buf(k.v_scale, 17);
    fill_buf(k.pooled, 33u << 20);   // crosses the 16 MiB segment boundary
}
static void fill_image(SavedConversation& s, size_t nids, size_t blobn) {
    for (auto& g : s.geometry) g = (int64_t) rng();
    s.layer_lo = 100; s.layer_hi = 200; s.cvec = true;
    fill_ckpt(s.live, nids, blobn);
    s.checkpoints.resize(3);
    fill_ckpt(s.checkpoints[0], 500, 20000); fill_ckpt(s.checkpoints[1], 5000, 60000); fill_ckpt(s.checkpoints[2], nids / 2, 30000);
    s.kv.resize(5);
    for (size_t i = 0; i < 5; ++i) fill_kv(s.kv[i], (int) i);
}
static bool eq(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple &&
           a.tails == b.tails && a.dead == b.dead && a.block_pos == b.block_pos;
}
static bool eq(const SavedConversation& a, const SavedConversation& b) {
    if (!(a.geometry == b.geometry && a.layer_lo == b.layer_lo && a.layer_hi == b.layer_hi && a.cvec == b.cvec)) return false;
    if (!eq(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i) if (!eq(a.checkpoints[i], b.checkpoints[i])) return false;
    if (a.kv.size() != b.kv.size() || a.stage_images.size() != b.stage_images.size()) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim) return false;
        if (!(x.k == y.k && x.v == y.v && x.k_scale == y.k_scale && x.v_scale == y.v_scale && x.pooled == y.pooled)) return false;
    }
    for (size_t i = 0; i < a.stage_images.size(); ++i) if (!eq(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

int main() {
    const std::string dir = "/tmp/strata-conv-disk-test";
    SavedConversation orig;
    fill_image(orig, 100000, 200000);
    orig.stage_images.resize(2);
    fill_image(orig.stage_images[0], 900, 9000);
    fill_image(orig.stage_images[1], 1200, 12000);

    SavedConversation copy = orig;   // a copy: the write shells it out
    const size_t before = copy.bytes();
    size_t freed = 0, fsize = 0;
    CHECK(conversation_cache_disk_write(dir, 42, copy, freed, fsize));
    CHECK(freed > before / 2);                          // the heavy parts really left RAM
    CHECK(copy.kv.empty() && copy.live.gdn.empty());    // the shell holds no K/V
    CHECK(copy.live.ids.size() == 100000);              // but keeps the ids matching reads
    CHECK(copy.checkpoints.size() == 3 && copy.checkpoints[1].ids.size() == 5000);

    std::string err;
    SavedConversation back;
    CHECK(conversation_cache_disk_read(dir + "/42.conv", back, err));
    if (!err.empty()) std::printf("read err: %s\n", err.c_str());
    CHECK(eq(orig, back));                              // byte-identical: ids, K/V, checkpoints, stages
    std::printf("round-trip: %zu MiB image -> %zu MiB file -> read back (freed %zu MiB)\n",
                before >> 20, fsize >> 20, freed >> 20);

    // A half-written file (a crash between the writes) must be refused, not crash.
    { std::string p = dir + "/42.conv";
      CHECK(truncate(p.c_str(), 5000000) == 0);
      SavedConversation half; std::string e2;
      CHECK(!conversation_cache_disk_read(p, half, e2));
      std::printf("truncated read refused: %s\n", e2.c_str()); }
    // An empty image round-trips too.
    { SavedConversation e; SavedConversation c2 = e; size_t fr = 0, fs = 0;
      CHECK(conversation_cache_disk_write(dir, 43, c2, fr, fs));
      SavedConversation b2; std::string e3;
      CHECK(conversation_cache_disk_read(dir + "/43.conv", b2, e3));
      CHECK(eq(e, b2)); }

    if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
    std::printf("ALL OK\n");
    return 0;
}
