// CPU-only tests for the disk parking store (--conversation-cache-storage disk,
// include/strata/core/conversation_disk.hpp): park writes a session-format file, restore reads it back byte for
// byte, eviction and the budget delete the oldest files, and the RAM index never holds a K/V byte.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_disk.hpp"

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

const SessionFileIdentity kId{0x1111, 0x2222};

// The engine's hooks, in test form: the plain session_file_* calls with a fixed identity and a free-space reserve.
ConversationDiskIo plain_io(uint64_t min_free_bytes = 0) {
    ConversationDiskIo io;
    io.write = [min_free_bytes](const std::string& path, const SavedConversation& image, size_t& bytes,
                                std::string& error) {
        SessionWriteOptions o;
        o.min_free_bytes = min_free_bytes;
        SessionStatus st;
        return session_file_write(path, image, kId, bytes, error, o, &st);
    };
    io.read = [](const std::string& path, SavedConversation& image, size_t& bytes, std::string& error) {
        SessionStatus st;
        return session_file_read(path, kId, image, bytes, error, SessionReadLimits{}, &st);
    };
    return io;
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

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed, bool pinned = false) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = std::vector<uint8_t>(4099, seed);
    c.ple = std::vector<uint8_t>(77, seed + 1);
    c.tails = std::vector<uint8_t>(301, seed + 2);
    c.dead = std::vector<uint8_t>(64, seed + 3);
    c.block_pos = std::vector<uint8_t>(8, seed + 4);
    c.used = 42 + seed;
    c.pinned = pinned;
    return c;
}

// A parked conversation: distinct token ids per `variant`, one K/V layer of `kb` KiB.
SavedConversation sample(int variant, size_t kb) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0;
    s.layer_hi = 48;
    s.cvec = true;
    s.live = checkpoint(size_t(100 + variant), uint8_t(1 + variant));
    for (size_t i = 0; i < s.live.ids.size(); ++i) s.live.ids[i] = int32_t(variant * 100000 + i);
    s.checkpoints.push_back(checkpoint(size_t(10 + variant), uint8_t(50 + variant)));
    for (size_t i = 0; i < s.checkpoints[0].ids.size(); ++i)
        s.checkpoints[0].ids[i] = int32_t(variant * 100000 + i);
    ConversationKv kv;
    kv.format = 2;
    kv.cells = 1024;
    kv.heads = 2;
    kv.head_dim = 256;
    kv.page_size = 64;
    kv.pooled_rows = 0;
    kv.idx_dim = 128;
    kv.k = pattern(kb * 1024, uint8_t(variant));
    kv.v = pattern(4096, uint8_t(variant + 10));
    kv.pooled = pattern(0, 0);
    s.kv.push_back(std::move(kv));
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() ||
        a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i];
        const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled))
            return false;
    }
    return true;
}

size_t dir_bytes(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file()) n += (size_t) e.file_size();
    return n;
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "strata_disk_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string d = dir.string();

    // ---- enabled(): the dir, the budget and the slots all have to be set
    check(!ConversationDiskCache(0, 4, 0, d).enabled(), "budget 0 disables");
    check(!ConversationDiskCache(4096, 0, 0, d).enabled(), "slots 0 disables");
    check(!ConversationDiskCache(4096, 4, 0, "").enabled(), "empty dir disables");
    check(!ConversationDiskCache(4096, 4, 500, d).wants(499), "min-tokens gate");
    check(ConversationDiskCache(4096, 4, 500, d).wants(500), "min-tokens pass");

    // ---- park: the file exists, take() reads the same bytes back, the file goes
    {
        ConversationDiskCache c(64u << 20, 4, 0, d);
        c.io(plain_io());
        SavedConversation a = sample(1, 64);
        const std::vector<int32_t> ids = a.live.ids;
        check(c.put(std::move(a)), "park 1");
        check(c.size() == 1 && c.bytes() == dir_bytes(dir), "bytes() counts the file");
        auto back = c.take(0);
        check(back.has_value(), "restore reads the file");
        check(same(*back, sample(1, 64)), "restored bytes match");
        check(back->live.ids == ids, "live ids match");
        check(c.size() == 0 && c.bytes() == 0, "entry gone after take");
        check(dir_bytes(dir) == 0, "file gone after take");
    }

    // ---- best(): the same prefix match as the RAM cache (checked against it)
    {
        ConversationDiskCache c(64u << 20, 4, 0, d);
        c.io(plain_io());
        ConversationCache ram(64u << 20, 4, 0);
        SavedConversation a = sample(1, 32), b = sample(2, 32);
        // give the two conversations a shared root: ids are variant*100000 + i, so remap b onto a's first 30
        for (size_t i = 0; i < 30; ++i) b.live.ids[i] = a.live.ids[i];
        for (size_t i = 0; i < 10 && i < b.checkpoints[0].ids.size(); ++i) b.checkpoints[0].ids[i] = a.live.ids[i];
        check(c.put(std::move(sample(1, 32))) && ram.put(sample(1, 32)), "park a");
        SavedConversation b2 = b;
        check(c.put(std::move(b)) && ram.put(std::move(b2)), "park b");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 120; ++i) prompt.push_back(100000 + i);   // a's ids, past both lives' ends
        prompt.push_back(-1);
        prompt.push_back(-2);
        // the image keys conversation_prefix matches on: a's (seed 2); b carries another (seed 3), which never
        // matches this prompt even where its tokens agree
        const std::vector<ConversationImageKey> imgs = {{3, 0x1234567890abcdefull ^ 2}};
        const auto dm = c.best(prompt, imgs, true);
        const auto rm = ram.best(prompt, imgs, true);
        // a's live state (101 tokens) is the longest prefix; b matches only the 30 shared root tokens
        check(dm.tokens == rm.tokens && dm.tokens == 101, "best matches the RAM cache");
        check(dm.index == rm.index && dm.live == rm.live, "best picks the RAM cache's entry");
        check(!c.best(prompt, imgs, false).tokens, "cvec mismatch never matches");
    }

    // ---- eviction: the oldest file goes, the budget holds, the counter counts
    {
        ConversationDiskCache probe(64u << 20, 4, 0, d);
        probe.io(plain_io());
        check(probe.put(std::move(sample(3, 64))), "park to measure");
        const size_t one_file = dir_bytes(dir);
        probe.clear();
        check(one_file > (64u << 10), "the file holds the payload");
        // a budget that fits one file but never two
        ConversationDiskCache cc(one_file + (one_file >> 3), 4, 0, d);
        cc.io(plain_io());
        check(cc.put(std::move(sample(3, 64))), "park 1 fits");
        check(cc.put(std::move(sample(4, 64))), "park 2 evicts park 1");
        check(cc.size() == 1 && cc.evictions() == 1, "one entry, one eviction");
        check(cc.bytes() == dir_bytes(dir), "bytes() is the files' bytes");
        cc.clear();
    }

    // ---- the budget refuses an oversized snapshot, and no file appears
    {
        const size_t one = sample(5, 256).bytes();
        ConversationDiskCache c(one - 1, 4, 0, d);
        c.io(plain_io());
        check(!c.put(std::move(sample(5, 256))), "oversized refused");
        check(dir_bytes(dir) == 0, "no file written for a refused park");
        check(c.size() == 0 && c.evictions() == 0, "nothing booked");
    }

    // ---- slots: the fourth park evicts the oldest even with disk room
    {
        ConversationDiskCache c(64u << 20, 2, 0, d);
        c.io(plain_io());
        check(c.put(std::move(sample(6, 16))), "slots park 1");
        check(c.put(std::move(sample(7, 16))), "slots park 2");
        check(c.put(std::move(sample(8, 16))), "slots park 3 (evicts)");
        check(c.size() == 2 && c.evictions() == 1, "slot pressure evicted the oldest");
        // what is left is 7 and 8: variant 7's ids no longer match
        std::vector<int32_t> p6;
        for (int i = 0; i < 20; ++i) p6.push_back(6 * 100000 + i);
        check(!c.best(p6, {}, true).tokens, "the evicted conversation is gone");
        c.clear();
    }

    // ---- pinned entries never leave by eviction
    {
        ConversationDiskCache c(64u << 20, 1, 0, d);
        c.io(plain_io());
        SavedConversation pin = sample(9, 16);
        pin.checkpoints[0].pinned = true;
        check(c.put(std::move(pin)), "pinned park");
        check(!c.put(std::move(sample(10, 16))), "no room over a pinned entry");
        check(c.size() == 1 && c.evictions() == 0, "pinned entry stayed");
        c.clear();
    }

    // ---- drop_superseded: a copy of this conversation a turn back goes, file and all
    {
        ConversationDiskCache c(64u << 20, 4, 0, d);
        c.io(plain_io());
        SavedConversation a = sample(11, 16);
        check(c.put(std::move(sample(11, 16))), "superseded park");
        const size_t before = dir_bytes(dir);
        check(before > 0, "a file to supersede");
        SavedConversation grown = sample(11, 16);
        grown.live = checkpoint(130, 77);
        grown.checkpoints.push_back(a.checkpoints[0]);   // the outgoing chain holds the parked copy's deepest
        const size_t dropped = c.drop_superseded(grown.live.ids, grown.live.imgs, grown.checkpoints, true);
        check(dropped == 1, "superseded copy dropped");
        check(dir_bytes(dir) == 0, "superseded file deleted");
        check(c.superseded() == 1, "superseded counted");
        c.clear();
    }

    // ---- a corrupt file: take() fails, the entry and the file go, the store lives on
    {
        ConversationDiskCache c(64u << 20, 4, 0, d);
        c.io(plain_io());
        check(c.put(std::move(sample(12, 64))), "park to corrupt");
        fs::path victim;
        for (const auto& e : fs::directory_iterator(dir)) victim = e.path();
        check(!victim.empty(), "found the file");
        {
            std::ofstream f(victim, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(200);
            f.put(0);
            f.put(0);
            f.put(0);
            f.put(0);
        }
        auto back = c.take(0);
        check(!back.has_value(), "corrupt file refused");
        check(!c.last_error().empty(), "the refusal says why");
        check(c.size() == 0 && dir_bytes(dir) == 0, "corrupt entry and file cleaned");
        check(c.put(std::move(sample(13, 16))), "the store still parks");
        c.clear();
    }

    // ---- an identity mismatch is a refusal too (another model's file)
    {
        ConversationDiskCache c(64u << 20, 4, 0, d);
        ConversationDiskIo io = plain_io();
        const SessionFileIdentity other{0x9999, 0x2222};
        io.read = [other](const std::string& path, SavedConversation& image, size_t& bytes, std::string& error) {
            SessionStatus st;
            return session_file_read(path, other, image, bytes, error, SessionReadLimits{}, &st);
        };
        c.io(std::move(io));
        check(c.put(std::move(sample(14, 32))), "park for identity check");
        auto back = c.take(0);
        check(!back.has_value() && dir_bytes(dir) == 0, "foreign identity refused and cleaned");
    }

    // ---- longest_tokens follows the index
    {
        ConversationDiskCache c(64u << 20, 4, 0, d);
        c.io(plain_io());
        check(c.longest_tokens() == 0, "empty store");
        SavedConversation big = sample(15, 16);
        big.live = checkpoint(4242, 5);
        check(c.put(std::move(big)), "park the long one");
        check(c.longest_tokens() == 4242, "longest tokens from the index");
        check(c.put(std::move(sample(16, 16))), "park a short one");
        check(c.longest_tokens() == 4242, "longest unchanged");
        c.clear();
    }

    // ---- the free-space preflight (min_free_bytes) refuses a park without leaving a file
    {
        ConversationDiskCache c(64u << 20, 4, 0, d);
        c.io(plain_io(UINT64_MAX / 2));   // a reserve no disk can meet
        check(!c.put(std::move(sample(17, 16))), "preflight refused");
        check(dir_bytes(dir) == 0 && c.size() == 0, "nothing written, nothing booked");
        check(!c.last_error().empty(), "the refusal says why");
    }

    fs::remove_all(dir);
    std::printf("conversation_disk_test: %d checks OK\n", checks);
    return 0;
}
