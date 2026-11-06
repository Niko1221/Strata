// File (disk) tier transport for --serve's parked conversations: a spilled conversation's heavy bytes (the
// running state, the checkpoint copies and the K/V pages) go to one file under the cache's overflow folder and
// the entry stays behind as an in-memory shell (its ids, images, pinned marks and geometry are all `best()`
// matching needs).  The format is internal to one engine version: a file written by another build is refused by
// the version tag, so no cross-version byte layout ever has to be stable (like the session files, a park that
// cannot read its file back just reads the prompt again).
// Copyright (c) 2026 the Strata authors.  Part of Strata under the same terms as the rest of the engine.
#include "strata/core/conversation_cache.hpp"

#include <cstdio>

namespace strata::core {
namespace {

constexpr uint64_t kSpillTag = 0x534B56435041524Bull;   // an internal tag: no cross-version stability promise
constexpr uint64_t kMaxField = 64ull << 30;             // 64 GiB: a corrupt length fails here, not in resize()

struct DiskWriter {
    std::FILE* f;
    bool ok = true;
    void u64(uint64_t v) { if (std::fwrite(&v, sizeof v, 1, f) != 1) ok = false; }
    void raw(const void* p, size_t n) { if (n && std::fwrite(p, 1, n, f) != n) ok = false; }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void buf(const ConversationBuffer& b) {
        u64(b.size());
        b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t) { return std::fwrite(p, 1, n, f) == n; });
    }
    void ckpt(const ConversationCheckpoint& c) {
        u64(c.ids.size());
        raw(c.ids.data(), c.ids.size() * sizeof(int32_t));
        u64(c.imgs.size());
        for (const auto& im : c.imgs) { u64((uint64_t) im.start); u64(im.hash); }
        u64(c.gdn.size()); raw(c.gdn.data(), c.gdn.size());
        u64(c.ple.size()); raw(c.ple.data(), c.ple.size());
        u64(c.tails.size()); raw(c.tails.data(), c.tails.size());
        u64(c.dead.size()); raw(c.dead.data(), c.dead.size());
        u64(c.block_pos.size()); raw(c.block_pos.data(), c.block_pos.size());
    }
    void kv(const ConversationKv& k) {
        u64((uint64_t) k.format); u64((uint64_t) k.cells); u64((uint64_t) k.heads); u64((uint64_t) k.head_dim);
        u64((uint64_t) k.page_size); u64((uint64_t) k.pooled_rows); u64((uint64_t) k.idx_dim);
        buf(k.k); buf(k.v); buf(k.k_scale); buf(k.v_scale); buf(k.pooled);
    }
    void image(const SavedConversation& s) {
        for (const auto g : s.geometry) u64((uint64_t) g);
        u64((uint64_t) s.layer_lo); u64((uint64_t) s.layer_hi); u64(s.cvec ? 1 : 0);
        ckpt(s.live);
        u64(s.checkpoints.size());
        for (const auto& c : s.checkpoints) ckpt(c);
        u64(s.kv.size());
        for (const auto& k : s.kv) kv(k);
        u64(s.stage_images.size());
        for (const auto& st : s.stage_images) image(st);
    }
};

struct DiskReader {
    std::FILE* f;
    bool ok = true;
    uint64_t u64() { uint64_t v = 0; if (std::fread(&v, sizeof v, 1, f) != 1) ok = false; return v; }
    size_t sz() { const uint64_t v = u64(); if (v > kMaxField) { ok = false; return 0; } return (size_t) v; }
    std::string str() { const size_t n = sz(); std::string s(n, 0); if (n && std::fread(s.data(), 1, n, f) != n) ok = false; return s; }
    void buf(ConversationBuffer& b) {
        const size_t n = sz();
        if (!ok) return;
        b.resize_for_overwrite(n);
        b.visit(0, n, [&](uint8_t* p, size_t c, size_t) { return std::fread(p, 1, c, f) == c; });
    }
    void heavy(std::vector<uint8_t>& v) { const size_t n = sz(); v.resize(n); if (n && std::fread(v.data(), 1, n, f) != n) ok = false; }
    void ckpt(ConversationCheckpoint& c) {
        const size_t n = sz(); c.ids.resize(n);
        if (n && std::fread(c.ids.data(), sizeof(int32_t), n, f) != n) ok = false;
        const size_t m = sz(); c.imgs.resize(m);
        for (auto& im : c.imgs) { im.start = (int64_t) u64(); im.hash = u64(); }
        heavy(c.gdn); heavy(c.ple); heavy(c.tails); heavy(c.dead); heavy(c.block_pos);
    }
    void kv(ConversationKv& k) {
        k.format = (int) u64(); k.cells = (int64_t) u64(); k.heads = (int64_t) u64(); k.head_dim = (int64_t) u64();
        k.page_size = (int64_t) u64(); k.pooled_rows = (int64_t) u64(); k.idx_dim = (int64_t) u64();
        buf(k.k); buf(k.v); buf(k.k_scale); buf(k.v_scale); buf(k.pooled);
    }
    void image(SavedConversation& s) {
        for (auto& g : s.geometry) g = (int64_t) u64();
        s.layer_lo = (int64_t) u64(); s.layer_hi = (int64_t) u64(); s.cvec = u64() != 0;
        ckpt(s.live);
        const size_t n = sz(); s.checkpoints.resize(n);
        for (auto& c : s.checkpoints) ckpt(c);
        const size_t m = sz(); s.kv.resize(m);
        for (auto& k : s.kv) kv(k);
        const size_t t = sz(); s.stage_images.resize(t);
        for (auto& st : s.stage_images) image(st);
    }
};

// The spilled entry keeps what matching reads (ids, images, the pinned marks, the geometry) and gives back the rest.
void free_ckpt(ConversationCheckpoint& c) {
    std::vector<uint8_t>().swap(c.gdn); std::vector<uint8_t>().swap(c.ple);
    std::vector<uint8_t>().swap(c.tails); std::vector<uint8_t>().swap(c.dead);
    std::vector<uint8_t>().swap(c.block_pos);
    for (auto& part : c.stage_parts) free_ckpt(part);
}
void free_heavy(SavedConversation& s) {
    free_ckpt(s.live);
    for (auto& c : s.checkpoints) free_ckpt(c);
    std::vector<ConversationKv>().swap(s.kv);
    for (auto& st : s.stage_images) free_heavy(st);
}

} // namespace

bool conversation_cache_disk_write(const std::string& dir, uint64_t serial, SavedConversation& image,
                                   size_t& freed, size_t& fsize) {
    freed = 0; fsize = 0;
    const size_t before = image.bytes();
    std::error_code mec;
    std::filesystem::create_directories(dir, mec);
    const std::string path = dir + "/" + std::to_string(serial) + ".conv";
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    DiskWriter w{f};
    w.u64(kSpillTag);
    w.str(STRATA_VERSION);
    w.image(image);
    if (std::fflush(f) != 0 || std::ferror(f) || std::fclose(f) != 0 || !w.ok) { std::remove(tmp.c_str()); return false; }
    std::error_code sec;
    const auto sz = std::filesystem::file_size(tmp, sec);
    if (sec || std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    free_heavy(image);
    freed = before > image.bytes() ? before - image.bytes() : 0;
    fsize = (size_t) sz;
    return true;
}

bool conversation_cache_disk_read(const std::string& path, SavedConversation& image, std::string& error) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { error = "cannot open " + path; return false; }
    DiskReader r{f};
    const bool tag_ok = r.u64() == kSpillTag;
    const std::string ver = r.str();
    SavedConversation back;
    r.image(back);
    if (std::ferror(f) != 0 || std::fclose(f) != 0) { error = "read failed: " + path; return false; }
    if (!tag_ok || !r.ok) { error = "bad spill file: " + path; return false; }
    if (ver != STRATA_VERSION) { error = "spill written by engine " + ver + ", this is " STRATA_VERSION; return false; }
    image = std::move(back);
    return true;
}

} // namespace strata::core
