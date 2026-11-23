#pragma once
// include/strata/core/conversation_spill_parts.hpp - V3 of the conversation cache's SSD tier (experimental, behind
// STRATA_SPILL_PARTS=1): an entry is kept as PARTS instead of one session file, so that a conversation written to the
// tier again - the usual case, one long conversation parked again on each turn - is extended in place instead of being
// written whole.  The session file format v1 has one contiguous payload whose offsets move as soon as the conversation
// grows, so a re-spill can reuse no block of the file it replaces.  Here:
//
//   <folder>/state-<gen>.session  the session file's own writer, with the K/V layers left out (live state + checkpoints;
//                                 about 236 MB on Qwen3.8-Flash-Next whatever the conversation's length).  Each generation
//                                 has its own name, so the previous generation's state file is not touched until the new
//                                 index names the new one (see the commit order below).
//   <folder>/kv-<l>-<p>.bin       one file per K/V layer and part (k, v, k_scale, v_scale, pooled): raw bytes, laid out
//                                 logically as [0,size) and only ever extended - a growing conversation writes just the
//                                 tail it changed; every earlier byte keeps its offset and its content.
//   <folder>/ids.bin              the ids sidecar: the live state's ids and every checkpoint's ids and picture keys, so a
//                                 process that adopts this folder can match a prompt against it without reading the
//                                 session file.  Bound to the index by ids_len/ids_hash (Meta); a mismatch only makes the
//                                 entry bytes-only, never wrong.
//   <folder>/parts.idx            the layer headers and, per part, the size, the committed size (bytes on the disk, which
//                                 may exceed `size` after an aborted append), the hash of each whole 1 MiB block and the
//                                 hash of the tail.  THE INDEX IS THE COMMIT POINT: an entry exists only once parts.idx is
//                                 in place, and it names the state generation, the ids and the block hashes of exactly the
//                                 bytes below.  A crash at any earlier moment leaves the previous generation complete.
//
// Writing a conversation that has a compatible earlier entry (same model/config/layer heads, every new part a byte
// extension of the old one) APPENDS to the same folder - same serial, same index - after writing only the bytes past the
// old size.  Anything else (a different conversation, a shrink, a layer geometry change, no earlier entry) becomes a
// FRESH folder written whole; the earlier entry is only removed after the new index is durable.
//
// Commit order (both modes; a crash between two steps leaves either the old index's generation or nothing new):
//   a) the new state generation's session file (session_file_write: temporary name, flush, rename, folder flush);
//   b) the part files' appended bytes, flushed (fsync) per file - the old generation's bytes [0, old size) are untouched;
//   c) ids.bin (temporary name, flush, rename) when it can be written;
//   d) the folder flush, so b)'s and c)'s names and data are durable BEFORE the index can name them;
//   e) parts.idx via a temporary name, flush, rename, folder flush - this is the commit;
//   f) only now: the old generation's state file and, for a replacement, the old folder are removed; the caller updates
//      the entry list and the budget.
// A failure before e) keeps the previous index (and so the previous generation) valid, except that an id sidecar of the
// previous generation may already have been replaced: the old entry then reads as bytes-only, not wrong.  Reading
// verifies the tail hash always (STRATA_SPILL_VERIFY=all verifies every block), and the process that adopts a folder
// from a previous process verifies every block once and drops the folder on the first mismatch.
//
// POSIX only for now; the Windows build keeps the session-file path.

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core::spill_parts {

inline constexpr size_t kBlock = 1u << 20;    // one K/V part is compared and written in 1 MiB blocks
inline constexpr uint32_t kVersion = 3;      // v3 adds the committed size, the ids binding and the state generation
inline constexpr uint32_t kVersionMinRead = 1;   // v1/v2 indexes are still read (state.session, no committed size)
inline constexpr char kMagic[8] = {'S', 'T', 'R', 'S', 'P', 'L', '0', '1'};
inline constexpr size_t kMaxIndexBytes = size_t(1) << 20;   // what read_meta() admits, and what write_meta() enforces
inline constexpr uint64_t kMaxPartBytes = uint64_t(1) << 40;   // sanity bounds for an index this process did not write

struct PartInfo {
    uint64_t size = 0;              // the part's logical size (bytes the reader wants)
    uint64_t committed = 0;         // bytes the file is known to hold: >= size, may be larger after an aborted append
    std::vector<uint64_t> blocks;   // one hash per whole block (size / kBlock of them)
    uint64_t tail_len = 0;          // the bytes after the last whole block
    uint64_t tail_hash = 0;         // their hash (0 when there are none)
    uint64_t reused = 0;            // debug only: bytes taken from the previous entry
};

struct LayerInfo {
    std::array<int64_t, 7> head{};   // format, cells, heads, head_dim, page_size, pooled_rows, idx_dim
    std::array<PartInfo, 5> parts;   // k, v, k_scale, v_scale, pooled
};

struct Meta {
    uint64_t model = 0, config = 0, chain = 0;
    uint32_t cvec = 0;        // the entry's picture-steering flag
    uint64_t ids_len = 0;     // v3: the ids sidecar's live-id length in bytes (0: no binding, or a pre-v3 index)
    uint64_t ids_hash = 0;    // v3: the hash of those bytes
    uint64_t state_gen = 0;   // v3: which state-<gen>.session the index names (0: the legacy "state.session")
    std::vector<LayerInfo> layers;
};

inline uint64_t block_hash(const uint8_t* p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

inline void put_u32(std::string& b, uint32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); }
inline void put_u64(std::string& b, uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); }
inline void put_i64(std::string& b, int64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); }

struct Cursor {
    const std::string& b;
    size_t at = 0;
    bool ok = true;
    bool need(size_t n) { if (!ok || b.size() - at < n) { ok = false; return false; } return true; }
    uint32_t u32() { if (!need(4)) return 0; uint32_t v; std::memcpy(&v, b.data() + at, 4); at += 4; return v; }
    uint64_t u64() { if (!need(8)) return 0; uint64_t v; std::memcpy(&v, b.data() + at, 8); at += 8; return v; }
    int64_t i64() { return (int64_t) u64(); }
};

// ---------------------------------------------------------------------------------------------------------------
// the state generation: each park writes its running state under a fresh name, so the previous generation's state file
// survives until the new index (the commit point) names the new one.  The generation is only a name: it says nothing
// about the K/V, which are append-only in their own files.
inline std::filesystem::path state_file(const std::filesystem::path& dir, uint64_t gen) {
    return gen == 0 ? dir / "state.session" : dir / ("state-" + std::to_string(gen) + ".session");
}

inline uint64_t next_state_gen(uint64_t avoid) {
    static std::atomic<uint64_t> counter{0};
    uint64_t g = ++counter;
    if (g == avoid) g = ++counter;
    return g;
}

#if !defined(_WIN32)
/// Flushes a directory's entries (a rename is only durable once its folder is flushed; conversation_file.cpp does the
/// same for the session file's own publish).
inline bool flush_dir(const std::filesystem::path& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// the index
inline bool write_meta(const std::filesystem::path& path, const Meta& m, std::string& error) {
    std::string b;
    b.append(kMagic, 8);
    put_u32(b, kVersion);
    put_u32(b, m.cvec ? 1u : 0u);
    put_u64(b, m.model);
    put_u64(b, m.config);
    put_u64(b, m.chain);
    put_u64(b, m.ids_len);      // v3
    put_u64(b, m.ids_hash);     // v3
    put_u64(b, m.state_gen);    // v3
    put_u32(b, (uint32_t) m.layers.size());
    put_u32(b, 0);
    for (const auto& l : m.layers) {
        for (int64_t v : l.head) put_i64(b, v);
        for (const auto& p : l.parts) {
            put_u64(b, p.size);
            put_u64(b, p.committed);   // v3
            put_u64(b, (uint64_t) p.blocks.size());
            put_u64(b, p.tail_len);
            put_u64(b, p.tail_hash);
            for (uint64_t h : p.blocks) put_u64(b, h);
        }
    }
    if (b.size() > kMaxIndexBytes) { error = "spill parts: index too large"; return false; }
    const std::filesystem::path tmp = path.string() + ".tmp";
#if defined(_WIN32)
    (void) tmp;
    error = "spill parts: not supported on this platform";
    return false;
#else
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) { error = "spill parts: cannot write " + tmp.string(); return false; }
    size_t at = 0;
    bool ok = true;
    while (at < b.size()) {
        const ssize_t n = ::write(fd, b.data() + at, b.size() - at);
        if (n <= 0) { ok = false; break; }
        at += (size_t) n;
    }
    ::fsync(fd);   // the index bytes are durable before the rename publishes them
    ::close(fd);
    if (!ok) { std::error_code ec; std::filesystem::remove(tmp, ec); error = "spill parts: short write of " + tmp.string(); return false; }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) { std::filesystem::remove(tmp, ec); error = "spill parts: cannot name " + path.string(); return false; }
    return true;
#endif
}

inline bool read_meta(const std::filesystem::path& path, Meta& m, std::string& error) {
    std::string b;
    {
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec || size < 32 || size > (uint64_t) kMaxIndexBytes) { error = "spill parts: no index"; return false; }
        b.resize((size_t) size);
        FILE* f = std::fopen(path.string().c_str(), "rb");
        if (!f) { error = "spill parts: cannot read " + path.string(); return false; }
        const size_t got = std::fread(b.data(), 1, b.size(), f);
        std::fclose(f);
        if (got != b.size()) { error = "spill parts: short index"; return false; }
    }
    if (std::memcmp(b.data(), kMagic, 8) != 0) { error = "spill parts: bad index magic"; return false; }
    Cursor c{b, 8};
    const uint32_t ver = c.u32();
    if (ver < kVersionMinRead || ver > kVersion) { error = "spill parts: another index version"; return false; }
    m.cvec = c.u32();
    m.model = c.u64();
    m.config = c.u64();
    m.chain = c.u64();
    if (ver >= 3) {
        m.ids_len = c.u64();
        m.ids_hash = c.u64();
        m.state_gen = c.u64();
    }
    const uint32_t layers = c.u32();
    (void) c.u32();
    if (!c.ok || layers > 1024) { error = "spill parts: bad index"; return false; }
    m.layers.resize(layers);
    for (auto& l : m.layers) {
        for (auto& v : l.head) v = c.i64();
        for (auto& p : l.parts) {
            p.size = c.u64();
            p.committed = ver >= 3 ? c.u64() : p.size;   // a v1/v2 index wrote exactly `size` bytes
            const uint64_t n = c.u64();
            if (!c.ok || n > (uint64_t(1) << 26)) { error = "spill parts: bad block table"; return false; }
            p.tail_len = c.u64();
            p.tail_hash = c.u64();
            if (p.size > kMaxPartBytes || p.committed > kMaxPartBytes || p.committed < p.size) {
                error = "spill parts: bad part extent";
                return false;
            }
            if ((uint64_t) n * kBlock + p.tail_len != p.size) { error = "spill parts: bad part blocks"; return false; }
            p.blocks.resize((size_t) n);
            for (auto& h : p.blocks) h = c.u64();
        }
    }
    if (!c.ok) { error = "spill parts: truncated index"; return false; }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// the ids sidecar: what lets a process that adopts a folder from a previous one match a prompt against it.  The live
// ids are bound to the index by Meta.ids_len/ids_hash (see write_entry: the sidecar is durable BEFORE the index names
// it, and a mismatch is treated as "no ids", never as a different conversation).  Ids identify the K/V by
// construction - they are read out of the same folder as the K/V - so the "never mount K/V that is not this token
// sequence" invariant holds even for an adopted folder.
inline constexpr char kIdsMagic[8] = {'S', 'T', 'R', 'I', 'D', 'S', '0', '1'};
inline constexpr uint32_t kIdsVersion = 1;

/// The binding of a checkpoint's ids: {bytes, fnv(bytes)}.  Empty ids bind to (0, 0).
inline void ids_identity(const ConversationCheckpoint& c, uint64_t& len, uint64_t& hash) {
    len = (uint64_t) c.ids.size() * 4;
    if (c.ids.empty()) { hash = 0; return; }
    hash = block_hash(reinterpret_cast<const uint8_t*>(c.ids.data()), (size_t) len);
}

inline bool write_ids(const std::filesystem::path& dir, const SavedConversation& image, uint64_t& live_len,
                      uint64_t& live_hash, std::string& error) {
    live_len = live_hash = 0;
    ids_identity(image.live, live_len, live_hash);
#if defined(_WIN32)
    (void) dir; (void) image;
    error = "spill parts: not supported on this platform";
    return false;
#else
    std::string b;
    b.append(kIdsMagic, 8);
    put_u32(b, kIdsVersion);
    put_u32(b, 0);
    auto put_checkpoint = [&](const ConversationCheckpoint& c) {
        put_u32(b, (uint32_t) c.ids.size());
        for (int32_t t : c.ids) put_u32(b, (uint32_t) t);
        put_u32(b, (uint32_t) c.imgs.size());
        for (const auto& im : c.imgs) { put_i64(b, im.start); put_u64(b, im.hash); }
    };
    put_checkpoint(image.live);
    put_u32(b, (uint32_t) image.checkpoints.size());
    for (const auto& c : image.checkpoints) put_checkpoint(c);
    const std::filesystem::path tmp = (dir / "ids.bin").string() + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) { error = "spill parts: cannot write " + tmp.string(); return false; }
    size_t at = 0;
    bool ok = true;
    while (at < b.size()) {
        const ssize_t n = ::write(fd, b.data() + at, b.size() - at);
        if (n <= 0) { ok = false; break; }
        at += (size_t) n;
    }
    if (ok) ::fsync(fd);   // durable before the index names it (commit order c)
    ::close(fd);
    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        error = "spill parts: short write of " + tmp.string();
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, dir / "ids.bin", ec);
    if (ec) { std::filesystem::remove(tmp, ec); error = "spill parts: cannot name ids.bin"; return false; }
    return true;
#endif
}

/// Reads the sidecar write_ids() wrote; false (with `error`) when it is absent or unreadable.  Only ids and image keys
/// are restored - the K/V itself always comes from the same folder's part files, bound by the index.
inline bool read_ids(const std::filesystem::path& dir, ConversationCheckpoint& live,
                     std::vector<ConversationCheckpoint>& checks, std::string& error) {
    std::string b;
    {
        std::error_code ec;
        const auto size = std::filesystem::file_size(dir / "ids.bin", ec);
        if (ec || size < 16 || size > (uint64_t(64) << 20)) { error = "spill parts: no ids sidecar"; return false; }
        b.resize((size_t) size);
        FILE* f = std::fopen((dir / "ids.bin").string().c_str(), "rb");
        if (!f) { error = "spill parts: cannot read ids.bin"; return false; }
        const size_t got = std::fread(b.data(), 1, b.size(), f);
        std::fclose(f);
        if (got != b.size()) { error = "spill parts: short ids.bin"; return false; }
    }
    if (std::memcmp(b.data(), kIdsMagic, 8) != 0) { error = "spill parts: bad ids magic"; return false; }
    Cursor c{b, 8};
    if (c.u32() != kIdsVersion) { error = "spill parts: another ids version"; return false; }
    (void) c.u32();
    auto get_checkpoint = [&](ConversationCheckpoint& out) -> bool {
        const uint32_t n = c.u32();
        if (!c.ok || n > (uint32_t) (32u << 20)) return false;
        out.ids.resize(n);
        for (auto& t : out.ids) t = (int32_t) c.u32();
        const uint32_t m = c.u32();
        if (!c.ok || m > (uint32_t) (1u << 20)) return false;
        out.imgs.resize(m);
        for (auto& im : out.imgs) { im.start = c.i64(); im.hash = c.u64(); }
        return c.ok;
    };
    if (!get_checkpoint(live)) { error = "spill parts: bad live ids"; return false; }
    const uint32_t n = c.u32();
    if (!c.ok || n > (uint32_t) (1u << 16)) { error = "spill parts: bad checkpoint count"; return false; }
    checks.resize(n);
    for (auto& ck : checks)
        if (!get_checkpoint(ck)) { error = "spill parts: bad checkpoint ids"; return false; }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// parts: hashing, writing, verifying
inline const ConversationBuffer& part_of(const ConversationKv& k, size_t part) {
    switch (part) {
        case 0: return k.k;
        case 1: return k.v;
        case 2: return k.k_scale;
        case 3: return k.v_scale;
        default: return k.pooled;
    }
}

inline ConversationBuffer& part_of(ConversationKv& k, size_t part) {
    switch (part) {
        case 0: return k.k;
        case 1: return k.v;
        case 2: return k.k_scale;
        case 3: return k.v_scale;
        default: return k.pooled;
    }
}

inline std::filesystem::path part_path(const std::filesystem::path& dir, size_t layer, size_t part) {
    return dir / ("kv-" + std::to_string(layer) + "-" + std::to_string(part) + ".bin");
}

inline uint64_t hash_range(const ConversationBuffer& b, uint64_t offset, size_t len) {
    uint64_t h = 0xcbf29ce484222325ull;
    if (len == 0) return 0;   // the convention PartInfo::tail_hash uses for "no tail"
    b.visit((size_t) offset, len, [&](const uint8_t* p, size_t n, size_t) { h = block_hash(p, n, h); return true; });
    return h;
}

/// Fills `out` with the description of `b` (size, block hashes, tail) - no disk access: the append decision and the
/// index both use these numbers.
inline void hash_part(const ConversationBuffer& b, PartInfo& out) {
    const uint64_t size = (uint64_t) b.size();
    const size_t nblocks = (size_t) (size / kBlock);
    const size_t tail_len = (size_t) (size - (uint64_t) nblocks * kBlock);
    out.size = size;
    out.committed = size;
    out.reused = 0;
    out.blocks.assign(nblocks, 0);
    out.tail_len = tail_len;
    for (size_t i = 0; i < nblocks; ++i) {
        uint64_t h = 0xcbf29ce484222325ull;
        b.visit(i * kBlock, kBlock, [&](const uint8_t* p, size_t n, size_t) { h = block_hash(p, n, h); return true; });
        out.blocks[i] = h;
    }
    out.tail_hash = tail_len ? hash_range(b, (uint64_t) nblocks * kBlock, tail_len) : 0;
}

/// Whether the previous entry's file is a byte prefix of `now` for this part - the only case in which the file may be
/// extended in place: every whole old block still hashes the same and the old tail's bytes (the part past its last whole
/// block) are the same at the same offsets.  `committed` is the extent the old file is known to hold; a file shorter
/// than that (a previous process lost it) is not extended.
inline bool append_safe(const PartInfo& old, const PartInfo& now, const ConversationBuffer& b,
                        const std::filesystem::path& path, std::string& why_not) {
    if (old.size > now.size) { why_not = "shrank"; return false; }
    if (old.blocks.size() > now.blocks.size()) { why_not = "fewer blocks"; return false; }
    for (size_t i = 0; i < old.blocks.size(); ++i)
        if (old.blocks[i] != now.blocks[i]) { why_not = "block " + std::to_string(i); return false; }
    if (old.tail_len != 0 && hash_range(b, (uint64_t) old.blocks.size() * kBlock, (size_t) old.tail_len) != old.tail_hash) {
        why_not = "tail";
        return false;
    }
    std::error_code ec;
    const auto on_disk = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec || on_disk < old.committed) { why_not = "file"; return false; }
    return true;
}

/// Writes `b` whole to `path` (a fresh folder: no earlier bytes to keep).  `out` already carries the hashes; its
/// `committed` becomes the file's size.
inline bool write_part_fresh(const std::filesystem::path& path, const ConversationBuffer& b, PartInfo& out,
                             uint64_t& wrote_bytes, std::string& error) {
#if defined(_WIN32)
    (void) path; (void) b; (void) out; (void) wrote_bytes;
    error = "spill parts: not supported on this platform";
    return false;
#else
    const uint64_t size = out.size;
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) { error = "spill parts: cannot write " + path.string(); return false; }
    bool ok = true;
    size_t at = 0;
    while (ok && at < size) {
        ok = b.visit(at, (size_t) (size - at), [&](const uint8_t* p, size_t n, size_t off) {
            size_t done = 0;
            while (done < n) {
                const ssize_t w = ::pwrite(fd, p + done, n - done, (off_t) (off + done));
                if (w <= 0) return false;
                done += (size_t) w;
            }
            return true;
        });
        at = size;   // visit covered the rest
    }
    if (ok) ok = ::fsync(fd) == 0;   // commit order b: the part bytes are durable before the index names them
    ::close(fd);
    if (!ok) { error = "spill parts: short write of " + path.string(); return false; }
    out.committed = size;
    wrote_bytes = size;
    return true;
#endif
}

/// Extends `path` in place with the bytes past the previous entry's size (caller has checked append_safe(): those bytes
/// are exactly the new content at the same offsets, and the file's earlier bytes are untouched, so a crash here leaves
/// the previous index's generation readable).
inline bool write_part_append(const std::filesystem::path& path, const ConversationBuffer& b, const PartInfo& old,
                              PartInfo& out, uint64_t& wrote_bytes, std::string& error) {
#if defined(_WIN32)
    (void) path; (void) b; (void) old; (void) out; (void) wrote_bytes;
    error = "spill parts: not supported on this platform";
    return false;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) { error = "spill parts: cannot extend " + path.string(); return false; }
    bool ok = true;
    if (out.size > old.size) {
        const size_t at = (size_t) old.size;
        ok = b.visit(at, (size_t) (out.size - at), [&](const uint8_t* p, size_t n, size_t off) {
            size_t done = 0;
            while (done < n) {
                const ssize_t w = ::pwrite(fd, p + done, n - done, (off_t) (off + done));
                if (w <= 0) return false;
                done += (size_t) w;
            }
            return true;
        });
    }
    if (ok) ok = ::fsync(fd) == 0;   // commit order b
    struct stat st {};
    const bool stated = ok && ::fstat(fd, &st) == 0;
    ::close(fd);
    if (!ok || !stated) { error = "spill parts: short extend of " + path.string(); return false; }
    const uint64_t on_disk = (uint64_t) st.st_size;
    out.committed = on_disk > out.size ? on_disk : out.size;   // a longer file (an aborted earlier append) is fine
    wrote_bytes = out.size - old.size;
    return true;
#endif
}

/// Verifies the tail hash of `p` against `b` (always: a part whose tail is wrong reads back as corrupted K/V) and, when
/// `full`, every whole block.  `b` holds exactly `p.size` bytes.
inline bool verify_part(const PartInfo& p, const ConversationBuffer& b, bool full, std::string& error) {
    if ((uint64_t) b.size() != p.size) { error = "size mismatch"; return false; }
    if (full) {
        for (size_t i = 0; i < p.blocks.size(); ++i) {
            uint64_t h = 0xcbf29ce484222325ull;
            b.visit(i * kBlock, kBlock, [&](const uint8_t* q, size_t n, size_t) { h = block_hash(q, n, h); return true; });
            if (h != p.blocks[i]) { error = "block " + std::to_string(i) + " hash mismatch"; return false; }
        }
    }
    if (p.tail_len != 0 && hash_range(b, (uint64_t) p.blocks.size() * kBlock, (size_t) p.tail_len) != p.tail_hash) {
        error = "tail hash mismatch";
        return false;
    }
    return true;
}

/// Reads `size` bytes of `path` into `b` (the index promises `committed` bytes on the disk, and at least `size`).
inline bool read_part(const std::filesystem::path& path, const PartInfo& p, ConversationBuffer& b, std::string& error) {
    const uint64_t size = p.size;
    std::error_code ec;
    const auto on_disk = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec || on_disk < p.committed || on_disk < size) {
        error = "spill parts: " + path.string() + " is shorter than the index names";
        return false;
    }
    b.resize((size_t) size);
    FILE* fh = std::fopen(path.string().c_str(), "rb");
    if (!fh) { error = "spill parts: cannot read " + path.string(); return false; }
    bool ok = true;
    size_t at = 0;
    while (ok && at < size) {
        ok = b.visit(at, (size_t) (size - at), [&](uint8_t* q, size_t n, size_t) { return std::fread(q, 1, n, fh) == n; });
        at = size;
    }
    std::fclose(fh);
    if (!ok) { error = "spill parts: short read of " + path.string(); return false; }
    return true;
}

/// The whole entry's bytes on the disk (state generation, parts, ids, index) - what an adopted folder costs.
inline uint64_t folder_bytes(const std::filesystem::path& dir) {
    std::error_code ec;
    uint64_t n = 0;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (it->is_regular_file(e2)) n += (uint64_t) it->file_size(e2);
    }
    return n;
}

/// The serial a parts folder is named after ("parts-<n>"), 0 when it does not carry one.
inline uint64_t serial_of(const std::filesystem::path& dir) {
    const std::string n = dir.filename().string();
    if (n.rfind("parts-", 0) != 0) return 0;
    try {
        return std::stoull(n.substr(6));
    } catch (...) {
        return 0;
    }
}

/// Removes what a crashed writer left in `dir` next to the committed generation `keep_gen`: temporary files and other
/// state generations.  Never touches the index, ids.bin or the K/V parts.
inline void cleanup_folder(const std::filesystem::path& dir, uint64_t keep_gen) {
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string n = it->path().filename().string();
        std::error_code e2;
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) {
            std::filesystem::remove(it->path(), e2);
            continue;
        }
        if (n == "state.session") {
            if (keep_gen != 0) std::filesystem::remove(it->path(), e2);
            continue;
        }
        if (n.rfind("state-", 0) == 0 && n.size() > 12 && n.compare(n.size() - 8, 8, ".session") == 0) {
            uint64_t gen = 0;
            try {
                gen = std::stoull(n.substr(6, n.size() - 6 - 8));
            } catch (...) {
                gen = ~uint64_t(0);
            }
            if (gen != keep_gen) std::filesystem::remove(it->path(), e2);
        }
    }
}

/// Verifies every block hash and the tail of an entry an earlier process wrote (the adoption check; it reads the parts
/// once, at start).  False with `error` naming the first bad block.
inline bool verify_entry(const std::filesystem::path& dir, const Meta& m, std::string& error) {
    for (size_t i = 0; i < m.layers.size(); ++i) {
        for (size_t p = 0; p < 5; ++p) {
            const PartInfo& part = m.layers[i].parts[p];
            ConversationBuffer b;
            if (!read_part(part_path(dir, i, p), part, b, error)) return false;
            std::string verr;
            if (!verify_part(part, b, true, verr)) {
                error = "layer " + std::to_string(i) + " part " + std::to_string(p) + ": " + verr;
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// the entry
struct Placement {
    std::filesystem::path dir;   // the entry's folder (append: the previous entry's own folder)
    uint64_t serial = 0;         // the entry's serial (append: the previous entry's serial)
    bool appended = false;       // true: the caller must not remove `dir` on failure (it is the previous generation)
};

/// Writes `image` as a parts entry under `ns_dir`, extending `prev` (its serial `prev_serial`) in place when it is a
/// compatible prefix of the new one.  On success `place` names the entry and `bytes_out` is what it takes on the disk.
/// On failure nothing new is listed anywhere and the previous generation (when there was one) is still readable; a
/// fresh folder is removed, and an append leaves at most unreferenced bytes past the previous index's extents.
inline bool write_entry(const std::filesystem::path& ns_dir, uint64_t fresh_serial, const std::filesystem::path& prev,
                        uint64_t prev_serial, const SavedConversation& image, const SessionFileIdentity& id,
                        uint64_t chain, Placement& place, uint64_t& bytes_out, std::string& error) {
#if defined(_WIN32)
    (void) ns_dir; (void) fresh_serial; (void) prev; (void) prev_serial; (void) image; (void) id; (void) chain;
    (void) place; (void) bytes_out;
    error = "spill parts: not supported on this platform";
    return false;
#else
    const bool debug = std::getenv("STRATA_SPILL_DEBUG") != nullptr;
    // The new K/V hashed once: these numbers decide the append and, when the write is fresh, become the index.
    std::vector<LayerInfo> now(image.kv.size());
    for (size_t i = 0; i < image.kv.size(); ++i) {
        const auto& k = image.kv[i];
        now[i].head = {int64_t(k.format), k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim};
        for (size_t p = 0; p < 5; ++p) hash_part(part_of(k, p), now[i].parts[p]);
    }
    // A previous entry may be extended in place only when its files are a byte prefix of every new part.
    Meta old;
    bool append = false;
    if (!prev.empty() && read_meta(prev / "parts.idx", old, error)) {
        // The chain is a lookup key only: append safety comes from the per-part byte-prefix check below, so a candidate
        // found by ids (a conversation shorter than chain_of's 4096-token window) may store a different chain - the index
        // written at commit carries the new one.
        append = old.layers.size() == now.size() && old.model == id.model && old.config == id.config &&
                 old.cvec == (image.cvec ? 1u : 0u);
        for (size_t i = 0; append && i < now.size(); ++i) {
            // The context growing is what an append is for: the cell counts may grow, the geometry may not change.
            const auto& oh = old.layers[i].head;
            const auto& nh = now[i].head;
            if (oh[0] != nh[0] || oh[2] != nh[2] || oh[3] != nh[3] || oh[4] != nh[4] || oh[6] != nh[6] || nh[1] < oh[1] ||
                nh[5] < oh[5]) {
                append = false;
                break;
            }
            for (size_t p = 0; append && p < 5; ++p) {
                std::string why;
                if (!append_safe(old.layers[i].parts[p], now[i].parts[p], part_of(image.kv[i], p), part_path(prev, i, p), why)) {
                    append = false;
                    if (debug)
                        std::fprintf(stderr, "spill parts: fresh (layer %zu part %zu: %s)\n", i, p, why.c_str());
                }
            }
        }
    }
    error.clear();
    if (append) {
        place.dir = prev;
        place.serial = prev_serial;
        place.appended = true;
    } else {
        uint64_t serial = fresh_serial;
        std::error_code ec;
        std::filesystem::path dir = ns_dir / ("parts-" + std::to_string(serial));
        while (std::filesystem::exists(dir, ec)) dir = ns_dir / ("parts-" + std::to_string(++serial));
        std::filesystem::create_directories(dir, ec);
        if (ec) { error = "spill parts: cannot make " + dir.string(); return false; }
        place.dir = dir;
        place.serial = serial;
    }
    // (a) this generation's running state, under its own name: the previous generation's file is untouched.
    const uint64_t state_gen = next_state_gen(append ? old.state_gen : 0);
    auto abandon = [&](const std::string& why) {
        error = why;
        std::error_code ec;
        if (place.appended) {
            // The previous generation's index, state file and part extents are untouched, so it is still what a reader
            // gets.  Take back this generation's running state (no index names it); bytes past the old extents are
            // unreferenced and get rewritten by the next append.
            std::filesystem::remove(state_file(place.dir, state_gen), ec);
            return;
        }
        std::filesystem::remove_all(place.dir, ec);
    };
    SavedConversation state;
    state.geometry = image.geometry;
    state.layer_lo = image.layer_lo;
    state.layer_hi = image.layer_hi;
    state.cvec = image.cvec;
    state.live = image.live;
    state.checkpoints = image.checkpoints;
    size_t state_bytes = 0;
    std::string werr;
    if (!session_file_write(state_file(place.dir, state_gen).string(), state, id, state_bytes, werr)) {
        abandon(werr);
        return false;
    }
    // (b) the parts: append the bytes past the old size, or write whole
    uint64_t total = state_bytes, wrote = state_bytes;
    for (size_t i = 0; i < image.kv.size(); ++i) {
        for (size_t p = 0; p < 5; ++p) {
            PartInfo& part = now[i].parts[p];
            uint64_t wrote_part = 0;
            const ConversationBuffer& b = part_of(image.kv[i], p);
            const bool ok = append ? write_part_append(part_path(place.dir, i, p), b, old.layers[i].parts[p], part, wrote_part, werr)
                                   : write_part_fresh(part_path(place.dir, i, p), b, part, wrote_part, werr);
            if (!ok) {
                abandon(werr);
                return false;
            }
            if (append) part.reused = old.layers[i].parts[p].size;
            total += part.size;
            wrote += wrote_part;
            if (debug)
                std::fprintf(stderr, "spill parts dbg: layer %zu part %d size %llu MiB reused %llu MiB old %llu MiB\n", i,
                             (int) p, (unsigned long long) (part.size >> 20), (unsigned long long) (part.reused >> 20),
                             (unsigned long long) (append ? old.layers[i].parts[p].size >> 20 : 0));
        }
    }
    // (c) the ids sidecar - what lets the next process match a prompt against this folder.  Not required: without it the
    // entry is bytes-only.  Its live ids are bound in the index below, so a stale sidecar can never claim this K/V.
    uint64_t ids_len = 0, ids_hash = 0;
    std::string ierr;
    if (!write_ids(place.dir, image, ids_len, ids_hash, ierr)) {
        ids_len = ids_hash = 0;
        if (debug) std::fprintf(stderr, "spill parts: %s: %s (the entry was written without its ids)\n",
                                place.dir.filename().string().c_str(), ierr.c_str());
    }
    // (d) everything above durable before the index may name it
    if (!flush_dir(place.dir)) {
        abandon("spill parts: cannot flush " + place.dir.string());
        return false;
    }
    // (e) THE COMMIT: the index, atomically renamed into place
    Meta meta;
    meta.model = id.model;
    meta.config = id.config;
    meta.chain = chain;
    meta.cvec = image.cvec ? 1u : 0u;
    meta.ids_len = ids_len;
    meta.ids_hash = ids_hash;
    meta.state_gen = state_gen;
    meta.layers = std::move(now);
    if (!write_meta(place.dir / "parts.idx", meta, werr)) {
        abandon(werr);
        return false;
    }
    if (!flush_dir(place.dir)) {
        // The index is renamed but its folder entry may not survive a power loss; the entry is the truth from here on.
        if (debug) std::fprintf(stderr, "spill parts: %s: parts.idx flush failed (published, durability unknown)\n",
                                place.dir.filename().string().c_str());
    }
    // (f) the previous generation is garbage now
    if (append && old.state_gen != state_gen) {
        std::error_code ec;
        std::filesystem::remove(state_file(place.dir, old.state_gen), ec);
    }
    {
        std::error_code ec;
        total += (uint64_t) std::filesystem::file_size(place.dir / "parts.idx", ec);
        total += (uint64_t) std::filesystem::file_size(place.dir / "ids.bin", ec);
    }
    bytes_out = total;
    if (debug)
        std::fprintf(stderr,
                     "spill parts: entry %s: %llu MiB on the disk, %llu MiB written (%llu MiB kept), state %llu MiB, %s\n",
                     place.dir.filename().string().c_str(), (unsigned long long) (total >> 20),
                     (unsigned long long) (wrote >> 20), (unsigned long long) ((total - wrote) >> 20),
                     (unsigned long long) (state_bytes >> 20), append ? "appended in place" : "written fresh");
    return true;
#endif
}

/// Reads a parts entry back into `out`.  The state file's read limits and identity apply; the parts are bounded by the
/// caller's per-layer limits like the session file's payload is.  The tail hash is always verified
/// (STRATA_SPILL_VERIFY=all additionally verifies every block), so a torn or edited part reads as an error, never as a
/// different conversation's K/V.
inline bool read_entry(const std::filesystem::path& dir, const SessionFileIdentity& id, const SessionReadLimits& limits,
                       SavedConversation& out, uint64_t& bytes_out, std::string& error) {
#if defined(_WIN32)
    (void) dir; (void) id; (void) limits; (void) out; (void) bytes_out;
    error = "spill parts: not supported on this platform";
    return false;
#else
    Meta meta;
    if (!read_meta(dir / "parts.idx", meta, error)) return false;
    if (meta.model != id.model || meta.config != id.config) { error = "spill parts: another model or config"; return false; }
    SavedConversation state;
    size_t state_bytes = 0;
    if (!session_file_read(state_file(dir, meta.state_gen).string(), id, state, state_bytes, error, limits)) return false;
    if (!state.kv.empty()) { error = "spill parts: the state file carries K/V"; return false; }
    if (meta.layers.size() > limits.max_kv_layers) { error = "spill parts: more K/V layers than this engine's limit"; return false; }
    state.kv.resize(meta.layers.size());
    const char* verify = std::getenv("STRATA_SPILL_VERIFY");
    const bool full = verify != nullptr && std::string(verify) == "all";
    uint64_t total = state_bytes;
    for (size_t i = 0; i < meta.layers.size(); ++i) {
        auto& k = state.kv[i];
        const auto& l = meta.layers[i];
        if (limits.geometry && state.geometry != *limits.geometry) { error = "spill parts: another geometry"; return false; }
        k.format = (int) l.head[0];
        k.cells = l.head[1]; k.heads = l.head[2]; k.head_dim = l.head[3];
        k.page_size = l.head[4]; k.pooled_rows = l.head[5]; k.idx_dim = l.head[6];
        for (size_t p = 0; p < 5; ++p) {
            const PartInfo& part = l.parts[p];
            const uint64_t cap = i < limits.max_kv_bytes.size() ? limits.max_kv_bytes[i][p] : UINT64_MAX;
            if (part.size > cap) { error = "spill parts: a K/V part exceeds this engine's limit"; return false; }
            ConversationBuffer& b = part_of(k, p);
            if (!read_part(part_path(dir, i, p), part, b, error)) return false;
            if (!verify_part(part, b, full, error)) {
                error = "spill parts: " + part_path(dir, i, p).filename().string() + ": " + error;
                return false;
            }
            total += part.size;
        }
    }
    bytes_out = total;
    out = std::move(state);
    return true;
#endif
}

}  // namespace strata::core::spill_parts
