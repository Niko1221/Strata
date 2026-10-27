// The kept-conversation (slot) bookkeeping that needs no GPU: a slot's .meta file (identity, size, tokens, checkpoint
// positions) and the choice of which kept conversation a prompt continues, and from where.  The engine's serve loop
// (ds4_main.cpp) uses these; tests/dsv41/slot_test.cpp checks them alone.
//
// Slot .meta layout (little endian), "DS4META2":
//   u64 magic, u64 identity length, identity bytes, u64 .bin bytes, u64 token count, i32 tokens[],
//   u64 checkpoint count (0..4), i64 checkpoints[], u64 magic
// The magic at both ends and the .bin size make a half-written or foreign file refused, never read.
//
// A slot is the user's own conversation (a .meta's tokens decode to the prompt, a .bin holds its state), so on POSIX
// every slot file is its owner's alone: created 0600 (create_private), and a previous version's files made so at start
// (make_private).  On Windows the folder's inherited ACL decides.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::ds4::slotmeta {

constexpr uint64_t kMagic = 0x324154454d345344ull;   // "DS4META2"

// A new, empty file only the owner can read and write, for a stream to open and truncate next (the mode stays): the
// mode is set as the file is made, so no other user can open it in between.  A file of that name is removed first - a
// mode only applies to a file being created, and an earlier interrupted write may have left one with a looser mode.
inline bool create_private(const std::string& path) {
#ifndef _WIN32
    ::unlink(path.c_str());
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    ::close(fd);
#else
    (void) path;
#endif
    return true;
}

// An existing slot file (a previous version wrote them with the umask's mode, often readable by every user) made the
// owner's alone; false when that failed.
inline bool make_private(const std::string& path) {
#ifndef _WIN32
    std::error_code ec;
    std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    return !ec;
#else
    (void) path;
    return true;
#endif
}

// Written under a temporary name and renamed over `path` (an interrupted write leaves no half file).
inline bool write(const std::string& path, const std::string& identity, const std::vector<int32_t>& toks,
                  uint64_t bytes, const std::vector<int64_t>& cks) {
    const std::string tmp = path + ".tmp";
    if (!create_private(tmp)) return false;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        const uint64_t il = identity.size(), nt = toks.size(), nc = cks.size();
        f.write((const char*) &kMagic, 8);
        f.write((const char*) &il, 8);
        f.write(identity.data(), (std::streamsize) il);
        f.write((const char*) &bytes, 8);
        f.write((const char*) &nt, 8);
        f.write((const char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t)));
        f.write((const char*) &nc, 8);
        f.write((const char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t)));
        f.write((const char*) &kMagic, 8);
        f.flush();
        if (!f) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

// false: foreign (another identity), damaged, cut short, or not plausible (no tokens, more than max_ctx, a checkpoint
// outside the tokens).
inline bool read(const std::string& path, const std::string& identity, int64_t max_ctx, std::vector<int32_t>& toks,
                 uint64_t& bytes, std::vector<int64_t>& cks) {
    std::ifstream f(path, std::ios::binary);
    uint64_t m0 = 0, il = 0, nt = 0, nc = 0, m1 = 0;
    if (!f.read((char*) &m0, 8) || m0 != kMagic || !f.read((char*) &il, 8) || il != identity.size()) return false;
    std::string id(il, '\0');
    if (!f.read(id.data(), (std::streamsize) il) || id != identity) return false;
    if (!f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (uint64_t) max_ctx) return false;
    toks.resize(nt);
    if (!f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) || !f.read((char*) &nc, 8) || nc > 4)
        return false;
    cks.resize(nc);
    if (!f.read((char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t))) || !f.read((char*) &m1, 8) || m1 != kMagic)
        return false;
    for (int64_t c : cks)
        if (c <= 0 || (uint64_t) c > nt) return false;
    return true;
}

// The kept conversation `prompt` continues the furthest, and where: *len is the number of the slot's tokens that are
// the prompt's (0: none).  A conversation the prompt continues whole comes back whole (len == its size); one the prompt
// leaves part way - a retried prompt, a history re-rendered without the earlier reasoning - comes back at its latest
// checkpoint the prompt still agrees with.  Only a match longer than `floor` (what the model itself can go back to)
// counts.  The last prompt token is never covered (a prompt always reads its last token).  Ties keep the earlier slot.
struct Cand {
    const std::vector<int32_t>* tokens;
    const std::vector<int64_t>* cks;
};
inline size_t pick(const std::vector<Cand>& slots, const std::vector<int>& prompt, size_t floor, size_t* len) {
    *len = 0;
    if (prompt.size() < 2) return SIZE_MAX;
    size_t best = SIZE_MAX;
    for (size_t i = 0; i < slots.size(); ++i) {
        const std::vector<int32_t>& t = *slots[i].tokens;
        const size_t lim = std::min(t.size(), prompt.size() - 1);
        size_t c = 0;
        while (c < lim && t[c] == prompt[c]) ++c;
        size_t u = c == t.size() ? c : 0;
        if (u == 0)
            for (int64_t k : *slots[i].cks)
                if ((size_t) k <= c && (size_t) k > u) u = (size_t) k;
        if (u > *len && u > floor) {
            best = i;
            *len = u;
        }
    }
    return best;
}

}  // namespace strata::ds4::slotmeta
