// The kept-conversation (slot) bookkeeping that needs no GPU: a slot's .meta file (identity, size, tokens, checkpoint
// positions, pictures) and the choice of which kept conversation a prompt continues, and from where.  The engine's
// serve loop (ds4_main.cpp) uses these; tests/dsv41/slot_test.cpp checks them alone.
//
// Slot .meta layout (little endian), "DS4META3":
//   u64 magic, u64 identity length, identity bytes, u64 .bin bytes, u64 token count, i32 tokens[],
//   u64 checkpoint count (0..19), i64 checkpoints[],
//   u64 picture count (0..token count), {i64 start, i64 len, u64 hash} pictures[], u64 magic
// The magic at both ends and the .bin size make a half-written or foreign file refused, never read.  The pictures are
// the slot's image spans, whole, sorted by start, not overlapping and inside the tokens; a file whose spans are not is
// refused.  "DS4META2" files (no picture list) are refused too: their image tokens could not be told from another
// picture's.
#pragma once

#include "ds4_images.hpp"   // ImageSpan, image_prefix, image_spans_valid
#include "ds4_marks.hpp"    // kMarksMax

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace strata::ds4::slotmeta {

constexpr uint64_t kMagic = 0x334154454d345344ull;   // "DS4META3"

// Written under a temporary name and renamed over `path` (an interrupted write leaves no half file).  false (nothing
// written) for pictures that are not whole spans of the tokens.
inline bool write(const std::string& path, const std::string& identity, const std::vector<int32_t>& toks,
                  uint64_t bytes, const std::vector<int64_t>& cks, const std::vector<ImageSpan>& images = {}) {
    if (!image_spans_valid(images, (int64_t) toks.size())) return false;
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        const uint64_t il = identity.size(), nt = toks.size(), nc = cks.size(), ni = images.size();
        f.write((const char*) &kMagic, 8);
        f.write((const char*) &il, 8);
        f.write(identity.data(), (std::streamsize) il);
        f.write((const char*) &bytes, 8);
        f.write((const char*) &nt, 8);
        f.write((const char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t)));
        f.write((const char*) &nc, 8);
        f.write((const char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t)));
        f.write((const char*) &ni, 8);
        for (const ImageSpan& s : images) {
            f.write((const char*) &s.start, 8);
            f.write((const char*) &s.len, 8);
            f.write((const char*) &s.hash, 8);
        }
        f.write((const char*) &kMagic, 8);
        f.flush();
        if (!f) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

// false: foreign (another identity or format), damaged, cut short, or not plausible (no tokens, more than max_ctx, a
// checkpoint outside the tokens, pictures that are not whole sorted spans of the tokens).  On false every output is
// emptied (bytes 0).
inline bool read(const std::string& path, const std::string& identity, int64_t max_ctx, std::vector<int32_t>& toks,
                 uint64_t& bytes, std::vector<int64_t>& cks, std::vector<ImageSpan>& images) {
    const auto fail = [&] {
        toks.clear();
        cks.clear();
        images.clear();
        bytes = 0;
        return false;
    };
    std::ifstream f(path, std::ios::binary);
    uint64_t m0 = 0, il = 0, nt = 0, nc = 0, ni = 0, m1 = 0;
    if (!f.read((char*) &m0, 8) || m0 != kMagic || !f.read((char*) &il, 8) || il != identity.size()) return fail();
    std::string id(il, '\0');
    if (!f.read(id.data(), (std::streamsize) il) || id != identity) return fail();
    if (max_ctx <= 0 || !f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (uint64_t) max_ctx)
        return fail();
    toks.resize(nt);
    if (!f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) || !f.read((char*) &nc, 8) ||
        nc > 3 + (uint64_t) kMarksMax)
        return fail();
    cks.resize(nc);
    if (!f.read((char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t))) || !f.read((char*) &ni, 8) || ni > nt)
        return fail();
    images.resize(ni);
    for (ImageSpan& s : images)
        if (!f.read((char*) &s.start, 8) || !f.read((char*) &s.len, 8) || !f.read((char*) &s.hash, 8)) return fail();
    if (!f.read((char*) &m1, 8) || m1 != kMagic) return fail();
    for (int64_t c : cks)
        if (c <= 0 || (uint64_t) c > nt) return fail();
    if (!image_spans_valid(images, (int64_t) nt)) return fail();
    return true;
}

// The old form, for callers that know nothing of pictures: a slot that holds pictures is refused (its image tokens
// alone would pass for any other picture's).
inline bool read(const std::string& path, const std::string& identity, int64_t max_ctx, std::vector<int32_t>& toks,
                 uint64_t& bytes, std::vector<int64_t>& cks) {
    std::vector<ImageSpan> images;
    if (!read(path, identity, max_ctx, toks, bytes, cks, images)) return false;
    if (images.empty()) return true;
    toks.clear();
    cks.clear();
    bytes = 0;
    return false;
}

// True when `pos` falls strictly inside one of `spans` (a resume there would split a picture).  Any order.
inline bool inside_image(const std::vector<ImageSpan>& spans, int64_t pos) {
    for (const ImageSpan& s : spans)
        if (s.start < pos && pos - s.start < s.len) return true;
    return false;
}

// The kept conversation `prompt` continues the furthest, and where: *len is the number of the slot's tokens that are
// the prompt's (0: none).  A conversation the prompt continues whole comes back whole (len == its size); one the prompt
// leaves part way - a retried prompt, a history re-rendered without the earlier reasoning - comes back at its latest
// checkpoint the prompt still agrees with.  Only a match longer than `floor` (what the model itself can go back to)
// counts.  The last prompt token is never covered (a prompt always reads its last token).  Ties keep the earlier slot.
// Pictures: the tokens agreed on are cut to the prefix where the slot's pictures and the request's (`images`, the
// prompt's spans) are the same pictures at the same places (image_prefix), and a resume point never falls inside a
// picture of either.  Without pictures on either side the choice is the text-only one.
struct Cand {
    const std::vector<int32_t>* tokens;
    const std::vector<int64_t>* cks;
    const std::vector<ImageSpan>* images = nullptr;   // the slot's pictures (nullptr: none)
};
inline size_t pick(const std::vector<Cand>& slots, const std::vector<int>& prompt, size_t floor, size_t* len,
                   const std::vector<ImageSpan>& images = {}) {
    *len = 0;
    if (prompt.size() < 2) return SIZE_MAX;
    static const std::vector<ImageSpan> kNone;
    size_t best = SIZE_MAX;
    for (size_t i = 0; i < slots.size(); ++i) {
        const std::vector<int32_t>& t = *slots[i].tokens;
        const std::vector<ImageSpan>& si = slots[i].images ? *slots[i].images : kNone;
        const size_t lim = std::min(t.size(), prompt.size() - 1);
        size_t c = 0;
        while (c < lim && t[c] == prompt[c]) ++c;
        if (!si.empty() || !images.empty()) {
            const int64_t p = image_prefix(si, images, (int64_t) c);
            c = p <= 0 ? 0 : std::min(c, (size_t) p);
        }
        // a resume point: agreed on, and splitting no picture of the slot or the request
        const auto ok_at = [&](size_t k) {
            return k <= c && !inside_image(si, (int64_t) k) && !inside_image(images, (int64_t) k);
        };
        size_t u = c == t.size() && ok_at(c) ? c : 0;
        if (u == 0)
            for (int64_t k : *slots[i].cks)
                if (k > 0 && (size_t) k > u && ok_at((size_t) k)) u = (size_t) k;
        if (u > *len && u > floor) {
            best = i;
            *len = u;
        }
    }
    return best;
}

}  // namespace strata::ds4::slotmeta
