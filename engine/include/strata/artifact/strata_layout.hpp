// include/strata/artifact/strata_layout.hpp - the two header layouts of a .strata v1.x file, and the one place that tells them apart.
//
//   4 KiB layout  (every file shipped up to 0.2.22): section table at 0x10, engram sha at 0x190, sections 4 KiB aligned.
//   16 KiB layout (the re-adjusted Inkling SQ-20 and later): section table at 0x20 (the 16 bytes at 0x10 hold a u16
//                 tile_layout_id and are otherwise not read), engram sha at 0x1A0, sections 16 KiB aligned, and
//                 Section 3 padded up to a 16 KiB multiple, so its byte length no longer says how many entries it holds.
//
// Both start with the same magic, flags, profile and num_sections, and nothing in them names the layout, so it is decided
// by checking both tables against what a real table must satisfy (strict: every section_id equals its index, every
// non-empty section is aligned to the layout's page, lies inside the file and does not overlap another one).
// A header that satisfies both, or neither, is refused with an error that says why, never guessed.  The old 4 KiB
// check (aligned, section 0 present) is kept as the fallback for a header that is old-looking but not strict, so a
// truncated old file still opens as it always did and fails later where it is short.
//
// Header only and free of zstd and of the file class: the Mac no-zstd reader, the SYCL / Metal Inkling loaders and the
// Hy4 CPU expert reader parse the same 4 KiB frame by hand and all call this instead of their own 0x10.
#pragma once

#include "strata/artifact/strata_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::artifact {

// Bytes of the frame the parser needs: magic, flags, profile, count, 0x20 + 6 x 64 section table, 32 B engram sha.
constexpr size_t STRATA_HEADER_NEED = 0x20 + STRATA_NUM_SECTIONS * sizeof(StrataSectionEntry) + 32;

namespace layout_detail {

struct Probe {
    StrataSectionEntry sec[STRATA_NUM_SECTIONS]{};
    bool aligned = true;   // every non-empty section starts on the layout's page
    bool ids = true;       // section_id == index for all six
    bool ranges = true;    // non-empty: >= 4 KiB, no overflow, ends inside the file (when its size is known), no overlap
    bool any_section = false;  // at least one section has bytes
    std::string why;       // first thing that failed
    bool strict() const { return aligned && ids && ranges && any_section; }
};

inline Probe probe_table(const uint8_t* frame, size_t table_off, uint64_t page, uint64_t file_size) {
    Probe p;
    for (int i = 0; i < STRATA_NUM_SECTIONS; ++i)
        std::memcpy(&p.sec[i], frame + table_off + i * sizeof(StrataSectionEntry), sizeof(StrataSectionEntry));
    auto fail = [&](bool& flag, const std::string& msg) {
        flag = false;
        if (p.why.empty()) p.why = msg;
    };
    for (int i = 0; i < STRATA_NUM_SECTIONS; ++i) {
        const auto& s = p.sec[i];
        if (s.section_id != static_cast<uint32_t>(i))
            fail(p.ids, "entry " + std::to_string(i) + " has section_id " + std::to_string(s.section_id));
        if (s.byte_length == 0) continue;
        p.any_section = true;
        if (s.file_offset % page != 0)
            fail(p.aligned, "Section " + std::to_string(i) + " is not " + std::to_string(page / 1024) + " KiB aligned (offset " +
                                std::to_string(s.file_offset) + ")");
        if (s.file_offset < STRATA_PAGE_SIZE || s.file_offset + s.byte_length < s.file_offset)
            fail(p.ranges, "Section " + std::to_string(i) + " has an impossible offset " + std::to_string(s.file_offset));
        else if (file_size != 0 && s.file_offset + s.byte_length > file_size)
            fail(p.ranges, "Section " + std::to_string(i) + " ends past the end of the file (truncated file?)");
    }
    if (p.ranges) {
        const StrataSectionEntry* v[STRATA_NUM_SECTIONS];
        size_t n = 0;
        for (int i = 0; i < STRATA_NUM_SECTIONS; ++i)
            if (p.sec[i].byte_length != 0) v[n++] = &p.sec[i];
        std::sort(v, v + n, [](const StrataSectionEntry* a, const StrataSectionEntry* b) { return a->file_offset < b->file_offset; });
        for (size_t k = 1; k < n; ++k)
            if (v[k]->file_offset < v[k - 1]->file_offset + v[k - 1]->byte_length) {
                fail(p.ranges, "Sections " + std::to_string(v[k - 1]->section_id) + " and " + std::to_string(v[k]->section_id) + " overlap");
                break;
            }
    }
    return p;
}

}  // namespace layout_detail

// Parse the 4 KiB header frame of a .strata file into `out` (magic, flags, profile, section table, engram sha, layout).
// `file_size` is the size of the file, or 0 when it is not known (then the in-file checks are skipped).
// False with `err` set on any header that is neither layout; `out` is only touched on success.
inline bool strata_parse_header_frame(const uint8_t* frame, size_t frame_len, uint64_t file_size, StrataHeader& out, std::string& err) {
    static const uint8_t kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};
    if (frame_len < STRATA_HEADER_NEED) {
        err = "Short .strata header frame";
        return false;
    }
    if (std::memcmp(frame, kMagic, 8) != 0) {
        err = "Invalid .strata magic (expected STRATA\\x00\\x01)";
        return false;
    }
    StrataHeader h{};
    uint16_t prof = 0;
    std::memcpy(&h.flags, frame + 8, 4);
    std::memcpy(&prof, frame + 12, 2);
    h.profile = static_cast<StrataProfile>(prof);
    std::memcpy(&h.num_sections, frame + 14, 2);
    if (h.num_sections != STRATA_NUM_SECTIONS) {
        err = "Unexpected .strata num_sections: " + std::to_string(h.num_sections);
        return false;
    }
    const layout_detail::Probe a = layout_detail::probe_table(frame, STRATA_TABLE_OFFSET_4K, STRATA_PAGE_SIZE, file_size);
    const layout_detail::Probe b = layout_detail::probe_table(frame, STRATA_TABLE_OFFSET_16K, STRATA_PAGE_SIZE_16K, file_size);
    const layout_detail::Probe* pick = nullptr;
    if (a.strict() && b.strict()) {
        err = "Ambiguous .strata header: the section table reads as both the 4 KiB layout (at 0x10) and the 16 KiB layout (at 0x20)";
        return false;
    }
    if (b.strict()) {
        pick = &b;
        h.layout = StrataLayout::Page16K;
    } else if (a.aligned && a.ids && a.any_section) {
        // the old rule (4 KiB aligned) plus section_id == index, which every writer sets, and at least one section: a torn
        // or foreign 16 KiB header read at 0x10 comes out as empty sections and must not pass as an old file with nothing in it
        pick = &a;
        h.layout = StrataLayout::Page4K;
    }
    if (!pick) {
        err = "Invalid .strata header: the section table is neither the 4 KiB layout (" + (a.why.empty() ? std::string("no section has any bytes") : a.why) +
              ") nor the 16 KiB layout (" + (b.why.empty() ? std::string("no section has any bytes") : b.why) + ")";
        return false;
    }
    h.page_size = static_cast<uint32_t>(h.layout == StrataLayout::Page16K ? STRATA_PAGE_SIZE_16K : STRATA_PAGE_SIZE);
    h.section_table_offset = static_cast<uint32_t>(h.layout == StrataLayout::Page16K ? STRATA_TABLE_OFFSET_16K : STRATA_TABLE_OFFSET_4K);
    if (h.layout == StrataLayout::Page16K) std::memcpy(&h.tile_layout_id, frame + 0x10, sizeof(uint16_t));
    for (int i = 0; i < STRATA_NUM_SECTIONS; ++i) h.sections[i] = pick->sec[i];
    std::memcpy(h.engram_shard_sha256, frame + h.section_table_offset + STRATA_NUM_SECTIONS * sizeof(StrataSectionEntry), 32);
    out = h;
    return true;
}

// How many directory entries Section 3 holds.  Entry = 32 B, or 34 B (32 B + a u16 vram_load_order kept after all the
// entries) when section 3's flag bit 0 is set (`has_order`).
//   4 KiB layout: the byte length is exact, as it always was (a length that is not a multiple is an error).
//   16 KiB layout: the length is padded up, so the count is the manifest's (num_physical_layers x num_experts_per_layer,
//                  `manifest_n`); it must fit with less than one page of padding.  Without a manifest count the length is
//                  taken as exact when it divides evenly (a caller may then trim an all-zero tail), and refused otherwise.
inline bool strata_section3_entry_count(const StrataHeader& h, uint64_t s3_bytes, bool has_order, uint64_t manifest_n, size_t& n, std::string& err) {
    const uint64_t entry = has_order ? sizeof(StrataExpertEntry) + sizeof(uint16_t) : sizeof(StrataExpertEntry);
    const uint64_t exact = s3_bytes / entry;
    if (h.layout == StrataLayout::Page16K && manifest_n != 0) {
        if (manifest_n > exact || s3_bytes - manifest_n * entry >= h.page_size) {
            err = "Section 3 (" + std::to_string(s3_bytes) + " B) does not match the manifest's " + std::to_string(manifest_n) + " expert entries";
            return false;
        }
        n = static_cast<size_t>(manifest_n);
        return true;
    }
    if (s3_bytes % entry != 0) {
        err = has_order ? "Section 3 v1.1 byte_length not a multiple of 34 B (32 B entry + 2 B vram_load_order)"
                        : "Section 3 byte_length not a multiple of 32 B StrataExpertEntry";
        return false;
    }
    n = static_cast<size_t>(exact);
    return true;
}

// Size of an open file (0 when it cannot be told); the position is put back.
inline uint64_t strata_file_size(std::FILE* fp) noexcept {
    if (!fp) return 0;
#if defined(_WIN32)
    const __int64 here = _ftelli64(fp);
    if (here < 0 || _fseeki64(fp, 0, SEEK_END) != 0) return 0;
    const __int64 end = _ftelli64(fp);
    _fseeki64(fp, here, SEEK_SET);
#else
    const off_t here = ftello(fp);
    if (here < 0 || fseeko(fp, 0, SEEK_END) != 0) return 0;
    const off_t end = ftello(fp);
    fseeko(fp, here, SEEK_SET);
#endif
    return end > 0 ? static_cast<uint64_t>(end) : 0;
}

// A top-level `"key": <number>` or `"key": "string"` of the manifest JSON, found by walking the text (strings and nesting
// skipped), so a key inside the chat template or a nested object is never taken for the top-level one.
// False when the key is absent or not that kind of value.  Not a general JSON parser: the manifest is already known to be
// a JSON object; this only reads two scalars before the full parse is available.
inline bool strata_manifest_top_value(const std::string& json, const char* key, bool want_string, std::string& text) {
    size_t i = 0;
    const size_t n = json.size();
    int depth = 0;
    auto skip_string = [&](size_t& p, std::string* out) {  // p at the opening quote; ends after the closing one
        ++p;
        while (p < n && json[p] != '"') {
            if (json[p] == '\\' && p + 1 < n) {
                if (out) out->push_back(json[p + 1]);
                p += 2;
            } else {
                if (out) out->push_back(json[p]);
                ++p;
            }
        }
        if (p < n) ++p;
    };
    while (i < n) {
        const char c = json[i];
        if (c == '{' || c == '[') {
            ++depth;
            ++i;
        } else if (c == '}' || c == ']') {
            --depth;
            ++i;
        } else if (c == '"') {
            std::string s;
            skip_string(i, depth == 1 ? &s : nullptr);
            if (depth != 1 || s != key) continue;
            size_t j = i;
            while (j < n && (json[j] == ' ' || json[j] == '\t' || json[j] == '\r' || json[j] == '\n')) ++j;
            if (j >= n || json[j] != ':') continue;  // a string value that happens to equal the key
            ++j;
            while (j < n && (json[j] == ' ' || json[j] == '\t' || json[j] == '\r' || json[j] == '\n')) ++j;
            if (j >= n) return false;
            text.clear();
            if (want_string) {
                if (json[j] != '"') return false;
                skip_string(j, &text);
                return true;
            }
            size_t e = j;
            while (e < n && json[e] >= '0' && json[e] <= '9') ++e;
            if (e == j) return false;
            if (e - j > 19) text.assign(19, '9');  // more digits than a u64 holds: some huge value, not an absent key
            else text.assign(json, j, e - j);
            return true;
        } else {
            ++i;
        }
    }
    return false;
}

// Entries Section 3 must hold according to the manifest (num_physical_layers x num_experts_per_layer), 0 when it says nothing,
// and UINT64_MAX when it says something that cannot be (a zero, or a factor above 2^20): strata_section3_entry_count refuses
// that, since no Section 3 can hold it.  Each factor is capped so the product and the product times 34 cannot wrap.
inline uint64_t strata_manifest_expert_count(const std::string& json) {
    std::string a, b;
    if (!strata_manifest_top_value(json, "num_physical_layers", false, a) || !strata_manifest_top_value(json, "num_experts_per_layer", false, b)) return 0;
    const unsigned long long nl = std::strtoull(a.c_str(), nullptr, 10), ne = std::strtoull(b.c_str(), nullptr, 10);
    if (nl == 0 || ne == 0 || nl > (1u << 20) || ne > (1u << 20)) return UINT64_MAX;
    return static_cast<uint64_t>(nl) * ne;
}

}  // namespace strata::artifact
