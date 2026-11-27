// tests/quant/strata_reader_layout_test.cpp - the .strata header layouts: the 4 KiB layout of every file up to 0.2.22 and
// the 16 KiB layout of the re-adjusted Inkling SQ-20 (section table at 0x20, 16 KiB aligned sections, padded Section 3).
// Synthetic files, no model and no GPU; the three golden header frames are copies of the real 0.2.22 files.
//
//   1. the real SQ-20, SQ-10 and Vision headers parse as the 4 KiB layout with the section table they always had;
//   2. a synthetic 4 KiB file opens through StrataFileReader as before (sections, manifest, directory, vram order);
//   3. a synthetic 16 KiB file opens: sections, padded Section 3 (count from the manifest, not length / 34), order, bodies;
//   4. the manifest is read by walking it: a chat template that spells a key, or a nested copy of it, is not the key;
//   5. neither layout is taken for the other, even when the bytes at 0x10 of a 16 KiB header look like an offset;
//   6. what fits neither layout, or both, is refused with a message, never opened and never a crash;
//   7. the Vision shard: header profile 3, or arch inkling2_vision in the manifest.
#include "strata/artifact/strata_layout.hpp"
#include "strata/artifact/strata_reader.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "strata_reader_layout_golden.inc"

using namespace strata::artifact;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

template <class T> void put(std::vector<uint8_t>& b, size_t at, T v) {
    if (b.size() < at + sizeof(T)) b.resize(at + sizeof(T), 0);
    std::memcpy(b.data() + at, &v, sizeof(T));
}
uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
// STRATA_LAYOUT_TEST_KEEP=1 keeps the two main synthetic files, so another build of the reader can be pointed at them.
void drop_keepable(const std::filesystem::path& p) {
    if (!std::getenv("STRATA_LAYOUT_TEST_KEEP")) std::filesystem::remove(p);
}

// A zstd frame holding `text` as one raw block (the bundled zstd is decode-only; the reader needs only a valid frame).
std::vector<uint8_t> zstd_raw_frame(const std::string& text) {
    std::vector<uint8_t> f;
    const uint32_t magic = 0xFD2FB528u, n = static_cast<uint32_t>(text.size());
    f.resize(4);
    std::memcpy(f.data(), &magic, 4);
    f.push_back(0xA0);  // single segment, 4-byte content size, no checksum, no dictionary
    for (int i = 0; i < 4; ++i) f.push_back(static_cast<uint8_t>(n >> (8 * i)));
    const uint32_t bh = 1u | (static_cast<uint32_t>(text.size()) << 3);  // last block, raw
    for (int i = 0; i < 3; ++i) f.push_back(static_cast<uint8_t>(bh >> (8 * i)));
    f.insert(f.end(), text.begin(), text.end());
    return f;
}

constexpr int kLayers = 4, kExperts = 3, kN = kLayers * kExperts;  // 12 directory entries

std::string manifest_text(const char* arch, int experts = kExperts) {
    // the chat template spells "num_physical_layers" with another value, and a nested object holds the key too: neither counts
    return std::string("{\"format\":\"strata-v1.1\",\"arch\":\"") + arch +
           "\",\"config\":{\"num_physical_layers\":99,\"vision_config\":{\"model_type\":\"inkling2_vision\"}},"
           "\"chat_template\":\"say \\\"num_physical_layers\\\": 77 here\","
           "\"num_physical_layers\":" + std::to_string(kLayers) + ",\"num_experts_per_layer\":" + std::to_string(experts) +
           ",\"build\":\"SQ-TEST\"}";
}

struct Spec {
    bool k16 = false;                  // the 16 KiB layout
    uint16_t profile = 0;
    bool order = true;                 // Section 3 carries the vram_load_order table (flags bit 0)
    int experts = kExperts;            // experts per layer: the directory holds kLayers x experts entries
    std::string manifest;              // empty: the standard one for `experts`
    bool pad_sec3 = true;              // 16 KiB layout: Section 3 padded to a page (always off for the 4 KiB layout)
    uint64_t s3_extra = 0;             // bytes added to Section 3's length (more padding than a page)
    uint16_t tile_id = 7;              // 16 KiB layout: the u16 at 0x10
    uint64_t ext0 = 0, ext1 = 0;       // 16 KiB layout: u64 at 0x10 (its low 16 bits replaced by tile_id) and u64 at 0x18, when not 0
    bool truncate_tail = false;        // cut the file inside Section 4
};

struct Built {
    std::filesystem::path path;
    uint64_t page = 0, off[6] = {}, len[6] = {}, size = 0;
};

// Order in the file, as in the shipped ones: Section 1, 4, 3, 5, 0 (the manifest last).
Built build(const char* name, const Spec& sp) {
    Built r;
    r.page = sp.k16 ? 16384 : 4096;
    std::vector<uint8_t> f(r.page, 0);
    uint64_t cur = r.page;
    auto place = [&](int id, uint64_t len) {
        r.off[id] = cur;
        r.len[id] = len;
        cur = align_up(cur + len, r.page);
    };
    const int n = kLayers * sp.experts;      // directory entries
    const int bodies = n < 12 ? n : 12;      // distinct bodies in Section 4 (entry k uses body k % 12)
    place(1, 100);
    place(4, static_cast<uint64_t>(bodies) * r.page);
    const uint64_t s3_entry = sp.order ? 34 : 32;
    place(3, ((sp.k16 && sp.pad_sec3) ? align_up(static_cast<uint64_t>(n) * s3_entry, r.page) : static_cast<uint64_t>(n) * s3_entry) + sp.s3_extra);
    place(5, 83);
    const std::vector<uint8_t> man = zstd_raw_frame(sp.manifest.empty() ? manifest_text("inkling2_loop", sp.experts) : sp.manifest);
    place(0, man.size());
    r.size = cur;
    f.resize(r.size, 0);

    // payloads
    std::memcpy(f.data() + r.off[0], man.data(), man.size());
    for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>(r.off[3]) + static_cast<size_t>(k) * 32;
        put<uint16_t>(f, e + 0, static_cast<uint16_t>((k % sp.experts + 1) % sp.experts));  // expert_id: a rank-ordered permutation
        f[e + 2] = 1;                                                                       // gu_code_id
        f[e + 3] = 2;                                                                       // dn_code_id
        put<uint64_t>(f, e + 8, static_cast<uint64_t>(k % bodies) * r.page);                // body_offset
        put<uint32_t>(f, e + 16, 64);                                                       // body_bytes
        put<uint32_t>(f, e + 20, 32);                                                       // dn_offset
        if (k < bodies) std::memset(f.data() + r.off[4] + static_cast<uint64_t>(k) * r.page, 0x40 + k, 64);  // the body
        if (sp.order) put<uint16_t>(f, static_cast<size_t>(r.off[3]) + static_cast<size_t>(n) * 32 + static_cast<size_t>(k) * 2, static_cast<uint16_t>(n - 1 - k));  // vram_load_order (mod 2^16)
    }

    // header frame
    const uint8_t magic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};
    std::memcpy(f.data(), magic, 8);
    put<uint32_t>(f, 8, 1);
    put<uint16_t>(f, 12, sp.profile);
    put<uint16_t>(f, 14, 6);
    const size_t table = sp.k16 ? 0x20 : 0x10;
    if (sp.k16) {
        if (sp.ext0) put<uint64_t>(f, 0x10, (sp.ext0 & ~0xFFFFull) | sp.tile_id);
        else put<uint16_t>(f, 0x10, sp.tile_id);
        if (sp.ext1) put<uint64_t>(f, 0x18, sp.ext1);
    }
    for (int i = 0; i < 6; ++i) {
        const size_t e = table + static_cast<size_t>(i) * 64;
        put<uint64_t>(f, e + 0, r.len[i] ? r.off[i] : 0);
        put<uint64_t>(f, e + 8, r.len[i]);
        for (int j = 0; j < 32; ++j) f[e + 16 + j] = static_cast<uint8_t>(0x10 + i * 7 + j);  // sha256 (not checked by the reader)
        put<uint32_t>(f, e + 48, static_cast<uint32_t>(i));
        put<uint32_t>(f, e + 52, (i == 3 && sp.order) ? 1u : 0u);
    }
    if (sp.truncate_tail) f.resize(r.off[4] + r.page);  // the directory and the manifest are gone: cut anywhere inside Section 4
    r.path = std::filesystem::temp_directory_path() / name;
    std::ofstream(r.path, std::ios::binary).write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return r;
}

// A frame with the magic and a table written by `fill` (offsets relative to 0), for the refusal cases.
std::vector<uint8_t> frame_with(uint16_t nsec = 6, uint16_t profile = 0) {
    std::vector<uint8_t> f(4096, 0);
    const uint8_t magic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};
    std::memcpy(f.data(), magic, 8);
    put<uint32_t>(f, 8, 1);
    put<uint16_t>(f, 12, profile);
    put<uint16_t>(f, 14, nsec);
    return f;
}
void entry(std::vector<uint8_t>& f, size_t table, int i, uint64_t off, uint64_t len, uint32_t sid) {
    const size_t e = table + static_cast<size_t>(i) * 64;
    put<uint64_t>(f, e, off);
    put<uint64_t>(f, e + 8, len);
    put<uint32_t>(f, e + 48, sid);
}
bool parse(const std::vector<uint8_t>& f, uint64_t size, StrataHeader& h, std::string& err) {
    err.clear();
    return strata_parse_header_frame(f.data(), f.size(), size, h, err);
}
std::vector<uint8_t> golden(const uint8_t* g) {
    std::vector<uint8_t> f(4096, 0);
    std::memcpy(f.data(), g, 0x1B0);
    return f;
}

struct Sec { uint64_t off, len; uint32_t flags; };

void check_golden(const char* name, const uint8_t* g, uint64_t size, uint16_t profile, const Sec (&want)[6]) {
    StrataHeader h;
    std::string err;
    const bool ok = parse(golden(g), size, h, err);
    std::printf("  %s\n", name);
    check(ok, "parses");
    if (!ok) {
        std::printf("    err: %s\n", err.c_str());
        return;
    }
    check(h.layout == StrataLayout::Page4K && h.page_size == 4096 && h.section_table_offset == 0x10, "4 KiB layout, table at 0x10");
    check(static_cast<uint16_t>(h.profile) == profile && h.flags == 1 && h.num_sections == 6, "profile, flags, num_sections");
    bool all = true;
    for (int i = 0; i < 6; ++i)
        all = all && h.sections[i].file_offset == want[i].off && h.sections[i].byte_length == want[i].len &&
              h.sections[i].flags == want[i].flags && h.sections[i].section_id == static_cast<uint32_t>(i);
    check(all, "all six section offsets, lengths, flags and ids as shipped");
    check(h.tile_layout_id == 0, "no 16 KiB id on a 4 KiB header");
}

}  // namespace

int main() {
    std::printf("strata_reader_layout_test\n");

    // 1. the real 0.2.22 headers -----------------------------------------------------------------------------------
    {
        const Sec sq20[6] = {{19346673664ull, 10375, 0}, {4096, 3670344256ull, 0}, {0, 0, 0},
                             {19346563072ull, 104448, 1}, {3670351872ull, 15676211200ull, 0}, {19346669568ull, 83, 0}};
        const Sec sq10[6] = {{10471514112ull, 10780, 0}, {4096, 1871628224ull, 0}, {0, 0, 0},
                             {10471403520ull, 104448, 1}, {1871634432ull, 8599769088ull, 0}, {10471510016ull, 138, 0}};
        const Sec vis[6] = {{636518400ull, 4097, 0}, {4096, 636512256ull, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        check_golden("Inkling-2-Mini-SQ-20.strata (19,346,685,952 B)", kGoldenSq20, 19346685952ull, 0, sq20);
        check_golden("Inkling-2-Mini-SQ-10.strata (10,471,526,400 B)", kGoldenSq10, 10471526400ull, 0, sq10);
        check_golden("Inkling-2-Mini-Vision.strata (636,526,592 B)", kGoldenVision, 636526592ull, 3, vis);
        StrataHeader h;
        std::string err;
        // the same frame with the file size not known, and with a file size that cuts Section 4 (a truncated download)
        check(parse(golden(kGoldenSq20), 0, h, err) && h.layout == StrataLayout::Page4K, "SQ-20 header with the file size unknown: 4 KiB layout");
        check(parse(golden(kGoldenSq20), 10000000000ull, h, err) && h.layout == StrataLayout::Page4K, "SQ-20 header on a truncated file: opens as before (reads fail later)");
        check(h.sections[3].byte_length == 104448 && h.sections[3].byte_length / 34 == 3072 && h.sections[3].byte_length % 34 == 0, "SQ-20 Section 3 = 3072 x 34 B exactly");
    }

    // 2. a synthetic 4 KiB file through the reader ---------------------------------------------------------------------
    {
        std::printf("  synthetic 4 KiB file\n");
        const Built b = build("strata_layout_4k.strata", Spec{});
        StrataFileReader rd;
        std::string err;
        const bool ok = rd.open(b.path.string(), err);
        check(ok, "opens");
        if (ok) {
            const auto& h = rd.header();
            check(h.layout == StrataLayout::Page4K && h.page_size == 4096 && h.tile_layout_id == 0, "4 KiB layout");
            bool secs = true;
            for (int i = 0; i < 6; ++i) secs = secs && h.sections[i].file_offset == (b.len[i] ? b.off[i] : 0) && h.sections[i].byte_length == b.len[i];
            check(secs, "section offsets and lengths");
            check(rd.manifest_json() == manifest_text("inkling2_loop"), "manifest text");
            check(rd.expert_directory().size() == kN && rd.vram_load_order().size() == kN, "12 directory entries, 12 order entries");
            bool dir = true;
            for (int k = 0; k < kN; ++k)
                dir = dir && rd.expert_directory()[k].expert_id == (k % kExperts + 1) % kExperts && rd.expert_directory()[k].body_offset == static_cast<uint64_t>(k) * 4096 &&
                      rd.vram_load_order()[k] == kN - 1 - k;
            check(dir, "directory entries and vram_load_order");
            std::vector<uint8_t> body;
            check(rd.read_expert_body(5, body, err) && body.size() == 64 && body[0] == 0x45 && body[63] == 0x45, "expert body read from Section 4");
            uint64_t v = 0;
            std::string s;
            check(rd.manifest_top_uint("num_physical_layers", v) && v == 4 && rd.manifest_top_string("arch", s) && s == "inkling2_loop", "top-level manifest keys");
            check(!rd.is_vision_shard(), "not a Vision shard");
        }
        rd.close();
        drop_keepable(b.path);
    }
    {
        std::printf("  synthetic 4 KiB file, no order table\n");
        Spec sp;
        sp.order = false;
        const Built b = build("strata_layout_4k_noorder.strata", sp);
        StrataFileReader rd;
        std::string err;
        const bool ok = rd.open(b.path.string(), err);
        check(ok && rd.expert_directory().size() == kN && rd.vram_load_order().empty(), "opens: 12 entries, no order");
        rd.close();
        std::filesystem::remove(b.path);
    }
    {
        std::printf("  synthetic 4 KiB file, cut inside Section 4 (the manifest at the end is gone)\n");
        Spec sp;
        sp.truncate_tail = true;
        const Built b = build("strata_layout_4k_trunc.strata", sp);
        StrataFileReader rd;
        std::string err;
        // the directory and manifest are past the cut: Section 0 cannot be read, a clean error, not a crash
        check(!rd.open(b.path.string(), err) && !err.empty(), "refused with an error (the manifest is cut off)");
        std::printf("    err: %s\n", err.c_str());
        rd.close();
        std::filesystem::remove(b.path);
    }

    // 3. a synthetic 16 KiB file through the reader --------------------------------------------------------------------
    {
        std::printf("  synthetic 16 KiB file\n");
        Spec sp;
        sp.k16 = true;
        const Built b = build("strata_layout_16k.strata", sp);
        StrataFileReader rd;
        std::string err;
        const bool ok = rd.open(b.path.string(), err);
        check(ok, "opens");
        if (!ok) std::printf("    err: %s\n", err.c_str());
        if (ok) {
            const auto& h = rd.header();
            check(h.layout == StrataLayout::Page16K && h.page_size == 16384 && h.section_table_offset == 0x20, "16 KiB layout, table at 0x20");
            check(h.tile_layout_id == 7, "the u16 at 0x10 is recorded");
            bool secs = true;
            for (int i = 0; i < 6; ++i) secs = secs && h.sections[i].file_offset == (b.len[i] ? b.off[i] : 0) && h.sections[i].byte_length == b.len[i];
            check(secs, "section offsets and lengths");
            check(b.len[3] == 16384 && b.len[3] % 34 != 0, "Section 3 is padded (16384 B, not a multiple of 34)");
            check(rd.manifest_json() == manifest_text("inkling2_loop"), "manifest text");
            check(rd.expert_directory().size() == kN && rd.vram_load_order().size() == kN, "12 directory entries and 12 order entries (not 16384 / 34 = 481)");
            bool dir = true;
            for (int k = 0; k < kN; ++k)
                dir = dir && rd.expert_directory()[k].expert_id == (k % kExperts + 1) % kExperts && rd.expert_directory()[k].body_offset == static_cast<uint64_t>(k) * 16384 &&
                      rd.vram_load_order()[k] == kN - 1 - k;
            check(dir, "directory entries and vram_load_order");
            std::vector<uint8_t> body;
            check(rd.read_expert_body(7, body, err) && body.size() == 64 && body[0] == 0x47 && body[63] == 0x47, "expert body read from Section 4");
            uint64_t v = 0;
            check(rd.manifest_top_uint("num_experts_per_layer", v) && v == 3, "top-level key found past a template that spells another one");
        }
        rd.close();
        drop_keepable(b.path);
    }
    {
        std::printf("  synthetic 16 KiB file, no order table, manifest without a count\n");
        Spec sp;
        sp.k16 = true;
        sp.order = false;
        sp.manifest = "{\"format\":\"strata-v1.1\",\"arch\":\"hy4\"}";
        const Built b = build("strata_layout_16k_hy4.strata", sp);
        StrataFileReader rd;
        std::string err;
        const bool ok = rd.open(b.path.string(), err);
        check(ok && rd.expert_directory().size() == kN, "opens: the zero entries at the tail of the padded Section 3 are dropped (12, not 512)");
        rd.close();
        std::filesystem::remove(b.path);
    }
    {
        std::printf("  synthetic 16 KiB file, Section 3 exactly one entry list (no padding)\n");
        Spec sp;
        sp.k16 = true;
        sp.pad_sec3 = false;  // 12 x 34 = 408 B, not a page: still accepted, the manifest count fits with 0 padding
        const Built b = build("strata_layout_16k_nopad.strata", sp);
        StrataFileReader rd;
        std::string err;
        check(rd.open(b.path.string(), err) && rd.expert_directory().size() == kN, "opens with 12 entries");
        rd.close();
        std::filesystem::remove(b.path);
    }

    // 4. the 0x10 bytes of a 16 KiB header can hold anything: it is still the 16 KiB layout ----------------------------
    {
        std::printf("  16 KiB header whose bytes at 0x10 look like a 4 KiB section 0 (offset 1 MiB, length 4096)\n");
        Spec sp;
        sp.k16 = true;
        sp.tile_id = 0;
        sp.ext0 = 1ull << 20;  // a reader that only asks "is the u64 at 0x10 below 4096?" calls this an old file
        sp.ext1 = 4096;
        const Built b = build("strata_layout_16k_ext.strata", sp);
        StrataFileReader rd;
        std::string err;
        const bool ok = rd.open(b.path.string(), err);
        check(ok && rd.header().layout == StrataLayout::Page16K && rd.expert_directory().size() == kN, "opened as the 16 KiB layout");
        rd.close();
        std::filesystem::remove(b.path);
    }
    {
        std::printf("  4 KiB header with no Section 0 at all (offset 0, length 0): not taken for the 16 KiB layout\n");
        std::vector<uint8_t> f = frame_with();
        entry(f, 0x10, 0, 0, 0, 0);
        entry(f, 0x10, 1, 4096, 50000, 1);
        entry(f, 0x10, 2, 0, 0, 2);
        entry(f, 0x10, 3, 0, 0, 3);
        entry(f, 0x10, 4, 0, 0, 4);
        entry(f, 0x10, 5, 0, 0, 5);
        StrataHeader h;
        std::string err;
        check(parse(f, 0, h, err) && h.layout == StrataLayout::Page4K && h.sections[1].file_offset == 4096, "4 KiB layout");
    }

    // 5. refusals -----------------------------------------------------------------------------------------------------
    {
        std::printf("  headers that are neither layout, or both\n");
        StrataHeader h;
        std::string err;
        std::vector<uint8_t> f = frame_with();
        check(!parse(f, 0, h, err) && has(err, "neither the 4 KiB layout"), "empty table: refused");
        std::printf("    err: %s\n", err.c_str());

        f = frame_with();
        for (int i = 0; i < 6; ++i) entry(f, 0x20, i, i == 1 ? 16384 + 1 : 0, i == 1 ? 100 : 0, i);  // 16 KiB table, section 1 not aligned
        check(!parse(f, 0, h, err) && has(err, "not 16 KiB aligned"), "16 KiB table with an unaligned section: refused, says why");
        std::printf("    err: %s\n", err.c_str());

        f = frame_with();
        for (int i = 0; i < 6; ++i) entry(f, 0x20, i, i == 1 ? 16384 : 0, i == 1 ? 100 : 0, i == 1 ? 9 : i);  // wrong section_id
        check(!parse(f, 0, h, err) && has(err, "section_id"), "16 KiB table with a wrong section_id: refused, says why");

        f = frame_with();
        for (int i = 0; i < 6; ++i) entry(f, 0x20, i, i == 1 ? 16384 : 0, i == 1 ? 100000 : 0, i);
        check(parse(f, 0, h, err) && h.layout == StrataLayout::Page16K, "the same table is the 16 KiB layout when the file size is unknown");
        check(!parse(f, 50000, h, err) && has(err, "past the end of the file"), "... and refused when the file is shorter than Section 1");
        std::printf("    err: %s\n", err.c_str());

        f = frame_with();
        for (int i = 0; i < 6; ++i) entry(f, 0x20, i, i <= 1 ? 16384 : 0, i <= 1 ? 16384 : 0, i);  // sections 0 and 1 on the same page
        check(!parse(f, 0, h, err) && has(err, "overlap"), "16 KiB table with overlapping sections: refused");

        f = frame_with();
        for (int i = 0; i < 6; ++i) entry(f, 0x10, i, i == 1 ? 4096 + 100 : 0, i == 1 ? 100 : 0, i);  // 4 KiB table, unaligned
        check(!parse(f, 0, h, err) && has(err, "not 4 KiB aligned"), "4 KiB table with an unaligned section: refused, says why");

        f = frame_with(5);
        check(!parse(f, 0, h, err) && has(err, "num_sections"), "5 sections: refused");
        f = frame_with(7);
        check(!parse(f, 0, h, err) && has(err, "num_sections"), "7 sections: refused");
        f = frame_with();
        f[0] = 'X';
        check(!parse(f, 0, h, err) && has(err, "magic"), "bad magic: refused");
        f = frame_with();
        check(!strata_parse_header_frame(f.data(), 100, 0, h, err) && has(err, "Short"), "a short frame: refused");

        // both tables strict at once: A_0 holds a real section, B_0 too, and every sid sits where both read it
        f = frame_with();
        for (int i = 0; i < 6; ++i) {
            put<uint32_t>(f, 0x10 + i * 64 + 48, i);   // 4 KiB table: section_id of entry i
            put<uint32_t>(f, 0x20 + i * 64 + 48, i);   // 16 KiB table: section_id of entry i
        }
        for (int i = 1; i < 6; ++i) put<uint32_t>(f, 0x10 + i * 64, i - 1);  // A_i offset low word = B_(i-1) section_id; length 0 (empty)
        put<uint64_t>(f, 0x10, 65536);                                     // A_0: offset 64 KiB
        put<uint64_t>(f, 0x18, 100);                                         //      length 100  (4 KiB aligned)
        put<uint64_t>(f, 0x20, 16384);                                       // B_0: offset 16 KiB  (= A_0 sha bytes 0..8)
        put<uint64_t>(f, 0x28, 100);                                         //      length 100
        check(!parse(f, 0, h, err) && has(err, "Ambiguous"), "a table that reads as both layouts: refused as ambiguous");
        std::printf("    err: %s\n", err.c_str());
    }
    {
        std::printf("  files through the reader\n");
        StrataFileReader rd;
        std::string err;
        check(!rd.open((std::filesystem::temp_directory_path() / "strata_layout_no_such_file.strata").string(), err) && has(err, "Failed to open"), "missing file: refused");
        const auto p = std::filesystem::temp_directory_path() / "strata_layout_short.strata";
        std::ofstream(p, std::ios::binary).write("STRATA\0\1short", 13);
        check(!rd.open(p.string(), err) && has(err, "Short read"), "a file shorter than the header: refused");
        std::filesystem::remove(p);

        Spec sp;
        sp.k16 = true;
        sp.truncate_tail = true;
        const Built b = build("strata_layout_16k_trunc.strata", sp);
        check(!rd.open(b.path.string(), err) && !err.empty(), "a 16 KiB file cut inside Section 4: refused, no crash");
        std::printf("    err: %s\n", err.c_str());
        rd.close();
        std::filesystem::remove(b.path);

        sp = Spec{};
        sp.k16 = true;
        sp.manifest = std::string(
            "{\"arch\":\"inkling2_loop\",\"num_physical_layers\":4,\"num_experts_per_layer\":2000}");  // 8000 entries do not fit the Section 3 the file has
        const Built c = build("strata_layout_16k_badcount.strata", sp);
        check(!rd.open(c.path.string(), err) && has(err, "does not match the manifest"), "padded Section 3 that cannot hold the manifest's count: refused");
        std::printf("    err: %s\n", err.c_str());
        rd.close();
        std::filesystem::remove(c.path);

        // A padded Section 3 whose length is a multiple of 34: 8000 entries x 34 B = 272000 B, padded to 17 pages = 278528 B = 8192 x 34.
        // Reading "length is a multiple of 34, so length / 34 is the count" gives 8192 entries, 192 of them padding (about 1 size in 17).
        sp = Spec{};
        sp.k16 = true;
        sp.experts = 2000;  // 4 layers x 2000 = 8000 entries
        const Built m = build("strata_layout_16k_mult34.strata", sp);
        const bool mok = rd.open(m.path.string(), err);
        check(m.len[3] == 278528 && m.len[3] % 34 == 0, "fixture: Section 3 is 278528 B, a multiple of 34 (8192 x 34)");
        check(mok && rd.expert_directory().size() == 8000 && rd.vram_load_order().size() == 8000, "padded Section 3 that is a multiple of 34: 8000 entries and 8000 order values, not 8192");
        if (mok) {
            bool ord = true;
            for (size_t k = 0; k < 8000; ++k) ord = ord && rd.vram_load_order()[k] == static_cast<uint16_t>(7999 - k);
            check(ord, "the order table is read from after the 8000th entry, not the 8192nd");
        } else {
            std::printf("    err: %s\n", err.c_str());
        }
        rd.close();
        drop_keepable(m.path);

        // A manifest whose count overflows.  2^32 x 2^31 = 2^63: times 34 that is 17 x 2^64, which wraps to 0, so "Section 3 holds
        // that many entries" passes and the count 2^63 goes to resize() (a throw, i.e. a crash).  The others wrap the product itself.
        const char* wraps[] = {
            "{\"arch\":\"inkling2_loop\",\"num_physical_layers\":4294967296,\"num_experts_per_layer\":2147483648}",
            "{\"arch\":\"inkling2_loop\",\"num_physical_layers\":4294967296,\"num_experts_per_layer\":4294967296}",
            "{\"arch\":\"inkling2_loop\",\"num_physical_layers\":576460752303423488,\"num_experts_per_layer\":32}",
            "{\"arch\":\"inkling2_loop\",\"num_physical_layers\":18446744073709551615,\"num_experts_per_layer\":2}",
            "{\"arch\":\"inkling2_loop\",\"num_physical_layers\":99999999999999999999,\"num_experts_per_layer\":2}",
            "{\"arch\":\"inkling2_loop\",\"num_physical_layers\":0,\"num_experts_per_layer\":128}",
        };
        int wi = 0;
        for (const char* w : wraps) {
            sp = Spec{};
            sp.k16 = true;
            sp.manifest = w;
            const Built o = build(("strata_layout_16k_overflow" + std::to_string(wi++) + ".strata").c_str(), sp);
            const bool refused = !rd.open(o.path.string(), err) && has(err, "does not match the manifest");
            check(refused, ("manifest count that wraps 64 bits: refused, no throw (" + std::to_string(wi) + ")").c_str());
            rd.close();
            if (wi == 1) drop_keepable(o.path); else std::filesystem::remove(o.path);
        }

        sp = Spec{};
        sp.k16 = true;
        sp.s3_extra = 16384;  // Section 3 two pages long for 12 entries: more padding than a page is not padding
        const Built d = build("strata_layout_16k_bigpad.strata", sp);
        check(!rd.open(d.path.string(), err) && has(err, "does not match the manifest"), "padding of a page or more after the manifest's count: refused");
        rd.close();
        std::filesystem::remove(d.path);

        sp = Spec{};
        sp.k16 = true;
        sp.manifest = "not json";  // not a zstd frame: the raw-frame writer still wraps it, so break the frame instead
        Built e = build("strata_layout_16k_badzstd.strata", sp);
        {
            std::fstream fs(e.path, std::ios::in | std::ios::out | std::ios::binary);
            fs.seekp(static_cast<std::streamoff>(e.off[0]));
            fs.write("garbage!", 8);
        }
        check(!rd.open(e.path.string(), err) && has(err, "ZSTD"), "Section 0 that is not zstd: refused");
        rd.close();
        std::filesystem::remove(e.path);
    }

    // 6. the manifest walk and the Vision shard --------------------------------------------------------------------------
    {
        std::printf("  manifest keys and the Vision shard\n");
        std::string t;
        check(strata_manifest_top_value("{\"a\":{\"k\":1},\"k\":22}", "k", false, t) && t == "22", "a nested key does not shadow the top-level one");
        check(!strata_manifest_top_value("{\"a\":\"k\",\"b\":\"x\"}", "k", false, t), "a string value equal to the key is not the key");
        check(!strata_manifest_top_value("{\"k\":\"text\"}", "k", false, t), "a string where a number is wanted: absent");
        check(strata_manifest_top_value("{\"k\" : \"te\\\"xt\" }", "k", true, t) && t == "te\"xt", "string value with an escaped quote");
        check(!strata_manifest_top_value("{\"k\":", "k", false, t), "cut off in the middle: absent, no crash");
        check(!strata_manifest_top_value("", "k", false, t), "empty text: absent");
        check(strata_manifest_expert_count(manifest_text("x")) == 12, "expert count = 4 layers x 3 experts, from the top level");
        check(strata_manifest_expert_count("{\"num_physical_layers\":4}") == 0, "one key only: 0");
        check(strata_manifest_expert_count("{\"num_physical_layers\":99999999999,\"num_experts_per_layer\":2}") == UINT64_MAX, "an absurd count: UINT64_MAX (refused later)");

        Spec sp;
        sp.k16 = true;
        sp.profile = 3;
        sp.manifest = "{\"format\":\"strata-v1.1-vision\",\"arch\":\"inkling2_vision\",\"vision_config\":{}}";
        sp.order = false;
        Built b = build("strata_layout_16k_vision.strata", sp);
        StrataFileReader rd;
        std::string err;
        check(rd.open(b.path.string(), err) && rd.header().profile == StrataProfile::VisionShard && rd.is_vision_shard(), "16 KiB Vision shard, header profile 3");
        rd.close();
        std::filesystem::remove(b.path);

        sp.profile = 0;  // the header profile of the shard says nothing, the manifest says what it is
        b = build("strata_layout_16k_vision0.strata", sp);
        check(rd.open(b.path.string(), err) && rd.is_vision_shard(), "arch inkling2_vision in the manifest is enough");
        rd.close();
        std::filesystem::remove(b.path);

        sp = Spec{};
        sp.k16 = true;  // a model file: its config names the vision model type, its arch is the loop
        b = build("strata_layout_16k_model.strata", sp);
        check(rd.open(b.path.string(), err) && !rd.is_vision_shard(), "a model whose config mentions inkling2_vision is not a Vision shard");
        rd.close();
        std::filesystem::remove(b.path);
    }

    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
