// --experts-wex4: the WEX4 container's header, index and expert records, checked against the native expert layout.
// The on-disk layout is the WEX4 spec of github.com/KairaCode/Wex-WRN2 (WEX4-SPEC.md); every integer is little-endian
// and read byte by byte.  docs/WEX4.md has the record convention this file relies on.
#include "strata/core/wex4_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace strata::core {

namespace {

constexpr uint32_t kMagic = 0x57455834u;   // "4XEW" on disk
constexpr uint64_t kPage = 4096, kHashed = 4064;
constexpr uint64_t kBlobRow = 64, kRecordRow = 72, kTensorRow = 96;
constexpr uint64_t kMaxIndex = 1ull << 28, kMaxRows = 1ull << 24, kMaxBlobs = 1ull << 16;
constexpr uint64_t kReqValueExact = 1;      // the one required-feature bit of WEX4 4.0 (WNR3 records only)
constexpr uint32_t kDtypeOpaque = 128;      // dtype 128 + t: opaque bytes of ggml type t

uint16_t u16(const uint8_t* p) { return (uint16_t) (p[0] | p[1] << 8); }
uint32_t u32(const uint8_t* p) { return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24; }
uint64_t u64(const uint8_t* p) { return (uint64_t) u32(p) | (uint64_t) u32(p + 4) << 32; }
uint64_t align_page(uint64_t n) { return (n + kPage - 1) & ~(kPage - 1); }

std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}

const char* codec_name(uint32_t c) {
    switch (c) {
        case 0: return "RAW";
        case 5: return "WNR3";
        case 6: return "WBP1";
        default: return "unknown";
    }
}

bool read_at(std::ifstream& f, uint64_t at, uint8_t* dst, uint64_t n) {
    f.seekg((std::streamoff) at);
    f.read((char*) dst, (std::streamsize) n);
    return (bool) f && (uint64_t) f.gcount() == n;
}

}  // namespace

namespace detail {
void wex4_sha256(const uint8_t* p, size_t n, uint8_t out[32]) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto ror = [](uint32_t x, int r) { return x >> r | x << (32 - r); };
    auto block = [&](const uint8_t* b) {
        uint32_t w[64], s[8];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t) b[4 * i] << 24 | (uint32_t) b[4 * i + 1] << 16 | (uint32_t) b[4 * i + 2] << 8 | b[4 * i + 3];
        for (int i = 16; i < 64; ++i)
            w[i] = w[i - 16] + (ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ w[i - 15] >> 3) + w[i - 7] +
                   (ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ w[i - 2] >> 10);
        std::memcpy(s, h, sizeof s);
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = s[7] + (ror(s[4], 6) ^ ror(s[4], 11) ^ ror(s[4], 25)) + ((s[4] & s[5]) ^ (~s[4] & s[6])) + k[i] + w[i];
            const uint32_t t2 = (ror(s[0], 2) ^ ror(s[0], 13) ^ ror(s[0], 22)) + ((s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]));
            s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1; s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
        }
        for (int i = 0; i < 8; ++i) h[i] += s[i];
    };
    size_t i = 0;
    for (; i + 64 <= n; i += 64) block(p + i);
    uint8_t tail[128] = {};
    const size_t rest = n - i;
    if (rest) std::memcpy(tail, p + i, rest);
    tail[rest] = 0x80;
    const size_t len = rest < 56 ? 64 : 128;
    const uint64_t bits = (uint64_t) n * 8;
    for (int b = 0; b < 8; ++b) tail[len - 1 - b] = (uint8_t) (bits >> (8 * b));
    block(tail);
    if (len == 128) block(tail + 64);
    for (int b = 0; b < 32; ++b) out[b] = (uint8_t) (h[b / 4] >> (24 - 8 * (b % 4)));
}
}  // namespace detail

bool wex4_root_ok(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    return true;
}

bool wex4_experts_open(const std::string& path, const strata::kernels::cpu::ExpertLayout& lay, const std::string& root_hex,
                       Wex4Experts& out, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    out = Wex4Experts{};
    auto fail = [&](const std::string& why) { err = "--experts-wex4: " + path + ": " + why; return false; };
    if (!lay.native || lay.fmt.size() != (size_t) lay.n_layers)
        return fail("needs a native pack (native_experts.txt); a canonical Q2_0 pack has no GGUF tensor names to match");
    if (!root_hex.empty() && !wex4_root_ok(root_hex)) return fail("--experts-wex4-root takes 64 hex digits");

    const std::filesystem::path fp(std::u8string(path.begin(), path.end()));
    std::error_code ec;
    const uint64_t size = (uint64_t) std::filesystem::file_size(fp, ec);
    if (ec) return fail("cannot read its size (" + ec.message() + ")");
    std::ifstream f(fp, std::ios::binary);
    if (!f) return fail("cannot open it");
    if (size < kPage) return fail("is " + std::to_string(size) + " B, smaller than a WEX4 header page");

    // ---- the header page
    uint8_t page[kPage], digest[32];
    if (!read_at(f, 0, page, kPage)) return fail("cannot read the header page");
    if (u32(page) != kMagic) return fail("is not a WEX4 file (no WEX4 magic)");
    detail::wex4_sha256(page, kHashed, digest);
    if (std::memcmp(digest, page + kHashed, 32) != 0) return fail("the header page is damaged (its SHA-256 does not match)");
    const std::string root = hex(digest, 32);
    if (u16(page + 4) != 4) return fail("is WEX version " + std::to_string(u16(page + 4)) + "; this engine reads version 4");
    if (u64(page + 8) & ~kReqValueExact) return fail("needs format features this engine does not know");
    const uint64_t file_bytes = u64(page + 24), index_offset = u64(page + 32), index_bytes = u64(page + 40);
    const uint64_t blobs = u32(page + 48), records = u32(page + 52), tensors = u32(page + 56);
    const uint64_t names_bytes = u64(page + 72), meta_bytes = u64(page + 80);
    if (file_bytes != size)
        return fail("is " + std::to_string(size) + " B but its header says " + std::to_string(file_bytes) + " B (truncated or extended)");
    if (blobs > kMaxBlobs || records > kMaxRows || tensors > kMaxRows || names_bytes > kMaxIndex || meta_bytes > kMaxIndex ||
        index_bytes != blobs * kBlobRow + records * kRecordRow + tensors * (kTensorRow + 4) + names_bytes + meta_bytes ||
        index_bytes > kMaxIndex || index_offset < kPage || index_offset % kPage != 0 || index_offset > size ||
        align_page(index_bytes) > size - index_offset)
        return fail("the header page is inconsistent (index size or position)");
    if (!root_hex.empty()) {
        std::string want = root_hex;
        for (char& c : want) if (c >= 'A' && c <= 'F') c = (char) (c - 'A' + 'a');
        if (want != root) return fail("its root is " + root + ", --experts-wex4-root asks for " + want);
    }

    // ---- the index
    std::vector<uint8_t> index((size_t) index_bytes);
    if (index_bytes && !read_at(f, index_offset, index.data(), index_bytes)) return fail("cannot read the index");
    detail::wex4_sha256(index.data(), index.size(), digest);
    if (std::memcmp(digest, page + 96, 32) != 0) return fail("the index is damaged (its SHA-256 does not match the header)");
    const uint8_t* rec_rows = index.data() + blobs * kBlobRow;
    const uint8_t* ten_rows = rec_rows + records * kRecordRow;
    const uint8_t* names = ten_rows + tensors * (kTensorRow + 4);   // after the tensor rows and the name order
    std::unordered_map<std::string, uint32_t> by_name;
    by_name.reserve((size_t) tensors);
    for (uint32_t t = 0; t < tensors; ++t) {
        const uint8_t* row = ten_rows + (uint64_t) t * kTensorRow;
        const uint64_t at = u32(row), len = u16(row + 4);
        if (at > names_bytes || len > names_bytes - at) return fail("tensor " + std::to_string(t) + " has a name outside the name table");
        if (!by_name.emplace(std::string((const char*) names + at, (size_t) len), t).second)
            return fail("two tensors have the same name");
    }

    // ---- one RAW record per expert, [gate | up | down], consecutive within a layer
    static const char* roles[3] = {"gate", "up", "down"};
    out.first.assign((size_t) lay.n_layers, 0);
    out.stride.assign((size_t) lay.n_layers, 0);
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        if (fm.tail_off != 0)   // the NVFP4 scale tail is not a GGUF tensor: no name to check it against
            return fail("layer " + std::to_string(l) + " is NVFP4 with a scale tail, which WEX4 records do not carry");
        if (fm.gu_type < 0 || fm.gu_type > 127 || fm.d_type < 0 || fm.d_type > 127)   // e.g. Gyro's trellis types 144-146
            return fail("layer " + std::to_string(l) + " has ggml type " + std::to_string(fm.gu_type > 127 ? fm.gu_type : fm.d_type) +
                        ", which does not fit WEX4's one-byte dtype code (128 + type)");
        const uint64_t blob = lay.bytes[(size_t) l];
        const uint64_t sizes[3] = {fm.up_off, fm.down_off - fm.up_off, blob - fm.down_off};
        const uint32_t dtypes[3] = {kDtypeOpaque + (uint32_t) fm.gu_type, kDtypeOpaque + (uint32_t) fm.gu_type,
                                    kDtypeOpaque + (uint32_t) fm.d_type};
        auto name = [&](int r, int64_t e) {
            return "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight#" + std::to_string(e);
        };
        auto at = [&](int64_t e) { return "layer " + std::to_string(l) + " expert " + std::to_string(e) + ": "; };
        if (by_name.count(name(0, lay.n_expert)))
            return fail(at(lay.n_expert) + "the file has more experts than this model (" + std::to_string(lay.n_expert) + ")");
        for (int64_t e = 0; e < lay.n_expert; ++e) {
            const auto it = by_name.find(name(0, e));
            if (it == by_name.end()) return fail(at(e) + "tensor " + name(0, e) + " is not in the file");
            const uint32_t t = it->second;
            const uint64_t r = u32(ten_rows + (uint64_t) t * kTensorRow + 72);
            if (r >= records) return fail(at(e) + "the index names a record that does not exist");
            const uint8_t* rec = rec_rows + r * kRecordRow;
            const uint64_t offset = u64(rec), stored = u32(rec + 8), decoded = u32(rec + 12);
            const uint32_t codec = u16(rec + 16), first_tensor = u32(rec + 28), count = u32(rec + 32);
            if (codec != 0)
                return fail(at(e) + "stored with codec " + codec_name(codec) + " (" + std::to_string(codec) +
                            "); this engine reads RAW records only");
            if (count != 3 || first_tensor != t || (uint64_t) t + 3 > tensors)
                return fail(at(e) + "its record does not hold exactly gate, up and down (" + std::to_string(count) + " tensors)");
            uint64_t cursor = 0;
            for (int k = 0; k < 3; ++k) {
                const uint8_t* row = ten_rows + (uint64_t) (t + k) * kTensorRow;
                const std::string got((const char*) names + u32(row), u16(row + 4));
                if (got != name(k, e) || row[6] != dtypes[k] || u32(row + 72) != r || u32(row + 76) != 1 ||
                    u64(row + 80) != cursor || u64(row + 88) != sizes[k]) {
                    char buf[320];
                    std::snprintf(buf, sizeof buf, "tensor %d of its record is %s (dtype %u, %llu B at %llu); expected %s (dtype %u, %llu B at %llu)",
                                  k, got.c_str(), (unsigned) row[6], (unsigned long long) u64(row + 88), (unsigned long long) u64(row + 80),
                                  name(k, e).c_str(), (unsigned) dtypes[k], (unsigned long long) sizes[k], (unsigned long long) cursor);
                    return fail(at(e) + buf);
                }
                cursor += sizes[k];
            }
            if (decoded != blob || stored != align_page(blob))
                return fail(at(e) + "its record is " + std::to_string(decoded) + " B (" + std::to_string(stored) +
                            " stored); the layout's blob is " + std::to_string(blob) + " B");
            if (offset < kPage || offset % kPage != 0 || offset > size || stored > size - offset)
                return fail(at(e) + "its record lies outside the file or off a 4096-byte boundary");
            if (e == 0) {
                out.first[(size_t) l] = offset;
                out.stride[(size_t) l] = stored;
            } else if (offset != out.first[(size_t) l] + (uint64_t) e * stored) {
                return fail(at(e) + "its record is at byte " + std::to_string(offset) + ", expected " +
                            std::to_string(out.first[(size_t) l] + (uint64_t) e * stored) +
                            ": a layer's records must be consecutive, one record size apart");
            }
        }
    }
    out.file_bytes = size;
    out.root = root;
    out.check_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::core
