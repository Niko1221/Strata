#pragma once
// A tiny WEX4 writer for the tests (RAW records only): the header page, the index right after it, then the records,
// each padded with zeros to 4096 bytes, as Wex-WRN2's wex4.py writes them.  It is written from the spec, not from
// wex4_experts.cpp, so a test reads a file that the engine's reader did not produce.  A test may break a rule on
// purpose: a record's codec field, a gap before a record (not a valid WEX4 file then, which is the point).
#include "strata/core/wex4_experts.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace wex4_test {

struct Tensor {
    std::string name;
    uint8_t dtype = 0;
    uint64_t nbytes = 0;
};
struct Record {
    std::vector<Tensor> tensors;   // tile `data` in order
    std::vector<uint8_t> data;
    uint16_t codec = 0;            // 0 RAW; a test sets 5 (WNR3) to check the refusal
    uint64_t gap_before = 0;       // zero bytes before the record (a multiple of 4096)
};

inline void put(std::vector<uint8_t>& b, size_t at, uint64_t v, int n) {
    for (int i = 0; i < n; ++i) b[at + (size_t) i] = (uint8_t) (v >> (8 * i));
}
inline uint64_t page_align(uint64_t n) { return (n + 4095) & ~4095ull; }

/// The whole file.  Its root is the SHA-256 of bytes [0, 4064).
inline std::vector<uint8_t> file(const std::vector<Record>& recs) {
    std::vector<const Tensor*> ts;
    std::string names;
    for (const Record& r : recs)
        for (const Tensor& t : r.tensors) { ts.push_back(&t); names += t.name; }
    const uint64_t n_rec = recs.size(), n_ten = ts.size();
    const uint64_t index_bytes = n_rec * 72 + n_ten * 100 + names.size();
    std::vector<uint8_t> index((size_t) index_bytes);
    uint64_t at = 4096 + page_align(index_bytes), name_at = 0, total = 0, max_decoded = 0;
    std::vector<std::pair<uint64_t, const Record*>> placed;
    uint32_t t = 0;
    for (uint64_t r = 0; r < n_rec; ++r) {
        const Record& rec = recs[(size_t) r];
        at += rec.gap_before;
        const uint64_t stored = page_align(rec.data.size());
        std::vector<uint8_t> padded(rec.data);
        padded.resize((size_t) stored, 0);
        uint8_t sha[32];
        strata::core::detail::wex4_sha256(padded.data(), padded.size(), sha);
        const size_t row = (size_t) (r * 72);
        put(index, row, at, 8);
        put(index, row + 8, stored, 4);
        put(index, row + 12, rec.data.size(), 4);
        put(index, row + 16, rec.codec, 2);
        put(index, row + 24, r, 4);                       // owner: the record itself
        put(index, row + 28, t, 4);
        put(index, row + 32, rec.tensors.size(), 4);
        std::memcpy(index.data() + row + 36, sha, 32);
        uint64_t cursor = 0;
        for (const Tensor& x : rec.tensors) {
            const size_t tr = (size_t) (n_rec * 72 + (uint64_t) t * 96);
            put(index, tr, name_at, 4);
            put(index, tr + 4, x.name.size(), 2);
            index[tr + 6] = x.dtype;
            index[tr + 7] = 1;                            // rank 1: shape = bytes (dtype >= 128 is opaque)
            put(index, tr + 8, x.nbytes, 8);
            put(index, tr + 72, r, 4);
            put(index, tr + 76, 1, 4);
            put(index, tr + 80, cursor, 8);
            put(index, tr + 88, x.nbytes, 8);
            name_at += x.name.size();
            cursor += x.nbytes;
            ++t;
        }
        placed.push_back({at, &rec});
        at += stored;
        total += rec.data.size();
        max_decoded = std::max<uint64_t>(max_decoded, rec.data.size());
    }
    std::vector<uint32_t> order(ts.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return ts[a]->name < ts[b]->name; });
    for (size_t i = 0; i < order.size(); ++i) put(index, (size_t) (n_rec * 72 + n_ten * 96 + i * 4), order[i], 4);
    std::memcpy(index.data() + n_rec * 72 + n_ten * 100, names.data(), names.size());

    std::vector<uint8_t> out((size_t) at, 0);
    put(out, 0, 0x57455834u, 4);
    put(out, 4, 4, 2);
    put(out, 24, at, 8);
    put(out, 32, 4096, 8);
    put(out, 40, index_bytes, 8);
    put(out, 52, n_rec, 4);
    put(out, 56, n_ten, 4);
    put(out, 68, max_decoded, 4);
    put(out, 72, names.size(), 8);
    put(out, 88, total, 8);
    strata::core::detail::wex4_sha256(index.data(), index.size(), out.data() + 96);
    strata::core::detail::wex4_sha256(out.data(), 4064, out.data() + 4064);
    std::memcpy(out.data() + 4096, index.data(), index.size());
    for (const auto& [offset, rec] : placed) std::memcpy(out.data() + offset, rec->data.data(), rec->data.size());
    return out;
}

/// Rehash the header page after a test edited it, so only the edit is wrong.
inline void reseal_header(std::vector<uint8_t>& f) { strata::core::detail::wex4_sha256(f.data(), 4064, f.data() + 4064); }

inline void save(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    o.write((const char*) bytes.data(), (std::streamsize) bytes.size());
}

/// One record per expert of a native layout, [gate | up | down], named and typed as the engine expects;
/// blob(l, e) gives its bytes.
template <class Layout, class Blob>
std::vector<Record> expert_records(const Layout& lay, Blob blob) {
    static const char* roles[3] = {"gate", "up", "down"};
    std::vector<Record> recs;
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const auto& f = lay.fmt[(size_t) l];
        const uint64_t sizes[3] = {f.up_off, f.down_off - f.up_off, f.bytes - f.down_off};
        const int types[3] = {f.gu_type, f.gu_type, f.d_type};
        for (int64_t e = 0; e < lay.n_expert; ++e) {
            Record r;
            for (int k = 0; k < 3; ++k)
                r.tensors.push_back({"blk." + std::to_string(l) + ".ffn_" + roles[k] + "_exps.weight#" + std::to_string(e),
                                     (uint8_t) (128 + types[k]), sizes[k]});
            r.data = blob(l, e);
            recs.push_back(std::move(r));
        }
    }
    return recs;
}

}  // namespace wex4_test
