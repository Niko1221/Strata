#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::program {

struct VisionRecord { int64_t n, nx, ny; size_t off; };

/// Read only as many finite embedding rows as the prompt can consume. Validate before allocating.
inline bool read_vision_records(const std::string& path, int64_t width, int64_t max_rows,
                                std::vector<float>& rows, std::vector<VisionRecord>& records, std::string& err) {
    rows.clear();
    records.clear();
    err.clear();
    if (width < 1 || max_rows < 0) { err = "invalid image embedding limits"; return false; }
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) { err = "cannot open " + path; return false; }
    struct Close { std::FILE* f; ~Close() { std::fclose(f); } } close{file};
    int64_t used = 0;
    while (true) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, 1, sizeof(hdr), file);
        if (got == 0 && std::feof(file) && !std::ferror(file)) break;
        if (got != sizeof(hdr) || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != width || width < 1 ||
            max_rows < used || hdr[1] > max_rows - used ||
            (uint64_t) hdr[1] > (rows.max_size() - rows.size()) / (uint64_t) width) {
            err = "bad or oversized image embeddings record";
            return false;
        }
        const size_t off = rows.size(), count = (size_t) hdr[1] * (size_t) width;
        rows.resize(off + count);
        if (std::fread(rows.data() + off, sizeof(float), count, file) != count) {
            err = "short image embeddings record";
            return false;
        }
        for (size_t i = off; i < rows.size(); ++i)
            if (!std::isfinite(rows[i])) { err = "nonfinite image embedding"; return false; }
        records.push_back({hdr[1], hdr[2], hdr[3], off});
        used += hdr[1];
    }
    return true;
}

}  // namespace strata::program
