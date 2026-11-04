#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::ds4 {

struct ImageSpan {
    int64_t start;
    int64_t len;
    uint64_t hash;
};

inline bool image_spans_valid(const std::vector<ImageSpan>& spans, int64_t end) {
    if (end < 0) return false;
    int64_t prev = 0;
    for (const ImageSpan& s : spans) {
        if (s.start < prev || s.len <= 0 || s.start > end || s.len > end - s.start) return false;
        prev = s.start + s.len;
    }
    return true;
}

// A prefix may never end inside an image, nor reuse an image with different pixels.
inline int64_t image_prefix(const std::vector<ImageSpan>& a, const std::vector<ImageSpan>& b, int64_t limit) {
    limit = std::max<int64_t>(0, limit);
    std::size_t i = 0, j = 0;
    while ((i < a.size() && a[i].start < limit) || (j < b.size() && b[j].start < limit)) {
        const bool aa = i < a.size() && a[i].start < limit;
        const bool bb = j < b.size() && b[j].start < limit;
        if (!aa || !bb) return std::max<int64_t>(0, aa ? a[i].start : b[j].start);
        const ImageSpan& x = a[i];
        const ImageSpan& y = b[j];
        if (x.start != y.start || x.len != y.len || x.hash != y.hash || x.start < 0 ||
            x.len <= 0 || x.len > limit - x.start)
            return std::max<int64_t>(0, std::min(x.start, y.start));
        ++i;
        ++j;
    }
    return limit;
}

}  // namespace strata::ds4
