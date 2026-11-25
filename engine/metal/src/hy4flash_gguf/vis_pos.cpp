// engine/metal/src/hy4flash_gguf/vis_pos.cpp - see vis_pos.h.
#include "vis_pos.h"

#include <algorithm>

namespace strata::hy4_flash {

std::string build_image_spans(std::vector<ImageSpan>& out, const int32_t* ids, int64_t n, int32_t image_token,
                              const std::vector<std::pair<int32_t, int32_t>>& pictures, int32_t merge) {
    out.clear();
    if (merge <= 0) return "the spatial merge size must be positive";
    size_t next = 0;
    for (int64_t i = 0; i < n;) {
        if (ids[i] != image_token) {
            ++i;
            continue;
        }
        int64_t j = i;
        while (j < n && ids[j] == image_token) ++j;
        if (next >= pictures.size())
            return "the prompt holds more image-token runs than pictures (" + std::to_string(out.size() + 1) +
                   " vs " + std::to_string(pictures.size()) + ")";
        const auto [gh, gw] = pictures[next++];
        if (gh <= 0 || gw <= 0 || gh % merge != 0 || gw % merge != 0)
            return "picture " + std::to_string(next - 1) + " has a " + std::to_string(gh) + "x" +
                   std::to_string(gw) + " patch grid, which is not a positive multiple of the merge size";
        const int64_t want = (int64_t) (gh / merge) * (gw / merge + 1);
        if (j - i != want)
            return "the image-token run at " + std::to_string(i) + " is " + std::to_string(j - i) +
                   " tokens, but picture " + std::to_string(next - 1) + "'s " + std::to_string(gh) + "x" +
                   std::to_string(gw) + " grid takes " + std::to_string(want);
        out.push_back({i, want, gh, gw});
        i = j;
    }
    if (next != pictures.size())
        return "the prompt holds " + std::to_string(next) + " image-token runs but " +
               std::to_string(pictures.size()) + " pictures were given";
    return {};
}

std::string build_mrope_positions(std::vector<int32_t>& pos, int64_t n, const std::vector<ImageSpan>& spans,
                                  int32_t merge) {
    if (merge <= 0) return "the spatial merge size must be positive";
    pos.assign((size_t) 3 * (size_t) n, 0);
    int32_t* t_axis = pos.data();
    int32_t* h_axis = pos.data() + n;
    int32_t* w_axis = pos.data() + 2 * n;

    int64_t cur = 0;        // the next position to hand out
    int64_t at = 0;         // the next token index
    size_t s = 0;
    while (at < n) {
        if (s < spans.size() && spans[s].start == at) {
            const ImageSpan& sp = spans[s++];
            const int64_t rows = sp.gh / merge, cols = sp.gw / merge + 1;
            if (rows * cols != sp.count) return "a span's count does not match its grid";
            if (at + sp.count > n) return "a span runs past the end of the prompt";
            // One block: the temporal axis is constant for a still picture, and the newline column is
            // column `cols - 1` of each row rather than a position of its own.
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < cols; ++c) {
                    const int64_t k = at + r * cols + c;
                    t_axis[k] = (int32_t) cur;
                    h_axis[k] = (int32_t) (cur + r);
                    w_axis[k] = (int32_t) (cur + c);
                }
            // The token after the picture resumes after the whole block, which is one past the largest of
            // the three axes - NOT one past the picture's last token.
            cur += std::max(rows - 1, cols - 1) + 1;
            at += sp.count;
            continue;
        }
        const int64_t until = s < spans.size() ? spans[s].start : n;
        if (until < at) return "the spans are not in prompt order";
        for (int64_t k = at; k < until; ++k) {
            t_axis[k] = h_axis[k] = w_axis[k] = (int32_t) cur;
            ++cur;
        }
        at = until;
    }
    if (s != spans.size()) return "a span starts past the end of the prompt";
    return {};
}

}  // namespace strata::hy4_flash
