#include "strata/core/batch_segments.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

static void require(bool value) { if (!value) { std::fprintf(stderr, "segment test failed\n"); std::exit(1); } }

int main() {
    using namespace strata::core;
    std::string err;
    // Sparse active sets, asymmetric demand and rotating spare-row allocation.
    // Assert capacity, no starvation, work conservation and valid dense tiles.
    for (int code = 0; code < 625; ++code) {
        int wanted[4], q = code, demand = 0;
        for (int i = 0; i < 4; ++i) { wanted[i] = q % 5; q /= 5; demand += wanted[i]; }
        for (int budget : {8, 12, 16}) for (int first = 0; first < 4; ++first) {
            int out[5] = {0, 0, 0, 0, 123456}, rows[kBatchMaxRows], used = 0;
            allocate_batch_rows(wanted, 4, budget, first, out);
            require(out[4] == 123456);
            for (int i = 0; i < 4; ++i) {
                require(out[i] >= (wanted[i] > 0 ? 1 : 0) && out[i] <= wanted[i]);
                for (int j = 0; j < out[i]; ++j) rows[used++] = i;
            }
            require(used == std::min(demand, budget));
            if (used) require(segment_group_boundary(rows, used) >= 0);
        }
    }
    // Rotation must give every equally demanding request the extra row equally.
    int demand[4] = {4, 4, 4, 4}, totals[4] = {};
    for (int first = 0; first < 4; ++first) {
        int out[4]; allocate_batch_rows(demand, 4, 9, first, out);
        for (int i = 0; i < 4; ++i) totals[i] += out[i];
    }
    for (int total : totals) require(total == 9);
    for (int width = 1; width <= 8; ++width) {
        int32_t inputs[8], outputs[8];
        for (int i = 0; i < width; ++i) { inputs[i] = 100 + i; outputs[i] = 101 + i; }
        require(accepted_segment_prefix(inputs, outputs, width, nullptr, 0) == width);
        for (int i = 0; i < width; ++i) {
            int64_t eos = outputs[i];
            require(accepted_segment_prefix(inputs, outputs, width, &eos, 1) == i + 1);
            int saved = outputs[i]; outputs[i] = -10;
            require(accepted_segment_prefix(inputs, outputs, width, nullptr, 0) == i + 1);
            outputs[i] = saved;
        }
    }
    for (int width = 1; width <= 8; ++width) {
        int32_t tokens[] = {10, 11, 248044, 13, 14, 15, 16, 17};
        int32_t initial[] = {-1, 7}, pairs[18];
        pairs[2 * width] = 123456;
        segment_predecessors(tokens, width, initial, pairs);
        for (int i = 0; i < width; ++i) {
            require(pairs[2 * i] == (i >= 2 ? tokens[i - 2] : initial[i]));
            require(pairs[2 * i + 1] == (i ? tokens[i - 1] : initial[1]));
        }
        require(pairs[2 * width] == 123456);
        require(initial[0] == -1 && initial[1] == 7);
    }
    int cases = 0;
    // Enumerate layouts independently: each slot can occur in only one run.
    // Includes ordinary, noncontiguous, repeated, reordered and eight-row layouts.
    for (int n = 1; n <= 8; ++n) {
        int layouts = 1;
        for (int i = 0; i < n; ++i) layouts *= 3;
        for (int code = 0; code < layouts; ++code) {
            std::array<int, 8> slots{}, keep{};
            std::array<int64_t, 8> pos{};
            bool seen[3] = {}, causal = true, ordinary = true;
            int q = code;
            for (int i = 0; i < n; ++i) {
                slots[i] = q % 3; q /= 3;
                if (i && slots[i] == slots[i - 1]) {
                    pos[i] = pos[i - 1] + 1;
                    ordinary = false;
                } else {
                    causal = causal && !seen[slots[i]];
                    seen[slots[i]] = true;
                    pos[i] = 100 * (slots[i] + 1);
                    keep[i] = 1;
                }
            }
            require(validate_segments(slots.data(), pos.data(), n, 8, 3, true, err) == causal);
            require(validate_segments(slots.data(), pos.data(), n, 8, 3, false, err) == (causal && ordinary));
            if (causal) {
                require(validate_segment_keeps(slots.data(), n, keep.data(), err));
                for (int i = 0; i < n; ++i) if (!i || slots[i] != slots[i - 1]) {
                    const int width = segment_width(slots.data(), n, i);
                    for (int k = 1; k <= width; ++k) {
                        keep[i] = k;
                        require(validate_segment_keeps(slots.data(), n, keep.data(), err));
                    }
                    keep[i] = width + 1;
                    require(!validate_segment_keeps(slots.data(), n, keep.data(), err));
                    keep[i] = 1;
                }
            }
            ++cases;
        }
    }
    // Every four-slot layout with 1..4 physical rows fits two kernel groups.
    for (int a = 1; a <= 4; ++a) for (int b = 1; b <= 4; ++b)
    for (int c = 1; c <= 4; ++c) for (int d = 1; d <= 4; ++d) {
        int rows[kBatchMaxRows]{};
        int n = 0, slot = 0;
        for (int width : {a, b, c, d}) {
            for (int j = 0; j < width; ++j) rows[n++] = slot;
            ++slot;
        }
        const int cut = segment_group_boundary(rows, n);
        require(cut >= 0);
        if (n > 8) require(cut > 0 && cut <= 8 && n - cut <= 8 && rows[cut - 1] != rows[cut]);
        else require(cut == 0);
    }
    int s[] = {0, 0, 1};
    int64_t p[] = {0, 2, 9};
    require(!validate_segments(s, p, 3, 8, 3, true, err));
    p[1] = 1;
    require(!validate_segments(s, p, 3, 2, 3, true, err));
    p[0] = std::numeric_limits<int64_t>::max();
    require(!validate_segments(s, p, 3, 8, 3, true, err));
    require(!validate_segments(nullptr, p, 3, 8, 3, true, err));
    int bad[] = {1, 1, 1};
    require(!validate_segment_keeps(s, 3, bad, err));
    std::printf("PASS: %d layouts, every accepted prefix, invalid positions and commits\n", cases);
}
