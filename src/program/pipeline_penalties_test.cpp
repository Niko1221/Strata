// The final-stage history must count the accepted prefix and this window,
// including its fed-back token, without retaining rejected speculative tokens.
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "pipeline_penalties_test: %s\n", why);
        std::exit(1);
    }
}

static void compare_rows(const std::vector<int32_t>& prefix, const std::vector<int32_t>& window, int history) {
    const size_t rows = window.size();
    std::vector<int32_t> staged(rows * (size_t) history, -99);
    strata::kernels::penalty_rows(prefix.data(), (int64_t) prefix.size(), window.data(),
                                  (int) rows, history, staged.data());
    // Plain decode consumes one token before each pick. Build that history
    // independently, by dropping old tokens as each new token arrives.
    std::vector<int32_t> recent(prefix);
    if (recent.size() > (size_t) history)
        recent.erase(recent.begin(), recent.end() - history);
    for (size_t row = 0; row < rows; ++row) {
        recent.push_back(window[row]);
        if (recent.size() > (size_t) history) recent.erase(recent.begin());
        const size_t padding = (size_t) history - recent.size();
        for (int col = 0; col < history; ++col) {
            const int32_t want = (size_t) col < padding ? -1 : recent[(size_t) col - padding];
            check(staged[row * (size_t) history + (size_t) col] == want,
                  "a window row differs from plain-decode history");
        }
    }
}

int main() {
    for (const int h : {1, 2, 7, 64, 4096}) {
        for (const int prefix_n : {0, 1, 5, 63, 65, 4100}) {
            std::vector<int32_t> consumed;
            for (int i = 0; i < prefix_n; ++i) consumed.push_back(i % 37);
            const std::vector<int32_t> a = {101, 102, 103, 104, 105, 106, 107, 108};
            for (int t = 1; t <= (int) a.size(); ++t) {
                const std::vector<int32_t> window(a.begin(), a.begin() + t);
                compare_rows(consumed, window, h);
                // Verdicts include first-draft rejection, partial acceptance
                // and full acceptance. Rebuild from the committed prefix, not
                // the history of the speculative continuation discarded here.
                for (int accepted = 1; accepted <= t; ++accepted) {
                    std::vector<int32_t> next(consumed);
                    next.insert(next.end(), window.begin(), window.begin() + accepted);
                    const std::vector<int32_t> b = {201, 202, 203};
                    compare_rows(next, b, h);
                    // A second parity follows the accepted part of B.
                    next.push_back(b[0]);
                    compare_rows(next, {301}, h);
                }
            }
        }
    }
    std::puts("pipeline_penalties_test: PASS (padding, cap, rejection, parity history)");
    return 0;
}
