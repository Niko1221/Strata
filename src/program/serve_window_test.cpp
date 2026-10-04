// Deterministic serving-prefix regressions: model the verifier's input/output pairing,
// then compare committed inputs with the exact tokens a continuation can send back.
#include "strata/program/serve_window.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
int checks = 0;
void require(bool ok, const char* label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
constexpr int32_t eos_token = -1;
struct Prefix {
    std::vector<int32_t> emitted, consumed;
    int64_t produced;
    int last_t = 0, last_keep = 0, last_emitted = 0;
    bool eos = false;

    Prefix(int prompt, int already = 0) : emitted((size_t) (prompt + already)), produced(already) {
        std::iota(emitted.begin(), emitted.end(), 0);
        consumed.assign(emitted.begin(), emitted.end() - 1);
    }

    void window(int proposed, int matched, int64_t max_new, int eos_row = -1, bool bounded = true) {
        last_t = bounded ? strata::program::serve_window_size(proposed, max_new - produced) : proposed;
        if (last_t == 0) return;
        std::vector<int32_t> input((size_t) last_t), output((size_t) last_t);
        std::iota(output.begin(), output.end(), (int32_t) emitted.size());
        if (eos_row >= 0 && eos_row < last_t) output[(size_t) eos_row] = eos_token;
        input[0] = emitted.back();
        for (int i = 1; i < last_t; ++i)
            input[(size_t) i] = i <= matched ? output[(size_t) i - 1] : -999;
        int a = 0;
        while (a < last_t - 1 && input[(size_t) a + 1] == output[(size_t) a]) ++a;
        if (bounded)
            a = strata::program::serve_output_count(output.data(), a + 1,
                    [](int32_t token) { return token == eos_token; }) - 1;
        last_keep = a + 1;
        consumed.insert(consumed.end(), input.begin(), input.begin() + last_keep);
        last_emitted = 0;
        for (int i = 0; i <= a && produced < max_new && !eos; ++i) {
            emitted.push_back(output[(size_t) i]);
            ++produced;
            ++last_emitted;
            eos = output[(size_t) i] == eos_token;
        }
    }

    bool exact() const {
        return consumed.size() + 1 == emitted.size() &&
               std::equal(consumed.begin(), consumed.end(), emitted.begin());
    }
    bool reusable() const {
        auto followup = emitted;
        followup.push_back(999999); // the next chat turn must not be mistaken for a hidden generated token
        return consumed.size() <= followup.size() &&
               std::equal(consumed.begin(), consumed.end(), followup.begin());
    }
};
} // namespace

int main() {
    try {
        // Exact real failure: prompt 8562, 62/64 outputs, p=8623, T=4, all drafts accepted.
        Prefix old(8562, 62);
        old.window(4, 3, 64, -1, false);
        require(old.consumed.size() == 8627 && old.emitted.size() == 8626 && !old.reusable(),
                "old final window commits an invisible token that breaks the next chat prefix");
        Prefix fixed(8562, 62);
        fixed.window(4, 3, 64);
        require(fixed.last_t == 2 && fixed.produced == 64 && fixed.consumed.size() == 8625 &&
                fixed.exact() && fixed.reusable(), "bounded final window keeps all 8625 reusable inputs");

        Prefix old_eos(7);
        old_eos.window(4, 3, 64, 0, false);
        require(old_eos.eos && !old_eos.reusable(), "old accepted tail after EOS also breaks reuse");
        Prefix fixed_eos(7);
        fixed_eos.window(4, 3, 64, 0);
        require(fixed_eos.eos && fixed_eos.last_keep == 1 && fixed_eos.exact() && fixed_eos.reusable(),
                "EOS commits its input only, leaving EOS as the unconsumed head");

        // Remaining output capacity, MTP/lookup proposals, partial/full matches, and EOS both
        // inside the accepted prefix and in rejected/unverified tails. Each case also models
        // STOP at this completed window: the exact emitted prefix is already reusable.
        for (int proposed = 1; proposed <= 8; ++proposed)
            for (int remaining = 1; remaining <= 9; ++remaining)
                for (int matched = 0; matched < proposed; ++matched)
                    for (int eos_row = -1; eos_row < proposed; ++eos_row) {
                        Prefix p(5, 2);
                        p.window(proposed, matched, 2 + remaining, eos_row);
                        require(p.exact() && p.reusable(), "every emitted prefix remains exact and reusable");
                        require(p.last_keep == p.last_emitted && p.produced <= 2 + remaining,
                                "commit count equals emission count without crossing the output cap");
                        const bool expected_eos = eos_row >= 0 && eos_row <= matched &&
                                                  eos_row < remaining && eos_row < proposed;
                        require(p.eos == expected_eos, "EOS in a rejected or unverified tail cannot stop the request");
                    }
        // Multiple windows preserve the fed-back head, including a partial match before the last cap.
        Prefix p(19);
        for (int step = 0; p.produced < 23; ++step) {
            p.window(step == 0 ? 1 : 4, step % 4, 23);
            require(p.exact() && p.reusable(), "successive windows preserve the same emitted history");
        }
        require(p.produced == 23 && p.consumed.size() == 41, "final head is not committed twice");
        require(strata::program::serve_window_size(8, std::numeric_limits<int64_t>::max()) == 8 &&
                strata::program::serve_window_size(8, 0) == 0,
                "large or exhausted output budgets do not overflow the window count");
        std::printf("serve window: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
