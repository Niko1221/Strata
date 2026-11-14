// Host-side regression harness for the SYCL device token mask (test_tmask_apply.sh drives it).
//
// The mask alignment lane found (2026-11-14, T2 decision memo in notes/pr3800-mask-alignment-findings.md)
// that tmask_apply_launch in sycl/src/core/verify.cpp read the MKN bitmap at words[v >> 5] without the row
// offset: every window row was masked with row 0's bitmap, so rows 1..T-1 of an MTP window lost their own
// masks (row 0 stayed correct, the host fallback fired on rows 1-3, forced lowest-ids produced padding
// strings, finish_reason=length).  The CUDA side (src/core/verify.cpp, mask_apply_row per row pointer) kept
// the per-row pointers and never corrupted.  The kernel body is extracted VERBATIM from
// sycl/src/core/verify.cpp by the driver into tmask_apply_extracted.inc, so this harness always tests the
// live source text, not a copy of it.
//
// What it asserts (the MKN layout the caller documents at the call site: "row t is words + t * n_words"):
//   1. every row of a multi-row window is masked with its OWN bitmap row (the MTP multi-row window holds);
//   2. the per-row allowed counts are the row's own counts;
//   3. a token at or past 32 * n_words stays masked on every row (the w < n_words guard, unchanged).
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

// ---- the minimal SYCL/dpct surface the extracted launch function touches ----
namespace sycl {
template <int D>
struct range {
    std::size_t e[2];
    range(std::size_t a, std::size_t b) : e{a, b} {}
    std::size_t get(int d) const { return e[d]; }
};
template <int D>
struct item {
    std::size_t c[2];
    std::size_t operator[](int d) const { return c[d]; }
};
struct handler {
    template <typename K, typename F>
    void parallel_for(range<2> r, F f) {  // sequential device: order cannot hide a row-offset bug here
        for (std::size_t t = 0; t < r.get(0); ++t)
            for (std::size_t v = 0; v < r.get(1); ++v) f(item<2>{{t, v}});
    }
};
namespace access {
enum class address_space { generic_space };
}  // namespace access
}  // namespace sycl

namespace dpct {
template <sycl::access::address_space AS, typename T, typename S>
T atomic_fetch_add(T* addr, S val) {
    T old = *addr;
    *addr = static_cast<T>(old + val);
    return old;
}
struct queue {
    template <typename F>
    void submit(F f) {
        sycl::handler h;
        f(h);
    }
};
using queue_ptr = queue*;
}  // namespace dpct

namespace strata {
namespace core {
constexpr float kMaskedLogit = -1e30f;  // the constant from include/strata/core/token_mask.hpp
}  // namespace core
}  // namespace strata

struct strata_tmask_apply_kernel;  // the kernel tag

#include "tmask_apply_extracted.inc"  // the live tmask_apply_launch from sycl/src/core/verify.cpp

// ---- assertions ----
static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

// One case: a rows x n_vocab window masked by the rows x n_words bitmap `words`, launched through the
// extracted kernel.  `allowed[t]` is the single token each row's own bitmap allows; every other token of
// every row must end up at kMaskedLogit, and every count must be 1.
static void run_case(const char* name, int rows, int64_t n_vocab, const std::vector<uint32_t>& words,
                     int64_t n_words, const std::vector<int64_t>& allowed) {
    std::vector<float> logits(static_cast<size_t>(rows) * static_cast<size_t>(n_vocab), 0.0f);
    std::vector<int32_t> counts(static_cast<size_t>(rows), 0);
    dpct::queue q;
    tmask_apply_launch(&q, logits.data(), n_vocab, words.data(), n_words, rows, counts.data());
    for (int t = 0; t < rows; ++t) {
        char msg[192];
        std::snprintf(msg, sizeof msg, "%s: row %d counts its own allowed token (got %d, want 1)", name, t,
                      counts[static_cast<size_t>(t)]);
        expect(counts[static_cast<size_t>(t)] == 1, msg);
        for (int64_t v = 0; v < n_vocab; ++v) {
            const float val = logits[static_cast<size_t>(t) * static_cast<size_t>(n_vocab) +
                                     static_cast<size_t>(v)];
            const bool keep = v == allowed[static_cast<size_t>(t)];
            if (keep) {
                std::snprintf(msg, sizeof msg, "%s: row %d keeps its own allowed token %lld unmasked", name, t,
                              static_cast<long long>(v));
            } else {
                std::snprintf(msg, sizeof msg, "%s: row %d masks token %lld (row %d's bitmap must not leak)",
                              name, t, static_cast<long long>(v), t);
            }
            expect(keep ? std::fabs(val) < 1e-6f : std::fabs(val - strata::core::kMaskedLogit) < 1e-20f, msg);
        }
    }
    std::printf("%s: %s\n", name, failures == 0 ? "ok" : "see FAIL lines above");
}

int main() {
    const int before = failures;
    // Case 1 (the reported shape, shrunk): a 4-row MTP window, one word per row, each row allowing a
    // different token.  With the missing row offset every row is masked by row 0's 0x01: rows 1-3 count 0
    // and token 0 of rows 1-3 stays unmasked while each row's own allowed token is crushed.
    run_case("multi-row window, one word", 4, 8, {0x01u, 0x02u, 0x04u, 0x08u}, 1, {0, 1, 2, 3});
    // Case 2: two words per row - the offset bug also breaks the second word of every row, so put the two
    // rows' allowed tokens in different words.
    // (the words are the row-major MKN layout: row 0's two words, then row 1's two words)
    run_case("multi-row window, two words", 2, 40,
             {0x00000000u, 0x00000002u /*row 0 allows token 33*/, 0x00000020u /*row 1 allows token 5*/,
              0x00000000u},
             2, {33, 5});
    // Case 3: the guard.  Tokens at or past 32 * n_words are masked on every row, whatever the row.
    run_case("past-the-bitmap tokens stay masked", 2, 70, {0x01u, 0x02u}, 1, {0, 1});
    // (case 3's rows only allow tokens 0/1; tokens 32..69 are past the bitmap and asserted masked inside
    // run_case by the kMaskedLogit check - counts stay 1 because only the allowed token bumps.)
    if (failures == before) {
        std::printf("PASS: tmask_apply_launch masks every window row with its own bitmap row\n");
        return 0;
    }
    std::printf("FAIL: %d assertion(s) failed - the window rows do not keep their own masks\n",
                failures - before);
    return 1;
}
