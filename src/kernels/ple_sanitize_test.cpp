// src/kernels/ple_sanitize_test.cpp - ple_sanitize_row (#3602/#879), CPU only, no model and no GPU.
//
// Every PleTable::Impl::decode() call (any table format: IQ4_NL, Q5_0, F8_E4M3, Q5_1, Q8_0, Q4_0, BF16, NVFP4)
// routes its output through this function before returning it. The cases below replay the exact decoded
// values STRATA_PLE_CHECK reported from the real on-disk table during the #879 investigation (a literal NaN
// and several multi-hundred-thousand-to-million magnitudes), plus the boundary the fix is built on: `> 1e5f`,
// the same threshold check_row already uses to flag a bad row.
#include "strata/kernels/ngram.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

using strata::kernels::PLE_HEAD_DIM;
using strata::kernels::ple_sanitize_row;

namespace {

int failures = 0;

void expect_row(const char* what, float* row, const float* want) {
    for (int j = 0; j < PLE_HEAD_DIM; ++j) {
        const bool want_nan = std::isnan(want[j]);
        const bool got_nan = std::isnan(row[j]);
        if (want_nan != got_nan || (!want_nan && row[j] != want[j])) {
            std::printf("  %-42s *** MISMATCH at [%d]: got %g, want %g\n", what, j, (double) row[j], (double) want[j]);
            ++failures;
            return;
        }
    }
    std::printf("  %-42s ok\n", what);
}

/// A row of unremarkable, in-range values (the kind every one of the table's ~320M good rows holds):
/// unaffected in every position.
void test_normal_row_untouched() {
    float row[PLE_HEAD_DIM], want[PLE_HEAD_DIM];
    for (int j = 0; j < PLE_HEAD_DIM; ++j) {
        row[j] = want[j] = (float) (j - PLE_HEAD_DIM / 2) * 0.37f;   // spans small negative and positive
    }
    ple_sanitize_row(row);
    expect_row("normal in-range row: untouched", row, want);
}

/// Values exactly at and either side of the 1e5 boundary: the comparison is strictly `> 1e5f`, so 1e5 itself
/// and anything under it must survive; anything over must not.
void test_boundary() {
    float row[PLE_HEAD_DIM], want[PLE_HEAD_DIM];
    for (int j = 0; j < PLE_HEAD_DIM; ++j) row[j] = want[j] = 0.0f;
    row[0] = want[0] = 99999.0f;          // just under: kept
    row[1] = want[1] = 100000.0f;         // exactly 1e5: kept (not `>`)
    row[2] = 100000.01f; want[2] = 0.0f;  // just over: zeroed
    row[3] = -99999.0f; want[3] = -99999.0f;   // the negative side is symmetric via fabs()
    row[4] = -100000.01f; want[4] = 0.0f;
    ple_sanitize_row(row);
    expect_row("1e5 boundary (> is strict)", row, want);
}

/// A literal NaN in one position - what STRATA_PLE_CHECK found coming straight out of the file's own bytes
/// for one real row (`gather_batch: row 25108390 ... value 0 of 160 decodes to nan`).
void test_nan() {
    float row[PLE_HEAD_DIM], want[PLE_HEAD_DIM];
    for (int j = 0; j < PLE_HEAD_DIM; ++j) row[j] = want[j] = 1.5f;
    row[0] = std::nanf("");
    want[0] = 0.0f;
    ple_sanitize_row(row);
    expect_row("a literal NaN value", row, want);
}

/// +Inf and -Inf: fabs(inf) > 1e5f is true, so both are caught by the same branch as a huge finite value.
void test_infinities() {
    float row[PLE_HEAD_DIM], want[PLE_HEAD_DIM];
    for (int j = 0; j < PLE_HEAD_DIM; ++j) row[j] = want[j] = 2.0f;
    row[10] = std::numeric_limits<float>::infinity();
    row[20] = -std::numeric_limits<float>::infinity();
    want[10] = want[20] = 0.0f;
    ple_sanitize_row(row);
    expect_row("+Inf and -Inf", row, want);
}

/// The exact magnitudes STRATA_PLE_CHECK reported from real, on-disk rows during the #879 investigation
/// (each "the delivered bytes ARE the file's bytes" - a genuine table defect, not a misread). All must zero.
void test_real_world_bad_values() {
    static const float kObserved[] = {
        262906.0f, -4130080.0f, -608224.0f, -197808.0f, 170768.0f, 688860.0f,
    };
    float row[PLE_HEAD_DIM], want[PLE_HEAD_DIM];
    for (int j = 0; j < PLE_HEAD_DIM; ++j) row[j] = want[j] = 3.25f;
    int j = 0;
    for (float v : kObserved) { row[j] = v; want[j] = 0.0f; ++j; }
    ple_sanitize_row(row);
    expect_row("observed real-world bad magnitudes", row, want);
}

/// Two bad values in the same row (the IQ4_NL format's 5 blocks of 32 can each carry their own scale, so more
/// than one block going bad in the same row is a real shape, not a hypothetical): both zero, the rest intact.
void test_two_bad_in_one_row() {
    float row[PLE_HEAD_DIM], want[PLE_HEAD_DIM];
    for (int j = 0; j < PLE_HEAD_DIM; ++j) row[j] = want[j] = (float) j * 0.01f;
    row[32] = std::nanf(""); want[32] = 0.0f;       // block 1
    row[96] = 9.0e6f; want[96] = 0.0f;              // block 3
    ple_sanitize_row(row);
    expect_row("two independent bad values, one row", row, want);
}

}  // namespace

int main() {
    test_normal_row_untouched();
    test_boundary();
    test_nan();
    test_infinities();
    test_real_world_bad_values();
    test_two_bad_in_one_row();
    std::printf("\nple_sanitize_test %s (%d failure%s)\n", failures == 0 ? "OK" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
