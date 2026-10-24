// src/core/verify_brow_at_test.cpp - Verifier::brow_row_at(): row t of the LAST window's head logits,
// the read the jev systemone endpoint (M3.1) asks for over the BROWAT wire line.
//
// brow_row() is brow_row_at(final row); this test pins the part that makes BROWAT safe: a row the
// window did not compute is refused by name, never answered with the final row.  The full Verifier
// needs a model, so - like verify_brow_test - the shippable unit here is the accessor contract: the
// fail-closed pre-run path and the row bound, both of which a real model cannot change.

#include "strata/core/verify.hpp"

#include <cstdio>
#include <string>

int main() {
    strata::core::Verifier v;
    std::string err;
    float out[1] = {0.0f};
    const int32_t one = 1;

    // No window has run: every row is refused, the final-row delegate included.
    if (v.brow_rows() != 0) {
        std::printf("FAIL: brow_rows() is %d before any window (want 0)\n", v.brow_rows());
        return 1;
    }
    if (v.brow_row_at(0, &one, 1, out, err)) {
        std::printf("FAIL: brow_row_at succeeded before any window ran\n");
        return 1;
    }
    if (err.empty()) {
        std::printf("FAIL: brow_row_at set no error before any window ran\n");
        return 1;
    }
    err.clear();

    // The row bound is checked before the tokens: row 5 of a window that never ran is out of range.
    if (v.brow_row_at(5, &one, 1, out, err)) {
        std::printf("FAIL: brow_row_at succeeded for a row of no window\n");
        return 1;
    }
    if (err.empty()) {
        std::printf("FAIL: brow_row_at set no error for a row of no window\n");
        return 1;
    }
    err.clear();

    // A negative row is never the final row in disguise.
    if (v.brow_row_at(-1, &one, 1, out, err)) {
        std::printf("FAIL: brow_row_at accepted row -1\n");
        return 1;
    }
    if (err.empty()) {
        std::printf("FAIL: brow_row_at set no error for row -1\n");
        return 1;
    }
    err.clear();

    // No tokens: refused, as BROW is.
    if (v.brow_row_at(0, nullptr, 0, nullptr, err)) {
        std::printf("FAIL: brow_row_at succeeded with no tokens\n");
        return 1;
    }
    if (err.empty()) {
        std::printf("FAIL: brow_row_at set no error with no tokens\n");
        return 1;
    }

    // The real window path needs a model; the accessor contract is the shippable unit here.
    std::printf("verify_brow_at_test: OK (fail-closed pre-run contract, row bound, no-token refusal)\n");
    return 0;
}
