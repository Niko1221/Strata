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
    err.clear();

    // ---- minor 7 (2026-11-05, expert review): a named position at/after the prompt end is a
    // DRAFT row (only a decode window would cover it) and is never captured - even when a
    // window offered to capture_window_rows covers it.  set_row_capture's third argument is
    // one past the last prompt position (generate.cpp passes the request's n).
    {
        strata::core::Verifier v2;
        std::string err2;
        const int64_t past[1] = {10};
        v2.set_row_capture(past, 1, /*limit: prompt length*/ 8);
        // a window [0, 64) WOULD cover position 10; the prompt ends at 8, so the row at 10 is
        // not a prompt row: nothing is captured (BROWAT then refuses the position by name).
        if (!v2.capture_window_rows(0, 64, err2)) {
            std::printf("FAIL: capture_window_rows failed on a past-the-end position: %s\n", err2.c_str());
            return 1;
        }
        if (v2.captured_rows() != 0) {
            std::printf("FAIL: a draft row (position >= n) was captured\n");
            return 1;
        }
        // and the refused position is not readable by name either
        if (v2.brow_row_captured(10, &one, 1, out, err2)) {
            std::printf("FAIL: brow_row_captured answered a position that was never captured\n");
            return 1;
        }
        err2.clear();
    }

    // ---- M10 (2026-11-05, expert review): clear_row_capture() drops the whole per-request
    // row state INCLUDING the last window's row count.  generate.cpp calls it at the top of
    // every request, so a request that fails before running a window leaves BROW answering
    // "no window" instead of the PREVIOUS request's row.  (Running a real window needs a
    // model; what is pinned here is the reset contract of the accessor generate.cpp relies on.)
    {
        strata::core::Verifier v3;
        std::string err3;
        const int64_t want[1] = {3};
        v3.set_row_capture(want, 1, /*limit*/ 8);
        v3.clear_row_capture();                    // what every request start does
        if (v3.brow_rows() != 0) {
            std::printf("FAIL: brow_rows() is %d after clear_row_capture (want 0)\n", v3.brow_rows());
            return 1;
        }
        if (v3.brow_row( &one, 1, out, err3)) {
            std::printf("FAIL: brow_row succeeded after clear_row_capture\n");
            return 1;
        }
        if (err3.empty()) {
            std::printf("FAIL: brow_row set no error after clear_row_capture\n");
            return 1;
        }
    }

    // The real window path needs a model; the accessor contract is the shippable unit here.
    std::printf("verify_brow_at_test: OK (fail-closed pre-run contract, row bound, no-token refusal, "
                "draft-row refusal, clear resets the window row count)\n");
    return 0;
}
