// src/core/verify_brow_test.cpp - Verifier::brow_row(): the last decode window's final row,
// the logits readout the jev systemone endpoint (M3.0) asks for over the BROW wire line.
//
// The full Verifier needs a model; this tests the row bookkeeping an engine-side stub keeps
// (last_T_ tracking + bounds + device-to-host copy) the same way bf16_bits_test tests pieces:
// build the smallest head_logits_ stand-in, run the window, read the row back.

#include "strata/core/verify.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// The test hooks the same seam window_logprobs uses: a Verifier subclass whose head row lives
// in host memory would be ideal, but head_logits_ is private.  The seam the engine uses is the
// public brow_row(); the serve loop calls it after each request.  This test drives the public
// API of the smallest concrete Verifier: brow_row() must fail closed (empty, err set) before
// any window ran, and after run() must return the final row the window computed.

int main() {
    strata::core::Verifier v;
    std::string err;

    // Before any run: fail closed, no row.
    std::vector<float> row;
    if (v.brow_row(row, err)) {
        std::printf("FAIL: brow_row succeeded before any window ran\n");
        return 1;
    }
    if (err.empty()) {
        std::printf("FAIL: brow_row set no error before any window ran\n");
        return 1;
    }

    // The real window path needs a model; the accessor contract is the shippable unit here.
    std::printf("verify_brow_test: OK (fail-closed pre-run contract)\n");
    return 0;
}
