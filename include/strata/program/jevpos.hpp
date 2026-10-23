// include/strata/program/jevpos.hpp - the `jevpos=` request key: the absolute prompt positions whose head rows the
// request keeps, so a later BROWAT can read one field's cue row out of the single prefill that ran N fields (jev fix
// B).  Header-only and dependency-free on purpose: this is pure string -> vector logic, and the round-2 review
// (2026-10-23) found it had no test on either side of the wire, which is how a dangling `else` pushed a position once
// per digit character and counted the 32-entry cap in digits.  tests/program/jevpos_test.cpp pins it with no model,
// no GPU and no server.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program::jevpos {

/// The most entries a list may name.  `serve/jev.py` gates a request on the same number (MAX_NAMED_ROWS) and a
/// longer list is refused WHOLE, never truncated: a truncated list would name fewer rows than the caller reads.
constexpr std::size_t kMaxEntries = 32;

/// A field longer than this cannot be a prompt position (a signed 64-bit value holds 19 digits), and it keeps the
/// accumulation below out of signed overflow, which would let an absurd field become a plausible position.
constexpr std::size_t kMaxDigits = 18;

/// Parse the text after `jevpos=` ("p1,p2,...") into `out`, which is cleared first.
///
/// The list is accepted only when EVERY field is digits, short enough, and there are at most kMaxEntries of them.
/// On any doubt `out` is left EMPTY and false is returned: a malformed list is no list, never a guessed position -
/// a guessed position answers a different field's question, and it looks right.  An empty field between commas is
/// skipped, as the engine has always done ("4,,9" is 4 and 9; "4," is 4).
inline bool parse(const std::string& list, std::vector<int64_t>& out) {
    std::vector<int64_t> kept;
    bool bad = false;
    std::size_t at = 0;
    while (at <= list.size() && !bad) {
        const std::size_t comma = list.find(',', at);
        const std::string one = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (!one.empty()) {
            if (one.size() > kMaxDigits) {
                bad = true;                       // not a position: refuse the list, do not wrap it
            } else {
                for (const char ch : one)
                    if (ch < '0' || ch > '9') { bad = true; break; }   // "12x" is not 12
                if (!bad && kept.size() >= kMaxEntries) bad = true;    // the 33rd entry: more than the cap
                if (!bad) {
                    int64_t v = 0;
                    for (const char ch : one) v = v * 10 + (int64_t) (ch - '0');
                    kept.push_back(v);
                }
            }
        }
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    if (bad) {
        out.clear();
        return false;
    }
    out.swap(kept);
    return true;
}

}  // namespace strata::program::jevpos
