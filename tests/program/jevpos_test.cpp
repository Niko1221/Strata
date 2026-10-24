// tests/program/jevpos_test.cpp - the `jevpos=` request key's parser (jev fix B).  Pure string -> vector: no model,
// no GPU, no server, so it runs anywhere.
//
// This file exists because of the round-2 review (2026-10-23): the parser had no test on EITHER side of the wire.
// The Python tests drive a fake engine whose INFO advertises `jevpos=1` and whose pins list is asserted in Python,
// so the real parser was never fed a single string.  A dangling `else` therefore pushed each position once per
// DIGIT character and counted the 32-entry cap in digits - `63,128` parsed to [63,63,128,128,128], and a legal
// 32-position list was refused whole, which silently dropped the consolidated path to one prompt per field.
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "strata/program/jevpos.hpp"

using strata::program::jevpos::kMaxEntries;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static std::vector<int64_t> range(int64_t lo, int64_t hi) {
    std::vector<int64_t> v;
    for (int64_t i = lo; i < hi; ++i) v.push_back(i);
    return v;
}

int main() {
    std::vector<int64_t> v;

    // --- the measured defect: multi-digit positions were pushed once per digit ---
    CHECK(strata::program::jevpos::parse("63,128", v));
    CHECK(v == std::vector<int64_t>({63, 128}));
    CHECK(strata::program::jevpos::parse("412,530,661", v));
    CHECK(v == std::vector<int64_t>({412, 530, 661}));
    CHECK(strata::program::jevpos::parse("7,8,9", v));
    CHECK(v == std::vector<int64_t>({7, 8, 9}));

    // --- the cap counts ENTRIES, not digits: 32 legal positions are kept, the 33rd refuses the list whole ---
    std::string list32;
    for (std::size_t i = 0; i < kMaxEntries; ++i) {
        if (i) list32 += ",";
        list32 += std::to_string(i);
    }
    CHECK(strata::program::jevpos::parse(list32, v));
    CHECK(v.size() == kMaxEntries);
    CHECK(v == range(0, kMaxEntries));

    // 32 positions of 3 digits each: 96 digits, still 32 entries -> accepted (this is the case a digit cap killed)
    std::string list100;
    for (std::size_t i = 0; i < kMaxEntries; ++i) {
        if (i) list100 += ",";
        list100 += std::to_string(100 + i);
    }
    CHECK(strata::program::jevpos::parse(list100, v));
    CHECK(v.size() == kMaxEntries);
    CHECK(v.front() == 100 && v.back() == 100 + (int64_t) kMaxEntries - 1);

    // the 33rd entry: refused WHOLE, never truncated
    std::string list33;
    for (std::size_t i = 0; i <= kMaxEntries; ++i) {
        if (i) list33 += ",";
        list33 += std::to_string(i);
    }
    CHECK(!strata::program::jevpos::parse(list33, v));
    CHECK(v.empty());

    // --- a malformed list is no list: never a guessed position ---
    CHECK(!strata::program::jevpos::parse("12x", v));      // used to parse as 12 (atoll stops at the junk)
    CHECK(v.empty());
    CHECK(!strata::program::jevpos::parse("4;9", v));
    CHECK(v.empty());
    CHECK(!strata::program::jevpos::parse("-4", v));
    CHECK(v.empty());
    CHECK(!strata::program::jevpos::parse(" 4", v));
    CHECK(v.empty());
    CHECK(!strata::program::jevpos::parse("9999999999999999999", v));   // 19 digits: past int64, refused
    CHECK(v.empty());

    // a refusal also empties a previously good vector (the caller reuses req_jevpos across requests)
    CHECK(strata::program::jevpos::parse("4,9", v));
    CHECK(!strata::program::jevpos::parse("4,x", v));
    CHECK(v.empty());

    // --- edges the engine has always allowed ---
    CHECK(strata::program::jevpos::parse("", v));          // no positions named: not an error
    CHECK(v.empty());
    CHECK(strata::program::jevpos::parse("4,,9", v));      // an empty field between commas is skipped
    CHECK(v == std::vector<int64_t>({4, 9}));
    CHECK(strata::program::jevpos::parse("4,", v));
    CHECK(v == std::vector<int64_t>({4}));
    CHECK(strata::program::jevpos::parse("0", v));
    CHECK(v == std::vector<int64_t>({0}));
    CHECK(strata::program::jevpos::parse("2147483000", v));   // 10 digits, one entry, no wrap
    CHECK(v == std::vector<int64_t>({2147483000LL}));

    if (failed) {
        std::fprintf(stderr, "jevpos_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("jevpos_test: all checks passed\n");
    return 0;
}
