// tests/core/decode_tuning_test.cpp - the decode tuning table: what it accepts, and every refusal (no GPU).
#include "strata/kernels/decode_tuning.hpp"

#include <cstdio>
#include <sstream>
#include <string>

using namespace strata::kernels;

namespace {
int failures = 0;
void expect(bool ok, const char* what, const std::string& detail = {}) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s %s\n", what, detail.c_str());
        ++failures;
    }
}

DecodeIdentity me() { return DecodeIdentity{"gfx1100", 70253211, "0123456789abcdef", kDecodeTuningSpace}; }

bool parse(const std::string& text, DecodeTuningTable& t, std::string& err) {
    std::istringstream in(text);
    return t.parse(in, me(), err);
}
}  // namespace

int main() {
    const std::string head = DecodeTuningTable::header(me()) + "\n";
    {
        DecodeTuningTable t;
        std::string err;
        const bool ok = parse("# measured\n" + head + "gu 18 2560 640 16  # gate/up\n down 20 640 2560 4\nmmvq 18 2560 4096 8\n", t, err);
        expect(ok, "a valid table parses", err);
        expect(t.entries().size() == 3, "three shapes");
        expect(t.rows(DecodeKernel::GateUp, 18, 2560, 640, 8) == 16, "gu lookup");
        expect(t.rows(DecodeKernel::Down, 20, 640, 2560, 8) == 4, "down lookup");
        expect(t.rows(DecodeKernel::Mmvq, 18, 2560, 4096, 4) == 8, "mmvq lookup");
        expect(t.rows(DecodeKernel::GateUp, 21, 2560, 640, 8) == 8, "another type keeps the default");
        expect(t.rows(DecodeKernel::Down, 18, 2560, 640, 8) == 8, "the kernel is part of the key");
        // writing what was read gives the same table
        std::string text = head;
        for (const DecodeTuningRow& r : t.entries()) text += DecodeTuningTable::line(r) + "\n";
        DecodeTuningTable t2;
        expect(parse(text, t2, err) && t2.entries().size() == 3, "header() and line() round-trip", err);
    }
    {
        DecodeTuningTable t;
        std::string err;
        expect(parse(head, t, err) && t.empty(), "a header alone is a valid, empty table", err);
    }
    struct Bad { const char* why; std::string text; const char* says; };
    const Bad bad[] = {
        {"no header", "gu 18 2560 640 8\n", "header"},
        {"empty file", "", "empty"},
        {"other arch", "STRATA_DECODE_TUNING_V1 gfx1101 70253211 0123456789abcdef 2\n", "gfx1101"},
        {"other runtime", "STRATA_DECODE_TUNING_V1 gfx1100 60000000 0123456789abcdef 2\n", "runtime"},
        {"other compiler", "STRATA_DECODE_TUNING_V1 gfx1100 70253211 fedcba9876543210 2\n", "compiler"},
        {"older table, no space", "STRATA_DECODE_TUNING_V1 gfx1100 70253211 0123456789abcdef\n", "run the tuner again"},
        {"other selection space", "STRATA_DECODE_TUNING_V1 gfx1100 70253211 0123456789abcdef 1\n", "selection space 1"},
        {"short header", "STRATA_DECODE_TUNING_V1 gfx1100\n", "header needs"},
        {"unknown kernel", head + "attn 18 2560 640 8\n", "expected"},
        {"missing field", head + "gu 18 2560 640\n", "expected"},
        {"extra field", head + "gu 18 2560 640 8 9\n", "expected"},
        {"rows not compiled", head + "gu 18 2560 640 32\n", "not compiled"},
        {"mmvq rows not compiled", head + "mmvq 18 2560 640 16\n", "not compiled"},
        {"zero shape", head + "gu 18 0 640 8\n", "positive"},
        {"duplicate", head + "gu 18 2560 640 8\ngu 18 2560 640 4\n", "twice"},
    };
    for (const Bad& b : bad) {
        DecodeTuningTable t;
        std::string err;
        const bool ok = parse(b.text, t, err);
        expect(!ok, b.why, "was accepted");
        expect(err.find(b.says) != std::string::npos, b.why, "message: " + err);
    }
    for (DecodeKernel k : {DecodeKernel::GateUp, DecodeKernel::Down, DecodeKernel::Mmvq}) {
        const std::vector<int> c = decode_rows_candidates(k);
        expect(!c.empty() && c.front() == decode_default_rows(k), "candidates start with the default");
        for (int r : c) expect(decode_rows_valid(k, r) && r * 32 <= 512, "every candidate is valid");
    }
    expect(decode_toolchain_hash("a").size() == 16 && decode_toolchain_hash("a") != decode_toolchain_hash("b"),
           "toolchain hash");
    DecodeTuningTable missing;
    std::string err;
    expect(!missing.load("/nonexistent/decode-tuning.txt", me(), err) && err.find("cannot open") != std::string::npos,
           "a missing file is refused");
    if (failures == 0) std::printf("decode_tuning_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
