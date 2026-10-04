// src/kernels/decode_tuning.cpp - the decode tuning table (see include/strata/kernels/decode_tuning.hpp).
// Plain C++: no GPU headers, so the parser is tested without a GPU (tests/core/decode_tuning_test.cpp).
#include "strata/kernels/decode_tuning.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace strata::kernels {

namespace {
const int kExpertRows[] = {8, 4, 16, 2};   // default first
const int kMmvqRows[] = {4, 2, 8, 1};

bool kernel_from_name(const std::string& s, DecodeKernel& k) {
    if (s == "gu") { k = DecodeKernel::GateUp; return true; }
    if (s == "down") { k = DecodeKernel::Down; return true; }
    if (s == "mmvq") { k = DecodeKernel::Mmvq; return true; }
    return false;
}

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
}  // namespace

bool decode_rows_valid(DecodeKernel k, int rows) noexcept {
    if (k == DecodeKernel::Mmvq) return std::find(std::begin(kMmvqRows), std::end(kMmvqRows), rows) != std::end(kMmvqRows);
    return std::find(std::begin(kExpertRows), std::end(kExpertRows), rows) != std::end(kExpertRows);
}

std::vector<int> decode_rows_candidates(DecodeKernel k) {
    if (k == DecodeKernel::Mmvq) return {std::begin(kMmvqRows), std::end(kMmvqRows)};
    return {std::begin(kExpertRows), std::end(kExpertRows)};
}

const char* decode_kernel_name(DecodeKernel k) noexcept {
    switch (k) {
        case DecodeKernel::GateUp: return "gu";
        case DecodeKernel::Down: return "down";
        case DecodeKernel::Mmvq: return "mmvq";
    }
    return "?";
}

int decode_default_rows(DecodeKernel k) noexcept {
    return k == DecodeKernel::Mmvq ? kDefaultMmvqRows : kDefaultExpertRows;
}

std::string decode_toolchain_hash(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    char b[17];
    std::snprintf(b, sizeof(b), "%016llx", (unsigned long long) h);
    return b;
}

std::string DecodeTuningTable::header(const DecodeIdentity& id) {
    std::ostringstream o;
    o << "STRATA_DECODE_TUNING_V1 " << id.arch << ' ' << id.runtime << ' ' << id.toolchain;
    return o.str();
}

std::string DecodeTuningTable::line(const DecodeTuningRow& r) {
    std::ostringstream o;
    o << decode_kernel_name(r.kernel) << ' ' << r.type << ' ' << r.n_in << ' ' << r.n_out << ' ' << r.rows;
    return o.str();
}

bool DecodeTuningTable::parse(std::istream& in, const DecodeIdentity& want, std::string& err) {
    rows_.clear();
    std::string raw;
    bool have_header = false;
    int lineno = 0;
    while (std::getline(in, raw)) {
        ++lineno;
        const size_t hash = raw.find('#');
        const std::string s = trim(hash == std::string::npos ? raw : raw.substr(0, hash));
        if (s.empty()) continue;
        std::istringstream ls(s);
        std::string word;
        ls >> word;
        if (!have_header) {
            if (word != "STRATA_DECODE_TUNING_V1") {
                err = "line " + std::to_string(lineno) + ": expected the STRATA_DECODE_TUNING_V1 header";
                return false;
            }
            DecodeIdentity got;
            if (!(ls >> got.arch >> got.runtime >> got.toolchain)) {
                err = "line " + std::to_string(lineno) + ": the header needs <arch> <runtime> <toolchain>";
                return false;
            }
            if (got.arch != want.arch) {
                err = "made for " + got.arch + ", this GPU is " + want.arch;
                return false;
            }
            if (got.runtime != want.runtime) {
                err = "made with HIP runtime " + std::to_string(got.runtime) + ", this one is " + std::to_string(want.runtime);
                return false;
            }
            if (got.toolchain != want.toolchain) {
                err = "made for an engine built by another compiler (" + got.toolchain + ", this build is " +
                      want.toolchain + "): run the tuner again";
                return false;
            }
            have_header = true;
            continue;
        }
        DecodeTuningRow r;
        long long n_in = 0, n_out = 0;
        std::string extra;
        if (!kernel_from_name(word, r.kernel) || !(ls >> r.type >> n_in >> n_out >> r.rows) || (ls >> extra)) {
            err = "line " + std::to_string(lineno) + ": expected `gu|down|mmvq <type> <n_in> <n_out> <rows>`";
            return false;
        }
        r.n_in = n_in;
        r.n_out = n_out;
        if (r.type < 0 || r.n_in <= 0 || r.n_out <= 0) {
            err = "line " + std::to_string(lineno) + ": type and shape must be positive";
            return false;
        }
        if (!decode_rows_valid(r.kernel, r.rows)) {
            err = "line " + std::to_string(lineno) + ": " + std::to_string(r.rows) + " rows per block is not compiled for " +
                  decode_kernel_name(r.kernel);
            return false;
        }
        for (const DecodeTuningRow& q : rows_)
            if (q.kernel == r.kernel && q.type == r.type && q.n_in == r.n_in && q.n_out == r.n_out) {
                err = "line " + std::to_string(lineno) + ": shape listed twice";
                return false;
            }
        rows_.push_back(r);
    }
    if (!have_header) {
        err = "empty table (no STRATA_DECODE_TUNING_V1 header)";
        return false;
    }
    return true;
}

bool DecodeTuningTable::load(const std::string& path, const DecodeIdentity& want, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    return parse(f, want, err);
}

int DecodeTuningTable::rows(DecodeKernel k, int type, int64_t n_in, int64_t n_out, int fallback) const noexcept {
    for (const DecodeTuningRow& r : rows_)
        if (r.kernel == k && r.type == type && r.n_in == n_in && r.n_out == n_out) return r.rows;
    return fallback;
}

}  // namespace strata::kernels
