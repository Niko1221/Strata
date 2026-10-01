#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
// The Python grammar matcher sends little-endian packed permission bits as hex.
// Reject malformed or empty masks; sampling an all-masked vocabulary is undefined.
inline bool decode_token_mask(const std::string& line, size_t vocab, std::vector<uint32_t>& out, std::string& err) {
    const size_t words = (vocab + 31) / 32;
    if (!vocab || line.rfind("MASK ", 0) != 0 || line.size() != 5 + words * 8) {
        err = "invalid grammar mask size"; return false;
    }
    out.assign(words, 0);
    auto digit = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < words * 4; ++i) {
        const int a = digit(line[5 + i * 2]), b = digit(line[6 + i * 2]);
        if (a < 0 || b < 0) { err = "invalid grammar mask hex"; return false; }
        out[i / 4] |= uint32_t((a << 4) | b) << ((i % 4) * 8);
    }
    if (vocab % 32) out.back() &= (uint32_t(1) << (vocab % 32)) - 1;
    for (uint32_t word : out) if (word) return true;
    err = "grammar mask permits no token"; return false;
}
}
