#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace p100_test {
struct Case { int a, b, c; };
inline int from_bits(uint32_t u) {
    int i;
    static_assert(sizeof(i) == sizeof(u), "32-bit int required");
    std::memcpy(&i, &u, sizeof(i));
    return i;
}
// Independent oracle: signed-byte array, widened sum, defined modulo conversion.
inline int oracle(int a, int b, int c) {
    int8_t av[4], bv[4];
    std::memcpy(av, &a, 4);
    std::memcpy(bv, &b, 4);
    int64_t sum = c;
    for (int j = 0; j < 4; ++j) sum += int64_t(av[j]) * bv[j];
    return from_bits(static_cast<uint32_t>(sum));
}
// Model of the four PTX operations, not execution of the production assembly.
inline int vmad_model(int a, int b, int c) {
    uint32_t result = static_cast<uint32_t>(c);
    for (int j = 0; j < 4; ++j) {
        int av = (static_cast<uint32_t>(a) >> (8 * j)) & 255;
        int bv = (static_cast<uint32_t>(b) >> (8 * j)) & 255;
        if (av >= 128) av -= 256;
        if (bv >= 128) bv -= 256;
        result += static_cast<uint32_t>(av * bv);
    }
    return from_bits(result);
}
inline std::vector<Case> cases() {
    std::vector<Case> out;
    // All 65,536 signed byte pairs in every selector, with hostile surrounding bytes.
    for (unsigned lane = 0; lane < 4; ++lane) {
        const uint32_t mask = ~(255u << (8 * lane));
        for (unsigned a = 0; a < 256; ++a)
            for (unsigned b = 0; b < 256; ++b)
                out.push_back({from_bits((0x807fff01u & mask) | (a << (8 * lane))),
                               from_bits((0xff80017fu & mask) | (b << (8 * lane))), 17});
    }
    const uint32_t edges[] = {0, 1, 0x01010101u, 0x7f7f7f7fu, 0x80808080u,
                             0xffffffffu, 0x7fffffffu, 0x80000000u};
    for (auto a : edges) for (auto b : edges) for (auto c : edges)
        out.push_back({from_bits(a), from_bits(b), from_bits(c)});
    // Intermediate overflow followed by cancellation back into the signed range.
    out.push_back({0x00000101, 0x0000ff01, from_bits(0x7fffffffu)});
    out.push_back({0x00000101, 0x000001ff, from_bits(0x80000000u)});
    uint32_t state = 0x70617363u;
    auto next = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (int j = 0; j < 100000; ++j) {
        const int a = from_bits(next()), b = from_bits(next()), c = from_bits(next());
        out.push_back({a, b, c});
    }
    return out;
}
}
