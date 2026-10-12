#pragma once
#include "strata/core/branch_digest.hpp"

// SHA-256, FIPS 180-4 sections 4.1.2, 4.2.2, 5 and 6.2:
// https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
// This implementation is checked against independent hashlib outputs. That
// comparison is not a claim of cryptographic-module certification.
#if defined(__CUDACC__)
#define STRATA_BRANCH_HD __host__ __device__
#else
#define STRATA_BRANCH_HD
#endif

namespace strata::core {
STRATA_BRANCH_HD inline uint32_t branch_rotr(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32 - shift));
}
STRATA_BRANCH_HD inline void branch_sha256(const uint8_t* input, uint64_t bytes, BranchDigest& digest) {
    constexpr uint32_t constants[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                         0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const uint64_t blocks = bytes / 64 + 1 + (bytes % 64 > 55);
    const uint64_t padded_bytes = blocks * 64, bits = bytes * 8;
    for (uint64_t block = 0; block < blocks; ++block) {
        uint32_t words[64]{};
        for (unsigned i = 0; i < 64; ++i) {
            const auto position = block * 64 + i;
            uint8_t value = 0;
            if (position < bytes) value = input[position];
            else if (position == bytes) value = 0x80;
            else if (position >= padded_bytes - 8)
                value = uint8_t(bits >> (8 * (padded_bytes - 1 - position)));
            words[i / 4] |= uint32_t(value) << (24 - 8 * (i % 4));
        }
        for (unsigned i = 16; i < 64; ++i) {
            const auto x = words[i - 15], y = words[i - 2];
            const auto s0 = branch_rotr(x, 7) ^ branch_rotr(x, 18) ^ (x >> 3);
            const auto s1 = branch_rotr(y, 17) ^ branch_rotr(y, 19) ^ (y >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
        uint32_t e=state[4], f=state[5], g=state[6], h=state[7];
        for (unsigned i = 0; i < 64; ++i) {
            const auto s1 = branch_rotr(e, 6) ^ branch_rotr(e, 11) ^ branch_rotr(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto t1 = h + s1 + choice + constants[i] + words[i];
            const auto s0 = branch_rotr(a, 2) ^ branch_rotr(a, 13) ^ branch_rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + majority;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    for (unsigned i = 0; i < 32; ++i) digest.bytes[i] = uint8_t(state[i / 4] >> (24 - 8 * (i % 4)));
}
} // namespace strata::core
#undef STRATA_BRANCH_HD
