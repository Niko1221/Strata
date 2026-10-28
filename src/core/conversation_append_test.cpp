// CPU-only tests for the session-file v3 primitives (include/strata/core/conversation_file.hpp):
// the serializable SessionHasher resume state and the fixed 128 B trailer record.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// A deterministic byte source for hashing.
std::vector<uint8_t> feed(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(seed * 131 + i * 17 + (i >> 5));
    return v;
}

void put64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }

// --- SessionHasher state -------------------------------------------------------

void hasher_state_tests() {
    // Every npending 0..31: state() captures lanes/pending/npending/total, zeroes the
    // pending area past npending, and a restored hasher continues to the same digest as
    // an uninterrupted one-shot hasher over prefix + appended bytes.
    for (int n = 0; n < 32; ++n) {
        const auto prefix = feed(size_t(n), uint8_t(9 + n));
        SessionHasher h(0x9E3779B97F4A7C15ull);
        h.update(prefix.data(), prefix.size());
        const SessionHasherState s = h.state();
        check(s.npending == uint8_t(n), "state npending matches");
        check(s.total == uint64_t(n), "state total matches");
        check(std::memcmp(s.pending, prefix.data(), prefix.size()) == 0, "state pending carries the tail bytes");
        bool tail_zeroed = true;
        for (int i = n; i < 32; ++i)
            if (s.pending[i] != 0) tail_zeroed = false;
        check(tail_zeroed, "state zeroes pending past npending");

        // Byte-exact comparison rule (S3 weakness 6): two hashers that saw the same bytes
        // agree over the WHOLE 32 B pending area, not just the [0, npending) prefix.
        SessionHasher h2(0x9E3779B97F4A7C15ull);
        h2.update(prefix.data(), prefix.size());
        const SessionHasherState s2 = h2.state();
        check(std::memcmp(&s, &s2, sizeof s) == 0, "state round-trip is byte-exact for equal inputs");

        // A restored hasher continues exactly: appended bytes give the one-shot digest.
        const auto more = feed(77, uint8_t(3));
        SessionHasher r(1);  // any seed: restore replaces all state
        r.restore(s);
        r.update(more.data(), more.size());
        SessionHasher one_shot(0x9E3779B97F4A7C15ull);
        one_shot.update(prefix.data(), prefix.size());
        one_shot.update(more.data(), more.size());
        check(r.digest() == one_shot.digest(), "restored hasher continues to the one-shot digest");
        check(r.state().total == uint64_t(n) + more.size(), "restored total continues");
    }

    // Multi-block prefix (exercises the lanes), then a mid-block tail.
    const auto big = feed(5000, 7), tail = feed(11, 8);
    SessionHasher h(0);
    h.update(big.data(), big.size());
    h.update(tail.data(), tail.size());
    SessionHasher r(0xABCDEF);
    r.restore(h.state());
    check(r.digest() == h.digest(), "state restore reproduces the digest for a multi-block stream");
}

// --- Trailer record -------------------------------------------------------------

SessionTrailer sample_trailer() {
    SessionTrailer t;
    t.gen = 3;
    t.committed_bytes = 1000000;
    t.saved_tokens = 12345;
    t.digest = 0x0123456789ABCDEFull;
    t.chain = 0xFEDCBA9876543210ull;
    for (int i = 0; i < 4; ++i) t.lanes[i] = 0x1000000000000000ull * (i + 1);
    for (int i = 0; i < 32; ++i) t.pending[i] = uint8_t(i * 8);
    t.npending = 21;
    t.hashed_bytes = 999936;
    return t;
}

void trailer_tests() {
    static_assert(sizeof(SessionTrailer) == 128, "SessionTrailer mirrors the 128 B record");
    const SessionTrailer t = sample_trailer();
    uint8_t bytes[128];
    trailer_bytes(t, bytes);
    SessionTrailer canon = t;
    std::memcpy(&canon.self_hash, bytes + 120, 8);  // the written record's self_hash

    // Fixed layout fields land at their offsets, little-endian.
    check(bytes[0] == 'S' && bytes[1] == 'T' && bytes[2] == 'R' && bytes[3] == 'T', "trailer magic bytes STRT");
    check(bytes[4] == 3, "trailer version 3");
    check(bytes[5] == 21, "trailer npending byte");
    check(bytes[6] == 0 && bytes[7] == 0, "trailer reserved is zero");
    uint64_t v;
    std::memcpy(&v, bytes + 8, 8);
    check(v == 3, "trailer gen offset");
    std::memcpy(&v, bytes + 16, 8);
    check(v == 1000000, "trailer committed_bytes offset");
    std::memcpy(&v, bytes + 24, 8);
    check(v == 12345, "trailer saved_tokens offset");
    std::memcpy(&v, bytes + 32, 8);
    check(v == 0x0123456789ABCDEFull, "trailer digest offset");
    std::memcpy(&v, bytes + 40, 8);
    check(v == 0xFEDCBA9876543210ull, "trailer chain offset");
    check(std::memcmp(bytes + 48, t.lanes, 32) == 0, "trailer lanes offset");
    check(std::memcmp(bytes + 80, t.pending, 32) == 0, "trailer pending offset");
    std::memcpy(&v, bytes + 112, 8);
    check(v == 999936, "trailer hashed_bytes offset");

    // self_hash covers every byte in use: [0, 120), seed 0.
    uint64_t self;
    std::memcpy(&self, bytes + 120, 8);
    check(self == session_hash64(bytes, 120, 0), "trailer self_hash covers bytes 0..119");

    SessionTrailer back;
    check(trailer_parse(bytes, back), "trailer parse accepts a written record");
    check(std::memcmp(&back, &canon, sizeof back) == 0, "trailer write/parse round-trip (self_hash recomputed consistently)");

    // Single-byte corruption at EVERY offset is rejected: bytes 0..119 by the self_hash,
    // 120..127 by the self_hash value itself (no uncovered pad).
    for (int off = 0; off < 128; ++off) {
        uint8_t bad[128];
        std::memcpy(bad, bytes, 128);
        bad[off] ^= 0xFF;
        SessionTrailer junk;
        if (trailer_parse(bad, junk)) {
            std::fprintf(stderr, "FAIL: trailer parse accepted a corruption at offset %d\n", off);
            std::exit(1);
        }
        ++checks;
    }

    // A reserved-slot nonzero (uncorrupted hash) is still refused: build a record whose
    // bytes[6] is 1, then fix bytes[6] back to 0 only in the self_hash input.
    {
        uint8_t bad[128];
        trailer_bytes(t, bad);  // canonical
        bad[6] = 1;             // reserved nonzero; self_hash now disagrees
        SessionTrailer junk;
        check(!trailer_parse(bad, junk), "trailer refuses a nonzero reserved byte");
    }

    // Genesis chain: hash64 over the 16 bytes model-LE then config-LE, seed 0; identity-dependent.
    const SessionFileIdentity a{0x1111222233334444ull, 0x5555666677778888ull}, b{0x1111222233334444ull, 1};
    uint8_t buf[16];
    put64(buf, a.model);
    put64(buf + 8, a.config);
    check(trailer_genesis_chain(a) == session_hash64(buf, 16, 0), "genesis chain = hash64(model||config)");
    check(trailer_genesis_chain(a) == trailer_genesis_chain(SessionFileIdentity{a.model, a.config}),
          "genesis chain is deterministic");
    check(trailer_genesis_chain(a) != trailer_genesis_chain(b), "genesis chain depends on the identity");
}
}  // namespace

int main() {
    hasher_state_tests();
    trailer_tests();
    std::printf("conversation_append_test: %d checks passed\n", checks);
    return 0;
}
