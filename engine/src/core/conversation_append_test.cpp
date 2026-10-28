// CPU-only tests for the session-file v3 primitives (include/strata/core/conversation_file.hpp):
// the serializable SessionHasher resume state and the fixed 128 B trailer record.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

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
// --- Header v3 ------------------------------------------------------------------

void header_v3_tests() {
    const SessionFileIdentity id{0xAABBCCDDEEFF0011ull, 0x0102030405060708ull};
    uint8_t h[64];
    header_bytes_v3(h, id, {4000000, 3999936, 4000128});  // payload, reserved0, reserved1
    check(std::memcmp(h, "STRSESS\x01", 8) == 0, "v3 header carries the session magic");
    uint32_t ver, size;
    std::memcpy(&ver, h + 8, 4);
    std::memcpy(&size, h + 12, 4);
    check(ver == 3, "v3 header version 3");
    check(size == 64, "v3 header size 64");
    uint64_t root;
    std::memcpy(&root, h + 32, 8);
    check(root == session_hash64(h, 32, 0), "v3 root hash covers the immutable bytes 0..31 only");

    SessionHeaderV3 body;
    std::string error;
    SessionHeaderV3 written{4000000, 3999936, 4000128};
    check(header_parse_v3(h, id, body, error), "v3 header parses");
    check(std::memcmp(&body, &written, sizeof body) == 0, "v3 advisory fields round-trip");
    check(header_parse_v3(h, SessionFileIdentity{id.model, id.config}, body, error), "v3 header identity match");

    // Advisory fields: tampering payload / reserved0 / reserved1 keeps the header TRUSTED
    // (a torn in-place header update must not distrust the file - the trailer is authoritative).
    for (int off = 40; off < 64; ++off) {
        uint8_t bad[64];
        std::memcpy(bad, h, 64);
        bad[off] ^= 0xFF;
        SessionHeaderV3 b2;
        std::string e2;
        if (!header_parse_v3(bad, id, b2, e2)) {
            std::fprintf(stderr, "FAIL: v3 header refused a tamper of the advisory byte %d\n", off);
            std::exit(1);
        }
        ++checks;
    }

    // Immutable region: a tamper of magic, version, header_size, model, config or the root
    // hash itself breaks trust.
    for (int off = 0; off < 32; ++off) {
        uint8_t bad[64];
        std::memcpy(bad, h, 64);
        bad[off] ^= 0xFF;
        SessionHeaderV3 b2;
        std::string e2;
        check(!header_parse_v3(bad, id, b2, e2), "v3 header refuses a tamper of the immutable region");
    }
    {
        uint8_t bad[64];
        std::memcpy(bad, h, 64);
        bad[35] ^= 0xFF;  // root hash field
        SessionHeaderV3 b2;
        std::string e2;
        check(!header_parse_v3(bad, id, b2, e2), "v3 header refuses a corrupted root hash");
    }
    {
        SessionHeaderV3 b2;
        std::string e2;
        check(!header_parse_v3(h, SessionFileIdentity{id.model, id.config + 1}, b2, e2),
              "v3 header refuses a foreign identity");
    }
    // A v2 header (version 2) is not a v3 header.
    {
        uint8_t v2[64];
        std::memset(v2, 0, 64);
        std::memcpy(v2, "STRSESS\x01", 8);
        uint32_t ver2 = 2, sz2 = 64;
        std::memcpy(v2 + 8, &ver2, 4);
        std::memcpy(v2 + 12, &sz2, 4);
        SessionHeaderV3 b2;
        std::string e2;
        check(!header_parse_v3(v2, id, b2, e2), "v3 header parse refuses a v2 header");
    }
}
// --- Last-checkpoint validator ---------------------------------------------------

namespace {
fs::path append_dir;
std::vector<uint8_t> current_bytes;  // staging for write_file

// Build a synthetic v3 file: header + [group region + 128 B trailer] x N.  The digest
// streams over the group regions in order (physical zone order); committed_bytes of group g
// is the file offset of its trailer record.  Regions are deterministic non-zero bytes.
struct SynthFile {
    std::vector<uint8_t> bytes;
    uint64_t committed[8] = {};  // per generation (1-based index)
};

SynthFile build_v3(const SessionFileIdentity& id, const std::vector<int>& region_sizes,
                   const SessionHeaderV3* header_override = nullptr, uint8_t version = 3) {
    SynthFile f;
    SessionHasher h(0);
    uint64_t chain = trailer_genesis_chain(id);
    auto put = [&](const void* p, size_t n) { f.bytes.insert(f.bytes.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    uint8_t head[64];
    std::memset(head, 0, 64);
    std::memcpy(head, "STRSESS\x01", 8);
    uint32_t ver = version, sz = 64;
    std::memcpy(head + 8, &ver, 4);
    std::memcpy(head + 12, &sz, 4);
    std::memcpy(head + 16, &id.model, 8);
    std::memcpy(head + 24, &id.config, 8);
    uint64_t root = session_hash64(head, 32, 0);
    std::memcpy(head + 32, &root, 8);
    put(head, 64);
    for (size_t g = 0; g < region_sizes.size(); ++g) {
        const int n = region_sizes[g];
        std::vector<uint8_t> region((size_t) n);
        for (int i = 0; i < n; ++i) region[(size_t) i] = uint8_t(0x41 + (g * 7 + i * 13) % 255);  // never zero: a torn record's data zone
        put(region.data(), region.size());
        h.update(region.data(), region.size());
        const SessionHasherState st = h.state();
        SessionTrailer t;
        t.gen = uint64_t(g) + 1;
        t.committed_bytes = f.bytes.size();  // the trailer's own offset
        t.saved_tokens = 1000 + 500 * uint64_t(g + 1);
        t.digest = h.digest();
        t.chain = chain;
        for (int i = 0; i < 4; ++i) t.lanes[i] = st.lanes[i];
        std::memcpy(t.pending, st.pending, 32);
        t.npending = st.npending;
        t.hashed_bytes = st.total;
        uint8_t tb[128];
        trailer_bytes(t, tb);
        f.committed[g] = t.committed_bytes;
        uint64_t self;
        std::memcpy(&self, tb + 120, 8);
        chain = self;
        put(tb, 128);
    }
    // header advisory slots (fresh header: newest trailer + next group start)
    SessionHeaderV3 body;
    if (header_override) {
        body = *header_override;
    } else if (!region_sizes.empty()) {
        body.payload = f.committed[region_sizes.size() - 1];
        body.reserved0 = f.committed[region_sizes.size() - 1];
        body.reserved1 = f.committed[region_sizes.size() - 1] + 128;
    }
    std::memcpy(f.bytes.data() + 40, &body.payload, 8);
    std::memcpy(f.bytes.data() + 48, &body.reserved0, 8);
    std::memcpy(f.bytes.data() + 56, &body.reserved1, 8);
    return f;
}

void write_file(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write((const char*) bytes.data(), (std::streamsize) bytes.size());
    o.flush();
}

fs::path uniq(const char* name) {
    static int seq = 0;
    return append_dir / (std::string(name) + "-" + std::to_string(seq++) + ".bin");
}

// Write the staged file and read the checkpoint list back.
bool read_checks(const fs::path& p, const SessionFileIdentity& id, std::vector<SessionCheckpoint>& out,
                 std::string& error) {
    write_file(p, current_bytes);
    return session_file_checkpoints(p.string(), id, out, error);
}

// A verifier-side recompute of the zone-order digest of generation g of a clean build.
uint64_t synth_digest(const SessionFileIdentity& id, const std::vector<int>& region_sizes, int upto) {
    SessionHasher h(0);
    for (int g = 0; g < upto; ++g) {
        std::vector<uint8_t> region((size_t) region_sizes[(size_t) g]);
        for (size_t i = 0; i < region.size(); ++i) region[i] = uint8_t(0x41 + (g * 7 + int(i) * 13) % 255);
        h.update(region.data(), region.size());
    }
    return h.digest();
}
}  // namespace

void validator_tests() {
    const SessionFileIdentity id{0x1234567890ABCDEFull, 0x0F0F0F0F0F0F0F0Full};
    std::string error;
    std::vector<SessionCheckpoint> out;

    // Clean two-generation file: newest first, fresh header fast path.
    SynthFile f = build_v3(id, {900, 700});
    current_bytes = f.bytes;
    fs::path p = uniq("clean");
    bool clean_ok = read_checks(p, id, out, error);
    if (!clean_ok) std::fprintf(stderr, "validator error: [%s]\n", error.c_str());
    check(clean_ok, "validator accepts a clean v3 file");
    check(out.front().trailer.gen == 2, "validator lands on the newest generation");
    check(out.front().trailer_offset == f.committed[1], "checkpoint trailer_offset is the trailer record");
    check(out.front().trailer.saved_tokens == 2000, "checkpoint carries saved_tokens");
    check(out.front().trailer.digest == synth_digest(id, {900, 700}, 2), "checkpoint digest is the zone-order digest");
    check(out.front().file_size == f.bytes.size(), "checkpoint carries the file size");

    // Foreign identity is refused outright.
    check(!read_checks(uniq("foreign"), SessionFileIdentity{id.model, id.config + 1}, out, error),
          "validator refuses a foreign identity");

    // Torn-tail proof: truncation at EVERY offset inside the newest group and its trailer
    // lands on gen 1 while gen 1's own trailer stands, or refuses; never a half state.
    for (uint64_t cut = f.committed[0]; cut < f.committed[1] + 128; ++cut) {  // cuts below the complete gen 2 trailer
        current_bytes.assign(f.bytes.begin(), f.bytes.begin() + (size_t) cut);
        bool ok = read_checks(uniq("torn"), id, out, error);
        bool gen1_trailer_present = cut >= f.committed[0] + 128;
        if (ok != gen1_trailer_present) {
            std::fprintf(stderr, "FAIL: torn cut %llu: ok=%d expected=%d\n", (unsigned long long) cut, int(ok),
                         int(gen1_trailer_present));
            std::exit(1);
        }
        if (ok && out.front().trailer.gen != 1) {
            std::fprintf(stderr, "FAIL: torn cut %llu trusted gen %llu\n", (unsigned long long) cut,
                         (unsigned long long) out.front().trailer.gen);
            std::exit(1);
        }
        ++checks;
    }

    // Corruption of one byte of the newest trailer: the scan lands on gen 1, not garbage.
    for (int off = 0; off < 128; ++off) {
        current_bytes = f.bytes;
        current_bytes[(size_t) f.committed[1] + (size_t) off] ^= 0xFF;
        check(read_checks(uniq("corrupt"), id, out, error) && out.front().trailer.gen == 1,
              "corrupted newest trailer falls back to gen 1");
    }

    // Corruption of the OLDER trailer (gen 1): the NEWEST checkpoint stands - its own record
    // is intact and the header pair corroborates it; the damaged older record costs nothing
    // on the save side (restore arbitrates older bytes).
    for (int off = 0; off < 128; ++off) {
        current_bytes = f.bytes;
        current_bytes[(size_t) f.committed[0] + (size_t) off] ^= 0xFF;
        check(read_checks(uniq("middle"), id, out, error) && out.front().trailer.gen == 2,
              "a damaged older trailer still yields the intact newest checkpoint");
    }

    // Inflated claim (drill F2): a trailer record at EOF whose committed_bytes exceeds its
    // position passes magic+chain arithmetic but fails the structural rule - gen 1 stands.
    {
        current_bytes = f.bytes;
        SessionTrailer t;  // re-sign a claim at a wrong position
        uint8_t tb[128];
        std::memcpy(tb, f.bytes.data() + f.committed[1], 128);
        check(trailer_parse(tb, t), "clean newest trailer parses");
        t.committed_bytes = f.bytes.size() + 4096;  // inflated claim, still chain-linked
        t.gen = 3;
        trailer_bytes(t, tb);
        current_bytes.insert(current_bytes.end(), tb, tb + 128);  // planted after the real EOF
        check(read_checks(uniq("f2"), id, out, error) && out.front().trailer.gen == 2,
              "inflated committed_bytes claim is structurally rejected");
    }

    // Verbatim replay of the newest trailer after EOF: same record at the wrong position is
    // rejected by the structural rule (committed_bytes must equal the record's own offset).
    {
        current_bytes = f.bytes;
        current_bytes.insert(current_bytes.end(), f.bytes.begin() + (long) f.committed[1],
                             f.bytes.begin() + (long) f.committed[1] + 128);
        check(read_checks(uniq("replay"), id, out, error) && out.front().trailer.gen == 2 &&
                  out.front().trailer_offset == f.committed[1],
              "a replayed trailer record at a new position is not trusted");
    }

    // Stale header fast path (drill scenario I): committed trailer fsynced, header never
    // updated - the newest VALID trailer still wins over the stale slot.
    {
        SynthFile stale = f;  // advisory slots are outside the root hash: patch freely
        const uint64_t body[3] = {f.committed[0], f.committed[0], f.committed[0] + 128};
        std::memcpy(stale.bytes.data() + 40, &body[0], 8);
        std::memcpy(stale.bytes.data() + 48, &body[1], 8);
        std::memcpy(stale.bytes.data() + 56, &body[2], 8);
        current_bytes = stale.bytes;
        check(read_checks(uniq("stale"), id, out, error) && out.front().trailer.gen == 2,
              "a stale header slot still resolves to the newest valid trailer");
    }

    // Garbage header slots: scan-back recovers both generations anyway.
    {
        SessionHeaderV3 junk{0xDEADBEEFDEADBEEFull, 7, 13};
        SynthFile g = build_v3(id, {900, 700}, &junk);
        current_bytes = g.bytes;
        check(read_checks(uniq("junkhdr"), id, out, error) && out.front().trailer.gen == 2,
              "garbage advisory slots still recover the newest trailer by scan");
    }

    // No trailers at all (gen 0 file): a group-less v3 file validates zero checkpoints ->
    // refused (full-rewrite self-heal verdict).
    {
        SynthFile bare = build_v3(id, {});
        current_bytes = bare.bytes;
        check(!read_checks(uniq("bare"), id, out, error), "a generation-less v3 file is refused");
    }

    // Not a session file / v2 header: refused.
    {
        SynthFile v2ish = build_v3(id, {900}, nullptr, 2);
        current_bytes = v2ish.bytes;
        check(!read_checks(uniq("v2"), id, out, error), "the v3 validator refuses a v2 header");
    }
    {
        current_bytes.assign(1024, 0x5A);
        check(!read_checks(uniq("garbage"), id, out, error), "random bytes are refused");
    }
}
}  // namespace

int main() {
    append_dir = fs::temp_directory_path() / "strata_append_test";
    fs::remove_all(append_dir);
    fs::create_directories(append_dir);
    hasher_state_tests();
    trailer_tests();
    header_v3_tests();
    validator_tests();
    fs::remove_all(append_dir);
    std::printf("conversation_append_test: %d checks passed\n", checks);
    return 0;
}
