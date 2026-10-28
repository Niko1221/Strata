// CPU-only tests for the session-file v3 primitives (include/strata/core/conversation_file.hpp):
// the serializable SessionHasher resume state and the fixed 128 B trailer record.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"

#include <array>
#include <cerrno>
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
// --- Full v3 write (generation 1) -------------------------------------------------

ConversationCheckpoint mk_ckpt(int ids_n, uint8_t seed) {
    ConversationCheckpoint c;
    for (int i = 0; i < ids_n; ++i) c.ids.push_back(int32_t(seed * 1000 + i));
    c.imgs = {{1, 0xA1}, {5, 0xB2}};
    c.gdn.assign(64, seed);
    c.ple.assign(32, uint8_t(seed ^ 1));
    c.tails.assign(16, uint8_t(seed ^ 2));
    c.dead.assign(24, uint8_t(seed ^ 3));
    c.block_pos.assign(8, uint8_t(seed ^ 4));
    c.used = seed;
    return c;
}

ConversationBuffer pattern_buf(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t(seed + (at + i) * 31);
        return true;
    });
    return b;
}

std::vector<uint8_t> pattern_vec(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(seed + i * 31);
    return v;
}

ConversationKv mk_layer(int format, int64_t cells, int64_t heads, int64_t head_dim, int64_t page, int64_t pooled_rows,
                        int64_t idx_dim, uint8_t seed) {
    ConversationKv k;
    k.format = format;
    k.cells = cells; k.heads = heads; k.head_dim = head_dim; k.page_size = page;
    k.pooled_rows = pooled_rows; k.idx_dim = idx_dim;
    const size_t cell_bytes = size_t(heads) * size_t(head_dim);  // int8-style fixture geometry
    const size_t scale_bytes = size_t(cells) * size_t(heads) * size_t(head_dim / 64) * 2;
    k.k = pattern_buf(size_t(cells) * cell_bytes, seed);
    k.v = pattern_buf(size_t(cells) * cell_bytes, uint8_t(seed + 7));
    k.k_scale = pattern_buf(scale_bytes, uint8_t(seed + 13));
    k.v_scale = pattern_buf(scale_bytes, uint8_t(seed + 19));
    if (pooled_rows) k.pooled = pattern_buf(size_t(pooled_rows) * size_t(idx_dim) * 4, uint8_t(seed + 23));
    return k;
}

SavedConversation make_conv(uint8_t seed, int64_t big_cells = 0) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(i) * 3 + 1;
    s.layer_lo = 0; s.layer_hi = 48; s.cvec = seed & 1;
    s.live = mk_ckpt(120, seed);
    s.checkpoints = {mk_ckpt(90, uint8_t(seed + 40))};  // the deepest one
    const int64_t cells = big_cells ? big_cells : 1000;
    s.kv.push_back(mk_layer(1, cells, 2, 256, 64, 9, 128, seed));
    s.kv.push_back(mk_layer(1, cells + 24, 2, 128, 64, 10, 128, uint8_t(seed + 3)));
    s.kv.push_back(mk_layer(2, cells, 2, 256, 64, 0, 0, uint8_t(seed + 6)));  // draft: no pooled part
    return s;
}

std::vector<SessionKvSource> make_sources(const SavedConversation& s) {
    std::vector<SessionKvSource> sources;
    for (const auto& k : s.kv) {
        SessionKvSource src;
        src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
        src.page_size = k.page_size; src.pooled_rows = k.pooled_rows; src.idx_dim = k.idx_dim;
        const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t i = 0; i < 5; ++i) src.sizes[i] = parts[i]->size();
        src.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
        sources.push_back(std::move(src));
    }
    return sources;
}

std::vector<uint8_t> slurp_bin(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void v3_write_tests() {
    const SessionFileIdentity id{0x1234567890ABCDEFull, 0x0F0F0F0F0F0F0F0Full};
    std::string error;
    SessionWriteOptions opt;
    opt.use_v3 = true;

    // Three fixture sizes: the group+trailer block is always padded so the file is 64 mod 512.
    std::vector<uint8_t> first_file;
    for (uint8_t seed : {1, 2, 3}) {
        const SavedConversation conv = make_conv(seed);
        const fs::path p = uniq("full");
        size_t bytes = 0;
        check(session_file_write(p.string(), conv, id, bytes, error, opt), "full v3 write (captured image)");
        const auto f = slurp_bin(p);
        check(bytes == f.size(), "v3 write reports the file size");
        check(f[8] == 3 && f[9] == 0, "v3 file carries version 3");
        check(f.size() % 512 == 64, "v3 group+trailer is padded to 512");

        std::vector<SessionCheckpoint> cps;
        check(session_file_checkpoints(p.string(), id, cps, error), "full v3 file validates");
        check(cps.front().trailer.gen == 1, "full write is generation 1");
        check(cps.front().trailer.saved_tokens == uint64_t(conv.live.ids.size()), "trailer saved_tokens = live ids");
        const uint64_t committed = cps.front().trailer_offset;
        // group + pad + trailer is a multiple of 512, so the group region (with its pad)
        // always ends exactly 128 B short of a 512 multiple:
        check((committed - 64) % 512 == 384, "group+trailer blocks are padded to 512");
        // digest = contiguous zone hash [64, committed), and the stored resume state continues it
        SessionHasher h(0);
        h.update(f.data() + 64, (size_t) committed - 64);
        check(h.digest() == cps.front().trailer.digest, "trailer digest is the zone-order digest");
        const SessionTrailer t = cps.front().trailer;
        SessionHasherState st{};
        for (int i = 0; i < 4; ++i) st.lanes[i] = t.lanes[i];
        std::memcpy(st.pending, t.pending, 32);
        st.npending = t.npending;
        st.total = t.hashed_bytes;
        SessionHasher r2(0);
        r2.restore(st);
        check(r2.digest() == t.digest, "stored resume state reproduces the digest");
        check(r2.state().total == committed - 64, "hashed_bytes counts the whole contiguous zone (group + pad)");
        check(cps.front().trailer.chain == trailer_genesis_chain(id), "generation 1 chains to genesis");

        // header advisory slots
        SessionHeaderV3 hv;
        std::string e2;
        check(header_parse_v3(f.data(), id, hv, e2), "v3 header parses from the file");
        check(hv.payload == committed && hv.reserved0 == committed && hv.reserved1 == committed + 128,
              "header slots point at the trailer and the next group start");

        // streamed sources produce the exact same bytes as the captured image
        SavedConversation meta = conv;
        meta.kv.clear();
        const auto sources = make_sources(conv);
        const fs::path q = uniq("streamed");
        check(session_file_write(q.string(), meta, sources, id, bytes, error, opt), "full v3 write (streamed sources)");
        check(slurp_bin(q) == f, "v3 streamed write equals the captured-image write");

        if (seed == 1) first_file = f;  // keep one fixture for the truncation checks
    }

    // Same fixture with the flag off: exact v2 output, which the v3 validator refuses.
    {
        const SavedConversation conv = make_conv(1);
        const fs::path p = uniq("v2mode");
        size_t bytes = 0;
        check(session_file_write(p.string(), conv, id, bytes, error), "flag-off write stays v2");
        const auto f = slurp_bin(p);
        check(f[8] == 2, "flag-off file carries version 2");
        std::vector<SessionCheckpoint> cps;
        check(!session_file_checkpoints(p.string(), id, cps, error), "the v3 validator refuses the v2 file");
    }

    // Truncating the real v3 file anywhere below its complete trailer refuses it (gen 1 is
    // the only checkpoint; the full torn matrix is pinned on synthetic files in Task 3).
    const uint64_t committed = first_file.size() - 128;
    for (uint64_t cut : {committed - 1, committed + 1, committed + 63, committed + 127, first_file.size() - 1}) {
        const fs::path p = uniq("trunc");
        write_file(p, std::vector<uint8_t>(first_file.begin(), first_file.begin() + (size_t) cut));
        std::vector<SessionCheckpoint> cps;
        check(!session_file_checkpoints(p.string(), id, cps, error), "truncated v3 file is refused");
    }

    // Fault seams on the v3 path: a failed flush or block write leaves no published file and
    // no temp (the existing R3/R5 semantics, now over the v3 byte stream).
    {
        const SavedConversation conv = make_conv(4);
        const fs::path p = uniq("fault-flush");
        size_t bytes = 0;
        SessionWriteOptions o = opt;
        o.fault = [](const char* step) { return std::strcmp(step, "file_flush") == 0 ? EIO : 0; };
        check(!session_file_write(p.string(), conv, id, bytes, error, o), "v3 flush fault fails the save");
        check(!fs::exists(p), "v3 flush fault publishes nothing");

        // a > 16 MiB fixture reaches the block writer, where the "write" seam fires
        const SavedConversation big = make_conv(5, 32768);
        const fs::path q = uniq("fault-write");
        SessionWriteOptions o2 = opt;
        o2.fault = [](const char* step) { return std::strcmp(step, "write") == 0 ? ENOSPC : 0; };
        check(!session_file_write(q.string(), big, id, bytes, error, o2), "v3 block-write fault fails the save");
        check(!fs::exists(q), "v3 block-write fault publishes nothing");
        int temps = 0;
        for (auto& e : fs::directory_iterator(append_dir))
            if (e.path().extension() == ".tmp" || e.path().string().find(".bin.") != std::string::npos) ++temps;
        check(temps == 0, "failed v3 saves leave no temp files");
    }
}
// --- v3 reader (region map, digest authority, cascade) ---------------------------

ConversationKv mk_layer_idx(int format, int64_t cells, int64_t heads, int64_t head_dim, int64_t page,
                            int64_t rows, int64_t idx_dim, uint8_t seed, int64_t idx_block) {
    (void) idx_block;  // fixture rows are explicit; the FILE carries idx_block through the source
    return mk_layer(format, cells, heads, head_dim, page, rows, idx_dim, seed);
}

// committed byte boundary of every appendable structure of a generation (test-side mirror of
// the engine rule, incl. the draft -1 and the pooled whole-row rule):
struct Bounds {
    uint64_t ids = 0, imgs = 0;
    std::vector<std::array<uint64_t, 5>> layer;
};
Bounds gen_bounds(const SavedConversation& s, const std::vector<int64_t>& idx_blocks) {
    Bounds b;
    const int64_t S = (int64_t) s.live.ids.size();
    b.ids = (uint64_t) S * 4;
    b.imgs = (uint64_t) s.live.imgs.size() * 16;
    for (size_t j = 0; j < s.kv.size(); ++j) {
        const ConversationKv& k = s.kv[j];
        const bool draft = j + 1 == s.kv.size() && k.pooled.empty() && k.pooled_rows == 0 && k.idx_dim == 0;
        const int64_t t = draft && S > 0 ? S - 1 : S;
        const int64_t W = k.page_size > 0 ? (t / k.page_size) * k.page_size : t;
        std::array<uint64_t, 5> v;
        const uint64_t bpc = (uint64_t) k.heads * (uint64_t) k.head_dim;               // k, v bytes per cell
        const uint64_t sbpc = (uint64_t) k.heads * (uint64_t) (k.head_dim / 64) * 2;   // scale bytes per cell
        v[0] = v[1] = bpc * (uint64_t) W;
        v[2] = v[3] = sbpc * (uint64_t) W;
        if (k.pooled_rows > 0 && idx_blocks[j] > 0) {
            const uint64_t rb = k.pooled.size() / (uint64_t) k.pooled_rows;
            v[4] = (uint64_t) (t / idx_blocks[j]) * rb;  // whole completed pooled rows below the boundary
        } else {
            v[4] = k.pooled.size();  // idx_block unknown: the pooled prefix is not append-trusted (ruling)
        }
        b.layer.push_back(v);
    }
    return b;
}

// conv2 from conv1: every appendable structure keeps its committed prefix bytes identical and
// grows with new values; the mutable re-emitted fields are freshly seeded (they may differ).
SavedConversation chain_next(const SavedConversation& c1, const Bounds& b1, int64_t cells2, int ids2,
                             int64_t rows2, uint8_t seed2, const std::vector<int64_t>& idx_blocks) {
    SavedConversation s = c1;
    s.live.ids.resize((size_t) ids2);
    for (int i = (int) c1.live.ids.size(); i < ids2; ++i)
        s.live.ids[(size_t) i] = int32_t(seed2 * 100000 + i);
    s.live.imgs.push_back({int64_t(ids2) + 3, 0xC3C3});
    s.live.gdn = pattern_vec(70, seed2);  // growing arrays too
    s.live.tails = pattern_vec(16, uint8_t(seed2 ^ 9));
    s.checkpoints = {mk_ckpt(ids2 - 10, uint8_t(seed2 + 77))};
    s.kv.clear();
    for (size_t j = 0; j < c1.kv.size(); ++j) {
        const ConversationKv& k = c1.kv[j];
        ConversationKv n;
        n.format = k.format; n.cells = cells2; n.heads = k.heads; n.head_dim = k.head_dim;
        n.page_size = k.page_size; n.pooled_rows = k.pooled_rows > 0 ? rows2 : 0; n.idx_dim = k.idx_dim;
        const size_t cell_bytes = size_t(k.heads) * size_t(k.head_dim);
        auto extend = [&](const ConversationBuffer& src, uint64_t bound, size_t new_size, uint8_t sd) {
            ConversationBuffer out = pattern_buf(new_size, sd);  // new values everywhere...
            // ...then the committed prefix must equal the previous generation's bytes exactly:
            out.visit(0, bound, [&](uint8_t* p, size_t c, size_t o) {
                src.read(p, o, c);  // same logical position
                return true;
            });
            return out;
        };
        n.k = extend(k.k, b1.layer[j][0], size_t(cells2) * cell_bytes, uint8_t(seed2 + 1));
        n.v = extend(k.v, b1.layer[j][1], size_t(cells2) * cell_bytes, uint8_t(seed2 + 8));
        const size_t scale2 = size_t(cells2) * size_t(k.heads) * size_t(k.head_dim / 64) * 2;
        n.k_scale = extend(k.k_scale, b1.layer[j][2], scale2, uint8_t(seed2 + 14));
        n.v_scale = extend(k.v_scale, b1.layer[j][3], scale2, uint8_t(seed2 + 20));
        if (rows2 > 0) {
            const size_t rb = size_t(k.idx_dim) * 4;
            n.pooled = extend(k.pooled, b1.layer[j][4], size_t(rows2) * rb, uint8_t(seed2 + 24));
        }
        s.kv.push_back(std::move(n));
    }
    return s;
}

// Hand-built generation-2 group appended to a real full-v3 file: mirrors put_group_v3
// (no preamble) with only the appended suffixes, continuing the digest from gen 1's trailer.
// b1 = generation 1's committed boundaries; ids_saved/imgs_saved = generation 1's record counts;
// idx_block = the per-layer idx_block the file's appendable geometry carries.
void append_group2(std::vector<uint8_t>& file, const SavedConversation& c2, const SessionCheckpoint& prev,
                   const Bounds& b1, size_t ids_saved, size_t imgs_saved, int64_t idx_block,
                   uint64_t& out_committed) {
    SessionHasher h(0);
    SessionHasherState st{};
    for (int i = 0; i < 4; ++i) st.lanes[i] = prev.trailer.lanes[i];
    std::memcpy(st.pending, prev.trailer.pending, 32);
    st.npending = prev.trailer.npending;
    st.total = prev.trailer.hashed_bytes;
    h.restore(st);
    auto put = [&](const void* p, size_t n) {
        file.insert(file.end(), (const uint8_t*) p, (const uint8_t*) p + n);
        h.update(p, n);
    };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    auto i64 = [&](int64_t v) { put(&v, 8); };
    auto vec = [&](const std::vector<uint8_t>& v) { u64(v.size()); put(v.data(), v.size()); };
    auto ckpt = [&](const ConversationCheckpoint& c) {
        u64(c.ids.size());
        put(c.ids.data(), c.ids.size() * 4);
        u64(c.imgs.size());
        for (const auto& k : c.imgs) { i64(k.start); u64(k.hash); }
        vec(c.gdn); vec(c.ple); vec(c.tails); vec(c.dead); vec(c.block_pos);
        u64(c.used);
        u64(0);  // no stage parts
    };
    // mutable block (no preamble): live counts, arrays, used, parts_n; full checkpoint records; kv headers+sizes
    u64(c2.live.ids.size());
    u64(c2.live.imgs.size());
    vec(c2.live.gdn); vec(c2.live.ple); vec(c2.live.tails); vec(c2.live.dead); vec(c2.live.block_pos);
    u64(c2.live.used);
    u64(0);
    u64(c2.checkpoints.size());
    for (const auto& c : c2.checkpoints) ckpt(c);
    for (size_t j = 0; j < c2.kv.size(); ++j) {
        const ConversationKv& k = c2.kv[j];
        for (int64_t x : {int64_t(k.format), k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim,
                          k.pooled_rows > 0 ? idx_block : 0})
            i64(x);
        const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t p = 0; p < 5; ++p) {
            u64(b1.layer[j][p]);            // append window [from, to): from = generation 1's committed boundary
            u64(parts[p]->size());
        }
    }
    // segment block: only the appended suffixes at the rounded boundaries of generation 1:
    const size_t new_ids = c2.live.ids.size() - ids_saved;
    put(c2.live.ids.data() + ids_saved, new_ids * 4);
    const size_t new_imgs = c2.live.imgs.size() - imgs_saved;
    put(c2.live.imgs.data() + imgs_saved, new_imgs * 16);
    for (size_t j = 0; j < c2.kv.size(); ++j) {
        const ConversationKv& k = c2.kv[j];
        const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t from = b1.layer[j][p];
            const uint64_t size = parts[p]->size();
            if (size <= from) continue;
            std::vector<uint8_t> buf(size_t(size - from));
            parts[p]->read(buf.data(), (size_t) from, buf.size());
            put(buf.data(), buf.size());
        }
    }
    // pad + trailer
    const uint64_t frame = file.size() - (prev.trailer_offset + 128);  // this group's bytes so far
    const uint64_t pad = (512 - (frame + 128) % 512) % 512;
    const uint8_t zeros[512] = {};
    put(zeros, pad);
    SessionTrailer t;
    t.gen = prev.trailer.gen + 1;
    t.committed_bytes = file.size();
    t.saved_tokens = c2.live.ids.size();
    t.digest = h.digest();
    t.chain = prev.trailer.self_hash;
    const SessionHasherState hs = h.state();
    for (int i = 0; i < 4; ++i) t.lanes[i] = hs.lanes[i];
    std::memcpy(t.pending, hs.pending, 32);
    t.npending = hs.npending;
    t.hashed_bytes = hs.total;
    uint8_t tb[128];
    trailer_bytes(t, tb);
    file.insert(file.end(), tb, tb + 128);
    // fresh header advisory slots
    std::memcpy(file.data() + 40, &t.committed_bytes, 8);
    std::memcpy(file.data() + 48, &t.committed_bytes, 8);
    const uint64_t next = t.committed_bytes + 128;
    std::memcpy(file.data() + 56, &next, 8);
    out_committed = t.committed_bytes;
}

bool same_buf(const ConversationBuffer& a, const ConversationBuffer& b) {
    if (a.size() != b.size()) return false;
    bool eq = true;
    std::vector<uint8_t> tmp;
    a.visit(0, a.size(), [&](const uint8_t* p, size_t c, size_t o) {
        tmp.resize(c);
        b.read(tmp.data(), o, c);
        if (std::memcmp(p, tmp.data(), c) != 0) eq = false;
        return eq;
    });
    return eq;
}
bool same_ckpt(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used &&
           a.stage_parts.size() == b.stage_parts.size();
}
bool same_conv(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_hi != b.layer_hi || a.layer_lo != b.layer_lo || a.cvec != b.cvec)
        return false;
    if (a.stage_table.size() != b.stage_table.size()) return false;
    if (!same_ckpt(a.live, b.live)) return false;
    if (a.checkpoints.size() != b.checkpoints.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_ckpt(a.checkpoints[i], b.checkpoints[i])) return false;
    if (a.kv.size() != b.kv.size()) return false;
    for (size_t j = 0; j < a.kv.size(); ++j) {
        const ConversationKv& x = a.kv[j];
        const ConversationKv& y = b.kv[j];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim)
            return false;
        if (!same_buf(x.k, y.k) || !same_buf(x.v, y.v) || !same_buf(x.k_scale, y.k_scale) ||
            !same_buf(x.v_scale, y.v_scale) || !same_buf(x.pooled, y.pooled))
            return false;
    }
    return true;
}

// One append chain: a real full-v3 write of conv1 + a hand-built gen-2 group. Read back and
// compare against conv2 (pins the draft -1 rule, the pooled row rule, ids/imgs streams, the
// mutable newest-copy rule and the digest authority).  With below_boundary, the hand-built
// group starts its window BELOW generation 1's committed boundary: the region-map builder
// must refuse that candidate before reading its segments (cascade lands on generation 1).
void chain_read_test(uint8_t seed, int64_t idx_block, int64_t page, int64_t heads_dim_pairs_dim,
                     bool below_boundary = false) {
    const SessionFileIdentity id{0x1234567890ABCDEFull, 0x0F0F0F0F0F0F0F0Full};
    std::string error;
    const int64_t cells1 = 1024, ids1 = 1024;
    SavedConversation c1;
    for (size_t i = 0; i < c1.geometry.size(); ++i) c1.geometry[i] = int64_t(i) * 3 + 1;
    c1.layer_lo = 0; c1.layer_hi = 48; c1.cvec = true;
    c1.live = mk_ckpt((int) ids1, seed);
    c1.checkpoints = {mk_ckpt((int) ids1 - 4, uint8_t(seed + 40))};
    const int64_t rows1 = ids1 / idx_block + 1;
    c1.kv.push_back(mk_layer_idx(1, cells1, 2, heads_dim_pairs_dim, page, rows1, 128, seed, idx_block));
    c1.kv.push_back(mk_layer_idx(1, cells1 + 64, 2, heads_dim_pairs_dim / 2, page, rows1, 128, uint8_t(seed + 3), idx_block));
    c1.kv.push_back(mk_layer_idx(2, cells1, 2, heads_dim_pairs_dim, page, 0, 0, uint8_t(seed + 6), idx_block));  // draft

    SessionWriteOptions opt;
    opt.use_v3 = true;
    SavedConversation meta = c1;
    meta.kv.clear();
    std::vector<SessionKvSource> sources = make_sources(c1);
    for (size_t j = 0; j < sources.size(); ++j)
        sources[j].idx_block = c1.kv[j].pooled_rows > 0 ? idx_block : 0;  // the draft layer has no indexer
    const fs::path p = uniq("chain");
    size_t bytes = 0;
    check(session_file_write(p.string(), meta, sources, id, bytes, error, opt), "chain: gen 1 written");

    std::vector<SessionCheckpoint> cps;
    check(session_file_checkpoints(p.string(), id, cps, error) && cps.front().trailer.gen == 1, "chain: gen 1 validates");
    const SessionCheckpoint gen1 = cps.front();
    const Bounds b1 = gen_bounds(c1, std::vector<int64_t>{idx_block, idx_block, idx_block});

    const int64_t ids2 = 1100, cells2 = 1100;
    const int64_t rows2 = ids2 / idx_block + 1;
    SavedConversation c2 = chain_next(c1, b1, cells2, (int) ids2, rows2, uint8_t(seed + 11),
                                      std::vector<int64_t>{idx_block, idx_block, idx_block});

    auto file = slurp_bin(p);
    uint64_t committed2 = 0;  // generation 2's committed trailer offset, reported by the builder
    Bounds bw = b1;
    if (below_boundary) {  // one cell below the committed boundary: an append must never start there
        const uint64_t bpc0 = c1.kv[0].k.size() / (uint64_t) c1.kv[0].cells;
        bw.layer[0][0] -= bpc0;
    }
    append_group2(file, c2, gen1, bw, c1.live.ids.size(), c1.live.imgs.size(), idx_block, committed2);
    write_file(p, file);

    SavedConversation parsed;
    check(session_file_read(p.string(), id, parsed, bytes, error), "chain: multi-generation file reads");
    if (below_boundary ? !same_conv(parsed, c1) : !same_conv(parsed, c2)) {
        std::fprintf(stderr, "FAIL: chain reassembly differs from the %s state\n",
                     below_boundary ? "generation-1 (a below-boundary append must be refused)" : "generation-2");
        std::fprintf(stderr, "  live ids %zu/%zu imgs %zu/%zu gdn %zu/%zu ple %zu/%zu tails %zu/%zu used %llu/%llu\n",
                     parsed.live.ids.size(), c2.live.ids.size(), parsed.live.imgs.size(), c2.live.imgs.size(),
                     parsed.live.gdn.size(), c2.live.gdn.size(), parsed.live.ple.size(), c2.live.ple.size(),
                     parsed.live.tails.size(), c2.live.tails.size(), (unsigned long long) parsed.live.used,
                     (unsigned long long) c2.live.used);
        std::fprintf(stderr, "  checkpoints %zu/%zu\n", parsed.checkpoints.size(), c2.checkpoints.size());
        for (size_t j = 0; j < c2.kv.size() && j < parsed.kv.size(); ++j) {
            const ConversationKv& x = parsed.kv[j];
            const ConversationKv& y = c2.kv[j];
            std::fprintf(stderr, "  kv[%zu] cells %lld/%lld heads %lld/%lld hd %lld/%lld page %lld/%lld rows %lld/%lld idx %lld/%lld | k %zu/%zu v %zu/%zu ks %zu/%zu vs %zu/%zu po %zu/%zu\n",
                         j, (long long) x.cells, (long long) y.cells, (long long) x.heads, (long long) y.heads,
                         (long long) x.head_dim, (long long) y.head_dim, (long long) x.page_size,
                         (long long) y.page_size, (long long) x.pooled_rows, (long long) y.pooled_rows,
                         (long long) x.idx_dim, (long long) y.idx_dim, x.k.size(), y.k.size(), x.v.size(),
                         y.v.size(), x.k_scale.size(), y.k_scale.size(), x.v_scale.size(), y.v_scale.size(),
                         x.pooled.size(), y.pooled.size());
        }
        std::exit(1);
    }
    if (!below_boundary) check(bytes == file.size(), "chain: read reports the file size");

    // Cascade: damage inside generation 2 (truncations and a trailer byte-flip) -> the reader
    // lands on the generation-1 checkpoint (== c1), never a half state; damage at or below
    // gen 1's own trailer refuses the file.
    const uint64_t g2start = gen1.trailer_offset + 128;
    for (uint64_t cut : {g2start + 30, g2start + (file.size() - g2start) / 2, committed2 + 1, committed2 + 63,
                         committed2 + 127, file.size() - 1}) {
        const fs::path q = uniq("cut");
        write_file(q, std::vector<uint8_t>(file.begin(), file.begin() + (size_t) cut));
        SavedConversation back;
        std::string e2;
        bool ok = session_file_read(q.string(), id, back, bytes, e2);
        if (!ok || !same_conv(back, c1)) {
            std::fprintf(stderr, "FAIL: cut %llu did not cascade to the generation-1 checkpoint (ok=%d)\n",
                         (unsigned long long) cut, int(ok));
            std::exit(1);
        }
        ++checks;
    }
    for (uint64_t cut : {gen1.trailer_offset + 64, gen1.trailer_offset - 1}) {  // gen 1 itself damaged
        const fs::path q = uniq("deepcut");
        write_file(q, std::vector<uint8_t>(file.begin(), file.begin() + (size_t) cut));
        SavedConversation back;
        std::string e2;
        check(!session_file_read(q.string(), id, back, bytes, e2), "damage below gen 1's trailer refuses the file");
    }
    // A byte flip inside generation 2's trailer (no truncation): gen 1 stands.
    {
        auto damaged = file;
        damaged[(size_t) committed2 + 37] ^= 0xFF;
        const fs::path q = uniq("flip");
        write_file(q, damaged);
        SavedConversation back;
        std::string e2;
        check(session_file_read(q.string(), id, back, bytes, e2) && same_conv(back, c1),
              "a damaged newest trailer demotes to the previous checkpoint");
    }
}

void reader_tests() {
    // A full single-generation v3 file reads back exactly (this is the v2-parity golden for v3:
    // the same SavedConversation the engine would have written).
    {
        const SessionFileIdentity id{0x1234567890ABCDEFull, 0x0F0F0F0F0F0F0F0Full};
        std::string error;
        SessionWriteOptions opt;
        opt.use_v3 = true;
        const SavedConversation conv = make_conv(9);
        const fs::path p = uniq("readfull");
        size_t bytes = 0;
        check(session_file_write(p.string(), conv, id, bytes, error, opt), "reader: full v3 write");
        SavedConversation parsed;
        bool rd = session_file_read(p.string(), id, parsed, bytes, error);
        if (!rd) std::fprintf(stderr, "read error: [%s]\n", error.c_str());
        check(rd, "reader: full v3 read");
        check(same_conv(parsed, conv), "reader: single-generation reassembly equals the saved state");
        check(bytes == (size_t) slurp_bin(p).size(), "reader: reports the file size");
    }
    // Two chains: engine-aligned idx_block 128 and the misaligned 100 (S1 scenario C).
    chain_read_test(7, 128, 64, 256);
    chain_read_test(13, 100, 64, 256);
    // A hand-built group whose window starts below the committed boundary is refused by the
    // region-map builder (never read); the restore lands on the intact generation 1.
    chain_read_test(19, 128, 64, 256, /*below_boundary = */ true);
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
    v3_write_tests();
    reader_tests();
    fs::remove_all(append_dir);
    std::printf("conversation_append_test: %d checks passed\n", checks);
    return 0;
}
