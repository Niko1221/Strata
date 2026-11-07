// CPU-only tests for the format-independent session digest (session_file_digest, conversation_file.hpp): the same
// conversation written as v1, as a v2 REF file, and as the K/V-first v3 layout (with and without REF) digests equal
// in tokens, K/V and decoded state; the lossy RHT codec marks the state unusable.  Built with
// -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"
#include "strata/core/rht_state_codec.hpp"

#include <array>
#include <chrono>
#include <cmath>
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

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

std::vector<uint8_t> named_bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

// finite f32 values: the RHT codec refuses a non-finite or wrong-size GDN
std::vector<uint8_t> finite_state(size_t n, uint8_t seed) {
    auto v = named_bytes_of(n, seed);
    for (size_t at = 0; at + sizeof(float) <= n; at += sizeof(float)) {
        const float f = float(std::sin(double(at / sizeof(float)) * 0.7 + double(seed)) * 3.0);
        std::memcpy(v.data() + at, &f, sizeof f);
    }
    return v;
}

ConversationBuffer buffer_of(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

ConversationCheckpoint digest_checkpoint(size_t tokens, uint8_t seed, size_t gdn_bytes) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i * 3 + seed));
    c.imgs = {{int64_t(seed), 0x1234567890abcdefull ^ seed}};
    c.gdn = finite_state(gdn_bytes, seed);
    c.ple = named_bytes_of(77, uint8_t(seed + 1));
    c.tails = bytes_of(301, uint8_t(seed + 2));
    c.dead = bytes_of(64, uint8_t(seed + 3));
    c.block_pos = bytes_of(8, uint8_t(seed + 4));
    c.used = 42 + seed;
    return c;
}

// A conversation with one homomorphic checkpoint (a REF hit) and two K/V layers; its geometry is one the RHT codec
// accepts, so the same content can also be written lossily.
SavedConversation digest_sample(size_t gdn_bytes) {
    SavedConversation s;
    s.geometry = {2560, 2, 4, 128, 16, 1, 2, 1, 6144, 24, 2, 256, 4, 128, 4, 320, 512, 640};
    s.layer_lo = 0;
    s.layer_hi = 2;
    s.live = digest_checkpoint(64, 1, gdn_bytes);
    ConversationCheckpoint cp = s.live;   // the same five arrays: the writer stores it as REF
    cp.ids.assign(s.live.ids.begin(), s.live.ids.begin() + 8);
    cp.imgs = {{9, 0x1234}};
    cp.used = 7;
    s.checkpoints = {cp};
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer;
        kv.cells = 64; kv.heads = 2; kv.head_dim = 128;
        kv.page_size = 16; kv.pooled_rows = 4; kv.idx_dim = 128;
        kv.k = buffer_of(4096 + 17 * size_t(layer), uint8_t(layer));
        kv.v = buffer_of(2048, uint8_t(layer + 10));
        kv.k_scale = buffer_of(layer ? 0 : 512, uint8_t(layer + 20));
        kv.v_scale = buffer_of(512, uint8_t(layer + 30));
        kv.pooled = buffer_of(4 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

std::vector<char> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<char>& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize) d.size());
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-session-digest-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x0f0f0f0f11112222ull, 0x3333444455556666ull};
    std::string error;
    size_t written = 0;

    SavedConversation content = digest_sample(0);
    rht::Layout layout{};
    check(rht::layout_from(content.geometry, content.layer_lo, content.layer_hi, layout),
          "fixture: the geometry takes the RHT codec");
    content = digest_sample(size_t(layout.logical));

    // the same content in every format the digest must see through
    const fs::path v1 = dir / "v1.sess";
    check(session_file_write(v1.string(), content, id, written, error), "v1: write");
    SessionFileDigest d_v1;
    SessionStatus st;
    check(session_file_digest(v1.string(), id, d_v1, error, &st), "v1: digest");
    check(d_v1.file_version == 1 && st.file_version == 1, "v1: the header version is reported");
    check(d_v1.state_exact && !d_v1.state_ref && !d_v1.state_rht, "v1: RAW state is exact");

    const fs::path v2ref = dir / "v2ref.sess";
    SessionWriteOptions ref;
    ref.state_ref = true;
    check(session_file_write(v2ref.string(), content, id, written, error, ref), "v2 REF: write");
    SessionFileDigest d_v2ref;
    check(session_file_digest(v2ref.string(), id, d_v2ref, error, &st), "v2 REF: digest");
    check(d_v2ref.file_version == 2 && d_v2ref.state_ref && d_v2ref.state_exact,
          "v2 REF: version 2, REF reported, state exact");

    const fs::path v3 = dir / "v3.sess";
    SessionWriteOptions v3o;
    v3o.layout_v3 = true;
    check(session_file_write(v3.string(), content, id, written, error, v3o), "v3: write");
    SessionFileDigest d_v3;
    check(session_file_digest(v3.string(), id, d_v3, error, &st), "v3: digest");
    check(d_v3.file_version == 3 && d_v3.state_exact && !d_v3.state_rht, "v3: version 3, state exact");

    const fs::path v3ref = dir / "v3ref.sess";
    SessionWriteOptions v3r;
    v3r.layout_v3 = true;
    v3r.state_ref = true;
    check(session_file_write(v3ref.string(), content, id, written, error, v3r), "v3 REF: write");
    SessionFileDigest d_v3ref;
    check(session_file_digest(v3ref.string(), id, d_v3ref, error, &st), "v3 REF: digest");
    check(d_v3ref.file_version == 3 && d_v3ref.state_ref, "v3 REF: version 3, REF reported");

    // the engine's own save path: the K/V comes from sources (device or pinned host) instead of captured buffers
    {
        SavedConversation meta = content;
        meta.kv.clear();
        std::vector<SessionKvSource> sources;
        for (const auto& k : content.kv) {
            SessionKvSource s;
            s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
            s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t part = 0; part < 5; ++part) s.sizes[part] = parts[part]->size();
            s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                return parts[part]->read(dst, offset, n);
            };
            sources.push_back(std::move(s));
        }
        const fs::path p = dir / "v1-source.sess";
        check(session_file_write(p.string(), meta, sources, id, written, error), "source K/V: write");
        SessionFileDigest d;
        check(session_file_digest(p.string(), id, d, error), "source K/V: digest");
        check(d.tokens == d_v1.tokens && d.kv == d_v1.kv && d.state == d_v1.state,
              "source K/V: the write path does not change the digest");
    }

    check(d_v1.tokens == d_v2ref.tokens && d_v1.tokens == d_v3.tokens && d_v1.tokens == d_v3ref.tokens,
          "tokens: v1, v2 REF, v3 and v3 REF digest equal");
    check(d_v1.kv == d_v2ref.kv && d_v1.kv == d_v3.kv && d_v1.kv == d_v3ref.kv,
          "K/V: v1, v2 REF, v3 and v3 REF digest equal (REF and the section order do not change the bytes)");
    check(d_v1.state == d_v2ref.state && d_v1.state == d_v3.state && d_v1.state == d_v3ref.state,
          "state: REF resolves and v3 reorders to the same decoded content");

    // RHT: the tokens and the K/V are untouched, the decoded GDN is not
    const fs::path rht2 = dir / "rht2.sess";
    SessionWriteOptions rht;
    rht.state_codec_rht = true;
    check(session_file_write(rht2.string(), content, id, written, error, rht), "RHT: write (v2)");
    SessionFileDigest d_rht;
    check(session_file_digest(rht2.string(), id, d_rht, error, &st), "RHT: digest");
    check(d_rht.file_version == 2 && d_rht.state_rht && !d_rht.state_exact && !d_rht.state_ref,
          "RHT: version 2, marked lossy, not exact");
    check(d_rht.tokens == d_v1.tokens && d_rht.kv == d_v1.kv, "RHT: tokens and K/V still digest equal");
    check(d_rht.state != d_v1.state, "RHT: the decoded state is not the stored state");

    const fs::path rht3 = dir / "rht3.sess";
    SessionWriteOptions rhtv3;
    rhtv3.layout_v3 = true;
    rhtv3.state_codec_rht = true;
    check(session_file_write(rht3.string(), content, id, written, error, rhtv3), "v3 RHT: write");
    SessionFileDigest d_rht3;
    check(session_file_digest(rht3.string(), id, d_rht3, error, &st), "v3 RHT: digest");
    check(d_rht3.file_version == 3 && d_rht3.state_rht && !d_rht3.state_exact,
          "v3 RHT: version 3, marked lossy");
    check(d_rht3.tokens == d_v1.tokens && d_rht3.kv == d_v1.kv && d_rht3.state == d_rht.state,
          "v3 RHT: the layout does not change tokens, K/V or the lossy decode");

    // the digest tells two conversations apart
    {
        SavedConversation other = content;
        other.live.ids[0] += 1;
        const fs::path p = dir / "other-token.sess";
        check(session_file_write(p.string(), other, id, written, error), "other token: write");
        SessionFileDigest d;
        check(session_file_digest(p.string(), id, d, error), "other token: digest");
        check(d.tokens != d_v1.tokens && d.kv == d_v1.kv && d.state == d_v1.state,
              "other token: only the token digest moves");

        other.live.ids[0] -= 1;
        other.kv[0].k.visit(7, 5, [](uint8_t* p, size_t c, size_t) {
            for (size_t i = 0; i < c; ++i) p[i] ^= 0x5a;
            return true;
        });
        const fs::path q = dir / "other-kv.sess";
        check(session_file_write(q.string(), other, id, written, error), "other K/V: write");
        check(session_file_digest(q.string(), id, d, error), "other K/V: digest");
        check(d.kv != d_v1.kv && d.tokens == d_v1.tokens && d.state == d_v1.state,
              "other K/V: only the K/V digest moves");
    }

    // the identity the file carries is checked before anything is digested
    {
        SessionFileDigest d;
        check(!session_file_digest(v1.string(), {id.model ^ 1, id.config}, d, error) &&
              error.find("another model") != std::string::npos, "identity: another model is refused");
        check(!session_file_digest(v1.string(), {id.model, id.config ^ 1}, d, error) &&
              error.find("another engine configuration") != std::string::npos,
              "identity: another configuration is refused");
    }

    // a delta chain's record stream is refused by name: the digest describes a base file
    {
        SavedConversation meta = content;
        meta.kv.clear();
        const fs::path dlog = dir / "chain.dlog";
        uint64_t record_bytes = 0;
        check(session_delta_write(dlog.string(), 0, meta, {}, 0, id, record_bytes, error),
              "delta: a record is written");
        SessionFileDigest d;
        check(!session_file_digest(dlog.string(), id, d, error) &&
              error.find("delta record stream") != std::string::npos,
              "delta: a record stream is refused by name");
    }

    // a corrupted payload is refused, not digested
    {
        std::vector<char> bytes = slurp(v1);
        check(bytes.size() > 17, "corrupt: the v1 file has a payload");
        bytes[bytes.size() - 17] = char(bytes[bytes.size() - 17] ^ 0x5a);   // inside the last K/V part
        const fs::path p = dir / "corrupt.sess";
        spit(p, bytes);
        SessionFileDigest d;
        check(!session_file_digest(p.string(), id, d, error) && error.find("checksum") != std::string::npos,
              "corrupt: a payload edit is refused by the checksum");
    }

    fs::remove_all(dir);
    std::printf("conversation_digest_test: %d checks passed\n", checks);
    return 0;
}
