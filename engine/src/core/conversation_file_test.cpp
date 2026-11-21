// CPU-only tests for the on-disk session format (include/strata/core/conversation_file.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"
#include "strata/core/session_save_reclaim.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <new>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

// v2 fixtures: attach exactly one stage part per stage-table entry (identity follows the parent, the
// split/merge rule), and compare a checkpoint including its parts (parts carry no parts of their own)
void attach_parts(ConversationCheckpoint& c, size_t stages) {
    for (size_t k = 0; k < stages; ++k) {
        ConversationCheckpoint part;
        part.ids = c.ids; part.imgs = c.imgs; part.used = c.used + uint64_t(k);
        part.gdn = bytes_of(999, uint8_t(21 + k)); part.ple = bytes_of(7, uint8_t(31 + k));
        part.tails = bytes_of(11, uint8_t(41 + k)); part.dead = bytes_of(3, uint8_t(51 + k));
        part.block_pos = bytes_of(2, uint8_t(61 + k));
        c.stage_parts.push_back(std::move(part));
    }
}
bool same_deep(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids != b.ids || a.imgs != b.imgs || a.gdn != b.gdn || a.ple != b.ple || a.tails != b.tails ||
        a.dead != b.dead || a.block_pos != b.block_pos || a.used != b.used ||
        a.stage_parts.size() != b.stage_parts.size())
        return false;
    for (size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!same_deep(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}

SavedConversation sample() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(1000, 1);
    s.checkpoints.push_back(checkpoint(10, 2));
    s.checkpoints.push_back(checkpoint(500, 3));
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = layer == 2 ? 0 : 9; kv.idx_dim = 128;
        // > 16 MiB so the buffer spans several segments
        kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(layer));
        kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(layer == 1 ? 0 : 512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(layer == 2 ? 0 : 9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// format v1 golden (the frozen predecessor): the fixed sample gave these exact bytes under version 1; the v2
// golden below is this sample plus its stage-table fields, pinned by the same test.
constexpr size_t kGoldenSize = 17878535;
constexpr uint64_t kGoldenHash = 0x70f812b350360d09ull;
constexpr uint64_t kGoldenV2Hash = 0xf80ccf8cf1b579f2ull;

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return same_deep(a, b);
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}

std::vector<char> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<char>& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize) d.size());
}

// no temporary file (any leftover of a write) in the directory
bool no_temp(const fs::path& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0 && n != "keep.bin.tmp") return false;
    }
    return true;
}

bool rejects(const fs::path& p, const SessionFileIdentity& id, const char* expect = nullptr,
             const SessionReadLimits& limits = {}) {
    SavedConversation out;
    size_t bytes = 0;
    std::string error;
    const bool ok = session_file_read(p.string(), id, out, bytes, error, limits);
    if (!ok && expect && error.find(expect) == std::string::npos) {
        std::fprintf(stderr, "  unexpected error text: %s (wanted %s)\n", error.c_str(), expect);
        return false;
    }
    return !ok && !error.empty();
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-session-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    const SavedConversation original = sample();
    const fs::path good = dir / "good.bin";

    // round trip
    size_t written = 0;
    std::string error;
    check(session_file_write(good.string(), original, id, written, error), "write succeeds");
    check(error.empty(), "write leaves no error");

    // golden: format v1 is frozen. The fixed sample must give these exact bytes (little-endian header fields at
    // fixed offsets, and a fixed hash of the whole file); a format change must bump the version and this test.
    {
        const std::vector<char> g = slurp(good);
        auto u32 = [&](size_t o) { uint32_t v = 0; for (int i = 3; i >= 0; --i) v = v << 8 | uint8_t(g[o + i]); return v; };
        auto u64 = [&](size_t o) { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = v << 8 | uint8_t(g[o + i]); return v; };
        check(g.size() >= 80 && std::memcmp(g.data(), "STRSESS\x01", 8) == 0, "golden: magic");
        check(u32(8) == 2 && u32(12) == 64, "golden: version 2, header 64 bytes (little-endian)");
        check(u64(16) == id.model && u64(24) == id.config, "golden: fingerprints at offsets 16 and 24");
        check(u64(32) == g.size() - 64 - 16, "golden: payload length at offset 32");
        check(u64(40) == 0 && u64(48) == 0, "golden: reserved fields are zero");
        check(std::memcmp(g.data() + g.size() - 8, "STRSEND\x01", 8) == 0, "golden: end marker");
        const uint64_t h = session_hash64(g.data(), g.size(), 0);
        const char* golden = std::getenv("STRATA_SESSION_GOLDEN_PRINT");
        if (golden) std::printf("golden size %zu hash %016llx\n", g.size(), (unsigned long long)h);
        check(g.size() == kGoldenSize + 8 + 3 * 8 && h == kGoldenV2Hash,
              "golden: the fixed sample's v2 bytes are the v1 bytes plus the stage-table count and one parts-count per checkpoint");
    }
    check(written == fs::file_size(good), "write reports the file size");
    check(no_temp(dir), "no temporary left behind");
    {
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(good.string(), id, back, read, error), "read succeeds");
        check(read == written, "read reports the file size");
        check(same(original, back), "round trip is identical");
    }
    // writing twice produces the same bytes (deterministic format)
    {
        const fs::path again = dir / "again.bin";
        check(session_file_write(again.string(), original, id, written, error), "second write");
        check(slurp(again) == slurp(good), "format is deterministic");
    }
    // saving over an existing file replaces it (MoveFileExW on Windows, rename elsewhere)
    {
        const fs::path over = dir / "over.bin";
        spit(over, std::vector<char>(100, 'o'));
        check(session_file_write(over.string(), original, id, written, error), "write over an existing file");
        check(slurp(over) == slurp(good), "existing file replaced by the session");
        check(no_temp(dir), "no temporary left behind after replace");
    }
    // buffered I/O (the path taken where direct I/O is refused or unavailable) writes and reads the same bytes
    {
        const char* inherited = std::getenv("STRATA_SESSION_BUFFERED");
        const std::string kept = inherited ? inherited : "";
        set_env("STRATA_SESSION_BUFFERED", "1");
        const fs::path p = dir / "buffered.bin";
        check(session_file_write(p.string(), original, id, written, error), "buffered write");
        check(slurp(p) == slurp(good), "buffered write equals the default write");
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(p.string(), id, back, read, error) && same(original, back), "buffered read");
        set_env("STRATA_SESSION_BUFFERED", inherited ? kept.c_str() : nullptr);   // the caller's choice again
    }

    const std::vector<char> image = slurp(good);
    // truncation at every structural boundary and in the middle
    for (size_t cut : {size_t(0), size_t(7), size_t(63), size_t(64), size_t(65), size_t(200), image.size() / 2,
                       image.size() - 17, image.size() - 16, image.size() - 8, image.size() - 1}) {
        const fs::path p = dir / "cut.bin";
        spit(p, std::vector<char>(image.begin(), image.begin() + (std::ptrdiff_t) cut));
        check(rejects(p, id), "truncated file rejected");
    }
    // trailing garbage
    {
        auto d = image; d.push_back('x');
        const fs::path p = dir / "tail.bin"; spit(p, d);
        check(rejects(p, id, "size"), "file with trailing bytes rejected");
    }
    // flipped payload byte (inside the big K buffer), flipped header byte, flipped trailer
    for (size_t at : {size_t(64 + 300), image.size() / 2, size_t(20), image.size() - 12, image.size() - 3}) {
        auto d = image; d[at] ^= 0x40;
        const fs::path p = dir / "flip.bin"; spit(p, d);
        check(rejects(p, id), "corrupted byte rejected");
    }
    // other model or other configuration
    check(rejects(good, {id.model ^ 1, id.config}, "model"), "different model fingerprint rejected");
    check(rejects(good, {id.model, id.config ^ 1}, "config"), "different configuration rejected");
    // wrong version: header field patched and header hash recomputed so only the version differs
    {
        auto d = image;
        uint32_t v = 3; std::memcpy(d.data() + 8, &v, 4);
        const uint64_t h = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &h, 8);
        const fs::path p = dir / "ver.bin"; spit(p, d);
        check(rejects(p, id, "version"), "unknown version rejected");
    }
    // format v2 refuses the frozen v1 file: header patched down to version 1 with the header hash recomputed
    {
        auto d = image;
        uint32_t v = 1; std::memcpy(d.data() + 8, &v, 4);
        const uint64_t h = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &h, 8);
        const fs::path p = dir / "ver1.bin"; spit(p, d);
        check(rejects(p, id, "version"), "a v1 file is refused by a v2 engine");
    }
    // a corrupted element count with a recomputed payload hash must fail on bounds, not allocate
    {
        auto d = image;
        // payload starts at 64: geometry 18*8, layer_lo, layer_hi, cvec, stage-table count = 22*8 bytes, then live.ids count
        const size_t at = 64 + 22 * 8;
        uint64_t huge = 0x0fffffffffffffffull; std::memcpy(d.data() + at, &huge, 8);
        uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
        const uint64_t h = session_hash64(d.data() + 64, payload, 0);
        std::memcpy(d.data() + 64 + payload, &h, 8);
        const fs::path p = dir / "count.bin"; spit(p, d);
        check(rejects(p, id, "count"), "oversized count rejected before allocation");
    }
    // missing file
    check(rejects(dir / "absent.bin", id), "missing file rejected");
    // a failed read leaves the destination untouched
    {
        SavedConversation keep = sample();
        size_t b = 0;
        check(!session_file_read((dir / "absent.bin").string(), id, keep, b, error), "missing read fails");
        check(same(keep, original), "failed read does not modify the destination");
    }
    // v2 stage consistency, checked before any disk access: a live checkpoint's stage parts must match the
    // stage table - this sample's table is empty, so a stray part is refused and no file appears
    {
        SavedConversation split = sample();
        split.live.stage_parts.push_back(checkpoint(1, 9));
        const fs::path p = dir / "split.bin";
        check(!session_file_write(p.string(), split, id, written, error), "stage parts outside the stage table are refused");
        check(!fs::exists(p) && no_temp(dir), "refused write leaves no file");
    }
    // a write into a missing directory fails cleanly
    check(!session_file_write((dir / "nope" / "x.bin").string(), original, id, written, error),
          "write into a missing directory fails");
    // the hash is position-sensitive and seed-sensitive
    {
        const char a[] = "abcdefghijklmnopqrstuvwxyz0123456789", b[] = "bacdefghijklmnopqrstuvwxyz0123456789";
        check(session_hash64(a, sizeof a, 0) != session_hash64(b, sizeof b, 0), "hash sees swapped bytes");
        check(session_hash64(a, sizeof a, 0) != session_hash64(a, sizeof a, 1), "hash sees the seed");
        SessionHasher h; h.update(a, 5); h.update(a + 5, sizeof a - 5);
        check(h.digest() == session_hash64(a, sizeof a, 0), "streaming hash equals one-shot hash");
    }
    // model fingerprint: changes with content at the head of a file and with the file list
    {
        const fs::path m1 = dir / "m1.bin", m2 = dir / "m2.bin";
        spit(m1, std::vector<char>(3u << 20, 'a'));
        spit(m2, std::vector<char>(100, 'b'));
        uint64_t f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0, f7 = 0, f8 = 0;
        const std::vector<SessionModelFile> both = {{"a", m1.string()}, {"b", m2.string()}};
        check(session_model_fingerprint(both, f1, error), "fingerprint ok");
        const std::vector<SessionModelFile> subset = {{"a", m1.string()}};
        const std::vector<SessionModelFile> roles = {{"b", m1.string()}, {"a", m2.string()}};
        check(session_model_fingerprint(subset, f2, error), "fingerprint subset ok");
        check(session_model_fingerprint(roles, f5, error), "fingerprint roles ok");
        check(f1 != f5, "fingerprint follows the role each file plays");
        // a moved folder: same roles, same bytes, other paths
        const fs::path moved = dir / "moved";
        fs::create_directories(moved);
        fs::copy_file(m1, moved / "m1.bin"); fs::copy_file(m2, moved / "m2.bin");
        const std::vector<SessionModelFile> moved_list = {{"a", (moved / "m1.bin").string()}, {"b", (moved / "m2.bin").string()}};
        check(session_model_fingerprint(moved_list, f6, error) && f6 == f1, "fingerprint does not follow the folder");
        // an optional input that is absent is recorded, not an error; a present one changes the fingerprint
        const std::vector<SessionModelFile> absent = {{"a", m1.string()}, {"b", m2.string()}, {"c", (dir / "nothere").string(), true}};
        const std::vector<SessionModelFile> present = {{"a", m1.string()}, {"b", m2.string()}, {"c", m2.string(), true}};
        check(session_model_fingerprint(absent, f7, error), "absent optional input ok");
        check(session_model_fingerprint(present, f8, error) &&
              f7 != f8 && f7 != f1, "optional input presence enters the fingerprint");
        auto d = slurp(m1); d[10] = 'z'; spit(m1, d);
        check(session_model_fingerprint(both, f3, error), "fingerprint after edit ok");
        check(f1 != f2 && f1 != f3, "fingerprint follows files and content");
        check(!session_model_fingerprint({{"x", (dir / "absent").string()}}, f4, error), "fingerprint of a missing file fails");
    }

    // only the deepest checkpoint is worth a disk write: the next turn resumes from it
    {
        std::vector<ConversationCheckpoint> chain;
        check(session_checkpoints_to_save(chain).empty(), "no checkpoints, none saved");
        chain.push_back(checkpoint(10, 2));
        chain.push_back(checkpoint(700, 3));
        chain.push_back(checkpoint(500, 4));
        const auto kept = session_checkpoints_to_save(chain);
        check(kept.size() == 1 && same_checkpoint(kept[0], chain[1]), "deepest checkpoint kept, alone");
    }
    // K/V streamed from its owner (no host copy): the same bytes as writing the captured image
    {
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> sources;
        for (const auto& k : original.kv) {
            SessionKvSource s;
            s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
            s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i]->size();
            s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                return parts[part]->read(dst, offset, n);
            };
            sources.push_back(std::move(s));
        }
        const fs::path p = dir / "streamed.bin";
        check(session_file_write(p.string(), meta, sources, id, written, error), "streamed write succeeds");
        check(slurp(p) == image, "streamed write equals the captured-image write");
        // a failing source leaves no file
        auto broken = sources;
        broken[1].read = [](size_t, size_t, void*, size_t) { return false; };
        const fs::path q = dir / "broken.bin";
        check(!session_file_write(q.string(), meta, broken, id, written, error), "failing source fails the write");
        check(!fs::exists(q) && no_temp(dir), "failed streamed write leaves no file");
        // K/V in the image and as sources at once is ambiguous: refused
        check(!session_file_write((dir / "both.bin").string(), original, sources, id, written, error),
              "image K/V plus sources refused");
    }

    // the configuration fingerprint: every field counts, doubles by their exact bits
    {
        SessionConfig c;
        c.engine_version = "0.1.38"; c.backend = "cuda"; c.kv = "int8"; c.max_context = 65536; c.kv_resident = 0;
        c.mtp_window = 4; c.rope.type = 0; c.rope.freq_base = 1e7; c.rope.factor = 1.0; c.rope.freq_scale = 1.0;
        c.rope.orig_ctx = 262144; c.rope.attn_factor = 1.0; c.rope.beta_fast = 32; c.rope.beta_slow = 1;
        c.switches = {{"STRATA_FAST_GDN", 0}};
        const uint64_t base = session_config_fingerprint(c);
        check(session_config_fingerprint(c) == base, "config fingerprint is deterministic");
        auto differs = [&](auto edit) { SessionConfig d = c; edit(d); return session_config_fingerprint(d) != base; };
        check(differs([](SessionConfig& d) { d.engine_version = "0.1.39"; }), "engine version enters the config");
        check(differs([](SessionConfig& d) { d.backend = "hip"; }), "backend enters the config");
        check(differs([](SessionConfig& d) { d.kv = "fp16"; }), "kv enters the config");
        check(differs([](SessionConfig& d) { d.max_context = 65537; }), "max context enters the config");
        check(differs([](SessionConfig& d) { d.kv_resident = 1; }), "kv residency enters the config");
        check(differs([](SessionConfig& d) { d.mtp_window = 0; }), "mtp window enters the config");
        check(differs([](SessionConfig& d) { d.kv_rot = true; }), "kv rotation enters the config");
        check(differs([](SessionConfig& d) { d.rope.factor = std::nextafter(1.0, 2.0); }),
              "a one-ulp rope change enters the config");
        check(differs([](SessionConfig& d) { d.rope.type = 2; }), "rope type enters the config");
        check(differs([](SessionConfig& d) { d.rope.beta_slow = 2; }), "rope beta enters the config");
        check(differs([](SessionConfig& d) { d.cvec = 7; }), "control vectors enter the config");
        check(differs([](SessionConfig& d) { d.switches[0].second = 1; }), "a switch value enters the config");
        check(differs([](SessionConfig& d) { d.switches.push_back({"X", 0}); }), "the switch list enters the config");
        // field boundaries: ("ab","c") and ("a","bc") hash differently
        SessionIdentityBuilder x(1), y(1);
        x.str("ab", "c"); y.str("a", "bc");
        check(x.digest() != y.digest(), "identity fields are length-delimited");
        SessionIdentityBuilder i(1), u(1);
        i.i64("n", 5); u.u64("n", 5);
        check(i.digest() != u.digest(), "identity fields are typed");
    }
    // bounds a read checks before it allocates, and the caller's admission
    {
        SessionReadLimits l;
        l.max_tokens = 999;   // the live state holds 1000
        check(rejects(good, id, "limit", l), "token count over the limit refused");
        l = {}; l.max_checkpoints = 1;
        check(rejects(good, id, "limit", l), "checkpoint count over the limit refused");
        l = {}; l.max_kv_layers = 2;
        check(rejects(good, id, "limit", l), "K/V layer count over the limit refused");
        l = {}; l.max_file_bytes = image.size() - 1;
        check(rejects(good, id, "limit", l), "file over the size limit refused");
        l = {};
        uint64_t asked = 0;
        l.admit = [&](uint64_t n, std::string& why) { asked = n; why = "no room for it"; return false; };
        check(rejects(good, id, "no room", l) && asked == session_read_peak_bytes(image.size()) &&
              asked > image.size(), "admission sees the parse's peak (more than the file) and can refuse");
        l = {};
        bool admitted = false;
        l.admit = [&](uint64_t, std::string&) { admitted = true; return true; };
        check(rejects(good, {id.model ^ 1, id.config}, "model", l) && !admitted,
              "admission is not asked for a foreign file");
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(good.string(), id, back, n, error, l) && admitted && same(back, original),
              "admitted file reads");
    }
    // format v2: the stage table round-trips, and the live layout must match the file's before anything is read
    {
        SavedConversation with_table = original;
        with_table.stage_table = {{12, 47, 9, 2, false}, {48, 60, 3, 1, true}};
        attach_parts(with_table.live, with_table.stage_table.size());
        for (auto& c : with_table.checkpoints) attach_parts(c, with_table.stage_table.size());
        const fs::path p = dir / "table.bin";
        check(session_file_write(p.string(), with_table, id, written, error), "stage table: written");
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(p.string(), id, back, n, error) && back.stage_table == with_table.stage_table &&
              same(back, with_table), "stage table: round trip keeps the geometry and the state");
        SessionReadLimits l;
        l.stages = with_table.stage_table;                        // a live engine with the same split
        check(session_file_read(p.string(), id, back, n, error, l) && back.stage_table == with_table.stage_table,
              "stage table: a matching live layout reads");
        l.stages = {SessionStageGeom{12, 40, 9, 2, false}};       // a different split
        check(rejects(p, id, "split", l), "stage table: another live split refused");
        l.stages = std::vector<SessionStageGeom>{};               // a single-device runtime (an ENGAGED empty vector:
        // an optional reset would skip the comparison, and `= {}` can mean either — engines must assign the vector)
        check(rejects(p, id, "split", l), "stage table: a split file on a single-device runtime refused");
        l.stages = {SessionStageGeom{12, 47, 9, 2, false}};       // one live stage, two in the file
        check(rejects(p, id, "split", l), "stage table: a shorter live split refused");
        l.stages = with_table.stage_table;                        // a single-device file on a split runtime
        check(rejects(good, id, "split", l), "stage table: a single-device file on a split runtime refused");
        l = {};
        l.max_stages = 1;                                         // fewer stages than the engine will allow
        check(rejects(p, id, "limit", l), "stage table: more stages than the runtime limit refused");
    }
    // format v2: every checkpoint carries exactly one stage part per table entry; the count and nesting are
    // checked on both sides, and parts' state arrays are bounded per stage
    {
        SavedConversation split = original;
        split.stage_table = {{12, 47, 9, 2, false}, {48, 60, 3, 1, true}};
        attach_parts(split.live, split.stage_table.size());
        for (auto& c : split.checkpoints) attach_parts(c, split.stage_table.size());
        const fs::path sp = dir / "parts.bin";
        check(session_file_write(sp.string(), split, id, written, error), "stage parts: written with a matching table");
        SavedConversation back;
        size_t n = 0;
        SessionReadLimits l;
        l.stages = split.stage_table;
        check(session_file_read(sp.string(), id, back, n, error, l) && back.stage_table == split.stage_table &&
              same_deep(back.live, split.live) && back.checkpoints.size() == split.checkpoints.size() &&
              same_deep(back.checkpoints[0], split.checkpoints[0]) &&
              same_deep(back.checkpoints[1], split.checkpoints[1]),
              "stage parts: round trip keeps every part");

        auto bad = split; bad.live.stage_parts.pop_back();
        check(!session_file_write((dir / "bad.bin").string(), bad, id, written, error) &&
              error.find("stage parts") != std::string::npos && !fs::exists(dir / "bad.bin") && no_temp(dir),
              "stage parts: a checkpoint disagreeing with the table is refused at the write, no file left");
        auto bad2 = split; bad2.stage_table.pop_back();
        check(!session_file_write((dir / "bad2.bin").string(), bad2, id, written, error) &&
              !fs::exists(dir / "bad2.bin") && no_temp(dir), "stage parts: an extra table entry refused too");
        auto nested = split; nested.live.stage_parts[0].stage_parts.push_back(checkpoint(1, 77));
        check(!session_file_write((dir / "nest.bin").string(), nested, id, written, error) &&
              error.find("nested") != std::string::npos && !fs::exists(dir / "nest.bin") && no_temp(dir),
              "stage parts: nesting is refused at the write");

        // reader: the live checkpoint's parts count tampered down, payload hash recomputed
        {
            auto d = slurp(sp);
            const size_t live_cp = 64 + 22 * 8 + 2 * 40;         // header, 22 u64s, the two stage-table entries
            const size_t at = live_cp + 8 + 1000 * 4 + 8 + 1 * 16 + (8 + 4099) + (8 + 77) + (8 + 301) + (8 + 64) +
                              (8 + 8) + 8;                        // ...used, then the stage-parts count field
            uint64_t was = 0; std::memcpy(&was, d.data() + at, 8);
            check(was == 2, "stage parts: test offset finds the parts count");
            was = 1; std::memcpy(d.data() + at, &was, 8);
            uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
            const uint64_t h = session_hash64(d.data() + 64, payload, 0); std::memcpy(d.data() + 64 + payload, &h, 8);
            const fs::path tp = dir / "tamper.bin"; spit(tp, d);
            check(rejects(tp, id, "stage parts", l), "stage parts: a count disagreeing with the table is refused");
        }
        // reader: a part's own parts count field set to 1 is refused as nesting
        {
            auto d = slurp(sp);
            const size_t live_cp = 64 + 22 * 8 + 2 * 40;
            const size_t parent_count = live_cp + 8 + 1000 * 4 + 8 + 1 * 16 + (8 + 4099) + (8 + 77) + (8 + 301) +
                                        (8 + 64) + (8 + 8) + 8;  // the parent's stage-parts count field
            const size_t at = parent_count + 8 + 8 + 1000 * 4 + 8 + 1 * 16 + (8 + 999) + (8 + 7) + (8 + 11) +
                              (8 + 3) + (8 + 2) + 8;             // first part: record, then its own count field
            uint64_t was = 0; std::memcpy(&was, d.data() + at, 8);
            check(was == 0, "stage parts: test offset finds the part's own count");
            was = 1; std::memcpy(d.data() + at, &was, 8);
            uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
            const uint64_t h = session_hash64(d.data() + 64, payload, 0); std::memcpy(d.data() + 64 + payload, &h, 8);
            const fs::path tp = dir / "nesttamper.bin"; spit(tp, d);
            check(rejects(tp, id, "nested", l), "stage parts: a tampered part nesting is refused");
        }
        // per-stage bounds: exact limits read, one byte less refuses, before any allocation
        {
            SessionReadLimits l2;
            l2.stages = split.stage_table;
            l2.max_stage_state_bytes = {{999, 7, 11, 3, 2}, {999, 7, 11, 3, 2}};
            check(session_file_read(sp.string(), id, back, n, error, l2) && same_deep(back.live, split.live),
                  "stage parts: exact per-stage limits read");
            l2.max_stage_state_bytes[1][0] = 998;
            check(rejects(sp, id, "limit", l2), "stage parts: a part state array over its stage limit refused");
        }
        // review battery: every new v2 section is inside the payload hash, and every structural field has
        // its own check - a raw byte flip fails the hash, a rehashed flip fails the structure.
        {
            auto rehash = [](std::vector<char>& d) {
                uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
                const uint64_t h = session_hash64(d.data() + 64, payload, 0);
                std::memcpy(d.data() + 64 + payload, &h, 8);
            };
            const size_t live_cp = 64 + 22 * 8 + 2 * 40;                 // the live checkpoint's record
            const size_t parts_at = live_cp + 8 + 1000 * 4 + 8 + 1 * 16 + (8 + 4099) + (8 + 77) + (8 + 301) +
                                    (8 + 64) + (8 + 8) + 8;              // ...used, then its parts count
            const size_t part_len = 8 + 1000 * 4 + 8 + 1 * 16 + (8 + 999) + (8 + 7) + (8 + 11) + (8 + 3) +
                                    (8 + 2) + 8 + 8;                     // a part: arrays, used, own count
            auto arrays = [](size_t rec) {                               // a part record's data arrays
                return std::vector<std::pair<const char*, size_t>>{
                    {"ids",       rec + 8 + 4},
                    {"imgs",      rec + 8 + 1000 * 4 + 8 + 4},
                    {"gdn",       rec + 8 + 1000 * 4 + 8 + 16 + 8 + 10},
                    {"ple",       rec + 8 + 1000 * 4 + 8 + 16 + (8 + 999) + 8 + 3},
                    {"tails",     rec + 8 + 1000 * 4 + 8 + 16 + (8 + 999) + (8 + 7) + 8 + 5},
                    {"dead",      rec + 8 + 1000 * 4 + 8 + 16 + (8 + 999) + (8 + 7) + (8 + 11) + 8 + 1},
                    {"block_pos", rec + 8 + 1000 * 4 + 8 + 16 + (8 + 999) + (8 + 7) + (8 + 11) + (8 + 3) + 8 + 1},
                };
            };
            const fs::path sp2 = dir / "battery.bin";
            check(session_file_write(sp2.string(), split, id, written, error), "battery: fixture written");
            auto flip = [&](size_t at, const char* name, const char* needle, const SessionReadLimits& lim) {
                auto d = slurp(sp2);
                d[at] ^= 0x40;
                const fs::path tp = dir / name; spit(tp, d);
                return rejects(tp, id, needle, lim);
            };
            // raw flips: every new section is caught - the stage table first by the live-layout comparison
            // (a wall in front of the hash), the checkpoint arrays and K/V bytes by the payload hash
            check(flip(64 + 22 * 8 + 4, "bat-table.bin", "split", l),
                  "battery: a stage table byte is caught by the layout comparison");
            for (size_t p = 0; p < 2; ++p) {
                const size_t rec = parts_at + 8 + p * part_len;
                for (const auto& [what, at] : arrays(rec)) {
                    const std::string nm = std::string("bat-") + (p ? "p1-" : "p0-") + what + ".bin";
                    check(flip(at, nm.c_str(), "checksum", l), ("battery: a byte in part " + std::to_string(p) + " " + what +
                                                                " is hashed").c_str());
                }
            }
            {
                uint64_t payload = 0; std::memcpy(&payload, slurp(sp2).data() + 32, 8);
                // the last kv entry (layer 2): ... v_scale count(8)+data(512), pooled count(8)+data(0)
                check(flip(64 + payload - 8 - 256, "bat-kvlast.bin", "checksum", l),
                      "battery: the last K/V bytes of the last stage group are hashed");
            }
            // rehashed flips: the structural checks stand on their own
            {   // a stage table entry grown past the live layout: the layout comparison refuses
                auto d = slurp(sp2);
                uint64_t cells = 0; std::memcpy(&cells, d.data() + 64 + 22 * 8 + 16, 8);
                check(cells == 9, "battery: offset finds the stage QSA carve size");
                cells += 1; std::memcpy(d.data() + 64 + 22 * 8 + 16, &cells, 8);
                rehash(d);
                const fs::path tp = dir / "bat-table-h.bin"; spit(tp, d);
                check(rejects(tp, id, "split", l), "battery: a rehashed stage table mismatch is refused");
            }
            {   // a chain checkpoint's parts count trimmed: the parts-vs-table rule refuses
                const size_t cp0 = parts_at + 8 + 2 * part_len + 8;     // the checkpoint-array count field
                uint64_t n0 = 0; std::memcpy(&n0, slurp(sp2).data() + cp0, 8);
                check(n0 == 10, "battery: offset finds the first chain checkpoint's ids count");
                const size_t at = cp0 + 8 + 10 * 4 + 8 + 1 * 16 + (8 + 4099) + (8 + 77) + (8 + 301) + (8 + 64) +
                                  (8 + 8) + 8;
                auto d = slurp(sp2);
                uint64_t cnt = 0; std::memcpy(&cnt, d.data() + at, 8);
                check(cnt == 2, "battery: offset finds a chain checkpoint's parts count");
                cnt = 1; std::memcpy(d.data() + at, &cnt, 8);
                rehash(d);
                const fs::path tp = dir / "bat-chain-h.bin"; spit(tp, d);
                check(rejects(tp, id, "stage parts", l),
                      "battery: a rehashed chain checkpoint parts mismatch is refused");
            }
            {   // a part's array count grown past its stage limit: the per-stage bound refuses
                SessionReadLimits l3;
                l3.stages = split.stage_table;
                l3.max_stage_state_bytes = {{999, 7, 11, 3, 2}, {999, 7, 11, 3, 2}};
                auto d = slurp(sp2);
                const size_t at = parts_at + 8 + 8 + 1000 * 4 + 8 + 16;   // part 0's gdn count field
                uint64_t n = 0; std::memcpy(&n, d.data() + at, 8);
                check(n == 999, "battery: offset finds a part array's count");
                n = 1001; std::memcpy(d.data() + at, &n, 8);
                rehash(d);
                const fs::path tp = dir / "bat-grown-h.bin"; spit(tp, d);
                check(rejects(tp, id, "limit", l3),
                      "battery: a rehashed grown part array is refused by the stage limit");
            }
            // a refused read admission is typed memory (the engine's RAM-floor refusal path)
            {
                SessionReadLimits lm;
                lm.admit = [](uint64_t, std::string& why) { why = "not enough RAM to read it (test)"; return false; };
                SavedConversation out; size_t bytes = 0; std::string error; SessionStatus st;
                check(!session_file_read(sp2.string(), id, out, bytes, error, lm, &st) &&
                      st.error == SessionError::memory && error.find("not enough RAM") != std::string::npos,
                      "battery: a refused read admission is typed memory");
            }
        }
    }
    // free-space reserve: an impossible reserve refuses the write and leaves the existing file untouched
    {
        const fs::path p = dir / "reserve.bin";
        spit(p, std::vector<char>(10, 'r'));
        SessionWriteOptions opt;
        opt.min_free_bytes = UINT64_MAX / 2;
        check(!session_file_write(p.string(), original, id, written, error, opt) &&
              error.find("disk space") != std::string::npos, "write over the free-space reserve refused");
        check(slurp(p) == std::vector<char>(10, 'r') && no_temp(dir), "refused write keeps the old file, no temporary");
        opt.min_free_bytes = 1;
        uint64_t calls = 0;
        opt.progress = [&](uint64_t, uint64_t) { ++calls; };
        check(session_file_write(p.string(), original, id, written, error, opt) && slurp(p) == image,
              "write with a small reserve succeeds");
    }
    // a failed write over an existing session keeps it, and touches no other file (a planted "<name>.tmp", the old
    // fixed temporary pattern included)
    {
        const fs::path p = dir / "keep.bin", planted = dir / "keep.bin.tmp";
        check(session_file_write(p.string(), original, id, written, error), "keep: first write");
        spit(planted, std::vector<char>(5, 'p'));
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> broken(1);
        broken[0].sizes = {100, 0, 0, 0, 0};
        broken[0].read = [](size_t, size_t, void*, size_t) { return false; };
        check(!session_file_write(p.string(), meta, broken, id, written, error), "keep: failing write fails");
        check(slurp(p) == image, "keep: the previous session survives a failed write");
        check(slurp(planted) == std::vector<char>(5, 'p') && no_temp(dir), "keep: no other file touched");
    }
    // R1: the input list carries every expert file the loader resolved, per (layer, role) - a GGUF that only
    // native_experts.txt names changes the fingerprint when it changes; one file in two roles is two entries
    {
        const fs::path g1 = dir / "ext-a.gguf", g2 = dir / "ext-b.gguf";
        spit(g1, std::vector<char>(4096, 'g'));
        spit(g2, std::vector<char>(4096, 'h'));
        SessionInputs in;
        in.native_shards = {g1.string()};
        in.experts = {{"expert blk.0.ffn_gate", g1.string()}, {"expert blk.0.ffn_up", g2.string()},
                      {"expert blk.0.ffn_down", g2.string()}};
        const auto list = session_model_inputs(in);
        check(list.size() == 4, "inputs: shard plus three expert roles, none merged");
        uint64_t a = 0, b = 0, c = 0;
        check(session_model_fingerprint(list, a, error), "inputs: fingerprint");
        auto d = slurp(g2); d[0] = 'X'; spit(g2, d);   // the expert-only file changes; the CLI shard does not
        check(session_model_fingerprint(list, b, error) && a != b, "inputs: an expert-only GGUF enters the fingerprint");
        SessionInputs swapped = in;
        std::swap(swapped.experts[0].second, swapped.experts[1].second);
        check(session_model_fingerprint(session_model_inputs(swapped), c, error) && c != b,
              "inputs: the role each expert file plays enters the fingerprint");
    }
    // R2: runtime limits bound every array before it is allocated, and the file size follows from them
    {
        auto exact = [&] {
            SessionReadLimits l;
            l.max_tokens = 1000; l.max_checkpoints = 2; l.max_kv_layers = 3;
            l.max_stages = 0;   // v2 stage table: this sample is single-device
            l.geometry = original.geometry;
            l.layer_range = std::make_pair(original.layer_lo, original.layer_hi);
            l.max_state_bytes = {4099, 77, 301, 64, 8};
            for (const auto& k : original.kv)
                l.max_kv_bytes.push_back({k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()});
            return l;
        };
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(good.string(), id, back, n, error, exact()) && same(back, original),
              "limits: a file within the exact runtime limits reads");
        check(session_read_max_file_bytes(exact()) >= image.size() &&
              session_read_max_file_bytes(exact()) < UINT64_MAX, "limits: the runtime limits bound the file size");
        check(session_read_max_file_bytes(SessionReadLimits{}) == UINT64_MAX, "limits: open limits leave it open");
        auto l = exact(); l.max_state_bytes[0] = 4098;
        check(rejects(good, id, "limit", l), "limits: a GDN state over the runtime size refused");
        l = exact(); l.max_state_bytes[4] = 7;
        check(rejects(good, id, "limit", l), "limits: block_pos over the runtime size refused");
        l = exact(); l.max_kv_bytes[0][0] = original.kv[0].k.size() - 1;
        check(rejects(good, id, "limit", l), "limits: a K buffer over the runtime size refused");
        l = exact(); l.max_kv_bytes[2][1] = original.kv[2].v.size() - 1;
        check(rejects(good, id, "limit", l), "limits: the draft's V over the runtime size refused");
        l = exact(); (*l.geometry)[3] ^= 1;
        check(rejects(good, id, "geometry", l), "limits: another geometry refused before the state is read");
        l = exact(); l.layer_range = std::make_pair(int64_t(0), int64_t(24));
        check(rejects(good, id, "layer", l), "limits: another layer range refused");
        // a huge count in a state array with a recomputed hash: refused by the bound, nothing allocated
        auto d = image;
        // live: ids count (8) + 1000*4, imgs count (8) + 1*16, then the gdn byte count
        const size_t at = 64 + 22 * 8 + 8 + 1000 * 4 + 8 + 16;
        uint64_t was = 0; std::memcpy(&was, d.data() + at, 8);
        check(was == 4099, "limits: test offset finds the GDN count");
        uint64_t huge = 3ull << 30; std::memcpy(d.data() + at, &huge, 8);
        uint64_t payload = 0; std::memcpy(&payload, d.data() + 32, 8);
        const uint64_t h = session_hash64(d.data() + 64, payload, 0);
        std::memcpy(d.data() + 64 + payload, &h, 8);
        const fs::path p = dir / "bigstate.bin"; spit(p, d);
        check(rejects(p, id, nullptr, exact()), "limits: an oversized state count refused before allocation");
        // status kinds of a read
        SessionStatus st;
        check(!session_file_read(p.string(), id, back, n, error, exact(), &st) && st.error == SessionError::invalid,
              "status: a bad file is 'invalid'");
        l = exact();
        l.admit = [](uint64_t, std::string& why) { why = "no RAM"; return false; };
        check(!session_file_read(good.string(), id, back, n, error, l, &st) && st.error == SessionError::memory,
              "status: the RAM preflight is 'memory'");
    }
    // R3/R5: injected failures at each write step - kind, publication and the old file
    {
        const fs::path p = dir / "fault.bin";
        const std::vector<char> old(10, 'q');
        struct Case { const char* step; int err; SessionError kind; bool published; };
        for (const Case c : {Case{"write", ENOSPC, SessionError::storage, false},
#ifdef EDQUOT
                             Case{"write", EDQUOT, SessionError::storage, false},
#endif
                             Case{"write", EIO, SessionError::io, false},
                             Case{"file_flush", EIO, SessionError::io, false},
                             Case{"rename", EACCES, SessionError::io, false},
                             Case{"dir_flush", EIO, SessionError::io, true}}) {
            spit(p, old);
            SessionWriteOptions opt;
            const std::string want = c.step;
            const int code = c.err;
            opt.fault = [want, code](const char* step) { return want == step ? code : 0; };
            SessionStatus st;
            const bool ok = session_file_write(p.string(), original, id, written, error, opt, &st);
            check(!ok && st.error == c.kind, "fault: the failure has its kind");
            check(st.published == c.published, "fault: publication reported exactly");
            if (c.published) {
                check(slurp(p) == image && error.find("replaced") != std::string::npos,
                      "fault: after a published failure the new file is there and the error says so");
            } else {
                check(slurp(p) == old, "fault: a failure before the rename keeps the old file");
            }
            check(no_temp(dir), "fault: no temporary left behind");
        }
#ifndef _WIN32
        // a filesystem that cannot flush a folder: saved, and said
        spit(p, old);
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::string(step) == "dir_flush" ? EINVAL : 0; };
        SessionStatus st;
        check(session_file_write(p.string(), original, id, written, error, opt, &st) && st.dir_flush_unsupported &&
              slurp(p) == image, "fault: EINVAL on the folder flush is not a failure, and is reported");
#endif
        // the free-space preflight is 'storage'
        SessionWriteOptions big;
        big.min_free_bytes = UINT64_MAX / 2;
        SessionStatus st2;
        check(!session_file_write(p.string(), original, id, written, error, big, &st2) &&
              st2.error == SessionError::storage, "status: the free-space preflight is 'storage'");
        check(std::string(session_error_name(SessionError::storage)) == "storage" &&
              std::string(session_error_name(SessionError::memory)) == "memory" &&
              std::string(session_error_name(SessionError::invalid)) == "invalid" &&
              std::string(session_error_name(SessionError::io)) == "io", "status: protocol names");
    }
    // B: progress after every block, the last partial one included (a file below 16 MiB reports too), and the
    // blocking steps announced before they start
    {
        const fs::path p = dir / "progress.bin";
        std::vector<std::pair<uint64_t, uint64_t>> seen;
        std::vector<std::string> phases;
        SessionWriteOptions opt;
        opt.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
        opt.phase = [&](const char* ph, uint64_t b) { phases.push_back(std::string(ph) + ":" + std::to_string(b)); };
        // a file below one block (the sample without its K/V layers) reports too: at the start and its one block
        SavedConversation small = original;
        small.kv.clear();
        check(session_file_write(p.string(), small, id, written, error, opt), "progress: small write");
        check(written > 0 && written < (16u << 20), "progress: the small file is below one block");
        check(seen.size() == 2 && seen[0] == std::make_pair(uint64_t(0), uint64_t(written)) &&
              seen[1] == std::make_pair(uint64_t(written), uint64_t(written)), "progress: start and the partial block");
        check(phases.size() == 2 && phases[0] == "flush:" + std::to_string(written) &&
              phases[1] == "publish:" + std::to_string(written), "progress: flush and publish announced, in order");
        opt.durable = false;
        phases.clear();
        check(session_file_write(p.string(), small, id, written, error, opt) && phases.size() == 1 &&
              phases[0].rfind("publish:", 0) == 0, "progress: no flush step announced without a flush");
        // a read reports every block, the first (the header's) included
        seen.clear();
        SessionReadLimits l;
        l.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
        SavedConversation back;
        size_t n = 0;
        check(session_file_read(p.string(), id, back, n, error, l) && seen.size() == 1 &&
              seen[0] == std::make_pair(uint64_t(n), uint64_t(n)), "progress: a small read reports its one block");
        // the fixed sample: one full block and a partial one, both reported on write and on read
        seen.clear();
        opt.durable = true;
        check(session_file_write(p.string(), original, id, written, error, opt) && written > (16u << 20) &&
              written < (32u << 20), "progress: sample write");
        check(seen.size() == 3 && seen[1].first == (16u << 20) && seen[2].first == written,
              "progress: the sample's full block and its partial last block");
        seen.clear();
        check(session_file_read(p.string(), id, back, n, error, l) && seen.size() == 2 &&
              seen[0].first == (16u << 20) && seen[1].first == n, "progress: the sample's read reports both blocks");
        // a multi-block streamed write: 16 MiB blocks then the partial one, never more than a block apart
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> big(1);
        big[0].sizes = {(40u << 20) + 123, 0, 0, 0, 0};
        big[0].read = [](size_t, size_t, void* d, size_t c) { std::memset(d, 0x5a, c); return true; };
        seen.clear();
        opt.durable = true;
        check(session_file_write(p.string(), meta, big, id, written, error, opt), "progress: 40 MiB write");
        bool steps = seen.size() == 4 && seen.front().first == 0 && seen.back().first == written;
        for (size_t i = 1; i < seen.size(); ++i)
            steps = steps && seen[i].first > seen[i - 1].first && seen[i].first - seen[i - 1].first <= (16u << 20);
        check(steps, "progress: every block reported, at most 16 MiB apart, the last one at the end");
        // the allowance of a blocking step: bounded, growing with its size
        check(session_phase_limit_s(0) == 60 && session_phase_limit_s(1198691396) == 60 + (1198691396 >> 22) &&
              session_phase_limit_s(UINT64_MAX) == 3600, "phase limit: 60 s + 1 s per 4 MiB, at most an hour");
    }
    // the pipeline witness: the OS write of a full block must begin while a source read is still running.
    // A serialized copy->write loop cannot produce this: its writes happen strictly between reads, never during one.
    {
        auto ns = []() {
            return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        };
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> src(1);
        src[0].format = 2; src[0].cells = 4096; src[0].heads = 2; src[0].head_dim = 256;
        src[0].page_size = 64; src[0].pooled_rows = 0; src[0].idx_dim = 0;
        src[0].sizes = {32u << 20, 16u << 20, 0, 0, 0};
        std::vector<std::pair<uint64_t, uint64_t>> reads;   // every source-read window [begin_ns, end_ns]
        src[0].read = [&](size_t, size_t, void* d, size_t c) {
            const uint64_t t0 = ns();
            std::memset(d, 0x5a, c);
            std::this_thread::sleep_for(std::chrono::milliseconds(40));   // a device copy's worth of time
            reads.push_back({t0, ns()});
            return true;
        };
        std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> spans;   // (byte_end, begin_ns, end_ns) per full block
        std::vector<uint64_t> seen;                                    // progress reports
        SessionWriteOptions opt;
        opt.block_span = [&](uint64_t byte_end, uint64_t b, uint64_t e) { spans.emplace_back(byte_end, b, e); };
        opt.progress = [&](uint64_t done, uint64_t) { seen.push_back(done); };
        const fs::path p = dir / "pipeline.bin";
        check(session_file_write(p.string(), meta, src, id, written, error, opt), "pipeline: 48 MiB streamed write");
        check(spans.size() == 3, "pipeline: three full 16 MiB blocks written");
        // the witness is a write BEGAN while a read was still running: a serialized loop's writes always sit
        // strictly between reads (its start can never fall inside a read window), a pipelined one's do
        bool overlap = false;
        for (const auto& sp : spans)
            for (const auto& w : reads)
                if (w.first <= std::get<1>(sp) && std::get<1>(sp) < w.second) overlap = true;
        if (!overlap) {
            for (const auto& w : reads) std::fprintf(stderr, "read window [%llu..%llu]\n", w.first, w.second);
            for (const auto& sp : spans) std::fprintf(stderr, "block span end %llu: [%llu..%llu]\n",
                                                       std::get<0>(sp), std::get<1>(sp), std::get<2>(sp));
        }
        check(overlap, "pipeline: a block's OS write began during a source read");
        // the bytes stay the ordinary streamed-write bytes: a captured-image twin must write the identical file
        SavedConversation twin = original;
        twin.kv.clear();
        ConversationKv kv;
        kv.format = 2; kv.cells = 4096; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = 0; kv.idx_dim = 0;
        auto flat = [](size_t n) {
            ConversationBuffer b;
            b.resize(n);
            b.visit(0, n, [](uint8_t* q, size_t c, size_t) { std::memset(q, 0x5a, c); return true; });
            return b;
        };
        kv.k = flat(32u << 20);
        kv.v = flat(16u << 20);
        twin.kv.push_back(std::move(kv));
        const fs::path t = dir / "pipeline-twin.bin";
        check(session_file_write(t.string(), twin, id, written, error), "pipeline: captured-image twin write");
        check(slurp(p) == slurp(t), "pipeline: the streamed file is the captured-image file, byte for byte");
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(p.string(), id, back, read, error), "pipeline: the file reads back");
        check(back.kv.size() == 1 && back.kv[0].k.size() == (32u << 20) && back.kv[0].v.size() == (16u << 20),
              "pipeline: the K/V parts round trip their sizes");
        bool monotonic = !seen.empty() && seen.front() == 0;
        for (size_t i = 1; i < seen.size(); ++i) monotonic = monotonic && seen[i] > seen[i - 1];
        check(monotonic && seen.back() == written, "pipeline: progress never goes back and ends at the file size");
    }
    // a block write failing mid-pipeline: the writer latches, the producer stops, and the error/cleanup channel
    // is the same one the serialized loop used (the test completes at once: no writer to outwait)
    {
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> src(1);
        src[0].format = 2; src[0].cells = 8192; src[0].heads = 2; src[0].head_dim = 256;
        src[0].page_size = 64; src[0].pooled_rows = 0; src[0].idx_dim = 0;
        src[0].sizes = {(64u << 20) + 4096, 0, 0, 0, 0};
        src[0].read = [](size_t, size_t, void* d, size_t c) { std::memset(d, 0x33, c); return true; };
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::string(step) == "write" ? EIO : 0; };
        SessionStatus st;
        const fs::path q = dir / "writer-fault.bin";
        check(!session_file_write(q.string(), meta, src, id, written, error, opt, &st), "writer fault: the save fails");
        check(error.find("session file:") != std::string::npos && error.find("failed") != std::string::npos,
              "writer fault: the error channel is unchanged");
        check(!st.published, "writer fault: nothing was published");
        check(!fs::exists(q) && no_temp(dir), "writer fault: no file and no temporary left behind");
    }
    // writer-thread starvation (tests seam: fault step "writer_thread"): the save must still complete,
    // serialized like the un-pipelined FileSink was - and the pipeline witness must NOT appear
    {
        auto ns = []() {
            return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        };
        SavedConversation meta = original;
        meta.kv.clear();
        std::vector<SessionKvSource> src(1);
        src[0].format = 2; src[0].cells = 4096; src[0].heads = 2; src[0].head_dim = 256;
        src[0].page_size = 64; src[0].pooled_rows = 0; src[0].idx_dim = 0;
        src[0].sizes = {32u << 20, 16u << 20, 0, 0, 0};
        std::vector<std::pair<uint64_t, uint64_t>> reads;
        src[0].read = [&](size_t, size_t, void* d, size_t c) {
            const uint64_t t0 = ns();
            std::memset(d, 0x5a, c);
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            reads.push_back({t0, ns()});
            return true;
        };
        std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> spans;
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::string(step) == "writer_thread" ? EAGAIN : 0; };
        opt.block_span = [&](uint64_t byte_end, uint64_t b, uint64_t e) { spans.emplace_back(byte_end, b, e); };
        const fs::path p = dir / "writer-starved.bin";
        error.clear();
        check(session_file_write(p.string(), meta, src, id, written, error, opt), "starved writer: the save still completes");
        check(error.empty(), "starved writer: no error text");
        check(spans.size() == 3, "starved writer: all three full blocks still written");
        bool witnessed = false;
        for (const auto& sp : spans)
            for (const auto& w : reads)
                if (w.first <= std::get<1>(sp) && std::get<1>(sp) < w.second) witnessed = true;
        check(!witnessed, "starved writer: blocks drained in the caller's thread, never during a read");
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(p.string(), id, back, read, error), "starved writer: the file reads back");
    }
    // A: the SAVE's RAM preflight comes BEFORE the checkpoint is copied
    {
        std::vector<ConversationCheckpoint> chain;
        chain.push_back(checkpoint(10, 2));
        chain.push_back(checkpoint(700, 3));
        chain.back().gdn.assign(8u << 20, 0x11);   // the deepest: an 8 MiB running state
        const auto before = chain;
        SessionSaveLive live;
        live.state_bytes = 5000; live.tokens = 1000; live.images = 1; live.kv_layers = 13;
        check(session_deepest_checkpoint(chain) == &chain[1], "save: deepest chosen by reference");
        {   // pin=N: the pinned checkpoint is the one kept, even when a deeper periodic one exists
            auto pinned_chain = chain;
            pinned_chain[0].pinned = true;
            check(session_deepest_checkpoint(pinned_chain) == &pinned_chain[0], "save: a pinned shared prefix beats a deeper checkpoint");
            check(session_deepest_checkpoint(chain) == &chain[1], "save: no pin, the deepest as before");
        }
        const uint64_t need = session_save_peak_bytes(&chain[1], live);
        check(need >= 2 * (uint64_t) chain[1].bytes() + 5000 + 4000 + (32u << 20) &&
              need <= 2 * (uint64_t) chain[1].bytes() + 5000 + 4000 + (34u << 20),
              "save: the peak counts two checkpoint copies, the live state, tokens and the two pipelined buffers");
        SessionSaveLive huge = live;
        huge.state_bytes = UINT64_MAX;
        check(session_save_peak_bytes(&chain[1], huge) == UINT64_MAX, "save: the peak saturates");
        std::vector<ConversationCheckpoint> out(1);
        std::string why;
        uint64_t asked = 0;
        bool empty_when_asked = false;
        const auto refuse = [&](uint64_t n, std::string& w) { asked = n; empty_when_asked = out.empty(); w = "low RAM"; return false; };
        check(!session_save_checkpoints(chain, live, refuse, out, why) && why == "low RAM" && asked == need,
              "save: the preflight is asked for the peak and can refuse");
        check(empty_when_asked && out.empty(), "save: refused before any copy (nothing copied)");
        bool same_chain = chain.size() == before.size();
        for (size_t i = 0; same_chain && i < chain.size(); ++i) same_chain = same_checkpoint(chain[i], before[i]);
        check(same_chain, "save: a refusal leaves the live checkpoints untouched");
        const auto agree = [&](uint64_t, std::string&) { return true; };
        check(session_save_checkpoints(chain, live, agree, out, why) && out.size() == 1 &&
              same_checkpoint(out[0], chain[1]), "save: admitted, the deepest checkpoint is copied");
        check(session_save_checkpoints({}, live, agree, out, why) && out.empty(), "save: no checkpoints, none copied");
#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__) && !defined(STRATA_NO_RLIMIT_TEST)
        // under a real address-space limit that cannot hold one more copy: the refusal comes first, no bad_alloc;
        // and the same limit with an admitting preflight does fail at the copy (the test can see the difference)
        std::vector<ConversationCheckpoint> fat(1);
        fat[0].ids = {1, 2, 3};
        fat[0].gdn.assign(256u << 20, 0x22);
        const fs::path old_file = dir / "old-session.bin";
        spit(old_file, std::vector<char>(9, 'o'));
        std::fflush(nullptr);
        const pid_t pid = ::fork();
        if (pid == 0) {
            long pages = 0, rss = 0;
            if (FILE* f = std::fopen("/proc/self/statm", "r")) { if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) pages = 0; std::fclose(f); }
            struct rlimit rl {};
            rl.rlim_cur = rl.rlim_max = (rlim_t) pages * (rlim_t) ::sysconf(_SC_PAGESIZE) + (64u << 20);
            if (pages <= 0 || ::setrlimit(RLIMIT_AS, &rl) != 0) ::_exit(10);
            std::vector<ConversationCheckpoint> o2;
            std::string w;
            const auto low = [](uint64_t n, std::string& m) { m = "need " + std::to_string(n); return false; };
            bool refused = false;
            try { refused = !session_save_checkpoints(fat, live, low, o2, w) && o2.empty(); } catch (...) { ::_exit(11); }
            if (!refused) ::_exit(12);
            try { session_save_checkpoints(fat, live, agree, o2, w); } catch (const std::bad_alloc&) { ::_exit(0); }
            ::_exit(13);   // the copy fitted: the limit did not bite, the test proves nothing
        }
        int status = 0;
        check(pid > 0 && ::waitpid(pid, &status, 0) == pid, "save: low-memory child ran");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "save: under a low address-space limit the preflight refuses before the copy (an admitted copy fails)");
        check(fat[0].gdn.size() == (256u << 20) && fat[0].gdn[12345] == 0x22 &&
              slurp(old_file) == std::vector<char>(9, 'o'), "save: live state and the old file intact");
#endif
    }
    // Opt-in pressure reclamation does not change the selected file bytes, even with pin=N.
    for (bool pin : {false, true}) {
        auto source = sample();
        source.checkpoints[0].pinned = pin;
        const auto live_before = source.live;
        auto expected = sample();
        expected.checkpoints = session_checkpoints_to_save(source.checkpoints);
        const fs::path before_path = dir / "reclaim-before.bin", after_path = dir / "reclaim-after.bin";
        check(session_file_write(before_path.string(), expected, id, written, error), "reclaim: baseline file written");
        ConversationCache parked(1u << 20, 4);
        SessionSaveLive sl; sl.state_bytes = source.live.bytes(); sl.tokens = source.live.ids.size();
        const auto freed = session_save_reclaim(source.checkpoints, parked, sl, 0,
                                                [] { return std::optional<uint64_t>(0); });
        check(freed.checkpoints == 1, "reclaim: a non-selected checkpoint released");
        check(same_checkpoint(source.live, live_before), "reclaim: live state remains intact");
        check(session_file_write(after_path.string(), source, id, written, error), "reclaim: file written after pressure");
        check(slurp(before_path) == slurp(after_path), "reclaim: file byte-identical to original selection");
        const auto old = slurp(after_path);
        SessionWriteOptions opt;
        opt.fault = [](const char* step) { return std::strcmp(step, "write") == 0 ? ENOSPC : 0; };
        SessionStatus status;
        check(!session_file_write(after_path.string(), source, id, written, error, opt, &status) &&
              status.error == SessionError::storage && !status.published, "reclaim: later write failure reported");
        check(slurp(after_path) == old && same_checkpoint(source.live, live_before) && no_temp(dir),
              "reclaim: later failure preserves old file and live state, no temporary left");
    }
    // R6: the folder the free-space query is asked about
    {
        check(session_free_space_dir("\\\\server\\share\\sessions\\chat.bin", true) == "\\\\server\\share\\sessions\\",
              "free space: UNC subfolder keeps its trailing backslash");
        check(session_free_space_dir("\\\\server\\share\\chat.bin", true) == "\\\\server\\share\\",
              "free space: UNC share root keeps its trailing backslash");
        check(session_free_space_dir("//server/share/s/chat.bin", true) == "\\\\server\\share\\s\\",
              "free space: forward slashes become backslashes");
        check(session_free_space_dir("C:\\chat.bin", true) == "C:\\", "free space: drive root");
        check(session_free_space_dir("C:chat.bin", true) == "C:\\", "free space: drive-relative name");
        check(session_free_space_dir("chat.bin", true).empty(), "free space: bare name is the current folder");
        check(session_free_space_dir("/a/b/chat.bin", false) == "/a/b", "free space: POSIX folder");
        check(session_free_space_dir("/chat.bin", false) == "/", "free space: POSIX root");
        check(session_free_space_dir("chat.bin", false) == ".", "free space: POSIX bare name");
    }
#ifndef _WIN32
    // what restore will not open: a symbolic link, a hard-linked file, a FIFO, a directory
    {
        const fs::path link = dir / "link.bin";
        fs::create_symlink(good, link);
        check(rejects(link, id, "symbolic link"), "a symbolic link is not followed");
        const fs::path hard = dir / "hard.bin";
        fs::create_hard_link(dir / "again.bin", hard);
        check(rejects(hard, id, "hard link"), "a hard-linked file is refused");
        fs::remove(hard);
        check(!rejects(dir / "again.bin", id), "the same file with one name again reads");
        const fs::path fifo = dir / "fifo.bin";
        check(::mkfifo(fifo.c_str(), 0600) == 0, "mkfifo");
        check(rejects(fifo, id, "regular"), "a FIFO is refused without blocking");
        check(rejects(dir / "moved", id, "regular"), "a directory is refused");
    }
    // saving over a symbolic link replaces the link, never writes through it; the new file is the owner's only
    {
        const fs::path target = dir / "target.bin", link = dir / "out-link.bin";
        spit(target, std::vector<char>(7, 't'));
        fs::create_symlink(target, link);
        check(session_file_write(link.string(), original, id, written, error), "write at a link name");
        check(!fs::is_symlink(link) && slurp(link) == image, "the link was replaced by the file");
        check(slurp(target) == std::vector<char>(7, 't'), "the link's target is untouched");
        struct stat st {};
        check(::stat(link.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600, "the session file is mode 0600");
    }
#endif

    fs::remove_all(dir);
    std::printf("conversation_file_test: %d checks passed\n", checks);
    return 0;
}
