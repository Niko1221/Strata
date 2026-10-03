// CPU-only tests for the on-disk session format (include/strata/core/conversation_file.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"

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
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
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

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
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

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used && b.stage_parts.empty();
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
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0 && n != "planted.bin.tmp") return false;
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
        set_env("STRATA_SESSION_BUFFERED", "1");
        const fs::path p = dir / "buffered.bin";
        check(session_file_write(p.string(), original, id, written, error), "buffered write");
        check(slurp(p) == slurp(good), "buffered write equals the default write");
        SavedConversation back;
        size_t read = 0;
        check(session_file_read(p.string(), id, back, read, error) && same(original, back), "buffered read");
        set_env("STRATA_SESSION_BUFFERED", nullptr);
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
        uint32_t v = 2; std::memcpy(d.data() + 8, &v, 4);
        const uint64_t h = session_hash64(d.data(), 56, 0); std::memcpy(d.data() + 56, &h, 8);
        const fs::path p = dir / "ver.bin"; spit(p, d);
        check(rejects(p, id, "version"), "unknown version rejected");
    }
    // a corrupted element count with a recomputed payload hash must fail on bounds, not allocate
    {
        auto d = image;
        // payload starts at 64: geometry 18*8, layer_lo, layer_hi, cvec = 21*8 bytes, then live.ids count
        const size_t at = 64 + 21 * 8;
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
    // layer-split images are not written; no file appears
    {
        SavedConversation split = sample();
        split.live.stage_parts.push_back(checkpoint(1, 9));
        const fs::path p = dir / "split.bin";
        check(!session_file_write(p.string(), split, id, written, error), "layer-split image refused");
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
        check(session_model_fingerprint({{"a", m1.string()}}, f2, error), "fingerprint subset ok");
        check(session_model_fingerprint({{"b", m1.string()}, {"a", m2.string()}}, f5, error), "fingerprint roles ok");
        check(f1 != f5, "fingerprint follows the role each file plays");
        // a moved folder: same roles, same bytes, other paths
        const fs::path moved = dir / "moved";
        fs::create_directories(moved);
        fs::copy_file(m1, moved / "m1.bin"); fs::copy_file(m2, moved / "m2.bin");
        check(session_model_fingerprint({{"a", (moved / "m1.bin").string()}, {"b", (moved / "m2.bin").string()}}, f6,
                                        error) && f6 == f1, "fingerprint does not follow the folder");
        // an optional input that is absent is recorded, not an error; a present one changes the fingerprint
        check(session_model_fingerprint({{"a", m1.string()}, {"b", m2.string()}, {"c", (dir / "nothere").string(), true}},
                                        f7, error), "absent optional input ok");
        check(session_model_fingerprint({{"a", m1.string()}, {"b", m2.string()}, {"c", m2.string(), true}}, f8, error) &&
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
        check(rejects(good, id, "no room", l) && asked == image.size(), "admission sees the file size and can refuse");
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
    // a failed write over an existing session keeps it, and touches no other file (a planted name of the old
    // fixed temporary pattern included)
    {
        const fs::path p = dir / "keep.bin", planted = dir / "planted.bin.tmp";
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
