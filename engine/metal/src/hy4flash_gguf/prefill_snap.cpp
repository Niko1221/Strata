// engine/metal/src/hy4flash_gguf/prefill_snap.cpp - see prefill_snap.h.
#include "prefill_snap.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace strata::hy4_flash {

namespace {

// ---- SHA-256 (FIPS 180-4).  Self-contained; no engine dependency has one. ------------------------------------
struct Sha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint8_t buf[64] = {};
    size_t n = 0;          // bytes in buf
    uint64_t total = 0;    // bytes absorbed

    static uint32_t ror(uint32_t x, int s) { return (x >> s) | (x << (32 - s)); }

    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
            0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
            0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
            0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
            0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
            0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t) p[4 * i] << 24) | ((uint32_t) p[4 * i + 1] << 16) | ((uint32_t) p[4 * i + 2] << 8) |
                   (uint32_t) p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void* data, size_t len) {
        const uint8_t* p = (const uint8_t*) data;
        total += len;
        while (len > 0) {
            const size_t take = len < 64 - n ? len : 64 - n;
            std::memcpy(buf + n, p, take);
            n += take; p += take; len -= take;
            if (n == 64) { block(buf); n = 0; }
        }
    }

    std::string hex() {
        const uint64_t bits = total * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        pad = 0;
        while (n != 56) update(&pad, 1);
        uint8_t be[8];
        for (int i = 0; i < 8; ++i) be[i] = (uint8_t) (bits >> (56 - 8 * i));
        // the length is absorbed WITHOUT counting towards `total`, which `update` would do
        std::memcpy(buf + 56, be, 8);
        block(buf);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

const char kMagic[12] = {'H', 'Y', '4', 'P', 'F', 'S', 'N', 'A', 'P', '0', '2', '\0'};

bool wr(std::FILE* f, const void* p, size_t n) { return n == 0 || std::fwrite(p, 1, n, f) == n; }
bool rd(std::FILE* f, void* p, size_t n) { return n == 0 || std::fread(p, 1, n, f) == n; }

bool wr_str(std::FILE* f, const std::string& s) {
    const uint32_t n = (uint32_t) s.size();
    return wr(f, &n, 4) && wr(f, s.data(), n);
}
bool rd_str(std::FILE* f, std::string& s) {
    uint32_t n = 0;
    if (!rd(f, &n, 4) || n > (1u << 20)) return false;
    s.assign(n, '\0');
    return rd(f, s.data(), n);
}

/// The binding is written AS ITS CANONICAL BLOB, and the payload digest is taken over that blob followed by
/// the regions.  Two consequences, both deliberate: a field left outside the digest could be edited while the
/// digest still matched - and `last_tok` is the FIRST token a continuation emits, so that is not academic - and
/// a struct's padding is not defined, so a memcpy of the struct would compare garbage between differently-
/// padded builds.
bool wr_binding(std::FILE* f, const SnapBinding& b) { return wr_str(f, b.blob()); }

bool rd_binding(std::FILE* f, SnapBinding& b, std::string& blob) {
    if (!rd_str(f, blob)) return false;
    size_t at = 0;
    auto raw = [&](void* p, size_t n) {
        if (at + n > blob.size()) return false;
        std::memcpy(p, blob.data() + at, n);
        at += n;
        return true;
    };
    auto str = [&](std::string& v) {
        uint32_t n = 0;
        if (!raw(&n, 4) || at + n > blob.size()) return false;
        v.assign(blob, at, n);
        at += n;
        return true;
    };
    if (!(raw(&b.model_bytes, 8) && raw(&b.model_mtime, 8) && raw(&b.model_mtime_ns, 8) && str(b.model_name) &&
          str(b.model_edges_sha) && raw(&b.engram_bytes, 8) && raw(&b.engram_mtime, 8) &&
          raw(&b.engram_mtime_ns, 8) && str(b.token_map_sha) && str(b.mode) && str(b.tokens_sha) &&
          raw(&b.n_tokens, 4) && raw(&b.context, 4) && raw(&b.chunk, 4) && raw(&b.blocks, 4) &&
          raw(&b.kv_w, 4) && raw(&b.idx_blocks, 4) && raw(&b.idx_dim, 4) && raw(&b.vocab, 4) &&
          raw(&b.used, 4) && raw(&b.mtp_rows, 4) && raw(&b.last_tok, 4) && raw(&b.bad_reported, 4)))
        return false;
    // Trailing bytes mean the writer had fields this reader does not: refuse rather than guess.
    return at == blob.size();
}

}  // namespace

std::string sha256_hex(const void* p, size_t n) {
    Sha256 s;
    s.update(p, n);
    return s.hex();
}

std::string SnapBinding::blob() const {
    std::string b;
    auto raw = [&](const void* p, size_t n) { b.append((const char*) p, n); };
    auto str = [&](const std::string& v) {
        const uint32_t n = (uint32_t) v.size();
        raw(&n, 4);
        b.append(v);
    };
    raw(&model_bytes, 8);
    raw(&model_mtime, 8);
    raw(&model_mtime_ns, 8);
    str(model_name);
    str(model_edges_sha);
    raw(&engram_bytes, 8);
    raw(&engram_mtime, 8);
    raw(&engram_mtime_ns, 8);
    str(token_map_sha);
    str(mode);
    str(tokens_sha);
    raw(&n_tokens, 4);
    raw(&context, 4);
    raw(&chunk, 4);
    raw(&blocks, 4);
    raw(&kv_w, 4);
    raw(&idx_blocks, 4);
    raw(&idx_dim, 4);
    raw(&vocab, 4);
    raw(&used, 4);
    raw(&mtp_rows, 4);
    raw(&last_tok, 4);
    raw(&bad_reported, 4);
    return b;
}

std::string SnapBinding::differs_from(const SnapBinding& o) const {
    auto i = [](const char* what, long long a, long long b) {
        return std::string(what) + " " + std::to_string(b) + ", the snapshot has " + std::to_string(a);
    };
    // The model first: a different model with the same prompt is the mistake that would otherwise produce
    // plausible tokens from the wrong weights.
    if (model_bytes != o.model_bytes) return i("the model is", (long long) model_bytes, (long long) o.model_bytes) + " bytes";
    if (model_mtime != o.model_mtime) return i("the model's mtime is", (long long) model_mtime, (long long) o.model_mtime);
    if (model_mtime_ns != o.model_mtime_ns)
        return i("the model's mtime nanoseconds are", (long long) model_mtime_ns, (long long) o.model_mtime_ns);
    if (model_name != o.model_name) return "the model file " + model_name + ", and this run has " + o.model_name;
    if (model_edges_sha != o.model_edges_sha)
        return "a model whose first and last MB hash to " + model_edges_sha.substr(0, 16) +
               "..., and this run's hash to " + o.model_edges_sha.substr(0, 16) + "...";
    if (engram_bytes != o.engram_bytes)
        return i("the Engram shard is", (long long) engram_bytes, (long long) o.engram_bytes) + " bytes";
    if (engram_mtime != o.engram_mtime || engram_mtime_ns != o.engram_mtime_ns)
        return i("the Engram shard's mtime is", (long long) engram_mtime, (long long) o.engram_mtime);
    if (token_map_sha != o.token_map_sha)
        return "a token map hashing to " + token_map_sha.substr(0, 16) + "..., and this run's hashes to " +
               o.token_map_sha.substr(0, 16) + "...";
    // A snapshot taken under a deliberate-wrongness switch must not load into a normal run: that run prints no
    // warning, its own switches being unset, so the mutated prompt state would pass for a clean one.
    if (mode != o.mode)
        return "the deliberate-wrongness switches [" + (mode.empty() ? std::string("none") : mode) +
               "], and this run has [" + (o.mode.empty() ? std::string("none") : o.mode) + "]";
    if (tokens_sha != o.tokens_sha)
        return "a different prompt: sha " + o.tokens_sha.substr(0, 16) + "..., the snapshot has " +
               tokens_sha.substr(0, 16) + "...";
    if (n_tokens != o.n_tokens) return i("the prompt is", n_tokens, o.n_tokens) + " tokens";
    // The context is the cache STRIDE, so a mismatch makes every block's offset wrong - it cannot be tolerated
    // even though the used prefix would be the same.
    if (context != o.context) return i("--context is", context, o.context);
    if (chunk != o.chunk) return i("the prefill chunk is", chunk, o.chunk);
    if (blocks != o.blocks) return i("the model has", blocks, o.blocks) + " blocks";
    if (kv_w != o.kv_w) return i("kv_a_width is", kv_w, o.kv_w);
    if (idx_blocks != o.idx_blocks) return i("the indexer covers", idx_blocks, o.idx_blocks) + " blocks";
    if (idx_dim != o.idx_dim) return i("idx_key_dim is", idx_dim, o.idx_dim);
    if (vocab != o.vocab) return i("the vocabulary is", vocab, o.vocab);
    if (mtp_rows != o.mtp_rows)
        return mtp_rows == 0 ? "no MTP hidden history (it was taken without --mtp), and this run needs " +
                                   std::to_string(o.mtp_rows) + " rows"
                             : std::to_string(mtp_rows) + " rows of MTP hidden history, and this run wants " +
                                   std::to_string(o.mtp_rows);
    return {};
}

std::string snap_save(const std::string& path, const SnapBinding& b, const std::vector<SnapSlice>& slices) {
    // THE DIGEST COVERS THE BINDING TOO, not only the regions: `last_tok` and `bad_reported` live in the
    // binding and `last_tok` is the first token a continuation emits, so a field outside the digest would be
    // editable with the digest still matching.
    Sha256 d;
    const std::string blob = b.blob();
    d.update(blob.data(), blob.size());
    for (const SnapSlice& s : slices) {
        if (s.ptr == nullptr && s.bytes != 0) return "prefill snapshot: region " + s.name + " has no address";
        d.update(s.name.data(), s.name.size());
        d.update(&s.bytes, sizeof s.bytes);
        d.update(s.ptr, (size_t) s.bytes);
    }
    const std::string digest = d.hex();

    // Written to a PRIVATE temporary and renamed.  Private because two saves to the same path would otherwise
    // open and truncate the same `.part` inode, and one could rename it while the other still held it open -
    // publishing a zero-byte file that the other then kept writing into.  rename(2) within one directory is
    // atomic, so an interrupted save leaves no file whose header passes.
    const std::string tmp = path + ".part." + std::to_string((long) ::getpid());
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) return "prefill snapshot: cannot write " + tmp + ": " + std::strerror(errno);
    const uint32_t n_slices = (uint32_t) slices.size();
    bool ok = wr(f, kMagic, sizeof kMagic) && wr_binding(f, b) && wr_str(f, digest) && wr(f, &n_slices, 4);
    for (const SnapSlice& s : slices) ok = ok && wr_str(f, s.name) && wr(f, &s.bytes, 8);
    for (const SnapSlice& s : slices) ok = ok && wr(f, s.ptr, (size_t) s.bytes);
    if (ok) ok = std::fflush(f) == 0;
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) {
        std::remove(tmp.c_str());
        return "prefill snapshot: writing " + tmp + " failed: " + std::strerror(errno);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return "prefill snapshot: cannot rename into " + path + ": " + std::strerror(errno);
    }
    return {};
}

std::string snap_load(const std::string& path, const SnapBinding& want, SnapBinding& got,
                      const std::vector<SnapSlice>& slices) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return "prefill snapshot: cannot read " + path + ": " + std::strerror(errno);
    bool installed = false;   // once true, the live buffers hold bytes from the file and are not the old state
    auto fail = [&](const std::string& m) {
        std::fclose(f);
        return "prefill snapshot " + path + ": " + m +
               (installed ? " - THE CACHES NOW HOLD PART OF THIS FILE AND MUST NOT BE USED: the caller has to "
                            "stop, not continue from them"
                          : "");
    };
    char magic[sizeof kMagic] = {};
    if (!rd(f, magic, sizeof magic)) return fail("too short to hold a header");
    if (std::memcmp(magic, kMagic, sizeof kMagic) != 0)
        return fail("not a snapshot of this format (expected " + std::string(kMagic) + ")");
    std::string blob;
    if (!rd_binding(f, got, blob)) return fail("the header is truncated or has fields this engine does not");
    std::string digest;
    uint32_t n_slices = 0;
    if (!rd_str(f, digest) || !rd(f, &n_slices, 4)) return fail("the header is truncated");
    if (const std::string d = got.differs_from(want); !d.empty()) return fail("it was taken with " + d);
    if (n_slices != slices.size())
        return fail("it holds " + std::to_string(n_slices) + " regions, this engine saves " +
                    std::to_string(slices.size()));
    for (size_t i = 0; i < slices.size(); ++i) {
        std::string name;
        uint64_t bytes = 0;
        if (!rd_str(f, name) || !rd(f, &bytes, 8)) return fail("the region table is truncated");
        if (name != slices[i].name)
            return fail("region " + std::to_string(i) + " is \"" + name + "\", this engine expects \"" +
                        slices[i].name + "\"");
        if (bytes != slices[i].bytes)
            return fail("region \"" + name + "\" is " + std::to_string(bytes) + " bytes, this engine needs " +
                        std::to_string(slices[i].bytes));
    }
    // ONE PASS: read straight into the live buffers and hash WHAT WAS INSTALLED.  Hashing in one pass and
    // re-reading in a second would leave a window in which the file could change between them, and a
    // modification there would be accepted because the second pass hashed nothing.  Hashing what is installed
    // closes that: the bytes being checked are the bytes now in memory.
    //
    // The cost is that a failure leaves the caches holding part of the file.  That is made explicit rather
    // than hidden: `installed` puts it in the message, `used` and `last_tok` are not updated by the caller on
    // an error, and the engine's own token check refuses the -1 that `last_token()` then returns.  Staging the
    // payload in memory would preserve the old state instead, at a second copy of it - half a gigabyte at 8K
    // context, a gigabyte at 16K - and the old state is not a state anyone wants to continue from either.
    Sha256 d;
    d.update(blob.data(), blob.size());
    for (const SnapSlice& s : slices) {
        d.update(s.name.data(), s.name.size());
        d.update(&s.bytes, sizeof s.bytes);
        if (s.bytes == 0) continue;
        installed = true;
        if (!rd(f, s.ptr, (size_t) s.bytes)) return fail("the payload ends inside region \"" + s.name + "\"");
        d.update(s.ptr, (size_t) s.bytes);
    }
    uint8_t extra = 0;
    if (std::fread(&extra, 1, 1, f) == 1) return fail("there are bytes after the payload");
    if (const std::string have = d.hex(); have != digest)
        return fail("the sha256 of the binding and payload is " + have.substr(0, 16) + "..., the header says " +
                    digest.substr(0, 16) + "... - the file is corrupt, was edited, or changed while being read");
    std::fclose(f);
    return {};
}

}  // namespace strata::hy4_flash
