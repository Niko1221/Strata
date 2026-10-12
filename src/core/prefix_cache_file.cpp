#include "strata/core/prefix_cache_file.hpp"

#include <bit>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// Windows types must be declared before bcrypt.h.
// clang-format off
#include <windows.h>
#include <bcrypt.h>
// clang-format on
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {
// Portable fallback follows FIPS 180-4; Windows uses the OS SHA-256 provider.
class Sha256 {
#if defined(_WIN32) && !defined(STRATA_PREFIX_PORTABLE_SHA)
  BCRYPT_ALG_HANDLE algorithm_ = nullptr;
  BCRYPT_HASH_HANDLE hash_ = nullptr;
  static void check(NTSTATUS s) {
    if (s < 0)
      throw std::runtime_error("SHA-256 provider failed");
  }

public:
  Sha256() {
    check(BCryptOpenAlgorithmProvider(&algorithm_, BCRYPT_SHA256_ALGORITHM,
                                      nullptr, 0));
    const auto s =
        BCryptCreateHash(algorithm_, &hash_, nullptr, 0, nullptr, 0, 0);
    if (s < 0) {
      BCryptCloseAlgorithmProvider(algorithm_, 0);
      algorithm_ = nullptr;
      check(s);
    }
  }
  ~Sha256() {
    if (hash_)
      BCryptDestroyHash(hash_);
    if (algorithm_)
      BCryptCloseAlgorithmProvider(algorithm_, 0);
  }
  void update(const void *data, size_t n) {
    auto p = static_cast<const uint8_t *>(data);
    while (n) {
      const ULONG part = static_cast<ULONG>(std::min<size_t>(n, 1U << 30));
      check(BCryptHashData(hash_, const_cast<PUCHAR>(p), part, 0));
      p += part;
      n -= part;
    }
  }
  PrefixDigest finish() {
    PrefixDigest out{};
    check(BCryptFinishHash(hash_, out.data(), ULONG(out.size()), 0));
    return out;
  }
#else
  std::array<uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<uint8_t, 64> buffer_{};
  uint64_t total_ = 0;
  size_t used_ = 0;
  void block(const uint8_t *p) {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t a = w[i - 15], b = w[i - 2];
      w[i] = w[i - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) +
             w[i - 7] + (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10));
    }
    auto [a, b, c, d, e, f, g, h] = h_;
    for (int i = 0; i < 64; ++i) {
      const uint32_t t =
          h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
          ((e & f) ^ ((~e) & g)) + k[i] + w[i];
      const uint32_t u =
          (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) +
          ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t;
      d = c;
      c = b;
      b = a;
      a = t + u;
    }
    const uint32_t v[8] = {a, b, c, d, e, f, g, h};
    for (size_t i = 0; i < 8; ++i)
      h_[i] += v[i];
  }

public:
  void update(const void *data, size_t n) {
    auto p = static_cast<const uint8_t *>(data);
    total_ += n;
    while (n) {
      const size_t part = std::min(n, 64 - used_);
      std::memcpy(buffer_.data() + used_, p, part);
      p += part;
      n -= part;
      used_ += part;
      if (used_ == 64) {
        block(buffer_.data());
        used_ = 0;
      }
    }
  }
  PrefixDigest finish() {
    const uint64_t bits = total_ * 8;
    const uint8_t one = 128, zero = 0;
    update(&one, 1);
    while (used_ != 56)
      update(&zero, 1);
    uint8_t length[8];
    for (int i = 0; i < 8; ++i)
      length[i] = uint8_t(bits >> (56 - 8 * i));
    update(length, 8);
    PrefixDigest out{};
    for (size_t i = 0; i < 8; ++i)
      for (size_t j = 0; j < 4; ++j)
        out[4 * i + j] = uint8_t(h_[i] >> (24 - 8 * j));
    return out;
  }
#endif
  Sha256(const Sha256 &) = delete;
  Sha256 &operator=(const Sha256 &) = delete;
#if !defined(_WIN32) || defined(STRATA_PREFIX_PORTABLE_SHA)
  Sha256() = default;
#endif
};

constexpr char magic[8] = {'S', 'T', 'R', 'A', 'P', 'F', 'X', '1'};
// v2: the payload ends with a layer split's stage images (a count, then each
// one in the top-level image's encoding). A v1 file is rejected by version
// before any payload byte is interpreted: a clean miss, never a misread.
constexpr uint64_t format_version = 2;
constexpr size_t header_bytes = 8 + 8 + 8 + 32 + 32;
// SavedConversation::kv holds at most n_layers + 1 (the draft) layers.
constexpr uint64_t max_kv_layers = 129;
void require(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
void u64_bytes(uint64_t x, uint8_t (&b)[8]) {
  for (size_t i = 0; i < 8; ++i)
    b[i] = uint8_t(x >> (8 * i));
}
uint64_t get_u64(const uint8_t *b) {
  uint64_t x = 0;
  for (size_t i = 0; i < 8; ++i)
    x |= uint64_t(b[i]) << (8 * i);
  return x;
}
void raw_write(std::ostream &out, const void *p, size_t n) {
  if (!n)
    return;
  out.write(static_cast<const char *>(p), std::streamsize(n));
  require(bool(out), "cache write failed");
}
void raw_read(std::istream &in, void *p, size_t n) {
  if (!n)
    return;
  in.read(static_cast<char *>(p), std::streamsize(n));
  require(bool(in), "truncated cache");
}
struct Writer {
  std::ostream &out;
  Sha256 hash;
  uint64_t size = 0;
  void bytes(const void *p, size_t n) {
    require(n <= prefix_cache_max_bytes - size, "cache exceeds byte limit");
    raw_write(out, p, n);
    hash.update(p, n);
    size += n;
  }
  void number(uint64_t x) {
    uint8_t b[8];
    u64_bytes(x, b);
    bytes(b, 8);
  }
  void blob(const std::vector<uint8_t> &v) {
    number(v.size());
    bytes(v.data(), v.size());
  }
  void blob(const ConversationBuffer &v) {
    number(v.size());
    require(v.visit(0, v.size(),
                    [&](const uint8_t *p, size_t n, size_t) {
                      bytes(p, n);
                      return true;
                    }),
            "invalid segmented buffer");
  }
};
struct Reader {
  std::istream &in;
  Sha256 hash;
  uint64_t remaining;
  void bytes(void *p, size_t n) {
    require(n <= remaining, "cache field exceeds payload");
    raw_read(in, p, n);
    hash.update(p, n);
    remaining -= n;
  }
  uint64_t number() {
    uint8_t b[8];
    bytes(b, 8);
    return get_u64(b);
  }
  size_t length() {
    const uint64_t n = number();
    require(n <= remaining && n <= prefix_cache_max_bytes,
            "invalid cache field length");
    return size_t(n);
  }
  void blob(std::vector<uint8_t> &v) {
    const size_t n = length();
    v.resize(n);
    bytes(v.data(), n);
  }
  void blob(ConversationBuffer &v) {
    const size_t n = length();
    v.resize(n);
    require(v.visit(0, n,
                    [&](uint8_t *p, size_t count, size_t) {
                      bytes(p, count);
                      return true;
                    }),
            "invalid segmented field");
  }
};
void hash_number(Sha256 &hash, uint64_t n) {
  uint8_t b[8];
  u64_bytes(n, b);
  hash.update(b, 8);
}
void hash_text(Sha256 &hash, const std::string &s) {
  hash_number(hash, s.size());
  hash.update(s.data(), s.size());
}
int64_t signed_number(Reader &r) {
  const auto n = r.number();
  require(n <= uint64_t(INT64_MAX), "invalid signed cache field");
  return int64_t(n);
}
std::filesystem::path temporary_path(const std::filesystem::path &path) {
#ifdef _WIN32
  const auto pid = GetCurrentProcessId();
#else
  const auto pid = getpid();
#endif
  auto tmp = path;
  tmp += std::string(".tmp-") + std::to_string(pid) + "-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count());
  return tmp;
}

// One image (the top-level one, or a stage image) in the shared encoding.
// `top` (non-null for a stage image) adds the stage-image rules.
void write_image(Writer &w, const SavedConversation &image, int32_t next_token,
                 const SavedConversation *top) {
  require(!image.live.ids.empty() &&
              image.live.ids.size() <= prefix_cache_max_tokens &&
              next_token >= 0,
          "invalid prefix length/frontier");
  require(image.live.imgs.empty() && image.live.stage_parts.empty() &&
              image.checkpoints.empty(),
          "only one text prefix is supported");
  require(image.kv.size() <= max_kv_layers, "invalid prefix snapshot size");
  if (top) {
    require(image.stage_images.empty(), "nested stage image");
    require(image.geometry == top->geometry && image.cvec == top->cvec &&
                image.live.ids == top->live.ids,
            "stage image differs from its prefix");
  }
  for (auto n : image.geometry) {
    require(n >= 0, "negative geometry");
    w.number(uint64_t(n));
  }
  require(image.layer_lo >= 0 && image.layer_hi > image.layer_lo,
          "invalid layer carve");
  w.number(uint64_t(image.layer_lo));
  w.number(uint64_t(image.layer_hi));
  w.number(image.cvec ? 1 : 0);
  w.number(uint64_t(next_token));
  w.number(image.live.ids.size());
  for (auto id : image.live.ids) {
    require(id >= 0, "negative token");
    w.number(uint64_t(id));
  }
  for (auto *v : {&image.live.gdn, &image.live.ple, &image.live.tails,
                  &image.live.dead, &image.live.block_pos})
    w.blob(*v);
  w.number(image.kv.size());
  for (const auto &k : image.kv) {
    require(k.format >= 0, "invalid KV format");
    w.number(uint64_t(k.format));
    for (auto n : {k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows,
                   k.idx_dim}) {
      require(n >= 0, "invalid KV geometry");
      w.number(uint64_t(n));
    }
    for (auto *v : {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled})
      w.blob(*v);
  }
}

// The exact expected prefix is checked before any large field is allocated.
// `top`: the already-decoded top-level image when this is a stage image.
void read_image(Reader &r, SavedConversation &image,
                const std::vector<int32_t> &prefix, int32_t next_token,
                bool cvec, const SavedConversation *top) {
  for (auto &n : image.geometry)
    n = signed_number(r);
  if (top)
    require(image.geometry == top->geometry, "stage image geometry differs");
  image.layer_lo = signed_number(r);
  image.layer_hi = signed_number(r);
  require(image.layer_hi > image.layer_lo, "invalid layer carve");
  const auto cv = r.number();
  require(cv <= 1 && bool(cv) == cvec, "steering mode changed");
  image.cvec = bool(cv);
  require(r.number() == uint64_t(next_token), "prefix frontier changed");
  require(r.number() == prefix.size(), "prefix length changed");
  image.live.ids.reserve(prefix.size());
  for (auto expected : prefix) {
    require(expected >= 0 && r.number() == uint64_t(expected),
            "prefix tokens changed");
    image.live.ids.push_back(expected);
  }
  for (auto *v : {&image.live.gdn, &image.live.ple, &image.live.tails,
                  &image.live.dead, &image.live.block_pos})
    r.blob(*v);
  const auto layers = r.number();
  require(layers <= max_kv_layers, "invalid KV layer count");
  image.kv.resize(size_t(layers));
  for (auto &k : image.kv) {
    const auto format = r.number();
    require(format <= INT_MAX, "invalid KV format");
    k.format = int(format);
    for (auto *n : {&k.cells, &k.heads, &k.head_dim, &k.page_size,
                    &k.pooled_rows, &k.idx_dim})
      *n = signed_number(r);
    for (auto *v : {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled})
      r.blob(*v);
  }
}
} // namespace

PrefixDigest prefix_cache_digest(const void *p, size_t n) {
  Sha256 h;
  h.update(p, n);
  return h.finish();
}

bool prefix_cache_identity(const std::vector<std::filesystem::path> &assets,
                           const std::vector<std::string> &settings,
                           PrefixDigest &identity, std::string &error) {
  error.clear();
  try {
    Sha256 h;
    hash_text(h, "Strata single text prefix v1; little endian IEEE754");
    require(std::endian::native == std::endian::little,
            "prefix cache requires little endian");
    for (const auto &s : settings)
      hash_text(h, s);
    std::set<std::filesystem::path> files;
    for (const auto &asset : assets) {
      if (asset.empty())
        continue;
      const auto path = std::filesystem::canonical(asset);
      if (std::filesystem::is_directory(path)) {
        for (const auto &entry :
             std::filesystem::recursive_directory_iterator(path)) {
          require(!entry.is_symlink(), "symlink in model assets");
          if (entry.is_regular_file())
            files.insert(entry.path());
          require(files.size() <= 100000, "too many model asset files");
        }
      } else {
        require(std::filesystem::is_regular_file(path),
                "model asset is not a regular file");
        files.insert(path);
      }
    }
    require(!files.empty(), "empty model identity");
    hash_number(h, files.size());
    std::array<uint8_t, 65536> data{};
    for (const auto &p : files) {
      const auto size = std::filesystem::file_size(p);
      const auto stamp = std::filesystem::last_write_time(p);
      const auto name = p.generic_u8string();
      hash_text(h, std::string(name.begin(), name.end()));
      hash_number(h, size);
      hash_number(h, uint64_t(stamp.time_since_epoch().count()));
      std::ifstream f(p, std::ios::binary);
      require(bool(f), "cannot read model identity");
      if (size <= 1024 * 1024) {
        for (uint64_t at = 0; at < size;) {
          const auto n = size_t(std::min<uint64_t>(size - at, data.size()));
          raw_read(f, data.data(), n);
          h.update(data.data(), n);
          at += n;
        }
      } else {
        for (uint64_t at : {uint64_t(0), size / 2, size - data.size()}) {
          f.seekg(std::streamoff(at));
          require(bool(f), "cannot seek model identity");
          raw_read(f, data.data(), data.size());
          h.update(data.data(), data.size());
        }
      }
      require(size == std::filesystem::file_size(p) &&
                  stamp == std::filesystem::last_write_time(p),
              "model asset changed while fingerprinting");
    }
    identity = h.finish();
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}

bool prefix_cache_write(const std::filesystem::path &path,
                        const PrefixDigest &identity, int32_t next_token,
                        const SavedConversation &image, std::string &error) {
  error.clear();
  std::filesystem::path tmp;
  try {
    require(std::endian::native == std::endian::little,
            "unsupported cache endianness");
    require(!image.live.ids.empty() &&
                image.live.ids.size() <= prefix_cache_max_tokens &&
                next_token >= 0,
            "invalid prefix length/frontier");
    require(image.live.imgs.empty() && image.live.stage_parts.empty() &&
                image.checkpoints.empty(),
            "only one text prefix is supported");
    require(image.stage_images.size() <= prefix_cache_max_stage_images,
            "too many stage images");
    // Every image's own rules are rechecked as it is written; here only what
    // spans them: at least one K/V layer, and the whole image's byte cap.
    uint64_t kv_layers = image.kv.size();
    for (const auto &stage : image.stage_images)
      kv_layers += stage.kv.size();
    require(kv_layers >= 1 && image.kv.size() <= max_kv_layers &&
                image.bytes() <= prefix_cache_max_bytes,
            "invalid prefix snapshot size");
    if (std::filesystem::exists(path)) {
      std::ifstream previous(path, std::ios::binary);
      char mark[8];
      raw_read(previous, mark, sizeof(mark));
      require(std::memcmp(mark, magic, sizeof(mark)) == 0,
              "refusing to replace a non-prefix-cache file");
    }
    if (!path.parent_path().empty())
      std::filesystem::create_directories(path.parent_path());
    tmp = temporary_path(path);
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    require(bool(out), "cannot create temporary prefix cache");
    std::array<uint8_t, header_bytes> header{};
    std::memcpy(header.data(), magic, 8);
    uint8_t b[8];
    u64_bytes(format_version, b);
    std::memcpy(header.data() + 8, b, 8);
    std::memcpy(header.data() + 24, identity.data(), identity.size());
    raw_write(out, header.data(), header.size());
    Writer w{out};
    w.hash.update(identity.data(), identity.size());
    write_image(w, image, next_token, nullptr);
    // v2: the later stages of a layer split, in stage order, each carving the
    // layers after the previous image's (no overlap; the restore checks the
    // exact carve against each stage's session)
    w.number(image.stage_images.size());
    int64_t previous_hi = image.layer_hi;
    for (const auto &stage : image.stage_images) {
      require(stage.layer_lo >= previous_hi, "stage images out of layer order");
      write_image(w, stage, next_token, &image);
      previous_hi = stage.layer_hi;
    }
    const auto digest = w.hash.finish();
    u64_bytes(w.size, b);
    std::memcpy(header.data() + 16, b, 8);
    std::memcpy(header.data() + 56, digest.data(), digest.size());
    out.seekp(0);
    raw_write(out, header.data(), header.size());
    out.flush();
    require(bool(out), "cache flush failed");
    out.close();
    require(!out.fail(), "cache close failed");
#ifdef _WIN32
    require(MoveFileExW(tmp.c_str(), path.c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) !=
                0,
            "cache atomic replacement failed");
#else
    std::filesystem::rename(tmp, path);
#endif
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    if (!tmp.empty()) {
      std::error_code ignored;
      std::filesystem::remove(tmp, ignored);
    }
    return false;
  }
}

bool prefix_cache_read(const std::filesystem::path &path,
                       const PrefixDigest &identity,
                       const std::vector<int32_t> &prefix, int32_t next_token,
                       bool cvec, SavedConversation &out, std::string &error) {
  error.clear();
  try {
    require(std::endian::native == std::endian::little,
            "unsupported cache endianness");
    require(!prefix.empty() && prefix.size() <= prefix_cache_max_tokens,
            "invalid expected prefix");
    const auto size = std::filesystem::file_size(path);
    require(size >= header_bytes &&
                size - header_bytes <= prefix_cache_max_bytes,
            "invalid cache file size");
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open prefix cache");
    std::array<uint8_t, header_bytes> header{};
    raw_read(in, header.data(), header.size());
    require(std::memcmp(header.data(), magic, 8) == 0 &&
                get_u64(header.data() + 8) == format_version,
            "unsupported cache format/version");
    require(get_u64(header.data() + 16) == size - header_bytes,
            "invalid cache payload size");
    require(std::equal(identity.begin(), identity.end(), header.begin() + 24),
            "model/config identity changed");
    Reader r{in, {}, size - header_bytes};
    r.hash.update(identity.data(), identity.size());
    SavedConversation image;
    read_image(r, image, prefix, next_token, cvec, nullptr);
    // v2: the stage images, bounded before any is allocated; each one is
    // checked against the same expected prefix, frontier and steering mode
    const auto stages = r.number();
    require(stages <= prefix_cache_max_stage_images,
            "invalid stage image count");
    image.stage_images.resize(size_t(stages));
    uint64_t kv_layers = image.kv.size();
    int64_t previous_hi = image.layer_hi;
    for (auto &stage : image.stage_images) {
      read_image(r, stage, prefix, next_token, cvec, &image);
      require(stage.layer_lo >= previous_hi,
              "stage images out of layer order");
      previous_hi = stage.layer_hi;
      kv_layers += stage.kv.size();
    }
    require(kv_layers >= 1, "invalid KV layer count");
    require(r.remaining == 0 && in.peek() == std::char_traits<char>::eof(),
            "trailing cache data");
    const auto digest = r.hash.finish();
    require(std::equal(digest.begin(), digest.end(), header.begin() + 56),
            "cache SHA-256 mismatch");
    require(image.bytes() <= prefix_cache_max_bytes,
            "decoded cache exceeds byte limit");
    out = std::move(image);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}
} // namespace strata::core
