#include "strata/core/prefix_cache_file.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

std::string hex(const PrefixDigest &digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

struct TempDirectory {
  fs::path path;
  TempDirectory() {
    const auto root = fs::current_path();
    for (unsigned attempt = 0; attempt < 32; ++attempt) {
      const auto stamp =
          std::chrono::steady_clock::now().time_since_epoch().count();
      auto candidate =
          root / ("prefix-cache-file-test-" + std::to_string(stamp) + "-" +
                  std::to_string(attempt));
      std::error_code ec;
      if (fs::create_directory(candidate, ec)) {
        path = std::move(candidate);
        return;
      }
      if (ec && ec != std::errc::file_exists) {
        std::fprintf(stderr, "cannot create test directory: %s\n",
                     ec.message().c_str());
        std::exit(1);
      }
    }
    std::fprintf(stderr, "cannot allocate a unique test directory\n");
    std::exit(1);
  }
  ~TempDirectory() {
    std::error_code ec;
    fs::remove_all(path, ec);
    if (ec)
      std::fprintf(stderr, "warning: test directory cleanup failed: %s\n",
                   ec.message().c_str());
  }
};

void fill_pattern(ConversationBuffer &buffer, uint8_t salt);

SavedConversation make_image() {
  SavedConversation image;
  image.geometry = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18};
  image.layer_lo = 2;
  image.layer_hi = 5;
  image.cvec = true;
  image.live.ids = {17, 23, 41, 59};
  image.live.gdn = {1, 2, 3};
  image.live.ple = {4, 5};
  image.live.tails = {6};
  image.live.dead = {7, 8, 9};
  image.live.block_pos = {10, 11};
  ConversationKv kv;
  kv.format = 3;
  kv.cells = 64;
  kv.heads = 8;
  kv.head_dim = 128;
  kv.page_size = 16;
  kv.pooled_rows = 12;
  kv.idx_dim = 32;
  kv.k.resize(ConversationBuffer::segment_bytes + 257);
  kv.v.resize(37);
  kv.k_scale = {12, 13, 14};
  kv.v_scale = {15, 16};
  kv.pooled.resize(19);
  fill_pattern(kv.k, 29);
  fill_pattern(kv.v, 47);
  fill_pattern(kv.pooled, 83);
  image.kv.push_back(std::move(kv));
  return image;
}

void fill_pattern(ConversationBuffer &buffer, uint8_t salt) {
  const bool ok = buffer.visit(
      0, buffer.size(), [=](uint8_t *data, size_t size, size_t offset) {
        for (size_t i = 0; i < size; ++i)
          data[i] = static_cast<uint8_t>(salt + (offset + i) * 17 +
                                         (offset + i) / 251);
        return true;
      });
  check(ok, "fill segmented conversation buffer");
}

// A later stage's image of `top`'s prefix: its own carve [lo, hi), running
// state and K/V layers (the last stage of a split also holds the draft's).
SavedConversation make_stage_image(const SavedConversation &top, int64_t lo,
                                   int64_t hi, size_t layers, uint8_t salt) {
  SavedConversation stage;
  stage.geometry = top.geometry;
  stage.layer_lo = lo;
  stage.layer_hi = hi;
  stage.cvec = top.cvec;
  stage.live.ids = top.live.ids;
  stage.live.gdn = {uint8_t(salt), uint8_t(salt + 1), uint8_t(salt + 2)};
  stage.live.ple = {uint8_t(salt + 3)};
  stage.live.dead = {uint8_t(salt + 4), uint8_t(salt + 5)};
  stage.live.block_pos = {uint8_t(salt + 6)};
  for (size_t i = 0; i < layers; ++i) {
    const bool draft = i + 1 == layers && layers > 1; // no indexer
    ConversationKv kv;
    kv.format = 3;
    kv.cells = 64;
    kv.heads = 8;
    kv.head_dim = 128;
    kv.page_size = 16;
    kv.pooled_rows = draft ? 0 : 12;
    kv.idx_dim = 32;
    kv.k.resize(41 + i);
    kv.v.resize(23 + i);
    kv.k_scale = {uint8_t(salt + 7), uint8_t(i)};
    kv.pooled.resize(draft ? 0 : 19);
    fill_pattern(kv.k, uint8_t(salt + 11 * i));
    fill_pattern(kv.v, uint8_t(salt + 13 * i + 1));
    fill_pattern(kv.pooled, uint8_t(salt + 17 * i + 2));
    stage.kv.push_back(std::move(kv));
  }
  return stage;
}

// Two later stages, as a three-GPU split parks them: [5, 9) and [9, 12),
// the last with an extra (draft) K/V layer.
SavedConversation make_split_image() {
  SavedConversation image = make_image();
  image.stage_images.push_back(
      make_stage_image(image, image.layer_hi, 9, 1, 101));
  image.stage_images.push_back(make_stage_image(image, 9, 12, 2, 151));
  return image;
}

bool equal_image(const SavedConversation &a, const SavedConversation &b) {
  if (a.geometry != b.geometry || a.layer_lo != b.layer_lo ||
      a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
      a.live.ids != b.live.ids || a.live.imgs != b.live.imgs ||
      a.live.gdn != b.live.gdn || a.live.ple != b.live.ple ||
      a.live.tails != b.live.tails || a.live.dead != b.live.dead ||
      a.live.block_pos != b.live.block_pos ||
      a.live.stage_parts.size() != b.live.stage_parts.size() ||
      a.checkpoints.size() != b.checkpoints.size() ||
      a.kv.size() != b.kv.size() ||
      a.stage_images.size() != b.stage_images.size())
    return false;
  for (size_t i = 0; i < a.stage_images.size(); ++i)
    if (!equal_image(a.stage_images[i], b.stage_images[i]))
      return false;
  for (size_t i = 0; i < a.kv.size(); ++i) {
    const auto &x = a.kv[i];
    const auto &y = b.kv[i];
    if (x.format != y.format || x.cells != y.cells || x.heads != y.heads ||
        x.head_dim != y.head_dim || x.page_size != y.page_size ||
        x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
        !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) ||
        !(x.v_scale == y.v_scale) || !(x.pooled == y.pooled))
      return false;
  }
  return true;
}

std::vector<uint8_t> read_file(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>());
}
void write_file(const fs::path &path, const std::vector<uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  check(bool(output), "open mutation fixture");
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  check(bool(output), "write mutation fixture");
}
void put_u64_le(std::vector<uint8_t> &data, size_t offset, uint64_t value) {
  check(offset <= data.size() && data.size() - offset >= 8,
        "u64 mutation offset in bounds");
  for (size_t i = 0; i < 8; ++i)
    data[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}

void preserved_failure(const fs::path &path, const PrefixDigest &identity,
                       const std::vector<int32_t> &prefix, int32_t frontier,
                       bool cvec, const SavedConversation &sentinel,
                       const char *message) {
  SavedConversation out = sentinel;
  std::string error;
  check(!prefix_cache_read(path, identity, prefix, frontier, cvec, out, error),
        message);
  check(!error.empty(), "failed read reports a diagnostic");
  check(equal_image(out, sentinel),
        "failed read leaves caller output unchanged");
}

uint64_t get_u64_le(const std::vector<uint8_t> &data, size_t offset) {
  check(offset <= data.size() && data.size() - offset >= 8,
        "u64 read offset in bounds");
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i)
    value |= uint64_t(data[offset + i]) << (i * 8);
  return value;
}

// The diagnostic of a read that must fail and leave `out` untouched.
std::string failure_reason(const fs::path &path, const PrefixDigest &identity,
                           const std::vector<int32_t> &prefix,
                           int32_t frontier, bool cvec,
                           const SavedConversation &sentinel,
                           const char *message) {
  SavedConversation out = sentinel;
  std::string error;
  check(!prefix_cache_read(path, identity, prefix, frontier, cvec, out, error),
        message);
  check(!error.empty(), "failed read reports a diagnostic");
  check(equal_image(out, sentinel),
        "failed read leaves caller output unchanged");
  return error;
}
} // namespace

int main() {
  check(hex(prefix_cache_digest(nullptr, 0)) ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 empty vector");
  const char abc[] = "abc";
  check(hex(prefix_cache_digest(abc, 3)) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc vector");
  const char multi[] =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  check(hex(prefix_cache_digest(multi, sizeof(multi) - 1)) ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "SHA-256 multiblock vector");

  TempDirectory temp;
  const fs::path weights = temp.path / "weights.bin";
  {
    std::ofstream file(weights, std::ios::binary);
    const char contents[] = "prefix-cache model bytes v1";
    file.write(contents, sizeof(contents) - 1);
    check(bool(file), "create identity asset");
  }
  const std::vector<std::string> settings = {"format=iq3", "context=262144",
                                             "prompt-hash=abc"};
  PrefixDigest identity{};
  std::string error;
  check(prefix_cache_identity({weights}, settings, identity, error),
        "compute model/config identity");
  check(error.empty(), "identity success clears diagnostic");
  PrefixDigest config_identity{};
  check(prefix_cache_identity(
            {weights}, {"format=iq3", "context=262144", "prompt-hash=def"},
            config_identity, error),
        "compute changed-prompt identity");
  check(config_identity != identity,
        "prompt/configuration participates in identity");
  PrefixDigest asset_identity{};
  {
    std::ofstream file(weights, std::ios::binary | std::ios::trunc);
    const char contents[] = "prefix-cache model bytes v2";
    file.write(contents, sizeof(contents) - 1);
  }
  check(prefix_cache_identity({weights}, settings, asset_identity, error),
        "recompute changed-asset identity");
  check(asset_identity != identity,
        "model asset content participates in identity");
  {
    std::ofstream file(weights, std::ios::binary | std::ios::trunc);
    const char contents[] = "prefix-cache model bytes v1";
    file.write(contents, sizeof(contents) - 1);
  }
  check(prefix_cache_identity({weights}, settings, identity, error),
        "restore original identity");

  const fs::path cache = temp.path / "saved-prefix.bin";
  const SavedConversation source = make_image();
  const std::vector<int32_t> prefix = source.live.ids;
  constexpr int32_t frontier = 101;
  check(prefix_cache_write(cache, identity, frontier, source, error),
        "write minimal text prefix cache");
  check(error.empty(), "write success clears diagnostic");
  SavedConversation restored;
  check(prefix_cache_read(cache, identity, prefix, frontier, source.cvec,
                          restored, error),
        "read valid cache");
  check(error.empty() && equal_image(source, restored),
        "roundtrip preserves metadata and all segmented KV bytes");
  check(restored.kv[0].k.size() > ConversationBuffer::segment_bytes,
        "roundtrip exercised multiple KV segments");

  const SavedConversation sentinel = make_image();
  const auto valid_bytes = read_file(cache);
  check(valid_bytes.size() > 88 + 208,
        "serialized fixture includes fixed header and first field");
  check(valid_bytes.size() - 88 <= prefix_cache_max_bytes,
        "valid fixture below file cap");
  check(valid_bytes[0] == 'S' && valid_bytes[1] == 'T',
        "serialized fixture magic sanity");

  check(!prefix_cache_read(cache, config_identity, prefix, frontier,
                           source.cvec, restored, error) &&
            equal_image(restored, source),
        "identity mismatch rejects without replacing output");
  check(!prefix_cache_read(cache, asset_identity, prefix, frontier, source.cvec,
                           restored, error) &&
            equal_image(restored, source),
        "changed model asset identity rejects without replacing output");
  auto changed_prefix = prefix;
  changed_prefix[1]++;
  check(!prefix_cache_read(cache, identity, changed_prefix, frontier,
                           source.cvec, restored, error) &&
            equal_image(restored, source),
        "prompt token mismatch rejects without replacing output");
  check(!prefix_cache_read(cache, identity, prefix, frontier + 1, source.cvec,
                           restored, error) &&
            equal_image(restored, source),
        "next-token frontier mismatch rejects without replacing output");
  check(!prefix_cache_read(cache, identity, prefix, frontier, false, restored,
                           error) &&
            equal_image(restored, source),
        "cvec mode mismatch rejects without replacing output");

  const auto reject_mutation = [&](const char *name, auto mutate) {
    auto bytes = valid_bytes;
    mutate(bytes);
    const fs::path bad = temp.path / (std::string(name) + ".bin");
    write_file(bad, bytes);
    preserved_failure(bad, identity, prefix, frontier, source.cvec, sentinel,
                      name);
  };
  reject_mutation("bad-magic", [](auto &b) { b[0] ^= 0x20; });
  reject_mutation("bad-version", [](auto &b) { b[8] ^= 1; });
  reject_mutation("truncated", [](auto &b) { b.pop_back(); });
  reject_mutation("bad-checksum", [](auto &b) { b[56] ^= 0x80; });
  reject_mutation("trailing-data", [](auto &b) {
    b.push_back(0x5a);
    put_u64_le(b, 16, static_cast<uint64_t>(b.size() - 88));
  });
  reject_mutation("oversized-field-length", [](auto &b) {
    // Still below the global byte cap, but larger than all remaining file
    // bytes.
    put_u64_le(b, 88 + 216, static_cast<uint64_t>(b.size()));
  });
  reject_mutation("field-over-512mib", [](auto &b) {
    // Header(88) + 18 geometry + layer bounds + cvec/frontier + token count +
    // IDs.
    put_u64_le(b, 88 + 216, prefix_cache_max_bytes + 1);
  });

  const fs::path atomic = temp.path / "atomic-preserve.bin";
  check(prefix_cache_write(atomic, identity, frontier, source, error),
        "seed atomic-write destination");
  const auto before_invalid_write = read_file(atomic);
  auto invalid = source;
  invalid.live.ids[0] = -1; // Fail after the temporary file has already
                            // received part of the payload.
  check(!prefix_cache_write(atomic, identity, frontier, invalid, error),
        "reject invalid token during atomic write");
  check(read_file(atomic) == before_invalid_write,
        "invalid write preserves prior file byte-for-byte");
  for (const auto &entry : fs::directory_iterator(temp.path))
    check(entry.path().filename().string().find(".tmp-") == std::string::npos,
          "atomic writer leaves no temp file");

  auto too_many_tokens = source;
  too_many_tokens.live.ids.resize(prefix_cache_max_tokens + 1, 1);
  const fs::path too_many = temp.path / "too-many-tokens.bin";
  check(
      !prefix_cache_write(too_many, identity, frontier, too_many_tokens, error),
      "reject token count above the cap");
  check(!fs::exists(too_many), "oversized token write publishes no file");
  check(!prefix_cache_read(cache, identity,
                           std::vector<int32_t>(prefix_cache_max_tokens + 1, 1),
                           frontier, source.cvec, restored, error) &&
            equal_image(restored, source),
        "reject expected prefix above the cap without changing output");

  // ---- v2: version, layer-split stage images, limits
  reject_mutation("version-1", [](auto &b) { put_u64_le(b, 8, 1); });
  {
    auto v1 = valid_bytes;
    put_u64_le(v1, 8, 1);
    const fs::path old_file = temp.path / "version-1-reason.bin";
    write_file(old_file, v1);
    check(failure_reason(old_file, identity, prefix, frontier, source.cvec,
                         sentinel, "v1 file is a miss") ==
              "unsupported cache format/version",
          "v1 file rejected by version, before any payload byte");
  }
  // Without a split the payload ends with a zero stage count; the top-level
  // encoding is the same with stages, so this is the count's offset in both.
  const size_t stage_count_at = valid_bytes.size() - 8;
  check(get_u64_le(valid_bytes, stage_count_at) == 0,
        "one-GPU file ends with a zero stage count");

  const fs::path split_cache = temp.path / "split-prefix.bin";
  const SavedConversation split = make_split_image();
  check(prefix_cache_write(split_cache, identity, frontier, split, error),
        "write a two-stage split prefix");
  check(error.empty(), "split write success clears diagnostic");
  SavedConversation split_restored;
  check(prefix_cache_read(split_cache, identity, prefix, frontier, split.cvec,
                          split_restored, error),
        "read a two-stage split prefix");
  check(error.empty() && equal_image(split, split_restored),
        "split roundtrip preserves every stage image byte for byte");
  check(split_restored.stage_images.size() == 2 &&
            split_restored.stage_images[1].kv.size() == 2 &&
            split_restored.stage_images[0].live.ids == prefix,
        "split roundtrip keeps stage order, draft layer and prefix");
  const auto split_bytes = read_file(split_cache);
  check(get_u64_le(split_bytes, stage_count_at) == 2,
        "split file carries its stage count after the top-level image");

  const auto reject_split_mutation = [&](const char *name, auto mutate) {
    auto bytes = split_bytes;
    mutate(bytes);
    const fs::path bad = temp.path / (std::string(name) + ".bin");
    write_file(bad, bytes);
    return failure_reason(bad, identity, prefix, frontier, split.cvec,
                          sentinel, name);
  };
  check(reject_split_mutation("stage-count-over-cap",
                              [&](auto &b) {
                                put_u64_le(b, stage_count_at,
                                           prefix_cache_max_stage_images + 1);
                              }) == "invalid stage image count",
        "forged stage count above the cap rejected before allocating");
  check(reject_split_mutation("stage-count-huge",
                              [&](auto &b) {
                                put_u64_le(b, stage_count_at, ~uint64_t(0));
                              }) == "invalid stage image count",
        "forged 2^64-1 stage count rejected before allocating");
  reject_split_mutation("stage-count-more-than-present", [&](auto &b) {
    put_u64_le(b, stage_count_at, 3);
  });
  check(reject_split_mutation("stage-count-fewer-than-present",
                              [&](auto &b) {
                                put_u64_le(b, stage_count_at, 1);
                              }) == "trailing cache data",
        "forged smaller stage count leaves trailing data");
  reject_split_mutation("stage-count-zero", [&](auto &b) {
    put_u64_le(b, stage_count_at, 0);
  });
  reject_split_mutation("split-truncated", [](auto &b) { b.pop_back(); });
  reject_split_mutation("split-truncated-mid-stage", [&](auto &b) {
    // Cut inside the first stage image and keep the header's payload size
    // consistent, so only the bounded field reads can catch it.
    b.resize(stage_count_at + 8 + 100);
    put_u64_le(b, 16, static_cast<uint64_t>(b.size() - 88));
  });
  reject_split_mutation("split-bad-checksum",
                        [](auto &b) { b[56] ^= 0x80; });
  reject_split_mutation("split-last-byte-flipped",
                        [](auto &b) { b[b.size() - 1] ^= 0x01; });
  reject_split_mutation("stage-image-token-forged", [&](auto &b) {
    // The first stage image's first token id: 18 geometry fields, the carve,
    // cvec, frontier and the id count precede it.
    put_u64_le(b, stage_count_at + 8 + 8 * (18 + 2 + 1 + 1 + 1), 99999);
  });

  const auto reject_split_write = [&](const char *name,
                                      const SavedConversation &bad) {
    const fs::path target = temp.path / (std::string(name) + ".bin");
    check(!prefix_cache_write(target, identity, frontier, bad, error), name);
    check(!error.empty(), "failed split write reports a diagnostic");
    check(!fs::exists(target), "failed split write publishes no file");
  };
  {
    auto bad = split;
    bad.stage_images[1].live.ids[2]++;
    reject_split_write("stage-prefix-differs", bad);
  }
  {
    auto bad = split;
    bad.stage_images[0].stage_images.push_back(make_image());
    reject_split_write("stage-image-nested", bad);
  }
  {
    auto bad = split;
    bad.stage_images[0].layer_lo = bad.layer_hi - 1;
    reject_split_write("stage-carve-overlap", bad);
  }
  {
    auto bad = split;
    bad.stage_images[1].checkpoints.push_back(ConversationCheckpoint{});
    reject_split_write("stage-image-checkpoint", bad);
  }
  {
    auto bad = split;
    bad.stage_images[0].cvec = !bad.cvec;
    reject_split_write("stage-image-steering-differs", bad);
  }
  {
    auto bad = split;
    while (bad.stage_images.size() <= prefix_cache_max_stage_images) {
      const int64_t lo = bad.stage_images.back().layer_hi;
      bad.stage_images.push_back(make_stage_image(bad, lo, lo + 1, 1, 7));
    }
    reject_split_write("too-many-stage-images", bad);
  }
  {
    auto bad = split;
    bad.kv.clear();
    for (auto &stage : bad.stage_images)
      stage.kv.clear();
    reject_split_write("split-without-kv", bad);
  }

  // Limits: a 131072-token prefix (twice the old 65536 cap) round-trips with
  // its stage images, and the byte cap holds the 131072-token fp16 estimate
  // of prefix_cache_file.hpp for the shipped geometry.
  check(prefix_cache_max_tokens == 262144, "token cap is 262144");
  {
    const uint64_t tokens = 131072, mib = 1024 * 1024;
    const uint64_t qsa_kv = 13 * tokens * (2 * 256 * 2) * 2;
    const uint64_t indexer = 12 * (tokens / 4 + 1) * 128 * 4;
    const uint64_t gdn = uint64_t(36) * (128 * 48 * 128 + 10240 * 3) * 4;
    const uint64_t estimate = qsa_kv + indexer + gdn +
                              uint64_t(3) * 9 * 10240 * 4 + 3 * tokens * 8 +
                              4 * mib;
    check(estimate > 3500 * mib && estimate < 3700 * mib,
          "131072-token fp16 estimate is about 3.55 GiB");
    check(2 * estimate <= prefix_cache_max_bytes,
          "byte cap holds the 131072-token fp16 image with 2x margin");
    check(2 * qsa_kv + 2 * indexer + gdn <= prefix_cache_max_bytes,
          "byte cap holds the 262144-token fp16 image");
  }
  {
    auto long_split = make_split_image();
    std::vector<int32_t> long_prefix(131072);
    for (size_t i = 0; i < long_prefix.size(); ++i)
      long_prefix[i] = int32_t((i * 7919) % 151643);
    long_split.live.ids = long_prefix;
    for (auto &stage : long_split.stage_images)
      stage.live.ids = long_prefix;
    const fs::path long_cache = temp.path / "split-131072.bin";
    check(prefix_cache_write(long_cache, identity, frontier, long_split,
                             error),
          "write a 131072-token split prefix");
    SavedConversation long_restored;
    check(prefix_cache_read(long_cache, identity, long_prefix, frontier,
                            long_split.cvec, long_restored, error) &&
              equal_image(long_split, long_restored),
          "131072-token split prefix round-trips");
    auto over_cap = long_split;
    over_cap.live.ids.resize(prefix_cache_max_tokens + 1, 1);
    for (auto &stage : over_cap.stage_images)
      stage.live.ids = over_cap.live.ids;
    reject_split_write("split-above-token-cap", over_cap);
  }

  std::printf("prefix_cache_file_test: %d checks passed\n", checks);
  return 0;
}
