// Optional, single text-prefix persistence. The codec has no CUDA dependency.
#pragma once
#include "strata/core/conversation_cache.hpp"
#include <array>
#include <filesystem>
#include <string>

namespace strata::core {
using PrefixDigest = std::array<uint8_t, 32>;
// Tokens: an agent client's system turn (Claude Code: tools + system prompt,
// ~40K tokens) fits many times over; 131072 is the target, 262144 the cap.
//
// Bytes: the whole image (top level plus every stage image of a layer split;
// the split only partitions the layers, so the sum is the one-GPU size).  For
// the shipped geometry (layout.hpp: 48 layers, 12 full-attention QSA layers,
// 36 GDN layers, n_head_kv 2, head_dim 256, idx_key_dim 128, idx_block 4) at
// 131072 tokens with --kv fp16, the largest K/V format:
//   QSA K/V   13 layers (12 + the MTP draft layer) x 131072 cells x
//             (2 heads x 256 x 2 B) x 2 (K and V)              = 3328 MiB
//   indexer   12 x (131072 / 4 + 1) rows x 128 x 4 B           ~  192 MiB
//   GDN       36 x (128 x 48 x 128 + 10240 x 3) x 4 B          ~  112 MiB
//   hyper-connection (PLE) history 9 x 10240 x 4 B per stage   ~  0.4 MiB
//   indexer tails/dead/block_pos, token ids (8 B each on disk,
//   repeated per stage image)                                   ~    2 MiB
//   total                                                       ~ 3.55 GiB
// The same formula gives the measured 4003-token int8 file of
// docs/sm75/BENCHMARKS.md (179,213,800 bytes) to within ~2 KB of metadata.
// No checkpoints are stored (the file holds one live prefix).  int8 K/V
// (1056 B/cell/layer) is ~2.0 GiB, q4_0/k8v4 less.  8 GiB holds 131072 fp16 tokens with > 2x margin
// and the 262144-token cap at fp16 (~7.0 GiB).  It is only a format bound:
// the physical-RAM admission checks at read and write are the real guard.
constexpr uint64_t prefix_cache_max_bytes = 8ULL * 1024 * 1024 * 1024;
constexpr size_t prefix_cache_max_tokens = 262144;
// A layer split runs at most 8 stages (SplitDrive::kMax), so at most 7 later
// stage images; the bound is kept at 8.
constexpr size_t prefix_cache_max_stage_images = 8;

// SHA-256 payload integrity and cache-key hashing (CNG on Windows).
PrefixDigest prefix_cache_digest(const void *bytes, size_t size);
// Local artifact identity: canonical paths, size/mtime, complete small files,
// and three 64 KiB samples of large files. This is NOT a full weights hash.
// Replacing weights while preserving metadata and unsampled bytes requires
// removing the cache; ordinary model/config/prompt edits invalidate it.
bool prefix_cache_identity(const std::vector<std::filesystem::path> &assets,
                           const std::vector<std::string> &settings,
                           PrefixDigest &identity, std::string &error);

// One minimal snapshot: no images, stage parts or duplicate checkpoints.  With
// a layer split, image.stage_images holds the later stages' images (same
// rules each, same prefix; format v2).
// next_token also keys MTP's one-token lookahead at the saved frontier.
bool prefix_cache_write(const std::filesystem::path &path,
                        const PrefixDigest &identity, int32_t next_token,
                        const SavedConversation &image, std::string &error);
// Bounded decode into a temporary CPU object; out is untouched on any failure.
// Exact expected prefix is checked before allocating the large state payload.
// Caller must still validate every image (the top level, and each stage image
// against its stage's session) before any GPU writes.
bool prefix_cache_read(const std::filesystem::path &path,
                       const PrefixDigest &identity,
                       const std::vector<int32_t> &expected_prefix,
                       int32_t next_token, bool cvec, SavedConversation &out,
                       std::string &error);
} // namespace strata::core
