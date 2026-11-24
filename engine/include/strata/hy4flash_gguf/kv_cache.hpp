#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash {

/// Store one compressed latent and rotary key per token and layer.
struct KvCache {
    int n_blocks = 0, kv_lora = 0, n_rot = 0, max_tokens = 0;
    int len = 0;                      ///< tokens appended so far
    /// In F16 mode both the compressed latent and rotary key use half precision.
    bool fp16 = true;
    /// Q8_0 mode quantizes only the latent. Rotary keys remain F16.
    bool q8 = false;
    std::vector<void*> kv_cmpr;       ///< per block, device [kv_lora, max_tokens] of __half / float, or Q8_0 rows
    std::vector<void*> k_pe;          ///< per block, device [n_rot, max_tokens], `fp16` ? __half : float

    ~KvCache();
    KvCache() = default;
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;
};

/// The cache formats `mla_attention` takes. Numbered so that a `bool fp16` still converts to the right one.
enum KvFormat : int { kKvF32 = 0, kKvF16 = 1, kKvQ8 = 2 };

inline int kv_format(const KvCache& c) { return c.q8 ? kKvQ8 : c.fp16 ? kKvF16 : kKvF32; }

/// Bytes one token's latent occupies in one block.
inline std::size_t kv_cache_lat_row_bytes(int kv_lora, int format) {
    return format == kKvQ8 ? (std::size_t) kv_lora + (std::size_t) kv_lora / 32 * 2
                           : (std::size_t) kv_lora * (format == kKvF16 ? 2 : 4);
}

/// Bytes one token's rope key occupies in one block; F16 for both F16 and Q8_0.
inline std::size_t kv_cache_pe_row_bytes(int n_rot, int format) {
    return (std::size_t) n_rot * (format == kKvF32 ? 4 : 2);
}

/// Allocate for `n_blocks` blocks and `max_tokens` positions. False (with `err`) when the device has no room.
/// The format is STRATA_HY4_FLASH_KV's (kv_cache_format_default); the second form names it.
bool kv_cache_init(KvCache& c, int n_blocks, int kv_lora, int n_rot, int max_tokens, std::string& err);
bool kv_cache_init(KvCache& c, int n_blocks, int kv_lora, int n_rot, int max_tokens, int format, std::string& err);

/// STRATA_HY4_FLASH_KV=q8_0 | f16 | f32, read once a process.
int kv_cache_format_default();

/// Append `n_tok` tokens' rows for one block at offset `at`, from the tensors `mla_project` produced.
/// `at + n_tok` must be within `max_tokens`.
bool kv_cache_append(KvCache& c, int block, int at, const float* kv_cmpr, const float* k_pe, int n_tok,
                     std::string& err, void* stream);

/// The same, with the position read from the DEVICE - what a captured graph needs, since the offset grows
/// by one every token and a graph bakes its node parameters.
bool kv_cache_append_dev(KvCache& c, int block, const int32_t* d_at, const float* kv_cmpr,
                         const float* k_pe, int n_tok, std::string& err, void* stream);

/// Bytes a cache of this shape occupies, for a budget line.
/// `format` is a KvFormat; a bool still means F32 / F16.
std::size_t kv_cache_bytes(int n_blocks, int kv_lora, int n_rot, int max_tokens, int format);

}  // namespace strata::hy4_flash
