#pragma once

#include "strata/hy4flash_gguf/mat.hpp"
#include <cstdint>
#include <string>
#include <vector>

/// Hash the current token and its history to select rows in the four Engram tables.
namespace strata::hy4_flash {

/// The per-table constants the hash needs, as read from the GGUF (see Geometry::engram_tables) plus the token
/// map, which is too long for the shared metadata reader's array cap and so comes from the pack.
struct EngramHashParams {
    int32_t ngram_size = 0;             ///< 3
    int32_t heads_per_ngram = 0;        ///< 8
    int32_t columns = 0;                ///< (ngram_size - 1) * heads_per_ngram = 16
    int64_t pad_token = -1;             ///< engram.pad_token_id, BEFORE the map is applied
    std::vector<uint64_t> multipliers;  ///< ngram_size entries for this table
    std::vector<uint64_t> primes;       ///< `columns` entries (ple.head_vocab_sizes)
    std::vector<uint64_t> offsets;      ///< `columns` entries (ple.head_offsets), global row numbers
    const int32_t* token_map = nullptr; ///< |vocab| entries; nullptr means the identity
    int64_t token_map_len = 0;

    /// Everything present and self-consistent, so the hash cannot run on a half-filled struct.
    bool check(std::string& err) const;
};

/// Compute per-token row indices and validity from the supplied positions and preceding tokens.
bool engram_row_ids(const EngramHashParams& p, const int32_t* tokens, const int32_t* pos, const int32_t* prev,
                    const uint8_t* is_embd, int64_t n_tok, std::vector<int64_t>& ids,
                    std::vector<float>& valid, std::string& err);

/// Read the packed int32 token remapping from engram_token_map.bin.
bool engram_token_map_load(const std::string& pack_dir, std::vector<int32_t>& out, std::string& err);

/// Inject Engram values into the residual streams before attention pre-mixing.
std::size_t engram_layer_scratch_bytes(int n_embd, int hc, int head_dim, int columns, int n_tok);

struct EngramLayerWeights {
    const void* kv = nullptr;       int kv_type = 0;   ///< [head_dim * columns, n_embd * (hc + 1)]
    const float* kv_b = nullptr;        ///< [n_embd, hc + 1]
    const float* gate = nullptr;        ///< [n_embd, hc]
    const float* lookup_norm = nullptr; ///< [head_dim, columns]
    const float* pre_gamma = nullptr;   ///< [n_embd]
    const float* hc_scale = nullptr;    ///< [n_embd, hc]
};

void engram_layer(const float* x, const float* rows, const float* valid, const EngramLayerWeights& w,
                  int n_embd, int hc, int head_dim, int columns, int n_tok, float* out, void* scratch,
                  void* stream, MatCtx* mc = nullptr);

}  // namespace strata::hy4_flash
