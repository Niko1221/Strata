// Compute Engram row hashes on the host before reading the selected rows.
#include "strata/hy4flash_gguf/engram.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace strata::hy4_flash {
namespace {

constexpr int64_t kDead = -1;

/// token -> compressed id, or kDead when the token is outside the map. An absent map is the identity, which is
/// what a checkpoint without vocabulary compression would mean.
inline int64_t cid(const EngramHashParams& p, int64_t t) {
    if (t < 0) return kDead;
    if (p.token_map == nullptr) return t;
    return t >= p.token_map_len ? kDead : (int64_t) p.token_map[t];
}

}  // namespace

bool EngramHashParams::check(std::string& err) const {
    if (ngram_size < 2 || heads_per_ngram < 1) {
        err = "engram: ngram_size " + std::to_string(ngram_size) + " and heads_per_ngram " +
              std::to_string(heads_per_ngram) + "; the hash needs at least one predecessor and one head";
        return false;
    }
    if (columns != (ngram_size - 1) * heads_per_ngram) {
        err = "engram: " + std::to_string(columns) + " columns but (ngram_size - 1) * heads_per_ngram is " +
              std::to_string((ngram_size - 1) * heads_per_ngram);
        return false;
    }
    if ((int32_t) multipliers.size() != ngram_size) {
        err = "engram: " + std::to_string(multipliers.size()) + " multipliers for an n-gram of " +
              std::to_string(ngram_size) + "; they come from the file and cannot be re-derived";
        return false;
    }
    if ((int32_t) primes.size() != columns || (int32_t) offsets.size() != columns) {
        err = "engram: " + std::to_string(primes.size()) + " primes and " + std::to_string(offsets.size()) +
              " offsets for " + std::to_string(columns) + " columns";
        return false;
    }
    for (int32_t c = 0; c < columns; ++c)
        if (primes[(size_t) c] == 0) {
            // A zero bucket count would be a division by zero, and it means the metadata was misread rather
            // than that the column is empty.
            err = "engram: column " + std::to_string(c) + " has a bucket count of 0";
            return false;
        }
    if (pad_token < 0) {
        err = "engram: no pad token; a history shorter than the n-gram is padded, not dropped";
        return false;
    }
    if (token_map != nullptr && pad_token >= token_map_len) {
        err = "engram: the pad token " + std::to_string(pad_token) + " is outside a map of " +
              std::to_string(token_map_len);
        return false;
    }
    return true;
}

bool engram_row_ids(const EngramHashParams& p, const int32_t* tokens, const int32_t* pos, const int32_t* prev,
                    const uint8_t* is_embd, int64_t n_tok, std::vector<int64_t>& ids,
                    std::vector<float>& valid, std::string& err) {
    if (!p.check(err)) return false;
    if (n_tok < 0 || (n_tok > 0 && (tokens == nullptr || pos == nullptr))) {
        err = "engram_row_ids: tokens and positions are required";
        return false;
    }
    const int64_t n_prev = p.ngram_size - 1;
    if (n_tok > 0 && n_prev > 0 && prev == nullptr) {
        err = "engram_row_ids: this model's hash reads " + std::to_string(n_prev) +
              " predecessors per token, so `prev` is required";
        return false;
    }

    // The pad goes through the map too: the reference takes engram_token_map[engram_pad_id].
    const int64_t pad = p.token_map != nullptr ? (int64_t) p.token_map[p.pad_token] : p.pad_token;

    ids.assign((size_t) (n_tok * p.columns), 0);
    valid.assign((size_t) (n_tok * p.columns), 0.f);
    std::vector<uint64_t> toks((size_t) p.ngram_size);

    for (int64_t i = 0; i < n_tok; ++i) {
        const bool embd = is_embd != nullptr && is_embd[i] != 0;
        int64_t c = embd ? kDead : cid(p, tokens[i]);
        const bool dead = c == kDead;
        toks[0] = (uint64_t) (dead ? pad : c);
        for (int64_t s = 1; s < p.ngram_size; ++s) {
            // `prev` is oldest-first, so the s-th predecessor back is at (n_prev - s).
            c = pos[i] - s < 0 ? kDead : cid(p, prev[i * n_prev + (n_prev - s)]);
            toks[(size_t) s] = (uint64_t) (c == kDead ? pad : c);
        }

        // Unsigned 64-bit, and the multiplication is meant to wrap.
        uint64_t rolling = toks[0] * p.multipliers[0];
        for (int64_t o = 1; o < p.ngram_size; ++o) {
            rolling ^= toks[(size_t) o] * p.multipliers[(size_t) o];
            // All the heads of this order share the rolling value; only the prime and the offset differ.
            for (int64_t h = 0; h < p.heads_per_ngram; ++h) {
                const int64_t col = (o - 1) * p.heads_per_ngram + h;
                ids[(size_t) (i * p.columns + col)] =
                    (int64_t) (rolling % p.primes[(size_t) col] + p.offsets[(size_t) col]);
                valid[(size_t) (i * p.columns + col)] = dead ? 0.f : 1.f;
            }
        }
    }
    return true;
}

bool engram_token_map_load(const std::string& pack_dir, std::vector<int32_t>& out, std::string& err) {
    const std::string path = pack_dir + "/engram_token_map.bin";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "engram: cannot open " + path + " (regenerate the pack with tools/iq_pack.py: the token map is "
              "too long for the GGUF metadata reader's array cap)";
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % (long) sizeof(int32_t) != 0) {
        std::fclose(f);
        err = "engram: " + path + " is " + std::to_string(bytes) + " bytes, not a whole number of int32";
        return false;
    }
    out.resize((size_t) bytes / sizeof(int32_t));
    const bool ok = std::fread(out.data(), 1, (size_t) bytes, f) == (size_t) bytes;
    std::fclose(f);
    if (!ok) {
        err = "engram: short read on " + path;
        out.clear();
        return false;
    }
    return true;
}

}  // namespace strata::hy4_flash
