// include/strata/core/token_mask.hpp - serve's constrained decoding (serve/constrain.py), the engine's half.
//
// With `mask=1` on GEN the engine asks the server before every verify window (`MQ`) and reads one line back:
//   MF                 free: the window as always (drafts, MTP)
//   MF cut=<id,...>    free, but the accepted tokens end at the first of these ids (the end of the thinking)
//   MK <base64>        a one-token window whose head logits outside the mask are set to kMaskedLogit before the
//                      request's own sampler picks; little-endian uint32 words, bit v = token v allowed
// Header-only and backend-neutral: the CUDA, HIP and SYCL builds use the same host code.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core {

constexpr float kMaskedLogit = -1e30f;   // finite: no inf arithmetic in the penalties or the softmax

/// base64 (standard alphabet, '=' padding) -> little-endian uint32 words.  False on a bad character or a length that
/// is not a whole number of words.
inline bool mask_b64_words(const char* s, std::vector<uint32_t>& out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<uint8_t> bytes;
    uint32_t acc = 0;
    int bits = 0;
    for (const char* p = s; *p != '\0' && *p != ' ' && *p != '\r' && *p != '\n'; ++p) {
        if (*p == '=') break;
        const int v = val(*p);
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            bytes.push_back((uint8_t) ((acc >> bits) & 0xFFu));
        }
    }
    if (bytes.size() % 4 != 0) return false;
    out.assign(bytes.size() / 4, 0u);
    for (size_t w = 0; w < out.size(); ++w)
        out[w] = (uint32_t) bytes[4 * w] | ((uint32_t) bytes[4 * w + 1] << 8) | ((uint32_t) bytes[4 * w + 2] << 16) |
                 ((uint32_t) bytes[4 * w + 3] << 24);
    return true;
}

/// The server's answer to MQ.  `masked`: an MK line (its words in `words`); else `cut` holds the MF line's cut ids.
inline bool mask_parse_reply(const std::string& line, bool& masked, std::vector<uint32_t>& words,
                             std::vector<int32_t>& cut, std::string& err) {
    masked = false;
    words.clear();
    cut.clear();
    if (line.rfind("MK ", 0) == 0) {
        if (!mask_b64_words(line.c_str() + 3, words) || words.empty()) { err = "a bad MK mask"; return false; }
        masked = true;
        return true;
    }
    if (line == "MF") return true;
    if (line.rfind("MF cut=", 0) == 0) {
        const char* p = line.c_str() + 7;
        while (*p != '\0') {
            char* e = nullptr;
            const long v = std::strtol(p, &e, 10);
            if (e == p || v < 0) { err = "a bad MF cut list"; return false; }
            cut.push_back((int32_t) v);
            p = *e == ',' ? e + 1 : e;
            if (*e != ',' && *e != '\0') { err = "a bad MF cut list"; return false; }
        }
        return true;
    }
    err = "expected MK or MF after MQ, got: " + line.substr(0, 40);
    return false;
}

/// One row of head logits: every token outside the mask to kMaskedLogit.  Returns how many tokens are allowed.
inline int64_t mask_apply_row(float* row, int64_t n_vocab, const uint32_t* words, int64_t n_words) {
    int64_t allowed = 0;
    for (int64_t v = 0; v < n_vocab; ++v) {
        const int64_t w = v >> 5;
        if (w < n_words && ((words[w] >> (v & 31)) & 1u)) ++allowed;
        else row[v] = kMaskedLogit;
    }
    return allowed;
}

/// The accepted-draft count `a` of a free window, ended at the first cut id among the emitted outv[0..a): the
/// window then emits outv[0..i] and commits window[0..i], exactly as if draft i+1 had been rejected.
inline int mask_cut_accepted(const int32_t* outv, int a, const std::vector<int32_t>& cut) {
    if (cut.empty()) return a;
    for (int i = 0; i < a; ++i)
        for (const int32_t c : cut)
            if (outv[i] == c) return i;
    return a;
}

/// Whether token v is allowed by the mask.
inline bool mask_allows(const uint32_t* words, int64_t n_words, int64_t v) {
    return v >= 0 && (v >> 5) < n_words && ((words[v >> 5] >> (v & 31)) & 1u);
}

/// The allowed token with the highest logit (row already masked); -1 if none.
inline int32_t mask_argmax(const float* row, int64_t n_vocab, const uint32_t* words, int64_t n_words) {
    int32_t best = -1;
    for (int64_t v = 0; v < n_vocab; ++v)
        if (mask_allows(words, n_words, v) && (best < 0 || row[v] > row[best])) best = (int32_t) v;
    return best;
}

}  // namespace strata::core
