// include/strata/core/token_mask.hpp - serve's constrained decoding (serve/constrain.py), the engine's half.
//
// With `mask=1` on GEN the engine asks the server before every verify window (`MQ`) and reads one line back:
//   MF                 free: the window as always (drafts, MTP)
//   MF cut=<id,...>    free, but the accepted tokens end at the first of these ids (the end of the thinking)
//   MK <base64>        a one-token window whose head logits outside the mask are set to kMaskedLogit before the
//                      request's own sampler picks; little-endian uint32 words, bit v = token v allowed
//   MKN <n> <b0> ...   a constrained window of up to n rows: row i is the mask for the token after the engine's
//                      i-th draft (the same encoding as MK, one after another).  `token_mask=2` engines send
//                      "MQ d1 .. dT-1" and the server answers MKN only when it looked the drafts through.
// Header-only and backend-neutral: the CUDA, HIP and SYCL builds use the same host code.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/kernels/verify_kernels.hpp"  // strata::kernels::kVerifyMaxT: the row cap an MKN answer may claim

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

/// The server's answer to MQ, multi-row form.  `masked`: the window is constrained and `words` holds `rows` mask
/// rows back to back (row i starts at words[i * n_words], with n_words = words.size() / rows); every row is encoded
/// exactly like a single MK line.  MF/MF cut leave `rows` at 0 and fill `cut`.  Errors: rows < 1, rows >
/// kVerifyMaxT, rows that disagree on their word count (so `words` is not rows * n_words), a bad base64 row.
inline bool mask_parse_reply_rows(const std::string& line, bool& masked, int& rows,
                                  std::vector<uint32_t>& words, std::vector<int32_t>& cut, std::string& err) {
    masked = false;
    rows = 0;
    words.clear();
    cut.clear();
    if (line.rfind("MKN ", 0) == 0) {
        const char* p = line.c_str() + 4;
        char* e = nullptr;
        const long n = std::strtol(p, &e, 10);
        if (e == p || *e != ' ') { err = "a bad MKN row count"; return false; }
        if (n < 1) { err = "MKN with rows < 1"; return false; }
        if (n > strata::kernels::kVerifyMaxT) { err = "MKN rows above kVerifyMaxT"; return false; }
        p = e + 1;
        std::vector<uint32_t> row;
        int64_t n_words = -1;
        for (int i = 0; i < n; ++i) {
            if (!mask_b64_words(p, row) || row.empty()) { err = "a bad MKN mask row"; return false; }
            if (n_words < 0) n_words = (int64_t) row.size();
            else if ((int64_t) row.size() != n_words) { err = "MKN rows of different widths"; return false; }
            words.insert(words.end(), row.begin(), row.end());
            for (; *p != '\0' && *p != ' '; ++p) {}
            if (i + 1 < n) {
                if (*p != ' ') { err = "MKN with fewer rows than its header"; return false; }
                ++p;
            }
        }
        if (*p != '\0' && *p != '\r' && *p != '\n') { err = "MKN with more rows than its header"; return false; }
        masked = true;
        rows = (int) n;
        return true;
    }
    if (line.rfind("MK ", 0) == 0) {
        if (!mask_b64_words(line.c_str() + 3, words) || words.empty()) { err = "a bad MK mask"; return false; }
        masked = true;
        rows = 1;
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
    err = "expected MK, MKN or MF after MQ, got: " + line.substr(0, 40);
    return false;
}

/// The server's answer to MQ.  `masked`: an MK line (its words in `words`); else `cut` holds the MF line's cut ids.
/// The thin rows==1 wrapper of mask_parse_reply_rows: an "MKN 1 ..." answer parses as the single row, an MKN with
/// more rows is an error here (the engine that cannot mask a window should not receive one).
inline bool mask_parse_reply(const std::string& line, bool& masked, std::vector<uint32_t>& words,
                             std::vector<int32_t>& cut, std::string& err) {
    int rows = 0;
    if (!mask_parse_reply_rows(line, masked, rows, words, cut, err)) return false;
    if (rows > 1) { err = "MKN with more than one row"; return false; }
    return true;
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

/// Whether token v is allowed by the mask.  For row i of an MKN answer pass words.data() + i * n_words; the row is
/// just a pointer offset, so the word count stays the per-row one.
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
