// src/grammar/grammar_vocab.cpp - build the GrammarVocab from GGUF tokenizer metadata.
//
// The pieces are expected in the GPT2 byte-level form byte-level-BPE vocabularies store (Qwen and
// the other supported families): every byte of the real piece rides the bytes_to_unicode symbol,
// which this loader decodes back.  A sentencepiece-style vocab holding literal U+0100-U+0143 text
// would be rewritten into raw bytes by that decode - not a supported family here.
#include "strata/grammar/grammar_vocab.hpp"

#include "strata/artifact/gguf_reader.hpp"

namespace strata {
namespace grammar {

namespace {
// SentencePiece stores byte tokens with the llama byte_decoder symbols: 'Ġ' (U+0120) is the
// space, 'Ċ' (U+010A) is the newline, and cp1252 symbols stand for their bytes; a GBNF grammar
// speaks real text, so the pieces are decoded back to bytes once at load (a grammar that wants
// a space then matches token 220, not a token whose piece spells 'Ġ').
std::string sp_piece_to_text(const std::string& piece) {
    std::string out;
    size_t i = 0;
    const size_t n = piece.size();
    while (i < n) {
        unsigned char c = (unsigned char)piece[i];
        uint32_t cp = c;
        size_t len = 1;
        if (c >= 0xF0 && i + 3 < n) { cp = ((c & 7u) << 18) | ((piece[i + 1] & 0x3F) << 12) | ((piece[i + 2] & 0x3F) << 6) | (piece[i + 3] & 0x3F); len = 4; }
        else if (c >= 0xE0 && i + 2 < n) { cp = ((c & 0xFu) << 12) | ((piece[i + 1] & 0x3F) << 6) | (piece[i + 2] & 0x3F); len = 3; }
        else if (c >= 0xC0 && i + 1 < n) { cp = ((c & 0x1Fu) << 6) | (piece[i + 1] & 0x3F); len = 2; }
        if (cp == 0x120u) { out += ' '; }            // the space symbol
        else if (cp == 0x10Au) { out += '\n'; }      // the newline symbol
        else if (cp >= 0xA1u && cp <= 0xFFu) { out += (char)(unsigned char)cp; }  // cp1252 symbols ride their bytes
        else if (cp >= 0x20u && cp <= 0x7Eu) { out += (char)(unsigned char)cp; }  // printable ascii
        else if (cp >= 0x100u && cp <= 0x143u) {
            // the rest of the GPT2 byte map (bytes_to_unicode, as HF's tokenizers and llama.cpp's
            // converter apply it): bytes 0-32 ride U+0100-U+0120, 127 rides U+0121, 128-160 ride
            // U+0122-U+0142 and 173 rides U+0143.  These are the second/third bytes of most CJK
            // pieces - leaving them as raw UTF-8 kept ~36% of this model's vocabulary (measured:
            // 88990 of 248320 tokens) out of every grammar mask, which degraded constrained
            // generation into filler characters exactly where the model wanted those tokens.
            uint32_t b = cp;
            if (cp <= 0x120u) b = cp - 0x100u;
            else if (cp == 0x121u) b = 0x7Fu;
            else if (cp <= 0x142u) b = cp - 0x122u + 0x80u;
            else b = 0xADu;
            out += (char)(unsigned char)b;
        }
        else { out.append(piece, i, len); }          // raw UTF-8 (CJK pieces) and everything else
        i += len;
    }
    return out;
}
} // namespace

bool load_grammar_vocab(const std::string& gguf_path, GrammarVocab& out, std::string& err) {
    // full_arrays: the sampled 64-item read cannot carry a 150k vocabulary
    GgufFile f(gguf_path, true);

    const MetaValue* tokens = f.get("tokenizer.ggml.tokens");
    if (tokens == nullptr || tokens->type != MetaType::ARRAY || tokens->elem != MetaType::STRING) {
        err = "tokenizer.ggml.tokens is not a string array in " + gguf_path;
        return false;
    }
    const size_t n = (size_t)tokens->count;
    out.pieces.clear();
    out.pieces.resize(n);
    for (size_t i = 0; i < tokens->items.size() && i < n; ++i) {
        out.pieces[i] = sp_piece_to_text(tokens->items[i].s);
    }

    out.control.assign(n, 0);
    if (const MetaValue* tt = f.get("tokenizer.ggml.token_type")) {
        if (tt->type != MetaType::ARRAY) {
            err = "tokenizer.ggml.token_type is not an array in " + gguf_path;
            return false;
        }
        // gguf-py SpecialToken: 1 NORMAL, 2 USER, 3 CONTROL, 4 UNUSED, 5 BYTE. A GGUF without
        // the key rides every token as text (llama.cpp's behavior before the flag existed).
        for (size_t i = 0; i < tt->items.size() && i < n; ++i) {
            out.control[i] = (int64_t)tt->items[i].u == 3 ? 1 : 0;
        }
    }

    out.eog.assign(n, 0);
    bool any = false;
    const MetaValue* eos = f.get("tokenizer.ggml.eos_token_id");
    if (eos != nullptr && eos->is_num() && eos->u < n) {
        out.eog[(size_t)eos->u] = 1;
        any = true;
    }
    if (const MetaValue* eos_ids = f.get("tokenizer.ggml.eos_token_ids")) {
        if (eos_ids->type == MetaType::ARRAY) {
            for (const MetaValue& e : eos_ids->items) {
                if (e.u < n) {
                    out.eog[(size_t)e.u] = 1;
                    any = true;
                }
            }
        }
    }
    if (!any) {
        err = "no tokenizer.ggml.eos_token_id(s) in " + gguf_path;
        return false;
    }
    return true;
}

} // namespace grammar
} // namespace strata
