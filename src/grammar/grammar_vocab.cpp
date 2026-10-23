// src/grammar/grammar_vocab.cpp - build the GrammarVocab from GGUF tokenizer metadata.
#include "strata/grammar/grammar_vocab.hpp"

#include "strata/artifact/gguf_reader.hpp"

namespace strata {
namespace grammar {

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
        out.pieces[i] = tokens->items[i].s;
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
