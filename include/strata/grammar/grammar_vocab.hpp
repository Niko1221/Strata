// include/strata/grammar/grammar_vocab.hpp - the GrammarVocab built from a model's GGUF.
//
// The engine keeps no tokenizer, and the model loader's metadata read samples array values at 64
// items (include/strata/artifact/gguf_reader.hpp:361-370) - the 150k-token list is not retained.
// The grammar vocab therefore re-opens the GGUF with full arrays and reads exactly:
//   tokenizer.ggml.tokens      - the piece per id
//   tokenizer.ggml.token_type  - the CONTROL flag (the vendored apply treats special tokens as
//                                text; Strata's mask bans CONTROL - spec §5.1, recorded divergence)
//   tokenizer.ggml.eos_token_id / eos_token_ids - the end-of-generation ids (EOG allowed at the
//                                grammar's terminal state only)
#pragma once

#include "strata/grammar/grammar.hpp"

#include <string>

namespace strata {
namespace grammar {

// false + a one-line err when the GGUF has no tokenizer metadata.
bool load_grammar_vocab(const std::string& gguf_path, GrammarVocab& out, std::string& err);

} // namespace grammar
} // namespace strata
