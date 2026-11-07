// Vendored verbatim from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (common/json-schema-to-grammar.h), MIT license (see LICENSE-llama.txt in this directory).
// Strata edits are marked with "STRATA:" and kept to the include-closure shims (json_compat.h, <cassert>)
// so the converter builds with zero llama.cpp headers.
#pragma once

#include "json-schema.h"
#include "json.h"

#include <functional>
#include <string>

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_chat_schema_document & schema);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)>    add_rule;
    std::function<std::string(const std::string &, const common_chat_schema &)> add_schema;
};

struct common_grammar_options {
    bool dotall = false;
    bool tool_value = false;                 // root tool strings are raw, unlike nested JSON strings
    std::string forbidden_sequence;          // never emit this raw inside serialized JSON strings
    // The document a `$ref` resolves against, when the schema being converted is only part of it.  The tool
    // builder compiles one parameter at a time and a JSON Pointer is rooted at the whole document, so
    // `#/properties/x` must still reach a sibling.  nullptr means "the schema is the whole document".
    const common_json * ref_root = nullptr;
};

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf,
                                   const common_grammar_options & options);
std::string gbnf_format_literal(const std::string & literal);

// Ported from llama.cpp common/peg-parser.cpp (gbnf_ac_grammar + gbnf_excluding_grammar, MIT): the
// complement of an Aho-Corasick automaton over `forbidden`, i.e. a grammar whose language is exactly the
// strings that contain none of those literals as a substring.  Returns a complete rule set whose start
// rule is `prefix`.  Safety is structural, as upstream: every state's right-hand side opens with an empty
// alternative (so it may always stop) and a character that would complete a forbidden literal gets no
// alternative at all, so no derivation can spell one.
std::string gbnf_excluding_grammar(const std::string & prefix, const std::vector<std::string> & forbidden);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
