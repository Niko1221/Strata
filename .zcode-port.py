import re, io

path = r"F:/Strata-dev/src/grammar/grammar.cpp"
s = io.open(path, encoding="utf-8").read()

# 1) includes
head_new = (
    "// Ported from llama.cpp (MIT, https://github.com/ggml-org/llama.cpp), src/llama-grammar.cpp.\n"
    "// Mechanical port for Strata (see include/strata/grammar/grammar.hpp): names under\n"
    "// strata::grammar, llama_vocab replaced by the GrammarVocab shim, GGML/LLAMA logging\n"
    "// replaced by compat shims. apply_impl/accept_impl (the llama_token_data_array entry\n"
    "// points) are dropped: Strata constrains sampling through the per-row bitmask instead.\n"
    "#include \"strata/grammar/grammar.hpp\"\n"
    "#include \"strata/grammar/compat.hpp\"\n"
)
i = s.index("#include <cmath>")
s = head_new + "\n" + s[i:]

# 2) parse_token: GrammarVocab has no tokenize - <name> is an invalid token (faithful to the
#    upstream "must tokenize to exactly 1 token" throw). <[id]> handled above needs no vocab.
old_tok = """    llama_token tokens[2];
    int32_t n_tokens = vocab->tokenize(src, static_cast<int32_t>(pos - src), tokens, 2, false, true);
    if (n_tokens != 1) {
        // must tokenize to exactly 1 token
        throw std::runtime_error("invalid token '" + std::string(src, pos - src) + "'");
    }
    return std::make_pair(tokens[0], pos);"""
assert old_tok in s
s = s.replace(old_tok, """    // Strata's GrammarVocab has no tokenize(): a <name> token is an invalid token.
    throw std::runtime_error("invalid token '" + std::string(src, pos - src) + "'");""")

# 3) drop apply_impl + accept_impl (dead in Strata: bitmask path replaces them)
start = s.index("void llama_grammar_apply_impl(")
end = s.index("void llama_grammar_accept_str(")
s = s[:start] + s[end:]

# 4) names - longest first
names = [
    ("llama_grammar_stacks", "grammar_stacks_vec"),
    ("llama_grammar_candidates", "grammar_candidates"),
    ("llama_grammar_candidate", "grammar_candidate"),
    ("llama_grammar_init_impl", "grammar_init_impl"),
    ("llama_grammar_free_impl", "grammar_free"),
    ("llama_grammar_clone_impl", "grammar_clone"),
    ("llama_grammar_get_rules", "grammar_get_rules"),
    ("llama_grammar_get_stacks", "grammar_stacks"),
    ("llama_grammar_is_end_of_sequence", "grammar_is_end_of_sequence"),
    ("llama_grammar_advance_stack", "grammar_advance_stack"),
    ("llama_grammar_reject_candidates_for_stack", "grammar_reject_candidates_for_stack"),
    ("llama_grammar_reject_candidates", "grammar_reject_candidates"),
    ("llama_grammar_detect_left_recursion", "grammar_detect_left_recursion"),
    ("llama_grammar_accept_chr", "grammar_accept_chr"),
    ("llama_grammar_accept_str", "grammar_accept_str"),
    ("llama_grammar_accept_token", "grammar_accept"),
    ("llama_grammar_accept", "grammar_accept"),
    ("llama_grammar_trigger_pattern", "grammar_trigger_pattern"),
    ("llama_grammar_parser", "grammar_parser"),
    ("llama_grammar_element", "grammar_element"),
    ("llama_grammar_rules", "grammar_rules"),
    ("llama_grammar_rule", "grammar_rule"),
    ("llama_grammar_stack", "grammar_stack"),
    ("llama_partial_utf8", "partial_utf8"),
    ("llama_gretype", "gretype"),
    ("LLAMA_GRETYPE_", "gretype::"),
    ("struct llama_grammar", "struct grammar"),
    ("llama_grammar", "grammar"),
    ("llama_vocab", "GrammarVocab"),
    ("llama_token", "int32_t"),
]
for a, b in names:
    s = s.replace(a, b)

# 5) shims
s = s.replace("GGML_ASSERT(", "GRAMMAR_ASSERT(")
s = s.replace("GGML_ABORT(", "GRAMMAR_ABORT(")
s = s.replace("LLAMA_LOG_ERROR(", "GRAMMAR_LOG_ERROR(")
s = s.replace("LLAMA_LOG_DEBUG(", "GRAMMAR_DEBUG(")

# 6) namespace wrap: after the last include line
last_inc = s.rindex("#include ")
eol = s.index("\n", last_inc)
s = (s[:eol + 1] +
     "\nnamespace strata {\nnamespace grammar {\n\n"
     "// vector of stacks (the upstream alias, kept out of the public header)\n"
     "using grammar_stacks_vec = std::vector<grammar_stack>;\n\n" +
     s[eol + 1:])

s = s.rstrip() + """

// the entry point: parse the text, start at the root rule, no triggers
grammar * grammar_init(const GrammarVocab * vocab, const char * text, const char * root) {
    return grammar_init_impl(vocab, text, root, /*lazy=*/false, nullptr, 0, nullptr, 0);
}

} // namespace grammar
} // namespace strata
"""

io.open(path, "w", encoding="utf-8", newline="\n").write(s)

# 7) compat.hpp: GRAMMAR_DEBUG no-op
cpath = r"F:/Strata-dev/include/strata/grammar/compat.hpp"
c = io.open(cpath, encoding="utf-8").read()
if "GRAMMAR_DEBUG" not in c:
    c = c.rstrip() + "\n#define GRAMMAR_DEBUG(...) ((void)0)\n"
    io.open(cpath, "w", encoding="utf-8", newline="\n").write(c)

print("port transform done")
