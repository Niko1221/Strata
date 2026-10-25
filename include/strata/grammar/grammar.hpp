// Ported from llama.cpp (MIT, https://github.com/ggml-org/llama.cpp), src/llama-grammar.h.
// Mechanical port for Strata: names live under strata::grammar, llama_token is int32_t, and the
// llama_vocab dependency is replaced by the GrammarVocab shim defined below. The parsing and
// stack logic is kept faithful to the upstream file; do not "improve" it here.
#pragma once

#include <cstdint>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include <cstdio>

namespace strata {
namespace grammar {

// grammar element type
enum class gretype {
    END,            // end of rule definition
    ALT,            // start of alternate definition for rule
    RULE_REF,       // non-terminal element: reference to rule
    CHAR,           // terminal element: character (code point)
    CHAR_NOT,       // inverse char(s) ([^a], [^a-b] [^abc])
    CHAR_RNG_UPPER, // modifies a preceding CHAR or CHAR_ALT to be an inclusive range ([a-z])
    CHAR_ALT,       // modifies a preceding CHAR or CHAR_RNG_UPPER to add an alternate char ([ab])
    CHAR_ANY,       // any character (.)
    TOKEN,          // terminal element: token (<[token-id]>)
    TOKEN_NOT,      // inverse token (!<[token-id]>)
};

struct grammar_element {
    gretype  type;
    uint32_t value; // Unicode code point, rule ID, or token ID
};

struct partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct grammar_candidate {
    size_t           index;
    const uint32_t * code_points;
    partial_utf8     partial_utf8;
    int32_t          id;
};

using grammar_rule  = std::vector<grammar_element>;
using grammar_stack = std::vector<const grammar_element *>;

using grammar_rules      = std::vector<grammar_rule>;
using grammar_candidates = std::vector<grammar_candidate>;

// The vocab shim for the ported grammar. Strata builds it from the model's GGUF metadata
// (token pieces, control-token flags, end-of-generation flags); it has no tokenize(): a
// <name> token in a grammar is an invalid token, while GBNF's numeric <[id]> form works.
struct GrammarVocab {
    std::vector<std::string> pieces;  // token id -> raw piece
    std::vector<char>        control; // 1 for control (special) tokens
    std::vector<char>        eog;     // 1 for end-of-generation tokens

    // out-of-range ids have no piece
    std::string token_to_piece(int32_t id) const {
        if (id < 0 || static_cast<size_t>(id) >= pieces.size()) {
            return "";
        }
        return pieces[static_cast<size_t>(id)];
    }

    bool is_eog(int32_t id) const {
        return id >= 0 && static_cast<size_t>(id) < eog.size() && eog[static_cast<size_t>(id)] != 0;
    }
};

struct grammar;

// TODO: remove, needed for tests atm
const grammar_rules & grammar_get_rules(const grammar * grammar);
std::vector<grammar_stack> & grammar_stacks(grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void grammar_accept(grammar * grammar, uint32_t chr);

std::vector<grammar_candidate> grammar_reject_candidates_for_stack(
        const grammar_rules      & rules,
        const grammar_stack      & stack,
        const grammar_candidates & candidates);

struct grammar_parser {
    const GrammarVocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;

    grammar_rules rules;

    grammar_parser(const GrammarVocab * vocab = nullptr) : vocab(vocab) {}

    grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char        * src,
            const std::string & rule_name,
            grammar_rule      & rule,
            bool                is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;

    size_t find(const std::string & input) const;
};

struct grammar {
    // maintain a list of tokens and their positions in the trigger_buffer
    using token_pos = std::pair<int32_t, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const GrammarVocab * vocab;

    const grammar_rules rules;  // TODO: shared ptr
    std::vector<grammar_stack> stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger
    // tokens. Kept for faithfulness to the upstream file; Strata's engine does not use them in v1.
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Cleared once trigger is found.
    std::vector<token_pos>   trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
    std::vector<int32_t>     trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing (even if special).
    std::vector<grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.
};

//
// API
//

// note: needed for tests and callers that build the rules by hand
grammar * grammar_init_impl(
        const GrammarVocab * vocab,
        const grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

grammar * grammar_init_impl(
        const GrammarVocab * vocab,
                 const char * grammar_str,
                 const char * grammar_root,
                         bool lazy,
                const char ** trigger_patterns,
                       size_t num_trigger_patterns,
               const int32_t * trigger_tokens,
                       size_t num_trigger_tokens);

// the entry point: parse the text, start at grammar_root, no triggers (trigger args zeroed);
// returns nullptr when the grammar fails to parse
grammar * grammar_init(const GrammarVocab * vocab, const char * text, const char * root);

void grammar_free(grammar * grammar);

grammar * grammar_clone(const grammar & grammar);

// advance every stack over the piece's code points; throws std::runtime_error when no stack
// survives ("Unexpected empty grammar stack after accepting piece: ...")
void grammar_accept_str(grammar & grammar, const std::string & piece);

// advance every stack over the token: a TOKEN/TOKEN_NOT stack top matches by token id, the
// others by the piece's code points; throws std::runtime_error when no stack survives
void grammar_accept(grammar & grammar, int32_t token, const std::string & piece);

// the reject fold over all stacks (a token survives when at least one stack accepts the whole
// piece); the mask builder uses it exactly as llama.cpp's apply_impl did
grammar_candidates grammar_reject_candidates(
        const grammar_rules      & rules,
        const std::vector<grammar_stack> & stacks,
        const grammar_candidates & candidates);

// full-piece UTF-8 decode into code points plus the trailing partial (accept_str's helper)
std::pair<std::vector<uint32_t>, partial_utf8> decode_utf8(
        const std::string & src, partial_utf8 partial_start);

} // namespace grammar
} // namespace strata
