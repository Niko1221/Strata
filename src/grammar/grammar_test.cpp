// src/grammar/grammar_test.cpp - parser goldens ported from llama.cpp tests/test-grammar-parser.cpp (MIT), plus accept smoke tests.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "strata/grammar/grammar.hpp"

#include <cassert>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::grammar;

static const char * type_str(gretype type) {
    switch (type) {
        case gretype::CHAR: return "gretype::CHAR";
        case gretype::CHAR_NOT: return "gretype::CHAR_NOT";
        case gretype::CHAR_ALT: return "gretype::CHAR_ALT";
        case gretype::CHAR_RNG_UPPER: return "gretype::CHAR_RNG_UPPER";
        case gretype::RULE_REF: return "gretype::RULE_REF";
        case gretype::ALT: return "gretype::ALT";
        case gretype::END: return "gretype::END";
        default: return "?";
    }
}

static void verify_parsing(const char *grammar_bytes, const std::vector<std::pair<std::string, uint32_t>> expected, const std::vector<grammar_element> &expected_rules) {
    uint32_t index = 0;
    grammar_parser parsed_grammar;
    parsed_grammar.parse(grammar_bytes);

    std::map<uint32_t, std::string> symbol_names;
    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
        symbol_names[it->second] = it->first;
    }

    auto print_all = [&]() {
        std::fprintf(stderr, "    verify_parsing(R\"\"\"(%s)\"\"\", {\n", grammar_bytes);
        for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
            std::fprintf(stderr, "        {\"%s\", %u},\n", it->first.c_str(), it->second);
        }
        std::fprintf(stderr, "    }, {\n");
        for (size_t i_rule = 0; i_rule < parsed_grammar.rules.size(); i_rule++) {
            std::fprintf(stderr, "        // %s (index %zu)\n", symbol_names[i_rule].c_str(), i_rule);
            auto & rule = parsed_grammar.rules[i_rule];
            for (uint32_t i = 0; i < rule.size(); i++) {
                std::string rule_str;
                std::fprintf(stderr, "        {%s, ", type_str(rule[i].type));
                if (rule[i].type == gretype::CHAR || rule[i].type == gretype::CHAR_ALT ||
                    rule[i].type == gretype::CHAR_NOT || rule[i].type == gretype::CHAR_RNG_UPPER) {
                    char c = rule[i].value;
                    if (c == '\n') {
                        std::fprintf(stderr, "'\\n'");
                    } else if (c == '\t') {
                        std::fprintf(stderr, "'\\t'");
                    } else if (c == '\r') {
                        std::fprintf(stderr, "'\\r'");
                    } else if (c == '\0') {
                        std::fprintf(stderr, "'\\0'");
                    } else {
                        std::fprintf(stderr, "'%c'", c);
                    }
                } else if (rule[i].type == gretype::RULE_REF) {
                    std::fprintf(stderr, "/* %s */ %u", symbol_names[rule[i].value].c_str(), rule[i].value);
                } else {
                    std::fprintf(stderr, "%u", rule[i].value);
                }
                std::fprintf(stderr, "},\n");
            }
        }
        std::fprintf(stderr, "    });\n");
    };

    if (getenv("TEST_GRAMMAR_PARSER_PRINT_ALL")) {
        print_all();
        std::fprintf(stderr, "\n");
        return;
    }

    std::fprintf(stderr, "Testing grammar:%s\n", grammar_bytes);

    if (parsed_grammar.symbol_ids.size() != expected.size()) {
        std::fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
        print_all();
        assert(parsed_grammar.symbol_ids.size() == expected.size());
    }

    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it)
    {
        std::string key = it->first;
        uint32_t value = it->second;
        std::pair<std::string, uint32_t> expected_pair = expected[index];

        // pretty print error message before asserting
        if (expected_pair.first != key || expected_pair.second != value)
        {
            std::fprintf(stderr, "index: %u\n", index);
            std::fprintf(stderr, "expected_pair: %s, %u\n", expected_pair.first.c_str(), expected_pair.second);
            std::fprintf(stderr, "actual_pair: %s, %u\n", key.c_str(), value);
            std::fprintf(stderr, "expected_pair != actual_pair\n");
            std::fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
            print_all();
        }

        assert(expected_pair.first == key && expected_pair.second == value);

        index++;
    }

    index = 0;
    for (auto rule : parsed_grammar.rules)
    {
        // compare rule to expected rule
        for (uint32_t i = 0; i < rule.size(); i++)
        {
            grammar_element element = rule[i];
            grammar_element expected_element = expected_rules[index];

            // pretty print error message before asserting
            if (expected_element.type != element.type || expected_element.value != element.value)
            {
                std::fprintf(stderr, "index: %u\n", index);
                std::fprintf(stderr, "expected_element: %s, %u\n", type_str(expected_element.type), expected_element.value);
                std::fprintf(stderr, "actual_element: %s, %u\n", type_str(element.type), element.value);
                std::fprintf(stderr, "expected_element != actual_element\n");
                std::fprintf(stderr, "all elements:\n");
                std::fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
                print_all();
            }

            assert(expected_element.type == element.type && expected_element.value == element.value);
            index++;
        }
    }
}

static void verify_failure(const char * grammar_bytes) {
    std::fprintf(stderr, "Testing expected failure:%s\n", grammar_bytes);
    grammar_parser result;
    result.parse(grammar_bytes);
    assert(result.rules.empty() && "should have failed");
}


// accept over the piece code points: init -> accept 'a','b' -> every stack empties -> the engine's
// token path (grammar_accept with a piece) throws; the chr path (advance_stack drop) stays silent
static void test_accept_ab(void) {
    grammar * g = grammar_init(nullptr, "root ::= \"ab\"\n", "root");
    assert(g != nullptr);
    grammar_accept(g, (uint32_t) 'a');
    grammar_accept(g, (uint32_t) 'b');
    bool any_empty = false;
    for (const grammar_stack & stack : grammar_stacks(g)) {
        if (stack.empty()) {
            any_empty = true;
        }
    }
    assert(any_empty && "a stack must be at the end of the grammar");
    bool threw = false;
    try {
        grammar_accept(*g, 0, "c");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw && "the token accept path past the grammar must throw");
    grammar_free(g);
}

// the token-syntax rule needs no vocab: <[id]> parses numerically
static void test_token_syntax_init(void) {
    grammar * g = grammar_init(nullptr, "root ::= <[1000]> <[1001]>\n", "root");
    assert(g != nullptr);
    grammar_free(g);
}

// ---- mask + vocab tests (Task 3) -------------------------------------------
#include "strata/grammar/grammar_mask.hpp"
#include "strata/grammar/grammar_vocab.hpp"

#include <cstring>
#include <fstream>

static bool bit_at(const uint32_t* words, int id) {
    return (words[id >> 5] & (1u << (id & 31))) != 0;
}

static void test_mask_ab(void) {
    GrammarVocab vocab;
    // id 3 rides a NUL piece (the byte-0x00 token once the GPT2 byte map is fully decoded) and id 4
    // an invalid-UTF-8 piece (the same token before the map was complete): both must stay banned in
    // every state - the accept walk stops at a NUL, so either would "match" as an empty piece
    vocab.pieces = { "a", "b", "<|im_end|>", std::string("\x00", 1), "\xc4\x8a" };
    vocab.control = { 0, 0, 1, 0, 0 };
    vocab.eog = { 0, 0, 1, 0, 0 };
    grammar* g = grammar_init(nullptr, "root ::= \"ab\"\n", "root");
    assert(g != nullptr);
    const grammar_rules& rules = grammar_get_rules(g);
    uint32_t words[1];

    grammar_build_mask(vocab, rules, grammar_stacks(g), words);
    assert(!bit_at(words, 0) && bit_at(words, 1) && bit_at(words, 2)); // only "a" rides
    assert(bit_at(words, 3) && bit_at(words, 4));                      // NUL and broken pieces stay banned
    assert((words[0] & 0xFFFFFFF8u) == 0xFFFFFFF8u);                   // padding stays banned

    grammar_accept(g, (uint32_t) 'a');
    grammar_build_mask(vocab, rules, grammar_stacks(g), words);
    assert(bit_at(words, 0) && !bit_at(words, 1) && bit_at(words, 2)); // only "b"
    assert(bit_at(words, 3) && bit_at(words, 4));

    grammar_accept(g, (uint32_t) 'b');
    grammar_build_mask(vocab, rules, grammar_stacks(g), words);
    assert(bit_at(words, 0) && bit_at(words, 1) && !bit_at(words, 2)); // terminal: EOG only
    grammar_free(g);
}

static void test_mask_cache(void) {
    GrammarVocab vocab;
    vocab.pieces = { "a", "b", "<|im_end|>" };
    vocab.control = { 0, 0, 1 };
    vocab.eog = { 0, 0, 1 };
    grammar* g = grammar_init(nullptr, "root ::= \"ab\"\n", "root");
    assert(g != nullptr);
    GrammarMaskCache cache(vocab, grammar_get_rules(g));
    const std::vector<uint32_t>& m1 = cache.mask_for(grammar_stacks(g));
    const std::vector<uint32_t>& m2 = cache.mask_for(grammar_stacks(g));
    assert(&m1 == &m2 && "the same stack state must return the same row");
    assert(cache.size() == 1);
    grammar_accept(g, (uint32_t) 'a');
    const std::vector<uint32_t>& m3 = cache.mask_for(grammar_stacks(g));
    assert(&m3 != &m1 && "an advanced stack is a new key");
    assert(cache.size() == 2);
    grammar_free(g);
}

// a minimal v3 GGUF: zero tensors, 100-token vocabulary, token 99 CONTROL + eos.  Ids 5-9 carry
// GPT2-byte-mapped pieces (the form Qwen-family GGUFs store) so the loader's byte map is pinned.
static void write_test_gguf(const std::string& path) {
    std::string b;
    auto u32 = [&b](uint32_t v) { b.append((const char*) &v, 4); };
    auto u64 = [&b](uint64_t v) { b.append((const char*) &v, 8); };
    auto str = [&](const std::string& s) { u64(s.size()); b += s; };
    u32(0x46554747u); // "GGUF"
    u32(3);           // version
    u64(0);           // n_tensors
    u64(3);           // n_kv
    str("tokenizer.ggml.tokens");
    u32(9); // ARRAY
    u32(8); // elem STRING
    u64(100);
    for (int i = 0; i < 100; ++i) {
        // the GPT2 byte-map forms: 'Ĥ' rides byte 0x82 (U+0124), 'Ń' rides 0xAD (U+0143),
        // "ä¸Ń" is the piece of a 一-class CJK character (bytes E4 B8 AD), 'Ċ' the newline,
        // 'Ġthe' the space-prefixed word
        std::string piece;
        switch (i) {
            case 5: piece = "\xc4\xa4"; break;
            case 6: piece = "\xc5\x83"; break;
            case 7: piece = "\xc3\xa4\xc2\xb8\xc5\x83"; break;
            case 8: piece = "\xc4\x8a"; break;
            case 9: piece = "\xc4\xa0the"; break;
            case 10: piece = "\xc4\x80"; break;   // 'Ā' rides byte 0x00 itself
            default: piece = "t" + std::to_string(i); break;
        }
        str(piece);
    }
    str("tokenizer.ggml.token_type");
    u32(9); // ARRAY
    u32(5); // elem I32
    u64(100);
    for (int i = 0; i < 100; ++i) {
        u32(i == 99 ? 3u : 1u);
    }
    str("tokenizer.ggml.eos_token_id");
    u32(5); // I32
    u32(99);
    std::ofstream out(path, std::ios::binary);
    out << b;
}

static void test_load_grammar_vocab(void) {
    const std::string path = "grammar_vocab_test.gguf";
    write_test_gguf(path);
    GrammarVocab vocab;
    std::string err;
    assert(load_grammar_vocab(path, vocab, err) && "a 100-token GGUF must load");
    assert(vocab.pieces.size() == 100);
    assert(vocab.pieces[3] == "t3");
    assert(vocab.pieces[99] == "t99" && "full arrays: the 100th token is not sampled away");
    assert(vocab.control[99] == 1 && vocab.control[0] == 0);
    assert(vocab.is_eog(99) && !vocab.is_eog(98));
    // the GPT2 byte map, complete: the pieces decode back to the raw bytes the grammar speaks
    const auto hex = [](const std::string& s) {
        std::string out;
        for (unsigned char c : s) {
            char b[4];
            std::snprintf(b, sizeof b, "%02x", c);
            out += b;
        }
        return out;
    };
    assert(hex(vocab.pieces[5]) == "82" && "'Ĥ' rides byte 0x82");
    assert(hex(vocab.pieces[6]) == "ad" && "'Ń' rides byte 0xAD");
    assert(hex(vocab.pieces[7]) == "e4b8ad" && "a CJK piece decodes to its own UTF-8 bytes");
    assert(hex(vocab.pieces[8]) == "0a" && "'Ċ' rides byte 0x0A");
    assert(hex(vocab.pieces[9]) == "20746865" && "'Ġthe' decodes to \" the\"");
    assert(vocab.pieces[10] == std::string("\x00", 1) && "'Ā' decodes to the NUL byte itself");
    std::remove(path.c_str());
}

int main(void)
{

    verify_failure(R"""(
        root ::= "a"{,}"
    )""");

    verify_failure(R"""(
        root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99}
    )""");

    verify_failure(R"""(
        root ::= "a"{,10}"
    )""");

    verify_failure(R"""(
        root ::= "a"{5000}
    )""");

    verify_failure(R"""(
        root ::= "a"{5000,}
    )""");

    verify_failure(R"""(
        root ::= "a"{5000,6000}
    )""");

    verify_parsing(R"""(
        root ::= "a"{0,5000}
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root ::= "a"{3,5000}
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a" | [bdx-z] | [^1-3]
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::ALT, 0},
        {gretype::CHAR, 'b'},
        {gretype::CHAR_ALT, 'd'},
        {gretype::CHAR_ALT, 'x'},
        {gretype::CHAR_RNG_UPPER, 'z'},
        {gretype::ALT, 0},
        {gretype::CHAR_NOT, '1'},
        {gretype::CHAR_RNG_UPPER, '3'},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= a+
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* a */ 1},
        {gretype::RULE_REF, /* root_2 */ 2},
        {gretype::END, 0},
        // a (index 1)
        {gretype::CHAR, 'a'},
        {gretype::END, 0},
        // root_2 (index 2)
        {gretype::RULE_REF, /* a */ 1},
        {gretype::RULE_REF, /* root_2 */ 2},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"+
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= a?
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* root_2 */ 2},
        {gretype::END, 0},
        // a (index 1)
        {gretype::CHAR, 'a'},
        {gretype::END, 0},
        // root_2 (index 2)
        {gretype::RULE_REF, /* a */ 1},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"?
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::CHAR, 'a'},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= a*
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* root_2 */ 2},
        {gretype::END, 0},
        // a (index 1)
        {gretype::CHAR, 'a'},
        {gretype::END, 0},
        // root_2 (index 2)
        {gretype::RULE_REF, /* a */ 1},
        {gretype::RULE_REF, /* root_2 */ 2},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"*
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2}
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,}
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{ 4}
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,4}
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
    }, {
        // root (index 0)
        {gretype::CHAR, 'a'},
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_2 */ 2},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::CHAR, 'a'},
        {gretype::ALT, 0},
        {gretype::END, 0},
        // root_2 (index 2)
        {gretype::CHAR, 'a'},
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= [0-9]+
    )""", {
        {"expr", 2},
        {"expr_5", 5},
        {"expr_6", 6},
        {"root", 0},
        {"root_1", 1},
        {"root_4", 4},
        {"term", 3},
        {"term_7", 7},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::RULE_REF, /* root_4 */ 4},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::RULE_REF, /* expr */ 2},
        {gretype::CHAR, '='},
        {gretype::RULE_REF, /* term */ 3},
        {gretype::CHAR, '\n'},
        {gretype::END, 0},
        // expr (index 2)
        {gretype::RULE_REF, /* term */ 3},
        {gretype::RULE_REF, /* expr_6 */ 6},
        {gretype::END, 0},
        // term (index 3)
        {gretype::CHAR, '0'},
        {gretype::CHAR_RNG_UPPER, '9'},
        {gretype::RULE_REF, /* term_7 */ 7},
        {gretype::END, 0},
        // root_4 (index 4)
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::RULE_REF, /* root_4 */ 4},
        {gretype::ALT, 0},
        {gretype::END, 0},
        // expr_5 (index 5)
        {gretype::CHAR, '-'},
        {gretype::CHAR_ALT, '+'},
        {gretype::CHAR_ALT, '*'},
        {gretype::CHAR_ALT, '/'},
        {gretype::RULE_REF, /* term */ 3},
        {gretype::END, 0},
        // expr_6 (index 6)
        {gretype::RULE_REF, /* expr_5 */ 5},
        {gretype::RULE_REF, /* expr_6 */ 6},
        {gretype::ALT, 0},
        {gretype::END, 0},
        // term_7 (index 7)
        {gretype::CHAR, '0'},
        {gretype::CHAR_RNG_UPPER, '9'},
        {gretype::RULE_REF, /* term_7 */ 7},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" ws term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= ident | num | "(" ws expr ")" ws
        ident ::= [a-z] [a-z0-9_]* ws
        num   ::= [0-9]+ ws
        ws    ::= [ \t\n]*
    )""", {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    }, {
        // root (index 0)
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::RULE_REF, /* root_5 */ 5},
        {gretype::END, 0},
        // root_1 (index 1)
        {gretype::RULE_REF, /* expr */ 2},
        {gretype::CHAR, '='},
        {gretype::RULE_REF, /* ws */ 3},
        {gretype::RULE_REF, /* term */ 4},
        {gretype::CHAR, '\n'},
        {gretype::END, 0},
        // expr (index 2)
        {gretype::RULE_REF, /* term */ 4},
        {gretype::RULE_REF, /* expr_7 */ 7},
        {gretype::END, 0},
        // ws (index 3)
        {gretype::RULE_REF, /* ws_12 */ 12},
        {gretype::END, 0},
        // term (index 4)
        {gretype::RULE_REF, /* ident */ 8},
        {gretype::ALT, 0},
        {gretype::RULE_REF, /* num */ 9},
        {gretype::ALT, 0},
        {gretype::CHAR, '('},
        {gretype::RULE_REF, /* ws */ 3},
        {gretype::RULE_REF, /* expr */ 2},
        {gretype::CHAR, ')'},
        {gretype::RULE_REF, /* ws */ 3},
        {gretype::END, 0},
        // root_5 (index 5)
        {gretype::RULE_REF, /* root_1 */ 1},
        {gretype::RULE_REF, /* root_5 */ 5},
        {gretype::ALT, 0},
        {gretype::END, 0},
        // expr_6 (index 6)
        {gretype::CHAR, '-'},
        {gretype::CHAR_ALT, '+'},
        {gretype::CHAR_ALT, '*'},
        {gretype::CHAR_ALT, '/'},
        {gretype::RULE_REF, /* term */ 4},
        {gretype::END, 0},
        // expr_7 (index 7)
        {gretype::RULE_REF, /* expr_6 */ 6},
        {gretype::RULE_REF, /* expr_7 */ 7},
        {gretype::ALT, 0},
        {gretype::END, 0},
        // ident (index 8)
        {gretype::CHAR, 'a'},
        {gretype::CHAR_RNG_UPPER, 'z'},
        {gretype::RULE_REF, /* ident_10 */ 10},
        {gretype::RULE_REF, /* ws */ 3},
        {gretype::END, 0},
        // num (index 9)
        {gretype::CHAR, '0'},
        {gretype::CHAR_RNG_UPPER, '9'},
        {gretype::RULE_REF, /* num_11 */ 11},
        {gretype::RULE_REF, /* ws */ 3},
        {gretype::END, 0},
        // ident_10 (index 10)
        {gretype::CHAR, 'a'},
        {gretype::CHAR_RNG_UPPER, 'z'},
        {gretype::CHAR_ALT, '0'},
        {gretype::CHAR_RNG_UPPER, '9'},
        {gretype::CHAR_ALT, '_'},
        {gretype::RULE_REF, /* ident_10 */ 10},
        {gretype::ALT, 0},
        {gretype::END, 0},
        // num_11 (index 11)
        {gretype::CHAR, '0'},
        {gretype::CHAR_RNG_UPPER, '9'},
        {gretype::RULE_REF, /* num_11 */ 11},
        {gretype::ALT, 0},
        {gretype::END, 0},
        // ws_12 (index 12)
        {gretype::CHAR, ' '},
        {gretype::CHAR_ALT, '\t'},
        {gretype::CHAR_ALT, '\n'},
        {gretype::RULE_REF, /* ws_12 */ 12},
        {gretype::ALT, 0},
        {gretype::END, 0},
    });

    // <[1000]> = "<think>"
    // <[1001]> = "</think>"
    verify_parsing(R"""(
        root  ::= <[1000]> !<[1001]> <[1001]>
    )""", {
        {"root", 0}
    }, {
        // root (index 0)
        {gretype::TOKEN, 1000},
        {gretype::TOKEN_NOT, 1001},
        {gretype::TOKEN, 1001},
        {gretype::END, 0},
    });

    test_accept_ab();
    test_token_syntax_init();
    test_mask_ab();
    test_mask_cache();
    test_load_grammar_vocab();
    std::printf("All grammar tests passed.\n");
    return 0;
}
