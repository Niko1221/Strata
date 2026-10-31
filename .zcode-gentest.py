import io

src = io.open(r"F:/Strata/third_party/llama.cpp/tests/test-grammar-parser.cpp", encoding="utf-8").read()

start = src.index("static const char * type_str")
helpers = src[start:src.index("int main()")]
main_all = src[src.index("int main()"):]
main_inner = main_all[main_all.index("{") + 1:main_all.rindex("return 0;")]

pairs = [
    ("llama_grammar_element", "grammar_element"),
    ("llama_grammar_parser", "grammar_parser"),
    ("llama_gretype", "gretype"),
    ("LLAMA_GRETYPE_", "gretype::"),
    ("fprintf(stderr", "std::fprintf(stderr"),
    ("printf(", "std::printf("),
]
for a, b in pairs:
    helpers = helpers.replace(a, b)
    main_inner = main_inner.replace(a, b)

accept_tests = r"""
// accept over the piece code points: init -> accept 'a','b' -> a stack is at the end -> accept 'c' throws
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
        grammar_accept(g, (uint32_t) 'c');
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw && "accepting a char past the grammar must throw");
    grammar_free(g);
}

// the token-syntax rule needs no vocab: <[id]> parses numerically
static void test_token_syntax_init(void) {
    grammar * g = grammar_init(nullptr, "root ::= <[1000]> <[1001]>\n", "root");
    assert(g != nullptr);
    grammar_free(g);
}

"""

out = (
    "// src/grammar/grammar_test.cpp - parser goldens ported from llama.cpp "
    "tests/test-grammar-parser.cpp (MIT), plus accept smoke tests.\n"
    "#ifdef NDEBUG\n#undef NDEBUG\n#endif\n\n"
    "#include \"strata/grammar/grammar.hpp\"\n\n"
    "#include <cassert>\n#include <cstdio>\n#include <map>\n#include <stdexcept>\n#include <string>\n#include <vector>\n\n"
    "using namespace strata::grammar;\n\n"
    + helpers
    + accept_tests
    + "int main(void)\n{\n" + main_inner
    + "    test_accept_ab();\n    test_token_syntax_init();\n"
    + "    std::printf(\"All grammar tests passed.\\n\");\n    return 0;\n}\n"
)

io.open(r"F:/Strata-dev/src/grammar/grammar_test.cpp", "w", encoding="utf-8", newline="\n").write(out)
print("grammar_test.cpp generated:", len(out), "bytes")
