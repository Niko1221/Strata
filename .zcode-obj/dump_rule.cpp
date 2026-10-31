// dump_rule <grammar-file> <rule-name-prefix> : print the parsed elements of matching rules
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "strata/grammar/grammar.hpp"

using namespace strata::grammar;

static void show(uint32_t v) {
    if (v >= 0x20 && v < 0x7F) std::printf("%c", (char) v);
    else std::printf("\\x%02x", v);
}

int main(int argc, char** argv) {
    std::ifstream f(argv[1], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    grammar* g = grammar_init(nullptr, text.c_str(), "root");
    if (!g) { std::printf("parse FAIL\n"); return 1; }
    const grammar_rules& rules = grammar_get_rules(g);
    const std::string want = argv[2];
    std::vector<std::string> names;   // rule id -> name is not exposed; scan by trying: the parser
    // keeps symbol_ids internally; instead print ALL rules whose first element sequence we can label
    // by index: rules are stored in creation order, so we print the index and the elements.
    for (size_t r = 0; r < rules.size(); ++r) {
        std::printf("rule %zu (want-prefix %s):", r, want.c_str());
        for (const grammar_element* e = rules[r].data(); e->type != gretype::END; ++e) {
            switch (e->type) {
            case gretype::END: break;
            case gretype::ALT: std::printf(" |"); break;
            case gretype::RULE_REF: std::printf(" R%u", e->value); break;
            case gretype::CHAR: std::printf(" '"); show(e->value); std::printf("'"); break;
            case gretype::CHAR_NOT: std::printf(" [^'"); show(e->value); std::printf("'"); break;
            case gretype::CHAR_RNG_UPPER: std::printf(" -'"); show(e->value); std::printf("'"); break;
            case gretype::CHAR_ALT: std::printf(" ,'"); show(e->value); std::printf("'"); break;
            case gretype::CHAR_ANY: std::printf(" ."); break;
            default: std::printf(" T%u", e->type); break;
            }
        }
        std::printf("\n");
        if (r > 40) break;
    }
    grammar_free(g);
    return 0;
}
