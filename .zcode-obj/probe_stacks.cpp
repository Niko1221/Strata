// offline replay: grammar + vocab (pieces json) + token ids -> stacks + mask per step
#include "strata/grammar/grammar.hpp"
#include "strata/grammar/grammar_mask.hpp"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace strata::grammar;

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: probe_stacks g.txt pieces.json id...\n"); return 2; }
    std::ifstream gf(argv[1]);
    if (!gf) { fprintf(stderr, "open fail: %s\n", argv[1]); return 2; }
    std::ostringstream ss;
    ss << gf.rdbuf();
    std::string text = ss.str();
    fprintf(stderr, "grammar text bytes=%zu\n", text.size());
    std::ifstream pf(argv[2]);
    if (!pf) { fprintf(stderr, "open fail: %s\n", argv[2]); return 2; }
    nlohmann::json j;
    pf >> j;
    fprintf(stderr, "pieces parsed=%zu\n", j.size());
    GrammarVocab v;
    for (auto& e : j) {
        v.pieces.push_back(e.at("p").get<std::string>());
        v.control.push_back((char) e.at("c").get<int>());
        v.eog.push_back((char) e.at("e").get<int>());
    }
    grammar* g = grammar_init(&v, text.c_str(), "root");
    if (g == nullptr) { fprintf(stderr, "grammar_init FAILED\n"); return 1; }
    fprintf(stderr, "grammar_init OK rules=%zu stacks=%zu\n", grammar_get_rules(g).size(), grammar_stacks(g).size());
    const grammar_rules& rules = grammar_get_rules(g);
    auto rule_of = [&](const grammar_element* el) -> int {
        for (size_t r = 0; r < rules.size(); ++r) {
            if (el >= rules[r].data() && el < rules[r].data() + rules[r].size()) {
                return (int) (el - rules[r].data()) + 10000 * (int) r;
            }
        }
        return -1;
    };
    auto dump = [&](const char* tag) {
        std::vector<uint32_t> mask(grammar_mask_words((int64_t) v.pieces.size()), 0u);
        grammar_build_mask(v, rules, grammar_stacks(g), mask.data());
        int words = (int) mask.size();
        long allowed = 0;
        int first = -1;
        for (int i = 0; i < (int) v.pieces.size(); ++i) {
            if (!(mask[(size_t) i >> 5] & (1u << (i & 31)))) { ++allowed; if (first < 0) first = i; }
        }
        printf("%s: stacks=%zu", tag, grammar_stacks(g).size());
        for (auto& st : grammar_stacks(g)) {
            int ri = -1;
            const char* tn = "?";
            if (!st.empty()) {
                ri = rule_of(st.back());
                static const char* ty[] = {"END","ALT","RULE_REF","CHAR","CHAR_NOT","CHAR_RNG_UPPER","CHAR_ALT","CHAR_ANY","TOKEN","TOKEN_NOT"};
                tn = ty[(int) st.back()->type];
            }
            printf(" [r=%d elem=%d %s:%u]", ri / 10000, ri % 10000, tn, st.empty() ? (unsigned) 0 : (st.empty() ? (unsigned) 0 : st.back()->value));
        }
        printf(" | allowed=%d first=%d words=%d\n", (int) allowed, first, words);
        fflush(stdout);
    };
    dump("init");
    for (int a = 3; a < argc; ++a) {
        int tok = atoi(argv[a]);
        try {
            grammar_accept(*g, tok, v.token_to_piece(tok));
        } catch (const std::exception& e) {
            printf("accept %d THREW: %s\n", tok, e.what());
            break;
        }
        char tag[64];
        snprintf(tag, sizeof tag, "after %d", tok);
        dump(tag);
    }
    grammar_free(g);
    return 0;
}
