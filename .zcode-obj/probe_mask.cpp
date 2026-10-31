#include "strata/grammar/grammar.hpp"
#include "strata/grammar/grammar_mask.hpp"
#include <cstdio>
#include <string>
#include <vector>
static void dump(const strata::grammar::GrammarVocab& vocab, strata::grammar::grammar* g, const char* name) {
    const auto& rules = strata::grammar::grammar_get_rules(g);
    std::printf("state '%s' stacks:\n", name);
    for (auto& s : strata::grammar::grammar_stacks(g)) {
        std::printf("  n=%zu:", s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            int off = -1;
            for (size_t r = 0; r < rules.size(); ++r)
                if (!rules[r].empty() && s[i] >= rules[r].data() && s[i] < rules[r].data() + rules[r].size())
                    off = (int) (s[i] - rules[r].data());
            std::printf(" r%d@%d(%d:%u)", -1, off, (int) s[i]->type, s[i]->value);
        }
        std::printf("\n");
    }
    std::vector<uint32_t> m(strata::grammar::grammar_mask_words(3), 0xFFFFFFFFu);
    strata::grammar::grammar_build_mask(vocab, rules, strata::grammar::grammar_stacks(g), m.data());
    std::printf("direct mask = %u\n", m[0]);
}
int main() {
    strata::grammar::GrammarVocab vocab;
    vocab.pieces = {"a", "b", "<|eog|>"};
    vocab.control = {0, 0, 1};
    vocab.eog = {0, 0, 1};
    strata::grammar::grammar* g = strata::grammar::grammar_init(&vocab, "root ::= pair*\npair ::= \"ab\" | \"ba\"\n", "root");
    if (!g) { std::printf("parse failed\n"); return 1; }
    const auto& rules = strata::grammar::grammar_get_rules(g);
    std::printf("rules=%zu\n", rules.size());
    for (size_t r = 0; r < rules.size(); ++r) {
        std::printf("rule %zu len %zu:", r, rules[r].size());
        for (size_t i = 0; i < rules[r].size(); ++i) std::printf(" [%d:%u]", (int) rules[r][i].type, rules[r][i].value);
        std::printf("\n");
    }
    dump(vocab, g, "root");
    strata::grammar::grammar_accept(*g, 1, "b");
    dump(vocab, g, "b");
    strata::grammar::grammar_accept(*g, 0, "a");
    dump(vocab, g, "ba");
    strata::grammar::grammar_accept(*g, 0, "a");
    dump(vocab, g, "baa");
    strata::grammar::grammar_free(g);
    return 0;
}
