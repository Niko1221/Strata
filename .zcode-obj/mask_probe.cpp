// mask_probe <grammar-file> <pieces-json>
// pieces-json: [{"p":"\n","c":1,"e":1}, ...]  - feeds token pieces through grammar_accept,
// prints per-accept masks (bit=0 allowed ids; ids: 0 nl, 1 im_end, 2 backtick, 3 brace).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include "strata/grammar/grammar.hpp"
#include "strata/grammar/grammar_mask.hpp"

using namespace strata::grammar;

int main(int argc, char** argv) {
    std::ifstream gf(argv[1], std::ios::binary);
    std::stringstream gs;
    gs << gf.rdbuf();
    std::string gtext = gs.str();

    std::ifstream pf(argv[2], std::ios::binary);
    std::stringstream ps;
    ps << pf.rdbuf();
    nlohmann::json pj = nlohmann::json::parse(ps.str());

    GrammarVocab vocab;
    for (auto& e : pj) {
        vocab.pieces.push_back(e.at("p").get<std::string>());
        vocab.control.push_back((char) e.at("c").get<int>());
        vocab.eog.push_back((char) e.at("e").get<int>());
    }
    grammar* g = grammar_init(&vocab, gtext.c_str(), "root");
    if (!g) { printf("PARSE FAIL\n"); return 1; }
    grammar_rules rules = grammar_get_rules(g);
    GrammarMaskCache cache(vocab, rules);
    const int n = (int) vocab.pieces.size();
    int words = (n + 31) / 32;
    printf("start:");
    {
        const std::vector<uint32_t>& m = cache.mask_for(grammar_stacks(g), rules);
        for (int id = 0; id < n; id++) printf(" %d=%d", id, (m[(size_t) id / 32] >> (id % 32)) & 1 ? 0 : 1);
        printf("\n");
    }
    // feed the token pieces in order, one per argv[3..] index
    for (int a = 3; a < argc; a++) {
        int id = atoi(argv[a]);
        try {
            grammar_accept(*g, id, vocab.pieces[(size_t) id]);
        } catch (const std::exception& e) {
            printf("accept %d THROW: %s\n", id, e.what());
            return 2;
        }
        const std::vector<uint32_t>& m = cache.mask_for(grammar_stacks(g), rules);
        printf("after %d '%s':", id, vocab.pieces[(size_t) id].c_str());
        for (int i = 0; i < words; i++) printf(" %08X", m[(size_t) i]);
        for (int id2 = 0; id2 < n; id2++) printf(" allow%d=%d", id2, (m[(size_t) id2 / 32] >> (id2 % 32)) & 1 ? 0 : 1);
        printf("\n");
    }
    grammar_free(g);
    return 0;
}
