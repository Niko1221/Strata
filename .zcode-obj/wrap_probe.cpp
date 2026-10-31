// wrap_probe <grammar-file> <text-file-with-one-segment-per-line>
// Walks each line's code points through a fresh grammar (grammar_accept chr variant).
// Prints OK or THROW per line index. Lines starting with '#' are skipped.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include "strata/grammar/grammar.hpp"

using namespace strata::grammar;

int main(int argc, char** argv) {
    std::ifstream gf(argv[1], std::ios::binary);
    std::stringstream gs;
    gs << gf.rdbuf();
    std::string gtext = gs.str();

    std::ifstream tf(argv[2], std::ios::binary);
    std::string line;
    int idx = 0;
    while (std::getline(tf, line)) {
        if (!line.empty() && line[0] == '#') continue;
        grammar* g = grammar_init(nullptr, gtext.c_str(), "root");
        if (!g) { std::printf("%d PARSE-FAIL\n", idx); ++idx; continue; }
        const char* p = line.c_str();
        const char* end = p + line.size();
        const char* res = nullptr;
        while (p < end) {
            uint32_t cp = (unsigned char) *p;
            int extra = 0;
            if (cp >= 0xF0) extra = 3; else if (cp >= 0xE0) extra = 2; else if (cp >= 0xC0) extra = 1;
            if (cp >= 0x80) {  // already a code point start: decode UTF-8 minimally
                if (extra == 0) { res = "bad-utf8"; break; }
                uint32_t v = cp & (extra == 3 ? 0x07u : extra == 2 ? 0x0Fu : 0x1Fu);
                for (int k = 1; k <= extra && p + k < end; ++k)
                    v = (v << 6) | ((unsigned char) p[k] & 0x3F);
                cp = v;
                p += extra;
            }
            ++p;
            try {
                grammar_accept(g, cp);
            } catch (const std::exception& e) {
                std::stringstream es;
                es << e.what();
                res = es.str().c_str();
                static std::string keep;
                keep = e.what();
                res = keep.c_str();
                break;
            }
        }
        if (res) std::printf("%d THROW %s\n", idx, res);
        else std::printf("%d OK\n", idx);
        grammar_free(g);
        ++idx;
    }
    return 0;
}
