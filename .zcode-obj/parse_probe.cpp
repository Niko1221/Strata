// probe: which rule of the think prologue grammar does the ported parser reject?
#include "strata/grammar/grammar.hpp"
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
int main(int argc, char** argv) {
    std::ifstream f(argv[1], std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    std::string text = ss.str();
    strata::grammar::grammar* g = strata::grammar::grammar_init(nullptr, text.c_str(), "root");
    std::printf("parse: %s\n", g ? "OK" : "FAIL");
    return g ? 0 : 1;
}
