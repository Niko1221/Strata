// src/qwen35/q36_main.cpp - strata-q36: the Qwen3.6 engine path on its own (the same entry point `strata` switches
// to for a qwen35moe model).
#include "strata/qwen35/engine.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    // Match strata's metadata query before model validation or GPU initialisation.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-V") == 0) {
            std::printf("strata-q36 %s\n", STRATA_VERSION);
            return 0;
        }
    }
    return strata::qwen35::run(argc, argv);
}
