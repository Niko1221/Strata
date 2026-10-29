// Experimental argv adapter. Links the unchanged, already-built engine libraries.
#include "strata/qwen35/engine.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--tokens-file" || arg == "--force-file") {
            if (++i >= argc) { std::cerr << "missing argument file\n"; return 2; }
            std::ifstream file(argv[i], std::ios::binary);
            if (!file) { std::cerr << "cannot open argument file\n"; return 2; }
            std::string contents((std::istreambuf_iterator<char>(file)), {});
            if (contents.empty() || contents.size() > 16 * 1024 * 1024) {
                std::cerr << "empty or oversized argument file\n"; return 2;
            }
            for (char& c : contents) if (c == '\r' || c == '\n' || c == '\t') c = ' ';
            args.push_back(arg == "--tokens-file" ? "--tokens" : "--force");
            args.push_back(std::move(contents));
        } else args.push_back(std::move(arg));
    }
    std::vector<char*> pointers;
    for (auto& arg : args) pointers.push_back(arg.data());
    pointers.push_back(nullptr);
    return strata::qwen35::run(static_cast<int>(args.size()), pointers.data());
}
