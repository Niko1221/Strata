// include/strata/artifact/gguf_split.hpp - the shards of a split model.  Separate from gguf_reader.hpp, which
// includes <windows.h>, so any translation unit can resolve a model's shards.
#pragma once

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata {

// Split models: `<name>-00001-of-0000N.gguf` ... `<name>-0000N-of-0000N.gguf`, as llama.cpp's gguf-split
// names them.  The shards in split order, from the first one's path; a file without that suffix is a model of
// one shard.  Throws when a shard is missing.
inline std::vector<std::string> gguf_split_paths(const std::string& first) {
    static const char pat[] = "-00001-of-";
    const size_t at = first.rfind(pat);
    const size_t digits = at == std::string::npos ? 0 : at + sizeof(pat) - 1;
    if (at == std::string::npos || first.size() != digits + 10 || first.compare(digits + 5, 5, ".gguf") != 0)
        return {first};
    int n = 0;
    for (size_t i = digits; i < digits + 5; ++i) {
        if (first[i] < '0' || first[i] > '9') return {first};
        n = n * 10 + (first[i] - '0');
    }
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char no[8];
        std::snprintf(no, sizeof no, "%05d", i);
        std::string p = first.substr(0, at + 1) + no + first.substr(at + 6);
        if (!std::ifstream(p, std::ios::binary)) throw std::runtime_error("missing model shard " + p);
        out.push_back(std::move(p));
    }
    return out;
}

}  // namespace strata
