// include/strata/artifact/gguf_split.hpp - the shards of a split model.  Separate from gguf_reader.hpp, which
// includes <windows.h>, so any translation unit can resolve a model's shards.
//
// From eddoursul/Strata 8029fa9 (gguf_split_paths), with one change of contract: upstream's model_shards()
// skipped a shard it could not open and returned the rest, so a download that was still running (or a shard
// deleted by hand) produced a model with some layers missing and an error much later, or none.  A missing shard
// is now an error that names it.
#pragma once

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata {

// Split models: `<name>-00001-of-0000N.gguf` ... `<name>-0000N-of-0000N.gguf`, as llama.cpp's gguf-split names
// them.  Given ANY shard's path, the shards in split order (the metadata shard, 00001, first); a file without that
// suffix is a model of one shard.  Only the five digits of the shard number are rewritten, so the directory and
// the stem are the caller's own spelling (a symlinked Hugging Face snapshot keeps its split names).  `skip_shard`
// names the one-based PLE-only shard replaced by an external PLE file; all other missing shards remain errors.
inline std::string gguf_split_shard_path(const std::string& any, int shard) {
    static const char tag[] = "-of-";
    const size_t at = any.rfind(tag);
    if (at == std::string::npos || at < 6 || any[at - 6] != '-' || any.size() != at + 4 + 5 + 5 ||
        any.compare(at + 9, 5, ".gguf") != 0)
        return {};
    int count = 0;
    for (size_t i = 0; i < 5; ++i) {
        const char digit = any[at + 4 + i];
        if (digit < '0' || digit > '9') return {};
        count = count * 10 + (digit - '0');
    }
    if (count < 1 || shard < 1 || shard > count) return {};
    char num[8];
    std::snprintf(num, sizeof num, "%05d", shard);
    std::string path = any;
    path.replace(at - 5, 5, num);
    return path;
}

inline std::vector<std::string> gguf_split_paths(const std::string& any, int skip_shard = 0) {
    static const char tag[] = "-of-";
    const size_t at = any.rfind(tag);
    // "...-NNNNN-of-MMMMM.gguf": five digits before the tag, five after it, then the extension
    if (at == std::string::npos || at < 6 || any[at - 6] != '-' || any.size() != at + 4 + 5 + 5 ||
        any.compare(at + 9, 5, ".gguf") != 0)
        return {any};
    int no = 0, n = 0;
    for (size_t i = 0; i < 5; ++i) {
        const char a = any[at - 5 + i], b = any[at + 4 + i];
        if (a < '0' || a > '9' || b < '0' || b > '9') return {any};
        no = no * 10 + (a - '0');
        n = n * 10 + (b - '0');
    }
    if (n < 1 || no < 1 || no > n) {
        if (skip_shard) throw std::runtime_error("--skip-ple-shard requires a split GGUF");
        return {any};
    }
    if (skip_shard < 0 || skip_shard > n)
        throw std::runtime_error("invalid skipped model shard " + std::to_string(skip_shard) + " of " +
                                 std::to_string(n));
    if (skip_shard && n < 2)
        throw std::runtime_error("cannot skip a shard from a single-file model");
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        if (i == skip_shard) continue;
        char num[8];
        std::snprintf(num, sizeof num, "%05d", i);
        std::string p = any;
        p.replace(at - 5, 5, num);
        if (!std::ifstream(p, std::ios::binary))
            throw std::runtime_error("missing model shard " + p + " (shard " + std::to_string(i) + " of " +
                                     std::to_string(n) + "; is the download complete?)");
        out.push_back(std::move(p));
    }
    return out;
}

}  // namespace strata
