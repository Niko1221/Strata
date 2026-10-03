#include "strata/ngram/prefault.hpp"
#include <cstdio>
#include <vector>

int main() {
    const size_t page = 4096;
    std::vector<uint8_t> bytes((129ull << 20) + 17);
    uint64_t expected = 0;
    for (size_t i = 0; i < bytes.size(); i += page) {
        bytes[i] = (uint8_t) ((i / page * 71 + 3) & 255);
        expected += bytes[i];
    }
    for (unsigned threads : {1u, 4u, 8u, 16u, 999u}) {
        auto result = strata::ngram::prefault_pages(bytes.data(), bytes.size(), page, threads);
        if (result.checksum != expected || result.pages != 1 + (bytes.size() - 1) / page ||
            !result.threads || result.threads > 64) return 1;
    }
    if (strata::ngram::prefault_pages(nullptr, 0, page, 16).pages) return 2;
    if (strata::ngram::prefault_pages(bytes.data(), bytes.size(), page, 0).pages) return 3;
    // A short unaligned readable region must not read beyond its final byte.
    for (size_t size : {size_t(1), page - 1, page, page + 1}) {
        auto result = strata::ngram::prefault_pages(bytes.data() + 1, size, page, 4);
        uint64_t want = bytes[1];
        if (size > page) want += bytes[page + 1];
        if (result.checksum != want) return 4;
    }
    std::puts("prefault: parallel checksum, tails, zero length and worker clamp PASS");
}
