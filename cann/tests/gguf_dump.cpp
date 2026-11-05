// cann/tests/gguf_dump.cpp - print what a GGUF file declares, without reading the weights.
//
// The metadata and the tensor index sit at the front of the file, so this works on a partially
// downloaded model: useful for checking a new architecture before the weights have arrived.
//
//   gguf_dump <model.gguf> [name filter]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strata/cann/gguf.hpp"

using namespace strata::cann;

namespace {

const char * type_name(uint32_t t) {
    switch (t) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 42: return "Q2_0";
        default: return "?";
    }
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        printf("usage: gguf_dump <model.gguf> [name filter]\n");
        return 2;
    }
    try {
        const Gguf g(argv[1]);
        printf("file      %s\n", argv[1]);
        printf("version   %u\n", g.version());
        printf("arch      %s\n", g.arch().c_str());
        printf("tensors   %zu\n", g.tensors().size());

        std::vector<std::pair<std::string, int64_t>> ints(g.int_metadata().begin(), g.int_metadata().end());
        std::sort(ints.begin(), ints.end());
        printf("\n-- integer metadata (%zu) --\n", ints.size());
        for (const auto & kv : ints) {
            printf("%-52s %lld\n", kv.first.c_str(), (long long) kv.second);
        }
        std::vector<std::pair<std::string, std::string>> strs(g.str_metadata().begin(), g.str_metadata().end());
        std::sort(strs.begin(), strs.end());
        printf("\n-- string metadata (%zu) --\n", strs.size());
        for (const auto & kv : strs) {
            printf("%-52s %.120s\n", kv.first.c_str(), kv.second.c_str());
        }

        // type histogram, plus the first tensors of every block so the layer layout is visible
        int counts[64] = { 0 };
        uint64_t elems[64] = { 0 };
        for (const auto & t : g.tensors()) {
            const uint32_t ty = t.type < 64 ? t.type : 0;
            counts[ty] += 1;
            elems[ty] += t.elements();
        }
        printf("\n-- types --\n");
        for (int i = 0; i < 64; ++i) {
            if (counts[i]) {
                printf("%-8s %5d tensors  %8.2f G elements\n", type_name((uint32_t) i), counts[i],
                       (double) elems[i] / 1e9);
            }
        }

        const char * filter = argc > 2 ? argv[2] : nullptr;
        printf("\n-- tensors%s --\n", filter ? " (filtered)" : "");
        size_t shown = 0;
        for (const auto & t : g.tensors()) {
            if (filter && t.name.find(filter) == std::string::npos) {
                continue;
            }
            printf("%-46s %-8s", t.name.c_str(), type_name(t.type));
            for (size_t d = 0; d < t.dims.size() && d < 4; ++d) {
                printf(" %6llu", (unsigned long long) t.dims[d]);
            }
            printf("\n");
            if (++shown >= 4000) {
                printf("... truncated\n");
                break;
            }
        }
        printf("\n%d tensors shown\n", (int) shown);
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
