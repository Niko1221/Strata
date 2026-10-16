// src/artifact/exl3_model_check.cpp - reconstruct EXL3 experts from the multi-shard model and print a
// deterministic checksum to compare against tools/exl3/check_model.py.  CPU-only.
//
// Usage:
//   exl3_model_check <model_dir> <base_name>
//   exl3_model_check <model_dir> --layer <L> --role <down_proj|gate_proj|up_proj> --count <N>
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels::cpu;

static uint64_t fnv1a(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t check_one(const Exl3Model& m, const std::string& base) {
    Exl3Linear L = m.linear(base);
    std::vector<uint16_t> w((size_t)L.in() * L.out());
    exl3_reconstruct_weight(L.trellis, L.ki, L.nj, L.bits, (Exl3Codebook)L.cb, L.suh, L.svh, w.data());
    std::printf("  %s: trellis[%d,%d] K=%d cb=%d -> %dx%d\n", base.c_str(), L.ki, L.nj, L.bits, L.cb,
                L.in(), L.out());
    return fnv1a(1469598103934665603ull, w.data(), w.size() * 2);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <dir> <base> | <dir> --layer L --role R --count N\n", argv[0]); return 2; }
    try {
        Exl3Model m(argv[1]);
        std::printf("model: %zu tensors\n", m.tensor_count());
        uint64_t h = 1469598103934665603ull;
        if (std::string(argv[2]) == "--layer") {
            int layer = std::stoi(argv[3]);
            std::string role = argv[5];
            int count = std::stoi(argv[7]);
            for (int e = 0; e < count; ++e) {
                std::string base = "model.language_model.layers." + std::to_string(layer) +
                                   ".mlp.experts." + std::to_string(e) + "." + role;
                Exl3Linear L = m.linear(base);
                std::vector<uint16_t> w((size_t)L.in() * L.out());
                exl3_reconstruct_weight(L.trellis, L.ki, L.nj, L.bits, (Exl3Codebook)L.cb, L.suh, L.svh, w.data());
                h = fnv1a(h, w.data(), w.size() * 2);
            }
            std::printf("layer %d %s x%d combined fnv1a=%016llx\n", layer, role.c_str(), count,
                        (unsigned long long)h);
        } else {
            h = check_one(m, argv[2]);
            std::printf("fnv1a=%016llx\n", (unsigned long long)h);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
