// src/artifact/safetensors_exl3_check.cpp - reconstruct one real EXL3 weight from a safetensors shard
// with the CPU reference kernel, and print a deterministic checksum for cross-checking against
// tools/exl3 (the numpy specification).  This is the real-model end-to-end path: reader -> decode ->
// Hadamard -> signs.
//
// Usage: safetensors_exl3_check <shard.safetensors> <tensor_base_name>
#include "strata/artifact/safetensors.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels::cpu;

static uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <shard.safetensors> <base_name>\n", argv[0]); return 2; }
    try {
        SafetensorsFile f(argv[1]);
        std::string base = argv[2];
        const auto* tr = f.find(base + ".trellis");
        const auto* suh = f.find(base + ".suh");
        const auto* svh = f.find(base + ".svh");
        if (!tr || !suh || !svh) { std::fprintf(stderr, "missing tensors for %s\n", base.c_str()); return 1; }
        int ki = (int)tr->shape[0], nj = (int)tr->shape[1];
        int words = (int)tr->shape[2];
        int bits = words * 16 / 256;
        Exl3Codebook cb = Exl3Codebook::ThreeInst;
        if (f.find(base + ".mul1")) cb = Exl3Codebook::Mul1;
        else if (f.find(base + ".mcg")) cb = Exl3Codebook::Mcg;
        std::printf("trellis %s [%d,%d,%d] K=%d cb=%d | suh %s[%zu] svh %s[%zu]\n",
                    tr->dtype.c_str(), ki, nj, words, bits, (int)cb,
                    suh->dtype.c_str(), suh->shape.size(), svh->dtype.c_str(), svh->shape.size());

        int k = ki * 16, n = nj * 16;
        std::vector<uint16_t> w((size_t)k * n);
        // suh/svh come as 16-bit values (fp16 or packed int16); the kernel reads fp16 bit patterns.
        const uint16_t* suh_p = (const uint16_t*)f.data(*suh);
        const uint16_t* svh_p = (const uint16_t*)f.data(*svh);
        const uint16_t* tr_p = (const uint16_t*)f.data(*tr);
        exl3_reconstruct_weight(tr_p, ki, nj, bits, cb, suh_p, svh_p, w.data());

        uint64_t h = fnv1a(w.data(), w.size() * 2);
        std::printf("reconstructed %dx%d fp16 | fnv1a=%016llx\n", k, n, (unsigned long long)h);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
