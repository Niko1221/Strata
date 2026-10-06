// Real CUDA VMM functional test. Small allocations; no pressure generator.
#include "strata/core/expert_cache.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr int64_t MiB = 1ll << 20;
    std::string err;
    strata::core::ExpertCache cache;
    int checks = 0;
    auto check = [&](bool ok, const char* name) {
        ++checks;
        if (!ok) std::fprintf(stderr, "FAIL %s: %s\n", name, err.c_str());
        return ok;
    };
    if (!cache.open_sized_elastic({MiB / 2, MiB / 2, MiB}, 4, 8, 8 * MiB, 2 * MiB, err)) {
        if (err.find("the driver has no virtual memory management") != std::string::npos) {
            std::fprintf(stderr, "VMM unsupported: skipped (%s)\n", err.c_str());
            return 77;
        }
        std::fprintf(stderr, "VMM unavailable: %s\n", err.c_str());
        return 1; // VMM exists: an allocation or implementation failure must fail.
    }
    if (!check(cache.elastic() && cache.slots() == 3 && cache.bytes() == 2 * MiB, "initial size")) return 1;
    if (!check(cache.elastic_mapped_bytes() == 2 * MiB, "initial physical map")) return 1;
    auto* base = cache.device_slot(0);
    const auto* offsets = cache.slot_offsets();
    std::vector<std::vector<uint8_t>> initial(3);
    for (int i = 0; i < 3; ++i) {
        initial[i].assign(i == 2 ? MiB : MiB / 2, (uint8_t) (17 + i));
        if (!check(cache.fill_slot_blocking(i, initial[i].data(), err, (int64_t) initial[i].size()), "fill initial")) return 1;
    }
    if (!check(cache.elastic_grow({MiB, MiB, MiB}, err), "grow across physical chunks")) return 1;
    if (!check(cache.slots() == 6 && cache.bytes() == 5 * MiB && cache.elastic_mapped_bytes() == 6 * MiB, "grown geometry")) return 1;
    if (!check(cache.device_slot(0) == base && cache.slot_offsets() == offsets, "stable base and offset table")) return 1;
    for (int i = 0; i < 3; ++i)
        if (!check(cache.verify_slot(i, initial[i].data(), err, (int64_t) initial[i].size()), "existing expert preserved")) return 1;
    std::vector<uint8_t> zeros(MiB, 0);
    for (int i = 3; i < 6; ++i)
        if (!check(cache.verify_slot(i, zeros.data(), err, MiB), "new expert zeroed")) return 1;
    if (!check(!cache.elastic_grow({8 * MiB}, err), "reject growth past address reservation")) return 1;
    if (!check(cache.slots() == 6 && cache.bytes() == 5 * MiB && cache.elastic_mapped_bytes() == 6 * MiB,
               "rejected growth leaves physical mapping unchanged")) return 1;
    if (!check(!cache.elastic_grow(std::vector<int64_t>(40, 256), err), "reject excess expert pairs")) return 1;
    if (!check(cache.elastic_grow({}, err), "empty growth")) return 1;
    if (!check(!cache.elastic_grow({0}, err) && !cache.elastic_grow({-256}, err), "reject nonpositive slot sizes")) return 1;
    if (!check(!cache.elastic_shrink(-1, err) && !cache.elastic_shrink(7, err), "reject invalid shrink")) return 1;
    if (!check(cache.elastic_shrink(2, err), "shrink removes whole chunks")) return 1;
    if (!check(cache.slots() == 2 && cache.bytes() == MiB && cache.elastic_mapped_bytes() == 2 * MiB, "shrunk geometry")) return 1;
    if (!check(cache.device_slot(0) == base && cache.slot_offsets() == offsets, "shrink retains pointers")) return 1;
    for (int i = 0; i < 2; ++i)
        if (!check(cache.verify_slot(i, initial[i].data(), err, (int64_t) initial[i].size()), "shrink preserves remaining bytes")) return 1;
    if (!check(cache.elastic_grow({MiB}, err), "regrow recycled physical map")) return 1;
    if (!check(cache.verify_slot(2, zeros.data(), err, MiB), "regrown bytes zeroed")) return 1;
    if (!check(cache.elastic_shrink(0, err) && cache.elastic_mapped_bytes() == 0, "shrink to empty")) return 1;
    if (!check(cache.elastic_grow({MiB}, err) && cache.device_slot(0) == base, "grow from empty same address")) return 1;
    if (!check(cache.verify_slot(0, zeros.data(), err, MiB), "grow from empty zeroed")) return 1;
    cache.close();
    if (!check(!cache.elastic() && !cache.valid() && cache.slots() == 0 && cache.elastic_mapped_bytes() == 0, "close resets state")) return 1;
    if (!check(cache.open_sized({1024}, 1, 1, err) && !cache.elastic(), "fixed cache after elastic close")) return 1;
    if (!check(!cache.elastic_grow({1024}, err) && !cache.elastic_shrink(0, err), "fixed cache rejects resizing")) return 1;
    if (!check(!cache.open_sized_elastic({0}, 1, 1, 8 * MiB, 2 * MiB, err) &&
               !cache.open_sized_elastic({-256}, 1, 1, 8 * MiB, 2 * MiB, err), "reject nonpositive initial slots")) return 1;
    if (!check(!cache.open_sized_elastic({256, 256}, 1, 1, 8 * MiB, 2 * MiB, err), "reject invalid initial pair count")) return 1;
    std::printf("elastic VMM: %d checks passed; largest successful physical mapping 6 MiB\n", checks);
    return 0;
}
