// prefill_gather_batch_test - mmq::gather_native_batch against one mmq::gather_native per expert, byte for byte.  The
// prompt path launches a group's expert gathers together (prefill.cpp), so the group buffers must hold exactly what
// the per-expert launches wrote: batches of 1..16 experts with blobs laid out as a native pack's (gate, up, down at
// their offsets), group positions that start mid-group (a batch launched early, before its group was full), and the
// unaligned case, which falls back to one launch per expert.  Synthetic bytes, no model, a second or two.
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    uint8_t* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc((void**) &p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

// `n` experts from group position `q0`; `shift` moves every source by that many bytes (1: the unaligned fallback)
void check(cudaStream_t s, const char* name, size_t gu_half, size_t d_bytes, int n, int q0, size_t shift, unsigned seed) {
    const size_t blob = (2 * gu_half + d_bytes + 255) / 256 * 256 + 256;   // a slot of the arena, with room to shift
    const size_t gub = 2 * gu_half;
    Dev src(blob * (size_t) n), a_gu(gub * mmq::kGatherMax), a_d(d_bytes * mmq::kGatherMax),
        b_gu(gub * mmq::kGatherMax), b_d(d_bytes * mmq::kGatherMax);
    std::mt19937 rng(seed);
    std::vector<uint8_t> h(blob * (size_t) n);
    for (auto& x : h) x = (uint8_t) rng();
    ck(cudaMemcpy(src.p, h.data(), h.size(), cudaMemcpyHostToDevice), "upload");
    // both destinations start from the same bytes, so whatever a launch did not write compares equal too
    std::vector<uint8_t> fill_gu(gub * mmq::kGatherMax), fill_d(d_bytes * mmq::kGatherMax);
    for (auto& x : fill_gu) x = (uint8_t) rng();
    for (auto& x : fill_d) x = (uint8_t) rng();
    for (uint8_t* p : {a_gu.p, b_gu.p}) ck(cudaMemcpy(p, fill_gu.data(), fill_gu.size(), cudaMemcpyHostToDevice), "fill");
    for (uint8_t* p : {a_d.p, b_d.p}) ck(cudaMemcpy(p, fill_d.data(), fill_d.size(), cudaMemcpyHostToDevice), "fill");
    mmq::GatherBatch b;
    for (int i = 0; i < n; ++i) {
        const uint8_t* base = src.p + blob * (size_t) i + shift;
        const int q = q0 + i;
        mmq::gather_native(base, base + gu_half, gu_half, base + gub, d_bytes, a_gu.p + gub * q, a_d.p + d_bytes * q, s);
        b.gate[b.n] = base;
        b.up[b.n] = base + gu_half;
        b.down[b.n] = base + gub;
        b.gu_dst[b.n] = b_gu.p + gub * q;
        b.d_dst[b.n] = b_d.p + d_bytes * q;
        ++b.n;
    }
    mmq::gather_native_batch(b, gu_half, d_bytes, s);
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<uint8_t> x(fill_gu.size()), y(fill_gu.size()), u(fill_d.size()), v(fill_d.size());
    ck(cudaMemcpy(x.data(), a_gu.p, x.size(), cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(y.data(), b_gu.p, y.size(), cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(u.data(), a_d.p, u.size(), cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(v.data(), b_d.p, v.size(), cudaMemcpyDeviceToHost), "download");
    if (x != y || u != v) throw std::runtime_error(std::string(name) + ": the batch differs from one gather per expert");
    // and the copies are the blobs' bytes (not merely equal to each other)
    for (int i = 0; i < n; ++i) {
        const uint8_t* hb = h.data() + blob * (size_t) i + shift;
        const int q = q0 + i;
        if (std::memcmp(y.data() + gub * q, hb, gub) != 0 || std::memcmp(v.data() + d_bytes * q, hb + gub, d_bytes) != 0)
            throw std::runtime_error(std::string(name) + ": expert " + std::to_string(i) + " is not its blob");
    }
    std::printf("  %-34s %2d experts from position %2d: identical\n", name, n, q0);
}

}  // namespace

int main() {
    try {
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        // an IQ3_XXS layer's sizes (gate/up 640 rows x 980 B each half, down 2,560 x 260 B) and a small odd one
        const size_t gu_half = 640 * 980, d_bytes = 2560 * 260;
        unsigned seed = 1;
        for (const int n : {1, 2, 5, 15, 16}) check(s, "IQ3_XXS-sized, aligned", gu_half, d_bytes, n, 0, 0, seed++);
        check(s, "IQ3_XXS-sized, mid-group", gu_half, d_bytes, 7, 9, 0, seed++);
        check(s, "unaligned sources (fallback)", gu_half, d_bytes, 4, 3, 1, seed++);
        check(s, "sizes not a multiple of 16 (fallback)", 1000 + 8, 3000 + 4, 6, 2, 0, seed++);
        check(s, "small, aligned", 4096, 2048, 16, 0, 0, seed++);
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill gather batch: identical to one gather per expert\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill gather batch test failed: %s\n", e.what());
        return 1;
    }
}
