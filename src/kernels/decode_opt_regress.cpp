// src/kernels/decode_opt_regress.cpp - plan v0.3 P6: the verify window's expert-row readers, pinned.
//
// The host expert pool no longer zeroes a published plan's GPU-owned rows: the readers are the ONLY reason
// that is safe, and they are asserted here with POISONED host rows so a regression to "read the host row" is
// caught.  These are real CUDA kernels on mapped pinned memory, synthetic data, no model artifact:
//
//   1. copy_rows_from_mapped with a published plan's hit list: hit rows must come out +0.0 on the device
//      whatever the host row holds, and the non-hit rows must copy verbatim (bitwise).
//   2. copy_rows_from_mapped with an empty hit list (a no-plan / STRATA_DEC_BATCH=0 group): every row copies
//      verbatim - the host's zeros are the source there, so they must be load-bearing.
//   3. copy_rows_or_zero_from_mapped, the device-plan reader: skip == ring -> the whole group +0.0; skip !=
//      ring (the host plan governs) -> plan rows +0.0 on the device, the rest verbatim from the host.
//
// Every assertion is bitwise (memcmp on float buffers): these kernels are deterministic, so a float-level
// comparison would let a rounding drift through.  Run: build/decode_opt_regress
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "decode_opt_regress: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int failures = 0;
void pin(bool ok, const char* what) {
    std::printf("%-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    constexpr int rows = 6, width = 256;
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream create");

    float* host = nullptr;
    float* mdev = nullptr;
    ck(cudaHostAlloc((void**) &host, (size_t) rows * width * 4, cudaHostAllocMapped), "host rows alloc");
    ck(cudaHostGetDevicePointer((void**) &mdev, host, 0), "host rows device alias");
    // Poisoned: values a +0.0-vs-copy regression must NOT survive on hit rows.  Small integers are exact.
    for (int i = 0; i < rows * width; ++i) host[i] = (float) (1000 + i);
    float* dst1 = nullptr;
    float* dst2 = nullptr;
    float* dst3 = nullptr;
    float* dst4 = nullptr;
    ck(cudaMalloc((void**) &dst1, (size_t) rows * width * 4), "dst1 alloc");
    ck(cudaMalloc((void**) &dst2, (size_t) rows * width * 4), "dst2 alloc");
    ck(cudaMalloc((void**) &dst3, (size_t) rows * width * 4), "dst3 alloc");
    ck(cudaMalloc((void**) &dst4, (size_t) rows * width * 4), "dst4 alloc");
    const int32_t hits[3] = {1, 3, 4};   // a published plan's GPU-owned rows
    const int32_t three = 3, zero = 0;
    int32_t *d_hits = nullptr, *d_cnt = nullptr;
    uint32_t* d_skip = nullptr;
    ck(cudaMalloc((void**) &d_hits, sizeof hits), "hits alloc");
    ck(cudaMalloc((void**) &d_cnt, 8), "counts alloc");
    ck(cudaMalloc((void**) &d_skip, 4), "skip alloc");
    ck(cudaMemcpy(d_hits, hits, sizeof hits, cudaMemcpyHostToDevice), "hits upload");
    const uint32_t ring = 7;
    // (i) published plan: hit rows come out +0.0 on the device whatever the host rows hold...
    ck(cudaMemcpy(d_cnt, &three, 4, cudaMemcpyHostToDevice), "count 3 upload");
    strata::kernels::copy_rows_from_mapped(dst1, mdev, rows, width, d_hits, d_cnt, s);
    // (ii) empty hit list (a no-plan / STRATA_DEC_BATCH=0 group): every row copies verbatim
    ck(cudaMemcpy(d_cnt, &zero, 4, cudaMemcpyHostToDevice), "count 0 upload");
    strata::kernels::copy_rows_from_mapped(dst2, mdev, rows, width, d_hits, d_cnt, s);
    // (iii) the device-plan reader with the host's plan governing (skip != ring): the same hit-aware contract
    const uint32_t no_plan = 0;
    ck(cudaMemcpy(d_skip, &no_plan, 4, cudaMemcpyHostToDevice), "skip 0 upload");
    ck(cudaMemcpy(d_cnt, &three, 4, cudaMemcpyHostToDevice), "count 3 re-upload");
    strata::kernels::copy_rows_or_zero_from_mapped(dst3, mdev, rows, width, d_skip, ring, d_hits, d_cnt, s);
    // (iv) the device planned the whole group (skip == ring): every row zeroed
    ck(cudaMemcpy(d_skip, &ring, 4, cudaMemcpyHostToDevice), "skip ring upload");
    strata::kernels::copy_rows_or_zero_from_mapped(dst4, mdev, rows, width, d_skip, ring, d_hits, d_cnt, s);
    std::vector<float> got1((size_t) rows * width), got2((size_t) rows * width);
    std::vector<float> got3((size_t) rows * width), got4((size_t) rows * width);
    ck(cudaMemcpy(got1.data(), dst1, got1.size() * 4, cudaMemcpyDeviceToHost), "dst1 readback");
    ck(cudaMemcpy(got2.data(), dst2, got2.size() * 4, cudaMemcpyDeviceToHost), "dst2 readback");
    ck(cudaMemcpy(got3.data(), dst3, got3.size() * 4, cudaMemcpyDeviceToHost), "dst3 readback");
    ck(cudaMemcpy(got4.data(), dst4, got4.size() * 4, cudaMemcpyDeviceToHost), "dst4 readback");
    bool hit_ok = true, miss_ok = true, all_verbatim = true, sl_hit_ok = true, sl_zero_ok = true;
    for (int r = 0; r < rows; ++r) {
        const bool hit = r == 1 || r == 3 || r == 4;
        for (int j = 0; j < width; ++j) {
            const float want = hit ? 0.0f : (float) (1000 + r * width + j);
            if (std::memcmp(&got1[(size_t) r * width + j], &want, 4) != 0) hit_ok = false;   // bitwise +0.0
            if (std::memcmp(&got2[(size_t) r * width + j], &host[(size_t) r * width + j], 4) != 0) miss_ok = false;
            if (std::memcmp(&got3[(size_t) r * width + j], &want, 4) != 0) sl_hit_ok = false;
        }
    }
    for (int i = 0; i < rows * width; ++i) {
        if (std::memcmp(&got2[i], &host[i], 4) != 0) all_verbatim = false;
        if (got4[(size_t) i] != 0.0f) sl_zero_ok = false;
    }
    pin(hit_ok, "1. published-plan rows: hit rows +0.0 on the device despite poisoned host values");
    pin(miss_ok, "1. published-plan rows: non-hit rows copied bitwise from the host");
    pin(all_verbatim, "1. empty hit list (no plan / STRATA_DEC_BATCH=0): every row copies verbatim");
    pin(sl_hit_ok, "1. device-plan reader (skip != ring): plan rows +0.0 on the device, host rows verbatim");
    pin(sl_zero_ok, "1. device-plan reader (skip == ring): the whole group's rows +0.0");
    ck(cudaFreeHost(host), "host rows free");
    ck(cudaFree(dst1), "dst1 free");
    ck(cudaFree(dst2), "dst2 free");
    ck(cudaFree(dst3), "dst3 free");
    ck(cudaFree(dst4), "dst4 free");
    ck(cudaFree(d_hits), "hits free");
    ck(cudaFree(d_cnt), "counts free");
    ck(cudaFree(d_skip), "skip free");
    ck(cudaStreamDestroy(s), "stream destroy");

    std::printf("decode_opt_regress: %d failures\n", failures);
    return failures ? 1 : 0;
}
