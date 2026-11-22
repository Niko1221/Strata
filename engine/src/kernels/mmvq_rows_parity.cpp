// src/kernels/mmvq_rows_parity.cpp - STRATA_MTP_HEAD_INDEX: native_mmvq_rows (the draft head's rows read in the main head
// by index) against what the drafter ran before, gather_rows into a copy then native_mmvq over it - bit for bit.
// Synthetic Q5_K matrices, no model.  Both single-column kernel shapes: n_in 2560 (Qwen3.8-Flash-Next's head, one row a
// block) and n_in 1024 (four rows a block), ids in unsorted runs with the first and last row among them.
// Exit 77 without a CUDA device, or on a backend without native_mmvq_rows (HIP).
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

void run(int n_in, int n_vocab, int n_ids, uint32_t seed) {
    std::mt19937 rng(seed);
    const int row_bytes = n_in / 256 * 176;
    std::vector<uint8_t> head((size_t) n_vocab * row_bytes);
    for (auto& b : head) b = (uint8_t) rng();
    for (size_t blk = 0; blk < head.size() / 176; ++blk) {   // finite scales: d and dmin in [2^-10, 2^-7)
        const uint16_t d = (uint16_t) (0x1400 + rng() % 0xC00), dmin = (uint16_t) (0x1400 + rng() % 0xC00);
        std::memcpy(&head[blk * 176], &d, 2);
        std::memcpy(&head[blk * 176 + 2], &dmin, 2);
    }
    std::vector<int32_t> ids;
    ids.push_back(n_vocab - 1);
    while ((int) ids.size() < n_ids - 1) {   // runs of 1-12 rows from random starts, like draft_vocab.bin
        const int start = (int) (rng() % (uint32_t) n_vocab), len = 1 + (int) (rng() % 12);
        for (int i = 0; i < len && start + i < n_vocab && (int) ids.size() < n_ids - 1; ++i) ids.push_back(start + i);
    }
    ids.push_back(0);
    std::vector<float> x((size_t) n_in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng);

    uint8_t *dhead = nullptr, *dcopy = nullptr, *dxq = nullptr;
    int32_t* dids = nullptr;
    float *dx = nullptr, *dy_copy = nullptr, *dy_rows = nullptr;
    ck(cudaMalloc((void**) &dhead, head.size()), "malloc head");
    ck(cudaMalloc((void**) &dcopy, (size_t) n_ids * row_bytes), "malloc copy");
    ck(cudaMalloc((void**) &dids, ids.size() * 4), "malloc ids");
    ck(cudaMalloc((void**) &dx, x.size() * 4), "malloc x");
    ck(cudaMalloc((void**) &dxq, k::native_q8_1_bytes(n_in, 1)), "malloc xq");
    ck(cudaMalloc((void**) &dy_copy, (size_t) n_ids * 4), "malloc y");
    ck(cudaMalloc((void**) &dy_rows, (size_t) n_ids * 4), "malloc y");
    ck(cudaMemcpy(dhead, head.data(), head.size(), cudaMemcpyHostToDevice), "upload head");
    ck(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "upload ids");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "upload x");
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    k::native_quantize_q8_1(dx, dxq, n_in, 1, s);
    k::gather_rows(dhead, row_bytes, dids, n_ids, dcopy, s);   // the copy bind made
    k::native_mmvq(13, dcopy, dxq, dy_copy, n_in, n_ids, 1, s);
    k::native_mmvq_rows(13, dhead, dids, dxq, dy_rows, n_in, n_ids, s);
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> a((size_t) n_ids), b((size_t) n_ids);
    ck(cudaMemcpy(a.data(), dy_copy, a.size() * 4, cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(b.data(), dy_rows, b.size() * 4, cudaMemcpyDeviceToHost), "download");
    int differ = 0, nonzero = 0;
    for (int i = 0; i < n_ids; ++i) {
        uint32_t ua, ub;
        std::memcpy(&ua, &a[(size_t) i], 4);
        std::memcpy(&ub, &b[(size_t) i], 4);
        if (ua != ub && differ++ < 4) std::printf("    row %d (id %d): copy 0x%08x, by index 0x%08x\n", i, ids[(size_t) i], ua, ub);
        nonzero += a[(size_t) i] != 0.0f;
    }
    const bool ok = differ == 0 && nonzero > n_ids / 2;
    std::printf("  n_in %4d, %5d of %d rows: %d logits differ, %d non-zero  %s\n", n_in, n_ids, n_vocab, differ, nonzero,
                ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
    cudaStreamDestroy(s);
    for (void* p : {(void*) dhead, (void*) dcopy, (void*) dids, (void*) dx, (void*) dxq, (void*) dy_copy, (void*) dy_rows}) cudaFree(p);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("mmvq_rows_parity: no CUDA device, skipped\n");
        return 77;
    }
    if (!k::native_mmvq_rows_supported(13)) {   // HIP: the drafter keeps its copy there
        std::printf("mmvq_rows_parity: native_mmvq_rows is not on this backend, skipped
");
        return 77;
    }
    std::printf("native_mmvq_rows against gather_rows + native_mmvq (Q5_K, one column), bit for bit\n");
    run(2560, 8192, 3001, 1u);
    run(1024, 4099, 1537, 2u);
    std::printf(g_fail ? "mmvq_rows_parity: %d FAILED\n" : "mmvq_rows_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
