// prefill_swiglu_quant_test - the prompt path's fused MMQ expert step against the kernels it replaces, byte for byte:
// mmq::swiglu_quant against swiglu + quantize (the down product's q8_1 rows of H), for down types the packs have
// (Q2_0, IQ4_NL, Q8_0), with the gate/up rows split (GGUF) and interleaved (the Strata pack), a zero row and a row
// with one huge value; NVFP4 experts against swiglu_scaled + quantize (each row's expert scales from a group's bounds,
// a group that starts past row 0).  Exit 77 without a CUDA device.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <algorithm>
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
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <class T> T* as() const { return (T*) p; }
};

int swiglu_quant_case(ggml_type t, int64_t rows, bool interleaved, uint32_t seed, int n_exp = 0) {
    constexpr int64_t NFF = 640;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.05f, 40.f);
    std::vector<float> gu((size_t) rows * 2 * NFF);
    for (int64_t r = 0; r < rows; ++r) {
        const float m = mag(rng);
        for (int64_t k = 0; k < 2 * NFF; ++k) gu[(size_t) (r * 2 * NFF + k)] = nd(rng) * m;
    }
    if (rows > 3) {   // a zero row, a row with one huge value
        std::memset(gu.data() + 2 * NFF, 0, 2 * NFF * sizeof(float));
        gu[(size_t) (3 * 2 * NFF + 5)] = 3e4f;
    }
    // n_exp > 0: an NVFP4 group of n_exp experts over the rows (absolute bounds from row0, as prefill.cpp passes them),
    // each with its {s_gate, s_up, s_down, 0}
    const int64_t row0 = n_exp > 0 ? 77 : 0;
    std::vector<int32_t> bounds;
    std::vector<float> tails;
    if (n_exp > 0) {
        std::uniform_int_distribution<int> cut(0, (int) rows);
        std::vector<int> c((size_t) n_exp - 1);
        for (int& v : c) v = cut(rng);
        std::sort(c.begin(), c.end());
        bounds.push_back((int32_t) row0);
        for (int v : c) bounds.push_back((int32_t) (row0 + v));
        bounds.push_back((int32_t) (row0 + rows));
        std::uniform_real_distribution<float> sc(0.01f, 4.f);
        for (int q = 0; q < n_exp; ++q) {
            tails.push_back(sc(rng));
            tails.push_back(sc(rng));
            tails.push_back(sc(rng));
            tails.push_back(0.f);
        }
    }
    const size_t qb = mmq::q8_bytes(rows, NFF);
    Dev dgu(gu.size() * 4), dh((size_t) rows * NFF * 4 + 16), dq0(qb), dq1(qb), db(bounds.size() * 4 + 4),
        dt(tails.size() * 4 + 4);
    ck(cudaMemcpy(dgu.p, gu.data(), gu.size() * 4, cudaMemcpyHostToDevice), "upload");
    if (n_exp > 0) {
        ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemcpy(dt.p, tails.data(), tails.size() * 4, cudaMemcpyHostToDevice), "upload");
    }
    ck(cudaMemset(dq0.p, 0x5a, qb), "memset");
    ck(cudaMemset(dq1.p, 0x5a, qb), "memset");
    // the two kernels it replaces, as prefill.cpp launches them
    if (n_exp > 0) {
        mmq::swiglu_scaled(dgu.as<float>(), dh.as<float>(), rows, NFF, db.as<int32_t>(), n_exp, dt.as<float>(), row0,
                           nullptr);
        mmq::quantize(dh.as<float>(), nullptr, dq0.p, (int) t, NFF, NFF, rows, nullptr, nullptr);
        mmq::swiglu_quant(dgu.as<float>(), dq1.p, rows, false, nullptr, db.as<int32_t>(), n_exp, dt.as<float>(), row0);
    } else {
        mmq::swiglu(dgu.as<float>(), dh.as<float>(), rows, NFF, interleaved, nullptr);
        mmq::quantize(dh.as<float>(), nullptr, dq0.p, (int) t, NFF, NFF, rows, nullptr);
        mmq::swiglu_quant(dgu.as<float>(), dq1.p, rows, interleaved, nullptr);
    }
    ck(cudaDeviceSynchronize(), "run");
    const size_t used = (size_t) rows * (1024 / 128) * 144;   // the rows' 8 blocks of 128 values (144 bytes each)
    std::vector<uint8_t> a(used), b(used);
    ck(cudaMemcpy(a.data(), dq0.p, used, cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(b.data(), dq1.p, used, cudaMemcpyDeviceToHost), "download");
    size_t bad = 0;
    for (size_t i = 0; i < used; ++i) bad += a[i] != b[i];
    std::printf("  swiglu_quant %-6s %5lld rows%s: %zu of %zu bytes differ\n", ggml_type_name(t), (long long) rows,
                n_exp > 0 ? " scaled (an NVFP4 group)" : interleaved ? " interleaved" : "", bad, used);
    return bad == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("prefill_swiglu_quant_test: no CUDA device (skipped)\n");
        return 77;
    }
    int fails = 0;
    try {
        for (ggml_type t : {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0}) {
            if (!mmq::swiglu_quant_ok((int) t)) { std::printf("  %s: not on the fused path here\n", ggml_type_name(t)); continue; }
            for (bool il : {false, true}) {
                fails += swiglu_quant_case(t, 1, il, 1);
                fails += swiglu_quant_case(t, 37, il, 2);
                fails += swiglu_quant_case(t, 4113, il, 3);
            }
        }
        if (mmq::swiglu_quant_ok((int) GGML_TYPE_NVFP4)) {   // the NVFP4 pack: split rows, the experts' scales
            fails += swiglu_quant_case(GGML_TYPE_NVFP4, 1, false, 4, 1);
            fails += swiglu_quant_case(GGML_TYPE_NVFP4, 37, false, 5, 3);
            fails += swiglu_quant_case(GGML_TYPE_NVFP4, 4113, false, 6, 16);
        } else {
            std::printf("  NVFP4: not on the fused path here\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_swiglu_quant_test: %s\n", e.what());
        return 1;
    }
    std::printf("prefill_swiglu_quant_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
