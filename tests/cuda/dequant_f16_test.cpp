// dequant_f16_test - strata::kernels::dequant_f16 / dequant_f16_ld of Q8_0 and IQ4_XS (the prompt path's dense
// weights before cuBLAS) against the exact value per element, rounded to nearest-even FP16: d * q (Q8_0) and
// d * (ls - 32) * kvalues_iq4nl[q] (IQ4_XS, ggml's dequantize_row_iq4_xs).  Random blocks with scales from zero and
// subnormal to large, row slices (row0), padded rows (ld), and an output pointer that is not 16-byte aligned (the
// generic kernel).  Exit 77 without a CUDA device.
#include "strata/kernels/dequant_bf16.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

float h2f(uint16_t h) {
    __half_raw r;
    r.x = h;
    return __half2float(__half(r));
}
uint16_t f2h(float f) { return __half_raw(__float2half_rn(f)).x; }

const int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// element k of a row (blocks at `row`)
float ref(int type, const uint8_t* row, int64_t k) {
    if (type == 8) {
        const uint8_t* b = row + (k / 32) * 34;
        uint16_t d;
        std::memcpy(&d, b, 2);
        return (float) (int8_t) b[2 + k % 32] * h2f(d);
    }
    const uint8_t* b = row + (k / 256) * 136;   // IQ4_XS: d scales_h scales_l[4] qs[128]
    const int ib = (int) (k % 256) / 32, j = (int) (k % 32);
    uint16_t d, sh;
    std::memcpy(&d, b, 2);
    std::memcpy(&sh, b + 2, 2);
    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = h2f(d) * (float) (ls - 32);
    const uint8_t q = b[8 + 16 * ib + j % 16];
    return dl * (float) kv_iq4nl[j < 16 ? (q & 0xf) : (q >> 4)];
}

int run_case(int type, int64_t rows_total, int64_t cols, int64_t row0, int64_t rows, int64_t ld, int64_t out_off,
             uint32_t seed) {
    std::mt19937 rng(seed);
    const int64_t be = type == 8 ? 32 : 256, bb = type == 8 ? 34 : 136, bpr = cols / be, row_bytes = bpr * bb;
    std::vector<uint8_t> w((size_t) (rows_total * row_bytes));
    std::uniform_int_distribution<int> byte(0, 255), pick(0, 9);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (int64_t i = 0; i < rows_total * bpr; ++i) {   // scales: mostly normal, some zero / subnormal / large
        uint16_t d;
        switch (pick(rng)) {
            case 0: d = 0; break;
            case 1: d = (uint16_t) (byte(rng) | 0x100); break;   // subnormal
            case 2: d = (uint16_t) (0x7000 + byte(rng)); break;  // ~8K: products past FP16's range become inf
            default: d = (uint16_t) (0x1c00 + byte(rng) * 16); break;
        }
        std::memcpy(w.data() + i * bb, &d, 2);
    }
    const size_t n_out = (size_t) (rows * ld + out_off + 64);
    uint8_t* dw = nullptr;
    uint16_t* dout = nullptr;
    cudaMalloc(&dw, w.size());
    cudaMalloc(&dout, n_out * 2);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0xee, n_out * 2);
    if (ld == cols) strata::kernels::dequant_f16(type, dw, row0, rows, cols, dout + out_off, nullptr);
    else if (!strata::kernels::dequant_f16_ld(type, dw, row0, rows, cols, ld, dout + out_off, nullptr)) {
        std::printf("  dequant_f16_ld refused ld %lld\n", (long long) ld);
        return 1;
    }
    std::vector<uint16_t> got(n_out);
    if (cudaMemcpy(got.data(), dout, n_out * 2, cudaMemcpyDeviceToHost) != cudaSuccess) { std::printf("  CUDA error\n"); return 1; }
    cudaFree(dw);
    cudaFree(dout);
    size_t bad = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            if (got[(size_t) (out_off + r * ld + k)] != f2h(ref(type, w.data() + (row0 + r) * row_bytes, k))) ++bad;
    for (int64_t r = 0; r < rows; ++r)   // the padding between rows untouched
        for (int64_t k = cols; k < ld; ++k)
            if (got[(size_t) (out_off + r * ld + k)] != 0xeeee) ++bad;
    std::printf("  %-6s %5lld x %5lld from row %4lld (ld %5lld, out +%lld): %zu of %lld differ\n",
                type == 8 ? "Q8_0" : "IQ4_XS", (long long) rows, (long long) cols, (long long) row0, (long long) ld,
                (long long) out_off, bad, (long long) (rows * cols));
    return bad ? 1 : 0;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("dequant_f16_test: no CUDA device (skipped)\n");
        return 77;
    }
    int fails = 0;
    for (int t : {8, 23}) {
        fails += run_case(t, 64, 2560, 0, 64, 2560, 0, 1);
        fails += run_case(t, 300, 10240, 37, 200, 10240, 0, 2);
        fails += run_case(t, 10, 256, 3, 7, 256, 0, 3);
        fails += run_case(t, 40, 768, 5, 33, 832, 0, 4);      // padded rows (the strided path)
        fails += run_case(t, 16, 2560, 1, 15, 2560, 1, 5);    // output not 16-byte aligned: the generic kernel
    }
    std::printf("dequant_f16_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
