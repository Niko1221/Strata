// Strata. The Engram reader's ggml Q5_K rows: q5k_value (ds4_q5k.cuh) through both of ds4_kernels.cu's branches - the
// window's engram_dequant (fmt 4) and the chunked prompt's engram_dequant_tok - against a copy of ggml's
// dequantize_row_q5_K.  Random legal superblocks (any bit pattern with a finite d and dmin is one), every value bit for
// bit.  No model; needs a GPU (77, skipped, where there is none).
//   q5k_engram_test
#include "../../src/dsv41/ds4_kernels.cuh"

#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("GPU error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float f = e == 0 ? std::ldexp((float) m, -24) : std::ldexp(1.0f + m / 1024.0f, e - 15);
    return s ? -f : f;
}

// ggml's get_scale_min_k4 (ggml/src/ggml-quants.c): the 6-bit scale and min of a 32-value block out of the superblock's
// 12 packed bytes
static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// a product rounded on its own before the subtraction: GCC and clang both fuse a * b - c into an FMA by default
// (-ffp-contract=fast), and ggml rounds each - the volatile store is that round
static inline float mul_sub(float prod, float m) {
    volatile float p = prod;
    return p - m;
}

// ggml's dequantize_row_q5_K (ggml/src/ggml-quants.c) on one 176-byte superblock: half d, half dmin, scales[12], qh[32],
// qs[128] -> 256 values
static void dequant_row_q5_K(const uint8_t* x, float* y) {
    const uint8_t* ql = x + 48;
    const uint8_t* qh = x + 16;
    const uint8_t* sc = x + 4;
    const float d = h2f((uint16_t) (x[0] | (x[1] << 8)));
    const float mn = h2f((uint16_t) (x[2] | (x[3] << 8)));
    int is = 0;
    uint8_t s, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, sc, &s, &m);
        const float d1 = d * s, m1 = mn * m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        const float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; ++l) *y++ = mul_sub(d1 * (float) ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)), m1);
        for (int l = 0; l < 32; ++l) *y++ = mul_sub(d2 * (float) ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)), m2);
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

template <typename T> static T* up(const std::vector<T>& v) {
    T* d; CK(hipMalloc(&d, std::max<size_t>(1, v.size() * sizeof(T)))); CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return d;
}
static std::vector<float> down(const float* d, size_t n) {
    std::vector<float> v(n); CK(hipMemcpy(v.data(), d, n * sizeof(float), hipMemcpyDeviceToHost)); return v;
}

// the values the device wrote against ggml's: the count that differs, the first few printed
static int compare(const char* what, const std::vector<float>& dev, const std::vector<float>& ref) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        if (dev[i] != ref[i]) {
            if (bad < 4) std::printf("%s: value %zu device %.9g, ggml %.9g\n", what, i, (double) dev[i], (double) ref[i]);
            ++bad;
        }
    std::printf("%s: %zu values, %d differ from ggml\n", what, ref.size(), bad);
    return bad;
}

int main() {
    void* probe = nullptr;   // (a device to run on: no GPU, skipped like the other device tests)
    if (hipMalloc(&probe, 16) != hipSuccess) {
        std::printf("q5k_engram_test: no GPU, skipped\n");
        return 77;
    }
    hipFree(probe);

    std::mt19937_64 rng(41);
    const int hd = 512, n = 4, cols = 2, T = 2;   // (4 rows of two superblocks; the chunked layout's 2 tokens of 2 rows)
    const int nb = hd / 256, per = cols * (hd + hd / 32);   // (a token's bytes: its cols rows, then their scales)
    std::vector<uint8_t> rows((size_t) n * hd, 0);
    for (int r = 0; r < n; ++r)
        for (int k = 0; k < nb; ++k) {
            uint8_t* b = &rows[(size_t) r * hd + (size_t) k * 176];
            for (int i = 0; i < 176; ++i) b[i] = (uint8_t) rng();
            for (int i = 0; i < 2; ++i) {   // (d and dmin: finite normal halves, so every block is one ggml can read)
                uint16_t h;
                do { h = (uint16_t) rng(); } while (((h >> 10) & 31) == 0 || ((h >> 10) & 31) == 31);
                b[2 * i] = (uint8_t) h;
                b[2 * i + 1] = (uint8_t) (h >> 8);
            }
        }
    std::vector<float> ref((size_t) n * hd);
    for (int r = 0; r < n; ++r)
        for (int k = 0; k < nb; ++k)
            dequant_row_q5_K(&rows[(size_t) r * hd + (size_t) k * 176], &ref[(size_t) r * hd + (size_t) k * 256]);

    uint8_t* drow = up(rows);
    float* dout = nullptr;
    CK(hipMalloc(&dout, ref.size() * sizeof(float)));
    int bad = 0;

    // ---- the window's rows: one after another, [n][hd] bytes with the superblocks at each row's start
    strata::ds4::engram_dequant(drow, nullptr, n, hd, dout, 0, 4);
    bad += compare("engram_dequant", down(dout, ref.size()), ref);

    // ---- the chunked prompt's: token t's cols rows, then their scale bytes (the type does not use them), `per` a token
    {
        std::vector<uint8_t> tok((size_t) T * per, 0);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < cols; ++c)
                std::memcpy(&tok[(size_t) t * per + (size_t) c * hd], &rows[(size_t) (t * cols + c) * hd],
                            (size_t) hd);
        uint8_t* dtok = up(tok);
        strata::ds4::engram_dequant_tok(dtok, (size_t) per, cols, T, hd, dout, 0, 4);
        bad += compare("engram_dequant_tok", down(dout, ref.size()), ref);
        hipFree(dtok);
    }

    hipFree(drow);
    hipFree(dout);
    std::printf("%s\n", bad == 0 ? "Q5_K Engram rows: every value equals ggml's" : "Q5_K Engram rows: FAILED");
    return bad == 0 ? 0 : 1;
}
