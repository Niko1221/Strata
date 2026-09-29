// Q2_K (GGML type 10) dequant and embedding-gather parity: bartowski IQ2_XS serves token_embd.weight
// as Q2_K, so the embed path must accept exactly that - a raw-GGUF block type the MMVQ set does NOT claim.
// The synthetic arm checks the decoder against a hand-computed reference; the optional two-file argument
// arm checks REAL rows of the IQ2_XS shard against `gguf.dequantize`.
#include "strata/kernels/iq_kernels.hpp"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define HIP_CHECK(x)                                                                                                    \
    do {                                                                                                                \
        const hipError_t e = (x);                                                                                       \
        if (e != hipSuccess) {                                                                                          \
            std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e));                                                 \
            return 2;                                                                                                   \
        }                                                                                                               \
    } while (0)

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        std::fprintf(stderr, "usage: hip_q2k_embed_parity [raw-two-rows.bin reference-two-rows.f32]\n");
        return 2;
    }
    constexpr int blocks = 4;
    constexpr int cols = 256 * blocks;
    constexpr int row_bytes = 84 * blocks;
    std::vector<uint8_t> raw(2 * row_bytes);
    std::vector<float> ref(2 * cols);
    for (int row = 0; row < 2; ++row) {
        for (int block = 0; block < blocks; ++block) {
            uint8_t* p = raw.data() + row * row_bytes + block * 84;
            for (int i = 0; i < 16; ++i) p[i] = (uint8_t) ((i * 17 + block * 31 + row * 7) & 255);
            for (int i = 0; i < 64; ++i) p[16 + i] = (uint8_t) ((i * 29 + block * 11 + row * 3) & 255);
            const uint16_t d = 0x3800;     // 0.5
            const uint16_t dmin = 0x3400;  // 0.25
            std::memcpy(p + 80, &d, 2);
            std::memcpy(p + 82, &dmin, 2);
            for (int n = 0; n < 2; ++n) for (int l = 0; l < 32; ++l) {
                const int is = 8 * n + l / 16;
                const uint8_t q = p[16 + 32 * n + l];
                for (int j = 0; j < 4; ++j) {
                    const uint8_t s = p[is + 2 * j];
                    ref[row * cols + block * 256 + 128 * n + l + 32 * j] =
                        0.5f * (s & 15) * ((q >> (2 * j)) & 3) - 0.25f * (s >> 4);
                }
            }
        }
    }
    if (strata::kernels::iq_row_bytes(10, cols) != row_bytes ||
        !strata::kernels::iq_dequant_supported(10) || strata::kernels::iq_supported(10)) return 1;

    // token ids 1 and 0, in that order: the two rows must be GATHERED, not replayed in order
    void* table = nullptr;
    int32_t* tokens = nullptr;
    float* out = nullptr;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK(hipMalloc(&table, raw.size()));
    HIP_CHECK(hipMalloc((void**) &tokens, 2 * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&out, 2 * cols * sizeof(float)));
    HIP_CHECK(hipMemcpy(table, raw.data(), raw.size(), hipMemcpyHostToDevice));
    const int32_t ids[2] = {1, 0};
    HIP_CHECK(hipMemcpy(tokens, ids, sizeof(ids), hipMemcpyHostToDevice));
    strata::kernels::iq_embed_rows(10, table, row_bytes, tokens, 2, cols, out, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<float> got(2 * cols);
    HIP_CHECK(hipMemcpy(got.data(), out, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    // gather must have ROW 1 then ROW 0: replay-in-order would hand the wrong rows per token
    int bad = 0;
    for (int row = 0; row < 2; ++row) for (int i = 0; i < cols; ++i)
        if (std::fabs(got[row * cols + i] - ref[(1 - row) * cols + i]) > 1e-6f) ++bad;
    strata::kernels::iq_dequant_f32(10, table, 2 * cols, out, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(got.data(), out, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-6f) ++bad;
    hipFree(table);
    hipFree(tokens);
    hipFree(out);
    if (argc == 3) {
        std::FILE* raw_file = std::fopen(argv[1], "rb");
        std::FILE* ref_file = std::fopen(argv[2], "rb");
        if (!raw_file || !ref_file) {
            if (raw_file) std::fclose(raw_file);
            if (ref_file) std::fclose(ref_file);
            return 2;
        }
        constexpr int real_cols = 2560;
        constexpr int real_row_bytes = real_cols / 256 * 84;
        std::vector<uint8_t> real_raw(2 * real_row_bytes);
        std::vector<float> real_ref(2 * real_cols), real_got(2 * real_cols);
        const bool read_ok = std::fread(real_raw.data(), 1, real_raw.size(), raw_file) == real_raw.size() &&
                             std::fread(real_ref.data(), sizeof(float), real_ref.size(), ref_file) == real_ref.size();
        std::fclose(raw_file);
        std::fclose(ref_file);
        if (!read_ok) return 2;
        void* real_table = nullptr;
        float* real_out = nullptr;
        HIP_CHECK(hipMalloc(&real_table, real_raw.size()));
        HIP_CHECK(hipMalloc(&real_out, real_got.size() * sizeof(float)));
        HIP_CHECK(hipMemcpy(real_table, real_raw.data(), real_raw.size(), hipMemcpyHostToDevice));
        strata::kernels::iq_dequant_f32(10, real_table, 2 * real_cols, real_out, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(real_got.data(), real_out, real_got.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < real_got.size(); ++i)
            if (std::fabs(real_got[i] - real_ref[i]) > 1e-6f) ++bad;
        hipFree(real_table);
        hipFree(real_out);
    }
    std::printf("q2k_embed_parity: %d mismatches\n", bad);
    return bad ? 1 : 0;
}