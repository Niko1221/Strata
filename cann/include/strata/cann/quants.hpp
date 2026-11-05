// cann/include/strata/cann/quants.hpp - the block formats the model's weights actually use.
//
// Q4_K and Q6_K are the two types that dominate Qwen3.6-35B-A3B's file. The layout is ggml's
// (ggml-common.h): 256 values per block, a shared fp16 scale (and, for Q4_K, a shared fp16
// min), 6-bit sub-block scales packed into 12 bytes for Q4_K and 16 signed bytes for Q6_K.
#pragma once

#include <cstdint>

namespace strata::cann {

constexpr int QK_K = 256;

#pragma pack(push, 1)
struct block_q4_K {
    uint16_t d;     // fp16 super-block scale
    uint16_t dmin;  // fp16 super-block min
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;  // fp16 super-block scale
};

struct block_q5_K {
    uint16_t d;    // fp16 super-block scale
    uint16_t dmin; // fp16 super-block min
    uint8_t scales[12];
    uint8_t qh[32]; // the fifth bit of each value, 8 values per byte
    uint8_t qs[128];
};

struct block_q8_0 {
    uint16_t d;  // fp16 scale
    int8_t qs[32];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "Q4_K block must be 144 bytes");
static_assert(sizeof(block_q6_K) == 210, "Q6_K block must be 210 bytes");
static_assert(sizeof(block_q5_K) == 176, "Q5_K block must be 176 bytes");
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block must be 34 bytes");

// Bytes one block of this type occupies; 0 when the type is not one of the two above.
int quant_type_size(int ggml_type);
// Elements per block (QK_K), 0 when unsupported.
int quant_block_elements(int ggml_type);

// Dequantize `k` values (a multiple of QK_K) into `out`. `blocks` points at the raw bytes the
// GGUF file holds. Returns false for an unsupported type or a length that is not a multiple.
bool dequantize(int ggml_type, const void * blocks, float * out, int64_t k);

// The implementations, exposed so the parity test can call them directly.
void dequantize_q4_K(const block_q4_K * blocks, float * out, int64_t k);
void dequantize_q5_K(const block_q5_K * blocks, float * out, int64_t k);
void dequantize_q6_K(const block_q6_K * blocks, float * out, int64_t k);
void dequantize_q8_0(const block_q8_0 * blocks, float * out, int64_t k);

} // namespace strata::cann
