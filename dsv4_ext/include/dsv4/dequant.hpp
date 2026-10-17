/**
 * @file dequant.hpp
 * @brief ggml quantization types: row sizes, dequantization and row dot products.
 *
 * This module implements faithful byte-offset ports of llama.cpp's `ggml/src/ggml-quants.c`
 * dequantization functions for all 31 supported ggml quantization types (F32 through MXFP4).
 * The CPU reference (`dequant_row()`, `row_dot()`) and the GPU kernel traits (`dq_traits.hpp`)
 * share identical math, ensuring bit-identical weight decoding across backends.
 *
 * @par IQ Codebooks
 *   IQ* codebook lookup tables are generated from llama.cpp's `ggml-common.h` by
 *   [`tools/extract_iq_tables.py`](../../tools/extract_iq_tables.py) into `src/iq_tables.cpp`.
 *   llama.cpp is never modified — only read.
 *
 * @par Prefill optimization (`row_dots()`)
 *   The prefill primitive decodes each weight block ONCE and dots it against T activation rows,
 *   saving the weight decode cost (not the multiply) which dominates at 1.8 GB/token for
 *   DeepSeek-V4-Flash's MoE path. With T==1 and contiguous rows, `row_dots()` is bit-identical
 *   to `row_dot()`.
 *
 * @par Supported types
 *   Scalar: F32, F16, BF16, F64, I8-I64
 *   Block 4-bit: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
 *   K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
 *   IQ quants: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS
 *   Mixed precision: MXFP4
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace dsv4 {

/**
 * @enum GgmlType
 * @brief Real ggml type IDs matching `ggml.h` and GGUF file on-disk values.
 *
 * These must stay in sync with llama.cpp's `ggml.h`. Types 0-30 cover the standard set;
 * MXFP4 (39) is added for DeepSeek-V4-Flash compatibility.
 */
enum GgmlType : uint32_t {
    T_F32 = 0,     T_F16 = 1,      T_Q4_0 = 2,   T_Q4_1 = 3,   T_Q5_0 = 6,   T_Q5_1 = 7,
    T_Q8_0 = 8,    T_Q2_K = 10,    T_Q3_K = 11,  T_Q4_K = 12,  T_Q5_K = 13,  T_Q6_K = 14,
    T_Q8_K = 15,   T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ4_NL = 20,
    T_IQ3_S = 21,  T_IQ2_S = 22,   T_IQ4_XS = 23, T_I8 = 24,   T_I16 = 25,   T_I32 = 26,
    T_I64 = 27,    T_F64 = 28,     T_IQ1_M = 29,  T_BF16 = 30,  T_MXFP4 = 39
};

/**
 * @brief Check if a ggml type can be dequantized by this module.
 * @param type  GgmlType enum value to check.
 * @return true if `dequant_row()` and `row_dot()` support this type.
 */
bool dequant_supported(uint32_t type);

/**
 * @brief Compute byte size of one row of `in` elements for the given ggml type.
 *
 * For block-compressed types (Q4_0, IQ1_M, etc.), returns `(in / block_elems) * block_bytes`.
 * Returns 0 if the type is unknown or `in` is not a whole number of blocks.
 *
 * @param type  GgmlType enum value.
 * @param in    Number of elements in the row.
 * @return Byte count, or 0 for unsupported/invalid inputs.
 */
size_t row_bytes(uint32_t type, int64_t in);

/**
 * @brief Get the number of elements per quantization block (1 for non-blocked types).
 * @param type  GgmlType enum value.
 * @return Block element count (e.g., 32 for Q8_0, 256 for IQ1_M), or 0 for unknown types.
 */
int dequant_block_elems(uint32_t type);

/**
 * @brief Decode one compressed row into float32.
 *
 * Unsupported types leave the output buffer untouched. The decode is a byte-offset port of
 * llama.cpp's `dequant_row_*` functions, operating directly on the on-disk GGUF layout
 * without depending on any struct alignment or padding.
 *
 * @param type  GgmlType enum value.
 * @param w     Pointer to compressed weight data (row-major).
 * @param in    Number of elements in the row.
 * @param y     Output buffer: must hold `in` floats, populated with dequantized values.
 */
void dequant_row(uint32_t type, const uint8_t* w, int64_t in, float* y);

/**
 * @brief Compute dot product of one compressed weight row with a float vector.
 *
 * Decodes the row on-the-fly (block by block) and accumulates in double precision to minimize
 * rounding error. Returns 0 for unsupported types or invalid inputs.
 *
 * @param type  GgmlType enum value.
 * @param w     Pointer to compressed weight data.
 * @param x     Float input vector.
 * @param in    Number of elements (must be a multiple of block_elems).
 * @return Dot product result, or 0 for unsupported types.
 */
float row_dot(uint32_t type, const uint8_t* w, const float* x, int64_t in);

/**
 * @brief Prefill primitive: one decoded weight row, T dot products against different activation rows.
 *
 * The core prefill optimization: each weight block is decoded once and reused for all T tokens,
 * saving the decode cost (not the multiply) which dominates MoE traffic at ~1.8 GB/token.
 *
 * @par Layout
 *   - `y[k * y_stride] = dot(w_row, x + rows[k] * x_stride)` for k in [0, T)
 *   - If `rows` is null, `rows[k] = k` (contiguous case).
 *   - With T==1 and rows==null, result is bit-identical to `row_dot()`.
 *
 * @param type      GgmlType enum value.
 * @param w         Pointer to compressed weight data.
 * @param x         Activation input buffer (TOKEN-MAJOR layout).
 * @param x_stride  Stride between consecutive token vectors in x (>= in).
 * @param rows      Optional gather indices (HOST memory); null = contiguous.
 * @param T         Number of tokens / dot products to compute.
 * @param in        Row dimension (elements per weight row).
 * @param y         Output buffer: T values, stride `y_stride`.
 * @param y_stride  Stride between consecutive output elements (>= 1).
 */
void row_dots(uint32_t type, const uint8_t* w, const float* x, int64_t x_stride, const int32_t* rows,
              int T, int64_t in, float* y, int64_t y_stride);

}  // namespace dsv4
