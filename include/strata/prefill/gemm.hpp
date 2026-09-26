// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  Tensor-core GEMM through cuBLAS.
#pragma once

#include <cstdint>
#include <string>

namespace strata::prefill {

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// The cuBLAS handle on `stream`, and with `side_streams` the side streams f16_grouped() spreads over;
    /// set_buffers() before the first product.
    bool init(void* stream, std::string& err, bool side_streams = false);
    /// Caller-owned device buffers (e.g. lent expert-cache slots): the dequantization scratch (`scratch_elems` FP16
    /// elements: the largest weight dequantized at once; may be null for no native() calls) and the cuBLAS workspace.
    void set_buffers(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// Y_i[rows[i], N] = X_i[rows[i], K] . W_i[N, K]^T (FP16 in, FP32 out) for n problems of one shape but their
    /// own row counts, in one cuBLAS grouped call; the pointer arrays are DEVICE memory, `rows` host memory.  Where
    /// cuBLAS does not group these types it runs f16() per problem, from the host copies of the pointers, spread
    /// over the side streams when the problems average 128 rows or more (a GEMM of a few hundred rows leaves most
    /// of the GPU idle; below that the launches, not the GPU, set the pace).
    void f16_grouped(const uint16_t* const* X, const uint16_t* const* W, float* const* Y, const uint16_t* const* X_host,
                     const uint16_t* const* W_host, float* const* Y_host, const int* rows, int n, int64_t N, int64_t K);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    static constexpr int AUX = 3;   // side streams for independent products
    void* handle_ = nullptr;
    void* stream_ = nullptr;
    void* aux_handle_[AUX] = {};
    void* aux_stream_[AUX] = {};
    void* fork_ = nullptr;
    void* join_[AUX] = {};
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    bool grouped_ = true;   // cuBLAS groups FP16 -> FP32 GEMMs (cleared at the first refusal)
};

}  // namespace strata::prefill
