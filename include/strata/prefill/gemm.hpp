// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  Tensor-core GEMM through cuBLAS.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::prefill {

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: BF16 elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with caller-owned device buffers (the prompt path borrowing expert-cache slots).
    bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                       std::string& err);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    /// Below sm_80 (no BF16 tensor cores): Volta converts both to FP16 and runs the FP16 tensor-core GEMM (Turing with
    /// STRATA_BF16_TC=1), Pascal widens both to fp32 (cublasSgemm); see gemm.cu.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    /// Caller-owned buffers only: the scratch and workspace moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    void* handle_ = nullptr;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    bool external_ = false;
    void* hipblaslt_state_ = nullptr;
    // below sm_80: FP16 (Pascal: fp32) copies of a BF16 product's weight and activation slice (Gemm::bf16)
    uint16_t* tc_w_ = nullptr;
    int64_t tc_w_elems_ = 0;
    uint16_t* tc_x_ = nullptr;
    int64_t tc_x_elems_ = 0;
};



}  // namespace strata::prefill
