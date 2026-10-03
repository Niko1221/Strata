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
    ///
    /// ON cc < 8.0 (V100 and older: no BF16 ALUs, and `cublasGemmEx` with `CUDA_R_16BF` measures 5.6x slower
    /// there than `CUDA_R_16F`) this multiplies in FP16 instead: both operands go through `f16_from_bf16`
    /// (exact for 2^-14 <= |x| <= 65280, clamped to +-65504 past that) and the existing FP16 GEMM.  The static
    /// W is converted ONCE into a bounded cache of FP16 twins (`STRATA_PREFILL_BF16_TWINS_MB`, default 256 MiB,
    /// 0 = convert per call into the scratch); X is converted per call into the scratch.  sm_75/80+ and HIP
    /// never enter this path (`STRATA_PREFILL_BF16_F16=0` forces the BF16 cuBLAS call anywhere, `=1` forces
    /// the FP16 path - the full-engine A/B).
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// The cc < 8.0 BF16-over-FP16 fallback is armed (see `bf16`).  For tests and the micro benchmark.
    bool bf16_as_f16() const { return bf16_as_f16_; }

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
    /// The cc < 8.0 fallback: Y = X . W^T with both operands rounded to FP16 first (see `bf16`).
    void bf16_via_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                      float beta);
    /// The FP16 twin of a static BF16 W, converted once on first use and cached until destruction; null when
    /// the cache is disabled or full (the caller then converts per call into the scratch).
    const uint16_t* f16_twin(const uint16_t* W, int64_t N, int64_t K);
    void free_twins();

    void* handle_ = nullptr;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    bool external_ = false;
    void* hipblaslt_state_ = nullptr;
    bool bf16_as_f16_ = false;   ///< cc < 8.0 (or STRATA_PREFILL_BF16_F16=1): the BF16 GEMM goes through FP16
    void* f16_twins_ = nullptr;  ///< private cache of converted Ws (struct F16Twins*), cc < 8.0 only
};



}  // namespace strata::prefill
