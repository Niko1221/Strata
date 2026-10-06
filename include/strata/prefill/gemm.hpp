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
              float beta = 0.0f, int64_t ldx = 0);

    /// S (STRATA_HCD_EXACT): the HC down projection (N 320, K 10240) by the WMMA kernel that reproduces hipBLASLt's
    /// solution 1176 / 1177 bit for bit; false (nothing launched) unless hipBLASLt would take one of those for this
    /// shape (so the caller falls back to bf16(), with the same ldx).
    bool bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.  `ldx` (> K) is
    /// X's padded row stride, taken only by STRATA_PF_PAD's path (0 = K).
    /// On an MMQ build a beta = 0 product whose type is covered, whose K is a multiple of 256 values and whose matrix fits
    /// the card's shared memory runs through llama.cpp's int8 MMQ instead (opt-in: STRATA_DENSE_MMQ=1).
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f, int64_t ldx = 0);

    /// HIP gfx103x (prompt_f16()): bf16() takes X as the FP16 image the prompt path's kernels write (set_act_f16),
    /// and bf16() / f16() / native() run the GEMM with FP16 out (rocBLAS's tuned kernels there), widened in place in
    /// Y's own rows.  Off (default): every call is what it was.  Set by Prefill::init, together with set_act_f16.
    void set_f16_io(bool on) { f16_io_ = on; }

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
    bool f16_io_ = false;
    // RDNA2 (gfx103x, HIP): the product as an SGEMM on FP32 copies (tc_w_, tc_x_); false: the native call runs (gemm.cu)
    bool rdna2_sgemm(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                     float beta, bool bf16);
    // below sm_80: FP16 (Pascal: fp32) copies of a BF16 product's weight and activation slice (Gemm::bf16);
    // RDNA2: FP32 copies of the weight and an activation slice (Gemm::rdna2_sgemm)
    uint16_t* tc_w_ = nullptr;
    int64_t tc_w_elems_ = 0;
    uint16_t* tc_x_ = nullptr;
    int64_t tc_x_elems_ = 0;
    bool native_mmq(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                    int64_t ldy);
    void* mmq_ctx_ = nullptr;
    void* mmq_buf_ = nullptr;
    bool mmq_failed_ = false;
    /// Y = X . W^T with FP16 out, written into Y's own rows and widened there (no buffer): prompt_f16() only.
    void f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy);
};

/// HIP on gfx103x (RDNA2), for the current device: rocBLAS's tuned GEMMs there are FP16 in -> FP16 out only (FP16 or
/// BF16 in -> FP32 out runs ~6x slower), so the prompt path's 16-bit GEMMs run in FP16 (Gemm::set_f16_io,
/// set_act_f16).  Cached per device (a layer split can mix cards).  STRATA_HIP_PROMPT_F16=0/1 overrides.
bool prompt_f16();



}  // namespace strata::prefill
