// include/strata/prefill/moe_mmq.hpp - prompt-speed plan step 2b: the prompt path's experts through llama.cpp's MMQ
// kernels (ggml-cuda mmq.cuh, MIT): the weights stay quantized and the activations are rounded to q8_1, the
// products run on int8 tensor cores.  The dequantize-to-FP16 + cuBLAS path wrote ~10 MB of FP16 per expert and
// multiplied in FP16; this reads the ~1.4-2 MB expert once.  A group of experts is gathered into one buffer
// (`gather_*`, one launch per expert as its blob arrives) and multiplied in one launch per product.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::mmq {

/// This build has the MMQ path (the ggml sources were available to the build).
bool built();
/// MMQ covers this ggml type (the i-quants and Q2_0 the packs use, Q8_0, NVFP4 unless fp16, and in a CUDA build
/// with STRATA_MMQ_KQUANTS the Q4_K / Q5_K / Q5_1 of Unsloth's UD-Q4_K_XL; IQ1_M is not covered).
bool supported(int ggml_type);
/// #420: `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of
/// `w_rows` rows that fits the card's shared memory - the same test its tile choice makes, which aborts the process
/// ("J_best=0") when nothing fits.  false (said once per type) keeps that product on the non-MMQ path.
bool fits(int ggml_type, int64_t w_rows);
/// NVFP4 prompt precision, STRATA_PREFILL_NVFP4: w4a8 (default) - int8 tensor cores, Q8_1 activations, what decode
/// runs at; w4a4 - Blackwell's FP4 x FP4 MMA, activations rounded to NVFP4 (faster, first-token KL up to 0.03 vs
/// fp16); fp16 - the dequantize + FP16 GEMM path (the reference).
enum class Nvfp4Mode { W4A8, W4A4, FP16 };
Nvfp4Mode nvfp4_mode();
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream, float* yscale = nullptr);
/// NVFP4 on Blackwell in w4a4 mode: the MMQ kernel multiplies FP4 x FP4, so its activations are NVFP4 too, with
/// one float scale per row that `quantize` writes to `yscale` and the product reads as Product::y_scale.  Decided
/// by the mode and a device probe of the same compile.
bool fp4_activations(int ggml_type);
/// NVFP4 expert tails: GU rows [row0, row0 + nrows) of a group of `n` experts (absolute `bounds`, n + 1 of them,
/// on the device) scaled by their expert's {s_gate, s_up, s_down, 0} (`tails`, 4 floats each, on the device): the
/// gate half by s_gate, the up half by s_up.
void scale_gu_rows(float* gu, int64_t ld, int64_t n_ff, const int32_t* bounds, int n, const float* tails,
                   int64_t row0, int64_t nrows, void* stream);
/// The down product's rows [0, nrows) (`ld` floats each, relative `bounds`) times their expert's s_down - on the FP32
/// output, not folded into up, where the hidden would sit near 1e-5 (FP16-subnormal block scales in q8 formats).
void scale_down_rows(float* d, int64_t ld, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
    const float* y_scale = nullptr;     ///< NVFP4 w4a4: the per-row activation scales (fp4_activations)
};

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);

private:
    void* ctx_ = nullptr;
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
/// `tail` (16 bytes, may be null) goes to `tail_dst`; `zero_bytes` zeroed right after gu_dst's and d_dst's copies
/// (the MMQ tail after a group's last expert).
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail = nullptr, void* tail_dst = nullptr,
                   size_t zero_bytes = 0);
/// A Strata-pack Q2_0 expert blob (codes and fp16 scales in separate planes, gate/up rows interleaved) into GGUF
/// Q2_0 blocks: gate/up [1280, 2560] at `gu_dst` (rows stay interleaved), down [2560, 640] at `d_dst`.  Same values.
void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Strata pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);
/// swiglu of an NVFP4 group's gate/up rows (split halves) with scale_gu_rows' scales applied as they are read.
void swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds, int n,
                   const float* tails, int64_t row0, void* stream);
/// sd[r] = the s_down of row r's expert (group-local bounds): the combine applies it as it reads the row.
void down_row_scales(float* sd, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

}  // namespace strata::prefill::mmq
