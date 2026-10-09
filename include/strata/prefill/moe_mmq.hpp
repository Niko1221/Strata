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
/// MMQ covers this ggml type (the i-quants and Q2_0 the packs use, Q8_0, and with STRATA_MMQ_KQUANTS the
/// K-quants Q4_K / Q5_K / Q5_1 / Q6_K: Unsloth's UD-Q4_K_XL experts (CUDA), and the dense GGUF projections of
/// the mixed-quant packs through Gemm::native's STRATA_DENSE_MMQ path (HIP); IQ1_M is not covered).
bool supported(int ggml_type);
/// #420: `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of
/// `w_rows` rows that fits the card's shared memory - the same test its tile choice makes, which aborts the process
/// ("J_best=0") when nothing fits.  false (said once per type) keeps that product on the non-MMQ path.
bool fits(int ggml_type, int64_t w_rows);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);

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

/// A DENSE product: ONE matrix against ALL of `T` activation rows in one GEMM.
///
/// `Product` above is shaped for a group of experts - `bounds` says which activation rows each one owns and
/// `ids` remaps the output rows.  A dense matrix is the degenerate case (one matrix, every row) and it is the
/// case that decides glm5-next's prompt path: its pre-section is entered once per EIGHT tokens, so a 4,096-token
/// chunk reads each dense weight matrix 512 times - 722 GB of reads on one stage whose weights are 1.4 GB - while
/// the fold that the eight columns exist to pay for only ever divides the read by eight.  One MMQ over all the
/// rows reads each of them once, which is the whole difference between a GEMV repeated T/8 times and a GEMM.
///
/// `w` is the native `ggml_type` blocks of an [N, K] matrix (GGUF's own layout: K the contiguous dimension,
/// `matrix_bytes(type, N, K)` bytes).  `x` is fp32 with `ldx` floats to a row (< 0 means K), `y` fp32 with `ldy`
/// floats to a row (< 0 means N).  Rows of `x` map to rows of `y` one for one.
///
/// Returns false - having written nothing - when this build has no MMQ, when `ggml_type` is not covered, when
/// `K` is not a whole number of 256-value chunks (llama.cpp's MMQ loads the weights in those, and the chunk past
/// a partial row would read the next row's bytes), when `fits` has no tile for an [N, K] matrix, or when the
/// allocation fails.  The caller falls back to its own path.  The state (the q8_1 activation image, the identity
/// row map, the bounds of one matrix) is per DEVICE and grown to the largest product asked of it.
bool dense(const void* w, int ggml_type, float* y, int64_t ldy, const float* x, int64_t ldx, int64_t T, int64_t N,
           int64_t K, void* stream);

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream);
/// gather_native for an MMQ group's experts [first, n) in ONE launch: expert q's blob (`blob[q]`; gate at +0, up at
/// +up_off, down at +down_off) to gu_dst + q * gu_stride and d_dst + q * d_stride - the same bytes as one gather_native
/// each.  Every pointer, offset and size 16-byte aligned (false otherwise: nothing launched, gather one at a time).
constexpr int kGatherGroupMax = 16;
struct GatherGroup {
    const uint8_t* blob[kGatherGroupMax] = {};
    int first = 0, n = 0;
};
bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);
/// A Strata-pack Q2_0 expert blob (codes and fp16 scales in separate planes, gate/up rows interleaved) into GGUF
/// Q2_0 blocks: gate/up [1280, 2560] at `gu_dst` (rows stay interleaved), down [2560, 640] at `d_dst`.  Same values.
void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Strata pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

/// y[i] = float(x[i]) for FP16 bits x.
void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream);
/// dst = {0, rows} (the bounds of one matrix; dst on the device).
void set_bounds(int32_t* dst, int32_t rows, void* stream);

}  // namespace strata::prefill::mmq
