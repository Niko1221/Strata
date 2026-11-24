// src/hy4flash_gguf/head.cu - the graph's two ends: the token embedding into the 4 streams, and the output head.
//
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/ihc.hpp"
#include "strata/hy4flash_gguf/mat.hpp"

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include <stdexcept>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace strata::hy4_flash {
namespace {

constexpr int kBlock = 256;

/// out[d, h, t] = row[d, t] for every stream h: the reference's hc_init is a BROADCAST of one embedding into
/// all hc streams, not a per-stream projection.
__global__ __launch_bounds__(kBlock) void broadcast_streams_k(const float* __restrict__ rows,
                                                              float* __restrict__ out, int n_embd, int hc,
                                                              int n_tok) {
    const long long n = (long long) n_embd * hc * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int d = (int) (i % n_embd);
        const long long rest = i / n_embd;
        const long long t = rest / hc;
        out[i] = rows[t * n_embd + d];
    }
}

unsigned grid_for(long long n) {
    const long long g = (n + kBlock - 1) / kBlock;
    return (unsigned) (g < 1 ? 1 : (g > 65535 ? 65535 : g));
}

}  // namespace

std::size_t embed_tokens_scratch_bytes(int n_embd, int n_tok) {
    return n_tok > 0 ? (std::size_t) n_embd * n_tok * sizeof(float) + 64 : 0;
}

void embed_tokens(const void* token_embd, int type, const int32_t* tokens, int n_embd, int hc, int n_tok,
                  float* out, void* scratch, void* stream) {
    if (token_embd == nullptr || tokens == nullptr || out == nullptr || scratch == nullptr ||
        stream == nullptr)
        throw std::invalid_argument("embed_tokens: a stream and non-null buffers");
    if (n_embd <= 0 || hc <= 0 || n_tok <= 0) throw std::invalid_argument("embed_tokens: positive extents");

    // One dequantise per token into the scratch: a row of a 120,832-row quantised table, so a gather kernel
    // would have to know the quantisation, while dequant_f32's row0 slice already does.
    float* rows = static_cast<float*>(scratch);
    for (int t = 0; t < n_tok; ++t) {
        if (tokens[t] < 0) throw std::invalid_argument("embed_tokens: a negative token id");
        kernels::dequant_f32(type, token_embd, tokens[t], 1, n_embd, rows + (std::size_t) t * n_embd, stream);
    }
    // Then the broadcast into all hc streams, reading the scratch and writing `out` - so no element is read
    // after being overwritten, which an in-place version would have had to reason about.
    broadcast_streams_k<<<grid_for((long long) n_embd * hc * n_tok), kBlock, 0,
                          static_cast<cudaStream_t>(stream)>>>(rows, out, n_embd, hc, n_tok);

    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("embed_tokens: ") + cudaGetErrorString(e));
}

std::size_t output_head_scratch_bytes(int n_embd, int hc, int vocab, int n_tok) {
    if (n_tok <= 0) return 0;
    const std::size_t f = sizeof(float);
    std::size_t b = hc_scratch_bytes(n_embd * hc, n_tok);        // the reduce's own norm + Q8_1
    b += (std::size_t) hc * n_tok * f;                            // the mixes
    b += (std::size_t) hc * n_tok * f;                            // the pre gate
    b += (std::size_t) n_embd * n_tok * f;                        // the reduced hidden
    b += kernels::native_q8_1_bytes_wide(n_embd, n_tok);               // for the LM head
    return b + 256;
}

void output_head(const float* x, const void* hc_fn, int hc_fn_type, const float* hc_scale,
                 const float* hc_base, const float* output_norm, const void* output, int output_type,
                 int n_embd, int hc, int vocab, int n_tok, float rms_eps, float hc_eps, float* logits,
                 void* scratch, void* stream, MatCtx* mc, float* hidden_out, int logits_rows,
                 float* export_hidden, int export_rows) {
    if (x == nullptr || hc_fn == nullptr || hc_scale == nullptr || hc_base == nullptr ||
        output_norm == nullptr || output == nullptr || (logits == nullptr && logits_rows != 0) || scratch == nullptr ||
        stream == nullptr)
        throw std::invalid_argument("output_head: a stream and every weight");
    if (n_embd <= 0 || hc <= 0 || vocab <= 0 || n_tok <= 0)
        throw std::invalid_argument("output_head: positive extents");

    char* p = static_cast<char*>(scratch);
    auto take = [&p](std::size_t bytes) {
        void* at = p;
        p += (bytes + 15) & ~(std::size_t) 15;
        return at;
    };
    const std::size_t f = sizeof(float);
    void* hc_sc = take(hc_scratch_bytes(n_embd * hc, n_tok));
    float* mixes = (float*) take((std::size_t) hc * n_tok * f);
    float* pre = (float*) take((std::size_t) hc * n_tok * f);
    float* reduced = (float*) take((std::size_t) n_embd * n_tok * f);
    void* q8 = take(kernels::native_q8_1_bytes_wide(n_embd, n_tok));

    // The final residual reduction uses only the pre-mix coefficients.
    hc_pre(x, hc_fn, hc_scale, hc_base, n_embd, hc, n_tok, /*n_out=*/hc, rms_eps, hc_eps,
           /*magnitude=*/0.f, hc_sc, mixes, pre, /*post=*/nullptr, reduced, stream, mc, hc_fn_type);

    kernels::rms_norm_weighted_wide(reduced, output_norm, n_tok, n_embd, rms_eps, stream);
    if (hidden_out)
        cudaMemcpyAsync(hidden_out, reduced, (std::size_t) n_embd * n_tok * sizeof(float),
                        cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
    if (export_hidden != nullptr) {
        if (export_rows < 1 || export_rows > n_tok)
            throw std::invalid_argument("output_head: export_rows outside the rows computed");
        if (cudaMemcpyAsync(export_hidden, reduced + (std::size_t) (n_tok - export_rows) * n_embd,
                            (std::size_t) export_rows * n_embd * f, cudaMemcpyDeviceToDevice,
                            static_cast<cudaStream_t>(stream)) != cudaSuccess)
            throw std::runtime_error("output_head: the hidden export copy failed");
    }
    if (logits_rows == 0) return;
    if (logits_rows < 0) logits_rows = n_tok;
    if (logits_rows > n_tok) throw std::invalid_argument("output_head: too many logit rows");
    reduced += (std::size_t) (n_tok - logits_rows) * n_embd;
    if (output_type != 30) kernels::native_quantize_q8_1_wide(reduced, q8, n_embd, logits_rows, stream);
    // Keep the output head on the matvec path to bound workspace use.
    if (output_type == 30)
        kernels::bf16_gemv_fp32_mmvf_cols(reduced, static_cast<const uint16_t*>(output), logits,
                                        n_embd, vocab, logits_rows, stream);
    else
        kernels::native_mmvq(output_type, output, q8, logits, n_embd, vocab, logits_rows, stream);

    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("output_head: ") + cudaGetErrorString(e));
}

}  // namespace strata::hy4_flash
