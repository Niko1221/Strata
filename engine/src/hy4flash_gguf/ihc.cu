#include "strata/hy4flash_gguf/ihc.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace strata::hy4_flash {
namespace {

using strata::kernels::launch_pdl;
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;

constexpr int kBlock = 256;
/// `reduce_streams_gated` holds the gates in shared memory; above this it falls back to two launches.
constexpr int kMaxGated = 64;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

/// pre and post from one mixes column. The orders differ deliberately: post multiplies by `magnitude` BEFORE
/// adding hc_eps, pre has no magnitude at all.
__global__ void gates(const float* __restrict__ mixes, const float* __restrict__ scale,
                      const float* __restrict__ base, int hc, int n_tok, int n_out, float hc_eps,
                      float magnitude, float* __restrict__ pre, float* __restrict__ post) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= hc * n_tok) return;
    const int ih = i % hc, t = i / hc;
    const float* col = mixes + (long long) t * n_out;

    // c50 rounds the scale and the add separately and takes a fast-math sigmoid. Written as intrinsics
    // because `a * b + c` would otherwise contract to one FMA, which is a different (closer) result.
    const float p = __fadd_rn(__fmul_rn(col[ih], scale[0]), base[ih]);
    pre[i] = __fdividef(1.f, 1.f + __expf(-p)) + hc_eps;
    if (post != nullptr) {
        const float q = __fadd_rn(__fmul_rn(col[hc + ih], scale[1]), base[hc + ih]);
        post[i] = __fdividef(1.f, 1.f + __expf(-q)) * magnitude + hc_eps;
    }
}

/// reduced[d, t] = sum_ih x[d, ih, t] * w[ih, t]
__global__ void reduce_streams(const float* __restrict__ x, const float* __restrict__ w, int hidden, int hc,
                              int n_tok, float* __restrict__ reduced) {
    const long long n = (long long) hidden * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int d = (int) (i % hidden);
        const long long t = i / hidden;
        const float* xr = x + t * (long long) hidden * hc + d;
        const float* wr = w + t * hc;
        float acc = 0.f;
        for (int ih = 0; ih < hc; ++ih) acc += xr[(long long) ih * hidden] * wr[ih];
        reduced[i] = acc;
    }
}

/// Fuse gate computation and reduction over the residual streams.
__global__ void reduce_streams_gated(const float* x_, const float* mixes_,
                                     const float* scale_, const float* base_,
                                     int hidden, int hc, int n_tok, int n_out, float hc_eps, float magnitude,
                                     float* post_, float* reduced_) {
    const float* STRATA_PDL_RESTRICT x = x_;
    const float* STRATA_PDL_RESTRICT mixes = mixes_;
    const float* STRATA_PDL_RESTRICT scale = scale_;   // a weight, but tiny: keep it behind the wait too
    const float* STRATA_PDL_RESTRICT base = base_;
    float* STRATA_PDL_RESTRICT post = post_;
    float* STRATA_PDL_RESTRICT reduced = reduced_;
    pdl_trigger();
    pdl_wait();
    __shared__ float s_pre[kMaxGated];
    for (int i = (int) threadIdx.x; i < hc * n_tok; i += (int) blockDim.x) {
        const int ih = i % hc, t = i / hc;
        const float* col = mixes + (long long) t * n_out;
        const float p = __fadd_rn(__fmul_rn(col[ih], scale[0]), base[ih]);
        s_pre[i] = __fdividef(1.f, 1.f + __expf(-p)) + hc_eps;
        if (post != nullptr && blockIdx.x == 0) {
            const float q = __fadd_rn(__fmul_rn(col[hc + ih], scale[1]), base[hc + ih]);
            post[i] = __fdividef(1.f, 1.f + __expf(-q)) * magnitude + hc_eps;
        }
    }
    __syncthreads();

    const long long n = (long long) hidden * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int d = (int) (i % hidden);
        const long long t = i / hidden;
        const float* xr = x + t * (long long) hidden * hc + d;
        const float* wr = s_pre + t * hc;
        float acc = 0.f;
        for (int ih = 0; ih < hc; ++ih) acc += xr[(long long) ih * hidden] * wr[ih];
        reduced[i] = acc;
    }
}

/// out[d, ih, t] = residual[d, ih, t] + y[d, t] * post[ih, t]
__global__ void post_streams(const float* y_, const float* residual_,
                            const float* post_, int hidden, int hc, int n_tok,
                            float* out_) {
    const float* STRATA_PDL_RESTRICT y = y_;
    const float* STRATA_PDL_RESTRICT residual = residual_;
    const float* STRATA_PDL_RESTRICT post = post_;
    float* STRATA_PDL_RESTRICT out = out_;
    pdl_trigger();
    pdl_wait();
    const long long n = (long long) hidden * hc * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int d = (int) (i % hidden);
        const long long rest = i / hidden;
        const int ih = (int) (rest % hc);
        const long long t = rest / hc;
        out[i] = residual[i] + y[t * hidden + d] * post[t * hc + ih];
    }
}

void launched(const char* what) {
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

unsigned grid_for(long long n) {
    const long long g = (n + kBlock - 1) / kBlock;
    return (unsigned) (g < 65535 ? (g < 1 ? 1 : g) : 65535);
}

}  // namespace

std::size_t hc_scratch_bytes(int hc_width, int n_tok) {
    if (hc_width <= 0 || n_tok <= 0) return 0;
    const std::size_t norm = (std::size_t) hc_width * (std::size_t) n_tok * sizeof(float);
    return norm + kernels::native_q8_1_bytes_wide(hc_width, n_tok);
}

void hc_flat_rms(const float* x, float* out, int hc_width, int n_tok, float eps, void* stream) {
    if (x == nullptr || out == nullptr || hc_width <= 0 || n_tok <= 0 || stream == nullptr)
        throw std::invalid_argument("hc_flat_rms: a stream, positive extents and non-null buffers");
    // The shared block-per-row RMS (kernels/elementwise): same shape, same butterfly reduction, same 256
    // threads, so this is bitwise what `flat_rms` produced - it WAS this kernel, generalised and moved so
    // that the weighted in-place form the decode path needs is not a second copy of it.
    kernels::rms_norm_block(x, nullptr, out, n_tok, hc_width, eps, stream);
}

void hc_gates(const float* mixes, const float* scale, const float* base, int hc, int n_tok, int n_out,
              float hc_eps, float magnitude, float* pre, float* post, void* stream) {
    if (mixes == nullptr || scale == nullptr || base == nullptr || pre == nullptr || hc <= 0 || n_tok <= 0 ||
        stream == nullptr)
        throw std::invalid_argument("hc_gates: a stream, positive extents and non-null buffers");
    // With a post the matrix must carry both halves; without one it carries only the pre coefficients.
    if (n_out != (post != nullptr ? 2 * hc : hc))
        throw std::invalid_argument("hc_gates: n_out must be 2 * hc with a post gate and hc without one");
    const long long n = (long long) hc * n_tok;
    gates<<<grid_for(n), kBlock, 0, static_cast<cudaStream_t>(stream)>>>(mixes, scale, base, hc, n_tok, n_out,
                                                                        hc_eps, magnitude, pre, post);
    launched("hc_gates");
}

void hc_reduce(const float* x, const float* w, int hidden, int hc, int n_tok, float* reduced, void* stream) {
    if (x == nullptr || w == nullptr || reduced == nullptr || hidden <= 0 || hc <= 0 || n_tok <= 0 ||
        stream == nullptr)
        throw std::invalid_argument("hc_reduce: a stream, positive extents and non-null buffers");
    reduce_streams<<<grid_for((long long) hidden * n_tok), kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        x, w, hidden, hc, n_tok, reduced);
    launched("hc_reduce");
}

void hc_post(const float* y, const float* residual, const float* post, int hidden, int hc, int n_tok,
             float* out, void* stream) {
    if (y == nullptr || residual == nullptr || post == nullptr || out == nullptr || hidden <= 0 || hc <= 0 ||
        n_tok <= 0 || stream == nullptr)
        throw std::invalid_argument("hc_post: a stream, positive extents and non-null buffers");
    launch_pdl(post_streams, grid_for((long long) hidden * hc * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), y, residual, post, hidden, hc, n_tok, out);
    launched("hc_post");
}

void hc_pre(const float* x, const void* fn, const float* scale, const float* base, int hidden, int hc,
            int n_tok, int n_out, float rms_eps, float hc_eps, float magnitude, void* scratch, float* mixes,
            float* pre, float* post, float* reduced, void* stream, MatCtx* mc, int fn_type) {
    if (x == nullptr || fn == nullptr || scratch == nullptr || mixes == nullptr || reduced == nullptr ||
        hidden <= 0 || hc <= 0 || n_tok <= 0 || stream == nullptr)
        throw std::invalid_argument("hc_pre: a stream, positive extents and non-null buffers");
    const int hc_width = hidden * hc;

    float* norm = static_cast<float*>(scratch);
    void* q8 = static_cast<void*>(norm + (std::size_t) hc_width * n_tok);

    // On the matvec path, produce Q8_1 directly from the norm without writing an FP32 intermediate.
    const bool gemm = mat_gemm_path(mc, n_tok, kSiteMixes);
    float* normed = gemm || fn_type == 0 || fn_type == 30 ? norm : nullptr;
    // Fuse flat RMS normalization and quantization only on the supported projection path.
    const int n4 = hc_width / 4;
    const bool fused_rmsq = hc_fuse_on() && normed == nullptr && n_tok <= 4 && hc_width % 1024 == 0 && n4 > 1024 &&
                            n4 <= 4096 && ((std::uintptr_t) x & 15) == 0;
    if (fused_rmsq) {
        hc_pre_rms_quant_fused(x, q8, hc_width, n_tok, rms_eps, stream);
    } else if (!kernels::native_rms_norm_quantize_q8_1(x, nullptr, normed, q8, n_tok, hc_width, rms_eps, stream)) {
        hc_flat_rms(x, norm, hc_width, n_tok, rms_eps, stream);
        kernels::native_quantize_q8_1_wide(norm, q8, hc_width, n_tok, stream);
        normed = norm;
    }
    // The shared pair, not a matvec of our own: Q4_K is type 12 in the engine's stable-id table. The Q8_1
    // came from the fused kernel above, so only the FP16 image is made here.
    const MatIn flat = mat_staged(mc, normed, q8, hc_width, n_tok, stream, kSiteMixes);
    mat_apply(flat, fn_type, fn, mixes, hc_width, n_out, n_tok, stream, kSiteMixes);
    // The ORIGINAL x: `norm` exists only to produce the mixes.
    if (hc * n_tok <= kMaxGated) {
        launch_pdl(reduce_streams_gated, grid_for((long long) hidden * n_tok), kBlock, 0,
                   static_cast<cudaStream_t>(stream),
                   x, mixes, scale, base, hidden, hc, n_tok, n_out, hc_eps, magnitude, post, reduced);
        launched("hc_pre gated reduce");
    } else {
        hc_gates(mixes, scale, base, hc, n_tok, n_out, hc_eps, magnitude, pre, post, stream);
        hc_reduce(x, pre, hidden, hc, n_tok, reduced, stream);
    }
}

}  // namespace strata::hy4_flash
