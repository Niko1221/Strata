// Gated MLA for hy4-flash.
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/window.hpp"

#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/hy4flash_gguf/prefill_moe.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace strata::hy4_flash {
namespace {

using strata::kernels::launch_pdl;
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;

constexpr int kBlock = 256;

constexpr int kAttnTile = 64;      ///< keys the scalar attention kernels score at once
constexpr int kAttnMinChunk = 16;  ///< and the fewest it is worth giving one
constexpr int kMaxSplit = 64;      ///< chunks the K/V range may be cut into; `mla_merge_k` sizes by it

/// Read latent rows from either FP16 or FP32 storage.
template <class T>
__device__ __forceinline__ float kv_at(const T* p, long long i) {
    return (float) p[i];
}

/// A KvQ8 pointer addresses raw Q8_0 blocks.
struct KvQ8 {
    unsigned char b;
};
template <class T> struct kv_pe { using type = T; };
template <> struct kv_pe<KvQ8> { using type = __half; };
template <class T> using kv_pe_t = typename kv_pe<T>::type;

/// Token `j`'s latent row, `w` = kv_lora wide.
template <class T>
__device__ __forceinline__ const T* kv_row(const T* p, long long j, int w) {
    return p + j * w;
}
template <>
__device__ __forceinline__ const KvQ8* kv_row<KvQ8>(const KvQ8* p, long long j, int w) {
    return p + j * (w + w / 16);
}

/// Element `d` of one latent row. Dequantised in F32, where code x scale is exact (7 + 11 bits).
template <class T>
__device__ __forceinline__ float kv_lat(const T* row, int d, int w) {
    (void) w;
    return (float) row[d];
}
template <>
__device__ __forceinline__ float kv_lat<KvQ8>(const KvQ8* row, int d, int w) {
    const int8_t* q = reinterpret_cast<const int8_t*>(row);
    return (float) q[d] * __half2float(reinterpret_cast<const __half*>(q + w)[d >> 5]);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

/// The whole warp takes part in both reductions; the callers pad with -inf / 0 past the live lanes rather
/// than narrowing the mask, because naming inactive threads in __shfl_*_sync is undefined and was measured to
/// hang a kernel in this very directory (see ihc.cu's flat_rms).
__device__ __forceinline__ float warp_max_f(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}

/// Rotate adjacent pairs using the CUDA RoPE angle calculation.
template<bool Picture>
__global__ __launch_bounds__(kBlock) void rope_norm_k(const float* x_, float* out_,
                                                      int head_dim, int n_rot, int n_head, int n_tok,
                                                      const int32_t* positions_,
                                                      float theta_scale, int src_head_dim, int src_off,
                                                      int dst_head_dim, int dst_off, ImageRope image) {
    const float* STRATA_PDL_RESTRICT x = x_;   // activations: no __restrict__ where PDL runs (pdl.hpp)
    float* STRATA_PDL_RESTRICT out = out_;
    const int32_t* STRATA_PDL_RESTRICT positions = positions_;
    pdl_trigger();
    pdl_wait();
    const long long pairs = (long long) (n_rot / 2) * n_head * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < pairs;
         i += (long long) gridDim.x * blockDim.x) {
        const int j = (int) (i % (n_rot / 2));          // which pair inside the rotated span
        const long long rest = i / (n_rot / 2);
        const int h = (int) (rest % n_head);
        const long long t = rest / n_head;

        int pos = positions[t];
        if constexpr (Picture) {
            // hy4-flash interleaves its 12/10/10 rotary pairs across time, height and width.
            const int axis = j < 30 ? j % 3 : 0;
            pos = pos < image.length ? image.axes[axis * image.length + pos] : pos + image.delta;
        }
        const float theta = (float) pos * __powf(theta_scale, (float) j);
        const float c = __cosf(theta), s = __sinf(theta);

        // The rotary tail may be a slice of a wider strided head.
        const long long src = (t * n_head + h) * (long long) src_head_dim + src_off;
        // Allow the rotated output to occupy a slice of a wider row.
        const long long dst = (t * n_head + h) * (long long) dst_head_dim + dst_off;
        // NORM: the pair is ADJACENT - 2j and 2j+1. NeoX would use j and j + n_rot/2.
        const float x0 = x[src + 2 * j];
        const float x1 = x[src + 2 * j + 1];
        out[dst + 2 * j] = x0 * c - x1 * s;
        out[dst + 2 * j + 1] = x0 * s + x1 * c;
    }
}

/// The indexer's pass-through half: channels [0, nope) copied, for `rope_norm_head_tail` where the ROTATED
/// span is the tail and the front is what survives untouched. Skipped entirely when out == x.
__global__ __launch_bounds__(kBlock) void rope_front_copy_k(const float* __restrict__ x,
                                                            float* __restrict__ out, int head_dim, int nope,
                                                            int n_head, int n_tok) {
    const long long n = (long long) nope * n_head * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int d = (int) (i % nope);
        const long long rest = i / nope;
        const int h = (int) (rest % n_head);
        const long long t = rest / n_head;
        const long long at = (t * n_head + h) * (long long) head_dim + d;
        out[at] = x[at];
    }
}

/// ggml copies the channels past n_rot through unchanged; skipped entirely when out == x.
__global__ __launch_bounds__(kBlock) void rope_tail_k(const float* __restrict__ x, float* __restrict__ out,
                                                      int head_dim, int n_rot, int n_head, int n_tok) {
    const int tail = head_dim - n_rot;
    const long long n = (long long) tail * n_head * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int d = (int) (i % tail);
        const long long rest = i / tail;
        const int h = (int) (rest % n_head);
        const long long t = rest / n_head;
        const long long at = (t * n_head + h) * (long long) head_dim + n_rot + d;
        out[at] = x[at];
    }
}

unsigned grid_for(long long n) {
    const long long g = (n + kBlock - 1) / kBlock;
    return (unsigned) (g < 1 ? 1 : (g > 65535 ? 65535 : g));
}

}  // namespace

void rope_norm(const float* x, float* out, int head_dim, int n_rot, int n_head, int n_tok,
               const int32_t* positions, float freq_base, void* stream, int src_head_dim, int src_off, ImageRope image) {
    if (src_head_dim <= 0) src_head_dim = head_dim;
    if (src_off < 0 || src_off + n_rot > src_head_dim)
        throw std::invalid_argument("rope_norm: the source slice must lie inside src_head_dim");
    if (x == nullptr || out == nullptr || positions == nullptr || stream == nullptr)
        throw std::invalid_argument("rope_norm: a stream and non-null buffers");
    if (head_dim <= 0 || n_rot <= 0 || n_rot > head_dim || n_rot % 2 != 0 || n_head <= 0 || n_tok <= 0)
        throw std::invalid_argument("rope_norm: n_rot must be even and at most head_dim, extents positive");
    if (freq_base <= 0.f) throw std::invalid_argument("rope_norm: freq_base must be positive");
    // Partial overlap would have threads reading channels another thread has already written. Equal pointers
    // are fine: each pair is read and written by one thread.
    if (out != x && src_head_dim == head_dim && src_off == 0) {
        const char* a = reinterpret_cast<const char*>(x);
        const char* b = reinterpret_cast<const char*>(out);
        const size_t bytes = (size_t) head_dim * n_head * n_tok * sizeof(float);
        if (a < b + bytes && b < a + bytes)
            throw std::invalid_argument("rope_norm: out must equal x or not overlap it");
    }

    const float theta_scale = powf(freq_base, -2.0f / (float) n_rot);
    const auto kernel = image.axes ? rope_norm_k<true> : rope_norm_k<false>;
    launch_pdl(kernel, grid_for((long long) (n_rot / 2) * n_head * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), x, out, head_dim, n_rot, n_head, n_tok, positions,
               theta_scale, src_head_dim, src_off, head_dim, 0, image);
    if (out != x && head_dim > n_rot)
        rope_tail_k<<<grid_for((long long) (head_dim - n_rot) * n_head * n_tok), kBlock, 0,
                      static_cast<cudaStream_t>(stream)>>>(x, out, head_dim, n_rot, n_head, n_tok);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("rope_norm: ") + cudaGetErrorString(e));
}

void rope_norm_head_tail(const float* x, float* out, int head_dim, int n_rot, int n_head, int n_tok,
                         const int32_t* positions, float freq_base, void* stream, ImageRope image) {
    if (x == nullptr || out == nullptr || positions == nullptr || stream == nullptr)
        throw std::invalid_argument("rope_norm_head_tail: a stream and non-null buffers");
    if (head_dim <= 0 || n_rot <= 0 || n_rot >= head_dim || n_rot % 2 != 0 || n_head <= 0 || n_tok <= 0)
        throw std::invalid_argument(
            "rope_norm_head_tail: an even n_rot strictly inside head_dim (the nope front must exist), and "
            "positive extents");
    if (freq_base <= 0.f) throw std::invalid_argument("rope_norm_head_tail: freq_base must be positive");
    const int nope = head_dim - n_rot;
    // Copy the unrotated prefix only when input and output differ.
    if (out != x)
        rope_front_copy_k<<<grid_for((long long) nope * n_head * n_tok), kBlock, 0,
                            static_cast<cudaStream_t>(stream)>>>(x, out, head_dim, nope, n_head, n_tok);
    const float theta_scale = powf(freq_base, -2.0f / (float) n_rot);
    // Both RoPE layouts share the same angle kernel.
    const auto kernel = image.axes ? rope_norm_k<true> : rope_norm_k<false>;
    launch_pdl(kernel, grid_for((long long) (n_rot / 2) * n_head * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), x, out, n_rot, n_rot, n_head, n_tok, positions,
               theta_scale, head_dim, nope, head_dim, nope, image);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("rope_norm_head_tail: ") + cudaGetErrorString(e));
}

namespace {
/// How long the per-block routed-id readback costs, cumulative. See the note at the readback itself.
double g_route_sync = 0.0;
double g_route_backlog = 0.0;
long g_route_syncs = 0;

/// The quantiser's multi-column cap (MAX_NCOLS in native_mmvq.cu). A whole-head or whole-expert batch is
/// wider than that - 64 heads, 16 experts - so those calls are split into runs of this many.
constexpr int kMaxNcols = 8;

/// out[j, h, t] = src[off + j, h, t]
__global__ __launch_bounds__(kBlock) void gather_slice_k(const float* src_, float* out_,
                                                         int src_head_dim, int off, int n, int n_head,
                                                         int n_tok) {
    const float* STRATA_PDL_RESTRICT src = src_;
    float* STRATA_PDL_RESTRICT out = out_;
    pdl_trigger();
    pdl_wait();
    const long long total = (long long) n * n_head * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const int j = (int) (i % n);
        const long long rest = i / n;
        const int h = (int) (rest % n_head);
        const long long t = rest / n_head;
        out[i] = src[(t * n_head + h) * (long long) src_head_dim + off + j];
    }
}

/// Gather into head-major storage so the values for each head are contiguous.
__global__ __launch_bounds__(kBlock) void gather_heads_k(const float* src_,
                                                         float* out_, int src_head_dim, int off,
                                                         int n, int n_head, int n_tok) {
    const float* STRATA_PDL_RESTRICT src = src_;
    float* STRATA_PDL_RESTRICT out = out_;
    pdl_trigger();
    pdl_wait();
    const long long total = (long long) n * n_head * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const int j = (int) (i % n);
        const long long rest = i / n;
        const int t = (int) (rest % n_tok);
        const int h = (int) (rest / n_tok);
        out[i] = src[((long long) t * n_head + h) * src_head_dim + off + j];
    }
}

/// The scatter for EVERY head in one launch. `src` is [v_head, n_tok, n_head]; `out` is [v_head, n_head,
/// n_tok], which is what `mla_output` reads.
__global__ __launch_bounds__(kBlock) void scatter_heads_k(const float* src_,
                                                          float* out_, int v_head, int n_head,
                                                          int n_tok) {
    const float* STRATA_PDL_RESTRICT src = src_;
    float* STRATA_PDL_RESTRICT out = out_;
    pdl_trigger();
    pdl_wait();
    const long long total = (long long) v_head * n_tok * n_head;
    for (long long x = blockIdx.x * (long long) blockDim.x + threadIdx.x; x < total;
         x += (long long) gridDim.x * blockDim.x) {
        const int e = (int) (x % v_head);
        const long long rest = x / v_head;
        const int t = (int) (rest % n_tok);
        const int h = (int) (rest / n_tok);
        out[((long long) t * n_head + h) * v_head + e] = src[x];
    }
}

/// Quantize all columns in one launch with the same layout as individual column calls.
void quantize_cols(const float* x, void* out, int n_in, int n_cols, void* stream) {
    kernels::native_quantize_q8_1_wide(x, out, n_in, n_cols, stream);
}

}  // namespace

void gather_slice(const float* src, float* out, int src_head_dim, int off, int n, int n_head, int n_tok,
                  void* stream) {
    if (src == nullptr || out == nullptr || stream == nullptr)
        throw std::invalid_argument("gather_slice: a stream and non-null buffers");
    if (src_head_dim <= 0 || n <= 0 || off < 0 || off + n > src_head_dim || n_head <= 0 || n_tok <= 0)
        throw std::invalid_argument("gather_slice: [off, off + n) must lie inside src_head_dim");
    launch_pdl(gather_slice_k, grid_for((long long) n * n_head * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), src, out, src_head_dim, off, n, n_head, n_tok);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("gather_slice: ") + cudaGetErrorString(e));
}

std::size_t mla_project_scratch_bytes(int n_embd, int q_lora, int kv_lora, int qk_head, int n_rot, int n_head,
                                      int n_tok) {
    if (n_tok <= 0) return 0;
    const std::size_t f = sizeof(float);
    std::size_t bytes = 0;
    bytes += (std::size_t) q_lora * n_tok * f;                      // q_resid
    bytes += (std::size_t) qk_head * n_head * n_tok * f;            // q
    bytes += (std::size_t) n_rot * n_head * n_tok * f;              // q_pe
    bytes += (std::size_t) (kv_lora + n_rot) * n_tok * f;           // kv_pe
    bytes += (std::size_t) kv_lora * n_tok * f;                     // kv_cmpr
    bytes += (std::size_t) n_rot * n_tok * f;                       // k_pe
    // The Q8_1 staging for the three matvecs: one at a time, so the largest suffices.
    const int widest = n_embd > q_lora ? n_embd : q_lora;
    bytes += kernels::native_q8_1_bytes_wide(widest, n_tok);
    return bytes + 256;   // slack for the 16-byte alignment each slice is rounded up to
}

void mla_project(const float* cur, const void* wq_a, int wq_a_type, const float* q_a_norm, const void* wq_b,
                 int wq_b_type, const void* wkv_a_mqa, int wkv_a_mqa_type, const float* kv_a_norm, int n_embd,
                 int q_lora, int kv_lora, int qk_head, int n_rot, int n_head, int n_tok,
                 const int32_t* positions, float rms_eps, float rope_base, void* scratch, MlaProjected& out,
                 const MatIn& cur_in,
                 void* stream, ImageRope image) {
    if (cur == nullptr || wq_a == nullptr || wq_b == nullptr || wkv_a_mqa == nullptr || q_a_norm == nullptr ||
        kv_a_norm == nullptr || positions == nullptr || scratch == nullptr || stream == nullptr)
        throw std::invalid_argument("mla_project: a stream and non-null buffers");
    if (n_embd <= 0 || q_lora <= 0 || kv_lora <= 0 || qk_head <= 0 || n_rot <= 0 || n_rot >= qk_head ||
        n_head <= 0 || n_tok <= 0)
        throw std::invalid_argument("mla_project: positive extents, and n_rot below qk_head");

    // Slices of the caller's scratch, each 16-byte aligned.
    char* p = static_cast<char*>(scratch);
    auto take = [&p](std::size_t bytes) {
        float* at = reinterpret_cast<float*>(p);
        p += (bytes + 15) & ~(std::size_t) 15;
        return at;
    };
    const std::size_t f = sizeof(float);
    out.q_resid = take((std::size_t) q_lora * n_tok * f);
    out.q = take((std::size_t) qk_head * n_head * n_tok * f);
    out.q_pe = take((std::size_t) n_rot * n_head * n_tok * f);
    float* kv_pe = take((std::size_t) (kv_lora + n_rot) * n_tok * f);
    out.kv_cmpr = take((std::size_t) kv_lora * n_tok * f);
    out.k_pe = take((std::size_t) n_rot * n_tok * f);
    void* q8 = static_cast<void*>(p);

    // ---- q: wq_a, its norm, then wq_b. `cur_in` is the caller's images of `cur`; see the header. `q8` is
    // still this function's own scratch, for q_resid below.
    MatIn cur_im = cur_in;
    if (cur_im.q8 == nullptr)
        cur_im = mat_stage(cur_in.mc, cur, q8, n_embd, n_tok, stream, kSiteQA);
    mat_apply(cur_im, wq_a_type, wq_a, out.q_resid, n_embd, q_lora, n_tok, stream, kSiteQA);
    // Apply weighted RMS normalization in place.
    if (!kernels::native_rms_norm_quantize_q8_1(out.q_resid, q_a_norm, out.q_resid, q8, n_tok, q_lora,
                                                rms_eps, stream)) {
        kernels::rms_norm_weighted_wide(out.q_resid, q_a_norm, n_tok, q_lora, rms_eps, stream);
        kernels::native_quantize_q8_1_wide(out.q_resid, q8, q_lora, n_tok, stream);
    }
    // Only stage the FP16 image here. The previous step already produced Q8_1.
    out.q_resid_q8 = cur_in.q8 != nullptr ? q8 : nullptr;
    const MatIn qr = mat_staged(cur_in.mc, out.q_resid, q8, q_lora, n_tok, stream, kSiteQB);
    mat_apply(qr, wq_b_type, wq_b, out.q, q_lora, qk_head * n_head, n_tok, stream, kSiteQB);

    // The rope half is channels [qk_head - n_rot, qk_head) of each head. Gathered contiguous first, which is
    // the shape the reference's roped view comes out as.
    const int nope = qk_head - n_rot;
    rope_norm(out.q, out.q_pe, n_rot, n_rot, n_head, n_tok, positions, rope_base, stream, qk_head, nope, image);

    // ---- kv: one projection giving the latent and the rope part side by side. The same `cur` as the q
    // half, so the same Q8 - but `q8` now holds q_resid's, hence `curq` and not `q8`.
    if (cur_in.q8 == nullptr)
        cur_im = mat_stage(cur_in.mc, cur, q8, n_embd, n_tok, stream, kSiteKVA);
    mat_apply(cur_im, wkv_a_mqa_type, wkv_a_mqa, kv_pe, n_embd, kv_lora + n_rot, n_tok, stream, kSiteKVA);
    // The latent is the FIRST kv_lora channels; only it is normed. Gathered rather than normed in place
    // because the rope part shares the row and must not be touched by the norm.
    gather_slice(kv_pe, out.kv_cmpr, kv_lora + n_rot, 0, kv_lora, 1, n_tok, stream);
    kernels::rms_norm_weighted_wide(out.kv_cmpr, kv_a_norm, n_tok, kv_lora, rms_eps, stream);
    rope_norm(kv_pe, out.k_pe, n_rot, n_rot, 1, n_tok, positions, rope_base, stream, kv_lora + n_rot,
              kv_lora, image);
}

namespace {
__global__ void repeated_head_ids(int32_t* ids, int count, int tokens) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)ids[i]=i/tokens;
}
bool verify_heads(int type,int nt) {
    static const bool on=[] {const char* v=std::getenv("STRATA_HY4_FLASH_VERIFY_HEADS");return v&&std::atoi(v)!=0;}();
    return on && nt>1 && nt<=8 && type==8;
}
/// Report the operation that invalidates stream capture when the capture probe is enabled.
void capture_probe(void* stream, const char* where) {
    static const bool on=[] {const char* v=std::getenv("STRATA_HY4_FLASH_CAPTURE_PROBE");return v&&std::atoi(v)!=0;}();
    if(!on) return;
    cudaStreamCaptureStatus st{};
    if(cudaStreamGetCaptureInfo(static_cast<cudaStream_t>(stream),&st,nullptr)!=cudaSuccess) {
        cudaGetLastError();
        static int bad=0;
        if(bad++==0)std::fprintf(stderr,"HY4_FLASH_CAPTURE_PROBE(moe) query refused at %s\n",where);
        return;
    }
    if(st==cudaStreamCaptureStatusNone) return;        // not capturing: nothing to report
    if(st!=cudaStreamCaptureStatusActive) {
        static int said=0;
        if(said++==0)std::fprintf(stderr,"HY4_FLASH_CAPTURE_PROBE(moe) invalidated after %s\n",where);
    }
}
/// Reuse each head matrix across the verify window. Disable to use one group per row.
/// The mmvq_multi_parity test checks equivalence across window widths.
bool verify_heads_multi() {
    static const bool on=[] {const char* v=std::getenv("STRATA_HY4_FLASH_VERIFY_HEADS_MULTI");return !v||std::atoi(v)!=0;}();
    return on;
}
// Group queries that share a projection matrix while retaining their individual masks.
void per_query_heads_product(int type, const void* w, std::size_t head_stride, const void* q8,
                              std::size_t q8_col, float* out, int n_in, int n_out, int n_head,
                              int n_tok, const int32_t* head_of, void* stream) {
    const bool multi = verify_heads_multi() &&
        kernels::native_mmvq_grouped_multi(type, w, head_stride, q8, q8_col * (std::size_t) n_tok,
                                           out, n_in, n_out, n_head, n_tok, stream, nullptr);
    if (!multi)
        kernels::native_mmvq_grouped(type, w, head_stride, q8, q8_col, out, n_in, n_out,
                                     n_head * n_tok, stream, head_of);
}
/// The shared body of both per-head sites: `repeated_head_ids` + a grouped product, with the matrix read
/// once when the multi-column entry takes the shape and T times when it declines.
bool verify_heads_product(int type, const void* w, std::size_t head_stride, const void* q8,
                          std::size_t q8_col, float* out, int n_in, int n_out, int n_head, int n_tok,
                          int32_t* ids, void* stream, const char* site) {
    repeated_head_ids<<<(n_head*n_tok+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        ids, n_head*n_tok, n_tok);
    const bool multi = verify_heads_multi() &&
        kernels::native_mmvq_grouped_multi(type, w, head_stride, q8, q8_col*(std::size_t)n_tok, out,
                                           n_in, n_out, n_head, n_tok, stream, nullptr);
    if (!multi)
        kernels::native_mmvq_grouped_context(type, w, head_stride, q8, q8_col, out, n_in, n_out,
                                             n_head*n_tok, stream, ids, n_tok);
    if (verify_heads_multi()) {
        static int said = 0;
        if (said++ < 2)
            std::fprintf(stderr, "HY4_FLASH_VERIFY_HEADS_MULTI %s at %s: type=%d %d->%d heads=%d T=%d\n",
                         multi ? "engaged" : "DECLINED -> grouped per row", site, type, n_in, n_out,
                         n_head, n_tok);
    }
    return multi;
}
}

std::size_t mla_absorb_scratch_bytes(int nope, int n_tok, int n_head) {
    if (nope <= 0 || n_tok <= 0 || n_head <= 0) return 0;
    // EVERY head's nope rows gathered contiguous, plus their Q8_1 form. Per head it was 2 launches x 2304
    // heads a token; batched it is one gather and n_head * n_tok / 8 quantise calls.
    return (std::size_t) nope * n_tok * n_head * sizeof(float) +
           (std::size_t) n_head * n_tok * kernels::native_q8_1_bytes(nope, 1) +
           // the batched per-head product's own buffers, and ZERO at or below kMoeBatchMin tokens
           grouped_mmq_scratch_bytes(12, nope, n_head, n_tok) + (std::size_t)n_head*n_tok*4 + 64;
}

void mla_absorb(const float* q, const void* wk_b, int wk_b_type, int qk_head, int nope, int kv_lora,
                int n_head, int n_tok, float* out, void* scratch, void* stream, const MlaPerQuery* per_query) {
    if (q == nullptr || wk_b == nullptr || out == nullptr || scratch == nullptr || stream == nullptr)
        throw std::invalid_argument("mla_absorb: a stream and non-null buffers");
    if (qk_head <= 0 || nope <= 0 || nope > qk_head || kv_lora <= 0 || n_head <= 0 || n_tok <= 0)
        throw std::invalid_argument("mla_absorb: positive extents, and nope at most qk_head");

    char* p = static_cast<char*>(scratch);
    float* gathered = reinterpret_cast<float*>(p);
    p += ((std::size_t) nope * n_tok * n_head * sizeof(float) + 15) & ~(std::size_t) 15;
    void* q8 = static_cast<void*>(p);
    p += ((std::size_t) n_head * n_tok * kernels::native_q8_1_bytes(nope, 1) + 15) & ~(std::size_t) 15;
    void* gsc = static_cast<void*>(p);   // the batched per-head product's, sized above

    // Respect the query and weight strides when indexing the per-head projections.
    launch_pdl(gather_heads_k, grid_for((long long) nope * n_head * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), q, gathered, qk_head, 0, nope, n_head, n_tok);
    quantize_cols(gathered, q8, nope, n_head * n_tok, stream);

    const std::size_t head_stride = wk_b_type == 30 ? (std::size_t) nope * kv_lora * 2 :
        kernels::native_mmvq_weight_bytes(wk_b_type, nope, kv_lora);
    const std::size_t q8_col = kernels::native_q8_1_bytes(nope, 1);
    // Every (head, token) column keeps the one-token reduction. Columns of a head
    // are contiguous, so a supported multi-column product reuses that head's
    // weights while preserving the [kv_lora, n_tok, n_head] output layout.
    if (per_query != nullptr) {
        if (per_query->head_of == nullptr || !kernels::native_mmvq_grouped_supported(wk_b_type, 1))
            throw std::invalid_argument("mla_absorb: per-query mode needs head_of and a grouped wk_b type");
        per_query_heads_product(wk_b_type, wk_b, head_stride, q8, q8_col, out, nope, kv_lora,
                                 n_head, n_tok, per_query->head_of, stream);
        const auto e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("mla_absorb: ") + cudaGetErrorString(e));
        return;
    }
    // Each head uses its own matrix and activation.
    const std::size_t gq = grouped_mmq_scratch_bytes(wk_b_type, nope, n_head, n_tok);
    const bool grouped_batched =
        gq != 0 && gq <= grouped_mmq_scratch_bytes(12, nope, n_head, n_tok) &&
        grouped_product_mmq(wk_b, wk_b_type, head_stride, gathered, nope, kv_lora, n_head, n_tok, out, gsc,
                            gq, stream);
    if (n_tok > 8) {
        static int said = 0;
        if (said++ == 0)
            std::fprintf(stderr, "hy4_flash: wk_b type %d (%d -> %d, %d heads, %d tokens) %s\n", wk_b_type,
                         nope, kv_lora, n_head, n_tok,
                         grouped_batched ? "BATCHED" : "on the per-head matvecs");
    }
    if (grouped_batched) {
        // nothing more
    } else if (verify_heads(wk_b_type,n_tok)) {
        auto* ids=static_cast<int32_t*>(gsc);
        verify_heads_product(wk_b_type,wk_b,head_stride,q8,q8_col,out,nope,kv_lora,n_head,n_tok,ids,
                             stream,"k_b");
    } else if (wk_b_type == 30) {
        for (int h = 0; h < n_head; ++h)
            kernels::bf16_gemv_fp32_mmvf_cols(gathered + (std::size_t) h * nope * n_tok,
                reinterpret_cast<const uint16_t*>(static_cast<const char*>(wk_b) + h * head_stride),
                out + (std::size_t) h * kv_lora * n_tok, nope, kv_lora, n_tok, stream);
    } else if (kernels::native_mmvq_grouped_supported(wk_b_type, n_tok)) {
        kernels::native_mmvq_grouped(wk_b_type, wk_b, head_stride, q8, n_tok * q8_col, out, nope, kv_lora,
                                     n_head, stream);
    } else {
        for (int h = 0; h < n_head; ++h)
            kernels::native_mmvq(wk_b_type, static_cast<const char*>(wk_b) + (std::size_t) h * head_stride,
                                 static_cast<const char*>(q8) + (std::size_t) h * n_tok * q8_col,
                                 out + (std::size_t) h * kv_lora * n_tok, nope, kv_lora, n_tok, stream);
    }
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("mla_absorb: ") + cudaGetErrorString(e));
}

namespace {

/// Score and normalize visible keys with the attention sink for each head and query.
template <class KvT>
__global__ __launch_bounds__(kBlock) void mla_scores_k(const float* __restrict__ q_absorbed,
                                                       const float* __restrict__ q_pe,
                                                       const KvT* __restrict__ kv_cmpr,
                                                       const kv_pe_t<KvT>* __restrict__ k_pe,
                                                       const float* __restrict__ sinks, int kv_lora, int n_rot,
                                                       int n_head, int n_tok, int n_kv_max, int kv_stride,
                                                       const int32_t* __restrict__ d_n_kv,
                                                       const int32_t* __restrict__ q_pos,
                                                       const int32_t* __restrict__ kv_pos, int causal_window,
                                                       float kq_scale, float* __restrict__ prob) {
    // Read the live KV length from device memory during graph replay.
    const int n_kv = d_n_kv != nullptr ? (int) *d_n_kv : n_kv_max;
    const int hi = (int) blockIdx.x;                 // head * n_tok + query
    const int h = hi / n_tok, i = hi % n_tok;
    if (h >= n_head) return;

    // q_absorbed is [kv_lora, n_tok, n_head] (heads slowest); q_pe is [n_rot, n_head, n_tok].
    const float* qa = q_absorbed + ((long long) h * n_tok + i) * kv_lora;
    const float* qp = q_pe + ((long long) i * n_head + h) * n_rot;
    float* p = prob + ((long long) h * n_tok + i) * kv_stride;
    const int pos_i = q_pos[i];

    __shared__ float s_max, s_sum;
    float local_max = -INFINITY;

    for (int j = threadIdx.x; j < n_kv; j += kBlock) {
        const int pos_j = kv_pos[j];
        // Causal, plus the window when one is given. `causal_window` counts the positions a query may see
        // INCLUDING itself, so 512 permits a lookback of 511 - which is what is_masked_swa does with
        // p1 - p0 >= n_swa.
        const bool masked = pos_j > pos_i || (causal_window > 0 && pos_i - pos_j >= causal_window);
        float s = -INFINITY;
        if (!masked) {
            float acc = 0.f;
            const KvT* kc = kv_row(kv_cmpr, j, kv_lora);
            for (int d = 0; d < kv_lora; ++d) acc += qa[d] * kv_lat(kc, d, kv_lora);
            for (int d = 0; d < n_rot; ++d) acc += qp[d] * kv_at(k_pe, (long long) j * n_rot + d);
            s = acc * kq_scale;
        }
        p[j] = s;
        local_max = fmaxf(local_max, s);
    }
    local_max = warp_max_f(local_max);
    __shared__ float warp_max_s[kBlock / 32];
    if ((threadIdx.x & 31) == 0) warp_max_s[threadIdx.x / 32] = local_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kBlock / 32 ? warp_max_s[threadIdx.x] : -INFINITY;
        v = warp_max_f(v);
        if (threadIdx.x == 0) {
            // ggml folds the sink into the max BEFORE exponentiating.
            if (sinks != nullptr) v = fmaxf(v, sinks[h]);
            s_max = v;
        }
    }
    __syncthreads();

    float local_sum = 0.f;
    for (int j = threadIdx.x; j < n_kv; j += kBlock) {
        const float e = p[j] == -INFINITY ? 0.f : __expf(p[j] - s_max);
        p[j] = e;
        local_sum += e;
    }
    local_sum = warp_sum(local_sum);
    __shared__ float warp_sum_s[kBlock / 32];
    if ((threadIdx.x & 31) == 0) warp_sum_s[threadIdx.x / 32] = local_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kBlock / 32 ? warp_sum_s[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if (threadIdx.x == 0) {
            // The sink adds to the DENOMINATOR only - it has no output slot.
            if (sinks != nullptr) v += __expf(sinks[h] - s_max);
            s_sum = v;
        }
    }
    __syncthreads();

    const float inv = s_sum > 0.f ? 1.f / s_sum : 0.f;
    for (int j = threadIdx.x; j < n_kv; j += kBlock) p[j] *= inv;
}

/// Fuse scoring, online softmax and latent reduction without storing the probability matrix.
template <class KvT, int kTile>
__global__ __launch_bounds__(kBlock) void mla_attend_k(
    const float* __restrict__ q_absorbed, const float* __restrict__ q_pe,
    const KvT* __restrict__ kv_cmpr, const kv_pe_t<KvT>* __restrict__ k_pe, const float* __restrict__ sinks,
    int kv_lora, int n_rot, int n_head, int n_tok, int n_kv_max, const int32_t* __restrict__ d_n_kv,
    const int32_t* __restrict__ q_pos, const int32_t* __restrict__ kv_pos, int causal_window,
    float kq_scale, float* __restrict__ latent) {
    const int n_kv = d_n_kv != nullptr ? (int) *d_n_kv : n_kv_max;
    const int hi = (int) blockIdx.x;
    const int h = hi / n_tok, i = hi % n_tok;
    if (h >= n_head) return;

    extern __shared__ float sm[];
    float* qs = sm;                 // the absorbed query, read once per block instead of once per key
    float* qps = qs + kv_lora;      // and its rope half
    float* sc = qps + n_rot;        // one tile of scores
    __shared__ float m_run_s, den_s, scale_s;

    const float* qa = q_absorbed + ((long long) h * n_tok + i) * kv_lora;
    const float* qp = q_pe + ((long long) i * n_head + h) * n_rot;
    for (int d = threadIdx.x; d < kv_lora; d += kBlock) qs[d] = qa[d];
    for (int d = threadIdx.x; d < n_rot; d += kBlock) qps[d] = qp[d];
    if (threadIdx.x == 0) {
        m_run_s = sinks != nullptr ? sinks[h] : -INFINITY;
        den_s = sinks != nullptr ? 1.0f : 0.0f;
    }
    __syncthreads();

    const int pos_i = q_pos[i];
    const int lane = (int) (threadIdx.x & 31), wp = (int) (threadIdx.x >> 5);
    constexpr int kWarps = kBlock / 32;
    constexpr int kPerThread = 4;   // kv_lora <= kPerThread * kBlock
    float o[kPerThread] = {0.f, 0.f, 0.f, 0.f};

    for (int t0 = 0; t0 < n_kv; t0 += kTile) {
        const int nt = min(kTile, n_kv - t0);
        // ---- scores: a WARP to a key, each lane a slice of the dot.
        for (int jj = wp; jj < nt; jj += kWarps) {
            const int j = t0 + jj;
            const int pos_j = kv_pos[j];
            // Causal, plus the window when one is given. `causal_window` counts the positions a query may
            // see INCLUDING itself, so 512 permits a lookback of 511 - which is what is_masked_swa does
            // with p1 - p0 >= n_swa.
            const bool masked = pos_j > pos_i || (causal_window > 0 && pos_i - pos_j >= causal_window);
            float acc = 0.f;
            if (!masked) {
                const KvT* kc = kv_row(kv_cmpr, (long long) j, kv_lora);
                for (int d = lane; d < kv_lora; d += 32) acc += qs[d] * kv_lat(kc, d, kv_lora);
                const kv_pe_t<KvT>* kp = k_pe + (long long) j * n_rot;
                for (int d = lane; d < n_rot; d += 32) acc += qps[d] * kv_at(kp, d);
                acc = warp_sum(acc) * kq_scale;
            } else {
                acc = -INFINITY;
            }
            if (lane == 0) sc[jj] = acc;
        }
        __syncthreads();
        // ---- online softmax over this tile: one warp does the two reductions.
        if (threadIdx.x < 32) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[j]);
            m = warp_max_f(m);
            const float m_new = fmaxf(m_run_s, m);
            // Everything masked so far: no rescale, no contribution, and no inf - inf.
            const float scale = (m_run_s == -INFINITY || m_new == -INFINITY) ? 1.0f : __expf(m_run_s - m_new);
            float sum = 0.f;
            for (int j = lane; j < nt; j += 32) {
                const float e = (sc[j] == -INFINITY || m_new == -INFINITY) ? 0.f : __expf(sc[j] - m_new);
                sc[j] = e;
                sum += e;
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                den_s = den_s * scale + sum;
                m_run_s = m_new;
                scale_s = scale;
            }
        }
        __syncthreads();
        // ---- and the weighted sum, straight into registers.
        const float scale = scale_s;
#pragma unroll
        for (int r = 0; r < kPerThread; ++r) o[r] *= scale;
        for (int jj = 0; jj < nt; ++jj) {
            const float p = sc[jj];
            if (p == 0.f) continue;
            const KvT* kc = kv_row(kv_cmpr, (long long) (t0 + jj), kv_lora);
#pragma unroll
            for (int r = 0; r < kPerThread; ++r) {
                const int d = (int) threadIdx.x + r * kBlock;
                if (d < kv_lora) o[r] += p * kv_lat(kc, d, kv_lora);
            }
        }
        __syncthreads();
    }

    const float inv = den_s > 0.f ? 1.f / den_s : 0.f;
    float* out = latent + ((long long) h * n_tok + i) * kv_lora;
#pragma unroll
    for (int r = 0; r < kPerThread; ++r) {
        const int d = (int) threadIdx.x + r * kBlock;
        if (d < kv_lora) out[d] = o[r] * inv;
    }
}

/// Split the KV range across blocks and merge their partial softmax reductions.
template <class KvT, int kTile>
__global__ __launch_bounds__(kBlock) void mla_attend_split_k(
    const float* __restrict__ q_absorbed, const float* __restrict__ q_pe,
    const KvT* __restrict__ kv_cmpr, const kv_pe_t<KvT>* __restrict__ k_pe, int kv_lora, int n_rot, int n_head,
    int n_tok, int n_kv_max, int n_split, const int32_t* __restrict__ d_n_kv,
    const int32_t* __restrict__ q_pos, const int32_t* __restrict__ kv_pos, int causal_window,
    float kq_scale, float* __restrict__ part, float* __restrict__ pmd) {
    const int n_kv = d_n_kv != nullptr ? (int) *d_n_kv : n_kv_max;
    const int bi = (int) blockIdx.x;
    const int s = bi % n_split;
    const int hi = bi / n_split;
    const int h = hi / n_tok, i = hi % n_tok;
    if (h >= n_head) return;

    // Contiguous slices of the LIVE range, so a short sequence leaves the tail chunks empty instead of
    // interleaving and reading the cache with a stride.
    const int per = (n_kv + n_split - 1) / n_split;
    const int begin = s * per;
    const int end = min(n_kv, begin + per);
    const long long pm = ((long long) hi * n_split + s) * 2;
    if (begin >= end) {
        if (threadIdx.x == 0) {
            pmd[pm] = -INFINITY;
            pmd[pm + 1] = 0.f;
        }
        return;
    }

    extern __shared__ float sm[];
    float* qs = sm;                 // the absorbed query, read once per block instead of once per key
    float* qps = qs + kv_lora;      // and its rope half
    float* sc = qps + n_rot;        // one tile of scores
    __shared__ float m_run_s, den_s, scale_s;

    const float* qa = q_absorbed + ((long long) h * n_tok + i) * kv_lora;
    const float* qp = q_pe + ((long long) i * n_head + h) * n_rot;
    for (int d = threadIdx.x; d < kv_lora; d += kBlock) qs[d] = qa[d];
    for (int d = threadIdx.x; d < n_rot; d += kBlock) qps[d] = qp[d];
    if (threadIdx.x == 0) {
        // No sink: this is a PARTIAL, and the sink belongs to the whole row. See the note above.
        m_run_s = -INFINITY;
        den_s = 0.f;
    }
    __syncthreads();

    const int pos_i = q_pos[i];
    const int lane = (int) (threadIdx.x & 31), wp = (int) (threadIdx.x >> 5);
    constexpr int kWarps = kBlock / 32;
    constexpr int kPerThread = 4;   // kv_lora <= kPerThread * kBlock
    float o[kPerThread] = {0.f, 0.f, 0.f, 0.f};

    for (int t0 = begin; t0 < end; t0 += kTile) {
        const int nt = min(kTile, end - t0);
        // ---- scores: a WARP to a key, each lane a slice of the dot.
        for (int jj = wp; jj < nt; jj += kWarps) {
            const int j = t0 + jj;
            const int pos_j = kv_pos[j];
            const bool masked = pos_j > pos_i || (causal_window > 0 && pos_i - pos_j >= causal_window);
            float acc = 0.f;
            if (!masked) {
                const KvT* kc = kv_row(kv_cmpr, (long long) j, kv_lora);
                for (int d = lane; d < kv_lora; d += 32) acc += qs[d] * kv_lat(kc, d, kv_lora);
                const kv_pe_t<KvT>* kp = k_pe + (long long) j * n_rot;
                for (int d = lane; d < n_rot; d += 32) acc += qps[d] * kv_at(kp, d);
                acc = warp_sum(acc) * kq_scale;
            } else {
                acc = -INFINITY;
            }
            if (lane == 0) sc[jj] = acc;
        }
        __syncthreads();
        // ---- online softmax over this tile: one warp does the two reductions.
        if (threadIdx.x < 32) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[j]);
            m = warp_max_f(m);
            const float m_new = fmaxf(m_run_s, m);
            const float scale = (m_run_s == -INFINITY || m_new == -INFINITY) ? 1.0f : __expf(m_run_s - m_new);
            float sum = 0.f;
            for (int j = lane; j < nt; j += 32) {
                const float e = (sc[j] == -INFINITY || m_new == -INFINITY) ? 0.f : __expf(sc[j] - m_new);
                sc[j] = e;
                sum += e;
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                den_s = den_s * scale + sum;
                m_run_s = m_new;
                scale_s = scale;
            }
        }
        __syncthreads();
        // ---- and the weighted sum, straight into registers.
        const float scale = scale_s;
#pragma unroll
        for (int r = 0; r < kPerThread; ++r) o[r] *= scale;
        for (int jj = 0; jj < nt; ++jj) {
            const float p = sc[jj];
            if (p == 0.f) continue;
            const KvT* kc = kv_row(kv_cmpr, (long long) (t0 + jj), kv_lora);
#pragma unroll
            for (int r = 0; r < kPerThread; ++r) {
                const int d = (int) threadIdx.x + r * kBlock;
                if (d < kv_lora) o[r] += p * kv_lat(kc, d, kv_lora);
            }
        }
        __syncthreads();
    }

    // UNNORMALISED, and with this chunk's own max: dividing here would throw away what the merge needs.
    if (threadIdx.x == 0) {
        pmd[pm] = m_run_s;
        pmd[pm + 1] = den_s;
    }
    if (m_run_s == -INFINITY) return;   // every key masked; the merge skips it and never reads `part`
    float* out = part + ((long long) hi * n_split + s) * kv_lora;
#pragma unroll
    for (int r = 0; r < kPerThread; ++r) {
        const int d = (int) threadIdx.x + r * kBlock;
        if (d < kv_lora) out[d] = o[r];
    }
}

/// Group heads within a block to share reads of the latent tile.
template <class KvT, int kTile, int kG>
__global__ __launch_bounds__(kBlock) void mla_attend_group_k(
    const float* __restrict__ q_absorbed, const float* __restrict__ q_pe,
    const KvT* __restrict__ kv_cmpr, const kv_pe_t<KvT>* __restrict__ k_pe, int kv_lora, int n_rot, int n_head,
    int n_tok, int n_kv_max, int n_split, const int32_t* __restrict__ d_n_kv,
    const int32_t* __restrict__ q_pos, const int32_t* __restrict__ kv_pos, int causal_window,
    float kq_scale, float* __restrict__ part, float* __restrict__ pmd) {
    const int n_kv = d_n_kv != nullptr ? (int) *d_n_kv : n_kv_max;
    const int bi = (int) blockIdx.x;
    const int s = bi % n_split;
    const int gi = bi / n_split;                  // (head group, token)
    const int grp = gi / n_tok, i = gi % n_tok;
    const int h0 = grp * kG;
    if (h0 >= n_head) return;

    const int per = (n_kv + n_split - 1) / n_split;
    const int begin = s * per;
    const int end = min(n_kv, begin + per);
    if (begin >= end) {
        if (threadIdx.x < kG) {
            const int h = h0 + (int) threadIdx.x;
            if (h < n_head) {
                const long long pm = ((long long) (h * n_tok + i) * n_split + s) * 2;
                pmd[pm] = -INFINITY;
                pmd[pm + 1] = 0.f;
            }
        }
        return;
    }

    extern __shared__ float sm[];
    float* qs = sm;                               // kG x (kv_lora + n_rot): the group's queries
    float* sc = qs + (std::size_t) kG * (kv_lora + n_rot);   // kG x kTile: their scores
    __shared__ float m_run_s[kG], den_s[kG], scale_s[kG];

    for (int g = 0; g < kG; ++g) {
        const int h = h0 + g;
        if (h >= n_head) break;
        const float* qa = q_absorbed + ((long long) h * n_tok + i) * kv_lora;
        const float* qp = q_pe + ((long long) i * n_head + h) * n_rot;
        float* dst = qs + (std::size_t) g * (kv_lora + n_rot);
        for (int d = threadIdx.x; d < kv_lora; d += kBlock) dst[d] = qa[d];
        for (int d = threadIdx.x; d < n_rot; d += kBlock) dst[kv_lora + d] = qp[d];
    }
    if (threadIdx.x < kG) {
        m_run_s[threadIdx.x] = -INFINITY;         // no sink: this is a partial, the merge folds it in
        den_s[threadIdx.x] = 0.f;
    }
    __syncthreads();

    const int pos_i = q_pos[i];
    const int lane = (int) (threadIdx.x & 31), wp = (int) (threadIdx.x >> 5);
    constexpr int kWarps = kBlock / 32;
    constexpr int kPerThread = 4;                 // kv_lora <= kPerThread * kBlock
    constexpr int kMaxLane = 18;                  // (kv_lora + n_rot) / 32, rounded up, at 576
    float o[kG][kPerThread] = {};

    for (int t0 = begin; t0 < end; t0 += kTile) {
        const int nt = min(kTile, end - t0);
        // ---- scores: a WARP owns one key, holds it in registers, and dots it against ALL kG queries.
        for (int jj = wp; jj < nt; jj += kWarps) {
            const int j = t0 + jj;
            const int pos_j = kv_pos[j];
            const bool masked = pos_j > pos_i || (causal_window > 0 && pos_i - pos_j >= causal_window);
            if (masked) {
                if (lane == 0)
                    for (int g = 0; g < kG; ++g) sc[(std::size_t) g * kTile + jj] = -INFINITY;
                continue;
            }
            // ONE read of this key, reused kG times. `kv_cmpr` then `k_pe`, in the queries' own order.
            float kr[kMaxLane];
            const KvT* kc = kv_row(kv_cmpr, (long long) j, kv_lora);
            const kv_pe_t<KvT>* kp = k_pe + (long long) j * n_rot;
            const int ntot = kv_lora + n_rot;
#pragma unroll
            for (int r = 0; r < kMaxLane; ++r) {
                const int d = lane + r * 32;
                kr[r] = d < kv_lora ? kv_lat(kc, d, kv_lora) : (d < ntot ? kv_at(kp, d - kv_lora) : 0.f);
            }
            for (int g = 0; g < kG; ++g) {
                if (h0 + g >= n_head) break;
                const float* q = qs + (std::size_t) g * ntot;
                float acc = 0.f;
#pragma unroll
                for (int r = 0; r < kMaxLane; ++r) {
                    const int d = lane + r * 32;
                    if (d < ntot) acc += q[d] * kr[r];
                }
                acc = warp_sum(acc) * kq_scale;
                if (lane == 0) sc[(std::size_t) g * kTile + jj] = acc;
            }
        }
        __syncthreads();
        // ---- online softmax, one warp a head.
        if (wp < kG && h0 + wp < n_head) {
            float* row = sc + (std::size_t) wp * kTile;
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, row[j]);
            m = warp_max_f(m);
            const float m_old = m_run_s[wp];
            const float m_new = fmaxf(m_old, m);
            const float scale = (m_old == -INFINITY || m_new == -INFINITY) ? 1.0f : __expf(m_old - m_new);
            float sum = 0.f;
            for (int j = lane; j < nt; j += 32) {
                const float e = (row[j] == -INFINITY || m_new == -INFINITY) ? 0.f : __expf(row[j] - m_new);
                row[j] = e;
                sum += e;
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                den_s[wp] = den_s[wp] * scale + sum;
                m_run_s[wp] = m_new;
                scale_s[wp] = scale;
            }
        }
        __syncthreads();
        // ---- the weighted sum, and the SAME read of the key serves kG accumulators.
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            const float sg = scale_s[g];
#pragma unroll
            for (int r = 0; r < kPerThread; ++r) o[g][r] *= sg;
        }
        for (int jj = 0; jj < nt; ++jj) {
            const KvT* kc = kv_row(kv_cmpr, (long long) (t0 + jj), kv_lora);
#pragma unroll
            for (int r = 0; r < kPerThread; ++r) {
                const int d = (int) threadIdx.x + r * kBlock;
                if (d >= kv_lora) continue;
                const float kv = kv_lat(kc, d, kv_lora);
#pragma unroll
                for (int g = 0; g < kG; ++g) {
                    const float p = sc[(std::size_t) g * kTile + jj];
                    if (p != 0.f) o[g][r] += p * kv;
                }
            }
        }
        __syncthreads();
    }

    for (int g = 0; g < kG; ++g) {
        const int h = h0 + g;
        if (h >= n_head) break;
        const long long hi = (long long) h * n_tok + i;
        if (threadIdx.x == 0) {
            pmd[(hi * n_split + s) * 2] = m_run_s[g];
            pmd[(hi * n_split + s) * 2 + 1] = den_s[g];
        }
        if (m_run_s[g] == -INFINITY) continue;
        float* out = part + (hi * n_split + s) * kv_lora;
#pragma unroll
        for (int r = 0; r < kPerThread; ++r) {
            const int d = (int) threadIdx.x + r * kBlock;
            if (d < kv_lora) out[d] = o[g][r];
        }
    }
}

/// Tensor-core MLA adapted from project-maya's mla_attn_mma_kernel (third_party/project-maya/NOTICE.md).
/// Each block processes 32 heads against shared KV rows.
constexpr int kMmaH = 32;             ///< heads a block
constexpr int kMmaC = 32;             ///< keys a tile
constexpr int kMmaV = 512;            ///< the latent, and the width of the weighted sum
constexpr int kMmaRot = 64;           ///< the rope key
constexpr int kMmaD = kMmaV + kMmaRot;  ///< 576: the score's whole k extent
constexpr int kMmaLd = kMmaD + 8;     ///< padded shared row, in halves (16 B keeps ldmatrix off one bank)
constexpr float kMmaPScale = 16384.f;  ///< 2^14: exp(s - m) is in [0, 1], so this uses FP16's whole range

#if !defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_USE_HIP)

/// ldmatrix needs sm_75 and this mma sm_80. Below that these compile to nothing and the dispatch never picks
/// the kernel (it asks the device for its compute capability first).
__device__ __forceinline__ void mma_ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#else
    (void) p;
    r0 = r1 = r2 = r3 = 0u;
#endif
}
__device__ __forceinline__ void mma_ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
#else
    (void) p;
    r0 = r1 = 0u;
#endif
}
/// The transposing form, which is what turns the [key][dim] latent tile into the weighted sum's B operand.
__device__ __forceinline__ void mma_ldsm_x2_t(uint32_t& r0, uint32_t& r1, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
#else
    (void) p;
    r0 = r1 = 0u;
#endif
}
__device__ __forceinline__ void mma_m16n8k16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

/// Eight cache elements to eight packed halves. The F16 cache is already the operand type and the operand
/// width, so that path is one 16-byte load; the F32 one (STRATA_HY4_FLASH_KV_F32=1) converts.
template <class KvT>
__device__ __forceinline__ uint4 mma_pack8(const KvT* p) {
    if constexpr (std::is_same<KvT, __half>::value) {
        return *(const uint4*) p;
    } else {
        const float4 a = ((const float4*) p)[0], b = ((const float4*) p)[1];
        __align__(16) __half h[8] = {__float2half_rn(a.x), __float2half_rn(a.y), __float2half_rn(a.z),
                                     __float2half_rn(a.w), __float2half_rn(b.x), __float2half_rn(b.y),
                                     __float2half_rn(b.z), __float2half_rn(b.w)};
        return *(const uint4*) h;
    }
}

/// Eight latent elements of token `slot`, from column `c` (a multiple of 8), as packed halves. For the Q8_0
/// latent the eight codes share one scale - 8 divides 32 - and code x scale is rounded to F16 once, the same
/// single rounding an F16 cache made when it stored the value.
template <class KvT>
__device__ __forceinline__ uint4 mma_lat8(const KvT* kv_cmpr, int slot, int c) {
    return mma_pack8(kv_cmpr + (long long) slot * kMmaV + c);
}
template <>
__device__ __forceinline__ uint4 mma_lat8<KvQ8>(const KvQ8* kv_cmpr, int slot, int c) {
    const unsigned char* row = reinterpret_cast<const unsigned char*>(kv_row(kv_cmpr, (long long) slot, kMmaV));
    const uint2 raw = *(const uint2*) (row + c);
    const float d = __half2float(reinterpret_cast<const __half*>(row + kMmaV)[c >> 5]);
    const int8_t* q = reinterpret_cast<const int8_t*>(&raw);
    __align__(16) __half h[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) h[k] = __float2half_rn((float) q[k] * d);
    return *(const uint4*) h;
}

/// The shared-memory bill: kQS query planes, one latent tile, kPS probability planes, one score tile.
template <int kQS, int kPS>
constexpr std::size_t mma_smem() {
    return ((std::size_t) kQS * kMmaH * kMmaLd + (std::size_t) kMmaC * kMmaLd +
            (std::size_t) kPS * kMmaH * (kMmaC + 8)) * sizeof(__half) +
           (std::size_t) kMmaH * kMmaC * sizeof(float);
}

/// One bit a cache slot, rounded up to a 32-bit word, a query row.
__host__ __device__ inline int sel_mask_words(int n_kv) { return (n_kv + 31) / 32; }

/// Encode selected absolute positions as a per-query mask over cache slots.
__global__ void sel_scatter_k(const int32_t* sel_, const int32_t* kv_pos_, uint32_t* mask_, int n_sel,
                              int n_tok, int n_kv_max, int words, const int32_t* d_n_kv_,
                              const int32_t* d_n_kv_q_) {
    const int32_t* STRATA_PDL_RESTRICT sel = sel_;
    const int32_t* STRATA_PDL_RESTRICT kv_pos = kv_pos_;
    uint32_t* STRATA_PDL_RESTRICT mask = mask_;
    const int32_t* STRATA_PDL_RESTRICT d_n_kv = d_n_kv_;
    const int32_t* STRATA_PDL_RESTRICT d_n_kv_q = d_n_kv_q_;   // MlaPerQuery::n_kv, or null
    pdl_trigger();
    pdl_wait();
    const int n_kv = d_n_kv_q != nullptr ? d_n_kv_q[blockIdx.x] : d_n_kv != nullptr ? *d_n_kv : n_kv_max;
    // Clear the query mask before setting any bits within the block.
    const int i = (int) blockIdx.x;                      // the query
    uint32_t* row = mask + (long long) i * words;
    for (int w = (int) threadIdx.x; w < words; w += (int) blockDim.x) row[w] = 0u;
    __syncthreads();
    // Read each query's selection from contiguous memory.
    for (int t = (int) threadIdx.x; t < n_sel; t += (int) blockDim.x) {
        const int32_t pos = sel[(long long) i * n_sel + t];
        if (pos < 0) continue;                           // a padding slot in a row shorter than n_sel
        int slot = pos < n_kv && kv_pos[pos] == pos ? pos : -1;
        if (slot < 0)
            for (int j = 0; j < n_kv; ++j)
                if (kv_pos[j] == pos) { slot = j; break; }
        if (slot >= 0) atomicOr(&row[slot >> 5], 1u << (slot & 31));
    }
}

/// Gather selected cache slots directly. A negative slot marks an invalid entry.
__global__ void sel_slots_k(const int32_t* sel_, const int32_t* kv_pos_, int32_t* slots_, int n_sel,
                            int n_tok, int n_kv_max, const int32_t* d_n_kv_, const int32_t* d_n_kv_q_) {
    const int32_t* STRATA_PDL_RESTRICT sel = sel_;
    const int32_t* STRATA_PDL_RESTRICT kv_pos = kv_pos_;
    int32_t* STRATA_PDL_RESTRICT slots = slots_;
    const int32_t* STRATA_PDL_RESTRICT d_n_kv = d_n_kv_;
    const int32_t* STRATA_PDL_RESTRICT d_n_kv_q = d_n_kv_q_;   // MlaPerQuery::n_kv, or null
    pdl_trigger();
    pdl_wait();
    const int n_kv_all = d_n_kv != nullptr ? *d_n_kv : n_kv_max;
    const long long n = (long long) n_sel * n_tok;
    for (long long x = (long long) blockIdx.x * blockDim.x + threadIdx.x; x < n;
         x += (long long) gridDim.x * blockDim.x) {
        const int n_kv = d_n_kv_q != nullptr ? d_n_kv_q[x / n_sel] : n_kv_all;
        const int32_t pos = sel[x];
        int slot = -1;
        if (pos >= 0) {
            // Same two-step as `sel_scatter_k`: the fast path for a cache filled in order, the search for
            // one that was ever rebased.
            if (pos < n_kv && kv_pos[pos] == pos) {
                slot = pos;
            } else {
                for (int j = 0; j < n_kv; ++j)
                    if (kv_pos[j] == pos) { slot = j; break; }
            }
        }
        slots[x] = (int32_t) slot;
    }
}

inline void sel_slots(const int32_t* sel, int n_sel, int n_tok, const int32_t* kv_pos, int n_kv,
                      int32_t* slots, void* stream, const int32_t* d_n_kv,
                      const int32_t* d_n_kv_q = nullptr) {
    launch_pdl(sel_slots_k, grid_for((long long) n_sel * n_tok), kBlock,
               0, static_cast<cudaStream_t>(stream), sel, kv_pos, slots, n_sel, n_tok, n_kv, d_n_kv,
               d_n_kv_q);
}

inline void sel_scatter(const int32_t* sel, int n_sel, int n_tok, const int32_t* kv_pos, int n_kv,
                        uint32_t* mask, int words, void* stream, const int32_t* d_n_kv,
                        const int32_t* d_n_kv_q = nullptr) {
    launch_pdl(sel_scatter_k, dim3((unsigned) n_tok), kBlock, 0, static_cast<cudaStream_t>(stream),
               sel, kv_pos, mask, n_sel, n_tok, n_kv, words, d_n_kv, d_n_kv_q);
}

/// Emit the same unnormalized partials as the scalar split path for mla_merge_k.
template <class KvT, int kQS, int kPS, bool kPQ = false>
__global__ __launch_bounds__(kBlock) void mla_attend_mma_k(
    const float* q_absorbed_, const float* q_pe_,
    const KvT* kv_cmpr_, const kv_pe_t<KvT>* k_pe_, int n_head, int n_tok, int n_kv_max,
    int n_split, const int32_t* d_n_kv_, const int32_t* q_pos_,
    const int32_t* kv_pos_, int causal_window, float kq_scale, float* part_,
    float* pmd_, const uint32_t* sel_mask_, int mask_words, const int32_t* sel_slots_, int n_sel,
    int skip_empty, bool decode_equivalent) {
    // Everything this kernel reads is an activation (the queries) or the KV cache a kernel just appended to,
    // so none of it may be loaded before the wait - see pdl.hpp on STRATA_PDL_RESTRICT.
    const float* STRATA_PDL_RESTRICT q_absorbed = q_absorbed_;
    const float* STRATA_PDL_RESTRICT q_pe = q_pe_;
    const KvT* STRATA_PDL_RESTRICT kv_cmpr = kv_cmpr_;
    const kv_pe_t<KvT>* STRATA_PDL_RESTRICT k_pe = k_pe_;
    const int32_t* STRATA_PDL_RESTRICT d_n_kv = d_n_kv_;
    const int32_t* STRATA_PDL_RESTRICT q_pos = q_pos_;
    const int32_t* STRATA_PDL_RESTRICT kv_pos = kv_pos_;
    float* STRATA_PDL_RESTRICT part = part_;
    float* STRATA_PDL_RESTRICT pmd = pmd_;
    const uint32_t* STRATA_PDL_RESTRICT sel_mask = sel_mask_;   // written by sel_scatter_k on this stream
    const int32_t* STRATA_PDL_RESTRICT sel_slots = sel_slots_;  // or by sel_slots_k; never both
    pdl_trigger();
    pdl_wait();
    const int s = (int) blockIdx.x, i = (int) blockIdx.y, h0 = (int) blockIdx.z * kMmaH;
    const int all_kv = d_n_kv != nullptr ? (int) *d_n_kv : n_kv_max;
    const int n_kv = kPQ ? (int) d_n_kv[i] : decode_equivalent ? min(all_kv, q_pos[i] + 1) : all_kv;
    const int tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // Bound the sliding-window scan by the earliest slot visible to any query in the chunk.
    int kv_lo = 0;
    if (causal_window > 0) {
        const int lo_pos = q_pos[(kPQ || decode_equivalent) ? i : 0] - causal_window + 1;
        if (lo_pos > 0) {
            int a = 0, b = n_kv;
            while (a < b) {                       // first slot with kv_pos[slot] >= lo_pos
                const int m = (a + b) >> 1;
                if (kv_pos[m] < lo_pos) a = m + 1; else b = m;
            }
            // Align the first slot to the tile boundary to preserve key indexing.
            kv_lo = (a / kMmaC) * kMmaC;
        }
    }
    // Empty tail chunks must still emit neutral partials for the merge.
    const int span_lo = sel_slots != nullptr ? 0 : kv_lo;
    const int span_hi = sel_slots != nullptr ? n_sel : n_kv;
    const int live = span_hi - span_lo;
    const int per = (live + n_split - 1) / n_split;
    const int begin = span_lo + s * per;
    const int end = min(span_hi, begin + per);
    if (begin >= end) {
        // Still has to say "nothing here": the merge reads every chunk's (max, denominator).
        if (tid < kMmaH) {
            const long long pm = ((long long) ((h0 + tid) * n_tok + i) * n_split + s) * 2;
            pmd[pm] = -INFINITY;
            pmd[pm + 1] = 0.f;
        }
        return;
    }

    extern __shared__ __align__(16) unsigned char mma_sm[];
    __half* sQ = (__half*) mma_sm;                                      // kQS x kMmaH x kMmaLd
    __half* sL = sQ + (std::size_t) kQS * kMmaH * kMmaLd;               // kMmaC x kMmaLd
    __half* sP = sL + (std::size_t) kMmaC * kMmaLd;                     // kPS x kMmaH x (kMmaC + 8)
    float* sS = (float*) (sP + (std::size_t) kPS * kMmaH * (kMmaC + 8));  // kMmaH x kMmaC
    __shared__ float s_m[kMmaH], s_l[kMmaH], s_sc[kMmaH], s_qs[kMmaH];
    __shared__ int s_ok[kMmaC];
    __shared__ int s_any;      ///< does this tile hold a single visible key? see the loop
    __shared__ int s_slot[kMmaC];  ///< the gathered path's cache slot per tile row; -1 where dead

    // Stage the absorbed and rotary queries once per head tile.
    constexpr int kQv = kMmaV / 32;               // 16 latent floats a lane
#pragma unroll
    for (int rr = 0; rr < kMmaH / 8; ++rr) {
        const int r = warp * (kMmaH / 8) + rr;
        const float4* qa4 =
            (const float4*) (q_absorbed + ((long long) (h0 + r) * n_tok + i) * kMmaV);
        const float2* qp2 = (const float2*) (q_pe + ((long long) i * n_head + h0 + r) * kMmaRot);
        float v[kQv + 2];
#pragma unroll
        for (int u = 0; u < kQv / 4; ++u) {
            const float4 t = qa4[u * 32 + lane];
            v[4 * u] = t.x;
            v[4 * u + 1] = t.y;
            v[4 * u + 2] = t.z;
            v[4 * u + 3] = t.w;
        }
        {
            const float2 t = qp2[lane];
            v[kQv] = t.x;
            v[kQv + 1] = t.y;
        }
        float mx = 0.f;
#pragma unroll
        for (int u = 0; u < kQv + 2; ++u) mx = fmaxf(mx, fabsf(v[u]));
        mx = warp_max_f(mx);
        int ex = 0;
        if (mx > 0.f) {
            frexpf(mx, &ex);                      // mx is in [2^(ex-1), 2^ex)
            ex = max(-60, min(60, 14 - ex));      // so mx * 2^ex lands in [2^13, 2^14)
        }
        const float f = ldexpf(1.0f, ex);
#pragma unroll
        for (int u = 0; u < kQv + 2; ++u) v[u] *= f;
        // Column of v[u]: the latent in float4 groups, then the rope pair. The planes of one column go out
        // together so that v[u] is read from the register file once.
#pragma unroll
        for (int pl = 0; pl < kQS; ++pl) {
            __half* row = sQ + (std::size_t) pl * kMmaH * kMmaLd + r * kMmaLd;
#pragma unroll
            for (int u = 0; u < kQv / 4; ++u) {
                __align__(8) __half h4[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    h4[k] = __float2half_rn(v[4 * u + k]);
                    v[4 * u + k] -= __half2float(h4[k]);   // the next plane carries what this one dropped
                }
                *(uint2*) (row + u * 128 + lane * 4) = *(const uint2*) h4;
            }
            __align__(4) __half h2[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                h2[k] = __float2half_rn(v[kQv + k]);
                v[kQv + k] -= __half2float(h2[k]);
            }
            *(uint*) (row + kMmaV + lane * 2) = *(const uint*) h2;
            // The eight columns past 576 are read by the 36th k-step like any other. The latent tile zeros
            // its own, but a NaN left in shared memory here would still come out of the MMA as a NaN.
            if (lane < 4) *(uint*) (row + kMmaD + lane * 2) = 0u;
        }
        if (lane == 0) s_qs[r] = ldexpf(1.0f, -ex);
    }
    if (tid < kMmaH) {
        s_m[tid] = -INFINITY;                     // a PARTIAL: the sink belongs to the row and goes in at the merge
        s_l[tid] = 0.0f;
    }

    // The C layout of m16n8: a thread owns rows lane / 4 and lane / 4 + 8, columns (lane % 4) * 2 and + 1.
    // Warp w owns the weighted sum's columns [64 w, 64 w + 64) - eight n-tiles - of both m-tiles.
    const int g = lane >> 2, q2 = (lane & 3) * 2;
    float acc[2][8][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 8; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.0f;

    const int pos_i = q_pos[i];
    for (int t0 = begin; t0 < end; t0 += kMmaC) {
        __syncthreads();
        if (tid == 0) s_any = 0;
        __syncthreads();
        // ---- which of this tile's keys exist and are visible. It is per key only, not per (key, head):
        // every head in the block shares the query position.
        if (tid < kMmaC) {
            int ok = 0;
            // The row's cache slot: the loop index itself on the dense path, an entry of the selection on
            // the gathered one. A `-1` entry is a dead slot of a short row and reads nothing.
            int slot = -1;
            if (t0 + tid < end)
                slot = sel_slots != nullptr ? (int) sel_slots[(long long) i * n_sel + t0 + tid]
                                            : t0 + tid;
            s_slot[tid] = slot;
            if (slot >= 0) {
                const int pos_j = kv_pos[slot];
                // `causal_window` counts the positions a query may see INCLUDING itself, so 512 permits a
                // lookback of 511 - which is what is_masked_swa does with p1 - p0 >= n_swa.
                ok = !(pos_j > pos_i || (causal_window > 0 && pos_i - pos_j >= causal_window));
                // Intersect the sparse selection with the causal window.
                if (ok && sel_mask != nullptr)
                    ok = (sel_mask[(long long) i * mask_words + (slot >> 5)] >> (slot & 31)) & 1u;
            }
            s_ok[tid] = ok;
            if (ok) atomicOr(&s_any, 1);
        }
        __syncthreads();
        // Skip key tiles invisible to every query in the block.
        const bool work = !skip_empty || s_any != 0;
        if (work) {
        // Pad the latent and rotary key tile to the matrix stride.
#pragma unroll
        for (int u = 0; u < kMmaC * (kMmaV / 8) / kBlock; ++u) {
            const int x = u * kBlock + tid;
            const int r = x / (kMmaV / 8), c8 = x & (kMmaV / 8 - 1);
            *(uint4*) (sL + r * kMmaLd + c8 * 8) =
                s_ok[r] ? mma_lat8(kv_cmpr, s_slot[r], c8 * 8)
                        : make_uint4(0u, 0u, 0u, 0u);
        }
        {
            const int r = tid / (kMmaRot / 8), c8 = tid & (kMmaRot / 8 - 1);
            *(uint4*) (sL + r * kMmaLd + kMmaV + c8 * 8) =
                s_ok[r] ? mma_pack8(k_pe + (long long) s_slot[r] * kMmaRot + c8 * 8)
                        : make_uint4(0u, 0u, 0u, 0u);
            // The eight columns past 576 are read by the 36th k-step like any other, and the query's own
            // are zero, but a NaN here would still come back out of the MMA.
            if (tid < kMmaC) *(uint4*) (sL + tid * kMmaLd + kMmaD) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        // Each warp computes a tile of query-key scores.
        {
            const int mt = warp >> 2, nt = warp & 3;
            constexpr int kAcc = 8;
            float sacc[kAcc][4] = {};
#pragma unroll
            for (int pl = 0; pl < kQS; ++pl) {
                const __half* Qp = sQ + (std::size_t) pl * kMmaH * kMmaLd;
#pragma unroll
                for (int ks = 0; ks < kMmaD / 16; ++ks) {
                    uint32_t a0, a1, a2, a3, b0, b1;
                    mma_ldsm_x4(a0, a1, a2, a3, Qp + (mt * 16 + (lane & 15)) * kMmaLd + ks * 16 + (lane >> 4) * 8);
                    mma_ldsm_x2(b0, b1, sL + (nt * 8 + (lane & 7)) * kMmaLd + ks * 16 + ((lane >> 3) & 1) * 8);
                    mma_m16n8k16(sacc[ks % kAcc], a0, a1, a2, a3, b0, b1);
                }
            }
            // Pairwise, so the eight are three levels and not a chain of eight.
#pragma unroll
            for (int w = kAcc / 2; w > 0; w >>= 1)
#pragma unroll
                for (int j = 0; j < w; ++j)
#pragma unroll
                    for (int k = 0; k < 4; ++k) sacc[j][k] += sacc[j + w][k];
            const float* tot = sacc[0];
            const int r0 = mt * 16 + g, c0 = nt * 8 + q2;
            const float fa = kq_scale * s_qs[r0], fb = kq_scale * s_qs[r0 + 8];
            sS[r0 * kMmaC + c0] = tot[0] * fa;
            sS[r0 * kMmaC + c0 + 1] = tot[1] * fa;
            sS[(r0 + 8) * kMmaC + c0] = tot[2] * fb;
            sS[(r0 + 8) * kMmaC + c0 + 1] = tot[3] * fb;
        }
        }   // if (work): the staging and the score MMA
        __syncthreads();
        // ---- the online softmax, eight threads to a head row and four keys each.
        {
            const int r = tid >> 3, j0 = (tid & 7) * 4;
            float v[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = s_ok[j0 + j] ? sS[r * kMmaC + j0 + j] : -INFINITY;
                mx = fmaxf(mx, v[j]);
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = s_m[r], m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // m_new == -inf means nothing visible yet anywhere in this row: no contribution, and no
                // inf - inf.
                float e = (v[j] == -INFINITY || m_new == -INFINITY) ? 0.f : __expf(v[j] - m_new);
                sum += e;
                e *= kMmaPScale;
#pragma unroll
                for (int pl = 0; pl < kPS; ++pl) {
                    const __half hv = __float2half_rn(e);
                    sP[(std::size_t) pl * kMmaH * (kMmaC + 8) + r * (kMmaC + 8) + j0 + j] = hv;
                    e -= __half2float(hv);
                }
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if ((tid & 7) == 0) {
                // m_old == -inf: the accumulator is zero, so a scale of zero is both right and NaN-free.
                const float sc = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
                s_sc[r] = sc;
                s_l[r] = s_l[r] * sc + sum;
                s_m[r] = m_new;
            }
        }
        __syncthreads();
        // ---- O = O * scale(row) + P . L, over the first 512 columns only.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const float sa = s_sc[mt * 16 + g], sb = s_sc[mt * 16 + g + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                acc[mt][nt][0] *= sa;
                acc[mt][nt][1] *= sa;
                acc[mt][nt][2] *= sb;
                acc[mt][nt][3] *= sb;
            }
        }
        if (work)
#pragma unroll
        for (int ks = 0; ks < kMmaC / 16; ++ks) {
            uint32_t pa[kPS][2][4];
#pragma unroll
            for (int pl = 0; pl < kPS; ++pl)
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                    mma_ldsm_x4(pa[pl][mt][0], pa[pl][mt][1], pa[pl][mt][2], pa[pl][mt][3],
                                sP + (std::size_t) pl * kMmaH * (kMmaC + 8) +
                                    (mt * 16 + (lane & 15)) * (kMmaC + 8) + ks * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                uint32_t b0, b1;
                mma_ldsm_x2_t(b0, b1, sL + (ks * 16 + (lane & 15)) * kMmaLd + warp * 64 + nt * 8);
#pragma unroll
                for (int pl = 0; pl < kPS; ++pl)
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                        mma_m16n8k16(acc[mt][nt], pa[pl][mt][0], pa[pl][mt][1], pa[pl][mt][2], pa[pl][mt][3],
                                     b0, b1);
            }
        }
    }
    __syncthreads();

    // ---- UNNORMALISED, with this chunk's own max, and the P planes' 2^14 taken back out.
    if (tid < kMmaH) {
        const long long pm = ((long long) ((h0 + tid) * n_tok + i) * n_split + s) * 2;
        pmd[pm] = s_m[tid];
        pmd[pm + 1] = s_l[tid];
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int ra = mt * 16 + g, rb = ra + 8;
        // Every key masked: the accumulator is already zero and the merge skips the chunk anyway, but a
        // zero factor is what keeps a stray inf out of `part` rather than relying on that.
        const float ia = s_m[ra] == -INFINITY ? 0.f : 1.f / kMmaPScale;
        const float ib = s_m[rb] == -INFINITY ? 0.f : 1.f / kMmaPScale;
        float* oa = part + ((long long) ((h0 + ra) * n_tok + i) * n_split + s) * kMmaV;
        float* ob = part + ((long long) ((h0 + rb) * n_tok + i) * n_split + s) * kMmaV;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int c = warp * 64 + nt * 8 + q2;
            *(float2*) (oa + c) = make_float2(acc[mt][nt][0] * ia, acc[mt][nt][1] * ia);
            *(float2*) (ob + c) = make_float2(acc[mt][nt][2] * ib, acc[mt][nt][3] * ib);
        }
    }
}

#endif  // !HIP

/// The S partials of one (head, token) into one latent row, and the sink in exactly once.
///
/// M = max(sink, max_s m_s); den = exp(sink - M) + sum_s den_s * exp(m_s - M); out = sum_s o_s * exp(m_s - M) / den.
__global__ __launch_bounds__(kBlock) void mla_merge_k(const float* part_,
                                                      const float* pmd_,
                                                      const float* sinks_, int kv_lora,
                                                      int n_head, int n_tok, int n_split,
                                                      float* latent_) {
    const float* STRATA_PDL_RESTRICT part = part_;
    const float* STRATA_PDL_RESTRICT pmd = pmd_;
    const float* STRATA_PDL_RESTRICT sinks = sinks_;
    float* STRATA_PDL_RESTRICT latent = latent_;
    pdl_trigger();
    pdl_wait();
    const int hi = (int) blockIdx.x;
    const int h = hi / n_tok;
    if (h >= n_head) return;
    const float* md = pmd + (long long) hi * n_split * 2;

    // Merge chunk maxima and denominators before normalizing the weighted values.
    __shared__ float inv_s;
    __shared__ float w_s[kMaxSplit];   // exp(m_s - M) a chunk, 0 for one with nothing in it
    if (threadIdx.x < 32) {
        const int lane = (int) threadIdx.x;
        float m = -INFINITY;
        for (int s = lane; s < n_split; s += 32) m = fmaxf(m, md[(long long) s * 2]);
        m = warp_max_f(m);
        // ggml folds the sink into the max BEFORE exponentiating.
        if (sinks != nullptr) m = fmaxf(m, sinks[h]);
        float den = 0.f;
        for (int s = lane; s < n_split; s += 32) {
            const float ms = md[(long long) s * 2];
            const float w = (ms == -INFINITY || m == -INFINITY) ? 0.f : __expf(ms - m);
            w_s[s] = w;
            den += md[(long long) s * 2 + 1] * w;
        }
        den = warp_sum(den);
        if (lane == 0) {
            // The sink adds to the DENOMINATOR only - it has no output slot.
            if (sinks != nullptr && m != -INFINITY) den += __expf(sinks[h] - m);
            inv_s = den > 0.f ? 1.f / den : 0.f;
        }
    }
    __syncthreads();
    const float inv = inv_s;
    const long long base = (long long) hi * n_split * kv_lora;
    float* out = latent + (long long) hi * kv_lora;
    // Two floats a lane a chunk, so the n_split loads of one output pair are independent and their latency
    // overlaps. kv_lora is 512 on every model this path serves; the scalar form is here for any other.
    if ((kv_lora & 1) == 0) {
        for (int d2 = (int) threadIdx.x; d2 < kv_lora / 2; d2 += kBlock) {
            float2 acc = make_float2(0.f, 0.f);
            for (int s = 0; s < n_split; ++s) {
                const float w = w_s[s];
                if (w == 0.f) continue;   // uniform across the block: an empty chunk is read by nobody
                const float2 v = ((const float2*) (part + base + (long long) s * kv_lora))[d2];
                acc.x += v.x * w;
                acc.y += v.y * w;
            }
            ((float2*) out)[d2] = make_float2(acc.x * inv, acc.y * inv);
        }
    } else {
        for (int d = (int) threadIdx.x; d < kv_lora; d += kBlock) {
            float acc = 0.f;
            for (int s = 0; s < n_split; ++s) {
                const float w = w_s[s];
                if (w != 0.f) acc += part[base + (long long) s * kv_lora + d] * w;
            }
            out[d] = acc * inv;
        }
    }
}

/// latent[d, (h, i)] = sum_j prob[h, i, j] * kv_cmpr[d, j], laid out per head as [kv_lora, n_tok] blocks so
/// each head's slab can go straight into a matvec.
template <class KvT>
__global__ __launch_bounds__(kBlock) void mla_weighted_v_k(const float* __restrict__ prob,
                                                           const KvT* __restrict__ kv_cmpr, int kv_lora,
                                                           int n_head, int n_tok, int n_kv_max,
                                                           int kv_stride,
                                                           const int32_t* __restrict__ d_n_kv,
                                                           float* __restrict__ latent) {
    const int n_kv = d_n_kv != nullptr ? (int) *d_n_kv : n_kv_max;
    const long long total = (long long) kv_lora * n_tok * n_head;
    for (long long x = blockIdx.x * (long long) blockDim.x + threadIdx.x; x < total;
         x += (long long) gridDim.x * blockDim.x) {
        const int d = (int) (x % kv_lora);
        const long long rest = x / kv_lora;
        const int i = (int) (rest % n_tok);
        const int h = (int) (rest / n_tok);
        const float* p = prob + ((long long) h * n_tok + i) * kv_stride;
        float acc = 0.f;
        for (int j = 0; j < n_kv; ++j) acc += p[j] * kv_lat(kv_row(kv_cmpr, (long long) j, kv_lora), d, kv_lora);
        latent[x] = acc;
    }
}

/// out[i * v_head * n_head + h * v_head + e] = src[(h, i) block][e]: scatter each head's matvec result into
/// the token-major, head-consecutive layout attn_kqv has.

/// More heads per block share a latent read but reduce the number of independent blocks.
int mla_head_group(int n_head) {
    static const int want = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ATTN_HEADS");
        return v != nullptr ? std::atoi(v) : -1;
    }();
    const int g = want > 0 ? want : 1;   // MEASURED SLOWER at 2, 4 and 8; see the kernel comment
    if (g <= 1 || n_head % g != 0) return 1;
    return g == 2 || g == 4 || g == 8 ? g : 1;
}

/// Choose the split count from available SMs and the number of attention heads.
int mla_splits(int n_head, int n_tok, int n_kv_reserved, int head_group = 1) {
    if (mat_decode_equivalent() && n_tok <= 8) n_tok = 1;
    static const int override_to = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ATTN_SPLIT");
        return v != nullptr ? std::atoi(v) : -1;
    }();
    if (override_to >= 1) return override_to < kMaxSplit ? override_to : kMaxSplit;
    static const int sms = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 1;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 1;
        return n > 0 ? n : 1;
    }();
    const long long rows = (long long) ((n_head + head_group - 1) / head_group) * n_tok;
    if (rows <= 0) return 1;
    const long long want = 13ll * sms;
    long long sp = (want + rows - 1) / rows;
    // A short final tile needs only the keys remaining in the chunk.
    const long long by_chunk = (n_kv_reserved + kAttnMinChunk - 1) / kAttnMinChunk;
    if (sp > by_chunk) sp = by_chunk;
    if (sp > kMaxSplit) sp = kMaxSplit;
    return sp < 1 ? 1 : (int) sp;
}

int mla_sm_count() {
    static const int sms = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 1;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 1;
        return n > 0 ? n : 1;
    }();
    return sms;
}

/// Number of FP16 planes used to represent each FP32 operand.
int mla_mma_prec(int n_tok) {
    static const int want = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ATTN_MMA_PREC");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n >= 1 && n <= 3 ? n : 0;
    }();
    if (want != 0) return want;   // forced: both phases, which is what the A/B and the audit use

    // Select precision planes separately for prompt and decode occupancy.
    static const int prompt = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ATTN_MMA_PREC_PREFILL");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n >= 1 && n <= 3 ? n : 1;
    }();
    if (n_tok > kGemmT) return prompt;
#if !defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_USE_HIP)
    // Ada's per-block limit fits one plane but not the default two-plane decode.
    // Cache the limit per calling thread and refresh it if that thread changes devices.
    static thread_local int cached_device = -1, decode_planes = 2;
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess && device != cached_device) {
        int bytes = 0;
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess) {
            decode_planes = size_t(bytes) >= mma_smem<2, 2>() ? 2 : 1;
            cached_device = device;
        }
    }
    return decode_planes;
#else
    return 2;
#endif
}

/// The tensor-core path requires a layout compatible with the ldmatrix indexing.
bool mla_mma_usable(int kv_lora, int n_rot, int n_head, bool kv_fp16) {
#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
    (void) kv_lora;
    (void) n_rot;
    (void) n_head;
    (void) kv_fp16;
    return false;
#else
    static const bool want = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ATTN_MMA");
        return v == nullptr || std::atoi(v) != 0;
    }();
    static const bool sm80 = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        return major >= 8;
    }();
    return want && sm80 && kv_fp16 && kv_lora == kMmaV && n_rot == kMmaRot && n_head % kMmaH == 0;
#endif
}

/// Size splits for the number of heads processed per tensor-core block.
int mla_mma_splits(int n_head, int n_tok, int n_kv_reserved) {
    if (mat_decode_equivalent() && n_tok <= 8) n_tok = 1;
    static const int override_to = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ATTN_SPLIT");
        return v != nullptr ? std::atoi(v) : -1;
    }();
    if (override_to >= 1) return override_to < kMaxSplit ? override_to : kMaxSplit;
    const long long rows = (long long) ((n_head + kMmaH - 1) / kMmaH) * n_tok;
    if (rows <= 0) return 1;
    long long sp = (2ll * mla_sm_count() + rows - 1) / rows;
    const long long by_tile = (n_kv_reserved + kMmaC - 1) / kMmaC;
    if (sp > by_tile) sp = by_tile;
    if (sp > kMaxSplit) sp = kMaxSplit;
    return sp < 1 ? 1 : (int) sp;
}

#if !defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_USE_HIP)

/// Set the dynamic shared-memory limit for each device and kernel specialization.
template <class KvT, int kQS, int kPS, bool kPQ = false>
bool mma_smem_ready() {
    static int ok[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (ok[dev] == 0) {
        ok[dev] = cudaFuncSetAttribute((const void*) mla_attend_mma_k<KvT, kQS, kPS, kPQ>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int) mma_smem<kQS, kPS>()) == cudaSuccess
                      ? 1
                      : -1;
        if (ok[dev] == -1) cudaGetLastError();
    }
    return ok[dev] == 1;
}

/// Explicit precision settings must fit the device as requested.
template <class KvT>
void mma_launch(int prec, dim3 grid, const float* q_absorbed, const float* q_pe, const KvT* kv_cmpr,
                const kv_pe_t<KvT>* k_pe, int n_head, int n_tok, int n_kv, int n_split, const int32_t* d_n_kv,
                const int32_t* q_pos, const int32_t* kv_pos, int causal_window, float kq_scale, float* part,
                float* pmd, void* stream, const uint32_t* sel_mask, int mask_words,
                const int32_t* sel_slots, int n_sel, int skip_empty, bool per_query = false) {
    const auto go = [&](auto qt, auto pt, auto pq) {
        constexpr int kQS = decltype(qt)::value, kPS = decltype(pt)::value;
        constexpr bool kPQ = decltype(pq)::value;
        if (!mma_smem_ready<KvT, kQS, kPS, kPQ>())
            throw std::runtime_error("mla_attention: the device refused " +
                                     std::to_string(mma_smem<kQS, kPS>() >> 10) +
                                     " KiB of shared memory for the tensor-core kernel");
        launch_pdl(mla_attend_mma_k<KvT, kQS, kPS, kPQ>, grid, kBlock, mma_smem<kQS, kPS>(),
                   static_cast<cudaStream_t>(stream),
                   q_absorbed, q_pe, kv_cmpr, k_pe, n_head, n_tok, n_kv, n_split, d_n_kv, q_pos, kv_pos,
                   causal_window, kq_scale, part, pmd, sel_mask, mask_words, sel_slots, n_sel,
                   skip_empty, mat_decode_equivalent() && n_tok<=8);
    };
    const auto by_prec = [&](auto pq) {
        if (prec <= 1) go(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}, pq);
        else if (prec == 2) go(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}, pq);
        else go(std::integral_constant<int, 3>{}, std::integral_constant<int, 2>{}, pq);
    };
    if (per_query) by_prec(std::true_type{});
    else by_prec(std::false_type{});
}

#endif  // !HIP

}  // namespace

namespace {

/// STRATA_HY4_FLASH_FUSED_ATTN: 0 takes the split path, which is what the fused kernel is compared against.
inline bool attn_fused_on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_HY4_FLASH_FUSED_ATTN");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
}

/// STRATA_HY4_FLASH_FUSED_CHECK=1: run the two-pass kernels beside the fused one and compare.
inline bool attn_fused_check() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_HY4_FLASH_FUSED_CHECK");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

/// STRATA_HY4_FLASH_ATTN_GATHER selects gathered slots or a bitmask scan.
inline int attn_gather() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_HY4_FLASH_ATTN_GATHER");
        return e == nullptr || std::atoi(e) != 0 ? 1 : 0;
    }();
    return v;
}

/// Skip tiles invisible to all queries unless explicitly disabled.
inline int attn_skip_empty() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_HY4_FLASH_ATTN_SKIP");
        return e == nullptr || std::atoi(e) != 0 ? 1 : 0;
    }();
    return v;
}

/// Reserve the probability matrix only for attention paths that materialize it.
inline bool attn_prob_needed(int kv_lora) {
    return !(attn_fused_on() && kv_lora <= 4 * kBlock) || attn_fused_check();
}

}  // namespace

std::size_t mla_attention_scratch_bytes(int kv_lora, int v_head, int n_head, int n_tok, int n_kv,
                                       int max_sel, bool per_query) {
    if (n_tok <= 0 || n_kv <= 0) return 0;
    const std::size_t f = sizeof(float);
    std::size_t b = 0;
    if (attn_prob_needed(kv_lora))
        b += (std::size_t) n_head * n_tok * n_kv * f;    // prob
    b += (std::size_t) kv_lora * n_tok * n_head * f;     // latent
    b += (std::size_t) v_head * n_tok * n_head * f;      // every head's v_mla result, scattered in one go
    b += (std::size_t) n_head * n_tok * kernels::native_q8_1_bytes(kv_lora, 1);   // their Q8_1 staging
    // Reserve a per-query bitmask over cache slots for sparse selection.
    b += (std::size_t) n_tok * sel_mask_words(n_kv) * sizeof(uint32_t);
    // and the gathered path's slot list, one int a selected cell a query.
    if (max_sel > 0) b += (std::size_t) n_tok * max_sel * sizeof(int32_t);
    // Each split stores a latent sum, maximum score and softmax denominator.
    const bool mma_maybe = kv_lora == kMmaV && n_head % kMmaH == 0;
    // A per-query call cuts each query's range the way a one-token call does, so it has that many chunks
    // for each of its n_tok queries - more than an n_tok call would.
    const int sp_tok = per_query ? 1 : n_tok;
    const int sp = std::max(mla_splits(n_head, sp_tok, n_kv, mla_head_group(n_head)),
                            mma_maybe ? mla_mma_splits(n_head, sp_tok, n_kv) : 0);
    if (sp > 1 || mma_maybe) {
        b += (std::size_t) n_head * n_tok * sp * kv_lora * f;
        b += (std::size_t) n_head * n_tok * sp * 2 * f;
    }
    // the batched per-head wv_b product's own buffers, and ZERO at or below kMoeBatchMin tokens
    b += grouped_mmq_scratch_bytes(12, kv_lora, n_head, n_tok);
    return b + (std::size_t)n_head*n_tok*4 + 256;
}

bool mla_per_query_supported(int kv_lora, int n_rot, int n_head, bool kv_fp16, int wk_b_type, int wv_b_type) {
    return attn_fused_on() && mla_mma_usable(kv_lora, n_rot, n_head, kv_fp16) &&
           kernels::native_mmvq_grouped_supported(wk_b_type, 1) &&
           kernels::native_mmvq_grouped_supported(wv_b_type, 1);
}

namespace {
__global__ void per_query_meta_k(const int32_t* __restrict__ d_len, int32_t* __restrict__ n_kv,
                                 int32_t* __restrict__ head_of, int n_tok, int n_groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_tok) n_kv[i] = d_len[0] + i + 1;
    if (i < n_groups) head_of[i] = i / n_tok;
}
}  // namespace

void mla_per_query_meta(const int32_t* d_len, int32_t* n_kv, int32_t* head_of, int n_tok, int n_head,
                        void* stream) {
    if (d_len == nullptr || n_kv == nullptr || head_of == nullptr || n_tok < 1 || n_head < 1 || stream == nullptr)
        throw std::invalid_argument("mla_per_query_meta: device arrays, a stream and positive extents");
    const int n = std::max(n_tok, n_head * n_tok);
    per_query_meta_k<<<grid_for(n), kBlock, 0, static_cast<cudaStream_t>(stream)>>>(d_len, n_kv, head_of, n_tok,
                                                                                    n_head * n_tok);
}

/// The body, once per cache element type. `mla_attention` below picks which.
template <class KvT>
void mla_attention_t(const float* q_absorbed, const float* q_pe, const KvT* kv_cmpr, const kv_pe_t<KvT>* k_pe,
                   const float* sinks, const void* wv_b, int wv_b_type, int kv_lora, int n_rot, int v_head,
                   int n_head, int n_tok, int n_kv, const int32_t* q_pos, const int32_t* kv_pos,
                   int causal_window, float kq_scale, float* out, void* scratch, void* stream,
                   int kv_stride, const int32_t* d_n_kv, const int32_t* sel, int n_sel, int max_sel,
                   const MlaPerQuery* pq) {
    if (q_absorbed == nullptr || q_pe == nullptr || kv_cmpr == nullptr || k_pe == nullptr || wv_b == nullptr ||
        q_pos == nullptr || kv_pos == nullptr || out == nullptr || scratch == nullptr || stream == nullptr)
        throw std::invalid_argument("mla_attention: a stream and non-null buffers");
    if (kv_lora <= 0 || n_rot <= 0 || v_head <= 0 || n_head <= 0 || n_tok <= 0 || n_kv <= 0 ||
        causal_window < 0)
        throw std::invalid_argument("mla_attention: positive extents, and a non-negative window");

    // `kv_stride` is how far apart `prob`'s rows are: the RESERVED context when the caller is going to
    // capture this, so neither the buffer nor the offsets inside it move as the sequence grows. 0 means
    // "as many as there are", which is what a one-shot caller wants.
    if (kv_stride <= 0) kv_stride = n_kv;
    if (kv_stride < n_kv) throw std::invalid_argument("mla_attention: kv_stride below n_kv");
    char* p = static_cast<char*>(scratch);
    auto take = [&p](std::size_t bytes) {
        float* at = reinterpret_cast<float*>(p);
        p += (bytes + 15) & ~(std::size_t) 15;
        return at;
    };
    // ONE kernel where there were two, unless the geometry outgrows its register budget - see mla_attend_k.
    constexpr int kTile = kAttnTile;
    const bool fused = attn_fused_on() && kv_lora <= 4 * kBlock;
    // Use tensor cores only for supported head geometry and device capabilities.
    const bool mma = fused && mla_mma_usable(kv_lora, n_rot, n_head, !std::is_same<KvT, float>::value);
    // Sized from `kv_stride`, the reservation, because a captured graph bakes the grid; see `mla_splits`.
    const int n_group = (fused && !mma) ? mla_head_group(n_head) : 1;
    // Per-query: every query takes the chunk count and the precision of a ONE-token call, which is what
    // makes each of them that call's answer. Only the tensor-core kernel has the per-query form.
    if (pq != nullptr && (!mma || pq->n_kv == nullptr || pq->head_of == nullptr ||
                          !kernels::native_mmvq_grouped_supported(wv_b_type, 1)))
        throw std::invalid_argument("mla_attention: the per-query mode needs the tensor-core kernel, both "
                                    "device arrays and a grouped wv_b type (mla_per_query_supported)");
    const int q_tok = pq != nullptr ? 1 : n_tok;
    const int n_split = mma ? mla_mma_splits(n_head, q_tok, kv_stride)
                            : (fused ? mla_splits(n_head, n_tok, kv_stride, n_group) : 1);
    const int32_t* d_n_kv_q = pq != nullptr ? pq->n_kv : nullptr;
    // Log the selected attention implementation once.
    [[maybe_unused]] static const bool said = [&] {
        std::fprintf(stderr, "hy4_flash attention: %s\n",
                     mma ? (mla_mma_prec(n_tok) == 1   ? "tensor cores, 32 heads a block (1 FP16 plane)"
                            : mla_mma_prec(n_tok) == 2 ? "tensor cores, 32 heads a block (2 FP16 planes)"
                                                  : "tensor cores, 32 heads a block (3 query planes)")
                     : fused ? "the scalar split-KV kernel"
                             : "the two-pass scalar kernels");
        return true;
    }();

    // Only the tensor-core path consumes sparse selections here.
    if (sel != nullptr && n_sel > 0 && !mma)
        throw std::runtime_error(
            "mla_attention: this block is sparse (the compressed indexer selected its keys) but the "
            "tensor-core kernel was not chosen, and no other kernel here implements the selection - running "
            "them would attend the WHOLE cache and look like a result. Leave STRATA_HY4_FLASH_FUSED_ATTN at its "
            "default on an sm_80-or-later card, or set STRATA_HY4_FLASH_INDEXER=0 to accept dense sparse blocks.");

    const std::size_t f = sizeof(float);
    // Fused attention does not materialize probabilities.
    float* prob = attn_prob_needed(kv_lora)
                      ? take((std::size_t) n_head * n_tok * kv_stride * f)
                      : nullptr;
    float* latent = take((std::size_t) kv_lora * n_tok * n_head * f);
    float* head_out = take((std::size_t) v_head * n_tok * n_head * f);
    // Allocate from the reserved KV stride so graph replay keeps stable addresses.
    const int mask_words = sel_mask_words(kv_stride);
    uint32_t* sel_mask = (uint32_t*) take((std::size_t) n_tok * mask_words * sizeof(uint32_t));
    // The selected-slot list shares the reserved workspace across layers.
    int32_t* sel_list = max_sel > 0 ? (int32_t*) take((std::size_t) n_tok * max_sel * sizeof(int32_t))
                                    : nullptr;
    if (sel != nullptr && max_sel > 0 && n_sel > max_sel)
        throw std::invalid_argument("mla_attention: n_sel above the max_sel the scratch was sized for");
    const bool gathered = sel != nullptr && n_sel > 0 && mma && sel_list != nullptr && attn_gather() != 0;
    if (sel == nullptr || n_sel <= 0) sel_mask = nullptr;
    if (!gathered) sel_list = nullptr;
    else sel_mask = nullptr;   // never both: the slot list already says which keys are in
    // Allocate fixed-size buffers before giving the remaining space to q8.
    float* part = nullptr;
    float* pmd = nullptr;
    const bool split_path = mma || (fused && n_split > 1);
    if (split_path) {
        part = take((std::size_t) n_head * n_tok * n_split * kv_lora * f);
        pmd = take((std::size_t) n_head * n_tok * n_split * 2 * f);
    }
    void* q8 = static_cast<void*>(p);
    p += ((std::size_t) n_head * n_tok * kernels::native_q8_1_bytes(kv_lora, 1) + 15) & ~(std::size_t) 15;
    void* vsc = static_cast<void*>(p);   // the batched per-head wv_b product's, sized in the bytes function

    // Optionally compare attention against the external GGML diagnostic adapter.
    static const char* diag_fa = std::getenv("STRATA_HY4_FLASH_DIAG_FA_LIBRARY");
    if (diag_fa != nullptr && std::is_same<KvT, float>::value) {
        using Fn = void (*)(const float*, const float*, const float*, const float*, const float*, int, int,
                            int, int, int, const int*, const int*, int, float, float*, void*);
        static void* handle = dlopen(diag_fa, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (handle == nullptr) throw std::runtime_error(std::string("diag FA: ") + dlerror());
        static Fn fn = reinterpret_cast<Fn>(dlsym(handle, "hy4_flash_diag_fa"));
        if (fn == nullptr) throw std::runtime_error("diag FA: no hy4_flash_diag_fa");
        fn(q_absorbed, q_pe, (const float*) kv_cmpr, (const float*) k_pe, sinks, kv_lora, n_rot, n_head,
           n_tok, n_kv, q_pos, kv_pos, causal_window, kq_scale, latent, stream);
    } else if (split_path) {
        // The selection arrives as absolute positions. Expand it to one bit a cache slot ONCE here, so the
        // inner loop's test is the register-cheap thing it already does for causality instead of a search
        // through n_sel ids a key. `sel_mask == nullptr` means this block is dense and nothing is launched.
        if (sel_list != nullptr) {
            sel_slots(sel, n_sel, n_tok, kv_pos, n_kv, sel_list, stream, d_n_kv, d_n_kv_q);
            // Cross-check gathered slots against the bitmask for the same sparse selection.
            static const bool diag = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_SEL_DIAG");
                return v != nullptr && std::atoi(v) != 0;
            }();
            if (diag) {
                std::vector<uint32_t> mrow((std::size_t) mask_words);
                std::vector<int32_t> lrow((std::size_t) n_sel);
                uint32_t* tmp = nullptr;
                if (cudaMalloc(&tmp, (std::size_t) n_tok * mask_words * sizeof(uint32_t)) == cudaSuccess) {
                    sel_scatter(sel, n_sel, n_tok, kv_pos, n_kv, tmp, mask_words, stream, d_n_kv,
                                d_n_kv_q);
                    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
                    cudaMemcpy(mrow.data(), tmp + (long long) (n_tok - 1) * mask_words,
                               mrow.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(lrow.data(), sel_list + (long long) (n_tok - 1) * n_sel,
                               lrow.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
                    cudaFree(tmp);
                    long set = 0;
                    for (uint32_t w : mrow) set += __builtin_popcount(w);
                    long valid = 0, bad = 0, unsorted = 0;
                    int prev = -1;
                    for (int32_t slot : lrow) {
                        if (slot < 0) continue;
                        ++valid;
                        if (slot <= prev) ++unsorted;
                        prev = slot;
                        if (slot >= n_kv || !((mrow[(std::size_t) (slot >> 5)] >> (slot & 31)) & 1u)) ++bad;
                    }
                    std::fprintf(stderr,
                                 "sel diag: query %d of %d, n_kv %d, n_sel %d -> list %ld valid, mask %ld "
                                 "set, %ld not in mask, %ld out of order -> %s\n",
                                 n_tok - 1, n_tok, n_kv, n_sel, valid, set, bad, unsorted,
                                 (valid == set && bad == 0 && unsorted == 0) ? "SAME SET, ASCENDING"
                                                                            : "MISMATCH");
                }
            }
        }
        if (sel_mask != nullptr) {
            sel_scatter(sel, n_sel, n_tok, kv_pos, n_kv, sel_mask, mask_words, stream, d_n_kv, d_n_kv_q);
            // Count the last query's selected slots when selection diagnostics are enabled.
            static const bool diag = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_SEL_DIAG");
                return v != nullptr && std::atoi(v) != 0;
            }();
            if (diag) {
                std::vector<uint32_t> row((std::size_t) mask_words);
                cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
                cudaMemcpy(row.data(), sel_mask + (long long) (n_tok - 1) * mask_words,
                           row.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                long set = 0;
                for (uint32_t w : row) set += __builtin_popcount(w);
                std::fprintf(stderr, "sel diag: query %d of %d, n_kv %d, n_sel %d -> %ld slots set\n",
                             n_tok - 1, n_tok, n_kv, n_sel, set);
            }
        }
#if !defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_USE_HIP)
        if (mma) {
            const dim3 grid((unsigned) n_split, (unsigned) n_tok, (unsigned) (n_head / kMmaH));
            mma_launch<KvT>(mla_mma_prec(q_tok), grid, q_absorbed, q_pe, kv_cmpr, k_pe, n_head, n_tok, n_kv,
                            n_split, pq != nullptr ? d_n_kv_q : d_n_kv, q_pos, kv_pos, causal_window, kq_scale,
                            part, pmd, stream, sel_mask, mask_words, sel_list, n_sel, attn_skip_empty(),
                            pq != nullptr);
        } else
#endif
        // The scalar split kernels below do not read the mask; the check beside `mma` has already refused a
        // sparse block that would reach them, so there is nothing to guard here.
        if (n_group > 1) {
            // kG queries and their score rows in shared; the key goes to registers. See mla_attend_group_k.
            const std::size_t shm =
                (std::size_t) n_group * ((std::size_t) kv_lora + n_rot + kTile) * sizeof(float);
            const unsigned grid =
                (unsigned) (((n_head + n_group - 1) / n_group) * n_tok * n_split);
            const auto go = [&](auto tag) {
                constexpr int kG = decltype(tag)::value;
                mla_attend_group_k<KvT, kTile, kG><<<grid, kBlock, shm, static_cast<cudaStream_t>(stream)>>>(
                    q_absorbed, q_pe, kv_cmpr, k_pe, kv_lora, n_rot, n_head, n_tok, n_kv, n_split, d_n_kv,
                    q_pos, kv_pos, causal_window, kq_scale, part, pmd);
            };
            if (n_group == 2) go(std::integral_constant<int, 2>{});
            else if (n_group == 4) go(std::integral_constant<int, 4>{});
            else go(std::integral_constant<int, 8>{});
        } else {
        const std::size_t shm = ((std::size_t) kv_lora + n_rot + kTile) * sizeof(float);
        mla_attend_split_k<KvT, kTile><<<(unsigned) (n_head * n_tok * n_split), kBlock, shm,
                                    static_cast<cudaStream_t>(stream)>>>(
            q_absorbed, q_pe, kv_cmpr, k_pe, kv_lora, n_rot, n_head, n_tok, n_kv, n_split, d_n_kv, q_pos,
            kv_pos, causal_window, kq_scale, part, pmd);
        }
        launch_pdl(mla_merge_k, (unsigned) (n_head * n_tok), kBlock, 0, static_cast<cudaStream_t>(stream),
                   part, pmd, sinks, kv_lora, n_head, n_tok, n_split, latent);
        // STRATA_HY4_FLASH_SPLIT_CHECK=1: run the one-block kernel over the same inputs and compare the latent.
        // The only way to tell a summation-order difference from a wrong answer, and the tokens cannot do
        // it: this model's router amplifies 1e-6 into a different argmax within a few steps.
        static const bool scheck = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_SPLIT_CHECK");
            return v != nullptr && std::atoi(v) != 0;
        }();
        if (scheck && pq == nullptr) {   // the one-block kernel has no per-query count
            const std::size_t n = (std::size_t) kv_lora * n_tok * n_head;
            std::vector<float> a(n), b(n);
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            cudaMemcpy(a.data(), latent, n * sizeof(float), cudaMemcpyDeviceToHost);
            const std::size_t shm_one = ((std::size_t) kv_lora + n_rot + kTile) * sizeof(float);
            mla_attend_k<KvT, kTile><<<(unsigned) (n_head * n_tok), kBlock, shm_one,
                                  static_cast<cudaStream_t>(stream)>>>(
                q_absorbed, q_pe, kv_cmpr, k_pe, sinks, kv_lora, n_rot, n_head, n_tok, n_kv, d_n_kv, q_pos,
                kv_pos, causal_window, kq_scale, latent);
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            cudaMemcpy(b.data(), latent, n * sizeof(float), cudaMemcpyDeviceToHost);
            double worst = 0.0, mag = 1e-9;
            std::size_t at = 0;
            for (std::size_t x = 0; x < n; ++x) {
                const double d = std::fabs((double) a[x] - (double) b[x]);
                if (d > worst) { worst = d; at = x; }
                mag = std::max(mag, std::fabs((double) b[x]));
            }
            std::fprintf(stderr,
                         "split check: %s n_kv %d split %d worst %.3e rel %.3e at %zu (h %zu d %zu) "
                         "split %.6f one %.6f mag %.3e\n",
                         mma ? "mma" : "scalar", n_kv, n_split, worst, worst / mag, at,
                         at / (std::size_t) kv_lora, at % (std::size_t) kv_lora, (double) a[at],
                         (double) b[at], mag);
        }
    } else if (fused) {
        const std::size_t shm = ((std::size_t) kv_lora + n_rot + kTile) * sizeof(float);
        mla_attend_k<KvT, kTile><<<(unsigned) (n_head * n_tok), kBlock, shm,
                              static_cast<cudaStream_t>(stream)>>>(
            q_absorbed, q_pe, kv_cmpr, k_pe, sinks, kv_lora, n_rot, n_head, n_tok, n_kv, d_n_kv, q_pos,
            kv_pos, causal_window, kq_scale, latent);
        // Run the split reference path when checking the fused attention output.
        if (attn_fused_check()) {
            const std::size_t n = (std::size_t) kv_lora * n_tok * n_head;
            std::vector<float> a(n), b(n);
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            cudaMemcpy(a.data(), latent, n * sizeof(float), cudaMemcpyDeviceToHost);
            mla_scores_k<KvT><<<(unsigned) (n_head * n_tok), kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
                q_absorbed, q_pe, kv_cmpr, k_pe, sinks, kv_lora, n_rot, n_head, n_tok, n_kv, kv_stride,
                d_n_kv, q_pos, kv_pos, causal_window, kq_scale, prob);
            mla_weighted_v_k<KvT><<<grid_for((long long) kv_lora * n_tok * n_head), kBlock, 0,
                               static_cast<cudaStream_t>(stream)>>>(prob, kv_cmpr, kv_lora, n_head, n_tok,
                                                                    n_kv, kv_stride, d_n_kv, latent);
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            cudaMemcpy(b.data(), latent, n * sizeof(float), cudaMemcpyDeviceToHost);
            double worst = 0.0, mag = 1e-9;
            std::size_t at = 0;
            for (std::size_t x = 0; x < n; ++x) {
                const double d = std::fabs((double) a[x] - (double) b[x]);
                if (d > worst) { worst = d; at = x; }
                mag = std::max(mag, std::fabs((double) b[x]));
            }
            std::fprintf(stderr, "fused check: n_kv %d worst %.3e at %zu (fused %.6f split %.6f) mag %.3e\n",
                         n_kv, worst, at, (double) a[at], (double) b[at], mag);
        }
    } else {
        mla_scores_k<KvT><<<(unsigned) (n_head * n_tok), kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
            q_absorbed, q_pe, kv_cmpr, k_pe, sinks, kv_lora, n_rot, n_head, n_tok, n_kv, kv_stride, d_n_kv,
            q_pos, kv_pos, causal_window, kq_scale, prob);
        mla_weighted_v_k<KvT><<<grid_for((long long) kv_lora * n_tok * n_head), kBlock, 0,
                           static_cast<cudaStream_t>(stream)>>>(prob, kv_cmpr, kv_lora, n_head, n_tok,
                                                                n_kv, kv_stride, d_n_kv, latent);
    }

    // Apply the head-specific value projection to the compressed latent.
    quantize_cols(latent, q8, kv_lora, n_head * n_tok, stream);
    const std::size_t head_stride = wv_b_type == 30 ? (std::size_t) kv_lora * v_head * 2 :
        kernels::native_mmvq_weight_bytes(wv_b_type, kv_lora, v_head);
    const std::size_t q8_col = kernels::native_q8_1_bytes(kv_lora, 1);
    // The same batched product as mla_absorb's: 73,728 launches and 184 ms on a 256-token prompt before.
    const std::size_t vq = grouped_mmq_scratch_bytes(wv_b_type, kv_lora, n_head, n_tok);
    const bool wv_batched =
        pq == nullptr && vq != 0 && vq <= grouped_mmq_scratch_bytes(12, kv_lora, n_head, n_tok) &&
        grouped_product_mmq(wv_b, wv_b_type, head_stride, latent, kv_lora, v_head, n_head, n_tok, head_out,
                            vsc, vq, stream);
    if (n_tok > 8) {
        static int said = 0;
        if (said++ == 0)
            std::fprintf(stderr, "hy4_flash: wv_b type %d (%d -> %d, %d heads, %d tokens) %s\n", wv_b_type,
                         kv_lora, v_head, n_head, n_tok,
                         wv_batched ? "BATCHED" : "on the per-head matvecs");
    }
    if (wv_batched) {
        // nothing more
    } else if (pq != nullptr) {
        per_query_heads_product(wv_b_type, wv_b, head_stride, q8, q8_col, head_out, kv_lora, v_head,
                                 n_head, n_tok, pq->head_of, stream);
    } else if (verify_heads(wv_b_type,n_tok)) {
        auto* ids=static_cast<int32_t*>(vsc);
        verify_heads_product(wv_b_type,wv_b,head_stride,q8,q8_col,head_out,kv_lora,v_head,n_head,n_tok,ids,
                             stream,"v_b");
    } else if (wv_b_type == 30) {
        for (int h = 0; h < n_head; ++h)
            kernels::bf16_gemv_fp32_mmvf_cols(latent + (std::size_t) h * kv_lora * n_tok,
                reinterpret_cast<const uint16_t*>(static_cast<const char*>(wv_b) + h * head_stride),
                head_out + (std::size_t) h * v_head * n_tok, kv_lora, v_head, n_tok, stream);
    } else if (kernels::native_mmvq_grouped_supported(wv_b_type, n_tok)) {
        kernels::native_mmvq_grouped(wv_b_type, wv_b, head_stride, q8, n_tok * q8_col, head_out, kv_lora,
                                     v_head, n_head, stream);
    } else {
        for (int h = 0; h < n_head; ++h)
            kernels::native_mmvq(wv_b_type, static_cast<const char*>(wv_b) + (std::size_t) h * head_stride,
                                 static_cast<const char*>(q8) + (std::size_t) h * n_tok * q8_col,
                                 head_out + (std::size_t) h * v_head * n_tok, kv_lora, v_head, n_tok,
                                 stream);
    }
    launch_pdl(scatter_heads_k, grid_for((long long) v_head * n_tok * n_head), kBlock, 0,
               static_cast<cudaStream_t>(stream), head_out, out, v_head, n_head, n_tok);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("mla_attention: ") + cudaGetErrorString(e));
}

void mla_attention(const float* q_absorbed, const float* q_pe, const void* kv_cmpr, const void* k_pe,
                   int kv_format, const float* sinks, const void* wv_b, int wv_b_type, int kv_lora, int n_rot,
                   int v_head, int n_head, int n_tok, int n_kv, const int32_t* q_pos, const int32_t* kv_pos,
                   int causal_window, float kq_scale, float* out, void* scratch, void* stream,
                   int kv_stride, const int32_t* d_n_kv, const int32_t* sel, int n_sel, int max_sel, const MlaPerQuery* per_query) {
    if (kv_format == kKvQ8) {
        if (kv_lora % 32 != 0) throw std::invalid_argument("mla_attention: a Q8_0 latent needs kv_lora % 32 == 0");
        mla_attention_t<KvQ8>(q_absorbed, q_pe, static_cast<const KvQ8*>(kv_cmpr),
                              static_cast<const __half*>(k_pe), sinks, wv_b, wv_b_type, kv_lora, n_rot,
                              v_head, n_head, n_tok, n_kv, q_pos, kv_pos, causal_window, kq_scale, out,
                              scratch, stream, kv_stride, d_n_kv, sel, n_sel, max_sel, per_query);
    } else if (kv_format == kKvF16) {
        mla_attention_t<__half>(q_absorbed, q_pe, static_cast<const __half*>(kv_cmpr),
                                static_cast<const __half*>(k_pe), sinks, wv_b, wv_b_type, kv_lora, n_rot,
                                v_head, n_head, n_tok, n_kv, q_pos, kv_pos, causal_window, kq_scale, out,
                                scratch, stream, kv_stride, d_n_kv, sel, n_sel, max_sel, per_query);
    } else {
        mla_attention_t<float>(q_absorbed, q_pe, static_cast<const float*>(kv_cmpr),
                               static_cast<const float*>(k_pe), sinks, wv_b, wv_b_type, kv_lora, n_rot,
                               v_head, n_head, n_tok, n_kv, q_pos, kv_pos, causal_window, kq_scale, out,
                               scratch, stream, kv_stride, d_n_kv, sel, n_sel, max_sel, per_query);
    }
}

namespace {

/// x *= sigmoid(g), elementwise over the whole [width, n_tok] block.
__global__ __launch_bounds__(kBlock) void sigmoid_gate_k(float* x_, const float* g_,
                                                         long long n) {
    float* STRATA_PDL_RESTRICT x = x_;
    const float* STRATA_PDL_RESTRICT g = g_;
    pdl_trigger();
    pdl_wait();
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x)
        x[i] *= __fdividef(1.f, 1.f + __expf(-g[i]));   // c50's divide, not IEEE: this TU has no --use_fast_math
}


}  // namespace

std::size_t mla_output_scratch_bytes(int n_embd, int attn_width, int n_tok) {
    if (n_tok <= 0) return 0;
    const std::size_t f = sizeof(float);
    std::size_t b = (std::size_t) attn_width * n_tok * f;                 // the gate
    const int widest = n_embd > attn_width ? n_embd : attn_width;
    b += kernels::native_q8_1_bytes_wide(widest, n_tok);
    return b + 128;
}

void mla_gate(const void* w_gate, int w_gate_type, const MatIn& cur_in, int n_embd, int attn_width,
              int n_tok, void* scratch, void* stream) {
    if (w_gate == nullptr || scratch == nullptr || stream == nullptr)
        throw std::invalid_argument("mla_gate: a stream and non-null buffers");
    if (n_embd <= 0 || attn_width <= 0 || n_tok <= 0)
        throw std::invalid_argument("mla_gate: positive extents");
    // The first region of mla_output's scratch - the same layout mla_output computes for itself.
    mat_apply(cur_in, w_gate_type, w_gate, reinterpret_cast<float*>(scratch), n_embd, attn_width, n_tok,
              stream, kSiteAttnGate);
}

void mla_output(float* attn_kqv, const float* cur, const void* w_gate, int w_gate_type, const void* wo,
                int wo_type, const float* post_norm, int n_embd, int attn_width, int n_tok, float rms_eps,
                float* out, void* scratch, void* stream, const MatIn& cur_in, void* gate_join) {
    if (attn_kqv == nullptr || cur == nullptr || w_gate == nullptr || wo == nullptr || post_norm == nullptr ||
        out == nullptr || scratch == nullptr || stream == nullptr)
        throw std::invalid_argument("mla_output: a stream and non-null buffers");
    if (n_embd <= 0 || attn_width <= 0 || n_tok <= 0)
        throw std::invalid_argument("mla_output: positive extents");

    char* p = static_cast<char*>(scratch);
    float* gate = reinterpret_cast<float*>(p);
    p += ((std::size_t) attn_width * n_tok * sizeof(float) + 15) & ~(std::size_t) 15;
    void* q8 = static_cast<void*>(p);

    // The gate is scored from the block's normed hidden, not from the attention output - the same hidden
    // mla_project read, so the same Q8 when the caller supplies one. With a fork the caller launched
    // `mla_gate` on a second stream after that Q8 existed; wait for it here instead of launching again.
    if (gate_join != nullptr) {
        if (cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(gate_join), 0) != cudaSuccess)
            throw std::runtime_error("mla_output: the forked gate could not join the pass");
    } else {
        MatIn cur_im = cur_in;
        if (cur_im.q8 == nullptr)
            cur_im = mat_stage(cur_in.mc, cur, q8, n_embd, n_tok, stream, kSiteAttnGate);
        mat_apply(cur_im, w_gate_type, w_gate, gate, n_embd, attn_width, n_tok, stream, kSiteAttnGate);
    }
    launch_pdl(sigmoid_gate_k, grid_for((long long) attn_width * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), attn_kqv, gate, (long long) attn_width * n_tok);

    // Apply the gate before projecting with wo.
    const MatIn kqv = mat_stage(cur_in.mc, attn_kqv, q8, attn_width, n_tok, stream, kSiteWO);
    mat_apply(kqv, wo_type, wo, out, attn_width, n_embd, n_tok, stream, kSiteWO);
    kernels::rms_norm_weighted_wide(out, post_norm, n_tok, n_embd, rms_eps, stream);

    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("mla_output: ") + cudaGetErrorString(e));
}

namespace {

/// ff = silu(gate) * up, in place on `gate`. NO CLAMP - see the header for why this architecture's dense FFN
/// is the unclamped one while its routed experts are not.
__global__ __launch_bounds__(kBlock) void swiglu_plain_k(float* gate_,
                                                         const float* up_, long long n) {
    float* STRATA_PDL_RESTRICT gate = gate_;
    const float* STRATA_PDL_RESTRICT up = up_;
    pdl_trigger();
    pdl_wait();
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const float g = gate[i];
        gate[i] = __fdividef(g, 1.f + __expf(-g)) * up[i];   // the fast divide, as c50
    }
}

}  // namespace

std::size_t dense_ffn_scratch_bytes(int n_embd, int n_ff, int n_tok) {
    if (n_tok <= 0) return 0;
    const std::size_t f = sizeof(float);
    std::size_t b = 2 * (std::size_t) n_ff * n_tok * f;     // gate and up
    const int widest = n_embd > n_ff ? n_embd : n_ff;
    b += kernels::native_q8_1_bytes_wide(widest, n_tok);
    return b + 128;
}

// Optional FP32 reference projection for the dense path.
bool f32_dense_enabled();
bool f32_matvec_dequant(const void* w, int type, const float* x, float* y, int n_in, int n_out, int n_tok,
                        void* stream);

void dense_ffn(const float* x, const void* w_gate, int w_gate_type, const void* w_up, int w_up_type,
               const void* w_down, int w_down_type, const float* post_norm, int n_embd, int n_ff, int n_tok,
               float rms_eps, float* out, void* scratch, void* stream, const MatIn& x_in) {
    if (x == nullptr || w_gate == nullptr || w_up == nullptr || w_down == nullptr || post_norm == nullptr ||
        out == nullptr || scratch == nullptr || stream == nullptr)
        throw std::invalid_argument("dense_ffn: a stream and non-null buffers");
    if (n_embd <= 0 || n_ff <= 0 || n_tok <= 0) throw std::invalid_argument("dense_ffn: positive extents");

    char* p = static_cast<char*>(scratch);
    const std::size_t ff_bytes = ((std::size_t) n_ff * n_tok * sizeof(float) + 15) & ~(std::size_t) 15;
    float* gate = reinterpret_cast<float*>(p);
    p += ff_bytes;
    float* up = reinterpret_cast<float*>(p);
    p += ff_bytes;
    void* q8 = static_cast<void*>(p);

    bool f32 = f32_dense_enabled();
    if (f32) {
        // All three or none: a half-converted FFN would attribute the difference to the wrong projection.
        f32 = f32_matvec_dequant(w_gate, w_gate_type, x, gate, n_embd, n_ff, n_tok, stream) &&
              f32_matvec_dequant(w_up, w_up_type, x, up, n_embd, n_ff, n_tok, stream);
    }
    if (!f32) {
        // One staging of x serves both projections - and when the caller already made one (the same images
        // its router and shared expert read), none at all.
        MatIn xi = x_in;
        if (xi.q8 == nullptr) xi = mat_stage(x_in.mc, x, q8, n_embd, n_tok, stream, kSiteFFNGU);
        // w13: gate and up read the same activation and have the same shape, so one launch does both. The
        // GEMM has no such pair, so there it is two products over the one activation image - which costs
        // nothing extra, the weights being read once each either way.
        if (mat_gemm_path(xi.mc, n_tok, kSiteFFNGU) || w_gate_type != w_up_type ||
            !kernels::native_mmvq_pair(w_gate_type, w_gate, w_up, xi.q8, gate, up, n_embd, n_ff, n_tok,
                                       stream)) {
            mat_apply(xi, w_gate_type, w_gate, gate, n_embd, n_ff, n_tok, stream, kSiteFFNGU);
            mat_apply(xi, w_up_type, w_up, up, n_embd, n_ff, n_tok, stream, kSiteFFNGU);
        }
    }
    // Fuse SwiGLU and Q8_1 quantization for the down projection.
    if (f32) {
        launch_pdl(swiglu_plain_k, grid_for((long long) n_ff * n_tok), kBlock, 0,
                   static_cast<cudaStream_t>(stream), gate, up, (long long) n_ff * n_tok);
        if (!f32_matvec_dequant(w_down, w_down_type, gate, out, n_ff, n_embd, n_tok, stream)) {
            kernels::native_quantize_q8_1_wide(gate, q8, n_ff, n_tok, stream);
            mat_apply(mat_staged(x_in.mc,gate,q8,n_ff,n_tok,stream),w_down_type,w_down,out,n_ff,n_embd,n_tok,stream,kSiteFFNDown);
        }
    } else if (mat_gemm_path(x_in.mc, n_tok, kSiteFFNDown)) {
        // GEMM consumes FP32 SwiGLU output. The fused quantizer supplies only Q8_1.
        launch_pdl(swiglu_plain_k, grid_for((long long) n_ff * n_tok), kBlock, 0,
                   static_cast<cudaStream_t>(stream), gate, up, (long long) n_ff * n_tok);
        const MatIn h = mat_stage(x_in.mc, gate, q8, n_ff, n_tok, stream, kSiteFFNDown);
        mat_apply(h, w_down_type, w_down, out, n_ff, n_embd, n_tok, stream, kSiteFFNDown);
    } else {
        kernels::native_swiglu_quantize_q8_1(gate, up, q8, n_ff, n_tok, stream);
        mat_apply(mat_staged(x_in.mc,gate,q8,n_ff,n_tok,stream),w_down_type,w_down,out,n_ff,n_embd,n_tok,stream,kSiteFFNDown);
    }
    kernels::rms_norm_weighted_wide(out, post_norm, n_tok, n_embd, rms_eps, stream);

    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dense_ffn: ") + cudaGetErrorString(e));
}

namespace {

/// Clamp the routed gate and up inputs before SwiGLU and overwrite the gate buffer.
__global__ void expert_publish_k(ExpertAsyncMailbox* m, const int32_t* ids) {
    if (threadIdx.x == 0) {
        for (int j=0;j<16;++j) m->ids[j]=ids[j];
        __threadfence_system();
        m->request = m->request + 1;
    }
}
// Exact partial-residency fast plan. This remains an ordinary launch: it must observe the
// completed router and must finish writing both device indices before dependent expert kernels run.
// No host may mutate the map except while this stream is blocked at a real-miss A/B handshake.
__global__ void expert_publish_plan_k(ExpertAsyncMailbox* m, const int32_t* ids,
                                      const int32_t* slot_of, int n_expert, int permanent,
                                      int32_t* hit, int32_t* miss) {
    if (threadIdx.x != 0) return;
    const uint32_t epoch = m->request + 1;
    int selected[16], slots[16];
    bool all_hit = m->error == 0 && slot_of != nullptr && n_expert > 0 && permanent > 0;
    const volatile int32_t* map = slot_of;
#pragma unroll
    for (int j=0; j<16; ++j) {
        selected[j] = ids[j];
        const bool valid = selected[j] >= 0 && selected[j] < n_expert;
        slots[j] = valid && map != nullptr ? map[selected[j]] : -1;
        all_hit = all_hit && valid && slots[j] >= 0 && slots[j] < permanent;
        m->ids[j] = selected[j];
    }
    if (all_hit) {
#pragma unroll
        for (int j=0; j<16; ++j) {
            // Host snapshots are CPU validation inputs, not a second mutable plan.
            m->hit[j] = slots[j]; m->miss[j] = -1;
            hit[j] = slots[j]; miss[j] = -1;
        }
    }
    __threadfence_system();
    m->fast_epoch = all_hit ? epoch : 0;
    __threadfence_system();
    m->request = epoch;
}
// Warp-coalesced candidate: request is the only host publication. All payload writers execute
// a system fence, then the entire warp reconverges before lane0 publishes the new request.
__global__ void expert_publish_plan_warp_k(ExpertAsyncMailbox* m, const int32_t* ids,
                                           const int32_t* slot_of, int n_expert, int permanent,
                                           int32_t* hit, int32_t* miss) {
    const int lane=threadIdx.x;
    uint32_t epoch=lane==0?m->request+1:0;
    epoch=__shfl_sync(0xffffffffu,epoch,0);
    unsigned healthy=lane==0?(m->error==0):0;
    healthy=__shfl_sync(0xffffffffu,healthy,0);
    int id=-1,slot=-1;
    bool valid=true;
    if(lane<16) {
        id=ids[lane];
        const bool id_ok=id>=0 && id<n_expert && slot_of!=nullptr;
        const volatile int32_t* map=slot_of;
        slot=id_ok?map[id]:-1;
        valid=id_ok && slot>=0 && slot<permanent;
        m->ids[lane]=id;
    }
    const bool fast=healthy && __all_sync(0xffffffffu,valid);
    if(fast && lane<16) {
        m->hit[lane]=slot;
        hit[lane]=slot;miss[lane]=-1;
        // No CPU fast-path consumer reads mapped m->miss; slow CPU planning overwrites all entries.
    }
    if(lane==0)m->fast_epoch=fast?epoch:0;
    __threadfence_system();
    __syncwarp(0xffffffffu);
    if(lane==0)m->request=epoch;
}
__global__ void expert_wait_plan_k(ExpertAsyncMailbox* m, int32_t* hit, int32_t* miss) {
    if (threadIdx.x == 0) {
        const uint32_t e=m->request;
        while (m->plan_ready < e && !m->error) __nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    const int j=threadIdx.x;
    const volatile int32_t* hp=m->hit;
    const volatile int32_t* mp=m->miss;
    if(j<16) { hit[j]=m->error?-1:hp[j]; miss[j]=m->error?-1:mp[j]; }
}
__global__ void expert_wait_miss_k(ExpertAsyncMailbox* m, int32_t* miss) {
    if(threadIdx.x==0) {
        const uint32_t e=m->request;
        while(m->miss_ready < e && !m->error) __nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    if(m->error && threadIdx.x<16) miss[threadIdx.x]=-1;
}
__global__ void expert_wait_plan_device_k(ExpertAsyncMailbox* m, int32_t* hit, int32_t* miss) {
    unsigned fast = threadIdx.x == 0 ? (m->fast_epoch == m->request && !m->error) : 0;
    fast = __shfl_sync(0xffffffffu, fast, 0);
    if (fast) return; // warp-uniform; indices were already published by the GPU.
    if (threadIdx.x == 0) {
        const uint32_t e=m->request;
        while (m->plan_ready < e && !m->error) __nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    const int j=threadIdx.x;
    const volatile int32_t* hp=m->hit;
    const volatile int32_t* mp=m->miss;
    if(j<16) { hit[j]=m->error?-1:hp[j]; miss[j]=m->error?-1:mp[j]; }
}
__global__ void expert_wait_miss_device_k(ExpertAsyncMailbox* m, int32_t* miss) {
    unsigned fast = threadIdx.x == 0 ? (m->fast_epoch == m->request && !m->error) : 0;
    fast = __shfl_sync(0xffffffffu, fast, 0);
    if (fast) return; // no DMA or CPU ready flag exists on this exact all-hit path.
    if(threadIdx.x==0) {
        const uint32_t e=m->request;
        while(m->miss_ready < e && !m->error) __nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    if(m->error && threadIdx.x<16) miss[threadIdx.x]=-1;
}
// Ordinary launch: no PDL trigger before B. All expert indices/weight reads stay behind
// DMA+map completion on a real miss; exact all-hit publication can return immediately.
__global__ void expert_wait_all_device_k(ExpertAsyncMailbox* m, int32_t* complete_index) {
    unsigned fast=threadIdx.x==0 ? (m->fast_epoch==m->request && !m->error) : 0;
    fast=__shfl_sync(0xffffffffu,fast,0);
    if(fast)return;
    if(threadIdx.x==0) {
        const uint32_t epoch=m->request;
        while(m->miss_ready<epoch && !m->error)__nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    const int j=threadIdx.x;
    const volatile int32_t* hp=m->hit;
    const volatile int32_t* mp=m->miss;
    if(j<16) {
        if(m->error)complete_index[j]=-1;
        else { const int h=hp[j];complete_index[j]=h>=0?h:mp[j]; }
    }
}
// Ordinary launches: publication/consumption must not cross the system-memory
// handshakes through an early PDL trigger. Exact all-hit epochs skip CPU traffic.
__global__ void expert_cpu_input_k(ExpertAsyncMailbox* m, const float* input,
                                   float* host_input, int width) {
    __shared__ int skip;
    if (threadIdx.x == 0) skip = m->error || m->fast_epoch == m->request;
    __syncthreads();
    if (skip) return;
    for (int i = threadIdx.x; i < width; i += blockDim.x) host_input[i] = input[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) m->input_ready = m->request;
}

__global__ void expert_cpu_merge_k(ExpertAsyncMailbox* m, const float* cpu_parts,
                                   float* parts, int width) {
    __shared__ unsigned mask;
    if (threadIdx.x == 0) {
        const uint32_t epoch = m->request;
        // A fast epoch bypasses CPU planning. Its previous mask may be stale.
        mask = m->fast_epoch == epoch ? 0u : m->cpu_mask;
        if (mask) {
            while (m->cpu_ready < epoch && !m->error) __nanosleep(128);
            __threadfence_system();
        }
        if (m->error) mask = 0;
    }
    __syncthreads();
    if (!mask) return;
    for (int i = threadIdx.x; i < 16 * width; i += blockDim.x)
        if (mask & (1u << (i / width))) parts[i] = cpu_parts[i];
}

// Ordinary launch: the complete predecessor and C must both precede combine.
// Direct-parts keeps the same epoch/fast/error handshake, with no staging copy.
__global__ void expert_cpu_wait_results_k(ExpertAsyncMailbox* m) {
    __shared__ unsigned mask;
    if (threadIdx.x == 0) {
        const uint32_t epoch = m->request;
        mask = m->fast_epoch == epoch ? 0u : m->cpu_mask;
        if (mask) {
            while (m->cpu_ready < epoch && !m->error) __nanosleep(128);
            __threadfence_system();
        }
        if (m->error) mask = 0;
    }
    __syncthreads();
}

__global__ void expert_done_k(ExpertAsyncMailbox* m) {
    if(threadIdx.x==0) { __threadfence_system(); m->done=m->request; }
}

/// Record device timestamps for estimating the CPU/GPU split.
__global__ void expert_split_stamp_k(unsigned long long* stats, int slot) {
    if (threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        stats[slot] = t;
        __threadfence_system();
    }
}
// Device-epoch variants retain all mapped CPU-authored fields and system fences.
__global__ void expert_publish_plan_warp_epoch_k(ExpertAsyncMailbox* m, const int32_t* ids,
                                           const int32_t* slot_of, int n_expert, int permanent,
                                           int32_t* hit, int32_t* miss, volatile uint32_t* shadow) {
    const int lane=threadIdx.x;
    uint32_t epoch=lane==0?shadow[0]+1:0;
    epoch=__shfl_sync(0xffffffffu,epoch,0);
    unsigned healthy=lane==0?(m->error==0):0;
    healthy=__shfl_sync(0xffffffffu,healthy,0);
    int id=-1,slot=-1;
    bool valid=true;
    if(lane<16) {
        id=ids[lane];
        const bool id_ok=id>=0 && id<n_expert && slot_of!=nullptr;
        const volatile int32_t* map=slot_of;
        slot=id_ok?map[id]:-1;
        valid=id_ok && slot>=0 && slot<permanent;
        m->ids[lane]=id;
    }
    const bool fast=healthy && __all_sync(0xffffffffu,valid);
    if(fast && lane<16) {
        m->hit[lane]=slot;
        hit[lane]=slot;miss[lane]=-1;
        // No CPU fast-path consumer reads mapped m->miss; slow CPU planning overwrites all entries.
    }
    if(lane==0) {
        m->fast_epoch=fast?epoch:0;
        shadow[0]=epoch;shadow[1]=fast?epoch:0;
    }
    __threadfence_system();
    __syncwarp(0xffffffffu);
    if(lane==0)m->request=epoch;
}
__global__ void expert_wait_plan_device_epoch_k(ExpertAsyncMailbox* m, int32_t* hit, int32_t* miss, volatile uint32_t* shadow) {
    unsigned fast = threadIdx.x == 0 ? (shadow[1] == shadow[0] && !m->error) : 0;
    fast = __shfl_sync(0xffffffffu, fast, 0);
    if (fast) return; // warp-uniform; indices were already published by the GPU.
    if (threadIdx.x == 0) {
        const uint32_t e=shadow[0];
        while (m->plan_ready < e && !m->error) __nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    const int j=threadIdx.x;
    const volatile int32_t* hp=m->hit;
    const volatile int32_t* mp=m->miss;
    if(j<16) { hit[j]=m->error?-1:hp[j]; miss[j]=m->error?-1:mp[j]; }
}
__global__ void expert_wait_miss_device_epoch_k(ExpertAsyncMailbox* m, int32_t* miss, volatile uint32_t* shadow) {
    unsigned fast = threadIdx.x == 0 ? (shadow[1] == shadow[0] && !m->error) : 0;
    fast = __shfl_sync(0xffffffffu, fast, 0);
    if (fast) return; // no DMA or CPU ready flag exists on this exact all-hit path.
    if(threadIdx.x==0) {
        const uint32_t e=shadow[0];
        while(m->miss_ready < e && !m->error) __nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    if(m->error && threadIdx.x<16) miss[threadIdx.x]=-1;
}
__global__ void expert_wait_all_device_epoch_k(ExpertAsyncMailbox* m, int32_t* complete_index, volatile uint32_t* shadow) {
    unsigned fast=threadIdx.x==0 ? (shadow[1]==shadow[0] && !m->error) : 0;
    fast=__shfl_sync(0xffffffffu,fast,0);
    if(fast)return;
    if(threadIdx.x==0) {
        const uint32_t epoch=shadow[0];
        while(m->miss_ready<epoch && !m->error)__nanosleep(128);
        __threadfence_system();
    }
    __syncthreads();
    const int j=threadIdx.x;
    const volatile int32_t* hp=m->hit;
    const volatile int32_t* mp=m->miss;
    if(j<16) {
        if(m->error)complete_index[j]=-1;
        else { const int h=hp[j];complete_index[j]=h>=0?h:mp[j]; }
    }
}
__global__ void expert_cpu_input_epoch_k(ExpertAsyncMailbox* m, const float* input,
                                   float* host_input, int width, volatile uint32_t* shadow) {
    __shared__ int skip;
    if (threadIdx.x == 0) skip = m->error || shadow[1] == shadow[0];
    __syncthreads();
    if (skip) return;
    for (int i = threadIdx.x; i < width; i += blockDim.x) host_input[i] = input[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) m->input_ready = shadow[0];
}
__global__ void expert_cpu_merge_epoch_k(ExpertAsyncMailbox* m, const float* cpu_parts,
                                   float* parts, int width, volatile uint32_t* shadow) {
    __shared__ unsigned mask;
    if (threadIdx.x == 0) {
        const uint32_t epoch = shadow[0];
        // A fast epoch bypasses CPU planning. Its previous mask may be stale.
        mask = shadow[1] == epoch ? 0u : m->cpu_mask;
        if (mask) {
            while (m->cpu_ready < epoch && !m->error) __nanosleep(128);
            __threadfence_system();
        }
        if (m->error) mask = 0;
    }
    __syncthreads();
    if (!mask) return;
    for (int i = threadIdx.x; i < 16 * width; i += blockDim.x)
        if (mask & (1u << (i / width))) parts[i] = cpu_parts[i];
}
__global__ void expert_cpu_wait_results_epoch_k(ExpertAsyncMailbox* m, volatile uint32_t* shadow) {
    __shared__ unsigned mask;
    if (threadIdx.x == 0) {
        const uint32_t epoch = shadow[0];
        mask = shadow[1] == epoch ? 0u : m->cpu_mask;
        if (mask) {
            while (m->cpu_ready < epoch && !m->error) __nanosleep(128);
            __threadfence_system();
        }
        if (m->error) mask = 0;
    }
    __syncthreads();
}
__global__ void expert_done_epoch_k(ExpertAsyncMailbox* m, volatile uint32_t* shadow) {
    if(threadIdx.x==0) { __threadfence_system(); m->done=shadow[0]; }
}
__global__ void expert_trace_masked_k(const float* g,const float* u,float* tg,float* tu,
                                     const int32_t* mask,int width) {
    if(mask[0]<0) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<width) { tg[i]=g[i]; tu[i]=u[i]; }
}

__global__ __launch_bounds__(kBlock) void swiglu_clamped_k(float* gate_,
                                                           const float* up_, float limit,
                                                           long long n) {
    float* STRATA_PDL_RESTRICT gate = gate_;
    const float* STRATA_PDL_RESTRICT up = up_;
    pdl_trigger();
    pdl_wait();
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        float g = gate[i], u = up[i];
        g = g < limit ? g : limit;
        u = u < -limit ? -limit : (u > limit ? limit : u);
        gate[i] = __fdividef(g, 1.f + __expf(-g)) * u;   // the fast divide, as c50
    }
}

/// out += add, elementwise.
__global__ __launch_bounds__(kBlock) void add_inplace_k(float* out_,
                                                        const float* add_, long long n) {
    float* STRATA_PDL_RESTRICT out = out_;
    const float* STRATA_PDL_RESTRICT add = add_;
    pdl_trigger();
    pdl_wait();
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x)
        out[i] += add[i];
}

/// Multiply FP32 weights by the activation rows.
__global__ __launch_bounds__(kBlock) void f32_matvec_k(const float* __restrict__ w,
                                                       const float* x_, float* y_,
                                                       int n_in, int n_out, int n_tok) {
    // `w` is a weight and may be loaded before the wait; `x` is the activation the kernel in front wrote.
    const float* STRATA_PDL_RESTRICT x = x_;
    float* STRATA_PDL_RESTRICT y = y_;
    pdl_trigger();
    pdl_wait();
    const int ot = (int) blockIdx.x;
    const int o = ot % n_out, t = ot / n_out;
    if (t >= n_tok) return;
    const float* wr = w + (long long) o * n_in;
    const float* xr = x + (long long) t * n_in;
    float acc = 0.f;
    // Preserve the reference's two-value grouping in the FP32 accumulation.
    if ((n_in & 1) == 0 && ((std::uintptr_t) wr & 7) == 0 && ((std::uintptr_t) xr & 7) == 0) {
        const float2* w2 = (const float2*) wr;
        const float2* x2 = (const float2*) xr;
        for (int i = (int) threadIdx.x; i < n_in / 2; i += kBlock) {
            const float2 a = w2[i], b = x2[i];
            acc += a.x * b.x;
            acc += a.y * b.y;
        }
    } else {
        for (int i = (int) threadIdx.x; i < n_in; i += kBlock) acc += wr[i] * xr[i];
    }
    acc = warp_sum(acc);
    __shared__ float part[kBlock / 32];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kBlock / 32 ? part[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if (threadIdx.x == 0) y[(long long) t * n_out + o] = v;
    }
}

/// Gather one token's column out of a [width, n_tok] block.
__global__ __launch_bounds__(kBlock) void take_token_k(const float* __restrict__ src, float* __restrict__ dst,
                                                       int width, int t) {
    for (int i = (int) (blockIdx.x * blockDim.x + threadIdx.x); i < width;
         i += (int) (gridDim.x * blockDim.x))
        dst[i] = src[(long long) t * width + i];
}

/// Translate expert IDs to resident slots on device.
__global__ __launch_bounds__(kBlock) void map_slots_k(const int32_t* ids_,
                                                      const int32_t* __restrict__ slot_of,
                                                      int32_t* out_, int n, int n_expert) {
    const int32_t* STRATA_PDL_RESTRICT ids = ids_;   // written by the routing kernel in front of this one
    int32_t* STRATA_PDL_RESTRICT out = out_;
    pdl_trigger();
    pdl_wait();
    for (int j = (int) (blockIdx.x * blockDim.x + threadIdx.x); j < n;
         j += (int) (gridDim.x * blockDim.x)) {
        const int e = ids[j];
        // A router that returned something out of range would otherwise index the table out of bounds; slot
        // 0 is wrong but bounded, and the parity tests compare ids directly so this cannot hide a bug.
        out[j] = e >= 0 && e < n_expert ? slot_of[e] : 0;
    }
}

}  // namespace

/// The diagnostic FP32 path dequantizes weights before the matvec.
bool f32_dense_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_F32_DENSE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

/// y[o, t] = sum_i w[o * n_in + i] * x[i, t], the weight dequantised from `type` first.
/// Returns false when the type cannot be dequantised here, so the caller keeps the MMVQ path.
bool f32_matvec_dequant(const void* w, int type, const float* x, float* y, int n_in, int n_out, int n_tok,
                        void* stream) {
    float* dw = nullptr;
    const std::size_t bytes = (std::size_t) n_in * n_out * sizeof(float);
    if (cudaMalloc(&dw, bytes) != cudaSuccess) return false;
    kernels::dequant_f32(type, w, 0, n_out, n_in, dw, stream);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(dw);
        return false;
    }
    launch_pdl(f32_matvec_k, (unsigned) (n_out * n_tok), kBlock, 0, static_cast<cudaStream_t>(stream),
               dw, x, y, n_in, n_out, n_tok);
    const auto e = cudaGetLastError();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    cudaFree(dw);
    return e == cudaSuccess;
}

namespace {


}  // namespace

namespace {

/// Optionally overlap the shared expert with routed experts on a separate stream.
struct ShexpStream {
    cudaStream_t stream = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    bool ok = false;
};

bool shexp_stream_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_SHEXP_STREAM");
        return v == nullptr || *v != '0';
    }();
    return on;
}

/// One per thread: a forward pass is serial within an engine, and the fork and the join are balanced inside
/// one `latent_moe` call, so a single stream serves every block. Two engines on two threads get two.
ShexpStream* shexp_stream() {
    static thread_local ShexpStream s;
    if (!s.ok) {
        if (cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.fork, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.join, cudaEventDisableTiming) != cudaSuccess)
            return nullptr;
        s.ok = true;
    }
    return &s;
}

}  // namespace

namespace {
#include "window_kernels.inc"

/// Sort the token-major tier slots for grouped window execution.
__global__ void moe_window_plan_k(const int32_t* slots, int n, int k, int32_t* w_index, int32_t* tok,
                                  int32_t* inv) {
    const int g = (int) threadIdx.x;
    if (g >= n) return;
    const int32_t s = slots[g];
    int r = 0;
    for (int o = 0; o < n; ++o) {
        const int32_t so = slots[o];
        r += so < s || (so == s && o < g);
    }
    w_index[r] = s;
    tok[r] = g / k;
    inv[g] = r;
}

}  // namespace

void latent_moe_route_sync_cost(double& secs, double& backlog, long& count) {
    secs = g_route_sync;
    backlog = g_route_backlog;
    count = g_route_syncs;
}

void mat_f32(const float* w, const float* x, float* y, int n_in, int n_out, int T, void* stream) {
    launch_pdl(f32_matvec_k, dim3((unsigned)(n_out*T)), kBlock, 0,
               static_cast<cudaStream_t>(stream), w, x, y, n_in, n_out, T);
}

/// Stripe groups for the resident verify window (default 16, zero disables striping).
/// A complete tier supplies the slots directly, so the stripe kernels can run without
/// the mailbox and staging used by the partial-resident window.
static int resident_stripe_groups() {
    static const int n = [] {
        const char* e = std::getenv("STRATA_HY4_FLASH_RESIDENT_STRIPES");
        const int v = e != nullptr ? std::atoi(e) : 16;
        return v < 0 ? 0 : (v > 32 ? 32 : v);
    }();
    return n;
}

std::size_t latent_moe_scratch_bytes(int n_embd, int expert_dim, int expert_ff, int n_expert, int k,
                                     int n_tok, int gate_type, int up_type, int down_type, int mmq_slots, int mmq_rows,
                                     bool exact_window) {
    if (n_tok <= 0) return 0;
    const std::size_t f = sizeof(float);
    std::size_t b = 0;
    b += (std::size_t) n_expert * n_tok * f;                    // router logits
    b += (std::size_t) k * n_tok * sizeof(int32_t);              // ids
    b += (std::size_t) k * n_tok * sizeof(int32_t);              // the resident path's ids -> slots
    b += (std::size_t) k * n_tok * f;                            // weights
    b += (std::size_t) expert_dim * n_tok * f;                   // routed_in
    b += (std::size_t) expert_dim * k * n_tok * f;               // the parts
    b += (std::size_t) expert_dim * n_tok * f;                   // the combined latent
    b += (std::size_t) n_embd * n_tok * f;                       // the shared expert's output
    b += 2 * (std::size_t) expert_ff * f;                        // one expert's gate / up (single token)
    b += (std::size_t) expert_dim * f;                           // one token of routed_in
    b += 2 * (std::size_t) expert_ff * n_tok * f;                // the shared expert's gate / up
    b += 2 * (std::size_t) expert_ff * f;                        // the traced first expert's gate / up
    const int widest = n_embd > expert_ff ? n_embd : expert_ff;
    b += kernels::native_q8_1_bytes_wide(widest, n_tok);
    b += kernels::native_q8_1_bytes_wide(expert_ff, n_tok);           // the shared expert's SwiGLU, its own buffer
    b += kernels::native_q8_1_bytes(expert_dim, 1);              // the token's input, read by gate and up
    // k columns, sized one at a time: native_q8_1_bytes validates ncols against MAX_NCOLS (8) and k is 16.
    b += (std::size_t) k * kernels::native_q8_1_bytes(expert_ff, 1);   // all k SwiGLU outputs, read by down
    b += 2 * (std::size_t) k * expert_ff * f;                    // all k experts' gate / up, in one block
    // Reserve grouped prompt scratch only above the batching threshold.
    b += latent_moe_mmq_scratch_bytes(expert_dim, expert_ff, std::max(n_expert, mmq_slots), k, n_tok, mmq_rows);
    // Append window scratch only when this execution path is enabled.
    if (resident_stripe_groups() > 0 && n_tok > 1) {
        const std::size_t wn = (std::size_t) k * (std::size_t) n_tok;
        b += wn * (std::size_t) expert_dim * f;                            // the activation, one per entry
        b += kernels::native_q8_1_bytes_wide(expert_dim, (int) wn);        // and its Q8_1 image
        b += 2 * wn * (std::size_t) expert_ff * f;                         // every entry's gate / up
        b += kernels::native_q8_1_bytes_wide(expert_ff, (int) wn);         // every entry's SwiGLU
    }
    // NO per-group expert arena any more: the matvecs read each expert where it already is, by index.
    (void) gate_type;
    (void) up_type;
    (void) down_type;
    // The exact window's own buffers, AFTER everything above so the default layout does not move.
    if (exact_window) {
        const std::size_t g = (std::size_t) k * n_tok;
        b += kernels::native_q8_1_bytes_wide(expert_dim, n_tok) + 16;    // every token's input
        b += g * kernels::native_q8_1_bytes(expert_dim, 1) + 16;        // a copy a (token, pick)
        b += 2 * (g * expert_ff * f + 16);                              // gate / up
        b += g * kernels::native_q8_1_bytes(expert_ff, 1) + 16;         // the SwiGLU's Q8_1
        b += g * expert_dim * f + 16;                                   // the parts, in sorted order
        b += 3 * (g * sizeof(int32_t) + 16);                            // the plan
    }
    return b + 1024;
}

void latent_moe(const float* cur, const LatentMoeWeights& w, int n_embd, int expert_dim, int expert_ff,
                int n_expert, int k, int n_tok, float routed_scale, float swiglu_clamp, float rms_eps,
                float* out, void* scratch, void* stream, LatentMoeTrace* trace, const MatIn& cur_in) {
    const int ctx_n_tok = w.context_n_tok > 0 ? w.context_n_tok : n_tok;
    const bool resident = w.resident != nullptr && w.resident_slot_of != nullptr;
    const bool asynchronous = !resident && w.async_group != nullptr;
    const bool single_phase=asynchronous && w.async_group->single_phase;
    if(single_phase && w.async_group->device_slot_of==nullptr)
        throw std::runtime_error("single-phase async requires device-plan");
    if (asynchronous && (n_tok != 1 || k != 16)) throw std::invalid_argument("async MoE requires T1 K16");
    const bool windowed = w.window != nullptr;
    const bool fetched = !windowed && !resident && !asynchronous && w.fetch_experts != nullptr;
    if (cur == nullptr || out == nullptr || scratch == nullptr || stream == nullptr || w.gate_inp == nullptr ||
        w.routed_down == nullptr || w.routed_up == nullptr || w.gate_shexp == nullptr ||
        w.up_shexp == nullptr || w.down_shexp == nullptr || w.post_norm == nullptr ||
        (!windowed && !fetched && !resident && !asynchronous && (w.gate_exps == nullptr || w.up_exps == nullptr || w.down_exps == nullptr)))
        throw std::invalid_argument("latent_moe: a stream and every weight (the expert tensors unless "
                                    "fetch_experts or resident is given)");
    if (n_embd <= 0 || expert_dim <= 0 || expert_ff <= 0 || n_expert <= 0 || k <= 0 || n_tok <= 0)
        throw std::invalid_argument("latent_moe: positive extents");
    if (swiglu_clamp <= 0.f)
        throw std::invalid_argument("latent_moe: this model's routed experts are clamped; a clamp of 0 would "
                                    "silently run the shared expert's activation on them");

    char* p = static_cast<char*>(scratch);
    auto take = [&p](std::size_t bytes) {
        void* at = p;
        p += (bytes + 15) & ~(std::size_t) 15;
        return at;
    };
    const std::size_t f = sizeof(float);
    float* logits = (float*) take((std::size_t) n_expert * n_tok * f);
    int32_t* ids = (int32_t*) take((std::size_t) k * n_tok * sizeof(int32_t));
    int32_t* slots = (int32_t*) take((std::size_t) k * n_tok * sizeof(int32_t));
    float* rw = (float*) take((std::size_t) k * n_tok * f);
    float* routed_in = (float*) take((std::size_t) expert_dim * n_tok * f);
    float* parts = (float*) take((std::size_t) expert_dim * k * n_tok * f);
    // Keep the original scratch reservation and every subsequent offset unchanged.
    // The per-layer result allocation is now final mixed output: CPU DMA writes only
    // CPU ranks, indexed GPU down kernels write only their other ranks.
    static const bool direct_cpu_parts = [] {
        const char* e = std::getenv("STRATA_HY4_FLASH_CPU_DIRECT_PARTS");
        return e != nullptr && std::atoi(e) != 0;
    }();
    const bool use_direct_cpu_parts = direct_cpu_parts && asynchronous &&
                                      w.async_group->cpu_parts != nullptr;
    if (use_direct_cpu_parts) {
        static int said = 0;
        if (said++ == 0) std::fprintf(stderr, "HY4_FLASH_CPU_DIRECT_PARTS engaged\n");
        if (n_tok != 1 || k != 16)
            throw std::invalid_argument("latent_moe: direct CPU parts requires T1/top16");
        parts = w.async_group->cpu_parts;
    }

    float* latent = (float*) take((std::size_t) expert_dim * n_tok * f);
    float* shexp = (float*) take((std::size_t) n_embd * n_tok * f);
    float* e_gate = (float*) take((std::size_t) k * expert_ff * f);   // ALL k experts' gates, one matvec
    float* e_up = (float*) take((std::size_t) k * expert_ff * f);
    float* one_in = (float*) take((std::size_t) expert_dim * f);
    float* sh_gate = (float*) take((std::size_t) expert_ff * n_tok * f);
    float* sh_up = (float*) take((std::size_t) expert_ff * n_tok * f);
    float* trace_gate = (float*) take((std::size_t) expert_ff * f);
    float* trace_up = (float*) take((std::size_t) expert_ff * f);
    void* q8 = take(kernels::native_q8_1_bytes_wide(n_embd > expert_ff ? n_embd : expert_ff, n_tok));
    // Give the shared expert its own Q8 buffer while it runs concurrently with routed experts.
    void* q8_sh = take(kernels::native_q8_1_bytes_wide(expert_ff, n_tok));
    // Keep q8_in separate from q8_ff so an expert's down projection cannot overwrite the input used by later
    // experts.
    void* q8_in = take(kernels::native_q8_1_bytes(expert_dim, 1));
    void* q8_ff = take((std::size_t) k * kernels::native_q8_1_bytes(expert_ff, 1));
    // Route tables are indexed by physical pool slots for the resident pass, not by model expert ID.
    // The caller reserves this extent separately from the model's n_expert dimension.
    const std::size_t mmq_scratch_bytes =
        latent_moe_mmq_scratch_bytes(expert_dim, expert_ff,
                                    std::max(n_expert, w.stage_scratch_slots), k, n_tok, w.stage_scratch_rows);
    void* mmq_scratch = mmq_scratch_bytes != 0 ? take(mmq_scratch_bytes) : nullptr;
    // Append resident-window buffers in the order used by the scratch size calculation.
    const int w_entries = k * n_tok;
    const bool size_resident_window = resident_stripe_groups() > 0 && n_tok > 1;
    const bool want_resident_window = size_resident_window && w.verify_exact;
    float* w_x = nullptr; void* w_xq = nullptr;
    float* w_gate = nullptr; float* w_up = nullptr; void* w_hq = nullptr;
    if (size_resident_window) {
        w_x = (float*) take((std::size_t) w_entries * expert_dim * f);
        w_xq = take(kernels::native_q8_1_bytes_wide(expert_dim, w_entries));
        w_gate = (float*) take((std::size_t) w_entries * expert_ff * f);
        w_up = (float*) take((std::size_t) w_entries * expert_ff * f);
        w_hq = take(kernels::native_q8_1_bytes_wide(expert_ff, w_entries));
    }
    // The exact window's buffers, taken last so nothing above moves (see latent_moe_scratch_bytes).
    const bool window = w.exact_window && w.context_n_tok == 1 && resident && !asynchronous && n_tok > 1 &&
                        trace == nullptr && n_tok <= 4 && (std::size_t) k * n_tok <= 1024;
    // The wide form: the same single-launch-per-stage shape as the window, but every (token, pick) group
    // keeps its own token's input and its own expert - no sorting, no sharing, the per-token kernels' grid.y
    // widened. Mutually exclusive with the window (the caller picks one).
    const bool wide = w.exact_wide && w.context_n_tok == 1 && resident && !asynchronous && n_tok > 1 &&
                      trace == nullptr && n_tok <= 4 && (std::size_t) k * n_tok <= 1024;
    void *wq8_all = nullptr, *wq8_rep = nullptr, *wq8_ff = nullptr;
    float *we_gate = nullptr, *we_up = nullptr, *wparts = nullptr;
    int32_t *w_idx = nullptr, *w_tok = nullptr, *w_inv = nullptr;
    if (window || wide) {
        const std::size_t g = (std::size_t) k * n_tok;
        wq8_all = take(kernels::native_q8_1_bytes_wide(expert_dim, n_tok));
        wq8_ff = take(g * kernels::native_q8_1_bytes(expert_ff, 1));
        we_gate = (float*) take(g * expert_ff * f);
        we_up = (float*) take(g * expert_ff * f);
        wparts = (float*) take(g * expert_dim * f);
        if (window) {
            wq8_rep = take(g * kernels::native_q8_1_bytes(expert_dim, 1));
            w_idx = (int32_t*) take(g * sizeof(int32_t));
            w_tok = (int32_t*) take(g * sizeof(int32_t));
            w_inv = (int32_t*) take(g * sizeof(int32_t));
        }
    }
    const std::size_t gu_b = kernels::native_mmvq_weight_bytes(w.gate_exps_type, expert_dim, expert_ff);
    const std::size_t up_b = kernels::native_mmvq_weight_bytes(w.up_exps_type, expert_dim, expert_ff);
    const std::size_t dn_b = kernels::native_mmvq_weight_bytes(w.down_exps_type, expert_ff, expert_dim);

    // ---- the router scores the FULL-WIDTH cur. gate_inp is F32 here, so this is a plain GEMV, not a matvec
    // through the quantised path.
    MatIn cur_im = cur_in;
    if (cur_im.q8 == nullptr) cur_im = mat_stage(cur_in.mc, cur, q8, n_embd, n_tok, stream, kSiteRoutedDn);

    // Compute the unclamped shared expert from the original hidden state.
    const auto shared_expert = [&](void* st, void* branch_q8) {
        // w13: gate and up read the same images of `cur` and have the same shape. The GEMM has no pair, so
        // there it is two products over one activation image.
        if (mat_gemm_path(cur_im.mc, n_tok, kSiteShexpGU) || w.gate_shexp_type != w.up_shexp_type ||
            !kernels::native_mmvq_pair(w.gate_shexp_type, w.gate_shexp, w.up_shexp, cur_im.q8, sh_gate,
                                       sh_up, n_embd, expert_ff, n_tok, st)) {
            mat_apply(cur_im, w.gate_shexp_type, w.gate_shexp, sh_gate, n_embd, expert_ff, n_tok, st, kSiteShexpGU);
            mat_apply(cur_im, w.up_shexp_type, w.up_shexp, sh_up, n_embd, expert_ff, n_tok, st, kSiteShexpGU);
        }
        // Produce FP32 SwiGLU output for GEMM and Q8_1 output for matvec.
        if (mat_gemm_path(cur_im.mc, n_tok, kSiteShexpDn)) {
            launch_pdl(swiglu_plain_k, grid_for((long long) expert_ff * n_tok), kBlock, 0,
                       static_cast<cudaStream_t>(st), sh_gate, sh_up, (long long) expert_ff * n_tok);
            const MatIn h = mat_stage(cur_im.mc, sh_gate, branch_q8, expert_ff, n_tok, st, kSiteShexpDn);
            mat_apply(h, w.down_shexp_type, w.down_shexp, shexp, expert_ff, n_embd, n_tok, st, kSiteShexpDn);
        } else {
            kernels::native_swiglu_quantize_q8_1(sh_gate, sh_up, branch_q8, expert_ff, n_tok, st);
            mat_apply(mat_staged(cur_im.mc,sh_gate,branch_q8,expert_ff,n_tok,st),w.down_shexp_type,
                      w.down_shexp,shexp,expert_ff,n_embd,n_tok,st,kSiteShexpDn);
        }
    };

    // Fork only when the caller supplies a separate Q8 copy of cur. The routed path
    // overwrites the fallback buffer while the shared stream would still be reading it.
    // Run the shared expert on exactly one branch to avoid adding its output twice.
    static const char* shared_stream_flag = std::getenv("STRATA_HY4_FLASH_ASYNC_SHARED_STREAM");
    const bool fork_shared = asynchronous && cur_in.q8 != nullptr && shared_stream_flag &&
        std::atoi(shared_stream_flag) != 0 && w.async_group->shared_stream &&
        w.async_group->shared_fork_event && w.async_group->shared_join_event && w.async_group->shared_q8;
    static const char* early_flag = std::getenv("STRATA_HY4_FLASH_ASYNC_SHARED_EARLY");
    const bool early_shared = !fork_shared && asynchronous && early_flag && std::atoi(early_flag) != 0;
    static const char* compact_flag = std::getenv("STRATA_HY4_FLASH_ASYNC_COMPACT_MISS");
    const bool compact_miss = asynchronous && compact_flag && std::atoi(compact_flag) != 0;

    // Keep GEMM on its bound stream when processing prompt chunks.
    ShexpStream* fork = nullptr;
    if (!asynchronous && shexp_stream_on() && cur_in.q8 != nullptr && !mat_gemm_path(cur_in.mc, n_tok))
        fork = shexp_stream();
    if (fork != nullptr) {
        if (cudaEventRecord(fork->fork, static_cast<cudaStream_t>(stream)) != cudaSuccess ||
            cudaStreamWaitEvent(fork->stream, fork->fork, 0) != cudaSuccess)
            throw std::runtime_error("latent_moe: the shared expert's stream could not join the pass");
        shared_expert(fork->stream, q8_sh);
    }
    // gate_inp is F32. The diagnostic hook substitutes ggml_mul_mat for the native router GEMV.
    static const char* diag_mm = std::getenv("STRATA_DIAG_ROUTER_LIBRARY");
    if (diag_mm != nullptr) {
        using Fn = void (*)(const float*, const float*, float*, int, int, int, void*);
        static void* h = dlopen(diag_mm, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (h == nullptr) throw std::runtime_error(std::string("diag router: ") + dlerror());
        static Fn fn = reinterpret_cast<Fn>(dlsym(h, "hy4_flash_diag_mm_f32"));
        if (fn == nullptr) throw std::runtime_error("diag router: no hy4_flash_diag_mm_f32");
        fn(w.gate_inp, cur, logits, n_embd, n_expert, n_tok, stream);
    } else if (!mat_router_f32(cur_im.mc, w.gate_inp, cur, logits, n_embd, n_expert, n_tok, stream))
    launch_pdl(f32_matvec_k, (unsigned) (n_expert * n_tok), kBlock, 0, static_cast<cudaStream_t>(stream),
               w.gate_inp, cur, logits, n_embd, n_expert, n_tok);
    static const bool fuse_router_publish=[] {
        static const char* e=std::getenv("STRATA_HY4_FLASH_ASYNC_ROUTER_PUBLISH");
        return e && std::atoi(e)!=0;
    }();
    const bool published_in_router=fuse_router_publish && asynchronous && n_tok==1 &&
        kernels::native_router_sigmoid_supported(n_expert,k) && w.async_group->device_slot_of!=nullptr &&
        w.async_group->permanent_slots>0;
    // Initialization enables this only for the supported fused device-plan path.
    // The warp publication fallback below also updates both copies defensively.
    uint32_t* device_epoch=asynchronous?w.async_group->device_epoch:nullptr;
    if(device_epoch && (!w.async_group->device_slot_of || n_tok!=1 || !kernels::native_router_sigmoid_supported(n_expert,k)))
        throw std::runtime_error("latent_moe: invalid device-epoch descriptor");
    if(published_in_router) {
        const auto& ag=*w.async_group;
        kernels::NativeRouterPublishPlan plan{
            &ag.mailbox->request,&ag.mailbox->fast_epoch,&ag.mailbox->error,
            ag.mailbox->ids,ag.mailbox->hit,ag.device_slot_of,ag.hit_index,ag.miss_index,
            ag.n_expert,ag.permanent_slots,device_epoch};
        kernels::native_router_sigmoid_publish(logits,w.exp_probs_b,ids,rw,routed_scale,plan,stream);
    } else
        kernels::native_router_sigmoid(logits, w.exp_probs_b, ids, rw, n_expert, k, n_tok, routed_scale, stream);
    // Async uses only preallocated resources, with a later fork after the routing doorbell.
    // It must not execute the product's lazy TLS shared branch as well.
    if(windowed)window_publish_k<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(w.window->mailbox,ids,n_tok*k);
    if (asynchronous && !published_in_router) {
        const ExpertAsyncGroup& ag = *w.async_group;
        if (ag.device_slot_of != nullptr)
            {
            if(device_epoch) expert_publish_plan_warp_epoch_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                ag.mailbox, ids, ag.device_slot_of, ag.n_expert, ag.permanent_slots,
                ag.hit_index, ag.miss_index,device_epoch);
            else expert_publish_plan_warp_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                ag.mailbox, ids, ag.device_slot_of, ag.n_expert, ag.permanent_slots,
                ag.hit_index, ag.miss_index);
        }
        else
            expert_publish_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(ag.mailbox, ids);
    }
    // Time GPU work from publication through completion of the layer's branch.
    if (asynchronous && w.async_group->split_stats != nullptr)
        expert_split_stamp_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(w.async_group->split_stats, 0);
    if (fork_shared) {
        const auto fork_event = static_cast<cudaEvent_t>(w.async_group->shared_fork_event);
        const auto join_event = static_cast<cudaEvent_t>(w.async_group->shared_join_event);
        const auto sh_stream = static_cast<cudaStream_t>(w.async_group->shared_stream);
        if (cudaEventRecord(fork_event, static_cast<cudaStream_t>(stream)) != cudaSuccess ||
            cudaStreamWaitEvent(sh_stream, fork_event, 0) != cudaSuccess)
            throw std::runtime_error("latent_moe: async shared fork");
        shared_expert(w.async_group->shared_stream, w.async_group->shared_q8);
        if (cudaEventRecord(join_event, sh_stream) != cudaSuccess)
            throw std::runtime_error("latent_moe: async shared join event");
    }


    // ---- the experts work in the latent space
    capture_probe(stream, "latent_moe entry (router + gate_inp)");
    mat_apply(cur_im, w.routed_down_type, w.routed_down, routed_in, n_embd, expert_dim, n_tok, stream,
              kSiteRoutedDn);
    capture_probe(stream, "routed_down");

    if (asynchronous && w.async_group->cpu_input != nullptr)
        {
            if(device_epoch) expert_cpu_input_epoch_k<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(
            w.async_group->mailbox, routed_in, w.async_group->cpu_input, expert_dim,device_epoch);
            else expert_cpu_input_k<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(
            w.async_group->mailbox, routed_in, w.async_group->cpu_input, expert_dim);
        }

    // Group the chunk's routed rows by expert before running MMQ.
    bool batched = false;
    // Report the expert kernel actually selected for the prompt.
    if (n_tok > 1) {
        static int said = 0;
        if (said++ == 0)
            std::fprintf(stderr,
                         "HY4_FLASH_PREFILL_MOE_PATH resident=%d resident_base=%d slot_of=%d fetch=%d async=%d "
                         "gate_exps=%d n_tok=%d k=%d -> %s\n",
                         resident ? 1 : 0, w.resident_base, w.resident_slot_of != nullptr ? 1 : 0,
                         w.fetch_experts != nullptr ? 1 : 0, asynchronous ? 1 : 0,
                         w.gate_exps != nullptr ? 1 : 0, n_tok, k,
                         (resident && w.resident_base >= 0) ? "BATCHED-MMQ"
                         : resident                        ? "per-token resident"
                         : w.fetch_experts != nullptr      ? "per-token fetch"
                                                           : "per-token direct");
    }
    if(windowed) {
        window_experts(routed_in,parts,w,expert_dim,expert_ff,n_tok,swiglu_clamp,stream);
        kernels::native_moe_combine_multi(parts,rw,nullptr,latent,expert_dim,k,n_tok,stream);
        batched=true;
    }
    if (w.draft_cpu != nullptr) {
        // No hits to separate from misses and no victims to choose: every rank is a host row, so the
        // whole top-k is one request and the GPU's only job is to hand over the input and wait.
        if (n_tok != 1 || k != kWindowTopK)
            throw std::invalid_argument("latent_moe: draft CPU experts require one token and top-16");
        const auto st = static_cast<cudaStream_t>(stream);
        draft_cpu_publish_k<<<1, 256, 0, st>>>(w.draft_cpu->mailbox, ids, routed_in,
                                               w.draft_cpu->input, expert_dim, k);
        draft_cpu_parts_k<<<1, 256, 0, st>>>(w.draft_cpu->mailbox, w.draft_cpu->parts, parts,
                                             k * expert_dim);
        kernels::native_moe_combine_multi(parts,rw,nullptr,latent,expert_dim,k,n_tok,stream);
        batched=true;
    }
    if (resident && n_tok > 1 && w.resident_base >= 0) {
        const std::size_t need =
            latent_moe_mmq_scratch_bytes(expert_dim, expert_ff, n_expert, k, n_tok);
        if (need != 0 && mmq_scratch != nullptr && mmq_scratch_bytes >= need)
            batched = latent_moe_experts_mmq(routed_in, *w.resident, w.resident_base, w.gate_exps_type,
                                             w.up_exps_type, w.down_exps_type, expert_dim, expert_ff,
                                             n_expert, k, n_tok, ids, rw, swiglu_clamp, latent,
                                             mmq_scratch, mmq_scratch_bytes, stream);
    }

    // The full tier consumes router IDs on device.
    if (!batched) {
    // The resident path keeps them there: one kernel turns the router's ids into slot numbers and the
    // matvecs index themselves off this block's planes. No host step, so nothing has to be waited for.
    if (resident) {
        const int n = k * n_tok;
        launch_pdl(map_slots_k, grid_for(n), kBlock, 0, static_cast<cudaStream_t>(stream),
                   ids, w.resident_slot_of, slots, n, n_expert);
        capture_probe(stream, "map_slots_k");
    }
    // Map each token's selected experts to slots even when all experts are resident.
    bool window_striped = false;
    if (resident && want_resident_window && k == kWindowTopK &&
        w.gate_exps_type == w.up_exps_type) {
        const auto st = static_cast<cudaStream_t>(stream);
        const auto xb = kernels::native_q8_1_bytes(expert_dim, 1);
        const auto hb = kernels::native_q8_1_bytes(expert_ff, 1);
        const int stripes = resident_stripe_groups();
        window_repeat_k<<<grid_for((long long) w_entries * expert_dim), kBlock, 0, st>>>(
            routed_in, w_x, expert_dim, w_entries);
        kernels::native_quantize_q8_1_wide(w_x, w_xq, expert_dim, w_entries, stream);
        window_striped = kernels::native_mmvq_window_striped_pair(
            w.gate_exps_type, w.resident->gate, w.resident->gate_stride, w.resident->up,
            w.resident->up_stride, w_xq, xb, w_gate, w_up, expert_dim, expert_ff, w_entries, stream,
            slots, stripes);
        if (window_striped) {
            kernels::native_swiglu_clamp_quantize_q8_1_masked(w_gate, w_up, w_hq, expert_ff, w_entries,
                                                              swiglu_clamp, slots, stream);
            window_striped = kernels::native_mmvq_window_striped(
                w.down_exps_type, w.resident->down, w.resident->down_stride, w_hq, hb, parts, expert_ff,
                expert_dim, w_entries, stream, slots, stripes);
        }
        // Fall back when the paired kernel declines the input shape.
        static int said = 0;
        if (said++ == 0)
            std::fprintf(stderr, "HY4_FLASH_RESIDENT_STRIPES %s: groups=%d entries=%d n_tok=%d k=%d\n",
                         window_striped ? "engaged" : "DECLINED -> per-token loop", stripes, w_entries,
                         n_tok, k);
    }
    std::vector<int32_t> host_ids;
    if (fetched) {
        host_ids.resize((std::size_t) k * n_tok);
        // Reading the selected IDs on the host synchronizes the routing stream.
        const auto t0 = std::chrono::steady_clock::now();
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        const auto t1 = std::chrono::steady_clock::now();
        // Report the CUDA error at the synchronization point.
        cudaError_t rc = cudaMemcpyAsync(host_ids.data(), ids, host_ids.size() * sizeof(int32_t),
                                         cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
        if (rc == cudaSuccess) rc = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        if (rc != cudaSuccess) {
            std::size_t mfree = 0, mtot = 0;
            cudaMemGetInfo(&mfree, &mtot);
            cudaGetLastError();
            throw std::runtime_error(std::string("latent_moe: reading the routed expert ids back: ") +
                                     cudaGetErrorString(rc) + " (" +
                                     std::to_string((long) (mfree / (1024 * 1024))) + " MiB free of " +
                                     std::to_string((long) (mtot / (1024 * 1024))) +
                                     "; this is the streaming expert path, which runs when the tier "
                                     "cannot hold every expert - raise --expert-vram or lower the context)");
        }
        const auto t2 = std::chrono::steady_clock::now();
        g_route_backlog += std::chrono::duration<double>(t1 - t0).count();
        g_route_sync += std::chrono::duration<double>(t2 - t1).count();
        ++g_route_syncs;
    }

    // Group prompt rows by expert and stage misses in batches that fit the ring.
    // Every distinct routed expert must belong to a batch: the combine skips entries
    // with inv == -1. Accumulate each batch into latent to recover the full sum.
    bool staged_batched = false;
    if (fetched && w.stage_experts != nullptr && w.stage_batch_max > 0 && n_tok > 1 &&
        moe_mmq_usable(w.gate_exps_type, w.up_exps_type, w.down_exps_type, expert_dim, expert_ff, n_tok)) {
        // The ring's slot count is the launch's expert count, so the scratch is sized for THAT, not for
        // `n_expert` - a 64-slot ring needs far less than 512 experts' worth of bounds and cursors.
        const std::size_t need =
            latent_moe_mmq_scratch_bytes(expert_dim, expert_ff, w.stage_n_slot, k, n_tok, w.stage_scratch_rows);
        const bool room = need != 0 && mmq_scratch != nullptr && mmq_scratch_bytes >= need;
        // The distinct routed experts of this chunk, in ascending order so the partition is deterministic
        // and a batch's contents do not depend on routing order.
        std::vector<int32_t> distinct;
        distinct.reserve((std::size_t) w.stage_batch_max);
        {
            std::vector<uint8_t> seen((std::size_t) n_expert, 0);
            for (std::size_t i = 0; i < host_ids.size(); ++i) {
                const int32_t e = host_ids[i];
                if (e >= 0 && e < n_expert) seen[(std::size_t) e] = 1;
            }
            for (int e = 0; e < n_expert; ++e)
                if (seen[(std::size_t) e]) distinct.push_back(e);
        }
        // Resident experts and staged misses use separate weight planes. Run the resident
        // set directly and split the misses into ring-sized batches. Assign each expert to
        // exactly one pass so every (token, pick) contribution is scattered once.
        std::vector<int32_t> resident_ids, miss_ids, host_slot;
        if (w.stage_slot_of != nullptr && w.stage_resident != nullptr && w.stage_resident_slots > 0) {
            host_slot.resize((std::size_t) n_expert);
            if (cudaMemcpyAsync(host_slot.data(), w.stage_slot_of, host_slot.size() * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)) == cudaSuccess &&
                cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess) {
                for (int32_t e : distinct) {
                    if (host_slot[(std::size_t) e] >= 0) resident_ids.push_back(e);
                    else miss_ids.push_back(e);
                }
            } else {
                cudaGetLastError();
                miss_ids = distinct;   // could not read the map: stage everything, as before
            }
        } else {
            miss_ids = distinct;
        }
        if (room && !distinct.empty()) {
            // Scatter each batch's outputs into parts before reducing over expert ranks.
            if (cudaMemsetAsync(parts, 0, (std::size_t) expert_dim * k * n_tok * sizeof(float),
                                static_cast<cudaStream_t>(stream)) != cudaSuccess)
                throw std::runtime_error("latent_moe: cannot clear the expert parts");
            std::size_t covered = 0;
            bool ok = true;
            // Resolve the resident workspace fallback BEFORE publishing one immutable row plan.
            if (!resident_ids.empty()) {
                const std::size_t rneed = latent_moe_mmq_scratch_bytes(expert_dim, expert_ff,
                    w.stage_resident_slots, k, n_tok, w.stage_scratch_rows);
                if (rneed == 0 || mmq_scratch_bytes < rneed) {
                    miss_ids.insert(miss_ids.end(), resident_ids.begin(), resident_ids.end());
                    std::sort(miss_ids.begin(), miss_ids.end()); resident_ids.clear();
                }
            }
            static const bool plan_on = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_ROW_PLAN");
                return v != nullptr && std::atoi(v) != 0;
            }();
            std::vector<MoeBatchPlan> row_plans;
            std::size_t plan_at = 0;
            if (plan_on) {
                std::vector<MoeBatchSpec> specs;
                if (!resident_ids.empty()) specs.push_back({host_slot, w.stage_resident_slots});
                for (std::size_t at = 0; at < miss_ids.size(); at += (std::size_t) w.stage_batch_max) {
                    MoeBatchSpec spec; spec.n_slots = w.stage_n_slot;
                    spec.slot_of.assign((std::size_t) n_expert, -1);
                    const std::size_t end = std::min(at + (std::size_t) w.stage_batch_max, miss_ids.size());
                    for (std::size_t j = at; j < end; ++j) spec.slot_of[(std::size_t) miss_ids[j]] = (int32_t) (j - at);
                    specs.push_back(std::move(spec));
                }
                row_plans = moe_prepare_batches(host_ids, specs, n_expert, k, n_tok, routed_in,
                                                w.gate_exps_type, expert_dim, stream);
            }
            // Run resident experts directly from the tier.
            if (!resident_ids.empty()) {
                const std::size_t rneed = latent_moe_mmq_scratch_bytes(expert_dim, expert_ff,
                                                                       w.stage_resident_slots, k, n_tok, w.stage_scratch_rows);
                if (rneed != 0 && mmq_scratch != nullptr && mmq_scratch_bytes >= rneed) {
                    ok = latent_moe_experts_mmq(routed_in, *w.stage_resident, 0, w.gate_exps_type,
                                                w.up_exps_type, w.down_exps_type, expert_dim, expert_ff,
                                                n_expert, k, n_tok, ids, rw, swiglu_clamp, latent,
                                                mmq_scratch, mmq_scratch_bytes, stream, w.stage_slot_of,
                                                w.stage_resident_slots, false, parts,
                                                plan_on ? &row_plans.at(plan_at++) : nullptr);
                    if (ok) covered += resident_ids.size();
                } else {
                    // no room for the pool-wide launch: fold them back into the staged batches
                    miss_ids.insert(miss_ids.end(), resident_ids.begin(), resident_ids.end());
                    std::sort(miss_ids.begin(), miss_ids.end());
                    resident_ids.clear();
                }
            }
            if (ok && !miss_ids.empty() && w.stage_plan != nullptr)
                w.stage_plan(w.stage_block, miss_ids.data(), (int) miss_ids.size(), stream, w.stage_user);
            // ---- pass 2: the misses, in ring-sized batches.
            for (std::size_t at = 0; at < miss_ids.size() && ok; at += (std::size_t) w.stage_batch_max) {
                const std::size_t n = std::min((std::size_t) w.stage_batch_max, miss_ids.size() - at);
                std::vector<int32_t> batch(miss_ids.begin() + (long) at, miss_ids.begin() + (long) (at + n));
                ExpertGroup rg{};
                const int32_t* slot_dev = nullptr;
                if (!w.stage_experts(w.stage_block, batch.data(), (int) n, &rg, &slot_dev, stream,
                                     w.stage_user)) {
                    ok = false;
                    break;
                }
                ok = latent_moe_experts_mmq(routed_in, rg, 0, w.gate_exps_type, w.up_exps_type,
                                            w.down_exps_type, expert_dim, expert_ff, n_expert, k, n_tok,
                                            ids, rw, swiglu_clamp, latent, mmq_scratch, mmq_scratch_bytes,
                                            stream, slot_dev, w.stage_n_slot, /*accumulate=*/false,
                                            /*parts_out=*/parts, plan_on ? &row_plans.at(plan_at++) : nullptr);
                if (w.stage_release != nullptr) {
                    if (!ok) throw std::runtime_error("latent_moe: pipelined MMQ failed after staging");
                    w.stage_release(stream, w.stage_user);
                }
                if (ok) covered += n;
            }
            // ONE COMBINE, after every batch has scattered. This is the same call the per-token path makes,
            // so the weighted sum is the same function with the same rounding.
            if (ok)
                kernels::native_moe_combine_multi(parts, rw, nullptr, latent, expert_dim, k, n_tok, stream);
            if (plan_on) moe_finish_batches(stream);
            if (ok && covered != distinct.size())
                throw std::runtime_error("latent_moe: the expert batches did not cover the routed set (" +
                                         std::to_string(covered) + " of " +
                                         std::to_string(distinct.size()) + ") - the result would be a "
                                         "silently short weighted sum");
            staged_batched = ok;
            // EVERY 41st CALL, i.e. once a chunk rather than once a process: the first chunk always finds
            // the cache empty, so "0 resident" there says nothing about whether the split works.
            static int said = 0;
            if (said++ % 41 == 0)
                std::fprintf(stderr,
                             "HY4_FLASH_PREFILL_BATCH %s: %zu distinct (%zu resident in place, %zu staged in "
                             "%zu batches of %d), n_tok=%d\n",
                             ok ? "engaged" : "DECLINED -> per-token", distinct.size(),
                             resident_ids.size(), miss_ids.size(),
                             (miss_ids.size() + (std::size_t) w.stage_batch_max - 1) /
                                 (std::size_t) w.stage_batch_max,
                             w.stage_batch_max, n_tok);
        }
    }
    // The stripe path has already written every entry of `parts`; it still owes the weighted sum, and it
    // owes it through the SAME call the loop below makes so the rounding is one function, not two.
    if (window_striped)
        kernels::native_moe_combine_multi(parts, rw, nullptr, latent, expert_dim, k, n_tok, stream);
    capture_probe(stream, "the stripes decision / combine");
    if (!staged_batched && !window_striped) {
    const std::size_t q8_col = kernels::native_q8_1_bytes(expert_ff, 1);
    if (window) {
        // Batch all token/expert pairs for exact window verification.
        const int G = k * n_tok;
        const std::size_t in_col = kernels::native_q8_1_bytes(expert_dim, 1);
        kernels::native_quantize_q8_1_wide(routed_in, wq8_all, expert_dim, n_tok, stream);
        moe_window_plan_k<<<1, (G + 31) / 32 * 32, 0, static_cast<cudaStream_t>(stream)>>>(slots, G, k, w_idx,
                                                                                          w_tok, w_inv);
        kernels::gather_rows(static_cast<const uint8_t*>(wq8_all), (int64_t) in_col, w_tok, G,
                             static_cast<uint8_t*>(wq8_rep), stream);
        ExpertGroup grp = *w.resident;
        if (w.gate_exps_type != w.up_exps_type ||
            !kernels::native_mmvq_window(w.gate_exps_type, grp.gate, grp.gate_stride, wq8_rep, in_col,
                                         we_gate, expert_dim, expert_ff, G, stream, w_idx,
                                         grp.up, grp.up_stride, we_up)) {
            kernels::native_mmvq_grouped_context(w.gate_exps_type, grp.gate, grp.gate_stride, wq8_rep, in_col,
                                                 we_gate, expert_dim, expert_ff, G, stream, w_idx, 1);
            kernels::native_mmvq_grouped_context(w.up_exps_type, grp.up, grp.up_stride, wq8_rep, in_col, we_up,
                                                 expert_dim, expert_ff, G, stream, w_idx, 1);
        }
        kernels::native_swiglu_clamp_quantize_q8_1(we_gate, we_up, wq8_ff, expert_ff, G, swiglu_clamp, stream);
        if (!kernels::native_mmvq_window(w.down_exps_type, grp.down, grp.down_stride, wq8_ff, q8_col, wparts,
                                         expert_ff, expert_dim, G, stream, w_idx))
            kernels::native_mmvq_grouped_context(w.down_exps_type, grp.down, grp.down_stride, wq8_ff, q8_col,
                                                 wparts, expert_ff, expert_dim, G, stream, w_idx, 1);
        kernels::gather_rows(reinterpret_cast<const uint8_t*>(wparts), (int64_t) expert_dim * (int64_t) f, w_inv,
                             G, reinterpret_cast<uint8_t*>(parts), stream);
    }
    bool wide_ok = false;
    if (wide) {
        // Launch the token-major pairs together without reordering the weights.
        const int G = k * n_tok;
        const std::size_t in_col = kernels::native_q8_1_bytes(expert_dim, 1);
        kernels::native_quantize_q8_1_wide(routed_in, wq8_all, expert_dim, n_tok, stream);
        ExpertGroup grp = *w.resident;
        grp.index = slots;
        // The gate/up pair through the reuse kernel's token-major wide form only: the old grouped kernel
        // reads a flat per-group stride, so a wide launch without it would feed a group its neighbour's
        // token. A decline runs the per-token loop instead.
        if (w.gate_exps_type == w.up_exps_type &&
            kernels::native_mmvq_grouped_pair_wide(w.gate_exps_type, grp.gate, grp.gate_stride, grp.up,
                                                   grp.up_stride, wq8_all, in_col, we_gate, we_up,
                                                   expert_dim, expert_ff, G, k, stream, grp.index, 1)) {
            wide_ok = true;
            kernels::native_swiglu_clamp_quantize_q8_1(we_gate, we_up, wq8_ff, expert_ff, G, swiglu_clamp, stream);
            kernels::native_mmvq_grouped_context(w.down_exps_type, grp.down, grp.down_stride, wq8_ff, q8_col,
                                                 parts, expert_ff, expert_dim, G, stream, grp.index, 1);
        }
    }
    if (window || wide_ok) {
        // STRATA_HY4_FLASH_DFLY_MOE_CHECK=1 (diagnostic; synchronises): the per-token resident calls the loop below
        // makes, on the same inputs, into the window's own buffers, and every part compared byte for byte.
        static const bool check = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_MOE_CHECK");
            return v != nullptr && std::atoi(v) != 0;
        }();
        if (check) {
            const std::size_t G = (std::size_t) k * n_tok;
            const std::size_t np = (std::size_t) expert_dim * G;
            std::vector<uint32_t> got(np), ref(np);
            cudaMemcpyAsync(got.data(), parts, np * 4, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
            for (int t = 0; t < n_tok; ++t) {
                kernels::native_quantize_q8_1(routed_in + (std::size_t) t * expert_dim, q8_in, expert_dim, 1, stream);
                const int32_t* idx = slots + (std::size_t) t * k;
                ExpertGroup lgrp = *w.resident;
                lgrp.index = idx;
                if (w.gate_exps_type != w.up_exps_type ||
                    !kernels::native_mmvq_grouped_pair(w.gate_exps_type, lgrp.gate, lgrp.gate_stride, lgrp.up,
                                                       lgrp.up_stride, q8_in, 0, e_gate, e_up, expert_dim, expert_ff,
                                                       k, stream, idx, 1)) {
                    kernels::native_mmvq_grouped_context(w.gate_exps_type, lgrp.gate, lgrp.gate_stride, q8_in, 0,
                                                         e_gate, expert_dim, expert_ff, k, stream, idx, 1);
                    kernels::native_mmvq_grouped_context(w.up_exps_type, lgrp.up, lgrp.up_stride, q8_in, 0, e_up,
                                                         expert_dim, expert_ff, k, stream, idx, 1);
                }
                kernels::native_swiglu_clamp_quantize_q8_1(e_gate, e_up, q8_ff, expert_ff, k, swiglu_clamp, stream);
                kernels::native_mmvq_grouped_context(w.down_exps_type, lgrp.down, lgrp.down_stride, q8_ff, q8_col,
                                                     wparts + (std::size_t) t * k * expert_dim, expert_ff, expert_dim,
                                                     k, stream, idx, 1);
            }
            cudaMemcpyAsync(ref.data(), wparts, np * 4, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            std::size_t bad = 0;
            for (std::size_t x = 0; x < np; ++x) bad += got[x] != ref[x];
            std::vector<int32_t> hs((std::size_t) G);
            cudaMemcpy(hs.data(), slots, hs.size() * 4, cudaMemcpyDeviceToHost);
            std::sort(hs.begin(), hs.end());
            const long distinct = std::unique(hs.begin(), hs.end()) - hs.begin();
            static std::size_t calls = 0, values = 0, bad_calls = 0;
            ++calls;
            values += np;
            bad_calls += bad != 0;
            std::fprintf(stderr, "DFLY_MOE_CHECK n_tok %d: %ld distinct of %zu picks, %zu/%zu parts differ "
                         "(%zu calls, %zu values, %zu calls differ)\n",
                         n_tok, distinct, G, bad, np, calls, values, bad_calls);
        }
    }
    for (int t = 0; t < ((window || wide_ok) ? 0 : n_tok); ++t) {
        // Token t's column is already contiguous in routed_in.
        (void) one_in;
        kernels::native_quantize_q8_1(routed_in + (std::size_t) t * expert_dim, q8_in, expert_dim, 1,
                                      stream);

        // Resolve the selected experts to their resident or staged slots.
        ExpertGroup grp;
        if (asynchronous) {
            grp = w.async_group->planes;
            if(single_phase)
                {
            if(device_epoch) expert_wait_all_device_epoch_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                    w.async_group->mailbox,w.async_group->hit_index,device_epoch);
            else expert_wait_all_device_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                    w.async_group->mailbox,w.async_group->hit_index);
        }
            else if (w.async_group->device_slot_of != nullptr)
                {
            if(device_epoch) expert_wait_plan_device_epoch_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                    w.async_group->mailbox,w.async_group->hit_index,w.async_group->miss_index,device_epoch);
            else expert_wait_plan_device_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                    w.async_group->mailbox,w.async_group->hit_index,w.async_group->miss_index);
        }
            else
                expert_wait_plan_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                    w.async_group->mailbox,w.async_group->hit_index,w.async_group->miss_index);
        } else if (resident) {
            grp = *w.resident;
            grp.index = slots + (std::size_t) t * k;
        } else if (fetched) {
            if (!w.fetch_experts(host_ids.data() + (std::size_t) t * k, k, &grp, stream, w.fetch_user))
                throw std::runtime_error("latent_moe: fetch_experts failed");
        } else {
            grp.gate = w.gate_exps;  grp.gate_stride = gu_b;
            grp.up = w.up_exps;      grp.up_stride = up_b;
            grp.down = w.down_exps;  grp.down_stride = dn_b;
            grp.index = ids + (std::size_t) t * k;
        }

        // Group the selected gate, up and down projections by expert.
        for (int phase=0;phase<(asynchronous && !single_phase?2:1);++phase) {
        if (asynchronous) {
            if (phase==1) {
                if (w.async_group->device_slot_of != nullptr)
                    {
            if(device_epoch) expert_wait_miss_device_epoch_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                        w.async_group->mailbox,w.async_group->miss_index,device_epoch);
            else expert_wait_miss_device_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                        w.async_group->mailbox,w.async_group->miss_index);
        }
                else
                    expert_wait_miss_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(
                        w.async_group->mailbox,w.async_group->miss_index);
            }
            grp.index=phase==0?w.async_group->hit_index:w.async_group->miss_index;
        }
        // w13: gate and up in one launch when they are the same type. Same activation, same shape, same
        // expert index - only the bank and the output differ.
        static const bool compact_pair_gu=[] {
            const char* e=std::getenv("STRATA_HY4_FLASH_COMPACT_PAIR_GU");return e&&std::atoi(e)!=0;
        }();
        static const bool resident_pair_gu=[] {
            const char* e=std::getenv("STRATA_HY4_FLASH_RESIDENT_PAIR_GU");return e&&std::atoi(e)!=0;
        }();
        // Separate opt-in for the resident half of the two-phase partial-cache path.
        // Single-phase and fully resident paths retain their original PDL dispatch.
        const bool pair_requested=(compact_pair_gu && compact_miss && phase==1) ||
            (resident_pair_gu && asynchronous && !single_phase && phase==0);
        const bool compact_paired=pair_requested && w.gate_exps_type==w.up_exps_type &&
            kernels::native_mmvq_grouped_compact_pair(w.gate_exps_type,grp.gate,grp.gate_stride,
                grp.up,grp.up_stride,q8_in,0,e_gate,e_up,expert_dim,expert_ff,k,stream,grp.index,ctx_n_tok);
        // Report the first use of this path when tracing is enabled.
        if (compact_paired) {
            static int said = 0;
            if (said++ == 0)
                std::fprintf(stderr, "HY4_FLASH_PAIR_GU engaged: type %d, phase %d, %s\n", w.gate_exps_type, phase,
                             asynchronous ? (single_phase ? "async single-phase" : "async two-phase")
                                          : "synchronous");
        }
        if (!compact_paired && ((compact_miss && phase == 1) || w.gate_exps_type != w.up_exps_type ||
            !kernels::native_mmvq_grouped_pair(w.gate_exps_type, grp.gate, grp.gate_stride, grp.up,
                                               grp.up_stride, q8_in, 0, e_gate, e_up, expert_dim,
                                               expert_ff, k, stream, grp.index, ctx_n_tok))) {
            (compact_miss && phase == 1 ? kernels::native_mmvq_grouped_compact_miss : kernels::native_mmvq_grouped_context)(w.gate_exps_type, grp.gate, grp.gate_stride, q8_in, 0,
                                                 e_gate, expert_dim, expert_ff, k, stream, grp.index, ctx_n_tok);
            (compact_miss && phase == 1 ? kernels::native_mmvq_grouped_compact_miss : kernels::native_mmvq_grouped_context)(w.up_exps_type, grp.up, grp.up_stride, q8_in, 0, e_up,
                                                 expert_dim, expert_ff, k, stream, grp.index, ctx_n_tok);
        }
        if (trace != nullptr && t == 0 && asynchronous) {
            expert_trace_masked_k<<<grid_for(expert_ff),kBlock,0,static_cast<cudaStream_t>(stream)>>>(
                e_gate,e_up,trace_gate,trace_up,grp.index,expert_ff);
        }
        if (trace != nullptr && t == 0 && !asynchronous) {
            // Save the first routed expert's projections before SwiGLU overwrites the gate.
            cudaMemcpyAsync(trace_gate, e_gate, (std::size_t) expert_ff * sizeof(float),
                            cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
            cudaMemcpyAsync(trace_up, e_up, (std::size_t) expert_ff * sizeof(float),
                            cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
        }
        // Fuse clamped SwiGLU and Q8_1 quantization across the selected experts.
        if (asynchronous && (!single_phase || w.async_group->cpu_parts != nullptr))
            kernels::native_swiglu_clamp_quantize_q8_1_masked(e_gate,e_up,q8_ff,expert_ff,k,swiglu_clamp,grp.index,stream);
        else kernels::native_swiglu_clamp_quantize_q8_1(e_gate, e_up, q8_ff, expert_ff, k, swiglu_clamp, stream);
        // The parts are [expert_dim, k, n_tok]: token-major with the k experts contiguous, which is both
        // what the grouped kernel writes and what native_moe_combine_multi reads.
        (compact_miss && phase == 1 ? kernels::native_mmvq_grouped_compact_miss : kernels::native_mmvq_grouped_context)(w.down_exps_type, grp.down, grp.down_stride, q8_ff, q8_col,
                                             parts + (std::size_t) t * k * expert_dim, expert_ff,
                                             expert_dim, k, stream, grp.index, w.verify_exact ? 1 : ctx_n_tok);
        if (early_shared && phase == 0) shared_expert(stream, q8_sh);
        } // original-rank resident/miss phases join before combine
        if (t == 0) capture_probe(stream, "the per-token loop's first token");
    }

    if (asynchronous && w.async_group->cpu_parts != nullptr) {
        // ...and stops here, BEFORE the join. Everything after this point is the GPU standing still until
        // the CPU's rows land, which is the other branch's time, not this one's.
        if (w.async_group->split_stats != nullptr)
            expert_split_stamp_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(w.async_group->split_stats, 1);
        if (use_direct_cpu_parts)
            {
            if(device_epoch) expert_cpu_wait_results_epoch_k<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(w.async_group->mailbox,device_epoch);
            else expert_cpu_wait_results_k<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(w.async_group->mailbox);
        }
        else
            {
            if(device_epoch) expert_cpu_merge_epoch_k<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(
                w.async_group->mailbox, w.async_group->cpu_parts, parts, expert_dim,device_epoch);
            else expert_cpu_merge_k<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(
                w.async_group->mailbox, w.async_group->cpu_parts, parts, expert_dim);
        }
    }

    // The weighted sum, in the latent space. NO shared row here - the shared expert is added after
    // routed_up. The batched path has already written `latent`, in this same k order and rounding.
    kernels::native_moe_combine_multi(parts, rw, nullptr, latent, expert_dim, k, n_tok, stream);
    }  // !staged_batched: the per-token path
    }  // !batched

    // ---- back to full width, applied ONCE to the sum
    const MatIn lat = mat_stage(cur_in.mc, latent, q8, expert_dim, n_tok, stream, kSiteRoutedUp);
    capture_probe(stream, "the expert path (loop or stripes) + combine");
    mat_apply(lat, w.routed_up_type, w.routed_up, out, expert_dim, n_embd, n_tok, stream, kSiteRoutedUp);
    capture_probe(stream, "routed_up");

    // Start the shared expert here if it was not already dispatched to its own stream.
    if (fork == nullptr && !fork_shared && !early_shared) {
        if (cur_in.q8 == nullptr)
            cur_im = mat_stage(cur_in.mc, cur, q8, n_embd, n_tok, stream, kSiteShexpGU);
        shared_expert(stream, q8_sh);
    }

    if (trace != nullptr) {
        trace->routed_in = routed_in;
        trace->parts = parts;
        trace->latent = latent;
        trace->shexp = shexp;
        trace->ids = ids;
        trace->weights = rw;
        trace->logits = logits;
        trace->first_gate = trace_gate;
        trace->first_up = trace_up;
    }

    // ---- JOIN, before `shexp` is read.
    if (fork != nullptr) {
        if (cudaEventRecord(fork->join, fork->stream) != cudaSuccess ||
            cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), fork->join, 0) != cudaSuccess)
            throw std::runtime_error("latent_moe: the shared expert's stream could not rejoin the pass");
    }
    if (fork_shared && cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
            static_cast<cudaEvent_t>(w.async_group->shared_join_event), 0) != cudaSuccess)
        throw std::runtime_error("latent_moe: async shared branch join");
    launch_pdl(add_inplace_k, grid_for((long long) n_embd * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), out, shexp, (long long) n_embd * n_tok);
    kernels::rms_norm_weighted_wide(out, w.post_norm, n_tok, n_embd, rms_eps, stream);

    if(windowed)window_done_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(w.window->mailbox);
    if (asynchronous) {
            if(device_epoch) expert_done_epoch_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(w.async_group->mailbox,device_epoch);
            else expert_done_k<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(w.async_group->mailbox);
        }
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("latent_moe: ") + cudaGetErrorString(e));
}

}  // namespace strata::hy4_flash
