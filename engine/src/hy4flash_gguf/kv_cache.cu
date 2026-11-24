// src/hy4flash_gguf/kv_cache.cu - the latent KV cache. See include/strata/hy4flash_gguf/kv_cache.hpp.
#include "strata/hy4flash_gguf/kv_cache.hpp"

#include "strata/kernels/pdl.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace strata::hy4_flash {

KvCache::~KvCache() {
    for (void* p : kv_cmpr)
        if (p) cudaFree(p);
    for (void* p : k_pe)
        if (p) cudaFree(p);
}

std::size_t kv_cache_bytes(int n_blocks, int kv_lora, int n_rot, int max_tokens, int format) {
    if (n_blocks <= 0 || max_tokens <= 0) return 0;
    return (std::size_t) n_blocks * (std::size_t) max_tokens *
           (kv_cache_lat_row_bytes(kv_lora, format) + kv_cache_pe_row_bytes(n_rot, format));
}

/// Select latent storage with STRATA_HY4_FLASH_KV.
int kv_cache_format_default() {
    static const int format = [] {
        if (const char* v = std::getenv("STRATA_HY4_FLASH_KV")) {
            const std::string f(v);
            if (f == "q8_0" || f == "q8" || f == "Q8_0") return (int) kKvQ8;
            if (f == "f16" || f == "fp16" || f == "F16") return (int) kKvF16;
            if (f == "f32" || f == "fp32" || f == "F32") return (int) kKvF32;
            std::fprintf(stderr, "hy4_flash: STRATA_HY4_FLASH_KV=%s is not q8_0, f16 or f32; using f16\n", v);
            return (int) kKvF16;
        }
        const char* v = std::getenv("STRATA_HY4_FLASH_KV_F32");
        if (v != nullptr && std::atoi(v) != 0) return (int) kKvF32;
        return (int) kKvF16;
    }();
    return format;
}

bool kv_cache_init(KvCache& c, int n_blocks, int kv_lora, int n_rot, int max_tokens, std::string& err) {
    return kv_cache_init(c, n_blocks, kv_lora, n_rot, max_tokens, kv_cache_format_default(), err);
}

bool kv_cache_init(KvCache& c, int n_blocks, int kv_lora, int n_rot, int max_tokens, int format,
                   std::string& err) {
    if (n_blocks <= 0 || kv_lora <= 0 || n_rot <= 0 || max_tokens <= 0) {
        err = "kv_cache_init: positive extents";
        return false;
    }
    c.n_blocks = n_blocks;
    c.kv_lora = kv_lora;
    c.n_rot = n_rot;
    c.max_tokens = max_tokens;
    c.len = 0;
    c.q8 = format == kKvQ8;
    if (c.q8 && kv_lora % 32 != 0) {
        err = "kv_cache_init: a Q8_0 latent needs kv_lora divisible by 32, not " + std::to_string(kv_lora);
        return false;
    }
    c.fp16 = format != kKvF32;
    const std::size_t lat = kv_cache_lat_row_bytes(kv_lora, format) * (std::size_t) max_tokens;
    const std::size_t pe = kv_cache_pe_row_bytes(n_rot, format) * (std::size_t) max_tokens;
    c.kv_cmpr.assign((std::size_t) n_blocks, nullptr);
    c.k_pe.assign((std::size_t) n_blocks, nullptr);
    for (int b = 0; b < n_blocks; ++b) {
        // Zeroed: a masked position is never read, but a NaN left in an unwritten row would reach the score
        // through the dot before the mask is applied, and a quiet 0 there is easier to reason about.
        if (cudaMalloc(&c.kv_cmpr[(std::size_t) b], lat) != cudaSuccess ||
            cudaMalloc(&c.k_pe[(std::size_t) b], pe) != cudaSuccess) {
            err = "kv_cache_init: no room for " +
                  std::to_string(kv_cache_bytes(n_blocks, kv_lora, n_rot, max_tokens, format) >> 20) +
                  " MiB of cache";
            return false;
        }
        cudaMemset(c.kv_cmpr[(std::size_t) b], 0, lat);
        cudaMemset(c.k_pe[(std::size_t) b], 0, pe);
    }
    // The launcher looks for this string in the binary to know the build has the Q8_0 cache.
    if (c.q8) std::fprintf(stderr, "HY4_FLASH_KV_Q8 enabled: %d blocks x %d positions\n", n_blocks, max_tokens);
    return true;
}

namespace {

using strata::kernels::launch_pdl;
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;

/// Read the KV destination offset on device so the graph can replay at new positions.
template <class T>
__global__ void append_k(T* __restrict__ dst, const float* __restrict__ src, int width, int n_tok,
                         const int32_t* __restrict__ at) {
    const long long total = (long long) width * n_tok;
    const long long base = (long long) (*at) * width;
    for (long long x = blockIdx.x * (long long) blockDim.x + threadIdx.x; x < total;
         x += (long long) gridDim.x * blockDim.x)
        dst[base + x] = (T) src[x];
}

/// The host-offset form, which also has to CONVERT when the rows are F16 - a memcpy cannot.
template <class T>
__global__ void append_at_k(T* __restrict__ dst, const float* __restrict__ src, int width, int n_tok,
                            int at) {
    const long long total = (long long) width * n_tok;
    const long long base = (long long) at * width;
    for (long long x = blockIdx.x * (long long) blockDim.x + threadIdx.x; x < total;
         x += (long long) gridDim.x * blockDim.x)
        dst[base + x] = (T) src[x];
}

/// Append latent and rotary key rows in the same launch.
template <class T>
__global__ void append_pair_k(T* d1_, const float* s1_, int w1,
                              T* d2_, const float* s2_, int w2, int n_tok,
                              const int32_t* at_) {
    T* STRATA_PDL_RESTRICT d1 = d1_;
    const float* STRATA_PDL_RESTRICT s1 = s1_;
    T* STRATA_PDL_RESTRICT d2 = d2_;
    const float* STRATA_PDL_RESTRICT s2 = s2_;
    const int32_t* STRATA_PDL_RESTRICT at = at_;
    pdl_trigger();
    pdl_wait();
    const long long n1 = (long long) w1 * n_tok, n2 = (long long) w2 * n_tok;
    const long long pos = *at;
    for (long long x = blockIdx.x * (long long) blockDim.x + threadIdx.x; x < n1 + n2;
         x += (long long) gridDim.x * blockDim.x) {
        if (x < n1) d1[pos * w1 + x] = (T) s1[x];
        else d2[pos * w2 + (x - n1)] = (T) s2[x - n1];
    }
}

/// Quantize 32 latent values per warp and store the scale in FP16.
__device__ __forceinline__ void q8_group(unsigned char* row, const float* src, int kv_lora, int grp,
                                         int lane) {
    const float x = src[grp * 32 + lane];
    float amax = fabsf(x);
#pragma unroll
    for (int m = 16; m; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m, 32));
    const __half dh = __float2half_rn(amax / 127.0f);
    const float d = __half2float(dh);
    const int q = d > 0.f ? max(-127, min(127, __float2int_rn(x / d))) : 0;
    reinterpret_cast<int8_t*>(row)[grp * 32 + lane] = (int8_t) q;
    if (lane == 0) reinterpret_cast<__half*>(row + kv_lora)[grp] = dh;
}

/// The Q8_0 form of `append_pair_k`: the latent quantised a warp a group, the rope key converted to F16 a
/// lane an element, in one launch. `at` is on the device when `d_at` is given, the host value otherwise.
__global__ void append_q8_pair_k(unsigned char* d1_, const float* s1_, int kv_lora, __half* d2_,
                                 const float* s2_, int n_rot, int n_tok, const int32_t* d_at_, int h_at) {
    unsigned char* STRATA_PDL_RESTRICT d1 = d1_;
    const float* STRATA_PDL_RESTRICT s1 = s1_;
    __half* STRATA_PDL_RESTRICT d2 = d2_;
    const float* STRATA_PDL_RESTRICT s2 = s2_;
    const int32_t* STRATA_PDL_RESTRICT d_at = d_at_;
    pdl_trigger();
    pdl_wait();
    const long long pos = d_at != nullptr ? (long long) *d_at : (long long) h_at;
    const int groups = kv_lora / 32;
    const std::size_t row_bytes = (std::size_t) kv_lora + (std::size_t) kv_lora / 16;   // kv_cache_lat_row_bytes
    const long long n_q = (long long) groups * n_tok;                    // warps of latent
    const long long n_pe = ((long long) n_rot * n_tok + 31) / 32;        // warps of rope key
    const int lane = (int) (threadIdx.x & 31);
    const long long warps = (long long) gridDim.x * (blockDim.x / 32);
    for (long long w = (long long) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; w < n_q + n_pe;
         w += warps) {
        if (w < n_q) {
            const int t = (int) (w / groups), grp = (int) (w % groups);
            q8_group(d1 + (std::size_t) (pos + t) * row_bytes, s1 + (long long) t * kv_lora, kv_lora, grp,
                     lane);
        } else {
            const long long x = (w - n_q) * 32 + lane;
            if (x < (long long) n_rot * n_tok) d2[pos * n_rot + x] = __float2half_rn(s2[x]);
        }
    }
}

}  // namespace

bool kv_cache_append_dev(KvCache& c, int block, const int32_t* d_at, const float* kv_cmpr,
                         const float* k_pe, int n_tok, std::string& err, void* stream) {
    if (block < 0 || block >= c.n_blocks) {
        err = "kv_cache_append_dev: block " + std::to_string(block) + " of " + std::to_string(c.n_blocks);
        return false;
    }
    // No bound check here: the position is on the device. The caller checks it against max_tokens before
    // the pass, which is where it can still refuse rather than wrap.
    const int threads = 256;
    const auto blocks = [&](long long n) {
        long long b = (n + threads - 1) / threads;
        return (unsigned) (b < 1 ? 1 : (b > 65535 ? 65535 : b));
    };
    const auto st = static_cast<cudaStream_t>(stream);
    const long long both = (long long) (c.kv_lora + c.n_rot) * n_tok;
    if (c.q8) {
        const long long warps = (long long) c.kv_lora / 32 * n_tok + ((long long) c.n_rot * n_tok + 31) / 32;
        launch_pdl(append_q8_pair_k, blocks(warps * 32), threads, 0, st,
                   static_cast<unsigned char*>(c.kv_cmpr[(std::size_t) block]), kv_cmpr, c.kv_lora,
                   static_cast<__half*>(c.k_pe[(std::size_t) block]), k_pe, c.n_rot, n_tok, d_at, 0);
    } else if (c.fp16) {
        launch_pdl(append_pair_k<__half>, blocks(both), threads, 0, st,
                   static_cast<__half*>(c.kv_cmpr[(std::size_t) block]), kv_cmpr, c.kv_lora,
                   static_cast<__half*>(c.k_pe[(std::size_t) block]), k_pe, c.n_rot, n_tok, d_at);
    } else {
        launch_pdl(append_pair_k<float>, blocks(both), threads, 0, st,
                   static_cast<float*>(c.kv_cmpr[(std::size_t) block]), kv_cmpr, c.kv_lora,
                   static_cast<float*>(c.k_pe[(std::size_t) block]), k_pe, c.n_rot, n_tok, d_at);
    }
    if (cudaGetLastError() != cudaSuccess) {
        err = "kv_cache_append_dev: the copy failed at block " + std::to_string(block);
        return false;
    }
    return true;
}

bool kv_cache_append(KvCache& c, int block, int at, const float* kv_cmpr, const float* k_pe, int n_tok,
                     std::string& err, void* stream) {
    if (block < 0 || block >= c.n_blocks) {
        err = "kv_cache_append: block " + std::to_string(block) + " of " + std::to_string(c.n_blocks);
        return false;
    }
    if (at < 0 || n_tok < 0 || at + n_tok > c.max_tokens) {
        // Refused rather than wrapped: this cache has no eviction, so running off the end means the caller
        // asked for a longer context than it reserved, and silently overwriting position 0 would be worse.
        err = "kv_cache_append: " + std::to_string(at) + " + " + std::to_string(n_tok) + " exceeds the " +
              std::to_string(c.max_tokens) + " positions reserved";
        return false;
    }
    // Both tensors are [width, n_tok] contiguous and the cache rows are [width, max_tokens], so the
    // destination is the slice starting at `at`. A kernel rather than a memcpy, because F16 rows have to be
    // converted and one path is better than two.
    const int threads = 256;
    const auto blocks = [&](long long n) {
        long long b = (n + threads - 1) / threads;
        return (unsigned) (b < 1 ? 1 : (b > 65535 ? 65535 : b));
    };
    const auto st = static_cast<cudaStream_t>(stream);
    if (c.q8) {
        const long long warps = (long long) c.kv_lora / 32 * n_tok + ((long long) c.n_rot * n_tok + 31) / 32;
        append_q8_pair_k<<<blocks(warps * 32), threads, 0, st>>>(
            static_cast<unsigned char*>(c.kv_cmpr[(std::size_t) block]), kv_cmpr, c.kv_lora,
            static_cast<__half*>(c.k_pe[(std::size_t) block]), k_pe, c.n_rot, n_tok, nullptr, at);
    } else if (c.fp16) {
        append_at_k<__half><<<blocks((long long) c.kv_lora * n_tok), threads, 0, st>>>(
            static_cast<__half*>(c.kv_cmpr[(std::size_t) block]), kv_cmpr, c.kv_lora, n_tok, at);
        append_at_k<__half><<<blocks((long long) c.n_rot * n_tok), threads, 0, st>>>(
            static_cast<__half*>(c.k_pe[(std::size_t) block]), k_pe, c.n_rot, n_tok, at);
    } else {
        append_at_k<float><<<blocks((long long) c.kv_lora * n_tok), threads, 0, st>>>(
            static_cast<float*>(c.kv_cmpr[(std::size_t) block]), kv_cmpr, c.kv_lora, n_tok, at);
        append_at_k<float><<<blocks((long long) c.n_rot * n_tok), threads, 0, st>>>(
            static_cast<float*>(c.k_pe[(std::size_t) block]), k_pe, c.n_rot, n_tok, at);
    }
    if (cudaGetLastError() != cudaSuccess) {
        err = "kv_cache_append: the copy failed at block " + std::to_string(block);
        return false;
    }
    return true;
}

}  // namespace strata::hy4_flash
