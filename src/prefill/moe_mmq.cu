// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::mmq {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// ... plus, in the same launch, the expert's 16-byte NVFP4 tail (tail -> tail_dst) and the zeroed MMQ tails after
// the group's last slot (nz uint4 after ab_dst's and c_dst's copies): one launch per expert instead of a kernel,
// a 16-byte copy and, per group, two memsets.
__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst,
                              const uint4* __restrict__ tail, uint4* __restrict__ tail_dst, int64_t nz) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = na + nb + nc;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < n) c_dst[i - na - nb] = c[i - na - nb];
    else if (i < n + nz) ab_dst[na + nb + (i - n)] = make_uint4(0, 0, 0, 0);
    else if (i < n + 2 * nz) c_dst[nc + (i - n - nz)] = make_uint4(0, 0, 0, 0);
    else if (i == n + 2 * nz && tail != nullptr) *tail_dst = *tail;
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
__global__ void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const uint4 q = *(const uint4*) codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

// swiglu_kernel on the MMQ gate/up output with each expert's gate and up scales applied as it is read: the products
// scale_gu_rows stored, without the pass that stored them
__global__ void swiglu_scaled_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                                     const int32_t* __restrict__ bounds, int n, const float* __restrict__ tails,
                                     int64_t row0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff, ra = row0 + r;
    int q = 0;
    while (q + 1 < n && ra >= bounds[q + 1]) ++q;
    const float* row = gu + r * 2 * n_ff;
    const float g = row[k] * tails[4 * q], u = row[n_ff + k] * tails[4 * q + 1];
    h[i] = g / (1.0f + __expf(-g)) * u;
}
__global__ void down_row_scales_kernel(float* __restrict__ sd, const int32_t* __restrict__ bounds, int n,
                                       const float* __restrict__ tails, int64_t nrows) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    sd[r] = tails[4 * q + 2];
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

// mmq_nvfp4_w4a8.cu: mmq.cuh's int8 NVFP4 path, compiled for this GPU with Blackwell's FP4 MMA hidden
void run_nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s);
// mmq_nvfp4_w4a4.cu (the one unit built for 12xa): FP4 x FP4 and its activation quantizer, if this card runs it
bool w4a4_available();
void run_nvfp4_w4a4(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s);
void quantize_nvfp4_w4a4(const float* x, const int32_t* ids, void* xq, float* yscale, bool aligned, int64_t cols,
                         int64_t ld, int64_t rows, int64_t padded, cudaStream_t s);

bool built() { return true; }

Nvfp4Mode nvfp4_mode() {
    static const Nvfp4Mode m = [] {
        const char* e = std::getenv("STRATA_PREFILL_NVFP4");
        if (e == nullptr || std::strcmp(e, "w4a8") == 0) return Nvfp4Mode::W4A8;
        if (std::strcmp(e, "w4a4") == 0) return Nvfp4Mode::W4A4;
        if (std::strcmp(e, "fp16") == 0) return Nvfp4Mode::FP16;
        std::fprintf(stderr, "STRATA_PREFILL_NVFP4=%s: expected w4a8, w4a4 or fp16\n", e);
        std::exit(1);
    }();
    return m;
}

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0:
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0:   // #296 (Q4_K and Q5_1: STRATA_MMQ_KQUANTS)
#endif
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1:   // Unsloth's UD-Q4_K_XL experts (CUDA)
#endif
            return true;
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_NVFP4:
            return nvfp4_mode() != Nvfp4Mode::FP16;
#endif
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    if (!supported(t)) return false;
    // mul_mat_q_case's choice: the "fallback" configs when the rows are not a multiple of 128; then
    // mul_mat_q_switch_J's loop - a tile size whose config exists for this card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (t >= 0 && t < GGML_TYPE_COUNT && !said[t]) {
                said[t] = true;
                std::fprintf(stderr, "strata: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path "
                                     "(#420)\n", ggml_type_name((ggml_type) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

namespace {
__global__ void scale_gu_rows_kernel(float* __restrict__ gu, int64_t ld, int64_t n_ff, const int32_t* __restrict__ bounds,
                                     int n, const float* __restrict__ tails, int64_t row0) {
    const int64_t r = row0 + blockIdx.x;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    const float sg = tails[4 * q], su = tails[4 * q + 1];
    float* row = gu + r * ld;
    for (int64_t k = threadIdx.x; k < 2 * n_ff; k += blockDim.x) row[k] *= k < n_ff ? sg : su;
}
__global__ void scale_down_rows_kernel(float* __restrict__ d, int64_t ld, const int32_t* __restrict__ bounds, int n,
                                       const float* __restrict__ tails) {
    const int64_t r = blockIdx.x;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    const float sd = tails[4 * q + 2];
    float* row = d + r * ld;
    for (int64_t k = threadIdx.x; k < ld; k += blockDim.x) row[k] *= sd;
}
}  // namespace

bool fp4_activations(int t) {
#if defined(GGML_USE_HIP)   // no NVFP4 MMQ on HIP (and no W4A4 unit)
    (void) t;
    return false;
#else
    if (t != GGML_TYPE_NVFP4 || nvfp4_mode() != Nvfp4Mode::W4A4) return false;
    static const bool on = [] {
        const bool ok = w4a4_available();
        if (!ok) std::fprintf(stderr, "prefill mmq: STRATA_PREFILL_NVFP4=w4a4 needs an sm_12x card in a 12x build; W4A8\n");
        return ok;
    }();
    return on;
#endif
}

void scale_gu_rows(float* gu, int64_t ld, int64_t n_ff, const int32_t* bounds, int n, const float* tails,
                   int64_t row0, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    scale_gu_rows_kernel<<<(unsigned) nrows, 256, 0, (cudaStream_t) stream>>>(gu, ld, n_ff, bounds, n, tails, row0);
    ck(cudaGetLastError(), "scale_gu_rows");
}

void scale_down_rows(float* d, int64_t ld, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    scale_down_rows_kernel<<<(unsigned) nrows, 256, 0, (cudaStream_t) stream>>>(d, ld, bounds, n, tails);
    ck(cudaGetLastError(), "scale_down_rows");
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream,
              float* yscale) {
    if (rows <= 0) return;
#if !defined(GGML_USE_HIP)
    if (fp4_activations(t)) {
        if (yscale == nullptr) { std::fprintf(stderr, "prefill mmq: NVFP4 activations need a scale buffer\n"); std::exit(1); }
        const bool aligned = ((uintptr_t) x % 32 == 0) && ((size_t) ld * sizeof(float)) % 32 == 0;
        quantize_nvfp4_w4a4(x, ids, xq, yscale, aligned, cols, ld, rows, pad512(cols), (cudaStream_t) stream);
        ck(cudaGetLastError(), "quantize");
        return;
    }
#endif
    (void) yscale;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, p.y_scale,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
#endif
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
#endif
#if !defined(GGML_USE_HIP)   // NVFP4: CUDA only (no HIP instance, no W4A8 unit)
        case GGML_TYPE_NVFP4:
            if (fp4_activations(t)) run_nvfp4_w4a4(ctx, a, s);
            else run_nvfp4_w4a8(ctx, a, s);
            break;
#endif
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail, void* tail_dst, size_t zero_bytes) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      (uintptr_t) tail | (uintptr_t) tail_dst | gu_half_bytes | d_bytes | zero_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16, nz = (int64_t) zero_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc + 2 * nz + 1), 256, 0, s>>>(
            (const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst, (const uint4*) down, nc, (uint4*) d_dst,
            (const uint4*) tail, (uint4*) tail_dst, nz);
    } else {
        if (tail) cudaMemcpyAsync(tail_dst, tail, 16, cudaMemcpyDeviceToDevice, s);
        if (zero_bytes) {
            cudaMemsetAsync((uint8_t*) gu_dst + 2 * gu_half_bytes, 0, zero_bytes, s);
            cudaMemsetAsync((uint8_t*) d_dst + d_bytes, 0, zero_bytes, s);
        }
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks(1280LL * 40 + 2560LL * 10), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds, int n,
                   const float* tails, int64_t row0, void* stream) {
    if (rows <= 0) return;
    swiglu_scaled_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, bounds, n, tails, row0);
    ck(cudaGetLastError(), "swiglu_scaled");
}
void down_row_scales(float* sd, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    down_row_scales_kernel<<<blocks(nrows), 256, 0, (cudaStream_t) stream>>>(sd, bounds, n, tails, nrows);
    ck(cudaGetLastError(), "down_row_scales");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

}  // namespace strata::prefill::mmq
