// Polymorphic dequantization + cuBLAS Tensor Core prefill for strata-q38.
#include "prefill_gemm.cuh"
#include "spin8_triality.cuh"
#include "ops.cuh"
#include "iq_tables.hpp"
#include "iq_decode.hpp"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen38 {
namespace {

inline void ck_cublas(cublasStatus_t st) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(st)));
    }
}

__device__ __forceinline__ float load_fp16_le(const uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8)));
}

__device__ __forceinline__ float dequant_elem(int type, const uint8_t* p, int i) {
    if (type == 0)  return reinterpret_cast<const float*>(p)[i];
    if (type == 1)  return __half2float(reinterpret_cast<const half*>(p)[i]);
    if (type == 30) return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(p)[i]) << 16);
    if (type == 142) {
        p += static_cast<size_t>(i / 128) * 34;
        i %= 128;
        return load_fp16_le(p) * static_cast<float>(static_cast<int>((p[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
    }
    if (type == 8) {
        p += static_cast<size_t>(i / 32) * 34;
        i %= 32;
        return load_fp16_le(p) * static_cast<float>(reinterpret_cast<const int8_t*>(p + 2)[i]);
    }
    if (type == 21 || type == 22) {
        p += static_cast<size_t>(i / 256) * (type == 21 ? 110 : 82);
        i %= 256;
        return iq_value(type, p, i, load_fp16_le(p), q38_iq2s_grid, q38_iq3s_grid);
    }
    if (type == 23) {
        p += static_cast<size_t>(i / 256) * 136;
        i %= 256;
        const int b = i / 32, j = i % 32;
        const int scale = ((p[4 + b / 2] >> (4 * (b % 2))) & 15) |
                          ((((static_cast<uint16_t>(p[2]) | (static_cast<uint16_t>(p[3]) << 8)) >> (2 * b)) & 3) << 4);
        constexpr int kTable[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        const int code = (p[8 + 16 * b + j % 16] >> (j >= 16 ? 4 : 0)) & 15;
        return load_fp16_le(p) * static_cast<float>(scale - 32) * static_cast<float>(kTable[code]);
    }
    if (type == 12 || type == 13) {
        const bool five = (type == 13);
        p += static_cast<size_t>(i / 256) * (five ? 176 : 144);
        i %= 256;
        const int b = i / 32, j = i % 32;
        const uint8_t* sc = p + 4;
        int d = 0, m = 0;
        if (b < 4) {
            d = sc[b] & 63;
            m = sc[b + 4] & 63;
        } else {
            d = (sc[b + 4] & 15) | ((sc[b - 4] >> 6) << 4);
            m = (sc[b + 4] >> 4) | ((sc[b] >> 6) << 4);
        }
        const uint8_t* qs = p + (five ? 48 : 16);
        int code = (qs[(b / 2) * 32 + j] >> (4 * (b % 2))) & 15;
        if (five) code |= ((p[16 + j] >> b) & 1) << 4;
        return load_fp16_le(p) * static_cast<float>(d * code) - load_fp16_le(p + 2) * static_cast<float>(m);
    }
    // Q6_K (type == 14)
    p += static_cast<size_t>(i / 256) * 210;
    i %= 256;
    const int ip = i / 128, j = i % 32, group = (i % 128) / 32;
    const int lo = (p[ip * 64 + (group % 2) * 32 + j] >> (group >= 2 ? 4 : 0)) & 15;
    const int hi = (p[128 + ip * 32 + j] >> (2 * group)) & 3;
    const int scale = reinterpret_cast<const int8_t*>(p + 192)[ip * 8 + group * 2 + j / 16];
    return load_fp16_le(p + 208) * static_cast<float>(scale) * static_cast<float>((lo | (hi << 4)) - 32);
}

template <int TYPE>
__global__ void __launch_bounds__(256)
dequant_tile_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst,
                    int tile_rows, int K, size_t row_stride) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= tile_rows || col >= K) return;
    const uint8_t* row_ptr = src + static_cast<size_t>(row) * row_stride;
    dst[static_cast<size_t>(row) * K + col] = __float2half(dequant_elem(TYPE, row_ptr, col));
}

__global__ void __launch_bounds__(256)
f32_to_fp16_kernel(const float* __restrict__ src, half* __restrict__ dst, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void launch_dequant_tile(int type, const uint8_t* src, half* dst, int tile_rows, int K,
                         size_t row_stride, cudaStream_t stream) {
    const dim3 grid((K + 255) / 256, tile_rows);
    const dim3 block(256);
    switch (type) {
        case 0:   dequant_tile_kernel<0><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 1:   dequant_tile_kernel<1><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 8:   dequant_tile_kernel<8><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 12:  dequant_tile_kernel<12><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 13:  dequant_tile_kernel<13><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 14:  dequant_tile_kernel<14><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 21:  dequant_tile_kernel<21><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 22:  dequant_tile_kernel<22><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 23:  dequant_tile_kernel<23><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 30:  dequant_tile_kernel<30><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 142: dequant_tile_kernel<142><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        default:  throw std::runtime_error("unsupported matrix type in prefill_gemm");
    }
    ck(cudaGetLastError());
}

} // namespace

struct LtPlanEntry {
    int tile_rows = 0;
    int n_tokens = 0;
    int K = 0;
    int N = 0;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    bool valid = false;
};

struct PrefillGemmWorkspace::Impl {
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    half* tile_buf[2] = {nullptr, nullptr};
    size_t tile_buf_bytes = 0;
    half* act_fp16 = nullptr;
    size_t act_elems = 0;
    __nv_fp8_e4m3* act_fp8 = nullptr;
    __nv_fp8_e4m3* w_fp8 = nullptr;
    float* act_scales = nullptr;
    float* w_scales = nullptr;
    void* lt_workspace = nullptr;
    size_t lt_workspace_bytes = 4ull * 1024 * 1024;
    bool use_fp8 = false;
    cudaStream_t dequant_stream = nullptr;
    cudaEvent_t tile_ready[2] = {nullptr, nullptr};
    cudaEvent_t tile_done[2] = {nullptr, nullptr};
    cudaEvent_t act_ready = nullptr;
    PrefillGemmStats stats{};
    std::vector<LtPlanEntry> lt_plans;
};

__global__ void __launch_bounds__(256)
scale_fp8_rows_kernel(float* __restrict__ y, const float* __restrict__ x_sc,
                      const float* __restrict__ w_sc, int n_tokens, int rows, int row0, int tile_rows) {
    const int tok = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= n_tokens || c >= tile_rows) return;
    y[static_cast<size_t>(tok) * rows + row0 + c] *= (x_sc[tok] * w_sc[c]);
}

PrefillGemmWorkspace::~PrefillGemmWorkspace() {
    release();
}

size_t PrefillGemmWorkspace::estimate_bytes(int max_tokens, int max_k, size_t tile_buf_bytes) {
    if (max_tokens <= 0 || max_k <= 0 || tile_buf_bytes < static_cast<size_t>(max_k) * sizeof(half)) {
        return 0;
    }
    const size_t act_elems = static_cast<size_t>(max_tokens) * static_cast<size_t>(max_k);
    const size_t max_tile_rows_est = (tile_buf_bytes / 2) / 512 + 1024;
    constexpr size_t lt_workspace_bytes = 4ull * 1024 * 1024;
    return tile_buf_bytes * 2 + act_elems * (sizeof(half) + sizeof(__nv_fp8_e4m3))
         + tile_buf_bytes / 2 + lt_workspace_bytes
         + (static_cast<size_t>(max_tokens) + max_tile_rows_est) * sizeof(float);
}

int PrefillGemmWorkspace::cublas_version() {
    cublasHandle_t h = nullptr;
    int ver = 0;
    if (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) {
        cublasGetVersion(h, &ver);
        cublasDestroy(h);
    }
    return ver;
}

PrefillGemmStats PrefillGemmWorkspace::stats() const {
    return impl_ ? impl_->stats : PrefillGemmStats{};
}

void PrefillGemmWorkspace::reset_stats() {
    if (impl_) impl_->stats = PrefillGemmStats{};
}

void PrefillGemmWorkspace::init(int max_tokens, int max_k, size_t tile_buf_bytes) {
    release();
    if (max_tokens <= 0 || max_k <= 0 || tile_buf_bytes < static_cast<size_t>(max_k) * sizeof(half)) {
        throw std::runtime_error("invalid PrefillGemmWorkspace dimensions");
    }
    impl_ = new Impl();
    try {
        ck_cublas(cublasCreate(&impl_->cublas));
        ck_cublas(cublasSetMathMode(impl_->cublas, CUBLAS_DEFAULT_MATH));
        ck_cublas(cublasLtCreate(&impl_->cublas_lt));
        impl_->tile_buf_bytes = tile_buf_bytes;
        for (int i = 0; i < 2; ++i) {
            ck(cudaMalloc(&impl_->tile_buf[i], tile_buf_bytes));
        }
        impl_->act_elems = static_cast<size_t>(max_tokens) * static_cast<size_t>(max_k);
        ck(cudaMalloc(&impl_->act_fp16, impl_->act_elems * sizeof(half)));
        // FP8 Spin(8) Hadamard Tensor Core staging (shares tile_buf_bytes / 2 for FP8 weights)
        ck(cudaMalloc(&impl_->act_fp8, impl_->act_elems * sizeof(__nv_fp8_e4m3)));
        ck(cudaMalloc(&impl_->w_fp8, tile_buf_bytes / 2));
        ck(cudaMalloc(&impl_->act_scales, static_cast<size_t>(max_tokens) * sizeof(float)));
        const size_t max_tile_rows_est = (tile_buf_bytes / 2) / 512 + 1024;
        ck(cudaMalloc(&impl_->w_scales, max_tile_rows_est * sizeof(float)));
        ck(cudaMalloc(&impl_->lt_workspace, impl_->lt_workspace_bytes));
        ck(cudaStreamCreateWithFlags(&impl_->dequant_stream, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) {
            ck(cudaEventCreateWithFlags(&impl_->tile_ready[i], cudaEventDisableTiming));
            ck(cudaEventCreateWithFlags(&impl_->tile_done[i], cudaEventDisableTiming));
        }
        ck(cudaEventCreateWithFlags(&impl_->act_ready, cudaEventDisableTiming));
        const char* fp8_env = std::getenv("STRATA_FP8_PREFILL");
        impl_->use_fp8 = (fp8_env && fp8_env[0] == '1');
        total_bytes_ = estimate_bytes(max_tokens, max_k, tile_buf_bytes);
    } catch (...) {
        release();
        throw;
    }
}

void PrefillGemmWorkspace::release() {
    if (!impl_) return;
    for (auto& plan : impl_->lt_plans) {
        if (plan.c_desc) cublasLtMatrixLayoutDestroy(plan.c_desc);
        if (plan.b_desc) cublasLtMatrixLayoutDestroy(plan.b_desc);
        if (plan.a_desc) cublasLtMatrixLayoutDestroy(plan.a_desc);
        if (plan.desc) cublasLtMatmulDescDestroy(plan.desc);
    }
    impl_->lt_plans.clear();
    if (impl_->act_ready) cudaEventDestroy(impl_->act_ready);
    for (int i = 0; i < 2; ++i) {
        if (impl_->tile_done[i]) cudaEventDestroy(impl_->tile_done[i]);
        if (impl_->tile_ready[i]) cudaEventDestroy(impl_->tile_ready[i]);
    }
    if (impl_->dequant_stream) cudaStreamDestroy(impl_->dequant_stream);
    if (impl_->lt_workspace) cudaFree(impl_->lt_workspace);
    if (impl_->w_scales) cudaFree(impl_->w_scales);
    if (impl_->act_scales) cudaFree(impl_->act_scales);
    if (impl_->w_fp8) cudaFree(impl_->w_fp8);
    if (impl_->act_fp8) cudaFree(impl_->act_fp8);
    if (impl_->act_fp16) cudaFree(impl_->act_fp16);
    for (int i = 0; i < 2; ++i) {
        if (impl_->tile_buf[i]) cudaFree(impl_->tile_buf[i]);
    }
    if (impl_->cublas_lt) cublasLtDestroy(impl_->cublas_lt);
    if (impl_->cublas) cublasDestroy(impl_->cublas);
    delete impl_;
    impl_ = nullptr;
    total_bytes_ = 0;
}

void dequant_rows_fp16(const StrataQuantTensor& weight, int row_start, int tile_rows,
                       half* dst_fp16, cudaStream_t stream) {
    if (!weight.d_data || !dst_fp16 || row_start < 0 || tile_rows <= 0 || row_start + tile_rows > weight.ne1) {
        throw std::runtime_error("invalid dequant_rows_fp16 arguments");
    }
    const auto* src = static_cast<const uint8_t*>(weight.d_data) + static_cast<size_t>(row_start) * weight.row_bytes;
    launch_dequant_tile(static_cast<int>(weight.type), src, dst_fp16, tile_rows, weight.ne0, weight.row_bytes, stream);
}

void PrefillGemmWorkspace::matmul_batch(const StrataQuantTensor& weight, const float* x_fp32, float* y_fp32,
                                        int n_tokens, cudaStream_t stream) {
    if (!weight.d_data || !x_fp32 || !y_fp32 || n_tokens <= 0 || weight.ne0 <= 0 || weight.ne1 <= 0) {
        throw std::runtime_error("invalid matmul_batch arguments");
    }
    const int K = weight.ne0;
    const int N = weight.ne1;
    if (n_tokens < 4 || !impl_) {
        if (impl_) impl_->stats.matvec_fallback_calls += static_cast<uint64_t>(n_tokens);
        for (int t = 0; t < n_tokens; ++t) {
            matvec(static_cast<int>(weight.type), weight.d_data,
                   x_fp32 + static_cast<size_t>(t) * K,
                   y_fp32 + static_cast<size_t>(t) * N, K, N, stream);
        }
        return;
    }
    const size_t x_elems = static_cast<size_t>(n_tokens) * K;
    if (x_elems > impl_->act_elems) {
        throw std::runtime_error("PrefillGemmWorkspace activation buffer too small");
    }
    f32_to_fp16_kernel<<<static_cast<unsigned>((x_elems + 255) / 256), 256, 0, stream>>>(x_fp32, impl_->act_fp16, x_elems);
    ck(cudaGetLastError());

    const bool fp8_eligible = impl_->use_fp8 && n_tokens >= 16 && (n_tokens % 16 == 0) && (K % 16 == 0);
    if (fp8_eligible) {
        spin8_fp8_hadamard_quant_kernel<<<n_tokens, 256, 0, stream>>>(
            impl_->act_fp16, impl_->act_fp8, impl_->act_scales, n_tokens, K);
        ck(cudaGetLastError());
    }

    ck_cublas(cublasSetStream(impl_->cublas, stream));
    int max_tile_rows = static_cast<int>(impl_->tile_buf_bytes / (static_cast<size_t>(K) * sizeof(half)));
    if (max_tile_rows <= 0) {
        throw std::runtime_error("PrefillGemmWorkspace tile buffer smaller than one row");
    }
    if (fp8_eligible && max_tile_rows >= 16) {
        max_tile_rows = (max_tile_rows / 16) * 16;
    }

    auto get_lt_plan = [&](int tile_rows) -> const LtPlanEntry* {
        for (const auto& entry : impl_->lt_plans) {
            if (entry.tile_rows == tile_rows && entry.n_tokens == n_tokens && entry.K == K && entry.N == N) {
                return &entry;
            }
        }
        LtPlanEntry entry{};
        entry.tile_rows = tile_rows;
        entry.n_tokens = n_tokens;
        entry.K = K;
        entry.N = N;
        cublasOperation_t op_a = CUBLAS_OP_T, op_b = CUBLAS_OP_N;
        ck_cublas(cublasLtMatmulDescCreate(&entry.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        ck_cublas(cublasLtMatmulDescSetAttribute(entry.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
        ck_cublas(cublasLtMatmulDescSetAttribute(entry.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
        ck_cublas(cublasLtMatrixLayoutCreate(&entry.a_desc, CUDA_R_8F_E4M3, K, tile_rows, K));
        ck_cublas(cublasLtMatrixLayoutCreate(&entry.b_desc, CUDA_R_8F_E4M3, K, n_tokens, K));
        ck_cublas(cublasLtMatrixLayoutCreate(&entry.c_desc, CUDA_R_32F, tile_rows, n_tokens, N));
        cublasLtMatmulPreference_t pref = nullptr;
        int found = 0;
        ck_cublas(cublasLtMatmulPreferenceCreate(&pref));
        ck_cublas(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &impl_->lt_workspace_bytes, sizeof(impl_->lt_workspace_bytes)));
        ck_cublas(cublasLtMatmulAlgoGetHeuristic(impl_->cublas_lt, entry.desc, entry.a_desc, entry.b_desc,
                                                entry.c_desc, entry.c_desc, pref, 1, &entry.heur, &found));
        cublasLtMatmulPreferenceDestroy(pref);
        entry.valid = (found > 0);
        if (!entry.valid) {
            cublasLtMatrixLayoutDestroy(entry.c_desc); entry.c_desc = nullptr;
            cublasLtMatrixLayoutDestroy(entry.b_desc); entry.b_desc = nullptr;
            cublasLtMatrixLayoutDestroy(entry.a_desc); entry.a_desc = nullptr;
            cublasLtMatmulDescDestroy(entry.desc); entry.desc = nullptr;
        }
        impl_->lt_plans.push_back(entry);
        return &impl_->lt_plans.back();
    };

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int n_tiles = (N + max_tile_rows - 1) / max_tile_rows;
    if (n_tiles > 1) {
        ck(cudaEventRecord(impl_->act_ready, stream));
        ck(cudaStreamWaitEvent(impl_->dequant_stream, impl_->act_ready, 0));
    }
    int slot = 0;
    for (int ti = 0; ti < n_tiles; ++ti) {
        const int row0 = ti * max_tile_rows;
        const int tile_rows = std::min(max_tile_rows, N - row0);
        half* w_fp16 = impl_->tile_buf[slot];
        if (n_tiles > 1) {
            if (ti >= 2) {
                ck(cudaStreamWaitEvent(impl_->dequant_stream, impl_->tile_done[slot], 0));
            }
            dequant_rows_fp16(weight, row0, tile_rows, w_fp16, impl_->dequant_stream);
            ck(cudaEventRecord(impl_->tile_ready[slot], impl_->dequant_stream));
            ck(cudaStreamWaitEvent(stream, impl_->tile_ready[slot], 0));
        } else {
            dequant_rows_fp16(weight, row0, tile_rows, w_fp16, stream);
        }
        if (fp8_eligible && (tile_rows % 16 == 0)) {
            const LtPlanEntry* plan = get_lt_plan(tile_rows);
            if (plan->valid) {
                spin8_fp8_hadamard_quant_kernel<<<tile_rows, 256, 0, stream>>>(
                    w_fp16, impl_->w_fp8, impl_->w_scales, tile_rows, K);
                ck(cudaGetLastError());
                ck_cublas(cublasLtMatmul(impl_->cublas_lt, plan->desc, &alpha, impl_->w_fp8, plan->a_desc,
                                         impl_->act_fp8, plan->b_desc, &beta, y_fp32 + row0, plan->c_desc,
                                         y_fp32 + row0, plan->c_desc, &plan->heur.algo,
                                         impl_->lt_workspace, impl_->lt_workspace_bytes, stream));
                scale_fp8_rows_kernel<<<dim3((tile_rows + 255) / 256, n_tokens), 256, 0, stream>>>(
                    y_fp32, impl_->act_scales, impl_->w_scales, n_tokens, N, row0, tile_rows);
                ck(cudaGetLastError());
                ++impl_->stats.fp8_gemm_calls;
            } else {
                ck_cublas(cublasGemmEx(
                    impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N, tile_rows, n_tokens, K, &alpha,
                    w_fp16, CUDA_R_16F, K, impl_->act_fp16, CUDA_R_16F, K, &beta,
                    y_fp32 + row0, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                ++impl_->stats.fp8_fallback_no_algo;
                ++impl_->stats.fp16_gemm_calls;
            }
        } else {
            if (fp8_eligible && (tile_rows % 16 != 0)) {
                ++impl_->stats.fp8_fallback_shape;
            }
            // Row-major Y[n_tokens, N] = X[n_tokens, K] * W_tile[tile_rows, K]^T
            // In column-major: C[tile_rows, n_tokens] = W_tile^T * X, ldc = N.
            ck_cublas(cublasGemmEx(
                impl_->cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                tile_rows, n_tokens, K,
                &alpha,
                w_fp16, CUDA_R_16F, K,
                impl_->act_fp16, CUDA_R_16F, K,
                &beta,
                y_fp32 + row0, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            ++impl_->stats.fp16_gemm_calls;
        }
        if (n_tiles > 1) {
            ck(cudaEventRecord(impl_->tile_done[slot], stream));
        }
        slot ^= 1;
    }
}

} // namespace strata::qwen38
