// BF16 projections vs a CPU double reference, including tiled output and init-time scratch failure.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t FP32_BYTES = 64u << 20;
bool fail_fp32 = false;
int fp32_allocations = 0;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void* ptr = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&ptr, bytes)); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

void parity(strata::prefill::Gemm& gemm, int t, int n, int k) {
    const int stride = n + 4;
    std::vector<uint16_t> x((size_t) t * k), w((size_t) n * k);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (uint16_t) (0x3c00 + (i * 37) % 1024);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint16_t) (0x3b00 + (i * 13) % 1024 + (i % 3 == 0 ? 0x8000 : 0));
    x[0] = 0x4780; // 65536: outside FP16's finite range.
    w[0] = 0x3780;
    Buffer dx(x.size() * 2), dw(w.size() * 2), dy((size_t) t * stride * sizeof(float));
    check(cudaMemcpy(dx.ptr, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw.ptr, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    for (float beta : {0.0f, 0.5f, 1.0f}) {
        std::vector<float> y((size_t) t * stride, beta == 0 ? std::numeric_limits<float>::quiet_NaN() : 0.25f);
        check(cudaMemcpy(dy.ptr, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice));
        gemm.bf16((uint16_t*) dx.ptr, (uint16_t*) dw.ptr, (float*) dy.ptr, t, n, k, stride, beta);
        check(cudaStreamSynchronize((cudaStream_t) gemm.stream()));
        check(cudaMemcpy(y.data(), dy.ptr, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int r = 0; r < t; ++r) {
            for (int c = 0; c < n; ++c) {
                double reference = beta * 0.25, norm = 0;
                for (int j = 0; j < k; ++j) {
                    const double product = (double) strata::kernels::f32_from_bf16(x[(size_t) r * k + j]) *
                                            strata::kernels::f32_from_bf16(w[(size_t) c * k + j]);
                    reference += product;
                    norm += std::abs(product);
                }
                const float actual = y[(size_t) r * stride + c];
                if (!std::isfinite(actual) || std::abs(actual - reference) > 1e-5 * (1 + norm))
                    throw std::runtime_error("BF16 differs from CPU double reference");
            }
            for (int c = n; c < stride; ++c) {
                const float padding = y[(size_t) r * stride + c];
                if (beta == 0 ? !std::isnan(padding) : padding != 0.25f)
                    throw std::runtime_error("output stride padding overwritten");
            }
        }
    }
}

void run(bool external, bool fail, bool pre_ampere) {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        Buffer workspace(32u << 20);
        fail_fp32 = fail;
        fp32_allocations = 0;
        {
            strata::prefill::Gemm gemm;
            std::string err;
            const bool ok = external ? gemm.init_external(stream, nullptr, 0, workspace.ptr, 32u << 20, err)
                                     : gemm.init(stream, 0, err);
            if (!ok) throw std::runtime_error(err);
#if defined(STRATA_TEST_WRAP_CUDA_MALLOC)
            const int expected = pre_ampere ? 1 : 0;
            if (fp32_allocations != expected)
                throw std::runtime_error("64 MiB scratch was not reserved exactly once at init");
#endif
            if (external) gemm.rebind(nullptr, 0, workspace.ptr, 32u << 20);
            for (const auto& shape : std::vector<std::vector<int>>{{1, 35, 47}, {17, 35, 47}, {33, 35, 47},
                                                                 {1031, 35, 47}, {3, 4103, 4097}, {17, 4, 47}})
                parity(gemm, shape[0], shape[1], shape[2]);
#if defined(STRATA_TEST_WRAP_CUDA_MALLOC)
            if (fp32_allocations != expected)
                throw std::runtime_error("BF16 allocated or retried scratch after init");
#endif
        }
        fail_fp32 = false;
        // The Gemm destructor must not free caller-owned workspace.
        check(cudaMemset(workspace.ptr, 0, 32u << 20));
        check(cudaDeviceSynchronize());
    }
    check(cudaStreamDestroy(stream));
    (void) pre_ampere;
}
}  // namespace

#if defined(STRATA_TEST_WRAP_CUDA_MALLOC)
extern "C" cudaError_t __real_cudaMalloc(void**, size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** ptr, size_t bytes) {
    if (bytes == FP32_BYTES) {
        ++fp32_allocations;
        if (fail_fp32) { *ptr = nullptr; return cudaErrorMemoryAllocation; }
    }
    return __real_cudaMalloc(ptr, bytes);
}
#endif

int main() {
    try {
        int devices = 0;
        check(cudaGetDeviceCount(&devices));
        if (devices == 0) throw std::runtime_error("BF16 GEMM test requires a CUDA GPU");
        for (int device = 0; device < devices; ++device) {
            check(cudaSetDevice(device));
            cudaDeviceProp prop{};
            check(cudaGetDeviceProperties(&prop, device));
            for (bool external : {false, true}) {
                run(external, false, prop.major == 7);
#if defined(STRATA_TEST_WRAP_CUDA_MALLOC)
                run(external, true, prop.major == 7);
#endif
            }
            std::printf("GPU %d: BF16 GEMM parity, ownership, and init-time allocation checks passed\n", device);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "BF16 GEMM test: %s\n", e.what());
        return 1;
    }
}
