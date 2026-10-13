// Exercise HIP BLAS and the GR SiLU kernel on a persistent thread per device, then return to device 0.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {
constexpr int kTokens = 8192;
constexpr int kN = 320;
constexpr int kK = 10240;

std::string base_arch(const char* name) {
    std::string arch(name);
    const size_t suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);
    return arch;
}

bool compiled_for(const std::string& arch) {
    std::string list = STRATA_HIP_ARCHS;
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t end = list.find(',', begin);
        if (list.substr(begin, end == std::string::npos ? end : end - begin) == arch) return true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return false;
}

struct DeviceState {
    int ordinal = -1;
    std::string arch;
    cudaStream_t stream = nullptr;
    uint16_t* x = nullptr;
    uint16_t* w = nullptr;
    float* y = nullptr;
    uint16_t* y16 = nullptr;
    std::unique_ptr<strata::prefill::Gemm> gemm;
};

bool init_buffers(DeviceState& d, int ordinal) {
    cudaDeviceProp prop{};
    if (cudaSetDevice(ordinal) != cudaSuccess || cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) return false;
    d.ordinal = ordinal;
    d.arch = base_arch(prop.gcnArchName);
    if (cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&d.x, (size_t) kTokens * kK * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d.w, (size_t) kN * kK * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d.y, (size_t) kTokens * kN * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d.y16, (size_t) kTokens * kN * sizeof(uint16_t)) != cudaSuccess)
        return false;
    if (cudaMemsetAsync(d.x, 0, (size_t) kTokens * kK * sizeof(uint16_t), d.stream) != cudaSuccess ||
        cudaMemsetAsync(d.w, 0, (size_t) kN * kK * sizeof(uint16_t), d.stream) != cudaSuccess ||
        cudaStreamSynchronize(d.stream) != cudaSuccess)
        return false;
    return true;
}

bool enable_prompt_f16_opt_in() {
#if defined(_WIN32)
    return _putenv_s("STRATA_HIP_PROMPT_F16", "1") == 0;
#else
    return setenv("STRATA_HIP_PROMPT_F16", "1", 1) == 0;
#endif
}

bool init_gemm(DeviceState& d) {
    if (cudaSetDevice(d.ordinal) != cudaSuccess) return false;
    const bool f16_io = strata::prefill::prompt_f16();
    const bool expected_f16 = d.arch.compare(0, 6, "gfx103") == 0;
    if (f16_io != expected_f16) {
        std::fprintf(stderr, "device %d (%s): STRATA_HIP_PROMPT_F16=1 selected %s; expected %s\n", d.ordinal,
                     d.arch.c_str(), f16_io ? "FP16" : "BF16", expected_f16 ? "FP16" : "BF16");
        return false;
    }
    strata::prefill::set_act_f16(f16_io);
    std::printf("device %d (%s): effective prompt GEMM mode %s\n", d.ordinal, d.arch.c_str(),
                f16_io ? "FP16-in/FP16-out" : "BF16-in/FP32-out");
    d.gemm = std::make_unique<strata::prefill::Gemm>();
    std::string err;
    if (!d.gemm->init(d.stream, (int64_t) kN * kK, err)) {
        std::fprintf(stderr, "device %d Gemm::init: %s\n", d.ordinal, err.c_str());
        return false;
    }
    d.gemm->set_f16_io(f16_io);
    return true;
}

bool step(DeviceState& d, int count, const char* phase) {
    if (cudaSetDevice(d.ordinal) != cudaSuccess) return false;
    int current_device = -1, stream_device = -1;
    if (hipGetDevice(&current_device) != hipSuccess || hipStreamGetDevice(d.stream, &stream_device) != hipSuccess ||
        current_device != d.ordinal || stream_device != d.ordinal) {
        std::fprintf(stderr, "device %d (%s), %s: device/stream mismatch (current %d, stream %d)\n", d.ordinal,
                     d.arch.c_str(), phase, current_device, stream_device);
        return false;
    }
    const bool verify_result = std::strcmp(phase, "return-to-first-device chunk") == 0;
    float sentinel[8];
    for (float& v : sentinel) v = 1.0f;
    if (verify_result && cudaMemcpyAsync(d.y, sentinel, sizeof(sentinel), cudaMemcpyHostToDevice, d.stream) != cudaSuccess)
        return false;
    for (int i = 0; i < count; ++i) {
        d.gemm->bf16(d.x, d.w, d.y, kTokens, kN, kK);
        const cudaError_t gemm_error = cudaStreamSynchronize(d.stream);
        if (gemm_error != cudaSuccess) {
            std::fprintf(stderr, "device %d (%s), stream %d, %s iteration %d: HIP BLAS stream completion error %d (%s)\n",
                         d.ordinal, d.arch.c_str(), stream_device, phase, i, (int) gemm_error,
                         cudaGetErrorString(gemm_error));
            return false;
        }
        strata::prefill::gr_silu(d.y, d.y16, kTokens, d.stream);
        const cudaError_t e = cudaStreamSynchronize(d.stream);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "device %d (%s), stream %d, %s iteration %d: GR kernel completion error %d (%s)\n",
                         d.ordinal, d.arch.c_str(), stream_device, phase, i, (int) e, cudaGetErrorString(e));
            return false;
        }
        if (verify_result) {
            float result[8]{};
            if (cudaMemcpy(result, d.y, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            for (float v : result) {
                if (std::fabs(v) > 1e-5f) {
                    std::fprintf(stderr, "device %d (%s): GEMM output after device switch is %g, expected 0\n",
                                 d.ordinal, d.arch.c_str(), (double) v);
                    return false;
                }
            }
        }
    }
    std::printf("device %d (%s): %s (%d GEMM/GR steps) PASS\n", d.ordinal, d.arch.c_str(), phase, count);
    return true;
}

void release(DeviceState& d) {
    if (d.ordinal < 0) return;
    (void) cudaSetDevice(d.ordinal);
    if (d.stream) (void) cudaStreamSynchronize(d.stream);
    d.gemm.reset();
    if (d.x) (void) cudaFree(d.x);
    if (d.w) (void) cudaFree(d.w);
    if (d.y) (void) cudaFree(d.y);
    if (d.y16) (void) cudaFree(d.y16);
    if (d.stream) (void) cudaStreamDestroy(d.stream);
}
} // namespace

int main() {
    if (!enable_prompt_f16_opt_in()) {
        std::puts("could not set STRATA_HIP_PROMPT_F16=1 for the per-device regression");
        return 1;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: fewer than two visible HIP devices");
        return 77;
    }
    int first_device = -1, second_device = -1;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp a{};
        if (cudaGetDeviceProperties(&a, i) != cudaSuccess) continue;
        const std::string arch_a = base_arch(a.gcnArchName);
        if (!compiled_for(arch_a)) continue;
        for (int j = 0; j < count; ++j) {
            cudaDeviceProp b{};
            if (j == i || cudaGetDeviceProperties(&b, j) != cudaSuccess) continue;
            const std::string arch_b = base_arch(b.gcnArchName);
            if (compiled_for(arch_b) && arch_a.compare(0, 6, "gfx103") != 0 &&
                arch_b.compare(0, 6, "gfx103") == 0) {
                first_device = i;
                second_device = j;
                break;
            }
        }
        if (first_device >= 0) break;
    }
    if (first_device < 0) {
        std::puts("SKIP: need one compiled non-gfx103x GPU and one compiled gfx103x GPU");
        return 77;
    }
    DeviceState first, second;
    if (!init_buffers(first, first_device) || !init_buffers(second, second_device)) {
        std::puts("could not initialize the mixed-architecture HIP devices");
        release(second);
        release(first);
        return 1;
    }

    std::mutex mutex;
    std::condition_variable cv;
    bool first_done = false, second_done = false;
    bool first_ok = false, second_ok = false, return_ok = false;
    std::thread first_thread([&] {
        first_ok = init_gemm(first) && step(first, 43, "first stage");
        std::unique_lock<std::mutex> lock(mutex);
        first_done = true;
        cv.notify_all();
        cv.wait(lock, [&] { return second_done; });
        lock.unlock();
        if (first_ok) return_ok = step(first, 1, "return-to-first-device chunk");
        first.gemm.reset();
    });
    std::thread second_thread([&] {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return first_done; });
        }
        second_ok = init_gemm(second) && step(second, 5, "second stage");
        std::lock_guard<std::mutex> lock(mutex);
        second_done = true;
        cv.notify_all();
        second.gemm.reset();
    });
    first_thread.join();
    second_thread.join();
    const bool ok = first_ok && second_ok && return_ok;
    release(second);
    release(first);
    return ok ? 0 : 1;
}
