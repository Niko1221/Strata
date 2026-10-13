// Check HIP prefill kernel dispatch across heterogeneous devices in one process.
// The sequence intentionally returns to device 0 after launching on device 1:
// mixed layer-split prompts switch back to an earlier stage between chunks.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int kLr = 320;
constexpr int kTokens = 8192;

bool compiled_for(const char* arch) {
    std::string list = STRATA_HIP_ARCHS;
    std::string target(arch);
    if (const size_t suffix = target.find(':'); suffix != std::string::npos) target.resize(suffix);
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t end = list.find(',', begin);
        const std::string item = list.substr(begin, end == std::string::npos ? end : end - begin);
        if (item == target) return true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return false;
}

struct DeviceBuffers {
    int ordinal = -1;
    std::string arch;
    cudaStream_t stream = nullptr;
    float* input = nullptr;
    uint16_t* output = nullptr;
};

bool setup(DeviceBuffers& d, int ordinal) {
    cudaDeviceProp prop{};
    if (cudaSetDevice(ordinal) != cudaSuccess || cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) return false;
    if (!compiled_for(prop.gcnArchName)) return true; // this executable intentionally skips unrelated visible devices
    d.ordinal = ordinal;
    d.arch = prop.gcnArchName;
    const bool f16_io = strata::prefill::prompt_f16();
    const bool expected_f16 = d.arch.compare(0, 6, "gfx103") == 0;
    if (f16_io != expected_f16) {
        std::fprintf(stderr, "HIP device %d (%s): expected prompt mode %s, got %s\n", ordinal, d.arch.c_str(),
                     expected_f16 ? "FP16" : "BF16", f16_io ? "FP16" : "BF16");
        return false;
    }
    strata::prefill::set_act_f16(f16_io);
    std::printf("HIP device %d (%s): effective prompt GEMM mode %s\n", ordinal, d.arch.c_str(),
                f16_io ? "FP16-in/FP16-out" : "BF16-in/FP32-out");
    if (cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&d.input, (size_t) kTokens * kLr * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d.output, (size_t) kTokens * kLr * sizeof(uint16_t)) != cudaSuccess)
        return false;
    std::vector<float> input((size_t) kTokens * kLr);
    for (size_t i = 0; i < input.size(); ++i) input[i] = ((int) (i % kLr) - kLr / 2) * 0.03125f;
    if (cudaMemcpyAsync(d.input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, d.stream) !=
            cudaSuccess || cudaStreamSynchronize(d.stream) != cudaSuccess)
        return false;
    return true;
}

bool run(DeviceBuffers& d, const char* phase) {
    if (cudaSetDevice(d.ordinal) != cudaSuccess) return false;
    strata::prefill::gr_silu(d.input, d.output, kTokens, d.stream);
    const cudaError_t e = cudaStreamSynchronize(d.stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "HIP device %d (%s), %s: %s\n", d.ordinal, d.arch.c_str(), phase, cudaGetErrorString(e));
        return false;
    }
    std::vector<uint16_t> output((size_t) kTokens * kLr);
    if (cudaMemcpy(output.data(), d.output, output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    for (uint16_t v : output)
        if ((v & 0x7f80u) == 0x7f80u) { // finite inputs must not produce NaNs or infinities
            std::fprintf(stderr, "HIP device %d (%s), %s: non-finite output\n", d.ordinal, d.arch.c_str(), phase);
            return false;
        }
    std::printf("HIP device %d (%s): %s PASS\n", d.ordinal, d.arch.c_str(), phase);
    return true;
}

void release(DeviceBuffers& d) {
    if (d.ordinal < 0) return;
    (void) cudaSetDevice(d.ordinal);
    if (d.stream) (void) cudaStreamSynchronize(d.stream);
    if (d.input) (void) cudaFree(d.input);
    if (d.output) (void) cudaFree(d.output);
    if (d.stream) (void) cudaStreamDestroy(d.stream);
}
} // namespace

int main() {
#if defined(_WIN32)
    const bool env_ok = _putenv_s("STRATA_HIP_PROMPT_F16", "1") == 0;
#else
    const bool env_ok = setenv("STRATA_HIP_PROMPT_F16", "1", 1) == 0;
#endif
    if (!env_ok) {
        std::puts("could not set STRATA_HIP_PROMPT_F16=1 for the per-device regression");
        return 1;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: fewer than two visible HIP devices");
        return 77;
    }
    std::vector<DeviceBuffers> devices;
    for (int i = 0; i < count; ++i) {
        DeviceBuffers d;
        if (!setup(d, i)) {
            std::fprintf(stderr, "could not initialize HIP device %d\n", i);
            for (auto& x : devices) release(x);
            return 1;
        }
        if (d.ordinal >= 0) devices.push_back(std::move(d));
    }
    if (devices.size() < 2) {
        for (auto& d : devices) release(d);
        std::puts("SKIP: fewer than two visible devices have compiled code objects");
        return 77;
    }
    bool ok = true;
    for (auto& d : devices) ok = run(d, "first pass") && ok;
    for (auto it = devices.rbegin(); it != devices.rend(); ++it) ok = run(*it, "reverse pass") && ok;
    // This final pass is the regression case: return to the first architecture after all other devices ran.
    ok = run(devices.front(), "return-to-first-device pass") && ok;
    for (auto& d : devices) release(d);
    return ok ? 0 : 1;
}
