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
    bool f16_io = false;
    cudaStream_t stream = nullptr;
    float* input = nullptr;
    uint16_t* output = nullptr;
};

bool finite_16(uint16_t bits, bool f16) {
    const uint16_t exponent = f16 ? 0x7c00u : 0x7f80u;
    return (bits & exponent) != exponent;
}

bool finite_encoding_tests() {
    return finite_16(0x3c00u, true) && finite_16(0x7bffu, true) && !finite_16(0x7c00u, true) &&
           !finite_16(0xfc00u, true) && !finite_16(0x7e00u, true) && finite_16(0x3f80u, false) &&
           finite_16(0x7c00u, false) && finite_16(0x7f7fu, false) && !finite_16(0x7f80u, false) &&
           !finite_16(0xff80u, false) && !finite_16(0x7fc1u, false);
}

bool expected_f16(const std::string& mode, const std::string& arch) {
    return mode == "1" || (mode == "gfx103x" && arch.compare(0, 6, "gfx103") == 0);
}

bool setup(DeviceBuffers& d, int ordinal, const std::string& mode) {
    cudaDeviceProp prop{};
    if (cudaSetDevice(ordinal) != cudaSuccess || cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) return false;
    if (!compiled_for(prop.gcnArchName)) return true; // this executable intentionally skips unrelated visible devices
    d.ordinal = ordinal;
    d.arch = prop.gcnArchName;
    d.f16_io = strata::prefill::prompt_f16();
    const bool expect_f16 = expected_f16(mode, d.arch);
    if (d.f16_io != expect_f16) {
        std::fprintf(stderr, "HIP device %d (%s), STRATA_HIP_PROMPT_F16=%s: expected %s, got %s\n", ordinal,
                     d.arch.c_str(), mode.c_str(), expect_f16 ? "FP16" : "BF16", d.f16_io ? "FP16" : "BF16");
        return false;
    }
    strata::prefill::set_act_f16(d.f16_io);
    std::printf("HIP device %d (%s): effective prompt GEMM mode %s\n", ordinal, d.arch.c_str(),
                d.f16_io ? "FP16-in/FP16-out" : "BF16-in/FP32-out");
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
        if (!finite_16(v, d.f16_io)) { // test the exponent layout used by this device's FP16/BF16 activation image
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

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "gfx103x";
    if (mode != "0" && mode != "1" && mode != "gfx103x") {
        std::fprintf(stderr, "usage: %s [0|1|gfx103x]\n", argv[0]);
        return 2;
    }
#if defined(_WIN32)
    const bool env_ok = _putenv_s("STRATA_HIP_PROMPT_F16", mode.c_str()) == 0;
#else
    const bool env_ok = setenv("STRATA_HIP_PROMPT_F16", mode.c_str(), 1) == 0;
#endif
    if (!env_ok) {
        std::fprintf(stderr, "could not set STRATA_HIP_PROMPT_F16=%s for the per-device regression\n", mode.c_str());
        return 1;
    }
    if (!finite_encoding_tests()) {
        std::puts("FP16/BF16 finite-value exponent checks failed");
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
        if (!setup(d, i, mode)) {
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
