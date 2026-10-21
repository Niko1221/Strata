// Exercise HIP BLAS and the GR SiLU kernel on a persistent thread per device, then return to the first device.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    bool f16_io = false;
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

uint16_t to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float from_bf16(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t to_f16(float value) {
    const __half half = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float from_f16(uint16_t value) {
    const __half_raw raw{value};
    return __half2float(__half(raw));
}

bool check_cpu_reference(DeviceState& d, bool f16_io) {
    constexpr int t = 5, n = 96, k = 128;
    std::vector<uint16_t> x((size_t) t * k), w((size_t) n * k);
    std::vector<float> xf(x.size()), wf(w.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float source = (static_cast<int>((i * 17 + 3) % 101) - 50) / 200.0f;
        x[i] = f16_io ? to_f16(source) : to_bf16(source);
        xf[i] = f16_io ? from_f16(x[i]) : from_bf16(x[i]);
    }
    for (size_t i = 0; i < w.size(); ++i) {
        const float source = (static_cast<int>((i * 29 + 7) % 97) - 48) / 250.0f;
        w[i] = to_bf16(source);
        wf[i] = from_bf16(w[i]);
    }
    if (cudaMemcpyAsync(d.x, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, d.stream) != cudaSuccess ||
        cudaMemcpyAsync(d.w, w.data(), w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, d.stream) != cudaSuccess)
        return false;
    d.gemm->bf16(d.x, d.w, d.y, t, n, k, n);
    if (cudaStreamSynchronize(d.stream) != cudaSuccess) return false;
    std::vector<float> got((size_t) t * n);
    if (cudaMemcpy(got.data(), d.y, got.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    double diff2 = 0.0, ref2 = 0.0, maximum = 0.0;
    bool finite = true;
    for (int row = 0; row < t; ++row)
        for (int col = 0; col < n; ++col) {
            double ref = 0.0;
            for (int i = 0; i < k; ++i)
                ref += (double) xf[(size_t) row * k + i] * wf[(size_t) col * k + i];
            const double delta = (double) got[(size_t) row * n + col] - ref;
            finite = finite && std::isfinite(got[(size_t) row * n + col]);
            diff2 += delta * delta;
            ref2 += ref * ref;
            maximum = std::max(maximum, std::abs(delta));
        }
    const double rel = std::sqrt(diff2 / std::max(ref2, 1e-300));
    const double rel_limit = f16_io ? 2e-3 : 1e-4;
    const bool ok = finite && ref2 > 0.0 && rel < rel_limit && maximum < 5e-3;
    std::printf("device %d (%s): nontrivial %s GEMM vs CPU %s rel_l2=%.3g max_abs=%.3g\n", d.ordinal,
                d.arch.c_str(), f16_io ? "FP16-out" : "BF16-in/FP32-out", ok ? "PASS" : "FAIL", rel, maximum);
    const bool cleared = cudaMemsetAsync(d.x, 0, x.size() * sizeof(uint16_t), d.stream) == cudaSuccess &&
                         cudaMemsetAsync(d.w, 0, w.size() * sizeof(uint16_t), d.stream) == cudaSuccess &&
                         cudaStreamSynchronize(d.stream) == cudaSuccess;
    return ok && cleared;
}

// Seed a deterministic sparse-but-nonzero slice in the full 8192 x 320 x 10240 stress GEMM.
// The rest of the already-zeroed buffers stay zero, so the GPU exercises the large shape without a huge CPU reference.
bool prepare_large_nonzero(DeviceState& d) {
    constexpr int active_k = 128, active_n = 16;
    std::vector<uint16_t> x(active_k), w((size_t) active_n * active_k);
    for (int i = 0; i < active_k; ++i) {
        const float source = 0.02f + (float) (i % 19) * 0.001f;
        x[i] = d.f16_io ? to_f16(source) : to_bf16(source);
    }
    for (int row = 0; row < active_n; ++row)
        for (int i = 0; i < active_k; ++i)
            w[(size_t) row * active_k + i] = to_bf16(0.002f + (float) ((row * active_k + i) % 23) * 0.0001f);
    if (cudaMemcpyAsync(d.x, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, d.stream) != cudaSuccess)
        return false;
    for (int row = 0; row < active_n; ++row)
        if (cudaMemcpyAsync(d.w + (size_t) row * kK, w.data() + (size_t) row * active_k,
                            (size_t) active_k * sizeof(uint16_t), cudaMemcpyHostToDevice, d.stream) != cudaSuccess)
            return false;
    return cudaStreamSynchronize(d.stream) == cudaSuccess;
}

bool check_large_reference(DeviceState& d, const char* phase) {
    constexpr int active_k = 128, active_n = 16, hc = 4;
    std::vector<uint16_t> x(active_k), w((size_t) active_n * active_k), got_gr(active_n);
    std::vector<float> xf(active_k), wf((size_t) active_n * active_k), got(active_n);
    for (int i = 0; i < active_k; ++i) {
        const float source = 0.02f + (float) (i % 19) * 0.001f;
        x[i] = d.f16_io ? to_f16(source) : to_bf16(source);
        xf[i] = d.f16_io ? from_f16(x[i]) : from_bf16(x[i]);
    }
    for (int row = 0; row < active_n; ++row)
        for (int i = 0; i < active_k; ++i) {
            const size_t q = (size_t) row * active_k + i;
            w[q] = to_bf16(0.002f + (float) (q % 23) * 0.0001f);
            wf[q] = from_bf16(w[q]);
        }
    if (cudaMemcpy(got.data(), d.y, got.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(got_gr.data(), d.y16, got_gr.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;

    double diff2 = 0.0, ref2 = 0.0, maximum = 0.0;
    bool finite = true;
    for (int row = 0; row < active_n; ++row) {
        double sum = 0.0;
        for (int i = 0; i < active_k; ++i)
            sum += (double) xf[i] * wf[(size_t) row * active_k + i];
        const double delta = (double) got[row] - sum;
        finite = finite && std::isfinite(got[row]) && got[row] > 0.0f && sum > 0.0;
        diff2 += delta * delta;
        ref2 += sum * sum;
        maximum = std::max(maximum, std::abs(delta));
    }
    const double rel = std::sqrt(diff2 / std::max(ref2, 1e-300));
    const bool gemm_ok = finite && rel < (d.f16_io ? 2e-3 : 1e-4) && maximum < 5e-3;
    std::printf("device %d (%s), %s: large-shape GEMM vs CPU %s [T=%d N=%d K=%d; checked %d outputs, active K=%d] "
                "rel_l2=%.3g max_abs=%.3g\n",
                d.ordinal, d.arch.c_str(), phase, gemm_ok ? "PASS" : "FAIL", kTokens, kN, kK, active_n,
                active_k, rel, maximum);
    if (!gemm_ok) return false;

    bool gr_ok = true;
    double gr_max_abs = 0.0, gr_max_ulp = 0.0;
    for (int i = 0; i < active_n; ++i) {
        const double v = (double) got[i] / hc;
        const double expected = v / (1.0 + std::exp(-v));
        const float actual = d.f16_io ? from_f16(got_gr[i]) : from_bf16(got_gr[i]);
        const double error = std::abs((double) actual - expected);
        const double ulp = std::ldexp(1.0, std::ilogb(std::abs(expected)) - (d.f16_io ? 10 : 7));
        gr_ok = gr_ok && std::isfinite(actual) && actual > 0.0f && error <= 1.25 * ulp + 1e-7;
        gr_max_abs = std::max(gr_max_abs, error);
        gr_max_ulp = std::max(gr_max_ulp, error / ulp);
    }
    std::printf("device %d (%s), %s: large-shape GR SiLU vs CPU %s max_abs=%.3g max_ulp=%.3g\n", d.ordinal,
                d.arch.c_str(), phase, gr_ok ? "PASS" : "FAIL", gr_max_abs, gr_max_ulp);
    return gr_ok;
}

// Re-run a small, nonzero GEMM plus GR SiLU after each device phase and after returning to the first device.
// Reuse the large stress buffers; the small host reference adds no device allocation.
bool check_transition_reference(DeviceState& d, const char* phase) {
    constexpr int t = 5, n = kN, k = 128, hc = 4, lr = 320;
    std::vector<uint16_t> x((size_t) t * k), w((size_t) n * k);
    std::vector<float> xf(x.size()), wf(w.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float source = 0.02f + (float) (i % 19) * 0.001f;
        x[i] = d.f16_io ? to_f16(source) : to_bf16(source);
        xf[i] = d.f16_io ? from_f16(x[i]) : from_bf16(x[i]);
    }
    for (size_t i = 0; i < w.size(); ++i) {
        const float source = 0.002f + (float) (i % 23) * 0.0001f;
        w[i] = to_bf16(source);
        wf[i] = from_bf16(w[i]);
    }
    const size_t x_bytes = x.size() * sizeof(uint16_t), w_bytes = w.size() * sizeof(uint16_t);
    const size_t w_clear_bytes = (size_t) kN * kK * sizeof(uint16_t);
    if (cudaMemcpyAsync(d.x, x.data(), x_bytes, cudaMemcpyHostToDevice, d.stream) != cudaSuccess ||
        cudaMemcpyAsync(d.w, w.data(), w_bytes, cudaMemcpyHostToDevice, d.stream) != cudaSuccess)
        return false;
    d.gemm->bf16(d.x, d.w, d.y, t, n, k, n);
    if (cudaStreamSynchronize(d.stream) != cudaSuccess) return false;
    std::vector<float> got((size_t) t * n);
    if (cudaMemcpy(got.data(), d.y, got.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    double diff2 = 0.0, ref2 = 0.0, maximum = 0.0;
    bool finite = true;
    for (int row = 0; row < t; ++row)
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int i = 0; i < k; ++i)
                sum += (double) xf[(size_t) row * k + i] * wf[(size_t) col * k + i];
            const size_t q = (size_t) row * n + col;
            const double delta = (double) got[q] - sum;
            finite = finite && std::isfinite(got[q]);
            diff2 += delta * delta;
            ref2 += sum * sum;
            maximum = std::max(maximum, std::abs(delta));
        }
    const double rel = std::sqrt(diff2 / std::max(ref2, 1e-300));
    const double rel_limit = d.f16_io ? 2e-3 : 1e-4;
    const bool gemm_ok = finite && ref2 > 0.0 && rel < rel_limit && maximum < 5e-3;
    std::printf("device %d (%s), %s: nonzero GEMM vs CPU %s rel_l2=%.3g max_abs=%.3g\n", d.ordinal,
                d.arch.c_str(), phase, gemm_ok ? "PASS" : "FAIL", rel, maximum);
    if (!gemm_ok) return false;

    strata::prefill::gr_silu(d.y, d.y16, t, d.stream);
    if (cudaStreamSynchronize(d.stream) != cudaSuccess) return false;
    std::vector<uint16_t> got_gr((size_t) t * lr);
    if (cudaMemcpy(got_gr.data(), d.y16, got_gr.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    bool gr_ok = true;
    double gr_max_abs = 0.0, gr_max_ulp = 0.0;
    for (size_t i = 0; i < got_gr.size(); ++i) {
        // GEMM is compared with its own CPU reference above; compare GR SiLU against its actual FP32 input.
        const double v = (double) got[i] / hc;
        const double expected = v / (1.0 + std::exp(-v));
        const float actual = d.f16_io ? from_f16(got_gr[i]) : from_bf16(got_gr[i]);
        const double error = std::abs((double) actual - expected);
        const int exponent = std::ilogb(std::abs(expected));
        const double ulp = std::ldexp(1.0, exponent - (d.f16_io ? 10 : 7));
        const double tolerance = 1.25 * ulp + 1e-7;
        gr_ok = gr_ok && std::isfinite(actual) && error <= tolerance;
        gr_max_abs = std::max(gr_max_abs, error);
        gr_max_ulp = std::max(gr_max_ulp, error / ulp);
    }
    std::printf("device %d (%s), %s: nonzero GR SiLU vs CPU %s max_abs=%.3g max_ulp=%.3g\n", d.ordinal,
                d.arch.c_str(), phase, gr_ok ? "PASS" : "FAIL", gr_max_abs, gr_max_ulp);

    // Restore the all-zero stress inputs so the original 43/5/1 large-shape sequence is unchanged.
    const bool cleared = cudaMemsetAsync(d.x, 0, x_bytes, d.stream) == cudaSuccess &&
                         cudaMemsetAsync(d.w, 0, w_clear_bytes, d.stream) == cudaSuccess &&
                         cudaStreamSynchronize(d.stream) == cudaSuccess;
    return gr_ok && cleared;
}

bool init_gemm(DeviceState& d, const std::string& mode) {
    if (cudaSetDevice(d.ordinal) != cudaSuccess) return false;
    const bool f16_io = strata::prefill::prompt_f16();
    d.f16_io = f16_io;
    const bool expected = (!mode.empty() && mode[0] == '1') || mode == "all" ||
                          (mode == "gfx103x" && d.arch.compare(0, 6, "gfx103") == 0);
    if (f16_io != expected) {
        std::fprintf(stderr, "device %d (%s): STRATA_HIP_PROMPT_F16=%s selected %s; expected %s\n", d.ordinal,
                     d.arch.c_str(), mode.c_str(), f16_io ? "FP16" : "BF16", expected ? "FP16" : "BF16");
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
    return check_cpu_reference(d, f16_io);
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

bool release(DeviceState& d) {
    if (d.ordinal < 0) return true;
    bool ok = true;
    auto check = [&](const char* operation, cudaError_t error) {
        if (error != cudaSuccess) {
            std::fprintf(stderr, "device %d (%s): cleanup %s failed: %s\n", d.ordinal, d.arch.c_str(), operation,
                         cudaGetErrorString(error));
            ok = false;
        }
    };
    check("cudaSetDevice", cudaSetDevice(d.ordinal));
    if (d.stream) check("cudaStreamSynchronize", cudaStreamSynchronize(d.stream));
    d.gemm.reset();
    check("Gemm destruction", cudaGetLastError());
    auto free_device = [&](const char* name, auto* pointer) {
        if (pointer) check(name, cudaFree(pointer));
    };
    free_device("cudaFree(x)", d.x);
    free_device("cudaFree(w)", d.w);
    free_device("cudaFree(y)", d.y);
    free_device("cudaFree(y16)", d.y16);
    if (d.stream) check("cudaStreamDestroy", cudaStreamDestroy(d.stream));
    std::printf("device %d (%s): resource cleanup %s\n", d.ordinal, d.arch.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "gfx103x";
    bool reference_only = false, reverse = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reference-only") == 0) reference_only = true;
        else if (std::strcmp(argv[i], "--reverse") == 0) reverse = true;
        else {
            std::fprintf(stderr, "usage: %s [0|1|all|gfx103x] [--reference-only] [--reverse]\n", argv[0]);
            return 2;
        }
    }
    const bool legacy_prefix = mode.size() > 1 && mode[0] == '1';
    if (mode != "0" && mode != "1" && !legacy_prefix && mode != "all" && mode != "gfx103x") {
        std::fprintf(stderr, "usage: %s [0|1|all|gfx103x] [--reference-only] [--reverse]\n", argv[0]);
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
    if (reverse) std::swap(first_device, second_device);
    DeviceState first, second;
    if (!init_buffers(first, first_device) || !init_buffers(second, second_device)) {
        std::puts("could not initialize the mixed-architecture HIP devices");
        (void) release(second);
        (void) release(first);
        return 1;
    }
    if (reference_only) {
        const bool first_ok = init_gemm(first, mode);
        const bool second_ok = init_gemm(second, mode);
        const bool second_clean = release(second);
        const bool first_clean = release(first);
        return first_ok && second_ok && second_clean && first_clean ? 0 : 1;
    }

    std::mutex mutex;
    std::condition_variable cv;
    bool first_done = false, second_done = false;
    bool first_ok = false, second_ok = false, return_ok = false;
    std::thread first_thread([&] {
        first_ok = init_gemm(first, mode) && prepare_large_nonzero(first) && step(first, 43, "first stage") &&
               check_large_reference(first, "first-device phase") &&
               check_transition_reference(first, "first-device phase");
        std::unique_lock<std::mutex> lock(mutex);
        first_done = true;
        cv.notify_all();
        cv.wait(lock, [&] { return second_done; });
        lock.unlock();
        if (first_ok) return_ok = step(first, 1, "return-to-first-device chunk") &&
                                  check_transition_reference(first, "return-to-first-device phase");
    });
    std::thread second_thread([&] {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return first_done; });
        }
        second_ok = init_gemm(second, mode) && prepare_large_nonzero(second) && step(second, 5, "second stage") &&
                    check_large_reference(second, "second-device phase") &&
                    check_transition_reference(second, "second-device phase");
        std::lock_guard<std::mutex> lock(mutex);
        second_done = true;
        cv.notify_all();
    });
    first_thread.join();
    second_thread.join();
    const bool ok = first_ok && second_ok && return_ok;
    const bool second_clean = release(second);
    const bool first_clean = release(first);
    return ok && second_clean && first_clean ? 0 : 1;
}
