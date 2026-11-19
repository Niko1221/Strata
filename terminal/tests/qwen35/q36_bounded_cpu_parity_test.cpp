// Synthetic CPU numerical parity for the three type pairs observed in the
// owned UD-Q4_K_XL pack header. No model payload, CUDA or inference server.
#include "strata/qwen35/bounded_experts.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
    if (!cpu::cpu_avx1_ok()) { std::fprintf(stderr, "AVX unavailable; NOT_RUN\n"); return 77; }
    constexpr int N = 2048, FF = 512, count = 19;
    for (const auto types : {std::pair{GGML_TYPE_Q4_K, GGML_TYPE_Q5_K},
                             std::pair{GGML_TYPE_Q4_K, GGML_TYPE_Q6_K},
                             std::pair{GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}}) {
        cpu::NativeFmt fmt;
        std::string err;
        if (!cpu::native_fmt(types.first, types.second, N, FF, fmt, err)) {
            std::fprintf(stderr, "%s\n", err.c_str()); return 1;
        }
        std::vector<uint8_t> blobs[2] = {std::vector<uint8_t>(fmt.bytes), std::vector<uint8_t>(fmt.bytes)};
        std::vector<float> weights(N * FF);
        for (int b = 0; b < 2; ++b) {
            for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin(float(i + b * 113) * 0.17f);
            ggml_quantize_chunk(types.first, weights.data(), blobs[b].data(), 0, FF, N, nullptr);
            ggml_quantize_chunk(types.first, weights.data(), blobs[b].data() + fmt.up_off, 0, FF, N, nullptr);
            ggml_quantize_chunk(types.second, weights.data(), blobs[b].data() + fmt.down_off, 0, N, FF, nullptr);
        }
        std::vector<std::vector<uint8_t>> acts(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<float> x(N), output(count * cpu::MAXT * N, 0);
        for (int t = 0; t < cpu::MAXT; ++t) {
            for (int i = 0; i < N; ++i) x[i] = std::sin(float(i + t * 13) * 0.031f);
            cpu::native_quant_act(fmt, x.data(), acts[t].data());
        }
        std::vector<cpu::ExpertJobMulti> jobs(count);
        for (int j = 0; j < count; ++j) {
            jobs[j].blob = blobs[j % 2].data();
            jobs[j].nt = j % 3 == 0 ? 1 : j % 3 == 1 ? 4 : 8;
            for (int t = 0; t < jobs[j].nt; ++t) {
                jobs[j].nact[t] = acts[t].data();
                jobs[j].out[t] = output.data() + (j * cpu::MAXT + t) * N;
            }
        }
        cpu::ExpertPool pool(3, false);
        pool.run_split_multi_native(fmt, jobs.data(), count);
        const auto expected = output;
        for (float value : expected) if (!std::isfinite(value)) return 1;
        strata::qwen35::ExpertCpuStaging stage(fmt.bytes);
        for (int repeat = 0; repeat < 2; ++repeat) {
            std::fill(output.begin(), output.end(), 0.0f);
            if (!stage.run(jobs.data(), jobs.size(), fmt.bytes,
                [](const uint8_t* src, uint8_t* dst, size_t bytes) { std::memcpy(dst, src, bytes); return true; },
                [&](cpu::ExpertJobMulti* wave, size_t n) { pool.run_split_multi_native(fmt, wave, (int) n); })) return 1;
            if (std::memcmp(expected.data(), output.data(), output.size() * sizeof(float))) {
                std::fprintf(stderr, "CPU wave mismatch types=%d/%d repeat=%d\n", types.first, types.second, repeat);
                return 1;
            }
        }
        std::printf("CPU staged parity types=%d/%d N=%d FF=%d jobs=%d nt=1,4,8 repeats=2: byte-identical\n",
                    types.first, types.second, N, FF, count);
    }
    return 0;
}
