// CPU-only end-to-end check: capacity-limited groups equal single-token rows.
#include "strata/host_expert_jobs.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
    if (!cpu::cpu_avx1_ok()) return 77;
    constexpr int H = 64, FF = 64, K = 10;
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, H, FF, fmt, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::vector<float> weights((size_t) H * FF);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.17f);
    std::vector<uint8_t> blob(fmt.bytes);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data(), 0, FF, H, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + fmt.up_off, 0, FF, H, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + fmt.down_off, 0, H, FF, nullptr);
    std::array<std::array<uint8_t, cpu::kNativeActBytes>, cpu::MAXT> act{};
    std::array<std::array<float, H>, cpu::MAXT> reference{};
    for (int t = 0; t < cpu::MAXT; ++t) {
        std::array<float, H> input;
        for (int i = 0; i < H; ++i) input[(size_t) i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(fmt, input.data(), act[(size_t) t].data());
        std::array<float, FF> hidden;
        const void* a[] = {act[(size_t) t].data()};
        float* h[] = {hidden.data()};
        cpu::native_gu_rows(fmt, blob.data(), a, 1, h, 0, FF);
        std::array<uint8_t, cpu::kNativeHBytes> hq;
        cpu::native_quant_h(fmt, hidden.data(), hq.data());
        const void* q[] = {hq.data()};
        float* output[] = {reference[(size_t) t].data()};
        cpu::native_down_rows(fmt, blob.data(), q, 1, output, 0, H);
        for (float value : reference[(size_t) t]) if (!std::isfinite(value)) return 1;
    }
    int checks = 0;
    for (int workers : {1, 3}) for (bool host : {false, true}) {
        cpu::ExpertPool pool(workers, false, host);
        for (int nt = 1; nt <= cpu::MAXT; ++nt) {
            std::vector<cpu::ExpertJobMulti> jobs((size_t) nt * K);
            std::vector<float> output((size_t) nt * K * H, NAN);
            int count = 0;
            int16_t last = -1;
            for (int t = 0; t < nt; ++t) for (int j = 0; j < K; ++j) {
                const int index = strata::core::detail::claim_host_expert_job(jobs, count, last);
                auto& job = jobs[(size_t) index];
                if (job.nt >= cpu::MAXT) return 1;
                const int slot = job.nt++;
                job.blob = blob.data();
                job.nact[slot] = act[(size_t) t].data();
                job.out[slot] = output.data() + ((size_t) t * K + j) * H;
            }
            pool.run_split_multi_native(fmt, jobs.data(), count);
            for (int t = 0; t < nt; ++t) for (int j = 0; j < K; ++j) {
                if (std::memcmp(output.data() + ((size_t) t * K + j) * H,
                                reference[(size_t) t].data(), (size_t) H * sizeof(float)) != 0) {
                    std::fprintf(stderr, "native grouped output differs (workers=%d host=%d nt=%d)\n", workers, host, nt);
                    return 1;
                }
            }
            ++checks;
        }
    }
    std::printf("PASS: %d native grouped evaluations match the single-token rows bit for bit\n", checks);
    return 0;
}
