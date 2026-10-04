#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: pool_pipeline_parity PACK GGUF_SHARD (real-model test)\n");
        return 77;
    }
    std::string error;
    if (!cpu::expert_layout_load(argv[1], 48, 512, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    const auto& layout = cpu::expert_layout();
    std::FILE* shard = std::fopen(argv[2], "rb");
    if (!shard) return 2;
    int checks = 0;
    const int max_jobs = 97;
    for (int layer : {0, 1, 2}) {
        const auto& fmt = layout.fmt[(size_t) layer];
        std::vector<std::vector<uint8_t>> blobs(7, std::vector<uint8_t>(fmt.bytes));
        for (int e = 0; e < 7; ++e) {
            const size_t sizes[] = {fmt.up_off, fmt.down_off - fmt.up_off, fmt.bytes - fmt.down_off};
            const size_t dst[] = {0, fmt.up_off, fmt.down_off};
            for (int role = 0; role < 3; ++role) {
                const auto offset = layout.gguf_off[(size_t) layer * 3 + role] + e * sizes[role];
                if (fseeko(shard, (off_t) offset, SEEK_SET) != 0 ||
                    std::fread(blobs[e].data() + dst[role], 1, sizes[role], shard) != sizes[role]) return 2;
            }
        }
        std::mt19937 random(51 + layer);
        std::normal_distribution<float> gauss(0.f, .2f);
        std::vector<std::vector<uint8_t>> acts(4, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<cpu::ActQ> q2_acts(4);
        for (int t = 0; t < 4; ++t) {
            std::vector<float> x(cpu::H);
            for (float& v : x) v = gauss(random);
            cpu::native_quant_act(fmt, x.data(), acts[t].data());
            cpu::act_quant_any(x.data(), cpu::H, q2_acts[t]);
        }
        std::vector<cpu::ExpertJobMulti> jobs(max_jobs);
        std::vector<float> result((size_t) max_jobs * 4 * cpu::H);
        for (int e = 0; e < max_jobs; ++e) {
            jobs[e].blob = blobs[e % 7].data();
            for (int t = 0; t < 4; ++t) {
                jobs[e].nact[t] = acts[t].data();
                jobs[e].act[t] = &q2_acts[t];
                jobs[e].out[t] = result.data() + ((size_t) e * 4 + t) * cpu::H;
            }
        }
        for (int workers : {1, 4, 17}) for (bool host : {false, true}) {
            setenv("STRATA_POOL_PIPELINE", "0", 1);
            cpu::ExpertPool baseline(workers, true, host);
            setenv("STRATA_POOL_PIPELINE", "1", 1);
            cpu::ExpertPool pipeline(workers, true, host);
            for (int nt : {1, 2, 4}) for (int count : {1, 2, 7, 18, 97}) {
                for (auto& j : jobs) j.nt = nt;
                std::fill(result.begin(), result.end(), -1.2345e33f);
                baseline.run_split_multi_native(fmt, jobs.data(), count);
                const auto expected = result;
                if (workers == 1 && !host) {
                    for (int e = 0; e < std::min(count, 7); ++e) {
                        std::vector<float> ff((size_t) nt * cpu::FF);
                        std::vector<float> out((size_t) nt * cpu::H);
                        std::vector<std::vector<uint8_t>> hq(nt, std::vector<uint8_t>(cpu::kNativeHBytes));
                        std::vector<cpu::ActQ> aq(nt);
                        float* fp[cpu::MAXT];
                        float* op[cpu::MAXT];
                        const void* hp[cpu::MAXT];
                        const cpu::ActQ* ap[cpu::MAXT];
                        for (int t = 0; t < nt; ++t) {
                            fp[t] = ff.data() + (size_t) t * cpu::FF;
                            op[t] = out.data() + (size_t) t * cpu::H;
                            hp[t] = hq[t].data();
                            ap[t] = &aq[t];
                        }
                        cpu::native_gu_rows(fmt, jobs[e].blob, jobs[e].nact, nt, fp, 0, cpu::FF);
                        for (int t = 0; t < nt; ++t)
                            if (fmt.d_type == 42) cpu::act_quant_any(fp[t], cpu::FF, aq[t]);
                            else cpu::native_quant_h(fmt, fp[t], hq[t].data());
                        if (fmt.d_type == 42)
                            cpu::q2_rows_any(jobs[e].blob + fmt.down_off, fmt.d_row, cpu::FF / 64,
                                            ap, nt, op, 0, cpu::H);
                        else cpu::native_down_rows(fmt, jobs[e].blob, hp, nt, op, 0, cpu::H);
                        for (int t = 0; t < nt; ++t) {
                            if (std::memcmp(op[t], jobs[e].out[t], cpu::H * sizeof(float)) != 0) {
                                std::fprintf(stderr, "SERIAL REFERENCE FAILED layer=%d nt=%d count=%d\n",
                                             layer, nt, count);
                                return 1;
                            }
                            for (int r = 0; r < cpu::H; ++r)
                                if (!std::isfinite(op[t][r]) || op[t][r] == -1.2345e33f) return 1;
                        }
                    }
                }
                for (int repeat = 0; repeat < 3; ++repeat) {
                    std::fill(result.begin(), result.end(), -1.2345e33f);
                    pipeline.run_split_multi_native(fmt, jobs.data(), count);
                    if (std::memcmp(result.data(), expected.data(), result.size() * sizeof(float)) != 0) {
                        std::fprintf(stderr, "PARITY FAILED layer=%d workers=%d host=%d nt=%d count=%d\n",
                                     layer, workers, (int) host, nt, count);
                        return 1;
                    }
                    ++checks;
                }
            }
        }
        std::printf("LAYER %d PARITY PASSED (%d checks)\n", layer, checks);
        std::fflush(stdout);
    }
    std::fclose(shard);
    std::printf("PIPELINE BITWISE PARITY %d/%d PASSED\n", checks, checks);
}
