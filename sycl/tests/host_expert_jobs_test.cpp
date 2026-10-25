// CPU-only regression: repeated routed IDs must not overfill a host expert job.
#include "strata/host_expert_jobs.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
    constexpr int K = 10, EXPERTS = 16;
    std::vector<cpu::ExpertJobMulti> jobs((size_t) cpu::MAXT * K);
    std::array<cpu::ActQ, cpu::MAXT> acts{};
    std::array<int, cpu::MAXT> tokens{};
    std::array<uint8_t, EXPERTS> blobs{};
    for (int e = 0; e < EXPERTS; ++e) blobs[(size_t) e] = (uint8_t) e;
    cpu::NativeFmt stale_format;
    int checks = 0;
    for (int nt = 1; nt <= cpu::MAXT; ++nt) {
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (auto& job : jobs) { job.nt = cpu::MAXT; job.fmt = &stale_format; }
            std::array<int16_t, EXPERTS> last;
            last.fill(-1);
            std::vector<float> out((size_t) nt * K, -1.0f);
            std::vector<int> expected(out.size());
            int count = 0;
            for (int t = 0; t < nt; ++t) {
                tokens[(size_t) t] = t;
                for (int j = 0; j < K; ++j) {
                    const int expert = pattern == 0 ? j : pattern == 1 ? 0 : pattern == 2 ? j / 2 : (j < 3 ? 0 : j);
                    const int index = strata::core::detail::claim_host_expert_job(jobs, count, last[(size_t) expert]);
                    auto& job = jobs[(size_t) index];
                    if (job.nt < 0 || job.nt >= cpu::MAXT || job.fmt != nullptr) {
                        std::fprintf(stderr, "FAIL: job token count %d, capacity %d (nt=%d pattern=%d)\n",
                                     job.nt, cpu::MAXT, nt, pattern);
                        return 1;
                    }
                    const int slot = job.nt++;
                    job.blob = &blobs[(size_t) expert];
                    job.act[slot] = &acts[(size_t) t];
                    job.nact[slot] = &tokens[(size_t) t];
                    job.out[slot] = &out[(size_t) t * K + j];
                    expected[(size_t) t * K + j] = expert * 100 + t;
                }
            }
            int entries = 0;
            for (int j = 0; j < count; ++j) {
                const auto& job = jobs[(size_t) j];
                if (job.nt < 1 || job.nt > cpu::MAXT || job.fmt != nullptr) return 1;
                const int expert = *job.blob;
                for (int t = 0; t < job.nt; ++t) {
                    const int token = *static_cast<const int*>(job.nact[t]);
                    if (job.act[t] != &acts[(size_t) token]) return 1;
                    *job.out[t] = (float) (expert * 100 + token);
                    ++entries;
                }
            }
            if (entries != nt * K) return 1;
            for (size_t i = 0; i < out.size(); ++i) if (out[i] != (float) expected[i]) return 1;
            ++checks;
        }
    }
    {   // No allocation or out-of-bounds access when the caller supplied too few jobs.
        std::vector<cpu::ExpertJobMulti> empty;
        int count = 0;
        int16_t last = -1;
        try { strata::core::detail::claim_host_expert_job(empty, count, last); return 1; }
        catch (const std::length_error&) {}
    }
    std::printf("PASS: %d grouping cases, including duplicate IDs, full windows and reused jobs\n", checks);
    return 0;
}
