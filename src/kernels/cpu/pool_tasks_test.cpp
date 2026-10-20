// CPU-only task partition parity. Optional real expert: pool_tasks_test GU_TYPE DOWN_TYPE BLOB_FILE
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static void put_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
// a synthetic Strata-form (s2) expert: random codes, fp16 scales ~0.004-0.016 (as expert_multi_test's)
static void make_s2_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}
static bool same_h(const float* a, const float* b) { return std::memcmp(a, b, (size_t) cpu::H * sizeof(float)) == 0; }
static void nan_h(float* a) { std::fill(a, a + cpu::H, std::numeric_limits<float>::quiet_NaN()); }
static void unset_env(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) return 2;
    // its activation quantiser is the AVX one at least (act_quant_any): a CPU without AVX (Westmere) would stop on an illegal instruction
    if (!cpu::cpu_avx1_ok()) { std::printf("pool_tasks_test: no AVX on this CPU, skipped\n"); return 0; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(argc == 4 ? std::atoi(argv[1]) : GGML_TYPE_Q4_0,
                         argc == 4 ? std::atoi(argv[2]) : GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    if (argc == 4) {
        std::FILE* file = std::fopen(argv[3], "rb");
        if (!file) return 2;
        const size_t got = std::fread(blob.data(), 1, blob.size(), file);
        std::fclose(file);
        if (got != blob.size()) return 2;
    } else {
        std::vector<float> weights((size_t) cpu::H * cpu::FF);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.17f);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    }
    for (int tasks : {-1, cpu::ExpertPool::kMaxTasks + 1}) {
        try { cpu::ExpertPool invalid(1, false, true, cpu::PoolAffinity::All, tasks); return 1; }
        catch (const std::invalid_argument&) {}
    }
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ActQ> qact(cpu::MAXT);
    std::vector<float> x(cpu::H);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[t].data());
        cpu::act_quant_any(x.data(), cpu::H, qact[t]);
    }
    int checks = 0, split_checks = 0, mixed_checks = 0;
    for (int n : {1, 7, cpu::ExpertPool::kMaxSplitMulti + 1}) {
        std::vector<cpu::ExpertJobMulti> jobs(n);
        std::vector<float> out((size_t) n * cpu::MAXT * cpu::H);
        for (int e = 0; e < n; ++e) {
            jobs[e].blob = blob.data();
            jobs[e].nt = e % 3 == 0 ? 1 : e % 3 == 1 ? 4 : cpu::MAXT;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = act[t].data();
                jobs[e].act[t] = &qact[t];
                jobs[e].out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
            }
        }
        { // The default partition is the arithmetic reference; inactive token slots remain zero.
            cpu::ExpertPool baseline(3, false);
            baseline.run_split_multi_native(f, jobs.data(), n);
        }
        const auto ref = out;
        for (float v : ref) if (!std::isfinite(v)) return 1;
        for (bool host : {false, true}) for (int workers : {1, 3})
            for (int tasks : {0, 1, 41, 128, 192, 256, cpu::ExpertPool::kMaxTasks}) {
                cpu::ExpertPool pool(workers, false, host, cpu::PoolAffinity::All, tasks);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    std::fill(out.begin(), out.end(), 0.0f);
                    for (const auto& job : jobs) for (int t = 0; t < job.nt; ++t)
                        std::fill(job.out[t], job.out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
                    pool.run_split_multi_native(f, jobs.data(), n);
                    if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) != 0) {
                        std::fprintf(stderr, "mismatch n=%d workers=%d host=%d tasks=%d repeat=%d\n",
                                     n, workers, host, tasks, repeat);
                        return 1;
                    }
                    ++checks;
                }
            }
        // STRATA_NUMA_POOL=test: two slices without a real second node (thread j -> slice j % 2), every worker
        // sleeping between batches (spin 0) so the late-wake path of issue #29 runs constantly
        put_env("STRATA_NUMA_POOL", "test");
        put_env("STRATA_POOL_SPIN_US", "0");
        for (bool host : {false, true}) for (int workers : {1, 2, 3, 5})
            for (int tasks : {0, 1, 41, cpu::ExpertPool::kMaxTasks}) {
                cpu::ExpertPool pool(workers, false, host, cpu::PoolAffinity::All, tasks);
                const int slices = workers + (host ? 1 : 0) >= 2 ? 2 : 0;
                if (pool.numa_slices() != slices || cpu::active_numa_split().on != (slices == 2) ||
                    (slices == 2 && !cpu::active_numa_split().test)) {
                    std::fprintf(stderr, "numa-split slices %d, expected %d (workers=%d host=%d)\n",
                                 pool.numa_slices(), slices, workers, host);
                    return 1;
                }
                for (int repeat = 0; repeat < 2; ++repeat) {
                    for (const auto& job : jobs) for (int t = 0; t < job.nt; ++t)
                        std::fill(job.out[t], job.out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
                    pool.run_split_multi_native(f, jobs.data(), n);
                    if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) != 0) {
                        std::fprintf(stderr, "numa-split mismatch n=%d workers=%d host=%d tasks=%d repeat=%d\n",
                                     n, workers, host, tasks, repeat);
                        return 1;
                    }
                    ++split_checks;
                }
                pool.diag(stderr);
            }
        // The split with modes 0-4 between the native batches, on one pool: a mode-0..4 batch lives on `head_`, a
        // native one on the slice heads, so this alternates the two families of heads (a stale claim must fail on
        // the other family) and runs the per-slice `parked` through the non-native modes.  The Strata-form kernels
        // need AVX-512 VBMI, the oracle contract included (a Xeon Gold 6226 stops on vpmultishiftqb inside
        // s2_expert_vnni_q with the oracle on), so a CPU without it skips this part.
        // CBA: no coverage on a VBMI-free CPU (the 6226 box); a test-only kernel seam in the pool if it must have it.
        const cpu::CpuFeatures feat = cpu::cpu_features();
        if (n == 7 && feat.usable()) {
            std::mt19937 rng(7);
            std::vector<std::vector<uint8_t>> s2(3, std::vector<uint8_t>(cpu::BLOB));
            for (auto& b : s2) make_s2_blob(b.data(), rng);
            const int NJ = 5;   // single-token jobs: run() and run_split()
            std::vector<cpu::ExpertJob> sj(NJ);
            std::vector<float> sout((size_t) NJ * cpu::H), sref(sout.size());
            std::vector<cpu::ExpertJobMulti> mj(3);   // multi-token jobs: run_split_multi(), 1, 2 and 3 tokens
            std::vector<float> mout((size_t) 3 * cpu::MAXT * cpu::H), mref(mout.size());
            {   // the serial reference: run_split and run_split_multi are bitwise run's, token for token (pool.hpp)
                cpu::ExpertScratch ws;
                for (int j = 0; j < NJ; ++j) {
                    sj[j].blob = s2[(size_t) j % 3].data();
                    sj[j].act = &qact[(size_t) j % cpu::MAXT];
                    sj[j].out = sout.data() + (size_t) j * cpu::H;
                    sj[j].slot = j;
                    cpu::s2_expert_vnni_q(sj[j].blob, *sj[j].act, sref.data() + (size_t) j * cpu::H, ws);
                }
                for (int e = 0; e < 3; ++e) {
                    mj[e].blob = s2[(size_t) e].data();
                    mj[e].nt = e + 1;
                    for (int t = 0; t < mj[e].nt; ++t) {
                        mj[e].act[t] = &qact[(size_t) t];
                        mj[e].out[t] = mout.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
                        cpu::s2_expert_vnni_q(mj[e].blob, qact[(size_t) t], mref.data() + ((size_t) e * cpu::MAXT + t) * cpu::H, ws);
                    }
                }
            }
            for (float v : sref) if (!std::isfinite(v)) { std::fprintf(stderr, "s2 reference not finite\n"); return 1; }
            for (bool host : {false, true}) for (int workers : {1, 2, 3, 5})
                for (int tasks : {0, 41}) {
                    cpu::ExpertPool pool(workers, false, host, cpu::PoolAffinity::All, tasks);
                    for (int repeat = 0; repeat < 3; ++repeat)
                        for (int step = 0; step < 6; ++step) {
                            bool ok = true;
                            if (step % 2 == 1) {   // the native batch between each pair
                                for (const auto& job : jobs) for (int t = 0; t < job.nt; ++t) nan_h(job.out[t]);
                                pool.run_split_multi_native(f, jobs.data(), n);
                                ok = std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) == 0;
                            } else if (step < 4) {   // 0: run (mode 0), 2: run_split (modes 1/2)
                                for (int j = 0; j < NJ; ++j) nan_h(sj[j].out);
                                if (step == 0) pool.run(sj.data(), NJ);
                                else pool.run_split(sj.data(), NJ);
                                for (int j = 0; j < NJ; ++j) ok = ok && same_h(sj[j].out, sref.data() + (size_t) j * cpu::H);
                            } else {   // 4: run_split_multi (modes 3/4)
                                for (const auto& job : mj) for (int t = 0; t < job.nt; ++t) nan_h(job.out[t]);
                                pool.run_split_multi(mj.data(), 3);
                                for (int e = 0; e < 3; ++e) for (int t = 0; t < mj[e].nt; ++t)
                                    ok = ok && same_h(mj[e].out[t], mref.data() + ((size_t) e * cpu::MAXT + t) * cpu::H);
                            }
                            if (!ok) {
                                std::fprintf(stderr, "numa-split mixed-mode mismatch step=%d workers=%d host=%d tasks=%d "
                                                     "repeat=%d\n", step, workers, host, tasks, repeat);
                                return 1;
                            }
                            ++mixed_checks;
                        }
                    pool.diag(stderr);
                }
        } else if (n == 7) {
            std::printf("pool tasks: modes 0-4 interleaved under the split SKIPPED (%s)\n", feat.reason());
        }
        unset_env("STRATA_NUMA_POOL");
        unset_env("STRATA_POOL_SPIN_US");
        if (cpu::active_numa_split().on) { std::fprintf(stderr, "numa-split plan outlived its pool\n"); return 1; }
    }
#if defined(__linux__)
    {   // the real split refuses STRATA_EXCHANGE_ROTATE=1 (experts move between slots, the slices need fixed ones)
        put_env("STRATA_NUMA_POOL", "1");
        put_env("STRATA_EXCHANGE_ROTATE", "1");
        bool refused = false;
        {
            cpu::ExpertPool pool(2, true, true);
            refused = !cpu::active_numa_split().on &&
                      cpu::active_numa_split().why.find("STRATA_EXCHANGE_ROTATE") != std::string::npos;
        }
        unset_env("STRATA_EXCHANGE_ROTATE");
        unset_env("STRATA_NUMA_POOL");
        if (!refused) { std::fprintf(stderr, "numa-split did not refuse STRATA_EXCHANGE_ROTATE=1\n"); return 1; }
    }
    {   // the real split on any host: on means two distinct nodes and cores on both slices, off always says why
        put_env("STRATA_NUMA_POOL", "1");
        bool ok = false;
        {
            cpu::ExpertPool pool(4, true, true);
            const auto& plan = cpu::active_numa_split();
            ok = plan.on ? plan.nodes[0] != plan.nodes[1] && !plan.cores[0].empty() && !plan.cores[1].empty()
                         : !plan.why.empty() && plan.nodes[0] == -1 && plan.nodes[1] == -1 && plan.cores[0].empty() &&
                           plan.cores[1].empty();
            std::printf("pool tasks: real split %s%s\n", plan.on ? "on" : "off: ", plan.why.c_str());
        }
        unset_env("STRATA_NUMA_POOL");
        if (!ok) { std::fprintf(stderr, "numa-split real-mode plan invariant broken\n"); return 1; }
        // the DEFAULT worker count under the split spans every socket (pool_plan.hpp span_packages): on a 2-socket
        // host the split must then be on, where 0.2.0's host-socket default alone would leave every thread on one node
        put_env("STRATA_NUMA_POOL", "1");
        {
            const int packages = (int) cpu::pool_plan_input(cpu::detect_cpu_topology(true, cpu::PoolAffinity::All))
                                     .cores_per_package.size();
            cpu::ExpertPool pool(0, true, true);
            const auto& plan = cpu::active_numa_split();
            std::printf("pool tasks: default pool under the split: %d workers on %d socket(s), split %s%s (%s)\n",
                        pool.workers(), packages, plan.on ? "on" : "off: ", plan.why.c_str(), pool.plan_limit());
            ok = packages != 2 || (plan.on && plan.cores[0].size() > 0 && plan.cores[1].size() > 0);
        }
        unset_env("STRATA_NUMA_POOL");
        if (!ok) { std::fprintf(stderr, "numa-split: the default pool on a 2-socket host did not span both nodes\n"); return 1; }
        // a value other than 0 / 1 / test is off, with the value in the reason
        for (const char* v : {"off", "yes"}) {
            put_env("STRATA_NUMA_POOL", v);
            bool bad = false;
            {
                cpu::ExpertPool pool(2, true, true);
                bad = cpu::active_numa_split().on || cpu::active_numa_split().why.find(std::string("'") + v + "'") == std::string::npos;
            }
            unset_env("STRATA_NUMA_POOL");
            if (bad) { std::fprintf(stderr, "numa-split did not refuse STRATA_NUMA_POOL=%s\n", v); return 1; }
        }
        // unset is the same decision as 1 (on by default with 2 nodes); 0 is off, and an unset switch that ends up
        // off (one node) keeps the same worker count as 0, so a one-node machine runs exactly as before
        const char* vals[3] = {nullptr, "1", "0"};
        int workers[3] = {0, 0, 0};
        bool on[3] = {false, false, false};
        for (int k = 0; k < 3; ++k) {
            if (vals[k] != nullptr) put_env("STRATA_NUMA_POOL", vals[k]);
            {
                cpu::ExpertPool pool(0, true, true);
                workers[k] = pool.workers();
                on[k] = cpu::active_numa_split().on || pool.numa_slices() != 0;
            }
            unset_env("STRATA_NUMA_POOL");
        }
        std::printf("pool tasks: default pool: unset %d workers split %s, 1 %d workers split %s, 0 %d workers\n",
                    workers[0], on[0] ? "on" : "off", workers[1], on[1] ? "on" : "off", workers[2]);
        if (on[0] != on[1] || workers[0] != workers[1] || on[2] || (!on[0] && workers[0] != workers[2])) {
            std::fprintf(stderr, "numa-split: the unset default does not match 1, or 0 / one node changed the pool\n");
            return 1;
        }
    }
#endif
    std::printf("pool tasks: %d bitwise comparisons passed, %d of them in NUMA-split mode (%d with modes 0-4 "
                "interleaved)\n", checks + split_checks + mixed_checks, split_checks + mixed_checks, mixed_checks);
    return 0;
}
