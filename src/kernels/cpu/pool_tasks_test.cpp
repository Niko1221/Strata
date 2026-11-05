// CPU-only task partition parity. Optional real expert: pool_tasks_test GU_TYPE DOWN_TYPE BLOB_FILE
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// what a park costs is a microbenchmark: it measures CPU time, and only its own build contains it.  Ask for it
// with -DSTRATA_POOL_SPIN_US_MICROBENCH=ON (option in CMakeLists.txt); an ordinary test build compiles none of
// this section or the headers it needs.  Built, it runs only when asked by name (`pool_tasks_test parkcost`),
// which CMake registers under its own test name, and it prints: only `parkcost assert`, or
// STRATA_POOL_PARK_ASSERT=1, can fail a run.
#ifdef STRATA_POOL_SPIN_US_MICROBENCH
#include <chrono>
#include <thread>
#include "strata/platform/cpu_relax.hpp"
#ifdef __linux__
#include <sys/resource.h>
#include <sys/time.h>
#endif
#endif
// the timing needs `getrusage`, and the change it prices is compiled on Linux only (the gate in CMakeLists.txt
// pairs with the `#ifdef __linux__` this file already uses), so the measurement is Linux-only; it also needs the
// pool in its own build to carry the change, because a build that compiles the park loop as the last release does
// has no pause stage left to price.
#if defined(STRATA_POOL_SPIN_US_MICROBENCH) && defined(__linux__) && defined(STRATA_POOL_SPIN_ZERO)
#define STRATA_POOL_PARK_COST 1
#endif

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

#if defined(STRATA_POOL_PARK_COST)
// What a park costs.  A parked worker used to run 1024 strata_cpu_pause() calls before the kernel sleep even with
// STRATA_POOL_SPIN_US=0, and the pauses change no output, so cost is the only thing that tells the two builds
// apart.  One tiny batch per phase with a gap between the phases parks every worker once per phase; that window
// minus the same phases run back to back (nothing parks) is what the parks cost, printed against what the same
// number of pause iterations costs in one thread.  The bound for the asserting arm is 1.00, and on a busy box the
// winning side has under 2x to spare (best 1.50 for the build that spins, worst 0.55 for the build that sleeps, on
// a Core Ultra 9 275HX with a serving strata-host; 0.38 against 2.04 idle).  It builds its own expert and its own
// pool, so it cannot disturb the parity run above.
static int park_cost(bool must_assert) {
    const char* ask = std::getenv("STRATA_POOL_PARK_ASSERT");
    if (ask && *ask && std::strcmp(ask, "0") != 0) must_assert = true;
    if (!cpu::cpu_avx1_ok()) { std::printf("park cost: no AVX on this CPU, skipped\n"); return 0; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    std::vector<float> weights((size_t) cpu::H * cpu::FF);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.17f);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data(), 0, cpu::FF, cpu::H, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ActQ> qact(cpu::MAXT);
    std::vector<float> x(cpu::H);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[t].data());
        cpu::act_quant_any(x.data(), cpu::H, qact[t]);
    }
    put_env("STRATA_POOL_SPIN_US", "0");
    const int pw = 8, pphases = 1000, pgap_us = 200, preps = 3;
    const double park_bound = 1.0;   // the parks must not add up to a second run of the pause loop, which is what
                                     // a worker that spins through its park does once per park
    const long pauses = (long) pw * pphases * 1024;   // one pause stage per idle worker per phase
    std::vector<cpu::ExpertJobMulti> pjob(1);
    std::vector<float> pout((size_t) cpu::MAXT * cpu::H, 0.0f);
    pjob[0].blob = blob.data();
    pjob[0].nt = 1;
    pjob[0].nact[0] = act[0].data();
    pjob[0].act[0] = &qact[0];
    pjob[0].out[0] = pout.data();
    const auto cpu_us = [] {
        struct rusage r;
        getrusage(RUSAGE_SELF, &r);
        return (double) r.ru_utime.tv_sec * 1e6 + (double) r.ru_utime.tv_usec
             + (double) r.ru_stime.tv_sec * 1e6 + (double) r.ru_stime.tv_usec;
    };
    const auto gap = std::chrono::microseconds(pgap_us);
    cpu::ExpertPool pool(pw, false, true, cpu::PoolAffinity::All, 0);
    const auto phase = [&] { nan_h(pout.data()); pool.run_split_multi_native(f, pjob.data(), 1); };
    for (int warm = 0; warm < 20; ++warm) { phase(); std::this_thread::sleep_for(gap); }
    if (!std::isfinite(pout[0])) { std::fprintf(stderr, "park cost: the batch left no output\n"); return 1; }
    double best_ratio = 0.0, best_parks_us = 0.0;
    int kept = 0, dropped = 0;   // `kept`, not a negative ratio as the sentinel: a rep of the build that sleeps
                                  // measures a cost of 0, and noise can make it read negative (-13 ms here, which
                                  // an old -1.0 sentinel took for "no rep measured anything" and failed on).
    for (int rep = 0; rep < preps; ++rep) {
        const double parked0 = cpu_us();
        for (int p = 0; p < pphases; ++p) { phase(); std::this_thread::sleep_for(gap); }
        const double parked1 = cpu_us();
        const double hot0 = cpu_us();
        for (int p = 0; p < pphases; ++p) phase();   // back to back: the workers never park between phases
        const double hot1 = cpu_us();
        const double parked_us = parked1 - parked0, hot_us = hot1 - hot0;
        const double parks_us = parked_us - hot_us;
        const double ref0 = cpu_us();
        for (long i = 0; i < pauses; ++i) strata_cpu_pause();
        const double ref_us = cpu_us() - ref0;
        const double ratio = ref_us > 0 ? parks_us / ref_us : 0.0;
        std::printf("pool tasks: park cost rep %d/%d: %d workers x %d phases at %d us: %.0f ms of CPU with the "
                    "gap, %.0f ms back to back, parks %.0f ms; %ld plain pauses alone %.0f ms; %.2f of that\n",
                    rep + 1, preps, pw, pphases, pgap_us, parked_us / 1000, hot_us / 1000, parks_us / 1000,
                    pauses, ref_us / 1000, ratio);
        // a rep whose gap window cost no more than the back-to-back one measured no parks at all, so it cannot tell
        // the two builds apart: drop it rather than fail on it, and say so in the same line as the rest.
        if (parked_us + 0.25 * ref_us < hot_us) {
            std::printf("pool tasks: park cost rep %d/%d dropped: the gap window used no more CPU than the "
                        "back-to-back window\n", rep + 1, preps);
            ++dropped;
            continue;
        }
        if (kept == 0 || ratio < best_ratio) { best_ratio = ratio; best_parks_us = parks_us; }
        ++kept;
    }
    unset_env("STRATA_POOL_SPIN_US");
    if (kept == 0) {
        std::fprintf(must_assert ? stderr : stdout, "pool tasks: park cost: none of the %d reps measured any "
                             "parks, so this run cannot tell the two builds apart%s\n", preps,
                     must_assert ? "" : " (measured, not asserted)");
        return must_assert ? 1 : 0;
    }
    std::printf("pool tasks: park cost: %d of %d reps measured parks; the best cost %.0f ms = %.2f of the cost of "
                "%ld plain pauses; ", kept, preps, best_parks_us / 1000, best_ratio, pauses);
    if (must_assert) std::printf("the asserted bound is %.2f\n", park_bound);
    else std::printf("printed only, not asserted (parkcost assert, or STRATA_POOL_PARK_ASSERT=1)\n");
    if (must_assert && best_ratio > park_bound) {
        std::fprintf(stderr, "park cost: a parked worker at STRATA_POOL_SPIN_US=0 cost %.0f ms of CPU in %d phases, "
                             "%.2f of the cost of %ld plain pauses: the worker spun before it slept, so the hot "
                             "stage the knob turns off was still running\n",
                     best_parks_us / 1000, pphases, best_ratio, pauses);
        return 1;
    }
    return 0;
}
#endif

int main(int argc, char** argv) {
#if defined(STRATA_POOL_PARK_COST)
    // asked for by name only, and ahead of the argument checks: with the option off this is not compiled, and
    // `pool_tasks_test parkcost` is a wrong argument count exactly as it was before.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "parkcost") != 0) continue;
        bool must_assert = false;
        for (int j = 1; j < argc; ++j) if (std::strcmp(argv[j], "assert") == 0) must_assert = true;
        return park_cost(must_assert);
    }
#endif
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
        // STRATA_NUMA_SPLIT=test: two slices without a real second node (thread j -> slice j % 2), every worker
        // sleeping between batches (spin 0) so the late-wake path of issue #29 runs constantly
        put_env("STRATA_NUMA_SPLIT", "test");
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
        unset_env("STRATA_NUMA_SPLIT");
        unset_env("STRATA_POOL_SPIN_US");
        if (cpu::active_numa_split().on) { std::fprintf(stderr, "numa-split plan outlived its pool\n"); return 1; }
    }
#if defined(__linux__)
    {   // the real split refuses STRATA_EXCHANGE_ROTATE=1 (experts move between slots, the slices need fixed ones)
        put_env("STRATA_NUMA_SPLIT", "1");
        put_env("STRATA_EXCHANGE_ROTATE", "1");
        bool refused = false;
        {
            cpu::ExpertPool pool(2, true, true);
            refused = !cpu::active_numa_split().on &&
                      cpu::active_numa_split().why.find("STRATA_EXCHANGE_ROTATE") != std::string::npos;
        }
        unset_env("STRATA_EXCHANGE_ROTATE");
        unset_env("STRATA_NUMA_SPLIT");
        if (!refused) { std::fprintf(stderr, "numa-split did not refuse STRATA_EXCHANGE_ROTATE=1\n"); return 1; }
    }
    {   // the real split on any host: on means two distinct nodes and cores on both slices, off always says why
        put_env("STRATA_NUMA_SPLIT", "1");
        bool ok = false;
        {
            cpu::ExpertPool pool(4, true, true);
            const auto& plan = cpu::active_numa_split();
            ok = plan.on ? plan.nodes[0] != plan.nodes[1] && !plan.cores[0].empty() && !plan.cores[1].empty()
                         : !plan.why.empty() && plan.nodes[0] == -1 && plan.nodes[1] == -1 && plan.cores[0].empty() &&
                           plan.cores[1].empty();
            std::printf("pool tasks: real split %s%s\n", plan.on ? "on" : "off: ", plan.why.c_str());
        }
        unset_env("STRATA_NUMA_SPLIT");
        if (!ok) { std::fprintf(stderr, "numa-split real-mode plan invariant broken\n"); return 1; }
        // the DEFAULT worker count under the split spans every socket (pool_plan.hpp span_packages): on a 2-socket
        // host the split must then be on, where 0.2.0's host-socket default alone would leave every thread on one node
        put_env("STRATA_NUMA_SPLIT", "1");
        {
            const int packages = (int) cpu::pool_plan_input(cpu::detect_cpu_topology(true, cpu::PoolAffinity::All))
                                     .cores_per_package.size();
            cpu::ExpertPool pool(0, true, true);
            const auto& plan = cpu::active_numa_split();
            std::printf("pool tasks: default pool under the split: %d workers on %d socket(s), split %s%s (%s)\n",
                        pool.workers(), packages, plan.on ? "on" : "off: ", plan.why.c_str(), pool.plan_limit());
            ok = packages != 2 || (plan.on && plan.cores[0].size() > 0 && plan.cores[1].size() > 0);
        }
        unset_env("STRATA_NUMA_SPLIT");
        if (!ok) { std::fprintf(stderr, "numa-split: the default pool on a 2-socket host did not span both nodes\n"); return 1; }
        // a value other than 1 / test is off, with the value in the reason; "0" and "" are off silently
        for (const char* v : {"off", "yes"}) {
            put_env("STRATA_NUMA_SPLIT", v);
            bool bad = false;
            {
                cpu::ExpertPool pool(2, true, true);
                bad = cpu::active_numa_split().on || cpu::active_numa_split().why.find(std::string("'") + v + "'") == std::string::npos;
            }
            unset_env("STRATA_NUMA_SPLIT");
            if (bad) { std::fprintf(stderr, "numa-split did not refuse STRATA_NUMA_SPLIT=%s\n", v); return 1; }
        }
    }
    {   // 0.2.2: STRATA_NUMA_POOL=0 is the off switch of the split, STRATA_NUMA_SPLIT=1 or test included; and with neither set the
        // plan is empty on a host with one NUMA node (the code path of the pool before the split) and, on two or more, on only
        // when it says why or says the nodes (the pool then runs as before)
        for (const char* v : {"1", "test"}) {
            put_env("STRATA_NUMA_POOL", "0");
            put_env("STRATA_NUMA_SPLIT", v);
            bool off = false;
            {
                cpu::ExpertPool pool(4, true, true);
                off = !cpu::active_numa_split().on && cpu::active_numa_split().why.empty() && pool.numa_slices() == 0;
            }
            unset_env("STRATA_NUMA_POOL");
            unset_env("STRATA_NUMA_SPLIT");
            if (!off) { std::fprintf(stderr, "STRATA_NUMA_POOL=0 did not switch the split off (STRATA_NUMA_SPLIT=%s)\n", v); return 1; }
        }
        unset_env("STRATA_NUMA_SPLIT");
        cpu::ExpertPool pool(4, true, true);
        const auto& plan = cpu::active_numa_split();
        std::printf("pool tasks: nothing set: split %s%s\n", plan.on ? "on" : "off", plan.why.empty() ? "" : (": " + plan.why).c_str());
        if (!plan.on && !plan.why.empty()) { std::fprintf(stderr, "an automatic split that did not run left a refusal reason\n"); return 1; }
    }
    {   // 0.2.2: a made-up two-node machine (STRATA_TEST_SYSFS_ROOT): the pool's own node lookup, the automatic split, its log lines
        const std::vector<int> cores = cpu::detect_cpu_topology(true, cpu::PoolAffinity::All).worker_cores;
        const int host = cpu::detect_cpu_topology(true, cpu::PoolAffinity::All).host_core;
        if (cores.size() < 4 || host < 0) {
            std::printf("pool tasks: fake two-node machine SKIPPED (needs 4 worker cores and a host core)\n");
        } else {
            namespace fs = std::filesystem;
            const fs::path root = fs::temp_directory_path() / ("strata_pool_fake_sysfs_" + std::to_string((long long) std::time(nullptr)));
            // node k of worker core i: the workers alternate between the nodes, the host core is on node 0
            auto build = [&](int n_nodes) {
                fs::remove_all(root);
                fs::create_directories(root / "devices/system/node");
                std::vector<std::string> list((size_t) n_nodes);
                auto put = [&](int cpu, int node) {
                    fs::create_directories(root / "devices/system/cpu" / ("cpu" + std::to_string(cpu)) / ("node" + std::to_string(node)));
                    list[(size_t) node] += (list[(size_t) node].empty() ? "" : ",") + std::to_string(cpu);
                };
                put(host, 0);
                for (size_t i = 0; i < cores.size(); ++i) put(cores[i], (int) (i % (size_t) n_nodes));
                for (int k = 0; k < n_nodes; ++k) {
                    fs::create_directories(root / "devices/system/node" / ("node" + std::to_string(k)));
                    std::ofstream(root / "devices/system/node" / ("node" + std::to_string(k)) / "cpulist") << list[(size_t) k] << "\n";
                }
            };
            put_env("STRATA_TEST_SYSFS_ROOT", root.string().c_str());
            unset_env("STRATA_NUMA_SPLIT");
            unset_env("STRATA_NUMA_POOL");
            const int workers = (int) std::min<size_t>(cores.size(), 8) & ~1;   // even: the nodes get the same count

            build(2);   // two nodes, nothing set: the split runs by itself
            {
                cpu::ExpertPool pool(workers, true, true);
                const auto& plan = cpu::active_numa_split();
                const bool ok = plan.on && !plan.test && plan.nodes[0] == 0 && plan.nodes[1] == 1 && pool.numa_slices() == 2 &&
                                !plan.cores[0].empty() && !plan.cores[1].empty();
                std::printf("pool tasks: fake two nodes, nothing set: split %s (%zu + %zu cores)\n", plan.on ? "on" : "off",
                            plan.cores[0].size(), plan.cores[1].size());
                if (!ok) { std::fprintf(stderr, "the automatic split did not run on a two-node machine\n"); return 1; }
            }
            put_env("STRATA_NUMA_POOL", "0");   // the off switch
            {
                cpu::ExpertPool pool(workers, true, true);
                if (cpu::active_numa_split().on || pool.numa_slices() != 0) { std::fprintf(stderr, "STRATA_NUMA_POOL=0 did not switch the automatic split off\n"); return 1; }
            }
            unset_env("STRATA_NUMA_POOL");
            put_env("STRATA_EXCHANGE_ROTATE", "1");   // the split cannot run: the pool stays as it was, with no half-made plan
            {
                cpu::ExpertPool pool(workers, true, true);
                if (cpu::active_numa_split().on || pool.numa_slices() != 0 || !cpu::active_numa_split().why.empty()) {
                    std::fprintf(stderr, "an automatic split that cannot run left a plan or a reason behind\n");
                    return 1;
                }
            }
            unset_env("STRATA_EXCHANGE_ROTATE");
            for (int nodes : {1, 3}) {   // one node: the pool before the split; three: not supported, the pool as it was
                build(nodes);
                cpu::ExpertPool pool(workers, true, true);
                if (cpu::active_numa_split().on || pool.numa_slices() != 0) { std::fprintf(stderr, "split on a %d-node machine\n", nodes); return 1; }
            }
            {   // the default worker count and cores of a one-node machine do not depend on the split code at all
                build(1);
                cpu::ExpertPool a(0, true, true);
                const std::vector<int> a_cores = a.worker_cores();
                const int a_workers = a.workers();
                unset_env("STRATA_TEST_SYSFS_ROOT");
                cpu::ExpertPool b(0, true, true);
                if (a_workers != b.workers() || a_cores != b.worker_cores()) {
                    std::fprintf(stderr, "a one-node fake machine changed the pool's workers or cores\n");
                    return 1;
                }
                std::printf("pool tasks: one node: %d workers and the same cores as with no topology override\n", a_workers);
            }
            unset_env("STRATA_TEST_SYSFS_ROOT");
            fs::remove_all(root);
        }
    }
#endif
    std::printf("pool tasks: %d bitwise comparisons passed, %d of them in NUMA-split mode (%d with modes 0-4 "
                "interleaved)\n", checks + split_checks + mixed_checks, split_checks + mixed_checks, mixed_checks);
    return 0;
}
