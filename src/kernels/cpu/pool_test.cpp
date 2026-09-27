// src/kernels/cpu/pool_test.cpp - P2.S3's test for the expert pool.
//
// The pool's correctness claims are small and specific, so they are checked directly rather than through a
// timing number:
//
//   1. THE SAME ANSWER AS SERIAL.  Every job's output must be bit-identical to running that expert serially.
//      Each worker owns its own `ExpertScratch`, and the jobs share one read-only activation, so there is no
//      legitimate source of difference - "close enough" here would be hiding a data race.
//   2. EVERY JOB RUNS EXACTLY ONCE.  Claiming is `head.fetch_add`, and an off-by-one in the bound is the
//      classic pool bug: it either drops a job or runs one twice.  Checked with a sentinel-filled output
//      buffer, so a dropped job is visible as an untouched slot rather than as a slightly wrong number.
//   3. REPEATED BATCHES.  A pool that works once and hangs or corrupts on the second `run()` is the failure
//      mode the park protocol exists to prevent, so `run()` is called many times in a row.
//   4. A BATCH BIGGER AND SMALLER THAN THE WORKER COUNT, because `n < workers` leaves most workers claiming
//      nothing and `n > workers` is the real case (10 experts, 5 workers).
//   5. (--stress, synthetic experts, no file) SLEEPING AND LATE WORKERS.  Idle workers sleep after `kIdleSpin`;
//      a phase opened while they sleep is often finished by the host alone before they wake, and a worker that
//      wakes then must not join it: the host may already be writing the next phase.  Thousands of batches, with
//      idle gaps either side of `kIdleSpin` and one-expert batches after the long ones, every output bitwise
//      against a serial run of the same kernels.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert.hpp"
#if defined(STRATA_NATIVE_EXPERTS)
#include "ggml.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    out.assign(cpu::BLOB, 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
#if defined(_MSC_VER)
    if (_fseeki64(f, index * (long long) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#else
    if (fseeko(f, (off_t) index * (off_t) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#endif
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---- 5: --stress
const float kSentinel = -1.2345e33f;

std::vector<float> gauss_vec(std::mt19937& rng, size_t n, float sd) {
    std::normal_distribution<float> g(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/// The pause before a batch: mostly none (generation's layers come back to back), sometimes up to 2 ms, and one
/// in 40 longer than `kIdleSpin`, so every worker is asleep when the batch is published.  True for a long one.
bool gap(std::mt19937& rng) {
    const unsigned r = rng() % 40;
    if (r == 0) {
        std::this_thread::sleep_for(cpu::ExpertPool::kIdleSpin + std::chrono::milliseconds(15));
        return true;
    }
    if (r < 5) std::this_thread::sleep_for(std::chrono::microseconds(100 + rng() % 1900));
    return false;
}

struct StressStats {
    long long batches = 0, outputs = 0, differ = 0;
    int long_gaps = 0;
    std::vector<double> ms_spinning, ms_after_sleep;   // one-expert batches
};

void stress_report(const char* what, const StressStats& s, uint32_t sleeps) {
    std::printf("  %-34s %s (%lld of %lld outputs differ, %lld batches; %d long gaps, %u worker sleeps)\n", what,
                s.differ ? "*** NO ***" : "identical to serial", s.differ, s.outputs, s.batches, s.long_gaps, sleeps);
    std::printf("  %-34s %.3f ms while the workers spin, %.3f ms right after they slept (median)\n",
                "  a one-expert batch", median(s.ms_spinning), median(s.ms_after_sleep));
}

/// `run()`: whole Q2_0 experts.  Random codes with finite scales: the protocol is under test, not the kernel.
StressStats stress_run(cpu::ExpertPool& pool, std::mt19937& rng, int iters) {
    const int NB = 4, NA = 4, MAXJ = 16;
    std::vector<std::vector<uint8_t>> blobs((size_t) NB, std::vector<uint8_t>(cpu::BLOB));
    for (auto& b : blobs) {
        for (auto& v : b) v = (uint8_t) rng();
        for (size_t o = cpu::O_GU_SCALES; o + 2 <= cpu::BLOB; o += 2) {
            const uint16_t h = (uint16_t) (0x2000 | (rng() & 0x3FF));   // fp16 in [2^-7, 2^-6)
            std::memcpy(b.data() + o, &h, 2);
        }
    }
    std::vector<cpu::ActQ> acts((size_t) NA);
    for (auto& a : acts) cpu::act_quant_q8_1(gauss_vec(rng, cpu::H, 1.0f).data(), cpu::H, a);
    std::vector<cpu::ExpertJob> jobs((size_t) MAXJ);
    std::vector<float> out((size_t) MAXJ * cpu::H), ref(cpu::H);
    cpu::ExpertScratch ws;
    StressStats s;
    for (int it = 0; it < iters; ++it) {
        const bool slept = gap(rng);
        s.long_gaps += slept;
        const bool tiny = slept ? rng() % 2 == 0 : rng() % 20 == 0;
        const int n = tiny ? 1 : 1 + (int) (rng() % MAXJ);
        for (int j = 0; j < n; ++j) {
            jobs[(size_t) j].blob = blobs[rng() % NB].data();
            jobs[(size_t) j].act = &acts[rng() % NA];
            jobs[(size_t) j].out = out.data() + (size_t) j * cpu::H;
            std::fill(jobs[(size_t) j].out, jobs[(size_t) j].out + cpu::H, kSentinel);
        }
        const double t0 = now_ms();
        pool.run(jobs.data(), n);
        const double ms = now_ms() - t0;
        if (tiny) (slept ? s.ms_after_sleep : s.ms_spinning).push_back(ms);
        for (int j = 0; j < n; ++j) {
            cpu::s2_expert_vnni_q(jobs[(size_t) j].blob, *jobs[(size_t) j].act, ref.data(), ws);
            s.differ += std::memcmp(ref.data(), jobs[(size_t) j].out, sizeof(float) * cpu::H) != 0;
        }
        s.outputs += n;
        ++s.batches;
    }
    return s;
}

#if defined(STRATA_NATIVE_EXPERTS)
/// `run_split_multi_native`: one layer's experts in GGUF formats, split by rows, several tokens per expert.
StressStats stress_native(cpu::ExpertPool& pool, std::mt19937& rng, int iters, ggml_type gu, ggml_type dn,
                          std::string& err) {
    StressStats s;
    cpu::NativeFmt f;
    if (!cpu::native_fmt((int) gu, (int) dn, cpu::H, cpu::FF, f, err)) return s;
    const int NB = 4, NA = 8, MAXE = 12, MAXNT = 4;
    // ggml's quantizers on random weights (valid blocks), one thread per expert: the i-quant ones are slow
    std::vector<std::vector<uint8_t>> blobs((size_t) NB, std::vector<uint8_t>(f.bytes));
    {
        std::vector<std::thread> th;
        for (int b = 0; b < NB; ++b)
            th.emplace_back([&f, &blobs, gu, dn, b, seed = rng()] {
                std::mt19937 r(seed);
                uint8_t* p = blobs[(size_t) b].data();
                ggml_quantize_chunk(gu, gauss_vec(r, (size_t) cpu::FF * cpu::H, 0.02f).data(), p, 0, cpu::FF, cpu::H, nullptr);
                ggml_quantize_chunk(gu, gauss_vec(r, (size_t) cpu::FF * cpu::H, 0.02f).data(), p + f.up_off, 0, cpu::FF,
                                    cpu::H, nullptr);
                ggml_quantize_chunk(dn, gauss_vec(r, (size_t) cpu::H * cpu::FF, 0.02f).data(), p + f.down_off, 0, cpu::H,
                                    cpu::FF, nullptr);
            });
        for (auto& t : th) t.join();
    }
    std::vector<std::vector<uint8_t>> acts((size_t) NA, std::vector<uint8_t>(cpu::kNativeActBytes));
    for (auto& a : acts) cpu::native_quant_act(f, gauss_vec(rng, cpu::H, 1.0f).data(), a.data());
    std::vector<cpu::ExpertJobMulti> jobs((size_t) MAXE);
    std::vector<float> out((size_t) MAXE * MAXNT * cpu::H), ref((size_t) MAXNT * cpu::H);
    std::vector<float> ff((size_t) MAXNT * cpu::FF);
    std::vector<uint8_t> hq((size_t) MAXNT * cpu::kNativeHBytes);
    for (int it = 0; it < iters; ++it) {
        const bool slept = gap(rng);
        s.long_gaps += slept;
        const bool tiny = slept ? rng() % 2 == 0 : rng() % 20 == 0;
        const int n = tiny ? 1 : 1 + (int) (rng() % MAXE);
        for (int e = 0; e < n; ++e) {
            cpu::ExpertJobMulti& j = jobs[(size_t) e];
            j = cpu::ExpertJobMulti{};
            j.blob = blobs[rng() % NB].data();
            j.nt = tiny ? 1 : 1 + (int) (rng() % MAXNT);
            int tok[NA];
            std::iota(tok, tok + NA, 0);
            std::shuffle(tok, tok + NA, rng);
            for (int t = 0; t < j.nt; ++t) {
                j.nact[t] = acts[(size_t) tok[t]].data();
                j.out[t] = out.data() + ((size_t) e * MAXNT + (size_t) t) * cpu::H;
                std::fill(j.out[t], j.out[t] + cpu::H, kSentinel);
            }
        }
        const double t0 = now_ms();
        pool.run_split_multi_native(f, jobs.data(), n);
        const double ms = now_ms() - t0;
        if (tiny) (slept ? s.ms_after_sleep : s.ms_spinning).push_back(ms);
        // the same kernels and token groups on this thread, every row at once
        for (int e = 0; e < n; ++e) {
            const cpu::ExpertJobMulti& j = jobs[(size_t) e];
            float* ffp[cpu::MAXT];
            float* rp[cpu::MAXT];
            const void* hqp[cpu::MAXT];
            for (int t = 0; t < j.nt; ++t) {
                ffp[t] = ff.data() + (size_t) t * cpu::FF;
                rp[t] = ref.data() + (size_t) t * cpu::H;
                hqp[t] = hq.data() + (size_t) t * cpu::kNativeHBytes;
            }
            cpu::native_gu_rows(f, j.blob, j.nact, j.nt, ffp, 0, cpu::FF);
            for (int t = 0; t < j.nt; ++t) cpu::native_quant_h(f, ffp[t], hq.data() + (size_t) t * cpu::kNativeHBytes);
            cpu::native_down_rows(f, j.blob, hqp, j.nt, rp, 0, cpu::H);
            for (int t = 0; t < j.nt; ++t) s.differ += std::memcmp(rp[t], j.out[t], sizeof(float) * cpu::H) != 0;
            s.outputs += j.nt;
        }
        ++s.batches;
    }
    return s;
}
#endif

int stress(int iters) {
    std::mt19937 rng(20260927);
    cpu::ExpertPool pool;
    std::printf("  %-34s %d + the host thread; they sleep after %lld ms without work\n", "workers", pool.workers(),
                (long long) cpu::ExpertPool::kIdleSpin.count());
    int bad = 0;
    if (cpu::cpu_features().usable()) {
        const uint32_t z0 = pool.sleeps();
        const StressStats s = stress_run(pool, rng, iters);
        stress_report("run(), whole Q2_0 experts", s, pool.sleeps() - z0);
        bad += s.differ != 0 || s.long_gaps == 0 || pool.sleeps() == z0;
    } else {
        std::printf("  run(): SKIPPED, the CPU lacks %s\n", cpu::cpu_features().reason());
    }
#if defined(STRATA_NATIVE_EXPERTS)
    const struct { ggml_type gu, dn; const char* name; } fmts[] = {
        {GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, "native rows, Q4_K / Q5_1"},             // ggml-cpu's dot products
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "native rows, IQ3_XXS / IQ4_NL"},   // + the AVX-512 i-quant kernel
    };
    for (const auto& ft : fmts) {
        std::string err;
        const uint32_t z0 = pool.sleeps();
        const StressStats s = stress_native(pool, rng, iters, ft.gu, ft.dn, err);
        if (!err.empty()) {
            std::printf("  %s: %s\n", ft.name, err.c_str());
            ++bad;
            continue;
        }
        stress_report(ft.name, s, pool.sleeps() - z0);
        bad += s.differ != 0 || s.long_gaps == 0 || pool.sleeps() == z0;
    }
#else
    std::printf("  native rows: SKIPPED, built without STRATA_NATIVE_EXPERTS\n");
#endif
    std::printf("\npool stress: %d failures\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    int stress_iters = 0;
    const char* path = "pack/full/experts.bin";
    long long layer = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--stress") stress_iters = 1500;
        else if (a == "--iters" && i + 1 < argc) stress_iters = std::atoi(argv[++i]);
        else if (a == "--file" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else {
            std::fprintf(stderr, "usage: pool_test [--selftest] [--file P] [--layer N]\n"
                                 "       pool_test --stress [--iters N]   (synthetic experts, no file)\n");
            return 2;
        }
    }
    if (stress_iters > 0) return stress(stress_iters);

    const cpu::CpuFeatures feat = cpu::cpu_features();
    if (!feat.usable()) {
        std::printf("  CPU lacks %s - the VNNI path cannot run here; pool test SKIPPED, not passed.\n",
                    feat.reason());
        std::printf("\npool: 0 failures, 1 SKIPPED\n");
        return 0;
    }

    int bad = 0;

    // ---- ten experts off one layer, which is exactly what a token uses
    const int NEXP = 10;
    std::vector<std::vector<uint8_t>> blobs((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e)
        if (!read_blob(path, layer * 512 + e, blobs[(size_t) e])) {
            std::fprintf(stderr, "cannot read expert %d of layer %lld from %s\n", e, layer, path);
            return 2;
        }

    std::mt19937 rng(99);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);

    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);

    // ---- serial reference
    std::vector<float> ref((size_t) NEXP * cpu::H);
    {
        cpu::ExpertScratch ws;
        for (int e = 0; e < NEXP; ++e)
            cpu::s2_expert_vnni_q(blobs[(size_t) e].data(), act, ref.data() + (size_t) e * cpu::H, ws);
    }

    // ---- the pool
    const std::vector<int> cores = cpu::physical_cores(true);
    const int hw = (int) std::thread::hardware_concurrency();
    std::printf("  %-44s %d logical, %d physical (skipping the first)\n", "cores the pool will use",
                hw, (int) cores.size());

    cpu::ExpertPool pool;
    std::printf("  %-44s %d\n", "workers", pool.workers());

    std::vector<cpu::ExpertJob> jobs((size_t) NEXP);
    std::vector<float> got((size_t) NEXP * cpu::H);
    std::vector<float> weights((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e) weights[(size_t) e] = 0.1f * (float) (e + 1);

    // ---- 1 + 2: same answer as serial, and every job exactly once.
    // The output buffer starts at a sentinel no real result can equal, so a job that never runs shows up as
    // an untouched slot instead of as a plausible number.
    const float SENTINEL = -1.2345e33f;
    for (int e = 0; e < NEXP; ++e) {
        jobs[(size_t) e].blob = blobs[(size_t) e].data();
        jobs[(size_t) e].act = &act;
        jobs[(size_t) e].out = got.data() + (size_t) e * cpu::H;
        jobs[(size_t) e].weight = weights[(size_t) e];
        jobs[(size_t) e].slot = e;
        std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
    }
    pool.run(jobs.data(), NEXP);

    long long not_run = 0, diff = 0;
    float worst = 0.f;
    for (int e = 0; e < NEXP; ++e)
        for (int i = 0; i < cpu::H; ++i) {
            const float a = ref[(size_t) e * cpu::H + i], b = got[(size_t) e * cpu::H + i];
            if (b == SENTINEL) { ++not_run; continue; }
            if (std::memcmp(&a, &b, 4) != 0) ++diff;
            worst = std::fmax(worst, std::fabs(a - b));
        }
    std::printf("  %-44s %s (%lld of %d outputs untouched)\n", "every job ran exactly once",
                not_run ? "*** NO ***" : "yes", not_run, NEXP * cpu::H);
    if (not_run) ++bad;
    // BIT-IDENTICAL, not "close".  Each worker has a private scratch and the activation is shared read-only,
    // so any difference at all is a race or a scratch collision - and a tolerance would hide exactly that.
    std::printf("  %-44s %s (%lld of %d differ, worst |d| %.3e)\n", "identical to the serial run",
                diff ? "*** NO ***" : "yes", diff, NEXP * cpu::H, (double) worst);
    if (diff) ++bad;

    // ---- 3: repeated batches.  A park protocol that works once and corrupts on the second call is the
    //        exact failure this loop is here to catch.
    {
        int repeats_bad = 0;
        for (int r = 0; r < 200; ++r) {
            for (int e = 0; e < NEXP; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), NEXP);
            for (int e = 0; e < NEXP && !repeats_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++repeats_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (200 consecutive batches)\n", "repeated batches stay correct",
                    repeats_bad ? "*** NO ***" : "yes");
        if (repeats_bad) ++bad;
    }

    // ---- 4: batch sizes either side of the worker count
    {
        int size_bad = 0;
        for (int n : {1, 2, pool.workers() - 1 > 0 ? pool.workers() - 1 : 1, pool.workers(),
                      pool.workers() + 1, NEXP}) {
            if (n < 1 || n > NEXP) continue;
            for (int e = 0; e < n; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), n);
            for (int e = 0; e < n && !size_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++size_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (1, w-1, w, w+1, 10)\n", "batch sizes around the worker count",
                    size_bad ? "*** NO ***" : "yes");
        if (size_bad) ++bad;
    }

    // ---- and the number that matters for the ledger: c for one layer, 10 experts over 48 layers.
    //
    // BEST OF N, not a mean.  The first version divided a 10-run total by 10 and reported 6.604 ms per layer -
    // 2.1 GB/s against L9's measured 42.55 - purely because a CUDA build was running on the same 6-core
    // machine at the time.  A mean over a contended machine measures the contention; a best-of measures the
    // instrument, and every other timing tool in this project already takes the best.  The spread is printed
    // so a contended run is visible rather than being read as a regression.
    {
        const int REPS = 20;
        std::vector<double> t((size_t) REPS, 0.0);
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now_ms();
            pool.run(jobs.data(), NEXP);
            t[(size_t) r] = now_ms() - t0;
        }
        std::vector<double> sorted = t;
        std::sort(sorted.begin(), sorted.end());
        const double best = sorted.front(), median = sorted[sorted.size() / 2], worst = sorted.back();
        const double gbs = (double) NEXP * cpu::BLOB / (best * 1e-3) / 1e9;
        std::printf("\n  %-44s %7.3f ms   (%.1f GB/s)\n", "10 experts, one layer (best of 20)",
                    best, gbs);
        std::printf("  %-44s %7.3f / %7.3f ms   (spread %.2fx)\n", "median / worst", median, worst,
                    median / (best > 0 ? best : 1));
        std::printf("  %-44s %7.3f ms   -> %.1f tok/s for the full 48 layers\n",
                    "extrapolated to 48 layers", best * 48, 1000.0 / (best * 48));
        std::printf("  L9 measured 42.55 GB/s on 6 cores; this pool uses %d workers (core 0 is left to the\n",
                    pool.workers());
        std::printf("  host loop), so %.1f GB/s x 6/%d = %.1f GB/s is the per-core comparison.\n",
                    gbs, pool.workers(), gbs * 6.0 / pool.workers());
        if (median / (best > 0 ? best : 1) > 1.5)
            std::printf("  *** the spread is over 1.5x: this machine was CONTENDED and the median is not a\n"
                        "      property of the pool.  Re-run on a quiet machine before quoting it. ***\n");
    }

    std::printf("\npool: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("pool_test OK\n");
    return 0;
}
