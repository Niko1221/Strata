// src/kernels/cpu/pool_quant_test.cpp - PR #500: the intermediate quantization of a multi-token layer, as a phase
// of the pool (drain mode 7) instead of a loop on the host with every worker parked.
//
// The step is small and easy to get subtly wrong, so it is checked three ways rather than one:
//
//   1. PARALLEL vs SEQUENTIAL, BIT FOR BIT.  `STRATA_POOL_PARALLEL_QUANT=0` runs the sequential host loop this
//      replaces; both arms quantize the same (expert, token) pair with the same function and the same buffers,
//      so ANY difference at all is a scheduling bug - a task mapped to the wrong pair, a phase published before
//      its inputs were complete, or two tasks writing the same bytes.  A tolerance here would hide exactly that.
//   2. THE POOL vs A SERIAL REFERENCE, so that the values are still the ones the single-token path produces: for
//      a Q2_0 pack, `s2_expert_vnni_q` per token - the contract the row-range kernels are documented to keep -
//      and for a native pack the same primitives with one thread running the FULL row range (rows are
//      independent of the range they are computed in, so that is bitwise the pool's answer).
//   3. EVERY WRITTEN ROW IS WRITTEN, against a sentinel: a task that never runs leaves the sentinel behind, and a
//      task that runs twice disagrees with the reference.
//
// The cases cover one expert and a full batch (kMaxSplitMulti), fewer tasks than threads and more, different
// token counts per expert (a verify window routes ~1.5 tokens to each expert, not the same number to all), the
// maximum task count (kMaxSplitMulti x MAXT), the fallback for more experts than one batch, pool sizes from one
// worker up, and a stretch with the workers asleep between phases - issue #29's timing hazard, applied to the
// phase this adds.
//
//   pool_quant_test             the checks above, one line each
//   pool_quant_test --bench     the measurement behind `quant_min_tasks()`: the quantization phase's own time
//                               (`ms_multi_q`, which the pool accumulates around exactly this step) for both
//                               arms, over a range of task counts and three pool sizes
//
// A native pack needs `STRATA_NATIVE_EXPERTS`; without it only the Q2_0 cases run (the build has no ggml-cpu).
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(STRATA_NATIVE_EXPERTS)
#include "ggml.h"   // a synthetic native pack is quantized by ggml itself, as native_expert_parity does
#endif

#if defined(_WIN32)
#define set_env(name, value) _putenv_s(name, value)
#else
#define set_env(name, value) setenv(name, value, 1)
#endif

namespace cpu = strata::kernels::cpu;

namespace {

/// A row value no real result can be, so a task that never runs is visible as an untouched row.
constexpr float kSentinel = -1.2345e33f;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---- one case: its jobs (blob, activations, one output row per token) and the rows themselves

struct Rows {
    std::vector<float> v;
    float* at(int e, int t) { return v.data() + ((size_t) e * cpu::MAXT + (size_t) t) * cpu::H; }
    const float* at(int e, int t) const { return v.data() + ((size_t) e * cpu::MAXT + (size_t) t) * cpu::H; }
    void reset(int n) { v.assign((size_t) n * cpu::MAXT * cpu::H, kSentinel); }
};

struct Case {
    int n = 0;
    std::vector<int> nt;                        ///< tokens per expert, in job order (each 1..MAXT)
    std::vector<cpu::ExpertJobMulti> jobs;

    void build() {
        jobs.assign((size_t) n, cpu::ExpertJobMulti{});
        for (int e = 0; e < n; ++e) jobs[(size_t) e].nt = nt[(size_t) e];
    }
    int tasks() const {
        int t = 0;
        for (int e = 0; e < n; ++e) t += nt[(size_t) e];
        return t;
    }
};

/// Point every task's output at its own row of `r`, and leave the sentinel in every row a task does not write.
void point_outputs(Case& c, Rows& r) {
    r.reset(c.n);
    for (int e = 0; e < c.n; ++e)
        for (int t = 0; t < c.nt[(size_t) e]; ++t) c.jobs[(size_t) e].out[(size_t) t] = r.at(e, t);
}

struct Cmp {
    long long cells = 0, diff = 0, missing = 0;
    bool ok() const { return diff == 0 && missing == 0; }
};

Cmp compare(const Case& c, const Rows& ref, const Rows& got) {
    Cmp x;
    for (int e = 0; e < c.n; ++e)
        for (int t = 0; t < c.nt[(size_t) e]; ++t) {
            const float* a = ref.at(e, t);
            const float* b = got.at(e, t);
            bool written = false;
            for (int i = 0; i < cpu::H; ++i) {
                ++x.cells;
                if (b[i] == kSentinel) continue;
                written = true;
                if (std::memcmp(&a[i], &b[i], 4) != 0) ++x.diff;
            }
            if (!written) ++x.missing;
        }
    return x;
}

int report(const char* what, const Case& c, const Cmp& x) {
    std::printf("  %-52s %s (%d experts, %d tasks, %lld cells, %lld differ, %lld rows not written)\n", what,
                x.ok() ? "yes" : "*** NO ***", c.n, c.tasks(), x.cells, x.diff, x.missing);
    return x.ok() ? 0 : 1;
}

// ---- the pools.  `parallel_quant_` and `quant_min_tasks_` are read at construction (like STRATA_POOL_SPIN_US),
//      so the two arms need two pools, and pinned workers of one arm must not be alive under the other's.
struct PoolOpts {
    bool parallel = true;
    int workers = 0;        ///< 0 = every physical core but the host's
    int min_tasks = 1;      ///< force the phase on for every count, so the checks cover both arms
    int spin_us = -1;       ///< -1 = the pool's default 20 ms before a parked worker sleeps
};

std::unique_ptr<cpu::ExpertPool> make_pool(const PoolOpts& o) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%d", o.min_tasks);
    set_env("STRATA_POOL_QUANT_MIN_TASKS", buf);
    set_env("STRATA_POOL_PARALLEL_QUANT", o.parallel ? "1" : "0");
    if (o.spin_us >= 0) {
        std::snprintf(buf, sizeof buf, "%d", o.spin_us);
        set_env("STRATA_POOL_SPIN_US", buf);
    } else {
        set_env("STRATA_POOL_SPIN_US", "");
    }
    auto pool = std::make_unique<cpu::ExpertPool>(o.workers, /*pin=*/true, /*host_works=*/true);
    if (pool->parallel_quant() != o.parallel || pool->quant_min_tasks() != o.min_tasks) {
        std::fprintf(stderr, "pool_quant_test: the pool did not take the settings (parallel %d, min %d)\n",
                     (int) pool->parallel_quant(), pool->quant_min_tasks());
        std::exit(2);
    }
    return pool;
}

// ---- the Q2_0 (canonical) path

void make_s2_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);   // fp16 scales ~0.004-0.016
        std::memcpy(b + i, &h, 2);
    }
}

/// The serial reference: the whole-expert single-token kernel, per token.  The row-range path the pool uses is
/// documented to be bitwise this, so a difference is the pool's - not a tolerance to widen.
void legacy_reference(const Case& c, const std::vector<cpu::ActQ>& acts) {
    cpu::ExpertScratch ws;
    for (int e = 0; e < c.n; ++e)
        for (int t = 0; t < c.nt[(size_t) e]; ++t) {
            const cpu::ExpertJobMulti& j = c.jobs[(size_t) e];
            cpu::s2_expert_vnni_q(j.blob, *j.act[(size_t) t], j.out[(size_t) t], ws);
        }
}

int check_legacy(const char* label, const std::vector<int>& nt, const std::vector<cpu::ActQ>& acts, int workers,
                 bool stress) {
    Case c;
    c.n = (int) nt.size();
    c.nt = nt;
    c.build();
    std::mt19937 rng(500 + c.n);
    std::vector<std::vector<uint8_t>> blobs((size_t) c.n, std::vector<uint8_t>(cpu::BLOB));
    for (int e = 0; e < c.n; ++e) make_s2_blob(blobs[(size_t) e].data(), rng);
    for (int e = 0; e < c.n; ++e) {
        c.jobs[(size_t) e].blob = blobs[(size_t) e].data();
        for (int t = 0; t < c.nt[(size_t) e]; ++t) c.jobs[(size_t) e].act[(size_t) t] = &acts[(size_t) t];
    }

    Rows ref, seq_rows, par_rows;
    point_outputs(c, ref);
    legacy_reference(c, acts);

    int bad = 0;
    long long seq_phases = 0, par_phases = 0, par_tasks = 0;
    {
        auto pool = make_pool({false, workers, 1, -1});
        point_outputs(c, seq_rows);
        pool->run_split_multi(c.jobs.data(), c.n);
        seq_phases = pool->quant_phases;
    }
    {
        auto pool = make_pool({true, workers, 1, -1});
        point_outputs(c, par_rows);
        pool->run_split_multi(c.jobs.data(), c.n);
        par_phases = pool->quant_phases;
        par_tasks = pool->quant_tasks;
    }
    const bool fallback = c.n > cpu::ExpertPool::kMaxSplitMulti;
    const bool arms_ok = seq_phases == 0 && (fallback ? par_phases == 0 : par_phases == 1 && par_tasks == c.tasks());
    std::printf("  %s: %d experts, %d tasks, sequential %lld phase(s), parallel %lld phase(s) / %lld tasks%s - %s\n",
                label, c.n, c.tasks(), seq_phases, par_phases, par_tasks,
                fallback ? ", single-token fallback" : "", arms_ok ? "as asked" : "*** NOT AS ASKED ***");
    if (!arms_ok) ++bad;
    bad += report("sequential (STRATA_POOL_PARALLEL_QUANT=0) vs the reference", c, compare(c, ref, seq_rows));
    bad += report("parallel (mode 7) vs the reference", c, compare(c, ref, par_rows));
    bad += report("parallel vs sequential, bit for bit", c, compare(c, seq_rows, par_rows));

    // ---- with every worker asleep (issue #29's timing hazard): a worker counted as parked may be about to wake
    //      on a NEW epoch, so the host's task table must be PUBLISHED, not merely written.  The pool's default
    //      spin is 20 ms, so STRATA_POOL_SPIN_US=1 puts a sleeper at every phase boundary.
    if (stress) {
        auto pool = make_pool({true, workers, 1, 1});
        long long mismatches = 0;
        for (int r = 0; r < 25; ++r) {
            point_outputs(c, par_rows);
            pool->run_split_multi(c.jobs.data(), c.n);
            if (!compare(c, ref, par_rows).ok()) ++mismatches;
        }
        std::printf("  %-52s %s (25 batches, workers asleep throughout)\n", "   the same with the workers asleep",
                    mismatches ? "*** NO ***" : "yes");
        bad += mismatches ? 1 : 0;
    }
    return bad;
}

#if defined(STRATA_NATIVE_EXPERTS)

// ---- a native (GGUF-form) path: the activation and the weights of one layer, quantized by ggml

void native_reference(const cpu::NativeFmt& f, const Case& c, const std::vector<std::vector<uint8_t>>& nact) {
    std::vector<float> ff((size_t) cpu::MAXT * cpu::FF);
    std::vector<cpu::ActQ> a2((size_t) cpu::MAXT);
    std::vector<std::vector<uint8_t>> hq((size_t) cpu::MAXT, std::vector<uint8_t>(cpu::kNativeHBytes));
    for (int e = 0; e < c.n; ++e) {
        const cpu::ExpertJobMulti& j = c.jobs[(size_t) e];
        const int nt = j.nt;
        const void* ap[cpu::MAXT];
        float* ffp[cpu::MAXT];
        for (int t = 0; t < nt; ++t) {
            ap[t] = nact[(size_t) t].data();
            ffp[t] = ff.data() + (size_t) t * cpu::FF;
        }
        cpu::native_gu_rows(f, j.blob, ap, nt, ffp, 0, cpu::FF);   // mode 5, one thread, the full row range
        if (f.d_type == 42) {                                      // mode 7 + 6 on the Q2_0 down kernels
            const cpu::ActQ* a2p[cpu::MAXT];
            for (int t = 0; t < nt; ++t) {
                cpu::act_quant_any(ff.data() + (size_t) t * cpu::FF, cpu::FF, a2[(size_t) t]);
                a2p[t] = &a2[(size_t) t];
            }
            cpu::q2_rows_any(j.blob + f.down_off, f.d_row, (int) (f.n_ff / 64), a2p, nt, j.out, 0, cpu::H);
        } else {                                                   // mode 7 + 6 on ggml-cpu
            const void* hp[cpu::MAXT];
            for (int t = 0; t < nt; ++t) {
                cpu::native_quant_h(f, ff.data() + (size_t) t * cpu::FF, hq[(size_t) t].data());
                hp[t] = hq[(size_t) t].data();
            }
            cpu::native_down_rows(f, j.blob, hp, nt, j.out, 0, cpu::H);
        }
    }
}

int check_native(const char* label, int gu_type, int d_type, const std::vector<int>& nt, int distinct_blobs,
                 int workers, bool stress) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, cpu::H, cpu::FF, f, err)) {
        std::printf("  %-52s skipped (%s)\n", label, err.c_str());
        return 0;
    }
    Case c;
    c.n = (int) nt.size();
    c.nt = nt;
    c.build();

    // The weights: ggml quantizes random floats into the layer's formats, which is exactly the byte layout a
    // native pack holds for that (gu_type, d_type).  Gate and up are two matrices of FF rows of H; down is H rows
    // of FF.  Distinct blobs per expert, so a task that takes the wrong expert's blob cannot pass.
    std::mt19937 rng(700 + gu_type * 4 + d_type);
    std::normal_distribution<float> nd(0.f, 0.4f);
    std::vector<float> w((size_t) cpu::FF * cpu::H), wd((size_t) cpu::H * cpu::FF);
    std::vector<std::vector<uint8_t>> blobs((size_t) distinct_blobs, std::vector<uint8_t>(f.bytes));
    for (auto& b : blobs) {
        for (auto& v : w) v = nd(rng);
        for (auto& v : wd) v = nd(rng);
        ggml_quantize_chunk((ggml_type) gu_type, w.data(), b.data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk((ggml_type) gu_type, w.data(), b.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk((ggml_type) d_type, wd.data(), b.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    }
    for (int e = 0; e < c.n; ++e) c.jobs[(size_t) e].blob = blobs[(size_t) (e % distinct_blobs)].data();

    const int maxt = *std::max_element(nt.begin(), nt.end());
    std::vector<std::vector<float>> x((size_t) maxt, std::vector<float>(cpu::H));
    std::vector<std::vector<uint8_t>> nact((size_t) maxt, std::vector<uint8_t>(cpu::kNativeActBytes));
    for (int t = 0; t < maxt; ++t) {
        std::normal_distribution<float> nx(0.f, 1.f);
        for (auto& v : x[(size_t) t]) v = nx(rng);
        cpu::native_quant_act(f, x[(size_t) t].data(), nact[(size_t) t].data());
    }
    for (int e = 0; e < c.n; ++e) {
        cpu::ExpertJobMulti& j = c.jobs[(size_t) e];
        for (int t = 0; t < j.nt; ++t) {
            j.act[(size_t) t] = nullptr;                 // a native layer's gate/up takes `nact`, not an ActQ
            j.nact[(size_t) t] = nact[(size_t) t].data();
        }
    }

    Rows ref, seq_rows, par_rows;
    point_outputs(c, ref);
    native_reference(f, c, nact);

    int bad = 0;
    long long seq_phases = 0, par_phases = 0, par_tasks = 0;
    {
        auto pool = make_pool({false, workers, 1, -1});
        point_outputs(c, seq_rows);
        pool->run_split_multi_native(f, c.jobs.data(), c.n);
        seq_phases = pool->quant_phases;
    }
    {
        auto pool = make_pool({true, workers, 1, -1});
        point_outputs(c, par_rows);
        pool->run_split_multi_native(f, c.jobs.data(), c.n);
        par_phases = pool->quant_phases;
        par_tasks = pool->quant_tasks;
    }
    std::printf("  %s: %d experts, %d tasks\n", label, c.n, c.tasks());
    const bool arms_ok = seq_phases == 0 && par_phases == 1 && par_tasks == c.tasks();
    std::printf("  %-52s %s (sequential %lld phases, parallel %lld phase(s) / %lld tasks)\n",
                "   both arms ran the one they were asked to", arms_ok ? "yes" : "*** NO ***", seq_phases,
                par_phases, par_tasks);
    if (!arms_ok) ++bad;
    bad += report("   parallel (mode 7) vs the serial full-range reference", c, compare(c, ref, par_rows));
    bad += report("   sequential (STRATA_POOL_PARALLEL_QUANT=0) vs the reference", c, compare(c, ref, seq_rows));
    bad += report("   parallel vs sequential, bit for bit", c, compare(c, seq_rows, par_rows));

    if (stress) {
        auto pool = make_pool({true, workers, 1, 1});
        long long mismatches = 0;
        for (int r = 0; r < 8; ++r) {
            point_outputs(c, par_rows);
            pool->run_split_multi_native(f, c.jobs.data(), c.n);
            if (!compare(c, ref, par_rows).ok()) ++mismatches;
        }
        std::printf("  %-52s %s (8 batches, workers asleep throughout)\n",
                    "   the same with the workers asleep", mismatches ? "*** NO ***" : "yes");
        bad += mismatches ? 1 : 0;
    }
    return bad;
}
#endif  // STRATA_NATIVE_EXPERTS

// ---- the measurement behind `quant_min_tasks()`

int run_bench() {
    // best-of, like every other timing in this project: a mean over a contended machine measures the contention.
    // The phase is a couple of microseconds, so the count is high enough that the host loop between phase 3 and
    // the next `run_split_multi` cannot land inside the measurement.
    constexpr int BENCH_REPS = 40;
    const int TASKS[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96};
    const int NT = (int) (sizeof(TASKS) / sizeof(TASKS[0]));
    const int WORKERS[] = {0, 6, 2};
    const int NW = (int) (sizeof(WORKERS) / sizeof(WORKERS[0]));

    // the per-task cost, best of many: one 640-element activation quantization, which is what one task is
    {
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> ff(cpu::FF);
        for (auto& v : ff) v = nd(rng);
        cpu::ActQ a2;
        double best = 1e9;
        for (int r = 0; r < 2000; ++r) {
            const double t0 = now_ms();
            cpu::act_quant_q8_1(ff.data(), cpu::FF, a2);
            best = std::fmin(best, now_ms() - t0);
        }
        std::printf("one task (act_quant_q8_1 of %d floats): best of 2000 = %.3f us\n", cpu::FF, best * 1e3);
    }

    const int max_tasks = TASKS[NT - 1];
    std::mt19937 rng(17);
    std::vector<std::vector<uint8_t>> blobs((size_t) max_tasks, std::vector<uint8_t>(cpu::BLOB));
    for (auto& b : blobs) make_s2_blob(b.data(), rng);
    std::vector<float> x((size_t) cpu::H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);

    Case c;
    c.n = max_tasks;
    c.nt.assign((size_t) max_tasks, 1);   // one token per expert: tasks == experts, so the count is the x axis
    c.build();
    for (int e = 0; e < c.n; ++e) {
        c.jobs[(size_t) e].blob = blobs[(size_t) e].data();
        c.jobs[(size_t) e].act[0] = &act;
    }
    Rows rows;

    std::vector<std::vector<double>> seq_us((size_t) NW), par_us((size_t) NW);
    std::vector<int> workers((size_t) NW), threads((size_t) NW);
    for (int wi = 0; wi < NW; ++wi) {
        for (int arm = 0; arm < 2; ++arm) {
            auto pool = make_pool({arm == 1, WORKERS[wi], 1, -1});
            workers[(size_t) wi] = pool->workers();
            threads[(size_t) wi] = pool->workers() + (pool->host_works() ? 1 : 0);
            std::vector<double>& out = arm == 1 ? par_us[(size_t) wi] : seq_us[(size_t) wi];
            for (int ti = 0; ti < NT; ++ti) {
                point_outputs(c, rows);
                double best = 1e9;
                for (int r = 0; r < BENCH_REPS; ++r) {
                    const double q0 = pool->ms_multi_q;
                    pool->run_split_multi(c.jobs.data(), TASKS[ti]);
                    best = std::fmin(best, pool->ms_multi_q - q0);
                }
                out.push_back(best * 1e3);   // us
            }
        }
    }
    std::printf("\nthe intermediate quantization alone (ms_multi_q, best of %d; a batch of T experts with one token\n"
                "each, so tasks == T)\n%8s", BENCH_REPS, "tasks");
    for (int wi = 0; wi < NW; ++wi)
        std::printf(" | %7s(w=%d,t=%d) %7s %7s", "seq us", workers[(size_t) wi], threads[(size_t) wi], "par us",
                    "faster");
    std::printf("\n");
    for (int ti = 0; ti < NT; ++ti) {
        std::printf("%8d", TASKS[ti]);
        for (int wi = 0; wi < NW; ++wi) {
            const double s = seq_us[(size_t) wi][(size_t) ti], p = par_us[(size_t) wi][(size_t) ti];
            std::printf(" | %13.1f %7.1f %6.2fx", s, p, p > 0 ? s / p : 0.0);
        }
        std::printf("\n");
    }
    std::printf("\nRead it off the table rather than guessing: below the crossover the parallel arm pays for a park\n"
                "barrier it cannot earn back, which is what `quant_min_tasks()` keeps the sequential loop for.  These\n"
                "are measurements on the machine this runs on, not properties of the code.\n");
    return 0;
}

int run_checks() {
    const cpu::CpuFeatures feat = cpu::cpu_features();
    if (!feat.usable()) {
        std::printf("  CPU lacks %s - the VNNI path cannot run here; pool_quant_test SKIPPED, not passed.\n",
                    feat.reason());
        return 0;
    }

    // one activation per token of the verify window, shared by every expert (which is what the engine passes)
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<float>> x((size_t) cpu::MAXT, std::vector<float>(cpu::H));
    std::vector<cpu::ActQ> acts((size_t) cpu::MAXT);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (auto& v : x[(size_t) t]) v = nd(rng);
        cpu::act_quant_q8_1(x[(size_t) t].data(), cpu::H, acts[(size_t) t]);
    }

    int bad = 0;
    std::printf("Q2_0 (canonical) layers - the pool's run_split_multi:\n");
    bad += check_legacy("one expert, one token (the smallest batch)", {1}, acts, 0, false);
    bad += check_legacy("one expert, a full window", {8}, acts, 0, false);
    bad += check_legacy("fewer tasks than threads (3)", {1, 1, 1}, acts, 0, false);
    bad += check_legacy("one task per thread (12)", std::vector<int>(12, 1), acts, 0, false);
    bad += check_legacy("a full window each (12 x 8 = 96 tasks)", std::vector<int>(12, 8), acts, 0, false);
    bad += check_legacy("different token counts per expert", {1, 2, 3, 5, 8, 4, 7, 6}, acts, 0, true);
    bad += check_legacy("the same, on two workers", {1, 2, 3, 5, 8, 4, 7, 6}, acts, 2, false);
    bad += check_legacy("the same, on one worker", {1, 2, 3, 5, 8, 4, 7, 6}, acts, 1, false);
    bad += check_legacy("a full batch: kMaxSplitMulti experts", std::vector<int>(cpu::ExpertPool::kMaxSplitMulti, 2),
                        acts, 0, false);
    bad += check_legacy("the maximum task count: every expert at MAXT tokens",
                        std::vector<int>(cpu::ExpertPool::kMaxSplitMulti,
                                         cpu::ExpertPool::kMaxQuantTasks / cpu::ExpertPool::kMaxSplitMulti),
                        acts, 0, false);
    bad += check_legacy("more experts than one batch (the fallback)", std::vector<int>(97, 2), acts, 0, false);

#if defined(STRATA_NATIVE_EXPERTS)
    std::printf("native (GGUF-form) layers - the pool's run_split_multi_native:\n");
    // IQ3_XXS gate/up with a Q2_0 down: mode 7 -> act_quant_any, mode 6 -> the Q2_0 down kernels
    bad += check_native("an IQ3_XXS / Q2_0 layer, one expert, a full window", 18, 42, {8}, 1, 0, false);
    bad += check_native("an IQ3_XXS / Q2_0 layer, mixed windows", 18, 42, {1, 2, 8, 3}, 4, 0, true);
    // IQ2_S gate/up with an IQ4_NL down: mode 7 -> native_quant_h, mode 6 -> ggml-cpu's down rows
    bad += check_native("an IQ2_S / IQ4_NL layer, one token", 22, 20, {1}, 1, 0, false);
    bad += check_native("an IQ2_S / IQ4_NL layer, mixed windows", 22, 20, {8, 1, 3, 8, 2, 5}, 6, 2, true);
    // IQ3_S gate/up, a whole batch of experts (one blob is reused, so this stays cheap)
    bad += check_native("an IQ3_S / IQ4_NL layer, a full batch", 21, 20,
                        std::vector<int>(cpu::ExpertPool::kMaxSplitMulti, 3), 4, 0, false);
#else
    std::printf("native layers: not built (STRATA_NATIVE_EXPERTS off), so no native case ran\n");
#endif

    std::printf("\npool_quant_test: %d failures\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") bench = true;
        else {
            std::fprintf(stderr, "usage: pool_quant_test [--bench]\n");
            return 2;
        }
    }
    return bench ? run_bench() : run_checks();
}
