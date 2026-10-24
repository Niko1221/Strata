// src/kernels/cpu/pool_warm_test.cpp - ExpertPool::warm_post + run_split_multi_native give the same bits as the plain call.
// Random Q4_K/Q4_K experts (Sage geometry), 1-3 tokens per job; reference = no warm set; then warm sets of 1..K experts, with a
// pause (the workers finish loading), with none (the call aborts the load), a warm set for other experts, and repeated.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
namespace cpu = strata::kernels::cpu;
static constexpr int64_t kH = 2048, kFF = 512;
static void make_blob(const cpu::NativeFmt& f, uint8_t* blob, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) std::max(kH, kFF));
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) f.d_type);
    for (int64_t r = 0; r < 2 * kFF; ++r) { for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng); tg->from_float(row.data(), blob + (size_t) r * f.gu_row, kH); }
    for (int64_t r = 0; r < kH; ++r) { for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng); td->from_float(row.data(), blob + f.down_off + (size_t) r * f.d_row, kFF); }
}
// returns 77 (ctest SKIP) when no pool of this machine can warm, 1 on a mismatch or a warm path that never ran, else 0
static int run_pool(int workers, const cpu::NativeFmt& f, std::vector<std::vector<uint8_t>>& blobs, std::vector<std::vector<uint8_t>>& act) {
    const int E = 24, MAXJ = 8;
    std::mt19937 rng(11);
    cpu::ExpertPool pool(workers);
    pool.enable_warm(true);
    if (!pool.warm_enabled()) { printf("pool_warm_test: %d workers: the pool cannot warm here (unpinned workers, a NUMA split or one worker)\n", workers); return 77; }
    std::vector<std::vector<float>> out_ref((size_t) E * 3, std::vector<float>(kH)), out((size_t) E * 3, std::vector<float>(kH));
    auto run = [&](const std::vector<int>& ids, std::vector<std::vector<float>>& dst) {
        std::vector<cpu::ExpertJobMulti> jobs;
        for (int e : ids) {
            cpu::ExpertJobMulti j; j.blob = blobs[e].data(); j.nt = 1 + e % 3;
            for (int t = 0; t < j.nt; ++t) { j.nact[t] = act[t].data(); j.out[t] = dst[(size_t) e * 3 + t].data(); }
            jobs.push_back(j);
        }
        pool.run_split_multi_native(f, jobs.data(), (int) jobs.size());
    };
    int bad = 0, runs = 0;
    std::mt19937 pick(5);
    for (int it = 0; it < 400; ++it) {
        std::vector<int> ids; int nj = 1 + pick() % MAXJ;
        while ((int) ids.size() < nj) { int e = pick() % E; bool dup = false; for (int x : ids) dup |= x == e; if (!dup) ids.push_back(e); }
        for (auto& v : out_ref) std::fill(v.begin(), v.end(), 0.f);
        for (auto& v : out) std::fill(v.begin(), v.end(), 0.f);
        run(ids, out_ref);   // no warm set
        const int mode = it % 4;   // 0 warm subset + wait, 1 warm subset no wait, 2 warm other experts + wait, 3 warm all + wait
        std::vector<const uint8_t*> wb;
        if (mode == 2) { for (int e = 0; e < E && (int) wb.size() < 3; ++e) { bool in = false; for (int x : ids) in |= x == e; if (!in) wb.push_back(blobs[e].data()); } }
        else for (size_t i = 0; i < ids.size() && (int) wb.size() < (mode == 3 ? 8 : 1 + (int) (pick() % 3)); ++i) wb.push_back(blobs[ids[i]].data());
        pool.warm_post(f, wb.data(), (int) wb.size());
        if (mode != 1) std::this_thread::sleep_for(std::chrono::microseconds(300));
        run(ids, out);
        for (int e : ids) for (int t = 0; t < 1 + e % 3; ++t)
            if (std::memcmp(out_ref[(size_t) e * 3 + t].data(), out[(size_t) e * 3 + t].data(), kH * 4) != 0) { ++bad; printf("MISMATCH it %d mode %d expert %d tok %d\n", it, mode, e, t); }
        ++runs;
    }
    printf("pool_warm_test: %d workers asked, %d running: %d runs, %d mismatches, warm sets posted %lld, complete at the call %lld; calls none/some/all %lld/%lld/%lld\n", workers, pool.workers(), runs, bad,
           (long long) pool.warm_posts, (long long) pool.warm_full, (long long) pool.wc_calls[0], (long long) pool.wc_calls[1], (long long) pool.wc_calls[2]);
    if (pool.warm_full == 0 || pool.wc_calls[1] + pool.wc_calls[2] == 0) { printf("pool_warm_test: %d workers: the warm path never ran\n", workers); return 1; }
    return bad ? 1 : 0;
}

int main() {
    ggml_cpu_init();
    cpu::NativeFmt f; std::string err;
    if (!cpu::native_fmt(12, 12, kH, kFF, f, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int E = 24;
    std::mt19937 rng(11);
    std::vector<std::vector<uint8_t>> blobs(E, std::vector<uint8_t>(f.bytes + 64));
    for (int e = 0; e < 4; ++e) make_blob(f, blobs[e].data(), rng);
    for (int e = 4; e < E; ++e) blobs[e] = blobs[e % 4];
    for (int e = 0; e < E; ++e) { std::mt19937 r2(100 + e);   // distinct weights: perturb scales
        for (size_t i = 0; i < f.bytes; i += 97) blobs[e][i] ^= (uint8_t) (r2() & 0x3); }
    std::vector<std::vector<uint8_t>> act(3, std::vector<uint8_t>(cpu::kNativeActBytes));
    for (int t = 0; t < 3; ++t) { std::vector<float> x(kH); std::normal_distribution<float> nd(0.f, 1.f); for (auto& v : x) v = nd(rng); cpu::native_quant_act(f, x.data(), act[t].data()); }
    // worker counts that divide n_ff / n_embd (4, 16) and ones that do not (3, 5, 7, 11): the row slices round differently
    int ran = 0, skipped = 0;
    for (int w : {4, 5, 7, 11, 3, 16}) {
        const int rc = run_pool(w, f, blobs, act);
        if (rc == 77) ++skipped; else if (rc != 0) return rc; else ++ran;
    }
    if (ran == 0) return 77;
    return 0;
}
