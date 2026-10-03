// src/kernels/qsa_grouped_attn_parity.cpp - the V100 grouped verify attention (qsa_grouped_attn.hpp) against the
// FP32 kernel it replaces (`qsa_decode_attn_batch`), a per-row replay of it (`qsa_decode_attn_step`, the "逐行"
// baseline) and an FP64 host reference (GPU, synthetic, no model).
//
// The shape is the verify window's: M = 2/4/8 rows at consecutive positions of a 32K context, each row's own
// top-k selection of 2,051 cells (a recent window plus older cells that drift a few percent per row, so the
// rows' selections overlap as they do in a real window), FP16 / int8 / q4_0 / K8V4 pools over a shuffled page
// table with a couple of non-resident pages.  Checks:
//   1. against FP64, the new kernel's error is no larger than a small multiple of the old kernel's;
//   2. the new and old outputs agree to the bound printed by the run and pinned below;
//   3. the old kernel in per-row replay mode agrees with the old kernel in batch mode (same math, sanity).
// then times grouped vs the batched FP32 path vs the per-row replay with CUDA events (one window each).
//
// PINNED BOUNDS (measured on 2x V100-SXM2-16GB, CUDA 12.8, this file as committed; a change that moves the
// numbers past them fails the test): gate 1 is `err_new <= max(4 * err_old, 1e-3 * ref_scale)` - the 1e-3 floor
// is the FP16 round of the int8/q4 V dequant (measured 2.2e-5 of scale on fp16 KV, 3.2e-4 on int8 KV, 3.5e-4 on
// k8v4; q/p enter the MMAs as hi+lo FP16 pairs, int8 K codes and their scales are exact).  Gate 2 is
// `diff <= 1e-3 * scale` (measured 7e-6 on fp16 KV, 3.3e-4 on int8).  `STRATA_GA_HILO=0` builds trade the
// hi+lo parts for one FP16 cast each: ~1.4x faster, measured agreement ~6e-4 of scale.
//
// Usage: qsa_grouped_attn_parity [--selftest | <ctx=32768> <windows=16> <reps=5>]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_grouped_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
float h2f(uint16_t b) { __half h; *reinterpret_cast<uint16_t*>(&h) = b; return __half2float(h); }
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

void grouped_env(const char* v) {   // the A/B switch qsa_decode_attn_batch dispatches on
#if defined(_WIN32)
    _putenv_s("STRATA_GROUPED_ATTN", v);
#else
    setenv("STRATA_GROUPED_ATTN", v, 1);
#endif
}

constexpr float kQkScale = 1.0f / 16.0f;   // 1/sqrt(256), what the kernels fold into the score

// one window's M rows of one format; returns 0 on PASS
int run(int fmt, int M, int64_t ctx, int windows, int reps) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(4321 + fmt * 17 + M);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    std::uniform_real_distribution<float> q4d(0.01f, 0.08f);
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> k4, v4;
    if (fmt == 1) {
        kq.resize((size_t) (rows * HD)); vq.resize((size_t) (rows * HD));
        ks.resize((size_t) (rows * 4)); vs.resize((size_t) (rows * 4));
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        for (auto& x : vs) x = f2h(sc(rng));
    } else if (fmt == 0) {
        kh.resize((size_t) (rows * HD)); vh.resize((size_t) (rows * HD));
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    } else {
        // q4_0 rows (fmt 2 both sides, fmt 3: K int8 + V q4_0)
        const size_t bh = (size_t) ((HD / 32) * 18);
        k4.resize((size_t) rows * bh); v4.resize((size_t) rows * bh);
        for (size_t r = 0; r < (size_t) rows; ++r) {
            for (int b = 0; b < HD / 32; ++b) {
                uint8_t* pk = k4.data() + r * bh + (size_t) b * 18;
                uint8_t* pv = v4.data() + r * bh + (size_t) b * 18;
                const uint16_t dk = f2h(q4d(rng)), dv = f2h(q4d(rng));
                std::memcpy(pk, &dk, 2); std::memcpy(pv, &dv, 2);
                for (int j = 0; j < 16; ++j) {
                    pk[2 + j] = (uint8_t) ((rng() & 0x0F) | ((rng() & 0x0F) << 4));
                    pv[2 + j] = (uint8_t) ((rng() & 0x0F) | ((rng() & 0x0F) << 4));
                }
            }
        }
        if (fmt == 3) {
            kq.resize((size_t) (rows * HD)); ks.resize((size_t) (rows * 4));
            for (auto& x : kq) x = (int8_t) code(rng);
            for (auto& x : ks) x = f2h(sc(rng));
        }
    }
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    table[pages / 3] = -1;   // two pages the KV streaming did not make resident: every reader must mask them
    table[pages / 2] = -1;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t nq = (int64_t) windows * M;
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> old_cells;
    std::uniform_int_distribution<int64_t> anyc;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + (size_t) (i * cap);
        steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) w;
        if (w == nkv) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[(size_t) c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1);
            anyc = std::uniform_int_distribution<int64_t>(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) anyc(rng);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[(size_t) pick(rng)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
    k::QsaAttnPools pl;
    if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else if (fmt == 0) { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    else if (fmt == 2) { pl.k_q4 = up(k4); pl.v_q4 = up(v4); }
    else { pl.k_q = up(kq); pl.k_scale = up(ks); pl.v_q4 = up(v4); }
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_old = nullptr, *d_new = nullptr, *scratch = nullptr, *scratch1 = nullptr;
    ck(cudaMalloc(&d_old, (size_t) (nq * NH * HD) * 4), "malloc");
    ck(cudaMalloc(&d_new, (size_t) (nq * NH * HD) * 4), "malloc");
    const int64_t stride = k::qsa_decode_attn_scratch_floats(cap, s);
    ck(cudaMalloc(&scratch, (size_t) (M * stride) * 4), "malloc");
    ck(cudaMalloc(&scratch1, (size_t) stride * 4), "malloc");
    auto old_run = [&]() {   // the engine's verify call shape, forced onto the FP32 kernel
        grouped_env("0");
        for (int w = 0; w < windows; ++w)
            k::qsa_decode_attn_batch(d_q + (size_t) w * M * NH * HD, pl, d_ids + (size_t) w * M * cap,
                                     d_steps + (size_t) w * M * k::kStepCount, cap, s, scratch,
                                     d_old + (size_t) w * M * NH * HD, M, nullptr);
    };
    auto row_run = [&]() {   // 逐行: the decode path replayed once per row
        grouped_env("0");
        for (int w = 0; w < windows; ++w)
            for (int i = 0; i < M; ++i)
                k::qsa_decode_attn_step(d_q + ((size_t) w * M + i) * NH * HD, pl,
                                        d_ids + ((size_t) w * M + i) * cap, d_steps + ((size_t) w * M + i) * k::kStepCount,
                                        cap, s, scratch1, d_old + ((size_t) w * M + i) * NH * HD, nullptr);
    };
    auto new_run = [&]() {
        grouped_env("1");   // force the grouped kernel on (the V100 gate would take it here anyway)
        for (int w = 0; w < windows; ++w) {
            const bool ok = k::qsa_grouped_attn_batch(d_q + (size_t) w * M * NH * HD, pl,
                                                      d_ids + (size_t) w * M * cap,
                                                      d_steps + (size_t) w * M * k::kStepCount, cap, s, scratch,
                                                      d_new + (size_t) w * M * NH * HD, M, nullptr);
            if (!ok) { std::fprintf(stderr, "qsa_grouped_attn_batch refused the pools\n"); std::exit(2); }
        }
    };
    old_run();
    std::vector<float> o((size_t) (nq * NH * HD)), nw(o.size()), rp(o.size());
    row_run();
    ck(cudaDeviceSynchronize(), "run1");
    ck(cudaMemcpy(rp.data(), d_old, rp.size() * 4, cudaMemcpyDeviceToHost), "down");
    old_run();
    new_run();
    ck(cudaDeviceSynchronize(), "run2");
    ck(cudaMemcpy(o.data(), d_old, o.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(nw.data(), d_new, nw.size() * 4, cudaMemcpyDeviceToHost), "down");
    // 3. per-row replay vs the batched old kernel: same math, different chunking
    double row_diff = 0;
    for (size_t i = 0; i < o.size(); ++i) row_diff = std::max(row_diff, (double) std::fabs(o[i] - rp[i]));
    // 1. FP64 reference over a sample of rows (all M rows of the first and last window, one row of the rest)
    double err_old = 0, err_new = 0, ref_scale = 0;
    auto ref_row = [&](int64_t i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        const int32_t* sel = ids.data() + (size_t) (i * cap);
        auto kv_val = [&](bool value, int64_t row, int64_t d) -> double {
            if (fmt == 1) return (double) (value ? vq : kq)[(size_t) (row * HD + d)] *
                                 h2f((value ? vs : ks)[(size_t) (row * 4 + d / 64)]);
            if (fmt == 0) return h2f((value ? vh : kh)[(size_t) (row * HD + d)]);
            if (fmt == 3 && !value) return (double) kq[(size_t) (row * HD + d)] * h2f(ks[(size_t) (row * 4 + d / 64)]);
            const uint8_t* blk = (value ? v4 : k4).data() + (size_t) (row * (HD / 32) * 18) + (size_t) (d / 32) * 18;
            const float ds = h2f(*(const uint16_t*) blk);
            const int rem = (int) (d % 32);
            const uint8_t byte = blk[2 + (rem < 16 ? rem : rem - 16)];
            const int nib = rem < 16 ? (int) (byte & 0x0F) : (int) (byte >> 4);
            return (double) (nib - 8) * ds;
        };
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w);
            double mx = -1e300;
            for (int64_t c = 0; c < w; ++c) {
                const int64_t cell = sel[(size_t) c];
                const int64_t page = table[(size_t) (cell / PS)];
                if (page < 0) { sco[(size_t) c] = -1e300; continue; }
                const int64_t row = (page * NKV + kvh) * PS + cell % PS;
                double a = 0;
                for (int64_t d = 0; d < HD; ++d) a += (double) q[((size_t) i * NH + h) * HD + d] * kv_val(false, row, d);
                sco[(size_t) c] = a * kQkScale;
                mx = std::max(mx, sco[(size_t) c]);
            }
            double l = 0;
            for (auto& x : sco) { x = x > -1e200 ? std::exp(x - mx) : 0.0; l += x; }
            for (int64_t d = 0; d < HD; ++d) {
                double a = 0;
                for (int64_t c = 0; c < w; ++c) {
                    if (sco[(size_t) c] == 0.0) continue;
                    const int64_t cell = sel[(size_t) c];
                    const int64_t page = table[(size_t) (cell / PS)];
                    const int64_t row = (page * NKV + kvh) * PS + cell % PS;
                    a += sco[(size_t) c] * kv_val(true, row, d);
                }
                const double r = l > 0 ? a / l : 0.0;
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                ref_scale = std::max(ref_scale, std::fabs(r));
                err_old = std::max(err_old, (double) std::fabs(o[at] - r));
                err_new = std::max(err_new, (double) std::fabs(nw[at] - r));
            }
        }
    };
    for (int w = 0; w < windows; ++w)
        for (int i = 0; i < M; ++i)
            if (w == 0 || w == windows - 1 || i == 0) ref_row((int64_t) w * M + i);
    // 2. new vs old everywhere
    double diff = 0, scale = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
        scale = std::max(scale, (double) std::fabs(o[i]));
    }
    // speed: one window's time in each shape
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](auto& fn) {
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) fn();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / reps;
    };
    const float ms_old = time(old_run), ms_new = time(new_run), ms_row = time(row_run);
    const bool ok1 = err_new <= std::max(4.0 * err_old, 1e-3 * ref_scale);
    const bool ok2 = diff <= 1e-3 * scale;
    const bool ok3 = row_diff <= 1e-5 * scale;
    static const char* fmts[4] = {"fp16", "int8", "q4_0", "k8v4"};
    std::printf("%s %s M=%d ctx %lld x %d windows: vs FP64 old %.3g new %.3g (scale %.3g); new vs old %.3g (%.2g); "
                "rows vs batch %.3g; per window: grouped %.3f ms, batch %.3f ms (%.2fx), per-row %.3f ms (%.2fx)\n",
                ok1 && ok2 && ok3 ? "PASS" : "FAIL", fmts[fmt], M, (long long) ctx, windows, err_old, err_new,
                ref_scale, diff, scale > 0 ? diff / scale : 0.0, row_diff, ms_new / windows, ms_old / windows,
                ms_old / ms_new, ms_row / windows, ms_row / ms_new);
    cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q);
    cudaFree(d_old); cudaFree(d_new); cudaFree(scratch); cudaFree(scratch1);
    cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
    cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool);
    cudaFree((void*) pl.k_q4); cudaFree((void*) pl.v_q4); cudaFree((void*) pl.page_table);
    return ok1 && ok2 && ok3 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    int64_t ctx = 32768;
    int windows = 16, reps = 5;
    if (argc > 1 && std::strcmp(argv[1], "--one") == 0) {   // one config for a sanitizer run
        return run(argc > 2 ? std::atoi(argv[2]) : 0, argc > 3 ? std::atoi(argv[3]) : 2, 32768, 4, 1);
    }
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        ctx = 32768;
        windows = 4;
        reps = 3;
    } else {
        if (argc > 1) ctx = std::atoll(argv[1]);
        if (argc > 2) windows = std::atoi(argv[2]);
        if (argc > 3) reps = std::atoi(argv[3]);
    }
    int fails = 0;
    for (int fmt = 0; fmt < 4; ++fmt)
        for (int M : {2, 4, 8}) fails += run(fmt, M, ctx, windows, reps);
    // short contexts: the selection is the identity (union == the row's list exactly)
    fails += run(1, 4, 1500, windows, reps);
    fails += run(0, 8, 2100, windows, reps);
    std::printf("FAILURES: %d\n", fails);
    return fails;
}
