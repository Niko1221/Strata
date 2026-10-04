// sel_scores_bench: the prompt path's QSA block scores, today's kernel (one sub-group per (query, block)) against
// oneMKL GEMMs (pooled keys x indexer queries, tiled over blocks) plus a relu-sum reduce, at the B70's long-context
// shapes. Accuracy against an FP64 reference, and the top-k ids each variant leads to.
//   sel_scores_bench [reps=10] [active blocks...=10000 32768 65536]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace k = strata::kernels;
namespace blas = oneapi::mkl::blas;
constexpr int D = 128, H = 4, R = 4, NQ = 256;

// out[q][b] for b <= n_bid(q): sum over heads of relu(S[(q*H + h)][b - b0]); block n_bid scores the dead key
static void reduce_tile(sycl::queue& q, const float* S, int64_t tb, int64_t b0, int64_t nb_tile, const float* dead,
                        const float* qidx, const int32_t* steps, int nq, int64_t max_blocks, float* out) {
    q.parallel_for(sycl::range<2>((size_t) nq, (size_t) nb_tile), [=](sycl::id<2> it) {
        const int qi = (int) it[0];
        const int64_t b = b0 + (int64_t) it[1];
        const int64_t n_kv = steps[qi * k::kStepCount + k::kStepNKv], n_bid = steps[qi * k::kStepCount + k::kStepNBid];
        if (b > n_bid) return;
        float score = 0.0f;
        if (b == n_bid) {
            for (int h = 0; h < H; ++h) {
                float d = 0.0f;
                for (int i = 0; i < D; ++i) d += dead[i] * qidx[(qi * H + h) * D + i];
                score += d > 0.0f ? d : 0.0f;
            }
            if (n_kv % R != 0) score += 1e9f;
        } else {
            for (int h = 0; h < H; ++h) {
                const float d = S[(size_t) (qi * H + h) * tb + (size_t) (b - b0)];
                score += d > 0.0f ? d : 0.0f;
            }
        }
        out[(size_t) qi * max_blocks + b] = score;
    });
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 10;
    std::vector<int64_t> actives;
    for (int i = 2; i < argc; ++i) actives.push_back(std::atoll(argv[i]));
    if (actives.empty()) actives = {10000, 32768, 65536};
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s\n", q->get_device().get_info<sycl::info::device::name>().c_str());
    const k::QsaShapes s = k::qsa_real_shapes();
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const int64_t active : actives) {
        const int64_t max_blocks = active + 2;
        std::vector<float> hp((size_t) max_blocks * D), hd(D), hq((size_t) NQ * H * D);
        for (auto& x : hp) x = nd(rng);
        for (auto& x : hd) x = nd(rng);
        for (auto& x : hq) x = nd(rng) * 0.1f;
        // consecutive positions ending at the last active block (a prompt batch)
        std::vector<int32_t> hs((size_t) NQ * k::kStepCount);
        for (int i = 0; i < NQ; ++i) {
            const int64_t n_kv = active * R - (NQ - 1 - i) - 1;
            hs[i * k::kStepCount + k::kStepPos] = (int32_t) (n_kv - 1);
            hs[i * k::kStepCount + k::kStepNKv] = (int32_t) n_kv;
            hs[i * k::kStepCount + k::kStepNBid] = (int32_t) (n_kv / R);
            hs[i * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
        }
        const int64_t act = hs[(NQ - 1) * k::kStepCount + k::kStepNBid] + 1;
        float* dp = sycl::malloc_device<float>(hp.size(), *q);
        float* dd = sycl::malloc_device<float>(D, *q);
        float* dq = sycl::malloc_device<float>(hq.size(), *q);
        int32_t* ds = sycl::malloc_device<int32_t>(hs.size(), *q);
        float* out_ref = sycl::malloc_device<float>((size_t) NQ * max_blocks, *q);
        float* out_g = sycl::malloc_device<float>((size_t) NQ * max_blocks, *q);
        const int64_t cap = k::qsa_selection_width(active * R, s) + 8;
        int32_t* ids_ref = sycl::malloc_device<int32_t>((size_t) NQ * cap, *q);
        int32_t* ids_g = sycl::malloc_device<int32_t>((size_t) NQ * cap, *q);
        const int64_t TB = 8192;
        float* S = sycl::malloc_device<float>((size_t) NQ * H * TB, *q);
        q->memcpy(dp, hp.data(), hp.size() * 4);
        q->memcpy(dd, hd.data(), D * 4);
        q->memcpy(dq, hq.data(), hq.size() * 4);
        q->memcpy(ds, hs.data(), hs.size() * 4).wait();
        q->memset(out_ref, 0, (size_t) NQ * max_blocks * 4).wait();
        auto time = [&](auto&& fn) {
            fn(); q->wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            q->wait();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        auto cur = [&] { k::qsa_block_scores(dp, dd, dq, ds, NQ, max_blocks, s, out_ref, q, act); };
        auto gemm = [&](blas::compute_mode mode) {
            for (int64_t b0 = 0; b0 < act; b0 += TB) {
                const int64_t nb = std::min(TB, act - b0);
                blas::column_major::gemm(*q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, nb, NQ * H, D,
                                         1.0f, dp + b0 * D, D, dq, D, 0.0f, S, TB, mode);
                reduce_tile(*q, S, TB, b0, nb, dd, dq, ds, NQ, max_blocks, out_g);
            }
        };
        const double t_cur = time(cur);
        // FP64 reference on 4 queries (first, middle, last)
        std::vector<float> r((size_t) NQ * max_blocks), g((size_t) NQ * max_blocks);
        q->memcpy(r.data(), out_ref, r.size() * 4).wait();
        k::qsa_block_topk(out_ref, ds, NQ, max_blocks, cap, s, ids_ref, q, act);
        std::vector<int32_t> hir((size_t) NQ * cap), hig((size_t) NQ * cap);
        q->memcpy(hir.data(), ids_ref, hir.size() * 4).wait();
        auto err_vs64 = [&](const std::vector<float>& v) {
            double emax = 0, smax = 0;
            for (int qi : {0, NQ / 2, NQ - 1}) {
                const int64_t n_bid = hs[qi * k::kStepCount + k::kStepNBid];
                for (int64_t b = 0; b < n_bid; b += 7) {
                    double sc = 0;
                    for (int h = 0; h < H; ++h) {
                        double d = 0;
                        for (int i = 0; i < D; ++i) d += (double) hp[(size_t) b * D + i] * hq[(size_t) (qi * H + h) * D + i];
                        sc += d > 0 ? d : 0;
                    }
                    emax = std::max(emax, std::fabs(sc - v[(size_t) qi * max_blocks + b]));
                    smax = std::max(smax, std::fabs(sc));
                }
            }
            return emax / smax;
        };
        std::printf("active %lld blocks (%lld cells), %d queries: current %.3f ms, rel err %.2e\n", (long long) act,
                    (long long) act * R, NQ, t_cur, err_vs64(r));
        struct V { const char* name; blas::compute_mode m; } vs[] = {{"sgemm fp32", blas::compute_mode::standard},
                                                                     {"bf16x3", blas::compute_mode::float_to_bf16x3},
                                                                     {"tf32", blas::compute_mode::float_to_tf32},
                                                                     {"bf16x2", blas::compute_mode::float_to_bf16x2}};
        for (const V& v : vs) {
            q->memset(out_g, 0, (size_t) NQ * max_blocks * 4).wait();
            double t;
            try { t = time([&] { gemm(v.m); }); } catch (const std::exception& e) { std::printf("  %-10s failed: %s\n", v.name, e.what()); continue; }
            q->memcpy(g.data(), out_g, g.size() * 4).wait();
            k::qsa_block_topk(out_g, ds, NQ, max_blocks, cap, s, ids_g, q, act);
            q->memcpy(hig.data(), ids_g, hig.size() * 4).wait();
            int64_t diff_ids = 0, diff_q = 0;
            for (int qi = 0; qi < NQ; ++qi) {
                const int w = hs[qi * k::kStepCount + k::kStepWidth];
                std::vector<int32_t> a(hir.begin() + (size_t) qi * cap, hir.begin() + (size_t) qi * cap + w);
                std::vector<int32_t> b(hig.begin() + (size_t) qi * cap, hig.begin() + (size_t) qi * cap + w);
                std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                std::vector<int32_t> x;
                std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(x));
                diff_ids += (int64_t) x.size(); diff_q += !x.empty();
            }
            std::printf("  %-10s %.3f ms (%.2fx), rel err %.2e, selection differs in %lld of %d queries (%lld cells)\n",
                        v.name, t, t_cur / t, err_vs64(g), (long long) diff_q, NQ, (long long) diff_ids);
        }
        for (void* p : {(void*) dp, (void*) dd, (void*) dq, (void*) ds, (void*) out_ref, (void*) out_g, (void*) ids_ref,
                        (void*) ids_g, (void*) S})
            sycl::free(p, *q);
    }
    return 0;
}
