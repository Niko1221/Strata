// CPU idle warm-up adapted from Basalt (MIT), ca7d07418cdf6ee08462c560102a7798c94886cc.
// See BASALT-LICENSE.txt. Multiple experts, mixed layouts and owned activation snapshots added here.
#include "strata/kernels/cpu/idle_prefetch.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/platform/cpu_relax.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace strata::kernels::cpu {
bool IdleExpertPrefetch::init(const std::vector<std::vector<uint16_t>>& routers, int N, int NE, int k,
                             std::string& error) {
    if (!cpu_avx2_ok() || count_ == 0 || pool_.workers() <= 1 ||
        N <= 0 || N % 32 || NE < 1 || NE > 512 || k < 1 || k > 16 || routers.empty()) {
        error = "requires AVX2, resident native experts, 2+ workers, N%32=0, NE<=512, k<=16";
        return false;
    }
    if (pool_.numa_slices() > 0 && !active_numa_split().test) {
        const auto& plan = active_numa_split();
        const auto& cores = pool_.worker_cores();
        for (int k = 0; k < 2; ++k) {
            bool worker = false;
            for (int c : plan.cores[k])
                worker |= std::find(cores.begin(), cores.begin() + std::min(cores.size(), (size_t) pool_.workers()), c) !=
                          cores.begin() + std::min(cores.size(), (size_t) pool_.workers());
            if (!worker) { error = "NUMA prefetch requires a worker on each node"; return false; }
        }
    }
    for (const auto& r : routers) if (r.size() != size_t(N) * NE) { error = "router shape"; return false; }
    N_ = N; NE_ = NE; K_ = k; L_ = int(routers.size());
    rq_.resize(size_t(L_) * NE * N / 2); rs_.resize(size_t(L_) * NE * N / 32);
    xq_.resize(size_t(MAXT) * (N / 32) * sizeof(block_q8_1));
    logits_.resize(size_t(MAXT) * NE); blobs_.resize(NE);
    for (int l = 0; l < L_; ++l) for (int e = 0; e < NE; ++e) for (int b = 0; b < N / 32; ++b) {
        float v[32], a = 0;
        for (int i = 0; i < 32; ++i) {
            uint32_t u = uint32_t(routers[l][size_t(e) * N + b * 32 + i]) << 16;
            std::memcpy(v + i, &u, 4);
            if (!std::isfinite(v[i])) { error = "non-finite router"; return false; }
            a = std::max(a, std::fabs(v[i]));
        }
        size_t blk = (size_t(l) * NE + e) * (N / 32) + b;
        float d = a / 7.f; rs_[blk] = d;
        for (int i = 0; i < 16; ++i) {
            long lo = d > 0 ? std::lrintf(v[i] / d) : 0, hi = d > 0 ? std::lrintf(v[i + 16] / d) : 0;
            rq_[blk * 16 + i] = uint8_t((std::clamp(lo, -7L, 7L) + 8) | ((std::clamp(hi, -7L, 7L) + 8) << 4));
        }
    }
    return true;
}

void IdleExpertPrefetch::start(int layer, const float* x, int nt, const std::vector<WarmBlob>& blobs) {
    stop();
    chosen_n_ = 0; layer_ = -1;
    if (N_ == 0 || layer < 0 || layer >= L_ || nt < 1 || nt > MAXT || blobs.size() != size_t(NE_)) return;
    layer_ = layer; nt_ = nt; blobs_ = blobs;
    for (int t = 0; t < nt; ++t)
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_1)->from_float(
            x + size_t(t) * N_, xq_.data() + size_t(t) * N_ / 32 * sizeof(block_q8_1), N_);
    arrived_.store(0, std::memory_order_relaxed); ready_.store(false, std::memory_order_relaxed);
    finished_.store(0, std::memory_order_relaxed); active_ = true;
    ++starts_;
    pool_.idle_start(job, this);
}

void IdleExpertPrefetch::stop() {
    pool_.idle_stop();
    if (active_) {
        if (finished_.load(std::memory_order_relaxed) < pool_.workers()) ++cancelled_;
        active_ = false;
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma,f16c")))
#endif
void IdleExpertPrefetch::logits(int first, int end, const std::atomic<bool>& stop) {
    int NB = N_ / 32;
    auto* R = rq_.data() + size_t(layer_) * NE_ * N_ / 2;
    auto* S = rs_.data() + size_t(layer_) * NE_ * NB;
    __m128i mask = _mm_set1_epi8(15), off = _mm_set1_epi8(8);
    for (int e = first; e < end; ++e) {
        if (stop.load(std::memory_order_relaxed)) return;
        __m256 acc[MAXT]; for (int t = 0; t < nt_; ++t) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < NB; ++b) {
            auto q = _mm_loadu_si128((const __m128i*)(R + size_t(e) * N_ / 2 + b * 16));
            auto wl = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_and_si128(q, mask), off));
            auto wh = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), mask), off));
            for (int t = 0; t < nt_; ++t) {
                const auto* blk = xq_.data() + (size_t(t) * NB + b) * sizeof(block_q8_1);
                uint16_t dh; std::memcpy(&dh, blk, 2);
                auto x = _mm256_loadu_si256((const __m256i*)(blk + 4));
                auto p = _mm256_add_epi32(_mm256_madd_epi16(wl, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x))),
                                         _mm256_madd_epi16(wh, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1))));
                float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dh)));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(S[size_t(e) * NB + b] * d), acc[t]);
            }
        }
        for (int t = 0; t < nt_; ++t) {
            auto v = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            logits_[size_t(t) * NE_ + e] = _mm_cvtss_f32(v);
        }
    }
}

void IdleExpertPrefetch::select() {
    float score[512] = {};
    for (int t = 0; t < nt_; ++t) {
        int top[16]; float val[16]; int m = 0;
        for (int e = 0; e < NE_; ++e) {
            float v = logits_[size_t(t) * NE_ + e];
            if (!std::isfinite(v) || (m == K_ && v <= val[K_ - 1])) continue;
            int j = m < K_ ? m++ : K_ - 1;
            while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; top[j] = top[j - 1]; --j; }
            val[j] = v; top[j] = e;
        }
        if (!m) continue;
        float w[16], sum = 0;
        for (int j = 0; j < m; ++j) sum += (w[j] = std::exp(val[j] - val[0]));
        for (int j = 0; j < m; ++j) if (blobs_[top[j]].data) score[top[j]] += w[j] / sum;
    }
    chosen_n_ = 0;
    for (int i = 0; i < count_; ++i) {
        int best = -1;
        for (int e = 0; e < NE_; ++e) if (score[e] > 0 && (best < 0 || score[e] > score[best])) best = e;
        if (best < 0) break;
        chosen_[chosen_n_++] = best; score[best] = 0;
    }
    ++predicted_;
}

void IdleExpertPrefetch::job(void* p, int w, int W, const std::atomic<bool>& stop) {
    auto& c = *(IdleExpertPrefetch*)p;
    c.logits(c.NE_ * w / W, c.NE_ * (w + 1) / W, stop);
    if (c.arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == W) {
        if (!stop.load(std::memory_order_relaxed)) c.select();
        c.ready_.store(true, std::memory_order_release);
    } else while (!c.ready_.load(std::memory_order_acquire)) {
        if (stop.load(std::memory_order_relaxed)) return;
        strata_cpu_pause();
    }
    int slice, rank, count;
    c.pool_.idle_partition(w, slice, rank, count);
    uint64_t sink = 0, lines = 0;
    // Warm each selected expert in rank order. Every worker touches disjoint 64-byte ranges.
    for (int i = 0; i < c.chosen_n_; ++i) {
        auto b = c.blobs_[c.chosen_[i]];
        if (!b.data || b.down_off > b.bytes || b.up_off > b.down_off) continue;
        // In a split, gate/up/down each own two row halves. Keep warming local;
        // legacy blobs without an up offset retain the old two-role path off NUMA.
        if (slice >= 0 && b.up_off == 0) continue;
        const size_t ends[4] = {0, b.up_off, b.down_off, b.bytes};
        for (int part = b.up_off ? 0 : 1; part < 3; ++part) {
            size_t r0 = b.up_off ? ends[part] : (part == 1 ? 0 : b.down_off);
            size_t r1 = ends[part + 1];
            if (slice >= 0) {
                const size_t middle = r0 + (r1 - r0) / 2;
                if (slice == 0) r1 = middle;
                else r0 = middle;
            }
            size_t per = ((r1 - r0 + count - 1) / count + 63) & ~size_t(63);
            size_t b0 = std::min(r1, r0 + per * rank), b1 = std::min(r1, b0 + per);
            for (size_t o = b0; o < b1; o += 4096) {
                if (stop.load(std::memory_order_relaxed)) goto done;
                size_t n = std::min(size_t(4096), b1 - o);
                for (size_t q = 0; q < n; q += 64) {
                    // A byte load is unaligned-safe and cannot cross the blob boundary.
                    sink += *(const volatile uint8_t*)(b.data + o + q); ++lines;
                }
            }
        }
    }
    if (!stop.load(std::memory_order_relaxed)) c.finished_.fetch_add(1, std::memory_order_relaxed);
done:
    c.lines_.fetch_add(lines, std::memory_order_relaxed);
    if (sink == 0x9e3779b97f4a7c15ull) c.sink_.fetch_add(sink, std::memory_order_relaxed);
}

void IdleExpertPrefetch::score(int layer, const int32_t* ids, int64_t n) {
    stop();
    if (layer != layer_ || !ready_.load(std::memory_order_acquire)) return;
    ++scored_;
    for (int j = 0; j < chosen_n_; ++j) {
        ++offered_;
        for (int64_t i = 0; i < n; ++i) if (ids[i] == chosen_[j]) { ++hit_; break; }
    }
    if (scored_ % 4096 == 0) report();
}

void IdleExpertPrefetch::report() const {
    std::fprintf(stderr, "strata cpu-prefetch: count=%d starts=%llu cancelled=%llu predictions=%llu scored=%llu selected=%llu routed_hits=%llu read_lines=%llu touched_MiB=%.3f\n",
        count_, (unsigned long long)starts_, (unsigned long long)cancelled_, (unsigned long long)predicted_, (unsigned long long)scored_,
        (unsigned long long)offered_, (unsigned long long)hit_, (unsigned long long)read_lines(), read_lines() * 64. / (1024 * 1024));
}
}
