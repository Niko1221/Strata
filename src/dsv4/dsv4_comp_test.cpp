// src/dsv4/dsv4_comp_test.cpp - standalone unit test for the dsv4 S4 compressor block
// (src/dsv4/dsv4_comp.cu), against a verbatim host port of src/dsv4/dsv4ref.cpp's
// compressor_forward (:623) on synthetic weights.
//
// Why this exists: at the golden prompt length (13 tokens) the r = 128 compressor pools nothing
// (ncomp = 0), so the S4 replay gates exercise the compressor's wiring but not its pooling numerics.
// This test drives the pooling paths directly:
//   case A: r=128 prefill s=13   -> no compressed token, state carry only
//   case B: r=128 prefill s=300  -> 2 compressed tokens + 44-token remainder state
//   case C: r=4 (coff=2 overlap) prefill s=13 -> 3 compressed tokens, overlap state layout
//   case D: r=4 decode start_pos=13..19 -> state fill, pooling at t%4==0, the overlap shift
//   case E: r=128 decode start_pos=300..383 -> 84 single-token steps, pooling fires at t=384
// Each case compares GPU vs host reference: kv_c rows, the carry state, and the cache slot writes.
//
// Also the memcheck small-geometry target for the S4 kernels:
//   compute-sanitizer --tool memcheck build-s4/strata-dsv4-comp-test
//
// usage: strata-dsv4-comp-test [--seed 7]

#include "strata/dsv4/attn.hpp"
#include "strata/dsv4/comp.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int64_t kNEmbd = 4096;
constexpr int64_t kHeadDim = 512;
constexpr int64_t kMaxPos = 4096;

#define CK(x)                                                                            \
    do {                                                                                 \
        cudaError_t err_ = (x);                                                          \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                         __FILE__, __LINE__);                                            \
            std::exit(3);                                                                \
        }                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Host reference: verbatim port of dsv4ref.cpp compressor_forward (:623) + the rmsnorm/rope tail.
// FP32 GEMM (float accumulation like the host oracle), pooling in double, state semantics identical.
// ---------------------------------------------------------------------------
float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct HostComp {
    int64_t ratio, d, coff, W;
    std::vector<float> wkv, wgate;   // dequantized bf16 rows [W][kNEmbd]
    std::vector<float> ape;          // [ratio][W]
    std::vector<float> norm;         // [d]
    std::vector<float> kv_state, score_state;  // [coff*ratio][W]
    // rope table (same builder as the device side)
    int64_t half = 32;
    std::vector<float> cosv, sinv;

    void init_state() {
        kv_state.assign((size_t) (coff * ratio * W), 0.0f);
        score_state.assign((size_t) (coff * ratio * W), -INFINITY);
    }

    void gemm(const std::vector<float>& w, const float* x, int64_t s, float* out) const {
        for (int64_t t = 0; t < s; ++t)
            for (int64_t r = 0; r < W; ++r) {
                float acc = 0.f;
                for (int64_t i = 0; i < kNEmbd; ++i) acc += x[t * kNEmbd + i] * w[(size_t) r * kNEmbd + i];
                out[t * W + r] = acc;
            }
    }

    void rope_row(float* row, int64_t pos) const {
        float* base = row + (d - 2 * half);
        for (int64_t i = 0; i < half; ++i) {
            const float c = cosv[(size_t) (pos * half + i)], sn = sinv[(size_t) (pos * half + i)];
            const float x0 = base[2 * i], x1 = base[2 * i + 1];
            base[2 * i] = x0 * c - x1 * sn;
            base[2 * i + 1] = x0 * sn + x1 * c;
        }
    }

    void rmsnorm(float* row) const {
        double sq = 0;
        for (int64_t i = 0; i < d; ++i) sq += (double) row[i] * row[i];
        const float inv = (float) (1.0 / std::sqrt(sq / d + 1e-6));
        for (int64_t i = 0; i < d; ++i) row[i] = norm[(size_t) i] * (row[i] * inv);
    }

    // returns compressed rows written into kv_c ([ncomp][d]); cache writes appended to cache_log as
    // (slot, row).  Mirrors dsv4_comp_forward's contract.
    int64_t forward(const float* x, int64_t s, int64_t start_pos, std::vector<float>& kv_c) {
        std::vector<float> kv((size_t) s * W), score((size_t) s * W);
        gemm(wkv, x, s, kv.data());
        gemm(wgate, x, s, score.data());
        const bool overlap = coff == 2;
        int64_t ncomp = 0;
        if (start_pos == 0) {
            const bool should = s >= ratio;
            const int64_t rem = s % ratio, cutoff = s - rem;
            const int64_t offset = overlap ? ratio : 0;
            if (overlap && cutoff >= ratio) {
                for (int64_t i = 0; i < ratio * W; ++i) {
                    kv_state[(size_t) i] = kv[(size_t) ((cutoff - ratio) * W + i)];
                    score_state[(size_t) i] = score[(size_t) ((cutoff - ratio) * W + i)] + ape[(size_t) i];
                }
            }
            if (rem > 0)
                for (int64_t i = 0; i < rem * W; ++i) {
                    kv_state[(size_t) (offset * W + i)] = kv[(size_t) (cutoff * W + i)];
                    score_state[(size_t) (offset * W + i)] =
                        score[(size_t) (cutoff * W + i)] + ape[(size_t) i];
                }
            if (!should) return 0;
            ncomp = cutoff / ratio;
            kv_c.resize((size_t) ncomp * d);
            const int64_t rows = coff * ratio;
            for (int64_t j = 0; j < ncomp; ++j)
                for (int64_t dd = 0; dd < d; ++dd) {
                    double mx = -1.0 / 0.0, denom = 0.0, acc = 0.0;
                    for (int64_t i = 0; i < rows; ++i) {
                        int64_t tok, off;
                        if (overlap) {
                            if (i < ratio) { tok = j * ratio + i - ratio; off = dd; }
                            else { tok = j * ratio + (i - ratio); off = d + dd; }
                        } else { tok = j * ratio + i; off = dd; }
                        if (overlap && tok < 0) continue;
                        const double v = (double) score[(size_t) (tok * W + off)] +
                                         (double) ape[(size_t) ((tok % ratio) * W + off)];
                        if (v > mx) mx = v;
                    }
                    for (int64_t i = 0; i < rows; ++i) {
                        int64_t tok, off;
                        if (overlap) {
                            if (i < ratio) { tok = j * ratio + i - ratio; off = dd; }
                            else { tok = j * ratio + (i - ratio); off = d + dd; }
                        } else { tok = j * ratio + i; off = dd; }
                        if (overlap && tok < 0) continue;
                        denom += exp((double) score[(size_t) (tok * W + off)] +
                                     (double) ape[(size_t) ((tok % ratio) * W + off)] - mx);
                    }
                    for (int64_t i = 0; i < rows; ++i) {
                        int64_t tok, off;
                        if (overlap) {
                            if (i < ratio) { tok = j * ratio + i - ratio; off = dd; }
                            else { tok = j * ratio + (i - ratio); off = d + dd; }
                        } else { tok = j * ratio + i; off = dd; }
                        if (overlap && tok < 0) continue;
                        const double p = exp((double) score[(size_t) (tok * W + off)] +
                                             (double) ape[(size_t) ((tok % ratio) * W + off)] - mx) /
                                         denom;
                        acc += p * (double) kv[(size_t) (tok * W + off)];
                    }
                    kv_c[(size_t) (j * d + dd)] = (float) acc;
                }
        } else {
            const bool should = (start_pos + 1) % ratio == 0;
            const int64_t rem = start_pos % ratio;
            for (int64_t c = 0; c < W; ++c) score[(size_t) c] += ape[(size_t) (rem * W + c)];
            const int64_t dst = overlap ? ratio + rem : rem;
            for (int64_t c = 0; c < W; ++c) {
                kv_state[(size_t) (dst * W + c)] = kv[(size_t) c];
                score_state[(size_t) (dst * W + c)] = score[(size_t) c];
            }
            if (!should) return 0;
            ncomp = 1;
            kv_c.resize((size_t) d);
            const int64_t rows = coff * ratio;
            for (int64_t dd = 0; dd < d; ++dd) {
                double mx = -1.0 / 0.0, denom = 0.0, acc = 0.0;
                for (int64_t i = 0; i < rows; ++i) {
                    const int64_t off = (!overlap || i < ratio) ? dd : d + dd;
                    const double v = (double) score_state[(size_t) (i * W + off)];
                    if (v > mx) mx = v;
                }
                for (int64_t i = 0; i < rows; ++i) {
                    const int64_t off = (!overlap || i < ratio) ? dd : d + dd;
                    const double p = exp((double) score_state[(size_t) (i * W + off)] - mx);
                    denom += p;
                    acc += p * (double) kv_state[(size_t) (i * W + off)];
                }
                kv_c[(size_t) dd] = (float) (acc / denom);
            }
            if (overlap) {
                std::memmove(kv_state.data(), kv_state.data() + ratio * W, (size_t) (ratio * W) * 4);
                std::memmove(score_state.data(), score_state.data() + ratio * W,
                             (size_t) (ratio * W) * 4);
            }
        }
        for (int64_t j = 0; j < ncomp; ++j) {
            float* row = kv_c.data() + j * d;
            rmsnorm(row);
            rope_row(row, (start_pos == 0 ? 0 : start_pos + 1 - ratio) + j * ratio);
        }
        return ncomp;
    }
};

// rope table builder, identical to dsv4_attn_test.cpp's (dsv4ref rope_build)
void rope_build(int64_t dim, int64_t maxpos, int64_t orig, double base, double factor,
                double beta_fast, double beta_slow, std::vector<float>& cosv,
                std::vector<float>& sinv) {
    const int64_t half = dim / 2;
    std::vector<double> f((size_t) half);
    for (int64_t i = 0; i < half; ++i)
        f[(size_t) i] = std::pow(base, -static_cast<double>(2 * i) / dim);
    if (orig > 0) {
        auto fcd = [&](double nr) {
            return (double) dim * std::log((double) orig / (nr * 2.0 * M_PI)) / (2.0 * std::log(base));
        };
        const int64_t low = std::max<int64_t>(0, (int64_t) std::floor(fcd(beta_fast)));
        const int64_t high = std::min<int64_t>(dim - 1, (int64_t) std::ceil(fcd(beta_slow)));
        for (int64_t i = 0; i < half; ++i) {
            double ramp = ((double) i - low) / (double) (high - low);
            ramp = std::min(1.0, std::max(0.0, ramp));
            const double smooth = 1.0 - ramp;
            f[(size_t) i] = f[(size_t) i] / factor * (1.0 - smooth) + f[(size_t) i] * smooth;
        }
    }
    cosv.resize((size_t) maxpos * half);
    sinv.resize((size_t) maxpos * half);
    for (int64_t p = 0; p < maxpos; ++p)
        for (int64_t i = 0; i < half; ++i) {
            const double ang = (double) p * f[(size_t) i];
            cosv[(size_t) (p * half + i)] = (float) std::cos(ang);
            sinv[(size_t) (p * half + i)] = (float) std::sin(ang);
        }
}

// ---------------------------------------------------------------------------
// comparison helpers
// ---------------------------------------------------------------------------
struct Cmp {
    double cos = 0, absmax = 0, rel_amax = 0;
    int64_t n = 0;
};

Cmp cmp(const std::vector<float>& a, const std::vector<float>& b, const char* tag) {
    Cmp r;
    if (a.size() != b.size()) {
        std::fprintf(stderr, "SIZE MISMATCH [%s] %zu vs %zu\n", tag, a.size(), b.size());
        r.n = -1;
        return r;
    }
    r.n = (int64_t) a.size();
    double dot = 0, na = 0, nb = 0, mx = 0, mg = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
        mx = std::max(mx, (double) std::fabs(a[i] - b[i]));
        mg = std::max(mg, (double) std::fabs(b[i]));
    }
    r.cos = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
    r.absmax = mx;
    r.rel_amax = mg > 0 ? mx / mg : 0.0;
    return r;
}

struct Gpu {
    // device buffers
    float *d_x = nullptr, *d_cache = nullptr, *d_kvc = nullptr;
    uint16_t *d_wkv = nullptr, *d_wgate = nullptr;
    float *d_ape = nullptr, *d_norm = nullptr;
    strata::dsv4::Dsv4CompState st;
    strata::dsv4::Dsv4CompWorkspace ws;
    strata::dsv4::Dsv4CompWeights w;
    cudaStream_t stream = nullptr;
    int64_t max_s, W, max_comp, cache_slots;

    void alloc(int64_t ratio, int64_t coff, int64_t max_s_, int64_t cache_slots_) {
        max_s = max_s_;
        W = coff * kHeadDim;
        max_comp = max_s / std::min<int64_t>(ratio, max_s) + 2;
        cache_slots = cache_slots_;
        CK(cudaStreamCreate(&stream));
        CK(cudaMalloc(&d_x, (size_t) max_s * kNEmbd * 4));
        CK(cudaMalloc(&d_wkv, (size_t) W * kNEmbd * 2));
        CK(cudaMalloc(&d_wgate, (size_t) W * kNEmbd * 2));
        CK(cudaMalloc(&d_ape, (size_t) ratio * W * 4));
        CK(cudaMalloc(&d_norm, (size_t) kHeadDim * 4));
        CK(cudaMalloc(&d_cache, (size_t) cache_slots * kHeadDim * 4));
        CK(cudaMalloc(&d_kvc, (size_t) max_comp * kHeadDim * 4));
        CK(cudaMemset(d_cache, 0, (size_t) cache_slots * kHeadDim * 4));
        if (!dsv4_comp_state_init(st, ratio, kHeadDim, coff, stream) ||
            !dsv4_comp_workspace_init(ws, (int) max_s, W, max_comp, 128, kNEmbd, coff * ratio)) {
            std::fprintf(stderr, "gpu alloc failed\n");
            std::exit(3);
        }
        CK(cudaStreamSynchronize(stream));
    }
    void free() {
        dsv4_comp_state_free(st);
        dsv4_comp_workspace_free(ws);
        cudaFree(d_x);
        cudaFree(d_wkv);
        cudaFree(d_wgate);
        cudaFree(d_ape);
        cudaFree(d_norm);
        cudaFree(d_cache);
        cudaFree(d_kvc);
        cudaStreamDestroy(stream);
    }
};

int g_fail = 0;

void report(const char* name, const Cmp& c, double cos_gate, double rel_gate) {
    const bool ok = c.n >= 0 && c.cos >= cos_gate && c.rel_amax <= rel_gate;
    if (!ok) ++g_fail;
    std::printf("  %-34s n=%7lld cos=%.9f absmax=%.3e rel_amax=%.3e %s\n", name, (long long) c.n,
                c.cos, c.absmax, c.rel_amax, ok ? "PASS" : "FAIL");
}

// run one case: random weights/x for the given geometry, a prefill of s tokens, then `ndec` decode
// steps from start_pos = s.  Compares kv_c, state, and the cache after every step.
void run_case(const char* name, int64_t ratio, int64_t coff, int64_t s, int64_t ndec, uint32_t seed,
              const std::vector<float>& cosv, const std::vector<float>& sinv,
              const strata::dsv4::Dsv4RopeTables& rope_dev) {
    std::printf("case %s (ratio=%lld coff=%lld prefill=%lld decode=%lld)\n", name, (long long) ratio,
                (long long) coff, (long long) s, (long long) ndec);
    const int64_t W = coff * kHeadDim;
    std::mt19937 rng(seed);
    auto randn = [&]() { return std::normal_distribution<float>(0.f, 1.f)(rng); };
    HostComp ref;
    ref.ratio = ratio;
    ref.d = kHeadDim;
    ref.coff = coff;
    ref.W = W;
    ref.cosv = cosv;
    ref.sinv = sinv;
    ref.wkv.resize((size_t) W * kNEmbd);
    ref.wgate.resize((size_t) W * kNEmbd);
    ref.ape.resize((size_t) ratio * W);
    ref.norm.resize((size_t) kHeadDim);
    for (auto& v : ref.wkv) v = randn() * 0.02f;
    for (auto& v : ref.wgate) v = randn() * 0.02f;
    for (auto& v : ref.ape) v = randn();
    for (auto& v : ref.norm) v = 1.0f + randn() * 0.1f;
    ref.init_state();
    // bf16 payloads for the GPU (round-trip through bf16 so both sides see the same weights)
    std::vector<uint16_t> wkv16((size_t) W * kNEmbd), wgate16((size_t) W * kNEmbd);
    for (size_t i = 0; i < ref.wkv.size(); ++i) {
        uint32_t u;
        float f = ref.wkv[i];
        std::memcpy(&u, &f, 4);
        wkv16[i] = (uint16_t) (u >> 16);
        ref.wkv[i] = bf16_to_f32(wkv16[i]);
        f = ref.wgate[i];
        std::memcpy(&u, &f, 4);
        wgate16[i] = (uint16_t) (u >> 16);
        ref.wgate[i] = bf16_to_f32(wgate16[i]);
    }
    std::vector<float> x((size_t) (s + ndec) * kNEmbd);
    for (auto& v : x) v = randn();

    Gpu gpu;
    gpu.alloc(ratio, coff, /*max_s=*/std::max<int64_t>(s, 1), /*cache_slots=*/kMaxPos / ratio);
    CK(cudaMemcpy(gpu.d_wkv, wkv16.data(), wkv16.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gpu.d_wgate, wgate16.data(), wgate16.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gpu.d_ape, ref.ape.data(), ref.ape.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gpu.d_norm, ref.norm.data(), ref.norm.size() * 4, cudaMemcpyHostToDevice));
    gpu.w.wkv = gpu.d_wkv;
    gpu.w.wgate = gpu.d_wgate;
    gpu.w.ape = gpu.d_ape;
    gpu.w.norm = gpu.d_norm;

    auto state_cmp = [&](const char* tag) {
        const size_t n = (size_t) (coff * ratio * W);
        std::vector<float> h_kv(n), h_sc(n);
        CK(cudaMemcpy(h_kv.data(), gpu.st.kv_state, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_sc.data(), gpu.st.score_state, n * 4, cudaMemcpyDeviceToHost));
        char b[80];
        std::snprintf(b, sizeof b, "%s state.kv", tag);
        report(b, cmp(h_kv, ref.kv_state, b), 0.999999, 1e-3);
        std::snprintf(b, sizeof b, "%s state.score", tag);
        // -inf padding rows: compare only finite entries
        std::vector<float> a2, b2;
        for (size_t i = 0; i < n; ++i)
            if (std::isfinite(ref.score_state[i])) {
                a2.push_back(h_sc[i]);
                b2.push_back(ref.score_state[i]);
            }
        report(b, cmp(a2, b2, b), 0.999999, 1e-3);
        // the -inf / zero padding must be exactly preserved
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(ref.score_state[i]) && h_sc[i] != ref.score_state[i]) ++bad;
        if (bad) {
            ++g_fail;
            std::printf("  %-34s %zu padding cells corrupted FAIL\n", "state.score padding", bad);
        }
    };
    auto step = [&](int64_t start_pos, int64_t len, const char* tag) {
        CK(cudaMemcpy(gpu.d_x, x.data() + (size_t) start_pos * kNEmbd, (size_t) len * kNEmbd * 4,
                      cudaMemcpyHostToDevice));
        std::vector<float> ref_kvc;
        const int64_t ref_n =
            ref.forward(x.data() + (size_t) start_pos * kNEmbd, len, start_pos, ref_kvc);
        const int64_t gpu_n =
            dsv4_comp_forward(gpu.w, gpu.st, gpu.ws, gpu.d_x, len, start_pos, rope_dev, 1e-6,
                              gpu.d_cache, gpu.d_kvc, gpu.stream);
        CK(cudaStreamSynchronize(gpu.stream));
        if (gpu_n != ref_n) {
            ++g_fail;
            std::printf("  ncomp mismatch gpu %lld vs ref %lld FAIL\n", (long long) gpu_n,
                        (long long) ref_n);
            return;
        }
        if (ref_n > 0) {
            std::vector<float> h_kvc((size_t) ref_n * kHeadDim);
            CK(cudaMemcpy(h_kvc.data(), gpu.d_kvc, h_kvc.size() * 4, cudaMemcpyDeviceToHost));
            char b[80];
            std::snprintf(b, sizeof b, "%s kv_c", tag);
            report(b, cmp(h_kvc, ref_kvc, b), 0.999999, 1e-3);
            // cache slots: slot0 = prefill ? 0 : start_pos/ratio
            const int64_t slot0 = start_pos == 0 ? 0 : start_pos / ratio;
            std::vector<float> h_cache((size_t) ref_n * kHeadDim);
            CK(cudaMemcpy(h_cache.data(), gpu.d_cache + (size_t) slot0 * kHeadDim,
                          h_cache.size() * 4, cudaMemcpyDeviceToHost));
            std::snprintf(b, sizeof b, "%s cache slots", tag);
            report(b, cmp(h_cache, h_kvc, b), 1.0, 0.0);   // cache slots are a bitwise copy of kv_c
        }
        state_cmp(tag);
    };

    step(0, s, "prefill");
    for (int64_t k = 0; k < ndec; ++k) {
        char tag[64];
        std::snprintf(tag, sizeof tag, "decode sp=%lld", (long long) (s + k));
        step(s + k, 1, tag);
    }
    gpu.free();
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep stdout/stderr interleaving truthful in logs
    uint32_t seed = 7;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t) std::stoul(argv[++i]);
    CK(cudaSetDevice(0));

    // YaRN compress rope table (theta 160000, orig 65536, factor 16, beta 32/1), n_rot = 64
    std::vector<float> cosv, sinv;
    rope_build(64, kMaxPos, 65536, 160000.0, 16.0, 32.0, 1.0, cosv, sinv);
    float *d_cos = nullptr, *d_sin = nullptr;
    CK(cudaMalloc(&d_cos, cosv.size() * 4));
    CK(cudaMalloc(&d_sin, sinv.size() * 4));
    CK(cudaMemcpy(d_cos, cosv.data(), cosv.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sin, sinv.data(), sinv.size() * 4, cudaMemcpyHostToDevice));
    strata::dsv4::Dsv4RopeTables rope_dev;
    rope_dev.cos = d_cos;
    rope_dev.sin = d_sin;
    rope_dev.half = 32;

    run_case("A r128 prefill 13 (no pooling)", 128, 1, 13, 0, seed, cosv, sinv, rope_dev);
    run_case("B r128 prefill 300", 128, 1, 300, 0, seed + 1, cosv, sinv, rope_dev);
    run_case("C r4 prefill 13 (overlap)", 4, 2, 13, 0, seed + 2, cosv, sinv, rope_dev);
    run_case("D r4 decode 13..19 (shift)", 4, 2, 13, 7, seed + 3, cosv, sinv, rope_dev);
    run_case("E r128 decode to t=384", 128, 1, 300, 84, seed + 4, cosv, sinv, rope_dev);

    cudaFree(d_cos);
    cudaFree(d_sin);
    if (g_fail) {
        std::printf("dsv4_comp_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("dsv4_comp_test: all cases PASS\n");
    return 0;
}
