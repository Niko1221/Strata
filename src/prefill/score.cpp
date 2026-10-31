// src/prefill/score.cpp - see include/strata/prefill/score.hpp.
//
// Pipelines per chunk (T positions):
//
//   R_rows (T, hc*n_embd)          the prompt path's final residual, one hyper-connection stack per position
//     | fused_gr_read_multi        the FINAL mixer: collapse hc streams to one n_embd vector (no injection)
//   mixed  (T, n_embd) fp32
//     | f32_to_f16
//   mixed16 (T, n_embd) fp16       what `Gemm` takes
//     | Gemm::f16 / Gemm::native   one batched product per row block: (rows, n_embd) x (n_embd, n_vocab)
//   logits (rows, n_vocab) fp32
//     | score_rows                 log p(target) = logits[target] - logsumexp over the WHOLE vocabulary
//   out[pos0 .. pos0+rows) fp32, host
//
// WHY THE HEAD IS A GEMM AND NOT `lm_head`.  `core::lm_head` is a single-row GEMV: 318 MB of `output.weight` per
// token.  A 128K-token prompt would read 40 TB (about four hours) to score it.  Here the weight is dequantized ONCE
// into an FP16 image (1.27 GB at this vocabulary) and every row block multiplies into it, so the same prompt costs
// 128K x 2560 x 248320 x 2 = 325 TFLOP: a few seconds at the card's FP16 throughput.  When that image does not fit,
// `Gemm::native` dequantizes per block instead - correct, and 40x slower on weight traffic, so it is the fallback.
#include "strata/prefill/score.hpp"

#include "strata/core/geometry.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/score_rows.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace strata::prefill {
namespace {

// The final mixer's RMSNorm epsilon, the same constant `core::layer`'s `lm_head_mix` uses.  Two places computing
// the head's normalisation must not disagree about it.
constexpr float kRmsEps = 1e-6f;

// Rows per head block before shrinking.  1024 rows x 248320 floats = 1.02 GB of logits; on a card that cannot give
// that, the block halves until the allocation succeeds (down to kMinRows).  The block matters because the weight is
// streamed once per block on the fallback path, and the product is one launch either way.
constexpr int64_t kFirstRows = 1024;
constexpr int64_t kMinRows = 16;

// VRAM this path always leaves to the ENGINE.  The expert cache takes nearly the whole card, so what a scoring run
// may use is only the slack - and the prompt path itself allocates per-request buffers on top (the MMQ expert
// staging, the verify window).  Taking that slack away does not fail loudly: the engine's expert path silently
// produces non-finite rows (measured: a 20-token prompt with 2 valid rows, layer 3's GU/H buffers half NaN, and a
// GEN over the same tokens still fine, because only a scoring run had taken the memory).
constexpr size_t kEngineFloorBytes = size_t{768} << 20;

int64_t env_i64(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    return (int64_t) std::atoll(v);
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

bool env_off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0';
}

void dev_free(void*& p) {
    if (p != nullptr) {
        cudaFree(p);
        p = nullptr;
    }
}

// cudaMalloc that says what it failed on.  A scoring run that silently used less memory than it asked for shows up
// as a mysterious slowdown, not as the allocation problem it is.
bool dev_alloc(void*& p, size_t bytes, const char* what, std::string& err) {
    p = nullptr;
    if (bytes == 0) return true;
    const cudaError_t rc = cudaMalloc(&p, bytes);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        err = std::string("score: ") + what + " (" + std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(rc);
        return false;
    }
    return true;
}

}  // namespace

struct Scorer::Impl {
    Prefill* pf = nullptr;
    const core::WeightTable* wt = nullptr;
    core::ModelGeometry g;
    void* stream = nullptr;

    Gemm gemm;
    bool gemm_ready = false;

    // The head's weights.
    const core::WeightRef* wo = nullptr;   ///< output.weight (its shape; its blocks unless `head` has them)
    const core::NativeHead* head = nullptr;  ///< --native's output projection: the GGUF blocks the engine actually uses
    const core::WeightRef* wn = nullptr;   ///< output_hc_norm.weight  (f32)
    const core::WeightRef* wd = nullptr;   ///< output_hc_down.weight  (bf16)
    const core::WeightRef* wu = nullptr;   ///< output_hc_up.weight    (bf16)
    uint16_t* w16 = nullptr;               ///< the FP16 image of output.weight, when it fit

    /// The GGUF blocks to multiply by, and their type: the native head's when it is loaded (`--native` reads
    /// `output.weight` into it and leaves the table's copy canonical), otherwise the table's own.
    const void* wblocks() const { return head != nullptr ? head->weights() : wo->native_data; }
    int wtype() const { return head != nullptr ? head->type() : wo->native_type; }

    // Per-chunk scratch, grown to the largest chunk seen.  `xn` is the multi read's scratch, which is per SLICE of
    // kFusedGrMaxT tokens - so it is that size, not the chunk's: at a 8192-token chunk the difference is 335 MB of
    // VRAM that the engine's own expert staging can use instead.
    float* mixed = nullptr;      ///< T x n_embd
    uint16_t* mixed16 = nullptr; ///< T x n_embd
    float* xn = nullptr;         ///< kFusedGrMaxT x hc*n_embd (fused_gr_read_multi's scratch, one slice)
    float* lo = nullptr;         ///< T x hc_lr
    float* rs = nullptr;         ///< T x hc
    float* inj = nullptr;        ///< T x hc
    int64_t tok_cap = 0;
    std::vector<kernels::FusedGrArgs> args;

    // The head's block.
    float* logits = nullptr;   ///< rows x n_vocab
    float* lse = nullptr;      ///< score_rows scratch
    float* lp = nullptr;       ///< rows, device
    int32_t* tgt = nullptr;    ///< n, device (the target token per position)
    int64_t tgt_cap = 0;
    int64_t rows = 0;

    float* out_rows = nullptr;   ///< the caller's host output for this run
    bool warned_loan = false;    ///< the lent-buffers warning went out (see run)
    int64_t scored = 0;
    double ms_head = 0;
    std::string note;            ///< what the head's layout cost, for the start-up line (see ensure_block)
    size_t free_before = 0;      ///< the free VRAM that choice was made from

    ~Impl() { release(); }

    bool ensure_tokens(int64_t T, std::string& err);
    bool ensure_targets(int64_t n, std::string& err);
    bool ensure_block(std::string& err);
    std::string head_why() const;
    bool score_chunk(const float* R_rows, int64_t T, int64_t pos0, std::string& err);
    void release_tokens();
    void release_head();
    void release();
};

void Scorer::Impl::release_tokens() {    dev_free((void*&) mixed);
    dev_free((void*&) mixed16);
    dev_free((void*&) xn);
    dev_free((void*&) lo);
    dev_free((void*&) rs);
    dev_free((void*&) inj);
    tok_cap = 0;
}

// The head's working set lives for ONE run: the expert cache has taken the card, so a scorer that holds 2.3 GB
// between runs leaves the engine's own per-request buffers nowhere to go.
void Scorer::Impl::release_head() {
    dev_free((void*&) logits);
    dev_free((void*&) lse);
    dev_free((void*&) lp);
    dev_free((void*&) w16);
    // `rows` and `note` stay: the caller reports them for the run that just finished.
}

void Scorer::Impl::release() {
    release_tokens();
    release_head();
    dev_free((void*&) tgt);
    tgt_cap = 0;
    out_rows = nullptr;
    scored = 0;
    ms_head = 0;
    rows = 0;
    note.clear();
    free_before = 0;
}

bool Scorer::Impl::ensure_tokens(int64_t T, std::string& err) {
    if (T <= tok_cap) return true;
    release_tokens();
    const int64_t hc = g.hc, ne = g.n_embd, lr = g.hc_lr;
    if (!dev_alloc((void*&) mixed, (size_t) T * ne * sizeof(float), "the mixer output", err) ||
        !dev_alloc((void*&) mixed16, (size_t) T * ne * sizeof(uint16_t), "the mixer's FP16 image", err) ||
        !dev_alloc((void*&) xn, (size_t) kernels::kFusedGrMaxT * hc * ne * sizeof(float),
                   "the mixer's stream scratch", err) ||
        !dev_alloc((void*&) lo, (size_t) T * lr * sizeof(float), "the mixer's bottleneck", err) ||
        !dev_alloc((void*&) rs, (size_t) T * hc * sizeof(float), "the mixer's norms", err) ||
        !dev_alloc((void*&) inj, (size_t) T * hc * sizeof(float), "the mixer's injections", err)) {
        release_tokens();
        return false;
    }
    tok_cap = T;
    args.resize((size_t) std::min<int64_t>(T, kernels::kFusedGrMaxT));
    return true;
}

bool Scorer::Impl::ensure_targets(int64_t n, std::string& err) {
    if (n <= tgt_cap) return true;
    dev_free((void*&) tgt);
    if (!dev_alloc((void*&) tgt, (size_t) n * sizeof(int32_t), "the target tokens", err)) return false;
    tgt_cap = n;
    return true;
}

bool Scorer::Impl::ensure_block(std::string& err) {
    if (logits != nullptr) return true;
    // What this run may take: the free VRAM less the engine's floor, less whatever this run already took (the token
    // scratch).  Taken NOW rather than at attach, because a scorer that holds 2.3 GB between runs leaves the
    // engine's expert path with nothing - and it then produces non-finite rows instead of failing.
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        cudaGetLastError();
        err = "score: the free VRAM is unknown";
        return false;
    }
    const size_t budget = free_b > kEngineFloorBytes ? free_b - kEngineFloorBytes : 0;
    free_before = free_b;
    const int64_t V = g.n_vocab, K = g.n_embd;
    const size_t image = (size_t) V * K * sizeof(uint16_t);
    const size_t min_block = (size_t) kMinRows * (size_t) V * sizeof(float);
    const int64_t cap = std::max<int64_t>(kMinRows, std::min<int64_t>(env_i64("STRATA_SCORE_ROWS", kFirstRows),
                                                                     kFirstRows));
    // The image first: with the whole weight in FP16 every row block is a pure product, so blocks may be small.  It
    // is only worth taking when a block still fits beside it - otherwise the card is better served by the largest
    // block it can hold and the per-block dequantize `Gemm::native` does (see head_why for the arithmetic).
    if (!env_off("STRATA_SCORE_WCACHE") && budget >= image + min_block && wo->ne1 == V && wo->ne0 == K &&
        wblocks() != nullptr && wtype() >= 0) {
        void* p = nullptr;
        std::string why;
        if (dev_alloc(p, image, "the head's FP16 image", why)) {
            // One launch for the whole head: `dequant_f16` writes at the buffer's start whatever `row0` is (it moves
            // the SOURCE window), so slicing here would overwrite the same place V/4096 times.
            kernels::dequant_f16(wtype(), wblocks(), 0, V, K, (uint16_t*) p, stream);
            if (cudaStreamSynchronize((cudaStream_t) stream) == cudaSuccess) {
                w16 = (uint16_t*) p;
            } else {
                const cudaError_t rc = cudaGetLastError();
                cudaFree(p);
                cudaGetLastError();
                note = std::string("the head's FP16 image: the dequantize failed (") + cudaGetErrorString(rc) + ")";
            }
        } else {
            note = why;
        }
    }
    // The logits block: as many rows as the rest of the budget holds, never more than `cap`.
    const size_t left = budget > image && w16 != nullptr ? budget - image : budget;
    int64_t want = std::min<int64_t>(cap, (int64_t) (left / ((size_t) V * sizeof(float))));
    want = std::max<int64_t>(kMinRows, want);
    std::string first_err;
    for (int64_t r = want; r >= kMinRows; r /= 2) {
        void *lg = nullptr, *sl = nullptr, *pl = nullptr;
        std::string why;
        if (dev_alloc(lg, (size_t) r * (size_t) V * sizeof(float), "the logits block", why) &&
            dev_alloc(sl, (size_t) score_rows_scratch_bytes((int) r), "the log-probability scratch", why) &&
            dev_alloc(pl, (size_t) r * sizeof(float), "the row log-probabilities", why)) {
            logits = (float*) lg;
            lse = (float*) sl;
            lp = (float*) pl;
            rows = r;
            char line[256];
            std::snprintf(line, sizeof line, "the head %s, %zu MiB free before the run, a %lld-row block",
                          w16 != nullptr ? ("as a " + std::to_string(image >> 20) + " MiB FP16 image").c_str()
                                         : ("dequantized per block - " + head_why()).c_str(),
                          free_before >> 20, (long long) r);
            note = line;
            return true;
        }
        dev_free(lg);
        dev_free(sl);
        dev_free(pl);
        if (first_err.empty()) first_err = why;
    }
    err = "score: no row block fits beside the engine's floor: " + first_err;
    return false;
}

// Why the head is not in FP16, when it is not: the arithmetic above decided it, and a scoring run 40x slower than
// it should be has to say why.
std::string Scorer::Impl::head_why() const {
    const int64_t V = g.n_vocab, K = g.n_embd;
    const size_t image = (size_t) V * K * sizeof(uint16_t);
    if (env_off("STRATA_SCORE_WCACHE")) return "STRATA_SCORE_WCACHE=0";
    if (wblocks() == nullptr || wtype() < 0) return "output.weight has no GGUF blocks";
    if (wo->ne1 != V || wo->ne0 != K)
        return "output.weight is " + std::to_string(wo->ne0) + " x " + std::to_string(wo->ne1) + ", not " +
               std::to_string(K) + " x " + std::to_string(V);
    char why[192];
    std::snprintf(why, sizeof why, "%zu MiB free, and the image plus a %lld-row block needs more",
                  (size_t) ((free_before >> 20)), (long long) kMinRows);
    return why;
}

bool Scorer::Impl::score_chunk(const float* R_rows, int64_t T, int64_t pos0, std::string& err) {
    if (T <= 0) return true;
    if (!ensure_tokens(T, err) || !ensure_block(err)) return false;
    const int64_t hc = g.hc, ne = g.n_embd, lr = g.hc_lr;
    const double t0 = now_ms();
    if (std::getenv("STRATA_SCORE_DEBUG") != nullptr) {
        // Per ROW, not per block: a row of R_rows is one position's hc*n_embd residuals, so this says directly how
        // many positions `on_chunk` actually handed over (a T that outruns the data is the bug this looks for).
        std::vector<float> hr((size_t) (T * hc * ne));
        cudaMemcpy(hr.data(), R_rows, (size_t) T * hc * ne * sizeof(float), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "score dbg: R_rows as %lld rows of %lld floats:\n", (long long) T, (long long) (hc * ne));
        int64_t bad = 0;
        for (int64_t t = 0; t < T; ++t) {
            const float* r = hr.data() + (size_t) t * hc * ne;
            int64_t nonfinite = 0;
            double first = 0.0;
            for (int64_t i = 0; i < hc * ne; ++i) {
                if (!std::isfinite(r[i])) ++nonfinite;
                if (i == 0) first = r[i];
            }
            if (nonfinite > 0) ++bad;
            if (t < 4 || nonfinite > 0 || t == T - 1)
                std::fprintf(stderr, "  row %3lld: %.4f ... %lld of %lld non-finite%s\n", (long long) t, first,
                             (long long) nonfinite, (long long) (hc * ne), nonfinite == 0 ? "" : "  <--");
        }
        std::fprintf(stderr, "score dbg: %lld of %lld rows have non-finite values\n", (long long) bad, (long long) T);
        std::fflush(stderr);
    }

    // 1. The final mixer, kFusedGrMaxT positions at a time (the multi read's own cap).
    for (int64_t base = 0; base < T; base += kernels::kFusedGrMaxT) {
        const int64_t nt = std::min<int64_t>(kernels::kFusedGrMaxT, T - base);
        for (int64_t i = 0; i < nt; ++i) {
            const int64_t t = base + i;
            kernels::FusedGrArgs& a = args[(size_t) i];
            a = kernels::FusedGrArgs{};
            a.R = R_rows + (size_t) t * hc * ne;
            a.R_out = const_cast<float*>(a.R);
            a.apply = false;   // nothing was computed after this read: no half to fold in
            a.w_norm = (const float*) wn->data;
            a.w_down = (const uint16_t*) wd->data;
            a.w_up = (const uint16_t*) wu->data;
            a.w_inject = nullptr;   // the FINAL mixer has no write-back
            a.eps = kRmsEps;
            a.lo = lo + (size_t) t * lr;
            a.rs = rs + (size_t) t * hc;
            a.inject_out = inj + (size_t) t * hc;
            a.mixed = mixed + (size_t) t * ne;
        }
        // The return value says whether it also wrote q8_1 images (STRATA_QFUSE), not whether it worked: this read
        // wants no q8 image, so the answer is false and that is fine.
        kernels::fused_gr_read_multi(args.data(), (int) nt, xn, stream);
        const cudaError_t rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            err = std::string("score: the final mixer failed: ") + cudaGetErrorString(rc);
            return false;
        }
    }

    // 2. FP16 image for the head's GEMM.
    if (!f32_to_f16(mixed, mixed16, T * ne, stream)) {
        err = "score: the mixer's FP16 conversion failed";
        return false;
    }
    if (std::getenv("STRATA_SCORE_DEBUG") != nullptr) {
        std::vector<float> hm((size_t) (T * ne));
        cudaMemcpy(hm.data(), mixed, (size_t) T * ne * sizeof(float), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "score dbg: w16 %s; mixed rows (all %lld):\n", w16 ? "cached" : "per block", (long long) T);
        int64_t nmix_bad = 0, nmix_zero = 0;
        for (int64_t t = 0; t < T; ++t) {
            const float* r = hm.data() + (size_t) t * ne;
            int64_t nz = 0, nf = 0;
            for (int64_t i = 0; i < ne; ++i) {
                if (r[i] == 0.0f) ++nz;
                if (!std::isfinite(r[i])) ++nf;
            }
            if (nz == ne) ++nmix_zero;
            if (nf > 0) ++nmix_bad;
            if (t < 3 || nz == ne || nf > 0)
                std::fprintf(stderr, "  row %3lld: %.4f %.4f %.4f | %lld non-finite, %lld zero of %lld\n", (long long) t,
                             (double) r[0], (double) r[1], (double) r[2], (long long) nf, (long long) nz,
                             (long long) ne);
        }
        std::fprintf(stderr, "score dbg: mixed: %lld rows all zero, %lld rows with non-finite values\n",
                     (long long) nmix_zero, (long long) nmix_bad);
        std::fflush(stderr);
    }

    // 3. The head, one row block at a time.
    for (int64_t r0 = 0; r0 < T; r0 += rows) {
        const int64_t n = std::min(rows, T - r0);
        if (w16 != nullptr) {
            // Y (n, V) = X (n, K) . W (V, K)^T, with W already dequantized.
            gemm.f16(mixed16 + (size_t) r0 * ne, w16, logits, n, g.n_vocab, ne);
        } else {
            gemm.native(mixed16 + (size_t) r0 * ne, wtype(), wblocks(), logits, n, g.n_vocab, ne);
        }
        if (!score_rows(logits, (int) n, (int) g.n_vocab, g.n_vocab, tgt + pos0 + r0, lse, lp, stream)) {
            err = "score: the log-probability kernel failed";
            return false;
        }
        // The row log-probabilities go straight to their host slots: rows x 4 bytes, nothing beside the product.
        if (cudaMemcpyAsync(out_rows + pos0 + r0, lp, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost,
                            (cudaStream_t) stream) != cudaSuccess) {
            err = "score: copying the row log-probabilities to the host failed";
            return false;
        }
        if (std::getenv("STRATA_SCORE_DEBUG") != nullptr) {
            std::vector<float> hl((size_t) n), hlg(4);
            std::vector<int32_t> ht((size_t) n);
            cudaMemcpy(hl.data(), lp, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(ht.data(), tgt + pos0 + r0, (size_t) n * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(hlg.data(), logits, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "score dbg: pos0 %lld T %lld block %lld n %lld vocab %lld logits[0][0..3] %.3f %.3f %.3f %.3f\n",
                         (long long) pos0, (long long) T, (long long) rows, (long long) n, (long long) g.n_vocab,
                         hlg[0], hlg[1], hlg[2], hlg[3]);
            for (int64_t i = 0; i < n; ++i)
                std::fprintf(stderr, "score dbg:   row %lld tgt %d lp %.6f\n", (long long) i, ht[(size_t) i],
                             (double) hl[(size_t) i]);
            // A/B against the DECODE path's own head on the SAME mixer output: `NativeHead::run` is the single-row
            // MMVQ the engine scores generated tokens with (Q8 activations where this path uses FP16).  If these
            // two disagree, the batched head is wrong; if they agree, any difference from a GEN logprob is the
            // prompt path and the decode path disagreeing about the residual, not about the head.
            if (n >= 1 && head != nullptr && head->loaded()) {
                float* dref = nullptr;
                if (cudaMalloc(&dref, (size_t) g.n_vocab * sizeof(float)) == cudaSuccess) {
                    std::string e2;
                    if (head->run(mixed, dref, stream, e2)) {
                        std::vector<float> ref((size_t) g.n_vocab), mine((size_t) g.n_vocab);
                        cudaMemcpy(ref.data(), dref, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(mine.data(), logits, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost);
                        double m1 = -1e30, s1 = 0.0, m2 = -1e30, s2 = 0.0, worst = 0.0;
                        int64_t wi = -1;
                        for (int64_t v = 0; v < g.n_vocab; ++v) {
                            m1 = std::max(m1, (double) ref[(size_t) v]);
                            m2 = std::max(m2, (double) mine[(size_t) v]);
                        }
                        for (int64_t v = 0; v < g.n_vocab; ++v) {
                            s1 += std::exp((double) ref[(size_t) v] - m1);
                            s2 += std::exp((double) mine[(size_t) v] - m2);
                            const double d = std::fabs((double) ref[(size_t) v] - (double) mine[(size_t) v]);
                            if (d > worst) { worst = d; wi = v; }
                        }
                        const double lse1 = m1 + std::log(s1), lse2 = m2 + std::log(s2);
                        const int t0 = ht[0];
                        const double lp1 = (double) ref[(size_t) t0] - lse1;
                        const double lp2 = (double) mine[(size_t) t0] - lse2;
                        std::fprintf(stderr,
                                     "score dbg: A/B row %lld tgt %d: decode head %.6f, batched %.6f, diff %.6f "
                                     "(logsumexp %.4f vs %.4f, worst logit %.4f at id %lld: %.3f vs %.3f)\n",
                                     (long long) pos0, t0, lp1, lp2, lp2 - lp1, lse1, lse2, worst, (long long) wi,
                                     (double) ref[(size_t) wi], (double) mine[(size_t) wi]);
                    } else {
                        std::fprintf(stderr, "score dbg: A/B head failed: %s\n", e2.c_str());
                    }
                    cudaFree(dref);
                }
            }
            std::fflush(stderr);
        }
    }
    ms_head += now_ms() - t0;
    scored += T;
    return true;
}

// ------------------------------------------------------------------ public

Scorer::Scorer() : impl_(std::make_unique<Impl>()) {}
Scorer::~Scorer() = default;

bool Scorer::attach(Prefill& pf, const core::WeightTable& wt, const core::ModelGeometry& g,
                    const core::NativeHead* head, void* stream, std::string& err) {
    Impl& s = *impl_;
    s.release();
    s.pf = &pf;
    s.wt = &wt;
    s.g = g;
    s.stream = stream;
    s.head = (head != nullptr && head->loaded()) ? head : nullptr;

    s.wo = wt.find("output.weight");
    s.wn = wt.find("output_hc_norm.weight");
    s.wd = wt.find("output_hc_down.weight");
    s.wu = wt.find("output_hc_up.weight");
    if (s.wo == nullptr) { err = "score: output.weight is missing"; return false; }
    if (s.wblocks() == nullptr || s.wtype() < 0) {
        err = "score: output.weight has no GGUF blocks and no native head (the engine must run with --native)";
        return false;
    }
    if (s.wn == nullptr || s.wd == nullptr || s.wu == nullptr) {
        err = "score: an output_hc_* weight is missing";
        return false;
    }
    if (s.wn->kind != core::WeightKind::F32 || s.wd->kind != core::WeightKind::Bf16InF32 ||
        s.wu->kind != core::WeightKind::Bf16InF32) {
        err = "score: the output_hc_* weights have the wrong engine forms";
        return false;
    }
    if (!kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr)) {
        err = "score: this geometry has no multi-token hyper-connection read";
        return false;
    }

    // The GEMM's scratch: enough for a slice of the head, which `Gemm::native` cuts further if it must.
    if (!s.gemm.init(stream, std::max<int64_t>(g.n_embd * 64, int64_t{8} << 20), err)) {
        err = "score: the head's GEMM: " + err;
        return false;
    }
    s.gemm_ready = true;
    // Nothing is allocated here: the head's working set is taken for one run and given back (see release_head),
    // because the expert cache has already claimed the card and the engine needs the slack between runs.
    return true;
}

bool Scorer::run(const int64_t* tokens, int64_t n, const int32_t* targets, float* out, std::string& err) {
    Impl& s = *impl_;
    if (s.pf == nullptr || !s.gemm_ready) { err = "score: not attached"; return false; }
    if (n <= 0 || tokens == nullptr || targets == nullptr || out == nullptr) {
        err = "score: nothing to score";
        return false;
    }
    // The prompt path's buffers are either its own or lent expert-cache slots.  Over a loan the ROWS are not ours to
    // trust: measured on this rig (4090, Swift IQ3_XXS, --prefill auto), an 8-token SCORE sums to -79.39 over a
    // 2,485-slot loan, to -56.60 over a 201-slot one and to -56.11 over the path's own 8192-token buffers (the
    // Hugging Face teacher-forced reference is -56.64) - and the engine's own GEN reads non-finite logits over the
    // same loan, so this is the prompt path, not this file.  Say it once, with the way out, instead of failing.
    if (s.pf->borrowed() && !s.warned_loan) {
        s.warned_loan = true;
        std::fprintf(stderr,
                     "strata score: the prompt path's buffers are lent expert-cache slots (a %lld-token layout).  "
                     "Scores over such a loan can be wrong and the engine's own prompt reads can return non-finite "
                     "rows.  Run the engine with --no-prefill-borrow for scores a training step consumes.\n",
                     (long long) s.pf->chunk());
        std::fflush(stderr);
    }
    if (!s.ensure_targets(n, err)) return false;
    if (cudaMemcpyAsync(s.tgt, targets, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice,
                        (cudaStream_t) s.stream) != cudaSuccess) {
        err = "score: uploading the target tokens failed";
        return false;
    }
    s.out_rows = out;
    s.scored = 0;
    s.ms_head = 0;

    auto previous = s.pf->on_chunk;
    s.pf->on_chunk = [&s](const float* R_rows, int64_t T, int64_t pos0, std::string& e) {
        return s.score_chunk(R_rows, T, pos0, e);
    };
    const bool ok = s.pf->run(tokens, n, 0, err);
    s.pf->on_chunk = std::move(previous);
    // Whatever the run did, the head's working set and the mixer scratch go back to the engine now: the expert cache
    // has the card, and holding 2.3 GB against it breaks its own prompt path (see kEngineFloorBytes).  The rows and
    // the note stay for the caller to report.
    if (ok && cudaStreamSynchronize((cudaStream_t) s.stream) != cudaSuccess) {
        err = "score: the device did not finish the run";
        s.release_head();
        s.release_tokens();
        return false;
    }
    s.release_head();
    s.release_tokens();
    if (!ok) return false;
    // A NaN here is never this path's arithmetic: the head runs in FP32 over rows the prompt path handed over, so a
    // gap means the prompt path itself did not produce a finite residual (a broken gather table, a short chunk read
    // in a layout sized for a longer one - #879 catches the same thing one request later, on the logits).  Say so
    // with the position, rather than handing a training loop a sum that poisons the whole step.
    for (int64_t t = 0; t + 1 < n; ++t) {
        if (!std::isfinite(out[t])) {
            err = "the prompt path's residual was not finite at position " + std::to_string(t) + " of " +
                  std::to_string(n - 1) + " scored (" + std::to_string(s.scored) + " rows came back)";
            return false;
        }
    }
    return true;
}

int64_t Scorer::scored() const { return impl_->scored; }
double Scorer::ms_head() const { return impl_->ms_head; }
int64_t Scorer::rows_per_block() const { return impl_->rows; }
const std::string& Scorer::note() const { return impl_->note; }

}  // namespace strata::prefill
