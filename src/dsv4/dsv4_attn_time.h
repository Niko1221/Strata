// src/dsv4/dsv4_attn_time.h - [M1] attention accounting, off by default.
//
// STRATA_DSV4_ATTNTIME=1 records one pooled CUDA event pair around every attention launch (and around
// the host-side topk table build) and prints, at process exit, the GPU time per category for each of
// the three attention layer classes (hash r=0 / compressor r=128 / indexer r=4).  The pairs are only
// read once, at exit, so the pipeline is never synchronised mid-run; the events themselves add a
// timestamp write per section (a fraction of a percent of a 4k prefill).
//
// Attribution only: like STRATA_DSV4_Q8TIME this is a diagnostic, not a benchmark path.
//
// The pool index is taken in begin() (never reused), so scopes may nest: each scope owns its pair.

#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::dsv4 {

enum Dsv4AttnTimeCat {
    kAtHc = 0,      // hc_mix + hc_fin (pre) / hc_post
    kAtNorm,        // rmsnorm / rmsnorm+rope
    kAtXpose,       // xpose1024 / xpose2048
    kAtRope,        // rope only (inverse rope included)
    kAtProj,        // the six Q8_0 projections (wq_a wq_b wkv wo_a wo_b, indexer q)
    kAtMqa,         // mqa512 launches, including the per-row ring copies inside the loop
    kAtCompGemm,    // compressor BF16 (double-accumulated) projections
    kAtCompPool,    // compressor ape store + pooling (prefill / chunk / decode)
    kAtIdx,         // indexer weights gemm + dp/score/select
    kAtMem,         // kv cache / staging device-to-device copies
    kAtTableH2D,    // topk table upload (pageable H2D / 2-D H2D)
    kAtN
};

enum Dsv4AttnTimeClass { kAtHash = 0, kAtR128 = 1, kAtIdxLayer = 2, kAtClsN = 3 };

struct AttnTimeAcc {
    static constexpr int kMax = 1 << 18;  // event pairs; 2 * 262144 timestamps
    bool on = false;
    int cls = kAtHash;
    int used = 0;
    int depth = 0;
    long nested = 0;
    long bad_elapsed = 0;
    cudaEvent_t* ev = nullptr;
    unsigned char* cat = nullptr;
    unsigned char* klass = nullptr;
    double ms[kAtClsN][kAtN] = {};
    long n[kAtClsN][kAtN] = {};
    double host_table_ms = 0.0;
    long host_table_n = 0;
    AttnTimeAcc() {
        const char* e = std::getenv("STRATA_DSV4_ATTNTIME");
        on = e != nullptr && e[0] != '\0' && e[0] != '0';
        if (!on) return;
        ev = (cudaEvent_t*) std::calloc((size_t) 2 * kMax, sizeof(cudaEvent_t));
        cat = (unsigned char*) std::malloc(kMax);
        klass = (unsigned char*) std::malloc(kMax);
        if (ev == nullptr || cat == nullptr || klass == nullptr) {
            std::fprintf(stderr, "dsv4_attntime: out of host memory, timing disabled\n");
            on = false;
        }
    }
    bool flushed = false;
    ~AttnTimeAcc() { flush(); }
    void flush() {
        if (!on || flushed) return;
        flushed = true;
        const char* first_err = "none";
        for (int i = 0; i < used; ++i) {
            if (ev[2 * i] == nullptr || ev[2 * i + 1] == nullptr) {
                ++bad_elapsed;
                if (first_err[0] == 'n') first_err = "null event";
                continue;
            }
            cudaError_t e0 = cudaEventSynchronize(ev[2 * i + 1]);
            float el = 0.0f;
            cudaError_t e1 = cudaEventElapsedTime(&el, ev[2 * i], ev[2 * i + 1]);
            if (e0 != cudaSuccess || e1 != cudaSuccess) {
                ++bad_elapsed;
                if (first_err[0] == 'n')
                    first_err = cudaGetErrorString(e1 != cudaSuccess ? e1 : e0);
                continue;
            }
            ms[klass[i]][cat[i]] += (double) el;
            ++n[klass[i]][cat[i]];
        }
        static const char* nm[kAtN] = {"hc",        "norm",      "xpose",     "rope",      "q8proj",
                                       "mqa512",    "comp-gemm", "comp-pool", "indexer",   "memcpy",
                                       "table-h2d"};
        static const char* cn[kAtClsN] = {"hash r=0", "comp r=128", "idx  r=4"};
        for (int c = 0; c < kAtClsN; ++c) {
            double tot = 0.0;
            long calls = n[c][kAtHc];  // one hc_pre per attention call
            for (int k = 0; k < kAtN; ++k) tot += ms[c][k];
            std::fprintf(stderr, "dsv4_attntime[%s]: gpu %9.3f ms over %ld calls\n", cn[c], tot,
                         calls);
            for (int k = 0; k < kAtN; ++k)
                if (n[c][k])
                    std::fprintf(stderr, "  %-10s n=%7ld total %9.3f ms (%8.4f ms/call)\n", nm[k],
                                 n[c][k], ms[c][k], ms[c][k] / (double) n[c][k]);
        }
        std::fprintf(stderr, "dsv4_attntime[host]: topk table build %.3f ms / %ld calls\n",
                     host_table_ms, host_table_n);
        std::fprintf(stderr,
                     "dsv4_attntime: event pairs %d / %d, nested begins %ld, unreadable pairs %ld (%s)\n",
                     used, kMax, nested, bad_elapsed, first_err);
    }
    int begin(int c, cudaStream_t st) {
        if (!on) return -1;
        if (depth++ > 0) ++nested;
        if (used >= kMax) return -1;
        const int i = used++;
        cat[i] = (unsigned char) c;
        klass[i] = (unsigned char) cls;
        if (ev[2 * i] == nullptr) cudaEventCreateWithFlags(&ev[2 * i], cudaEventDefault);
        if (ev[2 * i + 1] == nullptr) cudaEventCreateWithFlags(&ev[2 * i + 1], cudaEventDefault);
        cudaEventRecord(ev[2 * i], st);
        return i;
    }
    void end(int i, cudaStream_t st) {
        if (!on) return;
        if (depth > 0) --depth;
        if (i < 0) return;
        cudaEventRecord(ev[2 * i + 1], st);
    }
};

inline AttnTimeAcc g_attn_time;

// Records one pair for the enclosing scope (RAII: safe with early returns and with nesting).
struct AttnTimeScope {
    int idx;
    cudaStream_t st;
    AttnTimeScope(int c, cudaStream_t s) : idx(-1), st(s) {
        if (g_attn_time.on) idx = g_attn_time.begin(c, st);
    }
    ~AttnTimeScope() { g_attn_time.end(idx, st); }
    AttnTimeScope(const AttnTimeScope&) = delete;
    AttnTimeScope& operator=(const AttnTimeScope&) = delete;
};

inline void attn_time_host_table(double ms) {
    if (!g_attn_time.on) return;
    g_attn_time.host_table_ms += ms;
    ++g_attn_time.host_table_n;
}

inline void attn_time_set_class(int cls) { g_attn_time.cls = cls; }

// Reads the pooled pairs while the CUDA context is certainly alive (the engine calls this just before
// it prints its own summary); the destructor is only a fallback for paths that return early.
inline void attn_time_flush() { g_attn_time.flush(); }

}  // namespace strata::dsv4
