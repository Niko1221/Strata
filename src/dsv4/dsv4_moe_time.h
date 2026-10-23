// src/dsv4/dsv4_moe_time.h - [M1] MoE accounting, off by default.
//
// STRATA_DSV4_MOETIME=1 records one pooled CUDA event pair around every MoE GPU section (the three
// grouped GEMVs, swiglu, scatter, the staged blob upload, the in-stream wait for a staging event, the
// table upload, the peer launch set, the shared expert, the combine, hc_post) and a host-clock pair
// around the one blocking point of the host path (the route D2H drain).  At process exit it prints
// ms/call per category plus the host wall of the whole call, so the account closes:
//
//     host wall  =  (idle) + sum(gpu categories)
//
// and whatever is left over is the exposed staging / launch gap that M2 and M3 have to attack.
//
// The pairs are read ONCE, at exit, never mid-run: reading them needs cudaEventSynchronize and doing
// that per call would drain the pipeline and change what is being measured (the same lesson the
// attention accounting learned).  The events themselves add one timestamp write per section - a
// fraction of a percent of a 4k prefill - so a run with the switch on takes essentially the same path
// as one without.  Attribution only: this is a diagnostic, not a benchmark path.
//
// Nested scopes are safe: the pool index is taken in begin() and never reused, so each scope owns its
// pair, and the category only decides where the elapsed time is summed at exit.

#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv4 {

enum Dsv4MoeTimeCat {
    kMhRouteChain = 0,  // hc_pre + hc_fin + rmsnorm + xpose2048 + gate scores + route (one scope)
    kMhGateW,           // grouped GEMV, gate  (role 0)
    kMhUpW,             // grouped GEMV, up    (role 1)
    kMhSwiglu,          // swiglu clamp
    kMhDownW,           // grouped GEMV, down  (role 2)
    kMhScatter,         // scatter into the comb6 cells
    kMhStageDma,        // the staged blobs' H2D, queued on the KERNEL stream (the serial one)
    kMhPfWait,          // in-stream wait for a staging / prefetch event (backpressure, stream time)
    kMhTabH2D,          // the packed per-chunk table upload
    kMhPeer,            // device-1 launch set (gate..down on the peer stream) + its wait for y
    kMhShared,          // the shared expert (side stream or inline)
    kMhCombine,         // the deterministic combine
    kMhHcPost,          // hc_post + the d_h device-to-device copy
    kMhAdmitFill,       // an admission fill that had to go on the kernel stream (no pinned ring)
    kMhMem,             // bout memset, route row copies, peer copy-back
    kMhN
};

struct MoeTimeAcc {
    static constexpr int kMax = 1 << 18;  // event pairs; 2 * 262144 timestamps
    bool on = false;
    int used = 0;
    int depth = 0;
    long nested = 0;
    long skipped = 0;
    long bad_elapsed = 0;
    cudaEvent_t* ev = nullptr;
    unsigned char* cat = nullptr;
    double ms[kMhN] = {};
    long n[kMhN] = {};
    // host-clock side: the call's own wall and the one blocking point on the host path.
    double host_wall_ms = 0.0;
    double host_route_block_ms = 0.0;
    double host_stage_ms = 0.0;   // [M3] pinned copy of the staged blobs (host side)
    long host_calls = 0;
    MoeTimeAcc() {
        const char* e = std::getenv("STRATA_DSV4_MOETIME");
        on = e != nullptr && e[0] != '\0' && e[0] != '0';
        if (!on) return;
        ev = (cudaEvent_t*) std::calloc((size_t) 2 * kMax, sizeof(cudaEvent_t));
        cat = (unsigned char*) std::malloc(kMax);
        if (ev == nullptr || cat == nullptr) {
            std::fprintf(stderr, "dsv4_moetime: out of host memory, timing disabled\n");
            on = false;
        }
    }
    bool flushed = false;
    ~MoeTimeAcc() { flush(); }
    void flush() {
        if (!on || flushed) return;
        flushed = true;
        const char* first_err = "none";
        for (int i = 0; i < used; ++i) {
            if (ev[2 * i] == nullptr || ev[2 * i + 1] == nullptr) {
                ++bad_elapsed;
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
            ms[cat[i]] += (double) el;
            ++n[cat[i]];
        }
        static const char* nm[kMhN] = {
            "route-chain", "gemm-gate",  "gemm-up",   "swiglu",     "gemm-down",
            "scatter",     "stage-dma",  "pf-wait",   "table-h2d",  "peer",
            "shared",      "combine",    "hc-post",   "admit-fill", "mem"};
        const double calls = host_calls > 0 ? (double) host_calls : 1.0;
        double sum = 0.0;
        for (int k = 0; k < kMhN; ++k) sum += ms[k];
        std::fprintf(stderr,
                     "dsv4_moetime: gpu busy %.3f ms/call over %ld MoE calls (sum of %ld sections)\n",
                     sum / calls, host_calls, used);
        for (int k = 0; k < kMhN; ++k) {
            if (n[k] == 0) continue;
            std::fprintf(stderr, "  %-12s n=%7ld total %9.1f ms  %7.3f ms/call  %7.3f ms/section\n", nm[k],
                         n[k], ms[k], ms[k] / calls, ms[k] / (double) n[k]);
        }
        std::fprintf(stderr,
                     "dsv4_moetime: host wall %.3f ms/call (route drain %.3f) | gpu busy %.3f | "
                     "exposed %.3f ms/call\n",
                     host_wall_ms / calls, host_route_block_ms / calls, sum / calls,
                     (host_wall_ms - sum) / calls);
        if (host_stage_ms > 0.0)
            std::fprintf(stderr, "dsv4_moetime: host stage copy %.3f ms/call\n",
                         host_stage_ms / calls);
        std::fprintf(stderr,
                     "dsv4_moetime: pairs %d / %d, skipped %ld, nested begins %ld, unreadable %ld (%s)\n",
                     used, kMax, skipped, nested, bad_elapsed, first_err);
    }
    int begin(int c, cudaStream_t st) {
        if (!on) return -1;
        if (depth++ > 0) ++nested;
        if (used >= kMax) { ++skipped; return -1; }
        const int i = used++;
        cat[i] = (unsigned char) c;
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

inline MoeTimeAcc g_moe_time;

// One pair for the enclosing scope (RAII: safe with early returns and with nesting).
struct MoeTimeScope {
    int idx;
    cudaStream_t st;
    MoeTimeScope(int c, cudaStream_t s) : idx(-1), st(s) {
        if (g_moe_time.on) idx = g_moe_time.begin(c, s);
    }
    ~MoeTimeScope() { g_moe_time.end(idx, st); }
    MoeTimeScope(const MoeTimeScope&) = delete;
    MoeTimeScope& operator=(const MoeTimeScope&) = delete;
};

// Host-clock side: one call's wall, and the route D2H drain (the only place the host blocks on work
// it has just queued, so it is where a staging stall lands).  Callers accumulate these directly:
// host_wall_ms / host_calls is the ms per MoE call the account is closed against.
inline double moe_time_now_ms(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Reads the pooled pairs while the CUDA context is certainly alive (the driver calls this just before
// it prints its own summary); the destructor is only a fallback for paths that return early.
inline void moe_time_flush() { g_moe_time.flush(); }

}  // namespace strata::dsv4
