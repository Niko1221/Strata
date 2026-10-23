// include/strata/core/tail_prof.hpp - STRATA_DBG_TAIL=1: host timings of the batch window's tail (commit, the MTP drafts,
// the host's part between two windows), accumulated per window and printed with the batch summary. Debug only: off,
// every point is one predicted branch.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

struct TailProf {
    enum Slot {
        kLoop,          // batch_step's return -> the next batch_step (the main loop: try_next_line)
        kPlan,          // batch_step -> run_slot_rows (the rows, apply_pending, adapt_tick)
        kRunPrep,       // run_slot_rows: pre_launch + stage_batch
        kRunLaunch,     // run_slot_rows: the window graph's cudaGraphLaunch call
        kRunTotal,      // run_slot_rows, all of it
        kPreCommit,     // run_slot_rows' return -> the commit call (keep)
        kCommitLaunch,  // the commit graph's cudaGraphLaunch call
        kCommitSync,    // its cudaStreamSynchronize
        kCommitTotal,   // commit_slot_prefixes, all of it
        kEmitPrint,     // the BT lines of every slot
        kDraftCopy,     // the residual rows' cudaMemcpyAsync calls
        kDraftPrep,     // draft(): -> its first graph launch (captures, the host records, stage_source_R)
        kDraftLaunch,   // draft(): the round and step graphs' cudaGraphLaunch calls
        kDraftWait,     // draft(): the spin on the outputs and the sync
        kDraftTotal,    // draft(), all of it
        kFlush,         // std::fflush(stdout)
        kAdapt,         // the adaptive tier's join
        kReport,        // report_batch_window -> batch_step's return
        kRunPreLaunch,  // run_slot_rows: the pre_launch hook (res_sync)
        kSbRefresh,     // stage_batch: the checks + refresh_ar
        kSbCapture,     // stage_batch: the graph lookup / eviction / capture
        kSbRecords,     // stage_batch: the host records before the PLE gather
        kSbPle,         // stage_batch: the PLE rows' gather
        kN
    };
    double ms[kN] = {};
    long long windows = 0;
    long long ple_checked = 0, ple_bad = 0;   // STRATA_DBG_PLECHECK: gathered PLE rows re-read and compared
    long long pair_rounds = 0, draft_checked = 0, draft_bad = 0;   // STRATA_FUSE_DRAFTHEAD / STRATA_DBG_DRAFTCHECK
    long long early_rounds = 0, early_bad = 0;   // STRATA_FUSE_DRAFTEARLY: pairs launched before the commit, mispredicted
    // STRATA_FUSE_GDNDEFER: flushes (a slot's pending rows run by the commit's kernel); STRATA_DBG_B2CHECK: a pending
    // slot's state after the next window against a shadow copy committed the old way, compared bit for bit
    long long b2_flushes = 0, b2_checked = 0, b2_bad = 0;
    static bool ple_check() {
        static const bool v = [] { const char* e = std::getenv("STRATA_DBG_PLECHECK"); return e != nullptr && e[0] == '1'; }();
        return v;
    }
    std::chrono::steady_clock::time_point last_end{};
    static bool on() {
        static const bool v = [] { const char* e = std::getenv("STRATA_DBG_TAIL"); return e != nullptr && e[0] == '1'; }();
        return v;
    }
    static std::chrono::steady_clock::time_point now() { return std::chrono::steady_clock::now(); }
    void add(Slot s, std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        ms[s] += std::chrono::duration<double, std::milli>(b - a).count();
    }
};

inline TailProf& tail_prof() {
    static TailProf p;
    return p;
}

// STRATA_DBG_ACOMMIT=1 (with STRATA_FUSE_ACOMMIT): the sessions whose batch commit is still running (commit_slot_prefixes
// sets them, Verifier::wait_commit clears them); the session writers and readers outside the window check theirs
struct AcommitDbg {
    static constexpr int kN = 16;
    std::atomic<const void*> s[kN]{};
    std::atomic<long long> hits{0};
    // the keep words the batch commit read (an echo of its device copy) against the ones the host wrote for it;
    // STRATA_DBG_ACOMMIT=2 also falsifies the host's record every 100th commit (the check's own test)
    long long echo_checked = 0, echo_bad = 0;
    static int level() {
        static const int v = [] { const char* e = std::getenv("STRATA_DBG_ACOMMIT"); return e != nullptr ? std::atoi(e) : 0; }();
        return v;
    }
    static bool on() { return level() >= 1; }
    void set(int i, const void* p) { if (i >= 0 && i < kN) s[i].store(p); }
    void clear() { for (auto& x : s) x.store(nullptr); }
    void check(const void* p, const char* site) {
        if (!on() || p == nullptr) return;
        for (auto& x : s)
            if (x.load() == p) {
                if (hits.fetch_add(1) < 8) std::fprintf(stderr, "strata: DBG_ACOMMIT: %s with its batch commit pending\n", site);
                return;
            }
    }
};
inline AcommitDbg& acommit_dbg() {
    static AcommitDbg d;
    return d;
}

}  // namespace strata::core
