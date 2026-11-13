// include/strata/core/tp_host_runtime.hpp - the object that makes a rank's partial sums whole again.
//
// WHAT IT OWNS.  A tensor-parallel rank is one process holding two cards' worth of state and running two
// windows in lockstep, so `reduce()` has to be called from each rank's own thread with that rank's own device
// and stream.  This class owns the tested collective (`kernels/allreduce.hpp`, gate `tp_allreduce_test`) and
// nothing else: the two arenas, the two sessions and the two threads are the engine's business.
//
// WHAT `reduce` DOES, AND WHY IT IS TWO CALLS.
//
//   1. `launch(rank, y, n, stream)` starts the in-kernel handshake for this rank: it stages this rank's
//      half, signals its arrival, waits for the peer's, adds, and leaves the sum in `y`.  The handshake is on
//      the DEVICE, so the host does not block here - which is the whole reason the token lives in device
//      memory (`allreduce.hpp` documents the three deliberate departures from llama.cpp's original).
//   2. `flush(rank, stream)` is NOT optional.  Under WDDM a launch sits in the command buffer and does not
//      begin until a later submission arrives on that stream - measured while building this: one launch alone
//      did not start in 400 ms, and with two ranks launching only the first started.  Without the flush the
//      peer waits for an arrival that never comes and the window gives up.
//
// FAILURE IS LOUD.  A rank that cannot meet its peer must not carry on with half a sum, because half a sum
// looks plausible: `reduce` therefore returns false once the collective has given up (the kernel records that
// in a mapped word, `timed_out_rank()`), and it also refuses the calls it cannot make sense of - a runtime with
// no pair, a count past the staging the pair was built for, or a rank index that is not one of the two.
#pragma once

#include "strata/core/tp_runtime.hpp"
#include "strata/kernels/allreduce.hpp"
#include "strata/sycl_queue.hpp"

#include <cstdint>
#include <atomic>
#include <string>
#include <thread>

namespace strata::core {

class TpHostRuntime final : public TpRuntime {
public:
    /// `pair` must already be `init`ed for the two ordinals this rank set uses, with staging room for the
    /// widest tensor a layer reduces (the residual width: `g.n_embd`).
    TpHostRuntime(strata::kernels::ArPair* pair, int rank) : pair_(pair), rank_(rank) {}

    /// Both ranks wait here before each launch, so neither can be a call ahead of its peer when the kernels meet
    /// on the device.  One shared barrier for the pair (the two ranks are two threads of one process).
    void meeting() {
        static std::atomic<uint32_t> arrive{0};
        static std::atomic<uint32_t> epoch{0};
        const uint32_t my = epoch.load(std::memory_order_acquire);
        if (arrive.fetch_add(1, std::memory_order_acq_rel) + 1 == (uint32_t) strata::kernels::kArRanks) {
            arrive.store(0, std::memory_order_release);
            epoch.fetch_add(1, std::memory_order_release);
            return;
        }
        // the peer is on its way; the second rank of the pair is the one that spins
        for (int64_t spins = 0; epoch.load(std::memory_order_acquire) == my; ++spins) {
            if (spins > 2000000000ll) return;      // a broken run must not hang the whole engine here
            std::this_thread::yield();
        }
    }

    bool reduce(float* y, int64_t n, void* stream, std::string& err) override {
        // TPDIAG (temporary, STRATA_TP_NO_REDUCE=1): the collective is skipped entirely.  The answer is wrong (the
        // residual stays a partial), but a two-card run that then stops faulting says the fault is in the
        // collective's path rather than in the layers.
        if (const char* nr = std::getenv("STRATA_TP_NO_REDUCE"); nr != nullptr && nr[0] == (char) 49) return true;
        if (pair_ == nullptr || !pair_->ready()) {
            err = "tensor parallel: no collective is initialised";
            return false;
        }
        if (rank_ < 0 || rank_ >= strata::kernels::kArRanks) {
            err = "tensor parallel: rank " + std::to_string(rank_) + " is not one of the " +
                  std::to_string(strata::kernels::kArRanks);
            return false;
        }
        if (n <= 0 || n > pair_->max_floats()) {
            err = "tensor parallel: a reduce of " + std::to_string(n) + " floats against staging for " +
                  std::to_string(pair_->max_floats());
            return false;
        }
        // MEASURED AND REVERTED.  A host-side rendezvous before every launch (so neither rank could be a call
        // ahead of its peer) did NOT fix the divergence: eight identical requests then gave
        //   ['1','The','The','S','<|im_start|>','Thinking','The','The']  - 5 distinct of 8, and no 'We'.
        // That refutes "one rank a call ahead" as the mechanism, so the barrier is not kept - a change that
        // measures no better than the baseline is reverted rather than carried on a theory.  `meeting()` is left
        // above, unused by this path, for the next experiment.
        // What the evidence still says: after the reduce the two ranks' values DIFFER - `livepre` at layer 0
        // splits (rank0 14.5292413 / rank1 12.3435555) in every run whose answer was wrong, and is bit-identical
        // in the runs that returned the single card's token.
        // TPDIAG: the parity protocol requires both ranks to make the SAME calls in the SAME order
        // (`kernels/allreduce.hpp`), so a count per rank is the cheapest way to see that premise hold.  Every
        // call is printed: the interesting event is the FIRST call whose count differs between the two ranks.
        {
            static int calls[2] = {0, 0};
            const int r = rank_ < 2 ? rank_ : 0;
            ++calls[r];
            // TPDIAG (STRATA_TP_DIAG_CALLS=1): the staging buffer is chosen by `(calls_[r] & 1)` on the HOST
            // (`allreduce.cu:337`), so the two ranks silently use different buffers from the first call where
            // their counts differ by an odd number.  Counting the calls per rank finds the call where they part.
            static const bool diag_calls = [] {
                const char* v = std::getenv("STRATA_TP_DIAG_CALLS"); return v != nullptr && v[0] != (char) 48;
            }();
            if (diag_calls && calls[r] <= 160)
                std::fprintf(stderr, "TPDIAG rcall rank=%d n=%d n=%lld\n", r, calls[r], (long long) n);
            if (diag_calls && calls[r] % 48 == 1) {
                for (int q = 0; q < strata::kernels::kArRanks; ++q)
                    std::fprintf(stderr,
                                 "TPDIAG prog rank=%d peer=%d entered=%u staged=%u signalled=%u saw_peer=%u\n", r, q,
                                 pair_->peek_progress(q, 0), pair_->peek_progress(q, 1), pair_->peek_progress(q, 2),
                                 pair_->peek_progress(q, 3));
            }
        }
        const int late = pair_->timed_out_rank();
        // TPDIAG: STRATA_TP_NO_CLEAR_TIMEOUT=1 skips the clear, which is how the two arms of the test that
        // asks whether the clear itself is what broke the meeting are run.
        if (const char* nc = std::getenv("STRATA_TP_NO_CLEAR_TIMEOUT"); nc == nullptr || nc[0] != (char) 49)
            pair_->clear_timeout();
        pair_->launch(rank_, y, (int) n, stream);
        // see the header comment: without this the launch never begins under WDDM
        pair_->flush(rank_, stream);

        // DIAGNOSTIC (STRATA_TP_STAGE_DBG=1): read back what each rank actually published for this call.  The
        // region is mapped-pinned, so the host can read it once the stream is drained; this is the one link in
        // the reduce that had never been observed directly - the partials and the sum were both measured, and
        // the sum turned out to be neither rank's partial.
        if (const char* sd = std::getenv("STRATA_TP_STAGE_DBG"); sd != nullptr && sd[0] == (char) 49) {
            static unsigned dbg_call = 0;
            ++dbg_call;
            // PAST THE CAPTURE.  The first calls are the graph CAPTURE, which records without executing, so the
            // staging still reads zero there (measured: `call=1/2 ... sum=0 sumsq=0` on both ranks).  Sample a
            // window's worth of calls later, where the kernels really ran.
            if (dbg_call < 100 || dbg_call > 106) { /* skip */ } else {
            strata::q_of(stream)->wait_and_throw();
            for (int r = 0; r < strata::kernels::kArRanks; ++r) {
                const uint16_t* s = (const uint16_t*) pair_->host_staging(r, (int) (dbg_call & 1ull));
                if (s == nullptr) continue;
                double acc = 0.0, acc2 = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    const uint16_t b = s[i];
                    const uint32_t f = ((uint32_t) b) << 16;
                    float v;
                    std::memcpy(&v, &f, 4);
                    acc += (double) v; acc2 += (double) v * (double) v;
                }
                std::fprintf(stderr, "TPDIAG stage call=%u rank=%d n=%lld sum=%.9g sumsq=%.9g\n", dbg_call, r,
                             (long long) n, acc, acc2);
            }
            }
        }
        // The handshake is on the device, so a give-up is discovered rather than blocked on.  `late` was read
        // BEFORE the launch and the flag cleared there, so what it holds is the value the previous call left.
        if (late >= 0) {
            // TPDIAG: the four mapped progress words say WHICH stage of the meeting failed, without any sync:
            // entered / staged / signalled / saw-peer.  `entered == 0` means the launch never began (a WDDM
            // submission problem); `signalled != 0 && saw_peer == 0` means this rank arrived and the PEER never
            // did; both non-zero on both sides with a give-up means the visibility of the write, not the order.
            for (int r = 0; r < strata::kernels::kArRanks; ++r) {
                std::fprintf(stderr,
                             "TPDIAG stall rank=%d entered=%u staged=%u signalled=%u saw_peer=%u "
                             "arrival_self[p0]=%u arrival_peer[p0]=%u arrival_self[p1]=%u arrival_peer[p1]=%u\n",
                             r, pair_->peek_progress(r, 0), pair_->peek_progress(r, 1), pair_->peek_progress(r, 2),
                             pair_->peek_progress(r, 3), pair_->peek_arrival(r, false, 0, 0),
                             pair_->peek_arrival(r, true, 0, 0), pair_->peek_arrival(r, false, 1, 0),
                             pair_->peek_arrival(r, true, 1, 0));
            }
            static int calls2[2] = {0, 0};
            const int r2 = rank_ < 2 ? rank_ : 0;
            std::fprintf(stderr, "TPDIAG TIMEOUT seen-by-rank=%d at its call=%d; ranks timed out: %d\n", r2,
                         ++calls2[r2], late);
            err = "tensor parallel: rank " + std::to_string(late) +
                  " gave up waiting for its peer (the two ranks must call the same reduces in the same order)";
            return false;
        }
        return true;
    }

    int rank() const { return rank_; }
    strata::kernels::ArPair* pair() const { return pair_; }

private:
    strata::kernels::ArPair* pair_ = nullptr;
    int rank_ = 0;
};

}  // namespace strata::core
