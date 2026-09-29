// include/strata/kernels/cpu/pool.hpp - P2.S3: the CPU expert pool.
//
// A layer runs TEN experts against ONE activation, and the expert kernel is DRAM-bound (Memory/LEDGER.md L9:
// 42.55 GB/s on 6 cores, tracking core count almost exactly).  So the pool's job is not to be clever - it is
// to keep every physical core reading expert bytes for the whole layer, and to be cheap enough that ten
// dispatches per layer cost less than one expert.
//
// WHY A FLAT BATCH AND NOT A RING.  P2.S3 describes a "lock-free SPMC ring", which is what you need when
// jobs arrive while others are still being computed.  Here they cannot: the host must SUM all ten outputs
// before the next layer starts, so a layer is a barrier by construction and the queue never holds more than
// one batch.  A ring would add a wrap-around to get wrong and buy nothing.  What is kept from the phase is
// the part that matters: `head`/`done` are single fetch_add counters, one claim per worker, no lock.
//
// THE PHASE PROTOCOL, because this is where a pool usually goes wrong.  One atomic word holds the phase's
// epoch, whether it is open, and how many workers are parked.  The host writes a phase's jobs while the
// previous phase is closed, then opens the next epoch; a worker joins with a compare-exchange on the whole
// word, so it can only join the phase it saw open.  The host closes a phase once `done == n` AND every
// worker that joined has parked again.  Waiting only for `done` is not enough: a worker can still be inside
// the drain loop after its last `done` increment, and the host resetting `head` underneath it would let it
// claim a job from the NEXT batch.  Closing is what makes a late worker harmless: one that notices a phase
// only after the host finished it (it was asleep, or descheduled) finds it closed and waits for the next.
//
// IDLE WORKERS SLEEP.  A parked worker spins for `kIdleSpin`, then blocks on the word (`std::atomic::wait`:
// WaitOnAddress, a futex) until the host opens a phase; the host wakes them only when one sleeps.  Generation
// opens phases every few hundred microseconds, and a round's longest gap (head, drafts, the first layer) is
// a few ms, so they sleep only between requests and while a prompt runs on the GPUs.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace strata::kernels::cpu {

/// One expert evaluation.  `act` is SHARED and read-only across the whole batch - that sharing is the point
/// of `s2_expert_vnni_q` and it is what saves 480 redundant activation conversions per token.
struct ExpertJob {
    const uint8_t* blob = nullptr;   ///< one 1,382,400-byte expert
    const ActQ* act = nullptr;       ///< the layer's quantized activation, shared
    float* out = nullptr;            ///< H floats, written by exactly one worker
    float weight = 1.0f;             ///< the router weight; applied by the HOST, not here
    int slot = -1;                   ///< the job's index, for diagnostics
};

/// Plan v0.3 P6: one expert for the `nt` tokens of a verify window that were routed to it.  Every token's output
/// is bitwise the single-token job's.
struct ExpertJobMulti {
    const uint8_t* blob = nullptr;
    int nt = 0;
    const ActQ* act[MAXT] = {};
    float* out[MAXT] = {};
    /// Plan v0.3 P6: a native pack's activations (the layer's `vec_dot_type`), one per token.
    const void* nact[MAXT] = {};
};

/// One logical processor per PHYSICAL core, so a worker is never scheduled onto an SMT sibling of another
/// worker.  On the 6-core/12-thread machine this project measures on, `hardware_concurrency()/2` workers on
/// logical processors 0..5 would put every worker on a sibling pair and halve the useful bandwidth - which is
/// exactly the kind of error that shows up as "the CPU path is slower than the model says" with no clue why.
std::vector<int> physical_cores();

/// Where the host loop and the pool's workers run, one logical processor each: the host on the last physical core,
/// the workers on the others.  With `spare_first` (a second GPU) the first core stays with the OS: Windows sends the
/// GPUs' interrupts there (here the 5070 Ti's to its first logical processor, the 3090's to its second), and the
/// second GPU's copies and launches raise thousands a second during generation, each ~16 us there in the ISR and a
/// DPC.  With four physical cores or fewer the first core takes a worker anyway.
struct CorePlan {
    int host = -1;
    std::vector<int> workers;
};
CorePlan core_plan(bool spare_first);

/// **THE RESERVATION IS A FICTION UNLESS THE HOST IS ACTUALLY PUT THERE.**
///
/// `core_plan()` keeps the workers off the host's core so that the host loop can spin on `cudaEventQuery`
/// without stealing a worker's cycles.  Nothing in the pool can enforce the other half of that, so this is it:
/// the host loop calls this on entry and restores on exit.
///
/// MEASURED, and this is why it exists: the pool runs at **36.32 GB/s on 5 workers with nothing else running**
/// - exactly 5/6 of L9's 44.14 on 6 - and at **26.9 GB/s inside the host loop**, where the unpinned spinning
/// host is free to land on a worker's core or its SMT sibling.  That 1.35x is not the kernel.
///
/// Returns the PREVIOUS affinity mask, or -1 if the platform refused; pass it to `restore_thread_affinity`.
long long pin_current_thread(int core);
void restore_thread_affinity(long long previous);

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)   // the alignas(64) members pad the class on purpose (one cache line each)
#endif
class ExpertPool {
public:
    /// `n_workers <= 0` means a worker on each of `core_plan(spare_first).workers`.  Workers are pinned to those
    /// cores and each owns one `ExpertScratch`, so nothing in the token path allocates.
    ///
    /// **`host_works` PUTS THE HOST THREAD INTO THE DRAIN (R2.2's FIRST HALF).**
    ///
    /// The pool keeps a core for the host loop so the doorbell spin cannot steal a worker's cycles - but
    /// during `run()` the host does not spin, it waits, so that core is idle for the whole drain. Measured on the
    /// 6-core machine this project targets: the engine's pool drains at **33.7 GB/s** (663.6 MB of expert
    /// blobs in 19.71 ms/token) where the same kernel on 5 workers should reach 5/6 x 44.14 = 36.8 and the
    /// machine measures 44.14 GB/s on all six. So the sixth core is being paid for and not used.
    ///
    /// With `host_works`, `run()` claims jobs itself instead of spinning on `done_`, and the pool is six
    /// threads on six cores. `false` is the A/B arm and exists so the change is measurable rather than
    /// asserted - the counter it moves is `pool phases ... drain`, which is host-side and needs no profiler.
    explicit ExpertPool(int n_workers = 0, bool pin = true, bool host_works = true, bool spare_first = false);
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const { return n_; }
    /// Whether the host thread also drains.  Reported at startup, because "the engine adapts to the machine it
    /// is on" is only true if the engine says which adaptation it took.
    bool host_works() const { return host_works_; }

    /// Publish `n` jobs, then block until every one is done AND the phase is closed.
    /// `jobs` must outlive the call (it does, and the workers never touch it afterwards).
    void run(ExpertJob* jobs, int n);

    /// Plan v0.3 P4: the same outputs as `run`, bitwise, with every expert split by rows across all threads
    /// (gate/up rows, then the intermediate's quantization, then down rows).  With fewer experts than threads -
    /// the case once the VRAM tier takes half of them - `run` leaves cores idle and each expert streams at one
    /// core's bandwidth; this streams every expert at all of them.  At most `kMaxSplit` experts.
    void run_split(ExpertJob* jobs, int n);
    static constexpr int kMaxSplit = 16;
    /// Plan v0.3 P6: `run_split` for multi-token jobs (at most `kMaxSplitMulti`); the rows of each expert are
    /// read once for all of its tokens.
    void run_split_multi(ExpertJobMulti* jobs, int n);
    /// Plan v0.3 P6: the same for a native pack's layer (ggml-cpu arithmetic, `nact` activations).  `first`, when
    /// given, runs once on the host: after the workers have started on the first phase, before it joins them (or
    /// alone when there are no jobs).
    void run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n, void (*first)(void*) = nullptr,
                                void* ctx = nullptr);
    static constexpr int kMaxSplitMulti = 96;
    /// run_split_multi's phases, accumulated ms: gate/up rows, the intermediate quantization, down rows.
    double ms_multi_gu = 0, ms_multi_q = 0, ms_multi_down = 0;
    int64_t multi_bytes = 0;

    /// How long a parked worker spins before it sleeps.
    static constexpr std::chrono::milliseconds kIdleSpin{50};
    /// How many times a worker has gone to sleep (each worker counts once per idle stretch).
    uint32_t sleeps() const { return sleeps_.load(std::memory_order_relaxed); }

    /// **WHERE `run()` SPENDS ITS TIME, in milliseconds accumulated over its lifetime**: the drain, and closing
    /// the phase (the joined workers parking again).  Without the split there is no way to tell a pool that is
    /// slow at the WORK from one that is slow at the SYNCHRONISATION, and those need opposite fixes.
    ///
    /// Only the host thread touches these, in `run()`, so they need no atomics.
    void phase_ms(double& drain, double& close) const {
        drain = ms_drain_;
        close = ms_close_;
    }

private:
    void worker(int id);
    void drain(int id, ExpertScratch& scratch);
    void run_phase(int mode, int n_tasks, void (*first)(void*) = nullptr, void* ctx = nullptr);
    void open_phase();
    void close_phase();

    int n_ = 0;
    bool host_works_ = true;
    ExpertJob* jobs_ = nullptr;
    int njobs_ = 0;
    /// The host's own scratch when `host_works_`.  A separate object rather than a share of `scratch_[i]`,
    /// because a worker may own any index and the two must not be able to collide.
    ExpertScratch host_scratch_;
    // `run()`'s phases, accumulated.  Host-thread only; see `phase_ms`.
    double ms_drain_ = 0.0;
    double ms_close_ = 0.0;
    // ---- EACH ATOMIC GETS ITS OWN CACHE LINE, AND THE PARK LOOP WRITES NOTHING SHARED.  (Review finding C3.)
    //
    // Adjacent atomics that workers and the host contend on invalidate each other on every access, and a
    // diagnostic counter the park loop once incremented on every iteration hammered the very line the host
    // writes to publish work.  `alignas(64)` keeps them apart; the park loop only reads `state_` (a worker
    // writes it once to join a phase and once to park again, and `sleepers_` only when it goes to sleep).
    alignas(64) std::atomic<uint32_t> head_{0};
    alignas(64) std::atomic<uint32_t> done_{0};
    /// epoch << 32 | kClosed (bit 31) | parked workers
    alignas(64) std::atomic<uint64_t> state_{0};
    alignas(64) std::atomic<uint32_t> sleepers_{0};
    std::atomic<uint32_t> sleeps_{0};
    alignas(64) std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
    std::vector<ExpertScratch> scratch_;   // one per worker: no allocation, no false sharing of the hot data
    // run_split state: mode 0 = whole experts, 1 = gate/up row parts, 2 = down row parts
    int mode_ = 0;
    int parts_a_ = 1, parts_b_ = 1;
    struct SplitBuf {
        alignas(64) float ff[FF];
        ActQ a2;
    };
    std::vector<SplitBuf> split_;
    // run_split_multi state: mode 3 = gate/up row parts, 4 = down row parts
    ExpertJobMulti* mjobs_ = nullptr;
    int64_t mrows_ = 0;     // rows of the current multi phase across all its experts (n * FF, then n * H)
    int mtasks_ = 1;        // equal row ranges the phase is cut into
    struct SplitBufMulti {
        alignas(64) float ff[MAXT][FF];
        ActQ a2[MAXT];
        alignas(64) uint8_t hq[MAXT][kNativeHBytes];   // plan v0.3 P6: native down activations
    };
    const NativeFmt* nfmt_ = nullptr;
    std::vector<SplitBufMulti> split_multi_;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace strata::kernels::cpu
