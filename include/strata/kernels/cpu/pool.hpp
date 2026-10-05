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
// THE COMPLETION PROTOCOL, because this is where a pool usually goes wrong.  `run()` waits for `done == n`
// AND for every worker to PARK.  Waiting only for `done` is not enough: a worker can still be inside the
// drain loop after its last `done` increment, and the host resetting `head` underneath it would let that
// worker claim a job from the NEXT batch before the next batch has been published.
//
// **AND `parked` IS NOT ENOUGH EITHER (issue #29).**  A worker that went to sleep (after `kSpinBeforeSleep`) is
// still counted as parked when it wakes, so for a moment after it has seen a new epoch the host believes it is
// idle.  On a card with most experts in VRAM the workers sleep in the middle of a request, tiny batches finish
// before a sleeper is awake, and that moment comes round constantly: a late worker could claim from the NEXT
// batch while the host was still writing it, run a job twice, or add to `done` after the host had reset it -
// and `done != n` then never ended, with the GPU waiting on the pool forever.  So every claim carries its
// batch: `head` is one 64-bit word `epoch | njobs | index`, a claim is a CAS that only succeeds for the epoch
// the worker woke for, and a late worker's claim simply fails.  A claim that succeeds belongs to the current
// batch, whose description the host cannot change until that job's `done` has landed.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
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

/// How worker threads are allocated across physical/logical CPU cores (#272).  `All` is the layout the pool has
/// always used and the default; the hybrid-aware ones are opt-in (--pool-affinity auto|p-cores).
enum class PoolAffinity {
    Auto,      ///< Hybrid: prioritize physical P-cores, then SMT, then E-cores (defaults to P-core count)
    PCores,    ///< Restrict workers strictly to Performance cores and their SMT siblings
    All,       ///< The default: one worker per physical core in the OS's order, without hybrid distinction
};

struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0;                ///< Physical performance cores
    int p_threads = 0;              ///< Total logical threads on performance cores
    int e_cores = 0;                ///< Efficient cores
    std::vector<int> worker_cores;  ///< Ordered CPU IDs; on Windows, group * 64 + processor within the group
    int host_core = -1;             ///< Logical core reserved for host thread
};

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

/// One logical processor per PHYSICAL core, so a worker is never scheduled onto an SMT sibling of another
/// worker.  On the 6-core/12-thread machine this project measures on, `hardware_concurrency()/2` workers on
/// logical processors 0..5 would put every worker on a sibling pair and halve the useful bandwidth - which is
/// exactly the kind of error that shows up as "the CPU path is slower than the model says" with no clue why.
///
/// `skip_first` drops the first core, which P2.S3 reserves for the host loop.
std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

/// **THE RESERVATION IS A FICTION UNLESS THE HOST IS ACTUALLY PUT THERE.**
///
/// `physical_cores(true)` keeps the workers off the first physical core so that the host loop can spin on
/// `cudaEventQuery` without stealing a worker's cycles.  Nothing in the pool can enforce the other half of
/// that, so this is it: the host loop calls this on entry and restores on exit.
///
/// MEASURED, and this is why it exists: the pool runs at **36.32 GB/s on 5 workers with nothing else running**
/// - exactly 5/6 of L9's 44.14 on 6 - and at **26.9 GB/s inside the host loop**, where the unpinned spinning
/// host is free to land on a worker's core or its SMT sibling.  That 1.35x is not the kernel.
///
/// The previous host placement. `valid` is false when querying or setting placement failed.
/// Windows uses a reversible CPU Set selection, leaving hard affinity (including Windows 11's implicit
/// all-group eligibility) untouched. An empty selection restores inheritance from the process defaults.
/// Linux stores the complete, dynamically sized native CPU mask, including CPU IDs above 63.
struct ThreadAffinity {
#if defined(_WIN32)
    std::vector<unsigned long> cpu_sets;
#else
    std::vector<unsigned long> mask;
#endif
    bool valid = false;
};

/// Selects one encoded processor for the caller; Windows CPU Sets respect existing hard-affinity limits.
/// Restore on the same calling thread. Pool-owned workers use hard group affinity instead.
ThreadAffinity pin_current_thread(int core);
void restore_thread_affinity(const ThreadAffinity& previous);

class ExpertPool {
public:
    /// `n_workers <= 0` means "every physical core except the first".  Workers are pinned to physical cores
    /// (minus core 0 by default) and each owns one `ExpertScratch`, so nothing in the token path allocates.
    ///
    /// **`host_works` PUTS THE HOST THREAD INTO THE DRAIN (R2.2's FIRST HALF).**
    ///
    /// The pool reserves core 0 for the host loop so the doorbell spin cannot steal a worker's cycles - but
    /// during `run()` the host does not spin, it waits, so core 0 is idle for the whole drain. Measured on the
    /// 6-core machine this project targets: the engine's pool drains at **33.7 GB/s** (663.6 MB of expert
    /// blobs in 19.71 ms/token) where the same kernel on 5 workers should reach 5/6 x 44.14 = 36.8 and the
    /// machine measures 44.14 GB/s on all six. So the sixth core is being paid for and not used.
    ///
    /// With `host_works`, `run()` claims jobs itself instead of spinning on `done_`, and the pool is six
    /// threads on six cores. `false` is the A/B arm and exists so the change is measurable rather than
    /// asserted - the counter it moves is `pool phases ... drain`, which is host-side and needs no profiler.
    explicit ExpertPool(int n_workers = 0, bool pin = true, bool host_works = true,
                        PoolAffinity affinity = PoolAffinity::All);
    /// The watchdog's view of the pool (issue #31): the batch, the counters, every thread's state.
    void diag(std::FILE* f) const;
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const { return n_; }
    /// Whether the host thread also drains.  Reported at startup, because "the engine adapts to the machine it
    /// is on" is only true if the engine says which adaptation it took.
    bool host_works() const { return host_works_; }

    bool is_hybrid() const { return topo_.is_hybrid; }
    int p_cores() const { return topo_.p_cores; }
    int p_threads() const { return topo_.p_threads; }
    int e_cores() const { return topo_.e_cores; }
    PoolAffinity affinity() const { return affinity_; }

    /// Publish `n` jobs, then block until every one has been claimed AND every worker has parked.
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
    /// Plan v0.3 P6: the same for a native pack's layer (ggml-cpu arithmetic, `nact` activations).
    void run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n);
    static constexpr int kMaxSplitMulti = 96;

    // ---- PR #500: THE INTERMEDIATE QUANTIZATION AS A PHASE OF ITS OWN (drain mode 7).
    //
    // Between the gate/up phase and the down phase of a multi-token layer, every token's intermediate `ff[t]`
    // must be quantized into `a2[t]` (legacy Q8_1 / `act_quant_any`) or `hq[t]` (a native pack's `vec_dot_type`).
    // That used to be a nested host loop with every worker spinning on the park, because it cannot be folded
    // into the gate/up phase: an expert's rows are split across tasks, so `ff[e][t]` is incomplete until ALL
    // of phase 5's tasks have finished.  The barrier is required; running the work on ONE thread while the
    // others sit still is not.
    //
    // So the host and the workers that already claim whole experts - the same `head_`/`done_` protocol, the same
    // workers, the same pinning - claim quantization tasks instead.  One task is one (expert, token) pair of
    // 640 floats: `ff[t]` is read-only for the phase and the task writes only its own `a2[t]`/`hq[t]`, so no
    // two tasks touch the same bytes and no lock or barrier beyond the phase's own is needed.
    //
    // WHAT IT IS WORTH, AND WHERE IT IS NOT.  A phase costs two park barriers; a task is one 640-element
    // quantization (0.169 us alone, bench below).  Measured on the 12-core Ryzen 9 7900 this was written on, 11
    // workers + the host, as the ratio of the two arms' phase times (`ms_multi_q`, best of 40, median of three
    // runs - `pool_quant_test --bench` prints the whole table and the spread):
    //
    //     tasks      1     2     4     6     8    12    16    24    32    48    64    96
    //     faster  0.27x 0.5x  0.9x  0.9x  1.15x 1.2x  1.3x  1.45x 1.7x  1.8x  2.0x  2.05x
    //
    // End to end it is the same story: on the IQ3_S pack with --spec 4 (26 CPU experts a layer, 13xx tasks a
    // round) the engine's own report went from `pool multi quantize 0.457` to `0.280 ms/round` over eight runs -
    // 1.6x on this step, 0.18% of a ~100 ms/round decode, with byte-identical generated tokens in both arms.
    // Small because the step is only ~0.7% of the pool's work, and the pool is the dominant cost; real,
    // repeatable, and it scales with the task count.  It LOSES below about eight tasks, where the barrier costs
    // more than the work it distributes; `quant_min_tasks()` is that crossing, measured rather than guessed.  A
    // smaller pool crosses over earlier (6-8 tasks at 6 workers, 2-3 at 2), so one flat threshold is the
    // conservative side of every pool size measured.  `STRATA_POOL_PARALLEL_QUANT=0` runs the sequential loop at
    // every size: the A/B arm, and it changes nothing else.
    bool parallel_quant() const { return parallel_quant_; }
    /// The task count below which the host runs the quantization alone.  The native (ggml) path uses the same
    /// number: its task is one pass over the same 640 floats.  `STRATA_POOL_QUANT_MIN_TASKS` overrides it (a test
    /// knob for the measurement above, like `STRATA_POOL_SPIN_US`).
    int quant_min_tasks() const { return quant_min_tasks_; }
    /// One (expert, token) pair is one task, so a batch has at most this many.
    static constexpr int kMaxQuantTasks = kMaxSplitMulti * MAXT;
    /// run_split_multi's phases, accumulated ms: gate/up rows, the intermediate quantization, down rows.
    double ms_multi_gu = 0, ms_multi_q = 0, ms_multi_down = 0;
    int64_t multi_bytes = 0;
    /// PR #500: how many intermediate-quantization phases ran across the pool, and how many tasks they carried.
    /// Host-thread only, like `ms_multi_q`; the test reads them to check WHICH arm ran instead of assuming it.
    int64_t quant_phases = 0, quant_tasks = 0;

    /// Total `_mm_pause` iterations spent waiting, over all workers, is no longer counted - see the note on the
    /// atomics below.  It was a LOCKED read-modify-write in the park loop, so measuring the contention added to
    /// it.
    long long pauses() const { return 0; }

    /// **WHERE `run()` SPENDS ITS TIME, in milliseconds accumulated over its lifetime.**  Three phases per
    /// layer - wait for every worker to be parked, wait for the drain, wait for them to re-park - and until now
    /// all three were reported as one number.  Without the split there is no way to tell a pool that is slow at
    /// the WORK from one that is slow at the SYNCHRONISATION, and those need opposite fixes: the first is a
    /// kernel problem and the second is a barrier problem.
    ///
    /// Only the host thread touches these, in `run()`, so they need no atomics.
    void phase_ms(double& wait_park, double& drain, double& repark) const {
        wait_park = ms_wait_park_;
        drain = ms_drain_;
        repark = ms_repark_;
    }

    /// **A PARKED WORKER SPINS FOR THIS LONG, THEN SLEEPS.**  The park is a `_mm_pause` spin because a layer's
    /// batches are microseconds apart and a wake-up from the OS costs more than that.  But a spin that never ends
    /// keeps every worker's core at 100% while the engine waits for a request - issue #4, "CPU 50% even when
    /// doing nothing" (7 of the 5700X's 16 threads).  Between requests the workers block on `sleep_cv_`.  NOT only
    /// between requests: with most experts in VRAM (a 24 GB card) many layers have no CPU work, so the workers
    /// also sleep mid-request - which the claim protocol above must survive (issue #29).
    /// `STRATA_POOL_SPIN_US` overrides it (a test knob: a short spin makes the workers sleep constantly).
    static constexpr std::chrono::milliseconds kSpinBeforeSleep{20};
    /// A pool wait that sees no completion for this long is a bug; the engine stops with a message instead of
    /// spinning forever, and the server starts it again (issue #29).
    static constexpr std::chrono::seconds kStall{60};

private:
    void worker(int id);
    void drain(int id, ExpertScratch& scratch, uint32_t epoch);
    void run_phase(int mode, int n_tasks);
    /// Claim the next job of batch `epoch`, or -1 (that batch is exhausted, or it is not the current one).
    int claim(uint32_t epoch);
    /// PR #500: fill the quantization phase's task table from `mjobs_[0..n)`, and return the task count.  A
    /// running count per expert instead of a list of pairs: the task index alone says which pair it is (see
    /// `quant_expert_of`), so the token path allocates nothing and the host writes n+1 ints.
    int prepare_quant_tasks(int n);
    /// PR #500: quantize the intermediates of `mjobs_[0..n)` - mode 7 across the pool and the host, or the
    /// sequential host loop below `quant_min_tasks` / with `STRATA_POOL_PARALLEL_QUANT=0`.
    void run_quant(int n);
    /// The expert of quantization task `i` - the last one whose running count is not past it.
    int quant_expert_of(int i) const {
        return (int) (std::upper_bound(quant_start_, quant_start_ + quant_experts_ + 1, i) - quant_start_) - 1;
    }
    /// Publish the batch whose description the caller has just written: reset `done`, then `head`, then the epoch.
    uint32_t begin_batch(int n);
    /// The host's waits, bounded by `kStall`.
    void wait_parked(const char* what);
    void wait_done(int n);
    /// Bump `epoch_`, and wake the workers that went to sleep.  Every publish goes through here.
    void publish();

    int n_ = 0;
    bool host_works_ = true;
    ExpertJob* jobs_ = nullptr;
    int njobs_ = 0;
    /// The host's own scratch when `host_works_`.  A separate object rather than a share of `scratch_[i]`,
    /// because a worker may own any index and the two must not be able to collide.
    ExpertScratch host_scratch_;
    // `run()`'s three phases, accumulated.  Host-thread only; see `phase_ms`.
    double ms_wait_park_ = 0.0;
    double ms_drain_ = 0.0;
    double ms_repark_ = 0.0;
    // ---- EACH ATOMIC GETS ITS OWN CACHE LINE, AND THE SPIN COUNTER IS GONE.  (Review finding C3.)
    //
    // These were six adjacent atomics, which put `head_`, `done_`, `parked_` and `epoch_` on ONE cache line -
    // the four that workers and the host actually contend on, invalidating each other on every access.
    //
    // Worse, the parked spin did `pauses_.fetch_add(1)` on EVERY iteration: a LOCKED read-modify-write, five
    // workers against one line, at roughly one iteration per `_mm_pause`.  So the line the host must WRITE to
    // publish work (`epoch_`) and READ to confirm the workers are parked (`parked_`) was being hammered by the
    // very threads waiting for it.  That is contention the pool imposes on itself.
    //
    // The counter was diagnostic only - `pauses()` was read in one place, to print a number nothing branched on
    // - so it is deleted rather than amortised.  `alignas(64)` then stops the remaining four sharing.
    // issue #31 diagnostics: each worker's state (kParked, kSleeping, kBetween, or the job it runs) and the host's
    // (kIdle, kWaitParked, kWaitDone, or its job), printed by the serve watchdog through `diag`
    static constexpr int32_t kParked = -1, kSleeping = -2, kBetween = -3, kIdle = -10, kWaitParked = -11,
                             kWaitDone = -12;
    std::unique_ptr<std::atomic<int32_t>[]> wstate_;
    std::atomic<int32_t> hstate_{kIdle};
    std::atomic<int64_t> hstate_ms_{0};
    alignas(64) std::atomic<uint64_t> head_{0};   // epoch << 32 | njobs << 16 | next index (issue #29)
    alignas(64) std::atomic<uint32_t> done_{0};
    alignas(64) std::atomic<uint32_t> parked_{0};
    alignas(64) std::atomic<uint32_t> epoch_{0};
    alignas(64) std::atomic<bool> stop_{false};
    // The sleep after `kSpinBeforeSleep`.  `sleepers_` is how `publish` knows whether anyone needs waking, so the
    // token path pays one uncontended load per publish and never takes the mutex while the workers spin.
    alignas(64) std::atomic<uint32_t> sleepers_{0};
    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;
    std::chrono::microseconds spin_before_sleep_{kSpinBeforeSleep};
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
    // run_split_multi state: mode 3 = gate/up row parts, 4 = down row parts, 7 = intermediate quantization
    ExpertJobMulti* mjobs_ = nullptr;
    int64_t mrows_ = 0;     // rows of the current multi phase across all its experts (n * FF, then n * H)
    int mtasks_ = 1;        // equal row ranges the phase is cut into
    struct SplitBufMulti {
        alignas(64) float ff[MAXT][FF];
        ActQ a2[MAXT];
        alignas(64) uint8_t hq[MAXT][kNativeHBytes];   // plan v0.3 P6: native down activations
    };
    // mode 7's task table: `quant_start_[e]` is how many tasks experts 0..e-1 own, so task `i` belongs to
    // expert `e` with `quant_start_[e] <= i < quant_start_[e+1]` and is its token `i - quant_start_[e]`.
    int quant_start_[kMaxSplitMulti + 1] = {};
    int quant_experts_ = 0;   // experts in the current table (its entries are 0..quant_experts_)
    bool parallel_quant_ = true;      // STRATA_POOL_PARALLEL_QUANT=0: the sequential host loop
    int quant_min_tasks_ = 8;         // STRATA_POOL_QUANT_MIN_TASKS: the measured crossing above
    const NativeFmt* nfmt_ = nullptr;
    std::vector<SplitBufMulti> split_multi_;
    PoolAffinity affinity_ = PoolAffinity::All;
    CpuTopology topo_;
};

}  // namespace strata::kernels::cpu
