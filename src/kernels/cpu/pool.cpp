// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::kernels::cpu {

std::vector<std::vector<int>> physical_cores() {
    std::vector<std::vector<int>> cores;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (len > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                    (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        for (const char* p = buf.data(); p < buf.data() + len;) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
            if (e->Relationship == RelationProcessorCore) {
                const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
                std::vector<int> core;
                for (int bit = 0; bit < 64; ++bit)
                    if (g.Mask & (1ull << bit)) core.push_back((int) (g.Group * 64 + bit));
                if (!core.empty()) cores.push_back(core);
            }
            p += e->Size;
        }
    }
#else
    cpu_set_t set;   // each logical processor as a core of its own
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) cores.push_back({i});
    }
#endif
    if (cores.empty())
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cores.push_back({(int) i});
    return cores;
}

namespace {

#if defined(_WIN32)
// Each logical processor's time in interrupts and DPCs so far (100 ns units), from NtQuerySystemInformation's
// processor performance information (winternl.h names its DpcTime and InterruptTime Reserved1); the kernel adds to
// them at each clock tick, so over a second they resolve ~1.6%.
std::vector<uint64_t> interrupt_times() {
    struct Info { LARGE_INTEGER idle, kernel, user, dpc, interrupt; ULONG interrupts; };
    using Query = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
    static const Query query = reinterpret_cast<Query>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQuerySystemInformation")));
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    std::vector<Info> v(si.dwNumberOfProcessors);
    ULONG len = 0;
    std::vector<uint64_t> t;
    if (query == nullptr || query(8, v.data(), (ULONG) (v.size() * sizeof(Info)), &len) != 0) return t;
    for (const Info& i : v) t.push_back((uint64_t) (i.dpc.QuadPart + i.interrupt.QuadPart));
    return t;
}
#endif

constexpr double kBusy = 0.05;    // a processor over this share of a second in interrupts and DPCs takes them
constexpr double kQuiet = 0.02;   // one below it may take a thread

}  // namespace

CorePlacement::CorePlacement(bool spare_first) : cores_(physical_cores()) {
    for (size_t c = 0; c < cores_.size(); ++c)
        for (int p : cores_[c]) {
            if (p >= (int) core_of_.size()) core_of_.resize((size_t) p + 1, -1);
            core_of_[(size_t) p] = (int) c;
        }
    host_ = cores_.back().front();
    for (size_t c = spare_first && cores_.size() > 4 ? 1 : 0; c + 1 < cores_.size(); ++c)
        workers_.push_back(cores_[c].front());
}

// A quiet processor for a thread on `from`: the first of the quietest free core whose processors are all quiet, else
// the other processor of `from`'s own core, or -1.
int CorePlacement::quiet_processor(int from, const std::vector<double>& load) const {
    auto quiet = [&](int p) { return p < (int) load.size() && load[(size_t) p] < kQuiet && hot_[(size_t) p] == 0; };
    std::vector<char> taken(cores_.size(), 0);
    for (int p : workers_) taken[(size_t) core_of_[(size_t) p]] = 1;
    taken[(size_t) core_of_[(size_t) host_]] = 1;
    int best = -1;
    double best_load = 0;
    for (size_t c = 0; c < cores_.size(); ++c) {
        if (taken[c]) continue;
        bool ok = true;
        double sum = 0;
        for (int p : cores_[c]) {
            ok = ok && quiet(p);
            sum += p < (int) load.size() ? load[(size_t) p] : 0.0;
        }
        if (ok && (best < 0 || sum < best_load)) {
            best = cores_[c].front();
            best_load = sum;
        }
    }
    if (best >= 0) return best;
    for (int p : cores_[(size_t) core_of_[(size_t) from]])
        if (p != from && quiet(p)) return p;
    return -1;
}

std::string CorePlacement::tick(ExpertPool& pool) {
#if defined(_WIN32)
    const auto now = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(now - at_).count();
    if (!busy_.empty() && s < 1.0) return {};
    std::vector<uint64_t> cur = interrupt_times();
    // a first sample, or the engine idle since the last one (between requests): a new start, nothing to judge
    const bool fresh = busy_.empty() || s > 3.0 || cur.size() != busy_.size();
    std::vector<double> load(cur.size(), 0.0);
    if (!fresh)
        for (size_t p = 0; p < cur.size(); ++p) load[p] = (double) (cur[p] - busy_[p]) / 1e7 / s;
    busy_ = std::move(cur);
    at_ = now;
    hot_.resize(busy_.size(), 0);
    for (size_t p = 0; p < hot_.size(); ++p) hot_[p] = !fresh && load[p] > kBusy ? hot_[p] + 1 : 0;
    std::string moved;
    for (int t = -1; t < (int) workers_.size(); ++t) {   // the host, then the workers
        int& at = t < 0 ? host_ : workers_[(size_t) t];
        if (at >= (int) hot_.size() || hot_[(size_t) at] < 2) continue;
        const int to = quiet_processor(at, load);
        if (to < 0) continue;
        if (t < 0) pin_current_thread(to);
        else pool.pin_worker(t, to);
        char line[160];
        std::snprintf(line, sizeof line,
                      "%slogical processor %d spent %.0f%% of the last second in interrupts and DPCs: "
                      "the %s moved to %d", moved.empty() ? "" : "; ", at, 100.0 * load[(size_t) at],
                      t < 0 ? "host thread" : "pool worker", to);
        moved += line;
        hot_[(size_t) at] = 0;
        at = to;
        ++moves;
    }
    return moved;
#else
    (void) pool;
    return {};
#endif
}

namespace {

// `state_`: the epoch in the high half, then the closed bit, then the parked workers
constexpr uint64_t kClosed = 1ull << 31;
constexpr uint64_t kParked = kClosed - 1;
constexpr uint64_t kEpoch = 1ull << 32;

void pin_this_thread(int core) {
    if (core < 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

}  // namespace

long long pin_current_thread(int core) {
    if (core < 0) return -1;
#if defined(_WIN32)
    // `SetThreadAffinityMask` RETURNS the previous mask, or 0 on failure - so 0 doubles as the error, which is
    // why the caller must not treat it as a restorable value.
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
    return prev == 0 ? -1 : (long long) prev;
#else
    cpu_set_t prev;
    CPU_ZERO(&prev);
    if (pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) != 0) return -1;
    unsigned long mask = 0;
    for (int i = 0; i < CPU_SETSIZE && i < 64; ++i)
        if (CPU_ISSET(i, &prev)) mask |= 1ul << i;
    pin_this_thread(core);
    return (long long) mask;
#endif
}

void restore_thread_affinity(long long previous) {
    if (previous <= 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) previous);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < 64; ++i)
        if ((previous >> i) & 1) CPU_SET(i, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works, std::vector<int> cores) : host_works_(host_works) {
    if (cores.empty()) cores = CorePlacement(false).workers();
    n_ = n_workers > 0 ? n_workers : (int) cores.size();
    if (n_ < 1) n_ = 1;
    state_.store(kClosed | (uint64_t) n_, std::memory_order_relaxed);   // epoch 0, closed, everyone parked
    scratch_.resize((size_t) n_);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) cores.size() ? cores[(size_t) i] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core);
            worker(i);
        });
    }
}

void ExpertPool::pin_worker(int i, int core) {
    if (i < 0 || i >= (int) threads_.size() || core < 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask((HANDLE) threads_[(size_t) i].native_handle(), (DWORD_PTR) 1 << (core & 63));
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(threads_[(size_t) i].native_handle(), sizeof set, &set);
#endif
}

ExpertPool::~ExpertPool() {
    stop_.store(true, std::memory_order_seq_cst);
    // a new value, so a sleeping worker's wait returns and it sees the stop flag
    state_.fetch_add(kEpoch, std::memory_order_seq_cst);
    state_.notify_all();
    for (auto& t : threads_) t.join();
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;   // the last epoch this worker joined
    for (;;) {
        // Park: wait for an open phase this worker has not joined.  `_mm_pause` rather than a bare spin because
        // it yields the pipeline to the sibling hyperthread.  The loop writes nothing shared (see the note on
        // the atomics in pool.hpp) until it joins, or goes to sleep after `kIdleSpin` without work.
        uint64_t s = state_.load(std::memory_order_acquire);
        auto idle_from = std::chrono::steady_clock::now();
        for (uint32_t spins = 1;; ++spins) {
            if (stop_.load(std::memory_order_relaxed)) return;
            if (!(s & kClosed) && (uint32_t) (s >> 32) != seen) {
                // join; fails when the host has closed this phase or other workers joined since `s` was read
                if (state_.compare_exchange_weak(s, s - 1, std::memory_order_acq_rel, std::memory_order_acquire)) break;
                continue;
            }
            _mm_pause();
            if (spins % 1024 == 0 && std::chrono::steady_clock::now() - idle_from > kIdleSpin) {
                // seq_cst, against open_phase's store then load: the host sees this sleeper or this worker's
                // wait sees the new epoch
                sleepers_.fetch_add(1, std::memory_order_seq_cst);
                sleeps_.fetch_add(1, std::memory_order_relaxed);
                state_.wait(s, std::memory_order_seq_cst);
                sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                idle_from = std::chrono::steady_clock::now();
            }
            s = state_.load(std::memory_order_acquire);
        }
        seen = (uint32_t) (s >> 32);

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.
        drain(id, scratch_[(size_t) id]);
        state_.fetch_add(1, std::memory_order_release);   // parked again
    }
}

void ExpertPool::open_phase() {
    // The previous phase is closed, so no worker is inside a drain and none can join until this store.
    const uint64_t s = state_.load(std::memory_order_relaxed);
    state_.store(((s >> 32) + 1) << 32 | (uint64_t) n_, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) state_.notify_all();
}

void ExpertPool::close_phase() {
    // Every job is done; wait for the workers that joined to park again, and shut the phase in the same step,
    // so a worker that notices it late cannot join it.
    const uint64_t open = (state_.load(std::memory_order_relaxed) & ~kParked) | (uint64_t) n_;
    for (uint64_t s = open;
         !state_.compare_exchange_weak(s, open | kClosed, std::memory_order_acq_rel, std::memory_order_relaxed);
         s = open)
        _mm_pause();
}

void ExpertPool::drain(int id, ExpertScratch& scratch) {
    (void) id;
    for (;;) {
        const uint32_t i = head_.fetch_add(1, std::memory_order_relaxed);
        if (i >= (uint32_t) njobs_) break;
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 5 && nfmt_->gu_type == 42) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (nfmt_->d_type == 42) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks, void (*first)(void*), void* ctx) {
    mode_ = mode;
    njobs_ = n_tasks;
    head_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    open_phase();
    if (first != nullptr) first(ctx);
    if (host_works_) drain(-1, host_scratch_);
    while (done_.load(std::memory_order_acquire) != (uint32_t) n_tasks) _mm_pause();
    close_phase();
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = 3 * threads;
    mrows_ = (int64_t) n * FF;
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n, void (*first)(void*),
                                        void* ctx) {
    if (n <= 0) {
        if (first != nullptr) first(ctx);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        const int threads = n_ + (host_works_ ? 1 : 0);
        mtasks_ = 3 * threads;
        mrows_ = (int64_t) nb * FF;
        const auto a = std::chrono::steady_clock::now();
        run_phase(5, mtasks_, b0 == 0 ? first : nullptr, ctx);
        const auto b = std::chrono::steady_clock::now();
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (f.d_type == 42) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
        const auto c = std::chrono::steady_clock::now();
        mrows_ = (int64_t) nb * H;
        run_phase(6, mtasks_);
        const auto d = std::chrono::steady_clock::now();
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // The previous phase is closed, so no worker can be in the drain loop while the batch is written.
    //
    // THE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is slow at
    // the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    head_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    open_phase();   // seq_cst: jobs_/njobs_ are visible to a worker that joins

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) _mm_pause();`.  The host is pinned to its own core - the
    // one `CorePlacement` deliberately keeps the workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const uint32_t i = head_.fetch_add(1, std::memory_order_relaxed);
            if (i >= (uint32_t) n) break;
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    while (done_.load(std::memory_order_acquire) != (uint32_t) n) _mm_pause();
    // And close the phase, so the next `run` starts from a known state.  See the header for why `done` alone
    // is not enough.
    const auto t_c = std::chrono::steady_clock::now();
    close_phase();
    const auto t_d = std::chrono::steady_clock::now();

    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_close_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu
