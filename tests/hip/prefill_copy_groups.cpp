#include "copy_groups.hpp"
#include "stager.hpp"
#include <chrono>
#include <cstdio>
#include <string>

namespace {
using strata::prefill::detail::CopyGroups;
using strata::prefill::detail::Stager;
struct Entry { int job; };
void env(const char* name, int value) {
    const auto text = std::to_string(value);
#if defined(_WIN32)
    _putenv_s(name, text.c_str());
#else
    setenv(name, text.c_str(), 1);
#endif
}
bool check(cudaError_t status) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "copy groups: %s\n", cudaGetErrorString(status));
    return false;
}

// Independent maximal-group oracle: grow one entry at a time while every
// boundary predicate holds, rather than reproducing the planner's limit math.
bool plan_ok(const CopyGroups& groups, const std::vector<Entry>& seq,
             const std::vector<char>& pinned, int host_ring, int direct, int staged) {
    for (size_t begin = 0; begin < seq.size();) {
        const int job = seq[begin].job;
        const bool stage_pinned = job >= 0 && pinned[job % host_ring];
        const int batch = job < 0 ? direct : stage_pinned ? staged : 1;
        size_t end = begin + 1;
        while (end < seq.size() && end - begin < (size_t) batch) {
            if (end / groups.ring != begin / groups.ring) break;
            if (job < 0) { if (seq[end].job >= 0) break; }
            else if (seq[end].job != job + (int) (end - begin) ||
                     seq[end].job / host_ring != job / host_ring ||
                     !pinned[seq[end].job % host_ring]) break;
            ++end;
        }
        for (size_t i = begin; i < end; ++i) {
            const auto step = groups.at(i);
            if (step.end != end || step.first != (i == begin) || step.terminal != (i + 1 == end) ||
                step.last_slot != (int) ((end - 1) % groups.ring)) return false;
            const size_t release = end > (size_t) groups.ring ? end - groups.ring : 0;
            if (!step.fits(release, groups.ring) || (release && step.fits(release - 1, groups.ring))) return false;
        }
        begin = end;
    }
    return true;
}

// Hold only the terminal `used` event behind a CPU gate. Earlier slot events
// are complete. The copy stream must remain blocked, and terminal `copied`
// must be recorded (an unrecorded event would incorrectly query as complete).
bool terminal_gate() {
    constexpr int n = 8;
    cudaStream_t copy = nullptr, compute = nullptr;
    std::vector<cudaEvent_t> used(n), copied(n);
    if (!check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking)) ||
        !check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking))) return false;
    for (int i = 0; i < n; ++i) {
        if (!check(cudaEventCreateWithFlags(&used[i], cudaEventDisableTiming)) ||
            !check(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming))) return false;
        if (i + 1 < n && !check(cudaEventRecord(used[i], compute))) return false;
    }
    if (!check(cudaStreamSynchronize(compute))) return false;
    std::atomic<bool> release{false};
    auto hold = [](void* opaque) {
        auto& gate = *static_cast<std::atomic<bool>*>(opaque);
        while (!gate.load(std::memory_order_acquire)) std::this_thread::yield();
    };
    if (!check(cudaLaunchHostFunc(compute, hold, &release))) return false;
    const bool recorded = check(cudaEventRecord(used[n - 1], compute));
    const CopyGroups groups(std::vector<Entry>(n, Entry{-1}), std::vector<char>(16, 1), n, 16, n, 1);
    std::atomic<size_t> issued{0};
    bool ok = recorded && check(groups.wait_reuse(groups.at(0), true, used.data(), copy));
    for (int i = 0; i < n; ++i) {
        ok = check(groups.publish(groups.at(i), copied.data(), copy, issued)) && ok;
        ok = issued.load() == (i + 1 == n ? (size_t) n : 0) && ok;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    do {
        ok = cudaEventQuery(copied[n - 1]) == cudaErrorNotReady && ok;
        std::this_thread::yield();
    } while (ok && std::chrono::steady_clock::now() < deadline);
    release.store(true, std::memory_order_release);
    ok = check(cudaStreamSynchronize(copy)) && check(cudaStreamSynchronize(compute)) && ok;
    for (int i = 0; i < n; ++i)
        ok = check(cudaEventDestroy(used[i])) && check(cudaEventDestroy(copied[i])) && ok;
    ok = check(cudaStreamDestroy(copy)) && check(cudaStreamDestroy(compute)) && ok;
    if (!ok) std::fprintf(stderr, "copy groups: terminal event gate failed\n");
    return ok;
}

bool run(int gpu_ring, int host_ring, int direct, int staged, int mode, bool skip) {
    constexpr size_t bytes = 4096;
    const int count = 4 * std::max(gpu_ring, host_ring) + 1;
    env("STRATA_STAGER_RING", host_ring);
    env("STRATA_HIP_STAGE_BATCH", std::min(16, host_ring));
    Stager stage;
    if (!stage.init(bytes, 3)) return false;
    // Force the actual pageable fallback at selected host slots. The planner
    // must leave those sources in singleton GPU completion groups.
    for (int sl : {3, 11}) {
        if (stage.pinned[sl] && !check(cudaFreeHost(stage.buf[sl]))) return false;
        stage.pinned[sl] = 0;
        stage.pageable[sl].resize(bytes);
        stage.buf[sl] = stage.pageable[sl].data();
    }
    cudaStream_t copy = nullptr, compute = nullptr;
    uint8_t *source = nullptr, *ring = nullptr, *output = nullptr;
    if (!check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking)) ||
        !check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking)) ||
        !check(cudaHostAlloc((void**) &source, (size_t) count * bytes, cudaHostAllocDefault)) ||
        !check(cudaMalloc((void**) &ring, (size_t) gpu_ring * bytes)) ||
        !check(cudaMalloc((void**) &output, (size_t) count * bytes))) return false;
    std::vector<cudaEvent_t> copied(gpu_ring), used(gpu_ring);
    std::vector<char> live(gpu_ring, 0);
    for (int sl = 0; sl < gpu_ring; ++sl)
        if (!check(cudaEventCreateWithFlags(&copied[sl], cudaEventDisableTiming)) ||
            !check(cudaEventCreateWithFlags(&used[sl], cudaEventDisableTiming))) return false;
    for (int generation = 0; generation < 2; ++generation) {
        std::vector<Entry> seq;
        std::vector<Stager::Job> jobs;
        for (int i = 0; i < count; ++i) {
            for (size_t b = 0; b < bytes; ++b)
                source[(size_t) i * bytes + b] = (uint8_t) (i * 71 + b * 17 + generation * 23);
            // Direct/staged blocks cross host/GPU boundaries at differing offsets.
            const bool is_direct = mode == 0 || (mode == 2 && (i / 9) % 3 == 0);
            seq.push_back({is_direct ? -1 : (int) jobs.size()});
            if (!is_direct) jobs.push_back({source + (size_t) i * bytes, bytes});
        }
        stage.start(std::move(jobs), true);
        const CopyGroups groups(seq, stage.pinned, gpu_ring, host_ring, direct, staged);
        if (!plan_ok(groups, seq, stage.pinned, host_ring, direct, staged)) return false;
        std::atomic<size_t> issued{0}, consumed{0};
        std::atomic<bool> bad{false};
        uint64_t records = 0, waits = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        auto progress = [&] {
            if (std::chrono::steady_clock::now() > deadline) bad.store(true);
            std::this_thread::yield();
        };
        std::thread issuer([&] {
            cudaSetDevice(stage.device);
            size_t published = 0;
            for (size_t i = 0; i < seq.size() && !bad.load(); ++i) {
                const auto step = groups.at(i);
                while (!step.fits(consumed.load(std::memory_order_acquire), gpu_ring) && !bad.load()) progress();
                if (bad.load()) break;
                const int sl = (int) (i % gpu_ring);
                if (!check(groups.wait_reuse(step, live[step.last_slot], used.data(), copy))) bad.store(true);
                waits += step.first && live[step.last_slot];
                const uint8_t* src = seq[i].job < 0 ? source + i * bytes : stage.wait(seq[i].job);
                if (!check(cudaMemcpyAsync(ring + (size_t) sl * bytes, src, bytes, cudaMemcpyHostToDevice, copy))) bad.store(true);
                if (seq[i].job >= 0) stage.issued_one(seq[i].job, copy);
                if (!check(groups.publish(step, copied.data(), copy, issued))) bad.store(true);
                records += step.terminal;
                if (step.terminal) published = step.end;
                if (issued.load(std::memory_order_acquire) != published) bad.store(true);
                live[sl] = 1;
            }
        });
        for (size_t i = 0; i < seq.size() && !bad.load(); ++i) {
            // Production can release unrouted entries without a copied wait.
            if (!skip || i % 7 != 0) {
                while (issued.load(std::memory_order_acquire) <= i && !bad.load()) progress();
                if (bad.load()) break;
                if (!check(groups.wait_copy(i, copied.data(), compute)) ||
                    !check(cudaMemcpyAsync(output + i * bytes, ring + (i % gpu_ring) * bytes,
                                           bytes, cudaMemcpyDeviceToDevice, compute))) bad.store(true);
            }
            if (!check(cudaEventRecord(used[i % gpu_ring], compute))) bad.store(true);
            consumed.store(i + 1, std::memory_order_release);
        }
        issuer.join();
        stage.finish();
        if (bad.load() || !check(cudaStreamSynchronize(copy)) || !check(cudaStreamSynchronize(compute))) return false;
        std::vector<uint8_t> got((size_t) count * bytes);
        if (!check(cudaMemcpy(got.data(), output, got.size(), cudaMemcpyDeviceToHost))) return false;
        uint64_t expected_records = 0, expected_waits = 0;
        for (size_t i = 0; i < seq.size(); ++i) {
            const auto step = groups.at(i);
            expected_records += step.terminal;
            expected_waits += step.first && (generation || step.end > (size_t) gpu_ring);
            if ((!skip || i % 7 != 0) && std::memcmp(got.data() + i * bytes, source + i * bytes, bytes)) {
                std::fprintf(stderr, "copy groups: stale payload i=%zu generation=%d\n", i, generation);
                return false;
            }
        }
        if (records != expected_records || waits != expected_waits || issued.load() != seq.size() ||
            stage.fence_errors.load()) return false;
    }
    for (int sl = 0; sl < gpu_ring; ++sl)
        if (!check(cudaEventDestroy(copied[sl])) || !check(cudaEventDestroy(used[sl]))) return false;
    return check(cudaFree(ring)) && check(cudaFree(output)) && check(cudaFreeHost(source)) &&
           check(cudaStreamDestroy(copy)) && check(cudaStreamDestroy(compute));
}
}  // namespace

int main() {
    if (!terminal_gate()) return 1;
    int cases = 0;
    for (int gpu : {8, 17, 32}) for (int host : {16, 17})
        for (int batch : {1, 2, 8}) for (int mode : {0, 1, 2}) for (bool skip : {false, true}) {
            const int staged = batch == 1 ? 1 : std::min({16, gpu, host});
            if (!run(gpu, host, batch, staged, mode, skip)) {
                std::fprintf(stderr, "copy groups failed: gpu=%d host=%d batches=%d/%d mode=%d skip=%d\n",
                             gpu, host, batch, staged, mode, (int) skip);
                return 1;
            }
            cases += 2;
        }
    // Job discontinuities are not produced by today's stream walk, but must
    // still terminate a staged group if that walk changes later.
    const std::vector<Entry> gaps{{0}, {1}, {3}, {4}, {-1}, {-1}, {7}};
    const std::vector<char> pinned(16, 1);
    const CopyGroups groups(gaps, pinned, 17, 16, 8, 16);
    if (!plan_ok(groups, gaps, pinned, 16, 8, 16)) return 1;
    std::printf("HIP copy groups: %d GPU generations passed; mixed/pageable boundaries, wraps, publication, reuse and skipped entries\n", cases);
    return 0;
}
