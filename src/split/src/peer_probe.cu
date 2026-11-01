#include "peer_probe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fnx {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("peer probe ") + what + ": " + cudaGetErrorString(e));
}

__device__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// each word names its run and its place, so a word that lands elsewhere, tears or is left from another run reads as
// missing
__device__ __forceinline__ unsigned long long probe_word(size_t i, unsigned long long tag) { return tag << 32 | i; }

__global__ void k_probe_write(unsigned long long* dst, size_t n, unsigned long long tag) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        *(volatile unsigned long long*) (dst + i) = probe_word(i, tag);
}
// on the receiving card, its own memory: missing counts the words not seen within timeout_ns of the kernel's start
__global__ void k_probe_watch(const unsigned long long* src, size_t n, unsigned long long tag, unsigned long long timeout_ns,
                              unsigned* missing) {
    const unsigned long long t0 = now_ns();
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        while (*(const volatile unsigned long long*) (src + i) != probe_word(i, tag))
            if (now_ns() - t0 > timeout_ns) {
                atomicAdd(missing, 1u);
                break;
            }
}

struct Side {
    int dev = -1;
    unsigned long long *buf = nullptr, *src = nullptr;   // what others write into; what this card copies out of
    unsigned* missing = nullptr;
    unsigned* missing_host = nullptr;   // pinned, so reading the count back waits on this stream alone
    cudaStream_t st = nullptr;
};
// the first failure ("the call on card n: the error") after trying every handle, or empty. A card that will not take
// cudaSetDevice keeps its handles: freeing them would free them on whatever card is current
std::string free_side(Side& x) {
    std::string first;
    const auto keep = [&](cudaError_t e, const char* call) {
        if (e != cudaSuccess && first.empty()) first = std::string(call) + " on card " + std::to_string(x.dev) + ": " + cudaGetErrorString(e);
    };
    keep(cudaSetDevice(x.dev), "cudaSetDevice");
    if (!first.empty()) return first;
    if (x.st) {
        keep(cudaStreamDestroy(x.st), "cudaStreamDestroy");
        x.st = nullptr;
    }
    const auto free_dev = [&](auto*& p) {
        if (p) keep(cudaFree(p), "cudaFree");
        p = nullptr;
    };
    free_dev(x.buf);
    free_dev(x.src);
    free_dev(x.missing);
    if (x.missing_host) {
        keep(cudaFreeHost(x.missing_host), "cudaFreeHost");
        x.missing_host = nullptr;
    }
    return first;
}
struct Sides {
    std::vector<Side> s;
    void release() {
        for (Side& x : s) {
            const std::string err = free_side(x);
            if (!err.empty()) throw std::runtime_error("peer probe: " + err);
        }
    }
    // only while give_up's error unwinds, every card idle: that error is the one to report, so these frees' own are not
    ~Sides() {
        for (Side& x : s) free_side(x);
    }
};

// a card's probe stream within 2 s: cudaSuccess idle, cudaErrorNotReady still running, else the fault it hit
cudaError_t idle_within_2s(const Side& x) {
    const cudaError_t set = cudaSetDevice(x.dev);
    if (set != cudaSuccess) return set;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (;;) {
        const cudaError_t e = cudaStreamQuery(x.st);
        if (e != cudaErrorNotReady || std::chrono::steady_clock::now() > end) return e;
    }
}
// every wait in the probe: work that never retires throws (and give_up ends the process) rather than block the start
void require_idle(const Side& x, const char* what) {
    const cudaError_t e = idle_within_2s(x);
    if (e == cudaErrorNotReady)
        throw std::runtime_error("card " + std::to_string(x.dev) + ": the probe's " + what + " had not finished 2 s later");
    ck(e, what);
}

// Every failure inside the probe ends here. A card whose stream is still running 2 s later, or has faulted, is past
// anything this process can clean up after: freeing memory there waits on the stuck work, and the lab's and fnx_run's
// own cleanups would hang the same way. So the message goes out and the process ends, since a hung start says nothing
// to systemd, lab.sh or the front end. With every card idle the error is thrown like any other
[[noreturn]] void give_up(const Sides& sides, const std::string& msg) {
    std::string bad;
    for (const Side& x : sides.s) {
        if (!x.st) continue;
        const cudaError_t e = idle_within_2s(x);
        if (e == cudaErrorNotReady) bad += "; card " + std::to_string(x.dev) + " had not finished the probe 2 s later";
        else if (e != cudaSuccess) bad += "; card " + std::to_string(x.dev) + " faulted: " + cudaGetErrorString(e);
    }
    if (bad.empty()) throw std::runtime_error(msg);
    std::fprintf(stderr, "fnx: %s%s. Nothing on that card can be freed safely, so the process exits now; it may need a reset\n",
                 msg.c_str(), bad.c_str());
    std::fflush(stdout);   // a lab check's results so far, when stdout is a file or pipe
    std::fflush(stderr);
    std::_Exit(1);
}

}  // namespace

std::vector<PeerRate> probe_peer_stores(const std::vector<int>& devs) {
    // 20 x 256 threads writing and watching, as tools/p2p_store.cu measured the hops; 3 runs a pair to warm up, the
    // median of the next 15 kept, then one copy-engine run, checked and not timed. The tag runs on across calls (one
    // process makes many engines in the lab), so a buffer the allocator hands back holds no word a later run could take
    // for its own
    constexpr size_t words = PEER_PROBE_BYTES / sizeof(unsigned long long);
    constexpr int warm = 3, timed = 15, blocks = 20, threads = 256;
    static_assert(timed > 0, "the median needs a timed run");
    constexpr unsigned long long timeout_ns = 1000000000ull;
    static unsigned long long tag = 0;
    Sides sides;
    try {
        for (int d : devs) {
            sides.s.push_back(Side{d});
            Side& y = sides.s.back();
            ck(cudaSetDevice(d), "cudaSetDevice");
            ck(cudaStreamCreateWithFlags(&y.st, cudaStreamNonBlocking), "cudaStreamCreate");
            ck(cudaMalloc(&y.buf, PEER_PROBE_BYTES), "cudaMalloc");
            ck(cudaMalloc(&y.src, PEER_PROBE_BYTES), "cudaMalloc");
            ck(cudaMalloc(&y.missing, sizeof(unsigned)), "cudaMalloc");
            ck(cudaMallocHost(&y.missing_host, sizeof(unsigned)), "cudaMallocHost");
            ck(cudaMemsetAsync(y.buf, 0, PEER_PROBE_BYTES, y.st), "cudaMemsetAsync");
            require_idle(y, "zeroing");
        }
        // the watcher on x for tag, timeout_ns from now: the words of x.buf not seen by then
        const auto watch = [&](Side& x, unsigned long long timeout) {
            ck(cudaSetDevice(x.dev), "cudaSetDevice");
            ck(cudaMemsetAsync(x.missing, 0, sizeof(unsigned), x.st), "cudaMemsetAsync");
            k_probe_watch<<<blocks, threads, 0, x.st>>>(x.buf, words, tag, timeout, x.missing);
            ck(cudaGetLastError(), "watch launch");
        };
        // the count of a watcher that has finished (require_idle on its card first)
        const auto missing_on = [&](Side& x) {
            ck(cudaSetDevice(x.dev), "cudaSetDevice");
            ck(cudaMemcpyAsync(x.missing_host, x.missing, sizeof(unsigned), cudaMemcpyDeviceToHost, x.st), "cudaMemcpyAsync");
            require_idle(x, "count");
            return *x.missing_host;
        };
        // the watcher's own compare must be able to fail: on each card, a tag nobody writes has every word missing
        for (Side& x : sides.s) {
            ++tag;
            watch(x, 1000000);
            require_idle(x, "self-check");
            const unsigned missing = missing_on(x);
            if (missing != words)
                throw std::runtime_error("card " + std::to_string(x.dev) + ": the probe's watcher passed words nobody wrote (" +
                                         std::to_string(words - missing) + " of " + std::to_string(words) + "): its check is broken");
        }
        // one run a -> b: a's earlier work drained, then the watcher on b, then a's stores (copy false) or a copy of a's
        // own src (copy true); the time from a's launch until b has seen every word
        const auto run = [&](Side& from, Side& to, int a, int b, bool copy) {
            ++tag;
            if (copy) {
                ck(cudaSetDevice(from.dev), "cudaSetDevice");
                k_probe_write<<<blocks, threads, 0, from.st>>>(from.src, words, tag);
                ck(cudaGetLastError(), "source launch");
            }
            require_idle(from, copy ? "source write" : "earlier writes");
            watch(to, timeout_ns);
            ck(cudaSetDevice(from.dev), "cudaSetDevice");
            const auto t0 = std::chrono::steady_clock::now();
            if (copy)
                ck(cudaMemcpyPeerAsync(to.buf, to.dev, from.src, from.dev, PEER_PROBE_BYTES, from.st), "cudaMemcpyPeerAsync");
            else {
                k_probe_write<<<blocks, threads, 0, from.st>>>(to.buf, words, tag);
                ck(cudaGetLastError(), "write launch");
            }
            require_idle(to, "watch");
            const auto t1 = std::chrono::steady_clock::now();
            const unsigned missing = missing_on(to);
            if (missing) {
                PeerFailure f{};
                f.from = a;
                f.to = b;
                f.ncard = (int) devs.size();
                f.copy = copy;
                f.missing = missing;
                f.words = words;
                f.timeout_ms = timeout_ns / 1000000;
                throw std::runtime_error(peer_probe_failure(f));
            }
            return std::chrono::duration<double, std::micro>(t1 - t0).count();
        };
        std::vector<PeerRate> out;
        for (int a = 0; a < (int) devs.size(); ++a)
            for (int b = 0; b < (int) devs.size(); ++b) {
                if (a == b) continue;
                Side &from = sides.s[a], &to = sides.s[b];
                std::vector<double> us;
                for (int rep = 0; rep < warm + timed; ++rep) {
                    const double t = run(from, to, a, b, false);
                    if (rep >= warm) us.push_back(t);
                }
                run(from, to, a, b, true);
                require_idle(from, "copy");
                std::sort(us.begin(), us.end());
                out.push_back({a, b, us[us.size() / 2]});
            }
        sides.release();
        return out;
    } catch (const std::exception& e) {
        give_up(sides, e.what());
    }
}

}  // namespace fnx
