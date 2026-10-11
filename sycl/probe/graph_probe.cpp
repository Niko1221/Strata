// sycl/probe/graph_probe.cpp - do recorded command graphs replay on this card under this driver, how large may one
// be, how many may be alive, and does recording them leak?
//
// Strata's decode runs each verify window as one captured graph of about 2,400 nodes and keeps a graph per window
// shape.  SYCL's counterpart is sycl_ext_oneapi_graph record-and-replay, which XeStrata and PR #423 use on Linux.  On
// Windows the v1 Level Zero adapter leaks about 1.5 KiB of host memory per node for every finalize-and-destroy
// (intel/llvm#23073, an A770; the v2 adapter does not), and llama.cpp ships its SYCL graphs compiled in but disabled.
// This probe records a graph of 64 small kernels and 4 memcpy nodes and replays it 1,000 times (and runs the same
// work eagerly, for the gain), then grows graphs until finalize or submit throws, keeps many alive at once, and
// measures this process's private memory over 200 finalize-and-destroy cycles.
//
//   graph_probe
//
// Recordings here hold no host wait and no device-to-host copy, as the engine's must not: a host wait inside a
// recording throws, and a D2H copy appended during capture spun forever in the v2 adapter.
// VERDICT: graphs usable or not, the node ceiling reached, the leak per node.
#include "probe_common.hpp"

#if __has_include(<sycl/ext/oneapi/experimental/graph.hpp>)
#include <sycl/ext/oneapi/experimental/graph.hpp>
#define PROBE_HAVE_GRAPH 1
#endif

#if !defined(PROBE_HAVE_GRAPH)
int main() {
    probe::Gpu gpu;
    if (!probe::start("graph_probe", gpu)) return 2;
    probe::verdict("fail", "this compiler has no <sycl/ext/oneapi/experimental/graph.hpp>: no SYCL graphs");
    return 2;
}
#else

namespace sx = sycl::ext::oneapi::experimental;

namespace {

constexpr double kWaitSeconds = 60.0;
constexpr int kKernels = 64, kCopies = 4;
constexpr size_t kItems = 4096;              // the largest kernel node's range
constexpr size_t kCopyBytes = 1u << 20;      // each memcpy node
constexpr uint32_t kSumPerRound = 64 * 65 / 2;   // every node adds k+1 to buf[0]: 1 + 2 + ... + 64
constexpr int kReplays = 1000, kLatencyReplays = 200, kEagerRounds = 200, kLeakCycles = 200, kAliveMax = 64;
constexpr int kAliveNodes = 256;

using Exec = sx::command_graph<sx::graph_state::executable>;

// The work one recording holds: `kernels` small kernels adding to `buf`, each over a different range (4,096 down to
// 2,080 items, so no two neighbouring nodes have the same shape), and `copies` device-to-device memcpy nodes.  The
// same function submits the work eagerly.  Every range reaches buf[0], which therefore counts the rounds.
void submit_work(sycl::queue& q, uint32_t* buf, uint8_t* src, uint8_t* dst, int kernels, int copies, int shape) {
    for (int k = 0; k < kernels; ++k) {
        const size_t n = kItems - (size_t) ((k + shape) % 64) * 32;
        const uint32_t add = (uint32_t) (k % 64 + 1);
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { buf[i[0]] += add; });
    }
    for (int c = 0; c < copies; ++c) q.memcpy(dst + (size_t) c * kCopyBytes, src + (size_t) c * kCopyBytes, kCopyBytes);
}

// record on the in-order queue, finalize; the queue leaves the recording state even when the body throws
Exec record(sycl::queue& q, uint32_t* buf, uint8_t* src, uint8_t* dst, int kernels, int copies, int shape,
            size_t* nodes) {
    sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
    g.begin_recording(q);
    try {
        submit_work(q, buf, src, dst, kernels, copies, shape);
    } catch (...) {
        try {
            g.end_recording(q);
        } catch (...) {
        }
        throw;
    }
    g.end_recording(q);
    if (nodes) *nodes = g.get_nodes().size();
    return g.finalize();
}

uint32_t read_word(sycl::queue& q, const uint32_t* p) {
    uint32_t v = 0;
    q.memcpy(&v, p, sizeof v).wait();
    return v;
}

const char* what(const std::exception& e) { return e.what(); }

}  // namespace

int main() {
    probe::Gpu gpu;
    if (!probe::start("graph_probe", gpu)) return 2;
    std::printf("the environment that changes graphs:\n");
    probe::print_env("SYCL_UR_USE_LEVEL_ZERO_V2");
    probe::print_env("UR_L0_DISABLE_EVENTS_CACHING");
    probe::print_env("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS");
    try {
        std::printf("graph aspects: ext_oneapi_graph %s, ext_oneapi_limited_graph %s\n",
                    gpu.dev.has(sycl::aspect::ext_oneapi_graph) ? "yes" : "no",
                    gpu.dev.has(sycl::aspect::ext_oneapi_limited_graph) ? "yes" : "no");
    } catch (const sycl::exception& e) {
        std::printf("graph aspects: unreadable (%s)\n", e.what());
    }
    const bool v2 = probe::env_set("SYCL_UR_USE_LEVEL_ZERO_V2") && std::atoi(probe::env("SYCL_UR_USE_LEVEL_ZERO_V2")) != 0;

    try {
        sycl::queue q = probe::make_queue(gpu);
        uint32_t* buf = sycl::malloc_device<uint32_t>(kItems, q);
        uint8_t* src = (uint8_t*) sycl::malloc_device(kCopies * kCopyBytes, q);
        uint8_t* dst = (uint8_t*) sycl::malloc_device(kCopies * kCopyBytes, q);
        if (!buf || !src || !dst) {
            probe::verdict("fail", "device USM for the probe's buffers was refused");
            return 2;
        }
        q.memset(buf, 0, kItems * sizeof(uint32_t));
        q.memset(src, 7, kCopies * kCopyBytes);
        q.memset(dst, 0, kCopies * kCopyBytes).wait();

        // 1. the window-sized graph: record, finalize, replay, and the same work eagerly
        std::printf("\n1. a graph of %d kernel nodes + %d memcpy nodes (%zu KiB each), replayed %d times\n", kKernels,
                    kCopies, kCopyBytes >> 10, kReplays);
        double replay_us = 0, eager_us = 0, latency_med = 0, latency_p99 = 0;
        size_t nodes = 0;
        bool part1 = false;
        std::string part1_why;
        try {
            auto t0 = probe::clk::now();
            Exec exec = record(q, buf, src, dst, kKernels, kCopies, 0, &nodes);
            std::printf("  recorded %zu nodes, finalized in %.1f ms\n", nodes, probe::ms_since(t0));
            q.memset(buf, 0, kItems * sizeof(uint32_t)).wait();
            t0 = probe::clk::now();
            {
                probe::Watchdog wd(kWaitSeconds, "the replays");
                for (int i = 0; i < kReplays; ++i) q.ext_oneapi_graph(exec);
                q.wait();
            }
            replay_us = probe::us_since(t0) / kReplays;
            std::vector<double> lat;
            {
                probe::Watchdog wd(kWaitSeconds, "the waited replays");
                for (int i = 0; i < kLatencyReplays; ++i) {
                    const auto t1 = probe::clk::now();
                    q.ext_oneapi_graph(exec);
                    q.wait();
                    lat.push_back(probe::us_since(t1));
                }
            }
            latency_med = probe::percentile(lat, 50);
            latency_p99 = probe::percentile(lat, 99);
            const uint32_t got = read_word(q, buf), got_last = read_word(q, buf + kItems - 1);
            const uint32_t want = (uint32_t) (kReplays + kLatencyReplays) * kSumPerRound;
            std::printf("  %.1f us per replay queued back to back; submit + wait median %.1f us, p99 %.1f us\n", replay_us,
                        latency_med, latency_p99);
            std::printf("  result %s (buf[0] %u, expected %u; buf[last] %u)\n", got == want ? "correct" : "WRONG", got,
                        want, got_last);
            // the same work eagerly: what the graph saves
            q.memset(buf, 0, kItems * sizeof(uint32_t)).wait();
            t0 = probe::clk::now();
            {
                probe::Watchdog wd(kWaitSeconds, "the eager rounds");
                for (int i = 0; i < kEagerRounds; ++i) submit_work(q, buf, src, dst, kKernels, kCopies, 0);
                q.wait();
            }
            eager_us = probe::us_since(t0) / kEagerRounds;
            const uint32_t got_e = read_word(q, buf);
            std::printf("  eager: %.1f us per round of the same %d submissions (%s)\n", eager_us, kKernels + kCopies,
                        got_e == (uint32_t) kEagerRounds * kSumPerRound ? "correct" : "WRONG");
            part1 = got == want && got_e == (uint32_t) kEagerRounds * kSumPerRound;
            if (!part1) part1_why = "a replayed or eager result was wrong";
        } catch (const std::exception& e) {
            part1_why = what(e);
            std::printf("  threw: %s\n", part1_why.c_str());
        }

        // 2. one graph of growing node count, until something throws
        std::printf("\n2. one graph of N kernel nodes (ranges differ node to node), finalized and replayed once\n");
        int reached = 0;
        std::string ladder_stop;
        for (int n : {128, 256, 512, 1024, 2048, 4096, 8192}) {
            try {
                size_t count = 0;
                auto t0 = probe::clk::now();
                Exec e = record(q, buf, src, dst, n, 0, n, &count);
                const double fin_ms = probe::ms_since(t0);
                t0 = probe::clk::now();
                {
                    probe::Watchdog wd(kWaitSeconds, "the large graph's replay");
                    q.ext_oneapi_graph(e);
                    q.wait();
                }
                const double rep_ms = probe::ms_since(t0);
                std::printf("  %5d nodes: finalize %.1f ms, replay %.2f ms (%.1f us per node)\n", n, fin_ms, rep_ms,
                            rep_ms * 1000.0 / n);
                reached = n;
            } catch (const std::exception& e) {
                ladder_stop = what(e);
                std::printf("  %5d nodes: threw: %s\n", n, ladder_stop.c_str());
                break;
            }
        }

        // 3. many distinct graphs alive at once (the engine keeps one per window shape)
        std::printf("\n3. distinct graphs of %d nodes kept alive at once, each replayed once\n", kAliveNodes);
        size_t alive = 0;
        std::string alive_stop;
        {
            std::vector<Exec> execs;
            const uint64_t mem0 = probe::private_bytes();
            for (int i = 1; i <= kAliveMax; ++i) {
                try {
                    execs.push_back(record(q, buf, src, dst, kAliveNodes, 0, i, nullptr));
                    probe::Watchdog wd(kWaitSeconds, "a replay among many graphs");
                    q.ext_oneapi_graph(execs.back());
                    q.wait();
                } catch (const std::exception& e) {
                    alive_stop = what(e);
                    std::printf("  graph %d: threw: %s\n", i, alive_stop.c_str());
                    break;
                }
            }
            alive = execs.size();
            std::printf("  %zu graphs alive (%zu nodes); private memory grew %.1f MiB\n", alive, alive * kAliveNodes,
                        ((double) probe::private_bytes() - (double) mem0) / 1048576.0);
        }

        // 4. finalize-and-destroy cycles: the event-cache leak of intel/llvm#23073
        std::printf("\n4. %d finalize-and-destroy cycles of a %d-node graph (10 warm-up cycles first)\n", kLeakCycles,
                    kKernels);
        double per_node = 0;
        bool leak_measured = false;
        {
            auto cycle = [&] {
                Exec e = record(q, buf, src, dst, kKernels, 0, 0, nullptr);
                probe::Watchdog wd(kWaitSeconds, "a leak cycle's replay");
                q.ext_oneapi_graph(e);
                q.wait();
            };
            try {
                for (int i = 0; i < 10; ++i) cycle();
                const uint64_t before = probe::private_bytes();
                std::printf("  private memory before: %.2f MiB\n", (double) before / 1048576.0);
                for (int i = 1; i <= kLeakCycles; ++i) {
                    cycle();
                    if (i % 50 == 0) {
                        const double grown = (double) probe::private_bytes() - (double) before;
                        std::printf("  after %3d cycles: %+.2f MiB (%+.1f KiB per cycle, %+.0f B per node)\n", i,
                                    grown / 1048576.0, grown / 1024.0 / i, grown / i / kKernels);
                    }
                }
                const double grown = (double) probe::private_bytes() - (double) before;
                per_node = grown / kLeakCycles / kKernels;
                leak_measured = before != 0;
                if (!leak_measured) std::printf("  this process's private memory is not readable here\n");
            } catch (const std::exception& e) {
                std::printf("  threw: %s\n", what(e));
            }
        }

        sycl::free(buf, q);
        sycl::free(src, q);
        sycl::free(dst, q);

        std::printf("\n");
        if (!part1) {
            probe::verdict("fail", "graphs unusable: %s; run the engine's eager (graph-free) decode path", part1_why.c_str());
            return 1;
        }
        // 1.5 KiB per node is the measured leak (#23073); a few hundred bytes of allocator noise is not one
        const bool leak = leak_measured && per_node > 256.0;
        const bool ceiling_ok = reached >= 4096;   // a window graph has about 2,400 nodes
        const bool alive_ok = alive >= 16;
        std::string hint;
        if (leak && !v2) hint = " -> set SYCL_UR_USE_LEVEL_ZERO_V2=1 (or UR_L0_DISABLE_EVENTS_CACHING=1)";
        else if (leak) hint = " -> leaks even on the v2 adapter: re-record windows sparingly";
        char ceiling[128];
        if (ladder_stop.empty()) std::snprintf(ceiling, sizeof ceiling, "%d nodes in one graph (the ladder's end)", reached);
        else std::snprintf(ceiling, sizeof ceiling, "%d nodes in one graph, the next step threw", reached);
        probe::verdict(ceiling_ok && alive_ok && !leak ? "ok" : "partial",
                       "graphs replay (%zu nodes: %.1f us per replay, %.1f us eager); %s; %zu graphs alive%s; "
                       "%+.0f B per node per finalize-destroy%s",
                       nodes, replay_us, eager_us, ceiling, alive, alive_stop.empty() ? "" : " (then it threw)", per_node,
                       hint.c_str());
        return 0;
    } catch (const sycl::exception& e) {
        probe::verdict("fail", "SYCL threw: %s", e.what());
        return 2;
    }
}
#endif
