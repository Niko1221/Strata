// sycl/probe/doorbell_probe.cpp - does a running kernel on this card see the host's writes, and the host the kernel's?
//
// Strata's decode is a handshake through host memory: a kernel rings a sequence number the host polls (GPU -> host),
// and a one-work-item kernel spins on a flag the host raises (host -> GPU).  The CUDA engine does both with volatile
// accesses; on an Intel GPU a volatile load does not bypass the caches, and which read does see the host's store has
// differed per driver: on Linux a system-scope atomic load worked on an Arc Pro B70 (xe) and did not on a B60 (NEO
// 26.31, where an uncached read hint did), and on the Windows driver the atomic load saw nothing (0 of 2,000) while a
// volatile load followed by a system-scope acquire fence saw everything (sycl_doorbell.hpp has the history).  This
// probe measures the GPU -> host ring and five host -> GPU readings on the card and driver at hand, directly and
// inside a replayed command_graph (the engine runs its windows as graphs), before a model depends on any of them.
//
//   doorbell_probe [trips]        default 500 round trips per reading; 100 graph replays
//
// Per reading: round trips observed, median / p99 microseconds of launch -> ring seen and flag -> kernel done, and the
// most polls a successful wait took against the engine's spin bound (strata::spin_max).  A reading that misses three
// trips in a row is abandoned (each miss costs a whole spin bound).  VERDICT ok when the ring is seen every time and
// at least one reading saw every host write, naming the readings; partial or fail otherwise.
#include "probe_common.hpp"
#include "strata/sycl_doorbell.hpp"   // the engine's own sys_load / sys_store and spin_max: variant 5 IS sys_load

#if __has_include(<sycl/ext/oneapi/experimental/graph.hpp>)
#include <sycl/ext/oneapi/experimental/graph.hpp>
#define PROBE_HAVE_GRAPH 1
#endif

namespace {

constexpr int kGraphTrips = 100;
constexpr double kHostTimeout = 3.0;    // the host stops waiting for the ring after this; the kernel ends on its bound
constexpr double kWedgeSeconds = 30.0;  // a kernel still running this long past its bound: the queue is wedged
constexpr int kAbandonAfter = 3;        // consecutive misses before a reading is abandoned

// The five ways a kernel can read the host's flag.
enum Variant : int {
    kVolatile = 1,        // a volatile load, what nvcc made of CUDA's volatile
    kVolatileFence = 2,   // a volatile load + a system-scope acquire fence per poll (the Windows reading)
    kAtomic = 3,          // a system-scope relaxed atomic_ref load (the port's first reading)
    kUncached = 4,        // an uncached L1+L3 read hint behind an acquire fence (the Linux default since 0.1.40)
    kEngine = 5,          // whatever this build's sycl_doorbell.hpp chose
};

const char* variant_name(int v) {
    switch (v) {
        case kVolatile: return "1 (volatile load)";
        case kVolatileFence: return "2 (volatile load + acquire system fence)";
        case kAtomic: return "3 (atomic_ref system load)";
        case kUncached: return "4 (uncached L1+L3 read hint + acquire fence)";
        case kEngine: return "5 (this build's sys_load)";
    }
    return "?";
}

using uncached_read = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::read_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached, sycl::ext::oneapi::experimental::cache_level::L1,
        sycl::ext::oneapi::experimental::cache_level::L3>>));

template <int V>
inline uint32_t read_flag(const volatile uint32_t* p) {
    if constexpr (V == kVolatile) {
        return *p;
    } else if constexpr (V == kVolatileFence) {
        const uint32_t v = *p;
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        return v;
    } else if constexpr (V == kAtomic) {
        return sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>(
                   *const_cast<uint32_t*>(p)).load();
    } else if constexpr (V == kUncached) {
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, uncached_read> u(const_cast<uint32_t*>(p));
        return u[0];
    } else {
        return strata::sys_load(p);
    }
}

// The words the host and the kernel share, in host USM: [0] the flag the host raises (host -> GPU), [1] the ring the
// kernel writes (GPU -> host), [2] the polls the kernel made, [3] 1 when it saw the flag, [4] the trip a graph replay
// expects (a recorded kernel cannot capture it).
struct Words { uint32_t flag, ring, polls, seen, trip; };

inline uint32_t host_load(const volatile uint32_t* p) { return *p; }
inline void host_store(volatile uint32_t* p, uint32_t v) {
    *p = v;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

template <int V>
void submit(sycl::queue& q, Words* w, uint32_t trip, uint32_t bound, bool from_word) {
    q.single_task([=] {
        const uint32_t t = from_word ? strata::sys_load(&w->trip) : trip;
        strata::sys_store(&w->ring, t);
        uint32_t n = 0, seen = 0;
        while (n < bound) {
            ++n;
            if (read_flag<V>(&w->flag) == t) { seen = 1; break; }
        }
        strata::sys_store(&w->polls, n);
        strata::sys_store(&w->seen, seen);
    });
}

struct Result {
    int trips = 0, ring_seen = 0, flag_seen = 0, abandoned_at = -1;
    uint32_t max_polls = 0;
    std::vector<double> ring_us, done_us;
};

template <int V>
Result run_variant(sycl::queue& q, Words* w, int trips, uint32_t bound) {
    Result r;
    int misses = 0;
    for (int i = 1; i <= trips; ++i) {
        const uint32_t trip = (uint32_t) i * 7u + (uint32_t) V;
        host_store(&w->flag, 0); host_store(&w->ring, 0); host_store(&w->polls, 0); host_store(&w->seen, 0);
        probe::Watchdog dog(kWedgeSeconds, variant_name(V));
        const probe::clk::time_point t0 = probe::clk::now();
        submit<V>(q, w, trip, bound, false);
        bool rang = false;
        while (probe::s_since(t0) < kHostTimeout) {
            if (host_load(&w->ring) == trip) { rang = true; break; }
        }
        const probe::clk::time_point t1 = probe::clk::now();
        host_store(&w->flag, trip);
        q.wait_and_throw();
        ++r.trips;
        if (rang) { ++r.ring_seen; r.ring_us.push_back(probe::us_since(t0) - probe::us_since(t1)); }
        if (host_load(&w->seen) == 1) {
            ++r.flag_seen;
            r.done_us.push_back(probe::us_since(t1));
            r.max_polls = std::max(r.max_polls, host_load(&w->polls));
            misses = 0;
        } else if (++misses >= kAbandonAfter) {
            r.abandoned_at = i;
            break;
        }
    }
    return r;
}

void report(const Result& r, int trips, uint32_t bound) {
    std::vector<double> ring = r.ring_us, done = r.done_us;
    std::printf("  ring seen %d/%d (launch -> ring %.1f us median, %.1f p99); flag seen %d/%d (flag -> done %.1f us "
                "median, %.1f p99; at most %u of %u polls)%s\n",
                r.ring_seen, r.trips, probe::percentile(ring, 50), probe::percentile(ring, 99), r.flag_seen, r.trips,
                probe::percentile(done, 50), probe::percentile(done, 99), r.max_polls, bound,
                r.abandoned_at > 0 ? " - abandoned after three misses in a row" : "");
    (void) trips;
}

#if defined(PROBE_HAVE_GRAPH)
// The engine's own reading recorded once into a command_graph and replayed: the handshake inside a graph is what the
// engine does for every verify window.
int run_graph(sycl::queue& q, Words* w, uint32_t bound, int trips) {
    namespace sg = sycl::ext::oneapi::experimental;
    sg::command_graph<sg::graph_state::modifiable> g(q.get_context(), q.get_device());
    g.begin_recording(q);
    submit<kEngine>(q, w, 0, bound, true);
    g.end_recording();
    auto exec = g.finalize();
    int seen = 0;
    for (int i = 1; i <= trips; ++i) {
        const uint32_t trip = 100000u + (uint32_t) i;
        host_store(&w->flag, 0); host_store(&w->ring, 0); host_store(&w->polls, 0); host_store(&w->seen, 0);
        host_store(&w->trip, trip);
        probe::Watchdog dog(kWedgeSeconds, "the graph replay");
        const probe::clk::time_point t0 = probe::clk::now();
        q.ext_oneapi_graph(exec);
        while (probe::s_since(t0) < kHostTimeout && host_load(&w->ring) != trip) {}
        host_store(&w->flag, trip);
        q.wait_and_throw();
        if (host_load(&w->seen) == 1) ++seen;
    }
    return seen;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const int trips = argc > 1 ? std::max(1, std::atoi(argv[1])) : 500;
    probe::Gpu gpu;
    if (!probe::start("doorbell_probe", gpu)) return 2;
    try {
        sycl::queue q = probe::make_queue(gpu);
        Words* w = sycl::malloc_host<Words>(1, q);
        if (w == nullptr) { probe::verdict("fail", "malloc_host of five words failed"); return 2; }
        *w = Words{};
        const uint32_t bound = strata::spin_max(q);
        std::printf("spin bound %u polls (strata::spin_max; STRATA_SPIN_MAX overrides), %d round trips per reading\n\n",
                    bound, trips);
        Result r[6];
        bool complete[6] = {};
        std::printf("variant %s\n", variant_name(kVolatile));
        r[1] = run_variant<kVolatile>(q, w, trips, bound); report(r[1], trips, bound);
        std::printf("variant %s\n", variant_name(kVolatileFence));
        r[2] = run_variant<kVolatileFence>(q, w, trips, bound); report(r[2], trips, bound);
        std::printf("variant %s\n", variant_name(kAtomic));
        r[3] = run_variant<kAtomic>(q, w, trips, bound); report(r[3], trips, bound);
        std::printf("variant %s\n", variant_name(kUncached));
        r[4] = run_variant<kUncached>(q, w, trips, bound); report(r[4], trips, bound);
        std::printf("variant %s\n", variant_name(kEngine));
        r[5] = run_variant<kEngine>(q, w, trips, bound); report(r[5], trips, bound);
        std::string good;
        int rings = 0, ring_trips = 0;
        for (int v = 1; v <= 5; ++v) {
            complete[v] = r[v].trips == trips && r[v].flag_seen == trips;
            if (complete[v]) good += (good.empty() ? "" : ", ") + std::to_string(v);
            rings += r[v].ring_seen;
            ring_trips += r[v].trips;
        }
        int graph_seen = -1;
#if defined(PROBE_HAVE_GRAPH)
        std::printf("\ncommand_graph: variant 5 recorded once, replayed %d times\n", kGraphTrips);
        graph_seen = run_graph(q, w, bound, kGraphTrips);
        std::printf("  flag seen %d/%d inside the replayed graph\n", graph_seen, kGraphTrips);
#else
        std::printf("\n(no sycl_ext_oneapi_graph header: the graph replay was not measured)\n");
#endif
        std::printf("\n");
        if (rings < ring_trips) {
            probe::verdict("fail", "the host saw the kernel's ring %d of %d times: the engine cannot run on this driver "
                           "(its stager and token rings are read that way)", rings, ring_trips);
        } else if (good.empty()) {
            probe::verdict("partial", "no reading saw every host write; keep STRATA_VERIFY_NO_HOST=1 (the host waits "
                           "for the whole window; every expert must be in VRAM or the pinned mirror)");
        } else if (!complete[5]) {
            probe::verdict("partial", "variant(s) %s saw every host write but this build's sys_load (variant 5) did not: "
                           "rebuild with the matching -DSTRATA_DOORBELL_* option (sycl_doorbell.hpp)", good.c_str());
        } else if (graph_seen >= 0 && graph_seen < kGraphTrips) {
            probe::verdict("partial", "variant(s) %s complete when launched directly, but the graph replay saw %d/%d: "
                           "keep STRATA_VERIFY_NO_HOST=1", good.c_str(), graph_seen, kGraphTrips);
        } else {
            std::vector<double> done = r[5].done_us;
            probe::verdict("ok", "variant(s) %s saw every host write; this build's sys_load answers in %.1f us median "
                           "(at most %u polls)%s", good.c_str(), probe::percentile(done, 50), r[5].max_polls,
                           graph_seen >= 0 ? ", inside a replayed graph too" : "");
        }
        sycl::free(w, q);
    } catch (const sycl::exception& e) {
        probe::verdict("fail", "SYCL exception: %s", e.what());
        return 2;
    }
    return 0;
}
