// doorbell.cpp - does a host<->device flag handshake work through host USM on this card, and how?
//   mode 0: plain volatile loads/stores      mode 1: atomic_ref system scope     mode 2: L0 uncached host alloc + atomics
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
using clk = std::chrono::steady_clock;
using A = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>;
int main(int argc, char** argv) {
    const int mode = argc > 1 ? std::atoi(argv[1]) : 1;   // 3: like 1 but the kernel is recorded into a command_graph
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    uint32_t* m = nullptr;
    if (mode == 2 || mode == 6) {   // 6: uncached L0 allocation + cache-control hints
        auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        ze_host_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* p = nullptr;
        if (zeMemAllocHost(zctx, &d, 4096, 64, &p) != ZE_RESULT_SUCCESS) { std::puts("zeMemAllocHost failed"); return 2; }
        m = (uint32_t*) p;
    } else {
        m = sycl::malloc_host<uint32_t>(1024, q);
    }
    volatile uint32_t* flag = m;          // host -> device
    volatile uint32_t* seq = m + 16;      // device -> host
    *flag = 0; *seq = 0;
    std::printf("mode %d: %s\n", mode, q.get_device().get_info<sycl::info::device::name>().c_str());
    const auto t0 = clk::now();
    auto body = [=]() {
        uint32_t* f = const_cast<uint32_t*>(flag); uint32_t* s = const_cast<uint32_t*>(seq);
        if (mode == 0) { *s = 1; while (*flag != 7) {} *s = 2; }
        else if (mode == 4 || mode == 5 || mode == 6) {   // loads and stores that bypass every cache level
            namespace ie = sycl::ext::intel::experimental;
            namespace oe = sycl::ext::oneapi::experimental;
            using UC = decltype(oe::properties(
                ie::read_hint<ie::cache_control<ie::cache_mode::uncached, ie::cache_level::L1, ie::cache_level::L2, ie::cache_level::L3>>,
                ie::write_hint<ie::cache_control<ie::cache_mode::uncached, ie::cache_level::L1, ie::cache_level::L2, ie::cache_level::L3>>));
            oe::annotated_ptr<uint32_t, UC> F(f), S(s);
            *S = 1; sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            while (*F != 7) {}
            *S = 2; sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        }
        else { A(*s).store(1); sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
               while (A(*f).load() != 7) {} A(*s).store(2); sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system); }
    };
    namespace exp = sycl::ext::oneapi::experimental;
    sycl::event ev;
    exp::command_graph<exp::graph_state::modifiable> g(q.get_context(), q.get_device());
    if (mode == 9) {   // graph-launched, write only: is a completed graph's write to host USM visible to the host?
        uint32_t* dflag = sycl::malloc_device<uint32_t>(4, q);
        q.memset(dflag, 0, 16).wait();
        g.begin_recording(q);
        q.single_task([=]() { A(*const_cast<uint32_t*>(seq)).store(1); A(*dflag).store(7); });
        q.single_task([=]() { if (A(*dflag).load() == 7) A(*const_cast<uint32_t*>(seq)).store(2); else A(*const_cast<uint32_t*>(seq)).store(3); });
        g.end_recording();
        auto ge = g.finalize();
        q.ext_oneapi_graph(ge).wait();
        std::printf("  graph done: seq=%u (2 = host sees the graph's write and the second kernel saw the first's device write; 3 = device write not seen; 0 = host write not seen)\n", *seq);
        uint32_t back = 0; q.memcpy(&back, dflag, 4).wait();
        std::printf("  dflag read back by memcpy: %u\n", back);
        return *seq == 2 ? 0 : 1;
    }
    if (mode == 3 || mode == 5) {
        g.begin_recording(q);
        q.single_task(body);
        g.end_recording();
        auto ge = g.finalize();
        ev = q.ext_oneapi_graph(ge);
        std::printf("  graph: %zu nodes, launched\n", g.get_nodes().size());
    } else {
        ev = q.single_task(body);
    }
    // host: wait for the ring (1), then raise the flag, then wait for the second ring (2)
    auto wait_seq = [&](uint32_t want, const char* what) {
        while (*seq != want) {
            if (clk::now() - t0 > std::chrono::seconds(8)) { std::printf("  %s: TIMEOUT (seq=%u)\n", what, *seq); return false; }
        }
        std::printf("  %s: seen after %.1f ms\n", what, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        return true;
    };
    bool ok = wait_seq(1, "device->host ring 1");
    if (ok) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); *flag = 7; std::atomic_thread_fence(std::memory_order_seq_cst); ok = wait_seq(2, "host->device flag, ring 2"); }
    if (!ok) { std::printf("  (kernel still running - the driver will time it out)\n"); *flag = 7; }
    ev.wait();
    std::printf("  kernel done; %s\n", ok ? "HANDSHAKE OK" : "HANDSHAKE FAILED");
    return ok ? 0 : 1;
}
