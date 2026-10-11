// sycl/probe/probe_common.hpp - what the four decision probes share: the list of Level Zero GPUs and the choice of
// one, host clocks and percentiles, this process's memory, a watchdog for a wedged queue, and the one VERDICT line
// every probe ends with.
//
// Plain SYCL, no dpct: the probes run before the engine's design is trusted, so they depend on none of its code
// (doorbell_probe includes sycl_doorbell.hpp on purpose: its variant 5 is the engine's own sys_load).
#pragma once
#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace probe {

namespace syclex = sycl::ext::oneapi::experimental;

// --- clocks and figures --------------------------------------------------------------------------------------------
using clk = std::chrono::steady_clock;
inline double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}
inline double ms_since(clk::time_point t0) { return us_since(t0) / 1000.0; }
inline double s_since(clk::time_point t0) { return us_since(t0) / 1e6; }
inline double gib(uint64_t bytes) { return (double) bytes / 1073741824.0; }
inline double gb_per_s(uint64_t bytes, double seconds) { return seconds > 0 ? (double) bytes / seconds / 1e9 : 0.0; }

// percentile p (0..100) of a sample; sorts the sample in place
inline double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t) (p / 100.0 * (double) (v.size() - 1) + 0.5);
    if (i >= v.size()) i = v.size() - 1;
    return v[i];
}

// --- the environment -----------------------------------------------------------------------------------------------
inline const char* env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}
inline bool env_set(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != 0;
}
inline void print_env(const char* name) {
    const char* v = std::getenv(name);
    std::printf("  %s=%s\n", name, v ? v : "(unset)");
}

// --- the verdict ---------------------------------------------------------------------------------------------------
// The last line of every probe: "VERDICT: <ok|fail|partial> <one sentence>".  README.md says what each one changes.
inline void verdict(const char* state, const char* fmt, ...) {
    std::printf("VERDICT: %s ", state);
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
    std::fflush(stdout);
}

// --- a watchdog for waits that may never return --------------------------------------------------------------------
// A wedged Level Zero queue on Windows gives no TDR and no driver event (XxChonkExX/Vulkan-VM#4),
// so a queue wait that never returns would leave the probe silent.  The watchdog names what was waited for and ends
// the process with a fail verdict instead, as XeStrata's 120 s runtime watchdog does.
class Watchdog {
public:
    Watchdog(double seconds, const char* what)
        : what_(what),
          deadline_(clk::now() + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(seconds))) {
        thread_ = std::thread([this] { run(); });
    }
    ~Watchdog() {
        {
            std::lock_guard<std::mutex> g(mu_);
            done_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

private:
    void run() {
        std::unique_lock<std::mutex> lock(mu_);
        if (cv_.wait_until(lock, deadline_, [this] { return done_; })) return;
        std::printf("\nwatchdog: %s did not finish in time: the queue is wedged (recover the card with "
                    "Disable-PnpDevice / Enable-PnpDevice, or reboot)\n", what_);
        verdict("fail", "%s hung", what_);
        std::_Exit(3);
    }
    const char* what_;
    clk::time_point deadline_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread thread_;
};

// --- this process and this machine ----------------------------------------------------------------------------------
// This process's private (committed, unshared) memory, for the graph leak measurement; 0 when unavailable.
inline uint64_t private_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX c{};
    c.cb = sizeof c;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &c, sizeof c)) return 0;
    return (uint64_t) c.PrivateUsage;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long size = 0, resident = 0, shared = 0, text = 0, lib = 0, data = 0;
    const int n = std::fscanf(f, "%lu %lu %lu %lu %lu %lu", &size, &resident, &shared, &text, &lib, &data);
    std::fclose(f);
    if (n < 6) return 0;
    return (uint64_t) data * (uint64_t) sysconf(_SC_PAGESIZE);   // data + stack: the process's private pages
#endif
}

inline uint64_t physical_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? (uint64_t) m.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
#endif
}

// --- the GPUs ------------------------------------------------------------------------------------------------------
// The architecture enum's name.  The extension (sycl_ext_oneapi_device_architecture) has no to-string of its own, so
// this names the Intel GPUs a Strata user may have, DG1 on; "unknown" is the enum's own value, "arch <n>" an
// enumerator this list does not name.  Verified on the B70 (2026-10-04, oneAPI 2026.0.0): the runtime reports
// intel_gpu_bmg_g31 for it.
inline std::string arch_name(const sycl::device& d) {
    syclex::architecture a;
    try {
        a = d.get_info<syclex::info::device::architecture>();
    } catch (const sycl::exception& e) {
        return std::string("unreadable (") + e.what() + ")";
    }
    switch (a) {
        case syclex::architecture::intel_gpu_bmg_g31: return "intel_gpu_bmg_g31";
        case syclex::architecture::intel_gpu_bmg_g21: return "intel_gpu_bmg_g21";
        case syclex::architecture::intel_gpu_lnl_m: return "intel_gpu_lnl_m";
        case syclex::architecture::intel_gpu_mtl_h: return "intel_gpu_mtl_h";
        case syclex::architecture::intel_gpu_mtl_u: return "intel_gpu_mtl_u";   // also arl_u / arl_s / mtl_s
        case syclex::architecture::intel_gpu_arl_h: return "intel_gpu_arl_h";
        case syclex::architecture::intel_gpu_acm_g10: return "intel_gpu_acm_g10";
        case syclex::architecture::intel_gpu_acm_g11: return "intel_gpu_acm_g11";
        case syclex::architecture::intel_gpu_acm_g12: return "intel_gpu_acm_g12";
        case syclex::architecture::intel_gpu_pvc: return "intel_gpu_pvc";
        case syclex::architecture::intel_gpu_dg1: return "intel_gpu_dg1";
        case syclex::architecture::unknown: return "unknown";
        default: return "arch " + std::to_string((int) a);
    }
}

struct Gpu {
    sycl::device dev;
    int index = 0;   // its place among the Level Zero GPUs in SYCL's order: ONEAPI_DEVICE_SELECTOR=level_zero:<index>
    std::string name, arch, driver, platform;
    uint64_t global = 0, max_alloc = 0, local = 0;
    std::vector<size_t> sub_groups;
    bool usm_host = false, usm_device = false, fp16 = false;
};

// Only Level Zero devices: the engine runs on nothing else, and ONEAPI_DEVICE_SELECTOR=level_zero:N numbers these.
inline std::vector<Gpu> level_zero_gpus() {
    std::vector<Gpu> out;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) {
            Gpu g;
            g.dev = d;
            g.index = (int) out.size();
            g.name = d.get_info<sycl::info::device::name>();
            g.arch = arch_name(d);
            g.driver = d.get_info<sycl::info::device::driver_version>();
            g.platform = p.get_info<sycl::info::platform::name>();
            g.global = d.get_info<sycl::info::device::global_mem_size>();
            g.max_alloc = d.get_info<sycl::info::device::max_mem_alloc_size>();
            g.local = d.get_info<sycl::info::device::local_mem_size>();
            g.sub_groups = d.get_info<sycl::info::device::sub_group_sizes>();
            g.usm_host = d.has(sycl::aspect::usm_host_allocations);
            g.usm_device = d.has(sycl::aspect::usm_device_allocations);
            g.fp16 = d.has(sycl::aspect::fp16);
            out.push_back(g);
        }
    }
    return out;
}

inline void print_gpus(const std::vector<Gpu>& gpus) {
    std::printf("Level Zero GPUs (SYCL's order; ONEAPI_DEVICE_SELECTOR=level_zero:<n> picks one):\n");
    if (gpus.empty()) std::printf("  none\n");
    for (const auto& g : gpus) {
        std::string sg;
        for (size_t s : g.sub_groups) sg += (sg.empty() ? "" : "/") + std::to_string(s);
        std::printf("  device %d: %s\n", g.index, g.name.c_str());
        std::printf("    arch %s, %.2f GiB global, %.2f GiB max allocation, %llu KiB local, sub-groups %s, fp16 %s, "
                    "host USM %s, device USM %s\n    driver %s (%s)\n",
                    g.arch.c_str(), gib(g.global), gib(g.max_alloc), (unsigned long long) (g.local >> 10), sg.c_str(),
                    g.fp16 ? "yes" : "no", g.usm_host ? "yes" : "no", g.usm_device ? "yes" : "no", g.driver.c_str(),
                    g.platform.c_str());
    }
}

// ONEAPI_DEVICE_SELECTOR, when set, has already filtered the list: its first device is the user's choice.  Otherwise
// the first BMG-G31 (the Arc Pro B70 / B65 die), so that the processor's own GPU never takes the probe.
inline const Gpu* choose_gpu(const std::vector<Gpu>& gpus, std::string& why) {
    if (gpus.empty()) {
        why = "no Level Zero GPU: is the Intel driver installed, and does ONEAPI_DEVICE_SELECTOR (if set) name one?";
        return nullptr;
    }
    if (env_set("ONEAPI_DEVICE_SELECTOR")) {
        why = std::string("the first device ONEAPI_DEVICE_SELECTOR=") + env("ONEAPI_DEVICE_SELECTOR") + " exposes";
        return &gpus[0];
    }
    for (const auto& g : gpus)
        if (g.arch == "intel_gpu_bmg_g31") {
            why = "the first GPU whose architecture is intel_gpu_bmg_g31";
            return &g;
        }
    why = "no BMG-G31 among them: the first Level Zero GPU (ONEAPI_DEVICE_SELECTOR=level_zero:<n> picks another)";
    return &gpus[0];
}

// Every probe starts the same way: list the GPUs, choose one, say why.  false (after a fail verdict) when there is
// none.  stdout is unbuffered so that every line is out before a hang.
inline bool start(const char* probe, Gpu& chosen) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("%s (Strata, sycl/probe)\n", probe);
    print_env("ONEAPI_DEVICE_SELECTOR");
    print_env("SYCL_UR_USE_LEVEL_ZERO_V2");
    std::vector<Gpu> gpus;
    try {
        gpus = level_zero_gpus();
    } catch (const sycl::exception& e) {
        verdict("fail", "listing the Level Zero GPUs threw: %s", e.what());
        return false;
    }
    print_gpus(gpus);
    std::string why;
    const Gpu* g = choose_gpu(gpus, why);
    if (!g) {
        verdict("fail", "%s", why.c_str());
        return false;
    }
    chosen = *g;
    std::printf("chosen: device %d (%s): %s\n\n", g->index, g->name.c_str(), why.c_str());
    return true;
}

// Asynchronous errors (a kernel that faulted, a lost device) are printed, not thrown from a destructor.
inline sycl::async_handler async_report() {
    return [](sycl::exception_list l) {
        for (auto& ep : l) {
            try {
                std::rethrow_exception(ep);
            } catch (const sycl::exception& e) {
                std::printf("asynchronous SYCL exception: %s\n", e.what());
            } catch (const std::exception& e) {
                std::printf("asynchronous exception: %s\n", e.what());
            }
        }
    };
}

// An in-order queue in a context of its own on the chosen device: one device per context (compute-runtime#980: two
// Intel devices in one context mirror every VRAM allocation into host RAM).
inline sycl::queue make_queue(const Gpu& g) {
    sycl::context ctx(g.dev);
    return sycl::queue(ctx, g.dev, async_report(), sycl::property::queue::in_order{});
}

}  // namespace probe
