// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
//
// SYCL port: dpct's migration of upstream's file, with the device layer an Intel Arc needs (written for the Windows
// port on an Arc Pro B70, docs/INTEL_WINDOWS.md; the Linux paths keep their behaviour).
//   - strata::mem_get_info (strata/sycl_mem.hpp): the free-VRAM figure, on Windows lowered by what the WDDM budget
//     withholds, keyed by the SYCL device's LUID the way upstream does it for HIP (src/core/device.cu).
//   - The capability gate: 32-wide sub-groups, fp16, device and host USM, and the architecture enum.  dpct's
//     "compute capability" is the device's version string (Level Zero "1.6"), which passed upstream's kMinCc 75 on
//     Intel by accident; it stays in DeviceInfo::cc_major/minor for the prints, and decides nothing.
//   - The chunked DeviceArena and the VRAM heartbeat (strata/sycl_heartbeat.hpp).
// Windows-only code is under _WIN32.  The HIP branches of upstream's file (cudaDeviceProp, gcnArchName) are gone: a
// SYCL build never defines STRATA_USE_HIP and they did not compile as SYCL.
#if defined(_WIN32)
// Before <sycl/sycl.hpp>: dpct/device.hpp includes <windows.h>, whose min/max macros would break the SYCL headers.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/device.hpp"
#include "strata/sycl_heartbeat.hpp"
#include "strata/sycl_mem.hpp"
#include "strata/sycl_queue.hpp"   // malloc_device_guarded: the aliased-pages check of a big allocation (xe)

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <dxgi1_4.h>
#endif

namespace strata::core {

namespace {

void check(dpct::err0 e, const char *what) {
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw CudaError(
            std::string(what) + ": " + dpct::get_error_string_dummy(e), (int)e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.  (The arena fills with queue::fill below; this kernel stays as the function
// device_code_error() names, as dpct left it.)
constexpr uint32_t kPoisonBits = 0x7fc00000u;
__dpct_inline__ void poison_kernel(float *p, uint64_t n_floats) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint64_t i =
        (uint64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n_floats) p[i] = sycl::bit_cast<float>(kPoisonBits);
}

namespace syclex = sycl::ext::oneapi::experimental;

// The architecture enum's name (sycl_ext_oneapi_device_architecture).  The engine is compiled ahead of time for
// bmg-g31, the Arc Pro B70/B65 die; "bmg" alone is the B580's and that binary does not load on a B70, so the name
// is what the probe prints and what gpu_arch_problem() checks.  Anything the runtime cannot name is "unknown".
// unverified on the B70: that the Windows Level Zero driver reports intel_gpu_bmg_g31 (intel/llvm 6.2.0 knows the
// enumerator).
const char* arch_name(const sycl::device& dev) {
#if defined(SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE)
    try {
        switch (dev.get_info<syclex::info::device::architecture>()) {
            case syclex::architecture::intel_gpu_bmg_g31: return "intel_gpu_bmg_g31";
            case syclex::architecture::intel_gpu_bmg_g21: return "intel_gpu_bmg_g21";
            case syclex::architecture::intel_gpu_lnl_m: return "intel_gpu_lnl_m";
            case syclex::architecture::intel_gpu_acm_g10: return "intel_gpu_acm_g10";
            case syclex::architecture::intel_gpu_acm_g11: return "intel_gpu_acm_g11";
            case syclex::architecture::intel_gpu_acm_g12: return "intel_gpu_acm_g12";
            case syclex::architecture::intel_gpu_pvc: return "intel_gpu_pvc";
            default: return "unknown";
        }
    } catch (const sycl::exception&) {
        return "unknown";
    }
#else
    (void) dev;
    return "unknown";
#endif
}

// The Level Zero devices in SYCL's order.  ONEAPI_DEVICE_SELECTOR=level_zero:N names the Nth of these, which is how
// setup and the server address the card, so the ordinals of this file count the same way.  dpct's own numbering
// puts the default selector's choice first; device_info() and DeviceArena map an ordinal to dpct's id when they
// select the device.  Empty when the runtime has no Level Zero platform (the caller says "(no GPU device)").
const std::vector<sycl::device>& l0_devices() {
    static const std::vector<sycl::device> devs = [] {
        std::vector<sycl::device> r;
        try {
            for (const sycl::device& d : sycl::device::get_devices(sycl::info::device_type::all))
                if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) r.push_back(d);
        } catch (const sycl::exception&) {
            r.clear();
        }
        return r;
    }();
    return devs;
}

// Why the device cannot run the kernels, from what it reports ("" when it can): every kernel is built with
// -fsycl-default-sub-group-size=32 (lane = tid & 31 throughout), the attention and the quantizers use fp16, the
// expert arena and the doorbell are host USM.  Not a device-ID check: the integrated GPU beside the B70 is refused
// by what it lacks, or runs.
std::string capability_problem(const sycl::device& dev) {
    try {
        if (!dev.has(sycl::aspect::usm_device_allocations)) return "no device USM (aspect::usm_device_allocations)";
        if (!dev.has(sycl::aspect::usm_host_allocations))
            return "no host USM (aspect::usm_host_allocations); the expert arena and the doorbell live in host memory";
        if (!dev.has(sycl::aspect::fp16)) return "no fp16 (aspect::fp16)";
        const std::vector<size_t> sg = dev.get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(sg.begin(), sg.end(), (size_t) 32) == sg.end()) {
            std::string have;
            for (size_t s : sg) have += (have.empty() ? "" : ",") + std::to_string(s);
            return "no 32-wide sub-groups (this device offers " + have +
                   "); every kernel is built with -fsycl-default-sub-group-size=32";
        }
    } catch (const sycl::exception& e) {
        return std::string("the runtime cannot describe it: ") + e.what();
    }
    return "";
}

// An ahead-of-time binary carries device code for one die.  STRATA_SYCL_AOT_DEVICE is the -device name the build
// passed to ocloc (sycl/CMakeLists.txt); only bmg-g31 has a known architecture enumerator to check against.
std::string aot_problem(const sycl::device& dev) {
#if defined(STRATA_SYCL_AOT_DEVICE)
    if (std::strcmp(STRATA_SYCL_AOT_DEVICE, "bmg-g31") == 0) {
        const std::string arch = arch_name(dev);
        if (arch != "intel_gpu_bmg_g31")
            return "this engine's device code is compiled for bmg-g31 (Arc Pro B70/B65); this is " + arch;
    }
#else
    (void) dev;
#endif
    return "";
}

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_SYCL_AOT_DEVICE)
    return STRATA_SYCL_AOT_DEVICE;
#else
    return "";
#endif
}

int device_count() {
    try {
        return (int) l0_devices().size();
    } catch (const std::exception&) {   // a runtime that cannot enumerate is "no device", not an exit
        return 0;
    }
}

bool device_summary(int ordinal, std::string &name, std::string &detail) {
    const std::vector<sycl::device>& devs = l0_devices();
    if (ordinal < 0 || (size_t) ordinal >= devs.size()) return false;
    try {
        const sycl::device& dev = devs[(size_t) ordinal];
        name = dev.get_info<sycl::info::device::name>();
        char buf[160];
        std::snprintf(buf, sizeof(buf), "arch %s, %.1f GiB", arch_name(dev),
                      (double) dev.get_info<sycl::info::device::global_mem_size>() / (1024.0 * 1024 * 1024));
        detail = buf;
        return true;
    } catch (const sycl::exception&) {
        return false;
    }
}

}  // namespace strata::core

// ------------------------------------------------------------------------------------------ the free VRAM figure
// The free VRAM figure on Windows.  Upstream found this on HIP (src/core/device.cu 106-191): the
// runtime's free figure is the card's size minus this process's own allocations; it never counts what the desktop
// and other programs hold, and Windows gives each process a video memory budget (DXGI QueryVideoMemoryInfo) that
// does.  Allocations past the budget still succeed, and are what WDDM moves out to system RAM (measured there: decode
// at 30 tok/s instead of 41 once the expert cache had filled the card).  So the free figure is the runtime's lowered
// by what the budget withholds (the card's size minus the budget).  Subtracting from the runtime's own figure, rather
// than taking the budget minus DXGI's usage, keeps memory the runtime has freed and caches counted as free.
// dxgi.dll is loaded at run time, so nothing new is linked; when it, the LUID or the adapter cannot be found, the
// runtime's figure stands.  On the dedicated B70 the budget is close to the card's size (the Windows driver already
// advertises 0.98 of WDDM's dedicated memory as global_mem_size, so a budget at or above it withholds nothing).
// unverified on the B70: the budget figures themselves; measured on an RX 6800 under HIP (0.8-1.8 GiB withheld).
namespace strata {
namespace {
#if defined(_WIN32)
IDXGIAdapter3* budget_adapter(unsigned device) {
    constexpr unsigned kMaxDevices = 16;
    static std::mutex mu;
    static IDXGIAdapter3* adapters[kMaxDevices] = {};
    static bool tried[kMaxDevices] = {};
    if (device >= kMaxDevices) return nullptr;
    std::lock_guard<std::mutex> lock(mu);
    if (tried[device]) return adapters[device];
    tried[device] = true;
    unsigned char bytes[8];
    if (!device_luid(bytes)) return nullptr;
    LUID luid{};
    std::memcpy(&luid, bytes, sizeof luid);
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    static HMODULE dxgi = LoadLibraryW(L"dxgi.dll");   // kept for the process' lifetime, like the adapters
    const auto create = dxgi ? (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory4* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory4), (void**) &factory))) return nullptr;
    IDXGIAdapter3* adapter = nullptr;
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**) &adapter))) adapter = nullptr;
    factory->Release();
    adapters[device] = adapter;
    return adapter;
}

void lower_by_wddm_budget(size_t& free_bytes, size_t total_bytes) {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_WDDM_BUDGET");
        return v != nullptr && std::atoi(v) == 0;
    }();
    if (off || total_bytes == 0) return;
    unsigned device = 0;
    try {
        device = dpct::get_current_device_id();
    } catch (const std::exception&) {
        return;
    }
    IDXGIAdapter3* adapter = budget_adapter(device);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    if (adapter == nullptr || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) ||
        local.Budget == 0 || local.Budget >= total_bytes)
        return;
    const size_t withheld = total_bytes - (size_t) local.Budget;   // the desktop's and other programs' share
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) {
        std::fprintf(stderr, "strata: Windows budgets %llu of this card's %llu MiB for this process; free VRAM is "
                             "counted within that (STRATA_WDDM_BUDGET=0: off)\n",
                     (unsigned long long) (local.Budget >> 20), (unsigned long long) (total_bytes >> 20));
    }
    free_bytes = free_bytes > withheld ? free_bytes - withheld : 0;
}
#endif
}  // namespace

bool device_luid(unsigned char (&luid)[8]) {
#if defined(_WIN32) && defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 8
    // sycl_ext_intel_device_info version 8 (DPC++ 2025.3): the adapter LUID, 8 bytes, behind its own aspect
    try {
        const sycl::device& dev = dpct::get_current_device();
        if (!dev.has(sycl::aspect::ext_intel_device_info_luid)) return false;
        const std::array<unsigned char, 8> l = dev.get_info<sycl::ext::intel::info::device::luid>();
        std::memcpy(luid, l.data(), sizeof luid);
        return true;
    } catch (const std::exception&) {
        return false;
    }
#else
    (void) luid;
    return false;
#endif
}

bool mem_get_info(size_t& free_bytes, size_t& total_bytes) {
    free_bytes = total_bytes = 0;
    bool have_free = false;
    try {
        dpct::device_ext& dev = dpct::get_current_device();
        total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
        // the query behind dpct's get_memory_info (ext_intel_free_memory: Level Zero Sysman, ZES_ENABLE_SYSMAN=1),
        // without its stderr line: the caller decides what an unknown figure means
#if defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 4
        if (dev.has(sycl::aspect::ext_intel_free_memory)) {
            free_bytes = (size_t) dev.get_info<sycl::ext::intel::info::device::free_memory>();
            have_free = true;
        }
#endif
    } catch (const std::exception&) {
        return false;
    }
    if (!have_free) return false;
#if defined(_WIN32)
    lower_by_wddm_budget(free_bytes, total_bytes);
#endif
    return true;
}

// ------------------------------------------------------------------------------------------- the VRAM heartbeat
// Contract C9 (strata/sycl_heartbeat.hpp).  The registry is a vector under a mutex; a touch copies it, submits one
// read-only kernel per range and waits.  Unregister takes the touch mutex first, so a range is never freed while a
// kernel that reads it is in flight (the arena's destructor would otherwise race the heartbeat thread).
namespace {
struct HbRange {
    const void* ptr;
    size_t bytes;
    unsigned dev;   // dpct's id of the device the range lives on: the touch queue must be in that device's context
};
struct HbQueue {
    sycl::queue* q = nullptr;   // owned by dpct's device_ext, like every migrated queue
    uint32_t* scratch = nullptr;
};
std::mutex& hb_touch_mutex() { static std::mutex m; return m; }
std::mutex& hb_mutex() { static std::mutex m; return m; }
std::vector<HbRange>& hb_ranges() { static std::vector<HbRange> r; return r; }
constexpr size_t kHbStrideWords = (64 * 1024) / sizeof(uint32_t);   // one read every 64 KiB
constexpr size_t kHbGroup = 256;
}  // namespace
class strata_vram_heartbeat_kernel;

void heartbeat_register(const void* ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0) return;
    unsigned dev = 0;
    try {
        dev = dpct::get_current_device_id();
    } catch (const std::exception&) {
        return;
    }
    std::lock_guard<std::mutex> lock(hb_mutex());
    hb_ranges().push_back(HbRange{ptr, bytes, dev});
}

void heartbeat_unregister(const void* ptr) {
    std::lock_guard<std::mutex> touching(hb_touch_mutex());   // a touch in flight finishes first
    std::lock_guard<std::mutex> lock(hb_mutex());
    std::vector<HbRange>& r = hb_ranges();
    r.erase(std::remove_if(r.begin(), r.end(), [ptr](const HbRange& x) { return x.ptr == ptr; }), r.end());
}

size_t heartbeat_touch() {
    std::lock_guard<std::mutex> touching(hb_touch_mutex());
    std::vector<HbRange> ranges;
    {
        std::lock_guard<std::mutex> lock(hb_mutex());
        ranges = hb_ranges();
    }
    if (ranges.empty()) return 0;
    static std::map<unsigned, HbQueue> queues;   // under hb_touch_mutex
    size_t covered = 0;
    try {
        std::vector<sycl::queue*> used;
        for (const HbRange& r : ranges) {
            HbQueue& hq = queues[r.dev];
            if (hq.q == nullptr) {
                // its own in-order queue in dpct's context for that device, created on the first touch: the engine's
                // queues are never waited on from here, and the USM pointers are valid only in their context
                hq.q = dpct::get_device(r.dev).create_queue(true);
                hq.scratch = sycl::malloc_device<uint32_t>(1, *hq.q);
                if (hq.scratch == nullptr) throw std::runtime_error("no device memory for the heartbeat's scratch word");
                hq.q->memset(hq.scratch, 0, sizeof(uint32_t)).wait_and_throw();
            }
            const size_t words = r.bytes / sizeof(uint32_t);
            if (words == 0) continue;
            const size_t n = (words + kHbStrideWords - 1) / kHbStrideWords;   // the last read stays inside the range
            const size_t groups = (n + kHbGroup - 1) / kHbGroup;
            const uint32_t* base = static_cast<const uint32_t*>(r.ptr);
            uint32_t* acc = hq.scratch;
            // Every page of the range is read; the sum lands in the scratch word so the reads survive optimisation.
            // One atomic per work-group, not per read: 30 GiB is 480k reads.
            hq.q->parallel_for<strata_vram_heartbeat_kernel>(
                sycl::nd_range<1>(groups * kHbGroup, kHbGroup), [=](sycl::nd_item<1> it) {
                    const size_t i = it.get_global_id(0);
                    const uint32_t v = i < n ? base[i * kHbStrideWords] : 0u;
                    const uint32_t sum = sycl::reduce_over_group(it.get_group(), v, sycl::plus<uint32_t>());
                    if (it.get_local_id(0) == 0) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> a(*acc);
                        a.fetch_add(sum);
                    }
                });
            covered += r.bytes;
            if (std::find(used.begin(), used.end(), hq.q) == used.end()) used.push_back(hq.q);
        }
        for (sycl::queue* q : used) q->wait_and_throw();
    } catch (const std::exception& e) {
        // unverified on the B70: what a wedged or lost Level Zero queue reports here on Windows (no TDR, no event;
        // intel/llvm#23209 says DEVICE_LOST).  Reported, not fatal: the engine's own next submission decides.
        std::fprintf(stderr, "strata: VRAM heartbeat failed: %s\n", e.what());
        return 0;
    }
    return covered;
}

unsigned heartbeat_period_s() {
    static const unsigned period = [] {
#if defined(_WIN32)
        unsigned v = 10;   // WDDM evicts idle memory; 10 s is well inside the idle minutes of intel/llvm#23209
#else
        unsigned v = 0;    // Linux: no eviction of idle VRAM; off unless asked
#endif
        if (const char* e = std::getenv("STRATA_SYCL_HEARTBEAT_S"); e != nullptr && *e != '\0') {
            const long n = std::strtol(e, nullptr, 10);
            v = n <= 0 ? 0u : (unsigned) n;
        }
        return v;
    }();
    return period;
}
}  // namespace strata

namespace strata::core {

std::string gpu_arch_problem(int ordinal) {
    const std::vector<sycl::device>& devs = l0_devices();
    if (ordinal < 0 || (size_t) ordinal >= devs.size()) return "";
    const sycl::device& dev = devs[(size_t) ordinal];
    if (const std::string why = capability_problem(dev); !why.empty()) return why;
    return aot_problem(dev);
}

std::string device_code_error() try {
    // every .cu of the engine is compiled for the same CMAKE_CUDA_ARCHITECTURES, so this kernel stands for all
    // (gpu_arch_problem() checks the ahead-of-time die before this point; dpct's query here asks the device, not
    // the binary)
    dpct::kernel_function_info a{};
    const dpct::err0 e = DPCT_CHECK_ERROR(
        dpct::get_kernel_function_info(&a, (const void *)poison_kernel));
    if (e == 0) return {};
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    return dpct::get_error_string_dummy(e);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

DeviceInfo device_info(int ordinal) {
    const std::vector<sycl::device>& devs = l0_devices();
    if (devs.empty()) {
        throw CudaError("no Level Zero GPU is present; Strata's SYCL engine needs an Intel GPU on Level Zero (Arc Pro "
                        "B70: intel_gpu_bmg_g31, driver 32.0.101.9034 or newer; ONEAPI_DEVICE_SELECTOR=level_zero:N "
                        "chooses one)",
                        -1);
    }
    if (ordinal < 0 || (size_t) ordinal >= devs.size()) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(devs.size()) + ")",
                        -1);
    }
    const sycl::device& dev = devs[(size_t) ordinal];
    DeviceInfo d;
    d.ordinal = ordinal;
    unsigned id = 0;
    try {
        id = dpct::get_device_id(dev);   // dpct numbers the default selector's choice first; this is cudaSetDevice
        dpct::select_device(id);
    } catch (const std::exception& e) {
        throw CudaError(std::string("cudaSetDevice: ") + e.what(), -1);
    }

    dpct::device_info p{};
    check(DPCT_CHECK_ERROR(dpct::get_device(id).get_device_info(p)),
          "cudaGetDeviceProperties");
    d.name = p.get_name();
    /*
    DPCT1005: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    d.cc_major = p.get_major_version();   // the version string's major.minor (Level Zero "1.6"): a print, not a gate
    d.cc_minor = p.get_minor_version();
    d.multi_processor_count = p.get_max_compute_units();

    size_t free_b = 0, total_b = 0;
    if (strata::mem_get_info(free_b, total_b)) {
        d.free_bytes = free_b;
        d.total_bytes = total_b;
    } else {
        // llama.cpp's fallback (PR #28227): without Sysman's figure, count the card's size, and say so once
        d.total_bytes = total_b != 0 ? total_b : p.get_global_mem_size();
        d.free_bytes = d.total_bytes;
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) {
            std::fprintf(stderr, "strata: free VRAM is unknown (no ext_intel_free_memory; ZES_ENABLE_SYSMAN=1 enables "
                                 "Level Zero Sysman); counting the card's %llu MiB as free\n",
                         (unsigned long long) (d.total_bytes >> 20));
        }
    }

    /*
    DPCT1043: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.driver_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaDriverGetVersion");
    /*
    DPCT1043: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.runtime_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaRuntimeGetVersion");

    // The run-time half of the architecture policy: CMake compiles for one die, this refuses to RUN on another, or
    // on a device without the sub-group width, the fp16 and the USM the kernels assume.  A binary can be carried to
    // a machine with another card and would otherwise fail at its first launch, with nothing that names the cause.
    d.arch = arch_name(dev);
    if (const std::string why = gpu_arch_problem(ordinal); !why.empty()) {
        throw CudaError("device " + std::to_string(ordinal) + " (" + d.name + ", " + d.arch +
                            ") cannot run this engine: " + why,
                        -1);
    }
    return d;
}

// ------------------------------------------------------------------------------------------- the chunked arena
namespace {
// The chunk rule.  STRATA_ARENA_CHUNK_MIB when set (0 = one allocation).  Otherwise ONE allocation
// first, as upstream: measured on the Windows B70 (sycl/probe/alloc_probe.exe, 2026-10-04, a PC with 128 GiB of
// RAM), every single device allocation up to the reported maximum of 31.16 GiB succeeded and filled at 349-446 GB/s,
// so chunking has nothing to gain there, and a request larger than a chunk is the one thing alloc() cannot serve.
// When the one allocation is refused, the fallback is 60% of max_mem_alloc_size (llama.cpp PR #28953 on BMG-G31, for
// compute-runtime#998: one allocation above a threshold that tracks host RAM plus swap fails, 19.3 GiB on a 16 GiB
// box), and on Windows at most 4096 MiB (native-Windows B70 reports: UR errors 39/40 at and above 4 GiB even with
// UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1; not reproduced by the probe on the 128 GiB PC).  STRATA_MAX_ALLOC_MIB
// stands in for the reported maximum, as in XeStrata (a smaller card, for checking the rule).
uint64_t arena_chunk_bytes(const sycl::device& dev, uint64_t total, uint64_t& max_alloc, std::string& how,
                           bool& retry_chunked) {
    max_alloc = 0;
    retry_chunked = false;
    try {
        max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    } catch (const sycl::exception&) {
    }
    if (const char* v = std::getenv("STRATA_MAX_ALLOC_MIB"); v != nullptr && *v != '\0') {
        const uint64_t mib = std::strtoull(v, nullptr, 10);
        if (mib != 0) max_alloc = mib << 20;
    }
    if (const char* v = std::getenv("STRATA_ARENA_CHUNK_MIB"); v != nullptr && *v != '\0') {
        how = v;
        const uint64_t mib = std::strtoull(v, nullptr, 10);
        return mib == 0 ? total : mib << 20;
    }
    how = "auto";
    retry_chunked = max_alloc != 0;   // one allocation first; the fallback below only when that is refused
    return total;
}

// The fallback chunk when the one allocation was refused (see arena_chunk_bytes).
uint64_t fallback_chunk_bytes(uint64_t max_alloc, uint64_t total) {
    uint64_t chunk = max_alloc / 10 * 6;
#if defined(_WIN32)
    chunk = std::min<uint64_t>(chunk, 4096ull << 20);
#endif
    chunk = chunk >> 20 << 20;   // whole MiB, for the log line and the probes
    return chunk == 0 || chunk >= total ? total : chunk;
}
}  // namespace

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    const std::vector<sycl::device>& devs = l0_devices();
    if (ordinal < 0 || (size_t) ordinal >= devs.size()) {
        throw CudaError("DeviceArena: device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(devs.size()) + " Level Zero devices)",
                        -1);
    }
    const sycl::device& dev = devs[(size_t) ordinal];
    try {
        dev_id_ = dpct::get_device_id(dev);
        dpct::select_device(dev_id_);
    } catch (const std::exception& e) {
        throw CudaError(std::string("cudaSetDevice: ") + e.what(), -1);
    }
    sycl::queue& q = dpct::get_device(dev_id_).in_order_queue();

    // One allocation for the region when the device takes it (measured on the B70: it does, up to 31 GiB), else
    // several (arena_chunk_bytes).  They happen here, before anything depends on them.
    uint64_t max_alloc = 0;
    std::string how;
    bool retry_chunked = false;
    chunk_bytes_ = arena_chunk_bytes(dev, bytes, max_alloc, how, retry_chunked);
    // allocates every chunk of the current chunk_bytes_; false (with the arena released) when the device refuses one
    auto allocate_all = [&](uint64_t& n_chunks) -> bool {
        n_chunks = (bytes + chunk_bytes_ - 1) / chunk_bytes_;
        chunks_.reserve((size_t) n_chunks);
        for (uint64_t off = 0; off < bytes; off += chunk_bytes_) {
            Chunk c;
            c.bytes = std::min<uint64_t>(chunk_bytes_, bytes - off);
            // alloc() aligns on the absolute address, so the chunk's own alignment does not matter.  The guarded
            // allocation is upstream's for every big device buffer: on xe a 22 GiB arena came back with two pages
            // mapped onto the same memory (sycl_queue.hpp, arena_alias_check), and the check moves such a buffer.
            try {
                c.ptr = strata::malloc_device_guarded((size_t) c.bytes, q, "VRAM arena");
            } catch (const std::exception&) {   // Level Zero reports a refused size as an exception, not null: a
                c.ptr = nullptr;                 // sycl::exception, or std::bad_alloc from the runtime; both are the refusal
            }
            if (c.ptr == nullptr) {
                release();
                return false;
            }
            chunks_.push_back(c);
            strata::heartbeat_register(c.ptr, (size_t) c.bytes);
        }
        return true;
    };
    uint64_t n_chunks = 0;
    try {
        bool ok = allocate_all(n_chunks);
        if (!ok && retry_chunked) {
            const uint64_t one = chunk_bytes_;
            chunk_bytes_ = fallback_chunk_bytes(max_alloc, bytes);
            if (chunk_bytes_ < one) {
                std::fprintf(stderr, "strata: one VRAM allocation of %llu MiB was refused (compute-runtime#998?); "
                                     "trying %llu MiB chunks\n",
                             (unsigned long long) (one >> 20), (unsigned long long) (chunk_bytes_ >> 20));
                how = "auto, chunked after a refusal";
                ok = allocate_all(n_chunks);
            }
        }
        if (!ok) {
            char msg[320];
            std::snprintf(msg, sizeof msg,
                          "cudaMalloc: device allocation refused for a %llu MiB arena in %llu MiB chunks "
                          "(max_mem_alloc_size %llu MiB; compute-runtime#998; STRATA_ARENA_CHUNK_MIB=N sets the chunk)",
                          (unsigned long long) (bytes >> 20), (unsigned long long) (std::min(chunk_bytes_, bytes) >> 20),
                          (unsigned long long) (max_alloc >> 20));
            throw CudaError(msg, -1);
        }
        std::fprintf(stderr, "strata: VRAM arena %llu chunk%s of %llu MiB (max_mem_alloc_size %llu MiB, STRATA_ARENA_CHUNK_MIB=%s)\n",
                     (unsigned long long) n_chunks, n_chunks == 1 ? "" : "s",
                     (unsigned long long) (std::min(chunk_bytes_, bytes) >> 20), (unsigned long long) (max_alloc >> 20),
                     how.c_str());
        if (poison_) {
            // in pieces of at most 1 GiB: a single fill past 4 GiB never completed on the B70 (XeStrata, 2026-09-30)
            constexpr uint64_t kPiece = 1ull << 30;
            for (const Chunk& c : chunks_) {
                for (uint64_t o = 0; o < c.bytes; o += kPiece) {
                    const size_t n = (size_t) (std::min(kPiece, c.bytes - o) / sizeof(uint32_t));
                    q.fill<uint32_t>((uint32_t*) ((char*) c.ptr + o), kPoisonBits, n);
                }
            }
            q.wait_and_throw();
        }
    } catch (const sycl::exception& e) {
        release();
        throw CudaError(std::string("cudaMalloc: ") + e.what(), -1);
    } catch (...) {
        release();
        throw;
    }
}

DeviceArena::~DeviceArena() { release(); }

void DeviceArena::release() noexcept {
    if (chunks_.empty()) return;
    try {
        dpct::get_device(dev_id_).queues_wait_and_throw();   // cudaFree's implicit sync: no kernel still reads the chunks
    } catch (...) {
    }
    for (Chunk& c : chunks_) {
        if (c.ptr == nullptr) continue;
        strata::heartbeat_unregister(c.ptr);   // before the free: a touch in flight finishes first
        try {
            sycl::free(c.ptr, dpct::get_device(dev_id_).get_context());   // best effort: a destructor must not throw
        } catch (...) {
        }
        c.ptr = nullptr;
    }
    chunks_.clear();
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    // the first chunk, from the current one on, with room for the aligned request: found before anything changes,
    // so a refused request leaves the arena as it was
    size_t i = cur_;
    uint64_t padding = 0;
    for (; i < chunks_.size(); ++i) {
        const Chunk& c = chunks_[i];
        const uintptr_t address = (uintptr_t) c.ptr + c.used;
        padding = (align - (address & (align - 1))) & (align - 1);
        if (padding <= c.bytes - c.used && bytes <= c.bytes - c.used - padding) break;
    }
    if (i == chunks_.size()) {
        char msg[320];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B (align %llu) with %llu of %llu B used in %llu chunks "
                      "of %llu B - the plan from P1.S9 did not close, or the request is larger than a chunk "
                      "(STRATA_ARENA_CHUNK_MIB)",
                      (unsigned long long) bytes, (unsigned long long) align, (unsigned long long) used_,
                      (unsigned long long) capacity_, (unsigned long long) chunks_.size(),
                      (unsigned long long) chunk_bytes_);
        throw CudaError(msg, -1);
    }
    for (; cur_ < i; ++cur_) {   // the tails of the chunks passed over count as used: nothing allocates there later
        used_ += chunks_[cur_].bytes - chunks_[cur_].used;
        chunks_[cur_].used = chunks_[cur_].bytes;
    }
    Chunk& c = chunks_[cur_];
    const uint64_t start = c.used + padding;
    c.used = start + bytes;
    used_ += padding + bytes;
    return (char*) c.ptr + start;
}

}  // namespace strata::core
