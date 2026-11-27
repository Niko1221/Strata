#include "strata/core/remote_experts.hpp"
#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/helper_plugin_abi.h"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::kWindowMaxT * 10;   // a window's rows x top-10

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}
} // namespace


// ---- strata-mi50 (2026-10-23): the helper tier in a plugin DLL with its own HIP runtime (helper_plugin_abi.h) ----
namespace {
struct HelperPlugin {
    std::string dll, pack;
    int device = 0;
    bool loaded = false;
    strata_helper_init_t init = nullptr;
    strata_helper_open_t open = nullptr;
    strata_helper_close_t close = nullptr;
    strata_helper_begin_t begin = nullptr;
    strata_helper_finish_t finish = nullptr;
    strata_helper_owns_t owns = nullptr;
    strata_helper_holds_t holds = nullptr;
    strata_helper_stats_t stats = nullptr;
    strata_helper_open2_t open2 = nullptr;           // ABI 2 (2026-10-26): --remote-expert-opt
    strata_helper_begin_opt_t begin_opt = nullptr;
    strata_helper_finish_opt_t finish_opt = nullptr;
    strata_helper_adapt_t adapt = nullptr;           // ABI 3 (2026-10-26): adaptive swaps on the helper
    strata_helper_copy_blobs_t copy_blobs = nullptr; // ABI 4 (2026-10-27): the prompt path reads the helper's blobs
    strata_helper_prefill_reserve_t pf_reserve = nullptr;   // ABI 5 (2026-10-27): the helper computes prompt experts
    strata_helper_prefill_begin_t pf_begin = nullptr;
    strata_helper_prefill_finish_t pf_finish = nullptr;
};
HelperPlugin& helper_plugin() { static HelperPlugin p; return p; }

bool load_helper_plugin(std::string& err) {
    HelperPlugin& p = helper_plugin();
    if (p.loaded) return true;
#if defined(_WIN32)
    // 2026-10-26: the driver's HIP 5.x runtime first, with its dependencies from System32 only.  It imports amd_comgr.dll by
    // that name, and an engine package folder can hold another one (the official package's HIP 7 copy beside strata.exe):
    // loaded from there, the HIP 5.x runtime fails (cudaMemGetInfo failed in the plugin).  The plugin's import then binds
    // to this already loaded module.
    LoadLibraryExA("amdhip64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    // the plugin's own folder first: its HIP 5.x libraries (hipblas.dll, rocblas.dll) sit beside it or on PATH
    HMODULE m = LoadLibraryExA(p.dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (m == nullptr) m = LoadLibraryA(p.dll.c_str());
    if (m == nullptr) {
        err = "helper plugin: cannot load " + p.dll + " (error " + std::to_string((unsigned long) GetLastError()) + ")";
        return false;
    }
    auto sym = [&](const char* n) { return (void*) GetProcAddress(m, n); };
    auto ver = (strata_helper_abi_version_t) sym("strata_helper_abi_version");
    // 2026-10-27: ABI 4 plugins still load (no prefill_* functions); was: ver() != STRATA_HELPER_ABI_VERSION
    if (ver == nullptr || ver() < STRATA_HELPER_ABI_MIN || ver() > STRATA_HELPER_ABI_VERSION) { err = "helper plugin: " + p.dll + " has another ABI version"; return false; }
    p.init = (strata_helper_init_t) sym("strata_helper_init");
    p.open = (strata_helper_open_t) sym("strata_helper_open");
    p.close = (strata_helper_close_t) sym("strata_helper_close");
    p.begin = (strata_helper_begin_t) sym("strata_helper_begin");
    p.finish = (strata_helper_finish_t) sym("strata_helper_finish");
    p.owns = (strata_helper_owns_t) sym("strata_helper_owns");
    p.holds = (strata_helper_holds_t) sym("strata_helper_holds");
    p.stats = (strata_helper_stats_t) sym("strata_helper_stats");
    p.open2 = (strata_helper_open2_t) sym("strata_helper_open2");
    p.begin_opt = (strata_helper_begin_opt_t) sym("strata_helper_begin_opt");
    p.finish_opt = (strata_helper_finish_opt_t) sym("strata_helper_finish_opt");
    p.adapt = (strata_helper_adapt_t) sym("strata_helper_adapt");
    p.copy_blobs = (strata_helper_copy_blobs_t) sym("strata_helper_copy_blobs");
    if (ver() >= 5) {
        p.pf_reserve = (strata_helper_prefill_reserve_t) sym("strata_helper_prefill_reserve");
        p.pf_begin = (strata_helper_prefill_begin_t) sym("strata_helper_prefill_begin");
        p.pf_finish = (strata_helper_prefill_finish_t) sym("strata_helper_prefill_finish");
    }
    if (!p.init || !p.open || !p.close || !p.begin || !p.finish || !p.owns || !p.holds || !p.stats ||
        !p.open2 || !p.begin_opt || !p.finish_opt || !p.adapt || !p.copy_blobs) {
        err = "helper plugin: " + p.dll + " lacks an entry point";
        return false;
    }
    p.loaded = true;
    return true;
#else
    err = "helper plugin: only on Windows";
    return false;
#endif
}
struct PluginCtx { const ExpertCache* primary; ExpertSource* source; };
int plugin_primary_cb(void* ctx, int32_t layer, int32_t expert) {
    return ((PluginCtx*) ctx)->primary->slot_of(layer, expert) >= 0 ? 1 : 0;
}
const uint8_t* plugin_blob_cb(void* ctx, int64_t layer, int64_t expert) { return ((PluginCtx*) ctx)->source->blob(layer, expert); }
// adapt_swaps through the plugin: ctx is the caller's blob function
const uint8_t* plugin_blob_fn_cb(void* ctx, int64_t layer, int64_t expert) {
    return (*(const std::function<const uint8_t*(int64_t, int64_t)>*) ctx)(layer, expert);
}
}  // namespace

void RemoteExperts::set_plugin(const std::string& dll, const std::string& pack_dir, int plugin_device) {
    HelperPlugin& p = helper_plugin();
    p.dll = dll; p.pack = pack_dir; p.device = plugin_device;
}
bool RemoteExperts::plugin_configured() { return !helper_plugin().dll.empty(); }
bool RemoteExperts::plugin_owns(int64_t index) const { return helper_plugin().owns(plugin_, index) != 0; }
bool RemoteExperts::plugin_holds(int64_t layer, int32_t expert) const { return helper_plugin().holds(plugin_, layer, expert) != 0; }
int64_t RemoteExperts::plugin_i(int k) const { int64_t c[5] = {}; double d[3] = {}; helper_plugin().stats(plugin_, c, d); return c[k]; }
double RemoteExperts::plugin_d(int k) const { int64_t c[5] = {}; double d[3] = {}; helper_plugin().stats(plugin_, c, d); return d[k]; }

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    if (plugin_configured()) {   // strata-mi50: the plugin runtime's device; `device` (this runtime's ordinal) is unused
        if (!load_helper_plugin(err)) return false;
        char e[512] = {};
        const HelperPlugin& p = helper_plugin();
        if (p.init(p.pack.c_str(), 0, 0, p.device, &free_gib, e, (int) sizeof e) != 0) { err = std::string("helper plugin: ") + e; return false; }
        return true;
    }
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    if (device == primary_device()) {   // the primary may be any visible ordinal; the helper cannot be it
        err = "CUDA" + std::to_string(device) + " experts: the helper cannot be the primary device "
                                                 "(STRATA_PRIMARY_DEVICE)";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
    if (!(spin && spin[0] == '0')) cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
#endif
    // HIP has no cudaInitDevice equivalent; retain its default scheduling policy.
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}

void RemoteExperts::close() {
    if (plugin_) { helper_plugin().close(plugin_); plugin_ = nullptr; return; }
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) cudaStreamSynchronize(stream_);
        if (read_stream_) cudaStreamSynchronize(read_stream_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scales_) cudaFree(d_scales_);
        if (d_xr_) cudaFree(d_xr_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (d_meta_) cudaFree(d_meta_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (stream_) cudaStreamDestroy(stream_);
        if (read_stream_) cudaStreamDestroy(read_stream_);
    }
    device_ = -1;
    stream_ = nullptr;
    read_stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_xr_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool auto_size) {
    if (plugin_configured()) {   // strata-mi50: the tier lives in the plugin (its own HIP runtime and device)
        close();
        if (!load_helper_plugin(err)) return false;
        const HelperPlugin& p = helper_plugin();
        char e[512] = {};
        double free_gib = 0;
        if (p.init(p.pack.c_str(), layers, experts, p.device, &free_gib, e, (int) sizeof e) != 0) { err = std::string("helper plugin: ") + e; return false; }
        std::vector<int32_t> flat;
        flat.reserve(ranked.size() * 2);
        for (const auto& pr : ranked) { flat.push_back(pr.first); flat.push_back(pr.second); }
        PluginCtx ctx{&primary, &source};
        // 2026-10-26: open2 with reduce when --remote-expert-opt attached an optimizer (was p.open, no reduction)
        plugin_ = p.open2(slots, flat.data(), (int64_t) ranked.size(), plugin_primary_cb, plugin_blob_cb, &ctx, claimed.data(),
                          auto_size ? 1 : 0, remote_opt_ ? 1 : 0, e, (int) sizeof e);
        if (plugin_ == nullptr) { err = std::string("helper plugin: ") + e; return false; }
        n_expert_ = experts;
        if (remote_opt_) {
            plugin_sum_.assign((size_t) strata::kernels::cpu::MAXT * H, 0.0f);
            plugin_owned_.assign((size_t) CAP, 0);
        }
        return true;
    }
    return open_with(device, slots, layers, experts, ranked,
                     [&](int32_t l, int32_t e) { return primary.slot_of(l, e) >= 0; },
                     [&](int64_t l, int64_t e) { return source.blob(l, e); }, claimed, err, auto_size, false);
}

bool RemoteExperts::open_with(int device, int slots, int64_t layers, int64_t experts,
                              const std::vector<std::pair<int32_t, int32_t>>& ranked,
                              const std::function<bool(int32_t, int32_t)>& on_primary,
                              const std::function<const uint8_t*(int64_t, int64_t)>& blob,
                              std::vector<uint8_t>& claimed, std::string& err, bool auto_size, bool standalone) {
    close();
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts ||
        (!standalone && device == primary_device())) {   // the helper cannot be the primary device (any visible ordinal)
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (!on_primary(pair.first, pair.second) && !claimed[index] && !picked[index]) {
            const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
            if (auto_size && needed + bytes + (512ull << 20) > free_bytes) break;
            selected.push_back(pair);
            needed += bytes;
            picked[index] = 1;
            if ((int) selected.size() >= slots) break;
        }
    }
    if (selected.empty()) {
        err = "CUDA" + std::to_string(device) + (auto_size ? " experts: no unclaimed expert fits with 512 MiB free"
                                                          : " experts: no unclaimed experts remain");
        close(); return false;
    }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                     : cache_.open((int64_t) selected.size(), layers, experts,
                                                   (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* bytes = blob(pair.first, pair.second);
        if (slot < 0 || !bytes || !cache_.fill_slot_blocking(slot, bytes, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const size_t meta_bytes = sizeof(RemoteMeta) + (remote_opt_ ? remote_opt_->metadata_bytes() : 0);
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device) &&
        check(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device) &&
        check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device) &&
        check(cudaHostAlloc(&h_meta_, meta_bytes, cudaHostAllocPortable), "metadata staging", err, device) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err, device) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err, device) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err, device) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err, device) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err, device) &&
        check(cudaMalloc(&d_meta_, meta_bytes), "group metadata", err, device) &&
        (!strata::kernels::cpu::hadamard_spec().any() ||
         check(cudaMalloc((void**) &d_xr_, (size_t) CAP * H * sizeof(float)), "rotated input", err, device));
    if (!allocated) { close(); return false; }
    if (!check(cudaStreamCreateWithFlags(&read_stream_, cudaStreamNonBlocking), "cache read stream", err, device)) {
        close();
        return false;
    }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) FF), (int) FF);
    }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0') &&
                 cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess;
    cudaGetLastError();
    auto* meta = (RemoteMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    return true;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err) {
    if (plugin_) {
        char e[512] = {};
        // 2026-10-26: --remote-expert-opt - the plugin reduces with the window's router weights (begin_opt)
        plugin_reduced_ = remote_opt_ && remote_opt_->active();
        const int rc = plugin_reduced_
            ? helper_plugin().begin_opt(plugin_, layer, x, ids, n_tok, k, kind, primary_res, remote_opt_->weights_, e, (int) sizeof e)
            : helper_plugin().begin(plugin_, layer, x, ids, n_tok, k, kind, primary_res, e, (int) sizeof e);
        if (rc != 0) { err = e; plugin_reduced_ = false; return false; }
        return true;
    }
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::kWindowMaxT || k != (int) strata::core::kCompiledGeometry.n_expert_used || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    const bool reduce = remote_opt_ && remote_opt_->active();
    if (reduce) remote_opt_->prepare(*this, meta + 1);
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    groups_ = (int32_t) group_id_.size();
    const cudaStream_t s = stream_;
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(RemoteMeta) + (reduce ? remote_opt_->metadata_bytes() : 0),
                              cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    if (!staged) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        const auto& fmt = lay.fmt[(size_t) layer];
        const float* xe = zero_copy_ ? z_x_ : d_x_;
        if (fmt.had_gu) {   // APR: Hadamard-folded gate/up take H (s * x)
            strata::kernels::hadamard_rows(xe, d_xr_, n_tok, H, fmt.had_block, d_sx_, s);
            xe = d_xr_;
        }
        strata::kernels::quantize_q8_1_rows(xe, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        if (fmt.had_d) { L.had_block = fmt.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2_window(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s,
                                               (int) n_tok);
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    if (reduce) {
        if (!remote_opt_->reduce(*this, (RemoteMeta*) d_meta_ + 1, err)) return false;
    } else if (!zero_copy_ && !check(cudaMemcpyAsync(h_out_, d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return false;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * H * sizeof(float);
    return true;
}

bool RemoteExperts::finish(float* out, std::string& err) {
    if (plugin_) {
        char e[512] = {};
        if (plugin_reduced_) {   // 2026-10-26: one summed row per token back, added into the optimizer's mapped sum
            plugin_reduced_ = false;
            if (helper_plugin().finish_opt(plugin_, plugin_sum_.data(), plugin_owned_.data(), e, (int) sizeof e) != 0) { err = e; return false; }
            remote_opt_->accumulate(*this);
            return true;
        }
        if (helper_plugin().finish(plugin_, out, e, (int) sizeof e) != 0) { err = e; return false; }
        return true;
    }
    if (group_id_.empty()) return true;
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    // Spin on the stream like PeerExperts::finish instead of the blocking sync: the layer waits for this helper
    // on the CPU pool's critical path, and a blocking sync's wake-up costs more than the helper's small batch
    // takes - on HIP especially, where the CUDA branch's cudaInitDevice(ScheduleSpin) does not exist and the
    // device keeps its default policy.  STRATA_REMOTE_SPIN=0 keeps the blocking sync (the same switch the CUDA
    // branch reads).
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)   // HIP only (2026-10-23: also the gfx906 plugin build, previously only STRATA_USE_HIP): the CUDA branch's cudaInitDevice(ScheduleSpin) already makes the sync spin
    static const bool spin = [] {
        const char* v = std::getenv("STRATA_REMOTE_SPIN");
        return !(v != nullptr && v[0] == '0');
    }();
    if (spin) {
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!check(e, "finish", err, device_)) return false;
    } else if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) {
        return false;
    }
#else
    if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) return false;
#endif
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    if (remote_opt_ && remote_opt_->active()) { remote_opt_->accumulate(*this); return true; }
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}

// strata-mi50 (2026-10-26): the swap rule of RemoteExpertOpt::adapt (src/core/remote_expert_opt.cu) for this one tier, run where the
// tier's cache lives: in the plugin's runtime for the MI50.  The in-process tiers keep the original asynchronous path in adapt();
// here every fill is blocking, so a blob pointer from `blob` is never used after the next call (the callback crosses runtimes).
bool RemoteExperts::adapt_swaps(const float* usage, const int32_t* primary, uint8_t* resident, int64_t n, int max_swaps,
                                const std::function<const uint8_t*(int64_t, int64_t)>& blob, int& swapped, std::string& err) {
    swapped = 0;
    if (plugin_) {
        char e[512] = {};
        if (helper_plugin().adapt(plugin_, usage, primary, resident, n, max_swaps, plugin_blob_fn_cb, (void*) &blob, &swapped,
                                  e, (int) sizeof e) != 0) { err = e; return false; }
        return true;
    }
    if (max_swaps <= 0 || device_ < 0 || n != (int64_t) layers_present_.size() * n_expert_) return true;
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> candidates, victims;
    for (int32_t l = 0; l < (int32_t) layers_present_.size(); ++l) {
        candidates.clear(); victims.clear();
        const size_t base = (size_t) l * n_expert_;
        float primary_cold = 1.0e30f;
        for (int32_t e = 0; e < n_expert_; ++e) {
            const float u = usage[base + e];
            if (primary[base + e] >= 0) primary_cold = std::min(primary_cold, u);
            if (cache_.slot_of(l, e) >= 0) victims.emplace_back(u, e);
            else if (!resident[base + e] && u >= 2.0f) candidates.emplace_back(u, e);
        }
        const size_t m = std::min(candidates.size(), victims.size());
        if (!m) continue;
        std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::partial_sort(victims.begin(), victims.begin() + m, victims.end(), [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < m; ++i) {
            if (victims[i].first > primary_cold) break;
            if (candidates[i].first < victims[i].first + 1.5f) break;
            swaps.push_back({candidates[i].first - victims[i].first, l, candidates[i].second, victims[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    if (swaps.empty()) return true;
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    if (!check(cudaStreamSynchronize(stream_), "adapt", err, device_)) return false;   // nothing of this tier still running
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (const auto& s : swaps) {
        const uint8_t* in_blob = blob(s.layer, s.in);
        if (in_blob == nullptr) continue;
        if (!cache_.fill_slot_blocking(cache_.slot_of(s.layer, s.out), in_blob, err, (int64_t) lay.blob_bytes(s.layer))) {
            err = "CUDA" + std::to_string(device_) + " experts: adapt: " + err;
            return false;
        }
        cache_.replace(s.layer, s.out, s.in);
        const size_t base = (size_t) s.layer * n_expert_;
        resident[base + s.out] = 0;
        resident[base + s.in] = 1;
        ++swapped;
    }
    return true;
}

bool RemoteExperts::prefill_supported() const {
    const HelperPlugin& p = helper_plugin();
    return plugin_ != nullptr && p.pf_reserve && p.pf_begin && p.pf_finish;
}

bool RemoteExperts::prefill_reserve(int64_t t_max, std::vector<uint8_t>& layer_ok, uint16_t** x16, uint16_t** sum16, std::string& err) {
    if (!prefill_supported()) { err = "the helper plugin has no prefill entry points (ABI < 5)"; return false; }
    char e[512] = {};
    if (helper_plugin().pf_reserve(plugin_, t_max, layer_ok.data(), (int64_t) layer_ok.size(), x16, sum16, e, (int) sizeof e) != 0) { err = e; return false; }
    return true;
}

bool RemoteExperts::prefill_begin(int64_t layer, int64_t t, int64_t k, const uint16_t* x16, const int32_t* ids, const float* w,
                                  const uint8_t* take, std::string& err) {
    char e[512] = {};
    if (helper_plugin().pf_begin(plugin_, layer, t, k, x16, ids, w, take, e, (int) sizeof e) != 0) { err = e; return false; }
    return true;
}

bool RemoteExperts::prefill_finish(uint16_t* sum16, double* ms_wait, std::string& err) {
    char e[512] = {};
    if (helper_plugin().pf_finish(plugin_, sum16, ms_wait, e, (int) sizeof e) != 0) { err = e; return false; }
    return true;
}

bool RemoteExperts::copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                      size_t n, uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (plugin_ && n > 0) {   // 2026-10-27: the plugin copies from the helper's VRAM (ABI 4); was: every blob a miss, read from the files
        char e[512] = {};
        if (helper_plugin().copy_blobs(plugin_, layers, experts, dst, (int64_t) n, hit, e, (int) sizeof e) != 0) { err = e; return false; }
        return true;
    }
    if (plugin_ || device_ < 0 || read_stream_ == nullptr || n == 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    std::lock_guard<std::mutex> lock(read_mu_);
    bool queued = false;
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        const int32_t slot = cache_.slot_of(layers[i], experts[i]);
        if (slot < 0) continue;
        const size_t bytes = (size_t) lay.blob_bytes(layers[i]);
        if (!check(cudaMemcpyAsync(dst[i], cache_.device_slot(slot), bytes, cudaMemcpyDeviceToHost, read_stream_),
                   "cache read", err, device_)) {
            // Nothing may still be writing a caller's buffer when this returns.  If the drain succeeds, the
            // hits queued before the failure are complete and the caller may keep them; if it does not,
            // forget every hit and let the caller refill all of them from its fallback.
            if (cudaStreamSynchronize(read_stream_) != cudaSuccess) std::fill(hit, hit + n, 0);
            return false;
        }
        hit[i] = 1;
        queued = true;
    }
    if (queued && !check(cudaStreamSynchronize(read_stream_), "cache read wait", err, device_)) {
        std::fill(hit, hit + n, 0);   // a failed wait means no copied buffer may be trusted
        return false;
    }
    return true;
}

bool RemotePrefillSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (dst != nullptr && from_helper(layer, expert)) {
        const int32_t l = (int32_t) layer, e = (int32_t) expert;
        uint8_t* d = dst;
        std::string err;
        for (RemoteExperts* h : helpers_) {
            uint8_t got = 0;
            const bool ok = h->copy_cached_blobs(&l, &e, &d, 1, &got, err);
            if (got) {
                helper_blobs_.fetch_add(1, std::memory_order_relaxed);
                helper_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer),
                                        std::memory_order_relaxed);
                return true;
            }
            if (!ok) break;
        }
        warn(err);
    }
    return fallback_.copy_blob(layer, expert, dst);
}

bool RemotePrefillSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    if (helpers_.empty() || n == 0) return fallback_.copy_blobs(layers, experts, dst, n);
    thread_local std::vector<int32_t> hl, he;
    thread_local std::vector<uint8_t*> hd;
    thread_local std::vector<size_t> idx;
    thread_local std::vector<uint8_t> got;
    thread_local std::vector<uint8_t> served;
    thread_local std::vector<int32_t> ml, me;
    thread_local std::vector<uint8_t*> md;
    thread_local std::vector<int> which;
    which.assign(n, -1);
    served.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        for (size_t k = 0; k < helpers_.size(); ++k)
            if (helpers_[k]->holds(layers[i], experts[i])) { which[i] = (int) k; break; }
    }
    // One stream batch per helper, so several hits pay one wait; the misses go to the file source in one
    // call, which keeps its own batched-read merging for them.
    uint64_t blobs = 0, bytes = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t k = 0; k < helpers_.size(); ++k) {
        hl.clear(); he.clear(); hd.clear(); idx.clear();
        for (size_t i = 0; i < n; ++i)
            if (which[i] == (int) k) {
                hl.push_back(layers[i]); he.push_back(experts[i]); hd.push_back(dst[i]); idx.push_back(i);
            }
        if (hl.empty()) continue;
        got.assign(hl.size(), 0);
        std::string err;
        if (!helpers_[k]->copy_cached_blobs(hl.data(), he.data(), hd.data(), hl.size(), got.data(), err)) warn(err);
        for (size_t j = 0; j < got.size(); ++j)
            if (got[j]) {
                served[idx[j]] = 1;
                ++blobs;
                bytes += lay.blob_bytes(hl[j]);
            }
    }
    if (blobs != 0) {
        helper_blobs_.fetch_add(blobs, std::memory_order_relaxed);
        helper_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    ml.clear(); me.clear(); md.clear();
    for (size_t i = 0; i < n; ++i)
        if (!served[i]) { ml.push_back(layers[i]); me.push_back(experts[i]); md.push_back(dst[i]); }
    if (ml.empty()) return true;
    return fallback_.copy_blobs(ml.data(), me.data(), md.data(), ml.size());
}

void RemotePrefillSource::warn(const std::string& err) {
    if (err.empty() || warned_.exchange(true)) return;
    std::fprintf(stderr, "prefill: a helper cache read failed (%s); the file source fills that blob instead\n",
                 err.c_str());
}
} // namespace strata::core
