#include "strata/core/remote_experts.hpp"
#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/core/remote_expert_opt.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {
namespace {
// The window's entry count.  It follows the model's routing width k (Flash-Next's 10, Qwen3.6's 8);
// allocate the arrays for the widest k the engine admits - MAXT * 10 = 80 entries - so both fit.
constexpr int64_t kMaxCap = strata::kernels::cpu::MAXT * 10;
constexpr int kMaxRings = 3;

// Small enough to copy as one pinned buffer per layer, whatever the model's shape. The kernels read
// the individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[kMaxCap];
    int32_t start[kMaxCap + 1];
    int32_t dst[kMaxCap];
    int32_t tok[kMaxCap];
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
    /// Destruct the scope NOW instead of at the end of its enclosing block, running the restoring
    /// cudaSetDevice(previous) at this exact point.  STRATA_Q36_HELPER_TIMING=1 measures that call as part of a
    /// drain's host overhead, and that window must not also enclose the blocking sync that follows.
    void release() { if (previous >= 0) { cudaSetDevice(previous); previous = -1; } }
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

/// STRATA_Q36_HELPER_RING=<K>: integer-parsed; unset or 0 => off (legacy K=1). Cap at kMaxRings.
int parse_helper_ring() {
    const char* r = std::getenv("STRATA_Q36_HELPER_RING");
    if (!r || !r[0]) return 1;
    char* end = nullptr;
    const long v = std::strtol(r, &end, 10);
    if (end == r || v <= 0) return 1;   // =0 must DISABLE (not presence-based)
    return (int) std::min<long>(v, kMaxRings);
}
} // namespace

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
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

bool RemoteExperts::alloc_win(Win& w, size_t meta_bytes, size_t scratch, std::string& err) {
    const bool allocated =
        check(cudaHostAlloc((void**) &w.h_x_, (size_t) cap_ * n_embd_ * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device_) &&
        check(cudaHostAlloc((void**) &w.h_out_, (size_t) cap_ * n_embd_ * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device_) &&
        check(cudaHostAlloc(&w.h_meta_, meta_bytes, cudaHostAllocPortable), "metadata staging", err, device_) &&
        check(cudaMalloc((void**) &w.d_x_, (size_t) cap_ * n_embd_ * sizeof(float)), "input", err, device_) &&
        check(cudaMalloc((void**) &w.d_out_, (size_t) cap_ * n_embd_ * sizeof(float)), "result", err, device_) &&
        check(cudaMalloc((void**) &w.d_q8_, (size_t) cap_ * (n_embd_ / 32) * 36), "activation", err, device_) &&
        check(cudaMalloc((void**) &w.d_scales_, (size_t) cap_ * (n_embd_ / 32) * sizeof(float)), "activation scales", err, device_) &&
        check(cudaMalloc(&w.d_scratch_, scratch), "scratch", err, device_) &&
        check(cudaMalloc(&w.d_meta_, meta_bytes), "group metadata", err, device_);
    if (!allocated) return false;
    if (rings_ > 1) {
        if (!check(cudaMalloc((void**) &w.d_src_row_, (size_t) cap_ * sizeof(int32_t)), "src_row", err, device_))
            return false;
        if (!check(cudaEventCreateWithFlags(&w.ev_done_, cudaEventDisableTiming), "ring event", err, device_))
            return false;
    }
    if (zero_copy_) {
        if (cudaHostGetDevicePointer((void**) &w.z_x_, w.h_x_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &w.z_out_, w.h_out_, 0) != cudaSuccess) {
            zero_copy_ = false;
            w.z_x_ = w.z_out_ = nullptr;
            cudaGetLastError();
        }
    }
    auto* meta = (RemoteMeta*) w.d_meta_;
    w.d_ptr_ = meta->ptr;
    w.d_start_ = meta->start;
    w.d_dst_ = meta->dst;
    w.d_tok_ = meta->tok;
    w.d_count_ = &meta->count;
    w.owned_.resize(cap_);
    w.group_of_.resize(cap_);
    w.group_id_.reserve(cap_);
    w.ptr_.reserve(cap_);
    w.start_.reserve(cap_ + 1);
    w.dst_.reserve(cap_);
    w.tok_.reserve(cap_);
    w.original_row_.reserve(cap_);
    w.src_row_.assign((size_t) cap_, -1);
    return true;
}

void RemoteExperts::free_win(Win& w) {
    if (w.ev_done_) { cudaEventDestroy(w.ev_done_); w.ev_done_ = nullptr; }
    if (w.d_src_row_) cudaFree(w.d_src_row_);
    if (w.d_x_) cudaFree(w.d_x_);
    if (w.d_out_) cudaFree(w.d_out_);
    if (w.d_q8_) cudaFree(w.d_q8_);
    if (w.d_scales_) cudaFree(w.d_scales_);
    if (w.d_scratch_) cudaFree(w.d_scratch_);
    if (w.d_meta_) cudaFree(w.d_meta_);
    if (w.h_x_) cudaFreeHost(w.h_x_);
    if (w.h_out_) cudaFreeHost(w.h_out_);
    if (w.h_meta_) cudaFreeHost(w.h_meta_);
    w = Win{};
}

void RemoteExperts::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        drain_all();
        if (stream_) cudaStreamSynchronize(stream_);
        if (read_stream_) cudaStreamSynchronize(read_stream_);
        cache_.close();
        for (Win& w : win_) free_win(w);
        win_.clear();
        if (d_xr_) cudaFree(d_xr_);   // APR: the rotated-input staging is shared by the rings, so it is not in free_win
        if (stream_) cudaStreamDestroy(stream_);
        if (read_stream_) cudaStreamDestroy(read_stream_);
    }
    device_ = -1;
    rings_ = 1;
    rr_ = 0;
    stream_ = nullptr;
    read_stream_ = nullptr;
    zero_copy_ = false;
    d_xr_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool auto_size,
                         int64_t n_embd, int64_t n_ff, int64_t k,
                         const std::vector<strata::kernels::NativeExpertLayout>* layer_layouts) {
    close();
    if (n_embd <= 0 || n_ff <= 0 || k <= 0 || k > 10 || strata::kernels::cpu::MAXT * k > kMaxCap) {
        err = "experts: the model's row width, expert width and routing width are out of range";
        return false;
    }
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    k_ = k;
    cap_ = strata::kernels::cpu::MAXT * k_;
    layer_layouts_.clear();
    if (layer_layouts) layer_layouts_ = *layer_layouts;
    rings_ = parse_helper_ring();
    rr_ = 0;
    if (rings_ > 1 && remote_opt_ != nullptr) {
        err = "CUDA" + std::to_string(device) +
              " experts: STRATA_Q36_HELPER_RING>1 cannot be used with remote_opt reduce (single-window)";
        rings_ = 1;
        return false;
    }
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts ||
        device == primary_device()) {   // the helper cannot be the primary device (any visible ordinal)
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    // Whether the caller gave a step for EVERY layer.  No table at all, a stride of 0, or a table that does not
    // cover the layers means the engine's step is unknown - Flash-Next's native packs pass no table, q36's helpers
    // always pass one with every stride set.  When it is unknown the cache keeps its own 256-BYTE ALIGNMENT, the
    // layout it has always laid its slots out with: a slot narrower than that is not where the kernels' vector
    // loads expect it, and the VRAM budget would change for every existing configuration.
    step_known_ = lay.native && !layer_layouts_.empty() &&
                  std::all_of(layer_layouts_.begin(), layer_layouts_.end(),
                              [](const strata::kernels::NativeExpertLayout& l) { return l.stride > 0; });
    // ONE answer to "how wide is this layer's slot", used by the VRAM budget and by open_sized alike.  Two
    // separate expressions here drifted apart once already (the budget rounded to 256 and the sizes did not).
    const auto slot_bytes_at = [&](int64_t layer) -> uint64_t {
        const uint64_t blob = lay.blob_bytes(layer);
        if (!step_known_) return (blob + 255) / 256 * 256;   // the cache's own alignment, unchanged
        const uint64_t step = layer_layouts_[(size_t) layer].stride;
        return step > blob ? step : blob;
    };
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            // The bytes this expert takes in the cache: the layout's step when the caller gave one (the slot is
            // that wide so the next expert starts where MMQ expects it), else the cache's own 256-byte alignment.
            // The number here must be the one open_sized lays the slot out with, or the VRAM budget is wrong.
            const uint64_t bytes = lay.native ? slot_bytes_at(pair.first) : lay.max_blob;
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
        for (const auto& pair : selected) sizes.push_back((int64_t) slot_bytes_at(pair.first));
        // The slots are laid out with that step VERBATIM when the caller gave one: the cache's own 256-byte
        // alignment would move it (a Q6_K layer's 2041200 becomes 2041344) and the MMQ, which steps over the
        // experts in whole blocks, would then read every expert but the first from the wrong offset - NaN rows
        // rather than an error.  Without a step from the caller (every caller but q36's helpers) the cache's own
        // 256-byte alignment stays, so those caches keep the layout they had.
        cache_.set_slot_align(step_known_ ? 0 : 256);
    } else {
        cache_.set_slot_align(256);   // the default: blobs where the kernels' vector loads expect them
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
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(cap_, n_embd_, n_ff_),
        strata::kernels::native_expert_scratch_bytes(cap_, n_ff_));
    // remote_opt metadata only when rings==1 (reduce assumes a single window).
    const size_t meta_bytes = sizeof(RemoteMeta) + (remote_opt_ && rings_ == 1 ? remote_opt_->metadata_bytes() : 0);
    if (!check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device)) {
        close(); return false;
    }
    if (strata::kernels::cpu::hadamard_spec().any() &&
        !check(cudaMalloc((void**) &d_xr_, (size_t) kMaxCap * n_embd_ * sizeof(float)), "rotated input", err, device)) {
        close(); return false;   // APR: the rotated-input staging is shared by the rings, allocated once here
    }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) n_embd_), (int) n_embd_);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) n_ff_), (int) n_ff_);
    }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    // Mapping is attempted per-slot in alloc_win; any failure clears zero_copy_ below.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0');
    win_.assign((size_t) rings_, Win{});
    for (Win& w : win_) {
        if (!alloc_win(w, meta_bytes, scratch, err)) { close(); return false; }
    }
    // If any slot failed to map, fall back every slot to the copy path.
    if (zero_copy_) {
        for (Win& w : win_)
            if (!w.z_x_ || !w.z_out_) { zero_copy_ = false; break; }
        if (!zero_copy_)
            for (Win& w : win_) w.z_x_ = w.z_out_ = nullptr;
    }
    if (!check(cudaStreamCreateWithFlags(&read_stream_, cudaStreamNonBlocking), "cache read stream", err, device)) {
        close();
        return false;
    }
    layers_present_.assign((size_t) layers, 0);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    return true;
}

bool RemoteExperts::owns(int slot, int64_t index) const {
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size()) return false;
    const Win& w = win_[(size_t) slot];
    if (index < 0 || index >= w.n_ || (size_t) index >= w.owned_.size()) return false;
    return w.owned_[(size_t) index] != 0;
}

cudaEvent_t RemoteExperts::ev_done(int slot) const {
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size()) return nullptr;
    return win_[(size_t) slot].ev_done_;
}

float* RemoteExperts::h_out(int slot) const {
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size()) return nullptr;
    return win_[(size_t) slot].h_out_;
}

const int32_t* RemoteExperts::src_row(int slot) const {
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size()) return nullptr;
    return win_[(size_t) slot].src_row_.data();
}

int64_t RemoteExperts::compact_n(int slot) const {
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size()) return 0;
    return (int64_t) win_[(size_t) slot].original_row_.size();
}

int64_t RemoteExperts::window_n(int slot) const {
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size()) return 0;
    return win_[(size_t) slot].n_;
}

int RemoteExperts::begin_win(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                             int64_t k, const int32_t* kind, const int32_t* primary_res,
                             std::string& err) {
    // Ping-pong: next slot; if still busy, drain it (rings==1 always slot 0, never creates events).
    const int slot = rr_;
    rr_ = (rr_ + 1) % rings_;
    return begin_slot(slot, layer, x, ids, n_tok, k, kind, primary_res, err);
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err) {
    // Flash-Next / legacy: always slot 0 so finish()/owns() stay single-window.
    const int s = begin_slot(0, layer, x, ids, n_tok, k, kind, primary_res, err);
    if (s == -1) return true;
    if (s < 0) return false;
    return true;
}

int RemoteExperts::begin_slot(int slot, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                              int64_t k, const int32_t* kind, const int32_t* primary_res,
                              std::string& err) {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size() || n <= 0 || n > cap_ ||
        n_tok > strata::kernels::cpu::MAXT || k != k_ || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return -2;
    }
    if (win_[(size_t) slot].busy_) {
        if (!drain(slot, err)) return -2;
    }
    Win& w = win_[(size_t) slot];
    w.n_ = n;
    std::fill(w.owned_.begin(), w.owned_.begin() + n, 0);
    w.group_id_.clear(); w.ptr_.clear(); w.start_.clear(); w.dst_.clear(); w.tok_.clear(); w.original_row_.clear();
    std::fill(w.src_row_.begin(), w.src_row_.begin() + n, -1);
    if (!layers_present_[(size_t) layer]) return -1;
    std::fill(w.group_of_.begin(), w.group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t cache_slot = cache_.slot_of(layer, e);
        if (cache_slot < 0) continue;
        w.owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < w.group_id_.size(); ++g)
            if (w.group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) w.group_id_.size();
            w.group_id_.push_back(e);
            w.ptr_.push_back((unsigned long long) cache_.device_slot(cache_slot));
        }
        w.group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (w.group_id_.empty()) return -1;
    for (size_t g = 0; g < w.group_id_.size(); ++g) {
        w.start_.push_back((int32_t) w.dst_.size());
        for (int64_t i = 0; i < n; ++i) if (w.group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            const int32_t compact = (int32_t) w.dst_.size();
            w.original_row_.push_back((int32_t) i);
            w.src_row_[(size_t) i] = compact;
            w.dst_.push_back(compact);
            w.tok_.push_back((int32_t) (i / k));
        }
    }
    w.start_.push_back((int32_t) w.dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(w.h_x_, x, (size_t) n_tok * n_embd_ * sizeof(float));
    auto* meta = (RemoteMeta*) w.h_meta_;
    std::memcpy(meta->ptr, w.ptr_.data(), w.ptr_.size() * sizeof(w.ptr_[0]));
    std::memcpy(meta->start, w.start_.data(), w.start_.size() * sizeof(w.start_[0]));
    std::memcpy(meta->dst, w.dst_.data(), w.dst_.size() * sizeof(w.dst_[0]));
    std::memcpy(meta->tok, w.tok_.data(), w.tok_.size() * sizeof(w.tok_[0]));
    meta->count = (int32_t) w.group_id_.size();
    const bool reduce = remote_opt_ && remote_opt_->active() && rings_ == 1;
    if (reduce) remote_opt_->prepare(*this, meta + 1);
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return -2; }
    w.groups_ = (int32_t) w.group_id_.size();
    const cudaStream_t s = stream_;
    bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(w.d_x_, w.h_x_, (size_t) n_tok * n_embd_ * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(w.d_meta_, w.h_meta_, sizeof(RemoteMeta) + (reduce ? remote_opt_->metadata_bytes() : 0),
                              cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    // Ring device-merge needs helper-side src_row; rings==1 finish uses host scatter only.
    if (staged && rings_ > 1)
        staged = check(cudaMemcpyAsync(w.d_src_row_, w.src_row_.data(), (size_t) n * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, s), "copy src_row", err, device_);
    if (!staged) return -2;
    const auto& lay = strata::kernels::cpu::expert_layout();
    // Per-layer layouts (given to open() by an engine that never loads the global expert_layout table,
    // e.g. qwen35moe, whose own load() already computed one per layer) win over that table.  A layer past the
    // end of either fails rather than reading the wrong entry - an empty layout would silently mis-decode.
    // With neither, the canonical Q2_0 s2 path runs exactly as before.
    strata::kernels::NativeExpertLayout L;
    bool use_native = false;
    if (!layer_layouts_.empty()) {          // ours: per-layer layouts win over the global table
        if ((size_t) layer >= layer_layouts_.size()) {
            err = "CUDA" + std::to_string(device_) + " experts: no per-layer layout for layer " +
                  std::to_string(layer) + " (open() was given " + std::to_string(layer_layouts_.size()) +
                  ")";
            return -2;
        }
        L = layer_layouts_[(size_t) layer];
        use_native = true;
    } else if (lay.native) {                 // ours: global table, with the bounds check upstream lacked
        if ((size_t) layer >= lay.fmt.size()) {
            err = "CUDA" + std::to_string(device_) + " experts: the expert layout has no entry for layer " +
                  std::to_string(layer) + " (it has " + std::to_string(lay.fmt.size()) + ")";
            return -2;
        }
        const auto& fmt = lay.fmt[(size_t) layer];
        L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        use_native = true;
    }
    if (use_native) {
        // The gate/up activation is Hadamard-folded in an APR pack: quantize H (s * x), never x.  Both parts
        // of this are upstream's - on our path fmt.had_gu is always false (q36 loads no pack, so
        // expert_layout_set_native leaves had_gu/had_d false) and the branch costs one compare.
        const bool have_fmt = (size_t) layer < lay.fmt.size();
        const auto& fmt = have_fmt ? lay.fmt[(size_t) layer] : strata::kernels::cpu::NativeFmt{};
        const float* xe = zero_copy_ ? w.z_x_ : w.d_x_;
        if (fmt.had_gu) {   // APR: Hadamard-folded gate/up take the full row width (s * x)
            strata::kernels::hadamard_rows(xe, d_xr_, n_tok, n_embd_, fmt.had_block, d_sx_, s);
            xe = d_xr_;
        }
        strata::kernels::quantize_q8_1_rows(xe, n_tok, n_embd_, w.d_q8_, s);
        // The down weights are folded, so native_expert_grouped rotates h itself (iq_kernels.hpp:56-59).
        // Upstream set this only on its own path; on the layer_layouts_ path the flags come from the global
        // table too, or the kernel would decode a folded down matrix as if it were unfolded.
        if (have_fmt && fmt.had_d) { L.had_block = fmt.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, w.d_ptr_, w.d_start_, w.d_count_, w.d_dst_, w.d_tok_,
                                               w.groups_, (int64_t) w.dst_.size(), w.d_q8_, w.d_scratch_,
                                               zero_copy_ && !reduce ? w.z_out_ : w.d_out_, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? w.z_x_ : w.d_x_, w.d_q8_, w.d_scales_, n_tok * n_embd_, s);
        strata::kernels::moe_grouped_s2(w.d_ptr_, w.d_start_, w.d_count_, w.d_dst_, w.d_tok_,
                                        w.groups_, (int64_t) w.dst_.size(), w.d_q8_, w.d_scales_, w.d_scratch_,
                                        zero_copy_ && !reduce ? w.z_out_ : w.d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : w.dst_.size()) * n_embd_ * sizeof(float);
    if (reduce) {
        if (!remote_opt_->reduce(*this, (RemoteMeta*) w.d_meta_ + 1, err)) return -2;
    } else if (!zero_copy_ && !check(cudaMemcpyAsync(w.h_out_, w.d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return -2;
    // rings>1: record completion after the last kernel (+ optional D2H). rings==1: no events (byte-identical).
    if (rings_ > 1) {
        if (!check(cudaEventRecord(w.ev_done_, s), "ring event record", err, device_)) return -2;
    }
    w.busy_ = true;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * n_embd_ * sizeof(float);
    return slot;
}

bool RemoteExperts::drain(int slot, std::string& err) {
    if (slot < 0 || slot >= rings_ || (size_t) slot >= win_.size()) {
        err = "CUDA" + std::to_string(device_) + " experts: drain slot out of range";
        return false;
    }
    Win& w = win_[(size_t) slot];
    if (!w.busy_) return true;
    if (w.group_id_.empty()) { w.busy_ = false; return true; }
    // STRATA_Q36_HELPER_TIMING=1 only: the host-side prologue+epilogue drain spends around the blocking wait -
    // DeviceScope's three runtime calls (cudaGetDevice + 2x cudaSetDevice), the branches above and the two
    // steady_clock reads.  ms_wait_ times the blocking call alone and cannot see this, so the two together say
    // how much of a drain is driver overhead and how much is really GPU wait.  The gate is exactly "1", the same
    // test Engine::helper_timing() applies (this file cannot call it: it is private and in another
    // translation unit), so the two counters arm together.  A flag-off run keeps a nullopt and pays no clock read
    // beyond the one ms_wait_ already took.
    const char* ht = std::getenv("STRATA_Q36_HELPER_TIMING");
    const bool hton = ht && ht[0] == '1' && ht[1] == '\0';
    std::optional<std::chrono::steady_clock::time_point> oh0;
    if (hton) oh0 = std::chrono::steady_clock::now();
    // The overhead window closes BEFORE the blocking sync (so it times host cost only) and its destructor -
    // the epilogue's cudaSetDevice(previous), the third of DeviceScope's three runtime calls - runs inside it.
    // A scope brace alone cannot do this: it would also enclose the blocking call below, making the two counters
    // equal by construction.  Hence the explicit release(): DeviceScope is destructed here, and nothing that
    // follows is inside the window.
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    scope.release();
    if (hton) {
        ms_sync_overhead_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *oh0).count();
        ++drains_timed_;
    }
    {
    const auto w0 = std::chrono::steady_clock::now();
    if (rings_ > 1) {
        if (!check(cudaEventSynchronize(w.ev_done_), "drain", err, device_)) return false;
    } else {
        // Spin on the stream like PeerExperts::finish instead of the blocking sync: the layer waits for this helper
        // on the CPU pool's critical path, and a blocking sync's wake-up costs more than the helper's small batch
        // takes - on HIP especially, where the CUDA branch's cudaInitDevice(ScheduleSpin) does not exist and the
        // device keeps its default policy.  STRATA_REMOTE_SPIN=0 keeps the blocking sync (the same switch the CUDA
        // branch reads).
#if defined(STRATA_USE_HIP)   // HIP only: the CUDA branch's cudaInitDevice(ScheduleSpin) already makes the sync spin
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
    }
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    w.busy_ = false;
    }
    return true;
}

void RemoteExperts::drain_all() {
    if (device_ < 0 || win_.empty()) return;
    std::string err;
    for (int s = 0; s < rings_; ++s) (void) drain(s, err);
}

bool RemoteExperts::finish(float* out, std::string& err) {
    // Legacy wrapper: drain slot 0, then host-scatter / remote_opt accumulate (rings==1 order identical).
    if (win_.empty() || win_[0].group_id_.empty()) return true;
    if (!drain(0, err)) return false;
    Win& w = win_[0];
    if (remote_opt_ && remote_opt_->active()) { remote_opt_->accumulate(*this); return true; }
    for (size_t i = 0; i < w.original_row_.size(); ++i)
        std::memcpy(out + (size_t) w.original_row_[i] * n_embd_, w.h_out_ + i * n_embd_, (size_t) n_embd_ * sizeof(float));
    return true;
}

bool RemoteExperts::copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                      size_t n, uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (device_ < 0 || read_stream_ == nullptr || n == 0) return true;
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
