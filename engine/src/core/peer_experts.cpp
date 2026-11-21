// src/core/peer_experts.cpp - see include/strata/core/peer_experts.hpp.
#include "strata/core/peer_experts.hpp"

#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/core/flag_order.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace strata::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }

namespace {

constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::kWindowMaxT * 10;   // entries per layer: a window's rows x top-10
constexpr int kMetaSlots = 4;                                 // host staging ring (>= max in-flight G=2 groups)

struct Meta {                           // one block, uploaded with one copy per layer
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];                   // [0] groups, [1] entries
    int32_t out_row[CAP];               // multi-GPU: compact row -> entry row in the host `out`
};

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// #2666: `cudaDeviceCanAccessPeer` said yes and `cudaDeviceEnablePeerAccess` succeeded, and on a pair of GPUs behind a host bridge
/// (2x RTX A4000, "PHB" in nvidia-smi topo) `cudaMemcpyPeerAsync` then returned success having moved nothing: the peer's input rows
/// were zeros and the rows it sent back never arrived, so the first prompt's MoE output was garbage and the logits NaN.  Without
/// peer access enabled the same copies are right (the driver stages them).  So the pair is asked once, with a small copy each way,
/// whether the direct path really carries data.  true = both ways exact (or nothing could be tested: the old trust is kept).
bool p2p_copies_data(int device) {
    constexpr size_t kN = 1 << 14;   // floats
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    const int dev[2] = {0, device};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        On on(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess) { cudaGetLastError(); tested = false; }
    }
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {   // 0: device 0 -> peer, 1: peer -> device 0
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        {
            On on(dev[from]);
            if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        }
        {
            On on(dev[to]);
            if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
        }
        if (cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN * 4) != cudaSuccess) { tested = false; break; }
        {
            On on(dev[to]);
            if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
    }
    for (int k = 0; k < 2; ++k)
        if (d[k] != nullptr) { On on(dev[k]); cudaFree(d[k]); }
    if (!tested) { cudaGetLastError(); return true; }
    return ok;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    if (device_ < 0) return;
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_meta_) cudaFree(d_meta_);
        if (d_ident_) cudaFree(d_ident_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (d_gate_vals_) cudaFree(d_gate_vals_);
        if (h_gate_) cudaFreeHost((void*) h_gate_);
        if (d_q8_win_) cudaFree(d_q8_win_);
        if (d_flag_cnt_) cudaFree(d_flag_cnt_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_ident_ = nullptr;
    meta_slot_ = 0;
    d_q8_ = d_q8_win_ = nullptr;
    q8_win_stride_ = 0;
    q8_win_steps_ = 0;
    d_flag_cnt_ = h_sync_flag_ = d_sync_flag_ = nullptr;
    vram_sync_flag_ = false;
    async_in_flight_ = launched_async_ = false;
    d_sx_ = d_sh_ = nullptr;
    d_gate_vals_ = nullptr;
    h_gate_ = nullptr;
    stream_ = refill_ = nullptr;
    refill_ev_ = nullptr;
    device_ = -1;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err) {
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 1 || device >= count) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
              " devices; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, primary_device(), device);
        cudaDeviceCanAccessPeer(&b, device, primary_device());
        if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = 0;   // test knob: pretend the pair has no P2P
        p2p_ = a && b;
        if (p2p_) {
            On on0(primary_device());
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(primary_device(), 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
            // recommend, never force: the direct path is used only if a test copy each way arrives exact (#2666)
            if (p2p_ && !p2p_copies_data(device)) {
                std::fprintf(stderr, "strata: peer GPU %d: peer access is enabled but a test copy between the two GPUs arrived wrong "
                                     "(a host bridge between them, no real P2P): the prompt path takes the host route instead "
                                     "(STRATA_PEER_NO_P2P=1 does that by hand)\n", device);
                { On q0(0); cudaDeviceDisablePeerAccess(device); cudaGetLastError(); }
                { On q1(device); cudaDeviceDisablePeerAccess(0); cudaGetLastError(); }
                p2p_ = false;
            }
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    int64_t ff = strata::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaHostAlloc(&h_meta_, sizeof(Meta) * kMetaSlots, cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
        ck(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc(&d_meta_, std::max<size_t>(sizeof(Meta), strata::kernels::kPeerPutMetaMax)), "plan", err) &&
        ck(cudaMalloc((void**) &d_ident_, (size_t) CAP * sizeof(int32_t)), "ident", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err) &&
        ck(cudaMalloc((void**) &d_flag_cnt_, 64), "flag counter", err) &&
        ck(cudaMemset(d_flag_cnt_, 0, 64), "flag counter zero", err);
    if (!alloc_ok) { close(); return false; }
    {
        int32_t ident_h[CAP];
        for (int64_t i = 0; i < CAP; ++i) ident_h[i] = (int32_t) i;
        if (!ck(cudaMemcpy(d_ident_, ident_h, (size_t) CAP * sizeof(int32_t), cudaMemcpyHostToDevice), "ident init", err)) {
            close();
            return false;
        }
    }
    static const bool p2p_q8_opt_in = [] {
        const char* s = std::getenv("STRATA_P2P_DECODE_SCATTER");
        const char* q = std::getenv("STRATA_PEER_P2P_Q8");
        return s != nullptr && s[0] != '\0' && s[0] != '0' && q != nullptr && q[0] != '\0' && q[0] != '0';
    }();
    if (p2p_ && p2p_q8_opt_in) {
        q8_win_stride_ = (((size_t) strata::kernels::kWindowMaxT * (size_t) (H / 32) * 36u) + 255u) & ~255ull;
        q8_win_steps_ = n_layers * 2;
        if (cudaMalloc((void**) &d_q8_win_, (size_t) q8_win_steps_ * q8_win_stride_) != cudaSuccess) {
            (void) cudaGetLastError();
            d_q8_win_ = nullptr;
            q8_win_stride_ = 0;
            q8_win_steps_ = 0;
        }
    }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) ff), (int) ff);
    }

    // the pairs the primary does not hold, in rank order, as many as fit
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back((int64_t) b);
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    // mapped reads: advise the next kAhead pairs so their reads overlap
    constexpr size_t kAhead = 256;
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const int32_t slot = cache_.admit(pr.first, pr.second);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (slot < 0 || b == nullptr || !cache_.fill_slot(slot, b, refill_, err, (int64_t) lay.blob_bytes(pr.first))) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        src.note_async_read(b, refill_);   // #1237
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    if (!ck(cudaStreamSynchronize(refill_), "fill", err)) { close(); return false; }
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    res_ver_.fetch_add(1, std::memory_order_release);
    return true;
}

void PeerExperts::set_sync_flag(uint32_t* host_flag, uint32_t* dev_vram_flag) {
    h_sync_flag_ = nullptr;
    d_sync_flag_ = nullptr;
    vram_sync_flag_ = false;
    if (device_ < 0 || host_flag == nullptr) return;
    if (p2p_ && dev_vram_flag != nullptr) {
        h_sync_flag_ = host_flag;
        d_sync_flag_ = dev_vram_flag;
        vram_sync_flag_ = true;
        return;
    }
    On on(device_);
    void* dev_ptr = nullptr;
    if (cudaHostGetDevicePointer(&dev_ptr, host_flag, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    h_sync_flag_ = host_flag;
    d_sync_flag_ = (uint32_t*) dev_ptr;
}

bool PeerExperts::direct_mode() {
    static const bool v = [] { const char* e = std::getenv("STRATA_PEER_DIRECT"); return e == nullptr || std::atoi(e) != 0; }();
    return v;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out, int64_t q8_step, bool p2p_q8,
                         uint32_t sync_ring) {
    const bool direct = direct_mode() && out != nullptr;
    struct Timer {
        double& acc;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } timer{ms_launch};
    launched_direct_ = false;
    launched_async_ = false;
    launched_rows_ = 0;
    if (!direct) row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    unsigned long long l_ptr[CAP];
    int32_t l_start[CAP + 1];
    int32_t l_tok[CAP];
    int32_t l_out_row[CAP];
    int32_t g_eid[CAP];
    int16_t g_head[CAP], g_tail[CAP], g_next[CAP];
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows (single pass)
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        const int32_t eid = ids[i];
        int g = -1;
        for (int q = 0; q < groups; ++q) {
            if (g_eid[q] == eid) { g = q; break; }
        }
        if (g < 0) {
            const int32_t slot = res_[(size_t) (layer * n_expert_ + eid)];
            if (slot < 0) { err = "peer experts: an entry was planned for the peer but is not resident"; return false; }
            g = groups++;
            g_eid[g] = eid;
            l_ptr[g] = (unsigned long long) cache_.device_slot(slot);
            g_head[g] = g_tail[g] = (int16_t) i;
            ++experts_;
        } else {
            g_next[g_tail[g]] = (int16_t) i;
            g_tail[g] = (int16_t) i;
        }
        g_next[i] = -1;
    }
    for (int g = 0; g < groups; ++g) {
        l_start[g] = rows;
        for (int16_t j = g_head[g]; j >= 0; j = g_next[j]) {
            l_tok[rows] = (int32_t) ((int64_t) j / k);
            l_out_row[rows] = (int32_t) j;
            if (!direct) row_of_.push_back((int32_t) j);
            ++rows;
        }
    }
    const bool use_gpu_sync = direct && sync_ring > 0 && d_sync_flag_ != nullptr && h_sync_flag_ != nullptr &&
                              d_flag_cnt_ != nullptr;
    if (groups == 0) {
        if (use_gpu_sync) {
            if (!vram_sync_flag_ && !async_in_flight_) {
                strata::core::flag_raise_max(h_sync_flag_, sync_ring);
            } else {
                On on(device_);
                strata::kernels::signal_flag_max(d_sync_flag_, sync_ring, stream_);
                async_in_flight_ = true;
            }
            launched_async_ = true;
        }
        return true;
    }
    l_start[groups] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) {
        err = "peer experts: only native packs are supported";
        return false;
    }
    const auto& f = lay.fmt[(size_t) layer];
    uint8_t* q8_input = (direct && p2p_q8 && !f.had_gu && q8_step >= 0) ? win_q8_ptr(q8_step) : nullptr;
    On on(device_);
    const cudaStream_t s = stream_;
    if (q8_input == nullptr) {
        if (!direct) std::memcpy(h_x_, x, (size_t) (n_tok * H) * sizeof(float));
        if (!ck(cudaMemcpyAsync(d_x_, direct ? x : h_x_, (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s),
                "input", err))
            return false;
        // APR: Hadamard-folded gate/up take H (s * x); d_x_ is this device's private copy, rotated in place
        if (f.had_gu) strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, H, f.had_block, d_sx_, s);
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        q8_input = d_q8_;
    }
    const size_t off_count = 0;
    const size_t off_ptr = 16;
    const size_t off_start = off_ptr + (size_t) groups * sizeof(unsigned long long);
    const size_t off_tok = off_start + (size_t) (groups + 1) * sizeof(int32_t);
    const size_t off_out_row = off_tok + (size_t) rows * sizeof(int32_t);
    const size_t compact_bytes = off_out_row + (size_t) rows * sizeof(int32_t);
    const unsigned long long* dm_ptr = nullptr;
    const int32_t* dm_start = nullptr;
    const int32_t* dm_count = nullptr;
    const int32_t* dm_tok = nullptr;
    const int32_t* dm_out_row = nullptr;
    if (compact_bytes <= strata::kernels::kPeerPutMetaMax) {
        alignas(8) uint8_t compact[strata::kernels::kPeerPutMetaMax];
        const int32_t cnt4[4] = {groups, rows, 0, 0};
        std::memcpy(compact + off_count, cnt4, sizeof(cnt4));
        std::memcpy(compact + off_ptr, l_ptr, (size_t) groups * sizeof(unsigned long long));
        std::memcpy(compact + off_start, l_start, (size_t) (groups + 1) * sizeof(int32_t));
        std::memcpy(compact + off_tok, l_tok, (size_t) rows * sizeof(int32_t));
        std::memcpy(compact + off_out_row, l_out_row, (size_t) rows * sizeof(int32_t));
        if (!strata::kernels::peer_put_meta(d_meta_, compact, compact_bytes, s)) {
            err = "peer experts: plan put_meta failed";
            return false;
        }
        const uint8_t* dmb = (const uint8_t*) d_meta_;
        dm_count = (const int32_t*) (dmb + off_count);
        dm_ptr = (const unsigned long long*) (dmb + off_ptr);
        dm_start = (const int32_t*) (dmb + off_start);
        dm_tok = (const int32_t*) (dmb + off_tok);
        dm_out_row = (const int32_t*) (dmb + off_out_row);
    } else {
        Meta& m = ((Meta*) h_meta_)[meta_slot_];
        meta_slot_ = (meta_slot_ + 1) % kMetaSlots;
        std::memcpy(m.ptr, l_ptr, (size_t) groups * sizeof(unsigned long long));
        std::memcpy(m.start, l_start, (size_t) (groups + 1) * sizeof(int32_t));
        std::memcpy(m.tok, l_tok, (size_t) rows * sizeof(int32_t));
        std::memcpy(m.out_row, l_out_row, (size_t) rows * sizeof(int32_t));
        m.count[0] = groups;
        m.count[1] = rows;
        if (!ck(cudaMemcpyAsync(d_meta_, &m, sizeof(Meta), cudaMemcpyHostToDevice, s), "plan", err))
            return false;
        const Meta* dm = (const Meta*) d_meta_;
        dm_ptr = dm->ptr;
        dm_start = dm->start;
        dm_count = dm->count;
        dm_tok = dm->tok;
        dm_out_row = dm->out_row;
    }
    auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
    if (f.had_d) { L.had_block = f.had_block; L.d_signs = d_sh_; }
    if (direct && q8_step >= 0 && strata::kernels::native_expert_grouped_p2p(
                      L, dm_ptr, dm_start, dm_count, dm_out_row, dm_tok, groups, rows, q8_input, d_scratch_, out,
                      use_gpu_sync ? d_flag_cnt_ : nullptr, use_gpu_sync ? d_sync_flag_ : nullptr, sync_ring, s)) {
        if (use_gpu_sync) {
            launched_async_ = true;
            async_in_flight_ = true;
        }
        launched_direct_ = true;
        return true;
    }
    strata::kernels::native_expert_grouped(L, dm_ptr, dm_start, dm_count, d_ident_, dm_tok, groups, rows, q8_input,
                                           d_scratch_, d_out_, s);
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            if (use_gpu_sync) {
                strata::kernels::scatter_rows_f32_flag(d_out_, out, dm_out_row, rows, H, d_flag_cnt_, d_sync_flag_,
                                                       sync_ring, s);
                launched_async_ = true;
                async_in_flight_ = true;
            } else {
                strata::kernels::scatter_rows_f32(d_out_, out, dm_out_row, rows, H, s);
            }
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        return true;
    }
    return ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s),
              "results", err);
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (launched_rows_ == 0) { launched_async_ = false; return true; }
    ++launches;
    if (launched_async_) { launched_rows_ = 0; launched_async_ = false; return true; }
    const auto t0 = std::chrono::steady_clock::now();
    {
        On on(device_);
        // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, h_out_ + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    return true;
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!pending_.empty() || max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        const int32_t* r1 = res_.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    const bool gated = gate_landed_ != nullptr;
    if (gated) {   // the copies go in layer order and each layer's landing raises the gate's counter
        std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.layer < b.layer; });
        gate_landed_->store(-1, std::memory_order_release);
        if (h_gate_ != nullptr) *h_gate_ = -1;
        gate_has_->assign((size_t) n_layers_, 0);
        *gate_last_ = -1;
    }
    On on(device_);
    int32_t issued_layer = -1;   // the layer whose copies are on the stream and not yet ticked
    auto tick = [&](int32_t layer) -> bool {
        if ((h_gate_ != nullptr
                 ? cudaMemcpyAsync((void*) h_gate_, d_gate_vals_ + layer, 4, cudaMemcpyDeviceToHost, refill_)
                 : cudaLaunchHostFunc(refill_, [](void* p) {
                auto* t = (Tick*) p;
                int32_t cur = t->counter->load(std::memory_order_relaxed);
                while (cur < t->layer && !t->counter->compare_exchange_weak(cur, t->layer, std::memory_order_release)) {}
            }, &ticks_[(size_t) layer])) == cudaSuccess)
            return true;
        (void) cudaGetLastError();
        cudaStreamSynchronize(refill_);   // no callback: wait here, the gate opens at once
        gate_landed_->store((int32_t) n_layers_, std::memory_order_release);
        return false;
    };
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res_[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || b == nullptr) continue;
        if (gated && issued_layer >= 0 && issued_layer != s.layer) tick(issued_layer);
        if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), b, (size_t) lay.blob_bytes(s.layer), cudaMemcpyHostToDevice,
                                refill_), "refill", err))
            return false;
        if (gated) {
            (*gate_has_)[(size_t) s.layer] = 1;
            *gate_last_ = std::max(*gate_last_, s.layer);
            issued_layer = s.layer;
        }
        res_[out] = kNotResident;              // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        ++swaps_;
    }
    if (gated && issued_layer >= 0) tick(issued_layer);
    if (!ck(cudaEventRecord(refill_ev_, refill_), "refill event", err)) return false;
    // #2188: hand the copies and their landing signals to the GPU now (WDDM batches a stream's work until a driver
    // call flushes it, and the verify window only spins on the pinned int; Linux submits at once: a no-op)
    if (gated && h_gate_ != nullptr && issued_layer >= 0) (void) cudaStreamQuery(refill_);
    res_ver_.fetch_add(1, std::memory_order_release);
    return true;
}

volatile int32_t* PeerExperts::set_overlap_gate(std::atomic<int32_t>* landed, std::vector<uint8_t>* has, int32_t* last,
                                                bool copy_signal) {
    gate_landed_ = landed;
    gate_has_ = has;
    gate_last_ = last;
    ticks_.resize((size_t) n_layers_);
    for (int64_t l = 0; l < n_layers_; ++l) ticks_[(size_t) l] = {landed, (int32_t) l};
    if (copy_signal && h_gate_ == nullptr) {
        On on(device_);
        std::vector<int32_t> iota((size_t) n_layers_);
        for (size_t i = 0; i < iota.size(); ++i) iota[i] = (int32_t) i;
        int32_t* vals = nullptr;
        void* hp = nullptr;
        if (cudaMalloc((void**) &vals, iota.size() * 4) == cudaSuccess &&
            cudaMemcpy(vals, iota.data(), iota.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaHostAlloc(&hp, 64, cudaHostAllocPortable) == cudaSuccess) {
            *(volatile int32_t*) hp = -1;
            d_gate_vals_ = vals;
            h_gate_ = (volatile int32_t*) hp;
        } else {   // host functions then
            (void) cudaGetLastError();
            if (vals) cudaFree(vals);
        }
    }
    return h_gate_;
}

void PeerExperts::flip_pending() {
    if (pending_.empty()) return;
    for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot;
    res_ver_.fetch_add(1, std::memory_order_release);
}

void PeerExperts::apply_pending(bool wait) {
    if (pending_.empty()) return;
    {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot;
    pending_.clear();
    res_ver_.fetch_add(1, std::memory_order_release);
}

}  // namespace strata::core
