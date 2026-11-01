// src/core/dflash2.cpp - see include/strata/core/dflash2.hpp.
#include "strata/core/dflash2.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <atomic>
#include <initializer_list>
#include <stdexcept>

namespace strata::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

constexpr int GGML_Q8_0 = 8;

}  // namespace

Dflash2Drafter::~Dflash2Drafter() {
    if (device_ < 0) return;
    prof_report();
    prof_free();
    const OnDevice on_device(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& sl : slots_) free_slot(*sl);   // every conversation's graphs and rings, slot 0 first
    auto fr = [](void* p) { if (p) cudaFree(p); };
    fr(blob_); fr(d_meta_); fr(xq_); fr(xqp_); fr(xdp_); fr(xqc_); fr(xdc_); fr(rt_ctx_); fr(rt_blk_);
    fr(fcy_); fr(hctx_); fr(kc_); fr(vc_); fr(h_); fr(xn_); fr(xc_); fr(dyn_); fr(q_); fr(qn_); fr(k_); fr(v_); fr(kblk_); fr(ao_); fr(o_);
    fr(gate_); fr(up_); fr(mlp_); fr(part_); fr(fn_); fr(fn_batch_); fr(hsel_batch_); fr(logits_); fr(topval_); fr(hsel_); fr(toptok_); fr(topscratch_);
    fr(dvocab_); fr(dhead_);
    if (h_meta_) cudaFreeHost(h_meta_);
    if (h_out_) cudaFreeHost(h_out_);
    if (cs_) cudaStreamDestroy(cs_);
    if (cs2_) cudaStreamDestroy(cs2_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    for (cudaEvent_t e : ev_ctx_) if (e) cudaEventDestroy(e);
    if (ev_a_) cudaEventDestroy(ev_a_);
    if (ev_b_) cudaEventDestroy(ev_b_);
}

// STRATA_DF2_PROFILE=1 (measurement only): the round runs without its graph and an event follows every stage; the per-stage averages print at exit
struct Dflash2Drafter::Prof {
    std::vector<cudaEvent_t> ev;
    std::vector<std::string> name;
    std::vector<std::pair<std::string, double>> total;
    int64_t rounds = 0;
};

void Dflash2Drafter::prof_free() {
    delete prof_;
    prof_ = nullptr;
}

// STRATA_DF2_SKIP=attn,head,ctx,mlp,qkvo (measurement only): leaves those parts out of the round, so the graph's GPU time without them can be read
static bool skip_part(const char* what) {
    static const std::string list = [] { const char* e = std::getenv("STRATA_DF2_SKIP"); return std::string(e ? e : ""); }();
    return !list.empty() && list.find(what) != std::string::npos;
}

void Dflash2Drafter::stamp(const char* name) {
    if (!prof_) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, cs_);
    prof_->ev.push_back(e);
    prof_->name.push_back(name);
}

void Dflash2Drafter::prof_collect() {
    if (!prof_ || prof_->ev.size() < 2) return;
    cudaStreamSynchronize(cs_);
    for (size_t i = 1; i < prof_->ev.size(); ++i) {
        float ms = 0;
        cudaEventElapsedTime(&ms, prof_->ev[i - 1], prof_->ev[i]);
        auto it = std::find_if(prof_->total.begin(), prof_->total.end(), [&](const auto& t) { return t.first == prof_->name[i]; });
        if (it == prof_->total.end()) prof_->total.emplace_back(prof_->name[i], ms); else it->second += ms;
    }
    for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);
    prof_->ev.clear();
    prof_->name.clear();
    ++prof_->rounds;
}

void Dflash2Drafter::prof_report() const {
    if (!prof_ || prof_->rounds == 0) return;
    double sum = 0;
    for (const auto& t : prof_->total) sum += t.second;
    std::fprintf(stderr, "strata dflash2 profile (%lld rounds, ms per round, total %.3f):", (long long) prof_->rounds, sum / (double) prof_->rounds);
    for (const auto& t : prof_->total) std::fprintf(stderr, " %s %.3f", t.first.c_str(), t.second / (double) prof_->rounds);
    std::fputc(10, stderr);
}

const Dflash2Drafter::Tensor& Dflash2Drafter::find(const std::string& name) const {
    for (const Tensor& t : tensors_)
        if (t.name == name) return t;
    throw std::runtime_error("dflash2: tensor " + name + " is missing from the pack");
}

bool Dflash2Drafter::load(const std::string& path, int device, std::string& err) {
    device_ = device;
    const OnDevice on_device(device_);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "dflash2: cannot open " + path; return false; }
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) { err = "dflash2: " + path + " is not a DF2P v1 pack"; return false; }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    const long data_off = 12 + (long) hdr[2];
    uint64_t total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        long long v1, v2, v3, v4;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            const std::string key = b;
            if (key == "hidden") N_ = std::atoi(c);
            else if (key == "layers") L_ = std::atoi(c);
            else if (key == "heads") NH_ = std::atoi(c);
            else if (key == "kv_heads") NKV_ = std::atoi(c);
            else if (key == "head_dim") HD_ = std::atoi(c);
            else if (key == "inter") I_ = std::atoi(c);
            else if (key == "window") window_ = std::atoi(c);
            else if (key == "rope_theta") theta_ = (float) std::atof(c);
            else if (key == "eps") eps_ = (float) std::atof(c);
            else if (key == "conv_group") group_ = std::atoi(c);
            else if (key == "rank") rank_ = std::atoi(c);
            else if (key == "mask_token") mask_token_ = std::atoi(c);
            else if (key == "vocab") vocab_ = std::atoll(c);
            else if (key == "taps") {
                taps_.clear();
                for (const char* q = c; *q;) { taps_.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; }
            }
        } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
            Tensor t;
            t.name = a; t.dtype = b; t.rows = v1; t.cols = v2; t.off = (uint64_t) v3; t.bytes = (uint64_t) v4;
            tensors_.push_back(t);
            total = std::max(total, t.off + t.bytes);
        }
    }
    if (tensors_.empty() || NH_ != 32 || NKV_ != 8 || HD_ != 128 || rank_ != 256 || N_ != 2560 || L_ < 1 || L_ > 8) {
        err = "dflash2: the pack's shape is not the Flash-Next drafter this engine was written for";
        return false;
    }
    if (window_ < 1 || window_ > kMaxWindow) {
        err = "dflash2: pack attention window " + std::to_string(window_) + " is outside the supported 1.." + std::to_string(kMaxWindow) + " range";
        return false;
    }
    ring_slots_ = window_ + kRingSlack;
    // every slot's host mirror is ring_slots entries - the same df2_ring_mirror_entries sizing set_n_slots()
    // applies to slots added after this load, so neither path can drift from the other
    for (auto& sl : slots_) { sl->ring_pos_.assign((size_t) df2_ring_mirror_entries(ring_slots_), -1); sl->have_ctx_ = 0; }
    // The pack's identity for save_state/restore_state: a parked conversation's blob is only
    // restorable into a drafter built from this same pack (path hash, size, mtime).
    pack_path_ = path;
    {
        std::error_code ec;
        const std::filesystem::path p(path);
        pack_size_ = (uint64_t) std::filesystem::file_size(p, ec);
        if (ec) pack_size_ = 0;
        const auto wt = std::filesystem::last_write_time(p, ec);
        pack_mtime_ = ec ? 0 : (int64_t) wt.time_since_epoch().count();
    }
    if (cudaMalloc((void**) &blob_, total) != cudaSuccess) { err = "dflash2: the pack does not fit in VRAM (" + std::to_string(total >> 20) + " MiB)"; return false; }
    vram_ += total;
    uint64_t max_bytes = 0;
    for (const Tensor& t : tensors_) max_bytes = std::max(max_bytes, t.bytes);
    std::vector<uint8_t> stage(max_bytes);
    uint8_t* tmp = nullptr;
    if (cudaMalloc((void**) &tmp, max_bytes) != cudaSuccess) { err = "dflash2: upload scratch"; return false; }
    struct TmpFree { uint8_t* p; ~TmpFree() { cudaFree(p); } } tmp_free{tmp};
    for (const Tensor& t : tensors_) {
#if defined(_WIN32)
        _fseeki64(f, data_off + (long long) t.off, SEEK_SET);
#else
        fseeko(f, data_off + (off_t) t.off, SEEK_SET);
#endif
        if (std::fread(stage.data(), 1, t.bytes, f) != t.bytes) { err = "dflash2: the pack is shorter than its index"; return false; }
        if (t.dtype == "q8_0") {   // GGUF blocks -> the tiled qs plane at the tensor's offset, then the d plane (same total bytes)
            if (cudaMemcpy(tmp, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
            kernels::df2_q8_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols, (int) t.cols, t.rows * t.cols / 32, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: weight conversion failed"; return false; }
        } else if (cudaMemcpy(blob_ + t.off, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dflash2: weight upload failed";
            return false;
        }
    }
    // the draft round runs while the verifier's async commit still has work on the card: a high-priority stream keeps it from queueing behind that
    // (CUDA by default; on HIP only with STRATA_DF2_PRIO=1, where the engine's other streams keep the default priority)
    int prio_lo = 0, prio_hi = 0;
#if defined(STRATA_USE_HIP)
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return e && e[0] == '1'; }();
    if (hi_prio) hipDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (hipStreamCreateWithPriority(&cs_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        hipStreamCreateWithPriority(&cs2_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#else
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return !(e && e[0] == '0'); }();
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        cudaStreamCreateWithPriority(&cs2_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#endif
        err = "dflash2: stream";
        return false;
    }
    if (cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "dflash2: events";
        return false;
    }
    for (int l = 0; l < 8; ++l)
        if (cudaEventCreateWithFlags(&ev_ctx_[l], cudaEventDisableTiming) != cudaSuccess) { err = "dflash2: events"; return false; }
    if (const char* e = std::getenv("STRATA_DF2_PROFILE"); e && e[0] == '1') prof_ = new Prof();
    if (const char* e = std::getenv("STRATA_DF2_TIMING"); e && e[0] == '1') {
        timing_ = true;
        cudaEventCreate(&ev_a_);
        cudaEventCreate(&ev_b_);
    }
    return alloc(err);
}

bool Dflash2Drafter::alloc(std::string& err) {
    auto dev = [&](auto** p, size_t n) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
        if (cudaMalloc((void**) p, n * sizeof(T)) != cudaSuccess) return false;
        vram_ += n * sizeof(T);
        return true;
    };
    const size_t B = (size_t) kRows * 4;
    bool ok = dev(&d_meta_, 256) && dev(&xq_, kernels::native_q8_1_bytes(N_, kRows) + kernels::native_q8_1_il_bytes(N_, kRows) + 4096) && dev(&xqp_, (size_t) 5 * N_ * B + 4096) && dev(&xdp_, (size_t) 5 * N_ / 32 * B + 64) && dev(&xqc_, (size_t) 5 * N_ * B + 4096) && dev(&xdc_, (size_t) 5 * N_ / 32 * B + 64) && dev(&rt_ctx_, B * 128) && dev(&rt_blk_, B * 128) &&
              dev(&fcy_, B * N_) && dev(&hctx_, B * N_) && dev(&kc_, B * NKV_ * HD_) && dev(&vc_, B * NKV_ * HD_) && dev(&h_, B * N_) && dev(&xn_, B * N_) &&
              dev(&xc_, B * N_) && dev(&dyn_, B * 640) && dev(&q_, B * NH_ * HD_) && dev(&qn_, B * NH_ * HD_) && dev(&k_, B * NKV_ * HD_) && dev(&v_, B * NKV_ * HD_) &&
              dev(&kblk_, B * NKV_ * HD_) && dev(&ao_, B * NH_ * HD_) && dev(&o_, B * N_) && dev(&gate_, B * I_) && dev(&up_, B * I_) && dev(&mlp_, B * N_) &&
              dev(&part_, (size_t) kernels::df2_attention_chunks(window_) * 8 * 32 * 130) && dev(&fn_, B * N_) &&
              dev(&fn_batch_, (size_t) kBatchMaxRows * N_) && dev(&hsel_batch_, (size_t) kBatchMaxRows * rank_) &&
              dev(&topval_, (size_t) kBatchMaxRows * 16) && dev(&toptok_, (size_t) kBatchMaxRows * 16) &&
              dev(&hsel_, B * rank_) && dev(&topscratch_, (size_t) kDrafts * kernels::kDf2TopBlocks * kernels::kDf2TopK * 2);
    for (auto& sl : slots_)
        if (ok && !alloc_slot_rings(*sl)) ok = false;   // every slot the drafter was grown to before load()
    if (!ok) { err = "dflash2: the drafter's buffers do not fit in VRAM"; return false; }
    void* hd = nullptr;
    void* dd = nullptr;
    if (!mapped((kDf2MaxSlots * 32 + kDf2MaxSlots * 4 + 128) * 4, (void**) &hd, &dd)) { err = "dflash2: mapped staging"; return false; }
    h_meta_ = (int32_t*) hd;
    m_meta_ = (int32_t*) dd;
    if (!mapped(128 * 4, &hd, &dd)) { err = "dflash2: mapped output"; return false; }
    h_out_ = (int32_t*) hd;
    out_tok_ = (int32_t*) dd;            // the select kernel writes the picks here (8 ints per slot, up to 8 slots) ...
    out_prob_ = (float*) dd + 64;        // ... and their probabilities at +64 ints
    cudaMemset(rt_ctx_, 0, B * 128 * 4);
    cudaMemset(xqp_, 0, (size_t) 5 * N_ * B + 4096);
    cudaMemset(xdp_, 0, ((size_t) 5 * N_ / 32 * B + 64) * 4);
    cudaMemset(xqc_, 0, (size_t) 5 * N_ * B + 4096);
    cudaMemset(xdc_, 0, ((size_t) 5 * N_ / 32 * B + 64) * 4);
    return cudaDeviceSynchronize() == cudaSuccess;
}

// One conversation slot's K/V rings (the ring is the slot's only device memory; the pack, the head and the block
// scratch stay shared on the drafter).  Allocates and zeroes on the already-open device_ and tracks vram_, so
// vram_bytes() covers every slot's rings; a failed allocation unwinds what it made, and the caller drops the slot.
// The per-plane byte count is recorded in the slot (df2_ring_plane_bytes) so free_slot subtracts exactly the
// bytes this allocation added: a set_fp16_ring or a second load() between allocate and free would otherwise
// make vram_ drift, or underflow it (uint64_t) when the recompute comes out bigger.
bool Dflash2Drafter::alloc_slot_rings(Slot& sl) {
    const size_t ring_bytes = (size_t) df2_ring_plane_bytes(ring_slots_, NKV_, HD_, (int) ring_elem_bytes());
    sl.ring_plane_bytes_ = ring_bytes;   // set before the first cudaMalloc: the unwind below subtracts with it too
    for (int l = 0; l < L_; ++l) {
        if (cudaMalloc((void**) &sl.kring_[l], ring_bytes) == cudaSuccess) vram_ += ring_bytes; else break;
        if (cudaMalloc((void**) &sl.vring_[l], ring_bytes) == cudaSuccess) vram_ += ring_bytes; else break;
    }
    for (int l = 0; l < L_; ++l)
        if (sl.kring_[l] == nullptr || sl.vring_[l] == nullptr) { free_slot(sl); return false; }
    for (int l = 0; l < L_; ++l) { cudaMemset(sl.kring_[l], 0, ring_bytes); cudaMemset(sl.vring_[l], 0, ring_bytes); }
    return true;
}

void Dflash2Drafter::free_slot(Slot& sl) {
    for (cudaGraphExec_t& e : sl.exec_)
        if (e) { cudaGraphExecDestroy(e); e = nullptr; }
    for (cudaGraphExec_t& e : sl.body_exec_)
        if (e) { cudaGraphExecDestroy(e); e = nullptr; }
    // the bytes this slot's planes were cudaMalloc'd with, not a recompute from the current geometry.
    // up to 8, not L_: a half-loaded drafter can hold rings for layers it never validated
    const size_t ring_bytes = sl.ring_plane_bytes_;
    for (int l = 0; l < 8; ++l) {
        if (sl.kring_[l]) { cudaFree(sl.kring_[l]); sl.kring_[l] = nullptr; vram_ -= ring_bytes; }
        if (sl.vring_[l]) { cudaFree(sl.vring_[l]); sl.vring_[l] = nullptr; vram_ -= ring_bytes; }
    }
    sl.ring_plane_bytes_ = 0;
}

// One indexed write to the host ring-position mirror.  The mirror has ring_slots entries wherever a slot comes
// to exist (load(), set_n_slots(); df2_ring_mirror_entries sizes both paths), and df2_ring_mirror_index is the
// pos % ring_slots the kv_ring kernels use, so the assert only fires if that sizing ever regresses - finding 1's
// overrun was silent heap corruption under an 8k-window pack, not a visible failure.
int32_t& Dflash2Drafter::mirror_cell(Slot& sl, int64_t pos) {
    const size_t i = (size_t) df2_ring_mirror_index(pos, ring_slots_);
    assert(i < sl.ring_pos_.size());
    return sl.ring_pos_[i];
}

// Grow-only up to kDf2MaxSlots.  Extra slots are free until a round runs on them; after that their rings hold a
// conversation and shrinking would silently drop it, so it is refused.  Rings for a slot added after load() are
// allocated on the open device here; a slot added before load() gets its rings from alloc().
bool Dflash2Drafter::set_n_slots(int n, std::string& err) {
    if (n < 1 || n > kDf2MaxSlots) {
        err = "dflash2: slot count " + std::to_string(n) + " is outside 1.." + std::to_string(kDf2MaxSlots);
        return false;
    }
    const int have = (int) slots_.size();
    if (n == have) return true;
    if (n > have) {
        if (device_ >= 0) {
            const OnDevice on_device(device_);
            while ((int) slots_.size() < n) {
                auto sl = std::make_unique<Slot>();
                // a slot added after load() needs the same ring_slots_-sized host mirror load() gives every
                // slot: an 8k-window pack (ring 8208) overruns any other size, silently, in context()/draft()/
                // save_state() (dflash2_slots_test.cpp pins the sizing and the index bounds)
                sl->ring_pos_.assign((size_t) df2_ring_mirror_entries(ring_slots_), -1);
                if (!alloc_slot_rings(*sl)) {
                    err = "dflash2: the extra slot's rings do not fit in VRAM";
                    // all-or-nothing: the slots this call pushed (alloc_slot_rings already unwound the one it
                    // failed on) have run no round, so free and drop them too - a failed growth leaves
                    // n_slots() and vram_bytes() describing the drafter that served its slots before the call
                    for (size_t i = (size_t) have; i < slots_.size(); ++i) free_slot(*slots_[(size_t) i]);
                    slots_.resize((size_t) df2_slots_after_failed_grow(have, (int) slots_.size() - have));
                    return false;
                }
                slots_.push_back(std::move(sl));
            }
        } else {
            while ((int) slots_.size() < n) {
                auto sl = std::make_unique<Slot>();
                sl->ring_pos_.assign((size_t) df2_ring_mirror_entries(ring_slots_), -1);   // load() resizes it to the pack's ring
                slots_.push_back(std::move(sl));
            }
        }
        return true;
    }
    if (cur_ >= n) { err = "dflash2: cannot drop the active slot " + std::to_string(cur_); return false; }
    for (int i = n; i < have; ++i) {
        // "used" means a round ran on the slot: context()/draft() moved have_ctx_, or a capture ran the ring
        // kernels against the ring even if the round afterwards failed (exec_ non-null).
        const Slot& sl = *slots_[(size_t) i];
        bool captured = false;
        for (cudaGraphExec_t e : sl.exec_) captured = captured || e != nullptr;
        if (sl.have_ctx_ != 0 || captured) {
            err = "dflash2: slot " + std::to_string(i) + " has run a round; its ring holds a conversation";
            return false;
        }
    }
    if (device_ >= 0) {
        const OnDevice on_device(device_);
        for (int i = have - 1; i >= n; --i) free_slot(*slots_[(size_t) i]);
    }
    slots_.resize((size_t) n);
    return true;
}

bool Dflash2Drafter::set_slot(int i, std::string& err) {
    if (!df2_slot_ok(i, (int) slots_.size())) {
        err = "dflash2: slot " + std::to_string(i) + " is outside 0.." + std::to_string((int) slots_.size() - 1);
        return false;
    }
    cur_ = i;
    return true;
}

bool Dflash2Drafter::bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err) {
    const OnDevice on_device(device_);
    embed_ = std::move(embed);
    head_w_ = head_w;
    head_type_ = head_type;
    n_dvocab_ = head_rows;
    if (!vocab.empty()) {
        n_dvocab_ = (int64_t) vocab.size();
        if (cudaMalloc((void**) &dvocab_, vocab.size() * 4) != cudaSuccess) { err = "dflash2: draft vocabulary"; return false; }
        vram_ += vocab.size() * 4;
        cudaMemcpy(dvocab_, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
    }
    if ((n_dvocab_ + kernels::kDf2TopBlocks - 1) / kernels::kDf2TopBlocks > 4096) { err = "dflash2: the head has too many rows for the top-k"; return false; }
    if (!logits_ && cudaMalloc((void**) &logits_, (size_t) kDrafts * (size_t) n_dvocab_ * 4) != cudaSuccess) { err = "dflash2: the draft logits do not fit"; return false; }
    vram_ += (uint64_t) kDrafts * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

uint64_t Dflash2Drafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab, const std::string& vocab_file) const {
    int64_t n_dv = 0;
    if (!vocab_file.empty()) {
        if (std::FILE* vf = std::fopen(vocab_file.c_str(), "rb")) {
            std::fseek(vf, 0, SEEK_END);
            const long size = std::ftell(vf);
            std::fclose(vf);
            if (size >= 4 && size % 4 == 0) n_dv = (int64_t) (size / 4);
        }
    }
    return df2_bind_bytes(head_row_bytes, n_vocab, n_dv, kDrafts);
}

bool Dflash2Drafter::bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err) {
    const OnDevice on_device(device_);
    if (head == nullptr || !head->loaded()) { err = "dflash2: the drafter needs the native head (--native)"; return false; }
    const WeightRef* we = wt.find("token_embd.weight");
    if (we == nullptr) { err = "dflash2: token_embd.weight is missing"; return false; }
    EmbedFn embed = [we](const int32_t* tok, int n, float* out, cudaStream_t cs) {
        if (const NativeEmbed* ne = native_embed()) {
            ne->gather_dev(tok, n, out, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            kernels::embedding_gather_dev(codes, scales, offsets, tok, n, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                          (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), out, cs);
        }
    };
    std::vector<int32_t> vocab;
    if (!vocab_file.empty()) {
        if (std::FILE* vf = std::fopen(vocab_file.c_str(), "rb")) {
            std::fseek(vf, 0, SEEK_END);
            const long size = std::ftell(vf);
            std::fseek(vf, 0, SEEK_SET);
            if (size >= 4 && size % 4 == 0) {
                vocab.resize((size_t) size / 4);
                if (std::fread(vocab.data(), 4, vocab.size(), vf) != vocab.size()) vocab.clear();
            }
            std::fclose(vf);
        }
    }
    const void* hw = head->weights();
    const int64_t row_bytes = (int64_t) head->row_bytes();
    int64_t head_rows = row_bytes > 0 ? (int64_t) (head->weight_bytes() / (uint64_t) row_bytes) : 0;
    if (!vocab.empty()) {
        int32_t* dv = nullptr;
        if (cudaMalloc((void**) &dv, vocab.size() * 4) != cudaSuccess || cudaMalloc((void**) &dhead_, (size_t) ((int64_t) vocab.size() * row_bytes)) != cudaSuccess) {
            if (dv) cudaFree(dv);
            err = "dflash2: the draft vocabulary's head rows do not fit";
            return false;
        }
        cudaMemcpy(dv, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
        kernels::gather_rows((const uint8_t*) hw, row_bytes, dv, (int64_t) vocab.size(), dhead_, nullptr);
        cudaDeviceSynchronize();
        cudaFree(dv);
        vram_ += (uint64_t) ((int64_t) vocab.size() * row_bytes);
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    }
    if (!bind_raw(embed, hw, head->type(), head_rows, vocab, err)) return false;
    std::fprintf(stderr, "strata dflash2: %lld-token draft head (%.1f MiB), drafter %.0f MiB of VRAM\n", (long long) n_dvocab_,
                 (double) (n_dvocab_ * row_bytes) / 1048576.0, (double) vram_ / 1048576.0);
    return true;
}

void Dflash2Drafter::gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, cudaStream_t cs, const uint8_t* xq, const float* xd) {
    kernels::Df2Mat mats[3];
    int n = 0, n_in = 0;
    auto o = outs.begin();
    for (const char* name : names) {
        const Tensor& t = find(name);
        mats[n] = kernels::Df2Mat{blob_ + t.off, *o++, (int) t.rows};
        if (n == 0) n_in = (int) t.cols;
        ++n;
    }
    kernels::df2_gemm(mats, n, xq, xd, n_in, ncols, cs);
}

bool Dflash2Drafter::record_context(const float* taps, int n, cudaStream_t cs, std::string& err) {
    using namespace kernels;
    if (skip_part("ctx")) return true;
    Slot& sl = cur();   // the context update writes the active conversation's ring
    try {
        const int tap_n = (int) taps_.size() * N_;
        df2_rope_table(d_meta_ + 0, n, theta_, rt_ctx_, cs);
        stamp("df2_rope_table");
        df2_quant_rows(taps, tap_n, n, xqc_, xdc_, cs);
        stamp("df2_quant_rows");
        gemm({"fc.weight"}, {fcy_}, n, cs, xqc_, xdc_);
        stamp("gemm:fc");
        df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, n, N_, eps_, xqc_, xdc_, cs);
        stamp("df2_rms_quant");
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, n, cs, xqc_, xdc_);
            stamp("gemm:kp");
            if (fp16_ring_)
                df2_kv_ring_f16(kc_, vc_, f32(p + "self_attn.k_norm.weight"), d_meta_ + 0, rt_ctx_, sl.kring_[l], sl.vring_[l], ring_slots_, n, NKV_, eps_, cs);
            else
                df2_kv_ring(kc_, vc_, f32(p + "self_attn.k_norm.weight"), d_meta_ + 0, rt_ctx_, sl.kring_[l], sl.vring_[l], ring_slots_, n, NKV_, eps_, cs);
            stamp("df2_kv_ring");
            if (fork_) cudaEventRecord(ev_ctx_[l], cs);   // layer l's keys and values are in the ring
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block_body(cudaStream_t cs, std::string& err) {
    using namespace kernels;
    Slot& sl = cur();   // the block attends over the active conversation's ring
    try {
        const int B = kRows;
        df2_rope_table(d_meta_ + 8, B, theta_, rt_blk_, cs);
        stamp("df2_rope_table");
        embed_(d_meta_ + 24, B, h_, cs);
        stamp("embed_");
        df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, B, N_, eps_, xqp_, xdp_, cs);
        stamp("df2_rms_quant");
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const float* base_a = f32(p + "attention_conv.base_kernel");
            const float* base_m = f32(p + "mlp_conv.base_kernel");
            const std::string akp = p + "attention_conv.kernel_projection.weight", mkp = p + "mlp_conv.kernel_projection.weight";
            const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
            if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:akp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, cs, xqp_, xdp_);
            stamp("gemm:qp");
            if (!skip_part("small")) df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_, kblk_, B, NH_, NKV_, eps_, cs);
            stamp("df2_norm_rope_qk");
            if (fork_) cudaStreamWaitEvent(cs, ev_ctx_[l], 0);   // the context update has written this layer's ring
            if (!skip_part("attn")) {
                if (fp16_ring_)
                    df2_attention_f16(qn_, sl.kring_[l], sl.vring_[l], kblk_, v_, d_meta_ + 17, ring_slots_, window_, part_, xqp_, xdp_, cs);
                else
                    df2_attention(qn_, sl.kring_[l], sl.vring_[l], kblk_, v_, d_meta_ + 17, ring_slots_, window_, part_, xqp_, xdp_, cs);
            }
            stamp("df2_attention");
            if (!skip_part("qkvo")) gemm({(p + "self_attn.o_proj.weight").c_str()}, {o_}, B, cs, xqp_, xdp_);
            stamp("gemm:o_proj");
            if (!skip_part("small")) df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
            if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:mkp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, cs, xqp_, xdp_);
            stamp("gemm:gp");
            if (!skip_part("small")) df2_swiglu_quant(gate_, up_, B, I_, xqp_, xdp_, cs);
            stamp("df2_swiglu_quant");
            if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, B, cs, xqp_, xdp_);
            stamp("gemm:mlp.down_proj");
            // the next layer's input norm, or after the last layer the final norm (-> fn_, the rows the head and the selector read)
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            if (!skip_part("small")) df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
        }
        // df2_conv_rms_quant on the final layer has already quantized fn_ rows 1..7 into xqp_ + N_ and xdp_ + N_ / 32
        gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, kDrafts, cs, xqp_ + N_, xdp_ + N_ / 32);
        stamp("gemm:candidate_selector.hidden_projection");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block_head(cudaStream_t cs, std::string& err) {
    using namespace kernels;
    try {
        native_quantize_q8_1(fn_ + N_, xq_, N_, kDrafts, cs);   // rows 1..7, the head's format
        df2_quant_rows(fn_ + N_, N_, kDrafts, xqp_, xdp_, cs);   // and the selector projection's
        if (!skip_part("head")) {
            static const int head_il = [] {
                const char* e = std::getenv("STRATA_DFLASH_HEAD_IL");
                return e && e[0] ? (e[0] != '0' ? 1 : 0) : -1;
            }();
            bool use_il = (head_il != 0) && native_mmvq_il_supported(head_type_, kDrafts, (int) n_dvocab_);
            bool tuned_il = false;
            if (!use_il && head_il == 1) {
                native_mmvq_il_tune(4);
                use_il = native_mmvq_il_supported(head_type_, kDrafts, (int) n_dvocab_);
                tuned_il = use_il;
                if (!use_il) native_mmvq_il_tune(0);
            }
            if (use_il) {
                uint8_t* xil = xq_ + native_q8_1_bytes(N_, kRows);
                native_q8_1_interleave(xq_, xil, N_, kDrafts, cs);
                native_mmvq_il(head_type_, head_w_, xq_, xil, logits_, N_, (int) n_dvocab_, kDrafts, cs);
                if (tuned_il) native_mmvq_il_tune(0);
            } else {
                native_mmvq(head_type_, head_w_, xq_, logits_, N_, (int) n_dvocab_, kDrafts, cs);
            }
        }
        stamp("native_mmvq");
        df2_topk(logits_, (int) n_dvocab_, dvocab_, kDrafts, topscratch_, topval_, toptok_, cs);
        stamp("df2_topk");
        gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, kDrafts, cs, xqp_, xdp_);
        stamp("gemm:candidate_selector.hidden_projection");
        df2_select(hsel_, topval_, toptok_, reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
                   reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off), rank_, kDrafts, d_meta_ + 16, out_tok_,
                   out_prob_, cs);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block(cudaStream_t cs, std::string& err) {
    return record_block_body(cs, err) && record_block_head(cs, err);
}

// Captures against the active slot: the graph bakes this slot's ring pointers and the taps staging pointer,
// which it records in this slot's taps_, so a captured graph belongs to the slot it was captured on and to the
// one shared taps buffer every slot drafts into.  A slot that was never prepared captures lazily on its first
// draft(); one whose graphs live on another taps buffer captures nothing new (mixing graph sets across buffers
// would launch a graph that reads the old buffer's rows).
bool Dflash2Drafter::capture(int n, const float* taps, std::string& err) {
    Slot& sl = cur();
    if (sl.exec_[n]) return true;
    if (!df2_graphs_valid_for(sl.taps_, taps)) { err = "dflash2: capture: this slot's graphs are captured on another taps buffer"; return false; }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin capture"; return false; }
    kernels::copy_i32_from_mapped(d_meta_, m_meta_ + (size_t) cur_ * 32, 32, cs_);
    // the context update (fc, the keys and values) and the block's front run side by side: they meet before each layer's attention
    // measured: -0.3 ms a round on an RTX 3060, +0.0..0.7 on a Strix Halo (gfx1151), so the HIP builds keep one stream (STRATA_DF2_FORK=1 / 0 overrides)
#if defined(STRATA_USE_HIP)
    static const bool fork_default = false;
#else
    static const bool fork_default = true;
#endif
    static const bool fork_on = [] { const char* e = std::getenv("STRATA_DF2_FORK"); return e && e[0] ? e[0] == '1' : fork_default; }();
    fork_ = fork_on && prof_ == nullptr && cs2_ != nullptr;
    if (fork_) {
        cudaEventRecord(ev_fork_, cs_);
        cudaStreamWaitEvent(cs2_, ev_fork_, 0);
    }
    bool ok = record_context(taps, n, fork_ ? cs2_ : cs_, err);
    if (fork_) cudaEventRecord(ev_join_, cs2_);
    if (ok) ok = record_block(cs_, err);
    if (fork_) cudaStreamWaitEvent(cs_, ev_join_, 0);
    fork_ = false;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    if (ce != cudaSuccess || cudaGraphInstantiate(&sl.exec_[n], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(sl.exec_[n], cs_);
    cudaStreamSynchronize(cs_);
    sl.taps_ = taps;   // this slot's graphs now read `taps`; draft() checks it against this slot's pointer
    return true;
}

bool Dflash2Drafter::capture_body(int n, const float* taps, std::string& err) {
    Slot& sl = cur();
    if (sl.body_exec_[n]) return true;
    if (!df2_graphs_valid_for(sl.taps_, taps)) { err = "dflash2: capture_body: this slot's graphs are captured on another taps buffer"; return false; }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin capture"; return false; }
    kernels::copy_i32_from_mapped(d_meta_, m_meta_ + (size_t) cur_ * 32, 32, cs_);
#if defined(STRATA_USE_HIP)
    static const bool fork_default = false;
#else
    static const bool fork_default = true;
#endif
    const bool fork_on = [] { const char* e = std::getenv("STRATA_DF2_FORK"); return e && e[0] ? e[0] == '1' : fork_default; }();
    fork_ = fork_on && prof_ == nullptr && cs2_ != nullptr;
    if (fork_) {
        cudaEventRecord(ev_fork_, cs_);
        cudaStreamWaitEvent(cs2_, ev_fork_, 0);
    }
    bool ok = record_context(taps, n, fork_ ? cs2_ : cs_, err);
    if (fork_) cudaEventRecord(ev_join_, cs2_);
    if (ok) ok = record_block_body(cs_, err);
    if (fork_) cudaStreamWaitEvent(cs_, ev_join_, 0);
    fork_ = false;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    if (ce != cudaSuccess || cudaGraphInstantiate(&sl.body_exec_[n], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture_body: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(sl.body_exec_[n], cs_);
    cudaStreamSynchronize(cs_);
    sl.taps_ = taps;
    return true;
}

bool Dflash2Drafter::prepare(const float* taps, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (prof_ != nullptr) return true;   // the profile runs without graphs
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int saved_cur = cur_;
    if (slots_.size() > 1 && cur_ == 0) {
        for (int s = 0; s < (int) slots_.size(); ++s) {
            if (!df2_graphs_valid_for(slots_[(size_t) s]->taps_, taps)) continue;
            cur_ = s;
            for (int i = 0; i < 8; ++i) h_meta_[(size_t) s * 32 + i] = i;
            for (int n = 1; n <= kRows; ++n)
                if (!capture_body(n, taps, err)) { cur_ = saved_cur; return false; }
        }
        cur_ = saved_cur;
    } else {
        for (int i = 0; i < 8; ++i) h_meta_[(size_t) cur_ * 32 + i] = i;
        for (int n = 1; n <= kRows; ++n)
            if (!capture(n, taps, err)) return false;
    }
    ms_capture = ms_since(t0);
    return true;
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    Slot& sl = cur();
    const int tap_n = (int) taps_.size() * N_;
    int32_t* hm = h_meta_ + (size_t) cur_ * 32;
    for (int64_t r0 = 0; r0 < n; r0 += kRows) {
        const int m = (int) std::min<int64_t>(kRows, n - r0);
        for (int i = 0; i < m; ++i) hm[i] = (int32_t) (pos0 + r0 + i);
        for (int i = 0; i < m; ++i) mirror_cell(sl, pos0 + r0 + i) = (int32_t) (pos0 + r0 + i);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        kernels::copy_i32_from_mapped(d_meta_, m_meta_ + (size_t) cur_ * 32, 32, cs_);
        if (!record_context(taps + (size_t) r0 * (size_t) tap_n, m, cs_, err)) return false;
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    }
    sl.have_ctx_ = std::max<int64_t>(sl.have_ctx_, pos0 + n);
    return true;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (n_new < 1 || n_new > kRows) { err = "dflash2: draft rows out of range"; return false; }
    const OnDevice on_device(device_);
    Slot& sl = cur();
    const Clock::time_point t0 = Clock::now();
    // the guard is this slot's: its graphs read the buffer this slot captured on (Slot::taps_), so a
    // prepare() on another slot cannot move it out from under them (df2_graphs_valid_for)
    if (!df2_graphs_valid_for(sl.taps_, taps)) { err = "dflash2: the taps buffer moved after capture"; return false; }
    int32_t* hm = h_meta_ + (size_t) cur_ * 32;
    if (!prof_ && !sl.exec_[n_new]) {
        // the staging must hold valid positions while the capture runs the (empty) kernels' parameter checks;
        // an extra slot reaches here on its first draft() and captures its graphs against its own rings
        for (int i = 0; i < 8; ++i) hm[i] = (int32_t) (pos0 + i);
        if (!capture(n_new, taps, err)) return false;   // capture() records this slot's taps_ on success
    }
    const int64_t s = pos0 + n_new;
    for (int i = 0; i < 8; ++i) { hm[i] = (int32_t) (pos0 + std::min(i, n_new - 1)); hm[8 + i] = (int32_t) (s + i); }
    for (int i = 0; i < n_new; ++i) mirror_cell(sl, pos0 + i) = (int32_t) (pos0 + i);
    hm[16] = anchor;
    hm[17] = (int32_t) s;
    hm[24] = anchor;
    for (int i = 1; i < 8; ++i) hm[24 + i] = mask_token_;
    for (int i = 0; i < kDrafts; ++i) { ((volatile int32_t*) h_out_)[i] = -1; }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (timing_) cudaEventRecord(ev_a_, cs_);
    if (prof_) {   // the same round without its graph, an event after every stage
        cudaStreamSynchronize(cs_);
        for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);   // the stamps of context() calls are not a round
        prof_->ev.clear();
        prof_->name.clear();
        kernels::copy_i32_from_mapped(d_meta_, m_meta_ + (size_t) cur_ * 32, 32, cs_);
        stamp("start");
        if (!record_context(taps, n_new, cs_, err) || !record_block(cs_, err)) return false;
        prof_collect();
    } else
    if (cudaGraphLaunch(sl.exec_[n_new], cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    if (timing_) cudaEventRecord(ev_b_, cs_);
    (void) cudaStreamQuery(cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    for (int i = 0; i < kDrafts; ++i) {
        drafts[i] = ((volatile int32_t*) h_out_)[i];
        if (probs) probs[i] = ((volatile float*) h_out_)[64 + i];
    }
    if (timing_) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_a_, ev_b_) == cudaSuccess) ms_gpu += ms;
    }
    sl.have_ctx_ = std::max<int64_t>(sl.have_ctx_, s);
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool Dflash2Drafter::draft_batch_begin(const BatchItem* items, int n_items, float* stage_taps, size_t stage_stride, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (n_items == 0) return true;
    if (items == nullptr || n_items < 0 || n_items > kDf2MaxSlots) { err = "dflash2: invalid batch items"; return false; }
    const OnDevice on_device(device_);
    const int saved_cur = cur_;

    static const bool batch_body_env = [] {
        const char* e = std::getenv("STRATA_DF2_BATCHBODY");
        return e == nullptr || e[0] == '\0' || std::strcmp(e, "0") != 0;
    }();
    int total_rows = 0;
    int total_ctx = 0;
    int ctx_offs[kDf2MaxSlots] = {};
    for (int idx = 0; idx < n_items; ++idx) {
        const BatchItem& it = items[idx];
        if (!df2_slot_ok(it.slot, (int) slots_.size())) { err = "dflash2: batch slot out of range"; return false; }
        if (it.n_new < 1 || it.n_new > kRows) { err = "dflash2: batch draft rows out of range"; return false; }
        if (it.k_want < 1 || it.k_want > kDrafts) { err = "dflash2: batch k_want out of range"; return false; }
        ctx_offs[idx] = total_ctx;
        total_ctx += it.n_new;
        total_rows += it.k_want;
    }
    const bool use_batch_body = batch_body_env && !prof_ && n_items >= 2 && n_items <= 4 && total_ctx <= kRows * 4;
    if (!use_batch_body) {
        for (int idx = 0; idx < n_items; ++idx) {
            const BatchItem& it = items[idx];
            const float* cap_taps = stage_taps != nullptr ? stage_taps : it.taps;
            Slot& sl = *slots_[(size_t) it.slot];
            if (!df2_graphs_valid_for(sl.taps_, cap_taps)) { err = "dflash2: the taps buffer moved after capture"; return false; }
            if (!prof_ && !sl.body_exec_[it.n_new]) {
                cur_ = it.slot;
                int32_t* hm = h_meta_ + (size_t) it.slot * 32;
                for (int i = 0; i < 8; ++i) hm[i] = (int32_t) (it.pos0 + i);
                if (!capture_body(it.n_new, cap_taps, err)) { cur_ = saved_cur; return false; }
            }
        }
    }

    int row_off = 0;
    int32_t* bmeta = h_meta_ + (size_t) kDf2MaxSlots * 32;
    const size_t bbody_off = (size_t) kDf2MaxSlots * 32 + (size_t) kDf2MaxSlots * 4;
    int32_t* hb = h_meta_ + bbody_off;
    for (int idx = 0; idx < n_items; ++idx) {
        const BatchItem& it = items[idx];
        Slot& sl = *slots_[(size_t) it.slot];
        int32_t* hm = h_meta_ + (size_t) it.slot * 32;
        const int64_t s = it.pos0 + it.n_new;
        for (int i = 0; i < 8; ++i) {
            hm[i] = (int32_t) (it.pos0 + std::min(i, it.n_new - 1));
            hm[8 + i] = (int32_t) (s + i);
        }
        for (int i = 0; i < it.n_new; ++i) {
            mirror_cell(sl, it.pos0 + i) = (int32_t) (it.pos0 + i);
            if (use_batch_body) hb[ctx_offs[idx] + i] = (int32_t) (it.pos0 + i);
        }
        if (use_batch_body) {
            for (int i = 0; i < 8; ++i) {
                hb[32 + idx * 8 + i] = (int32_t) (s + i);
                hb[64 + idx * 8 + i] = i == 0 ? it.anchor : mask_token_;
            }
            hb[96 + idx] = (int32_t) s;
        }
        hm[16] = it.anchor;
        hm[17] = (int32_t) s;
        hm[24] = it.anchor;
        for (int i = 1; i < 8; ++i) hm[24 + i] = mask_token_;
        bmeta[idx * 4 + 0] = row_off;
        bmeta[idx * 4 + 1] = it.k_want;
        bmeta[idx * 4 + 2] = it.anchor;
        bmeta[idx * 4 + 3] = 0;
        for (int i = 0; i < it.k_want; ++i) ((volatile int32_t*) h_out_)[idx * 8 + i] = -1;
        row_off += it.k_want;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);

    if (timing_) cudaEventRecord(ev_a_, cs_);
    if (use_batch_body) {
        try {
            using namespace kernels;
            const int tap_n = (int) taps_.size() * N_;
            const int total_B = n_items * kRows;
            int32_t* d_bctx_pos = d_meta_ + 32;
            int32_t* d_bblk_pos = d_meta_ + 64;
            int32_t* d_bblk_tok = d_meta_ + 96;
            int32_t* d_banchor_pos = d_meta_ + 128;
            copy_i32_from_mapped(d_meta_ + 32, m_meta_ + bbody_off, 104, cs_);

            if (!skip_part("ctx")) {
                df2_rope_table(d_bctx_pos, total_ctx, theta_, rt_ctx_, cs_);
                for (int idx = 0; idx < n_items; ++idx) {
                    const BatchItem& it = items[idx];
                    const float* src_taps = stage_taps != nullptr ? stage_taps : it.taps;
                    if (stage_taps != nullptr && it.taps != stage_taps)
                        copy_from_mapped(stage_taps, it.taps, (int64_t) ((size_t) it.n_new * stage_stride), cs_);
                    df2_quant_rows(src_taps, tap_n, it.n_new,
                                   xqc_ + (size_t) ctx_offs[idx] * tap_n,
                                   xdc_ + (size_t) ctx_offs[idx] * (tap_n / 32), cs_);
                }
                gemm({"fc.weight"}, {fcy_}, total_ctx, cs_, xqc_, xdc_);
                df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, total_ctx, N_, eps_, xqc_, xdc_, cs_);
                for (int l = 0; l < L_; ++l) {
                    const std::string p = "layers." + std::to_string(l) + ".";
                    const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
                    gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, total_ctx, cs_, xqc_, xdc_);
                    const float* kn_w = f32(p + "self_attn.k_norm.weight");
                    for (int idx = 0; idx < n_items; ++idx) {
                        Slot& sl = *slots_[(size_t) items[idx].slot];
                        const int c0 = ctx_offs[idx], m = items[idx].n_new;
                        if (fp16_ring_)
                            df2_kv_ring_f16(kc_ + (size_t) c0 * NKV_ * HD_, vc_ + (size_t) c0 * NKV_ * HD_,
                                            kn_w, d_bctx_pos + c0, rt_ctx_ + (size_t) c0 * 128,
                                            sl.kring_[l], sl.vring_[l], ring_slots_, m, NKV_, eps_, cs_);
                        else
                            df2_kv_ring(kc_ + (size_t) c0 * NKV_ * HD_, vc_ + (size_t) c0 * NKV_ * HD_,
                                        kn_w, d_bctx_pos + c0, rt_ctx_ + (size_t) c0 * 128,
                                        sl.kring_[l], sl.vring_[l], ring_slots_, m, NKV_, eps_, cs_);
                    }
                }
            }

            df2_rope_table(d_bblk_pos, total_B, theta_, rt_blk_, cs_);
            embed_(d_bblk_tok, total_B, h_, cs_);
            df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, total_B, N_, eps_, xqp_, xdp_, cs_);
            for (int l = 0; l < L_; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".";
                const float* base_a = f32(p + "attention_conv.base_kernel");
                const float* base_m = f32(p + "mlp_conv.base_kernel");
                const std::string akp = p + "attention_conv.kernel_projection.weight", mkp = p + "mlp_conv.kernel_projection.weight";
                const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
                const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
                if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, total_B, cs_, xqp_, xdp_);
                if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, total_B, N_, group_, xqp_, xdp_, cs_);
                if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, total_B, cs_, xqp_, xdp_);
                if (!skip_part("small"))
                    df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"),
                                     rt_blk_, qn_, kblk_, total_B, NH_, NKV_, eps_, cs_);
                if (!skip_part("attn")) {
                    for (int idx = 0; idx < n_items; ++idx) {
                        Slot& sl = *slots_[(size_t) items[idx].slot];
                        const size_t b0 = (size_t) idx * kRows;
                        if (fp16_ring_)
                            df2_attention_f16(qn_ + b0 * NH_ * HD_, sl.kring_[l], sl.vring_[l],
                                              kblk_ + b0 * NKV_ * HD_, v_ + b0 * NKV_ * HD_, d_banchor_pos + idx,
                                              ring_slots_, window_, part_, xqp_ + b0 * NH_ * HD_,
                                              xdp_ + b0 * (NH_ * HD_ / 32), cs_);
                        else
                            df2_attention(qn_ + b0 * NH_ * HD_, sl.kring_[l], sl.vring_[l],
                                          kblk_ + b0 * NKV_ * HD_, v_ + b0 * NKV_ * HD_, d_banchor_pos + idx,
                                          ring_slots_, window_, part_, xqp_ + b0 * NH_ * HD_,
                                          xdp_ + b0 * (NH_ * HD_ / 32), cs_);
                    }
                }
                if (!skip_part("qkvo")) gemm({(p + "self_attn.o_proj.weight").c_str()}, {o_}, total_B, cs_, xqp_, xdp_);
                if (!skip_part("small"))
                    df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"),
                                       xn_, total_B, N_, group_, eps_, xqp_, xdp_, cs_);
                if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, total_B, cs_, xqp_, xdp_);
                if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, total_B, N_, group_, xqp_, xdp_, cs_);
                if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, total_B, cs_, xqp_, xdp_);
                if (!skip_part("small")) df2_swiglu_quant(gate_, up_, total_B, I_, xqp_, xdp_, cs_);
                if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, total_B, cs_, xqp_, xdp_);
                const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
                if (!skip_part("small"))
                    df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma),
                                       l + 1 < L_ ? xn_ : fn_, total_B, N_, group_, eps_, xqp_, xdp_, cs_);
            }
            gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, total_B, cs_, xqp_, xdp_);
            row_off = 0;
            for (int idx = 0; idx < n_items; ++idx) {
                const BatchItem& it = items[idx];
                const size_t b1 = (size_t) idx * kRows + 1;
                df2_pack_head_rows(fn_ + b1 * N_, hsel_ + b1 * rank_,
                                   fn_batch_ + (size_t) row_off * N_, hsel_batch_ + (size_t) row_off * rank_,
                                   it.k_want, N_, rank_, cs_);
                row_off += it.k_want;
            }
        } catch (const std::exception& e) {
            cur_ = saved_cur;
            err = e.what();
            return false;
        }
    } else {
        row_off = 0;
        for (int idx = 0; idx < n_items; ++idx) {
            const BatchItem& it = items[idx];
            Slot& sl = *slots_[(size_t) it.slot];
            const float* cap_taps = stage_taps != nullptr ? stage_taps : it.taps;
            if (stage_taps != nullptr && it.taps != stage_taps) {
                try {
                    kernels::copy_from_mapped(stage_taps, it.taps, (int64_t) ((size_t) it.n_new * stage_stride), cs_);
                } catch (const std::exception& e) {
                    cur_ = saved_cur;
                    err = e.what();
                    return false;
                }
            }
            cur_ = it.slot;
            if (prof_) {
                kernels::copy_i32_from_mapped(d_meta_, m_meta_ + (size_t) it.slot * 32, 32, cs_);
                if (!record_context(cap_taps, it.n_new, cs_, err) || !record_block_body(cs_, err)) {
                    cur_ = saved_cur;
                    return false;
                }
            } else if (cudaGraphLaunch(sl.body_exec_[it.n_new], cs_) != cudaSuccess) {
                cur_ = saved_cur;
                err = std::string("dflash2 draft_batch: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            try {
                kernels::df2_pack_head_rows(fn_ + N_, hsel_, fn_batch_ + (size_t) row_off * N_,
                                            hsel_batch_ + (size_t) row_off * rank_, it.k_want, N_, rank_, cs_);
            } catch (const std::exception& e) {
                cur_ = saved_cur;
                err = e.what();
                return false;
            }
            row_off += it.k_want;
        }
        cur_ = saved_cur;
    }

    try {
        using namespace kernels;
        static const int head_il = [] {
            const char* e = std::getenv("STRATA_DFLASH_HEAD_IL");
            return e && e[0] ? (e[0] != '0' ? 1 : 0) : -1;
        }();
        for (int c0 = 0; c0 < total_rows; c0 += kDrafts) {
            const int nc = std::min((int) kDrafts, total_rows - c0);
            native_quantize_q8_1(fn_batch_ + (size_t) c0 * N_, xq_, N_, nc, cs_);
            if (!skip_part("head")) {
                bool use_il = (head_il != 0) && native_mmvq_il_supported(head_type_, nc, (int) n_dvocab_);
                bool tuned_il = false;
                if (!use_il && head_il != 0) {
                    native_mmvq_il_tune(4);
                    use_il = native_mmvq_il_supported(head_type_, nc, (int) n_dvocab_);
                    tuned_il = use_il;
                    if (!use_il) native_mmvq_il_tune(0);
                }
                if (use_il) {
                    uint8_t* xil = xq_ + native_q8_1_bytes(N_, kRows);
                    if (nc >= 2) native_q8_1_interleave(xq_, xil, N_, nc, cs_);
                    native_mmvq_il(head_type_, head_w_, xq_, xil, logits_, N_, (int) n_dvocab_, nc, cs_);
                    if (tuned_il) native_mmvq_il_tune(0);
                } else {
                    native_mmvq(head_type_, head_w_, xq_, logits_, N_, (int) n_dvocab_, nc, cs_);
                }
            }
            df2_topk(logits_, (int) n_dvocab_, dvocab_, nc, topscratch_,
                     topval_ + (size_t) c0 * 16, toptok_ + (size_t) c0 * 16, cs_);
        }
        df2_select_multi(hsel_batch_, topval_, toptok_,
                         reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
                         reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off),
                         rank_, m_meta_ + (size_t) kDf2MaxSlots * 32, n_items, out_tok_, out_prob_, cs_);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    if (timing_) cudaEventRecord(ev_b_, cs_);
    (void) cudaStreamQuery(cs_);
    return true;
}

bool Dflash2Drafter::draft_batch_end(const BatchItem* items, int n_items, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (n_items == 0) return true;
    const OnDevice on_device(device_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("dflash2 draft_batch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    for (int idx = 0; idx < n_items; ++idx) {
        const BatchItem& it = items[idx];
        for (int i = 0; i < it.k_want; ++i) {
            it.drafts[i] = ((volatile int32_t*) h_out_)[idx * 8 + i];
            if (it.probs) it.probs[i] = ((volatile float*) h_out_)[64 + idx * 8 + i];
        }
        Slot& sl = *slots_[(size_t) it.slot];
        sl.have_ctx_ = std::max<int64_t>(sl.have_ctx_, it.pos0 + it.n_new);
    }
    if (timing_) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_a_, ev_b_) == cudaSuccess) ms_gpu += ms;
    }
    rounds += n_items;
    return true;
}

bool Dflash2Drafter::draft_batch(const BatchItem* items, int n_items, float* stage_taps, size_t stage_stride, std::string& err) {
    const Clock::time_point t0 = Clock::now();
    if (!draft_batch_begin(items, n_items, stage_taps, stage_stride, err) ||
        !draft_batch_end(items, n_items, err))
        return false;
    ms_draft += ms_since(t0);
    return true;
}

bool Dflash2Drafter::debug_rows(std::vector<float>& rows, std::string& err) {
    rows.resize((size_t) kRows * (size_t) N_);
    if (cudaMemcpy(rows.data(), fn_, rows.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug copy"; return false; }
    return true;
}

// ---- the drafter's context: park/restore ----------------------------------------------------------
// The blob: a 128-byte self-describing header, the kRing slot positions, then L layers of planar K/V
// in the device's own layout. Everything a restore needs to refuse a foreign drafter: geometry ints,
// the pack's path hash / size / mtime, and the slot table that makes a resumed checkpoint's rewind
// safe (cells written past the resumed prefix are zeroed, the drafter keeps the cells below it).
namespace {
#pragma pack(push, 1)
struct Df2StateHeader {
    uint32_t magic, version;
    int64_t L, NKV, HD, ring, window, N, NH;
    int64_t have;             // last committed position (exclusive) at save time
    int64_t prefix;           // conversation length the blob was parked at
    uint32_t theta_bits, eps_bits;
    uint64_t pack_size;
    int64_t pack_mtime;
    uint64_t pack_path_hash;
    uint64_t ring_bytes;      // L * 2 * ring * NKV * HD * 4
    uint8_t reserved[8];
};
#pragma pack(pop)
static_assert(sizeof(Df2StateHeader) == 128, "df2 state header is the blob's fixed contract");

uint64_t df2_path_hash(const std::string& s) {   // FNV-1a-64
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}
}  // namespace

// save/restore always mean the active slot: the blob format (kStateVersion 1) describes one conversation's rings,
// and a parked conversation restores into the slot the engine is about to serve.
bool Dflash2Drafter::save_state(std::vector<uint8_t>& out, std::string& err, int64_t conversation_len) {
    Slot& sl = cur();
    if (blob_ == nullptr || sl.kring_[0] == nullptr) { err = "dflash2: save_state on an un-loaded drafter"; return false; }
    const OnDevice on_device(device_);
    const size_t elem_bytes = ring_elem_bytes();
    const size_t slice = (size_t) ring_slots_ * NKV_ * HD_ * elem_bytes;
    const size_t rings = (size_t) L_ * 2 * slice;
    out.clear();
    try { out.resize(128 + (size_t) ring_slots_ * 4 + rings); }
    catch (const std::bad_alloc&) { err = "dflash2: save_state out of host memory"; return false; }
    Df2StateHeader h{};
    h.magic = kStateMagic; h.version = kStateVersion;
    h.L = L_; h.NKV = NKV_; h.HD = HD_; h.ring = ring_slots_; h.window = window_; h.N = N_; h.NH = NH_;
    h.have = sl.have_ctx_;
    h.prefix = conversation_len < 0 ? sl.have_ctx_ : conversation_len;
    // No equality check against have_ctx_: in serve mode the drafter's context is rebuilt per request
    // from the cache-NEW rows only (the ring holds the window around the tail at absolute positions),
    // so the conversation is always longer than the rows this drafter has seen. restore_state() zeroes
    // slots at positions >= the resumed checkpoint, which is what makes a rewound branch correct.
    std::memcpy(&h.theta_bits, &theta_, 4);
    std::memcpy(&h.eps_bits, &eps_, 4);
    h.pack_size = pack_size_; h.pack_mtime = pack_mtime_; h.pack_path_hash = df2_path_hash(pack_path_);
    h.ring_bytes = rings;
    std::memcpy(out.data(), &h, 128);
    assert(sl.ring_pos_.size() >= (size_t) ring_slots_);   // the mirror is ring_slots entries (df2_ring_mirror_entries); copying more would overrun it
    std::memcpy(out.data() + 128, sl.ring_pos_.data(), (size_t) ring_slots_ * 4);
    uint8_t* p = out.data() + 128 + (size_t) ring_slots_ * 4;
    if (cs_ != nullptr && cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("dflash2: save_state stream: ") + cudaGetErrorString(cudaGetLastError());
        out.clear();
        return false;
    }
    for (int l = 0; l < L_; ++l)
        for (int kv = 0; kv < 2; ++kv) {
            const void* src = kv ? sl.vring_[l] : sl.kring_[l];
            if (cudaMemcpy(p, src, slice, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = std::string("dflash2: save_state ring copy: ") + cudaGetErrorString(cudaGetLastError());
                out.clear();
                return false;
            }
            p += slice;
        }
    return true;
}

bool Dflash2Drafter::restore_state(const std::vector<uint8_t>& in, int64_t prefix_len, std::string& err) {
    const auto refuse = [&](const char* why) { err = std::string("dflash2: restore_state: ") + why; return false; };
    Slot& sl = cur();
    if (blob_ == nullptr || sl.kring_[0] == nullptr) return refuse("the drafter is not loaded");
    if (in.size() < 128 + (size_t) ring_slots_ * 4) return refuse("the blob is short");
    Df2StateHeader h;
    std::memcpy(&h, in.data(), 128);
    if (h.magic != kStateMagic) return refuse("not a DFlash2 context blob");
    if (h.version != kStateVersion) return refuse("blob version is not this engine's");
    if (h.L != L_ || h.NKV != NKV_ || h.HD != HD_ || h.ring != ring_slots_ || h.window != window_ ||
        h.N != N_ || h.NH != NH_) return refuse("the drafter's geometry changed");
    float ts = 0, es = 0;
    std::memcpy(&ts, &h.theta_bits, 4); std::memcpy(&es, &h.eps_bits, 4);
    if (ts != theta_ || es != eps_) return refuse("the drafter's RoPE/eps changed");
    std::error_code ec;
    const std::filesystem::path p(pack_path_);
    const uint64_t size_now = (uint64_t) std::filesystem::file_size(p, ec);
    if (ec || size_now != h.pack_size) return refuse("the drafter pack changed size");
    const auto wt = std::filesystem::last_write_time(p, ec);
    if (ec || (int64_t) wt.time_since_epoch().count() != h.pack_mtime) return refuse("the drafter pack changed");
    if (h.pack_path_hash != df2_path_hash(pack_path_)) return refuse("the drafter pack moved");
    const size_t elem_bytes = ring_elem_bytes();
    const size_t slice = (size_t) ring_slots_ * NKV_ * HD_ * elem_bytes;
    const size_t rings = (size_t) L_ * 2 * slice;
    if (h.ring_bytes != rings) return refuse("the blob's ring size disagrees with the header");
    if (in.size() != 128 + (size_t) ring_slots_ * 4 + rings) return refuse("the blob's size disagrees with the header");
    if (h.have < 0 || h.prefix < h.have || prefix_len < 0 || prefix_len > h.prefix)
        return refuse("the saved/resumed positions are out of range");
    // Serve parks stamp h.prefix with the CONVERSATION length while the drafter has seen only the
    // cache-new rows of the request that parked it (have << prefix), and the ring slots hold
    // absolute conversation positions. Every range check therefore goes against h.prefix.
    if (sl.ring_pos_.size() < (size_t) ring_slots_) return refuse("the drafter's host ring mirror is not sized for its ring");
    const int32_t* table = reinterpret_cast<const int32_t*>(in.data() + 128);
    for (int s = 0; s < ring_slots_; ++s)
        if (table[s] < -1 || table[s] >= h.prefix || (table[s] >= 0 && table[s] % ring_slots_ != s))
            return refuse("the ring slot table is invalid");
    // Rewind: cells this park committed past the resumed checkpoint go; the drafter must not draft
    // from tokens of a branch the conversation did not take.
    const uint8_t* src = in.data() + 128 + (size_t) ring_slots_ * 4;
    std::vector<uint8_t> staging;
    try { staging.resize(slice); }
    catch (const std::bad_alloc&) { return refuse("out of host memory"); }
    const OnDevice on_device(device_);
    for (int l = 0; l < L_; ++l)
        for (int kv = 0; kv < 2; ++kv) {
            const uint8_t* slab = src + ((size_t) l * 2 + kv) * slice;
            std::memcpy(staging.data(), slab, slice);
            // The ring is head-major ([NKV][kRing][HD], see kv_ring_kernel): zero every head's cell for
            // slots that were never written or were written at positions >= prefix_len.
            for (int s = 0; s < ring_slots_; ++s)
                if (table[s] < 0 || table[s] >= prefix_len)
                    for (int hk = 0; hk < NKV_; ++hk)
                        std::memset(staging.data() + ((size_t) hk * ring_slots_ + s) * (size_t) HD_ * elem_bytes, 0, (size_t) HD_ * elem_bytes);
            void* dst = kv ? sl.vring_[l] : sl.kring_[l];
            if (cudaMemcpy(dst, staging.data(), slice, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = std::string("dflash2: restore_state ring copy: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
    for (int s = 0; s < ring_slots_; ++s) sl.ring_pos_[(size_t) s] = (table[s] >= 0 && table[s] < prefix_len) ? table[s] : -1;
    sl.have_ctx_ = prefix_len;
    err.clear();
    return true;
}

bool Dflash2Drafter::copy_slot(int from_slot, int to_slot, int64_t prefix_len, std::string& err) {
    if (!bound_) { err = "dflash2: copy_slot before load"; return false; }
    if (from_slot < 0 || from_slot >= (int) slots_.size() || to_slot < 0 || to_slot >= (int) slots_.size()) {
        err = "dflash2: copy_slot: slot out of range";
        return false;
    }
    const OnDevice on_device(device_);
    const Slot& src = *slots_[(size_t) from_slot];
    Slot& dst = *slots_[(size_t) to_slot];
    const int64_t end = prefix_len >= 0 ? std::min(prefix_len, src.have_ctx_) : src.have_ctx_;
    const int64_t begin = std::max<int64_t>(0, end - ring_slots_);
    if (from_slot == to_slot) {
        const int saved = cur_;
        cur_ = to_slot;
        rewind(end);
        cur_ = saved;
        return true;
    }
    const size_t eb = ring_elem_bytes();
    const size_t slot_bytes = (size_t) HD_ * eb;
    const size_t head_stride = (size_t) ring_slots_ * slot_bytes;
    const size_t ring_bytes = (size_t) NKV_ * head_stride;

    bool any_dropped = false;
    dst.ring_pos_.assign((size_t) ring_slots_, -1);
    for (int s = 0; s < ring_slots_; ++s) {
        const int64_t p = src.ring_pos_[(size_t) s];
        if (p >= begin && p < end && end > begin) {
            dst.ring_pos_[(size_t) s] = p;
        } else if (p >= 0) {
            any_dropped = true;
        }
    }

    for (int l = 0; l < L_; ++l) {
        if (end <= begin) {
            (void) cudaMemsetAsync(dst.kring_[l], 0, ring_bytes, cs_);
            (void) cudaMemsetAsync(dst.vring_[l], 0, ring_bytes, cs_);
            continue;
        }
        if (!any_dropped) {
            (void) cudaMemcpyAsync(dst.kring_[l], src.kring_[l], ring_bytes, cudaMemcpyDeviceToDevice, cs_);
            (void) cudaMemcpyAsync(dst.vring_[l], src.vring_[l], ring_bytes, cudaMemcpyDeviceToDevice, cs_);
            continue;
        }
        (void) cudaMemsetAsync(dst.kring_[l], 0, ring_bytes, cs_);
        (void) cudaMemsetAsync(dst.vring_[l], 0, ring_bytes, cs_);
        for (int s0 = 0; s0 < ring_slots_;) {
            if (dst.ring_pos_[(size_t) s0] < 0) { ++s0; continue; }
            int s1 = s0 + 1;
            while (s1 < ring_slots_ && dst.ring_pos_[(size_t) s1] >= 0) ++s1;
            const size_t off = (size_t) s0 * slot_bytes;
            const size_t width = (size_t) (s1 - s0) * slot_bytes;
            (void) cudaMemcpy2DAsync((uint8_t*) dst.kring_[l] + off, head_stride,
                                     (const uint8_t*) src.kring_[l] + off, head_stride,
                                     width, (size_t) NKV_, cudaMemcpyDeviceToDevice, cs_);
            (void) cudaMemcpy2DAsync((uint8_t*) dst.vring_[l] + off, head_stride,
                                     (const uint8_t*) src.vring_[l] + off, head_stride,
                                     width, (size_t) NKV_, cudaMemcpyDeviceToDevice, cs_);
            s0 = s1;
        }
    }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "dflash2: copy_slot: device copy failed";
        return false;
    }
    dst.have_ctx_ = prefix_len >= 0 ? prefix_len : end;
    err.clear();
    return true;
}

void Dflash2Drafter::reset() {
    Slot& sl = cur();   // a new request on the active slot; the other slots keep their conversations
    sl.have_ctx_ = 0;
    for (int32_t& p : sl.ring_pos_) p = -1;
    if (device_ >= 0 && sl.kring_[0] != nullptr) {
        const OnDevice on_device(device_);
        const size_t bytes = (size_t) ring_slots_ * NKV_ * HD_ * ring_elem_bytes();
        for (int l = 0; l < L_; ++l) {
            cudaMemsetAsync(sl.kring_[l], 0, bytes, cs_);
            cudaMemsetAsync(sl.vring_[l], 0, bytes, cs_);
        }
        if (cs_) cudaStreamSynchronize(cs_);
    }
}

void Dflash2Drafter::rewind(int64_t prefix_len) {
    if (prefix_len <= 0) { reset(); return; }
    Slot& sl = cur();
    if (device_ >= 0 && sl.kring_[0] != nullptr) {
        const OnDevice on_device(device_);
        assert(sl.ring_pos_.size() >= (size_t) ring_slots_);   // the scan below reads ring_slots entries
        const size_t elem_bytes = ring_elem_bytes();
        bool any = false;
        for (int s0 = 0; s0 < ring_slots_;) {
            if (sl.ring_pos_[(size_t) s0] < prefix_len) { ++s0; continue; }
            int s1 = s0;
            while (s1 < ring_slots_ && sl.ring_pos_[(size_t) s1] >= prefix_len) {
                sl.ring_pos_[(size_t) s1] = -1;
                ++s1;
            }
            const size_t span_bytes = (size_t) (s1 - s0) * (size_t) HD_ * elem_bytes;
            for (int l = 0; l < L_; ++l)
                for (int hk = 0; hk < NKV_; ++hk) {
                    const size_t off_bytes = ((size_t) hk * ring_slots_ + s0) * (size_t) HD_ * elem_bytes;
                    cudaMemsetAsync(reinterpret_cast<uint8_t*>(sl.kring_[l]) + off_bytes, 0, span_bytes, cs_);
                    cudaMemsetAsync(reinterpret_cast<uint8_t*>(sl.vring_[l]) + off_bytes, 0, span_bytes, cs_);
                }
            any = true;
            s0 = s1;
        }
        if (any && cs_) cudaStreamSynchronize(cs_);
    } else {
        for (int32_t& p : sl.ring_pos_)
            if (p >= prefix_len) p = -1;
    }
    if (sl.have_ctx_ > prefix_len) sl.have_ctx_ = prefix_len;
}

}  // namespace strata::core
