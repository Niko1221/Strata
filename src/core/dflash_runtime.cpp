// src/core/dflash_runtime.cpp - the DFlashDrafter's device half: weights, K/V pools, the fusion
// and the block forward (the artifact loader lives in dflash.cpp).  Semantics: docs/DFLASH.md.
#include "strata/core/dflash.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/native_rope.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ============================================================================================
// The runtime: weights, the drafter's own K/V pools, the fusion (context cells) and the block
// forward.  Eager, one stream, greedy.  All semantics: docs/DFLASH.md.
// ============================================================================================

namespace strata::core {

namespace {

constexpr float kEps = 1e-6f;   // the architecture's rms_norm eps (the config's rms_norm_eps)

// The stage-parity fixture (STRATA_DF_PARITY=<dir>): when the drafter's cycle counter reaches
// STRATA_DF_PARITY_CYCLE (default 1), every named stage is written as [u32 float count][f32
// payload] into <dir>/<name>.bin for tools/dflash_stage_parity.py.  No effect otherwise.
int parity_want(int64_t cycle) {
    static const char* dir = std::getenv("STRATA_DF_PARITY");
    static const int64_t want = [] {
        const char* c = std::getenv("STRATA_DF_PARITY_CYCLE");
        return c ? std::atoll(c) : 1;
    }();
    return dir && cycle == want;
}

void parity_dump_u16_as_f32(const char* dir, const char* name, const void* dev, int64_t n, cudaStream_t cs) {
    std::vector<uint16_t> host((size_t) n);
    if (cudaMemcpyAsync(host.data(), dev, (size_t) n * 2, cudaMemcpyDeviceToHost, cs) != cudaSuccess) return;
    cudaStreamSynchronize(cs);
    std::vector<float> wide((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        // TRUE fp16 -> f32 (re-bias the exponent): the bits<<16 shift is the BF16 widening and
        // turns every dumped pool into plausible-looking nonsense
        const __half h = __ushort_as_half(host[(size_t) i]);
        wide[(size_t) i] = __half2float(h);
    }
    char path[600];
    std::snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    if (FILE* f = std::fopen(path, "wb")) {
        const uint32_t cnt = (uint32_t) n;
        std::fwrite(&cnt, 4, 1, f);
        std::fwrite(wide.data(), 4, (size_t) n, f);
        std::fclose(f);
        std::fprintf(stderr, "dflash parity: dumped %s (fp16->f32, %lld)\n", name, (long long) n);
    }
}

void parity_dump(const char* dir, const char* name, const void* dev, int64_t floats, cudaStream_t cs) {
    std::vector<float> host((size_t) floats);
    if (cudaMemcpyAsync(host.data(), dev, (size_t) floats * 4, cudaMemcpyDeviceToHost, cs) != cudaSuccess) return;
    cudaStreamSynchronize(cs);
    char path[600];
    std::snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    const uint32_t n = (uint32_t) floats;
    std::fwrite(&n, 4, 1, f);
    std::fwrite(host.data(), 4, (size_t) floats, f);
    std::fclose(f);
    std::fprintf(stderr, "dflash parity: dumped %s (%lld floats)\n", name, (long long) floats);
}

uint64_t mapped_bytes(int64_t n) { return ((uint64_t) n + 63) & ~uint64_t(63); }

// ------------------------------------------------------------------------------------
// STRATA_DF_EVENTS=1: per-section GPU time (cudaEvent pairs on the drafter's stream) and
// wall time, accumulated across cycles and dumped by release().  Off: one static-bool
// branch per section, no events, no clocks.  One drafter per process today (generate.cpp),
// so the accumulators are file-local.
struct DFlashTiming {
    struct Acc { double gpu_ms = 0, wall_ms = 0; int64_t n = 0; };
    bool on = false;
    bool inited = false;
    std::vector<cudaEvent_t> ev;                       // pairs, reused every cycle
    size_t ev_used = 0;                                // next free event index (even = starts)
    std::map<std::string, Acc> acc;                    // section -> totals
    std::vector<std::pair<std::string, size_t>> open;  // started sections awaiting their GPU time
    void init() {
        if (inited) return;
        inited = true;
        ev.resize(512);   // 256 sections per cycle is well above propose()'s ~80 + fusion's ~40
        for (cudaEvent_t& e : ev) {
            if (cudaEventCreateWithFlags(&e, 0) != cudaSuccess) { ev.resize(0); return; }   // timing events
        }
    }
};

DFlashTiming& df_timing() {
    static DFlashTiming t;
    return t;
}

/// A timed region of the drafter's stream: the start event is recorded when the guard opens,
/// the end event when it closes, so everything launched inside lands between them (the stream
/// is serial) and the pair's elapsed time is the region's GPU time.
struct DFlashSection {
    DFlashSection(const char* name) : name_(name) {
        DFlashTiming& t = df_timing();
        if (!t.on) return;
        if (t.ev_used + 2 > t.ev.size()) return;   // pool exhausted: drop the section, never block
        idx_ = t.ev_used;
        cudaEventRecord(t.ev[idx_], df_stream());
        t.open.emplace_back(name, idx_);
        t.ev_used += 2;
        wall0_ = std::chrono::steady_clock::now();
    }
    ~DFlashSection() {
        if (idx_ == SIZE_MAX) return;
        DFlashTiming& t = df_timing();
        cudaEventRecord(t.ev[idx_ + 1], df_stream());
        DFlashTiming::Acc& a = t.acc[name_];
        a.wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0_).count();
        a.n += 1;
    }
    static cudaStream_t& df_stream();   // the drafter's stream (set by upload)
    const char* name_;
    size_t idx_ = SIZE_MAX;   // the pair's first event; SIZE_MAX = timing off / pool exhausted
    std::chrono::steady_clock::time_point wall0_{};
};

/// The GPU half of the sections opened since the last drain: after a full sync every recorded
/// event pair is complete, so fold the elapsed times in and recycle the pool.
void df_timing_drain() {
    DFlashTiming& t = df_timing();
    if (!t.on || !t.inited) return;
    for (const auto& [name, idx] : t.open) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, t.ev[idx], t.ev[idx + 1]) == cudaSuccess) t.acc[name].gpu_ms += ms;
    }
    t.open.clear();
    t.ev_used = 0;
}

void df_timing_dump() {
    DFlashTiming& t = df_timing();
    if (!t.on) return;
    std::fprintf(stderr, "dflash timing (STRATA_DF_EVENTS=1), section totals: %18s %10s %10s %8s\n",
                 "gpu ms", "wall ms", "gpu/wall", "calls");
    double gpu_sum = 0, wall_sum = 0;
    for (auto& [name, a] : t.acc) { gpu_sum += a.gpu_ms; wall_sum += a.wall_ms; }
    for (auto& [name, a] : t.acc) {
        std::fprintf(stderr, "  %-28s %10.2f %10.2f %8.3f %8lld\n", name.c_str(), a.gpu_ms, a.wall_ms,
                     a.wall_ms > 0 ? a.gpu_ms / a.wall_ms : 0.0, (long long) a.n);
    }
    std::fprintf(stderr, "  %-28s %10.2f %10.2f\n", "SUM (not decode wall)", gpu_sum, wall_sum);
}

cudaStream_t& DFlashSection::df_stream() {
    static cudaStream_t s = nullptr;   // one drafter per process; upload() sets it
    return s;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}


/// Mapped pinned host memory + its device alias (the MTP staging's shape, local here).
bool dflash_mapped(int64_t n, void** host, void** dev) {
    void* h = nullptr;
    if (cudaHostAlloc(&h, mapped_bytes(n), cudaHostAllocMapped) != cudaSuccess) return false;
    if (cudaHostGetDevicePointer(dev, h, 0) != cudaSuccess) { cudaFreeHost(h); return false; }
    *host = h;
    return true;
}

}  // namespace

void DFlashDrafter::release() {
    df_timing_dump();
    df_timing().acc.clear();
    df_timing().open.clear();
    df_timing().ev_used = 0;
    if (device_ >= 0) cudaSetDevice(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    auto free_dev = [](void* p) { if (p) cudaFree(p); };
    free_dev(w_);
    for (auto& [name, p] : wf_) free_dev((void*) p);
    for (const auto& st : st_) {
        if (st.host_step) cudaFreeHost(st.host_step);   // qsa_state_init allocates one pair per state
        if (st.host_pos) cudaFreeHost(st.host_pos);
        if (st.owns_rope) {                             // defensive: the drafter borrows the session's tables
            if (st.cos_tab) cudaFree(st.cos_tab);
            if (st.sin_tab) cudaFree(st.sin_tab);
        }
    }
    for (void* a : arenas_) free_dev(a);
    free_dev(tok_); free_dev(step_); free_dev(pos_); free_dev(poskv_); free_dev(attn_step_);
    free_dev(tapin_); free_dev(xn16_); free_dev(attn16_);
    free_dev(tapf_); free_dev(emb_); free_dev(h_); free_dev(xn_); free_dev(ctx_);
    free_dev(q_); free_dev(kc_); free_dev(vc_); free_dev(attn_); free_dev(bo_);
    free_dev(gate_); free_dev(up_); free_dev(logits_);
    free_dev(xq_); free_dev(attn_scratch_); free_dev(arg_scratch_);
    if (h_out_) cudaFreeHost(h_out_);
    if (h_tok_) cudaFreeHost(h_tok_);
    delete owned_head_; owned_head_ = nullptr;
    if (cs_) cudaStreamDestroy(cs_);
    w_ = nullptr; wf_.clear(); wt_.clear();
    st_.clear(); arenas_.clear();
    tok_ = step_ = pos_ = poskv_ = attn_step_ = nullptr;
    tapin_ = xn16_ = attn16_ = nullptr;
    tapf_ = emb_ = h_ = xn_ = ctx_ = nullptr;
    q_ = kc_ = vc_ = attn_ = bo_ = nullptr;
    gate_ = up_ = logits_ = nullptr;
    xq_ = nullptr;
    attn_scratch_ = nullptr;
    arg_scratch_ = nullptr;
    out_ = h_out_ = nullptr; h_tok_ = nullptr;
    cs_ = nullptr;
    vram_ = 0;
    device_ = -1;
    emb_ref_ = nullptr;
    head_ = nullptr;
    window_ = cap_ = attn_scratch_floats_ = 0;
    parity_dir_[0] = 0;
}

bool DFlashDrafter::upload(const ModelGeometry& target_g, SessionState& ss, int device, int64_t window,
                           int64_t mask_override, std::string& err) {
    const DFlashGeometry& dg = artifact_.geom();
    device_ = device;
    // the effective mask token, resolved once: the CLI override wins over the artifact's metadata
    // (generate.cpp has validated it against the target's vocabulary)
    mask_ = mask_override >= 0 ? mask_override : dg.mask_token_id;
    if (cudaSetDevice(device_) != cudaSuccess) { err = "dflash: no such device"; return false; }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "dflash: cannot create its stream";
        return false;
    }
    DFlashSection::df_stream() = cs_;
    if (std::getenv("STRATA_DF_EVENTS") != nullptr && std::getenv("STRATA_DF_EVENTS")[0] == '1') {
        df_timing().on = true;
        df_timing().init();
    }

    // any failure past this point leaves a partially-built drafter: release() takes it back
    auto bail = [&](const std::string& why) { release(); err = why; return false; };

    // ---- the weights: one device block, the GGUF layout preserved (row-major rows of bf16).
    // Earlier phases may have left a sticky error in the per-thread state: start clean.
    cudaGetLastError();
    if (cudaMalloc(&w_, artifact_.weight_bytes()) != cudaSuccess)
        return bail("dflash: the BF16 weights do not fit in VRAM");
    vram_ += artifact_.weight_bytes();
    size_t at = 0;
    for (const auto& t : artifact_.tensors()) {
        wt_.push_back({t.name, w_ + at / 2});
        if (cudaMemcpyAsync(w_ + at / 2, artifact_.host_data(t), (size_t) t.rows * t.cols * 2,
                            cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            return bail(std::string("dflash: the weight upload of '") + t.name + "' failed: " +
                        cudaGetErrorString(cudaGetLastError()));
        }
        at += (size_t) t.rows * t.cols * 2;
    }
    // The rms_norm gammas run as F32 device vectors (the kernel's contract); widen the small ones
    // at upload.  Qwen3 norms apply `y * w` with NO Gemma +1.
    auto is_norm = [](const std::string& n) {
        return n == "hidden_norm" || n == "norm" || n.find("layernorm") != std::string::npos ||
               n.find("q_norm") != std::string::npos || n.find("k_norm") != std::string::npos;
    };
    for (const auto& t : artifact_.tensors()) {
        if (!is_norm(t.name)) continue;
        float* f32d = nullptr;
        if (cudaMalloc(&f32d, (size_t) t.rows * t.cols * 4) != cudaSuccess)
            return bail("dflash: the norm weights do not fit");
        vram_ += (size_t) t.rows * t.cols * 4;
        const uint16_t* host = (const uint16_t*) artifact_.host_data(t);
        std::vector<float> wide((size_t) t.rows * t.cols);
        for (int64_t i = 0; i < t.rows * t.cols; ++i) {
            uint32_t bits = (uint32_t) host[i] << 16;
            std::memcpy(&wide[(size_t) i], &bits, 4);
        }
        if (cudaMemcpyAsync(f32d, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess)
            return bail("dflash: the norm weight upload failed");
        wf_.push_back({t.name, f32d});
    }

    // ---- the drafter's own K/V: the target's QSA pool shapes, but only `layers` pools.  The pool
    // geometry is the target's with n_layers shrunk so is_qsa_layer counts exactly 5 pools
    // (layers 3, 7, 11, ... under the interval-4 layout).
    pool_g_ = target_g;
    if (target_g.qsa_interval <= 0 || dg.layers * target_g.qsa_interval > target_g.n_layers)
        return bail("dflash: cannot size the draft K/V pools from the target geometry");
    pool_g_.n_layers = dg.layers * target_g.qsa_interval;
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    // ONE STATE PER DRAFT LAYER: a QsaState is a single layer's pools, and the five draft layers
    // must not share them (a shared pool made every layer attend layer 0's K/V).
    if (pool_g_.n_qsa_layers() != dg.layers)
        return bail("dflash: the pool geometry does not give one state per draft layer");
    // The attention capacity decides the pools' depth too: the drafter never reads or writes a
    // cell at or past cap_, so cells past it would only be allocated to rot.  The window (default
    // 32768) is what keeps the five FP16 pools affordable (32768 cells x 20 KiB = 640 MiB); at the
    // session's full max_cells a long-context session would ask for many GiB of drafter K/V alone.
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = std::min<int64_t>(((window_ > 0 ? window_ : max_cells) + 63) / 64 * 64, max_cells);
    // The allocation policy is EXPLICIT, not the session's: whole-resident FP16 pools carved from
    // this drafter's own arena, no ring, no streaming host copy, no elastic VMM.  qsa_state_bytes
    // and qsa_state_init consume the same options, so the arena holds every logical page the
    // identity page table names (n_slots == n_pages) - the runtime never repairs a state that a
    // different policy has already carved.  ring = -1 is what kv_plan reads as "always fully
    // resident": disable_streaming only kills the ring and would leave the session's --kv-resident
    // cap to hand back a mode-1 (streamed) state the ownership invariant refuses.
    strata::core::QsaStateInitOptions kv_opts;
    kv_opts.force_owned_kv = true;
    kv_opts.force_f16_kv = true;
    kv_opts.disable_elastic = true;
    const int64_t ring = -1;
    const auto pool_shapes = strata::core::shapes_of(pool_g_);
    const uint64_t sb = strata::core::qsa_state_bytes(pool_g_, cap_, false, ring, kv_opts);
    st_.assign((size_t) dg.layers, QsaState{});
    arenas_.assign((size_t) dg.layers, nullptr);
    for (int l = 0; l < (int) dg.layers; ++l) {
        if (cudaMalloc(&arenas_[(size_t) l], sb) != cudaSuccess)
            return bail("dflash: the draft K/V states do not fit in VRAM (lower --dflash-window)");
        if (strata::core::qsa_state_init(pool_g_, cap_, arenas_[(size_t) l], st_[(size_t) l],
                                         &ss.qsa_states[ss.qsa_primary()], ring, kv_opts) == 0)
            return bail("dflash: the draft K/V state init failed");
        {
            // the page table starts as the identity, uploaded on the drafter's own stream (the init
            // wrote it on the default stream)
            std::vector<int32_t> ident_page((size_t) st_[(size_t) l].n_pages);
            for (int64_t pg = 0; pg < st_[(size_t) l].n_pages; ++pg) ident_page[(size_t) pg] = (int32_t) pg;
            if (cudaMemcpyAsync(st_[(size_t) l].page_table, ident_page.data(),
                                (size_t) st_[(size_t) l].n_pages * 4, cudaMemcpyHostToDevice,
                                cs_) != cudaSuccess)
                return bail("dflash: the draft page table init failed");
        }
        strata::core::qsa_state_zero(st_[(size_t) l], pool_g_, nullptr);
        vram_ += sb;
        // OWNERSHIP INVARIANT (docs/DFLASH.md): the state must be the owned whole-resident FP16 one
        // the policy asked for - checked as metadata AND as the full physical byte ranges: a pointer
        // inside the arena does not prove every logical page has backing (a ring or streamed state
        // addresses pages this arena never carved).
        {
            const QsaState& s = st_[(size_t) l];
            const uintptr_t a0 = reinterpret_cast<uintptr_t>(arenas_[(size_t) l]);
            const uintptr_t a1 = a0 + sb;
            const int64_t pool_bytes = (int64_t) s.n_slots * pool_shapes.n_head_kv * pool_shapes.page_size *
                                       dg.head_dim * 2;
            const auto inside = [&](const void* p, int64_t bytes) {
                const uintptr_t x = reinterpret_cast<uintptr_t>(p);
                return bytes >= 0 && x >= a0 && x + (uint64_t) bytes <= a1;
            };
            if (s.kv_elastic != -1 || s.kv_mode != 0 || s.kv_int8 || s.kv_q4 || s.kv_hybrid ||
                s.host.k_pool != nullptr || s.map.slot_block != nullptr || s.n_slots < s.n_pages ||
                !inside(s.k_pool, pool_bytes) || !inside(s.v_pool, pool_bytes)) {
                return bail("dflash: the draft K/V state is not the owned whole-resident FP16 allocation "
                            "the drafter requires (n_slots " + std::to_string((long long) s.n_slots) +
                            ", n_pages " + std::to_string((long long) s.n_pages) + ", mode " +
                            std::to_string(s.kv_mode) + ")");
            }
            if (std::getenv("STRATA_DF_DBG"))
                std::fprintf(stderr, "dflash dbg: layer %d owned: k_pool=%p v_pool=%p arena=[%p,%p) "
                             "slots=%lld pages=%lld mode=%d\n",
                             l, (void*) s.k_pool, (void*) s.v_pool, (void*) a0, (void*) a1,
                             (long long) s.n_slots, (long long) s.n_pages, s.kv_mode);
        }
    }
    cudaDeviceSynchronize();
    if (std::getenv("STRATA_DF_DBG")) {
        const QsaState& s0 = st_[0];
        const QsaState& ref = ss.qsa_states[ss.qsa_primary()];
        std::fprintf(stderr, "dflash dbg: arena0=%p st0.k_pool=%p (offset %lld) ref.k_pool=%p\n",
                     arenas_[0], (void*) s0.k_pool, (long long) ((uint8_t*) s0.k_pool - (uint8_t*) arenas_[0]),
                     (void*) ref.k_pool);
        std::vector<int32_t> pt((size_t) std::min<int64_t>(s0.n_pages, 40));
        cudaMemcpyAsync(pt.data(), s0.page_table, pt.size() * 4, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        std::fprintf(stderr, "dflash dbg: state kv_mode=%d n_slots=%lld n_pages=%lld ring table[:12]=", (int) s0.kv_mode,
                     (long long) s0.n_slots, (long long) s0.n_pages);
        for (size_t i = 0; i < pt.size() && i < 12; ++i) std::fprintf(stderr, " %d", pt[(size_t) i]);
        std::fprintf(stderr, "\n");
    }

    // ---- buffers: at most 8 rows ride through the forward at once
    max_rows_ = 8;
    shapes_ = strata::core::shapes_of(pool_g_);
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, shapes_);
    const int64_t R = max_rows_, N = dg.hidden, F = dg.fusion_in(), I = dg.intermediate;
    const int64_t Q = dg.n_head * dg.head_dim, KVW = dg.n_head_kv * dg.head_dim;
    if (!dflash_mapped(R * 4 + 64, (void**) &h_out_, (void**) &out_) ||
        cudaHostAlloc(&h_tok_, (size_t)(R + 4) * 4, cudaHostAllocDefault) != cudaSuccess) {
        return bail("dflash: mapped staging failed");
    }
    auto take = [&](size_t n, void** p) -> bool {
        if (cudaMalloc(p, n) != cudaSuccess) {
            err = "dflash: the drafter buffers do not fit in VRAM";
            return false;
        }
        vram_ += n;
        return true;
    };
    const int64_t W16 = std::max(N, std::max(I, Q));   // the widest bf16 activation (MLP down's input)
    bool ok = take((R + 4) * 4, (void**) &tok_) && take(R * 4 * 4, (void**) &step_) &&
              take(R * 4 * 4, (void**) &attn_step_) &&
              take(R * (int64_t) dg.n_head * 4, (void**) &pos_) &&
              take(R * (int64_t) dg.n_head_kv * 4, (void**) &poskv_) &&
              take(R * F * 2, (void**) &tapin_) && take(R * W16 * 2, (void**) &xn16_) &&
              take(R * W16 * 2, (void**) &attn16_) && take(R * F * 4, (void**) &tapf_) &&
              take(R * N * 4, (void**) &emb_) && take(R * N * 4, (void**) &h_) &&
              take(R * N * 4, (void**) &xn_) && take(R * N * 4, (void**) &ctx_) &&
              take(R * Q * 4, (void**) &q_) &&
              take(R * KVW * 4, (void**) &kc_) && take(R * KVW * 4, (void**) &vc_) &&
              take(R * Q * 4, (void**) &attn_) && take(R * N * 4, (void**) &bo_) &&
              take(R * I * 4, (void**) &gate_) && take(R * I * 4, (void**) &up_) &&
              take(R * dg.vocab * 4, (void**) &logits_) &&
              take(strata::kernels::argmax_rows_scratch_bytes((int) R), (void**) &arg_scratch_) &&
              take((size_t) strata::kernels::native_q8_1_bytes((int) N, (int) R), (void**) &xq_) &&
              take((size_t) max_rows_ * (size_t) attn_scratch_floats_ * 4, (void**) &attn_scratch_);
    if (!ok) return bail(err);
    // no selection table: the drafter's attention reads the identity cells straight from the
    // block position (dflash_attn_batch); dflash_identity_fill stays for the load test
    if (cudaMemset(arg_scratch_, 0, strata::kernels::argmax_rows_scratch_bytes((int) R)) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess)
        return bail("dflash: the upload did not land");
    if (const char* pd = std::getenv("STRATA_DF_PARITY"))
        std::snprintf(parity_dir_, sizeof parity_dir_, "%s", pd);
    return true;
}

bool DFlashDrafter::load_head(const std::string& path, std::string& err) {
    if (owned_head_) { err = "dflash: draft head already loaded"; return false; }
    const DFlashGeometry& dg = artifact_.geom();
    auto candidate = std::make_unique<NativeHead>();
    if (!candidate->load({path}, dg.hidden, dg.vocab, err)) return false;
    vram_ += candidate->weight_bytes();
    std::fprintf(stderr, "dflash: draft-only head %s, type %d, +%.1f MiB weights; verifier head unchanged\n",
                 path.c_str(), candidate->type(), candidate->weight_bytes() / 1048576.0);
    owned_head_ = candidate.release();
    return true;
}

bool DFlashDrafter::bind(const WeightTable& wt, const NativeHead* head, std::string& err) {
    const DFlashGeometry& dg = artifact_.geom();
    head_ = owned_head_ ? owned_head_ : head;
    if (head_ == nullptr || !head_->loaded()) {
        err = "dflash: the target's native head is required (the full-vocabulary draft head)";
        return false;
    }
    // the mask id must sit inside the vocabulary the artifact declares (the target's; the embedding
    // gather would read past the table otherwise - propose() re-checks every row it stages)
    if (mask_ < 0 || mask_ >= dg.vocab) {
        err = "dflash: the mask token id " + std::to_string(mask_) + " is outside the artifact's vocabulary (" +
              std::to_string(dg.vocab) + ")";
        return false;
    }
    emb_ref_ = wt.find("token_embd.weight");
    if (emb_ref_ == nullptr) { err = "dflash: the target's token_embd.weight is missing"; return false; }
    // GGUF shape [n_embd, n_vocab]: ne0 is the embedding length, ne1 the table's row count
    if (emb_ref_->ne1 < (uint64_t) dg.vocab) {
        err = "dflash: the target's embedding table holds " + std::to_string(emb_ref_->ne1) +
              " rows, less than the artifact's vocabulary " + std::to_string(dg.vocab);
        return false;
    }
    return true;
}

/// One batch of up to 8 rows: ctx = hidden_norm(fc(taps)) into `ctx_`, then each draft layer's
/// K/V projected, k-normed, roped and appended at [pos0, pos0+rows).
bool DFlashDrafter::add_context(const uint16_t* taps, int n_taps, int64_t stride_rows, int64_t pos0, int64_t rows,
                                std::string& err) {
    if (std::getenv("STRATA_DF_DBG")) std::fprintf(stderr, "dflash dbg: add_context rows=%lld pos0=%lld\n",
                                                   (long long) rows, (long long) pos0);
    // The prompt path's capture: [n_taps][stride_rows][n_embd] bf16.  The fusion wants [rows][F]
    // (tap-major inside a row): transpose each 8-row batch into tapin_, then fuse + append.
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, F = dg.fusion_in();
    if (n_taps * N != F) { err = "dflash: the tap count does not match the fusion input"; return false; }
    for (int64_t r0 = 0; r0 < rows; r0 += 8) {
        const int nr = (int) std::min<int64_t>(8, rows - r0);
        {
        DFlashSection section("prompt.tap_gather");
        for (int t = 0; t < n_taps; ++t)
            strata::kernels::bf16_gather_strided(taps + (size_t) ((int64_t) t * stride_rows + r0) * N, N,
                                                 tapin_ + (size_t) t * N, F, (int) N, nr, cs_);
        }
        if (!fusion_rows(pos0 + r0, nr, err)) return false;
        if (df_timing().on) {
            if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash: profiling sync failed"; return false; }
            df_timing_drain();
        }
    }
    // The prefill callback lends its tap buffer only until this call returns. Its
    // producer runs on another stream and can reuse/free the buffer immediately.
    // Keep this ownership boundary even though fusion no longer stages on the host.
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "dflash: prompt context consumption failed";
        return false;
    }
    if (cycle_ == 0 && parity_dir_[0]) {
        // the prompt's context cells, layer 0: the first 32 pages raw ([page][kvh][slot][hd] fp16)
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash: sync failed"; return false; }
        if (std::getenv("STRATA_DF_DBG")) {
            std::vector<uint16_t> probe(8 * 256);
            cudaMemcpyAsync(probe.data(), st_[0].k_pool, probe.size() * 2, cudaMemcpyDeviceToHost, cs_);
            cudaStreamSynchronize(cs_);
            double a0 = 0;
            for (uint16_t b : probe) a0 += std::abs((int) b);
            std::fprintf(stderr, "dflash dbg: pool page0 abs=%.1f k_pool=%p table0=%d\n", a0, (void*) st_[0].k_pool,
                         [&] { int32_t t = -5; cudaMemcpyAsync(&t, st_[0].page_table, 4, cudaMemcpyDeviceToHost, cs_);
                               cudaStreamSynchronize(cs_); return t; }());
        }
        parity_dump_u16_as_f32(parity_dir_, "pool_cells0", arenas_[0], (int64_t) st_[0].max_cells * 2 * 256, cs_);
    }
    return true;
}

bool DFlashDrafter::add_context_f32(const float* taps, int n_taps, int64_t stride_floats, int64_t pos0, int64_t rows,
                                    std::string& err) {
    // The verify window's capture: [n_taps][stride_floats] f32 (stride_floats = max_t*n_embd), the
    // rows [0, rows) of each tap valid.  (The prompt path's add_context takes ROWS instead: its
    // tap stride is the chunk capacity in rows.)
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, F = dg.fusion_in();
    if (n_taps * N != F) { err = "dflash: the tap count does not match the fusion input"; return false; }
    if (rows > max_rows_) { err = "dflash: more context rows than the forward's width"; return false; }
    // f32 source: the fusion input is [row][tap][hidden] (rows x F) - tap t's hidden goes at
    // r * F + t * N, NOT [tap][row][hidden].  (Rows of one tap are consecutive in the window's
    // buffer; a 2-D copy here trips the driver's pitch rules for no gain.)
    {
        DFlashSection s("ctx.taps_d2d");
        for (int t = 0; t < n_taps; ++t) {
            for (int r = 0; r < rows; ++r) {
                if (cudaMemcpyAsync(tapf_ + (size_t) ((int64_t) r * F + (int64_t) t * N),
                                    taps + (size_t) ((int64_t) t * stride_floats + r * N), (size_t) N * 4,
                                    cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
                    err = std::string("dflash: the tap gather failed: ") + cudaGetErrorString(cudaGetLastError()) +
                          " (t=" + std::to_string(t) + " r=" + std::to_string(r) + " stride=" +
                          std::to_string(stride_floats) + " rows=" + std::to_string(rows) + ")";
                    return false;
                }
            }
        }
        strata::kernels::f32_to_bf16_bulk(tapf_, tapin_, (int64_t) rows * F, cs_);
    }
    if (!fusion_rows(pos0, (int) rows, err)) return false;
    {
        DFlashSection s("ctx.sync");
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("dflash: the context update failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    cudaGetLastError();
    df_timing_drain();
    return true;
}

bool DFlashDrafter::fusion_rows(int64_t pos0, int rows, std::string& err) {
    using namespace strata::kernels;
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, F = dg.fusion_in(), KVW = dg.n_head_kv * dg.head_dim;
    // the pools hold exactly the attention window's cells: a context cell past cap_ has no page
    // (propose() refuses the same bound before its own appends)
    if (pos0 < 0 || rows < 1 || pos0 + rows > cap_) {
        err = "dflash: context cells past the window cap (" + std::to_string(pos0 + rows) + " > " +
              std::to_string(cap_) + ") - raise --dflash-window";
        return false;
    }
    auto wp = [&](const char* name) -> const uint16_t* {
        for (auto& [n, p] : wt_)
            if (n == name) return p;
        return nullptr;
    };
    auto wf = [&](const char* name) -> const float* {
        for (auto& [n, p] : wf_)
            if (n == name) return p;
        return nullptr;
    };
    // ctx = hidden_norm(fc(taps)); one batched launch per chunk (the weights were read once per
    // row before)
    {
        DFlashSection s("fusion.fc_gemv");
        bf16_gemv_batch(tapin_, wp("fc"), ctx_, F, N, rows, cs_);
    }
    {
        DFlashSection section("fusion.rmsnorm");
        native_qsa_rms_norm_weighted(ctx_, wf("hidden_norm"), ctx_, (int) N, rows, kEps, cs_);
    }
    if (parity_want(cycle_)) {
        parity_dump(parity_dir_, "tapsin", tapf_, rows * F, cs_);
        parity_dump(parity_dir_, "ctx", ctx_, rows * N, cs_);
    }
    if (cycle_ == 0 && parity_dir_[0]) {
        char path[600];
        std::snprintf(path, sizeof path, "%s/ctx_b%lld.bin", parity_dir_, (long long) pos0);
        std::vector<float> host((size_t) rows * N);
        if (cudaMemcpyAsync(host.data(), ctx_, host.size() * 4, cudaMemcpyDeviceToHost, cs_) != cudaSuccess) return false;
        cudaStreamSynchronize(cs_);
        if (FILE* f = std::fopen(path, "wb")) {
            const uint32_t n = (uint32_t) (rows * N);
            std::fwrite(&n, 4, 1, f);
            std::fwrite(host.data(), 4, host.size(), f);
            std::fclose(f);
        }
    }
    if (std::getenv("STRATA_DF_DBG")) {
        std::vector<float> probe(rows * N);
        cudaMemcpyAsync(probe.data(), ctx_, probe.size() * 4, cudaMemcpyDeviceToHost, cs_);
        cudaStreamSynchronize(cs_);
        double s2 = 0;
        for (double v : probe) s2 += v * v;
        std::fprintf(stderr, "dflash dbg: fusion ctx norm pos0=%lld rows=%d rms=%.4f\n", (long long) pos0, rows,
                     std::sqrt(s2 / probe.size()));
    }
    {
        DFlashSection section("fusion.to_bf16");
        f32_to_bf16_bulk(ctx_, xn16_, (int64_t) rows * N, cs_);
    }
    dflash_build_positions(poskv_, rows, (int) dg.n_head_kv, pos0, cs_);
    dflash_build_steps(step_, rows, pos0, (int) shapes_.page_size, cs_);

    for (int64_t l = 0; l < dg.layers; ++l) {
        const std::string pre = "layers." + std::to_string(l);
        {
            {
                DFlashSection section("fusion.k_proj");
                bf16_gemv_batch(xn16_, wp((pre + ".self_attn.k_proj").c_str()), kc_, N, KVW, rows, cs_);
            }
            {
                DFlashSection section("fusion.v_proj");
                bf16_gemv_batch(xn16_, wp((pre + ".self_attn.v_proj").c_str()), vc_, N, KVW, rows, cs_);
            }
        }
        native_qsa_rms_norm_weighted(kc_, wf((pre + ".self_attn.k_norm").c_str()), kc_, (int) dg.head_dim,
                                     (int) (rows * dg.n_head_kv), kEps, cs_);

        // rope at each row's own position (k rows of one row sit NKV apart: [row r][head][hd]);
        // the positions and the append cells are built on the DEVICE - no pinned staging whose
        // pending copy a rewrite could overtake, so the per-layer syncs are gone with it
        {
            DFlashSection section("fusion.rope");
            dflash_rope_neox_apply(kc_, kc_, (int) (rows * dg.n_head_kv), (int) dg.head_dim, dg.rope_theta, poskv_, cs_);
        }

        // append at the true cells
        if (std::getenv("STRATA_DF_DBG") && l == 0)
            std::fprintf(stderr, "dflash dbg: fusion append pos0=%lld rows=%d cells [%lld..%lld)\n",
                         (long long) pos0, rows, (long long) pos0, (long long) (pos0 + rows));
        const QsaState& stl = st_[(size_t) l];
        const QsaAttnPools pools = qsa_attn_pools(stl);

        if (stl.kv_q4)
            kv_append_q4_steps(stl.k_q4, stl.v_q4, stl.page_table, step_, 4, rows, kc_, vc_, shapes_, cs_, &stl.host);
        else if (stl.kv_int8)
            kv_append_q8_steps(stl.k_q, stl.v_q, stl.k_scale, stl.v_scale, stl.page_table, step_, 4, kc_, vc_,
                               (int) KVW, rows, shapes_, cs_, &stl.host);
        else {
            DFlashSection s("fusion.kv_append");
            kv_append_f16_steps(stl.k_pool, stl.v_pool, stl.page_table, step_, 4, kc_, vc_, (int) KVW, rows,
                                shapes_, cs_, nullptr);
            // the pool rows the appends landed in, for the harness (after a sync so the appends are done)
            if (parity_want(cycle_) && l == 0) {
                if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash: its stream failed"; return false; }
                {
                    const QsaState& s0 = st_[0];
                    std::vector<int32_t> pt((size_t) std::min<int64_t>(s0.n_pages, 40));
                    cudaMemcpyAsync(pt.data(), s0.page_table, pt.size() * 4, cudaMemcpyDeviceToHost, cs_);
                    cudaStreamSynchronize(cs_);
                    std::fprintf(stderr, "dflash dbg: table at cycle-1 ctx[:16]=");
                    for (size_t i = 0; i < pt.size() && i < 16; ++i) std::fprintf(stderr, " %d", pt[(size_t) i]);
                    std::fprintf(stderr, "\n");
                }
                for (int r = 0; r < rows; ++r) {
                    const int64_t cell = pos0 + r;
                    int32_t page_h = -1;
                    cudaMemcpyAsync(&page_h, stl.page_table + cell / shapes_.page_size, 4,
                                    cudaMemcpyDeviceToHost, cs_);
                    cudaStreamSynchronize(cs_);
                    const int64_t page = page_h;
                    const int64_t row = (page * dg.n_head_kv) * shapes_.page_size + (cell % shapes_.page_size);
                    parity_dump_u16_as_f32(parity_dir_, "pool_k0", stl.k_pool + row * dg.head_dim, dg.head_dim, cs_);
                    parity_dump_u16_as_f32(parity_dir_, "pool_head", stl.k_pool, 163840, cs_);
                    char path[600];
                    std::snprintf(path, sizeof path, "%s/pool_meta.bin", parity_dir_);
                    if (FILE* f = std::fopen(path, "wb")) {
                        const uint32_t m[3] = {(uint32_t) cell, (uint32_t) page, (uint32_t) row};
                        std::fwrite(m, 4, 3, f);
                        std::fclose(f);
                    }
                }
            }
        }
    }
    return true;
}

bool DFlashDrafter::propose(int32_t x, int64_t pos, int block, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, Q = dg.n_head * dg.head_dim, KVW = dg.n_head_kv * dg.head_dim, I = dg.intermediate;
    auto wp = [&](const char* name) -> const uint16_t* {
        for (auto& [n, p] : wt_)
            if (n == name) return p;
        return nullptr;
    };
    auto wf = [&](const char* name) -> const float* {
        for (auto& [n, p] : wf_)
            if (n == name) return p;
        return nullptr;
    };
    if (block < 1 || block > max_rows_) { err = "dflash: bad block size"; return false; }
    if (x < 0 || x >= dg.vocab || mask_ < 0 || mask_ >= dg.vocab) {
        err = "dflash: the anchor or mask token id sits outside the vocabulary";
        return false;
    }
    const int K = block;

    // ---- the query rows: [x, mask x (K-1)]; row 0 = the anchor's own embedding
    h_tok_[0] = x;
    for (int r = 1; r < K; ++r) h_tok_[r] = (int32_t) mask_;
    if (cudaMemcpyAsync(tok_, h_tok_, (size_t) K * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "dflash: the token staging failed";
        return false;
    }
    if (const strata::core::NativeEmbed* ne = strata::core::native_embed()) {
        ne->gather_dev(tok_, K, emb_, cs_);
    } else {
        const auto* codes = (const uint8_t*) emb_ref_->data;
        const auto* scales = (const float*) (codes + emb_ref_->codes_bytes);
        const auto* offsets = emb_ref_->has_offset ? (const float*) (codes + emb_ref_->codes_bytes + emb_ref_->scales_bytes)
                                                   : nullptr;
        DFlashSection s("propose.emb_gather");
        embedding_gather_dev(codes, scales, offsets, tok_, K, emb_ref_->ne0, emb_ref_->code_bits, emb_ref_->code_bias,
                             emb_ref_->group_elems, (uint64_t) (emb_ref_->ne0 / (8 / emb_ref_->code_bits)),
                             (uint64_t) (emb_ref_->ne0 / emb_ref_->group_elems), emb_, cs_);
    }
    if (cudaMemcpyAsync(h_, emb_, (size_t) K * N * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "dflash: the residual init failed";
        return false;
    }
    if (parity_want(cycle_)) {
        parity_dump(parity_dir_, "emb", emb_, K * N, cs_);
        const uint32_t meta[5] = {(uint32_t) pos, (uint32_t) K, (uint32_t) x, (uint32_t) mask_,
                                  (uint32_t) shapes_.page_size};
        char path[600];
        std::snprintf(path, sizeof path, "%s/meta.bin", parity_dir_);
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(meta, 4, 5, f); std::fclose(f); }
    }
    // the query rows' and the KV rows' rope positions, the appends' cells and the attention's
    // window are all built ON THE DEVICE once for the whole block: the values are the same every
    // layer, and nothing here is host memory any more, so there is no staging copy a rewrite
    // could overtake (the old two-region pinned h_pos_ dance existed only for that race)
    dflash_build_positions(pos_, K, (int) dg.n_head, pos, cs_);
    dflash_build_positions(poskv_, K, (int) dg.n_head_kv, pos, cs_);
    dflash_build_steps(step_, K, pos, (int) shapes_.page_size, cs_);
    dflash_build_attn_steps(attn_step_, K, pos + K, (int) shapes_.page_size, cs_);

    for (int64_t l = 0; l < dg.layers; ++l) {
        const std::string pre = "layers." + std::to_string(l);
        // ---- attention half
        {
            DFlashSection s("L*.attn_norm");
            native_qsa_rms_norm_weighted(h_, wf((pre + ".input_layernorm").c_str()), xn_, (int) N, K, kEps, cs_);
            if (parity_want(cycle_)) {
                char name[32];
                std::snprintf(name, sizeof name, "xn%d", (int) l);
                parity_dump(parity_dir_, name, xn_, K * N, cs_);
            }
            f32_to_bf16_bulk(xn_, xn16_, (int64_t) K * N, cs_);
        }
        {
            {
                DFlashSection section("L*.q_proj");
                bf16_gemv_batch(xn16_, wp((pre + ".self_attn.q_proj").c_str()), q_, N, Q, K, cs_);
            }
            {
                DFlashSection section("L*.k_proj");
                bf16_gemv_batch(xn16_, wp((pre + ".self_attn.k_proj").c_str()), kc_, N, KVW, K, cs_);
            }
            {
                DFlashSection section("L*.v_proj");
                bf16_gemv_batch(xn16_, wp((pre + ".self_attn.v_proj").c_str()), vc_, N, KVW, K, cs_);
            }
        }
        // per-head q/k norms, then rope (q rows: NH heads at [pos..pos+K); append uses true cells)
        if (parity_want(cycle_)) {
            char name[32];
            std::snprintf(name, sizeof name, "qraw%d", (int) l);
            parity_dump(parity_dir_, name, q_, K * Q, cs_);
        }
        native_qsa_rms_norm_weighted(q_, wf((pre + ".self_attn.q_norm").c_str()), q_, (int) dg.head_dim,
                                     (int) (K * dg.n_head), kEps, cs_);
        if (parity_want(cycle_)) {
            char name[32];
            std::snprintf(name, sizeof name, "qnormed%d", (int) l);
            parity_dump(parity_dir_, name, q_, K * Q, cs_);
        }
        native_qsa_rms_norm_weighted(kc_, wf((pre + ".self_attn.k_norm").c_str()), kc_, (int) dg.head_dim,
                                     (int) (K * dg.n_head_kv), kEps, cs_);
        {
            DFlashSection s("L*.rope_q");
            dflash_rope_neox_apply(q_, q_, (int) (K * dg.n_head), (int) dg.head_dim, dg.rope_theta, pos_, cs_);
        }
        {
            DFlashSection s("L*.rope_kv");
            dflash_rope_neox_apply(kc_, kc_, (int) (K * dg.n_head_kv), (int) dg.head_dim, dg.rope_theta, poskv_, cs_);
        }
        // append the queries' own cells at their true positions (step_, built once above), then
        // every query reads [0, pos+K) (attn_step_)
        const QsaState& stl = st_[(size_t) l];
        const QsaAttnPools pools = qsa_attn_pools(stl);
        {
            DFlashSection s("L*.kv_append");
            if (stl.kv_q4)
                kv_append_q4_steps(stl.k_q4, stl.v_q4, stl.page_table, step_, 4, K, kc_, vc_, shapes_, cs_, &stl.host);
            else if (stl.kv_int8)
                kv_append_q8_steps(stl.k_q, stl.v_q, stl.k_scale, stl.v_scale, stl.page_table, step_, 4, kc_, vc_,
                                   (int) KVW, K, shapes_, cs_, &stl.host);
            else
                kv_append_f16_steps(stl.k_pool, stl.v_pool, stl.page_table, step_, 4, kc_, vc_, (int) KVW, K,
                                    shapes_, cs_, &stl.host);
        }
        if (parity_want(cycle_)) {
            char name[32];
            std::snprintf(name, sizeof name, "steps%d", (int) l);
            parity_dump(parity_dir_, name, attn_step_, K * 4, cs_);   // i32 bits reinterpreted as f32
        }
        if ((int64_t)(pos + K) > cap_) {
            err = "dflash: the window cap is exceeded (raise --dflash-window)";
            return false;
        }
        if (parity_want(cycle_)) {
            // the attention oracle's inputs: whole pool pages covering every visible cell (the
            // pools are read-only for the kernel, so dumping before or after it is the same), the
            // page table's entries for those pages, and the per-row steps the kernel is handed
            const int64_t pages = (pos + K + shapes_.page_size - 1) / shapes_.page_size;
            const int64_t prows = pages * shapes_.page_size * dg.n_head_kv;
            char name[32];
            std::snprintf(name, sizeof name, "kpool%d", (int) l);
            parity_dump_u16_as_f32(parity_dir_, name, pools.k_pool, prows * dg.head_dim, cs_);
            std::snprintf(name, sizeof name, "vpool%d", (int) l);
            parity_dump_u16_as_f32(parity_dir_, name, pools.v_pool, prows * dg.head_dim, cs_);
            std::snprintf(name, sizeof name, "pt%d", (int) l);
            parity_dump(parity_dir_, name, pools.page_table, pages, cs_);
        }
        static const bool df_dbg = std::getenv("STRATA_DF_DBG") != nullptr;
        if (l == 0 && df_dbg) {
            const QsaAttnPools pl = qsa_attn_pools(st_[0]);
            std::fprintf(stderr,
                         "dflash dbg: attn cap=%lld K=%d attn_end=%lld pools k=%p kq=%p kq4=%p pt=%p "
                         "kv_mode=%d rot=%d int8=%d q4=%d shapes hd=%lld nh=%lld nkv=%lld ps=%lld rot_bits=%lld\n",
                         (long long) cap_, K, (long long) (pos + K), (const void*) pl.k_pool,
                         (const void*) pl.k_q, (const void*) pl.k_q4, (const void*) pl.page_table, (int) st_[0].kv_mode,
                         (int) st_[0].kv_rot, (int) st_[0].kv_int8, (int) st_[0].kv_q4, (long long) shapes_.head_dim,
                         (long long) shapes_.n_head, (long long) shapes_.n_head_kv, (long long) shapes_.page_size,
                         (long long) shapes_.n_rot);
        }
        {
            DFlashSection s("L*.attn");
            // the drafter's cells are the identity [0, pos+K): no selection table, and only their
            // chunks launch (the configured window cap stays the scratch stride)
            dflash_attn_batch(q_, pools, attn_step_, pos + K, cap_, shapes_, (float*) attn_scratch_, attn_, K, cs_);
        }
        if (parity_want(cycle_)) {
            char name[32];
            std::snprintf(name, sizeof name, "q%d", (int) l);
            parity_dump(parity_dir_, name, q_, K * Q, cs_);
            std::snprintf(name, sizeof name, "k%d", (int) l);
            parity_dump(parity_dir_, name, kc_, K * KVW, cs_);
            std::snprintf(name, sizeof name, "v%d", (int) l);
            parity_dump(parity_dir_, name, vc_, K * KVW, cs_);
            std::snprintf(name, sizeof name, "attn%d", (int) l);
            parity_dump(parity_dir_, name, attn_, K * Q, cs_);
        }
        f32_to_bf16_bulk(attn_, attn16_, (int64_t) K * Q, cs_);
        {
            DFlashSection s("L*.o_proj");
            bf16_gemv_batch(attn16_, wp((pre + ".self_attn.o_proj").c_str()), bo_, Q, N, K, cs_);
        }
        if (l == 0 && df_dbg) {
            std::vector<float> hb(4), ab(4);
            cudaMemcpyAsync(hb.data(), h_, 16, cudaMemcpyDeviceToHost, cs_);
            cudaMemcpyAsync(ab.data(), bo_, 16, cudaMemcpyDeviceToHost, cs_);
            cudaStreamSynchronize(cs_);
            std::fprintf(stderr, "df dbg: after attn h0=%.3e bo0=%.3e\n", hb[0], ab[0]);
        }
        {
            DFlashSection section("L*.residual_add");
            add_inplace(h_, bo_, K * N, cs_);
        }
        if (parity_want(cycle_)) {
            char name[32];
            std::snprintf(name, sizeof name, "h_attn%d", (int) l);
            parity_dump(parity_dir_, name, h_, K * N, cs_);
        }
        // ---- MLP half
        {
            DFlashSection s("L*.mlp_norm");
            native_qsa_rms_norm_weighted(h_, wf((pre + ".post_attention_layernorm").c_str()), xn_, (int) N, K, kEps, cs_);
            f32_to_bf16_bulk(xn_, xn16_, (int64_t) K * N, cs_);
        }
        {
            {
                DFlashSection section("L*.gate_proj");
                bf16_gemv_batch(xn16_, wp((pre + ".mlp.gate_proj").c_str()), gate_, N, I, K, cs_);
            }
            {
                DFlashSection section("L*.up_proj");
                bf16_gemv_batch(xn16_, wp((pre + ".mlp.up_proj").c_str()), up_, N, I, K, cs_);
            }
        }
        {
            DFlashSection s("L*.swiglu");
            swiglu_inplace(gate_, up_, K * I, cs_);
            f32_to_bf16_bulk(gate_, xn16_, (int64_t) K * I, cs_);
        }
        {
            DFlashSection s("L*.down_gemv");
            bf16_gemv_batch(xn16_, wp((pre + ".mlp.down_proj").c_str()), bo_, I, N, K, cs_);
        }
        {
            DFlashSection section("L*.residual_add");
            add_inplace(h_, bo_, K * N, cs_);
        }
        if (parity_want(cycle_)) {
            char name[32];
            std::snprintf(name, sizeof name, "h_mlp%d", (int) l);
            parity_dump(parity_dir_, name, h_, K * N, cs_);
        }
    }
    // ---- final norm, the target's head, the row argmaxes
    static const bool df_dbg2 = std::getenv("STRATA_DF_DBG") != nullptr;
    if (df_dbg2) {
        std::vector<float> hb(4), eb(4);
        cudaMemcpyAsync(hb.data(), h_, 16, cudaMemcpyDeviceToHost, cs_);
        cudaMemcpyAsync(eb.data(), emb_, 16, cudaMemcpyDeviceToHost, cs_);
        cudaStreamSynchronize(cs_);
        std::fprintf(stderr, "df dbg: final h0=%.3e emb0=%.3e\n", hb[0], eb[0]);
    }
    {
        DFlashSection section("out.rmsnorm");
        native_qsa_rms_norm_weighted(h_, wf("norm"), xn_, (int) N, K, kEps, cs_);
    }
    if (parity_want(cycle_)) parity_dump(parity_dir_, "final_norm", xn_, K * N, cs_);
    const bool output_parity = parity_want(cycle_);
    ++cycle_;
    {
        DFlashSection s("out.quantize");
        native_quantize_q8_1(xn_, xq_, (int) N, K, cs_);
    }
    if (output_parity)
        parity_dump(parity_dir_, "head_act_q8", (const float*) xq_,
                    strata::kernels::native_q8_1_bytes((int) N, K) / 4, cs_);
    {
        DFlashSection s("out.head");
        native_mmvq(head_->type(), head_->weights(), xq_, logits_, (int) N, (int) dg.vocab, K, cs_);
    }
    if (output_parity) parity_dump(parity_dir_, "head_logits", logits_, K * dg.vocab, cs_);
    {
        DFlashSection s("out.argmax");
        argmax_rows(logits_, K, (int) dg.vocab, arg_scratch_, out_, cs_);
    }
    if (output_parity) parity_dump(parity_dir_, "head_picks", (const float*) out_, K, cs_);
    {
        DFlashSection s("out.sync");
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("dflash: its stream failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    df_timing_drain();
    std::memcpy(out, h_out_, (size_t) K * 4);
    return true;
}

}  // namespace strata::core
