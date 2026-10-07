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

#include <algorithm>
#include <cstdio>
#include <cstring>
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
        const uint32_t bits = (uint32_t) host[(size_t) i] << 16;
        std::memcpy(&wide[(size_t) i], &bits, 4);
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

uint64_t DFlashDrafter::bind_bytes(int64_t n_vocab, int max_t) const {
    // The draft logits (max_t rows over the full vocabulary) plus the block's scratch (a few MiB
    // of q/k/v, attention and MLP intermediates for at most 8 rows).
    return (uint64_t) max_t * (uint64_t) n_vocab * 4 + ((uint64_t) 24 << 20);
}

bool DFlashDrafter::upload(const ModelGeometry& target_g, SessionState& ss, int device, int64_t window,
                           std::string& err) {
    const DFlashGeometry& dg = artifact_.geom();
    device_ = device;
    mask_ = dg.mask_token_id;
    if (cudaSetDevice(device_) != cudaSuccess) { err = "dflash: no such device"; return false; }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "dflash: cannot create its stream";
        return false;
    }

    // ---- the weights: one device block, the GGUF layout preserved (row-major rows of bf16).
    // Earlier phases may have left a sticky error in the per-thread state: start clean.
    cudaGetLastError();
    if (cudaMalloc(&w_, artifact_.weight_bytes()) != cudaSuccess) {
        err = "dflash: the BF16 weights do not fit in VRAM";
        return false;
    }
    vram_ += artifact_.weight_bytes();
    size_t at = 0;
    for (const auto& t : artifact_.tensors()) {
        wt_.push_back({t.name, w_ + at / 2});
        if (cudaMemcpyAsync(w_ + at / 2, artifact_.host_data(t), (size_t) t.rows * t.cols * 2,
                            cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = std::string("dflash: the weight upload of '") + t.name + "' failed: " +
                  cudaGetErrorString(cudaGetLastError());
            return false;
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
        if (cudaMalloc(&f32d, (size_t) t.rows * t.cols * 4) != cudaSuccess) {
            err = "dflash: the norm weights do not fit";
            return false;
        }
        vram_ += (size_t) t.rows * t.cols * 4;
        const uint16_t* host = (const uint16_t*) artifact_.host_data(t);
        std::vector<float> wide((size_t) t.rows * t.cols);
        for (int64_t i = 0; i < t.rows * t.cols; ++i) {
            uint32_t bits = (uint32_t) host[i] << 16;
            std::memcpy(&wide[(size_t) i], &bits, 4);
        }
        if (cudaMemcpyAsync(f32d, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the norm weight upload failed";
            return false;
        }
        wf_.push_back({t.name, f32d});
    }

    // ---- the drafter's own K/V: the target's QSA pool shapes, but only `layers` pools.  The pool
    // geometry is the target's with n_layers shrunk so is_qsa_layer counts exactly 5 pools
    // (layers 3, 7, 11, ... under the interval-4 layout).
    pool_g_ = target_g;
    if (target_g.qsa_interval <= 0 || dg.layers * target_g.qsa_interval > target_g.n_layers) {
        err = "dflash: cannot size the draft K/V pools from the target geometry";
        return false;
    }
    pool_g_.n_layers = dg.layers * target_g.qsa_interval;
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * dg.layers + 64 : 0;
    const bool kv_int8_was = strata::core::qsa_kv_int8();
    const bool kv_hybrid_was = strata::core::qsa_kv_hybrid();
    strata::core::qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) strata::core::qsa_set_kv_int8(true);   // K8V4 targets run their drafter INT8
    const uint64_t sb = strata::core::qsa_state_bytes(pool_g_, max_cells, false, ring);
    // ONE STATE PER DRAFT LAYER: a QsaState is a single layer's pools, and the five draft layers
    // must not share them (a shared pool made every layer attend layer 0's K/V).
    if (pool_g_.n_qsa_layers() != dg.layers) {
        err = "dflash: the pool geometry does not give one state per draft layer";
        return false;
    }
    st_.assign((size_t) dg.layers, QsaState{});
    arenas_.assign((size_t) dg.layers, nullptr);
    for (int l = 0; l < (int) dg.layers; ++l) {
        if (cudaMalloc(&arenas_[(size_t) l], sb) != cudaSuccess) {
            strata::core::qsa_set_kv_int8(kv_int8_was);
            strata::core::qsa_set_kv_hybrid(kv_hybrid_was);
            err = "dflash: the draft K/V states do not fit in VRAM";
            return false;
        }
        if (strata::core::qsa_state_init(pool_g_, max_cells, arenas_[(size_t) l], st_[(size_t) l],
                                         &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
            strata::core::qsa_set_kv_int8(kv_int8_was);
            strata::core::qsa_set_kv_hybrid(kv_hybrid_was);
            err = "dflash: the draft K/V state init failed";
            return false;
        }
        // The init COPIES the session reference's streaming/elastic plumbing: the page table's -1
        // residency entries, the host-copy pointers, the stream map, and (worst) `kv_elastic` - an
        // INDEX INTO THE SESSION'S elastic-pool registry.  With it copied, the drafter's k_pool
        // pointed at the session's VMM range, which kvg_start/kvg_ensure re-map and zero while the
        // prompt is read: context cells vanished in irregular holes.  The drafter streams nothing
        // and owns its plain cudaMalloc arena: identity page table, no host copy, no map, no
        // elastic registration.
        st_[(size_t) l].kv_elastic = -1;
        st_[(size_t) l].host = strata::kernels::KvHostPools{};
        st_[(size_t) l].map = strata::kernels::KvStreamMap{};
        st_[(size_t) l].kv_mode = 0;
        {
            std::vector<int32_t> ident_page((size_t) st_[(size_t) l].n_pages);
            for (int64_t pg = 0; pg < st_[(size_t) l].n_pages; ++pg) ident_page[(size_t) pg] = (int32_t) pg;
            if (cudaMemcpyAsync(st_[(size_t) l].page_table, ident_page.data(),
                                (size_t) st_[(size_t) l].n_pages * 4, cudaMemcpyHostToDevice,
                                cs_) != cudaSuccess) {
                err = "dflash: the draft page table init failed";
                return false;
            }
        }
        strata::core::qsa_state_zero(st_[(size_t) l], pool_g_, nullptr);
        vram_ += sb;
        // OWNERSHIP INVARIANT (docs/DFLASH.md): the pools must live in THIS state's arena - not in
        // an elastic VMM range, a streaming host copy or another layer's slab.  Checked as pointers,
        // not metadata: qsa_state_bytes/init skip the arena's K/V storage when the process elastic
        // K/V is on, and a metadata reset cannot repair that.
        {
            const uintptr_t a0 = reinterpret_cast<uintptr_t>(arenas_[(size_t) l]);
            const uintptr_t a1 = a0 + sb;
            const auto inside = [&](const void* p) {
                const uintptr_t x = reinterpret_cast<uintptr_t>(p);
                return x >= a0 && x < a1;
            };
            const QsaState& s = st_[(size_t) l];
            if (!inside(s.k_pool) || !inside(s.v_pool) || s.kv_elastic != -1 || s.kv_mode != 0 ||
                s.kv_int8 || s.kv_q4 || s.kv_hybrid || s.host.k_pool != nullptr || s.map.slot_block != nullptr) {
                err = "dflash: draft K/V pools are not owned by the DFlash state arena "
                      "(the elastic K/V (--kv-grow) must be off for the DFlash drafter)";
                return false;
            }
            if (std::getenv("STRATA_DF_DBG"))
                std::fprintf(stderr, "dflash dbg: layer %d owned: k_pool=%p v_pool=%p arena=[%p,%p) elastic=%d mode=%d\n",
                             l, (void*) s.k_pool, (void*) s.v_pool, (void*) a0, (void*) a1, s.kv_elastic, s.kv_mode);
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
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    shapes_ = strata::core::shapes_of(pool_g_);
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, shapes_);
    const int64_t R = max_rows_, N = dg.hidden, F = dg.fusion_in(), I = dg.intermediate;
    const int64_t Q = dg.n_head * dg.head_dim, KVW = dg.n_head_kv * dg.head_dim;
    if (!dflash_mapped(R * 4 + 64, (void**) &h_out_, (void**) &out_) ||
        cudaHostAlloc(&h_tok_, (size_t)(R + 4) * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_step_, (size_t) R * 16, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_pos_, (size_t) R * dg.n_head * 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "dflash: mapped staging failed";
        return false;
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
              take(R * (int64_t) dg.n_head * 4, (void**) &pos_) && take(R * cap_ * 4, (void**) &ident_) &&
              take(R * F * 2, (void**) &tapin_) && take(R * W16 * 2, (void**) &xn16_) &&
              take(R * W16 * 2, (void**) &attn16_) && take(R * F * 4, (void**) &tapf_) &&
              take(R * N * 4, (void**) &emb_) && take(R * N * 4, (void**) &h_) &&
              take(R * N * 4, (void**) &xn_) && take(R * N * 4, (void**) &ctx_) &&
              take(R * Q * 4, (void**) &q_) &&
              take(R * KVW * 4, (void**) &kc_) && take(R * KVW * 4, (void**) &vc_) &&
              take(R * Q * 4, (void**) &attn_) && take(R * N * 4, (void**) &bo_) &&
              take(R * I * 4, (void**) &gate_) && take(R * I * 4, (void**) &up_) &&
              take(R * dg.vocab * 4, (void**) &logits_) && take(N * 4, (void**) &mask_row_) &&
              take(strata::kernels::argmax_rows_scratch_bytes((int) R), (void**) &arg_scratch_) &&
              take((size_t) strata::kernels::native_q8_1_bytes((int) N, (int) R), (void**) &xq_) &&
              take((size_t) max_rows_ * (size_t) attn_scratch_floats_ * 4, (void**) &attn_scratch_);
    if (!ok) return false;
    // the identity cell selection, once, for EVERY query row: the batch attention offsets the
    // table by row * cap (ids += blockIdx.z * cap), so rows 1..K-1 read garbage when only row 0
    // is initialized - the constant-mask-row symptom.  [r][i] = i, duplicated per row on purpose
    // (no optimization before correctness).
    {
        std::vector<int32_t> id_host((size_t) max_rows_ * (size_t) cap_);
        dflash_identity_fill(id_host.data(), (int) max_rows_, cap_);
        if (cudaMemcpy(ident_, id_host.data(), id_host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dflash: the identity selection upload failed";
            return false;
        }
    }
    if (cudaMemset(arg_scratch_, 0, strata::kernels::argmax_rows_scratch_bytes((int) R)) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "dflash: the upload did not land";
        return false;
    }
    if (const char* pd = std::getenv("STRATA_DF_PARITY"))
        std::snprintf(parity_dir_, sizeof parity_dir_, "%s", pd);
    return true;
}

bool DFlashDrafter::bind(const WeightTable& wt, const NativeHead* head, std::string& err) {
    head_ = head;
    if (head_ == nullptr || !head_->loaded()) {
        err = "dflash: the target's native head is required (the full-vocabulary draft head)";
        return false;
    }
    if (mask_ < 0) { err = "dflash: no mask token id (metadata or --dflash-mask-token)"; return false; }
    if (const strata::core::NativeEmbed* ne = strata::core::native_embed()) {
        ne->gather_one(mask_, mask_row_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash: the mask row gather failed"; return false; }
        return true;
    }
    emb_ref_ = wt.find("token_embd.weight");
    if (emb_ref_ == nullptr) { err = "dflash: the target's token_embd.weight is missing"; return false; }
    int32_t* id_dev = nullptr;
    if (cudaMalloc(&id_dev, 4) != cudaSuccess) { err = "dflash: the mask row staging failed"; return false; }
    const int32_t one = (int32_t) mask_;
    const auto* codes = (const uint8_t*) emb_ref_->data;
    const auto* scales = (const float*) (codes + emb_ref_->codes_bytes);
    const auto* offsets = emb_ref_->has_offset ? (const float*) (codes + emb_ref_->codes_bytes + emb_ref_->scales_bytes)
                                               : nullptr;
    strata::kernels::embedding_gather_dev(codes, scales, offsets, id_dev, 1, emb_ref_->ne0, emb_ref_->code_bits,
                                          emb_ref_->code_bias, emb_ref_->group_elems,
                                          (uint64_t) (emb_ref_->ne0 / (8 / emb_ref_->code_bits)),
                                          (uint64_t) (emb_ref_->ne0 / emb_ref_->group_elems), mask_row_, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        cudaFree(id_dev);
        err = "dflash: the mask row gather failed";
        return false;
    }
    cudaFree(id_dev);
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
        for (int t = 0; t < n_taps; ++t)
            strata::kernels::bf16_gather_strided(taps + (size_t) ((int64_t) t * stride_rows + r0) * N, N,
                                                 tapin_ + (size_t) t * N, F, (int) N, nr, cs_);
        if (!fusion_rows(pos0 + r0, nr, err)) return false;
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
    if (!fusion_rows(pos0, (int) rows, err)) return false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("dflash: the context update failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaGetLastError();
    return true;
}

bool DFlashDrafter::fusion_rows(int64_t pos0, int rows, std::string& err) {
    using namespace strata::kernels;
    const DFlashGeometry& dg = artifact_.geom();
    const int64_t N = dg.hidden, F = dg.fusion_in(), KVW = dg.n_head_kv * dg.head_dim;
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
    // ctx = hidden_norm(fc(taps)); the projections run one row per launch (the bf16 path has no
    // multi-row variant yet - the prompt batches loop, the cycle needs at most 8)
    for (int r = 0; r < rows; ++r)
        bf16_gemv(tapin_ + (size_t) r * F, wp("fc"), ctx_ + (size_t) r * N, F, N, cs_);
    native_qsa_rms_norm_weighted(ctx_, wf("hidden_norm"), ctx_, (int) N, rows, kEps, cs_);
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
    f32_to_bf16_bulk(ctx_, xn16_, (int64_t) rows * N, cs_);

    for (int64_t l = 0; l < dg.layers; ++l) {
        const std::string pre = "layers." + std::to_string(l);
        for (int r = 0; r < rows; ++r) {
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.k_proj").c_str()), kc_ + (size_t) r * KVW, N, KVW, cs_);

            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.v_proj").c_str()), vc_ + (size_t) r * KVW, N, KVW, cs_);
        }
        native_qsa_rms_norm_weighted(kc_, wf((pre + ".self_attn.k_norm").c_str()), kc_, (int) dg.head_dim,
                                     (int) (rows * dg.n_head_kv), kEps, cs_);

        // rope at each row's own position (k rows of one row sit NKV apart: [row r][head][hd]);
        // the rope reads DEVICE positions
        for (int r = 0; r < rows; ++r)
            for (int64_t hh = 0; hh < dg.n_head_kv; ++hh) h_pos_[(size_t) r * dg.n_head_kv + hh] = (int32_t)(pos0 + r);
        if (cudaMemcpyAsync(pos_, h_pos_, (size_t) rows * dg.n_head_kv * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the position staging failed";
            return false;
        }
        dflash_rope_neox_apply(kc_, kc_, (int) (rows * dg.n_head_kv), (int) dg.head_dim, dg.rope_theta, pos_, cs_);

        // append at the true cells
        for (int r = 0; r < rows; ++r) {
            const int64_t cell = pos0 + r;
            h_step_[(size_t) r * 4 + 0] = (int32_t) cell;
            h_step_[(size_t) r * 4 + 1] = (int32_t)(cell + 1);
            h_step_[(size_t) r * 4 + 2] = (int32_t)((cell + 1) / 4);
            h_step_[(size_t) r * 4 + 3] = (int32_t)(cell + 1);
        }
        if (cudaMemcpyAsync(step_, h_step_, (size_t) rows * 16, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the step staging failed";
            return false;
        }
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
            for (int r = 0; r < rows; ++r)
                kv_append_step(stl.k_pool, stl.v_pool, stl.page_table, step_ + r * 4, kc_ + (size_t) r * KVW,
                               vc_ + (size_t) r * KVW, shapes_, cs_, nullptr);
            if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash: its stream failed"; return false; }
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
        const uint32_t meta[2] = {(uint32_t) pos, (uint32_t) K};
        char path[600];
        std::snprintf(path, sizeof path, "%s/meta.bin", parity_dir_);
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(meta, 4, 2, f); std::fclose(f); }
    }
    // per-head rope positions of the query rows: row r at pos+r
    for (int r = 0; r < K; ++r)
        for (int64_t hh = 0; hh < dg.n_head; ++hh) h_pos_[(size_t) r * dg.n_head + hh] = (int32_t)(pos + r);

    for (int64_t l = 0; l < dg.layers; ++l) {
        const std::string pre = "layers." + std::to_string(l);
        // ---- attention half
        native_qsa_rms_norm_weighted(h_, wf((pre + ".input_layernorm").c_str()), xn_, (int) N, K, kEps, cs_);
        if (parity_want(cycle_) && l == 0) parity_dump(parity_dir_, "xn0", xn_, K * N, cs_);
        f32_to_bf16_bulk(xn_, xn16_, (int64_t) K * N, cs_);
        for (int r = 0; r < K; ++r) {
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.q_proj").c_str()), q_ + (size_t) r * Q, N, Q, cs_);
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.k_proj").c_str()), kc_ + (size_t) r * KVW, N, KVW, cs_);
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".self_attn.v_proj").c_str()), vc_ + (size_t) r * KVW, N, KVW, cs_);
        }
        // per-head q/k norms, then rope (q rows: NH heads at [pos..pos+K); append uses true cells)
        if (parity_want(cycle_) && l == 0) parity_dump(parity_dir_, "q_raw", q_, K * Q, cs_);
        native_qsa_rms_norm_weighted(q_, wf((pre + ".self_attn.q_norm").c_str()), q_, (int) dg.head_dim,
                                     (int) (K * dg.n_head), kEps, cs_);
        if (parity_want(cycle_) && l == 0) parity_dump(parity_dir_, "q_normed", q_, K * Q, cs_);
        native_qsa_rms_norm_weighted(kc_, wf((pre + ".self_attn.k_norm").c_str()), kc_, (int) dg.head_dim,
                                     (int) (K * dg.n_head_kv), kEps, cs_);
        if (cudaMemcpyAsync(pos_, h_pos_, (size_t) K * dg.n_head * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the position staging failed";
            return false;
        }
        dflash_rope_neox_apply(q_, q_, (int) (K * dg.n_head), (int) dg.head_dim, dg.rope_theta, pos_, cs_);
        for (int r = 0; r < K; ++r)
            for (int64_t hh = 0; hh < dg.n_head_kv; ++hh) h_pos_[(size_t) r * dg.n_head_kv + hh] = (int32_t)(pos + r);
        if (cudaMemcpyAsync(pos_, h_pos_, (size_t) K * dg.n_head_kv * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the position staging failed";
            return false;
        }
        dflash_rope_neox_apply(kc_, kc_, (int) (K * dg.n_head_kv), (int) dg.head_dim, dg.rope_theta, pos_, cs_);
        // append the queries' own cells at their true positions, then every query reads [0, pos+K)
        for (int r = 0; r < K; ++r) {
            const int64_t cell = pos + r;
            h_step_[(size_t) r * 4 + 0] = (int32_t) cell;
            h_step_[(size_t) r * 4 + 1] = (int32_t)(cell + 1);
            h_step_[(size_t) r * 4 + 2] = (int32_t)((cell + 1) / 4);
            h_step_[(size_t) r * 4 + 3] = (int32_t)(cell + 1);
        }
        if (cudaMemcpyAsync(step_, h_step_, (size_t) K * 16, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "dflash: the step staging failed";
            return false;
        }
        const QsaState& stl = st_[(size_t) l];
        const QsaAttnPools pools = qsa_attn_pools(stl);
        if (stl.kv_q4)
            kv_append_q4_steps(stl.k_q4, stl.v_q4, stl.page_table, step_, 4, K, kc_, vc_, shapes_, cs_, &stl.host);
        else if (stl.kv_int8)
            kv_append_q8_steps(stl.k_q, stl.v_q, stl.k_scale, stl.v_scale, stl.page_table, step_, 4, kc_, vc_,
                               (int) KVW, K, shapes_, cs_, &stl.host);
        else
            for (int r = 0; r < K; ++r)
                kv_append_step(stl.k_pool, stl.v_pool, stl.page_table, step_ + r * 4, kc_ + (size_t) r * KVW,
                               vc_ + (size_t) r * KVW, shapes_, cs_, &stl.host);
        // non-causal over every cell: each row's record reads [0, pos+K)
        {
            std::vector<int32_t> attn_steps((size_t) K * 4);
            for (int r = 0; r < K; ++r) {
                attn_steps[(size_t) r * 4 + 0] = (int32_t)(pos + K - 1);
                attn_steps[(size_t) r * 4 + 1] = (int32_t)(pos + K);
                attn_steps[(size_t) r * 4 + 2] = (int32_t)((pos + K) / 4);
                attn_steps[(size_t) r * 4 + 3] = (int32_t)(pos + K);
            }
            if (cudaMemcpyAsync(step_, attn_steps.data(), (size_t) K * 16, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
                err = "dflash: the attention staging failed";
                return false;
            }
        }
        if ((int64_t)(pos + K) > cap_) {
            err = "dflash: the window cap is exceeded (raise --dflash-window)";
            return false;
        }
        static const bool df_dbg = std::getenv("STRATA_DF_DBG") != nullptr;
        if (l == 0 && df_dbg) {
            const QsaAttnPools pl = qsa_attn_pools(st_[0]);
            std::fprintf(stderr,
                         "dflash dbg: attn cap=%lld K=%d step0=[%d %d %d %d] pools k=%p kq=%p kq4=%p pt=%p "
                         "kv_mode=%d rot=%d int8=%d q4=%d shapes hd=%lld nh=%lld nkv=%lld ps=%lld rot_bits=%lld\n",
                         (long long) cap_, K, h_step_[0], h_step_[1], h_step_[2], h_step_[3], (const void*) pl.k_pool,
                         (const void*) pl.k_q, (const void*) pl.k_q4, (const void*) pl.page_table, (int) st_[0].kv_mode,
                         (int) st_[0].kv_rot, (int) st_[0].kv_int8, (int) st_[0].kv_q4, (long long) shapes_.head_dim,
                         (long long) shapes_.n_head, (long long) shapes_.n_head_kv, (long long) shapes_.page_size,
                         (long long) shapes_.n_rot);
        }
        qsa_decode_attn_batch(q_, pools, ident_, step_, cap_, shapes_, (float*) attn_scratch_, attn_, K, cs_);
        if (parity_want(cycle_) && l <= 1) {
            const char* tag = l == 0 ? "0" : "1";
            char name[32];
            std::snprintf(name, sizeof name, "q%s", tag);
            parity_dump(parity_dir_, name, q_, K * Q, cs_);
            std::snprintf(name, sizeof name, "k%s", tag);
            parity_dump(parity_dir_, name, kc_, K * KVW, cs_);
            std::snprintf(name, sizeof name, "v%s", tag);
            parity_dump(parity_dir_, name, vc_, K * KVW, cs_);
            std::snprintf(name, sizeof name, "attn%s", tag);
            parity_dump(parity_dir_, name, attn_, K * Q, cs_);
        }
        f32_to_bf16_bulk(attn_, attn16_, (int64_t) K * Q, cs_);
        for (int r = 0; r < K; ++r)
            bf16_gemv(attn16_ + (size_t) r * Q, wp((pre + ".self_attn.o_proj").c_str()), bo_ + (size_t) r * N, Q, N, cs_);
        if (l == 0 && df_dbg) {
            std::vector<float> hb(4), ab(4);
            cudaMemcpyAsync(hb.data(), h_, 16, cudaMemcpyDeviceToHost, cs_);
            cudaMemcpyAsync(ab.data(), bo_, 16, cudaMemcpyDeviceToHost, cs_);
            cudaStreamSynchronize(cs_);
            std::fprintf(stderr, "df dbg: after attn h0=%.3e bo0=%.3e\n", hb[0], ab[0]);
        }
        add_inplace(h_, bo_, K * N, cs_);
        if (parity_want(cycle_) && l == 0) parity_dump(parity_dir_, "h_attn0", h_, K * N, cs_);
        // ---- MLP half
        native_qsa_rms_norm_weighted(h_, wf((pre + ".post_attention_layernorm").c_str()), xn_, (int) N, K, kEps, cs_);
        f32_to_bf16_bulk(xn_, xn16_, (int64_t) K * N, cs_);
        for (int r = 0; r < K; ++r) {
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".mlp.gate_proj").c_str()), gate_ + (size_t) r * I, N, I, cs_);
            bf16_gemv(xn16_ + (size_t) r * N, wp((pre + ".mlp.up_proj").c_str()), up_ + (size_t) r * I, N, I, cs_);
        }
        swiglu_inplace(gate_, up_, K * I, cs_);
        f32_to_bf16_bulk(gate_, xn16_, (int64_t) K * I, cs_);
        for (int r = 0; r < K; ++r)
            bf16_gemv(xn16_ + (size_t) r * I, wp((pre + ".mlp.down_proj").c_str()), bo_ + (size_t) r * N, I, N, cs_);
        add_inplace(h_, bo_, K * N, cs_);
        if (parity_want(cycle_) && l == 0) parity_dump(parity_dir_, "h_mlp0", h_, K * N, cs_);
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
    native_qsa_rms_norm_weighted(h_, wf("norm"), xn_, (int) N, K, kEps, cs_);
    if (parity_want(cycle_)) parity_dump(parity_dir_, "final_norm", xn_, K * N, cs_);
    ++cycle_;
    native_quantize_q8_1(xn_, xq_, (int) N, K, cs_);
    native_mmvq(head_->type(), head_->weights(), xq_, logits_, (int) N, (int) dg.vocab, K, cs_);
    argmax_rows(logits_, K, (int) dg.vocab, arg_scratch_, out_, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("dflash: its stream failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    std::memcpy(out, h_out_, (size_t) K * 4);
    return true;
}

}  // namespace strata::core
