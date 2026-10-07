// src/core/glm_prefill.cu - the glm5-next batched PROMPT path (Glm5Model members).
//
// The token path (glm_fast_path.cu) reads a prompt one token at a time: every weight of the model crosses the
// memory bus once per token, ~31-55 ms each.  Here a chunk of T tokens goes through one layer at a time:
//
//   * projections are GEMMs - quantized weights dequantized to FP16 into a scratch and multiplied on the tensor
//     cores (cuBLAS), BF16 weights widened to F32 (SGEMM, the router and the hyper-connections stay F32-exact);
//   * the recurrences (KDA's delta rule, the causal convs) walk the chunk inside one kernel, the DSA caches are
//     written for every position before any token of the chunk attends;
//   * the routed experts are grouped by expert: the ones RESIDENT in the VRAM tier are multiplied where they lie
//     (llama.cpp's MMQ over the layer's whole pool partition in one launch - the slot stride is MMQ-addressable,
//     glmfast::expert_stride), the others are copied from the pinned RAM tier (or read from disk) into a small ring
//     of staging groups on the copy stream while the resident ones compute.
//
// The kernels are the token path's arithmetic (strata/kernels/glm_batch.hpp); what differs is the projections'
// rounding (FP16 activations instead of q8_1), as between llama.cpp's batched and one-token paths.  The prompt's
// routing also feeds the expert tiers' LFU counts, so the decode that follows starts from this conversation's
// experts.  STRATA_GLM_NO_PREFILL=1 keeps the token-at-a-time prompt (A/B).
#include "glm_fast_state.hpp"

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

namespace gb = strata::kernels::glmb;
namespace mmq = strata::prefill::mmq;

namespace {

// carves 256-byte-aligned buffers off a base; with base == nullptr it only measures
struct Carve {
    uint8_t* base = nullptr;
    size_t off = 0;
    template <class Tp>
    Tp* take(size_t n) {
        Tp* p = base ? (Tp*) (base + off) : nullptr;
        off += (n * sizeof(Tp) + 255u) & ~(size_t) 255u;
        return p;
    }
};

constexpr int kSub = 256;   // the mixers and the dense FFN run the chunk in sub-batches of this many tokens
// resident experts routed by at most kLightRows rows of a chunk take the light kernels (gf::rows_experts) instead of
// MMQ (STRATA_GLM_PREFILL_LIGHT overrides, 0 = MMQ only); their {slot, r0, nr} ride the bounds array from kLightOff
constexpr int kLightRows = 32, kLightOff = 6144, kLightMax = (8192 - kLightOff) / 3;

struct KdaBufs {
    float* proj[3];
    float* conv[3];
    float *fa, *ga, *beta, *g1, *g2;
    uint16_t* out16;
};
KdaBufs carve_kda(Carve& c, size_t T, const Glm5Geometry& g) {
    KdaBufs b{};
    const size_t DI = (size_t) g.d_inner();
    for (int i = 0; i < 3; ++i) b.proj[i] = c.take<float>(T * DI);
    for (int i = 0; i < 3; ++i) b.conv[i] = c.take<float>(T * DI);
    b.fa = c.take<float>(T * (size_t) g.kda_head_dim);
    b.ga = c.take<float>(T * (size_t) g.kda_head_dim);
    b.beta = c.take<float>(T * (size_t) g.n_head);
    b.g1 = c.take<float>(T * DI);
    b.g2 = c.take<float>(T * DI);
    b.out16 = c.take<uint16_t>(T * DI);
    return b;
}

struct DsaBufs {
    float *qr_raw, *qr, *kv_raw, *ik_raw, *ig_raw, *iw, *q, *iq, *score, *q_abs, *ctx, *attn;
    uint16_t *qr16, *attn16;
    int *cells, *n_sel;
    int score_ld;
};
DsaBufs carve_dsa(Carve& c, size_t T, const Glm5Geometry& g, int max_pools) {
    DsaBufs b{};
    b.qr_raw = c.take<float>(T * g.q_lora);
    b.qr = c.take<float>(T * g.q_lora);
    b.qr16 = c.take<uint16_t>(T * g.q_lora);
    b.kv_raw = c.take<float>(T * g.kv_lora);
    b.ik_raw = c.take<float>(T * g.idx_key);
    b.ig_raw = c.take<float>(T * g.idx_key);
    b.iw = c.take<float>(T * g.idx_heads);
    b.q = c.take<float>(T * (size_t) g.n_head * g.qk_nope);
    b.iq = c.take<float>(T * (size_t) g.idx_heads * g.idx_key);
    b.cells = c.take<int>(T * (size_t) g.n_sel_max());
    b.n_sel = c.take<int>(T);
    b.score_ld = std::max(1, max_pools);
    b.score = c.take<float>(T * (size_t) b.score_ld);
    b.q_abs = c.take<float>(T * (size_t) g.n_head * g.kv_lora);
    b.ctx = c.take<float>(T * (size_t) g.n_head * g.kv_lora);
    b.attn = c.take<float>(T * (size_t) g.n_head * g.v_head);
    b.attn16 = c.take<uint16_t>(T * (size_t) g.n_head * g.v_head);
    return b;
}

struct DenseBufs {
    float *dg, *du;
    uint16_t* dh16;
};
DenseBufs carve_dense(Carve& c, size_t T, const Glm5Geometry& g) {
    DenseBufs b{};
    b.dg = c.take<float>(T * g.n_ff_dense);
    b.du = c.take<float>(T * g.n_ff_dense);
    b.dh16 = c.take<uint16_t>(T * g.n_ff_dense);
    return b;
}

// the MoE runs on the WHOLE chunk (every expert's weights are read once per chunk); the shared expert in sub-batches
struct MoeBufs {
    float *logits, *rw, *H, *OUT;
    float *sh_g, *sh_u;
    uint16_t* sh16;
    int *ids, *rank, *counts, *base, *row_tok, *pos, *bounds;
    uint8_t *Xq, *Hq;
};
MoeBufs carve_moe(Carve& c, size_t T, const Glm5Geometry& g) {
    MoeBufs b{};
    const size_t rows = T * (size_t) g.n_exp_used;
    const size_t FF = (size_t) g.n_ff_exp * g.n_shared;
    b.logits = c.take<float>(T * g.n_expert);
    b.ids = c.take<int>(rows);
    b.rw = c.take<float>(rows);
    b.rank = c.take<int>(rows);
    b.pos = c.take<int>(rows);
    b.row_tok = c.take<int>(rows);
    b.counts = c.take<int>((size_t) g.n_expert);
    b.base = c.take<int>((size_t) g.n_expert);
    b.bounds = c.take<int>(8192);
    b.Xq = c.take<uint8_t>(mmq::q8_bytes((int64_t) rows, g.n_embd));
    b.H = c.take<float>(rows * g.n_ff_exp);
    b.Hq = c.take<uint8_t>(mmq::q8_bytes((int64_t) rows, g.n_ff_exp));
    b.OUT = c.take<float>(rows * g.n_embd);   // each set's gate/up rows live in its own OUT rows until the down product
    const size_t ts = std::min<size_t>(T, kSub);
    b.sh_g = c.take<float>(ts * FF);
    b.sh_u = c.take<float>(ts * FF);
    b.sh16 = c.take<uint16_t>(ts * FF);
    return b;
}

void blas_ck(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "glm prefill: %s: cuBLAS status %d\n", what, (int) st);
}

}  // namespace

struct Glm5Model::PrefillState {
    int T = 0;                       // tokens per chunk
    int max_pools = 0;
    cublasHandle_t blas = nullptr;
    void* ws = nullptr;
    std::unique_ptr<mmq::Context> mq;
    uint8_t* arena = nullptr;
    size_t arena_bytes = 0;
    // every device buffer lives in the pool's lendable tail (prefill_bind): valid only while lent
    size_t borrow_bytes = 0, ws_bytes = 0, gbuf_bytes = 0;
    bool lent = false;
    bool has_kda = false, has_dsa = false, has_dense = false, has_moe = false;
    uint8_t* region = nullptr;       // the pool's lendable tail
    size_t region_bytes = 0;
    int T_bound = 0;                 // the chunk the pointers are carved for (<= T)
    size_t need = 0;                 // ... and the bytes of the region that layout uses
    uint64_t dropped = 0;            // experts the lending evicted from VRAM (cumulative)
    uint64_t moved = 0;              // ... of which a lent slot's expert took a colder one's slot instead
    // per chunk, across layers
    float *R = nullptr, *x = nullptr, *mixer = nullptr, *ffn = nullptr;
    float *pre = nullptr, *post = nullptr, *comb = nullptr, *ss = nullptr, *mix = nullptr;
    uint16_t* x16 = nullptr;
    int* iota = nullptr;
    uint8_t* uni = nullptr;          // the per-layer region (KDA | DSA | dense | MoE)
    // weight scratch
    uint16_t* w16 = nullptr;
    int64_t w16_elems = 0;
    float* w32 = nullptr;
    int64_t w32_elems = 0;
    // non-resident experts: a ring of NG staging groups of GE slots each (the layer's slot stride apart)
    static constexpr int NG = 3, GE = 4;
    uint8_t* gbuf = nullptr;
    size_t gstride = 0;
    uint8_t* gpin = nullptr;         // pinned: the disk reads' landing ring (NG * GE slots of gstride)
    cudaEvent_t ev_ready[NG] = {}, ev_free[NG] = {};
    cudaEvent_t ev_land[NG * GE] = {};   // a landing slot's copy to the device ran: the reader may refill it
    // pinned host staging
    float* emb_h = nullptr;          // T x n_embd
    float* hop_h = nullptr;          // 2 x T x hc x n_embd: the split's hand-over (first half), double-buffered
    int* h_counts = nullptr;
    int* h_base = nullptr;
    int* h_bounds = nullptr;
    cudaEvent_t ev_hop = nullptr;
    // stats
    double ms = 0, ms_plan = 0, ms_disk = 0;
    int64_t tokens = 0, chunks = 0;
    // STRATA_GLM_PREFILL_PROF=1: events between the phases of every layer, summed per phase (debug)
    bool prof = false;
    std::vector<cudaEvent_t> pev;
    std::vector<const char*> pname;
    size_t pn = 0;
    std::map<std::string, double> pacc;
    void mark(const char* nm, cudaStream_t st) {
        if (!prof) return;
        if (pn >= pev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            pev.push_back(e);
            pname.push_back(nm);
        }
        cudaEventRecord(pev[pn], st);
        pname[pn] = nm;
        ++pn;
    }
    void collect() {
        if (!prof || pn == 0) return;
        cudaEventSynchronize(pev[pn - 1]);
        for (size_t i = 1; i < pn; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, pev[i - 1], pev[i]) == cudaSuccess) pacc[pname[i]] += t;
        }
        pn = 0;
    }
    uint64_t staged_ram = 0, staged_disk = 0, rows_resident = 0, rows_staged = 0, disk_issued = 0;
};

int Glm5Model::prefill_chunk() const { return pf_ ? pf_->T : 0; }

// ---------------------------------------------------------------- setup
bool Glm5Model::prefill_setup(std::string& err) {
    (void) err;
    if (getenv("STRATA_GLM_NO_PREFILL") != nullptr) return true;
    if (!mmq::built()) {
        std::fprintf(stderr, "glm prefill: this build has no MMQ kernels - the prompt runs token by token\n");
        return true;
    }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    bool has_kda = false, has_dsa = false, has_moe = false, has_dense = false;
    size_t gstride = 0;
    std::string why;
    const auto deq_ok = [&](const WSlot& w, const char* nm, int il) {
        if (w.q == nullptr || !strata::kernels::dequant_bf16_supported(w.type))
            why += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + ") ";
    };
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (Ly.recr) {
            has_kda = true;
            deq_ok(Ly.q, "attn_q", il);
            deq_ok(Ly.k, "attn_k", il);
            deq_ok(Ly.v, "attn_v", il);
        } else {
            has_dsa = true;
            deq_ok(Ly.q_a, "attn_q_a", il);
            deq_ok(Ly.q_b, "attn_q_b", il);
            deq_ok(Ly.kv_a, "attn_kv_a", il);
        }
        deq_ok(Ly.out, "attn_output", il);
        if (Ly.moe) {
            has_moe = true;
            deq_ok(Ly.sh_gate, "ffn_gate_shexp", il);
            deq_ok(Ly.sh_up, "ffn_up_shexp", il);
            deq_ok(Ly.sh_down, "ffn_down_shexp", il);
            if (!mmq::supported(Ly.gu_type) || !mmq::supported(Ly.d_type))
                why += "blk." + std::to_string(il) + " experts (types " + std::to_string(Ly.gu_type) + "/" +
                       std::to_string(Ly.d_type) + ") ";
            gstride = std::max(gstride, glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type));
        } else {
            has_dense = true;
            deq_ok(Ly.ffn_gate, "ffn_gate", il);
            deq_ok(Ly.ffn_up, "ffn_up", il);
            deq_ok(Ly.ffn_down, "ffn_down", il);
        }
    }
    if (!why.empty()) {
        std::fprintf(stderr, "glm prefill: CUDA%d has no prompt path (not covered: %s) - token by token\n", dev_,
                     why.substr(0, 400).c_str());
        return true;
    }
    if (has_dsa && (g.kv_lora != 512 || g.n_head % 16 != 0 || g.kda_head_dim != 128)) {
        std::fprintf(stderr, "glm prefill: geometry not covered by the batched kernels - token by token\n");
        return true;
    }
    auto* S = new PrefillState();
    pf_ = S;
    S->has_kda = has_kda;
    S->has_dsa = has_dsa;
    S->has_dense = has_dense;
    S->has_moe = has_moe;
    S->max_pools = (int) std::max<int64_t>(1, max_ctx_ / g.idx_kpool);
    S->prof = getenv("STRATA_GLM_PREFILL_PROF") != nullptr;
    S->gstride = gstride;

    // ---- sizes: the chunk T from the budget (per device)
    const int64_t E = g.n_embd;
    int64_t w32_elems = 2 << 20;
    if (has_dsa)
        w32_elems = std::max<int64_t>({w32_elems, (int64_t) g.n_head * g.kv_lora * g.qk_nope,
                                       (int64_t) g.n_head * g.v_head * g.kv_lora});
    const int64_t w16_elems = std::max<int64_t>((int64_t) g.d_inner() * E, (int64_t) 2048 * E);
    const size_t ws_bytes = (size_t) 16 << 20;
    const size_t fixed = (size_t) w16_elems * 2 + (size_t) w32_elems * 4 + ws_bytes +
                         (size_t) PrefillState::NG * PrefillState::GE * gstride + ((size_t) 64 << 20);
    const auto bytes_for = [&](size_t T) {
        Carve c;
        c.take<float>(T * 4 * E);                    // R
        c.take<float>(T * E);                        // x
        c.take<uint16_t>(T * E);                     // x16
        c.take<float>(T * E);                        // mixer
        c.take<float>(T * E);                        // ffn
        c.take<float>(T * 4);
        c.take<float>(T * 4);
        c.take<float>(T * 16);
        c.take<float>(T);
        c.take<float>(T * 24);
        c.take<int>(T * (size_t) g.n_exp_used);      // iota
        size_t uni = 0;
        const size_t ts = std::min<size_t>(T, kSub);
        if (has_kda) { Carve k; carve_kda(k, ts, g); uni = std::max(uni, k.off); }
        if (has_dsa) { Carve d; carve_dsa(d, ts, g, S->max_pools); uni = std::max(uni, d.off); }
        if (has_dense) { Carve d; carve_dense(d, ts, g); uni = std::max(uni, d.off); }
        if (has_moe) { Carve m; carve_moe(m, T, g); uni = std::max(uni, m.off); }
        return std::make_pair(c.off, uni);
    };
    double budget_mb = 1024.0;
    if (const char* b = getenv("STRATA_GLM_PREFILL_MB")) budget_mb = std::atof(b);
    int T = 2048;
    if (const char* c = getenv("STRATA_GLM_PREFILL_CHUNK")) T = std::max(16, std::atoi(c));
    while (T > 64) {
        const auto pu = bytes_for((size_t) T);
        if ((double) (fixed + pu.first + pu.second) <= budget_mb * 1048576.0) break;
        T -= 64;
    }
    S->T = T;
    const auto pu = bytes_for((size_t) T);
    S->arena_bytes = pu.first + pu.second;
    S->w16_elems = w16_elems;
    S->w32_elems = w32_elems;
    S->ws_bytes = ws_bytes;
    S->gbuf_bytes = has_moe ? (size_t) PrefillState::NG * PrefillState::GE * gstride : 0;
    {
        Carve c;
        c.take<uint8_t>(S->arena_bytes);
        c.take<uint16_t>((size_t) w16_elems);
        c.take<float>((size_t) w32_elems);
        c.take<uint8_t>(ws_bytes);
        c.take<uint8_t>(S->gbuf_bytes);
        S->borrow_bytes = c.off;
    }
    if (cudaHostAlloc((void**) &S->emb_h, (size_t) T * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_counts, 4096 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_base, 4096 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_bounds, 8192 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        (has_moe && cudaHostAlloc((void**) &S->gpin, (size_t) PrefillState::NG * PrefillState::GE * gstride,
                                  cudaHostAllocDefault) != cudaSuccess) ||
        (l1_ < g.n_layers && cudaHostAlloc((void**) &S->hop_h, (size_t) 2 * T * 4 * E * sizeof(float),
                                           cudaHostAllocDefault) != cudaSuccess)) {
        cudaGetLastError();
        std::fprintf(stderr, "glm prefill: CUDA%d pinned staging did not allocate - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    if (cublasCreate(&S->blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "glm prefill: CUDA%d cublasCreate failed - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    cublasSetStream(S->blas, F->cs);
    cublasSetMathMode(S->blas, CUBLAS_DEFAULT_MATH);
    S->mq = std::make_unique<mmq::Context>();
    for (int b = 0; b < PrefillState::NG; ++b) {
        cudaEventCreateWithFlags(&S->ev_ready[b], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&S->ev_free[b], cudaEventDisableTiming);
    }
    for (auto& e : S->ev_land) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&S->ev_hop, cudaEventDisableTiming);
    std::fprintf(stderr, "glm prefill: CUDA%d chunks of %d tokens, borrowing %.0f MB of the expert pool while a prompt "
                         "runs (activations %.0f, weight scratch %.0f, expert staging %.0f)\n",
                 dev_, T, (double) S->borrow_bytes / 1048576.0, (double) S->arena_bytes / 1048576.0,
                 (double) (w16_elems * 2 + w32_elems * 4) / 1048576.0, (double) S->gbuf_bytes / 1048576.0);
    (void) fixed;
    return true;
}

size_t Glm5Model::prefill_borrow_bytes() const { return pf_ ? pf_->borrow_bytes : 0; }

bool Glm5Model::prefill_bind(uint8_t* region, size_t bytes, std::string& err) {
    PrefillState* S = pf_;
    if (S == nullptr) return true;
    if (bytes < S->borrow_bytes) {
        err = "glm prefill: the lendable tail (" + std::to_string(bytes >> 20) + " MB) is smaller than the prompt path (" +
              std::to_string(S->borrow_bytes >> 20) + " MB)";
        return false;
    }
    S->region = region;
    S->region_bytes = bytes;
    prefill_carve(S->T);
    return true;
}

// the prompt path's device buffers for chunks of T tokens, carved from the start of the lendable region; returns the
// bytes that layout uses (a short prompt borrows - and evicts - only that much of the pool's tail)
size_t Glm5Model::prefill_carve(int Tn) {
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const size_t T = (size_t) Tn, E = (size_t) g.n_embd;
    size_t uni = 0;
    {
        const size_t ts = std::min<size_t>(T, kSub);
        if (S->has_kda) { Carve k; carve_kda(k, ts, g); uni = std::max(uni, k.off); }
        if (S->has_dsa) { Carve d; carve_dsa(d, ts, g, S->max_pools); uni = std::max(uni, d.off); }
        if (S->has_dense) { Carve d; carve_dense(d, ts, g); uni = std::max(uni, d.off); }
        if (S->has_moe) { Carve m; carve_moe(m, T, g); uni = std::max(uni, m.off); }
        // the NextN block's cache fill (the last half) carves 4 n_embd + 1.5 n_embd rows of its own
        if (mtp_il_ >= 0) uni = std::max(uni, T * (size_t) (6 * g.n_embd + g.kv_lora + 2 * g.idx_key) * 4 + 8 * 256);
    }
    size_t pers = 0;
    {
        Carve a;
        a.take<float>(T * 4 * E);
        a.take<float>(T * E);
        a.take<uint16_t>(T * E);
        a.take<float>(T * E);
        a.take<float>(T * E);
        a.take<float>(T * 4);
        a.take<float>(T * 4);
        a.take<float>(T * 16);
        a.take<float>(T);
        a.take<float>(T * 24);
        a.take<int>(T * (size_t) g.n_exp_used);
        pers = a.off;
    }
    Carve c{S->region};
    S->arena = c.take<uint8_t>(pers + uni);
    S->w16 = c.take<uint16_t>((size_t) S->w16_elems);
    S->w32 = c.take<float>((size_t) S->w32_elems);
    S->ws = c.take<uint8_t>(S->ws_bytes);
    S->gbuf = S->gbuf_bytes ? c.take<uint8_t>(S->gbuf_bytes) : nullptr;
    Carve a{S->arena};
    S->R = a.take<float>(T * 4 * E);
    S->x = a.take<float>(T * E);
    S->x16 = a.take<uint16_t>(T * E);
    S->mixer = a.take<float>(T * E);
    S->ffn = a.take<float>(T * E);
    S->pre = a.take<float>(T * 4);
    S->post = a.take<float>(T * 4);
    S->comb = a.take<float>(T * 16);
    S->ss = a.take<float>(T);
    S->mix = a.take<float>(T * 24);
    S->iota = a.take<int>(T * (size_t) g.n_exp_used);
    S->uni = S->arena + a.off;
    cublasSetWorkspace(S->blas, S->ws, S->ws_bytes);
    S->T_bound = Tn;
    S->need = c.off;
    return c.off;
}

// The pool's tail -> the prompt path: every lendable slot's expert leaves VRAM (it is only on disk until the decode
// fetches it again), a spare there leaves the spare table, and the device tables learn it before any prompt work.
void Glm5Model::prefill_lend() {
    FastState* F = fast_;
    PrefillState* S = pf_;
    if (F == nullptr || S == nullptr || S->lent) return;
    cudaSetDevice(dev_);
    lend_tail(S->need, S->moved, S->dropped);
    // the borrowed memory holds expert bytes: what the prompt path reads before writing is set up again
    mmq::iota(S->iota, (int64_t) S->T_bound * g_.n_exp_used, F->cs);
    for (int b = 0; b < PrefillState::NG; ++b) cudaEventRecord(S->ev_free[b], F->cs);
    cudaStreamSynchronize(F->cs);
    S->lent = true;
}

// The on-demand vision encoder (serve VLEND): the whole tail is emptied the same way and its memory freed, so the
// encoder's process can allocate it; the prompt path waits (token by token) until vision_reclaim gives it back.
bool Glm5Model::vision_lend(size_t& bytes, std::string& err) {
    FastState* F = fast_;
    bytes = 0;
    if (F == nullptr || !vis_lend_ok_) {
        err = "this engine has no pool tail to lend to the vision encoder";
        return false;
    }
    if (vis_lent_) {
        bytes = F->xpool_bytes;
        return true;
    }
    cudaSetDevice(dev_);
    uint64_t moved = 0, dropped = 0;
    lend_tail(F->xpool_bytes, moved, dropped);
    cudaStreamSynchronize(F->cs);
    cudaStreamSynchronize(F->copy);
    cudaFree(F->xpool);
    F->xpool = nullptr;
    for (auto& P : F->lp) P.xbase = nullptr;
    vis_lent_ = true;
    bytes = F->xpool_bytes;
    std::fprintf(stderr, "glm fast: CUDA%d %.2f GB lent to the vision encoder (%llu experts moved to colder slots, "
                         "%llu dropped)\n", dev_, (double) bytes / 1073741824.0, (unsigned long long) moved,
                 (unsigned long long) (dropped - moved));
    return true;
}

size_t Glm5Model::vision_lend_bytes() const { return fast_ != nullptr && vis_lend_ok_ ? fast_->xpool_bytes : 0; }

// ... and back once the encoder has ended: the tail is allocated again (false while the memory is still taken) and
// its slots are free - the decode's misses refill them
bool Glm5Model::vision_reclaim(std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || !vis_lent_) return true;
    cudaSetDevice(dev_);
    if (cudaMalloc(&F->xpool, F->xpool_bytes) != cudaSuccess) {
        cudaGetLastError();
        F->xpool = nullptr;
        err = "the pool's tail is not free yet";
        return false;
    }
    uint8_t* b = F->xpool;
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        auto& P = F->lp[(size_t) il];
        P.xbase = b;
        b += (size_t) (P.n - P.n_main) * P.stride;
    }
    if (pf_ != nullptr && !prefill_bind(F->xpool, F->xpool_bytes, err)) return false;
    {
        std::lock_guard<std::mutex> lk(F->mu);
        for (int il = l0_; il < lt_; ++il) {
            auto& P = F->lp[(size_t) il];
            for (int s = P.n_main; s < P.n; ++s)
                if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
        }
    }
    vis_lent_ = false;
    std::fprintf(stderr, "glm fast: CUDA%d the vision encoder's %.2f GB are back in the expert pool\n", dev_,
                 (double) F->xpool_bytes / 1073741824.0);
    return true;
}

// The tail's slots below xpool + limit leave the decode: a resident expert takes the slot of a colder main-slot
// resident (a device copy) or goes; a spare there leaves the spare table; the slots are kLent after this.
void Glm5Model::lend_tail(size_t limit, uint64_t& moved, uint64_t& dropped) {
    FastState* F = fast_;
    fast_boundary();                   // finished promotions and demotions go live (the plan reads the tables)
    cudaStreamSynchronize(F->copy);    // every demotion issued so far has read its slot
    const int NE = g_.n_expert;
    {
        std::lock_guard<std::mutex> lk(F->mu);
        int nu = 0;
        const auto flush = [&]() {
            if (nu == 0) return;
            cudaMemcpyAsync(F->upd_key_d, F->upd_key_h, (size_t) nu * sizeof(int), cudaMemcpyHostToDevice, F->cs);
            cudaMemcpyAsync(F->upd_val_d, F->upd_val_h, (size_t) nu * sizeof(unsigned long long),
                            cudaMemcpyHostToDevice, F->cs);
            gf::tab_update(F->tab, F->upd_key_d, F->upd_val_d, nu, F->cs);
            cudaStreamSynchronize(F->cs);   // the host buffers are reused after it ran
            nu = 0;
        };
        // (every key is edited at most once here, so a batch never carries two values for one key)
        const auto upd = [&](size_t k, unsigned long long v) {
            if (nu == FastState::kMaxUpd) flush();
            F->upd_key_h[nu] = (int) k;
            F->upd_val_h[nu] = v;
            ++nu;
        };
        // STRATA_GLM_LEND_DROP=1: drop the lent slots' experts as before (instead of moving them, below)
        static const bool lend_drop = getenv("STRATA_GLM_LEND_DROP") != nullptr;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            // the layer's main-slot residents, coldest first: a lent slot's expert moves over the coldest one that is
            // colder than it (a device copy) - the prompt costs the decode its coldest experts, not whichever sat in
            // the borrowed tail (a chat turn used to drop up to ~145 warm experts to disk)
            std::vector<std::pair<uint64_t, int>> cold;   // (count << 32 | recency rank, slot)
            if (!lend_drop)
                for (int s = 0; s < P.n_main; ++s)
                    if (P.st[(size_t) s] == FastState::kResident && P.key[(size_t) s] >= 0)
                        cold.push_back({(uint64_t) F->cnt[(size_t) il * NE + P.key[(size_t) s]], s});
            std::sort(cold.begin(), cold.end(), [&](const auto& a, const auto& b) {
                return a.first != b.first ? a.first < b.first : P.tick[(size_t) a.second] < P.tick[(size_t) b.second];
            });
            size_t ci = 0;
            for (int s = P.n_main; s < P.n; ++s) {
                // a slot past the bytes this lending uses (a prompt's layout) stays with the decode
                if (P.slot_ptr(s) >= F->xpool + limit) continue;
                char& st = P.st[(size_t) s];
                if (st == FastState::kResident && P.key[(size_t) s] >= 0) {
                    const int key = il * NE + P.key[(size_t) s];
                    if (ci < cold.size() && cold[ci].first < (uint64_t) F->cnt[(size_t) key]) {
                        const int v = cold[ci++].second;   // the coldest: it goes, this expert takes its slot
                        const int vkey = il * NE + P.key[(size_t) v];
                        cudaMemcpyAsync(P.slot_ptr(v), P.slot_ptr(s), F->L[(size_t) il].blob, cudaMemcpyDeviceToDevice,
                                        F->cs);
                        upd(F->tab_key((size_t) vkey), 0ull);
                        F->slot_of[(size_t) vkey] = -1;
                        F->left[(size_t) vkey] = 3;
                        upd(F->tab_key((size_t) key), (unsigned long long) P.slot_ptr(v));
                        F->slot_of[(size_t) key] = v;
                        P.key[(size_t) v] = P.key[(size_t) s];
                        P.tick[(size_t) v] = P.tick[(size_t) s];
                        ++moved;
                    } else {
                        upd(F->tab_key((size_t) key), 0ull);
                        F->slot_of[(size_t) key] = -1;
                        F->left[(size_t) key] = 3;
                    }
                    ++dropped;
                    ++F->diag_lend;
                } else if (st == FastState::kSpare) {
                    for (int j = 0; j < gf::kSpares; ++j)
                        if (P.spare[j] == s) {
                            P.spare[j] = -1;
                            upd(F->spare_key(il, j), 0ull);
                        }
                }
                P.key[(size_t) s] = -1;
                st = FastState::kLent;
            }
        }
        flush();
    }
}

// ... and back: the slots are free again; the decode's boundaries hand them out as spares (misses refill them)
void Glm5Model::prefill_return() {
    FastState* F = fast_;
    PrefillState* S = pf_;
    if (F == nullptr || S == nullptr || !S->lent) return;
    cudaSetDevice(dev_);
    cudaStreamSynchronize(F->cs);
    cudaStreamSynchronize(F->copy);
    std::lock_guard<std::mutex> lk(F->mu);
    for (int il = l0_; il < lt_; ++il) {
        auto& P = F->lp[(size_t) il];
        for (int s = P.n_main; s < P.n; ++s)
            if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
    }
    S->lent = false;
}

void Glm5Model::prefill_destroy() {
    PrefillState* S = pf_;
    if (S == nullptr) return;
    cudaSetDevice(dev_);
    if (fast_ && fast_->cs) cudaStreamSynchronize(fast_->cs);
    if (fast_ && fast_->copy) cudaStreamSynchronize(fast_->copy);
    if (S->prof && !S->pacc.empty()) {
        double tot = 0;
        for (auto& kv : S->pacc) tot += kv.second;
        std::fprintf(stderr, "glm prefill prof CUDA%d (%lld tokens, %.0f ms of stream time; waits on disk %.0f ms, plan %.0f "
                             "ms host; disk experts used %llu, read %llu):\n", dev_, (long long) S->tokens, tot, S->ms_disk,
                     S->ms_plan, (unsigned long long) S->staged_disk, (unsigned long long) S->disk_issued);
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : S->pacc) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        for (auto& e : v)
            std::fprintf(stderr, "  %-14s %9.1f ms  %5.1f%%  (%.3f ms/token)\n", e.second.c_str(), e.first,
                         100.0 * e.first / std::max(1e-9, tot), e.first / (double) std::max<int64_t>(1, S->tokens));
    }
    for (auto e : S->pev) cudaEventDestroy(e);
    S->mq.reset();
    if (S->blas) cublasDestroy(S->blas);
    for (int b = 0; b < PrefillState::NG; ++b) {
        if (S->ev_ready[b]) cudaEventDestroy(S->ev_ready[b]);
        if (S->ev_free[b]) cudaEventDestroy(S->ev_free[b]);
    }
    for (auto e : S->ev_land)
        if (e) cudaEventDestroy(e);
    if (S->ev_hop) cudaEventDestroy(S->ev_hop);
    if (S->gpin) cudaFreeHost(S->gpin);
    if (S->emb_h) cudaFreeHost(S->emb_h);
    if (S->hop_h) cudaFreeHost(S->hop_h);
    if (S->h_counts) cudaFreeHost(S->h_counts);
    if (S->h_base) cudaFreeHost(S->h_base);
    if (S->h_bounds) cudaFreeHost(S->h_bounds);
    delete S;
    pf_ = nullptr;
}

// ---------------------------------------------------------------- one half's layers over a chunk
bool Glm5Model::prefill_half(int64_t p0, int T, std::string& err, const int32_t* next_ids) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const cudaStream_t s = F->cs;
    const int E = g.n_embd, DI = g.d_inner(), HD = g.kda_head_dim, K = g.n_exp_used;
    const float one = 1.0f, zero = 0.0f;

    // Y[t][N] (row stride ldy) = alpha * X[t][K] (F32, row stride ldx) . W[N][K]^T, W BF16 (widened in slices)
    const auto sgemm_bf16 = [&](const uint16_t* W, int N, int Kd, const float* X, int ldx, float* Y, int ldy, int Tn,
                                float alpha) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w32_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            gb::bf16_to_f32(W + (size_t) r0 * Kd, S->w32, (int64_t) n * Kd, s);
            blas_ck(cublasSgemm(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &alpha, S->w32, Kd, X, ldx, &zero, Y + r0,
                                ldy),
                    "sgemm");
        }
    };
    // Y[t][N] = X16[t][K] . deq(W)[N][K]^T (tensor cores, F32 accumulate); beta = 1 adds to Y
    const auto hgemm_q = [&](const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y, int ldy, int Tn,
                             float beta) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w16_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, S->w16, s);
            blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &one, S->w16, CUDA_R_16F, Kd, X16,
                                 CUDA_R_16F, ldx, &beta, Y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "gemm f16");
        }
    };
    // the mHC read half for the whole chunk: (write half of the previous block,) mix dots, gates, x / x16
    const auto hc_read = [&](const float* block_out, const uint16_t* fn, const float* scale, const float* base,
                             const float* norm) {
        gb::hc_update(block_out, S->R, S->post, S->comb, S->ss, T, E, s);
        sgemm_bf16(fn, 24, 4 * E, S->R, 4 * E, S->mix, 24, T, 1.0f);
        gb::HcFinishArgs h;
        h.mix = S->mix;
        h.ss = S->ss;
        h.R = S->R;
        h.w_scale = scale;
        h.w_base = base;
        h.norm_w = norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = E;
        h.T = T;
        h.pre = S->pre;
        h.post = S->post;
        h.comb = S->comb;
        h.x = S->x;
        h.x16 = S->x16;
        gb::hc_finish(h, s);
    };

    // GLM_CB_DIR seam dumps of the chunk's LAST row, under the token path's names (debug only, synchronous)
    static const char* cb_dir = getenv("GLM_CB_DIR");
    const auto dump_row = [&](const std::string& name, const float* rows, int width, int row = -1) {
        if (cb_dir == nullptr || !cb_dir[0]) return;
        cudaStreamSynchronize(s);
        std::vector<float> buf((size_t) width);
        cudaMemcpy(buf.data(), rows + (size_t) (row < 0 ? T - 1 : row) * width, (size_t) width * sizeof(float),
                   cudaMemcpyDeviceToHost);
        if (FILE* f = std::fopen((std::string(cb_dir) + "/" + name + ".f32").c_str(), "wb")) {
            std::fwrite(buf.data(), 4, buf.size(), f);
            std::fclose(f);
        }
    };

    // ---- the disk reader: one thread for the chunk, reading the experts the MoE layers append - (layer, expert) in
    //      order - into the pinned landing ring, while the resident product and the RAM-tier groups run (Mercury,
    //      2.6k-token prompt: 5.1 -> 4.5 ms/token).  A layer appends its disk-only experts once its plan is known.
    //      STRATA_GLM_PREFILL_PRED_T=<tokens>: from that chunk size on, a layer also appends the NEXT MoE layer's
    //      whole disk-only set (a superset of what it will route), so those reads overlap its attention too - measured
    //      SLOWER on Mercury (a 2k chunk left ~40% of the disk-only experts unrouted: wasted reads on the one NVMe), so
    //      off by default.  A slot is refilled once the copy that read it ran (ev_land) and the layers moved past it.
    constexpr int KL = PrefillState::NG * PrefillState::GE;
    constexpr int kPredT = 0;
    struct DiskQ {
        std::mutex mu;
        std::vector<std::pair<int, int>> q;
        std::atomic<int> n{0}, landed{0}, copied{0};
        std::atomic<bool> stop{false};
        int push(int il, int e) {
            std::lock_guard<std::mutex> lk(mu);
            q.emplace_back(il, e);
            n.store((int) q.size(), std::memory_order_release);
            return (int) q.size() - 1;
        }
    } dq;
    // STRATA_GLM_DISK_QD=<n>: experts read at once (one batch across the workers) when they are queued and their
    // slots free - an NVMe gives more with a few in flight (Uranus 1.87 -> 2.2 GB/s at 2, Mercury 2.65 -> 2.96)
    static const int disk_qd = [] {
        const char* v = getenv("STRATA_GLM_DISK_QD");
        return std::max(1, std::min(4, v ? std::atoi(v) : 2));
    }();
    std::thread reader([&] {
        cudaSetDevice(dev_);
        // each part in pieces across the workers, as the decode's disk path (STRATA_GLM_READ_CHUNKS)
        static const int kChunks = [] {
            const char* v = getenv("STRATA_GLM_READ_CHUNKS");
            return std::max(1, std::min(16, v ? std::atoi(v) : 8));
        }();
        for (int k = 0;;) {
            while (dq.n.load(std::memory_order_acquire) <= k ||
                   (k >= KL && dq.copied.load(std::memory_order_acquire) < k - KL + 1)) {
                if (dq.stop.load()) return;
                std::this_thread::yield();
            }
            // this one, and the ones after it that are queued and whose slots are free already
            int nb = 1;
            while (nb < disk_qd && dq.n.load(std::memory_order_acquire) > k + nb &&
                   (k + nb < KL || dq.copied.load(std::memory_order_acquire) >= k + nb - KL + 1))
                ++nb;
            std::pair<int, int> it[4];
            uint8_t* land[4];
            for (int b = 0; b < nb; ++b) {
                cudaEventSynchronize(S->ev_land[(k + b) % KL]);   // the slot's last copy (this chunk's or earlier)
                std::lock_guard<std::mutex> lk(dq.mu);
                it[b] = dq.q[(size_t) (k + b)];
                land[b] = S->gpin + (size_t) ((k + b) % KL) * S->gstride;
            }
            F->workers->run(nb * 3 * kChunks, [&](int job) {
                const int b = job / (3 * kChunks), part = job % (3 * kChunks);
                fast_read_part(it[b].first, it[b].second, part / kChunks, land[b], part % kChunks, kChunks);
            });
            k += nb;
            dq.landed.store(k, std::memory_order_release);
        }
    });
    struct Joiner {
        std::thread& t;
        DiskQ& q;
        uint64_t& issued;
        ~Joiner() {
            q.stop.store(true);
            if (t.joinable()) t.join();
            issued += (uint64_t) q.n.load();
        }
    } joiner{reader, dq, S->disk_issued};
    // STRATA_GLM_PREFILL_PRED_T=<tokens>: the chunk size from which the next layer's disk set is read ahead (0 = never)
    static const int pred_t = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_PRED_T");
        return v ? std::atoi(v) : kPredT;
    }();
    int pred_layer = -1, pred0 = 0;   // the predicted segment: its layer, first index and experts (ascending)
    std::vector<int> pred_e;
    // the experts of layer il that are neither resident in its main slots nor held by the RAM tier (ascending)
    const auto disk_only = [&](int il) {
        std::vector<int> v;
        const auto& P = F->lp[(size_t) il];
        const auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        std::vector<char> res((size_t) g.n_expert, 0);
        std::lock_guard<std::mutex> lk(F->mu);
        for (int sl = 0; sl < P.n_main; ++sl)
            if (P.st[(size_t) sl] == FastState::kResident && P.key[(size_t) sl] >= 0) res[(size_t) P.key[(size_t) sl]] = 1;
        for (int e = 0; e < g.n_expert; ++e) {
            if (res[(size_t) e]) continue;
            const int rs = F->ram_of[(size_t) il * g.n_expert + e];
            if (rs >= 0 && RC.st[(size_t) rs] == FastState::kRHold) continue;
            v.push_back(e);
        }
        return v;
    };

    S->mark("start", s);
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        hc_read(il == l0_ ? nullptr : S->ffn, Ly.hc_attn_fn, Ly.hc_attn_scale, Ly.hc_attn_base, Ly.attn_norm);
        S->mark("hc", s);
        dump_row("attn_norm-" + std::to_string(il), S->x, E);

        // ---- the mixer, in sub-batches of kSub tokens (the recurrences carry their state across)
        for (int t0 = 0; t0 < T; t0 += kSub) {
            const int tn = std::min(kSub, T - t0);
            const float* xs = S->x + (size_t) t0 * E;
            const uint16_t* x16 = S->x16 + (size_t) t0 * E;
            float* mixer = S->mixer + (size_t) t0 * E;
            if (Ly.recr) {
                Carve c{S->uni};
                KdaBufs B = carve_kda(c, (size_t) tn, g);
                hgemm_q(Ly.q, DI, E, x16, E, B.proj[0], DI, tn, 0.0f);
                hgemm_q(Ly.k, DI, E, x16, E, B.proj[1], DI, tn, 0.0f);
                hgemm_q(Ly.v, DI, E, x16, E, B.proj[2], DI, tn, 0.0f);
                sgemm_bf16(Ly.f_a, HD, E, xs, E, B.fa, HD, tn, 1.0f);
                sgemm_bf16(Ly.g_a, HD, E, xs, E, B.ga, HD, tn, 1.0f);
                sgemm_bf16(Ly.beta, g.n_head, E, xs, E, B.beta, g.n_head, tn, 1.0f);
                const float* pr[3] = {B.proj[0], B.proj[1], B.proj[2]};
                float* cv[3] = {B.conv[0], B.conv[1], B.conv[2]};
                float* cst = state_ + kda_conv_[(size_t) il];
                gb::kda_conv(pr, Ly.conv, cst, cv, tn, DI, g.d_conv, s);
                gb::kda_conv_state(pr, cst, tn, DI, g.d_conv, s);
                sgemm_bf16(Ly.f_b, DI, HD, B.fa, HD, B.g1, DI, tn, 1.0f);
                sgemm_bf16(Ly.g_b, DI, HD, B.ga, HD, B.g2, DI, tn, 1.0f);
                gb::kda_rec(B.conv[0], B.conv[1], B.conv[2], B.g1, Ly.dt_bias, Ly.ssm_a, g.kda_lb, B.beta,
                            state_ + kda_S_[(size_t) il], B.g2, Ly.ssm_norm, g.norm_eps, g.n_head, tn, B.out16, s);
                hgemm_q(Ly.out, E, DI, B.out16, DI, mixer, E, tn, 0.0f);
            } else {
                Carve c{S->uni};
                DsaBufs B = carve_dsa(c, (size_t) tn, g, S->max_pools);
                const int pt = (int) (p0 + t0);
                const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
                hgemm_q(Ly.q_a, g.q_lora, E, x16, E, B.qr_raw, g.q_lora, tn, 0.0f);
                hgemm_q(Ly.kv_a, g.kv_lora, E, x16, E, B.kv_raw, g.kv_lora, tn, 0.0f);
                sgemm_bf16(Ly.idx_k, g.idx_key, E, xs, E, B.ik_raw, g.idx_key, tn, 1.0f);
                sgemm_bf16(Ly.idx_gate, g.idx_key, E, xs, E, B.ig_raw, g.idx_key, tn, 1.0f);
                sgemm_bf16(Ly.idx_proj, g.idx_heads, E, xs, E, B.iw, g.idx_heads, tn, prescale);
                gb::DsaPrepArgs d;
                d.qr_raw = B.qr_raw;
                d.q_a_norm = Ly.q_a_norm;
                d.qr = B.qr;
                d.qr16 = B.qr16;
                d.q_lora = g.q_lora;
                d.kv_raw = B.kv_raw;
                d.kv_norm = Ly.kv_a_norm;
                d.lat = state_ + dsa_lat_[(size_t) il];
                d.kv_lora = g.kv_lora;
                d.ik_raw = B.ik_raw;
                d.k_norm_w = Ly.k_norm_w;
                d.k_norm_b = Ly.k_norm_b;
                d.ik_cache = state_ + dsa_ik_[(size_t) il];
                d.ig_raw = B.ig_raw;
                d.ig_cache = state_ + dsa_ig_[(size_t) il];
                d.idx_key = g.idx_key;
                d.p0 = pt;
                d.T = tn;
                d.eps = g.norm_eps;
                gb::dsa_prep(d, s);
                // the pools completed inside this sub-batch: pool pi ends at position (pi + 1) * kpool - 1
                const int kp = g.idx_kpool;
                const int pool_lo = (pt + kp) / kp - 1, pool_hi = (pt + tn) / kp - 1;
                if (pool_hi >= pool_lo)
                    gb::dsa_pool(state_ + dsa_ik_[(size_t) il], state_ + dsa_ig_[(size_t) il], Ly.ape,
                                 state_ + dsa_pool_[(size_t) il], g.idx_key, kp, pool_lo, pool_hi - pool_lo + 1, s);
                hgemm_q(Ly.q_b, g.n_head * g.qk_nope, g.q_lora, B.qr16, g.q_lora, B.q, g.n_head * g.qk_nope, tn, 0.0f);
                sgemm_bf16(Ly.idx_q_b, g.idx_heads * g.idx_key, g.q_lora, B.qr, g.q_lora, B.iq, g.idx_heads * g.idx_key,
                           tn, 1.0f);
                const int max_vis = (pt + tn) / kp;
                if (max_vis > g.top_pools_max())
                    gb::dsa_score(B.iq, state_ + dsa_pool_[(size_t) il], B.iw, g.idx_key, g.idx_heads, pt, kp, tn,
                                  max_vis, B.score, B.score_ld, s);
                gb::dsa_select(B.score, B.score_ld, pt, kp, g.top_pools_max(), g.idx_select_tail, tn, g.n_sel_max(),
                               B.cells, B.n_sel, s);
                // q_abs[t][h] = wk_b[h] (kv_lora x qk_nope) . q[t][h]
                gb::bf16_to_f32(Ly.k_b, S->w32, (int64_t) g.n_head * g.kv_lora * g.qk_nope, s);
                blas_ck(cublasSgemmStridedBatched(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.kv_lora, tn, g.qk_nope, &one,
                                                  S->w32, g.qk_nope, (long long) g.kv_lora * g.qk_nope, B.q,
                                                  g.n_head * g.qk_nope, g.qk_nope, &zero, B.q_abs, g.n_head * g.kv_lora,
                                                  g.kv_lora, g.n_head),
                        "q_abs");
                gb::mla_attn(B.q_abs, state_ + dsa_lat_[(size_t) il], B.cells, B.n_sel, g.n_sel_max(), g.n_head,
                             g.kv_lora, 1.0f / std::sqrt((float) g.qk_nope), tn, B.ctx, s);
                // out[t][h] = wv_b[h] (v_head x kv_lora) . ctx[t][h]
                gb::bf16_to_f32(Ly.v_b, S->w32, (int64_t) g.n_head * g.v_head * g.kv_lora, s);
                blas_ck(cublasSgemmStridedBatched(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.v_head, tn, g.kv_lora, &one,
                                                  S->w32, g.kv_lora, (long long) g.v_head * g.kv_lora, B.ctx,
                                                  g.n_head * g.kv_lora, g.kv_lora, &zero, B.attn, g.n_head * g.v_head,
                                                  g.v_head, g.n_head),
                        "mla out");
                gb::f32_to_f16(B.attn, B.attn16, (int64_t) tn * g.n_head * g.v_head, s);
                hgemm_q(Ly.out, E, g.n_head * g.v_head, B.attn16, g.n_head * g.v_head, mixer, E, tn, 0.0f);
                if (t0 + tn == T) {
                    const std::string Ls = std::to_string(il);
                    dump_row("dsa_qr-" + Ls, B.qr, g.q_lora, tn - 1);
                    dump_row("dsa_q-" + Ls, B.q, g.n_head * g.qk_nope, tn - 1);
                    dump_row("dsa_iq-" + Ls, B.iq, g.idx_heads * g.idx_key, tn - 1);
                    dump_row("pf_qabs-" + Ls, B.q_abs, g.n_head * g.kv_lora, tn - 1);
                    dump_row("pf_ctx-" + Ls, B.ctx, g.n_head * g.kv_lora, tn - 1);
                    dump_row("pf_attn-" + Ls, B.attn, g.n_head * g.v_head, tn - 1);
                }
            }
        }

        dump_row("mixer-" + std::to_string(il), S->mixer, E);
        S->mark(Ly.recr ? "kda" : "dsa", s);
        hc_read(S->mixer, Ly.hc_ffn_fn, Ly.hc_ffn_scale, Ly.hc_ffn_base, Ly.ffn_norm);
        S->mark("hc", s);
        dump_row("ffn_norm-" + std::to_string(il), S->x, E);

        if (!Ly.moe) {
            for (int t0 = 0; t0 < T; t0 += kSub) {
                const int tn = std::min(kSub, T - t0);
                Carve c{S->uni};
                DenseBufs B = carve_dense(c, (size_t) tn, g);
                const uint16_t* x16 = S->x16 + (size_t) t0 * E;
                hgemm_q(Ly.ffn_gate, g.n_ff_dense, E, x16, E, B.dg, g.n_ff_dense, tn, 0.0f);
                hgemm_q(Ly.ffn_up, g.n_ff_dense, E, x16, E, B.du, g.n_ff_dense, tn, 0.0f);
                gb::swiglu_f16(B.dg, B.du, B.dh16, (int64_t) tn * g.n_ff_dense, g.swiglu_shexp, s);
                hgemm_q(Ly.ffn_down, E, g.n_ff_dense, B.dh16, g.n_ff_dense, S->ffn + (size_t) t0 * E, E, tn, 0.0f);
            }
            S->mark("dense_ffn", s);
            dump_row("ffn_out-" + std::to_string(il), S->ffn, E);
            continue;
        }

        // ---- MoE: the shared expert (sub-batches, straight into ffn), the routes, the expert groups
        Carve c{S->uni};
        MoeBufs M = carve_moe(c, (size_t) T, g);
        const int FF = g.n_ff_exp * g.n_shared, NE = g.n_expert, nff = g.n_ff_exp;
        for (int t0 = 0; t0 < T; t0 += kSub) {
            const int tn = std::min(kSub, T - t0);
            const uint16_t* x16 = S->x16 + (size_t) t0 * E;
            hgemm_q(Ly.sh_gate, FF, E, x16, E, M.sh_g, FF, tn, 0.0f);
            hgemm_q(Ly.sh_up, FF, E, x16, E, M.sh_u, FF, tn, 0.0f);
            gb::swiglu_f16(M.sh_g, M.sh_u, M.sh16, (int64_t) tn * FF, g.swiglu_shexp, s);
            hgemm_q(Ly.sh_down, E, FF, M.sh16, FF, S->ffn + (size_t) t0 * E, E, tn, 0.0f);
        }
        S->mark("shexp", s);
        sgemm_bf16(Ly.router, NE, E, S->x, E, M.logits, NE, T, 1.0f);
        gb::route(M.logits, Ly.router_bias, NE, K, g.w_scale, g.norm_w != 0, T, M.ids, M.rw, s);
        gb::expert_count(M.ids, T * K, NE, M.counts, M.rank, s);
        cudaMemcpyAsync(S->h_counts, M.counts, (size_t) NE * sizeof(int), cudaMemcpyDeviceToHost, s);
        S->mark("route", s);
        cudaStreamSynchronize(s);
        const auto tp = std::chrono::steady_clock::now();

        // ---- the plan (host): resident experts in slot order, then the staged ones in groups
        auto& P = F->lp[(size_t) il];
        auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        struct Staged {
            int e;
            const uint8_t* ram;   // the RAM tier slot, or null: disk
        };
        std::vector<Staged> staged;
        int nb = 0;   // h_bounds fill
        int rows_res = 0, max_res = 0;
        const int b_res = nb;
        // resident experts with at most light_rows rows go to the light kernels (gf::rows_experts): their rows come
        // after the MMQ product's, which sees them as empty
        static const int light_env = [] {
            const char* v = getenv("STRATA_GLM_PREFILL_LIGHT");
            return v ? std::atoi(v) : kLightRows;
        }();
        const auto lt_ok = [](int t) { return t == 16 || t == 18 || t == 19; };
        static const int light_layer = getenv("STRATA_GLM_PREFILL_LIGHT_LAYER") ? std::atoi(getenv("STRATA_GLM_PREFILL_LIGHT_LAYER")) : -1;
        const int light_rows = lt_ok(Ly.gu_type) && lt_ok(Ly.d_type) && (light_layer < 0 || il == light_layer) ? light_env : 0;
        int n_light = 0, rows_mmq = 0;
        int* h_light = S->h_bounds + kLightOff;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            S->h_bounds[nb++] = 0;
            for (int sl = 0; sl < P.n_main; ++sl) {
                int cnt = 0;
                if (P.st[(size_t) sl] == FastState::kResident && P.key[(size_t) sl] >= 0) {
                    const int e = P.key[(size_t) sl];
                    cnt = S->h_counts[e];
                    if (cnt > 0 && cnt <= light_rows && n_light < kLightMax) {
                        S->h_counts[e] = -cnt;   // claimed (rows assigned below)
                        h_light[3 * n_light] = sl;
                        h_light[3 * n_light + 1] = e;   // (the expert until its rows are known)
                        h_light[3 * n_light + 2] = cnt;
                        ++n_light;
                        cnt = 0;
                    } else if (cnt > 0) {
                        S->h_base[e] = rows_res;
                        S->h_counts[e] = -cnt;   // claimed
                    }
                }
                rows_res += std::max(0, cnt);
                max_res = std::max(max_res, cnt);
                S->h_bounds[nb++] = rows_res;
            }
            rows_mmq = rows_res;
            for (int i = 0; i < n_light; ++i) {
                const int e = h_light[3 * i + 1];
                S->h_base[e] = rows_res;
                h_light[3 * i + 1] = rows_res;
                rows_res += h_light[3 * i + 2];
            }
            for (int e = 0; e < NE; ++e) {
                if (S->h_counts[e] <= 0) continue;
                const int key = il * NE + e;
                const int rs = F->ram_of[(size_t) key];
                const uint8_t* src = nullptr;
                if (rs >= 0 && RC.st[(size_t) rs] == FastState::kRHold) src = RC.base + (size_t) rs * RC.stride;
                staged.push_back(Staged{e, src});
            }
            // the prompt's routing feeds the LFU counts the decode's tiers evict by
            uint64_t ev = 0;
            for (int e = 0; e < NE; ++e) {
                const int cnt = std::abs(S->h_counts[e]);
                if (cnt == 0) continue;
                uint32_t& cc = F->cnt[(size_t) il * NE + e];
                cc = (uint32_t) std::min<uint64_t>((uint64_t) cc + (uint64_t) cnt, 1u << 24);
                ev += (uint64_t) cnt;
                uint32_t& uu = F->usage[(size_t) il * NE + e];
                uu = (uint32_t) std::min<uint64_t>((uint64_t) uu + (uint64_t) cnt, 1u << 30);
                if (uu >= (1u << 30))
                    for (auto& u : F->usage) u >>= 1;
            }
            F->cnt_events += ev;
            while (F->cnt_events >= 32768) {
                F->cnt_events -= 32768;
                for (auto& cc : F->cnt) cc >>= 1;
            }
        }
        // RAM-tier experts first (DMA), disk ones last (their reads overlap the earlier groups)
        std::stable_partition(staged.begin(), staged.end(), [](const Staged& x) { return x.ram != nullptr; });
        int rows = rows_res;
        struct Group {
            int first, n, r0, nrows, max_rows, b_off;
        };
        std::vector<Group> groups;
        for (size_t i = 0; i < staged.size(); i += PrefillState::GE) {
            Group gr{(int) i, (int) std::min<size_t>(PrefillState::GE, staged.size() - i), rows, 0, 0, nb};
            S->h_bounds[nb++] = 0;
            for (int j = 0; j < gr.n; ++j) {
                const int e = staged[(size_t) gr.first + j].e;
                const int cnt = S->h_counts[e];
                S->h_base[e] = rows;
                rows += cnt;
                gr.nrows += cnt;
                gr.max_rows = std::max(gr.max_rows, cnt);
                S->h_bounds[nb++] = gr.nrows;
            }
            groups.push_back(gr);
        }
        if (rows != T * K) {
            err = "glm prefill: layer " + std::to_string(il) + " planned " + std::to_string(rows) + " of " +
                  std::to_string(T * K) + " routed rows";
            return false;
        }
        if (nb > kLightOff) {
            err = "glm prefill: the expert bounds overflowed";
            return false;
        }
        cudaMemcpyAsync(M.base, S->h_base, (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(M.bounds, S->h_bounds, (size_t) nb * sizeof(int), cudaMemcpyHostToDevice, s);
        if (n_light > 0)
            cudaMemcpyAsync(M.bounds + kLightOff, h_light, (size_t) 3 * n_light * sizeof(int), cudaMemcpyHostToDevice, s);
        gb::expert_scatter(M.ids, M.rank, M.base, T * K, K, M.row_tok, M.pos, s);
        S->mark("plan", s);
        S->ms_plan += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();

        // one expert set through gate/up, swiglu and down: rows [r0, r0 + nrows) of the sorted order
        const auto run_set = [&](const uint8_t* wbase, int n_exp, size_t stride, const int* d_bounds, int r0, int nrows,
                                 int max_rows) {
            if (nrows <= 0) return;
            mmq::quantize(S->x, M.row_tok + r0, M.Xq, Ly.gu_type, E, E, nrows, s);
            float* GU = M.OUT + (size_t) r0 * E;   // 2 * n_ff == n_embd: the set's own OUT rows hold its gate/up
            mmq::Product gu;
            gu.w = wbase;
            gu.type = Ly.gu_type;
            gu.w_rows = 2 * nff;
            gu.w_cols = E;
            gu.expert_bytes = stride;
            gu.n = n_exp;
            gu.xq = M.Xq;
            gu.bounds = d_bounds;
            gu.ids = S->iota;
            gu.total_rows = nrows;
            gu.max_rows = max_rows;
            gu.dst = GU;
            gu.ld_dst = 2 * nff;
            S->mq->run(gu, s);
            gb::swiglu_rows(GU, M.H, nrows, nff, g.swiglu_exp, s);
            mmq::quantize(M.H, nullptr, M.Hq, Ly.d_type, nff, nff, nrows, s);
            mmq::Product dn;
            dn.w = wbase + Ly.down_off;
            dn.type = Ly.d_type;
            dn.w_rows = E;
            dn.w_cols = nff;
            dn.expert_bytes = stride;
            dn.n = n_exp;
            dn.xq = M.Hq;
            dn.bounds = d_bounds;
            dn.ids = S->iota;
            dn.total_rows = nrows;
            dn.max_rows = max_rows;
            dn.dst = M.OUT + (size_t) r0 * E;
            dn.ld_dst = E;
            S->mq->run(dn, s);
        };
        if (2 * nff != E) {
            err = "glm prefill: 2 * n_ff_exp != n_embd (the gate/up rows would not fit the set's output rows)";
            return false;
        }
        // the disk-only experts (staged last, ascending): their indices in the reader's order - inside the segment
        // predicted one layer ago, or appended now
        std::vector<int> dlist, gidx;   // staged indices of the disk experts; their reader indices
        for (size_t i = 0; i < staged.size(); ++i)
            if (staged[i].ram == nullptr) dlist.push_back((int) i);
        int seg_end = dq.n.load();
        if (pred_layer == il) {
            size_t p = 0;
            for (int k : dlist) {
                const int e = staged[(size_t) k].e;
                while (p < pred_e.size() && pred_e[p] != e) ++p;
                if (p == pred_e.size()) break;
                gidx.push_back(pred0 + (int) p);
            }
            seg_end = pred0 + (int) pred_e.size();
            if (gidx.size() != dlist.size()) {   // (cannot happen: the plan's disk set is within the prediction)
                dq.copied.store(seg_end, std::memory_order_release);   // the predicted slots are free again
                gidx.clear();
            }
        }
        if (gidx.size() != dlist.size()) {
            for (int k : dlist) gidx.push_back(dq.push(il, staged[(size_t) k].e));
            seg_end = dq.n.load();
        }
        pred_layer = -1;
        if (pred_t > 0 && T >= pred_t && il + 1 < l1_ && F->L[(size_t) il + 1].moe) {
            pred_e = disk_only(il + 1);
            pred0 = dq.n.load();
            for (int e : pred_e) dq.push(il + 1, e);
            pred_layer = il + 1;
        }
        // the resident experts: the lightly routed ones through the light kernels, the rest in one MMQ product over
        // the layer's whole pool partition (the light ones empty there), while the copy stream stages
        if (n_light > 0 &&
            !gf::rows_experts(Ly.gu_type, Ly.d_type, P.base, P.stride, Ly.down_off, M.bounds + kLightOff, n_light,
                              M.row_tok, S->x, T, E, nff, g.swiglu_exp, rows_mmq, rows_res, M.Xq, M.Hq, M.OUT, E, s)) {
            err = "glm prefill: the light expert kernels refused the layer's types";
            return false;
        }
        S->mark("moe_light", s);
        // STRATA_GLM_PREFILL_LIGHT_CHECK=1 (debug): the light rows again through MMQ, compared row by row
        static const bool light_check = getenv("STRATA_GLM_PREFILL_LIGHT_CHECK") != nullptr;
        if (light_check && n_light > 0) {
            const int nl_rows = rows_res - rows_mmq;
            std::vector<float> A((size_t) nl_rows * E), B((size_t) nl_rows * E);
            cudaStreamSynchronize(s);
            cudaMemcpy(A.data(), M.OUT + (size_t) rows_mmq * E, A.size() * sizeof(float), cudaMemcpyDeviceToHost);
            int* hb = S->h_bounds + 4096;
            int nbc = 0, max_l = 0, li = 0;
            hb[nbc++] = 0;
            for (int sl = 0; sl < P.n_main; ++sl) {
                int add = 0;
                if (li < n_light && h_light[3 * li] == sl) {
                    add = h_light[3 * li + 2];
                    max_l = std::max(max_l, add);
                    ++li;
                }
                hb[nbc] = hb[nbc - 1] + add;
                ++nbc;
            }
            cudaMemcpy(M.bounds + 4096, hb, (size_t) nbc * sizeof(int), cudaMemcpyHostToDevice);
            run_set(P.base, P.n_main, P.stride, M.bounds + 4096, rows_mmq, nl_rows, max_l);
            cudaStreamSynchronize(s);
            cudaMemcpy(B.data(), M.OUT + (size_t) rows_mmq * E, B.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double num = 0, den = 0, worst = 0;
            for (int r = 0; r < nl_rows; ++r) {
                double rn = 0, rd = 0;
                for (int j = 0; j < E; ++j) {
                    const double dlt = (double) A[(size_t) r * E + j] - B[(size_t) r * E + j];
                    rn += dlt * dlt;
                    rd += (double) B[(size_t) r * E + j] * B[(size_t) r * E + j];
                }
                num += rn;
                den += rd;
                worst = std::max(worst, std::sqrt(rn / std::max(1e-30, rd)));
            }
            std::fprintf(stderr, "glm prefill light check layer %d: %d experts, %d rows, rel L2 %.3e, worst row %.3e\n", il,
                         n_light, nl_rows, std::sqrt(num / std::max(1e-30, den)), worst);
        }
        run_set(P.base, P.n_main, P.stride, M.bounds + b_res, 0, rows_mmq, max_res);
        S->rows_resident += (uint64_t) rows_res;
        S->mark("moe_resident", s);
        int dk = 0;   // the next disk expert (index into dlist / gidx)
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const Group& gr = groups[gi];
            const int b = (int) (gi % PrefillState::NG);
            uint8_t* gdst = S->gbuf + (size_t) b * PrefillState::GE * P.stride;
            cudaStreamWaitEvent(F->copy, S->ev_free[b], 0);
            const int dk0 = dk;
            for (int j = 0; j < gr.n; ++j) {
                const Staged& st = staged[(size_t) gr.first + j];
                const uint8_t* src = st.ram;
                if (src == nullptr) {
                    const int gi2 = gidx[(size_t) dk];
                    // every index below this one was copied (its event recorded) or predicted and never routed:
                    // released, so the reader can get to this one however far the prediction's gaps put it
                    if (dq.copied.load() < gi2) dq.copied.store(gi2, std::memory_order_release);
                    if (dq.landed.load(std::memory_order_acquire) <= gi2) {
                        const auto td = std::chrono::steady_clock::now();
                        while (dq.landed.load(std::memory_order_acquire) <= gi2) glmfast::cpu_relax();
                        S->ms_disk += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
                    }
                    src = S->gpin + (size_t) (gi2 % KL) * S->gstride;
                    ++dk;
                }
                cudaMemcpyAsync(gdst + (size_t) j * P.stride, src, Ly.blob, cudaMemcpyHostToDevice, F->copy);
                if (st.ram == nullptr) {   // its landing slot is free once this copy ran
                    cudaEventRecord(S->ev_land[gidx[(size_t) dk - 1] % KL], F->copy);
                    dq.copied.store(gidx[(size_t) dk - 1] + 1, std::memory_order_release);
                }
                S->staged_ram += st.ram != nullptr;
            }
            (void) dk0;
            cudaEventRecord(S->ev_ready[b], F->copy);
            cudaStreamWaitEvent(s, S->ev_ready[b], 0);
            run_set(gdst, gr.n, P.stride, M.bounds + gr.b_off, gr.r0, gr.nrows, gr.max_rows);
            cudaEventRecord(S->ev_free[b], s);
            S->rows_staged += (uint64_t) gr.nrows;
        }
        dq.copied.store(std::max(dq.copied.load(), seg_end), std::memory_order_release);   // predicted, never routed
        S->staged_disk += dlist.size();
        S->mark("moe_staged", s);
        dump_row("pf_shexp-" + std::to_string(il), S->ffn, E);
        gb::moe_combine(M.OUT, M.pos, M.rw, S->ffn, T, K, E, S->ffn, s);
        S->mark("combine", s);
        dump_row("ffn_out-" + std::to_string(il), S->ffn, E);
    }
    // the last layer's write half: R = post x ffn + comb . R
    gb::hc_update(S->ffn, S->R, S->post, S->comb, nullptr, T, E, s);
    S->mark("hc", s);
    // ---- the NextN block's caches for these positions (the half that carries it): position p reads h_p and the
    //      embedding of the token at p + 1, so only the cache-writing half of the block runs - eh_proj, the attention
    //      norm, kv_a and the indexer's key/gate - no queries, no attention, no FFN
    if (mtp_il_ >= 0 && next_ids != nullptr) {
        int Tv = 0;   // rows whose next token is known (all but possibly the prompt's last)
        while (Tv < T && next_ids[Tv] >= 0) ++Tv;
        if (Tv > 0) {
            const auto& Ly = F->L[(size_t) mtp_il_];
            const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
            if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
                err = "glm prefill: no embedding dequantizer for the draft block";
                return false;
            }
            const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, E);
            cudaStreamSynchronize(s);   // emb_h: the host writes it now
            for (int t = 0; t < Tv; ++t)
                tt->to_float(pack_emb_src_ + (size_t) next_ids[t] * row_b, S->emb_h + (size_t) t * E, E);
            Carve c{S->uni};
            float* h = c.take<float>((size_t) Tv * E);
            float* emb = c.take<float>((size_t) Tv * E);
            uint16_t* cat16 = c.take<uint16_t>((size_t) Tv * 2 * E);
            float* hid = c.take<float>((size_t) Tv * E);
            float* kv = c.take<float>((size_t) Tv * g.kv_lora);
            float* ik = c.take<float>((size_t) Tv * g.idx_key);
            float* ig = c.take<float>((size_t) Tv * g.idx_key);
            cudaMemcpyAsync(emb, S->emb_h, (size_t) Tv * E * sizeof(float), cudaMemcpyHostToDevice, s);
            gb::head_rows(S->R, w_.at("output_norm.weight"), g.norm_eps, Tv, E, h, s);
            gb::mtp_in_rows(emb, h, Ly.enorm, Ly.hnorm, g.norm_eps, Tv, E, cat16, s);
            hgemm_q(Ly.eh, E, 2 * E, cat16, 2 * E, hid, E, Tv, 0.0f);
            gb::rms_rows(hid, Ly.attn_norm, g.norm_eps, Tv, E, S->x, S->x16, s);
            hgemm_q(Ly.kv_a, g.kv_lora, E, S->x16, E, kv, g.kv_lora, Tv, 0.0f);
            sgemm_bf16(Ly.idx_k, g.idx_key, E, S->x, E, ik, g.idx_key, Tv, 1.0f);
            sgemm_bf16(Ly.idx_gate, g.idx_key, E, S->x, E, ig, g.idx_key, Tv, 1.0f);
            gb::DsaPrepArgs d;
            d.kv_raw = kv;
            d.kv_norm = Ly.kv_a_norm;
            d.lat = state_ + dsa_lat_[(size_t) mtp_il_];
            d.kv_lora = g.kv_lora;
            d.ik_raw = ik;
            d.k_norm_w = Ly.k_norm_w;
            d.k_norm_b = Ly.k_norm_b;
            d.ik_cache = state_ + dsa_ik_[(size_t) mtp_il_];
            d.ig_raw = ig;
            d.ig_cache = state_ + dsa_ig_[(size_t) mtp_il_];
            d.idx_key = g.idx_key;
            d.p0 = (int) p0;
            d.T = Tv;
            d.eps = g.norm_eps;
            gb::dsa_prep(d, s);
            const int kp = g.idx_kpool;
            const int pool_lo = (int) ((p0 + kp) / kp - 1), pool_hi = (int) ((p0 + Tv) / kp - 1);
            if (pool_hi >= pool_lo)
                gb::dsa_pool(state_ + dsa_ik_[(size_t) mtp_il_], state_ + dsa_ig_[(size_t) mtp_il_], Ly.ape,
                             state_ + dsa_pool_[(size_t) mtp_il_], g.idx_key, kp, pool_lo, pool_hi - pool_lo + 1, s);
            S->mark("mtp_cache", s);
        }
    }
    S->collect();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm prefill: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::dump_state(const std::string& path) {
    int half = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get(), ++half) {
        cudaSetDevice(m->dev_);
        cudaDeviceSynchronize();
        std::vector<float> buf(m->state_bytes_ / sizeof(float));
        if (cudaMemcpy(buf.data(), m->state_, buf.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        const std::string base = path + "." + std::to_string(half);
        FILE* f = std::fopen((base + ".bin").c_str(), "wb");
        if (f == nullptr) return false;
        std::fwrite(buf.data(), sizeof(float), buf.size(), f);
        std::fclose(f);
        FILE* t = std::fopen((base + ".txt").c_str(), "w");
        if (t == nullptr) return false;
        std::fprintf(t, "max_ctx %lld pos %lld\n", (long long) m->max_ctx_, (long long) m->pos_);
        for (int il = m->l0_; il < m->l1_; ++il)
            std::fprintf(t, "%d %lld %lld %lld %lld %lld %lld\n", il, (long long) m->kda_S_[(size_t) il],
                         (long long) m->kda_conv_[(size_t) il], (long long) m->dsa_lat_[(size_t) il],
                         (long long) m->dsa_ik_[(size_t) il], (long long) m->dsa_ig_[(size_t) il],
                         (long long) m->dsa_pool_[(size_t) il]);
        std::fclose(t);
    }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the prompt
bool Glm5Model::prefill(const std::vector<int32_t>& tokens, std::string& err, int32_t next_token) {
    err.clear();
    prefill_next_ = next_token;
    if (fast_ == nullptr || tokens.empty()) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())   // (a tail lent to the vision encoder: the
        if (m->pf_ == nullptr || m->fast_ == nullptr || m->vis_lent_) return false;   // token path, until it is back)
    // a handful of tokens costs less through the token path (the batched path streams every expert they touch)
    int64_t min_n = 32;
    if (const char* mn = getenv("STRATA_GLM_PREFILL_MIN")) min_n = std::max<int64_t>(1, std::atoll(mn));
    if ((int64_t) tokens.size() < min_n) return false;
    // the chunk for this prompt: no bigger than it needs (rounded up to 64), so it borrows only that much; a prompt
    // longer than the largest chunk is cut into EQUAL chunks (the two halves pipeline chunks)
    int Tmax = pf_->T;
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) Tmax = std::min(Tmax, m->pf_->T);
    const int64_t nn = (int64_t) tokens.size(), nchunks = (nn + Tmax - 1) / Tmax;
    const int Trun = (int) std::min<int64_t>(Tmax, ((nn + nchunks - 1) / nchunks + 63) / 64 * 64);
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        m->prefill_carve(Trun);
        m->prefill_lend();
    }
    const bool ok = prefill_run(tokens, err);
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->prefill_return();
    cudaSetDevice(dev_);
    return ok;
}

bool Glm5Model::prefill_run(const std::vector<int32_t>& tokens, std::string& err) {
    const Glm5Geometry& g = g_;
    const int64_t n = (int64_t) tokens.size();
    if (pos_ + n > max_ctx_) {
        err = "glm prefill: the prompt (" + std::to_string(pos_ + n) + " positions) exceeds the context (" +
              std::to_string(max_ctx_) + ")";
        return false;
    }
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm prefill: no embedding dequantizer";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int E = g.n_embd;
    const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, E);
    int Tc = pf_->T_bound;
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) Tc = std::min(Tc, m->pf_->T_bound);
    const int64_t pos_start = pos_;
    const int nc = (int) ((n + Tc - 1) / Tc);
    const auto chunk_len = [&](int c) { return (int) std::min<int64_t>(Tc, n - (int64_t) c * Tc); };
    // the token after every position of chunk c (the draft block's caches read it): the prompt's own next token,
    // the caller's for the last position (-1: unknown - that position is filled by the first draft of the decode)
    const auto next_of = [&](int c, std::vector<int32_t>& out) {
        const int T = chunk_len(c);
        out.resize((size_t) T);
        for (int t = 0; t < T; ++t) {
            const int64_t j = (int64_t) c * Tc + t + 1;
            out[(size_t) t] = j < n ? tokens[(size_t) j] : prefill_next_;
        }
    };
    // chunk c's embedding rows into this half's R (host-dequantized from the shard mapping)
    const auto embed = [&](int c) {
        PrefillState* S = pf_;
        const int T = chunk_len(c);
        const int64_t i = (int64_t) c * Tc;
        cudaStreamSynchronize(fast_->cs);   // emb_h may still feed the previous chunk's copy
        for (int t = 0; t < T; ++t) {
            tt->to_float(pack_emb_src_ + (size_t) tokens[(size_t) (i + t)] * row_b, S->emb_h + (size_t) t * E, E);
            if (const float* img = image_row(pos_start + i + t))   // an image's row in place of its <|image|> token
                std::memcpy(S->emb_h + (size_t) t * E, img, (size_t) E * sizeof(float));
        }
        // the embedding lands in x (free until the first layer's gates write it) and fans out to the 4 streams
        cudaMemcpyAsync(S->x, S->emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, fast_->cs);
        gb::embed_rows(S->x, S->R, T, E, fast_->cs);
    };
    Glm5Model* B = split_next_.get();
    const bool pipelined = B != nullptr && B->split_next_ == nullptr && nc > 1 &&
                           getenv("STRATA_GLM_PREFILL_SERIAL") == nullptr;
    bool cancelled = false;
    if (!pipelined) {
        // one half after the other, chunk by chunk
        for (int c = 0; c < nc; ++c) {
            const int T = chunk_len(c);
            const int64_t p0 = pos_start + (int64_t) c * Tc;
            cudaSetDevice(dev_);
            embed(c);
            std::vector<int32_t> nx0;
            if (mtp_il_ >= 0) next_of(c, nx0);   // a single device carries the draft block itself
            if (!prefill_half(p0, T, err, nx0.empty() ? nullptr : nx0.data())) return false;
            Glm5Model* prev = this;
            for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) {
                PrefillState* SP = prev->pf_;
                const size_t hop = (size_t) T * 4 * E * sizeof(float);
                cudaSetDevice(prev->dev_);
                cudaMemcpyAsync(SP->hop_h, SP->R, hop, cudaMemcpyDeviceToHost, prev->fast_->cs);
                cudaEventRecord(SP->ev_hop, prev->fast_->cs);
                cudaSetDevice(m->dev_);
                cudaStreamWaitEvent(m->fast_->cs, SP->ev_hop, 0);
                cudaMemcpyAsync(m->pf_->R, SP->hop_h, hop, cudaMemcpyHostToDevice, m->fast_->cs);
                m->pos_ = p0;
                std::vector<int32_t> nx;
                if (m->mtp_il_ >= 0) next_of(c, nx);
                if (!m->prefill_half(p0, T, err, nx.empty() ? nullptr : nx.data())) return false;
                prev = m;
            }
            pos_ = p0 + T;
            for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) m->pos_ = pos_;
            ++pf_->chunks;
            if (prefill_progress && c + 1 < nc && !prefill_progress(pos_ - pos_start, n)) {
                cancelled = true;
                break;
            }
        }
    } else {
        // THE PIPELINE: this half reads chunk c + 1 while the tail half reads chunk c - the residual rows cross over
        // through two pinned host slots (slot c & 1 is free again once the tail has copied chunk c - 2 in)
        std::mutex mu;
        std::condition_variable cv;
        int produced = 0, consumed = 0;
        bool stop = false;
        std::string tail_err;
        const size_t slot = (size_t) Tc * 4 * E;
        std::thread tail([&] {
            cudaSetDevice(B->dev_);
            for (int c = 0; c < nc; ++c) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return produced > c || stop; });
                    if (produced <= c) return;
                }
                const int T = chunk_len(c);
                const int64_t p0 = pos_start + (int64_t) c * Tc;
                cudaMemcpyAsync(B->pf_->R, pf_->hop_h + (size_t) (c & 1) * slot, (size_t) T * 4 * E * sizeof(float),
                                cudaMemcpyHostToDevice, B->fast_->cs);
                cudaStreamSynchronize(B->fast_->cs);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    consumed = c + 1;
                }
                cv.notify_all();
                B->pos_ = p0;
                std::string e2;
                std::vector<int32_t> nx;
                if (B->mtp_il_ >= 0) next_of(c, nx);
                if (!B->prefill_half(p0, T, e2, nx.empty() ? nullptr : nx.data())) {
                    std::lock_guard<std::mutex> lk(mu);
                    tail_err = e2.empty() ? "glm prefill: the tail half failed" : e2;
                    stop = true;
                    cv.notify_all();
                    return;
                }
                cudaStreamSynchronize(B->fast_->cs);
                if (prefill_progress && c + 1 < nc && !prefill_progress(std::min<int64_t>(n, (int64_t) (c + 1) * Tc), n)) {
                    std::lock_guard<std::mutex> lk(mu);
                    tail_err = "cancelled";
                    stop = true;
                    cv.notify_all();
                    return;
                }
            }
        });
        cudaSetDevice(dev_);
        for (int c = 0; c < nc; ++c) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return consumed >= c - 1 || stop; });
                if (stop) break;
            }
            const int T = chunk_len(c);
            const int64_t p0 = pos_start + (int64_t) c * Tc;
            embed(c);
            std::string e1;
            if (!prefill_half(p0, T, e1)) {
                std::lock_guard<std::mutex> lk(mu);
                if (tail_err.empty()) tail_err = e1.empty() ? "glm prefill: the head half failed" : e1;
                stop = true;
                cv.notify_all();
                break;
            }
            cudaMemcpyAsync(pf_->hop_h + (size_t) (c & 1) * slot, pf_->R, (size_t) T * 4 * E * sizeof(float),
                            cudaMemcpyDeviceToHost, fast_->cs);
            cudaStreamSynchronize(fast_->cs);
            {
                std::lock_guard<std::mutex> lk(mu);
                produced = c + 1;
            }
            cv.notify_all();
            ++pf_->chunks;
        }
        tail.join();
        if (stop) {
            if (tail_err == "cancelled") {
                cancelled = true;
            } else {
                err = tail_err;
                return false;
            }
        }
        pos_ = pos_start + n;
        B->pos_ = pos_;
    }
    if (cancelled) {
        err = "cancelled";
        return false;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("glm prefill: ") + cudaGetErrorString(e);
            cudaSetDevice(dev_);
            return false;
        }
    }
    if (gb::launch_errors() > 0 || strata::kernels::glmf::launch_errors() > 0) {
        err = "glm prefill: " + std::to_string(gb::launch_errors() + strata::kernels::glmf::launch_errors()) +
              " kernel launch(es) failed (see stderr)";
        return false;
    }
    cudaSetDevice(dev_);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pf_->ms += ms;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pf_->tokens += n;
    static const bool verbose = getenv("STRATA_GLM_PREFILL_VERBOSE") != nullptr;
    if (verbose) {
        uint64_t sr = 0, sd = 0, rr = 0, rs = 0;
        double plan = 0;
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            sr += m->pf_->staged_ram;
            sd += m->pf_->staged_disk;
            rr += m->pf_->rows_resident;
            rs += m->pf_->rows_staged;
            plan += m->pf_->ms_plan;
        }
        std::fprintf(stderr, "glm prefill: %lld tokens in %.1f ms (%.2f ms/token, %.0f tok/s) | staged experts: %llu ram, "
                             "%llu disk | rows resident %.1f%% | plan %.1f ms (cumulative)\n",
                     (long long) n, ms, ms / (double) n, 1000.0 * (double) n / ms, (unsigned long long) sr,
                     (unsigned long long) sd, 100.0 * (double) rr / (double) std::max<uint64_t>(1, rr + rs), plan);
    }
    return true;
}

}  // namespace strata::core
