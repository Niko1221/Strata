// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

}  // namespace

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[STAGE] = {};
    uint8_t* stage_host[STAGE] = {};
    cudaEvent_t copied[STAGE] = {}, used[STAGE] = {};
    bool stage_live[STAGE] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_emb_host;
    std::vector<uint32_t> ple_rows;
    float* ple_norm = nullptr;
    PrefillStats* stats = nullptr;
};

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() {
    if (!impl_) return;
    if (impl_->cs) cudaStreamSynchronize(impl_->cs);
    for (int i = 0; i < STAGE; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
        if (impl_->stage_host[i]) cudaFreeHost(impl_->stage_host[i]);
    }
    if (impl_->copy) cudaStreamDestroy(impl_->copy);
    for (void* p : impl_->owned) cudaFree(p);
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert != NE || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        if (!m.gemm.init_external(stream, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
    }
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok); m.xn = o.take<float>(T * D, ok);
    m.xn16 = o.take<uint16_t>(T * D, ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    m.qkv = o.take<float>(T * C, ok); m.z = o.take<float>(T * ZV, ok); m.ab = o.take<float>(T * 2 * HV, ok);
    m.gate = o.take<float>(T * HV, ok); m.beta = o.take<float>(T * HV, ok); m.hbuf = o.take<float>(T * C, ok);
    m.y = o.take<float>(T * ZV, ok); m.y_h = o.take<uint16_t>(T * ZV, ok);
    m.Kc = o.take<float>(T * 512, ok); m.Vc = o.take<float>(T * 512, ok); m.Qf = o.take<float>(T * 12288, ok);
    m.q = o.take<float>(T * ZV, ok); m.idx_raw = o.take<float>(T * 128, ok); m.q_idx = o.take<float>(T * 512, ok);
    m.attn = o.take<float>(T * ZV, ok); m.attn_h = o.take<uint16_t>(T * ZV, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    m.steps_host.resize(T * strata::kernels::kStepCount);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.sel_ids = o.take<int32_t>(T * (size_t) m.cap, ok);
    m.max_blocks = ss.qsa_states[0].max_cells / s.idx_block + 2;
    m.sel_scores = o.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
    m.attn_scratch = o.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
    m.logits = o.take<float>(T * NE, ok); m.w = o.take<float>(T * K, ok); m.ids = o.take<int32_t>(T * K, ok);
    m.slot_dev = o.take<int32_t>(T * K, ok); m.src_dev = o.take<int32_t>(T * K, ok);
    m.Xs = o.take<uint16_t>(T * K * N, ok); m.GU = o.take<float>(T * K * 1280, ok);
    m.Hh = o.take<uint16_t>(T * K * 640, ok); m.Dm = o.take<float>(T * K * N, ok);
    m.sgate = o.take<float>(T * 640, ok); m.sup = o.take<float>(T * 640, ok); m.sh_h = o.take<uint16_t>(T * 640, ok);
    m.shared = o.take<float>(T * N, ok); m.sg = o.take<float>(T, ok);
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    for (int i = 0; i < STAGE; ++i) {
        m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
        if (cudaHostAlloc((void**) &m.stage_host[i], (size_t) MAXBLOB(), cudaHostAllocDefault) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(NE); m.off.resize(NE + 1);
    m.ple_emb_host.resize(T * N); m.ple_rows.resize(T * strata::kernels::PLE_N_HEADS);
    if (!ok) { err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit"; return false; }
    return true;
}

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    (void) g;
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    f(T * N); f(T * D); f(T * D); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    f(T * C); f(T * ZV); f(T * 2 * HV); f(T * HV); f(T * HV); f(T * C); f(T * ZV); o.take<uint16_t>(T * ZV, ok);
    f(T * 512); f(T * 512); f(T * 12288); f(T * ZV); f(T * 128); f(T * 512); f(T * ZV); o.take<uint16_t>(T * ZV, ok);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    o.take<int32_t>(T * (size_t) cap, ok);
    const int64_t max_blocks = ss.qsa_states[0].max_cells / s.idx_block + 2;
    f(256 * (size_t) max_blocks);
    f(32 * strata::kernels::qsa_decode_attn_scratch_floats(cap, s));
    f(T * NE); f(T * K); o.take<int32_t>(T * K, ok); o.take<int32_t>(T * K, ok); o.take<int32_t>(T * K, ok);
    o.take<uint16_t>(T * K * N, ok); f(T * K * 1280); o.take<uint16_t>(T * K * 640, ok); f(T * K * N);
    f(T * 640); f(T * 640); o.take<uint16_t>(T * 640, ok); f(T * N); f(T);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    for (int i = 0; i < STAGE; ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    return o.used + (8u << 20);   // alignment slack
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}  // namespace

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    Impl& m = *impl_;
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};

    for (int64_t c0 = 0; c0 < n; c0 += m.T) {
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        ++stats_.chunks;
        // ---- embeddings, broadcast to the four streams
        for (int64_t t = 0; t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                if (cudaMemcpyAsync(m.emb + t * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        gr_broadcast(m.emb, m.R, T, m.cs);
        // ---- the PLE rows of the whole chunk, one batched SSD request
        const bool ple_on = ss.ple.ready();
        if (ple_on) {
            const auto tp = Clock::now();
            for (int64_t t = 0; t < T; ++t) {
                const int32_t tok = (int32_t) tokens[c0 + t];
                strata::kernels::ngram_rows(&tok, prev, 1, ss.ple.consts, m.ple_rows.data() + t * strata::kernels::PLE_N_HEADS);
                prev[0] = prev[1];
                prev[1] = tok;
            }
            if (!ss.ple.table->gather_batch(m.ple_rows.data(), (size_t) T, m.ple_emb_host.data(), err)) return false;
            cudaMemcpyAsync(m.ple_emb, m.ple_emb_host.data(), (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs);
            stats_.ms_ple += ms_since(tp);
        } else {
            for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        }
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
        cudaMemcpyAsync(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount * 4,
                        cudaMemcpyHostToDevice, m.cs);

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const core::LayerView v(*m.wt, l);
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
            if (l == 1 && ple_on) {
                const auto tp = Clock::now();
                for (int64_t t = 0; t < T; ++t) {
                    strata::kernels::PleOut po;
                    po.normalized = m.ple_norm;
                    po.result = m.R + t * D;
                    try {
                        strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                   ss.ple.scratch, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                }
                stats_.ms_ple += ms_since(tp);
            }
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs);
                if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs);
                if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err)) return false;
                gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    float* state = ss.gdn_state + (size_t) gdn_index * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                    kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                              st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs);
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    for (int64_t t = 0; t < T; ++t) {
                        const int32_t* step_t = m.steps_dev + t * strata::kernels::kStepCount;
                        try {
                            strata::kernels::native_qsa_indexer_append(m.idx_raw + t * 128, step_t + strata::kernels::kStepPos, 0,
                                                                       (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                                       (float) strata::kernels::qsa_freq_base(), m.cs);
                        } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    }
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                          m.max_blocks, s, m.sel_scores, m.cs);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs);
                    }
                    strata::kernels::QsaAttnPools pools;
                    pools.page_table = st.page_table;
                    if (st.kv_int8) { pools.k_q = st.k_q; pools.v_q = st.v_q; pools.k_scale = st.k_scale; pools.v_scale = st.v_scale; }
                    else { pools.k_pool = st.k_pool; pools.v_pool = st.v_pool; }
                    for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                        const int64_t nb = std::min(m.attn_batch, T - t0);
                        strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                               m.steps_dev + t0 * strata::kernels::kStepCount, m.cap, s,
                                                               m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                    }
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                    ++qsa_index;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err)) return false;
                    route(m.logits, m.ids, m.w, T, m.cs);
                    // the shared expert and its scalar gate
                    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                    // group the (token, k) pairs by expert on the host
                    cudaMemcpyAsync(m.ids_host.data(), m.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, m.cs);
                    cudaStreamSynchronize(m.cs);
                    std::fill(m.cnt.begin(), m.cnt.end(), 0);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = m.ids_host[(size_t) i];
                        if (e < 0 || e >= NE) { err = "prefill: routed id out of range"; return false; }
                        ++m.cnt[(size_t) e];
                    }
                    m.off[0] = 0;
                    for (int64_t e = 0; e < NE; ++e) m.off[(size_t) e + 1] = m.off[(size_t) e] + m.cnt[(size_t) e];
                    std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = m.ids_host[(size_t) i];
                        const int32_t p = fill[(size_t) e]++;
                        m.slot_host[(size_t) i] = p;
                        m.src_host[(size_t) p] = (int32_t) (i / K);
                    }
                    cudaMemcpyAsync(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                    cudaMemcpyAsync(m.src_dev, m.src_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                    gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                    // the experts, in id order: resident ones from VRAM, the others through the staging ring
                    std::vector<int32_t> order;
                    for (int32_t e = 0; e < NE; ++e) if (m.cnt[(size_t) e] > 0) order.push_back(e);
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                    int stage_next = 0;
                    std::vector<int> stage_of(order.size(), -1);
                    auto stage_one = [&](size_t j) -> bool {
                        const int32_t e = order[j];
                        const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * NE + e] >= 0;
                        if (resident) return true;
                        const int sl = stage_next;
                        stage_next = (stage_next + 1) % STAGE;
                        const auto th = Clock::now();
                        const uint8_t* b = m.src->blob(l, e);
                        if (!b) { err = "prefill: expert source has no blob"; return false; }
                        if (m.src->pinned(l, e)) {
                            // DMA straight from the page-locked arena: the copy stream only waits for the slot
                            if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                            cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                            ++stats_.experts_dma;
                        } else {
                            if (m.stage_live[sl]) cudaEventSynchronize(m.used[sl]);   // its previous blob was dequantized
                            std::memcpy(m.stage_host[sl], b, (size_t) lay.blob_bytes(l));
                            cudaMemcpyAsync(m.stage_dev[sl], m.stage_host[sl], (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                        }
                        cudaEventRecord(m.copied[sl], m.copy);
                        m.stage_live[sl] = true;
                        stage_of[j] = sl;
                        stats_.ms_experts_host += ms_since(th);
                        ++stats_.experts_streamed;
                        return true;
                    };
                    size_t staged = 0;
                    const size_t lookahead = STAGE - 1;
                    for (size_t j = 0; j < order.size(); ++j) {
                        while (staged < order.size() && staged <= j + lookahead) {
                            if (!stage_one(staged)) return false;
                            ++staged;
                        }
                        const int32_t e = order[j];
                        const uint8_t* blob_dev = nullptr;
                        if (stage_of[j] < 0) {
                            blob_dev = m.cache->device_slot(m.host_res[(size_t) l * NE + e]);
                            ++stats_.experts_resident;
                        } else {
                            cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                            blob_dev = m.stage_dev[stage_of[j]];
                        }
                        const int q = (int) (j % DQ);
                        if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                            const auto& f = lay.fmt[(size_t) l];
                            strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                               m.dq_gu[q], m.cs);
                            strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs);
                        } else {
                            blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                        }
                        if (stage_of[j] >= 0) cudaEventRecord(m.used[stage_of[j]], m.cs);
                        const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                        m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                        swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                        m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                    }
                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                }
                // ---- the hyper-connection write of this half
                gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
            }
        }
        stats_.tokens += T;
        if (on_chunk) {
            if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (!on_chunk(m.R, T, p0, err)) return false;
        }
    }
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    stats_.ms_total += ms_since(t_start);
    return true;
}

}  // namespace strata::prefill
