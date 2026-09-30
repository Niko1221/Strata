// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
constexpr int64_t SUB = 2048;                 // tokens of a sub-chunk: everything but the routed experts
constexpr int64_t SEL_BATCH = 256;            // queries per block-score launch
// A layer's quantized projections, dequantized to FP16 once for all its sub-chunks: the PLE key, or the mixer's
// (GDN: qkv, gate, out; QSA: q, k, v, out) and the shared expert's
constexpr int64_t W16 = std::max(D * N, std::max(C + ZV + ZV, 12288 + 512 + 512 + ZV) * N + 3 * 640 * N);
constexpr size_t GEMM_WS = 32u << 20;         // cuBLAS workspace
constexpr int GGML_Q8_0 = 8;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Bump allocation from a region; without one it only counts (bytes_needed).
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool ok = true;
    template <typename T> T* take(size_t n) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        T* p = base != nullptr ? (T*) (base + used) : nullptr;
        used += bytes;
        if (base != nullptr && used > cap) ok = false;
        return p;
    }
};

// The device buffers of one binding.
struct Bufs {
    uint16_t* w16 = nullptr;
    void* gemm_ws = nullptr;
    // the chunk's: residual stream, the MoE's input (FP16) and output, the FFN half's injection, routing, steps, PLE
    // rows, the second GPU's sums (FP16)
    float *R = nullptr, *bo_moe = nullptr, *inj_f = nullptr, *w = nullptr, *ple_emb = nullptr;
    uint16_t *mixed_h = nullptr, *sums = nullptr;
    int32_t *ids = nullptr, *steps = nullptr;
    // a sub-chunk's: the mixer's BF16 input, its output, the mixer half's injection, a read's row scales
    uint16_t* mixed_bf = nullptr;
    float *bo = nullptr, *inj_a = nullptr, *rs = nullptr;
    // the union: what one step needs at a time
    float* emb = nullptr;
    float *xn = nullptr, *lo = nullptr, *gated = nullptr;
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float *ple_kp = nullptr, *ple_key = nullptr, *ple_q = nullptr, *ple_gated = nullptr;
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr,
          *attn = nullptr, *sel_scores = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* sel_ids = nullptr;
    float *logits = nullptr, *sgate = nullptr, *sup = nullptr, *sg = nullptr;
    uint16_t* sh_h = nullptr;
    uint8_t* local = nullptr;     // this GPU's expert runner
    uint64_t local_bytes = 0;
};

// Chunks of T tokens: the chunk-long buffers, the sub-chunk ones, then the union of the steps a layer takes one after
// the other on the stream.  (Staging copies run ahead on a copy stream only while the routed experts run, after the
// host has waited for the stream, and the stream waits for them before the next step.)
Bufs layout(Alloc& a, int64_t T, int64_t cap, int64_t max_blocks, bool offload) {
    const size_t t = (size_t) T, p = (size_t) std::min(T, SUB);
    Bufs b;
    b.w16 = a.take<uint16_t>((size_t) W16);
    b.gemm_ws = a.take<uint8_t>(GEMM_WS);
    b.R = a.take<float>(t * D);
    b.mixed_h = a.take<uint16_t>(t * N);
    b.bo_moe = a.take<float>(t * N);
    b.inj_f = a.take<float>(t * HC);
    b.ids = a.take<int32_t>(t * K);
    b.w = a.take<float>(t * K);
    b.steps = a.take<int32_t>(t * strata::kernels::kStepCount);
    b.ple_emb = a.take<float>(t * N);
    b.sums = offload ? a.take<uint16_t>(t * N) : nullptr;   // read by the next layer's sub-chunks
    b.mixed_bf = a.take<uint16_t>(p * N);
    b.bo = a.take<float>(p * N);
    b.inj_a = a.take<float>(p * HC);
    b.rs = a.take<float>(p * HC);
    const uint64_t u0 = a.used;
    uint64_t u1 = u0;
    auto next = [&] { u1 = std::max(u1, a.used); a.used = u0; };
    b.emb = a.take<float>(p * N);
    next();   // the hyper-connection read
    b.xn = a.take<float>(p * D); b.xn16 = a.take<uint16_t>(p * D); b.lo = a.take<float>(p * LR);
    b.lo16 = a.take<uint16_t>(p * LR); b.gated = a.take<float>(p * D);
    next();   // the PLE block
    b.ple_kp = a.take<float>(p * D); b.ple_key = a.take<float>(p * D);
    b.ple_q = a.take<float>(p * D); b.ple_gated = a.take<float>(p * D);
    next();   // GDN
    b.qkv = a.take<float>(p * C); b.z = a.take<float>(p * ZV); b.ab = a.take<float>(p * 2 * HV);
    b.gate = a.take<float>(p * HV); b.beta = a.take<float>(p * HV); b.hbuf = a.take<float>(p * C);
    b.y = a.take<float>(p * ZV); b.y_h = a.take<uint16_t>(p * ZV);
    next();   // QSA
    b.Kc = a.take<float>(p * 512); b.Vc = a.take<float>(p * 512); b.Qf = a.take<float>(p * 12288);
    b.q = a.take<float>(p * ZV); b.idx_raw = a.take<float>(p * 128); b.q_idx = a.take<float>(p * 512);
    b.attn = a.take<float>(p * ZV); b.attn_h = a.take<uint16_t>(p * ZV); b.sel_ids = a.take<int32_t>(p * (size_t) cap);
    b.sel_scores = a.take<float>((size_t) SEL_BATCH * (size_t) max_blocks);
    next();   // the router and the shared expert
    b.logits = a.take<float>(p * NE); b.sgate = a.take<float>(p * 640); b.sup = a.take<float>(p * 640);
    b.sh_h = a.take<uint16_t>(p * 640); b.sg = a.take<float>(p);
    next();   // the routed experts
    b.local_bytes = ExpertRunner::bytes_needed(T, NE, false, !offload);
    b.local = a.take<uint8_t>(b.local_bytes);
    next();
    a.used = u1;
    return b;
}

strata::kernels::QsaShapes shapes_of(const core::ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

}  // namespace

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    ExpertRunner* offload = nullptr;
    ExpertRunner local;
    int64_t max_chunk = 0, T = 0;   // T: the chunk the buffers are bound for
    strata::kernels::QsaShapes s;
    int64_t cap = 0, max_blocks = 0;
    cudaStream_t cs = nullptr;
    // the second GPU's share, a sub-chunk at a time: the input goes out on `xfer`, the sums come back on `xsum`
    cudaStream_t xfer = nullptr, xsum = nullptr;
    cudaEvent_t ev_mixed = nullptr;
    std::vector<cudaEvent_t> ev_in, ev_piece;   // a sub-chunk's input is in host memory, its sums on this GPU
    Gemm gemm;
    void* owned = nullptr;   // the buffers' own allocation (no region)
    Bufs b;
    std::vector<std::pair<const void*, const uint16_t*>> w16_have;   // this layer's dequantized weights
    int64_t w16_used = 0;
    int32_t* ids_host = nullptr;   // pinned [max_chunk, K]
    float* w_host = nullptr;
    std::vector<int32_t> steps_host, cnt, off, entry_of;
    std::vector<int32_t> ex1, off1, src1, ex2, off2, src2;   // this GPU's experts, the second GPU's
    std::vector<float> w1, w2;
    // a chunk's PLE rows, read on a thread while the chunk before runs: two pinned buffers, their row indices
    float* ple_host[2] = {};
    std::vector<uint32_t> ple_rows[2];
};

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() {
    if (!impl_) return;
    Impl& m = *impl_;
    if (m.cs) cudaStreamSynchronize(m.cs);
    if (m.xfer) cudaStreamSynchronize(m.xfer);
    if (m.xsum) cudaStreamSynchronize(m.xsum);
    if (m.ev_mixed) cudaEventDestroy(m.ev_mixed);
    for (cudaEvent_t e : m.ev_in) if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : m.ev_piece) if (e) cudaEventDestroy(e);
    if (m.xfer) cudaStreamDestroy(m.xfer);
    if (m.xsum) cudaStreamDestroy(m.xsum);
    for (float* h : m.ple_host) if (h) cudaFreeHost(h);
    if (m.ids_host) cudaFreeHost(m.ids_host);
    if (m.w_host) cudaFreeHost(m.w_host);
    if (m.owned) cudaFree(m.owned);
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t max_chunk,
                   void* stream, ExpertRunner* offload, std::string& err) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.cache = cache; m.host_res = host_res; m.offload = offload;
    m.max_chunk = max_chunk; m.cs = (cudaStream_t) stream;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert != NE || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    m.s = shapes_of(g);
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, m.s);
    m.max_blocks = ss.qsa_states[0].max_cells / m.s.idx_block + 2;
    m.ev_in.assign((size_t) ((max_chunk + SUB - 1) / SUB), nullptr);
    m.ev_piece.assign(m.ev_in.size(), nullptr);
    bool pieces = true;
    for (auto* v : {&m.ev_in, &m.ev_piece})
        for (cudaEvent_t& e : *v)
            pieces = pieces && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    if (!pieces || cudaStreamCreateWithFlags(&m.xfer, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&m.xsum, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&m.ev_mixed, cudaEventDisableTiming) != cudaSuccess ||
        cudaHostAlloc((void**) &m.ids_host, (size_t) max_chunk * K * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &m.w_host, (size_t) max_chunk * K * 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "prefill: streams, events and pinned routing buffers";
        return false;
    }
    const size_t T = (size_t) max_chunk;
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.cnt.resize(NE);
    m.off.resize(NE + 1);
    m.entry_of.resize(T * K);
    for (int i = 0; i < 2; ++i) {
        m.ple_rows[i].resize(T * strata::kernels::PLE_N_HEADS);
        if (cudaHostAlloc((void**) &m.ple_host[i], T * N * 4, cudaHostAllocDefault) != cudaSuccess) {
            err = "prefill: pinned PLE row buffers";
            return false;
        }
    }
    int dev = 0;
    cudaGetDevice(&dev);
    return m.gemm.init(stream, err) &&
           m.local.init(dev, dev, stream, src, cache, host_res, NE, max_chunk, offload == nullptr, err);
}

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk, bool offload) {
    const strata::kernels::QsaShapes s = shapes_of(g);
    Alloc a;
    layout(a, chunk, strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s),
           ss.qsa_states[0].max_cells / s.idx_block + 2, offload);
    return a.used;
}

uint64_t Prefill::bytes_for(int64_t chunk) const {
    const Impl& m = *impl_;
    return bytes_needed(*m.g, *m.ss, chunk, m.offload != nullptr) +
           (m.offload == nullptr && chunk >= ExpertRunner::kPrefetchMin ? m.local.prefetch_need(nullptr) : 0);
}

bool Prefill::bind(void* region, uint64_t bytes, int64_t chunk, std::string& err) {
    Impl& m = *impl_;
    if (region == nullptr) {
        if (m.owned != nullptr) return true;   // allocated once, for the longest chunk
        chunk = m.max_chunk;
        bytes = bytes_needed(*m.g, *m.ss, chunk, m.offload != nullptr);
        if (cudaMalloc(&m.owned, bytes) != cudaSuccess) {
            m.owned = nullptr;
            err = "prefill: " + std::to_string(bytes >> 20) + " MiB of device buffers";
            return false;
        }
        region = m.owned;
    }
    if (chunk <= 0 || chunk > m.max_chunk) {
        err = "prefill: a chunk of " + std::to_string(chunk) + " tokens (set up for " + std::to_string(m.max_chunk) + ")";
        return false;
    }
    Alloc a;
    a.base = (uint8_t*) region;
    a.cap = bytes;
    m.b = layout(a, chunk, m.cap, m.max_blocks, m.offload != nullptr);
    if (!a.ok) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    m.gemm.set_buffers(nullptr, 0, m.b.gemm_ws, GEMM_WS);
    // the rest of a lent region takes this GPU's prefetched experts (outside the union: copies run beside the steps)
    const bool area = region != m.owned && bytes > a.used;
    if (!m.local.bind(m.b.local, m.b.local_bytes, chunk, err, area ? (uint8_t*) region + a.used : nullptr,
                      area ? bytes - a.used : 0))
        return false;
    m.T = chunk;
    return true;
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
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
    const Bufs& b = m.b;
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const strata::kernels::QsaShapes& s = m.s;
    constexpr int64_t SC = strata::kernels::kStepCount;
    if (m.T <= 0) { err = "prefill: no buffers bound"; return false; }
    const auto t_start = Clock::now();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    // Y[T, rows] = X . W^T for a quantized W [rows, cols]: dequantized at its first use in the layer
    auto w16 = [&](int type, const void* blocks, int64_t rows, int64_t cols) -> const uint16_t* {
        for (const auto& [k, p] : m.w16_have)
            if (k == blocks) return p;
        if (m.w16_used + rows * cols > W16) return nullptr;
        uint16_t* p = b.w16 + m.w16_used;
        strata::kernels::dequant_f16(type, blocks, 0, rows, cols, p, m.cs);
        m.w16_used += (rows * cols + 127) / 128 * 128;
        m.w16_have.emplace_back(blocks, p);
        return p;
    };
    auto native_proj = [&](const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                           std::string& e) {
        const uint16_t* W = w->native_data ? w16(w->native_type, w->native_data, w->ne1, w->ne0) : nullptr;
        if (W == nullptr) {
            e = "prefill: " + name + (w->native_data ? " does not fit the dequantization buffer"
                                                     : " has no native GGUF blocks (run with --native)");
            return false;
        }
        m.gemm.f16(X, W, Y, T, w->ne1, w->ne0);
        return true;
    };
    auto new_layer = [&] {
        m.w16_have.clear();
        m.w16_used = 0;
    };
    // --prefill-profile: an event at the start of each section; the time to the next event is charged to it
    std::vector<std::pair<int, cudaEvent_t>> ev;
    auto mark = [&](int section) {
        if (!profile) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, m.cs);
        ev.push_back({section, e});
    };

    // The MTP draft layer's K/V over a chunk's cells: its input norms and projections (e = fc_embedding(rms(the
    // embedding of the token at t + 1)), h = fc_hidden(rms(R_t)) per stream, h + e on every stream), the attention
    // hyper-connection read, then K (normed and rotated) and V into its cells, from the window's first cell on.  The
    // chunk's buffers are free once its last layer is done; its residual rows are normed in place.
    const core::MtpPromptKv dk = draft != nullptr ? draft->prompt_kv() : core::MtpPromptKv{};
    if (draft != nullptr && (!dk.norm_emb || !dk.norm_hidden || !dk.fc_emb || !dk.fc_hidden || !dk.hc_norm ||
                             !dk.hc_down || !dk.hc_up || !dk.k_proj || !dk.v_proj || !dk.k_norm)) {
        err = "prefill: a draft layer weight is missing";
        return false;
    }
    auto draft_pass = [&](const int64_t* toks, int64_t T, int64_t p0) -> bool {
        const int64_t from = std::clamp<int64_t>(dk.first_cell - p0, 0, T);
        if (from == T) return true;
        mark(kPsDraft);
        new_layer();
        const uint16_t *We = w16(GGML_Q8_0, dk.fc_emb, N, N), *Wh = w16(GGML_Q8_0, dk.fc_hidden, N, N),
                       *Wk = w16(GGML_Q8_0, dk.k_proj, 512, N), *Wv = w16(GGML_Q8_0, dk.v_proj, 512, N);
        if (!We || !Wh || !Wk || !Wv) {
            err = "prefill: the draft layer's projections do not fit the dequantization buffer";
            return false;
        }
        for (int64_t t = from; t < T; ++t) m.ids_host[t - from] = (int32_t) toks[t + 1];
        cudaMemcpyAsync(b.ids, m.ids_host, (size_t) (T - from) * 4, cudaMemcpyHostToDevice, m.cs);
        const core::QsaState& st = *dk.kv;
        for (int64_t t0 = from; t0 < T; t0 += SUB) {
            const int64_t P = std::min(SUB, T - t0), pp = p0 + t0;
            float* R = b.R + t0 * D;
            if (!core::embed_rows(*m.wt, g, b.ids + (t0 - from), P, b.emb, m.cs, err)) return false;
            rms_rows(b.emb, dk.norm_emb, P, N, N, EPS, m.cs);
            to_f16(b.emb, b.mixed_h, P * N, m.cs);
            m.gemm.f16(b.mixed_h, We, b.bo, P, N, N);
            rms_rows(R, dk.norm_hidden, P, D, D, EPS, m.cs);
            to_f16(R, b.xn16, P * D, m.cs);
            m.gemm.f16(b.xn16, Wh, b.gated, P * HC, N, N);
            strata::kernels::add_streams_broadcast(b.gated, b.bo, b.gated, N, (int) HC, (int) P, m.cs);
            gr_norm(b.gated, dk.hc_norm, EPS, b.xn, b.xn16, P, m.cs);
            m.gemm.bf16(b.xn16, dk.hc_down, b.lo, P, LR, D);
            gr_silu(b.lo, b.lo16, P, m.cs);
            m.gemm.bf16(b.lo16, dk.hc_up, b.gated, P, D, LR);
            gr_mix(b.xn, b.gated, nullptr, nullptr, P, m.cs, b.mixed_h);
            m.gemm.f16(b.mixed_h, Wk, b.Kc, P, 512, N);
            m.gemm.f16(b.mixed_h, Wv, b.Vc, P, 512, N);
            rms_rows(b.Kc, dk.k_norm, P * 2, 256, 256, EPS, m.cs);
            rope(b.Kc, P, 2, 256, 512, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
            kv_append(b.Kc, b.Vc, P, pp, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs);
        }
        return true;
    };

    // ---- a chunk's PLE rows (one batched SSD request) on a thread: the first chunk's while its layer 0 runs, each
    // next chunk's while the one before runs; waited for before the PLE block at layer 1
    struct PleRead {
        std::thread th;
        bool ok = true;
        std::string err;
        bool wait(std::string& e) {
            if (th.joinable()) th.join();
            if (!ok) e = err;
            return ok;
        }
        ~PleRead() { if (th.joinable()) th.join(); }
    } ple_read;
    int32_t pa[2] = {prev[0], prev[1]};   // the rows' tokens before the next chunk to read
    int ple_cur = 0;
    auto ple_start = [&](int64_t c, int buf) {
        const int64_t Tc = std::min(m.T, n - c);
        uint32_t* rows = m.ple_rows[buf].data();
        for (int64_t t = 0; t < Tc; ++t) {
            const int32_t tok = (int32_t) tokens[c + t];
            strata::kernels::ngram_rows(&tok, pa, 1, ss.ple.consts, rows + t * strata::kernels::PLE_N_HEADS);
            pa[0] = pa[1];
            pa[1] = tok;
        }
        ple_read.ok = true;
        ple_read.th = std::thread([&m, &ss, &ple_read, rows, Tc, out = m.ple_host[buf]] {
            ple_read.ok = ss.ple.table->gather_batch(rows, (size_t) Tc, out, ple_read.err);
        });
    };
    if (ss.ple.ready() && n > 0) ple_start(0, 0);

    static const bool trace = std::getenv("STRATA_TRACE") != nullptr;
    processed_ = 0;
    int64_t c0 = 0;
    for (; c0 < n; c0 += m.T) {
        if (should_stop && should_stop()) break;   // the chunks before stay: `prev` and the state hold them
        if (trace) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        ++stats_.chunks;
        // ---- the embeddings, broadcast to the four streams
        for (int64_t t0 = 0; t0 < T; t0 += SUB) {
            const int64_t P = std::min(SUB, T - t0);
            for (int64_t t = t0; t < t0 + P; ++t) {
                const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
                if (row) {
                    if (cudaMemcpyAsync(b.emb + (t - t0) * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                        err = "prefill: the image embedding upload failed";
                        return false;
                    }
                } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], b.emb + (t - t0) * N, m.cs, err)) {
                    return false;
                }
            }
            gr_broadcast(b.emb, b.R + t0 * D, P, m.cs);
        }
        const bool ple_on = ss.ple.ready();
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * SC, p0 + t, s);
        cudaMemcpyAsync(b.steps, m.steps_host.data(), (size_t) T * SC * 4, cudaMemcpyHostToDevice, m.cs);

        // ---- a layer's FFN write, deferred to the sub-chunk that reads its rows next: the second GPU's sums come back
        // a sub-chunk at a time while this GPU works on the sub-chunks before (with `norm`, the next read's norm in
        // the same pass)
        bool pending = false;
        auto ffn_write = [&](int64_t t0, int64_t P, const float* norm) {
            const uint16_t* part = nullptr;
            if (m.offload) {
                cudaStreamWaitEvent(m.cs, m.ev_piece[(size_t) (t0 / SUB)], 0);
                part = b.sums + t0 * N;
            }
            float *R = b.R + t0 * D, *bo = b.bo_moe + t0 * N, *inj = b.inj_f + t0 * HC;
            if (norm) gr_write_norm_rs(R, bo, inj, HC, norm, EPS, b.rs, b.xn16, P, m.cs, part);
            else gr_write(R, bo, inj, HC, P, m.cs, part);
        };
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const core::LayerView v(*m.wt, l);
            const bool qsa = core::is_qsa_layer(g, l);
            new_layer();
            // a long chunk uses nearly every expert: the ones this GPU's cache lacks are copied ahead, here while
            // the dense steps run, or on the second GPU
            if (T >= ExpertRunner::kPrefetchMin) {
                m.ex1.clear();
                for (int32_t e = 0; e < NE; ++e)
                    if (!(m.host_res && m.cache && m.host_res[(size_t) l * NE + e] >= 0)) m.ex1.push_back(e);
                if (!(m.offload ? m.offload : &m.local)->prefetch(l, m.ex1, err)) return false;
            }
            mark(kPsPle);
            // ---- the PLE block at layer 1: the key and value projections, then the decode path's per-token
            // arithmetic (the conv reads the previous tokens' normalized rows)
            if (l == 1 && ple_on) {
                // the chunk's rows, then the next chunk's read on the thread (the other buffer)
                const auto tp = Clock::now();
                if (!ple_read.wait(err)) return false;
                cudaMemcpyAsync(b.ple_emb, m.ple_host[ple_cur], (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs);
                if (c0 + T < n) ple_start(c0 + T, 1 - ple_cur);
                ple_cur = 1 - ple_cur;
                stats_.ms_ple += ms_since(tp);
                const strata::kernels::PleWeights& pw = ss.ple.w;
                for (int64_t t0 = 0; t0 < T; t0 += SUB) {
                    const int64_t P = std::min(SUB, T - t0);
                    if (pending) ffn_write(t0, P, nullptr);
                    const float* ple_emb = b.ple_emb + t0 * N;
                    to_bf16(ple_emb, b.mixed_bf, P * N, m.cs);
                    if (pw.key_bf16 != nullptr) {
                        m.gemm.bf16(b.mixed_bf, pw.key_bf16, b.ple_kp, P, D, N);
                    } else if (const uint16_t* W = pw.key_native_data ? w16(pw.key_native_type, pw.key_native_data, D, N)
                                                                      : nullptr) {
                        to_f16(ple_emb, b.mixed_h, P * N, m.cs);
                        m.gemm.f16(b.mixed_h, W, b.ple_kp, P, D, N);
                    } else {
                        err = "prefill: the PLE key has neither a BF16 nor a native GGUF form (run with --native)";
                        return false;
                    }
                    m.gemm.bf16(b.mixed_bf, pw.value_bf16, b.bo, P, N, N);
                    try {
                        strata::kernels::native_ple_postops_tokens(
                            b.ple_kp, b.R + t0 * D, b.bo, ss.ple.hist, pw,
                            {b.ple_key, b.ple_q, b.inj_a, b.ple_gated, b.ple_q, b.R + t0 * D}, (int) P, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                pending = false;
                new_layer();
            }
            const core::WeightRef *wr = nullptr, *wgi = nullptr, *wsg = nullptr, *wsu = nullptr, *wsd = nullptr;
            if (!(wr = need(v, "ffn_gate_inp.weight", err)) || !(wgi = need(v, "ffn_gate_inp_shexp.weight", err)) ||
                !(wsg = need(v, "ffn_gate_shexp.weight", err)) || !(wsu = need(v, "ffn_up_shexp.weight", err)) ||
                !(wsd = need(v, "ffn_down_shexp.weight", err)))
                return false;
            if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
            // ================ per sub-chunk: the mixer half, then the FFN half up to the routing ================
            for (int64_t t0 = 0; t0 < T; t0 += SUB) {
                const int64_t P = std::min(SUB, T - t0), pp = p0 + t0;
                uint16_t* mixed_h = b.mixed_h + t0 * N;
                for (int half = 0; half < 2; ++half) {
                    mark(half == 0 ? kPsHcRead : kPsHcFfn);
                    // ---- the hyper-connection read of this half (the FFN half's norm came with the mixer's write)
                    const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                    const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                      su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                    const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                          *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                    if (!wn || !wd || !wu || !wi) return false;
                    float* inj = half == 0 ? b.inj_a : b.inj_f + t0 * HC;
                    if (half == 0 && pending) {
                        mark(kPsCombine);
                        ffn_write(t0, P, (const float*) wn->data);
                        mark(kPsHcRead);
                    } else if (half == 0) {
                        gr_norm_rs(b.R + t0 * D, (const float*) wn->data, EPS, b.rs, b.xn16, P, m.cs);
                    }
                    if (!bf16_proj(m.gemm, wd, b.xn16, b.lo, P, sd, err)) return false;
                    gr_silu(b.lo, b.lo16, P, m.cs);
                    if (!bf16_proj(m.gemm, wu, b.lo16, b.gated, P, su, err)) return false;
                    if (!bf16_proj(m.gemm, wi, b.xn16, inj, P, si, err)) return false;
                    gr_mix_r(b.R + t0 * D, b.rs, (const float*) wn->data, b.gated, nullptr, b.mixed_bf, P, m.cs,
                             mixed_h);
                    if (half == 1) break;

                    if (!qsa) {
                        // ======================= GDN =======================
                        const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                              *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                              *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                              *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                              *wsa = need(v, "ssm_a", err);
                        if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                        mark(kPsGdn);
                        float* state = ss.gdn_state + (size_t) gdn_index * gdn_floats;
                        float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        if (!native_proj(wqkv, mixed_h, b.qkv, P, v.name("attn_qkv.weight"), err)) return false;
                        if (!native_proj(wg, mixed_h, b.z, P, v.name("attn_gate.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wa, b.mixed_bf, b.ab, P, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
                        if (!bf16_proj(m.gemm, wb, b.mixed_bf, b.ab + HV, P, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
                        gdn_gates(b.ab, (const float*) wdt->data, (const float*) wsa->data, b.gate, b.beta, P, m.cs);
                        mark(kPsGdnConv);
                        gdn_conv(conv, b.qkv, (const float*) wc->data, b.hbuf, P, EPS, m.cs);
                        mark(kPsGdnScan);
                        gdn_scan(state, b.hbuf, b.gate, b.beta, b.y, P, m.cs);
                        mark(kPsGdnNorm);
                        gdn_out_norm(b.y, b.z, (const float*) wnm->data, EPS, b.y_h, P, m.cs);
                        mark(kPsGdnOut);
                        if (!native_proj(wo, b.y_h, b.bo, P, v.name("ssm_out.weight"), err)) return false;
                    } else {
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
                        const int32_t* steps = b.steps + t0 * SC;
                        mark(kPsQsaProj);
                        if (!native_proj(wk, mixed_h, b.Kc, P, v.name("attn_k.weight"), err)) return false;
                        if (!native_proj(wv, mixed_h, b.Vc, P, v.name("attn_v.weight"), err)) return false;
                        if (!native_proj(wq, mixed_h, b.Qf, P, v.name("attn_q.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wik, b.mixed_bf, b.idx_raw, P, v.name("indexer.k_proj.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wiq, b.mixed_bf, b.q_idx, P, v.name("indexer.q_proj.weight"), err)) return false;
                        rms_rows(b.Kc, (const float*) wkn->data, P * 2, 256, 256, EPS, m.cs);
                        rope(b.Kc, P, 2, 256, 512, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
                        kv_append(b.Kc, b.Vc, P, pp, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                  st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs);
                        split_q(b.Qf, b.q, P, m.cs);
                        rms_rows(b.q, (const float*) wqn->data, P * 24, 256, 256, EPS, m.cs);
                        rope(b.q, P, 24, 256, 6144, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
                        rms_rows(b.q_idx, (const float*) wiqn->data, P * 4, 128, 128, EPS, m.cs);
                        rope(b.q_idx, P, 4, 128, 512, pp, (float) strata::kernels::qsa_freq_base(), m.cs);
                        // the indexer appends of the sub-chunk; then scores + selection for many queries at once:
                        // a query reads completed blocks (final once completed) and `dead` for its own tail block
                        const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        mark(kPsQsaIndexer);
                        try {
                            strata::kernels::native_qsa_indexer_append_multi(
                                b.idx_raw, steps + strata::kernels::kStepPos, (int) SC, (int) P, 0,
                                (const float*) wikn->data, EPS, ib, s, st.max_cells, (float) strata::kernels::qsa_freq_base(), m.cs);
                        } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                        mark(kPsQsaScores);
                        for (int64_t q0 = 0; q0 < P; q0 += SEL_BATCH) {
                            const int64_t nb = std::min(SEL_BATCH, P - q0);
                            // the batch's last query has completed the most blocks
                            const int64_t grid_blocks =
                                m.steps_host[(size_t) (t0 + q0 + nb - 1) * SC + strata::kernels::kStepNBid] + 1;
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, b.q_idx + q0 * 512, steps + q0 * SC,
                                                              nb, m.max_blocks, s, b.sel_scores, m.cs, grid_blocks);
                            strata::kernels::qsa_block_topk(b.sel_scores, steps + q0 * SC, nb, m.max_blocks, m.cap, s,
                                                            b.sel_ids + q0 * m.cap, m.cs);
                        }
                        strata::kernels::QsaAttnPools pools;
                        pools.page_table = st.page_table;
                        if (st.kv_int8) { pools.k_q = st.k_q; pools.v_q = st.v_q; pools.k_scale = st.k_scale; pools.v_scale = st.v_scale; }
                        else { pools.k_pool = st.k_pool; pools.v_pool = st.v_pool; }
                        mark(kPsQsaAttn);
                        strata::kernels::qsa_prefill_attn(b.q, pools, b.sel_ids, steps, m.cap, s, b.attn, P, m.cs);
                        mark(kPsQsaOut);
                        gate_attn(b.attn, b.Qf, b.attn_h, P, m.cs);
                        if (!native_proj(wo, b.attn_h, b.bo, P, v.name("attn_output.weight"), err)) return false;
                    }
                    // ---- the hyper-connection write of the mixer half, with the FFN half's norm
                    mark(kPsHcFfn);
                    const core::WeightRef* wfn = need(v, "hc_ffn_norm.weight", err);
                    if (!wfn) return false;
                    gr_write_norm_rs(b.R + t0 * D, b.bo, b.inj_a, HC, (const float*) wfn->data, EPS, b.rs, b.xn16, P,
                                     m.cs);
                }
                // ---- the FFN half's input is ready: its trip to the second GPU starts here
                if (m.offload) {
                    cudaEventRecord(m.ev_mixed, m.cs);
                    cudaStreamWaitEvent(m.xfer, m.ev_mixed, 0);
                    cudaMemcpyAsync(m.offload->host_input() + t0 * N, mixed_h, (size_t) P * N * 2,
                                    cudaMemcpyDeviceToHost, m.xfer);
                    cudaEventRecord(m.ev_in[(size_t) (t0 / SUB)], m.xfer);
                    m.offload->stage_input(t0, P, m.ev_in[(size_t) (t0 / SUB)]);
                }
                // ---- the router, and the shared expert (its gated output starts the MoE sum)
                mark(kPsRouter);
                if (!bf16_proj(m.gemm, wr, b.mixed_bf, b.logits, P, v.name("ffn_gate_inp.weight"), err)) return false;
                route(b.logits, b.ids + t0 * K, b.w + t0 * K, P, m.cs);
                if (!native_proj(wsg, mixed_h, b.sgate, P, v.name("ffn_gate_shexp.weight"), err)) return false;
                if (!native_proj(wsu, mixed_h, b.sup, P, v.name("ffn_up_shexp.weight"), err)) return false;
                swiglu_pair(b.sgate, b.sup, b.sh_h, P, m.cs);
                if (!native_proj(wsd, b.sh_h, b.bo_moe + t0 * N, P, v.name("ffn_down_shexp.weight"), err)) return false;
                m.gemm.bf16(b.mixed_bf, (const uint16_t*) wgi->data, b.sg, P, 1, N);
                moe_shared_gate(b.bo_moe + t0 * N, b.sg, P, m.cs);
                cudaMemcpyAsync(m.ids_host + t0 * K, b.ids + t0 * K, (size_t) P * K * 4, cudaMemcpyDeviceToHost, m.cs);
                cudaMemcpyAsync(m.w_host + t0 * K, b.w + t0 * K, (size_t) P * K * 4, cudaMemcpyDeviceToHost, m.cs);
            }
            pending = false;
            if (!qsa) ++gdn_index;
            else ++qsa_index;
            // ================ the routed experts of the whole chunk ================
            if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            mark(kPsExperts);
            // group the (token, k) entries by expert: this GPU takes the experts its cache holds (all of them without
            // a second GPU), the second GPU the others
            std::fill(m.cnt.begin(), m.cnt.end(), 0);
            for (int64_t i = 0; i < T * K; ++i) {
                const int32_t e = m.ids_host[i];
                if (e < 0 || e >= NE) { err = "prefill: routed id out of range"; return false; }
                ++m.cnt[(size_t) e];
            }
            m.off[0] = 0;
            for (int64_t e = 0; e < NE; ++e) m.off[(size_t) e + 1] = m.off[(size_t) e] + m.cnt[(size_t) e];
            {
                std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                for (int64_t i = 0; i < T * K; ++i) m.entry_of[(size_t) fill[(size_t) m.ids_host[i]]++] = (int32_t) i;
            }
            for (auto* v2 : {&m.ex1, &m.src1, &m.ex2, &m.src2}) v2->clear();
            m.w1.clear();
            m.w2.clear();
            m.off1.assign(1, 0);
            m.off2.assign(1, 0);
            for (int32_t e = 0; e < NE; ++e) {
                if (m.cnt[(size_t) e] == 0) continue;
                const bool here = !m.offload || (m.host_res && m.cache && m.host_res[(size_t) l * NE + e] >= 0);
                std::vector<int32_t>& ex = here ? m.ex1 : m.ex2;
                std::vector<int32_t>& of = here ? m.off1 : m.off2;
                std::vector<int32_t>& sr = here ? m.src1 : m.src2;
                std::vector<float>& wv = here ? m.w1 : m.w2;
                ex.push_back(e);
                for (int32_t p = m.off[(size_t) e]; p < m.off[(size_t) e + 1]; ++p) {
                    const int32_t i = m.entry_of[(size_t) p];
                    sr.push_back(i / (int32_t) K);
                    wv.push_back(m.w_host[i]);
                }
                of.push_back((int32_t) sr.size());
            }
            if (m.offload && !m.offload->run_layer(l, T, m.ex2, m.off2, m.src2, m.w2, SUB, nullptr, nullptr, err))
                return false;
            if (!m.local.run_layer(l, T, m.ex1, m.off1, m.src1, m.w1, SUB, b.mixed_h, b.bo_moe, err)) return false;
            // ---- the MoE output (+ the second GPU's sums) goes into the residual stream as the next layer reads it
            if (m.offload) {
                for (int64_t t0 = 0; t0 < T; t0 += SUB) {
                    cudaStreamWaitEvent(m.xsum, m.offload->piece_done(t0 / SUB), 0);
                    const size_t bytes = (size_t) std::min(SUB, T - t0) * N * 2;
                    cudaMemcpyAsync(b.sums + t0 * N, m.offload->host_sum() + t0 * N, bytes, cudaMemcpyHostToDevice,
                                    m.xsum);
                    cudaEventRecord(m.ev_piece[(size_t) (t0 / SUB)], m.xsum);
                }
            }
            pending = true;
        }
        mark(kPsCombine);   // the last layer's write
        for (int64_t t0 = 0; t0 < T; t0 += SUB) ffn_write(t0, std::min(SUB, T - t0), nullptr);
        if (draft != nullptr && !draft_pass(tokens + c0, T, p0)) return false;
        if (profile) {
            mark(kPsPle);
            cudaEventSynchronize(ev.back().second);
            for (size_t i = 0; i + 1 < ev.size(); ++i) {
                float ms = 0;
                cudaEventElapsedTime(&ms, ev[i].second, ev[i + 1].second);
                stats_.ms_section[ev[i].first] += ms;
            }
            for (auto& e : ev) cudaEventDestroy(e.second);
            ev.clear();
        }
        stats_.tokens += T;
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
    }
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    stats_.experts_streamed = m.local.experts_streamed;
    stats_.experts_prefetched = m.local.experts_prefetched;
    stats_.experts_resident = m.local.experts_resident;
    stats_.ms_experts_host = m.local.ms_host;
    stats_.ms_total += ms_since(t_start);
    processed_ = std::min(c0, n);
    if (c0 < n) {
        err = "cancelled";
        return false;
    }
    return true;
}

const char* prefill_section_name(int section) {
    static const char* const names[kPsCount] = {
        "PLE block", "HC read (mixer)", "GDN projections", "GDN conv", "GDN recurrence", "GDN output norm",
        "GDN output projection",
        "QSA projections, norms, KV", "QSA indexer appends",
        "QSA block scores + top-k", "QSA attention", "QSA gate + output", "HC write + HC read (FFN)",
        "router, shared expert", "grouping, experts", "combine + HC write", "draft layer K/V"};
    return section >= 0 && section < kPsCount ? names[section] : "?";
}

}  // namespace strata::prefill
