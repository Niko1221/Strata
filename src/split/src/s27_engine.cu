// The Engine (engine.hpp) for Qwen3.8-27B (make FNX_MODEL=27b): dense, split across the cards as Flash-Next is. Every
// layer on every card: the recurrent heads, the FFN by FF width (ffd_lo), the attention by heads (s27_attn.cuh: each
// card's K/V for its own KV heads (s27_attn.cuh counts them for the build's card count), no exchange), the
// vocabulary; the residual and the norms copied, two sums a layer (s27_ring3.cuh). The draft layer is the GGUF's
// nextn block (CardW::mtp), its head the Mtp's rows.
// Its layers, embeddings, pages and prompt path are here and in s27_prefill.cu; lanes, windows, commits, states, the
// draft layer's window, the sampler's candidates and the window graphs are engine_common.cu's (engine_parts.hpp).
#include "engine.hpp"
#include "engine_parts.hpp"
#include "env.hpp"
#include "peer_probe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fnx {

namespace {
// R[t] += pend[t] (a half's sum, when pend is not null), then y[t] = rms(R[t]) * w: the next half's input; with tq, also
// quantized as quantize_q8_1 does into the split twin (tq, td) of a q8_1 buffer of N-wide rows, which is all q8s reads
__global__ void k_add_norm(float* R, const float* pend, const float* w, float* y, int8_t* tq, float* td, float eps) {
    __shared__ float sh[32];
    float* r = R + (size_t) blockIdx.x * N;
    float q = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = r[i];
        if (pend) r[i] = x += pend[(size_t) blockIdx.x * N + i];
        q += x * x;
    }
    const float s = rsqrtf(s27_block_sum(q, sh) / (float) N + eps);
    const int lane = threadIdx.x & 31;
    for (int bl = threadIdx.x >> 5; bl < N / 32; bl += blockDim.x >> 5) {
        const int i = bl * 32 + lane;
        const float v = r[i] * s * w[i];
        y[(size_t) blockIdx.x * N + i] = v;
        if (!tq) continue;
        float amax = fabsf(v);
        for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        tq[(size_t) blockIdx.x * N + i] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        if (lane == 0) td[(size_t) blockIdx.x * (N / 32) + bl] = __half2float(__float2half(d));
    }
}
void add_norm(CardBuf& b, float* R, const float* pend, const float* w, int T, float eps, bool twin) {
    k_add_norm<<<T, 1024, 0, b.st>>>(R, pend, w, b.mixed, twin ? b.xtq : nullptr, twin ? b.xtd : nullptr, eps);
    ck(cudaGetLastError(), "add_norm");
}
// out[t] = [rms(e[t]) * we | rms(h[t]) * wh]: the draft layer's input before its projection
__global__ void k_norm_cat(const float* e, const float* h, const float* we, const float* wh, float* out, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x[2] = {e + (size_t) t * N, h + (size_t) t * N};
    const float* w[2] = {we, wh};
    for (int s = 0; s < 2; ++s) {
        float q = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) q += x[s][i] * x[s][i];
        const float r = rsqrtf(s27_block_sum(q, sh) / (float) N + eps);
        for (int i = threadIdx.x; i < N; i += blockDim.x) out[(size_t) t * 2 * N + s * N + i] = x[s][i] * r * w[s][i];
    }
}

// x summed across the cards (every card ends with the sum), three fp16 values a word (s27_ring3.cuh). On the box GPU 0
// sits on the other socket from 1 and 2: storing straight to both peers (2 hops, not the ring's 4) measured slower at
// every window size, 28.9 against 19.4 ms at 4 rows, from the traffic it sends across the sockets; the ring with two
// values a word (kernels.cu's) took 19.4 ms against 18.5 at 4 rows, 25.9 against 23.8 at 8
void sum_cards(CardBuf& b, float* x, int T, const int* ctl, int call) {
    ring_call_check(call, "sum_cards");
    if (b.ring32) {
        ring_sum(b.ring, x, T, (const unsigned*) (ctl + 1), call, b.st);
        return;
    }
    k_ring3<<<R3_BLOCKS, 256, 0, b.st>>>(b.r3, x, T, (const unsigned*) (ctl + 1), call);
    ck(cudaGetLastError(), "ring3");
}

// out[t][r] = w[r] . y[t] for a Q8_0 matrix (the 27B's are all Q8_0: the loader checks). q8s beat three kernels written
// for these shapes (activations in registers with split columns; a lane on 64 columns; activations in shared memory,
// whole rows a warp): at 4 rows 13.8 ms of matrices a window against 14.9, 21.9 and ~15.2. The rest of the gap to the
// same bytes in a few big matrices (11.5 ms at 4 rows) is at the boundaries between the window's 257 dependent
// matrices, and stayed there as one persistent launch with a grid barrier between matrices (13.8 ms), with each block
// prefetching its next rows into L2 before the barrier (15.1), with an L2 prefetch on a second stream (14.8 to 15.7)
// and with capped persistent grids (13.2 to 16.8)
void mv(CardBuf& b, const DW& w, const void* y, float* out, int T) { q8s_gemv(w.s, w.n_out, w.n_in, y, out, w.n_out, T, b.st); }

// the rows of a window: one sequence's at *ctl + t on the current lane's pages (lr empty), or a batched window's
struct Rows {
    const int* ctl;
    LaneRows lr;
    const int* const* tabs;   // a batched window's rows' rotary tables (null: rope's own, the current lane's)
};

// an attention layer's mixer half on one card: the last half's sum folded into R (pend) and the norm, the card's q heads
// and their share of the KV heads projected, the window's K/V stored, attention over every position, the card's heads through
// wo, summed across the cards into b.bo (folded into R by the next half)
void attn_half(CardBuf& b, const LayerW& L, float* R, bool pend, int slot, const Rows& rows, int q0, int kv0, int call, int T, float eps,
               const Rope& rope) {
    add_norm(b, R, pend ? b.bo : nullptr, L.attn_norm, T, eps, true);
    mv(b, L.wq, b.xq, b.qkv, T);
    k_s27_prep<<<dim3(S27_NQ + 2 * S27_NKV, T), HD, 0, b.st>>>(b.qkv, S27_QKV, L.q_norm, L.k_norm, b.pool, slot, rows.ctl, b.pt, rows.lr,
                                                                rows.tabs, eps, rope);
    k_s27_attn<<<dim3(S27_ASPLIT, T, S27_NQ), 128, 0, b.st>>>(b.qkv, S27_QKV, b.pool, slot, rows.ctl, b.pt, rows.lr, kv0, q0, b.apart);
    k_s27_merge<<<dim3(T, S27_NQ), HD, 0, b.st>>>(b.qkv, S27_QKV, b.apart, b.att);
    ck(cudaGetLastError(), "attention");
    quantize_q8_1(b.att, b.xq, S27_NQ * HD, T, b.st);
    mv(b, L.wo, b.xq, b.bo, T);
    sum_cards(b, b.bo, T, rows.ctl, call);
}

void ffn27(CardBuf& b, const CardW& w, const LayerW& L, float* R, const int* ctl, int call, int T, float eps) {
    const int ffs = w.fd1 - w.fd0;
    add_norm(b, R, b.bo, L.ffn_norm, T, eps, true);
    mv(b, L.ffn_gu, b.xq, b.gu, T);
    swiglu_stacked_q8_1(b.gu, b.hq, ffs, T, b.st);
    mv(b, L.ffn_down, b.hq, b.bo, T);
    sum_cards(b, b.bo, T, ctl, call);
}
}  // namespace

Engine::Engine(Model& m, int max_ctx, int lanes) : m_(m), max_ctx_(max_ctx) {
    rope_.base = m_.rope_base;
    long long env_orig = 0;
    if (std::getenv("FNX_YARN_ORIG_CTX")) {
        env_orig = env_int("FNX_YARN_ORIG_CTX", 0);
        if (env_orig <= 0 || env_orig > INT32_MAX)
            throw std::runtime_error("FNX_YARN_ORIG_CTX: want a trained context length from 1 to " + std::to_string(INT32_MAX) + ", got " +
                                     std::to_string(env_orig));
    }
    const RopeChoice rc = choose_rope(m_.yarn_factor, m_.yarn_orig, m_.ctx_train, std::getenv("FNX_ROPE_SCALE") ? rope_scale() : -1.0, env_orig);
    if (max_ctx > rc.limit)
        throw std::runtime_error("context " + std::to_string(max_ctx) + " is past the " + std::to_string(rc.limit) + " the rotary covers (" +
                                 (rc.S ? "trained " + std::to_string(rc.orig) + " x YaRN " + std::to_string(rc.S)
                                       : "the model's trained length; FNX_ROPE_SCALE=S (--rope-scale S) covers trained x S") +
                                 ")");
    if (rc.S) {
        rope_ = yarn_rope(m_.rope_base, rc.orig, rc.S, NROT);
        std::fprintf(stderr, "27b: rope scaling YaRN x%.4f%s: n_ctx_orig %lld, pairs %.0f..%.0f of %d blended, mscale %.4f\n", rc.S,
                     rc.from_model ? " (the model's own)" : "", rc.orig, rope_.c_low, rope_.c_high, NROT / 2, rope_.mscale);
    }
    if (std::getenv("FNX_KV8") && env_int("FNX_KV8", 0) != 0)
        throw std::runtime_error("FNX_KV8: the 27B keeps its K/V in f16 (llama.cpp's default); int8 K/V is Flash-Next's");
    // Flash-Next's engine settings (experts, the arena, hyper-connections, QSA, its prompt path's knobs): the 27B has none of
    // what they set, so one given is a mistake, not a run of the default
    for (const char* k : {"FNX_EXPERT_RAM_GIB", "FNX_EXPERT_PROFILE", "FNX_ARENA", "FNX_ARENA_RESERVE_MIB", "FNX_ARENA_BASE_PAGES",
                          "FNX_ARENA_CHURN", "FNX_HC1", "FNX_HC_BAR_WRAP", "FNX_NO_TWIN", "FNX_RING_TAG_START", "FNX_PF_OVERLAP",
                          "FNX_PF_PASS", "FNX_PF_STAGE_ROWS", "FNX_PF_STAGE_MIB", "FNX_PF_RING_SM", "FNX_PF_ATT_PIECES", "FNX_PF_QSA_ROWS",
                          "FNX_PF_ATTN_DEC", "FNX_PF_F16ACC", "FNX_PF_MOE_Q8", "FNX_PF_DRAFT8", "FNX_ROUTE_COUNT", "FNX_ROUTE_TRACE",
                          "FNX_PREFETCH_PROBE", "FNX_PF_SEQ_START", "FNX_QSA_DENSE", "FNX_MTP_EXPERTS", "FNX_PF_GEMM_LOG",
                          "FNX_PF_HC_CHECK"})
        if (std::getenv(k)) throw std::runtime_error(std::string(k) + " is Flash-Next's: the 27B has nothing it sets");
    const long long pf_rows = env_int("FNX_PF_ROWS", pf_c_);
    if (pf_rows < MAXT || pf_rows > 8192 || pf_rows % MAXT)
        throw std::runtime_error("FNX_PF_ROWS: a multiple of " + std::to_string(MAXT) + " from " + std::to_string(MAXT) + " to 8192, got " +
                                 std::to_string(pf_rows));
    pf_c_ = (int) pf_rows;
    const int extra_pages = (std::max(lanes, 1) - 1) * lane_block_pages();
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m.cards[c];
        CardBuf* b = new CardBuf();
        b->dev = w.dev;
        ck(cudaSetDevice(w.dev), "cudaSetDevice");
        for (int p = 0; p < NCARD; ++p)
            if (p != c) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer access");
                cudaGetLastError();
            }
        ck(cudaStreamCreateWithFlags(&b->st, cudaStreamNonBlocking), "stream");
        own_stream(b->st);
        ck(cudaEventCreate(&b->ev0), "event");
        own_event(b->ev0);
        ck(cudaEventCreate(&b->ev1), "event");
        own_event(b->ev1);
        if (w.nq != S27_NQ || w.nkv != S27_NKV || w.q0 != c * S27_NQ || w.kv0 != w.q0 / (NQH / NKVH) || w.L[3].wq.n_out != S27_QKV)
            throw std::runtime_error("27b: card " + std::to_string(c) + "'s attention heads are not the split s27_attn.cuh runs");
        const int ffs = w.fd1 - w.fd0;
        b->R = dalloc<float>((size_t) MAXT * N);
        b->mixed = dalloc<float>((size_t) MAXT * N);
        b->bo = dalloc<float>((size_t) MAXT * N);
        b->hfin = dalloc<float>((size_t) MAXT * N);
        b->xq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(S27_XQ_COLS));
        b->xtq = dalloc<int8_t>((size_t) MAXT * S27_XQ_COLS);
        b->xtd = dalloc<float>((size_t) MAXT * S27_XQ_COLS / 32);
        q8_twin(b->xq, (size_t) MAXT * q8_1_bytes(S27_XQ_COLS), b->xtq, b->xtd);
        b->hq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(ffs));
        q8_twin(b->hq, (size_t) MAXT * q8_1_bytes(ffs), dalloc<int8_t>((size_t) MAXT * ffs), dalloc<float>((size_t) MAXT * ffs / 32));
        for (const void* q : {(const void*) b->xq, (const void*) b->hq}) own([q] { q8_untwin(q); });
        b->qkvz = dalloc<float>((size_t) MAXT * (w.conv_ch() + w.nv() * HS));
        b->ab = dalloc<float>((size_t) MAXT * 2 * w.nv());
        b->gdn_scratch = dalloc<float>(gdn_scratch_floats(w.nv()));
        b->qkv = dalloc<float>((size_t) MAXT * S27_QKV);
        b->att = dalloc<float>((size_t) MAXT * S27_NQ * HD);
        b->apart = dalloc<float>((size_t) MAXT * S27_NQ * S27_ASPLIT * (HD + 2));
        b->gu = dalloc<float>((size_t) MAXT * 2 * ffs);
        b->head_logits = dalloc<float>((size_t) MAXT * (w.v1 - w.v0));
        b->best_v = dalloc<float>(MAXT);
        b->best_i = dalloc<int>(MAXT);
        b->am_keys = dalloc<unsigned long long>(MAXT);
        b->err = dalloc<unsigned>(1);
        b->S.assign(NL, nullptr);
        b->chist.assign(NL, nullptr);
        b->snap.assign(NL, nullptr);
        b->hsnap.assign(NL, nullptr);
        const size_t snap_floats = (size_t) (MAXT - 1) * w.nv() * HS * HS;
        int n_rec = 0;
        for (int l = 0; l < NL; ++l) n_rec += is_attn(l) ? 0 : 1;
        b->snap_block = dalloc<float>(snap_floats * n_rec);
        b->snap_block_bytes = snap_floats * n_rec * sizeof(float);
        for (int l = 0, k = 0; l < NL; ++l)
            if (!is_attn(l)) {
                b->S[l] = dalloc<float>((size_t) w.nv() * HS * HS);
                b->chist[l] = dalloc<float>((size_t) 2 * 3 * w.conv_ch());
                b->snap[l] = b->snap_block + snap_floats * k++;
                b->hsnap[l] = dalloc<float>((size_t) MAXT * 3 * w.conv_ch());
            }
        b->ctl = dalloc<int>(2);
        b->r3.card = c;
        b->r3.in = dalloc<unsigned long long>(r3_in_words());
        b->r3.ag = dalloc<unsigned long long>(r3_ag_words());
        b->r3.err = b->err;
        if ((b->ring32 = ring_f32())) {
            b->ring.card = c;
            b->ring.in = dalloc<unsigned long long>(ring_in_words(MAXT));
            b->ring.ag = dalloc<unsigned long long>(ring_ag_words(MAXT));
            b->ring.f32 = true;
            b->ring.err = b->err;
        }
        cb_.push_back(b);
    }
    {   // every link carries a probe's stores before the ring's (src/peer_probe.hpp)
        std::vector<int> devs;
        for (int c = 0; c < NCARD; ++c) devs.push_back(m.cards[c].dev);
        for (const std::string& line : peer_report(probe_peer_stores(devs), NCARD)) std::fprintf(stderr, "%s\n", line.c_str());
    }
    for (int c = 0; c < NCARD; ++c) {
        cb_[c]->r3.next_in = cb_[(c + 1) % NCARD]->r3.in;
        cb_[c]->r3.next_ag = cb_[(c + 1) % NCARD]->r3.ag;
        cb_[c]->ring.next_in = cb_[(c + 1) % NCARD]->ring.in;
        cb_[c]->ring.next_ag = cb_[(c + 1) % NCARD]->ring.ag;
    }
    // the prompt path's buffers now, so the pool below is sized from what is left after them
    prefill_alloc();
    // the K/V pool: as many pages as every card's free VRAM holds beside FNX_KV_RESERVE_MIB (default 1024: the window
    // graphs captured at first use, cuBLAS's workspaces, the lanes' page tables and rotary tables), at least the whole
    // context and the extra lanes' blocks. A page is the card's own KV heads' worth (s27_attn.cuh counts them per card
    // count): three cards, 8.9 MB — 34.8 KB a position, 16 attention layers and the draft layer, two heads of 256 f16
    // for K and V; four cards, one head, half that.
    const long long reserve_mib = env_int("FNX_KV_RESERVE_MIB", 1024);
    if (reserve_mib < 0) throw std::runtime_error("FNX_KV_RESERVE_MIB: want MiB >= 0, got " + std::to_string(reserve_mib));
    const size_t page_bytes = S27_PAGE * sizeof(__half);
    const int ctx_pages = max_ctx / S27_PP + 2;
    int n_pages = INT32_MAX;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
        size_t fr = 0, tot = 0;
        ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
        const size_t reserve = (size_t) reserve_mib << 20;
        const int fit = fr > reserve ? (int) ((fr - reserve) / page_bytes) : 0;
        if (fit < ctx_pages + extra_pages)
            throw std::runtime_error("27b: card " + std::to_string(c) + " has " + std::to_string(fr >> 20) + " MiB free: " +
                                     std::to_string(ctx_pages + extra_pages) + " K/V pages (" + std::to_string(max_ctx) + " positions and " +
                                     std::to_string(std::max(lanes, 1) - 1) + " more lanes' states) and " + std::to_string(reserve_mib) +
                                     " MiB in reserve (FNX_KV_RESERVE_MIB) do not fit");
        n_pages = std::min(n_pages, fit);
    }
    if (std::getenv("FNX_KV_PAGES")) {
        const long long want = env_int("FNX_KV_PAGES", 0);
        if (want < ctx_pages + extra_pages || want > n_pages)
            throw std::runtime_error("FNX_KV_PAGES: want " + std::to_string(ctx_pages + extra_pages) + " to " + std::to_string(n_pages));
        n_pages = (int) want;
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf* b = cb_[c];
        ck(cudaSetDevice(b->dev), "cudaSetDevice");
        b->n_pages = n_pages;
        b->pool = dalloc<__half>((size_t) n_pages * S27_PAGE);
        b->pt = dalloc<int>((size_t) n_pages);
        LaneDev L;
        L.pt = b->pt;
        L.S = b->S;
        L.chist = b->chist;
        L.ctl = dalloc<int>(2);
        b->lanes.push_back(L);
        size_t fr = 0, tot = 0;
        ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
        std::fprintf(stderr, "27b: card %d: %d K/V pages (%d positions), %zu MiB free after them\n", c, n_pages, n_pages * S27_PP, fr >> 20);
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "init sync"); }
    lanes_.emplace_back();
    for (int i = n_pages; i-- > 0;) free_pages_.push_back(i);   // taken from the back: low pages first
}

// the ring words' 16-bit tags (s27_ring3.cuh) wrap every 256 tags: every card idle, every buffer zeroed each time the
// tags cross a multiple of 128, so no word left in a slot is 256 tags old
void Engine::clear_rings() {
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        ck(cudaMemset(b->r3.in, 0, r3_in_words() * 8), "ring clear");
        ck(cudaMemset(b->r3.ag, 0, r3_ag_words() * 8), "ring clear");
        if (b->ring32) {
            ck(cudaMemset(b->ring.in, 0, ring_in_words(MAXT) * 8), "ring clear");
            ck(cudaMemset(b->ring.ag, 0, ring_ag_words(MAXT) * 8), "ring clear");
        }
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
}

void Engine::tags_room(unsigned n) {
    if (tag_max_ + n + 1u >= TAG_END) {
        clear_rings();
        std::fprintf(stderr, "27b: ring tags reached %u: tags start again\n", tag_max_);
        tag_max_ = 1;
        return;
    }
    if ((tag_max_ + n) / 128u != tag_max_ / 128u) clear_rings();
}

// the 27B has no expert arena: the pool's pages are its own
void Engine::arena_reclaim(double) {}
void Engine::arena_churn() {}
std::vector<int> Engine::arena_lent() const { return {}; }

void Engine::ensure_pages(int n_pos) { ensure_pages_of(cur_, n_pos); }

void Engine::ensure_pages_of(int lane, int n_pos) {
    LaneHost& L = lanes_[(size_t) lane];
    const int need = n_pos / S27_PP + 1;
    if (need > cb_[0]->n_pages) throw Refused("context full: " + std::to_string(n_pos) + " positions");
    const size_t had = L.pages.size();
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && on_pool_short && !in_pool_short_ && pending_.empty()) {
        const int was = cur_;
        in_pool_short_ = true;
        try {
            on_pool_short(lane, short_by);
        } catch (...) {
            in_pool_short_ = false;
            throw;
        }
        in_pool_short_ = false;
        if (cur_ != was) throw std::runtime_error("on_pool_short left lane " + std::to_string(cur_) + " current, not " + std::to_string(was));
    }
    if (need - (int) had > (int) free_pages_.size())
        throw Refused("the K/V pool is full (" + std::to_string(n_pos) + " positions on this lane, " + std::to_string(need - (int) had) +
                      " more pages wanted, " + std::to_string(free_pages_.size()) + " free)");
    while ((int) L.pages.size() < need) {
        L.pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    if (L.pages.size() == had) return;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        ck(cudaMemcpy(b->lanes[(size_t) lane].pt + had, L.pages.data() + had, (L.pages.size() - had) * sizeof(int), cudaMemcpyHostToDevice),
           "page table");
    }
}

// the 27B routes no experts
std::vector<unsigned long long> Engine::probe_counts() const { return {}; }
std::vector<unsigned> Engine::route_counts() const { return {}; }
std::vector<unsigned> Engine::route_trace(size_t) const { return {}; }

std::vector<int> Engine::graph_key(const std::vector<int>& lanes) const {
    std::vector<int> k(lanes);
    // a lane's rotary table is either none or its own buffer (made once), so whether it has one names the pointer a
    // graph holds: in a batched window's rows, or a single lane's window as an argument
    for (int l : lanes) k.push_back(cb_[0]->lanes[(size_t) l].mrope != nullptr);
    return k;
}

// a lane's states on each card, carved from a run of pool pages: recurrent states, conv histories (two halves)
int Engine::lane_block_pages() const {
    int n = 0;
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        size_t bytes = 256 * 64;   // alignment
        for (int l = 0; l < NL; ++l)
            if (!is_attn(l)) bytes += (size_t) w.nv() * HS * HS * 4 + (size_t) 2 * 3 * w.conv_ch() * 4 + 512;
        const size_t page = S27_PAGE * sizeof(__half);
        n = std::max(n, (int) ((bytes + page - 1) / page));
    }
    return n;
}

int Engine::add_lane() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("add_lane: a window is not committed");
    const int n = lane_block_pages();
    std::vector<int> fr(free_pages_);
    std::sort(fr.begin(), fr.end());
    int start = -1;
    for (int i = (int) fr.size() - n; i >= 0; --i)
        if (fr[(size_t) i + n - 1] - fr[(size_t) i] == n - 1) {
            start = fr[(size_t) i];
            break;
        }
    if (start < 0) throw std::runtime_error("add_lane: no run of " + std::to_string(n) + " free pages in the K/V pool");
    free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int p) { return p >= start && p < start + n; }),
                      free_pages_.end());
    LaneHost h;
    h.block0 = start;
    h.block_n = n;
    lanes_.push_back(h);
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        char* at = (char*) (b.pool + (size_t) start * S27_PAGE);
        ck(cudaMemset(at, 0, (size_t) n * S27_PAGE * sizeof(__half)), "lane block");
        auto take = [&](size_t bytes) {
            char* p = at;
            at += (bytes + 255) & ~(size_t) 255;
            return p;
        };
        LaneDev L;
        L.pt = dalloc<int>((size_t) b.n_pages);
        L.S.assign(NL, nullptr);
        L.chist.assign(NL, nullptr);
        for (int l = 0; l < NL; ++l)
            if (!is_attn(l)) {
                L.S[l] = (float*) take((size_t) w.nv() * HS * HS * 4);
                L.chist[l] = (float*) take((size_t) 2 * 3 * w.conv_ch() * 4);
            }
        L.ctl = dalloc<int>(2);
        b.lanes.push_back(L);
    }
    return (int) lanes_.size() - 1;
}

// the 27B has no PLE rows: ple must be null
void Engine::embed_rows(int p0, int T, const std::function<int(int)>& tok_at_cell, float* emb, float* ple) const {
    if (ple) throw std::runtime_error("27b: embed_rows takes no PLE rows");
    for (int t = 0; t < T; ++t) embed_token(tok_at_cell(p0 + t), p0 + t, emb + (size_t) t * N);
}

// a batched window carries each lane's next token and drafts, never a picture's cells
void Engine::embed_into(int r, int k, int p0, const std::function<int(int)>& tok_at, const char* batched) {
    if (batched)
        for (int j = 0; j < k; ++j)
            if (tok_at(p0 + j) >= IMAGE_TOK) throw std::runtime_error(std::string(batched) + ": an image cell in a batched window");
    embed_rows(p0, k, tok_at, h_.R0 + (size_t) r * N, nullptr);
}

void Engine::rows_upload(CardBuf& b, int T) {
    ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "R0");
}

// a sequence's device state on one card, in a fixed order: (pointer, bytes) pieces
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, bool, int pos, bool kv, unsigned hist_half,
                                                   const std::vector<int>& pages) {
    std::vector<std::pair<char*, size_t>> v;
    for (int l = 0; l < NL; ++l)
        if (!is_attn(l)) {
            v.emplace_back((char*) b.S[l], (size_t) w.nv() * HS * HS * sizeof(float));
            v.emplace_back((char*) (b.chist[l] + (size_t) hist_half * 3 * w.conv_ch()), (size_t) 3 * w.conv_ch() * sizeof(float));
        }
    if (kv)   // the pages holding positions 0 .. pos (every layer's K/V, the draft layer's too)
        for (int i = 0; i <= pos / S27_PP && i < (int) pages.size(); ++i)
            v.emplace_back((char*) (b.pool + (size_t) pages[(size_t) i] * S27_PAGE), S27_PAGE * sizeof(__half));
    return v;
}

// the layers of a window of T rows (one sequence's, or a batched window's from lanes: b.lane_rows[T], with the
// recurrent core in gdn), then the head: the output norm with the last sum folded in (kept in hfin for the draft
// layer), this card's vocabulary, its argmax
void Engine::enqueue_layers(int T, bool lanes, const GdnFn& gdn, bool dumps) {
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    auto dd = [&](const std::string& name, int layer, const float* p, int n) {
        if (dumps) dump_dev(name, layer, p, T, n);
    };
    // a card's rows: its window's at *b.ctl + t on its current lane's pages and rotary table, or a batched window's from
    // b.lane_rows, each with its own lane's
    auto rows_of = [&](CardBuf& b) {
        const int* const* lr = b.lane_rows[(size_t) T];
        return lanes ? Rows{b.ctl, LaneRows{lr, lr + MAXT}, lr + 2 * MAXT} : Rows{b.ctl, LaneRows{}, nullptr};
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            if (!is_attn(l)) {
                const int nv = w.nv();
                add_norm(b, b.R, l > 0 ? b.bo : nullptr, L.attn_norm, T, eps, true);
                q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
                gdn(b, w, L, l);
                mv(b, L.ssm_out, b.xq, b.bo, T);
                sum_cards(b, b.bo, T, b.ctl, call);
            } else {
                attn_half(b, L, b.R, l > 0, s27_slot(l), rows_of(b), w.q0, w.kv0, call, T, eps, rope_of(lanes ? nullptr : b.mrope));
            }
        });
        ++call;
        dd("attn_out", l, cb_[0]->bo, N);
        each([&](int, CardBuf& b, const CardW& w) { ffn27(b, w, w.L[l], b.R, b.ctl, call, T, eps); });
        ++call;
        dd("ffn_out", l, cb_[0]->bo, N);
    }
    each([&](int, CardBuf& b, const CardW& w) {
        add_norm(b, b.R, b.bo, w.out_norm, T, eps, true);
        ck(cudaMemcpyAsync(b.hfin, b.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, b.st), "hfin");
        mv(b, w.out, b.xq, b.head_logits, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.err, b.st);
    });
    dd("result_norm", -1, cb_[0]->hfin, N);
}

// every card's work for one window of T rows of the current lane, in stream order (captured into a graph unless dumping)
void Engine::enqueue(int T, bool dumps) {
    enqueue_layers(
        T, false,
        [&](CardBuf& b, const CardW& w, const LayerW& L, int l) {
            gdn_core(b.qkvz, b.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], b.xq, b.ctl, b.gdn_scratch, b.snap[l],
                     b.hsnap[l], w.nk, w.nv(), w.vmap, T, m_.eps, b.st);
        },
        dumps);
}

}  // namespace fnx

namespace fnx {

// enqueue()'s layers for a window of one row per lane (row r: lanes[r]): each row's recurrent core on its lane's state,
// its attention at its lane's position on its lane's pages (b.lane_rows[T])
void Engine::enqueue_batch(const std::vector<int>& lanes) {
    const int T = (int) lanes.size();
    enqueue_layers(
        T, true,
        [&](CardBuf& b, const CardW& w, const LayerW& L, int l) {
            float** rec = b.lane_gdn[(size_t) T];
            const GdnLanes gl{rec + (size_t) l * MAXT, rec + (size_t) (NL + l) * MAXT, b.lane_rows[(size_t) T]};
            gdn_core_lanes(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, w.nv(), w.vmap, gl, T, m_.eps, b.st);
        },
        false);
}

// enqueue_batch()'s layers for a window of several rows a lane (verify_rows): each lane's recurrent rows chained through
// its state as a window of its own (snapshots kept for commit_rows), every row's attention at its own position
// (b.row_pos) on its lane's pages
void Engine::enqueue_rows(const std::vector<int>& lanes, const std::vector<int>& counts) {
    const RowSegs sg = row_segs(counts);
    const int T = sg.T;
    (void) lanes;
    enqueue_layers(
        T, true,
        [&](CardBuf& b, const CardW& w, const LayerW& L, int l) {
            float** rec = b.lane_gdn[(size_t) T];
            for (const RowGroup& g : sg.groups) {
                const GdnLanes gl{rec + (size_t) l * MAXT + g.s0, rec + (size_t) (NL + l) * MAXT + g.s0, b.seg_ctl[(size_t) T] + g.s0,
                                  b.seg_row0[(size_t) T] + g.s0, b.snap[l], b.hsnap[l]};
                gdn_core_segs(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, w.nv(), w.vmap, gl, g.n,
                              g.count, m_.eps, b.st);
            }
        },
        false);
}

void Engine::mtp_attach(const Mtp& mtp) {
    if (mtp.cards.size() != (size_t) NCARD) throw std::runtime_error("mtp_attach: the draft layer is not on every card");
    mtp_ = &mtp;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        b.mtp_emb = dalloc<float>((size_t) MAXT * N);
        b.mtp_src = dalloc<float>((size_t) MAXT * N);
        b.mtp_cat = dalloc<float>((size_t) MAXT * 2 * N);
        b.mtp_R = dalloc<float>((size_t) MAXT * N);
        b.mtp_out = dalloc<float>((size_t) MAXT * N);
        b.mtp_logits = dalloc<float>((size_t) std::max(1, mc.dv1 - mc.dv0));
        b.mtp_bv = dalloc<float>(1);
        b.mtp_bi = dalloc<int>(1);
        b.mtp_ctl = dalloc<int>(2);
        b.mtp_pen = dalloc<int>(1 + 2 * PEN_CAP);
        b.mtp_pen_par = dalloc<float>(3);
        b.mtp_tk_v = dalloc<float>(TOPK_N);
        b.mtp_tk_i = dalloc<int>(TOPK_N);
        b.mtp_tk_err = dalloc<unsigned>(1);
        b.mtp_zero = dalloc<int>(1);
    }
    if (!h_.mtp_emb) {
        h_.mtp_emb = halloc<float>((size_t) MAXT * N);
        h_.mtp_ctl = halloc<int>(2);
        h_.mtp_pen = halloc<int>((size_t) NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par = halloc<float>(3);
        h_.mtp_tk_v = halloc<float>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_i = halloc<int>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_err = halloc<unsigned>(NCARD);
    }
    draft_index_.assign((size_t) m_.vocab, -1);
    for (size_t i = 0; i < mtp.vocab.size(); ++i) draft_index_[(size_t) mtp.vocab[i]] = (int) i;
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
}

// one draft-layer window of T rows of (token embedding mtp_emb, source mtp_src) at cells mtp_ctl[0]..: the nextn
// projection of [enorm(e) | hnorm(h)], an attention layer and an FFN as the trunk's (the draft layer's slot of the pages),
// the head norm (every row into mtp_out: a chained pass reads its own row there), then the last row through this card's rows of
// the draft head, its penalties, its argmax
void Engine::mtp_enqueue(int T) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        k_norm_cat<<<T, 1024, 0, b.st>>>(b.mtp_emb, b.mtp_src, w.mtp_enorm, w.mtp_hnorm, b.mtp_cat, eps);
        ck(cudaGetLastError(), "norm_cat");
        quantize_q8_1(b.mtp_cat, b.xq, 2 * N, T, b.st);
        mv(b, w.mtp_eh, b.xq, b.mtp_R, T);
        attn_half(b, w.mtp, b.mtp_R, false, S27_SLOTS - 1, Rows{b.mtp_ctl, LaneRows{}, nullptr}, w.q0, w.kv0, 0, T, eps, rope_of(b.mrope));
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn27(b, w, w.mtp, b.mtp_R, b.mtp_ctl, 1, T, eps);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const int nv = mc.dv1 - mc.dv0;
        ck(cudaSetDevice(b.dev), "set");
        add_norm(b, b.mtp_R, b.bo, w.mtp_head_norm, T, eps, false);
        ck(cudaMemcpyAsync(b.mtp_out, b.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, b.st), "draft out");
        quantize_q8_1(b.mtp_out + (size_t) (T - 1) * N, b.xq, N, 1, b.st);
        mv(b, mc.head, b.xq, b.mtp_logits, 1);
        penalize_row(b.mtp_logits, nv, b.mtp_pen, b.mtp_pen_par, b.st);
        argmax(b.mtp_logits, nv, 1, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.err, b.st);
    }
}

}  // namespace fnx
