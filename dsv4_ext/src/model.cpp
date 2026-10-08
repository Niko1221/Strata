#include "dsv4/model.hpp"

#include "dsv4/dequant.hpp"
#include "dsv4/gpu.hpp"
#include "dsv4/mem_plan.hpp"
#include "dsv4/ops.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <thread>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "dsv4/ops.hpp"
namespace dsv4 {
namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// ---------------------------------------------------------------- activation "QAT" simulation (model.py act_quant / fp4_act_quant)
float pow2_ceil(float v) { int e; float m = std::frexp(v, &e); return std::ldexp(1.f, m == 0.5f ? e - 1 : e); }
float fp8_round(float t) {
    if (t == 0.f) return 0.f;
    const float a = std::fabs(t); int e; std::frexp(a, &e);
    const int E = e - 1;
    const float step = E < -6 ? std::ldexp(1.f, -9) : std::ldexp(1.f, E - 3);
    float r = std::nearbyint(a / step) * step;
    if (r > 448.f) r = 448.f;
    return std::copysign(r, t);
}
void fp8_sim(float* x, int n, int blk) {
    for (int b = 0; b < n; b += blk) {
        float amax = 1e-4f;
        for (int i = b; i < std::min(n, b + blk); ++i) amax = std::max(amax, std::fabs(x[i]));
        const float s = pow2_ceil(amax / 448.f);
        for (int i = b; i < std::min(n, b + blk); ++i) x[i] = fp8_round(std::min(448.f, std::max(-448.f, x[i] / s))) * s;
    }
}
float fp4_round(float t) {
    const float a = std::fabs(t);
    float r;
    if (a <= 0.25f) r = 0.f; else if (a < 0.75f) r = 0.5f; else if (a <= 1.25f) r = 1.f; else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.f; else if (a < 3.5f) r = 3.f; else if (a <= 5.f) r = 4.f; else r = 6.f;
    return std::copysign(r, t);
}
void fp4_sim(float* x, int n, int blk) {
    for (int b = 0; b < n; b += blk) {
        float amax = 6.f * std::ldexp(1.f, -126);
        for (int i = b; i < std::min(n, b + blk); ++i) amax = std::max(amax, std::fabs(x[i]));
        const float s = pow2_ceil(amax / 6.f);
        for (int i = b; i < std::min(n, b + blk); ++i) x[i] = fp4_round(std::min(6.f, std::max(-6.f, x[i] / s))) * s;
    }
}
void hadamard(float* x, int n) {  // natural-order Walsh-Hadamard, scaled by n^-0.5 (rotate_activation)
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; ++j) { const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
    const float s = 1.f / std::sqrt((float) n);
    for (int i = 0; i < n; ++i) x[i] *= s;
}

// ---------------------------------------------------------------- tensors
struct Tn {
    const GgufTensor* g = nullptr;
    const uint8_t* h = nullptr;   // host (mmap)
    uint8_t* d = nullptr;         // device copy (only for matrices that are matvec'd on the GPU)
    bool no_gpu = false;          // the device has no kernel for this type: never uploaded, host only
    uint32_t type = 0;
    int64_t in = 0, out = 1, ne = 1;
    size_t rb = 0;                // bytes per row
    explicit operator bool() const { return g != nullptr; }
    const float* f() const { return (const float*) h; }
};

struct Comp {  // KV compressor state (model.py Compressor), decode-only
    int ratio = 0, d = 0, coff = 1;
    bool overlap = false, rotate = false;
    Tn wkv, wg, ape, norm;
    std::vector<float> kvs, scs;   // [coff*ratio][coff*d]
    float* cache = nullptr;        // [ctx/ratio + 1][d]
    std::vector<float> own;        // storage when cache is not shared with the attention kv buffer
};

// A layer's window ring is written at pos % win, so a chunk of T tokens would let token p see the
// rows of tokens p+1..p+T-1. The snapshot below is what prevents that: the ring is copied aside
// before the chunk is written, and an index whose slot now holds a FUTURE token is redirected into
// the snapshot, which holds exactly what the token-by-token path would have read.
struct Layer {
    int ratio = 0;
    int snap_base = 0;      // row index in kvbuf where the pre-chunk window snapshot starts
    std::vector<int> ring_pos;   // slot -> position of the token currently in it (-1 = never written)
    Tn attn_norm, q_a, q_a_norm, q_b, kv, kv_norm, o_a, o_b, sinks;
    Tn ffn_norm, gate_inp, exp_bias, tid2eid;
    Tn sh_gate, sh_up, sh_down;
    Tn hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    Comp ac, ic;
    Tn i_q_b, i_proj;
    bool has_idx = false;
    std::vector<float> kvbuf;      // [win][ncomp_max + win][hd]: ring, compressed entries, window snapshot
    int ncomp_max = 0;
    int max_idx = 0;              // widest attention index list one token can produce
    // routed experts
    Tn eg, eu, ed;
    size_t bg = 0, bu = 0, bd = 0, bpe = 0;     // bytes per expert: gate, up, down, total
    int slots = 0;
    bool gpu_experts = false;   // the device has kernels for eg/eu/ed: only then are VRAM slots worth anything
    uint8_t* dpool = nullptr;
    std::vector<int> slot_of, occ;              // expert -> slot, slot -> expert
    std::vector<uint64_t> freq;                 // routing frequency (profile counts + runtime)
    std::vector<int> ram_idx;                   // expert -> index in the RAM arena (-1 = not in RAM)
    std::vector<uint8_t> ram;                   // RAM arena: only experts the profile marks as used
    // LRU arena for MISS experts (neither in `ram` nor in a VRAM slot). Without it every miss re-reads
    // bpe bytes from the mmap'ed shard: for a model larger than RAM that is a disk read per expert.
    std::vector<uint8_t> cpool;                 // ncache slots of bpe bytes each
    std::vector<int> cslot_of, cocc;            // expert -> slot, slot -> expert
    std::vector<uint64_t> cstamp;               // slot -> last use (LRU clock)
    int ncache = 0;
};

struct ExpSrc { const uint8_t *g, *u, *d; const uint8_t* packed; };  // packed != null: contiguous [g|u|d]

uint64_t fnv(uint64_t h, uint64_t v) { for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xff; h *= 1099511628211ULL; } return h; }

void cpu_mv(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* x, float* y) {
    const size_t rb = row_bytes(type, in);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < rows; ++r) y[r] = row_dot(type, W + (size_t) r * rb, x, in);
}

}  // namespace

std::string decode_token_text(const std::string& tok) {
    static std::map<uint32_t, uint8_t> inv;
    if (inv.empty()) {
        std::vector<int> bs, cs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        cs = bs; int n = 0;
        for (int b = 0; b < 256; ++b) if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
        for (size_t i = 0; i < bs.size(); ++i) inv[(uint32_t) cs[i]] = (uint8_t) bs[i];
    }
    std::string out;
    for (size_t i = 0; i < tok.size();) {
        const unsigned char c = (unsigned char) tok[i];
        uint32_t cp; int len;
        if (c < 0x80) { cp = c; len = 1; } else if ((c >> 5) == 6) { cp = c & 0x1f; len = 2; } else if ((c >> 4) == 14) { cp = c & 0x0f; len = 3; } else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < tok.size(); ++k) cp = (cp << 6) | ((unsigned char) tok[i + k] & 0x3f);
        auto it = inv.find(cp);
        if (it != inv.end()) out.push_back((char) it->second); else out.append(tok, i, (size_t) len);
        i += (size_t) len;
    }
    return out;
}

// ================================================================================================
struct Model::Impl {
    RunOpts o;
    Dsv4Config c;
    GgufHeader h;
    ExpertInventory inv;
    std::unordered_map<std::string, const GgufTensor*> by_name;
    std::vector<const uint8_t*> maps; std::vector<size_t> map_len;
    std::vector<Layer> L;
    int nrun = 0, hd = 0, rd = 0, win = 0, dim = 0, hc = 4;
    Tn embd, outw, outnorm, out_hc_fn, out_hc_base, out_hc_scale;
    std::vector<float> cos_y, sin_y, cos_p, sin_p;   // YaRN (compressed layers) / plain (window-only layers) rope tables
    uint64_t model_hash = 0;
    std::vector<std::string> toks;
    // device scratch
    float *d_x = nullptr, *d_y = nullptr, *d_g = nullptr, *d_u = nullptr, *d_a = nullptr, *d_yh = nullptr;
    std::vector<void*> allocs;
    uint8_t* d_stage = nullptr;      // device pool MISS experts are staged through (see gpu::staging)
    // ---- prefill scratch (see forward_chunk) ----
    // A chunk of T tokens shares every weight row: the projections become [rows]x[in] times [T], and
    // the MoE evaluates each routed expert ONCE for all the tokens of the chunk that picked it.
    int pf_cap = 0;                  // tokens per chunk the scratch was sized for; 0 = no batching
    int64_t pf_xcap = 0, pf_ycap = 0;   // widest input / output block the device scratch holds
    float *d_pX = nullptr, *d_pY = nullptr;               // device activation in / out blocks
    float *d_pg = nullptr, *d_pu = nullptr, *d_pa = nullptr, *d_pd = nullptr, *d_pYm = nullptr;
    // host arena, sized once in load() for pf_cap tokens
    std::vector<float> pf, pf_w;
    std::vector<int32_t> pf_i;
    std::vector<int> pf_ic;
    std::vector<gpu::ExpBatch> pf_b;
    float *h_h = nullptr, *h_hn = nullptr, *h_yv = nullptr, *h_nrm = nullptr;
    float *h_qa = nullptr, *h_qr = nullptr, *h_q = nullptr, *h_kv = nullptr, *h_kvn = nullptr;
    float *h_o = nullptr, *h_t = nullptr, *h_lg = nullptr, *h_mixes = nullptr, *h_pre = nullptr;
    float *h_post = nullptr, *h_comb = nullptr, *h_acc = nullptr, *h_mg = nullptr, *h_mu = nullptr;
    float *h_ma = nullptr, *h_md = nullptr, *h_e = nullptr, *h_x = nullptr, *h_xn = nullptr;
    float *h_sg = nullptr, *h_su = nullptr, *h_sa = nullptr, *h_sy = nullptr;
    float *h_w = nullptr, *h_wt = nullptr;
    int32_t *h_attn_idx = nullptr; int *h_attn_n = nullptr;
    int32_t *h_sel = nullptr, *h_tok = nullptr, *h_tk = nullptr;
    int *h_cnt = nullptr, *h_off = nullptr, *h_at = nullptr, *h_eid = nullptr;
    int pf_mix = 0, pf_gin = 0, pf_max_list = 0;
    uint64_t pf_host_bytes = 0, pf_dev_bytes = 0;
    // The device block d_pX holds whatever pf_up() last copied. Re-uploading the same bytes is pure
    // PCIe waste, so the last upload is remembered - but ONLY until the next layer writes its
    // activations, which is what pf_clear_note() is for. Forgetting to call it would hand a kernel a
    // stale block, so it is called at the top of every block_batch().
    const float* pf_note_host = nullptr;
    size_t pf_note_bytes = 0;
    /// Upload the used part of a [T][stride] activation block (in elements per row).
    /// Returns the device pointer, or hx itself when the device cannot take it.
    const float* pf_up(const float* hx, int64_t stride, int T, int64_t in) {
        if (!o.gpu || !d_pX) return hx;
        const size_t bytes = ((size_t) (T - 1) * stride + (size_t) in) * 4;
        if (bytes > (size_t) pf_xcap * 4) return hx;
        if (pf_note_host == hx && pf_note_bytes == bytes) return d_pX;
        gpu::h2d(d_pX, hx, bytes);
        pf_note_host = hx; pf_note_bytes = bytes;
        return d_pX;
    }
    void pf_clear_note() { pf_note_host = nullptr; pf_note_bytes = 0; }
    /// True when a batched matmul over this block can run on the device at all. y_stride is the
    /// stride the CALLER wants its output rows at: the kernel writes them there directly, so no
    /// repacking pass is needed on the host (and none of the caller's other rows get clobbered).
    bool pf_dev(int64_t stride, int T, int64_t in, int64_t rows, int64_t y_stride) const {
        // Diagnostics only: DSV4_NO_DEVICE_MATMUL=1 keeps the batched projections on the host, so a parity
        // failure can be pinned on the matmul path or on the expert path (DSV4_NO_DEVICE_EXPERTS).
        static const bool off = std::getenv("DSV4_NO_DEVICE_MATMUL") != nullptr;
        return !off && o.gpu && d_pX && d_pY && T > 0 && T <= pf_cap && T <= gpu::kMaxSel &&
               (size_t) ((T - 1) * stride + in) * 4 <= (size_t) pf_xcap * 4 &&
               (size_t) ((T - 1) * y_stride + rows) * 4 <= (size_t) pf_ycap * 4;
    }
    /// Host floats per token of a chunk, matching the arena layout in pf_alloc() exactly.
    int64_t pf_per_token() const {
        const int64_t nh_hd = (int64_t) c.n_head * hd;
        return (int64_t) hc * dim * 2                      // h_h, h_hn
             + (int64_t) dim * 6                           // yv, nrm, e, x, xn, acc
             + (int64_t) c.q_lora * 2                      // q_a, q_r
             + nh_hd * 2                                   // q, o
             + (int64_t) hd * 2                            // kv, kv_norm
             + (int64_t) c.o_groups * c.o_lora             // the o_a group scratch
             + c.n_expert                                  // router logits
             + (int64_t) pf_mix * 2 + hc * 2               // hyper-connections
             + (int64_t) c.ff_exp * 6 + (int64_t) dim * 2; // shared expert + host expert buckets
    }
    /// The largest chunk that fits the memory this build is willing to spend on it.
    int pf_pick(int want) const {
        if (want <= 1) return 0;
        int t = std::min(want, 4096);
        const int64_t host_budget = (int64_t) 512 << 20;   // half a GiB of prefill arena, at most
        while (t > 8 && (int64_t) t * pf_per_token() * 4 > host_budget) t /= 2;
        return t;
    }
    /// Allocate the prefill arena. On the device the buffers are sized from pf_cap; if a cudaMalloc
    /// fails the chunk is halved and retried, because a smaller chunk is still far better than none.
    void pf_alloc() {
        // The widest attention index list a token can produce, straight from the config: the window
        // plus every compressed entry the context can hold. Computed here rather than from the Layer
        // array so the arena can be sized BEFORE the layers allocate their kv buffers.
        pf_max_list = win;
        for (int l = 0; l < nrun; ++l) {
            const int r = c.compress_ratios[(size_t) l];
            if (r > 0) pf_max_list = std::max(pf_max_list, win + o.ctx / r + 1);
        }
        // An indexer layer can select idx_topk compressed entries whatever the context says.
        pf_max_list = std::max(pf_max_list, win + c.idx_topk);
        while (pf_cap > 0) {
            const int T = pf_cap;
            const int64_t nh_hd = (int64_t) c.n_head * hd;
            // The widest row any batched matmul reads or writes, in elements. Both the input block
            // and the output block are sized from it, so pf_dev() can never be handed a block that
            // runs past either buffer.
            int64_t max_span = (int64_t) hc * dim;                 // the residual streams
            max_span = std::max(max_span, nh_hd);                  // q / o
            max_span = std::max(max_span, (int64_t) c.o_groups * c.o_lora);
            max_span = std::max(max_span, (int64_t) c.ff_exp);
            max_span = std::max(max_span, (int64_t) dim);
            max_span = std::max(max_span, (int64_t) c.q_lora);
            max_span = std::max(max_span, (int64_t) c.n_expert);
            max_span = std::max(max_span, (int64_t) pf_mix);
            pf_xcap = (int64_t) (T - 1) * max_span + max_span;
            pf_ycap = (int64_t) (T - 1) * max_span + max_span;
            bool ok = true;
            if (o.gpu && !gpu::is_emulated()) {
                d_pX = (float*) galloc((size_t) pf_xcap * 4);
                d_pY = (float*) galloc((size_t) pf_ycap * 4);
                d_pg = (float*) galloc((size_t) T * c.ff_exp * 4);
                d_pu = (float*) galloc((size_t) T * c.ff_exp * 4);
                d_pa = (float*) galloc((size_t) T * c.ff_exp * 4);
                d_pd = (float*) galloc((size_t) T * dim * 4);
                d_pYm = (float*) galloc((size_t) T * dim * 4);
                ok = d_pX && d_pY && d_pg && d_pu && d_pa && d_pd && d_pYm;
            }
            if (!ok) {
                // Give the VRAM back before retrying smaller: galloc() counts every byte it hands
                // out, and without the release the retry would only see a card that is fuller still.
                for (float** pp : {&d_pX, &d_pY, &d_pg, &d_pu, &d_pa, &d_pd, &d_pYm}) {
                    gfree(*pp);
                    *pp = nullptr;
                }
                pf_cap /= 2;
                continue;
            }
            // host arena
            const int64_t n = hc * dim;
            size_t at = 0;
            auto take = [&](size_t k) { size_t base = at; at += k; return base; };
            const size_t b_h = take((size_t) T * n), b_hn = take((size_t) T * n);
            const size_t b_yv = take((size_t) T * dim), b_nrm = take((size_t) T * dim);
            const size_t b_e = take((size_t) T * dim), b_x = take((size_t) T * dim), b_xn = take((size_t) T * dim);
            const size_t b_acc = take((size_t) T * dim);
            const size_t b_qa = take((size_t) T * c.q_lora), b_qr = take((size_t) T * c.q_lora);
            const size_t b_q = take((size_t) T * nh_hd), b_o = take((size_t) T * nh_hd);
            const size_t b_kv = take((size_t) T * hd), b_kvn = take((size_t) T * hd);
            const size_t b_t = take((size_t) T * c.o_groups * c.o_lora);
            const size_t b_lg = take((size_t) T * c.n_expert);
            const size_t b_mix = take((size_t) T * pf_mix), b_comb = take((size_t) T * pf_mix);
            const size_t b_pre = take((size_t) T * hc), b_post = take((size_t) T * hc);
            const size_t b_sg = take((size_t) T * c.ff_exp), b_su = take((size_t) T * c.ff_exp);
            const size_t b_sa = take((size_t) T * c.ff_exp), b_sy = take((size_t) T * dim);
            const size_t b_mg = take((size_t) T * c.ff_exp), b_mu = take((size_t) T * c.ff_exp);
            const size_t b_ma = take((size_t) T * c.ff_exp), b_md = take((size_t) T * dim);
            pf.assign(at, 0.f);
            auto P = [&](size_t b) { return pf.data() + b; };
            h_h = P(b_h); h_hn = P(b_hn); h_yv = P(b_yv); h_nrm = P(b_nrm);
            h_e = P(b_e); h_x = P(b_x); h_xn = P(b_xn); h_acc = P(b_acc);
            h_qa = P(b_qa); h_qr = P(b_qr); h_q = P(b_q); h_kv = P(b_kv); h_kvn = P(b_kvn);
            h_o = P(b_o); h_t = P(b_t); h_lg = P(b_lg);
            h_mixes = P(b_mix); h_comb = P(b_comb); h_pre = P(b_pre); h_post = P(b_post);
            h_sg = P(b_sg); h_su = P(b_su); h_sa = P(b_sa); h_sy = P(b_sy);
            h_mg = P(b_mg); h_mu = P(b_mu); h_ma = P(b_ma); h_md = P(b_md);
            pf_host_bytes = at * 4;
            pf_dev_bytes = (size_t) (pf_xcap + pf_ycap + (size_t) T * c.ff_exp * 3 + (size_t) T * dim * 2) * 4;
            // index / routing scratch. The attention index list is sized for the widest list a token
            // can produce in ANY layer (window + compressed entries), and pf_max_list is the stride.
            pf_i.assign((size_t) T * (c.n_expert_used * 2 + pf_max_list + 1), 0);
            h_tok = pf_i.data();
            h_sel = h_tok + (size_t) T * c.n_expert_used;
            h_attn_idx = h_sel + (size_t) T * c.n_expert_used;
            h_tk = h_attn_idx + (size_t) T * pf_max_list;
            pf_ic.assign((size_t) T + (size_t) (c.n_expert + 1) * 3 + (size_t) c.n_expert, 0);
            h_attn_n = pf_ic.data();
            h_cnt = h_attn_n + T;
            h_off = h_cnt + (c.n_expert + 1);
            h_at = h_off + (c.n_expert + 1);
            h_eid = h_at + (c.n_expert + 1);
            pf_w.assign((size_t) T * c.n_expert_used * 2, 0.f);
            h_w = pf_w.data();
            h_wt = h_w + (size_t) T * c.n_expert_used;
            pf_b.assign((size_t) c.n_expert, gpu::ExpBatch{});
            chunk_routing.assign((size_t) T * (size_t) c.n_layer * c.n_expert_used, -1);
            return;
        }
        pf_cap = 0;
        // Even with batching off, forward_chunk() may be asked for a whole prompt and the sequential
        // fallback still has somewhere to put the per-token routing trace.
        chunk_routing.assign((size_t) std::max(1, o.ctx) * (size_t) c.n_layer * c.n_expert_used, -1);
    }
    /// Routing of every token of the last forward_chunk(): [t][layer*K + k]. Empty for a sequential
    /// forward(). dsv4_run --dump-routing writes one row per token from this.
    std::vector<int> chunk_routing;
    int chunk_routing_n = 0;
    size_t last_routing_size_ = 0;   // n_layer * n_expert_used, set with last_routing_p
    bool have_profile = false;
    uint64_t cache_clock = 0;                   // LRU clock of the MISS arenas
    std::vector<std::pair<int, int>> missed_now;      // (layer, expert) missed during the current token
    int cur_pos = 0;

    ~Impl() {
        for (void* p : allocs) gpu::release(p);
        for (size_t i = 0; i < maps.size(); ++i) if (maps[i]) munmap((void*) maps[i], map_len[i]);
    }
    // vram_used counts every byte handed out by the device layer, so the residency plan can subtract the
    // Sizes of the allocations in allocs[], index for index: the prefill arena has to give its VRAM
    // back when it is resized down, and guessing at the size would corrupt the plan.
    std::vector<size_t> alloc_sz;
    // vram_used counts every byte handed out by the device layer, so the residency plan can subtract the
    // weights and scratch that are ALREADY on the card instead of guessing at them. On the CUDA backend
    // cudaMemGetInfo reports what the driver has left, not what this model still needs: without this the
    // planner hands out more expert slots than fit and the dpool allocation fails.
    void* galloc(size_t n) { void* p = gpu::alloc(n); if (p) { allocs.push_back(p); alloc_sz.push_back(n); vram_used += n; } return p; }
    void gfree(void* p) {
        if (!p) return;
        for (size_t k = 0; k < allocs.size(); ++k)
            if (allocs[k] == p) { vram_used -= alloc_sz[k]; allocs.erase(allocs.begin() + (long) k); alloc_sz.erase(alloc_sz.begin() + (long) k); break; }
        gpu::release(p);
    }
    uint64_t vram_used = 0, vram_scratch = 0;

    Tn tn(const std::string& name, bool required, std::string& err) {
        Tn t;
        auto it = by_name.find(name);
        if (it == by_name.end()) { if (required && err.empty()) err = "missing tensor: " + name; return t; }
        t.g = it->second; t.h = maps[(size_t) t.g->shard] + t.g->abs_offset; t.type = t.g->type;
        t.in = (int64_t) t.g->dims[0]; t.out = t.g->dims.size() > 1 ? (int64_t) t.g->dims[1] : 1; t.ne = t.g->dims.size() > 2 ? (int64_t) t.g->dims[2] : 1;
        t.rb = row_bytes(t.type, t.in);
        return t;
    }
    bool need_supported(const Tn& t, std::string& err) {
        if (!t.g) return true;
        if (t.type == T_I32) return true;  // the tid2eid hash table is read raw
        if (!dequant_supported(t.type)) {
            err = "tensor " + t.g->name + " has ggml type " + ggml_type_str(t.type) + " which is not supported" +
                  (t.g->known_type ? "" : " (unknown ggml id: this build knows ids 0..30 and 39)");
            return false;
        }
        if (!row_bytes(t.type, t.in)) {
            err = "tensor " + t.g->name + ": " + std::to_string(t.in) + " elements per row is not a whole number of " +
                  ggml_type_str(t.type) + " blocks";
            return false;
        }
        return true;
    }
    void upload(Tn& t) {
        if (!o.gpu || !t.g || t.d) return;
        // Uploading weights that no device kernel can decode would only burn VRAM: leave them on the host.
        if (!gpu::type_supported(t.type)) { t.no_gpu = true; return; }
        t.d = (uint8_t*) galloc(t.rb * (size_t) t.out);
        if (!t.d) { std::fprintf(stderr, "warning: VRAM allocation failed for %s: staying on the CPU\n", t.g->name.c_str()); return; }
        gpu::h2d(t.d, t.h, t.rb * (size_t) t.out);
    }
    void mv(const Tn& w, const float* x, float* y, int64_t row0 = 0, int64_t nrows = -1) {
        if (nrows < 0) nrows = w.out;
        if (o.gpu && w.d) {
            gpu::h2d(d_x, x, (size_t) w.in * 4);
            // matvec returns false when this backend has no kernel for the type: fall back to the host
            // rather than reading back a d_y that was never written.
            if (gpu::matvec(w.type, w.d + (size_t) row0 * w.rb, nrows, w.in, d_x, d_y)) {
                gpu::d2h(y, d_y, (size_t) nrows * 4);
                return;
            }
            if (!warned_host_fallback) {
                warned_host_fallback = true;
                std::fprintf(stderr, "warning: no device kernel for ggml type %s (%s): those matvecs run on the host\n",
                             ggml_type_str(w.type).c_str(), w.g ? w.g->name.c_str() : "?");
            }
        }
        cpu_mv(w.type, w.h + (size_t) row0 * w.rb, nrows, w.in, x, y);
    }
    bool warned_host_fallback = false;

    // ------------------------------------------------------------ prefill: batched matmul
    // Y[t*y_stride + r] = W_row_r . X[t*x_stride ..], for t in [0,T). Same math as mv(), over T tokens:
    // the weight row is decoded once and dotted T times, which is the whole point of a batched prefill.
    // X and Y are HOST buffers here; the device path copies the activation block up and back, which is
    // kilobytes against the megabytes of weights a matvec() over the same tensor would move per token.
    bool mvb(const Tn& w, const float* X, int64_t x_stride, int T, float* Y, int64_t y_stride,
             int64_t row0 = 0, int64_t nrows = -1) {
        if (T <= 0) return true;
        if (nrows < 0) nrows = w.out;
        if (x_stride <= 0) x_stride = w.in;
        if (y_stride <= 0) y_stride = nrows;
        if (o.gpu && w.d && pf_dev(x_stride, T, w.in, nrows, y_stride)) {
            const float* dX = pf_up(X, x_stride, T, w.in);
            // dX must really be the device block: pf_up falls back to the host pointer when the block
            // does not fit, and handing that to a kernel would read host addresses as VRAM.
            if (dX && dX != X && gpu::matmul(w.type, w.d + (size_t) row0 * w.rb, nrows, w.in, dX, T, nullptr, d_pY,
                                  x_stride, y_stride)) {
                // The kernel wrote row t at d_pY + t*y_stride, so the meaningful floats are exactly
                // [0,nrows) of each row. Copying one flat span instead would also move the gaps
                // between rows, which hold whatever an earlier matmul left there - and when
                // y_stride > nrows those gaps are the neighbouring group's output.
                for (int t = 0; t < T; ++t)
                    gpu::d2h(Y + (size_t) t * y_stride, d_pY + (size_t) t * y_stride, (size_t) nrows * 4);
                return true;
            }
            if (!warned_host_fallback) {
                warned_host_fallback = true;
                std::fprintf(stderr, "warning: no device kernel for ggml type %s (%s): those matmuls run on the host\n",
                             ggml_type_str(w.type).c_str(), w.g ? w.g->name.c_str() : "?");
            }
        }
        const size_t rb = row_bytes(w.type, w.in);
        if (!rb) return false;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < nrows; ++r)
            row_dots(w.type, w.h + (size_t) (row0 + r) * rb, X, x_stride, nullptr, T, w.in, Y + (size_t) r, y_stride);
        return true;
    }
    // mvb() never gathers: the gathered expert path goes straight through row_dots()/experts_batch().

    /// Row of the window ring that holds position `pos` as seen by a query at position `pos_q`.
    /// A chunk writes the ring ahead of the queries that come later in the same chunk; when the slot
    /// already holds a FUTURE token, the pre-chunk snapshot is the row the token-per-token path read.
    int win_row(const Layer& y, int slot, int pos_q) const {
        if (y.snap_base <= 0) return slot;
        return y.ring_pos[(size_t) slot] <= pos_q ? slot : y.snap_base + slot;
    }

    // ------------------------------------------------------------ expert sources / residency
    static ExpSrc packed(const Layer& l, const uint8_t* p) {
        ExpSrc s; s.packed = p; s.g = p; s.u = p + l.bg; s.d = p + l.bg + l.bu; return s;
    }
    ExpSrc src(const Layer& l, int e) const {
        if (l.ram_idx[(size_t) e] >= 0) return packed(l, l.ram.data() + (size_t) l.ram_idx[(size_t) e] * l.bpe);
        ExpSrc s{};
        s.g = l.eg.h + (size_t) e * l.bg; s.u = l.eu.h + (size_t) e * l.bu; s.d = l.ed.h + (size_t) e * l.bd;
        return s;
    }
    /// Weights of a MISS expert: the profile's RAM copy, else the LRU arena (copied out of the mmap once),
    /// else the mmap itself when the arena is disabled.
    ExpSrc src_miss(Layer& y, int e) {
        if (y.ncache <= 0 || y.ram_idx[(size_t) e] >= 0) return src(y, e);
        const int have = y.cslot_of[(size_t) e];
        if (have >= 0) { y.cstamp[(size_t) have] = ++cache_clock; ++st->cache_hits; return packed(y, y.cpool.data() + (size_t) have * y.bpe); }
        int slot = -1;
        for (int s = 0; s < y.ncache; ++s) if (y.cocc[(size_t) s] < 0) { slot = s; break; }
        if (slot < 0) {  // full: evict the least recently used
            uint64_t best = UINT64_MAX;
            for (int s = 0; s < y.ncache; ++s) if (y.cstamp[(size_t) s] < best) { best = y.cstamp[(size_t) s]; slot = s; }
            if (y.cocc[(size_t) slot] >= 0) y.cslot_of[(size_t) y.cocc[(size_t) slot]] = -1;
            ++st->cache_evictions;
        }
        uint8_t* dst = y.cpool.data() + (size_t) slot * y.bpe;
        std::memcpy(dst, y.eg.h + (size_t) e * y.bg, y.bg);
        std::memcpy(dst + y.bg, y.eu.h + (size_t) e * y.bu, y.bu);
        std::memcpy(dst + y.bg + y.bu, y.ed.h + (size_t) e * y.bd, y.bd);
        y.cocc[(size_t) slot] = e; y.cslot_of[(size_t) e] = slot; y.cstamp[(size_t) slot] = ++cache_clock;
        ++st->cache_loads; st->cache_bytes += y.bpe;
        return packed(y, dst);
    }
    void upload_slot(Layer& l, int slot, int e) {
        ExpSrc s = src(l, e);
        uint8_t* dst = l.dpool + (size_t) slot * l.bpe;
        if (s.packed) gpu::h2d(dst, s.packed, l.bpe);
        else { gpu::h2d(dst, s.g, l.bg); gpu::h2d(dst + l.bg, s.u, l.bu); gpu::h2d(dst + l.bg + l.bu, s.d, l.bd); }
        st->h2d_bytes += l.bpe;
        if (o.verify_slots) {
            std::vector<uint8_t> back(l.bpe);
            gpu::d2h(back.data(), dst, l.bpe);
            if (std::memcmp(back.data(), s.g, l.bg) || std::memcmp(back.data() + l.bg, s.u, l.bu) || std::memcmp(back.data() + l.bg + l.bu, s.d, l.bd)) {
                std::fprintf(stderr, "FATAL: verify_slot failed (slot %d, expert %d)\n", slot, e); std::abort();
            }
        }
        if (l.occ[(size_t) slot] >= 0) l.slot_of[(size_t) l.occ[(size_t) slot]] = -1;
        l.occ[(size_t) slot] = e; l.slot_of[(size_t) e] = slot;
    }

    // ------------------------------------------------------------ profile
    bool profile_load(const std::string& path, std::string& err) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open profile " + path; return false; }
        char magic[8]; uint32_t ver, nl, ne; uint64_t mh;
        bool ok = std::fread(magic, 1, 8, f) == 8 && std::fread(&ver, 4, 1, f) == 1 && std::fread(&nl, 4, 1, f) == 1 && std::fread(&ne, 4, 1, f) == 1 && std::fread(&mh, 8, 1, f) == 1;
        if (!ok || std::memcmp(magic, "DSV4PROF", 8) != 0 || ver != 1) { std::fclose(f); err = "not a DSV4 expert profile (version 1)"; return false; }
        if ((int) nl != c.n_layer || (int) ne != c.n_expert) { std::fclose(f); err = "profile layer/expert counts do not match the model"; return false; }
        if (mh != model_hash) { std::fclose(f); err = "profile was built for a different model (hash mismatch)"; return false; }
        std::vector<uint64_t> cnt((size_t) nl * ne);
        ok = std::fread(cnt.data(), 8, cnt.size(), f) == cnt.size();
        std::fclose(f);
        if (!ok) { err = "truncated profile"; return false; }
        for (int l = 0; l < c.n_layer; ++l) for (int e = 0; e < c.n_expert; ++e) L[(size_t) l].freq[(size_t) e] = cnt[(size_t) l * ne + e];
        return true;
    }

    Stats* st = nullptr;
    // ------------------------------------------------------------ load
    bool load(const std::vector<std::string>& shards, const RunOpts& opts, std::string& err) {
        const auto T0 = Clock::now();
        auto ph = [&](const char* what) { std::fprintf(stderr, "[load %6.1fs] %s\n", secs(T0, Clock::now()), what); };
        o = opts;
        if (!gguf_read_header(shards, h, err)) return false;
        if (h.tensors.empty()) { err = "no tensors found: pass the first shard of a multi-file GGUF (…-00001-of-0000N.gguf) so all shards are loaded"; return false; }
        if (!config_from_gguf(h, c, err)) return false;
        if (!inventory_from_gguf(h, c, inv, err)) return false;
        if (c.gating_func != 1 && c.gating_func != 2 && c.gating_func != 4) { err = "unsupported expert_gating_func " + std::to_string(c.gating_func); return false; }
        if (c.idx_key_len & (c.idx_key_len - 1)) { err = "indexer key length must be a power of two (Hadamard)"; return false; }
#ifdef _OPENMP
        // Every matvec here is memory-bound and embarrassingly parallel over the output rows: leaving this on
        // one core costs roughly 20x on a 28-thread machine. 0 means "use the whole machine".
        if (o.threads <= 0) {
            int hw = (int) std::thread::hardware_concurrency();
            o.threads = hw > 0 ? hw : 8;
        }
        omp_set_num_threads(o.threads);
#else
        if (o.verbose) std::fprintf(stderr, "warning: this build has NO OpenMP: every matvec runs on a single thread (expect ~20x slower)\n");
#endif
        for (const GgufTensor& t : h.tensors) by_name[t.name] = &t;
        for (const std::string& f : h.files) {
            int fd = ::open(f.c_str(), O_RDONLY);
            struct stat sb;
            if (fd < 0 || fstat(fd, &sb) != 0) { err = "cannot open " + f; return false; }
            void* m = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            if (m == MAP_FAILED) { err = "mmap failed for " + f; return false; }
            // MADV_RANDOM switches the kernel's readahead OFF. We never touch a single byte at a time: an
            // expert is three contiguous runs (gate|up|down, ~5.5 MiB) and a matvec walks whole rows, so
            // demand paging at page granularity is already efficient.
            //
            // WILLNEED was the wrong advice here: it asks the kernel to prefetch the WHOLE shard, which on
            // an 80 GiB model in front of a machine with less free RAM than that evicts the pages the very
            // next forward pass needs. The page cache then thrashes on every token and the miss path reads
            // the disk again. RANDOM keeps the cache for what is actually touched.
            madvise(m, (size_t) sb.st_size, MADV_RANDOM);
            maps.push_back((const uint8_t*) m); map_len.push_back((size_t) sb.st_size);
        }
        // Every tensor must lie inside its shard: a truncated or mismatched file would otherwise
        // turn into a segfault deep inside the dequantiser.
        for (const GgufTensor& t : h.tensors) {
            if (t.shard < 0 || (size_t) t.shard >= maps.size()) { err = "tensor " + t.name + " refers to a missing shard"; return false; }
            const uint64_t need = t.nbytes ? t.abs_offset + t.nbytes : t.abs_offset;
            if (need > map_len[(size_t) t.shard]) {
                err = "tensor " + t.name + " (" + ggml_type_str(t.type) + ", " + std::to_string(t.nbytes) +
                      " bytes at offset " + std::to_string(t.abs_offset) + ") is outside " + h.files[(size_t) t.shard] +
                      " (" + std::to_string(map_len[(size_t) t.shard]) + " bytes): the shard is truncated or the GGUF is inconsistent";
                return false;
            }
        }
        ph("mmap + header checks done");
        model_hash = 1469598103934665603ULL;
        for (uint64_t v : {(uint64_t) c.n_layer, (uint64_t) c.n_expert, (uint64_t) c.ff_exp, (uint64_t) c.n_embd, (uint64_t) h.tensors.size()}) model_hash = fnv(model_hash, v);
        for (uint64_t b : inv.bytes_per_expert) model_hash = fnv(model_hash, b);
        if (const GgufValue* t = h.find("tokenizer.ggml.tokens")) toks = t->strs;

        dim = c.n_embd; hd = c.key_len; rd = c.rope_dim; win = c.sliding_window; hc = c.hc_count;
        nrun = (o.max_layers > 0 && o.max_layers < c.n_layer) ? o.max_layers : c.n_layer;
        if (o.ctx < 8) o.ctx = 8;

        if (o.gpu) {
            if (!gpu::init(err)) return false;
            const uint64_t before_scratch = vram_used;
            d_x = (float*) galloc(16384 * 4); d_y = (float*) galloc((size_t) std::max<int64_t>(c.vocab, 16384) * 4 + 4096);
            d_g = (float*) galloc((size_t) gpu::kMaxHit * c.ff_exp * 4); d_u = (float*) galloc((size_t) gpu::kMaxHit * c.ff_exp * 4);
            d_a = (float*) galloc((size_t) gpu::kMaxHit * c.ff_exp * 4); d_yh = (float*) galloc((size_t) dim * 4);
            if (!d_x || !d_y || !d_g || !d_u || !d_a || !d_yh) { err = "cannot allocate GPU scratch"; return false; }
            vram_scratch = vram_used - before_scratch;
        }
        // ---- prefill scratch: sized from --prefill-chunk, shrunk to what fits below ----
        pf_mix = (2 + hc) * hc;
        pf_gin = c.n_head * hd / std::max(1, c.o_groups);
        {
            int want = o.prefill_chunk;
            if (want > o.ctx) want = o.ctx;
            if (want > gpu::kMaxSel) want = gpu::kMaxSel;
            // The window ring is written at pos % win, so a chunk longer than the ring would need
            // more than one snapshot to stay exact. Cap it at the window.
            if (win > 0 && want > win) want = win;
            pf_cap = pf_pick(want);
        }
        pf_alloc();
        if (o.verbose) {
            if (pf_cap > 0)
                std::fprintf(stderr, "prefill: chunks of %d tokens | host arena %.1f MiB | device %.1f MiB\n",
                             pf_cap, pf_host_bytes / 1048576.0, pf_dev_bytes / 1048576.0);
            else
                std::fprintf(stderr, "prefill: batching OFF (--prefill-chunk 0): one token per pass\n");
        }
        embd = tn("token_embd.weight", true, err); outw = tn("output.weight", true, err); outnorm = tn("output_norm.weight", true, err);
        out_hc_fn = tn("output_hc_fn.weight", true, err); out_hc_base = tn("output_hc_base.weight", true, err); out_hc_scale = tn("output_hc_scale.weight", true, err);
        if (!err.empty()) return false;
        if (!need_supported(embd, err) || !need_supported(outw, err)) return false;

        yarn_table(rd, o.ctx, (int) c.yarn_orig_ctx, c.rope_base_compress, c.yarn_factor, c.yarn_beta_fast, c.yarn_beta_slow, cos_y, sin_y);
        yarn_table(rd, o.ctx, 0, c.rope_base, c.yarn_factor, c.yarn_beta_fast, c.yarn_beta_slow, cos_p, sin_p);

        L.assign((size_t) c.n_layer, Layer());
        auto init_comp = [&](Comp& cp, int ratio, int d, bool rotate, const Tn& wkv, const Tn& wg, const Tn& ape, const Tn& norm, float* cache) {
            cp.ratio = ratio; cp.d = d; cp.overlap = ratio == 4; cp.coff = 1 + (cp.overlap ? 1 : 0); cp.rotate = rotate;
            cp.wkv = wkv; cp.wg = wg; cp.ape = ape; cp.norm = norm; cp.cache = cache;
            return wkv.out == (int64_t) cp.coff * d && wg.out == (int64_t) cp.coff * d;
        };
        for (int l = 0; l < nrun; ++l) {
            Layer& y = L[(size_t) l];
            auto N = [&](const char* s, bool req = true) { return tn("blk." + std::to_string(l) + "." + s, req, err); };
            y.ratio = c.compress_ratios[(size_t) l];
            y.attn_norm = N("attn_norm.weight"); y.q_a = N("attn_q_a.weight"); y.q_a_norm = N("attn_q_a_norm.weight"); y.q_b = N("attn_q_b.weight");
            y.kv = N("attn_kv.weight"); y.kv_norm = N("attn_kv_a_norm.weight"); y.o_a = N("attn_output_a.weight"); y.o_b = N("attn_output_b.weight"); y.sinks = N("attn_sinks.weight");
            y.ffn_norm = N("ffn_norm.weight"); y.gate_inp = N("ffn_gate_inp.weight");
            if (l < c.n_hash_layers) y.tid2eid = N("ffn_gate_tid2eid.weight"); else y.exp_bias = N("exp_probs_b.bias", false);
            y.sh_gate = N("ffn_gate_shexp.weight"); y.sh_up = N("ffn_up_shexp.weight"); y.sh_down = N("ffn_down_shexp.weight");
            y.hc_attn_fn = N("hc_attn_fn.weight"); y.hc_attn_base = N("hc_attn_base.weight"); y.hc_attn_scale = N("hc_attn_scale.weight");
            y.hc_ffn_fn = N("hc_ffn_fn.weight"); y.hc_ffn_base = N("hc_ffn_base.weight"); y.hc_ffn_scale = N("hc_ffn_scale.weight");
            y.eg = N("ffn_gate_exps.weight"); y.eu = N("ffn_up_exps.weight"); y.ed = N("ffn_down_exps.weight");
            if (!err.empty()) return false;
            y.ncomp_max = y.ratio ? o.ctx / y.ratio + 1 : 0;
            // [ring win][compressed ncomp_max][window snapshot win]: the last block only exists when
            // prefill batching is on, and holds the ring as it looked before the current chunk.
            y.max_idx = win + y.ncomp_max;
            y.kvbuf.assign((size_t) (win + y.ncomp_max + (pf_cap > 0 ? win : 0)) * hd, 0.f);
            y.snap_base = pf_cap > 0 ? win + y.ncomp_max : 0;
            y.ring_pos.assign((size_t) win, -1);
            if (y.ratio) {
                if (!init_comp(y.ac, y.ratio, hd, false, N("attn_compressor_kv.weight"), N("attn_compressor_gate.weight"), N("attn_compressor_ape.weight"), N("attn_compressor_norm.weight"), y.kvbuf.data() + (size_t) win * hd) && err.empty())
                    err = "compressor tensor shapes do not match ratio " + std::to_string(y.ratio) + " at layer " + std::to_string(l);
                y.has_idx = by_name.count("blk." + std::to_string(l) + ".indexer.attn_q_b.weight") > 0;
                if (y.has_idx) {
                    y.i_q_b = N("indexer.attn_q_b.weight"); y.i_proj = N("indexer.proj.weight");
                    y.ic.own.assign((size_t) (o.ctx / y.ratio + 1) * c.idx_key_len, 0.f);
                    if (!init_comp(y.ic, y.ratio, c.idx_key_len, true, N("indexer_compressor_kv.weight"), N("indexer_compressor_gate.weight"), N("indexer_compressor_ape.weight"), N("indexer_compressor_norm.weight"), y.ic.own.data()) && err.empty())
                        err = "indexer compressor shapes mismatch at layer " + std::to_string(l);
                }
            }
            if (!err.empty()) return false;
            for (Tn* t : {&y.q_a, &y.q_b, &y.kv, &y.o_a, &y.o_b, &y.gate_inp, &y.sh_gate, &y.sh_up, &y.sh_down, &y.ac.wkv, &y.ac.wg, &y.i_q_b, &y.ic.wkv, &y.ic.wg, &y.eg, &y.eu, &y.ed})
                if (!need_supported(*t, err)) return false;
            y.bg = y.eg.rb * (size_t) y.eg.out; y.bu = y.eu.rb * (size_t) y.eu.out; y.bd = y.ed.rb * (size_t) y.ed.out; y.bpe = y.bg + y.bu + y.bd;
            if (y.eg.ne != c.n_expert) { err = "expert tensor expert-count mismatch at layer " + std::to_string(l); return false; }
            y.slot_of.assign((size_t) c.n_expert, -1); y.ram_idx.assign((size_t) c.n_expert, -1); y.freq.assign((size_t) c.n_expert, 0);
            // A VRAM slot only pays off if experts_hit() can run this layer's three matrices on the device.
            y.gpu_experts = o.gpu && gpu::experts_supported(y.eg.type, y.eu.type, y.ed.type);
            // hc_*_fn is deliberately NOT uploaded: hc_half() runs hc_pre() per token, which reads
            // the host weights and keeps the exact (double) rounding of the sequential path. Uploading
            // 130 MiB of them would only take VRAM away from the experts that do go to the device.
            for (Tn* t : {&y.q_a, &y.q_b, &y.kv, &y.o_a, &y.o_b, &y.gate_inp, &y.sh_gate, &y.sh_up, &y.sh_down, &y.ac.wkv, &y.ac.wg, &y.i_q_b, &y.ic.wkv, &y.ic.wg}) upload(*t);
        }
        upload(outw);
        ph("non-expert weights uploaded to GPU");
        // ---- staging pool for MISS experts.
        // A MISS used to be decoded on the host: an IQ1_M expert is ~5.5 MiB of packed weights that
        // row_dot() expands element by element, which at 6 experts x 43 layers is several billion
        // decodes per token - more than the rest of the forward pass combined. With a device pool the
        // same expert is copied up and decoded by the kernels that already exist, so the expert cost
        // drops to a PCIe transfer that overlaps the other launches. Sized for one layer's worth of
        // kMaxHit experts: layers are evaluated one at a time, so the pool is reused, not duplicated.
        if (o.gpu) {
            size_t maxb = 0;
            for (int l = 0; l < nrun; ++l) maxb = std::max(maxb, L[(size_t) l].bpe);
            if (maxb > 0) {
                const size_t need = (size_t) gpu::kMaxHit * maxb;
                d_stage = (uint8_t*) galloc(need);
                if (d_stage && !gpu::staging(d_stage, need)) d_stage = nullptr;
                if (!d_stage && o.verbose)
                    std::fprintf(stderr, "note: no staging pool for MISS experts (%zu bytes): they stay on the host\n", need);
            }
        }
        reset();

        // ---- expert residency plan: VRAM slots per layer from --expert-vram-pct / --expert-cache
        std::string perr;
        if (!o.profile_in.empty()) {
            if (profile_load(o.profile_in, perr)) have_profile = true;
            else std::fprintf(stderr, "warning: incompatible expert profile (%s): starting without profile\n", perr.c_str());
        }
        if (o.gpu) {
            size_t fb = 0, tb = 0; gpu::mem_info(&fb, &tb);
            MemPlanInput in;
            in.vram_total = o.vram_total_mib >= 0 ? o.vram_total_mib << 20 : (int64_t) tb;
            in.vram_free = o.vram_free_mib >= 0 ? o.vram_free_mib << 20 : (int64_t) fb;
            in.reserve = o.reserve_mib << 20; in.n_expert = c.n_expert; in.pct = o.vram_pct; in.abs_slots = o.abs_slots;
            // No fixed costs here, and that is not an oversight: BOTH backends report free VRAM already net
            // of what this model has put on the card (the emulated one subtracts its own pool, cudaMemGetInfo
            // subtracts every cudaMalloc including ours and other processes'). The KV cache, the activations
            // and the prefill buffers are host memory (std::vector<float>), so they cost no VRAM at all.
            // Counting them again is what made the plan die with "fixed costs exceed the usable VRAM" on a
            // 24 GiB card that had plenty of room for experts. dsv4_plan still estimates them, because there
            // it is a what-if tool with no device in front of it.
            in.weights = 0; in.kv = 0; in.act = 0; in.prefill = 0; in.mtp = 0; in.scratch = 0;
            in.bytes_per_expert.assign((size_t) c.n_layer, 0);
            // A layer the device cannot evaluate gets 0 bytes here, so the planner gives it no slots: VRAM
            // would be spent on experts that always end up computed on the host anyway.
            for (int l = 0; l < nrun; ++l) in.bytes_per_expert[(size_t) l] = L[(size_t) l].gpu_experts ? L[(size_t) l].bpe : 0;
            in.ram_expert_bytes = inv.total_expert_bytes;
            MemPlan pl = plan_expert_memory(in);
            if (!pl.ok) { err = pl.err; return false; }
            if (o.verbose) std::fprintf(stderr, "%s", plan_report(in, pl).c_str());
            for (int l = 0; l < nrun; ++l) {
                Layer& y = L[(size_t) l];
                y.slots = pl.slots_layer[(size_t) l];
                if (y.slots <= 0) { y.slots = 0; continue; }
                y.dpool = (uint8_t*) galloc((size_t) y.slots * y.bpe);
                if (!y.dpool) { err = "VRAM allocation for the expert cache failed at layer " + std::to_string(l) + ": lower --expert-vram-pct or raise --expert-vram-reserve-mib"; return false; }
                y.occ.assign((size_t) y.slots, -1);
            }
        }
        // ---- RAM: only experts the profile marks as used are copied into RAM (the rest stay on disk via mmap)
        // uint64_t ram_bytes = 0; int ram_n = 0;
        // for (int l = 0; l < nrun; ++l) {
        //     Layer& y = L[(size_t) l];
        //     int n = 0;
        //     for (int e = 0; e < c.n_expert; ++e) if (o.ram_all || (have_profile && y.freq[(size_t) e] > 0)) y.ram_idx[(size_t) e] = n++;
        //     y.ram.resize((size_t) n * y.bpe);
        //     for (int e = 0; e < c.n_expert; ++e) if (y.ram_idx[(size_t) e] >= 0) {
        //         uint8_t* dst = y.ram.data() + (size_t) y.ram_idx[(size_t) e] * y.bpe;
        //         std::memcpy(dst, y.eg.h + (size_t) e * y.bg, y.bg); std::memcpy(dst + y.bg, y.eu.h + (size_t) e * y.bu, y.bu); std::memcpy(dst + y.bg + y.bu, y.ed.h + (size_t) e * y.bd, y.bd);
        //     }
        //     ram_bytes += y.ram.size(); ram_n += n;
        // }
        //        // ---- RAM: only experts the profile marks as used are copied into RAM (the rest stay on disk via mmap)
        // uint64_t ram_bytes = 0; int ram_n = 0;
        // // The mapping is MADV_RANDOM (no readahead), so a plain memcpy reads 4 KiB per page fault.
        // // WILLNEED asks the kernel for big sequential reads instead; it works despite MADV_RANDOM.
        // auto willneed = [](const uint8_t* p, size_t len) {
        //     const uintptr_t a = (uintptr_t) p & ~(uintptr_t) 4095;
        //     madvise((void*) a, len + ((uintptr_t) p - a), MADV_WILLNEED);
        // };
        // for (int l = 0; l < nrun; ++l) {
        //     Layer& y = L[(size_t) l];
        //     int n = 0;
        //     for (int e = 0; e < c.n_expert; ++e) if (o.ram_all || (have_profile && y.freq[(size_t) e] > 0)) y.ram_idx[(size_t) e] = n++;
        //     y.ram.resize((size_t) n * y.bpe);
        //     if (n == c.n_expert) {   // whole layer wanted (--expert-ram all): prefetch it in one go
        //         willneed(y.eg.h, (size_t) c.n_expert * y.bg);
        //         willneed(y.eu.h, (size_t) c.n_expert * y.bu);
        //         willneed(y.ed.h, (size_t) c.n_expert * y.bd);
        //     }
        //     #pragma omp parallel for schedule(static)
        //     for (int e = 0; e < c.n_expert; ++e) if (y.ram_idx[(size_t) e] >= 0) {
        //         uint8_t* dst = y.ram.data() + (size_t) y.ram_idx[(size_t) e] * y.bpe;
        //         std::memcpy(dst, y.eg.h + (size_t) e * y.bg, y.bg);
        //         std::memcpy(dst + y.bg, y.eu.h + (size_t) e * y.bu, y.bu);
        //         std::memcpy(dst + y.bg + y.bu, y.ed.h + (size_t) e * y.bd, y.bd);
        //     }
        //     ram_bytes += y.ram.size(); ram_n += n;
        // }

        // ---- RAM: only experts the profile marks as used are copied into RAM (the rest stay on disk via mmap)
        // Copy with pread() from every OpenMP thread instead of memcpy out of the mmap. The mapping is MADV_RANDOM,
        // so a memcpy page-faults 4 KiB at a time and a cold start crawled at 70-300 MB/s. An expert is three
        // contiguous runs, so pread() issues MiB-sized reads and keeps the NVMe queue full.
        uint64_t ram_bytes = 0; int ram_n = 0;
        std::vector<int> sfd;
        for (const std::string& f : h.files) sfd.push_back(::open(f.c_str(), O_RDONLY));
        auto rd = [&](const Tn& t, size_t off, uint8_t* dst, size_t len) {
            const int fd = sfd[(size_t) t.g->shard];
            size_t done = 0;
            while (done < len) {
                const ssize_t n = fd >= 0 ? ::pread(fd, dst + done, len - done, (off_t) (t.g->abs_offset + off + done)) : -1;
                if (n <= 0) { std::memcpy(dst + done, t.h + off + done, len - done); break; }   // fallback: the old mmap path
                done += (size_t) n;
            }
        };
        for (int l = 0; l < nrun; ++l) {
            Layer& y = L[(size_t) l];
            int n = 0;
            for (int e = 0; e < c.n_expert; ++e) if (o.ram_all || (have_profile && y.freq[(size_t) e] > 0)) y.ram_idx[(size_t) e] = n++;
            y.ram.resize((size_t) n * y.bpe);
            #pragma omp parallel for schedule(dynamic, 1)
            for (int e = 0; e < c.n_expert; ++e) if (y.ram_idx[(size_t) e] >= 0) {
                uint8_t* dst = y.ram.data() + (size_t) y.ram_idx[(size_t) e] * y.bpe;
                rd(y.eg, (size_t) e * y.bg, dst, y.bg);
                rd(y.eu, (size_t) e * y.bu, dst + y.bg, y.bu);
                rd(y.ed, (size_t) e * y.bd, dst + y.bg + y.bu, y.bd);
            }
            ram_bytes += y.ram.size(); ram_n += n;
        }
        for (int fd : sfd) if (fd >= 0) ::close(fd);
        ph("experts copied to RAM");
        // // ---- LRU arena for MISS experts: a host-RAM budget split over the layers, one slot per expert.
        // Without it a miss is a fresh mmap read of bpe bytes per expert per token, which on a model bigger
        // than RAM is a synchronous NTFS read on the critical path.
        uint64_t cache_bytes_plan = 0; int cache_slots_plan = 0;
        if (o.ram_cache_mib > 0) {
            uint64_t sumb = 0;
            for (int l = 0; l < nrun; ++l) sumb += L[(size_t) l].bpe;
            if (sumb) {
                const uint64_t budget = (uint64_t) o.ram_cache_mib << 20;
                const int q = (int) std::min<uint64_t>(budget / sumb, (uint64_t) c.n_expert);
                for (int l = 0; l < nrun; ++l) L[(size_t) l].ncache = q;
                uint64_t left = budget - (uint64_t) q * sumb;
                bool grew = true;
                while (grew) {  // spend the remainder one slot at a time, last layers first
                    grew = false;
                    for (int l = nrun - 1; l >= 0; --l) {
                        Layer& y = L[(size_t) l];
                        if (y.bpe == 0 || y.ncache >= c.n_expert || left < y.bpe) continue;
                        ++y.ncache; left -= y.bpe; grew = true;
                    }
                }
                for (int l = 0; l < nrun; ++l) {
                    Layer& y = L[(size_t) l];
                    if (y.ncache <= 0) continue;
                    y.cpool.resize((size_t) y.ncache * y.bpe);
                    y.cslot_of.assign((size_t) c.n_expert, -1);
                    y.cocc.assign((size_t) y.ncache, -1);
                    y.cstamp.assign((size_t) y.ncache, 0);
                    cache_bytes_plan += y.cpool.size(); cache_slots_plan += y.ncache;
                }
            }
        }
        // ---- VRAM preload: hottest experts per layer
        int preloaded = 0;
        if (o.gpu && have_profile)
            for (int l = 0; l < nrun; ++l) {
                Layer& y = L[(size_t) l];
                std::vector<int> ord(c.n_expert); std::iota(ord.begin(), ord.end(), 0);
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return y.freq[(size_t) a] > y.freq[(size_t) b]; });
                for (int k = 0; k < y.slots && k < c.n_expert && y.freq[(size_t) ord[(size_t) k]] > 0; ++k) { upload_slot(y, k, ord[(size_t) k]); ++preloaded; }
            }
        if (o.verbose) {
            int64_t sl = 0; for (int l = 0; l < nrun; ++l) sl += L[(size_t) l].slots;
            std::fprintf(stderr, "expert profile: %s\nexperts in RAM: %d (%.2f GiB) | VRAM slots: %lld, preloaded: %d | rest: mmap on disk\n",
                         have_profile ? ("loaded " + o.profile_in).c_str() : "none", ram_n, ram_bytes / 1073741824.0, (long long) sl, preloaded);
            std::fprintf(stderr, "miss LRU arena: %d slots (%.2f GiB) | threads: %d\n",
                         cache_slots_plan, cache_bytes_plan / 1073741824.0, o.threads);
            if (o.gpu) {
                size_t fb2 = 0, tb2 = 0; gpu::mem_info(&fb2, &tb2);
                std::fprintf(stderr, "device: %s | VRAM %.2f/%.2f GiB used, %.2f GiB free (weights %.2f GiB, scratch %.2f GiB)\n",
                             gpu::is_emulated() ? "CPU-EMULATED" : "CUDA",
                             vram_used / 1073741824.0, tb2 / 1073741824.0, fb2 / 1073741824.0,
                             (vram_used - vram_scratch) / 1073741824.0, vram_scratch / 1073741824.0);
            }
        }
        return true;
    }

    void reset() {
        for (Layer& y : L) {
            std::fill(y.ring_pos.begin(), y.ring_pos.end(), -1);
            for (Comp* cp : {&y.ac, &y.ic}) if (cp->ratio) {
                cp->kvs.assign((size_t) cp->coff * cp->ratio * cp->coff * cp->d, 0.f);
                cp->scs.assign((size_t) cp->coff * cp->ratio * cp->coff * cp->d, -INFINITY);
            }
        }
    }

    // ------------------------------------------------------------ attention
    bool comp_step(Comp& cp, const float* x, int pos, const float* cosT, const float* sinT) {
        const int ratio = cp.ratio, d = cp.d, W = cp.coff * d;
        std::vector<float> kv((size_t) W), sc((size_t) W);
        mv(cp.wkv, x, kv.data()); mv(cp.wg, x, sc.data());
        const float* ape = cp.ape.f() + (size_t) (pos % ratio) * W;
        for (int i = 0; i < W; ++i) sc[(size_t) i] += ape[i];
        const int row = (cp.overlap ? ratio : 0) + pos % ratio;
        std::memcpy(&cp.kvs[(size_t) row * W], kv.data(), (size_t) W * 4);
        std::memcpy(&cp.scs[(size_t) row * W], sc.data(), (size_t) W * 4);
        if ((pos + 1) % ratio) return false;
        std::vector<float> out((size_t) d);
        const int T = cp.overlap ? 2 * ratio : ratio;
        for (int ch = 0; ch < d; ++ch) {
            double mx = -INFINITY;
            auto v_at = [&](int t) { return cp.kvs[(size_t) t * W + ((cp.overlap && t >= ratio) ? d : 0) + ch]; };
            auto s_at = [&](int t) { return cp.scs[(size_t) t * W + ((cp.overlap && t >= ratio) ? d : 0) + ch]; };
            for (int t = 0; t < T; ++t) mx = std::max(mx, (double) s_at(t));
            double z = 0, a = 0;
            for (int t = 0; t < T; ++t) { const double e = std::exp((double) s_at(t) - mx); z += e; a += e * v_at(t); }
            out[(size_t) ch] = (float) (a / z);
        }
        if (cp.overlap) {
            std::memmove(&cp.kvs[0], &cp.kvs[(size_t) ratio * W], (size_t) ratio * W * 4);
            std::memmove(&cp.scs[0], &cp.scs[(size_t) ratio * W], (size_t) ratio * W * 4);
        }
        rmsnorm(out.data(), cp.norm.f(), c.rms_eps, d, out.data());
        const int fr = pos + 1 - ratio;
        rotary(out.data(), d, rd, cosT + (size_t) fr * (rd / 2), sinT + (size_t) fr * (rd / 2), false);
        if (cp.rotate) { hadamard(out.data(), d); if (o.qat_sim) fp4_sim(out.data(), d, 32); }
        else if (o.qat_sim) fp8_sim(out.data(), d - rd, 64);
        std::memcpy(cp.cache + (size_t) (pos / ratio) * d, out.data(), (size_t) d * 4);
        return true;
    }

    std::vector<int> indexer(Layer& y, const float* x, const float* qr, int pos, const float* cs, const float* sn, const float* cosT, const float* sinT) {
        const int ih = c.idx_heads, id = c.idx_key_len;
        std::vector<float> q((size_t) ih * id);
        mv(y.i_q_b, qr, q.data());
        for (int hh = 0; hh < ih; ++hh) {
            float* qh = &q[(size_t) hh * id];
            rotary(qh, id, rd, cs, sn, false);
            hadamard(qh, id);
            if (o.qat_sim) fp4_sim(qh, id, 32);
        }
        comp_step(y.ic, x, pos, cosT, sinT);
        std::vector<float> wgt((size_t) ih);
        cpu_mv(y.i_proj.type, y.i_proj.h, ih, dim, x, wgt.data());
        const float ws = std::pow((float) id, -0.5f) * std::pow((float) ih, -0.5f);
        for (float& v : wgt) v *= ws;
        const int ncomp = (pos + 1) / y.ratio, k = std::min(c.idx_topk, ncomp);
        std::vector<int> sel;
        if (k <= 0) return sel;
        std::vector<float> score((size_t) ncomp, 0.f);
        for (int t = 0; t < ncomp; ++t) {
            const float* kt = y.ic.cache + (size_t) t * id; float acc = 0.f;
            for (int hh = 0; hh < ih; ++hh) {
                float dot = 0.f; const float* qh = &q[(size_t) hh * id];
                for (int i = 0; i < id; ++i) dot += qh[i] * kt[i];
                acc += (dot > 0.f ? dot : 0.f) * wgt[(size_t) hh];
            }
            score[(size_t) t] = acc;
        }
        std::vector<int> ord((size_t) ncomp); std::iota(ord.begin(), ord.end(), 0);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) { return score[(size_t) a] != score[(size_t) b] ? score[(size_t) a] > score[(size_t) b] : a < b; });
        for (int i = 0; i < k; ++i) sel.push_back(ord[(size_t) i] + win);
        return sel;
    }

    void attention(int l, const float* x, int pos, float* out) {
        Layer& y = L[(size_t) l];
        const int nh = c.n_head;
        const bool yarn = y.ratio > 0;
        const float* cosT = yarn ? cos_y.data() : cos_p.data(); const float* sinT = yarn ? sin_y.data() : sin_p.data();
        const float* cs = cosT + (size_t) pos * (rd / 2); const float* sn = sinT + (size_t) pos * (rd / 2);
        std::vector<float> qa((size_t) c.q_lora), qr((size_t) c.q_lora), q((size_t) nh * hd);
        mv(y.q_a, x, qa.data());
        rmsnorm(qa.data(), y.q_a_norm.f(), c.rms_eps, c.q_lora, qr.data());
        mv(y.q_b, qr.data(), q.data());
        for (int hh = 0; hh < nh; ++hh) { float* qh = &q[(size_t) hh * hd]; rmsnorm(qh, nullptr, c.rms_eps, hd, qh); rotary(qh, hd, rd, cs, sn, false); }
        std::vector<float> kv((size_t) hd), kvn((size_t) hd);
        mv(y.kv, x, kv.data());
        rmsnorm(kv.data(), y.kv_norm.f(), c.rms_eps, hd, kvn.data());
        rotary(kvn.data(), hd, rd, cs, sn, false);
        if (o.qat_sim) fp8_sim(kvn.data(), hd - rd, 64);
        std::memcpy(&y.kvbuf[(size_t) (pos % win) * hd], kvn.data(), (size_t) hd * 4);
        // Keep ring_pos in step with the ring (attention_batch() does the same): win_row() trusts it whenever
        // snap_base > 0, which is for the whole life of the layer.
        if (!y.ring_pos.empty()) y.ring_pos[(size_t) (pos % win)] = pos;
        std::vector<int> idx, tmp; int r, cc;
        window_topk(win, 1, pos, idx, r, cc);
        if (y.ratio) {
            comp_step(y.ac, x, pos, cosT, sinT);
            if (y.has_idx) tmp = indexer(y, x, qr.data(), pos, cs, sn, cosT, sinT);
            else compress_topk(y.ratio, 1, pos, win, tmp, r, cc);
            idx.insert(idx.end(), tmp.begin(), tmp.end());
        }
        std::vector<float> o_(( size_t) nh * hd);
        sparse_attn_token(q.data(), nh, hd, y.kvbuf.data(), idx.data(), (int) idx.size(), y.sinks.f(), 1.f / std::sqrt((float) hd), o_.data());
        for (int hh = 0; hh < nh; ++hh) rotary(&o_[(size_t) hh * hd], hd, rd, cs, sn, true);
        const int G = c.o_groups; const int gin = nh * hd / G;
        std::vector<float> t((size_t) G * c.o_lora);
        for (int g = 0; g < G; ++g) mv(y.o_a, &o_[(size_t) g * gin], &t[(size_t) g * c.o_lora], (int64_t) g * c.o_lora, c.o_lora);
        mv(y.o_b, t.data(), out);
    }

    // ------------------------------------------------------------ MoE (router -> HIT on GPU / MISS on CPU -> shared expert)
    void moe(int l, const float* x, int token, float* y_out) {
        Layer& y = L[(size_t) l];
        const int K = c.n_expert_used, ff = c.ff_exp, E = c.n_expert;
        std::vector<float> lg((size_t) E);
        mv(y.gate_inp, x, lg.data());
        int32_t idx[16]; float w[16];
        const bool hash = l < c.n_hash_layers;
        const Score fn = c.gating_func == 1 ? Score::Softmax : c.gating_func == 2 ? Score::Sigmoid : Score::SqrtSoftplus;
        gate_route(lg.data(), E, K, fn, (!hash && y.exp_bias) ? y.exp_bias.f() : nullptr, hash ? (const int32_t*) y.tid2eid.h + (size_t) token * K : nullptr, c.exp_w_scale, idx, w);
        for (int k = 0; k < K; ++k) { last_routing_p[(size_t) l * K + k] = idx[k]; y.freq[(size_t) idx[k]]++; }

        // Every routed expert goes to the device when it can: the resident ones by VRAM pointer, the
        // others by host pointer through the staging pool. Only when the device refuses (no kernel for
        // these types, or no staging pool) does an expert fall back to the host loop below.
        gpu::ExpPtrs p{};   // value-initialised: n = 0, pointers unused beyond n
        int miss[16], nmiss = 0, nres = 0, nstaged = 0, nstaged_dev = 0;
        const bool device_experts = o.gpu && y.gpu_experts && (d_stage || gpu::is_emulated());
        p.bpe = y.bpe;
        for (int k = 0; k < K; ++k) {
            const int e = idx[k];
            int s = (o.gpu && y.slots > 0) ? y.slot_of[(size_t) e] : -1;
            if (s < 0 && o.gpu && y.slots > 0) {   // admit into a free slot (static profile mode fills the rest on first use)
                for (int q = 0; q < y.slots; ++q) if (y.occ[(size_t) q] < 0) { upload_slot(y, q, e); ++st->admits; s = q; break; }
            }
            if (s >= 0) {
                if (p.n >= gpu::kMaxHit) { miss[nmiss++] = k; continue; }
                uint8_t* base = y.dpool + (size_t) s * y.bpe;
                p.gate[p.n] = base; p.up[p.n] = base + y.bg; p.down[p.n] = base + y.bg + y.bu; p.w[p.n] = w[k]; ++p.n;
                ++nres;
            } else {
                missed_now.push_back({l, e});
                if (!device_experts || p.n >= gpu::kMaxHit) { miss[nmiss++] = k; ++nstaged; continue; }
                ExpSrc s2 = src_miss(y, e);
                p.hg[p.n] = s2.g; p.hu[p.n] = s2.u; p.hd[p.n] = s2.d; p.w[p.n] = w[k]; ++p.n;
                ++nstaged_dev;
            }
        }
        // hits/misses count VRAM residency, not which unit did the arithmetic: an expert staged from
        // host memory is still a miss for the residency plan, and the hit rate the report shows is what
        // tells the operator whether the slot budget is right.
        st->hits += (uint64_t) nres;
        st->misses += (uint64_t) (nstaged + nmiss + nstaged_dev);
        const float lim = c.swiglu_clamp_exp[(size_t) l];
        auto t0 = Clock::now();
        bool hit_ok = false;
        if (p.n > 0) {  // launch the experts asynchronously...
            gpu::h2d(d_x, x, (size_t) dim * 4);
            hit_ok = gpu::experts_hit(p, y.eg.type, y.eu.type, y.ed.type, ff, dim, lim, d_x, d_g, d_u, d_a, d_yh);
            if (!hit_ok) {   // no kernel, or a device error: those experts are computed on the host instead
                nmiss = 0;
                for (int k = 0; k < K; ++k) miss[nmiss++] = k;
                st->hits -= (uint64_t) nres;   // they were already counted as misses: residency, not device
                p.n = 0;
                if (!warned_host_fallback) {
                    warned_host_fallback = true;
                    std::fprintf(stderr, "warning: no device kernel for the experts of layer %d (%s/%s/%s): they run on the host\n",
                                 l, ggml_type_str(y.eg.type).c_str(), ggml_type_str(y.eu.type).c_str(), ggml_type_str(y.ed.type).c_str());
                }
            }
        }
        std::vector<float> acc((size_t) dim, 0.f), g((size_t) ff), u((size_t) ff), a((size_t) ff), yy((size_t) dim);
        for (int mi = 0; mi < nmiss; ++mi) {  // ...while the CPU computes whatever the device could not take
            const int k = miss[mi]; ExpSrc s = src_miss(y, idx[k]);
            cpu_mv(y.eg.type, s.g, ff, dim, x, g.data()); cpu_mv(y.eu.type, s.u, ff, dim, x, u.data());
            swiglu_clamped(g.data(), u.data(), ff, lim, a.data());
            for (float& v : a) v *= w[k];
            cpu_mv(y.ed.type, s.d, dim, ff, a.data(), yy.data());
            for (int i = 0; i < dim; ++i) acc[(size_t) i] += yy[(size_t) i];
        }
        auto t1 = Clock::now();
        if (hit_ok && p.n > 0) {
            std::vector<float> yh((size_t) dim);
            gpu::d2h(yh.data(), d_yh, (size_t) dim * 4);   // blocks until the GPU is done
            for (int i = 0; i < dim; ++i) acc[(size_t) i] += yh[(size_t) i];
        }
        auto t2 = Clock::now();
        st->cpu_miss_s += secs(t0, t1); st->gpu_hit_s += secs(t1, t2);
        // shared expert
        std::vector<float> sg((size_t) ff), su((size_t) ff), sa((size_t) ff), sy((size_t) dim);
        mv(y.sh_gate, x, sg.data()); mv(y.sh_up, x, su.data());
        swiglu_clamped(sg.data(), su.data(), ff, c.swiglu_clamp_shexp[(size_t) l], sa.data());
        mv(y.sh_down, sa.data(), sy.data());
        for (int i = 0; i < dim; ++i) y_out[i] = acc[(size_t) i] + sy[(size_t) i];
    }
    int32_t* last_routing_p = nullptr;

    // ============================================================ prefill: one chunk of T tokens
    // The token-per-token path above re-decodes every weight row once per token. Over a chunk of T
    // prompt tokens the same row is needed T times, so the batched path decodes it once and dots it
    // against T activation rows (row_dots / gpu::matmul), and evaluates each routed expert ONCE for
    // all the tokens of the chunk that picked it. The arithmetic per token is untouched: row_dots()
    // walks the same blocks with the same double accumulator as row_dot(), so the logits match.
    void attention_batch(int l, const float* X, int T, int pos0, float* out) {
        Layer& y = L[(size_t) l];
        const int nh = c.n_head;
        const bool yarn = y.ratio > 0;
        const float* cosT = yarn ? cos_y.data() : cos_p.data();
        const float* sinT = yarn ? sin_y.data() : sin_p.data();
        // The ring is written at pos % win, so a later token of the chunk would see an earlier slot
        // already holding a FUTURE token. Snapshot the ring first; win_row() redirects those reads.
        if (y.snap_base > 0)
            std::memcpy(y.kvbuf.data() + (size_t) y.snap_base * hd, y.kvbuf.data(), (size_t) win * hd * 4);

        mvb(y.q_a, X, dim, T, h_qa, c.q_lora);
        for (int t = 0; t < T; ++t)
            rmsnorm(h_qa + (size_t) t * c.q_lora, y.q_a_norm.f(), c.rms_eps, c.q_lora, h_qr + (size_t) t * c.q_lora);
        mvb(y.q_b, h_qr, c.q_lora, T, h_q, nh * hd);
        for (int t = 0; t < T; ++t) {
            const int pos = pos0 + t;
            const float* cs = cosT + (size_t) pos * (rd / 2);
            const float* sn = sinT + (size_t) pos * (rd / 2);
            for (int hh = 0; hh < nh; ++hh) {
                float* qh = h_q + (size_t) t * nh * hd + (size_t) hh * hd;
                rmsnorm(qh, nullptr, c.rms_eps, hd, qh);
                rotary(qh, hd, rd, cs, sn, false);
            }
        }
        mvb(y.kv, X, dim, T, h_kv, hd);
        for (int t = 0; t < T; ++t) {
            const int pos = pos0 + t;
            const float* cs = cosT + (size_t) pos * (rd / 2);
            const float* sn = sinT + (size_t) pos * (rd / 2);
            float* kvn = h_kvn + (size_t) t * hd;
            rmsnorm(h_kv + (size_t) t * hd, y.kv_norm.f(), c.rms_eps, hd, kvn);
            rotary(kvn, hd, rd, cs, sn, false);
            if (o.qat_sim) fp8_sim(kvn, hd - rd, 64);
            const int slot = pos % win;
            std::memcpy(&y.kvbuf[(size_t) slot * hd], kvn, (size_t) hd * 4);
            y.ring_pos[(size_t) slot] = pos;
        }
        // Compressor: strictly in position order, exactly as the sequential path.
        if (y.ratio)
            for (int t = 0; t < T; ++t) comp_step(y.ac, X + (size_t) t * dim, pos0 + t, cosT, sinT);

        const float scale = 1.f / std::sqrt((float) hd);
        std::vector<int> idx, tmp;   // reused across the chunk: this loop runs T times per layer
        for (int t = 0; t < T; ++t) {
            const int pos = pos0 + t;
            int n = 0;
            int r, cc;
            window_topk(win, 1, pos, idx, r, cc);
            // -1 is a masked slot: it stays -1, exactly as the sequential path leaves it.
            for (int s : idx) h_attn_idx[(size_t) t * pf_max_list + n++] = s < 0 ? -1 : win_row(y, s, pos);
            if (y.ratio) {
                if (y.has_idx) tmp = indexer(y, X + (size_t) t * dim, h_qr + (size_t) t * c.q_lora, pos,
                                             cosT + (size_t) pos * (rd / 2), sinT + (size_t) pos * (rd / 2), cosT, sinT);
                else compress_topk(y.ratio, 1, pos, win, tmp, r, cc);
                for (int s : tmp) h_attn_idx[(size_t) t * pf_max_list + n++] = s;
            }
            h_attn_n[t] = n;
            sparse_attn_token(h_q + (size_t) t * nh * hd, nh, hd, y.kvbuf.data(),
                              h_attn_idx + (size_t) t * pf_max_list, n, y.sinks.f(), scale,
                              h_o + (size_t) t * nh * hd);
        }
        for (int t = 0; t < T; ++t) {
            const int pos = pos0 + t;
            const float* cs = cosT + (size_t) pos * (rd / 2);
            const float* sn = sinT + (size_t) pos * (rd / 2);
            for (int hh = 0; hh < nh; ++hh) rotary(h_o + (size_t) t * nh * hd + (size_t) hh * hd, hd, rd, cs, sn, true);
        }
        const int G = c.o_groups;
        for (int g = 0; g < G; ++g)
            mvb(y.o_a, h_o + (size_t) g * pf_gin, nh * hd, T, h_t + (size_t) g * c.o_lora, G * c.o_lora,
                (int64_t) g * c.o_lora, c.o_lora);
        mvb(y.o_b, h_t, G * c.o_lora, T, out, dim);
    }

    /// True when this build can run a batched MoE layer on the device at all.
    bool pf_device_experts() const {
        static const bool off = std::getenv("DSV4_NO_DEVICE_EXPERTS") != nullptr;   // diagnostics, see pf_dev()
        return !off && o.gpu && !gpu::is_emulated() && d_pYm != nullptr;
    }

    void moe_batch(int l, const float* X, int T, const int32_t* toks, float* out) {
        Layer& y = L[(size_t) l];
        const int K = c.n_expert_used, ff = c.ff_exp, E = c.n_expert;
        mvb(y.gate_inp, X, dim, T, h_lg, E);
        const bool hash = l < c.n_hash_layers;
        const Score fn = c.gating_func == 1 ? Score::Softmax : c.gating_func == 2 ? Score::Sigmoid : Score::SqrtSoftplus;
        std::fill(h_cnt, h_cnt + E, 0);
        for (int t = 0; t < T; ++t) {
            int32_t idx[16]; float w[16];
            gate_route(h_lg + (size_t) t * E, E, K, fn, (!hash && y.exp_bias) ? y.exp_bias.f() : nullptr,
                       hash ? (const int32_t*) y.tid2eid.h + (size_t) toks[t] * K : nullptr, c.exp_w_scale, idx, w);
            for (int k = 0; k < K; ++k) {
                h_tok[(size_t) t * K + k] = idx[k];
                h_wt[(size_t) t * K + k] = w[k];
                ++h_cnt[idx[k]];
                y.freq[(size_t) idx[k]]++;
            }
            // The routing trace of the LAST token of the chunk is what --dump-routing and the
            // adaptive swaps read, so it is kept in the same place the sequential path leaves it.
            // (A trace row is n_layer*K wide like last_routing, so --max-layers leaves the unused layers at -1.)
            // Every token's routing also goes into chunk_routing, which lets a chunked prefill write
            // the same per-token trace file the token-per-token path does.
            for (int k = 0; k < K; ++k) {
                chunk_routing[(size_t) t * (size_t) c.n_layer * K + (size_t) l * K + k] = idx[k];
                if (t == T - 1) last_routing_p[(size_t) l * K + k] = idx[k];
            }
        }
        // Bucket the chunk's (token, expert) pairs by expert: h_sel/h_w become one contiguous run per
        // expert, which is exactly what experts_batch() and row_dots() want.
        int run = 0;
        for (int e = 0; e < E; ++e) { h_off[e] = run; run += h_cnt[e]; }
        h_off[E] = run;
        std::copy(h_off, h_off + E + 1, h_at);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                const int e = h_tok[(size_t) t * K + k];
                const int p = h_at[e]++;
                h_sel[p] = t;
                h_w[p] = h_wt[(size_t) t * K + k];
            }

        int nexp = 0;
        for (int e = 0; e < E; ++e) {
            if (!h_cnt[e]) continue;
            gpu::ExpBatch& b = pf_b[(size_t) nexp];
            int s = (y.slots > 0) ? y.slot_of[(size_t) e] : -1;
            if (s < 0 && y.slots > 0) {   // admit into a free slot, as the sequential path does
                for (int q = 0; q < y.slots; ++q)
                    if (y.occ[(size_t) q] < 0) { upload_slot(y, q, e); ++st->admits; s = q; break; }
            }
            if (s >= 0) {
                uint8_t* base = y.dpool + (size_t) s * y.bpe;
                b.gate = base; b.up = base + y.bg; b.down = base + y.bg + y.bu;
                b.hg = b.hu = b.hd = nullptr;
                st->hits += (uint64_t) h_cnt[e];
            } else {
                missed_now.push_back({l, e});
                ExpSrc src = src_miss(y, e);
                b.gate = b.up = b.down = nullptr;
                b.hg = src.g; b.hu = src.u; b.hd = src.d;
                st->misses += (uint64_t) h_cnt[e];
            }
            b.tok = h_sel + h_off[e];
            b.w = h_w + h_off[e];
            b.n = h_cnt[e];
            h_eid[nexp] = e;
            ++nexp;
        }
        std::fill(h_acc, h_acc + (size_t) T * dim, 0.f);
        const float lim = c.swiglu_clamp_exp[(size_t) l];
        bool ok = false;
        if (pf_device_experts() && y.gpu_experts && nexp > 0) {
            // X is the same block the gate_inp matmul above just uploaded, so this is usually free.
            const float* dX = pf_up(X, dim, T, dim);
            if (dX && dX != X) {   // pf_up falls back to the host pointer: a kernel must not read it
                ok = gpu::experts_batch(pf_b.data(), nexp, y.eg.type, y.eu.type, y.ed.type, ff, dim, lim,
                                        dX, T, d_pg, d_pu, d_pa, d_pd, d_pYm);
                if (ok) gpu::d2h(h_acc, d_pYm, (size_t) T * dim * 4);
            }
        }
        if (!ok) {
            // No device kernels for this layer: run the same buckets on the host, where row_dots()
            // still decodes each expert row once for all of its tokens.
            ok = true;
            for (int ei = 0; ei < nexp; ++ei) {
                const gpu::ExpBatch& b = pf_b[(size_t) ei];
                // A resident expert's device pointer cannot be read on the host: re-read it from its
                // source (RAM arena, LRU arena or mmap), which is the same bytes that were uploaded.
                const ExpSrc s2 = b.gate ? src(y, h_eid[ei]) : src_miss(y, h_eid[ei]);
                if (!s2.g || !s2.u || !s2.d) { ok = false; break; }
                host_expert(y, b, s2.g, s2.u, s2.d, ff, lim, X);
            }
            if (!ok) {   // last resort: the exact sequential path, token by token
                for (int t = 0; t < T; ++t) moe(l, X + (size_t) t * dim, toks[t], out + (size_t) t * dim);
                return;
            }
        }
        // shared expert, batched
        mvb(y.sh_gate, X, dim, T, h_sg, ff);
        mvb(y.sh_up, X, dim, T, h_su, ff);
        for (int t = 0; t < T; ++t)
            swiglu_clamped(h_sg + (size_t) t * ff, h_su + (size_t) t * ff, ff, c.swiglu_clamp_shexp[(size_t) l],
                           h_sa + (size_t) t * ff);
        mvb(y.sh_down, h_sa, ff, T, h_sy, dim);
        for (size_t i = 0; i < (size_t) T * dim; ++i) out[i] = h_acc[i] + h_sy[i];
    }

    /// One expert, all of the chunk's tokens that routed to it, on the host. Each of the three
    /// matrices is walked once for the whole bucket instead of once per token.
    void host_expert(Layer& y, const gpu::ExpBatch& b, const uint8_t* g, const uint8_t* u, const uint8_t* d,
                     int ff, float lim, const float* X) {
        const int n = b.n;
        const size_t rb_g = row_bytes(y.eg.type, dim), rb_u = row_bytes(y.eu.type, dim), rb_d = row_bytes(y.ed.type, ff);
        if (!rb_g || !rb_u || !rb_d) return;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < ff; ++r) {
            row_dots(y.eg.type, g + (size_t) r * rb_g, X, dim, b.tok, n, dim, h_mg + (size_t) r, ff);
            row_dots(y.eu.type, u + (size_t) r * rb_u, X, dim, b.tok, n, dim, h_mu + (size_t) r, ff);
        }
        for (int t = 0; t < n; ++t) {
            swiglu_clamped(h_mg + (size_t) t * ff, h_mu + (size_t) t * ff, ff, lim, h_ma + (size_t) t * ff);
            for (int i = 0; i < ff; ++i) h_ma[(size_t) t * ff + i] *= b.w[t];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < dim; ++r)
            row_dots(y.ed.type, d + (size_t) r * rb_d, h_ma, ff, nullptr, n, ff, h_md + (size_t) r, dim);
        // Same expert-order accumulation the device path uses; within one expert a token appears once.
        for (int t = 0; t < n; ++t) {
            const int32_t tk = b.tok[t];
            for (int i = 0; i < dim; ++i) h_acc[(size_t) tk * dim + i] += h_md[(size_t) t * dim + i];
        }
    }

    /// One half of a block for the whole chunk: hc_pre per token (24 rows, cheap, and it keeps the
    /// exact summation order of the sequential path), then the batched body, then hc_post.
    void hc_half(int l, int T, bool ffn, float* nrm_out) {
        Layer& y = L[(size_t) l];
        const Tn& fn = ffn ? y.hc_ffn_fn : y.hc_attn_fn;
        const Tn& sc = ffn ? y.hc_ffn_scale : y.hc_attn_scale;
        const Tn& bs = ffn ? y.hc_ffn_base : y.hc_attn_base;
        const Tn& an = ffn ? y.ffn_norm : y.attn_norm;
        for (int t = 0; t < T; ++t) {
            float* yv = h_yv + (size_t) t * dim;
            hc_pre(h_h + (size_t) t * hc * dim, hc, dim, fn.f(), sc.f(), bs.f(), c.rms_eps,
                   c.hc_sinkhorn_iters, c.hc_eps, yv, h_post + (size_t) t * hc, h_comb + (size_t) t * pf_mix);
            rmsnorm(yv, an.f(), c.rms_eps, dim, nrm_out + (size_t) t * dim);
        }
        // nrm_out is the block every following matmul reads, and it has just been rewritten: any
        // device copy of it from before this call is stale.
        pf_clear_note();
    }
    void hc_join(int T, const float* a) {
        for (int t = 0; t < T; ++t)
            hc_post(a + (size_t) t * dim, h_h + (size_t) t * hc * dim, h_post + (size_t) t * hc,
                    h_comb + (size_t) t * pf_mix, hc, dim, h_hn + (size_t) t * hc * dim);
        std::swap(h_h, h_hn);   // h_h always holds the running residual streams of this chunk
    }

    void block_batch(int l, int T, const int32_t* toks, int pos0) {
        hc_half(l, T, false, h_nrm);
        attention_batch(l, h_nrm, T, pos0, h_x);
        hc_join(T, h_x);
        hc_half(l, T, true, h_nrm);
        moe_batch(l, h_nrm, T, toks, h_x);
        hc_join(T, h_x);
    }

    int forward_chunk(const int* tokens, int T, int pos0, std::vector<float>* logits) {
        if (T <= 0) return 0;
        if (pf_cap <= 0 || T > pf_cap) {   // no batched scratch: fall back to the sequential path.
            // end_token() is deliberately NOT called here: the caller closes the chunk, and closing
            // it twice would count every token twice.
            chunk_routing_n = 0;
            std::fill(chunk_routing.begin(), chunk_routing.end(), -1);
            const size_t row = last_routing_size_;
            const size_t room = row ? chunk_routing.size() / row : 0;   // tokens the trace can hold
            for (int t = 0; t < T; ++t) {
                std::vector<float>* lg = (t + 1 == T) ? logits : nullptr;
                forward(tokens[t], pos0 + t, lg);
                // Keep the per-token trace the same shape the batched path produces, for as many
                // tokens as the arena can hold (a caller may ask for more than pf_cap).
                if ((size_t) t < room)
                    for (size_t i = 0; i < row; ++i)
                        chunk_routing[(size_t) t * row + i] = last_routing_p[i];
            }
            chunk_routing_n = (int) std::min<size_t>((size_t) T, room);
            return T;
        }
        // The chunk trace starts empty; a layer that runs the sequential fallback leaves -1 behind.
        chunk_routing_n = 0;
        std::fill(chunk_routing.begin(), chunk_routing.end(), -1);
        auto t0 = Clock::now();
        const int n = hc * dim;
        for (int t = 0; t < T; ++t) {
            float* e = h_e + (size_t) t * dim;
            dequant_row(embd.type, embd.h + (size_t) tokens[t] * embd.rb, dim, e);
            for (int j = 0; j < hc; ++j) std::memcpy(h_h + (size_t) t * n + (size_t) j * dim, e, (size_t) dim * 4);
        }
        for (int t = 0; t < T; ++t) h_tk[t] = tokens[t];
        for (int l = 0; l < nrun; ++l) block_batch(l, T, h_tk, pos0);
        chunk_routing_n = T;
        // Only the last token of the chunk needs logits: it is the one that starts the answer.
        if (logits) {
            std::vector<float> x((size_t) dim), xn((size_t) dim);
            hc_head(h_h + (size_t) (T - 1) * n, hc, dim, out_hc_fn.f(), out_hc_scale.f(), out_hc_base.f(),
                    c.rms_eps, c.hc_eps, x.data());
            // output_norm.weight, exactly as forward() does: without it the logits are the projection of an
            // unnormalised vector (wrong scale, wrong argmax).
            rmsnorm(x.data(), outnorm.f(), c.rms_eps, dim, xn.data());
            logits->assign((size_t) c.vocab, 0.f);
            mv(outw, xn.data(), logits->data());
        }
        st->total_s += secs(t0, Clock::now());
        ++st->prefill_chunks;
        st->prefill_tokens += (uint64_t) T;
        return T;
    }

    // ------------------------------------------------------------ block / forward
    void block(int l, std::vector<float>& hres, int token, int pos) {
        Layer& y = L[(size_t) l];
        std::vector<float> yv((size_t) dim), post((size_t) hc), comb((size_t) hc * hc), nrm((size_t) dim), a((size_t) dim), hn((size_t) hc * dim);
        hc_pre(hres.data(), hc, dim, y.hc_attn_fn.f(), y.hc_attn_scale.f(), y.hc_attn_base.f(), c.rms_eps, c.hc_sinkhorn_iters, c.hc_eps, yv.data(), post.data(), comb.data());
        rmsnorm(yv.data(), y.attn_norm.f(), c.rms_eps, dim, nrm.data());
        attention(l, nrm.data(), pos, a.data());
        hc_post(a.data(), hres.data(), post.data(), comb.data(), hc, dim, hn.data());
        hres = hn;
        hc_pre(hres.data(), hc, dim, y.hc_ffn_fn.f(), y.hc_ffn_scale.f(), y.hc_ffn_base.f(), c.rms_eps, c.hc_sinkhorn_iters, c.hc_eps, yv.data(), post.data(), comb.data());
        rmsnorm(yv.data(), y.ffn_norm.f(), c.rms_eps, dim, nrm.data());
        moe(l, nrm.data(), token, a.data());
        hc_post(a.data(), hres.data(), post.data(), comb.data(), hc, dim, hn.data());
        hres = hn;
    }

    void forward(int token, int pos, std::vector<float>* logits) {
        auto t0 = Clock::now();
        std::vector<float> e((size_t) dim), hres((size_t) hc * dim), x((size_t) dim), xn((size_t) dim);
        dequant_row(embd.type, embd.h + (size_t) token * embd.rb, dim, e.data());
        for (int j = 0; j < hc; ++j) std::memcpy(&hres[(size_t) j * dim], e.data(), (size_t) dim * 4);
        for (int l = 0; l < nrun; ++l) block(l, hres, token, pos);
        hc_head(hres.data(), hc, dim, out_hc_fn.f(), out_hc_scale.f(), out_hc_base.f(), c.rms_eps, c.hc_eps, x.data());
        rmsnorm(x.data(), outnorm.f(), c.rms_eps, dim, xn.data());
        if (logits) { logits->assign((size_t) c.vocab, 0.f); mv(outw, xn.data(), logits->data()); }
        st->total_s += secs(t0, Clock::now());
    }

    void end_token() { end_chunk(1); }

    /// Called once per chunk (a chunk of one token is exactly the old end_token).
    void end_chunk(int n) {
        if (o.adapt_swaps > 0 && o.gpu) {
            std::stable_sort(missed_now.begin(), missed_now.end(), [&](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return L[(size_t) a.first].freq[(size_t) a.second] > L[(size_t) b.first].freq[(size_t) b.second]; });
            int budget = o.adapt_swaps;
            for (auto& m : missed_now) {
                if (budget <= 0) break;
                Layer& y = L[(size_t) m.first];
                if (y.slots == 0 || y.slot_of[(size_t) m.second] >= 0) continue;
                int vs = 0; uint64_t vf = UINT64_MAX;
                for (int s = 0; s < y.slots; ++s) { const uint64_t f = y.occ[(size_t) s] < 0 ? 0 : y.freq[(size_t) y.occ[(size_t) s]]; if (f < vf) { vf = f; vs = s; } }
                if (y.freq[(size_t) m.second] > vf) { upload_slot(y, vs, m.second); ++st->swaps; --budget; }
            }
        }
        missed_now.clear();
        st->tokens += (uint64_t) (n > 0 ? n : 1);
    }

    bool save_profile(std::string& err) const {
        if (o.profile_out.empty()) return true;
        const std::string tmp = o.profile_out + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) { err = "cannot write " + tmp; return false; }
        const uint32_t ver = 1, nl = (uint32_t) c.n_layer, ne = (uint32_t) c.n_expert;
        std::fwrite("DSV4PROF", 1, 8, f); std::fwrite(&ver, 4, 1, f); std::fwrite(&nl, 4, 1, f); std::fwrite(&ne, 4, 1, f); std::fwrite(&model_hash, 8, 1, f);
        for (int l = 0; l < c.n_layer; ++l) std::fwrite(L[(size_t) l].freq.data(), 8, (size_t) c.n_expert, f);
        std::fflush(f); fsync(fileno(f)); std::fclose(f);
        if (std::rename(tmp.c_str(), o.profile_out.c_str()) != 0) { err = "rename failed for " + o.profile_out; return false; }
        return true;
    }

    void selfcheck() const {
        std::map<uint32_t, const GgufTensor*> first;
        for (const GgufTensor& t : h.tensors) if (t.dims.size() >= 2 && dequant_supported(t.type) && !first.count(t.type) && t.name.find("tid2eid") == std::string::npos) first[t.type] = &t;
        std::printf("dequantisation self-check (first 3 rows of the first tensor of each type; weights should be finite, ~zero mean, small):\n");
        for (auto& kv : first) {
            const GgufTensor& t = *kv.second;
            const uint8_t* base = maps[(size_t) t.shard] + t.abs_offset;
            const int64_t in = (int64_t) t.dims[0]; const size_t rb = row_bytes(t.type, in);
            std::vector<float> v((size_t) in * 3);
            for (int r = 0; r < 3; ++r) dequant_row(t.type, base + (size_t) r * rb, in, &v[(size_t) r * in]);
            double s = 0, s2 = 0, mx = 0; bool fin = true;
            for (float f : v) { fin = fin && std::isfinite(f); s += f; s2 += (double) f * f; mx = std::max(mx, (double) std::fabs(f)); }
            const double m = s / v.size();
            std::printf("  %-8s %-40s finite=%d mean=%+.5f std=%.5f absmax=%.4f\n", ggml_type_str(kv.first).c_str(), t.name.c_str(), (int) fin, m, std::sqrt(s2 / v.size() - m * m), mx);
        }
    }
};

// ================================================================================================
Model::Model() : p_(new Impl()) { p_->st = &stats; }
Model::~Model() {}
bool Model::load(const std::vector<std::string>& s, const RunOpts& o, std::string& err) {
    if (!p_->load(s, o, err)) return false;
    last_routing.assign((size_t) p_->c.n_layer * p_->c.n_expert_used, -1);
    p_->last_routing_p = last_routing.data();
    p_->last_routing_size_ = last_routing.size();
    return true;
}
void Model::reset() { p_->reset(); }
void Model::forward(int token, int pos, std::vector<float>* logits) {
    p_->forward(token, pos, logits);
    // A single token has no chunk trace: last_routing already carries it, as it always did.
    chunk_routing_.clear();
    chunk_routing_n_ = 0;
}
int Model::forward_chunk(const int* tokens, int n, int pos0, std::vector<float>* logits) {
    // Only the last token of a batched chunk writes last_routing: start from a clean row so nothing left over
    // from the previous chunk survives in the slots this chunk does not touch.
    std::fill(last_routing.begin(), last_routing.end(), -1);
    const int done = p_->forward_chunk(tokens, n, pos0, logits);
    // The per-token routing trace of the chunk, handed out so --dump-routing writes the same rows
    // the token-per-token path writes.
    chunk_routing_.assign(p_->chunk_routing.begin(), p_->chunk_routing.end());
    chunk_routing_n_ = p_->chunk_routing_n;
    return done;
}
int Model::max_chunk() const { return p_->pf_cap; }
void Model::end_token() { p_->end_token(); }
void Model::end_chunk(int n) { p_->end_chunk(n); }
bool Model::save_profile(std::string& err) const { return p_->save_profile(err); }
void Model::selfcheck() const { p_->selfcheck(); }
uint64_t Model::model_hash() const { return p_->model_hash; }
const Dsv4Config& Model::config() const { return p_->c; }
const std::vector<std::string>& Model::tokens() const { return p_->toks; }

}  // namespace dsv4
