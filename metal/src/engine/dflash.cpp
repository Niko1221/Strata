// metal/src/engine/dflash.cpp - the native DFlash2 drafter: the same graph as llama.cpp's DFLASH (src/models/dflash.cpp,
// native-spec/graph.md), fused into Strata's Metal kernels (kernels/dflash.metal + gemv.metal's Q8_0 tiles) on its
// own mtl::Context.  DflashDrafter's default backend (llama.cpp's graph, when built in: STRATA_DFLASH_NATIVE=0).
//
//   inject  fc + norm of the target features, then every layer's k/v (norm, RoPE) into a per-layer f16 ring of 2064
//           slots (position p lives in slot p % 2064; SP[slot] = the position it holds, -1 empty)
//   draft   [last token, mask x (nb - 1)] through the 5 layers (the block's own k/v stay out of the ring), the draft
//           head + selector gate, the top-16 per row and the candidate lattice; the host walks the lattice
// The weights are the GGUF itself (mmap, wrapped, bound at their tensor offsets); only d2t (I64 to i32) and the
// mask-token embedding (one token_embd row in the target's type, dequantized once) are converted.
#include "strata/spec/dflash_backend.hpp"

#include "gguf_lite.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace strata::spec {
namespace {

constexpr int D = 2560, NL = 5, MAXR = 8, RING = 2064, FF = 7680, FEAT = 5 * D, TOPK = 16, STAGE_ROWS = 64, VOCAB = 248320;
constexpr int Q8 = 8;

struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct QkvArgs { int32_t pos0, nb; };
struct SelArgs { int32_t anchor; uint32_t off_pre, off_next; };

struct Layer {
    const gguf::Tensor *q, *k, *v, *o, *acp, *fcp, *gate, *up, *down;
    size_t attn_norm, q_norm, k_norm, ffn_norm, abase, fbase;   // byte offsets of the F32 tensors
};

class NativeDflash final : public DflashBackend {
public:
    bool load(const std::string& path, std::string& err) override;
    int target_hidden() const override { return D; }
    int block() const override { return block_; }
    bool inject(const float* feats, int n, int pos0, std::string& err) override;
    int draft(int32_t id_last, int pos0, int max_n, float p_min, int32_t* out, float* conf) override;
    void trim(int pos) override;
    int pos_max() const override { return pos_max_; }
    const float* lattice() const override { return lat_.as<float>(); }
    ~NativeDflash() override;

private:
    const gguf::Tensor* tn(const std::string& name);
    const gguf::Tensor* mat(const std::string& name, uint64_t n_in, uint64_t n_out);
    size_t f32(const std::string& name, uint64_t n);   // byte offset of an F32 tensor of n values
    const mtl::Pipeline* P(const std::string& fn, int ntok = 0);
    void push(mtl::Dispatch d, bool concurrent);
    void gemv(const gguf::Tensor& t, const mtl::Buffer& x, size_t x_off, const mtl::Buffer& y, size_t y_off, int y_stride,
              int ntok, bool concurrent = false);
    void kern(const char* fn, std::vector<mtl::Arg> args, uint32_t g0, uint32_t g1, uint32_t tg, bool concurrent = false);
    void enc_inject(size_t stage_off, int n, int pos0);
    void enc_draft(int nb, int pos0, int32_t anchor);

    gguf::File fm_;
    std::unique_ptr<mtl::Context> cx_;
    int lib_ = -1, block_ = 8, nv_ = 0, pos_max_ = -1;
    int32_t mask_id_ = 248077;
    mtl::Buffer wm_;
    std::map<std::string, mtl::Pipeline> pc_;
    std::vector<mtl::Dispatch> ds_;
    std::string missing_;
    const gguf::Tensor *fc_ = nullptr, *head_ = nullptr, *selh_ = nullptr, *pre_ = nullptr, *next_ = nullptr, *emb_ = nullptr;
    size_t enc_norm_ = 0, out_norm_ = 0;
    Layer ly_[NL] = {};
    mtl::Buffer stage_, g_, kraw_, vraw_, x0_, hres_[2], hn_, hc_, ffninp_, fn_, fc_o_, o_, m_, hfin_, dyna_, dynf_, qraw_,
        qn_, kr_, vr_, part_, attn_, gt_, up_, hh_, logits_, selg_, cand_, unary_, lat_, d2t_, sp_, ringk_, ringv_, nk_, nv_buf_;
};

NativeDflash::~NativeDflash() {
    if (!cx_) return;
    // mtl::Buffer is a plain handle: free each while cx_ is alive (wm_ wraps fm_'s mmap, which fm_ unmaps after)
    for (mtl::Buffer* b : {&wm_, &stage_, &g_, &kraw_, &vraw_, &x0_, &hres_[0], &hres_[1], &hn_, &hc_, &ffninp_, &fn_, &fc_o_,
                           &o_, &m_, &hfin_, &dyna_, &dynf_, &qraw_, &qn_, &kr_, &vr_, &part_, &attn_, &gt_, &up_, &hh_,
                           &logits_, &selg_, &cand_, &unary_, &lat_, &d2t_, &sp_, &ringk_, &ringv_, &nk_, &nv_buf_})
        if (b->handle) cx_->free(*b);
}

const gguf::Tensor* NativeDflash::tn(const std::string& name) {
    const gguf::Tensor* t = fm_.find(name);
    if (!t) missing_ += " " + name;
    return t;
}

const gguf::Tensor* NativeDflash::mat(const std::string& name, uint64_t n_in, uint64_t n_out) {
    const gguf::Tensor* t = tn(name);
    if (t && (t->type != Q8 || t->shape.size() != 2 || t->shape[0] != n_in || t->shape[1] != n_out))
        missing_ += " " + name + "(not a Q8_0 [" + std::to_string(n_in) + ", " + std::to_string(n_out) + "])";
    return t;
}

size_t NativeDflash::f32(const std::string& name, uint64_t n) {
    const gguf::Tensor* t = tn(name);
    if (!t) return 0;
    uint64_t ne = 1;
    for (uint64_t s : t->shape) ne *= s;
    if (t->type != 0 || ne != n) missing_ += " " + name + "(not F32 x " + std::to_string(n) + ")";
    return (size_t) t->offset;
}

bool NativeDflash::load(const std::string& path, std::string& err) {
    if (!fm_.open(path, err)) { err = "--dflash: " + err; return false; }
    auto num = [&](const char* k, double def) { auto it = fm_.meta_num.find(k); return it == fm_.meta_num.end() ? def : it->second; };
    block_ = (int) num("dflash.block_size", 8);
    if (num("dflash.block_count", NL) != NL || num("dflash.embedding_length", D) != D || num("dflash.feed_forward_length", FF) != FF ||
        num("dflash.selector_top_k", TOPK) != TOPK || num("dflash.selector_rank", 256) != 256 || block_ < 2 || block_ > MAXR) {
        err = "--dflash: native drafter: the draft's config is not the 5-layer 2560-wide DFlash2 with top-k 16 and a block of 2..8";
        return false;
    }
    mask_id_ = (int32_t) num("tokenizer.ggml.mask_token_id", 248077);
    fc_ = mat("fc.weight", FEAT, D);
    selh_ = mat("selector_hidden.weight", D, 256);
    pre_ = mat("selector_predecessor.weight", 256, 248320);
    next_ = mat("selector_successor.weight", 256, 248320);
    emb_ = tn("token_embd.weight");
    head_ = tn("output.weight");
    const gguf::Tensor* d2t = tn("d2t");
    enc_norm_ = f32("enc.output_norm.weight", D);
    out_norm_ = f32("output_norm.weight", D);
    for (int il = 0; il < NL; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        Layer& l = ly_[il];
        l.q = mat(b + "attn_q.weight", D, 4096);
        l.k = mat(b + "attn_k.weight", D, 1024);
        l.v = mat(b + "attn_v.weight", D, 1024);
        l.o = mat(b + "attn_output.weight", 4096, D);
        l.acp = mat(b + "attn_conv_proj.weight", D, 640);
        l.fcp = mat(b + "ffn_conv_proj.weight", D, 640);
        l.gate = mat(b + "ffn_gate.weight", D, FF);
        l.up = mat(b + "ffn_up.weight", D, FF);
        l.down = mat(b + "ffn_down.weight", FF, D);
        l.attn_norm = f32(b + "attn_norm.weight", D);
        l.ffn_norm = f32(b + "ffn_norm.weight", D);
        l.q_norm = f32(b + "attn_q_norm.weight", 128);
        l.k_norm = f32(b + "attn_k_norm.weight", 128);
        l.abase = f32(b + "attn_conv_base", 4 * D);
        l.fbase = f32(b + "ffn_conv_base", 4 * D);
    }
    // the head over a draft vocabulary (d2t) is what tools/dflash_head.py --draft-vocab writes
    const char* fix = "; make the draft GGUF with tools/dflash_head.py --draft-vocab <ids> (e.g. data/draft_vocab_en.bin)";
    if (!missing_.empty()) { err = "--dflash: native drafter: missing or unexpected tensors:" + missing_ + fix; return false; }
    if (std::string(ref::type_name(emb_->type)) == "?" || emb_->shape[0] != D || head_->type != Q8 || head_->shape[0] != D ||
        d2t->type != 27 || d2t->shape[0] != head_->shape[1]) {
        err = std::string("--dflash: native drafter: token_embd must be a known type, output Q8_0 and d2t I64 over the draft vocab") + fix;
        return false;
    }
    nv_ = (int) head_->shape[1];
    if (nv_ > 40 * 1024) {
        err = std::string("--dflash: native drafter: the draft vocab exceeds the top-k kernel's 40960 rows") + fix;
        return false;
    }

    cx_ = std::make_unique<mtl::Context>();
    wm_ = cx_->wrap(fm_.map, fm_.map_bytes);
    // Q8_0 is the only gemv format here; the full ggml-common stays in (quant.metal's i-quant paths name its tables)
    lib_ = cx_->compile({"quant.metal", "units.metal", "gemv.metal", "dflash.metal"}, "#define GEMV_TYPES GV_SET(T_Q8_0)\n");
    if (P("df_topk")->max_threads < 1024) { err = "--dflash: native drafter: this GPU runs fewer than 1024 threads per threadgroup"; return false; }

    auto A = [&](size_t floats) { return cx_->alloc(floats * 4); };
    stage_ = A((size_t) STAGE_ROWS * FEAT);
    g_ = A((size_t) MAXR * D);
    kraw_ = A((size_t) NL * MAXR * 1024);
    vraw_ = A((size_t) NL * MAXR * 1024);
    x0_ = A((size_t) MAXR * D);
    hres_[0] = A((size_t) MAXR * D);
    hres_[1] = A((size_t) MAXR * D);
    hn_ = A((size_t) MAXR * D);
    hc_ = A((size_t) MAXR * D);
    ffninp_ = A((size_t) MAXR * D);
    fn_ = A((size_t) MAXR * D);
    fc_o_ = A((size_t) MAXR * D);
    o_ = A((size_t) MAXR * D);
    m_ = A((size_t) MAXR * D);
    hfin_ = A((size_t) MAXR * D);
    dyna_ = A((size_t) MAXR * 640);
    dynf_ = A((size_t) MAXR * 640);
    qraw_ = A((size_t) MAXR * 4096);
    qn_ = A((size_t) MAXR * 4096);
    kr_ = A((size_t) MAXR * 1024);
    vr_ = A((size_t) MAXR * 1024);
    part_ = A((size_t) MAXR * 32 * 9 * 130);
    attn_ = A((size_t) MAXR * 4096);
    gt_ = A((size_t) MAXR * FF);
    up_ = A((size_t) MAXR * FF);
    hh_ = A((size_t) MAXR * FF);
    logits_ = A((size_t) (MAXR - 1) * nv_);
    selg_ = A((size_t) MAXR * 256);
    cand_ = A((size_t) MAXR * TOPK);
    unary_ = A((size_t) MAXR * TOPK);
    lat_ = A((size_t) MAXR * D);
    d2t_ = A((size_t) nv_);
    sp_ = A(RING);
    ringk_ = cx_->alloc((size_t) NL * 8 * RING * 128 * 2);
    ringv_ = cx_->alloc((size_t) NL * 8 * RING * 128 * 2);
    nk_ = cx_->alloc((size_t) NL * 8 * MAXR * 128 * 2);
    nv_buf_ = cx_->alloc((size_t) NL * 8 * MAXR * 128 * 2);

    const int64_t* src = (const int64_t*) ((const uint8_t*) fm_.map + d2t->offset);
    int32_t* dst = d2t_.as<int32_t>();
    for (int i = 0; i < nv_; ++i) {
        dst[i] = (int32_t) src[i];
        if (src[i] < 0 || src[i] >= VOCAB) { err = "--dflash: native drafter: d2t maps outside the target vocab"; return false; }
    }
    if (mask_id_ < 0 || mask_id_ >= emb_->shape[1] || mask_id_ >= VOCAB) { err = "--dflash: native drafter: mask token id outside the vocab"; return false; }
    std::fill_n(sp_.as<int32_t>(), RING, -1);
    // rows 1.. of the block's input are the mask token's embedding; row 0 (the anchor) is written per draft
    const size_t rb = ref::row_bytes(emb_->type, D);
    ref::dequant(emb_->type, (const uint8_t*) fm_.map + emb_->offset + (size_t) mask_id_ * rb, x0_.as<float>() + D, D);
    for (int r = 2; r < MAXR; ++r) std::memcpy(x0_.as<float>() + (size_t) r * D, x0_.as<float>() + D, D * 4);
    return true;
}

const mtl::Pipeline* NativeDflash::P(const std::string& fn, int ntok) {
    const std::string key = ntok ? fn + "#" + std::to_string(ntok) : fn;
    auto it = pc_.find(key);
    if (it == pc_.end()) it = pc_.emplace(key, cx_->pipeline(lib_, fn, ntok ? mtl::Constants().i(0, ntok) : mtl::Constants())).first;
    return &it->second;
}

void NativeDflash::push(mtl::Dispatch d, bool concurrent) {
    d.barrier = !concurrent;
    mtl::require_fit(d, "dflash dispatch");
    ds_.push_back(std::move(d));
}

// y[t][r] = W[r] . x[t] for t < ntok: x rows of n_in floats from x + x_off, y rows y_stride apart from y + y_off.
// Tiles as Window::gemv picks them for Q8_0.
void NativeDflash::gemv(const gguf::Tensor& t, const mtl::Buffer& x, size_t x_off, const mtl::Buffer& y, size_t y_off,
                        int y_stride, int ntok, bool concurrent) {
    const int n_in = (int) t.shape[0], n_out = (int) t.shape[1];
    int L = 16, R = 2, S = 4, K = 1;   // R = 2: two rows share each x load (measured 4.4 vs 5.7 ms a draft(3), shared GPU)
    // fc (12800 in): 32,2,4,2 (32,1,8,2's sums in the same order, 0.07 ms less an inject(3)); ffn_down (7680 in) keeps
    // the 16,2,4,1 default (0.43 ms less a draft(3) than 32,1,8,2): M4 Max, idle GPU, interleaved, min of 300
    if (n_in >= 12800) { L = 32; R = 2; S = 4; K = 2; }
    else if (n_out <= 640) { L = 32; R = 1; }
    const std::string fn = "gemv_8_" + std::to_string(L) + "_" + std::to_string(R) + "_" + std::to_string(S) + "_" + std::to_string(K);
    mtl::Dispatch d;
    d.p = P(fn, ntok);
    const GemvArgs a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) ntok, (uint32_t) ref::row_bytes(Q8, n_in), (uint32_t) n_in,
                     (uint32_t) y_stride};
    d.args = {mtl::Arg::B(wm_, t.offset), mtl::Arg::B(x, x_off), mtl::Arg::B(x, x_off), mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
    const int rpt = (S / K) * (32 / L) * R;
    d.grid[0] = (n_out + rpt - 1) / rpt;
    d.tg[0] = S * 32;
    push(std::move(d), concurrent);
}

void NativeDflash::kern(const char* fn, std::vector<mtl::Arg> args, uint32_t g0, uint32_t g1, uint32_t tg, bool concurrent) {
    mtl::Dispatch d;
    d.p = P(fn);
    d.args = std::move(args);
    d.grid[0] = g0;
    d.grid[1] = g1;
    d.tg[0] = tg;
    push(std::move(d), concurrent);
}

// one group of n <= 8 feature rows (stage rows from stage_off bytes) at positions pos0..: k/v of every layer into the rings
void NativeDflash::enc_inject(size_t stage_off, int n, int pos0) {
    using mtl::Arg;
    gemv(*fc_, stage_, stage_off, g_, 0, D, n);
    kern("df_rms", {Arg::B(g_), Arg::B(wm_, enc_norm_), Arg::B(g_)}, n, 1, 256);
    for (int il = 0; il < NL; ++il) {
        gemv(*ly_[il].k, g_, 0, kraw_, (size_t) il * MAXR * 1024 * 4, 1024, n, il > 0);
        gemv(*ly_[il].v, g_, 0, vraw_, (size_t) il * MAXR * 1024 * 4, 1024, n, true);
    }
    const QkvArgs qa{pos0, n};
    for (int il = 0; il < NL; ++il)
        kern("df_qkv_inject",
             {Arg::B(kraw_, (size_t) il * MAXR * 1024 * 4), Arg::B(vraw_, (size_t) il * MAXR * 1024 * 4), Arg::B(wm_, ly_[il].k_norm),
              Arg::B(ringk_, (size_t) il * 8 * RING * 128 * 2), Arg::B(ringv_, (size_t) il * 8 * RING * 128 * 2), Arg::V(qa)},
             16, n, 128, il > 0);
}

// the block of nb rows (row 0's embedding already in x0_) at positions pos0..
void NativeDflash::enc_draft(int nb, int pos0, int32_t anchor) {
    using mtl::Arg;
    const QkvArgs qa{pos0, nb};
    kern("df_rms", {Arg::B(x0_), Arg::B(wm_, ly_[0].attn_norm), Arg::B(hn_)}, nb, 1, 256);
    for (int il = 0; il < NL; ++il) {
        const Layer& l = ly_[il];
        const mtl::Buffer& inp = il == 0 ? x0_ : hres_[(il - 1) & 1];
        const size_t nkv = (size_t) il * 8 * MAXR * 128 * 2, ring = (size_t) il * 8 * RING * 128 * 2;
        gemv(*l.acp, hn_, 0, dyna_, 0, 640, nb);
        kern("df_conv", {Arg::B(hn_), Arg::B(dyna_), Arg::B(wm_, l.abase), Arg::B(hc_)}, 10, nb, 256);
        gemv(*l.q, hc_, 0, qraw_, 0, 4096, nb);
        gemv(*l.k, hc_, 0, kr_, 0, 1024, nb, true);
        gemv(*l.v, hc_, 0, vr_, 0, 1024, nb, true);
        kern("df_qkv_decode",
             {Arg::B(qraw_), Arg::B(kr_), Arg::B(vr_), Arg::B(wm_, l.q_norm), Arg::B(wm_, l.k_norm), Arg::B(qn_), Arg::B(nk_, nkv),
              Arg::B(nv_buf_, nkv), Arg::V(qa)},
             48, nb, 128);
        kern("df_attn",
             {Arg::B(qn_), Arg::B(ringk_, ring), Arg::B(ringv_, ring), Arg::B(nk_, nkv), Arg::B(nv_buf_, nkv), Arg::B(sp_), Arg::B(part_),
              Arg::V(qa)},
             8, 9, 256);
        kern("df_attn_merge", {Arg::B(part_), Arg::B(attn_)}, nb * 32, 1, 128);
        gemv(*l.o, attn_, 0, o_, 0, D, nb);
        kern("df_conv_res_rms",
             {Arg::B(o_), Arg::B(dyna_), Arg::B(wm_, l.abase), Arg::B(inp), Arg::B(ffninp_), Arg::B(wm_, l.ffn_norm), Arg::B(fn_)}, nb, 1,
             256);
        gemv(*l.fcp, fn_, 0, dynf_, 0, 640, nb);
        kern("df_conv", {Arg::B(fn_), Arg::B(dynf_), Arg::B(wm_, l.fbase), Arg::B(fc_o_)}, 10, nb, 256);
        gemv(*l.gate, fc_o_, 0, gt_, 0, FF, nb);
        gemv(*l.up, fc_o_, 0, up_, 0, FF, nb, true);
        const uint32_t total = (uint32_t) nb * FF;
        kern("df_swiglu", {Arg::B(gt_), Arg::B(up_), Arg::B(hh_), Arg::V(total)}, (total + 255) / 256, 1, 256);
        gemv(*l.down, hh_, 0, m_, 0, D, nb);
        const bool last = il == NL - 1;
        kern("df_conv_res_rms",
             {Arg::B(m_), Arg::B(dynf_), Arg::B(wm_, l.fbase), Arg::B(ffninp_), Arg::B(hres_[il & 1]),
              Arg::B(wm_, last ? out_norm_ : ly_[il + 1].attn_norm), Arg::B(last ? hfin_ : hn_)},
             nb, 1, 256);
    }
    // the draft head and the selector gate on block rows 1..
    gemv(*head_, hfin_, (size_t) D * 4, logits_, 0, nv_, nb - 1);
    gemv(*selh_, hfin_, (size_t) D * 4, selg_, 0, 256, nb - 1, true);
    const uint32_t nvv = (uint32_t) nv_;
    kern("df_topk", {Arg::B(logits_), Arg::B(d2t_), Arg::B(cand_), Arg::B(unary_), Arg::V(nvv)}, nb - 1, 1, 1024);
    const SelArgs sa{anchor, (uint32_t) pre_->offset, (uint32_t) next_->offset};
    kern("df_selector", {Arg::B(wm_), Arg::B(cand_), Arg::B(unary_), Arg::B(selg_), Arg::B(lat_), Arg::V(sa)}, nb - 1, 1, 256);
}

void NativeDflash::trim(int pos) {
    if (pos_max_ < 0) return;
    int32_t* sp = sp_.as<int32_t>();
    if (pos <= 0) {
        std::fill_n(sp, RING, -1);
        pos_max_ = -1;
        return;
    }
    if (pos_max_ - pos + 1 > RING - 2048) {   // an overwritten in-window position is gone: the ring cannot be rewound this far
        std::fill_n(sp, RING, -1);
        pos_max_ = -1;
        return;
    }
    // slots whose position is rewound are forgotten (an older position they overwrote is out of the window while the rewind is at most RING - 2048 = 16)
    for (int p = std::max(pos, pos_max_ - RING + 1); p <= pos_max_; ++p)
        if (sp[p % RING] == p) sp[p % RING] = -1;
    pos_max_ = std::min(pos_max_, pos - 1);
}

bool NativeDflash::inject(const float* feats, int n, int pos0, std::string&) {
    if (n <= 0) return true;
    if (pos0 <= pos_max_) trim(pos0);
    else if (pos_max_ >= 0 && pos0 != pos_max_ + 1) trim(0);   // a gap: positions it never saw
    int32_t* sp = sp_.as<int32_t>();
    for (int off = 0; off < n; off += STAGE_ROWS) {
        const int c = std::min(STAGE_ROWS, n - off);
        std::memcpy(stage_.ptr, feats + (size_t) off * FEAT, (size_t) c * FEAT * sizeof(float));
        ds_.clear();
        for (int g = 0; g < c; g += MAXR) enc_inject((size_t) g * FEAT * 4, std::min(MAXR, c - g), pos0 + off + g);
        for (int i = 0; i < c; ++i) sp[(pos0 + off + i) % RING] = pos0 + off + i;
        cx_->run_graph(ds_);
    }
    pos_max_ = pos0 + n - 1;
    return true;
}

int NativeDflash::draft(int32_t id_last, int pos0, int max_n, float p_min, int32_t* out, float* conf) {
    if (id_last < 0 || id_last >= emb_->shape[1] || id_last >= VOCAB) return 0;
    if (max_n <= 0 || pos_max_ != pos0 - 1) return 0;   // the features up to id_last's position are not all in
    const int nb = std::min(block_, max_n + 1);
    const size_t rb = ref::row_bytes(emb_->type, D);
    ref::dequant(emb_->type, (const uint8_t*) fm_.map + emb_->offset + (size_t) id_last * rb, x0_.as<float>(), D);
    ds_.clear();
    enc_draft(nb, pos0, id_last);
    cx_->run_graph(ds_);
    const float* lat = lat_.as<float>();
    int n = 0, pred = 0;
    for (int i = 1; i < nb; ++i) {
        const float* row = lat + (size_t) i * D;
        const float* sc = row + TOPK + (size_t) pred * TOPK;
        pred = (int) (std::max_element(sc, sc + TOPK) - sc);
        float sum = 0.0f;   // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
        for (int k = 0; k < TOPK; ++k) sum += std::exp(sc[k] - sc[pred]);
        if (1.0f / sum < p_min) break;
        if (conf) conf[n] = 1.0f / sum;
        out[n++] = (int32_t) row[pred];
    }
    return n;
}

}  // namespace

std::unique_ptr<DflashBackend> make_dflash_native() { return std::make_unique<NativeDflash>(); }

}  // namespace strata::spec
