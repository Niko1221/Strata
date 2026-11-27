// engine/metal/src/hy4flash_gguf/mtp.cpp - see mtp.h.
//
// The block's arithmetic is the trunk's (forward.cpp encode_block), node for node, with the hyper-connections replaced
// by the plain residual the MTP block has: hc_pre is "cur = state" and hc_post is "state += out".  The kernels and
// their arg structs are the trunk's; the structs are repeated here (they mirror msl/hy4_flash_*.metal, like forward.cpp's).
#include "mtp.h"

#include "attn.h"
#include "dispatch.h"
#include "ref_quant.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace strata::hy4_flash {
namespace {

struct NormArgs {
    uint32_t n, x_stride, y_stride;
    float eps;
};
struct RopeArgs {
    uint32_t n_rot, n_head, head_dim, off, x_stride, y_stride;
    float theta_scale;
};
struct GluArgs {
    uint32_t n, g_stride, u_stride, y_stride;
    float limit;
};
struct RouterArgs {
    uint32_t n_expert, topk, l_stride, id_stride, w_stride, n_tok;
    float w_scale, floor;
};
struct IdxWeightArgs {
    uint32_t n, x_stride, y_stride;
    float denom;
};
struct KpoolArgs {
    uint32_t dim, kpool, g0, n_group, iv_stride, ig_stride, ape_stride, out_stride, out16_stride, n_rot;
    float eps, theta_scale;
};
struct ScoreArgs {
    uint32_t dim, n_head, n_pool, n_tok, kpool, q_stride, pool_stride, w_stride, s_stride;
};
struct TopkArgs {
    uint32_t kpool, top_k, n_pool, n_tok, s_stride, sel_stride;
};
struct AttnArgs {
    uint32_t n_head, d_qk, d_v, kv_stride, q_stride, o_stride, sel_stride, sc_stride, window;
    float kq_scale;
};
struct ArgmaxArgs {
    uint32_t n, stride;
};
struct ElemArgs {
    uint32_t n, a_stride, b_stride, y_stride;
};

constexpr int kSelCap = Forward::kSelCap;

/// HY4_FLASH_MTP_ONE_CB (default 1): a draft step's full row in one command buffer with the expert load at a cut (Mtp::step).
/// 0 = two command buffers instead.  Either way the same dispatches and the same values.
bool one_cb() {
    static const bool on = [] {
        const char* e = std::getenv("HY4_FLASH_MTP_ONE_CB");
        return e == nullptr || std::strcmp(e, "0") != 0;
    }();
    return on;
}

}  // namespace

// ---- the expert pool's bookkeeping ---------------------------------------------------------------------------------
void ExpertLru::reset(int slots, int n_experts) {
    n_slots_ = slots;
    clock_ = 0;
    hits = misses = 0;
    slot_of_.assign((size_t) std::max(n_experts, 0), -1);
    expert_.assign((size_t) std::max(slots, 0), -1);
    used_.assign((size_t) std::max(slots, 0), 0);
}

bool ExpertLru::place(const std::vector<int32_t>& want, std::vector<int>& slot,
                      std::vector<std::pair<int, int32_t>>& load) {
    slot.assign(want.size(), -1);
    load.clear();
    if ((int) want.size() > n_slots_) return false;
    const uint64_t now = ++clock_;
    // pass 1: the resident ones, stamped now so pass 2 cannot evict them
    for (size_t i = 0; i < want.size(); ++i) {
        const int32_t e = want[i];
        if (e < 0 || (size_t) e >= slot_of_.size()) return false;
        const int s = slot_of_[(size_t) e];
        if (s >= 0) {
            slot[i] = s;
            used_[(size_t) s] = now;
            ++hits;
        }
    }
    // pass 2: the rest into the least recently used slots (never one stamped now)
    for (size_t i = 0; i < want.size(); ++i) {
        if (slot[i] >= 0) continue;
        int best = -1;
        for (int s = 0; s < n_slots_; ++s) {
            if (used_[(size_t) s] == now) continue;
            if (best < 0 || used_[(size_t) s] < used_[(size_t) best]) best = s;
        }
        if (best < 0) return false;
        if (expert_[(size_t) best] >= 0) slot_of_[(size_t) expert_[(size_t) best]] = -1;
        expert_[(size_t) best] = want[i];
        slot_of_[(size_t) want[i]] = best;
        used_[(size_t) best] = now;
        slot[i] = best;
        load.push_back({best, want[i]});
        ++misses;
    }
    return true;
}

bool ExpertLru::preload(const std::vector<int32_t>& ranked, std::vector<std::pair<int, int32_t>>& load) {
    load.clear();
    for (int32_t e : expert_)
        if (e >= 0) return false;
    // the WHOLE ranking is checked, not only the prefix that fits: a profile's validity must not depend on the pool size
    std::vector<char> seen(slot_of_.size(), 0);
    for (int32_t e : ranked) {
        if (e < 0 || (size_t) e >= slot_of_.size() || seen[(size_t) e]) return false;   // out of range, or a repeat
        seen[(size_t) e] = 1;
    }
    const int n = std::min<int>(n_slots_, (int) ranked.size());
    for (int i = 0; i < n; ++i) {
        slot_of_[(size_t) ranked[(size_t) i]] = i;
        expert_[(size_t) i] = ranked[(size_t) i];
        used_[(size_t) i] = (uint64_t) (n - i);   // rank 0 the most recent; every later use is stamped above n
        load.push_back({i, ranked[(size_t) i]});
    }
    clock_ = std::max<uint64_t>(clock_, (uint64_t) n);
    return true;
}

// ---- the expert reads' threads ------------------------------------------------------------------------------------
/// The draft step's preads on threads that live as long as the module, instead of creating io_threads threads per
/// draft step.  `run` hands out the jobs and returns when all are done; false = a read failed.
struct ReadPool {
    struct Job {
        uint8_t* dst;
        uint64_t off, len;
    };
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    const std::vector<Job>* jobs = nullptr;
    size_t next = 0, left = 0;
    uint64_t gen = 0;
    bool stop = false, ok = true;
    int fd = -1;

    void start(int n, int fd_) {
        fd = fd_;
        for (int i = 0; i < n; ++i) th.emplace_back([this] { loop(); });
    }
    ~ReadPool() {
        {
            std::lock_guard<std::mutex> g(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto& x : th) x.join();
    }
    void loop() {
        std::unique_lock<std::mutex> g(mu);
        for (;;) {
            cv.wait(g, [&] { return stop || (jobs != nullptr && next < jobs->size()); });
            if (stop) return;
            const Job j = (*jobs)[next++];
            g.unlock();
            bool good = true;
            for (uint64_t done = 0; done < j.len;) {
                const ssize_t r = pread(fd, j.dst + done, (size_t) (j.len - done), (off_t) (j.off + done));
                if (r <= 0) { good = false; break; }
                done += (uint64_t) r;
            }
            g.lock();
            if (!good) ok = false;
            if (--left == 0) done_cv.notify_all();
        }
    }
    bool run(const std::vector<Job>& js) {
        if (js.empty()) return true;
        std::unique_lock<std::mutex> g(mu);
        jobs = &js;
        next = 0;
        left = js.size();
        ok = true;
        cv.notify_all();
        done_cv.wait(g, [&] { return left == 0; });
        jobs = nullptr;
        return ok;
    }
};

// ---- the block -----------------------------------------------------------------------------------------------------
struct Mtp::Impl {
    Model* m = nullptr;
    mtl::Context* cx = nullptr;
    int lib = -1;
    int C = 1024;            // rows a cache-only command buffer holds
    int rows_cap = 0;        // rows the state holds (max_context + a draft chain)
    int32_t n = 0, vocab = 0, kv_lora = 0, rope_dim = 0, kv_w = 0, qk = 0, v_mla = 0, H = 0, ql = 0;
    int32_t topk = 0, n_experts = 0, latent = 0, eff = 0, idx_dim = 0, idx_heads = 0, kpool = 0, idx_topk = 0;
    float eps = 0, tscale = 0, clamp = 0;
    int pools_done = 0;

    mtl::Pipeline p_rms_w, p_rope, p_glu, p_router, p_idx_w, p_idx_kpool, p_idx_score, p_idx_topk, p_argmax,
        p_copy, p_store_half, p_add, p_gate_mul;
    AttnPipes attn_sel;   // attn.h: the score row, the softmax, PV
    std::map<std::array<uint64_t, 3>, DenseGemvF32> gemv_;
    MoeF32 moe;

    enum {
        ENORM, HNORM, EH, SHN, ATTN_NORM, Q_A, Q_A_NORM, Q_B, KV_A, KV_A_NORM, K_B, V_B, SINKS, AGATE, AOUT, POST_ATTN,
        FFN_NORM, POST_FFW, GATE_INP, PROBS_B, RDOWN, RUP, SH_G, SH_U, SH_D, IQ_B, IK, IK_NORM, IPROJ, ICV, ICG, IAPE,
        ICNORM, OUT, EMBD, W_COUNT
    };
    const gguf::Tensor* t[W_COUNT] = {nullptr};

    mtl::Buffer hist, kv, idxc, pool32, pos, pos1, emb, joined, state, anorm, kvp, kvn, ik, iv, ig;
    mtl::Buffer q_resid, qb, Q, attn_o, kqv, gproj, gated, attn_y, attn_out, iq, iwraw, iw, score, sel, nsel, sc;
    mtl::Buffer fnorm, mlogits, ids, rw, rdown, xp, hbuf, hp, ye, mlat, upo, shg, shu, sha, sho, ffn_sum, fout, snorm;
    mtl::Buffer logits, amax, plan, badflag;
    size_t n_pool_max = 0;

    // the expert pool
    mtl::Buffer pool;
    ExpertLru lru;
    uint64_t slot_bytes = 0, sl_off[3] = {0, 0, 0}, sl_bytes[3] = {0, 0, 0};
    uint64_t file_off[3] = {0, 0, 0};
    int fd = -1;
    int io_threads = 8;
    std::unique_ptr<ReadPool> rpool;   // HY4_FLASH_MTP_IO_POOL=0: threads per step, as before

    const DenseGemvF32& gv(const gguf::Tensor* w) {
        const uint32_t n_in = (uint32_t) w->shape[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < w->shape.size(); ++i) rows *= w->shape[i];
        return gv(w->type, n_in, (uint32_t) rows);
    }
    const DenseGemvF32& gv(int type, uint32_t n_in, uint32_t n_out) {
        const std::array<uint64_t, 3> k{(uint64_t) type, n_in, n_out};
        auto it = gemv_.find(k);
        if (it == gemv_.end())
            it = gemv_.emplace(k, DenseGemvF32(*cx, lib, type, n_in, n_out, (uint32_t) ref::row_bytes(type, (int) n_in)))
                     .first;
        return it->second;
    }
    /// One head's shape of a per-head weight, all heads in one dispatch (DenseGemvF32::append_heads - bitwise the
    /// per-head dispatches; forward.cpp does the same for the trunk's wk_b / wv_b).
    std::map<std::array<uint64_t, 6>, DenseGemvF32> gemvh_;
    const DenseGemvF32& gvh(int type, uint32_t n_in, uint32_t n_out, GemvHeads h) {
        const std::array<uint64_t, 6> k{(uint64_t) type, n_in, n_out, h.w, h.x, h.y};
        auto it = gemvh_.find(k);
        if (it == gemvh_.end())
            it = gemvh_
                     .emplace(k, DenseGemvF32(*cx, lib, type, n_in, n_out,
                                              (uint32_t) ref::row_bytes(type, (int) n_in), h))
                     .first;
        return it->second;
    }
    mtl::Arg W(int slot, size_t extra = 0) const { return mtl::Arg::B(*wr[slot].buf, wr[slot].off + extra); }
    /// Where a weight is read from: the resident GGUF bytes, or (HY4_FLASH_MTP_QUANT / HY4_FLASH_MTP_HEAD) a quantized copy.
    struct WRef {
        const mtl::Buffer* buf = nullptr;
        uint64_t off = 0;
        int type = 0;
    };
    WRef wr[64];
    const DenseGemvF32& gvs(int slot) {
        const gguf::Tensor* w = t[slot];
        uint64_t rows = 1;
        for (size_t i = 1; i < w->shape.size(); ++i) rows *= w->shape[i];
        return gv(wr[slot].type, (uint32_t) w->shape[0], (uint32_t) rows);
    }
    // the quantized copies: the block's dense matrices and the MTP head; the experts come from a file
    mtl::Buffer qdense, qhead;
    int quant = 30;         // the experts' and dense matrices' type: 30 = as in the GGUF, 8 = Q8_0
    int head_type = 30;     // the MTP head: 30 = the trunk's output.weight itself, 8 = a Q8_0 copy
    std::string quantize();
    static constexpr uint64_t kQ8Header = 4096;
    uint64_t q8_matrix_bytes(int k) const {   // gate / up: [eff rows][latent], down: [latent rows][eff]
        return k < 2 ? (uint64_t) eff * ref::row_bytes(8, latent) : (uint64_t) latent * ref::row_bytes(8, eff);
    }
    std::string expert_file_q8(std::string& err);

    void rms(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, size_t x_off, uint32_t x_stride, int w,
             const mtl::Buffer& y, size_t y_off, uint32_t y_stride, uint32_t nn, int rows) {
        mtl::Dispatch d;
        d.p = &p_rms_w;
        NormArgs a{nn, x_stride, y_stride, eps};
        d.args = {mtl::Arg::B(x, x_off), W(w), mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
        d.grid[0] = (uint32_t) rows;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_rmsnorm (mtp)");
        ds.push_back(d);
    }
    void rope(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, uint32_t n_head, uint32_t head_dim, uint32_t off,
              uint32_t stride, const mtl::Buffer& pb, int rows) {
        mtl::Dispatch d;
        d.p = &p_rope;
        RopeArgs a{(uint32_t) rope_dim, n_head, head_dim, off, stride, stride, tscale};
        d.args = {mtl::Arg::B(x), mtl::Arg::B(x), mtl::Arg::B(pb), mtl::Arg::V(a)};
        d.grid[0] = (uint32_t) rope_dim / 2;
        d.grid[1] = n_head;
        d.grid[2] = (uint32_t) rows;
        d.tg[0] = 32;
        mtl::require_fit(d, "hy4_flash_rope_norm (mtp)");
        ds.push_back(d);
    }
    void elem(std::vector<mtl::Dispatch>& ds, const mtl::Pipeline& p, std::vector<mtl::Arg> args, uint32_t nn,
              int rows, const char* where) {
        mtl::Dispatch d;
        d.p = &p;
        d.args = std::move(args);
        d.tg[0] = 256;
        d.grid[0] = (nn + 255) / 256;
        d.grid[1] = (uint32_t) rows;
        mtl::require_fit(d, where);
        ds.push_back(d);
    }

    std::string open(Model& mm, mtl::Context& c, int lib_, int chunk);
    void encode_rows(std::vector<mtl::Dispatch>& ds, int q0, int T);
    void encode_full_a(std::vector<mtl::Dispatch>& ds, int q, int r);
    void encode_full_b(std::vector<mtl::Dispatch>& ds, int r);
    void encode_full_b0(std::vector<mtl::Dispatch>& ds);          // rdown and the shared expert: no pool, no plan
    void encode_full_b1(std::vector<mtl::Dispatch>& ds, int r);   // the routed experts onwards
    std::string embed(const int32_t* tokens, int q0, int T);
    std::string load_experts(MtpStats& st);
    std::string read_slots(const std::vector<std::pair<int, int32_t>>& load);
    std::string preload_pool();
    bool preload_pending = false;
    std::string preload_now() {
        if (!preload_pending) return {};
        preload_pending = false;
        return preload_pool();
    }
};

std::string Mtp::Impl::open(Model& mm, mtl::Context& c, int lib_, int chunk) {
    m = &mm;
    cx = &c;
    lib = lib_;
    C = std::max(8, chunk);
    const hy4_flash::Geometry& g = mm.geo;
    if (g.n_mtp_blocks != 1) return "this model has " + std::to_string(g.n_mtp_blocks) + " MTP blocks, not 1";
    const int L = g.trunk_blocks;
    const hy4_flash::LayerSpec& s = g.block(L);
    if (!s.has_indexer || s.attention != hy4_flash::Attention::Sparse || s.expert_ordinal < 0)
        return "the MTP block is not a sparse MoE block with an indexer, which this implementation assumes";
    n = g.hidden_dim;
    vocab = g.vocab_size;
    kv_lora = g.kv_lora_rank;
    rope_dim = g.rope_dim;
    kv_w = g.kv_a_width;
    qk = g.qk_head_dim;
    v_mla = g.v_head_dim;
    H = s.n_heads;
    ql = g.q_lora_rank;
    topk = g.top_k;
    n_experts = g.n_experts;
    latent = g.expert_dim;
    eff = g.expert_ff;
    idx_dim = g.idx_key_dim;
    idx_heads = g.idx_heads;
    kpool = g.idx_kpool;
    idx_topk = g.idx_top_k;
    eps = g.rms_norm_eps;
    tscale = std::pow(g.rope_theta_sparse, -2.0f / (float) g.rope_dim);
    clamp = g.swiglu_clamp;
    rows_cap = g.max_context + 16;

    const std::string p = "blk." + std::to_string(L) + ".";
    std::string missing;
    auto need = [&](int slot, const std::string& name) {
        const gguf::Tensor* w = m->find(name);
        if (w == nullptr && missing.empty()) missing = "no tensor " + name;
        // a tensor of the skip list has offset 0 in the dense buffer: it was never read (model.h)
        if (w != nullptr && std::find(m->skipped.begin(), m->skipped.end(), name) != m->skipped.end() &&
            m->on_file.count(name) == 0 && missing.empty())
            missing = name + " is not resident: open the model with the MTP block's dense tensors (with_mtp)";
        t[slot] = w;
    };
    need(ENORM, p + "nextn.enorm.weight");
    need(HNORM, p + "nextn.hnorm.weight");
    need(EH, p + "nextn.eh_proj.weight");
    need(SHN, p + "nextn.shared_head_norm.weight");
    need(ATTN_NORM, p + "attn_norm.weight");
    need(Q_A, p + "attn_q_a.weight");
    need(Q_A_NORM, p + "attn_q_a_norm.weight");
    need(Q_B, p + "attn_q_b.weight");
    need(KV_A, p + "attn_kv_a_mqa.weight");
    need(KV_A_NORM, p + "attn_kv_a_norm.weight");
    need(K_B, p + "attn_k_b.weight");
    need(V_B, p + "attn_v_b.weight");
    need(SINKS, p + "attn_sinks.weight");
    need(AGATE, p + "attn_gate.weight");
    need(AOUT, p + "attn_output.weight");
    need(POST_ATTN, p + "post_attention_norm.weight");
    need(FFN_NORM, p + "ffn_norm.weight");
    need(POST_FFW, p + "post_ffw_norm.weight");
    need(GATE_INP, p + "ffn_gate_inp.weight");
    need(PROBS_B, p + "exp_probs_b.bias");
    need(RDOWN, p + "ffn_routed_down.weight");
    need(RUP, p + "ffn_routed_up.weight");
    need(SH_G, p + "ffn_gate_shexp.weight");
    need(SH_U, p + "ffn_up_shexp.weight");
    need(SH_D, p + "ffn_down_shexp.weight");
    need(IQ_B, p + "indexer.attn_q_b.weight");
    need(IK, p + "indexer.attn_k.weight");
    need(IK_NORM, p + "indexer.k_norm.weight");
    need(IPROJ, p + "indexer.proj.weight");
    need(ICV, p + "indexer_compressor_value.weight");
    need(ICG, p + "indexer_compressor_gate.weight");
    need(IAPE, p + "indexer_compressor_ape.weight");
    need(ICNORM, p + "indexer_compressor_norm.weight");
    need(OUT, "output.weight");
    need(EMBD, "token_embd.weight");
    if (!missing.empty()) return missing;
    if (t[EH]->shape[0] != (uint64_t) (2 * n) || t[EH]->shape[1] != (uint64_t) n)
        return "nextn.eh_proj is not [2 x hidden, hidden]";
    for (int i = 0; i < W_COUNT; ++i) wr[i] = WRef{&m->dense, t[i]->offset, t[i]->type};
    // Q8_0 by default; "bf16" runs the block as the GGUF stores it
    auto env_type = [](const char* name) {
        const char* e = std::getenv(name);
        return e != nullptr && std::strcmp(e, "bf16") == 0 ? 30 : 8;
    };
    quant = env_type("HY4_FLASH_MTP_QUANT");
    head_type = env_type("HY4_FLASH_MTP_HEAD");
    if (const std::string e = quantize(); !e.empty()) return e;

    p_rms_w = cx->pipeline(lib, "hy4_flash_rmsnorm", mtl::Constants().b(40, true));
    p_rope = cx->pipeline(lib, "hy4_flash_rope_norm");
    p_glu = cx->pipeline(lib, "hy4_flash_swiglu_clamp");
    p_router = cx->pipeline(lib, "hy4_flash_router");
    p_idx_w = cx->pipeline(lib, "hy4_flash_indexer_weights");
    p_idx_kpool = cx->pipeline(lib, "hy4_flash_indexer_kpool");
    p_idx_score = cx->pipeline(lib, "hy4_flash_indexer_score");
    p_idx_topk = cx->pipeline(lib, "hy4_flash_indexer_topk");
    attn_sel = attn_pipes(cx, lib, true);
    p_argmax = cx->pipeline(lib, "hy4_flash_argmax");
    p_copy = cx->pipeline(lib, "hy4_flash_copy_f32");
    p_store_half = cx->pipeline(lib, "hy4_flash_store_half");
    p_add = cx->pipeline(lib, "hy4_flash_add_f32");
    p_gate_mul = cx->pipeline(lib, "hy4_flash_gate_mul");

    // the experts: BF16 gate / up / down, one slot holding all three
    const ExpertSlices& es = m->experts[(size_t) s.expert_ordinal];
    for (int k = 0; k < 3; ++k)
        if (es.type[k] != 30) return "the MTP block's experts are not BF16, which is what this implementation reads";
    moe = MoeF32(*cx, lib, quant, quant, (uint32_t) latent, (uint32_t) eff);
    std::string expert_file = m->shard1_path;
    if (quant == 8) {
        // Q8_0 experts from a derived file, [all gates][all ups][all downs] like the GGUF (expert_file_q8)
        std::string err;
        expert_file = expert_file_q8(err);
        if (!err.empty()) return err;
    }
    uint64_t at = 0;
    for (int k = 0; k < 3; ++k) {
        sl_off[k] = at;
        sl_bytes[k] = quant == 8 ? q8_matrix_bytes(k) : es.stride[k];
        file_off[k] = quant == 8 ? kQ8Header + (uint64_t) k * (uint64_t) n_experts * sl_bytes[k] : es.off[k];
        at += (sl_bytes[k] + 16383) / 16384 * 16384;
    }
    slot_bytes = at;
    double gb = 2.0;
    if (const char* e = std::getenv("HY4_FLASH_MTP_EXPERT_GB"))
        if (const double v = std::atof(e); v > 0) gb = v;
    int slots = (int) std::min<double>(n_experts, std::floor(gb * 1e9 / (double) slot_bytes));
    slots = std::max(slots, topk);
    if (const char* e = std::getenv("HY4_FLASH_MTP_IO_THREADS"))
        if (const int v = std::atoi(e); v > 0) io_threads = v;
    pool = cx->alloc((size_t) slots * slot_bytes);
    lru.reset(slots, n_experts);
    fd = ::open(expert_file.c_str(), O_RDONLY);
    if (fd < 0) return "cannot open " + expert_file;
    fcntl(fd, F_NOCACHE, 1);
    if (const char* e = std::getenv("HY4_FLASH_MTP_IO_POOL"); e == nullptr || std::strcmp(e, "0") != 0) {
        rpool = std::make_unique<ReadPool>();
        rpool->start(io_threads, fd);
    }
    // the pool's pre-fill waits for the first catch_up / step (preload_pending): by then the trunk's arena is filled
    // and locked.  Doing it here would make the pool's gigabytes resident BEFORE the arena fill - the same bytes,
    // earlier - and that alone is enough to push macOS into swap at a large arena.
    preload_pending = true;

    auto f32 = [&](size_t k) { return cx->alloc(4 * k); };
    hist = f32((size_t) (rows_cap + 1) * n);
    kv = cx->alloc(2 * (size_t) rows_cap * kv_w);
    idxc = cx->alloc(2 * (size_t) rows_cap * 3 * idx_dim);
    n_pool_max = (size_t) rows_cap / (size_t) kpool + 1;
    pool32 = f32(n_pool_max * (size_t) idx_dim);
    pos = cx->alloc(4 * (size_t) C);
    pos1 = cx->alloc(16);
    emb = f32((size_t) C * n);
    joined = f32((size_t) C * 2 * n);
    state = f32((size_t) C * n);
    anorm = f32((size_t) C * n);
    kvp = f32((size_t) C * kv_w);
    kvn = f32((size_t) C * kv_lora);
    ik = f32((size_t) C * idx_dim);
    iv = f32((size_t) C * idx_dim);
    ig = f32((size_t) C * idx_dim);
    q_resid = f32((size_t) ql);
    qb = f32((size_t) H * qk);
    Q = f32((size_t) H * kv_w);
    attn_o = f32((size_t) H * kv_lora);
    kqv = f32((size_t) H * v_mla);
    gproj = f32((size_t) H * v_mla);
    gated = f32((size_t) H * v_mla);
    attn_y = f32((size_t) n);
    attn_out = f32((size_t) n);
    iq = f32((size_t) idx_heads * idx_dim);
    iwraw = f32((size_t) idx_heads);
    iw = f32((size_t) idx_heads);
    score = f32(n_pool_max);
    sel = cx->alloc(4 * (size_t) kSelCap);
    nsel = cx->alloc(16);
    sc = f32((size_t) H * kSelCap);
    fnorm = f32((size_t) n);
    mlogits = f32((size_t) n_experts);
    ids = cx->alloc(4 * (size_t) topk);
    rw = f32((size_t) topk);
    rdown = f32((size_t) latent);
    xp = f32(16);
    hp = f32(16);
    hbuf = f32((size_t) topk * eff);
    ye = f32((size_t) topk * latent);
    mlat = f32((size_t) latent);
    upo = f32((size_t) n);
    shg = f32((size_t) eff);
    shu = f32((size_t) eff);
    sha = f32((size_t) eff);
    sho = f32((size_t) n);
    ffn_sum = f32((size_t) n);
    fout = f32((size_t) n);
    snorm = f32((size_t) n);
    logits = f32((size_t) vocab);
    amax = cx->alloc(16);
    plan = cx->alloc(4 * (size_t) (2 * topk + 1 + 3 * topk + 16));
    badflag = cx->alloc(16);
    std::fprintf(stderr,
                 "strata_hy4_flash_metal: MTP block %d: %d heads, its own MLA / indexer cache for %d rows, expert pool %d of %d "
                 "BF16 experts (%.2f GB, %.2f MB each, HY4_FLASH_MTP_EXPERT_GB), cache-only rows %d a command buffer\n",
                 L, H, rows_cap, slots, n_experts, (double) slots * (double) slot_bytes / 1e9,
                 (double) slot_bytes / 1e6, C);
    return {};
}

namespace {

/// BF16 rows -> Q8_0 with ggml's reference quantizer (ref::quantize_q8_0), `rows` rows of `n_in` values, on threads.
void bf16_to_q8(const uint16_t* src, uint64_t rows, uint32_t n_in, uint8_t* dst, int threads) {
    const size_t rb = ref::row_bytes(8, (int) n_in);
    std::atomic<uint64_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < threads; ++i)
        th.emplace_back([&] {
            std::vector<float> f(n_in);
            for (;;) {
                const uint64_t r0 = next.fetch_add(64);
                if (r0 >= rows) return;
                for (uint64_t r = r0; r < std::min(rows, r0 + 64); ++r) {
                    const uint16_t* row = src + r * n_in;
                    for (uint32_t j = 0; j < n_in; ++j) {
                        const uint32_t b = (uint32_t) row[j] << 16;
                        std::memcpy(&f[j], &b, 4);
                    }
                    ref::quantize_q8_0(f.data(), n_in, dst + r * rb);
                }
            }
        });
    for (auto& x : th) x.join();
}

}  // namespace

/// HY4_FLASH_MTP_QUANT (default q8_0, or bf16): every BF16 matrix of the block (not the 1-D norms, not the F32 router)
/// as a Q8_0 copy, the experts from a derived file; HY4_FLASH_MTP_HEAD (default q8_0, or bf16): a Q8_0 copy of output.weight
/// as the MTP's head (the trunk keeps its BF16 head).  Draft-only changes: they move the acceptance, never the
/// output - the verify is the trunk's.
std::string Mtp::Impl::quantize() {
    const int threads = std::max(4, (int) std::thread::hardware_concurrency() - 2);
    auto rows_of = [](const gguf::Tensor* w) {
        uint64_t r = 1;
        for (size_t i = 1; i < w->shape.size(); ++i) r *= w->shape[i];
        return r;
    };
    if (quant == 8) {
        std::vector<int> slots;
        uint64_t total = 0;
        for (int i = 0; i < W_COUNT; ++i) {
            if (i == OUT || i == EMBD || t[i]->type != 30 || t[i]->shape.size() < 2) continue;
            if (t[i]->shape[0] % 32 != 0) return std::string("MTP: ") + t[i]->name + " cannot be Q8_0 (row not 32-aligned)";
            slots.push_back(i);
            total += (rows_of(t[i]) * ref::row_bytes(8, (int) t[i]->shape[0]) + 255) / 256 * 256;
        }
        qdense = cx->alloc((size_t) total);
        uint64_t at = 0;
        for (int i : slots) {
            const uint32_t n_in = (uint32_t) t[i]->shape[0];
            // the BF16 source: resident, or read once from the file (model.h on_file: then it never was resident)
            std::vector<uint8_t> src((size_t) (rows_of(t[i]) * n_in * 2));
            if (!m->tensor_bytes(t[i]->name, 0, src.size(), src.data()))
                return std::string("MTP: cannot read ") + t[i]->name;
            bf16_to_q8((const uint16_t*) src.data(), rows_of(t[i]), n_in, qdense.as<uint8_t>() + at, threads);
            wr[i] = WRef{&qdense, at, 8};
            at += (rows_of(t[i]) * ref::row_bytes(8, (int) n_in) + 255) / 256 * 256;
        }
        std::fprintf(stderr, "strata_hy4_flash_metal: MTP block's %zu dense matrices as Q8_0 (%.1f MB)\n", slots.size(), total / 1e6);
    }
    if (head_type == 8) {
        const gguf::Tensor* o = t[OUT];
        if (o->type != 30) return "MTP: output.weight is not BF16, HY4_FLASH_MTP_HEAD=q8_0 expects it";
        const uint32_t n_in = (uint32_t) o->shape[0];
        qhead = cx->alloc((size_t) (rows_of(o) * ref::row_bytes(8, (int) n_in)));
        bf16_to_q8((const uint16_t*) ((const uint8_t*) m->dense.ptr + o->offset), rows_of(o), n_in, qhead.as<uint8_t>(),
                   threads);
        wr[OUT] = WRef{&qhead, 0, 8};
        std::fprintf(stderr, "strata_hy4_flash_metal: MTP head = a Q8_0 copy of output.weight (%.1f MB); the trunk's head is unchanged\n",
                     qhead.bytes / 1e6);
    }
    return {};
}

/// The Q8_0 experts' file: models/derived/<model dir>/mtp_experts_q8_0.bin, built here when it is missing or was made
/// from another source file (the header binds it to the shard's size and mtime).  [all gates][all ups][all downs],
/// expert e's matrix k at kQ8Header + (k * n_experts + e) * q8_matrix_bytes(k).
std::string Mtp::Impl::expert_file_q8(std::string& err) {
    const std::string& src = m->shard1_path;
    const size_t s1 = src.find_last_of('/');
    const std::string dir = src.substr(0, s1);
    const size_t s2 = dir.find_last_of('/');
    std::string path = dir.substr(0, s2) + "/derived/" + dir.substr(s2 + 1) + "/mtp_experts_q8_0.bin";
    if (const char* e = std::getenv("HY4_FLASH_MTP_Q8_FILE")) path = e;
    struct stat st{};
    if (::stat(src.c_str(), &st) != 0) { err = "cannot stat " + src; return {}; }
    uint64_t want[8] = {0x3051385150544D48ull /* "HMTPQ8Q0" */, 1, (uint64_t) n_experts, (uint64_t) latent,
                        (uint64_t) eff, (uint64_t) st.st_size,
                        (uint64_t) st.st_mtimespec.tv_sec * 1000000000ull + (uint64_t) st.st_mtimespec.tv_nsec, 0};
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        uint64_t got[8] = {};
        const bool ok = std::fread(got, 8, 8, f) == 8 && std::memcmp(got, want, sizeof want) == 0;
        std::fclose(f);
        if (ok) return path;
        std::fprintf(stderr, "strata_hy4_flash_metal: %s was made from another source; rebuilding\n", path.c_str());
    }
    const ExpertSlices& es = m->experts[(size_t) m->geo.block(m->geo.trunk_blocks).expert_ordinal];
    const std::string tmp = path + ".part";
    const int out = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) { err = "cannot write " + tmp + " (does the directory exist?)"; return {}; }
    fcntl(out, F_NOCACHE, 1);   // without it the whole file's dirty cache pushes this build into swap
    const int in = ::open(src.c_str(), O_RDONLY);
    fcntl(in, F_NOCACHE, 1);
    const double t0 = mtl::now_s();
    std::atomic<int> next{0};
    std::atomic<bool> ok{true};
    std::vector<std::thread> th;
    const int threads = std::max(4, (int) std::thread::hardware_concurrency() - 2);
    for (int i = 0; i < threads; ++i)
        th.emplace_back([&] {
            std::vector<uint16_t> bf;
            std::vector<uint8_t> q;
            std::vector<float> f;
            for (;;) {
                const int job = next++;
                if (job >= 3 * n_experts || !ok) return;
                const int k = job / n_experts, e = job % n_experts;
                const uint32_t n_in = k < 2 ? (uint32_t) latent : (uint32_t) eff;
                const uint64_t rows = k < 2 ? (uint64_t) eff : (uint64_t) latent;
                bf.resize((size_t) (rows * n_in));
                q.resize((size_t) q8_matrix_bytes(k));
                f.resize(n_in);
                const uint64_t bytes = 2 * rows * n_in;
                if (bytes != es.stride[k] ||
                    pread(in, bf.data(), (size_t) bytes, (off_t) (es.off[k] + (uint64_t) e * es.stride[k])) != (ssize_t) bytes) {
                    ok = false;
                    return;
                }
                const size_t rb = ref::row_bytes(8, (int) n_in);
                for (uint64_t r = 0; r < rows; ++r) {
                    for (uint32_t j = 0; j < n_in; ++j) {
                        const uint32_t b = (uint32_t) bf[(size_t) (r * n_in + j)] << 16;
                        std::memcpy(&f[j], &b, 4);
                    }
                    ref::quantize_q8_0(f.data(), n_in, q.data() + r * rb);
                }
                const off_t at = (off_t) (kQ8Header + ((uint64_t) k * (uint64_t) n_experts + (uint64_t) e) * q8_matrix_bytes(k));
                if (pwrite(out, q.data(), q.size(), at) != (ssize_t) q.size()) ok = false;
            }
        });
    for (auto& x : th) x.join();
    ::close(in);
    if (ok && pwrite(out, want, sizeof want, 0) != (ssize_t) sizeof want) ok = false;
    ::close(out);
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        err = "MTP: building " + path + " failed";
        return {};
    }
    std::fprintf(stderr, "strata_hy4_flash_metal: built %s (the MTP experts as Q8_0) in %.1f s\n", path.c_str(), mtl::now_s() - t0);
    return path;
}

/// The token embeddings of rows [q0, q0 + T) into `emb`: ref::dequant is ggml's own arithmetic (forward.cpp's
/// embed_window does the same for the trunk).
std::string Mtp::Impl::embed(const int32_t* tokens, int q0, int T) {
    const gguf::Tensor* te = t[EMBD];
    const size_t rb = ref::row_bytes(te->type, n);
    float* dst = emb.as<float>();
    std::vector<uint8_t> raw(rb);
    for (int i = 0; i < T; ++i) {
        const int32_t tok = tokens[q0 + i];
        if (tok < 0 || tok >= vocab) return "MTP: token id " + std::to_string(tok) + " is outside the vocabulary";
        if (!m->tensor_bytes("token_embd.weight", (uint64_t) tok * rb, rb, raw.data()))
            return "MTP: cannot read row " + std::to_string(tok) + " of token_embd.weight";
        ref::dequant(te->type, raw.data(), dst + (size_t) i * n, (size_t) n);
    }
    int32_t* pb = pos.as<int32_t>();
    for (int i = 0; i < T; ++i) pb[i] = q0 + i;
    return {};
}

/// Rows [q0, q0 + T), cache-only: eh_proj, attn_norm, the MLA latent and the indexer's key side into the caches,
/// and the pooled key of every group they complete.  The block's own `anorm` / `state` rows stay for a full row.
void Mtp::Impl::encode_rows(std::vector<mtl::Dispatch>& ds, int q0, int T) {
    const uint32_t uT = (uint32_t) T, un = (uint32_t) n;
    // [rms(embd) * enorm ; rms(h) * hnorm] -> eh_proj (llama.cpp's ggml_concat(enorm(tok), hnorm(h)))
    rms(ds, emb, 0, un, ENORM, joined, 0, 2 * un, un, T);
    rms(ds, hist, 4 * (size_t) q0 * n, un, HNORM, joined, 4 * (size_t) n, 2 * un, un, T);
    gvs(EH).append(ds, *wr[EH].buf, wr[EH].off, joined, 2 * un, state, un, 0, uT);
    rms(ds, state, 0, un, ATTN_NORM, anorm, 0, un, un, T);
    // the shared key: rope on its tail, the latent normed, both into the f16 cache
    gvs(KV_A).append(ds, *wr[KV_A].buf, wr[KV_A].off, anorm, un, kvp, (uint32_t) kv_w, 0, uT);
    rope(ds, kvp, 1, (uint32_t) kv_w, (uint32_t) kv_lora, (uint32_t) kv_w, pos, T);
    rms(ds, kvp, 0, (uint32_t) kv_w, KV_A_NORM, kvn, 0, (uint32_t) kv_lora, (uint32_t) kv_lora, T);
    const size_t kv_row = (size_t) q0 * (size_t) kv_w;
    {
        ElemArgs a{(uint32_t) kv_lora, (uint32_t) kv_lora, 0, (uint32_t) kv_w};
        elem(ds, p_store_half, {mtl::Arg::B(kvn), mtl::Arg::B(kv, 2 * kv_row), mtl::Arg::V(a), mtl::Arg::B(badflag)},
             a.n, T, "hy4_flash_store_half (mtp kv)");
        ElemArgs a2{(uint32_t) rope_dim, (uint32_t) kv_w, 0, (uint32_t) kv_w};
        elem(ds, p_store_half,
             {mtl::Arg::B(kvp, 4 * (size_t) kv_lora), mtl::Arg::B(kv, 2 * (kv_row + (size_t) kv_lora)),
              mtl::Arg::V(a2), mtl::Arg::B(badflag)},
             a2.n, T, "hy4_flash_store_half (mtp k_pe)");
    }
    // the indexer's key side
    gvs(IK).append(ds, *wr[IK].buf, wr[IK].off, anorm, un, ik, (uint32_t) idx_dim, 0, uT);
    rms(ds, ik, 0, (uint32_t) idx_dim, IK_NORM, ik, 0, (uint32_t) idx_dim, (uint32_t) idx_dim, T);
    gvs(ICV).append(ds, *wr[ICV].buf, wr[ICV].off, ik, (uint32_t) idx_dim, iv, (uint32_t) idx_dim, 0, uT);
    gvs(ICG).append(ds, *wr[ICG].buf, wr[ICG].off, ik, (uint32_t) idx_dim, ig, (uint32_t) idx_dim, 0, uT);
    const size_t irow = (size_t) q0 * 3ull * (size_t) idx_dim;
    ElemArgs ia{(uint32_t) idx_dim, (uint32_t) idx_dim, 0, 3u * (uint32_t) idx_dim};
    elem(ds, p_store_half, {mtl::Arg::B(iv), mtl::Arg::B(idxc, 2 * irow), mtl::Arg::V(ia), mtl::Arg::B(badflag)},
         ia.n, T, "hy4_flash_store_half (mtp indexer value)");
    elem(ds, p_store_half,
         {mtl::Arg::B(ig), mtl::Arg::B(idxc, 2 * (irow + (size_t) idx_dim)), mtl::Arg::V(ia), mtl::Arg::B(badflag)},
         ia.n, T, "hy4_flash_store_half (mtp indexer gate)");
    // pooled keys of every group these rows complete (pools_done was clamped to q0 / kpool by the caller)
    const int done = pools_done;
    const int have = (q0 + T) / kpool;
    if (have > done) {
        mtl::Dispatch d;
        d.p = &p_idx_kpool;
        KpoolArgs a{(uint32_t) idx_dim,
                    (uint32_t) kpool,
                    (uint32_t) done,
                    (uint32_t) (have - done),
                    3u * (uint32_t) idx_dim,
                    3u * (uint32_t) idx_dim,
                    (uint32_t) idx_dim,
                    (uint32_t) idx_dim,
                    3u * (uint32_t) idx_dim * (uint32_t) kpool,
                    (uint32_t) rope_dim,
                    eps,
                    tscale};
        const size_t pool16 = (size_t) done * (size_t) kpool * 3ull * (size_t) idx_dim + 2ull * (size_t) idx_dim;
        d.args = {mtl::Arg::B(idxc, 0),
                  mtl::Arg::B(idxc, 2 * (size_t) idx_dim),
                  W(IAPE),
                  W(ICNORM),
                  mtl::Arg::B(pool32, 4 * (size_t) done * (size_t) idx_dim),
                  mtl::Arg::B(idxc, 2 * pool16),
                  mtl::Arg::V(a)};
        d.grid[0] = (uint32_t) (have - done);
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_indexer_kpool (mtp)");
        ds.push_back(d);
    }
    pools_done = have;
}

/// The full row's first half, up to and including the router: row r of the window, at position q.
void Mtp::Impl::encode_full_a(std::vector<mtl::Dispatch>& ds, int q, int r) {
    const uint32_t un = (uint32_t) n, uH = (uint32_t) H;
    const size_t ar = (size_t) r * n;   // the row's element offset into anorm / state
    pos1.as<int32_t>()[0] = q;
    // ---- MLA, the query side
    gvs(Q_A).append(ds, *wr[Q_A].buf, wr[Q_A].off, anorm, un, q_resid, (uint32_t) ql, 0, 1, ar, 0);
    rms(ds, q_resid, 0, (uint32_t) ql, Q_A_NORM, q_resid, 0, (uint32_t) ql, (uint32_t) ql, 1);
    gvs(Q_B).append(ds, *wr[Q_B].buf, wr[Q_B].off, q_resid, (uint32_t) ql, qb, uH * (uint32_t) qk, 0, 1);
    rope(ds, qb, uH, (uint32_t) qk, (uint32_t) (qk - rope_dim), uH * (uint32_t) qk, pos1, 1);
    const uint32_t nope = (uint32_t) (qk - rope_dim);
    const size_t kb_head = (size_t) kv_lora * ref::row_bytes(wr[K_B].type, (int) nope);
    gvh(wr[K_B].type, nope, (uint32_t) kv_lora, GemvHeads{(uint32_t) kb_head, (uint32_t) qk, (uint32_t) kv_w})
        .append_heads(ds, *wr[K_B].buf, wr[K_B].off, qb, uH * (uint32_t) qk, Q, uH * (uint32_t) kv_w, 0, 1, uH);
    {
        ElemArgs a{(uint32_t) rope_dim, (uint32_t) qk, 0, (uint32_t) kv_w};
        elem(ds, p_copy, {mtl::Arg::B(qb, 4 * (size_t) nope), mtl::Arg::B(Q, 4 * (size_t) kv_lora), mtl::Arg::V(a)},
             a.n, H, "hy4_flash_copy_f32 (mtp q_pe)");
    }
    // ---- the indexer's query side, its score over the pooled keys and the selection
    const uint32_t n_pool = (uint32_t) ((q + 1) / kpool);
    gvs(IQ_B).append(ds, *wr[IQ_B].buf, wr[IQ_B].off, q_resid, (uint32_t) ql, iq, (uint32_t) (idx_heads * idx_dim), 0,
                       1);
    rope(ds, iq, (uint32_t) idx_heads, (uint32_t) idx_dim, (uint32_t) (idx_dim - rope_dim),
         (uint32_t) (idx_heads * idx_dim), pos1, 1);
    gvs(IPROJ).append(ds, *wr[IPROJ].buf, wr[IPROJ].off, anorm, un, iwraw, (uint32_t) idx_heads, 0, 1, ar, 0);
    {
        IdxWeightArgs a{(uint32_t) idx_heads, (uint32_t) idx_heads, (uint32_t) idx_heads,
                        std::sqrt((float) (idx_dim * idx_heads))};
        elem(ds, p_idx_w, {mtl::Arg::B(iwraw), mtl::Arg::B(iw), mtl::Arg::V(a)}, a.n, 1, "hy4_flash_indexer_weights (mtp)");
    }
    if (n_pool > 0) {
        mtl::Dispatch d;
        d.p = &p_idx_score;
        ScoreArgs a{(uint32_t) idx_dim,
                    (uint32_t) idx_heads,
                    n_pool,
                    1,
                    (uint32_t) kpool,
                    (uint32_t) (idx_heads * idx_dim),
                    3u * (uint32_t) idx_dim * (uint32_t) kpool,
                    (uint32_t) idx_heads,
                    (uint32_t) n_pool_max};
        d.args = {mtl::Arg::B(iq), mtl::Arg::B(idxc, 2 * (2ull * (size_t) idx_dim)), mtl::Arg::B(iw),
                  mtl::Arg::B(pos1), mtl::Arg::B(score), mtl::Arg::V(a)};
        d.grid[0] = n_pool;
        d.grid[1] = 1;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_indexer_score (mtp)");
        ds.push_back(d);
    }
    {
        mtl::Dispatch d;
        d.p = &p_idx_topk;
        TopkArgs a{(uint32_t) kpool, (uint32_t) idx_topk, n_pool, 1, (uint32_t) n_pool_max, (uint32_t) kSelCap};
        d.args = {mtl::Arg::B(score), mtl::Arg::B(pos1), mtl::Arg::B(sel), mtl::Arg::B(nsel), mtl::Arg::V(a)};
        d.grid[0] = 1;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_indexer_topk (mtp)");
        ds.push_back(d);
    }
    // ---- the attention over the MTP's own cache
    {
        AttnArgs a{uH,
                   (uint32_t) kv_w,
                   (uint32_t) kv_lora,
                   (uint32_t) kv_w,
                   uH * (uint32_t) kv_w,
                   uH * (uint32_t) kv_lora,
                   (uint32_t) kSelCap,
                   (uint32_t) kSelCap,
                   0u,
                   1.0f / std::sqrt((float) qk)};
        append_attn(ds, attn_sel,
                    {mtl::Arg::B(Q), mtl::Arg::B(kv), W(SINKS), mtl::Arg::B(sel), mtl::Arg::B(nsel),
                     mtl::Arg::B(pos1), mtl::Arg::B(sc), mtl::Arg::B(attn_o), mtl::Arg::V(a)},
                    uH, 1, (uint32_t) kSelCap, (uint32_t) kSelCap, (uint32_t) kv_w, (uint32_t) kv_lora, (uint32_t) kv_w,
                    "hy4_flash_mla_attn (mtp)");
    }
    const size_t vb_head = (size_t) v_mla * ref::row_bytes(wr[V_B].type, kv_lora);
    gvh(wr[V_B].type, (uint32_t) kv_lora, (uint32_t) v_mla,
        GemvHeads{(uint32_t) vb_head, (uint32_t) kv_lora, (uint32_t) v_mla})
        .append_heads(ds, *wr[V_B].buf, wr[V_B].off, attn_o, uH * (uint32_t) kv_lora, kqv, uH * (uint32_t) v_mla, 0, 1,
                      uH);
    gvs(AGATE).append(ds, *wr[AGATE].buf, wr[AGATE].off, anorm, un, gproj, uH * (uint32_t) v_mla, 0, 1, ar, 0);
    {
        ElemArgs a{uH * (uint32_t) v_mla, uH * (uint32_t) v_mla, uH * (uint32_t) v_mla, uH * (uint32_t) v_mla};
        elem(ds, p_gate_mul, {mtl::Arg::B(kqv), mtl::Arg::B(gproj), mtl::Arg::B(gated), mtl::Arg::V(a)}, a.n, 1,
             "hy4_flash_gate_mul (mtp)");
    }
    gvs(AOUT).append(ds, *wr[AOUT].buf, wr[AOUT].off, gated, uH * (uint32_t) v_mla, attn_y, un, 0, 1);
    rms(ds, attn_y, 0, un, POST_ATTN, attn_out, 0, un, un, 1);
    // the plain residual: state += attention
    {
        ElemArgs a{un, un, un, un};
        elem(ds, p_add, {mtl::Arg::B(state, 4 * ar), mtl::Arg::B(attn_out), mtl::Arg::B(state, 4 * ar), mtl::Arg::V(a)},
             a.n, 1, "hy4_flash_add_f32 (mtp attn residual)");
    }
    // ---- the FFN's input and the router
    rms(ds, state, 4 * ar, un, FFN_NORM, fnorm, 0, un, un, 1);
    gvs(GATE_INP).append(ds, *wr[GATE_INP].buf, wr[GATE_INP].off, fnorm, un, mlogits, (uint32_t) n_experts, 0, 1);
    {
        mtl::Dispatch d;
        d.p = &p_router;
        RouterArgs a{(uint32_t) n_experts, (uint32_t) topk, (uint32_t) n_experts, (uint32_t) topk,
                     (uint32_t) topk,      1u,              m->geo.routed_scale,  6.103515625e-5f};
        d.args = {mtl::Arg::B(mlogits), W(PROBS_B), mtl::Arg::B(ids), mtl::Arg::B(rw), mtl::Arg::V(a)};
        d.grid[0] = 1;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_router (mtp)");
        ds.push_back(d);
    }
}

/// The full row's second half: the experts (from the pool, placed by load_experts), the shared expert, the residual,
/// shared_head_norm and the head's argmax.
void Mtp::Impl::encode_full_b(std::vector<mtl::Dispatch>& ds, int r) {
    encode_full_b0(ds);
    encode_full_b1(ds, r);
}

/// The FFN's work that reads neither the pool nor the plan: the latent down-projection and the shared expert.  In a
/// draft step it runs between the router's signal and the GPU's wait, i.e. while the host reads the missing experts.
/// Moving them ahead of the routed experts changes no value: each dispatch writes its own buffer from fnorm / sha,
/// and the sum below (upo + sho) is formed after both.
void Mtp::Impl::encode_full_b0(std::vector<mtl::Dispatch>& ds) {
    const uint32_t un = (uint32_t) n;
    gvs(RDOWN).append(ds, *wr[RDOWN].buf, wr[RDOWN].off, fnorm, un, rdown, (uint32_t) latent, 0, 1);
    gvs(SH_G).append(ds, *wr[SH_G].buf, wr[SH_G].off, fnorm, un, shg, (uint32_t) eff, 0, 1);
    gvs(SH_U).append(ds, *wr[SH_U].buf, wr[SH_U].off, fnorm, un, shu, (uint32_t) eff, 0, 1);
    {
        GluArgs a{(uint32_t) eff, (uint32_t) eff, (uint32_t) eff, (uint32_t) eff, INFINITY};   // no clamp
        elem(ds, p_glu, {mtl::Arg::B(shg), mtl::Arg::B(shu), mtl::Arg::B(sha), mtl::Arg::V(a)}, a.n, 1,
             "hy4_flash_swiglu_clamp (mtp shexp)");
    }
    gvs(SH_D).append(ds, *wr[SH_D].buf, wr[SH_D].off, sha, (uint32_t) eff, sho, un, 0, 1);
}

/// The routed experts (from the pool, placed by load_experts), the residual, shared_head_norm and the head's argmax.
/// The plan's layout is FIXED (plan_layout): its offsets and its group count are known before the router has run, so
/// this can be encoded into the same command buffer as the router.
void Mtp::Impl::encode_full_b1(std::vector<mtl::Dispatch>& ds, int r) {
    const uint32_t un = (uint32_t) n;
    const size_t ar = (size_t) r * n;
    MoeArgs ma{};
    ma.gate_off = sl_off[0];
    ma.up_off = sl_off[1];
    ma.down_off = sl_off[2];
    ma.gu_stride = slot_bytes;
    ma.d_stride = slot_bytes;
    ma.gu_row = (uint32_t) ref::row_bytes(quant, latent);
    ma.d_row = (uint32_t) ref::row_bytes(quant, eff);
    ma.n_latent = (uint32_t) latent;
    ma.n_ff = (uint32_t) eff;
    ma.n_tok = 1;
    ma.topk = (uint32_t) topk;
    ma.n_group = (uint32_t) topk;   // plan_layout: topk group slots, the unused ones empty
    ma.off_gexp = 0;
    ma.off_gstart = (uint32_t) topk;
    ma.off_etok = 2 * (uint32_t) topk + 1;
    ma.off_ew = 3 * (uint32_t) topk + 1;
    ma.off_tokent = 4 * (uint32_t) topk + 1;
    ma.limit = clamp;
    moe.append(ds, ma, pool, plan, rdown, 0, xp, hbuf, hp, ye, mlat, 0);
    gvs(RUP).append(ds, *wr[RUP].buf, wr[RUP].off, mlat, (uint32_t) latent, upo, un, 0, 1);
    {
        ElemArgs a{un, un, un, un};
        elem(ds, p_add, {mtl::Arg::B(upo), mtl::Arg::B(sho), mtl::Arg::B(ffn_sum), mtl::Arg::V(a)}, a.n, 1,
             "hy4_flash_add_f32 (mtp ffn)");
    }
    rms(ds, ffn_sum, 0, un, POST_FFW, fout, 0, un, un, 1);
    {
        ElemArgs a{un, un, un, un};
        elem(ds, p_add, {mtl::Arg::B(state, 4 * ar), mtl::Arg::B(fout), mtl::Arg::B(state, 4 * ar), mtl::Arg::V(a)},
             a.n, 1, "hy4_flash_add_f32 (mtp ffn residual)");
    }
    rms(ds, state, 4 * ar, un, SHN, snorm, 0, un, un, 1);
    gvs(OUT).append(ds, *wr[OUT].buf, wr[OUT].off, snorm, un, logits, (uint32_t) vocab, 0, 1);
    {
        mtl::Dispatch d;
        d.p = &p_argmax;
        ArgmaxArgs a{(uint32_t) vocab, (uint32_t) vocab};
        d.args = {mtl::Arg::B(logits), mtl::Arg::B(amax), mtl::Arg::V(a)};
        d.grid[0] = 1;
        d.tg[0] = 1024;   // any multiple of 32 gives the same winner (hy4_flash_argmax)
        mtl::require_fit(d, "hy4_flash_argmax (mtp)");
        ds.push_back(d);
    }
}

/// The router's top-k into pool slots (reading the misses from the SSD) and the plan the MoE kernels read.
/// THE PLAN'S LAYOUT IS FIXED: gexp [topk] / gstart [topk + 1] / etok [topk] / ew [topk] / tokent [topk], the offsets
/// encode_full_b1 binds.  One token's router picks topk distinct experts, so MoePlan::build makes exactly topk groups
/// and the layout is MoePlan's own; only when a non-finite router repeats an id (the clamp below) are there fewer,
/// and the slots past them are left EMPTY (gstart[g] == gstart[g + 1]: hy4_flash_moe.metal leaves such a group at once).
/// Every group is one expert's entries computed alone and the combine adds them back in slot order through tokent,
/// so the padding moves no value.
std::string Mtp::Impl::load_experts(MtpStats& st) {
    const double t0 = mtl::now_s();
    int32_t* id = ids.as<int32_t>();
    const float* w = rw.as<float>();
    for (int k = 0; k < topk; ++k)
        if (id[k] < 0 || id[k] >= n_experts) id[k] = 0;   // a non-finite router; the draft is junk, verify rejects it
    MoePlan pl;
    pl.build(id, w, 1, (uint32_t) topk);
    std::vector<int32_t> want(pl.n_group);
    for (uint32_t g = 0; g < pl.n_group; ++g) want[g] = pl.buf[pl.off_gexp + g];
    std::vector<int> slot;
    std::vector<std::pair<int, int32_t>> load;
    const uint64_t h0 = lru.hits, m0 = lru.misses;
    if (!lru.place(want, slot, load)) return "MTP: the expert pool cannot hold one step's experts";
    st.expert_hits += lru.hits - h0;
    st.expert_misses += lru.misses - m0;
    st.half_hits[st.half & 1] += lru.hits - h0;
    st.half_misses[st.half & 1] += lru.misses - m0;
    if (const std::string e = read_slots(load); !e.empty()) return e;
    st.expert_bytes += (double) load.size() * (double) (sl_bytes[0] + sl_bytes[1] + sl_bytes[2]);
    for (uint32_t g = 0; g < pl.n_group; ++g) pl.buf[pl.off_gexp + g] = slot[g];
    int32_t* pb = plan.as<int32_t>();
    const uint32_t K = (uint32_t) topk, ng = pl.n_group;
    for (uint32_t g = 0; g < K; ++g) pb[g] = g < ng ? pl.buf[pl.off_gexp + g] : 0;
    for (uint32_t g = 0; g <= K; ++g) pb[K + g] = pl.buf[pl.off_gstart + std::min(g, ng)];
    std::memcpy(pb + 2 * K + 1, pl.buf.data() + pl.off_etok, 4 * (size_t) K);
    std::memcpy(pb + 3 * K + 1, pl.buf.data() + pl.off_ew, 4 * (size_t) K);
    std::memcpy(pb + 4 * K + 1, pl.buf.data() + pl.off_tokent, 4 * (size_t) K);
    st.expert_read_ms += (mtl::now_s() - t0) * 1e3;
    return {};
}

/// Every slice of every (slot, expert) in `load`, in 4 MiB pieces, on io_threads threads (F_NOCACHE preads).
std::string Mtp::Impl::read_slots(const std::vector<std::pair<int, int32_t>>& load) {
    struct Job {
        uint8_t* dst;
        uint64_t off, len;
    };
    std::vector<Job> jobs;
    for (const auto& ld : load)
        for (int k = 0; k < 3; ++k) {
            uint8_t* base = pool.as<uint8_t>() + (uint64_t) ld.first * slot_bytes + sl_off[k];
            const uint64_t src = file_off[k] + (uint64_t) ld.second * sl_bytes[k];
            for (uint64_t o = 0; o < sl_bytes[k]; o += (4ull << 20))
                jobs.push_back({base + o, src + o, std::min<uint64_t>(4ull << 20, sl_bytes[k] - o)});
        }
    if (!jobs.empty() && rpool) {
        std::vector<ReadPool::Job> pj;
        pj.reserve(jobs.size());
        for (const Job& j : jobs) pj.push_back({j.dst, j.off, j.len});
        if (!rpool->run(pj)) return "MTP: an expert read failed";
    } else if (!jobs.empty()) {
        std::atomic<size_t> next{0};
        std::atomic<bool> ok{true};
        std::vector<std::thread> th;
        const int nt = std::min<int>(io_threads, (int) jobs.size());
        for (int i = 0; i < nt; ++i)
            th.emplace_back([&] {
                for (;;) {
                    const size_t j = next++;
                    if (j >= jobs.size()) return;
                    uint64_t done = 0;
                    while (done < jobs[j].len) {
                        const ssize_t r = pread(fd, jobs[j].dst + done, jobs[j].len - done, (off_t) (jobs[j].off + done));
                        if (r <= 0) { ok = false; return; }
                        done += (uint64_t) r;
                    }
                }
            });
        for (auto& x : th) x.join();
        if (!ok) return "MTP: an expert read from " + m->shard1_path + " failed";
    }
    return {};
}

/// THE POOL'S STARTING CONTENTS (run once, at the first catch_up / step), from a profile - the treatment the
/// trunk's arena gets from its expert profile.  Without it the pool starts empty and most of a short run's expert
/// reads are first-touch misses.  The profile ranks the MTP router's experts by use over prompts that are NOT the
/// speed prompt.  Which slot an expert sits in moves no value: the plan names the slot and the bytes are the same
/// file's.  HY4_FLASH_MTP_POOL_PROFILE=PATH or `none`.
std::string Mtp::Impl::preload_pool() {
    std::string path;
    if (const char* e = std::getenv("HY4_FLASH_MTP_POOL_PROFILE")) path = e;
    else {
        const std::string& src = m->shard1_path;
        const std::string dir = src.substr(0, src.find_last_of('/'));
        const size_t s2 = dir.find_last_of('/');
        path = dir.substr(0, s2) + "/derived/" + dir.substr(s2 + 1) + "/mtp_pool_profile.bin";
    }
    if (path == "none") {
        std::fprintf(stderr, "strata_hy4_flash_metal: MTP expert pool starts empty (HY4_FLASH_MTP_POOL_PROFILE=none)\n");
        return {};
    }
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "strata_hy4_flash_metal: no MTP pool profile at %s: the pool starts empty\n", path.c_str());
        return {};
    }
    uint64_t hdr[4] = {};
    std::vector<int32_t> ranked;
    bool ok = std::fread(hdr, 8, 4, f) == 4 && hdr[0] == 0x4C4F4F5050544D48ull /* "HMTPPOOL" */ && hdr[1] == 1 &&
              hdr[2] == (uint64_t) n_experts && hdr[3] <= (uint64_t) n_experts;
    if (ok) {
        ranked.resize((size_t) hdr[3]);
        ok = std::fread(ranked.data(), 4, ranked.size(), f) == ranked.size();
    }
    std::fclose(f);
    if (!ok) return "MTP: " + path + " is not an MTP pool profile for " + std::to_string(n_experts) + " experts";
    std::vector<std::pair<int, int32_t>> load;
    if (!lru.preload(ranked, load)) return "MTP: " + path + " repeats an expert or names one out of range";
    const double t0 = mtl::now_s();
    if (const std::string e = read_slots(load); !e.empty()) return e;
    std::fprintf(stderr, "strata_hy4_flash_metal: MTP expert pool pre-filled with the profile's top %zu of %zu (%s) in %.2f s\n",
                 load.size(), ranked.size(), path.c_str(), mtl::now_s() - t0);
    return {};
}

// ---- the public face -----------------------------------------------------------------------------------------------
Mtp::~Mtp() {
    if (im_ != nullptr) im_->rpool.reset();   // join the readers before their fd closes
    if (im_ != nullptr && im_->fd >= 0) ::close(im_->fd);
    delete im_;
}

std::string Mtp::open(Model& m, mtl::Context& cx, int lib, int chunk) {
    if (im_ != nullptr) return "already open";
    im_ = new Impl();
    const std::string e = im_->open(m, cx, lib, chunk);
    if (!e.empty()) {
        im_->rpool.reset();   // join the readers, then close their fd (a profile error returns after it was opened)
        if (im_->fd >= 0) ::close(im_->fd);
        delete im_;
        im_ = nullptr;
        return e;
    }
    reset();
    return {};
}

void Mtp::reset() {
    len_ = 0;
    im_->pools_done = 0;
    std::memset(im_->hist.ptr, 0, 4 * (size_t) im_->n);   // h_{-1} = 0 (llama.cpp's pending_h, CUDA's memset)
    st_ = MtpStats{};
}

void Mtp::set_hidden(int q, int n, const float* h) {
    std::memcpy(hidden(q), h, 4 * (size_t) n * (size_t) im_->n);
}
float* Mtp::hidden(int q) { return im_->hist.as<float>() + (size_t) q * (size_t) im_->n; }
const float* Mtp::state() const { return im_->snorm.as<float>(); }
uint64_t Mtp::expert_pool_bytes() const { return im_ == nullptr ? 0 : im_->pool.bytes; }

void Mtp::truncate(int n) {
    len_ = std::min(len_, std::max(n, 0));
    im_->pools_done = std::min(im_->pools_done, len_ / im_->kpool);
}

std::string Mtp::catch_up(const int32_t* tokens, int q_end) {
    Impl& im = *im_;
    if (const std::string e = im.preload_now(); !e.empty()) return e;
    if (q_end > im.rows_cap) return "MTP: row " + std::to_string(q_end) + " is past the cache";
    const double t0 = mtl::now_s();
    im.pools_done = std::min(im.pools_done, len_ / im.kpool);
    while (len_ < q_end) {
        const int T = std::min(im.C, q_end - len_);
        if (const std::string e = im.embed(tokens, len_, T); !e.empty()) return e;
        std::vector<mtl::Dispatch> ds;
        im.encode_rows(ds, len_, T);
        im.cx->run_graph(ds);
        len_ += T;
        st_.cache_rows += (uint64_t) T;
    }
    st_.catchup_ms += (mtl::now_s() - t0) * 1e3;
    return {};
}

std::string Mtp::step(const int32_t* tokens, int q, int32_t* draft) {
    Impl& im = *im_;
    if (const std::string e = im.preload_now(); !e.empty()) return e;
    if (q < len_) truncate(q);
    if (q + 1 > im.rows_cap) return "MTP: row " + std::to_string(q) + " is past the cache";
    // the rows before q that are not in yet, cache-only, in command buffers of at most C
    if (q + 1 - len_ > im.C)
        if (const std::string e = catch_up(tokens, q + 1 - im.C); !e.empty()) return e;
    // now [len_, q] fits one command buffer: cache-only rows and the full row q as its last
    const int q0 = len_, T = q - q0 + 1;
    im.pools_done = std::min(im.pools_done, q0 / im.kpool);
    if (const std::string e = im.embed(tokens, q0, T); !e.empty()) return e;
    std::vector<mtl::Dispatch> ds;
    im.encode_rows(ds, q0, T);
    im.encode_full_a(ds, q, T - 1);
    if (!one_cb()) {   // HY4_FLASH_MTP_ONE_CB=0: two command buffers, the host's turn between them
        im.cx->run_graph(ds);
        if (const std::string e = im.load_experts(st_); !e.empty()) return e;
        ds.clear();
        im.encode_full_b(ds, T - 1);
        im.cx->run_graph(ds);
    } else {
        // ONE command buffer: after the router the GPU signals; the host looks the experts up and reads the misses
        // (load_experts) while the GPU runs rdown and the shared expert, which read neither the pool nor the plan;
        // then the GPU waits for the host and runs the routed experts and the head.  The same dispatches with the
        // same inputs as the two-buffer path, so the same bits; the GPU just does not go back to the queue twice.
        const int cut = (int) ds.size();
        im.encode_full_b0(ds);
        const int wait = (int) ds.size();
        im.encode_full_b1(ds, T - 1);
        struct Turn {
            Impl* im;
            MtpStats* st;
            std::string err;
        } turn{&im, &st_, {}};
        // the draft's own command buffer is not one of the trunk's windows: keep its segments out of the decode
        // line's spin / cut / count accumulators (main.cpp divides those by the trunk's positions and cut count)
        const mtl::Context::SegTiming saved = im.cx->seg_timing();
        im.cx->run_graph_seg(ds, {cut}, {wait}, [](void* u, int) {
            Turn& t = *(Turn*) u;
            t.err = t.im->load_experts(*t.st);
            // an error leaves the plan as it was; the step still runs to the end of the buffer, then reports
        }, &turn);
        mtl::Context::SegTiming keep = saved;
        keep.n_dis_plain += ds.size();
        im.cx->restore_seg_timing(keep);
        if (!turn.err.empty()) return turn.err;
    }
    const int32_t d = im.amax.as<int32_t>()[0];
    *draft = d >= 0 && d < im.vocab ? d : 0;
    len_ = q + 1;
    st_.cache_rows += (uint64_t) (T - 1);
    st_.full_rows++;
    return {};
}

MtpProbe mtp_probe_from_env() {
    const char* e = std::getenv("HY4_FLASH_MTP_PROBE");
    if (e == nullptr || *e == 0 || std::strcmp(e, "none") == 0) return MtpProbe::none;
    if (std::strcmp(e, "draft_random") == 0) return MtpProbe::draft_random;
    if (std::strcmp(e, "accept_all") == 0) return MtpProbe::accept_all;
    std::fprintf(stderr, "strata_hy4_flash_metal: HY4_FLASH_MTP_PROBE=%s is not draft_random / accept_all\n", e);
    std::exit(2);
}

std::string speculative_greedy(Forward& fw, Mtp& mtp, std::vector<int32_t>& seq, int gen, int k, int context,
                               MtpProbe probe, const std::function<void(int32_t)>& emit) {
    MtpStats& st = mtp.stats();
    const int vocab = fw.vocab();
    if (st.accept_hist.size() < (size_t) k + 1) st.accept_hist.assign((size_t) k + 1, 0);
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    int made = 0;
    auto push = [&](int32_t tok) {
        seq.push_back(tok);
        emit(tok);
        ++made;
    };
    // the token the prompt (or the previous call) left
    {
        const int32_t y = fw.last_token();
        if (y < 0 || y >= vocab)
            return std::string(Forward::kNonfinite) + ": the head's argmax returned " + std::to_string(y);
        push(y);
    }
    std::vector<int32_t> tmp, drafts, verify;
    // HY4_FLASH_MTP_DUMP=DIR: every round's first draft row (true token, trunk h) as (q, draft) + its s, and at the end
    // the tokens and the h history - what tmp/tests/hy4_flash/pyref/mtp_parity.py recomputes with the numpy MTP block
    const char* dump_dir = std::getenv("HY4_FLASH_MTP_DUMP");
    std::vector<int32_t> dump_rows, dump_chain;
    std::vector<float> dump_s, dump_chain_h;
    const size_t nh = (size_t) (mtp.hidden(1) - mtp.hidden(0));
    while (made < gen && (int) seq.size() < context) {
        const int L = (int) seq.size() - 1;   // the trunk has run [0, L); seq[L] is the token to run next
        int kk = std::min({k, gen - made - 1, context - (int) seq.size() - 1, Forward::kMaxWindow - 1});
        kk = std::max(kk, 0);
        drafts.assign((size_t) kk, 0);
        // ---- draft: rows [len, L] with row L full, then the chain from the MTP's own s
        const double td = mtl::now_s();
        tmp = seq;
        for (int j = 0; j < kk; ++j) {
            const int q = L + j;
            if (j > 0) {
                std::memcpy(mtp.hidden(q), mtp.state(), 4 * (size_t) (mtp.hidden(1) - mtp.hidden(0)));
                tmp.push_back(drafts[(size_t) j - 1]);
            }
            st.half = 2 * (made + j) >= gen ? 1 : 0;   // the pool's statistics by the half the predicted token is in
            int32_t d = 0;
            if (dump_dir != nullptr && j == 1) dump_chain_h.insert(dump_chain_h.end(), mtp.hidden(q), mtp.hidden(q) + nh);
            if (const std::string e = mtp.step(tmp.data(), q, &d); !e.empty()) return e;
            if (probe == MtpProbe::draft_random) {
                rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                d = (int32_t) ((rng >> 33) % (uint64_t) vocab);
            }
            drafts[(size_t) j] = d;
            if (dump_dir != nullptr && j == 0) {
                dump_rows.push_back(q);
                dump_rows.push_back(d);
                dump_s.insert(dump_s.end(), mtp.state(), mtp.state() + nh);
            }
            if (dump_dir != nullptr && j == 1) {   // the chain's first step: (q, its token, its draft)
                dump_chain.push_back(q);
                dump_chain.push_back(tmp[(size_t) q]);
                dump_chain.push_back(d);
            }
        }
        if (kk > 0) tmp.push_back(drafts[(size_t) kk - 1]);
        // rows past L were built from the MTP's own s; only row L (true token, trunk h) stays (mtp.h)
        mtp.truncate(L + 1);
        st.draft_ms += (mtl::now_s() - td) * 1e3;
        // ---- verify: the trunk at positions L .. L + kk in one window
        const double tv = mtl::now_s();
        if (const std::string e = fw.window(tmp.data(), L, kk + 1); !e.empty()) return e;
        st.verify_ms += (mtl::now_s() - tv) * 1e3;
        verify.assign(fw.argmax(), fw.argmax() + kk + 1);
        for (int32_t v : verify)
            if (v < 0 || v >= vocab)
                return std::string(Forward::kNonfinite) + ": the head's argmax returned " + std::to_string(v);
        int acc = probe == MtpProbe::accept_all ? kk : accept_prefix(drafts.data(), verify.data(), kk);
        // ---- commit: the trunk's h of positions L .. L + acc become hidden(L + 1 .. L + acc + 1)
        const double tc = mtl::now_s();
        mtp.set_hidden(L + 1, acc + 1, fw.result_norm());
        st.rounds++;
        st.offered += (uint64_t) kk;
        st.accepted += (uint64_t) acc;
        if ((size_t) acc < st.accept_hist.size()) st.accept_hist[(size_t) acc]++;
        for (int j = 0; j < acc && made < gen; ++j) push(drafts[(size_t) j]);
        if (made < gen) push(verify[(size_t) acc]);
        st.commit_ms += (mtl::now_s() - tc) * 1e3;
    }
    if (dump_dir != nullptr) {
        auto put = [&](const char* name, const void* p, size_t bytes) {
            const std::string f = std::string(dump_dir) + "/" + name;
            if (FILE* o = std::fopen(f.c_str(), "wb")) {
                std::fwrite(p, 1, bytes, o);
                std::fclose(o);
            }
        };
        put("rows.i32", dump_rows.data(), 4 * dump_rows.size());
        put("s.f32", dump_s.data(), 4 * dump_s.size());
        put("chain.i32", dump_chain.data(), 4 * dump_chain.size());
        put("chain_h.f32", dump_chain_h.data(), 4 * dump_chain_h.size());
        put("tokens.i32", seq.data(), 4 * seq.size());
        put("hidden.f32", mtp.hidden(0), 4 * nh * seq.size());   // hidden(q) = h_{q-1}, q < seq.size()
    }
    return {};
}

}  // namespace strata::hy4_flash
