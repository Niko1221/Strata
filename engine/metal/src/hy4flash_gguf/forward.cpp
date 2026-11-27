// engine/metal/src/hy4flash_gguf/forward.cpp - see forward.h.
//
// The node order is hy4_block.py's, which is llama.cpp's hy-v4-flash.cpp read into numpy for gate 1, so the two
// can be compared node by node.  Every arithmetic choice lives in the kernels; what is here is the wiring, the
// state and the host's turn at a cut.
#include "forward.h"

#include "attn.h"

#include "ref_quant.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

namespace strata::hy4_flash {
namespace {

/// HY4_FLASH_SHARED_EARLY=0 puts the shared expert after the routed experts, inside the wait (an A/B switch).
/// Either way every value is the same; only whether the GPU computes it while the host is at the cut changes.
constexpr int kMoeSplitDefault = 2;

bool shared_early() {
    static const bool on = [] {
        const char* e = std::getenv("HY4_FLASH_SHARED_EARLY");
        return !(e != nullptr && *e != '\0' && std::atoi(e) == 0);
    }();
    return on;
}

/// HY4_FLASH_MOE_SPLIT=<T>: windows of at least T positions run each layer's routed experts in two halves - the
/// ones already in the arena first, while the host waits for the rest, then the rest.  0 turns it off.
/// Bitwise neutral: an expert's entries are computed alone whichever half they are in, and the combine adds the
/// top-k back in slot order.
/// HY4_FLASH_MOE_INDIRECT=0 launches the routed experts' gate/up and down over every group slot a window could fill; the
/// default launches exactly the groups the plan has, per arena chunk and per half, from a grid the host writes at
/// the cut.  Only the number of thread groups that find nothing to do changes, never a value.
bool moe_indirect() {
    static const bool on = [] {
        const char* e = std::getenv("HY4_FLASH_MOE_INDIRECT");
        return !(e != nullptr && *e != '\0' && std::atoi(e) == 0);
    }();
    return on;
}

/// HY4_FLASH_PREFILL_MOE_MM: run a PREFILL chunk's routed experts as a grouped GEMM on the matrix units
/// (hy4_flash_moe_mm.metal) instead of the per-entry GEMVs.  Prefill may sum in another order than decode and may round
/// the operands to fp16; decode is never affected - the switch is read only for T > kMaxWindow and only for
/// IQ2_XXS gate/up with a Q4_K down, the pair the GEMM kernels exist for.
///   0 / off  the GEMVs (the control)        f32    f32 operands, one running sum  (THE DEFAULT)
///   f16      f16 operands, one running sum  f32x2  f32 operands, each K step summed on its own
/// NOT the same name as HY4_FLASH_PREFILL_MOE, which is the MoE sub-batch size (forward.h moe_sub).
///
/// WHY f32 IS THE DEFAULT: f16 operands are allowed and are slightly faster over a whole prompt, but f32 keeps
/// the intermediate h orders of magnitude closer to a float64 reference, which is worth more here than that
/// margin.  f16 stays selectable for when the tolerance judge passes it.
enum class MoeMm { Off, F16, F32, F32x2 };
MoeMm moe_mm_mode() {
    static const MoeMm m = [] {
        // reject a typo here (hy4_flash_kernels.h's parser resolves it to the default), then take ITS answer so the
        // kernel list, this mode and the snapshot binding are always the same decision
        if (const char* e = std::getenv("HY4_FLASH_PREFILL_MOE_MM")) {
            const std::string v = e;
            if (!v.empty() && v != "0" && v != "off" && v != "gemv" && v != "f16" && v != "f32" && v != "f32x2") {
                std::fprintf(stderr, "strata_hy4_flash_metal: HY4_FLASH_PREFILL_MOE_MM=%s is not 0 / f16 / f32 / f32x2\n", v.c_str());
                std::exit(2);
            }
        }
        const std::string n = strata_hy4_flash_metal::prefill_moe_mm_name();
        if (n == "0") return MoeMm::Off;
        if (n == "f16") return MoeMm::F16;
        if (n == "f32x2") return MoeMm::F32x2;
        return MoeMm::F32;
    }();
    return m;
}
/// The tile each mode runs, and whether each K step gets its own sum (function constant 60 = 1 is one running
/// sum).
strata::hy4_flash::MoeMmTile moe_mm_tile(MoeMm m) {
    strata::hy4_flash::MoeMmTile t;
    t.mpp = true;
    t.tk = 32;
    t.sgm = 4;
    t.sgn = 1;
    if (m == MoeMm::F16) { t.tm = 64; t.tn = 32; t.half = true; }
    else { t.tm = 64; t.tn = 64; t.half = false; }
    return t;
}

/// HY4_FLASH_PRUNE=0 keeps every buffer barrier the encoders put in; the default drops the ones between dispatches that
/// share no buffer the GPU writes (prune_barriers).  Which dispatch may overlap which changes, never a value.
/// HY4_FLASH_NORM_SPLIT=0 runs every RMSNorm as one thread group a row; the default spreads a decode row's norm over
/// more cores along the same reduction tree (hy4_flash_norm.metal hy4_flash_rms_part / hy4_flash_rms_apply).  Same values.
bool norm_split() {
    static const bool on = [] {
        const char* e = std::getenv("HY4_FLASH_NORM_SPLIT");
        return !(e != nullptr && *e != '\0' && std::atoi(e) == 0);
    }();
    return on;
}

bool prune_on() {
    static const bool on = [] {
        const char* e = std::getenv("HY4_FLASH_PRUNE");
        return !(e != nullptr && *e != '\0' && std::atoi(e) == 0);
    }();
    return on;
}

int moe_split_min() {
    static const int t = [] {
        const char* e = std::getenv("HY4_FLASH_MOE_SPLIT");
        return e != nullptr && *e != '\0' ? std::atoi(e) : kMoeSplitDefault;
    }();
    return t;
}

constexpr float kEngramEps = 1.1920929e-07f;   // FLT_EPSILON: the Engram path's own, not the model's

/// THE THIRD GATE'S MUTATIONS, by the reference's own names.  Each one is a single wrong
/// decision that a correct implementation does not make, and gate 1 must catch every one of them; a mutation that
/// slips through means the gate measures nothing there.  The reference carries them as nine git branches of
/// llama.cpp; here they are one environment variable, because the alternative - nine builds of the engine - would
/// make "the binary that passed the gate" and "the binary that ships" different objects.
///
/// A mutated run SAYS SO, loudly, on stdout and in every report line: a silent mutation switch is a way to make a
/// failing engine look passing, which is the one thing the gates exist to prevent.
struct Mut {
    bool none = true;
    bool no_sinks = false;      ///< drop the attention sink (block 21)
    bool rope_theta = false;    ///< a sliding block rotates with the SPARSE theta (block 2)
    bool engram_delta2 = false; ///< the Engram injection, doubled (block 11)
    bool router_biased = false; ///< the routing weight from the BIASED probability (block 6)
    bool clamp7 = false;        ///< SwiGLU clamp 10 -> 7 (block 41)
    bool no_ape = false;        ///< the kpool compressor without its position-in-group embedding
    bool no_tail = false;       ///< the indexer selection without the forced tail
    bool pool_rope_last = false;///< a pooled key rotated to its group's LAST position
    bool swa_plus1 = false;     ///< the sliding window one position too wide
    std::string name;

    static Mut from_env() {
        Mut m;
        const char* e = std::getenv("HY4_FLASH_MUT");
        if (e == nullptr || *e == 0 || std::strcmp(e, "none") == 0) return m;
        m.name = e;
        m.none = false;
        if (m.name == "no_sinks") m.no_sinks = true;
        else if (m.name == "rope_theta") m.rope_theta = true;
        else if (m.name == "engram_delta2") m.engram_delta2 = true;
        else if (m.name == "router_biased") m.router_biased = true;
        else if (m.name == "clamp7") m.clamp7 = true;
        else if (m.name == "no_ape") m.no_ape = true;
        else if (m.name == "no_tail") m.no_tail = true;
        else if (m.name == "pool_rope_last") m.pool_rope_last = true;
        else if (m.name == "swa_plus1") m.swa_plus1 = true;
        else m.name = "?" + m.name;   // an unknown name is reported by `open`, not ignored
        return m;
    }
};

// mirrors the arg structs in msl/hy4_flash_*.metal
struct NormArgs {
    uint32_t n, x_stride, y_stride;
    float eps;
};
struct RopeArgs {
    uint32_t n_rot, n_head, head_dim, off, x_stride, y_stride;
    float theta_scale;
    uint32_t pos_stride;        // M-RoPE only; 0 on the text path
    uint32_t mlen1, mlen2;
};
struct GluArgs {
    uint32_t n, g_stride, u_stride, y_stride;
    float limit;
};
struct HcArgs {
    uint32_t n_embd, x_stride, mix_stride, cur_stride;
    float hc_eps, hc_mag;
};
struct HcNormArgs {
    uint32_t y_stride;
    float eps;
};
struct KvFinishArgs {
    uint32_t n, n_rot, row, kvn_stride;
    float eps, theta_scale;
    uint32_t pos_stride;        // M-RoPE only; 0 on the text path
    uint32_t mlen1, mlen2;
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
struct EngramRowArgs {
    uint32_t dim, columns, row_bytes, out_stride;
    float eps;
};
struct EngramArgs {
    uint32_t n_embd, hc, x_stride, kv_stride;
    float sqrt_n, eps;
};

/// ggml's own theta_scale: powf(base, -2 / n_rot) in f32 (hy4_flash_rope.metal's note).
float theta_scale_of(float base, int n_rot) { return std::pow(base, -2.0f / (float) n_rot); }

}  // namespace

// ---- the implementation --------------------------------------------------------------------------------------------
struct Forward::Impl {
    // one request's pictures (set_image_prompt); empty for every text request, which is what
    // keeps the text path on the kernels it has always used
    const std::vector<Forward::ImageSpanRows>* img = nullptr;
    const int32_t* img_axes = nullptr;
    int64_t img_axes_len = 0;
    int32_t img_delta = 0, img_mlen1 = 0, img_mlen2 = 0;
    bool images() const { return img != nullptr && !img->empty(); }
    /// Floats between the M-RoPE planes, read off the buffer rather than from `tcap`: with two
    /// scratch sets the decode set is allocated at kMaxWindow while tcap keeps the prefill chunk,
    /// so striding by tcap would address past the end of the decode set's buffer.
    uint32_t rope_plane() const { return (uint32_t) (rope_pos.bytes / (4 * 3)); }
    Model* m = nullptr;
    mtl::Context* cx = nullptr;
    int lib = -1;
    std::string pack;
    EngramSupply eng;
    bool eng_ready = false;
    Timing tm;
    int n_tok = 0, p0 = 0;              // the window being run
    std::vector<int32_t> routed, predicted;
    std::vector<float> all_logits;      // [ordinal][T][n_experts], filled at each cut for the dump
    std::vector<int32_t> pools_done;    // per sparse block: pooled keys already in its cache
    std::vector<float> score_dump;      // the indexer score, trimmed to each query's own extent, for the dump
    Mut mut;
    mtl::Buffer mut_sinks, mut_zero;    // -inf sinks and a zeroed ape, for two of the mutations

    // geometry shorthands
    int32_t n_embd = 0, hc = 0, hcw = 0, vocab = 0, kv_lora = 0, rope_dim = 0, kv_w = 0, qk = 0, v_mla = 0;
    int32_t topk = 0, n_experts = 0, latent = 0, expert_ff = 0, ctx = 0, max_heads = 0;
    int32_t idx_dim = 0, idx_heads = 0, idx_kpool = 0, idx_topk = 0;
    float eps = 0;

    // pipelines
    mtl::Pipeline p_rope_mrope, p_kv_finish_mrope;
    mtl::Pipeline p_rms_w, p_rms_nw, p_rope, p_glu, p_hc_mix, p_hc_mix_norm, p_kv_finish, p_hc_head, p_hc_post, p_router, p_idx_w, p_idx_kpool,
        p_idx_score, p_idx_score_pf, p_idx_topk, p_argmax, p_copy, p_store_half, p_add_row, p_add,
        p_gate_mul, p_eng_rows, p_eng_gate, p_eng_delta;
    AttnPipes attn_range, attn_sel;   // attn.h: the score row, the softmax, PV
    mtl::Pipeline attn_pf_range, attn_pf_sel;   // the prefill MMA attention (hy4_flash_attn_pf.metal), prompt chunks only
    bool pf_mma_ok = false, pf_mma_idx_ok = false;
    std::map<std::array<uint64_t, 3>, DenseGemvF32> gemv_;
    std::map<std::array<uint64_t, 6>, DenseGemvF32> gemvh_;   // per-head weights, one dispatch for all heads
    std::map<std::pair<int, int>, MoeF32> moe_;

    // state
    mtl::Buffer kv, idxc;               // the MLA latent cache and the indexer cache, all blocks back to back
    // scratch, one window wide
    mtl::Buffer hcA, hcB, hc_eng, hc_mid, flat, mixes_a, mixes_f, pre_a, post_a, pre_f, post_f, cur;
    mtl::Buffer anorm, fnorm, fout;   // attn_norm / ffn_norm / ffn_out: three nodes gate 1's neighbours look at
    mtl::Buffer q_resid, qb, Q, qpe, kpe, attn_o, kqv, gproj, gated, attn_y, attn_out, kvp, kvn, sc, pos_b, rope_pos;
    mtl::Buffer ik, iv, ig, iq, iwraw, iw, pool32, score, sel, nsel;
    mtl::Buffer mlogits, ids, rw, pred, pred2, rdown, xp, hbuf, hp, ye, mlat, upo, shg, shu, sha, sho, ffn_sum;
    mtl::Buffer dg, du, da;
    mtl::Buffer plan, logits_b, argmax_b, badflag;
    uint32_t bad_reported = 0;        // how much of badflag the caller has already been told about
    std::string nonfinite_check();    // "" or the `nonfinite` sentinel for what this window added
    mtl::Buffer eng_rows, eng_valid, eng_any, eng_r, eng_kv, eng_dot, eng_gate, eng_delta;
    MoeSlots slots;
    // prefill: its own plan and buffer, so a decode window keeps the tighter g_max = T * topk and its MoE
    // dispatches launch only as many group thread groups as that plan needs.
    mtl::Buffer pplan;
    MoeSlots pslots;
    Prefill pf;
    int tcap = Forward::kMaxWindow;   ///< positions the scratch is allocated for
    int asub = Forward::kMaxWindow;   ///< positions per attention dispatch (the score scratch)
    int msub = Forward::kMaxWindow;   ///< positions per MoE plan (the entry-indexed scratch)
    int hsub = Forward::kMaxWindow;   ///< positions per head dispatch (the logits buffer)
    std::vector<float> last_log;      ///< the last position the head ran for
    int32_t last_tok = -1;
    int32_t snap_used = 0;            ///< positions the caches hold: set by prompt/window, saved and restored
    void snap_fill(SnapBinding& b) const;   ///< the geometry half of a snapshot's binding (prefill_snap.h)
    void snap_regions(std::vector<SnapSlice>& v);   ///< the state a decode continuation reads

    MoeSlots& plan_of(int T) { return T > Forward::kMaxWindow ? pslots : slots; }
    const mtl::Buffer& plan_buf(int T) const { return T > Forward::kMaxWindow ? pplan : plan; }

    /// One block's weights and shape, resolved once.
    struct BW {
        int id = 0, heads = 0, ordinal = -1, engram_table = -1, idx_slot = -1;
        bool sliding = false, has_indexer = false, dense_ffn = false;
        float base = 0, tscale = 0, clamp = 0;
        uint64_t kv_off = 0, idx_off = 0;   // in halves
        const gguf::Tensor* t[64] = {nullptr};
    };
    // the slots of BW::t
    enum {
        W_HCA_FN, W_HCA_SCALE, W_HCA_BASE, W_ATTN_NORM, W_Q_A, W_Q_A_NORM, W_Q_B, W_KV_A, W_KV_A_NORM, W_K_B, W_V_B,
        W_SINKS, W_AGATE, W_AOUT, W_POST_ATTN, W_HCF_FN, W_HCF_SCALE, W_HCF_BASE, W_FFN_NORM, W_POST_FFW, W_FGATE,
        W_FUP, W_FDOWN, W_GATE_INP, W_PROBS_B, W_RDOWN, W_RUP, W_SH_G, W_SH_U, W_SH_D, W_IQ_B, W_IK, W_IK_NORM,
        W_IPROJ, W_ICV, W_ICG, W_IAPE, W_ICNORM, W_E_LOOK, W_E_KV, W_E_KVB, W_E_GATE, W_E_HCS, W_E_GAMMA, W_COUNT
    };
    std::vector<BW> blk;
    // the head
    const gguf::Tensor* h_fn = nullptr;
    const gguf::Tensor* h_scale = nullptr;
    const gguf::Tensor* h_base = nullptr;
    const gguf::Tensor* h_norm = nullptr;
    const gguf::Tensor* h_out = nullptr;
    mtl::Buffer h_mixes, h_cur, h_rnorm;

    // the cut bookkeeping of the window being run.  cut_wait[s] is where the GPU waits for the host's turn s
    // (mtl::Context::run_graph_seg with waits): after the router's signal it runs the work that does not depend on
    // the routing - the shared expert and the latent down-projection - while the host is at the cut.
    std::vector<int> cut_block;
    std::vector<int> cut_wait;
    // what cut s is: 0 a router cut (at_layer whole), 1 a router cut whose layer runs its experts in two halves
    // (begin_layer: no wait), 2 that layer's second cut (finish_layer: the wait, then the read-ahead)
    std::vector<int8_t> cut_role;
    // prune_barriers: the buffers no decode dispatch writes (the weights, the arena, the host-written plans)
    std::vector<const void*> ro_keys;
    uint64_t pr_seen = 0, pr_dropped = 0;
    void prune_barriers(std::vector<mtl::Dispatch>& ds, const std::vector<int>& cuts, const std::vector<int>& waits);
    mtl::Pipeline p_rms_part, p_rms_apply_w, p_rms_apply_nw;
    mtl::Buffer rms_part;   // hy4_flash_rms_part's partials: kMaxWindow rows x 8
    mtl::Buffer plan_b;   // the late half's plan (MoeSlots::split_plans), decode windows only
    // moe_indirect: the thread-group grids of the decode MoE dispatches, [half][chunk][gate/up, down] x 3 uint32,
    // written by at_cut for the layer it is about to release.  gx: each kind's grid width (encode time), PER
    // ORDINAL - the width is n_out / the tile's rows a threadgroup, and the down's tile follows its format
    // (hy4_flash_kernels.h moe_down_tile: Q4_K 32 rows, Q6_K 16), which a checkpoint can mix across layers.  One width
    // for every layer would launch half the Q6_K rows.
    mtl::Buffer moe_ind;
    std::vector<std::array<uint32_t, 2>> moe_gx;
    /// The prefill grouped GEMM (moe_mm_mode): one MoeMmF32 per (gate/up, down) type pair, the tile lists of the
    /// layer the host is about to release - one per (sub-batch, arena chunk), `mm_cap` tiles each - and their
    /// indirect grids, written by at_cut like moe_ind.  mm_gx is PER ORDINAL for the reason moe_gx is: the grid
    /// width is n_out / the tile's TN, and only the layers whose types the GEMM covers write it.
    std::map<std::pair<int, int>, MoeMmF32> moemm_;
    mtl::Buffer mm_tiles, mm_ind;
    bool mm_all = false;      ///< EVERY routed layer is one the GEMM covers (so no layer needs xprep)
    uint32_t mm_cap = 0;      ///< tiles one (sub-batch, chunk) list has room for
    uint32_t mm_slots = 0;    ///< (sub-batch, chunk) pairs the buffers hold
    std::vector<std::array<uint32_t, 2>> mm_gx;

    /// Does ordinal `o` run the GEMM for a window of T?  The encoder and at_cut must agree, so both ask here.
    bool mm_active(int o, int T, uint32_t nch) const {
        if (moe_mm_mode() == MoeMm::Off || T <= Forward::kMaxWindow || mm_tiles.ptr == nullptr) return false;
        if (nch > 4 || (size_t) o >= m->experts.size()) return false;
        const ExpertSlices& es = m->experts[(size_t) o];
        return es.type[kGate] == ref::IQ2_XXS && es.type[kUp] == ref::IQ2_XXS && es.type[kDown] == ref::Q4_K;
    }
    const MoeMmF32& moe_mm(int gu_type, int d_type) {
        const std::pair<int, int> k{gu_type, d_type};
        auto it = moemm_.find(k);
        if (it == moemm_.end()) {
            const MoeMm md = moe_mm_mode();
            const strata::hy4_flash::MoeMmTile t = moe_mm_tile(md);
            // one running sum unless the mode asks for a sum per K step
            mtl::Constants c;
            if (md != MoeMm::F32x2) c.i(60, 1);
            it = moemm_.emplace(k, MoeMmF32(*cx, lib, (uint32_t) latent, (uint32_t) expert_ff, t, t, c)).first;
        }
        return it->second;
    }
    std::vector<int32_t> plan_a_host, plan_b_host;
    const Inject* inj = nullptr;
    // --prefill-consistency's routing controls (Forward::set_prompt_routing): while a prompt chunk runs, its routed
    // ids are copied to pr_record ([ordinal][pr_n][top_k]) and, if pr_force is set, REPLACED by pr_force's (the
    // same layout) the way --force-topk does it for one block.  Both null on every other run.
    std::vector<int32_t>* pr_record = nullptr;
    const int32_t* pr_force = nullptr;
    int pr_n = 0;
    bool pr_in_chunk = false;
    std::string cut_err;

    const DenseGemvF32& gv(int type, uint32_t n_in, uint32_t n_out) {
        const std::array<uint64_t, 3> k{(uint64_t) type, n_in, n_out};
        auto it = gemv_.find(k);
        if (it == gemv_.end())
            it = gemv_.emplace(k, DenseGemvF32(*cx, lib, type, n_in, n_out, (uint32_t) ref::row_bytes(type, (int) n_in),
                                               GemvHeads{}, /*allow_mm=*/true))
                     .first;
        return it->second;
    }
    /// One head's shape of a per-head weight, for DenseGemvF32::append_heads.
    const DenseGemvF32& gvh(int type, uint32_t n_in, uint32_t n_out, GemvHeads h) {
        const std::array<uint64_t, 6> k{(uint64_t) type, n_in, n_out, h.w, h.x, h.y};
        auto it = gemvh_.find(k);
        if (it == gemvh_.end())
            it = gemvh_
                     .emplace(k, DenseGemvF32(*cx, lib, type, n_in, n_out,
                                              (uint32_t) ref::row_bytes(type, (int) n_in), h, /*allow_mm=*/true))
                     .first;
        return it->second;
    }
    const DenseGemvF32& gv(const gguf::Tensor* t) {
        const uint32_t n_in = (uint32_t) t->shape[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < t->shape.size(); ++i) rows *= t->shape[i];
        return gv(t->type, n_in, (uint32_t) rows);
    }
    const MoeF32& moe(int gu_type, int d_type) {
        const std::pair<int, int> k{gu_type, d_type};
        auto it = moe_.find(k);
        if (it == moe_.end())
            it = moe_.emplace(k, MoeF32(*cx, lib, gu_type, d_type, (uint32_t) latent, (uint32_t) expert_ff)).first;
        return it->second;
    }

    // ---- dispatch helpers (one place each, so a stride mistake shows up in every user at once)
    void rms(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, size_t x_off, uint32_t x_stride,
             const gguf::Tensor* w, const mtl::Buffer& y, size_t y_off, uint32_t y_stride, uint32_t n, int rows,
             float e) {
        NormArgs a{n, x_stride, y_stride, e};
        if (norm_split() && rows <= Forward::kMaxWindow && n >= 2048 && rms_part.ptr != nullptr) {
            // hy4_flash_norm.metal: the same tree over HY4_FLASH_TG_SG thread groups, then the scale over n / 256 of them
            mtl::Dispatch p;
            p.p = &p_rms_part;
            p.args = {mtl::Arg::B(x, x_off), mtl::Arg::B(rms_part), mtl::Arg::V(a)};
            p.grid[0] = 8;
            p.grid[1] = (uint32_t) rows;
            p.tg[0] = 32;
            mtl::require_fit(p, "hy4_flash_rms_part");
            ds.push_back(p);
            mtl::Dispatch q;
            q.p = w != nullptr ? &p_rms_apply_w : &p_rms_apply_nw;
            q.args = {mtl::Arg::B(x, x_off), w != nullptr ? mtl::Arg::B(m->dense, w->offset) : mtl::Arg::B(x, x_off),
                      mtl::Arg::B(y, y_off), mtl::Arg::V(a), mtl::Arg::B(rms_part)};
            q.grid[0] = (n + 255) / 256;
            q.grid[1] = (uint32_t) rows;
            q.tg[0] = 256;
            mtl::require_fit(q, "hy4_flash_rms_apply");
            ds.push_back(q);
            return;
        }
        mtl::Dispatch d;
        d.p = w != nullptr ? &p_rms_w : &p_rms_nw;
        d.args = {mtl::Arg::B(x, x_off), w != nullptr ? mtl::Arg::B(m->dense, w->offset) : mtl::Arg::B(x, x_off),
                  mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
        d.grid[0] = (uint32_t) rows;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_rmsnorm");
        ds.push_back(d);
    }
    void rope(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, uint32_t n_head, uint32_t head_dim, uint32_t off,
              uint32_t stride, float tscale, int rows) {
        mtl::Dispatch d;
        const bool mr = images();
        d.p = mr ? &p_rope_mrope : &p_rope;
        RopeArgs a{(uint32_t) rope_dim, n_head, head_dim, off, stride, stride, tscale,
                   mr ? rope_plane() : 0u, mr ? (uint32_t) img_mlen1 : 0u, mr ? (uint32_t) img_mlen2 : 0u};
        d.args = {mtl::Arg::B(x), mtl::Arg::B(x), mtl::Arg::B(mr ? rope_pos : pos_b), mtl::Arg::V(a)};
        d.grid[0] = (uint32_t) rope_dim / 2;
        d.grid[1] = n_head;
        d.grid[2] = (uint32_t) rows;
        d.tg[0] = 32;
        mtl::require_fit(d, "hy4_flash_rope_norm");
        ds.push_back(d);
    }
    void elem(std::vector<mtl::Dispatch>& ds, const mtl::Pipeline& p, std::vector<mtl::Arg> args, uint32_t n,
              int rows, const char* where) {
        mtl::Dispatch d;
        d.p = &p;
        d.args = std::move(args);
        d.tg[0] = 256;
        d.grid[0] = (n + 255) / 256;
        d.grid[1] = (uint32_t) rows;
        mtl::require_fit(d, where);
        ds.push_back(d);
    }
    void glu(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& g, const mtl::Buffer& u, const mtl::Buffer& y,
             uint32_t n, int rows, float limit) {
        GluArgs a{n, n, n, n, limit};
        elem(ds, p_glu, {mtl::Arg::B(g), mtl::Arg::B(u), mtl::Arg::B(y), mtl::Arg::V(a)}, n, rows,
             "hy4_flash_swiglu_clamp");
    }
    // `norm_w` / `norm_y`: the block's RMSNorm of `cur` (attn_norm or ffn_norm) fused into the mix - one dispatch,
    // bitwise the hy4_flash_hc_mix + hy4_flash_rmsnorm pair it replaces (hy4_flash_hc.metal, hy4_flash_hc_mix_norm)
    void hc_pre(std::vector<mtl::Dispatch>& ds, const BW& b, const mtl::Buffer& x, int which, int T,
                const gguf::Tensor* norm_w = nullptr, const mtl::Buffer* norm_y = nullptr, float norm_eps = 0) {
        // flat_norm over the WHOLE hc * n_embd row, no weight
        rms(ds, x, 0, (uint32_t) hcw, nullptr, flat, 0, (uint32_t) hcw, (uint32_t) hcw, T, eps);
        const gguf::Tensor* fn = b.t[which == 0 ? W_HCA_FN : W_HCF_FN];
        const mtl::Buffer& mx = which == 0 ? mixes_a : mixes_f;
        gv(fn).append(ds, m->dense, fn->offset, flat, (uint32_t) hcw, mx, 2 * (uint32_t) hc, 0, (uint32_t) T);
        mtl::Dispatch d;
        d.p = norm_w != nullptr ? &p_hc_mix_norm : &p_hc_mix;
        HcArgs a{(uint32_t) n_embd, (uint32_t) hcw, 2 * (uint32_t) hc, (uint32_t) n_embd, m->geo.hc_eps,
                 m->geo.hc_magnitude};
        d.args = {mtl::Arg::B(x), mtl::Arg::B(mx),
                  mtl::Arg::B(m->dense, b.t[which == 0 ? W_HCA_SCALE : W_HCF_SCALE]->offset),
                  mtl::Arg::B(m->dense, b.t[which == 0 ? W_HCA_BASE : W_HCF_BASE]->offset),
                  mtl::Arg::B(which == 0 ? pre_a : pre_f), mtl::Arg::B(which == 0 ? post_a : post_f),
                  mtl::Arg::B(cur), mtl::Arg::V(a)};
        if (norm_w != nullptr) {
            HcNormArgs na{(uint32_t) n_embd, norm_eps};
            d.args.push_back(mtl::Arg::B(m->dense, norm_w->offset));
            d.args.push_back(mtl::Arg::B(*norm_y));
            d.args.push_back(mtl::Arg::V(na));
        }
        d.grid[0] = (uint32_t) T;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_hc_mix");
        ds.push_back(d);
    }
    void hc_post(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& y, const mtl::Buffer& residual,
                 const mtl::Buffer& post, const mtl::Buffer& out, int T) {
        mtl::Dispatch d;
        d.p = &p_hc_post;
        HcArgs a{(uint32_t) n_embd, (uint32_t) hcw, 2 * (uint32_t) hc, (uint32_t) n_embd, m->geo.hc_eps,
                 m->geo.hc_magnitude};
        d.args = {mtl::Arg::B(y), mtl::Arg::B(residual), mtl::Arg::B(post), mtl::Arg::B(out), mtl::Arg::V(a)};
        d.grid[0] = ((uint32_t) n_embd + 255) / 256;
        d.grid[1] = (uint32_t) hc;
        d.grid[2] = (uint32_t) T;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_hc_post");
        ds.push_back(d);
    }

    std::string open(Model& mm, mtl::Context& c, const std::string& pack_dir, const Prefill& p);
    std::string alloc();
    void alloc_scratch(int Tm, int as, int ms, const std::function<mtl::Buffer(size_t)>& get);
    std::string attach_prefill();
    void swap_scratch();                    ///< the decode set <-> the prompt's (the GPU must be idle)
    bool split = false, big_in = false;     ///< two scratch sets; the prompt's is the one in the members now
    std::vector<mtl::Buffer> big;           ///< the set that is NOT in the members
    std::vector<mtl::Buffer*> scratch_ptrs() {
        return {&hcA, &hcB, &hc_eng, &hc_mid, &flat, &mixes_a, &mixes_f, &pre_a, &post_a, &pre_f, &post_f, &cur, &anorm, &fnorm, &fout, &q_resid, &qb, &Q, &qpe, &kpe, &attn_o, &kqv, &gproj, &gated, &attn_y, &attn_out, &kvp, &kvn, &sc, &pos_b, &rope_pos, &ik, &iv, &ig, &iq, &iwraw, &iw, &score, &sel, &nsel, &mlogits, &ids, &rw, &pred, &pred2, &rdown, &xp, &hbuf, &hp, &ye, &mlat, &upo, &shg, &shu, &sha, &sho, &ffn_sum, &dg, &du, &da, &eng_rows, &eng_valid, &eng_any, &eng_r, &eng_kv, &eng_dot, &eng_gate, &eng_delta};
    }
    static bool scratch_split_on() {
        const char* e = std::getenv("HY4_FLASH_SCRATCH_SPLIT");
        return e == nullptr || std::atoi(e) != 0;
    }
    std::string resolve();
    void encode_block(std::vector<mtl::Dispatch>& ds, std::vector<int>& cuts, const BW& b, const mtl::Buffer& x,
                      const mtl::Buffer& out, int T, bool want_dump);
    void encode_head(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, int x_t0, int T, bool with_logits = true);
    std::string run_head(const mtl::Buffer& x, int x_t0, int T);
    std::string run_hidden(const mtl::Buffer& x, int x_t0, int T);
    void clamp_pools(int p0_);
    void at_cut(int s);
    std::string prepare_window(const int32_t* all_tokens, int p0_, int T, bool need_engram);
    std::string engram_rows_for(int block_id, int T);
    std::string embed_window(const int32_t* all_tokens, int T);
    void dump_block(const BW& b, int T, Dump* dump, const Inject& inj, const mtl::Buffer& in, const mtl::Buffer& out);
};

// ---- opening -------------------------------------------------------------------------------------------------------
std::string Forward::Impl::open(Model& mm, mtl::Context& c, const std::string& pack_dir, const Prefill& p) {
    m = &mm;
    cx = &c;
    pack = pack_dir;
    pf = p;
    // The sub-batch sizes are clamped to [kMaxWindow, tcap]: a decode window must fit in ONE sub-batch of each
    // stage, or the "prefill is bitwise decode" claim would rest on a path decode never takes.
    tcap = std::max(Forward::kMaxWindow, pf.chunk);
    auto clampsub = [&](int v) { return std::min(tcap, std::max(Forward::kMaxWindow, v)); };
    asub = clampsub(pf.attn_sub);
    msub = clampsub(pf.moe_sub);
    // THE PREFILL GEMM WANTS THE WHOLE CHUNK IN ONE PLAN.  A small sub-batch leaves a few entries per expert, so
    // the tiles are mostly padding and the GEMM is SLOWER than the GEMVs.  The GEMVs need the sub-batches because
    // their prepared h plane (hp) is 4 planes of entry x n_ff - 1.64 GB at a 4096 chunk - but the GEMM reads h as
    // the f32 it is and the gate/up reads the raw x plane, so with every routed layer covered neither plane exists
    // and the whole chunk costs h + ye alone.  Only then is the sub-batch raised, and only for the prompt.
    // The TRUNK's expert ordinals only: n_expert_ordinals() counts the MTP draft block's too (geometry.hpp), and
    // that one is BF16, which would answer "not covered" for every model.  The MTP block has its own expert path
    // (mtp.cpp), never this one.
    const int32_t n_trunk_ord = mm.geo.n_expert_ordinals() - mm.geo.n_mtp_blocks;
    mm_all = moe_mm_mode() != MoeMm::Off && n_trunk_ord > 0 && (size_t) n_trunk_ord <= mm.experts.size();
    for (int32_t o = 0; mm_all && o < n_trunk_ord; ++o) {
        const ExpertSlices& es = mm.experts[(size_t) o];
        if (es.type[kGate] != ref::IQ2_XXS || es.type[kUp] != ref::IQ2_XXS || es.type[kDown] != ref::Q4_K)
            mm_all = false;
    }
    if (mm_all) msub = tcap;
    hsub = clampsub(pf.head_sub);
    const hy4_flash::Geometry& g = mm.geo;
    n_embd = g.hidden_dim;
    hc = g.hc_mult > 0 ? g.hc_mult : 1;
    hcw = g.hc_width();
    vocab = g.vocab_size;
    kv_lora = g.kv_lora_rank;
    rope_dim = g.rope_dim;
    kv_w = g.kv_a_width;
    qk = g.qk_head_dim;
    v_mla = g.v_head_dim;
    topk = g.top_k;
    n_experts = g.n_experts;
    latent = g.expert_dim;
    expert_ff = g.expert_ff;
    ctx = g.max_context;
    eps = g.rms_norm_eps;
    idx_dim = g.idx_key_dim;
    idx_heads = g.idx_heads;
    idx_kpool = g.idx_kpool;
    idx_topk = g.idx_top_k;
    max_heads = 0;
    for (int i = 0; i < g.trunk_blocks; ++i) max_heads = std::max(max_heads, g.block(i).n_heads);
    if (kv_w != kv_lora + rope_dim)
        return "attention.key_length (" + std::to_string(kv_w) + ") is not kv_lora_rank + rope_dim";

    mut = Mut::from_env();
    if (!mut.none && mut.name[0] == '?')
        return "HY4_FLASH_MUT=" + mut.name.substr(1) + " is not one of the third gate's mutations (plan P0.3 §6: "
               "no_sinks, rope_theta, engram_delta2, router_biased, clamp7, no_ape, no_tail, pool_rope_last, "
               "swa_plus1)";
    if (!mut.none)
        std::fprintf(stdout, "strata_hy4_flash_metal: MUTATED BUILD, HY4_FLASH_MUT=%s - this run is the third gate's negative "
                             "control and its output is WRONG ON PURPOSE (goal §1)\n", mut.name.c_str());
    lib = cx->compile(strata_hy4_flash_metal::hy4_flash_kernel_files(), strata_hy4_flash_metal::hy4_flash_prelude(), false);
    p_rms_w = cx->pipeline(lib, "hy4_flash_rmsnorm", mtl::Constants().b(40, true));
    p_rms_nw = cx->pipeline(lib, "hy4_flash_rmsnorm", mtl::Constants().b(40, false));
    p_rms_part = cx->pipeline(lib, "hy4_flash_rms_part");
    p_rms_apply_w = cx->pipeline(lib, "hy4_flash_rms_apply", mtl::Constants().b(40, true));
    p_rms_apply_nw = cx->pipeline(lib, "hy4_flash_rms_apply", mtl::Constants().b(40, false));
    p_rope = cx->pipeline(lib, "hy4_flash_rope_norm");
    // The image variants are built always and dispatched only when a picture is attached, so a
    // text-only request cannot reach them.
    p_rope_mrope = cx->pipeline(lib, "hy4_flash_rope_mrope");
    p_kv_finish_mrope = cx->pipeline(lib, "hy4_flash_kv_finish_mrope");
    p_glu = cx->pipeline(lib, "hy4_flash_swiglu_clamp");
    p_hc_mix = cx->pipeline(lib, "hy4_flash_hc_mix", mtl::Constants().b(41, true));
    p_hc_mix_norm = cx->pipeline(lib, "hy4_flash_hc_mix_norm", mtl::Constants().b(41, true));
    p_kv_finish = cx->pipeline(lib, "hy4_flash_kv_finish");
    p_hc_head = cx->pipeline(lib, "hy4_flash_hc_mix", mtl::Constants().b(41, false));
    p_hc_post = cx->pipeline(lib, "hy4_flash_hc_post");
    p_router = cx->pipeline(lib, "hy4_flash_router");
    p_idx_w = cx->pipeline(lib, "hy4_flash_indexer_weights");
    p_idx_kpool = cx->pipeline(lib, "hy4_flash_indexer_kpool",
                               mut.pool_rope_last ? mtl::Constants().i(53, 1) : mtl::Constants());
    p_idx_score = cx->pipeline(lib, "hy4_flash_indexer_score");
    // the prefill MMA kernels (hy4_flash_attn_pf.metal): built only when a prompt chunk can reach them, and only for the
    // shapes they support (32 heads a threadgroup; the indexer's 16 heads x 128)
    // HY4_FLASH_PREFILL_PROBE=1 is the attention SUB-BATCH offset's negative control, and the MMA attention runs a chunk
    // in one dispatch - there are no sub-batches for it to get wrong, so the probe would inject nothing and the
    // check would be vacuous.  The probe therefore keeps the decode attention (probe 3, the window one key too
    // wide, goes through AttnArgs::window and does reach the MMA kernel).
    const bool mma_asked = pf.chunk > Forward::kMaxWindow && pf.mma != 0 && pf.probe != 1;
    // HY4_FLASH_PREFILL_MMA: 1 = both, 2 = the attention only, 3 = the indexer score only (bisecting a judge failure)
    pf_mma_ok = mma_asked && pf.mma != 3 && kv_w == 576 && kv_lora == 512 && max_heads % 32 == 0;
    pf_mma_idx_ok = mma_asked && pf.mma != 2 && idx_heads == 16 && idx_dim == 128;
    // WHICH PREFILL ATTENTION A RUN IS ON, said once at startup.  A shape this engine's MMA kernels do not cover
    // falls back to the decode kernels, and a run that believes it is testing the MMA path must not be able to be
    // quietly on them instead.
    if (pf.chunk > Forward::kMaxWindow) {
        // ONE line, with the reason for each half when it is off: the judge greps every log for it.
        auto why = [&](bool on, bool asked_off, const char* only, const std::string& shape) {
            if (on) return std::string();
            if (pf.mma == 0) return std::string(" [HY4_FLASH_PREFILL_MMA=0]");
            if (pf.probe == 1) return std::string(" [HY4_FLASH_PREFILL_PROBE=1 needs the sub-batched attention]");
            if (asked_off) return std::string(" [") + only + "]";
            return " [" + shape + "]";
        };
        const std::string ashape = "shape: shared key " + std::to_string(kv_w) + " (576), latent " +
                                   std::to_string(kv_lora) + " (512), max heads " + std::to_string(max_heads) +
                                   " (a multiple of 32)";
        const std::string ishape = "indexer shape: " + std::to_string(idx_heads) + " x " + std::to_string(idx_dim) +
                                   " (16 x 128)";
        std::fprintf(stdout, "  prefill attn   mma %s%s, indexer score mma %s%s (chunk %d)\n",
                     pf_mma_ok ? "on" : "off", why(pf_mma_ok, pf.mma == 3, "HY4_FLASH_PREFILL_MMA=3, indexer only", ashape).c_str(),
                     pf_mma_idx_ok ? "on" : "off",
                     why(pf_mma_idx_ok, pf.mma == 2, "HY4_FLASH_PREFILL_MMA=2, attention only", ishape).c_str(), pf.chunk);
    }
    if (pf_mma_ok) {
        attn_pf_range = cx->pipeline(lib, "hy4_flash_mla_pf5_s32", mtl::Constants().b(43, false));
        attn_pf_sel = cx->pipeline(lib, "hy4_flash_mla_pf5_s32", mtl::Constants().b(43, true));
    }
    if (pf_mma_idx_ok) p_idx_score_pf = cx->pipeline(lib, "hy4_flash_indexer_score_pf");
    p_idx_topk = cx->pipeline(lib, "hy4_flash_indexer_topk",
                              mut.no_tail ? mtl::Constants().i(51, 3) : mtl::Constants());
    attn_range = attn_pipes(cx, lib, false);
    attn_sel = attn_pipes(cx, lib, true);
    p_argmax = cx->pipeline(lib, "hy4_flash_argmax");
    p_copy = cx->pipeline(lib, "hy4_flash_copy_f32");
    p_store_half = cx->pipeline(lib, "hy4_flash_store_half");
    p_add_row = cx->pipeline(lib, "hy4_flash_add_row");
    p_add = cx->pipeline(lib, "hy4_flash_add_f32");
    p_gate_mul = cx->pipeline(lib, "hy4_flash_gate_mul");
    p_eng_rows = cx->pipeline(lib, "hy4_flash_engram_rows");
    p_eng_gate = cx->pipeline(lib, "hy4_flash_engram_gate");
    p_eng_delta = cx->pipeline(lib, "hy4_flash_engram_delta");

    if (const std::string e = resolve(); !e.empty()) return e;
    if (const std::string e = alloc(); !e.empty()) return e;
    if (!pack.empty()) {
        if (const std::string e = eng.open(mm, pack); !e.empty()) return "Engram: " + e;
        eng_ready = true;
    }
    return {};
}

std::string Forward::Impl::resolve() {
    const hy4_flash::Geometry& g = m->geo;
    blk.assign((size_t) g.trunk_blocks, BW());
    uint64_t kv_at = 0, idx_at = 0;
    int idx_n = 0;
    std::string missing;
    auto need = [&](BW& b, int slot, const std::string& name) {
        const gguf::Tensor* t = m->find(name);
        if (t == nullptr && missing.empty()) missing = "no tensor " + name;
        b.t[slot] = t;
    };
    for (int l = 0; l < g.trunk_blocks; ++l) {
        BW& b = blk[(size_t) l];
        const hy4_flash::LayerSpec& s = g.block(l);
        b.id = l;
        b.heads = s.n_heads;
        b.ordinal = s.expert_ordinal;
        b.engram_table = s.engram_table;
        b.sliding = s.attention == hy4_flash::Attention::Sliding;
        b.has_indexer = s.has_indexer;
        b.dense_ffn = s.expert_ordinal < 0;
        b.base = b.sliding ? g.rope_theta_sliding : g.rope_theta_sparse;
        if (mut.rope_theta && b.sliding) b.base = g.rope_theta_sparse;   // the mutation, on a sliding block
        b.tscale = theta_scale_of(b.base, g.rope_dim);
        b.clamp = mut.clamp7 ? 7.0f : g.swiglu_clamp;
        b.kv_off = kv_at;
        kv_at += (uint64_t) ctx * (uint64_t) kv_w;
        const std::string p = "blk." + std::to_string(l) + ".";
        need(b, W_HCA_FN, p + "hc_attn_fn.weight");
        need(b, W_HCA_SCALE, p + "hc_attn_scale.weight");
        need(b, W_HCA_BASE, p + "hc_attn_base.weight");
        need(b, W_ATTN_NORM, p + "attn_norm.weight");
        need(b, W_Q_A, p + "attn_q_a.weight");
        need(b, W_Q_A_NORM, p + "attn_q_a_norm.weight");
        need(b, W_Q_B, p + "attn_q_b.weight");
        need(b, W_KV_A, p + "attn_kv_a_mqa.weight");
        need(b, W_KV_A_NORM, p + "attn_kv_a_norm.weight");
        need(b, W_K_B, p + "attn_k_b.weight");
        need(b, W_V_B, p + "attn_v_b.weight");
        need(b, W_SINKS, p + "attn_sinks.weight");
        need(b, W_AGATE, p + "attn_gate.weight");
        need(b, W_AOUT, p + "attn_output.weight");
        need(b, W_POST_ATTN, p + "post_attention_norm.weight");
        need(b, W_HCF_FN, p + "hc_ffn_fn.weight");
        need(b, W_HCF_SCALE, p + "hc_ffn_scale.weight");
        need(b, W_HCF_BASE, p + "hc_ffn_base.weight");
        need(b, W_FFN_NORM, p + "ffn_norm.weight");
        need(b, W_POST_FFW, p + "post_ffw_norm.weight");
        if (b.dense_ffn) {
            need(b, W_FGATE, p + "ffn_gate.weight");
            need(b, W_FUP, p + "ffn_up.weight");
            need(b, W_FDOWN, p + "ffn_down.weight");
        } else {
            need(b, W_GATE_INP, p + "ffn_gate_inp.weight");
            need(b, W_PROBS_B, p + "exp_probs_b.bias");
            need(b, W_RDOWN, p + "ffn_routed_down.weight");
            need(b, W_RUP, p + "ffn_routed_up.weight");
            need(b, W_SH_G, p + "ffn_gate_shexp.weight");
            need(b, W_SH_U, p + "ffn_up_shexp.weight");
            need(b, W_SH_D, p + "ffn_down_shexp.weight");
        }
        if (b.has_indexer) {
            b.idx_slot = idx_n++;
            b.idx_off = idx_at;
            idx_at += (uint64_t) ctx * 3ull * (uint64_t) idx_dim;
            need(b, W_IQ_B, p + "indexer.attn_q_b.weight");
            need(b, W_IK, p + "indexer.attn_k.weight");
            need(b, W_IK_NORM, p + "indexer.k_norm.weight");
            need(b, W_IPROJ, p + "indexer.proj.weight");
            need(b, W_ICV, p + "indexer_compressor_value.weight");
            need(b, W_ICG, p + "indexer_compressor_gate.weight");
            need(b, W_IAPE, p + "indexer_compressor_ape.weight");
            need(b, W_ICNORM, p + "indexer_compressor_norm.weight");
        }
        if (b.engram_table >= 0) {
            need(b, W_E_LOOK, p + "engram_lookup_norm.weight");
            need(b, W_E_KV, p + "engram_kv.weight");
            need(b, W_E_KVB, p + "engram_kv_b.weight");
            need(b, W_E_GATE, p + "engram_gate.weight");
            need(b, W_E_HCS, p + "engram_hc_scale.weight");
            need(b, W_E_GAMMA, p + "engram_pre_gamma.weight");
        }
    }
    pools_done.assign((size_t) idx_n, 0);
    h_fn = m->find("output_hc_fn.weight");
    h_scale = m->find("output_hc_scale.weight");
    h_base = m->find("output_hc_base.weight");
    h_norm = m->find("output_norm.weight");
    h_out = m->find("output.weight");
    if (h_fn == nullptr || h_scale == nullptr || h_base == nullptr || h_norm == nullptr || h_out == nullptr)
        return "the output head's tensors are not all present";
    return missing;
}

/// The scratch a run of `Tm` positions needs (attention sub-batch `as`, MoE sub-batch `ms`), from allocator `get`.
/// Every buffer here is written before it is read within one window or chunk, so it carries no state between runs.
void Forward::Impl::alloc_scratch(int Tm, int as, int ms, const std::function<mtl::Buffer(size_t)>& get) {
    auto f32 = [&](size_t n) { return get(4 * n); };
    const size_t n_pool_max = (size_t) ctx / (size_t) idx_kpool + 1;
    const int cols = m->geo.engram_heads;
    hcA = f32((size_t) Tm * hcw);
    hcB = f32((size_t) Tm * hcw);
    hc_eng = f32((size_t) Tm * hcw);
    hc_mid = f32((size_t) Tm * hcw);
    flat = f32((size_t) Tm * hcw);
    mixes_a = f32((size_t) Tm * 2 * hc);
    mixes_f = f32((size_t) Tm * 2 * hc);
    pre_a = f32((size_t) Tm * hc);
    post_a = f32((size_t) Tm * hc);
    pre_f = f32((size_t) Tm * hc);
    post_f = f32((size_t) Tm * hc);
    cur = f32((size_t) Tm * n_embd);
    anorm = f32((size_t) Tm * n_embd);
    fnorm = f32((size_t) Tm * n_embd);
    fout = f32((size_t) Tm * n_embd);
    q_resid = f32((size_t) Tm * m->geo.q_lora_rank);
    qb = f32((size_t) Tm * max_heads * qk);
    Q = f32((size_t) Tm * max_heads * kv_w);
    qpe = f32((size_t) Tm * max_heads * rope_dim);
    kpe = f32((size_t) Tm * rope_dim);
    attn_o = f32((size_t) Tm * max_heads * kv_lora);
    kqv = f32((size_t) Tm * max_heads * v_mla);
    gproj = f32((size_t) Tm * max_heads * v_mla);
    gated = f32((size_t) Tm * max_heads * v_mla);
    attn_y = f32((size_t) Tm * n_embd);
    attn_out = f32((size_t) Tm * n_embd);
    kvp = f32((size_t) Tm * kv_w);
    kvn = f32((size_t) Tm * kv_lora);
    sc = f32((size_t) as * max_heads * Forward::kSelCap);
    pos_b = get(4 * (size_t) Tm);
    rope_pos = get(4 * 3 * (size_t) Tm);   // [3][Tm], the M-RoPE planes
    ik = f32((size_t) Tm * idx_dim);
    iv = f32((size_t) Tm * idx_dim);
    ig = f32((size_t) Tm * idx_dim);
    iq = f32((size_t) Tm * idx_heads * idx_dim);
    iwraw = f32((size_t) Tm * idx_heads);
    iw = f32((size_t) Tm * idx_heads);
    score = f32((size_t) Tm * n_pool_max);
    sel = get(4 * (size_t) Tm * Forward::kSelCap);
    nsel = get(4 * (size_t) Tm);
    mlogits = f32((size_t) Tm * n_experts);
    ids = get(4 * (size_t) Tm * topk);
    rw = f32((size_t) Tm * topk);
    pred = f32((size_t) Tm * n_experts);
    pred2 = f32((size_t) Tm * n_experts);   // only written when STRATA_METAL_PREFETCH2 > 0 (stream.h ahead2)
    rdown = f32((size_t) Tm * latent);
    // THE PREPARED PLANES.  With every routed layer on the GEMM (mm_all) a prompt-sized plane is dead weight -
    // 1.64 GB of h plane at a 4096 chunk - because the GEMM reads h as the f32 it is and IQ2_XXS gate/up reads the
    // raw x plane.  But they are NOT dropped: `mm_active` picks the GEMM per WINDOW, from the real T, and a window
    // of T <= kMaxWindow takes the GEMV path even while this set is attached - the last chunk of a 4097-token
    // prompt is one position, and with HY4_FLASH_SCRATCH_SPLIT=0 ordinary decode runs on this set too.  So they are
    // sized for a kMaxWindow window instead of the chunk: 3.3 MB instead of 1.74 GB, and every path stays in
    // bounds.  Sizing them for the GEMM alone would let a GEMV tail write past them.
    const bool mm_raw = mm_all && ms > Forward::kMaxWindow;
    const uint32_t ms_prep = mm_raw ? (uint32_t) Forward::kMaxWindow : (uint32_t) ms;
    xp = f32(strata_hy4_flash_metal::xprep_floats((uint32_t) latent, ms_prep));
    hbuf = f32((size_t) ms * topk * expert_ff);
    hp = f32(strata_hy4_flash_metal::xprep_floats((uint32_t) expert_ff, (uint32_t) (ms_prep * topk)));
    ye = f32((size_t) ms * topk * latent);
    mlat = f32((size_t) Tm * latent);
    upo = f32((size_t) Tm * n_embd);
    shg = f32((size_t) Tm * expert_ff);
    shu = f32((size_t) Tm * expert_ff);
    sha = f32((size_t) Tm * expert_ff);
    sho = f32((size_t) Tm * n_embd);
    ffn_sum = f32((size_t) Tm * n_embd);
    if (m->geo.dense_blocks > 0) {
        dg = f32((size_t) Tm * m->geo.dense_ff);
        du = f32((size_t) Tm * m->geo.dense_ff);
        da = f32((size_t) Tm * m->geo.dense_ff);
    }
    eng_rows = get((size_t) Tm * (size_t) cols * (m->engram_row_bytes != 0 ? m->engram_row_bytes : 1));
    eng_valid = f32((size_t) Tm * cols);
    eng_any = f32((size_t) Tm);
    eng_r = f32((size_t) Tm * m->geo.engram_lookup_dim());
    eng_kv = f32((size_t) Tm * (hc + 1) * n_embd);
    eng_dot = f32((size_t) Tm * hc);
    eng_gate = f32((size_t) Tm * hc);
    eng_delta = f32((size_t) Tm * hcw);
}

/// The prompt's scratch in a region of the arena (stream.h region/hold_region): the region's slots are emptied while a
/// prompt runs and are ordinary expert slots afterwards.  Sized by a dry pass of the same alloc_scratch.
std::string Forward::Impl::attach_prefill() {
    if (!split || !big.empty()) return {};
    const std::vector<mtl::Buffer*> ps = scratch_ptrs();
    std::vector<mtl::Buffer> dec;
    for (mtl::Buffer* b : ps) dec.push_back(*b);
    auto put = [&](const std::vector<mtl::Buffer>& v) { for (size_t i = 0; i < ps.size(); ++i) *ps[i] = v[i]; };
    constexpr size_t pg = 16384;
    auto round = [](size_t n) { return (std::max<size_t>(n, 1) + pg - 1) / pg * pg; };
    size_t need = 0;
    alloc_scratch(tcap, asub, msub, [&](size_t n) { need += round(n); return mtl::Buffer(); });
    uint8_t* at = m->stream.region(need);
    if (at == nullptr) { put(dec); return "no memory for the prompt's scratch"; }
    uint8_t* const end = at + need;
    alloc_scratch(tcap, asub, msub, [&](size_t n) {
        const mtl::Buffer b = cx->wrap(at, round(n));
        at += round(n);
        return b;
    });
    for (mtl::Buffer* b : ps) big.push_back(*b);
    put(dec);
    return at == end ? std::string() : "the prompt's scratch was not laid out as it was sized";
}
void Forward::Impl::swap_scratch() {
    const std::vector<mtl::Buffer*> ps = scratch_ptrs();
    for (size_t i = 0; i < ps.size(); ++i) std::swap(*ps[i], big[i]);
    big_in = !big_in;
}

std::string Forward::Impl::alloc() {
    // Tm is the widest run the scratch must hold: a decode window (8) with prefill off, the chunk size with it on.
    // Three stages are allocated for their SUB-BATCH instead, because at Tm = 1024 they are what does not fit:
    // the score rows (537 GB), the MoE's entry-indexed buffers (635 MB) and the logits (495 MB) - see forward.h.
    // HY4_FLASH_SCRATCH_SPLIT (default on): with a prefill chunk, only the DECODE scratch is allocated here; the prompt's
    // (1.47 GB at 1024 positions) goes into a region of the expert arena while a prompt runs (attach_prefill).
    split = tcap > Forward::kMaxWindow && scratch_split_on();
    if (split) alloc_scratch(Forward::kMaxWindow, Forward::kMaxWindow, Forward::kMaxWindow, [&](size_t n) { return cx->alloc(n); });
    else alloc_scratch(tcap, asub, msub, [&](size_t n) { return cx->alloc(n); });
    auto f32 = [&](size_t n) { return cx->alloc(4 * n); };
    const size_t kv_total = (size_t) blk.size() * (size_t) ctx * (size_t) kv_w;
    size_t idx_blocks = 0;
    for (const BW& b : blk)
        if (b.has_indexer) ++idx_blocks;
    kv = cx->alloc(2 * kv_total);
    idxc = cx->alloc(2 * idx_blocks * (size_t) ctx * 3ull * (size_t) idx_dim);
    const size_t n_pool_max = (size_t) ctx / (size_t) idx_kpool + 1;
    pool32 = f32(n_pool_max * (size_t) idx_dim);
    slots.layout((uint32_t) Forward::kMaxWindow, (uint32_t) topk, 4, (uint32_t) n_experts);
    plan = cx->alloc(4 * slots.buf.size());
    plan_b = cx->alloc(4 * slots.buf.size());
    rms_part = cx->alloc(4 * 8 * Forward::kMaxWindow);
    moe_ind = cx->alloc(4 * 3 * 2 * 4 * 2);
    moe_gx.assign((size_t) m->stream.n_layers(), {0u, 0u});

    if (tcap > Forward::kMaxWindow) {
        pslots.layout((uint32_t) tcap, (uint32_t) topk, 4, (uint32_t) n_experts, (uint32_t) msub);
        pplan = cx->alloc(4 * pslots.buf.size());
        if (moe_mm_mode() != MoeMm::Off) {
            // the tile lists and their grids: (sub-batch, chunk), worst case tiles each, (x, y, z) per kind
            const int tm = moe_mm_tile(moe_mm_mode()).tm;
            mm_cap = MoeMmF32::tile_cap(pslots.g_max, pslots.n_entry, tm);
            mm_slots = pslots.subs_for((uint32_t) tcap) * pslots.n_chunk;
            mm_tiles = cx->alloc(4ull * 2 * mm_cap * mm_slots);
            mm_ind = cx->alloc(4ull * 3 * 2 * mm_slots);
            mm_gx.assign((size_t) m->stream.n_layers(), {0u, 0u});
            const size_t ent = (size_t) msub * topk;
            // What this costs, both ways round.  Against the GEMVs AT THE SAME SUB-BATCH the GEMM saves the
            // prepared h plane; against the GEMVs AS THEY SHIP (a sub-batch of HY4_FLASH_PREFILL_MOE) the prompt's MoE
            // scratch GROWS, because the whole chunk is one plan.
            const double plane = 4.0 * (double) strata_hy4_flash_metal::xprep_floats((uint32_t) expert_ff, (uint32_t) ent);
            const size_t shipped_ent = (size_t) std::min(msub, pf.moe_sub) * topk;
            const double shipped = (double) (shipped_ent * (expert_ff + latent) * 4) +
                                   4.0 * (double) strata_hy4_flash_metal::xprep_floats((uint32_t) expert_ff,
                                                                           (uint32_t) shipped_ent) +
                                   4.0 * (double) strata_hy4_flash_metal::xprep_floats((uint32_t) latent,
                                                                           (uint32_t) std::min(msub, pf.moe_sub));
            const double mine = (double) (ent * (expert_ff + latent) * 4) + (double) (mm_tiles.bytes + mm_ind.bytes);
            const std::string saved =
                mm_all ? "; no prompt-sized prepared planes (" + std::to_string((long) (plane / 1e6)) +
                             " MB of h plane not allocated), prompt MoE scratch " +
                             std::to_string((long) ((mine - shipped) / 1e6)) + " MB against the GEMVs at sub-batch " +
                             std::to_string(std::min(msub, pf.moe_sub))
                       : " (a layer the GEMM does not cover keeps the GEMV sub-batch)";
            std::printf("  prefill moe    grouped GEMM %s: tile %dx%d, sub-batch %d of %d, %u tiles a (sub-batch, "
                        "chunk) list\n                 scratch h %.0f + ye %.0f MB, lists + grids %.2f MB%s\n",
                        strata_hy4_flash_metal::prefill_moe_mm_name(), tm,
                        moe_mm_tile(moe_mm_mode()).tn, msub, tcap, mm_cap,
                        (double) (ent * expert_ff * 4) / 1e6, (double) (ent * latent * 4) / 1e6,
                        (double) (mm_tiles.bytes + mm_ind.bytes) / 1e6,
                        saved.c_str());
        }
    }
    logits_b = f32((size_t) hsub * vocab);
    argmax_b = cx->alloc(4 * (size_t) hsub);
    last_log.assign((size_t) vocab, 0.0f);
    badflag = cx->alloc(16);
    mut_sinks = f32((size_t) max_heads);
    mut_zero = f32((size_t) idx_kpool * (size_t) idx_dim);
    for (int i = 0; i < max_heads; ++i) mut_sinks.as<float>()[i] = -INFINITY;
    h_mixes = f32((size_t) hsub * hc);
    h_cur = f32((size_t) hsub * n_embd);
    h_rnorm = f32((size_t) hsub * n_embd);
    return {};
}

// ---- one block -----------------------------------------------------------------------------------------------------
void Forward::Impl::encode_block(std::vector<mtl::Dispatch>& ds, std::vector<int>& cuts, const BW& b,
                                 const mtl::Buffer& x_in, const mtl::Buffer& out, int T, bool want_dump) {
    const uint32_t uT = (uint32_t) T;
    const uint32_t H = (uint32_t) b.heads;
    const mtl::Buffer* resid = &x_in;

    // ---- Engram, BEFORE the attention pre-mix
    if (b.engram_table >= 0) {
        const int cols = m->geo.engram_heads;
        {
            mtl::Dispatch d;
            d.p = &p_eng_rows;
            EngramRowArgs a{(uint32_t) m->geo.engram_head_dim, (uint32_t) cols, m->engram_row_bytes,
                            (uint32_t) m->geo.engram_lookup_dim(), kEngramEps};
            d.args = {mtl::Arg::B(eng_rows), mtl::Arg::B(m->dense, b.t[W_E_LOOK]->offset), mtl::Arg::B(eng_valid),
                      mtl::Arg::B(eng_r), mtl::Arg::V(a)};
            d.grid[0] = uT * (uint32_t) cols;
            d.tg[0] = 256;
            mtl::require_fit(d, "hy4_flash_engram_rows");
            ds.push_back(d);
        }
        const gguf::Tensor* ekv = b.t[W_E_KV];
        gv(ekv).append(ds, m->dense, ekv->offset, eng_r, (uint32_t) m->geo.engram_lookup_dim(), eng_kv,
                       (uint32_t) ((hc + 1) * n_embd), 0, uT);
        {
            ElemArgs a{(uint32_t) ((hc + 1) * n_embd), (uint32_t) ((hc + 1) * n_embd), 0,
                       (uint32_t) ((hc + 1) * n_embd)};
            elem(ds, p_add_row,
                 {mtl::Arg::B(eng_kv), mtl::Arg::B(m->dense, b.t[W_E_KVB]->offset), mtl::Arg::B(eng_kv),
                  mtl::Arg::V(a)},
                 a.n, T, "hy4_flash_add_row");
        }
        EngramArgs ea{(uint32_t) n_embd, (uint32_t) hc, (uint32_t) hcw, (uint32_t) ((hc + 1) * n_embd),
                      std::sqrt((float) n_embd), kEngramEps};
        {
            mtl::Dispatch d;
            d.p = &p_eng_gate;
            d.args = {mtl::Arg::B(x_in), mtl::Arg::B(eng_kv), mtl::Arg::B(m->dense, b.t[W_E_GATE]->offset),
                      mtl::Arg::B(eng_dot), mtl::Arg::B(eng_gate), mtl::Arg::V(ea)};
            d.grid[0] = (uint32_t) hc;
            d.grid[1] = uT;
            d.tg[0] = 256;
            mtl::require_fit(d, "hy4_flash_engram_gate");
            ds.push_back(d);
        }
        {
            mtl::Dispatch d;
            d.p = &p_eng_delta;
            d.args = {mtl::Arg::B(x_in), mtl::Arg::B(eng_kv), mtl::Arg::B(m->dense, b.t[W_E_GAMMA]->offset),
                      mtl::Arg::B(m->dense, b.t[W_E_HCS]->offset), mtl::Arg::B(eng_gate), mtl::Arg::B(eng_any),
                      mtl::Arg::B(eng_delta), mtl::Arg::B(hc_eng), mtl::Arg::V(ea)};
            d.grid[0] = (uint32_t) hc;
            d.grid[1] = uT;
            d.tg[0] = 256;
            mtl::require_fit(d, "hy4_flash_engram_delta");
            ds.push_back(d);
        }
        resid = &hc_eng;
    }

    // ---- attention pre-mix
    hc_pre(ds, b, *resid, 0, T, b.t[W_ATTN_NORM], &anorm, eps);   // + attn_norm(cur) -> anorm

    // ---- MLA
    const gguf::Tensor* qa = b.t[W_Q_A];
    gv(qa).append(ds, m->dense, qa->offset, anorm, (uint32_t) n_embd, q_resid, (uint32_t) m->geo.q_lora_rank, 0, uT);
    rms(ds, q_resid, 0, (uint32_t) m->geo.q_lora_rank, b.t[W_Q_A_NORM], q_resid, 0,
        (uint32_t) m->geo.q_lora_rank, (uint32_t) m->geo.q_lora_rank, T, eps);
    const gguf::Tensor* qbw = b.t[W_Q_B];
    gv(qbw).append(ds, m->dense, qbw->offset, q_resid, (uint32_t) m->geo.q_lora_rank, qb, H * (uint32_t) qk, 0, uT);
    rope(ds, qb, H, (uint32_t) qk, (uint32_t) (qk - rope_dim), H * (uint32_t) qk, b.tscale, T);
    // q_abs = wk_b[h] . q_nope[h], every head in one dispatch (grid.y = head, bitwise the per-head dispatches it
    // replaces - gemv.metal's note on function constants 10-12); then the rotated tail is copied in beside it
    const gguf::Tensor* kb = b.t[W_K_B];
    const uint32_t nope = (uint32_t) (qk - rope_dim);
    const size_t kb_head = (size_t) kv_lora * ref::row_bytes(kb->type, (int) nope);
    gvh(kb->type, nope, (uint32_t) kv_lora, GemvHeads{(uint32_t) kb_head, (uint32_t) qk, (uint32_t) kv_w})
        .append_heads(ds, m->dense, kb->offset, qb, H * (uint32_t) qk, Q, H * (uint32_t) kv_w, 0, uT, H);
    {
        ElemArgs a{(uint32_t) rope_dim, (uint32_t) qk, 0, (uint32_t) kv_w};
        elem(ds, p_copy,
             {mtl::Arg::B(qb, 4 * (size_t) nope), mtl::Arg::B(Q, 4 * (size_t) kv_lora), mtl::Arg::V(a)}, a.n,
             T * (int) H, "hy4_flash_copy_f32 (q_pe)");
        // and a packed copy of the same values, so the dump can carry the reference's q_pe node - only when a
        // dump is taken: nothing else reads qpe
        if (want_dump) {
            ElemArgs a2{(uint32_t) rope_dim, (uint32_t) qk, 0, (uint32_t) rope_dim};
            elem(ds, p_copy, {mtl::Arg::B(qb, 4 * (size_t) nope), mtl::Arg::B(qpe), mtl::Arg::V(a2)}, a2.n,
                 T * (int) H, "hy4_flash_copy_f32 (q_pe dump)");
        }
    }
    // the shared key: rope on its tail, the latent normed, both into the f16 cache
    const gguf::Tensor* kva = b.t[W_KV_A];
    gv(kva).append(ds, m->dense, kva->offset, anorm, (uint32_t) n_embd, kvp, (uint32_t) kv_w, 0, uT);
    // rope on the tail, rms on the latent, both stores: one dispatch (hy4_flash_attn.metal hy4_flash_kv_finish, bitwise the
    // hy4_flash_rope_norm + hy4_flash_rmsnorm + 2 x hy4_flash_store_half it replaces)
    const size_t kv_row = (size_t) b.kv_off + (size_t) p0 * (size_t) kv_w;
    {
        mtl::Dispatch d;
        const bool mr = images();
        d.p = mr ? &p_kv_finish_mrope : &p_kv_finish;
        KvFinishArgs a{(uint32_t) kv_lora, (uint32_t) rope_dim, (uint32_t) kv_w, (uint32_t) kv_lora, eps,
                       b.tscale, mr ? rope_plane() : 0u, mr ? (uint32_t) img_mlen1 : 0u,
                       mr ? (uint32_t) img_mlen2 : 0u};
        d.args = {mtl::Arg::B(kvp), mtl::Arg::B(m->dense, b.t[W_KV_A_NORM]->offset), mtl::Arg::B(kvn),
                  mtl::Arg::B(kv, 2 * kv_row), mtl::Arg::B(mr ? rope_pos : pos_b), mtl::Arg::B(badflag),
                  mtl::Arg::V(a)};
        d.grid[0] = uT;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_kv_finish");
        ds.push_back(d);
    }
    if (want_dump) {   // the reference's k_pe node; nothing else reads kpe
        ElemArgs a{(uint32_t) rope_dim, (uint32_t) kv_w, 0, (uint32_t) rope_dim};
        elem(ds, p_copy, {mtl::Arg::B(kvp, 4 * (size_t) kv_lora), mtl::Arg::B(kpe), mtl::Arg::V(a)}, a.n, T,
             "hy4_flash_copy_f32 (k_pe dump)");
    }

    // ---- the indexer, on the sparse blocks
    uint32_t n_pool = 0;
    if (b.has_indexer) {
        const gguf::Tensor* ikw = b.t[W_IK];
        gv(ikw).append(ds, m->dense, ikw->offset, anorm, (uint32_t) n_embd, ik, (uint32_t) idx_dim, 0, uT);
        rms(ds, ik, 0, (uint32_t) idx_dim, b.t[W_IK_NORM], ik, 0, (uint32_t) idx_dim, (uint32_t) idx_dim, T, eps);
        const gguf::Tensor* cv = b.t[W_ICV];
        const gguf::Tensor* cg = b.t[W_ICG];
        gv(cv).append(ds, m->dense, cv->offset, ik, (uint32_t) idx_dim, iv, (uint32_t) idx_dim, 0, uT);
        gv(cg).append(ds, m->dense, cg->offset, ik, (uint32_t) idx_dim, ig, (uint32_t) idx_dim, 0, uT);
        const size_t irow = (size_t) b.idx_off + (size_t) p0 * 3ull * (size_t) idx_dim;
        ElemArgs ia{(uint32_t) idx_dim, (uint32_t) idx_dim, 0, 3u * (uint32_t) idx_dim};
        elem(ds, p_store_half, {mtl::Arg::B(iv), mtl::Arg::B(idxc, 2 * irow), mtl::Arg::V(ia), mtl::Arg::B(badflag)},
             ia.n, T, "hy4_flash_store_half (indexer value)");
        elem(ds, p_store_half,
             {mtl::Arg::B(ig), mtl::Arg::B(idxc, 2 * (irow + (size_t) idx_dim)), mtl::Arg::V(ia),
              mtl::Arg::B(badflag)},
             ia.n, T, "hy4_flash_store_half (indexer gate)");
        // the pooled keys of every group this window completed
        const int done = pools_done[(size_t) b.idx_slot];
        const int have = (p0 + T) / idx_kpool;
        n_pool = (uint32_t) have;
        if (have > done) {
            mtl::Dispatch d;
            d.p = &p_idx_kpool;
            KpoolArgs a{(uint32_t) idx_dim,
                        (uint32_t) idx_kpool,
                        (uint32_t) done,
                        (uint32_t) (have - done),
                        3u * (uint32_t) idx_dim,
                        3u * (uint32_t) idx_dim,
                        (uint32_t) idx_dim,
                        (uint32_t) idx_dim,
                        3u * (uint32_t) idx_dim * (uint32_t) idx_kpool,
                        (uint32_t) rope_dim,
                        eps,
                        b.tscale};
            const size_t pool16 = (size_t) b.idx_off + (size_t) done * (size_t) idx_kpool * 3ull * (size_t) idx_dim +
                                  2ull * (size_t) idx_dim;
            d.args = {mtl::Arg::B(idxc, 2 * (size_t) b.idx_off),
                      mtl::Arg::B(idxc, 2 * ((size_t) b.idx_off + (size_t) idx_dim)),
                      mut.no_ape ? mtl::Arg::B(mut_zero) : mtl::Arg::B(m->dense, b.t[W_IAPE]->offset),
                      mtl::Arg::B(m->dense, b.t[W_ICNORM]->offset),
                      mtl::Arg::B(pool32, 4 * (size_t) done * (size_t) idx_dim),
                      mtl::Arg::B(idxc, 2 * pool16),
                      mtl::Arg::V(a)};
            d.grid[0] = (uint32_t) (have - done);
            d.tg[0] = 256;
            mtl::require_fit(d, "hy4_flash_indexer_kpool");
            ds.push_back(d);
        }
        const gguf::Tensor* iqb = b.t[W_IQ_B];
        gv(iqb).append(ds, m->dense, iqb->offset, q_resid, (uint32_t) m->geo.q_lora_rank, iq,
                       (uint32_t) (idx_heads * idx_dim), 0, uT);
        rope(ds, iq, (uint32_t) idx_heads, (uint32_t) idx_dim, (uint32_t) (idx_dim - rope_dim),
             (uint32_t) (idx_heads * idx_dim), b.tscale, T);
        const gguf::Tensor* ip = b.t[W_IPROJ];
        gv(ip).append(ds, m->dense, ip->offset, anorm, (uint32_t) n_embd, iwraw, (uint32_t) idx_heads, 0, uT);
        {
            IdxWeightArgs a{(uint32_t) idx_heads, (uint32_t) idx_heads, (uint32_t) idx_heads,
                            std::sqrt((float) (idx_dim * idx_heads))};
            elem(ds, p_idx_w, {mtl::Arg::B(iwraw), mtl::Arg::B(iw), mtl::Arg::V(a)}, a.n, T,
                 "hy4_flash_indexer_weights");
        }
        // The indexer's score, in SUB-BATCHES of `isub` positions: the buffer is n_tok x (ctx / kpool + 1) floats -
        // 134 MB for a chunk of 4096 at 32K.  Every value is a function of one (token, pool) pair, so splitting the
        // dispatch moves no value; `score` is bound at the sub-batch's own rows, so the sub-batches are independent
        // (unlike the attention's `sc`, which is reused).  A decode window is one sub-batch.
        if (n_pool > 0) {
            const int isub = T > Forward::kMaxWindow ? std::max(1, pf.idx_sub) : T;
            for (int t0 = 0; t0 < T; t0 += isub) {
                const uint32_t nt = (uint32_t) std::min(isub, T - t0);
                ScoreArgs a{(uint32_t) idx_dim,
                            (uint32_t) idx_heads,
                            n_pool,
                            nt,
                            (uint32_t) idx_kpool,
                            (uint32_t) (idx_heads * idx_dim),
                            3u * (uint32_t) idx_dim * (uint32_t) idx_kpool,
                            (uint32_t) idx_heads,
                            (uint32_t) ((size_t) ctx / (size_t) idx_kpool + 1)};
                mtl::Dispatch d;
                const bool mma = pf_mma_idx_ok && T > Forward::kMaxWindow;
                d.p = mma ? &p_idx_score_pf : &p_idx_score;
                d.args = {mtl::Arg::B(iq, 4 * (size_t) t0 * (size_t) (idx_heads * idx_dim)),
                          mtl::Arg::B(idxc, 2 * ((size_t) b.idx_off + 2ull * (size_t) idx_dim)),
                          mtl::Arg::B(iw, 4 * (size_t) t0 * (size_t) idx_heads),
                          mtl::Arg::B(pos_b, 4 * (size_t) t0),
                          mtl::Arg::B(score, 4 * (size_t) t0 * ((size_t) ctx / (size_t) idx_kpool + 1)),
                          mtl::Arg::V(a)};
                if (mma) {
                    d.grid[0] = (n_pool + 31) / 32;
                    d.grid[1] = (nt + 3) / 4;
                } else {
                    d.grid[0] = n_pool;
                    d.grid[1] = nt;
                }
                d.tg[0] = 256;
                mtl::require_fit(d, "hy4_flash_indexer_score");
                ds.push_back(d);
            }
        }
        if (inj == nullptr || inj->sel == nullptr) {
            mtl::Dispatch d;
            d.p = &p_idx_topk;
            TopkArgs a{(uint32_t) idx_kpool, (uint32_t) idx_topk, n_pool, uT,
                       (uint32_t) ((size_t) ctx / (size_t) idx_kpool + 1), (uint32_t) Forward::kSelCap};
            d.args = {mtl::Arg::B(score), mtl::Arg::B(pos_b), mtl::Arg::B(sel), mtl::Arg::B(nsel), mtl::Arg::V(a)};
            d.grid[0] = uT;
            d.tg[0] = 256;
            mtl::require_fit(d, "hy4_flash_indexer_topk");
            ds.push_back(d);
        }
    }

    // ---- attention itself (three dispatches: the score row, the softmax, PV - attn.h).  In SUB-BATCHES of `asub`
    // positions, because the score scratch is n_tok x n_head x 2051 floats - 4.2 MB for a decode window but 537 GB
    // for a prefill chunk of 1024.  Every value is a function of one (head, position) row of q / sel / sc, so
    // splitting the dispatch moves no value (hy4_flash_attn.metal's T-invariance note); a decode window is one sub-batch.
    //
    // THESE DISPATCHES MUST NOT BE RELAXED (no `relax_after` over this loop).  `sc` is sized `asub x max_heads x
    // kSelCap` and bound WITHOUT a t0 offset, so the kernel's `(t * n_head + h)` uses the position's index WITHIN
    // the sub-batch: successive sub-batches deliberately reuse the same scratch rows.  That is safe only while
    // they are serialised by the barrier each dispatch carries.  Letting them overlap - as the MoE and the GEMV
    // splits now do - would make two sub-batches write the same `sc` rows concurrently.  To relax them,
    // `sc` has to be bound at `4 * t0 * max_heads * kSelCap` and sized for the whole window first.
    // A PROMPT CHUNK runs the MMA attention instead (hy4_flash_attn_pf.metal): one dispatch for the whole chunk, no score
    // scratch (its softmax is online), so neither `sc` nor the sub-batch loop below applies to it.
    // PER BLOCK, not max_heads: the kernel stores 8-row blocks of 32 heads, so a block whose head count is not a
    // multiple of 32 keeps the decode attention (its last partial block would write the next token's rows).
    const bool attn_mma = pf_mma_ok && T > Forward::kMaxWindow && H % 32 == 0;
    for (int t0 = 0; t0 < T; t0 += (attn_mma ? T : asub)) {
        const uint32_t nt = (uint32_t) std::min(attn_mma ? T : asub, T - t0);
        AttnArgs a{H,
                   (uint32_t) kv_w,
                   (uint32_t) kv_lora,
                   (uint32_t) kv_w,
                   H * (uint32_t) kv_w,
                   H * (uint32_t) kv_lora,
                   (uint32_t) Forward::kSelCap,
                   (uint32_t) Forward::kSelCap,
                   b.sliding ? (uint32_t) (m->geo.sliding_window +
                                         (mut.swa_plus1 || (pf.probe == 3 && pr_in_chunk) ? 1 : 0)) : 0u,
                   1.0f / std::sqrt((float) qk)};
        const size_t probe_pos = pf.probe == 1 && pr_in_chunk ? 0 : 4 * (size_t) t0;
        // the bound on a row's key count: the window for a sliding block, the scratch row for a selection
        const uint32_t n_cap = !b.has_indexer && a.window != 0 ? a.window : (uint32_t) Forward::kSelCap;
        const std::vector<mtl::Arg> args = {mtl::Arg::B(Q, 4 * (size_t) t0 * H * (size_t) kv_w),
                  mtl::Arg::B(kv, 2 * (size_t) b.kv_off),
                  mut.no_sinks ? mtl::Arg::B(mut_sinks) : mtl::Arg::B(m->dense, b.t[W_SINKS]->offset),
                  mtl::Arg::B(sel, 4 * (size_t) t0 * (size_t) Forward::kSelCap),
                  mtl::Arg::B(nsel, 4 * (size_t) t0),
                  mtl::Arg::B(pos_b, probe_pos),
                  mtl::Arg::B(sc),
                  mtl::Arg::B(attn_o, 4 * (size_t) t0 * H * (size_t) kv_lora),
                  mtl::Arg::V(a)};
        if (attn_mma) {
            // the MMA kernel's eight arguments: q, kv, sinks, sel, n_sel, pos, o, the args bytes (no score scratch)
            mtl::Dispatch d;
            d.p = b.has_indexer ? &attn_pf_sel : &attn_pf_range;
            d.args = {args[0], args[1], args[2], args[3], args[4], args[5], args[7], args[8]};
            d.grid[0] = nt;
            d.grid[1] = (H + 31) / 32;
            d.tg[0] = 1024;
            mtl::require_fit(d, "hy4_flash_mla_pf5_s32");
            ds.push_back(d);
        } else {
            append_attn(ds, b.has_indexer ? attn_sel : attn_range, args, H, nt, n_cap, a.sc_stride, (uint32_t) kv_w,
                        (uint32_t) kv_lora, (uint32_t) kv_w, "hy4_flash_mla_attn");
        }
    }

    // kqv = wv_b[h] . o[h]
    const gguf::Tensor* vb = b.t[W_V_B];
    const size_t vb_head = (size_t) v_mla * ref::row_bytes(vb->type, kv_lora);
    gvh(vb->type, (uint32_t) kv_lora, (uint32_t) v_mla,
        GemvHeads{(uint32_t) vb_head, (uint32_t) kv_lora, (uint32_t) v_mla})
        .append_heads(ds, m->dense, vb->offset, attn_o, H * (uint32_t) kv_lora, kqv, H * (uint32_t) v_mla, 0, uT, H);
    const gguf::Tensor* ag = b.t[W_AGATE];
    gv(ag).append(ds, m->dense, ag->offset, anorm, (uint32_t) n_embd, gproj, H * (uint32_t) v_mla, 0, uT);
    {
        ElemArgs a{H * (uint32_t) v_mla, H * (uint32_t) v_mla, H * (uint32_t) v_mla, H * (uint32_t) v_mla};
        elem(ds, p_gate_mul, {mtl::Arg::B(kqv), mtl::Arg::B(gproj), mtl::Arg::B(gated), mtl::Arg::V(a)}, a.n, T,
             "hy4_flash_gate_mul");
    }
    const gguf::Tensor* ao = b.t[W_AOUT];
    gv(ao).append(ds, m->dense, ao->offset, gated, H * (uint32_t) v_mla, attn_y, (uint32_t) n_embd, 0, uT);
    rms(ds, attn_y, 0, (uint32_t) n_embd, b.t[W_POST_ATTN], attn_out, 0, (uint32_t) n_embd, (uint32_t) n_embd, T,
        eps);
    hc_post(ds, attn_out, *resid, post_a, hc_mid, T);

    // ---- the FFN half
    hc_pre(ds, b, hc_mid, 1, T, b.t[W_FFN_NORM], &fnorm, eps);   // + ffn_norm(cur) -> fnorm
    const mtl::Buffer* ffn_result = nullptr;
    if (b.dense_ffn) {
        const gguf::Tensor* wg = b.t[W_FGATE];
        const gguf::Tensor* wu = b.t[W_FUP];
        const gguf::Tensor* wd = b.t[W_FDOWN];
        gv(wg).append(ds, m->dense, wg->offset, fnorm, (uint32_t) n_embd, dg, (uint32_t) m->geo.dense_ff, 0, uT);
        gv(wu).append(ds, m->dense, wu->offset, fnorm, (uint32_t) n_embd, du, (uint32_t) m->geo.dense_ff, 0, uT);
        glu(ds, dg, du, da, (uint32_t) m->geo.dense_ff, T, INFINITY);   // the dense FFN does NOT clamp
        gv(wd).append(ds, m->dense, wd->offset, da, (uint32_t) m->geo.dense_ff, ffn_sum, (uint32_t) n_embd, 0, uT);
        ffn_result = &ffn_sum;
    } else {
        // the router, then the cut: the host reads the ids, puts the experts in the arena and writes the plan
        const gguf::Tensor* gi = b.t[W_GATE_INP];
        gv(gi).append(ds, m->dense, gi->offset, fnorm, (uint32_t) n_embd, mlogits, (uint32_t) n_experts, 0, uT);
        {
            mtl::Dispatch d;
            d.p = &p_router;
            RouterArgs a{(uint32_t) n_experts, (uint32_t) topk,   (uint32_t) n_experts, (uint32_t) topk,
                         (uint32_t) topk,      uT,                m->geo.routed_scale,  6.103515625e-5f};
            d.args = {mtl::Arg::B(mlogits), mtl::Arg::B(m->dense, b.t[W_PROBS_B]->offset), mtl::Arg::B(ids),
                      mtl::Arg::B(rw), mtl::Arg::V(a)};
            d.grid[0] = uT;
            d.tg[0] = 256;
            mtl::require_fit(d, "hy4_flash_router");
            ds.push_back(d);
        }
        // the next ordinal's router on THIS ordinal's FFN input: the read-ahead's prediction.  Not in a
        // prefill chunk: `ExpertStream::at_layer`'s prompt path streams the WHOLE next layer in rather than a
        // predicted top-k, so the prediction would be computed and thrown away.
        if (prefetch_enabled() && T <= Forward::kMaxWindow && b.ordinal + 1 < m->stream.n_layers()) {
            const BW* nb = nullptr;
            for (const BW& c : blk)
                if (c.ordinal == b.ordinal + 1) nb = &c;
            if (nb != nullptr && nb->t[W_GATE_INP] != nullptr) {
                const gguf::Tensor* ngi = nb->t[W_GATE_INP];
                gv(ngi).append(ds, m->dense, ngi->offset, fnorm, (uint32_t) n_embd, pred, (uint32_t) n_experts, 0,
                               uT);
            }
            // and the ordinal after it, from the same input (STRATA_METAL_PREFETCH2 = its top-k): a second, longer
            // lead for the read-ahead.  A GEMV into a buffer only the host reads - no trunk value depends on it.
            if (m->stream.ahead2() > 0 && b.ordinal + 2 < m->stream.n_layers()) {
                const BW* nb2 = nullptr;
                for (const BW& c : blk)
                    if (c.ordinal == b.ordinal + 2) nb2 = &c;
                if (nb2 != nullptr && nb2->t[W_GATE_INP] != nullptr) {
                    const gguf::Tensor* ngi = nb2->t[W_GATE_INP];
                    gv(ngi).append(ds, m->dense, ngi->offset, fnorm, (uint32_t) n_embd, pred2, (uint32_t) n_experts,
                                   0, uT);
                }
            }
        }
        cuts.push_back((int) ds.size());
        cut_block.push_back(b.id);
        // the MoE split (moe_split_min): decided here, from T alone, so the encoder and at_cut agree
        const bool split = moe_split_min() > 0 && T >= moe_split_min() && T <= Forward::kMaxWindow &&
                           plan_of(T).subs_for(uT) == 1;
        cut_role.push_back(split ? 1 : 0);

        // Between the router's signal and the wait for the host: what reads only fnorm.  The shared expert
        // (plain SiLU on the FULL-WIDTH cur, no clamp) and the routed experts' latent down-projection
        // depend on the FFN input alone, not on the ids, the weights, the plan or the arena the host is about to
        // write - so the GPU computes them while the host waits for the SSD instead of idling.  Only WHEN they run
        // moves; each writes its own buffers (shg/shu/sha/sho, rdown), so every value is the same bit for bit.
        const gguf::Tensor* sg = b.t[W_SH_G];
        const gguf::Tensor* su = b.t[W_SH_U];
        const gguf::Tensor* sd = b.t[W_SH_D];
        auto shared_expert = [&] {
            gv(sg).append(ds, m->dense, sg->offset, fnorm, (uint32_t) n_embd, shg, (uint32_t) expert_ff, 0, uT);
            gv(su).append(ds, m->dense, su->offset, fnorm, (uint32_t) n_embd, shu, (uint32_t) expert_ff, 0, uT);
            glu(ds, shg, shu, sha, (uint32_t) expert_ff, T, INFINITY);
            gv(sd).append(ds, m->dense, sd->offset, sha, (uint32_t) expert_ff, sho, (uint32_t) n_embd, 0, uT);
        };
        const bool early = shared_early();
        if (early) shared_expert();
        const gguf::Tensor* rd = b.t[W_RDOWN];
        gv(rd).append(ds, m->dense, rd->offset, fnorm, (uint32_t) n_embd, rdown, (uint32_t) latent, 0, uT);
        cut_wait.push_back(early ? (int) ds.size() : cuts.back());
        const ExpertSlices& es = m->experts[(size_t) b.ordinal];
        const MoeF32& mo = moe(es.type[kGate], es.type[kDown]);
        MoeArgs ma{};
        ma.gate_off = 0;
        ma.up_off = m->stream.up_in_slot(b.ordinal);
        ma.down_off = m->stream.down_in_slot(b.ordinal);
        ma.gu_stride = 16384;   // the arena addresses a slot by its 16 KiB page (stream.h's smap)
        ma.d_stride = 16384;
        ma.gu_row = (uint32_t) ref::row_bytes(es.type[kGate], latent);
        ma.d_row = (uint32_t) ref::row_bytes(es.type[kDown], expert_ff);
        ma.n_latent = (uint32_t) latent;
        ma.n_ff = (uint32_t) expert_ff;
        ma.topk = (uint32_t) topk;
        ma.limit = b.clamp;
        // The experts in SUB-BATCHES of positions (MoeSlots' note): h, its prepared plane and ye are indexed by
        // entry, so a chunk of 1024 x 16 entries would need 635 MB of them.  Every sub-batch reads the same
        // experts out of the arena - the SSD read happened once per chunk, at the cut above - and an entry's
        // arithmetic cannot see which sub-batch it is in, so this changes no value.
        MoeSlots& ps = plan_of(T);
        const mtl::Buffer& pb = plan_buf(T);
        // One dispatch per arena BUFFER, and only for the buffers the arena really has: the plan is laid out for
        // the maximum of four (its sections must exist before the arena is built - main.cpp's ordering), but a
        // dispatch for a buffer that does not exist has every group slot empty and STILL launches g_max x
        // (n_ff / rows per group) thread groups, 49k of them at a prefill chunk's g_max of 512.
        const uint32_t nch = std::min(ps.n_chunk, (uint32_t) std::max(1, m->stream.n_chunks()));
        for (uint32_t s = 0; s < ps.subs_for(uT); ++s) {
            const uint32_t t0 = ps.sub_first(s), nt = ps.sub_tokens(uT, s);
            MoeArgs sa = ma;
            sa.n_tok = nt;
            const size_t xo = 4 * (size_t) t0 * (size_t) latent;
            const size_t xo_in = pf.probe == 2 && pr_in_chunk ? 0 : xo;
            mo.append_xprep_x(ds, rdown, xo_in, xp, nt);   // nothing for IQ2_XXS gate/up (raw plane)
            // gate/up, prepare h, down - over the plan `p` (the whole plan, or one half of a split one)
            // a decode window's grids come from moe_ind (moe_indirect): exactly the groups of this half and chunk
            const bool ind = moe_indirect() && T <= Forward::kMaxWindow && ps.subs_for(uT) == 1 && nch <= 4;
            auto indirect = [&](int h, uint32_t c, int kind, size_t before) {
                if (!ind || ds.size() == before) return;   // append_* emitted nothing
                mtl::Dispatch& d = ds.back();
                moe_gx.at((size_t) b.ordinal)[(size_t) kind] = d.grid[0];
                d.indirect = &moe_ind;
                d.ind_off = 4 * 3 * (((size_t) h * 4 + c) * 2 + (size_t) kind);
            };
            auto half = [&](const mtl::Buffer& p, int h) {
                size_t at = ds.size();
                for (uint32_t c = 0; c < nch; ++c) {
                    const size_t n0 = ds.size();
                    mo.append_gu(ds, ps.for_chunk(sa, c, s), m->stream.chunk((int) c), p, rdown, xo_in, xp, hbuf);
                    indirect(h, c, 0, n0);
                }
                relax_after(ds, at, kOverlapMoeGu);   // the arena chunks hold disjoint groups, so disjoint entry rows
                mo.append_xprep_h(ds, hbuf, hp, nt * (uint32_t) topk);
                at = ds.size();
                for (uint32_t c = 0; c < nch; ++c) {
                    const size_t n0 = ds.size();
                    mo.append_down(ds, ps.for_chunk(sa, c, s), m->stream.chunk((int) c), p, hbuf, hp, ye);
                    indirect(h, c, 1, n0);
                }
                relax_after(ds, at, kOverlapMoeDown);
            };
            // A PREFILL chunk's experts as a grouped GEMM (mm_active): the same plan, plus a tile list per
            // (sub-batch, chunk) that at_cut fills, and an indirect grid of exactly that list's length.  No
            // xprep of h - the down GEMM dequantizes Q4_K itself and reads h as the f32 it is - and gate/up reads
            // the raw x plane (IQ2_XXS always did).  The combine below is decode's, in slot order.
            if (mm_active(b.ordinal, T, nch)) {
                const MoeMmF32& mm = moe_mm(es.type[kGate], es.type[kDown]);
                Hy4FlashMmArgs mmb{};
                mmb.gate_off = ma.gate_off;
                mmb.up_off = ma.up_off;
                mmb.down_off = ma.down_off;
                mmb.gu_stride = ma.gu_stride;
                mmb.d_stride = ma.d_stride;
                mmb.gu_row = ma.gu_row;
                mmb.d_row = ma.d_row;
                mmb.n_latent = ma.n_latent;
                mmb.n_ff = ma.n_ff;
                mmb.n_tile = mm_cap;   // the guard's bound; the real count is the indirect grid's y
                mmb.limit = ma.limit;
                auto mm_ind_at = [&](uint32_t c, int kind, size_t before) {
                    if (ds.size() == before) return;
                    const size_t slot = (size_t) s * ps.n_chunk + c;
                    mtl::Dispatch& d = ds.back();
                    mm_gx.at((size_t) b.ordinal)[(size_t) kind] = d.grid[0];
                    d.indirect = &mm_ind;
                    d.ind_off = 4 * 3 * (slot * 2 + (size_t) kind);
                };
                size_t at = ds.size();
                for (uint32_t c = 0; c < nch; ++c) {
                    const MoeArgs ca = ps.for_chunk(sa, c, s);
                    Hy4FlashMmArgs a = mmb;
                    a.off_gexp = ca.off_gexp;
                    a.off_gstart = ca.off_gstart;
                    a.off_etok = ca.off_etok;
                    a.off_ew = ca.off_ew;
                    const size_t to = 4ull * 2 * mm_cap * ((size_t) s * ps.n_chunk + c);
                    const size_t n0 = ds.size();
                    mm.append_gu(ds, a, m->stream.chunk((int) c), pb, mm_tiles, rdown, xo_in, hbuf, to);
                    mm_ind_at(c, 0, n0);
                }
                relax_after(ds, at, kOverlapMoeGu);
                at = ds.size();
                for (uint32_t c = 0; c < nch; ++c) {
                    const MoeArgs ca = ps.for_chunk(sa, c, s);
                    Hy4FlashMmArgs a = mmb;
                    a.off_gexp = ca.off_gexp;
                    a.off_gstart = ca.off_gstart;
                    a.off_etok = ca.off_etok;
                    a.off_ew = ca.off_ew;
                    const size_t to = 4ull * 2 * mm_cap * ((size_t) s * ps.n_chunk + c);
                    const size_t n0 = ds.size();
                    mm.append_down(ds, a, m->stream.chunk((int) c), pb, mm_tiles, hbuf, ye, to);
                    mm_ind_at(c, 1, n0);
                }
                relax_after(ds, at, kOverlapMoeDown);
            } else if (split) {
                // The resident half runs between the second cut's signal and its wait, i.e. while the host waits
                // for the SSD (finish_layer); the late half after it.  The late half's prepare rewrites the resident
                // rows of the prepared plane from the same h, so they come out the same; Metal's hazard tracking
                // orders the two halves' encoders (every buffer here is a tracked shared buffer).
                cuts.push_back(cut_wait.back());
                cut_block.push_back(b.id);
                cut_role.push_back(2);
                half(pb, 0);
                cut_wait.push_back((int) ds.size());
                half(plan_b, 1);
            } else {
                half(pb, 0);
            }
            mo.append_combine(ds, ps.for_chunk(sa, 0, s), ye, pb, mlat, xo);
        }

        const gguf::Tensor* ru = b.t[W_RUP];
        gv(ru).append(ds, m->dense, ru->offset, mlat, (uint32_t) latent, upo, (uint32_t) n_embd, 0, uT);
        if (!early) shared_expert();   // HY4_FLASH_SHARED_EARLY=0: after the routed experts, inside the wait
        {
            ElemArgs a{(uint32_t) n_embd, (uint32_t) n_embd, (uint32_t) n_embd, (uint32_t) n_embd};
            elem(ds, p_add, {mtl::Arg::B(upo), mtl::Arg::B(sho), mtl::Arg::B(ffn_sum), mtl::Arg::V(a)}, a.n, T,
                 "hy4_flash_add_f32");
        }
        ffn_result = &ffn_sum;
    }
    rms(ds, *ffn_result, 0, (uint32_t) n_embd, b.t[W_POST_FFW], fout, 0, (uint32_t) n_embd, (uint32_t) n_embd, T,
        eps);
    hc_post(ds, fout, hc_mid, post_f, out, T);
    (void) want_dump;
}

void Forward::Impl::encode_head(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, int x_t0, int T,
                                bool with_logits) {
    const uint32_t uT = (uint32_t) T;
    const size_t xo = 4 * (size_t) x_t0 * (size_t) hcw;   // the head's own buffers are indexed from 0
    rms(ds, x, xo, (uint32_t) hcw, nullptr, flat, 0, (uint32_t) hcw, (uint32_t) hcw, T, eps);
    gv(h_fn).append(ds, m->dense, h_fn->offset, flat, (uint32_t) hcw, h_mixes, (uint32_t) hc, 0, uT);
    {
        mtl::Dispatch d;
        d.p = &p_hc_head;
        HcArgs a{(uint32_t) n_embd, (uint32_t) hcw, (uint32_t) hc, (uint32_t) n_embd, m->geo.hc_eps,
                 m->geo.hc_magnitude};
        d.args = {mtl::Arg::B(x, xo), mtl::Arg::B(h_mixes),    mtl::Arg::B(m->dense, h_scale->offset),
                  mtl::Arg::B(m->dense, h_base->offset), mtl::Arg::B(pre_a), mtl::Arg::B(post_a),
                  mtl::Arg::B(h_cur),  mtl::Arg::V(a)};
        d.grid[0] = uT;
        d.tg[0] = 256;
        mtl::require_fit(d, "hy4_flash_hc_mix (head)");
        ds.push_back(d);
    }
    rms(ds, h_cur, 0, (uint32_t) n_embd, h_norm, h_rnorm, 0, (uint32_t) n_embd, (uint32_t) n_embd, T, eps);
    if (!with_logits) return;   // the MTP block's `h` only (PrefillSink::hidden)
    gv(h_out).append(ds, m->dense, h_out->offset, h_rnorm, (uint32_t) n_embd, logits_b, (uint32_t) vocab, 0, uT);
    {
        mtl::Dispatch d;
        d.p = &p_argmax;
        ArgmaxArgs a{(uint32_t) vocab, (uint32_t) vocab};
        d.args = {mtl::Arg::B(logits_b), mtl::Arg::B(argmax_b), mtl::Arg::V(a)};
        d.grid[0] = uT;
        d.tg[0] = 1024;   // hy4_flash_attn.metal: the winner does not depend on the thread count
        mtl::require_fit(d, "hy4_flash_argmax");
        ds.push_back(d);
    }
}

/// One head sub-batch as its own command buffer, and the last row kept (`last_token`).  T <= hsub.
std::string Forward::Impl::run_head(const mtl::Buffer& x, int x_t0, int T) {
    if (T < 1 || T > hsub)
        return "the head was asked for " + std::to_string(T) + " positions, its buffers hold " +
               std::to_string(hsub);
    std::vector<mtl::Dispatch> ds;
    encode_head(ds, x, x_t0, T);
    tm.gpu_ms += cx->run_graph(ds) * 1e3;
    std::memcpy(last_log.data(), logits_b.as<float>() + (size_t) (T - 1) * (size_t) vocab, 4 * (size_t) vocab);
    last_tok = argmax_b.as<int32_t>()[T - 1];
    return {};
}

/// The HC head and output_norm for T <= hsub positions, without the vocabulary GEMV: h_rnorm holds the rows after.
std::string Forward::Impl::run_hidden(const mtl::Buffer& x, int x_t0, int T) {
    if (T < 1 || T > hsub)
        return "the head was asked for " + std::to_string(T) + " positions, its buffers hold " +
               std::to_string(hsub);
    std::vector<mtl::Dispatch> ds;
    encode_head(ds, x, x_t0, T, false);
    tm.gpu_ms += cx->run_graph(ds) * 1e3;
    return {};
}

/// Every position >= p0 is about to be (re)written, so a pooled key whose group reaches p0 or past it is stale:
/// a rejected speculative window may have completed that group with a token that is not the sequence's.
/// For a run that only moves forward this is a no-op (pools_done is exactly p0 / kpool after the previous run).
void Forward::Impl::clamp_pools(int p0_) {
    const int keep = p0_ / idx_kpool;
    for (int32_t& d : pools_done) d = std::min(d, keep);
}

// ---- the host's turn -----------------------------------------------------------------------------------------------
void Forward::Impl::at_cut(int s) {
    const double t0 = mtl::now_s();
    if (cut_role[(size_t) s] == 2) {   // the split layer's second cut: wait for its late experts, then read ahead
        m->stream.finish_layer();
        tm.host_cut_ms += (mtl::now_s() - t0) * 1e3;
        return;
    }
    const bool split = cut_role[(size_t) s] == 1;
    const int l = cut_block[(size_t) s];
    const BW& b = blk[(size_t) l];
    const int o = b.ordinal;
    int32_t* id = ids.as<int32_t>();
    float* w = rw.as<float>();
    // THE ROUTER'S IDS CROSS FROM THE GPU TO THE HOST HERE, so they are checked here, BEFORE anything indexes
    // with them.  A nan anywhere upstream reaches the router as a nan logit, and a top-k scan whose comparisons
    // are all false yields values that are not expert ids at all; using one indexes `smap`, `mlogits` or the
    // router's bias row out of bounds, which SEGFAULTS the engine instead of ending the request.  The offenders
    // are clamped to 0 - the window's numbers are discarded by the caller anyway - and the request ends with the
    // `nonfinite` sentinel.
    //
    // It runs TWICE on purpose: once on what the GPU produced, and again after `--force-topk` has overwritten the
    // ids from a file, because that file is an untrusted input too.  Both calls must stay AHEAD of every use of
    // the ids: the forced path and the router_biased mutation below each index `mlogits` and the bias row with
    // them.
    auto guard_ids = [&](const char* whose) {
        int bad_id = -1, bad_at = -1;
        for (int i = 0; i < n_tok * topk; ++i)
            if (id[i] < 0 || id[i] >= n_experts) {
                if (bad_at < 0) { bad_id = id[i]; bad_at = i; }
                id[i] = 0;
            }
        if (bad_at >= 0 && cut_err.empty())
            cut_err = std::string(Forward::kNonfinite) + ": ordinal " + std::to_string(o) + " token " +
                      std::to_string(bad_at / topk) + ": " + whose + " expert " + std::to_string(bad_id) +
                      ", outside 0.." + std::to_string(n_experts - 1) + " - the request ends here (ADR §4.2)";
        return bad_at < 0;
    };
    if (!guard_ids("the router chose")) {
        tm.host_cut_ms += (mtl::now_s() - t0) * 1e3;
        return;
    }
    const int32_t* ftk = inj != nullptr ? inj->topk : nullptr;
    if (ftk == nullptr && pr_in_chunk && pr_force != nullptr) {
        if (p0 + n_tok <= pr_n) ftk = pr_force + ((size_t) o * pr_n + p0) * topk;
        else if (cut_err.empty())   // never silently unforced: the judge would compare a run it did not ask for
            cut_err = "set_prompt_routing was given " + std::to_string(pr_n) + " positions but the prompt reached " +
                      std::to_string(p0 + n_tok);
    }
    if (ftk != nullptr) {
        // the reference's routing, with OUR probabilities behind the weights (--force-topk removes the
        // near-tie flips without replacing the arithmetic)
        const float* lg = mlogits.as<float>();
        for (int t = 0; t < n_tok; ++t)
            for (int k = 0; k < topk; ++k) id[(size_t) t * topk + k] = ftk[(size_t) t * topk + k];
        if (!guard_ids("the injected --force-topk file names")) {
            tm.host_cut_ms += (mtl::now_s() - t0) * 1e3;
            return;
        }
        for (int t = 0; t < n_tok; ++t) {
            double sum = 0;
            for (int k = 0; k < topk; ++k) {
                const int32_t e = id[(size_t) t * topk + k];
                w[(size_t) t * topk + k] = 1.0f / (1.0f + std::exp(-lg[(size_t) t * n_experts + e]));
            }
            for (int k = 0; k < topk; ++k) sum += w[(size_t) t * topk + k];
            const float den = std::max((float) sum, 6.103515625e-5f);
            for (int k = 0; k < topk; ++k) w[(size_t) t * topk + k] = w[(size_t) t * topk + k] / den * m->geo.routed_scale;
        }
    }
    // for the read-ahead accuracy measurement: what this ordinal really wanted, and what the previous one predicted
    if (!routed.empty()) {
        const size_t at = (size_t) o * n_tok * topk;
        if (at + (size_t) n_tok * topk <= routed.size())
            std::memcpy(&routed[at], id, (size_t) n_tok * topk * 4);
    }
    if (mut.router_biased) {
        // the mutation: the weight comes from the BIASED probability instead of the unbiased one, which costs the
        // weights most of their accuracy, so gate 1 must catch it.
        const float* lg = mlogits.as<float>();
        const float* bias = (const float*) ((const uint8_t*) m->dense.ptr + b.t[W_PROBS_B]->offset);
        for (int t = 0; t < n_tok; ++t) {
            float sum = 0;
            for (int k = 0; k < topk; ++k) {
                const int32_t e = id[(size_t) t * topk + k];
                w[(size_t) t * topk + k] = 1.0f / (1.0f + std::exp(-lg[(size_t) t * n_experts + e])) + bias[e];
                sum += w[(size_t) t * topk + k];
            }
            const float den = std::max(sum, 6.103515625e-5f);
            for (int k = 0; k < topk; ++k)
                w[(size_t) t * topk + k] = w[(size_t) t * topk + k] / den * m->geo.routed_scale;
        }
    }
    const float* pr = prefetch_enabled() ? pred.as<float>() : nullptr;
    if (pr != nullptr && !predicted.empty() && o + 1 < m->stream.n_layers()) {
        // the top-k of the prediction, the same rule the stream's read-ahead uses
        const size_t at = (size_t) (o + 1) * n_tok * topk;
        if (at + (size_t) n_tok * topk <= predicted.size())
            for (int t = 0; t < n_tok; ++t) {
                std::vector<int32_t> idx((size_t) n_experts);
                for (int e = 0; e < n_experts; ++e) idx[(size_t) e] = e;
                const float* row = pr + (size_t) t * n_experts;
                std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                                  [&](int32_t x, int32_t y) { return row[x] > row[y] || (row[x] == row[y] && x < y); });
                for (int k = 0; k < topk; ++k) predicted[at + (size_t) t * topk + k] = idx[(size_t) k];
            }
    }
    if (!all_logits.empty()) {
        const size_t at = (size_t) o * n_tok * n_experts;
        if (at + (size_t) n_tok * n_experts <= all_logits.size())
            std::memcpy(&all_logits[at], mlogits.ptr, (size_t) n_tok * n_experts * 4);
    }
    const float* pr2 = pr != nullptr && m->stream.ahead2() > 0 ? pred2.as<float>() : nullptr;
    if (split) m->stream.begin_layer(o, n_tok, topk, id, prefetch_enabled() ? pr : nullptr,
                                     prefetch_enabled() ? pr2 : nullptr);
    else strata::hy4_flash::at_cut(*m, o, n_tok, topk, id, pr, pr2);
    bool absent = false;
    MoeSlots& ps = plan_of(n_tok);
    std::function<bool(int32_t)> late;
    // THE PARTITION: an expert whose bytes are still on their way goes to the late half.  It must reach fill() -
    // without it every group lands in the resident half, which then reads slots the SSD is still writing.
    if (split) late = [&](int32_t ex) { return m->stream.pending(o, ex); };
    const std::string e = ps.fill(id, w, (uint32_t) n_tok, (uint32_t) topk, [&](int32_t ex) {
        // split: a late expert has its slot already, its bytes arrive before the late half runs
        const uint32_t code = split ? m->stream.slot_code_of(o, ex) : m->stream.smap_of(o, ex);
        if (code == 0xFFFFFFFFu) {
            absent = true;   // at_layer should have put every demanded expert in place; reading slot 0 would be
            return MoeSlots::Place{0, 0};   // another expert's bytes, so this ends the run instead
        }
        return MoeSlots::Place{(int32_t) (code & 0x0FFFFFFFu), (int) (code >> 28)};
    }, late);
    if (absent && cut_err.empty())
        cut_err = "ordinal " + std::to_string(o) + ": an expert the window asked for is not in the arena";
    if (!e.empty() && cut_err.empty()) cut_err = "ordinal " + std::to_string(o) + ": " + e;
    if (split) {
        ps.split_plans(plan_a_host, plan_b_host);
        std::memcpy(plan_buf(n_tok).ptr, plan_a_host.data(), plan_a_host.size() * 4);
        std::memcpy(plan_b.ptr, plan_b_host.data(), plan_b_host.size() * 4);
    } else {
        std::memcpy(plan_buf(n_tok).ptr, ps.buf.data(), ps.buf.size() * 4);
    }
    // the prefill GEMM's tile lists and grids, for the layer this cut is about to release
    if (mm_active(o, n_tok, std::min(ps.n_chunk, (uint32_t) std::max(1, m->stream.n_chunks())))) {
        const int tm = moe_mm_tile(moe_mm_mode()).tm;
        for (uint32_t s = 0; s < ps.subs_for((uint32_t) n_tok); ++s)
            for (uint32_t c = 0; c < ps.n_chunk; ++c) {
                const size_t slot = (size_t) s * ps.n_chunk + c;
                if (slot >= mm_slots) break;   // laid out for tcap; a shorter window uses fewer slots
                const int32_t* gs = ps.buf.data() + (size_t) s * ps.plan_stride + ps.off_gstart + c * (ps.g_max + 1);
                const uint32_t n = MoeMmF32::tiles_into(gs, ps.g_max, tm,
                                                        mm_tiles.as<int32_t>() + 2 * mm_cap * slot, mm_cap);
                if (n > mm_cap) {
                    // tile_cap is provably enough, so this is unreachable - but the recovery must not leave the
                    // REST of the grids at the previous layer's values, because cut_err is only read after the
                    // command buffer has run.  Empty every slot, then stop.
                    std::memset(mm_ind.ptr, 0, mm_ind.bytes);
                    if (cut_err.empty())
                        cut_err = "ordinal " + std::to_string(o) + ": the MoE tile list of sub-batch " +
                                  std::to_string(s) + " chunk " + std::to_string(c) + " needs more than " +
                                  std::to_string(mm_cap) + " tiles";
                    return;
                }
                uint32_t* g = mm_ind.as<uint32_t>() + 3 * slot * 2;
                for (int kind = 0; kind < 2; ++kind) {
                    g[3 * kind] = mm_gx.at((size_t) o)[(size_t) kind];
                    g[3 * kind + 1] = n;
                    g[3 * kind + 2] = 1;
                }
            }
    }
    if (moe_indirect() && n_tok <= Forward::kMaxWindow && ps.subs_for((uint32_t) n_tok) == 1) {
        // the grids of this layer's MoE dispatches: (width, groups of that half in that chunk, 1).  The groups of a
        // chunk are packed from slot 0 (MoeSlots::fill), resident ones first when split (n_res).
        uint32_t* g = moe_ind.as<uint32_t>();
        for (uint32_t c = 0; c < ps.n_chunk && c < 4; ++c) {
            uint32_t total = 0;
            const int32_t* gs = ps.buf.data() + ps.off_gstart + c * (ps.g_max + 1);
            while (total < ps.g_max && gs[total + 1] > gs[total]) ++total;
            const uint32_t res = split ? ps.n_res[c] : total;
            for (int h = 0; h < 2; ++h)
                for (int kind = 0; kind < 2; ++kind) {
                    uint32_t* x = g + 3 * ((((size_t) h * 4) + c) * 2 + (size_t) kind);
                    x[0] = moe_gx.at((size_t) o)[(size_t) kind];
                    x[1] = h == 0 ? res : total - res;
                    x[2] = 1;
                }
        }
    }
    tm.host_cut_ms += (mtl::now_s() - t0) * 1e3;
}

// ---- windows -------------------------------------------------------------------------------------------------------
std::string Forward::Impl::prepare_window(const int32_t* all_tokens, int p0_, int T, bool need_engram) {
    p0 = p0_;
    n_tok = T;
    cut_err.clear();
    // The indexer's scratch is only written over the extent each query really has: the score kernel's grid is
    // `n_pool` wide and the selection is `nsel` long, so the rest of each row keeps whatever the last window left
    // there.  That is harmless for the arithmetic - nothing reads past the extent - but it makes the DUMP
    // non-deterministic, and a T-invariance check comparing dumped bytes then reports a difference that is
    // nothing but stale scratch.  Zeroing here makes the dumped bytes a function of the inputs alone.  Only the
    // FIRST T ROWS are cleared, not the whole buffer: these three are allocated for the largest window
    // (Tm = tcap, which `--prefill` raises to the chunk size) while decode runs T = 1, so clearing all of it
    // would spend most of a decode window wiping rows nobody touches.  Row t sits at a FIXED stride: the score
    // kernel's `s_stride` is `ctx / kpool + 1` = n_pool_max (alloc_scratch) and sel/nsel use kSelCap and 1,
    // so the first T rows are exactly the first T * stride entries.
    const size_t rows = (size_t) T;
    if (score.ptr != nullptr)
        std::memset(score.ptr, 0, std::min(score.bytes, 4 * rows * ((size_t) ctx / (size_t) idx_kpool + 1)));
    if (sel.ptr != nullptr) std::memset(sel.ptr, 0, std::min(sel.bytes, 4 * rows * (size_t) Forward::kSelCap));
    if (nsel.ptr != nullptr) std::memset(nsel.ptr, 0, std::min(nsel.bytes, 4 * rows));
    int32_t* pb = pos_b.as<int32_t>();
    for (int t = 0; t < T; ++t) pb[t] = p0 + t;
    if (images()) {
        // The three RoPE planes, beside the sequential positions rather than instead of them: the indexer,
        // the KV cache and the causal bound all keep reading pos_b.  Inside the prompt the planes come from
        // the table the request supplied; past it every axis is `index + delta`, which is upstream's
        // "constant position offset after the prompt" and is why decode needs no table at all.
        int32_t* rp = rope_pos.as<int32_t>();
        const size_t plane = rope_plane();
        for (int t = 0; t < T; ++t) {
            const int64_t i = (int64_t) p0 + t;
            for (int ax = 0; ax < 3; ++ax)
                rp[(size_t) ax * plane + (size_t) t] =
                    i < img_axes_len ? img_axes[(size_t) ax * (size_t) img_axes_len + (size_t) i]
                                     : (int32_t) i + img_delta;
        }
    }
    if (!need_engram) return {};
    if (!eng_ready) return "a block with an Engram table was asked for, but --pack was not given (ADR §3)";
    const int np = m->geo.engram_predecessors();
    std::vector<int32_t> prev((size_t) T * (size_t) std::max(np, 1), 0);
    for (int t = 0; t < T; ++t)
        for (int s = 1; s <= np; ++s) {
            const int p = p0 + t - s;
            prev[(size_t) t * np + (np - s)] = p >= 0 ? all_tokens[p] : 0;
        }
    if (const std::string e = eng.issue(all_tokens + p0, pb, prev.data(), T); !e.empty()) return e;
    return {};
}

void Forward::Impl::dump_block(const BW& b, int T, Dump* dump, const Inject& injected,
                              const mtl::Buffer& in_buf, const mtl::Buffer& out_buf) {
    if (dump == nullptr || !dump->on()) return;
    const std::string sfx = "-" + std::to_string(b.id);
    const int H = b.heads;
    auto per = [&](const char* name, const mtl::Buffer& buf, std::vector<int64_t> ne_one, int pos_axis,
                   size_t floats_per_pos) {
        dump->add(std::string(name) + sfx, "f32", std::move(ne_one), pos_axis, buf.ptr, 4 * floats_per_pos, T);
    };
    // the INPUT node under the reference's name for it, so gate 1 can check that the injection took (it demands
    // that this node be bitwise the reference's)
    const void* in_ptr = injected.hc != nullptr ? (const void*) injected.hc : (const void*) in_buf.ptr;
    if (b.id == 0)
        dump->add("hc_init", "f32", {n_embd, hc}, 1, in_ptr, 4 * (size_t) hcw, T);
    else
        dump->add("l_out-" + std::to_string(b.id - 1), "f32", {n_embd, hc}, 1, in_ptr, 4 * (size_t) hcw, T);
    if (b.engram_table >= 0) {
        dump->add("engram_rows_normed" + sfx, "f32", {m->geo.engram_lookup_dim()}, 2, eng_r.ptr,
                  4 * (size_t) m->geo.engram_lookup_dim(), T);
        per("engram_kv", eng_kv, {n_embd, hc + 1}, 1, (size_t) (hc + 1) * n_embd);
        per("engram_dot", eng_dot, {1, hc}, 1, (size_t) hc);
        per("engram_gate", eng_gate, {1, hc}, 1, (size_t) hc);
        per("engram_delta", eng_delta, {n_embd, hc}, 1, (size_t) hcw);
        per("engram_out", hc_eng, {n_embd, hc}, 1, (size_t) hcw);
    }
    // the reference dumps hc_mixes / hc_pre / hc_post_gate TWICE a block (once per half) and check_block.py's
    // Dump.get() returns the first, which is the attention half's - so that is the one written here.
    per("hc_mixes", mixes_a, {2 * hc}, 2, (size_t) 2 * hc);
    per("hc_pre", pre_a, {hc}, 2, (size_t) hc);
    per("hc_post_gate", post_a, {hc}, 2, (size_t) hc);
    per("attn_norm", anorm, {n_embd}, 2, (size_t) n_embd);
    per("q_resid", q_resid, {m->geo.q_lora_rank}, 2, (size_t) m->geo.q_lora_rank);
    per("kv_cmpr", kvn, {kv_lora}, 2, (size_t) kv_lora);
    per("k_pe", kpe, {rope_dim, 1}, 1, (size_t) rope_dim);
    per("q_pe", qpe, {rope_dim, H}, 1, (size_t) H * rope_dim);
    per("attn_kqv", kqv, {H * v_mla}, 2, (size_t) H * v_mla);
    per("attn_gated", gated, {H * v_mla}, 2, (size_t) H * v_mla);
    per("attn_out", attn_out, {n_embd}, 2, (size_t) n_embd);
    per("hc_attn_out", hc_mid, {n_embd, hc}, 1, (size_t) hcw);
    per("ffn_norm", fnorm, {n_embd}, 2, (size_t) n_embd);
    if (b.has_indexer) {
        per("indexer_k", ik, {idx_dim}, 2, (size_t) idx_dim);
        per("indexer_q", iq, {idx_dim, idx_heads}, 1, (size_t) idx_heads * idx_dim);
        per("indexer_weights", iw, {idx_heads}, 2, (size_t) idx_heads);
        // The score row is this engine's own width (max_context / kpool + 1) rather than the reference's padding.
        //
        // DUMPED ONLY OVER THE EXTENT THE QUERY ACTUALLY HAS, which is `nv = pos / kpool` pools: the score
        // buffer is ONE row per query but the dispatch is `n_pool = (p0 + T) / kpool` columns wide, shared by the
        // whole window, so columns in (nv, n_pool) hold -INF for this query (the mask) and exist at all only
        // because some LATER query in the same window needed them.  `hy4_flash_indexer_topk` reads g < nv and nothing
        // else reads the buffer, so those columns change nothing - but they make the dumped bytes a function of
        // the window size, which turns a T-invariance check on the dump into a false alarm.  Zeroing past nv
        // makes the artifact a function of the inputs alone; it does not hide a difference, because the extent
        // itself is compared.
        {
            const size_t w = (size_t) ctx / (size_t) idx_kpool + 1;
            score_dump.assign(w * (size_t) T, 0.0f);
            const float* src = score.as<float>();
            for (int t = 0; t < T; ++t) {
                const size_t nv = (size_t) ((p0 + t) / idx_kpool);
                const size_t take = nv < w ? nv : w;
                std::memcpy(&score_dump[(size_t) t * w], src + (size_t) t * w, 4 * take);
            }
            dump->add("indexer_score" + sfx, "f32", {(int64_t) w}, 2, score_dump.data(), 4 * w, T);
        }
        // THE SELECTED SET ITSELF.  The score is a continuous quantity and the selection is the discrete
        // decision taken from it, so a tie broken the other way shows up here and nowhere else - which is
        // exactly what a T-invariance check has to see ("same score, lower pool index wins").  The engine's own
        // buffers, so this is empty on a --force-sel run by construction: there the host wrote them.
        // i32, not f32: `per` is the float helper, and declaring int bytes as f32 would mislead any reader of
        // the manifest (the bytes would compare fine, the metadata would be a lie).
        dump->add("indexer_nsel" + sfx, "i32", {1}, 2, nsel.ptr, 4, T);
        dump->add("indexer_pool_sel" + sfx, "i32", {Forward::kSelCap}, 2, sel.ptr,
                  4 * (size_t) Forward::kSelCap, T);
    }
    if (!b.dense_ffn) {
        per("ffn_moe_logits", mlogits, {n_experts}, 2, (size_t) n_experts);
        dump->add("ffn_moe_topk" + sfx, "i32", {topk}, 2, ids.ptr, 4 * (size_t) topk, T);
        dump->add("ffn_moe_weights_scaled" + sfx, "f32", {1, topk}, 1, rw.ptr, 4 * (size_t) topk, T);
        per("ffn_moe_out", mlat, {latent}, 2, (size_t) latent);
        per("ffn_shexp", sho, {n_embd}, 2, (size_t) n_embd);
    } else {
        per("ffn_gate", dg, {m->geo.dense_ff}, 2, (size_t) m->geo.dense_ff);
        per("ffn_up", du, {m->geo.dense_ff}, 2, (size_t) m->geo.dense_ff);
        per("ffn_swiglu", da, {m->geo.dense_ff}, 2, (size_t) m->geo.dense_ff);
    }
    per("ffn_out", fout, {n_embd}, 2, (size_t) n_embd);
    per("l_out", out_buf, {n_embd, hc}, 1, (size_t) hcw);
}

// ---- the public face -----------------------------------------------------------------------------------------------
Forward::~Forward() { delete im_; }

std::string Forward::open(Model& m, mtl::Context& cx, const std::string& pack_dir, const Prefill& pf) {
    if (im_ != nullptr) return "already open";
    im_ = new Impl();
    const std::string e = im_->open(m, cx, pack_dir, pf);
    if (!e.empty()) {
        delete im_;
        im_ = nullptr;
    }
    return e;
}

// Called after the LAST flush of a window, i.e. after the command buffer that wrote the cache has completed -
// before that the flag is not readable.  Returns the `nonfinite` sentinel when this window added any: the request
// has to end HERE, not with a warning at the end of the run, by which time the value has been read by every
// later position.
std::string Forward::Impl::nonfinite_check() {
    if (badflag.ptr == nullptr) return {};
    const uint32_t now = *badflag.as<uint32_t>();
    if (now <= bad_reported) return {};
    const uint32_t added = now - bad_reported;
    bad_reported = now;
    return std::string(Forward::kNonfinite) + ": " + std::to_string(added) +
           " value(s) of this window could not enter an f16 cache (not finite, or past f16's 65504); they were "
           "stored as zero and the request ends here (ADR 0001 §4.2)";
}

/// The sub-batch sizes, overridable for a measurement (the defaults are in forward.h).
Prefill Prefill::from_env(int chunk) {
    Prefill p;
    p.chunk = chunk;
    auto num = [](const char* name, int& v) {
        if (const char* e = std::getenv(name))
            if (const int n = std::atoi(e); n > 0) v = n;
    };
    num("HY4_FLASH_PREFILL_ATTN", p.attn_sub);
    num("HY4_FLASH_PREFILL_MOE", p.moe_sub);
    num("HY4_FLASH_PREFILL_HEAD", p.head_sub);
    num("HY4_FLASH_PREFILL_IDX", p.idx_sub);
    // not num(): 0 is a value here (the decode kernels for the prompt), and num() only takes positives
    // *e != 0: an EMPTY value means unset (the default), as it does for the snapshot's mode list (main.cpp) -
    // atoi("") would have read 0 and silently selected the decode kernels for a snapshot recorded as default
    if (const char* e = std::getenv("HY4_FLASH_PREFILL_MMA"); e != nullptr && *e != 0) p.mma = std::atoi(e);
    num("HY4_FLASH_PREFILL_PROBE", p.probe);
    if (const char* e = std::getenv("HY4_FLASH_PREFILL_NOISE"))
        if (const double v = std::atof(e); v > 0) p.noise = v;
    if (p.noise > 0)
        std::fprintf(stdout, "strata_hy4_flash_metal: HY4_FLASH_PREFILL_NOISE=%g - the PREFILL path's block outputs carry a deliberate "
                             "relative perturbation of that size (a benign stand-in for a different fp32 summation "
                             "order; forward.h)\n", p.noise);
    if (p.probe != 0)
        std::fprintf(stdout, "strata_hy4_flash_metal: HY4_FLASH_PREFILL_PROBE=%d - the PREFILL path carries a deliberate %s and its "
                             "logits are WRONG ON PURPOSE; --prefill-consistency must FAIL (forward.h)\n", p.probe,
                     p.probe == 3 ? "sliding window one key too wide" : "sub-batch offset error");
    return p;
}

// ---- the prefill snapshot (prefill_snap.h) -------------------------------------------------------------------
//
// The regions are the whole `kv` and `idxc` ALLOCATIONS, not the used prefix of each block's slice.  The used
// prefix would be smaller, but it means knowing each buffer's internal layout, and `idxc`'s is set by the
// indexer kernels rather than here: a wrong stride would save the right number of bytes from the wrong places
// and only the bit-equality test would notice.  The whole allocation is 0.5 GB at 8K, which is cheap beside the
// prefill it saves.  `context` is part of the binding, which is what makes this safe - the stride is
// identical by construction.
void Forward::Impl::snap_regions(std::vector<SnapSlice>& v) {
    v.push_back({"kv", kv.ptr, kv.bytes});
    v.push_back({"idxc", idxc.ptr, idxc.bytes});
    // Per sparse block: how many pooled keys its cache already holds.  `clamp_pools` resets these to p0/kpool
    // at the start of every window, so a load that forgot them would be silently repaired for a forward-only
    // run and silently wrong for anything else - saved rather than recomputed.
    v.push_back({"pools_done", pools_done.data(), 4 * (uint64_t) pools_done.size()});
    v.push_back({"last_logits", last_log.data(), 4 * (uint64_t) last_log.size()});
    // THE NON-FINITE FLAG IS STATE, not a derived number.  `cache_nonfinite()` reads this buffer, and
    // `prefill()` does not call `nonfinite_check()`, so a prompt can leave the flag above `bad_reported`:
    // zeroing it on load would delete the only record that cached values were replaced by zero, and the
    // continuation that would otherwise have reported them on its next window would stay silent.  Saved and
    // restored verbatim, with `bad_reported` beside it so the pair stays consistent.
    v.push_back({"badflag", badflag.ptr, badflag.ptr == nullptr ? 0ull : 16ull});
}

void Forward::Impl::snap_fill(SnapBinding& b) const {
    b.context = ctx;
    b.blocks = (int32_t) blk.size();
    b.kv_w = kv_w;
    b.idx_blocks = (int32_t) pools_done.size();
    b.idx_dim = idx_dim;
    b.vocab = vocab;
    b.last_tok = last_tok;
    b.bad_reported = bad_reported;
}

std::string Forward::snapshot_save(const std::string& path, SnapBinding b,
                                   const std::vector<SnapSlice>& extra) const {
    if (im_ == nullptr) return "the engine is not open";
    im_->snap_fill(b);
    b.used = im_->snap_used;
    if (b.used <= 0) return "nothing to save: no prompt has run in this process";
    std::vector<SnapSlice> v;
    im_->snap_regions(v);
    for (const SnapSlice& s : extra) v.push_back(s);
    return snap_save(path, b, v);
}

std::string Forward::snapshot_load(const std::string& path, SnapBinding want,
                                   const std::vector<SnapSlice>& extra) {
    if (im_ == nullptr) return "the engine is not open";
    im_->snap_fill(want);
    // `used` and `last_tok` come FROM the file - they are what is being restored, so they are not compared.
    want.used = 0;
    want.last_tok = -1;
    want.bad_reported = 0;
    std::vector<SnapSlice> v;
    im_->snap_regions(v);
    for (const SnapSlice& s : extra) v.push_back(s);
    SnapBinding got;
    if (const std::string e = snap_load(path, want, got, v); !e.empty()) {
        // A failed load may have installed part of the file (snap_load says so when it did), so the caches are
        // not the state they were.  POISON rather than leave them looking usable: `used` 0 and `last_tok` -1,
        // which the engine's own token check turns into a clean refusal if a caller ignores this error.
        im_->snap_used = 0;
        im_->last_tok = -1;
        return e;
    }
    im_->last_tok = got.last_tok;
    im_->bad_reported = got.bad_reported;
    im_->snap_used = got.used;
    return {};
}

int32_t Forward::snapshot_used() const { return im_ == nullptr ? -1 : im_->snap_used; }

void Forward::reset() {
    if (im_ == nullptr) return;
    im_->snap_used = 0;
    std::fill(im_->pools_done.begin(), im_->pools_done.end(), 0);
    if (im_->badflag.ptr != nullptr) std::memset(im_->badflag.ptr, 0, 16);
    im_->bad_reported = 0;
}

const float* Forward::logits() const { return im_ == nullptr ? nullptr : im_->logits_b.as<float>(); }
const float* Forward::result_norm() const { return im_ == nullptr ? nullptr : im_->h_rnorm.as<float>(); }
int Forward::library() const { return im_ == nullptr ? -1 : im_->lib; }
const int32_t* Forward::argmax() const { return im_ == nullptr ? nullptr : im_->argmax_b.as<int32_t>(); }
int32_t Forward::last_token() const { return im_ == nullptr ? -1 : im_->last_tok; }
const float* Forward::last_logits() const { return im_ == nullptr ? nullptr : im_->last_log.data(); }
int Forward::prefill_chunk() const { return im_ == nullptr ? 0 : im_->pf.chunk; }
int32_t Forward::vocab() const { return im_ == nullptr ? 0 : im_->vocab; }
const Timing& Forward::timing() const {
    // Engram keeps its own split; copy it in on read so Timing stays one struct for main.cpp.  `hash_ms` is the
    // self-check of this measurement: it is pure CPU over the window's tokens, so it must NOT grow with context.
    const EngramSupply::Timing& e = im_->eng.timing();
    im_->tm.eng_tail_ms = e.tail_us / 1e3;
    im_->tm.eng_hash_ms = e.hash_us / 1e3;
    im_->tm.eng_join_ms = e.join_us / 1e3;
    return im_->tm;
}

uint32_t Forward::cache_nonfinite() const {
    return im_ == nullptr || im_->badflag.ptr == nullptr ? 0 : *im_->badflag.as<uint32_t>();
}
const std::vector<int32_t>& Forward::routed_ids() const { return im_->routed; }
void Forward::set_prompt_routing(std::vector<int32_t>* record, const int32_t* force, int n) {
    im_->pr_n = n;
    im_->pr_record = record;
    im_->pr_force = force;
    if (record != nullptr) record->assign((size_t) im_->m->stream.n_layers() * (size_t) n * (size_t) im_->topk, -1);
}

void Forward::set_image_prompt(const std::vector<ImageSpanRows>* spans, const int32_t* axes,
                               int64_t axes_len, int32_t delta, int32_t mrope_len1, int32_t mrope_len2) {
    im_->img = spans;
    im_->img_axes = axes;
    im_->img_axes_len = axes_len;
    im_->img_delta = delta;
    im_->img_mlen1 = mrope_len1;
    im_->img_mlen2 = mrope_len2;
}

bool Forward::has_image_prompt() const { return im_->images(); }
const std::vector<int32_t>& Forward::predicted_ids() const { return im_->predicted; }

namespace {
void cut_tramp(void* u, int s) { ((Forward::Impl*) u)->at_cut(s); }
}  // namespace

/// The window's Engram rows and validity into the shared buffers the kernels read.  One window is one command
/// buffer per Engram block (the four tables would otherwise overwrite each other's rows), which `window` handles.
std::string Forward::Impl::engram_rows_for(int block_id, int T) {
    std::string err;
    const uint8_t* rows = eng.join(block_id, err);
    if (rows == nullptr) return "Engram block " + std::to_string(block_id) + ": " + err;
    const uint64_t cols = eng.rows_per_token();
    std::memcpy(eng_rows.ptr, rows, (size_t) T * cols * eng.row_bytes());
    const float* v = eng.valid(block_id);
    std::memcpy(eng_valid.ptr, v, (size_t) T * cols * 4);
    float* any = eng_any.as<float>();
    for (int t = 0; t < T; ++t) {
        float s = 0;
        for (uint64_t c = 0; c < cols; ++c) s += v[(size_t) t * cols + c];
        // the reference masks the whole delta when no column is valid; the engram_delta2 mutation doubles it
        any[t] = s > 0 ? (mut.engram_delta2 ? 2.0f : 1.0f) : 0.0f;
    }
    return {};
}

/// hc_init: ONE embedding broadcast into all hc streams (hy-v4-flash.cpp:1241-1244, head.cu's note - not a
/// per-stream projection).  The rows are dequantized here rather than on the GPU: a window is at most 8 rows of
/// 3072 Q8_0 values, and `ref::dequant` is ggml's own arithmetic (d * q), so the result is the reference's.
std::string Forward::Impl::embed_window(const int32_t* all_tokens, int T) {
    const gguf::Tensor* te = m->find("token_embd.weight");
    if (te == nullptr) return "no tensor token_embd.weight";
    const size_t rb = ref::row_bytes(te->type, n_embd);
    std::vector<float> row((size_t) n_embd);
    std::vector<uint8_t> raw(rb);
    float* dst = hcA.as<float>();
    for (int t = 0; t < T; ++t) {
        const int32_t tok = all_tokens[p0 + t];
        if (tok < 0 || tok >= vocab) return "token id " + std::to_string(tok) + " is outside the vocabulary";
        // the row's bytes from the dense buffer or, when token_embd is not resident (model.h on_file), the file
        if (!m->tensor_bytes("token_embd.weight", (uint64_t) tok * rb, rb, raw.data()))
            return "cannot read row " + std::to_string(tok) + " of token_embd.weight";
        ref::dequant(te->type, raw.data(), row.data(), (size_t) n_embd);
        // A picture's positions take the tower's row instead of the placeholder token's embedding, into all
        // four hyper-connection planes exactly as the token embedding goes.  Done here rather than with a
        // kernel because hcA is shared host-visible memory: upstream needs a scatter kernel only because
        // its state lives in device memory the host cannot write.
        const float* src = row.data();
        if (images()) {
            const int64_t at = (int64_t) p0 + t;
            for (const Forward::ImageSpanRows& s : *img)
                if (at >= s.start && at < s.start + s.count) {
                    src = s.rows + (size_t) (at - s.start) * (size_t) n_embd;
                    break;
                }
        }
        for (int i = 0; i < hc; ++i)
            std::memcpy(dst + (size_t) t * hcw + (size_t) i * n_embd, src, 4 * (size_t) n_embd);
    }
    return {};
}

/// THE BARRIERS A DISPATCH LIST REALLY NEEDS.  Every append puts a buffer-scope memory barrier before its dispatch
/// (Dispatch::barrier defaults to true) whether or not it depends on what ran since the last one, so two GEMVs that
/// read the same input into different outputs - the attention's q / kv projections, the shared expert's gate and up -
/// still run one after the other.  This drops a barrier when the dispatch shares NO written buffer with any dispatch
/// since the last barrier that stays.  "Written" is conservative: every buffer is assumed read AND written except the
/// ones in ro_keys, which no decode dispatch writes; two arguments are the same buffer when they are the same
/// MTLBuffer (a view or an offset into it counts as the same - no byte ranges are trusted).  So a barrier is only
/// removed between dispatches whose memory cannot interact, and no value can change.  The set restarts at every
/// encoder boundary (a cut or a wait), where the encoders are ordered anyway.
void Forward::Impl::prune_barriers(std::vector<mtl::Dispatch>& ds, const std::vector<int>& cuts,
                                   const std::vector<int>& waits) {
    if (!prune_on()) return;
    auto key = [](const mtl::Buffer* b) -> const void* { return b->chunks ? (const void*) b->chunks.get() : b->handle; };
    if (ro_keys.empty()) {   // here, not at alloc(): the arena is built after the forward's buffers
        // mm_tiles / mm_ind are written by the HOST at the cut and only read by the GPU, like plan and moe_ind
        for (const mtl::Buffer* b : {&m->dense, &plan, &plan_b, &moe_ind, &mm_tiles, &mm_ind})
            if (b->ptr != nullptr) ro_keys.push_back(key(b));   // mm_* exist only with the GEMM on
        for (int c = 0; c < m->stream.n_chunks(); ++c) ro_keys.push_back(key(&m->stream.chunk(c)));
    }
    std::vector<size_t> bounds;
    for (int c : cuts) bounds.push_back((size_t) c);
    for (int w : waits) bounds.push_back((size_t) w);
    std::sort(bounds.begin(), bounds.end());
    size_t nb = 0;
    std::vector<const void*> live;   // the non-read-only buffers of every dispatch since the last kept barrier
    auto ro = [&](const void* k) { return std::find(ro_keys.begin(), ro_keys.end(), k) != ro_keys.end(); };
    std::vector<const void*> mine;
    for (size_t i = 0; i < ds.size(); ++i) {
        bool boundary = i == 0;
        while (nb < bounds.size() && bounds[nb] <= i) { if (bounds[nb] == i) boundary = true; ++nb; }
        mine.clear();
        for (const mtl::Arg& a : ds[i].args)
            if (a.buf != nullptr && !ro(key(a.buf))) mine.push_back(key(a.buf));
        if (ds[i].indirect != nullptr && !ro(key(ds[i].indirect))) mine.push_back(key(ds[i].indirect));
        if (boundary) { live = mine; continue; }
        if (!ds[i].barrier) { live.insert(live.end(), mine.begin(), mine.end()); continue; }
        ++pr_seen;
        bool clash = false;
        for (const void* k : mine)
            if (std::find(live.begin(), live.end(), k) != live.end()) { clash = true; break; }
        if (clash) { live = mine; continue; }
        ds[i].barrier = false;
        ++pr_dropped;
        live.insert(live.end(), mine.begin(), mine.end());
    }
    static int calls = 0;
    if (++calls == 200 && std::getenv("HY4_FLASH_PRUNE_STATS"))
        std::fprintf(stderr, "strata_hy4_flash_metal: prune_barriers: %llu of %llu barriers dropped over %d dispatch lists\n",
                     (unsigned long long) pr_dropped, (unsigned long long) pr_seen, calls);
}

std::string Forward::window(const int32_t* all_tokens, int p0, int T, Dump* dump) {
    Impl& im = *im_;
    if (T < 1 || T > kMaxWindow)
        return "the window is " + std::to_string(T) + ", outside 1.." + std::to_string(kMaxWindow);
    if (!im.m->stream.ready()) return "the expert arena was not built (--no-arena cannot run a forward pass)";
    im.inj = nullptr;
    im.clamp_pools(p0);
    bool need_engram = false;
    for (const Impl::BW& b : im.blk)
        if (b.engram_table >= 0) need_engram = true;
    const double te = mtl::now_s();
    if (const std::string e = im.prepare_window(all_tokens, p0, T, need_engram); !e.empty()) return e;
    if (need_engram) {
        // The whole window is one command buffer, so the rows have to be in their buffer before it is
        // committed: all four tables are joined here.
        for (int i = 0; i < im.eng.n_tables(); ++i) {
            std::string err;
            if (im.eng.join(im.eng.block_of(i), err) == nullptr)
                return "Engram block " + std::to_string(im.eng.block_of(i)) + ": " + err;
        }
        im.tm.engram_wait_ms += (mtl::now_s() - te) * 1e3;
    }
    if (const std::string e = im.embed_window(all_tokens, T); !e.empty()) return e;
    im.routed.assign((size_t) im.m->stream.n_layers() * (size_t) T * (size_t) im.topk, -1);
    im.predicted.assign((size_t) im.m->stream.n_layers() * (size_t) T * (size_t) im.topk, -1);
    if (dump != nullptr && dump->on())
        im.all_logits.assign((size_t) im.m->stream.n_layers() * (size_t) T * (size_t) im.n_experts, 0.0f);
    else
        im.all_logits.clear();

    std::vector<mtl::Dispatch> ds;
    std::vector<int> cuts;
    im.cut_block.clear();
    im.cut_wait.clear();
    im.cut_role.clear();
    // The inter-block state alternates between two buffers.  The dispatches hold POINTERS to them, so they are
    // never swapped by value - the pointers are.
    const mtl::Buffer* in = &im.hcA;
    const mtl::Buffer* out = &im.hcB;
    auto flush = [&]() -> std::string {
        if (ds.empty()) return {};
        im.prune_barriers(ds, cuts, im.cut_wait);
        const double t0 = mtl::now_s();
        im.tm.gpu_ms += im.cx->run_graph_seg(ds, cuts, im.cut_wait, cut_tramp, &im) * 1e3;
        im.tm.wall_ms += (mtl::now_s() - t0) * 1e3;
        ds.clear();
        cuts.clear();
        im.cut_block.clear();
        im.cut_wait.clear();
        im.cut_role.clear();
        return im.cut_err;
    };
    for (size_t l = 0; l < im.blk.size(); ++l) {
        const Impl::BW& b = im.blk[l];
        if (b.engram_table >= 0) {
            // one Engram block's rows at a time in the shared buffer, so what is already encoded runs first
            if (const std::string e = flush(); !e.empty()) return e;
            if (const std::string e = im.engram_rows_for(b.id, T); !e.empty()) return e;
        }
        im.encode_block(ds, cuts, b, *in, *out, T, false);
        std::swap(in, out);
    }
    im.encode_head(ds, *in, 0, T);
    if (const std::string e = flush(); !e.empty()) return e;
    // The cache is written; the flag is readable now and not before.
    if (const std::string e = im.nonfinite_check(); !e.empty()) return e;
    std::memcpy(im.last_log.data(), im.logits_b.as<float>() + (size_t) (T - 1) * (size_t) im.vocab,
                4 * (size_t) im.vocab);
    im.last_tok = im.argmax_b.as<int32_t>()[T - 1];
    im.snap_used = p0 + T;   // what a snapshot would record: the caches now hold positions 0 .. p0 + T - 1
    for (size_t i = 0; i < im.pools_done.size(); ++i) im.pools_done[i] = (p0 + T) / im.idx_kpool;
    if (dump != nullptr && dump->on()) {
        for (const Impl::BW& b : im.blk) {
            if (b.ordinal < 0) continue;
            const std::string sfx = "-" + std::to_string(b.id);
            dump->add("ffn_moe_topk" + sfx, "i32", {im.topk}, 2,
                      &im.routed[(size_t) b.ordinal * T * im.topk], 4 * (size_t) im.topk, T);
            dump->add("ffn_moe_logits" + sfx, "f32", {im.n_experts}, 2,
                      &im.all_logits[(size_t) b.ordinal * T * im.n_experts], 4 * (size_t) im.n_experts, T);
        }
        dump->add("hc_head", "f32", {im.n_embd}, 2, im.h_cur.ptr, 4 * (size_t) im.n_embd, T);
        dump->add("result_norm", "f32", {im.n_embd}, 2, im.h_rnorm.ptr, 4 * (size_t) im.n_embd, T);
    }
    im.tm.windows++;
    im.tm.positions += (uint64_t) T;
    im.tm.expert_wait_ms = im.m->stream.wait_ms();
    return {};
}

// ---- prefill -------------------------------------------------------------------------------------------------------
// One chunk, block by block.  Each block is its own command buffer with the SAME cut after its router as decode has,
// so `at_cut` -> `ExpertStream::at_layer` sees T = 1024 and takes its prompt path: the chunk's 16384 selections are
// reduced to the distinct experts of that layer and each is read from the SSD once, while the next layer's whole
// expert set streams in behind the compute (stream.cpp's `prompt` branch).
//
// WHY A COMMAND BUFFER PER BLOCK rather than one per chunk: a dense GEMV dispatch covers 8 positions, so one block
// of 1024 positions is already about 18,000 dispatches and the whole chunk would be 770,000 in a single encoder.
// The flush costs 42 commit/wait round trips a chunk, which is nothing beside the seconds a chunk takes.
std::string Forward::prefill(const int32_t* all_tokens, int p0, int T, const PrefillSink& sink, Dump* dump) {
    Impl& im = *im_;
    if (im.tcap <= kMaxWindow) return "this engine was opened without prefill (--prefill-chunk 0)";
    if (im.split && !im.big_in)
        return "the prompt's scratch is not attached (call attach_prefill after the arena is built, and prompt())";
    if (T < 1 || T > im.tcap)
        return "the chunk is " + std::to_string(T) + ", outside 1.." + std::to_string(im.tcap);
    if (!im.m->stream.ready()) return "the expert arena was not built (--no-arena cannot run a forward pass)";
    if (p0 + T > im.ctx)
        return "the chunk ends at position " + std::to_string(p0 + T) + ", past --context " +
               std::to_string(im.ctx);
    im.inj = nullptr;
    im.clamp_pools(p0);
    const double t_chunk = mtl::now_s();
    const double wait0 = im.m->stream.prompt_wait_ms(), host0 = im.m->stream.prompt_host_ms();
    // NO FENCE LAG WHILE A CHUNK RUNS, and this is what makes the arena's slot budget work.  A slot stays safe
    // from eviction for `fence_lag + 1` handshakes; decode needs 1 because it hands the GPU a whole window's
    // command buffer and the host's turn at ordinal o + 1 can start while the GPU is inside o.  Prefill commits
    // ONE BLOCK at a time and `run_graph_seg` waits for it, so when the next block's cut runs the GPU is
    // provably finished - and with lag 1 the layer before would stay pinned too, which together with this
    // layer's 512 experts and the next layer's 512 streamed ahead would need 1536 slots of one size class
    // rather than 1024 (`ExpertStream::victim` ends the process when a class runs dry: "no free slot").
    struct Lag {
        metal_engine::ExpertStream* s;
        ~Lag() { s->set_fence_lag(1); }
    } lag{&im.m->stream};
    im.m->stream.set_fence_lag(0);
    bool need_engram = false;
    for (const Impl::BW& b : im.blk)
        if (b.engram_table >= 0) need_engram = true;
    const double te = mtl::now_s();
    if (const std::string e = im.prepare_window(all_tokens, p0, T, need_engram); !e.empty()) return e;
    if (need_engram) {
        for (int i = 0; i < im.eng.n_tables(); ++i) {
            std::string err;
            if (im.eng.join(im.eng.block_of(i), err) == nullptr)
                return "Engram block " + std::to_string(im.eng.block_of(i)) + ": " + err;
        }
        im.tm.pf_engram_ms += (mtl::now_s() - te) * 1e3;   // the PROMPT's, not a decode position's
    }
    if (const std::string e = im.embed_window(all_tokens, T); !e.empty()) return e;
    const bool want_dump = dump != nullptr && dump->on();
    im.predicted.clear();
    if (want_dump) {
        im.routed.assign((size_t) im.m->stream.n_layers() * (size_t) T * (size_t) im.topk, -1);
        im.all_logits.assign((size_t) im.m->stream.n_layers() * (size_t) T * (size_t) im.n_experts, 0.0f);
    } else {
        im.routed.clear();
        im.all_logits.clear();
    }
    if (im.pr_record != nullptr && im.routed.empty())
        im.routed.assign((size_t) im.m->stream.n_layers() * (size_t) T * (size_t) im.topk, -1);
    struct InChunk {
        bool& f;
        ~InChunk() { f = false; }
    } in_chunk{im.pr_in_chunk};
    im.pr_in_chunk = true;

    const mtl::Buffer* in = &im.hcA;
    const mtl::Buffer* out = &im.hcB;
    std::vector<mtl::Dispatch> ds;
    std::vector<int> cuts;
    for (size_t l = 0; l < im.blk.size(); ++l) {
        const Impl::BW& b = im.blk[l];
        if (b.engram_table >= 0)
            if (const std::string e = im.engram_rows_for(b.id, T); !e.empty()) return e;
        const double t_enc = mtl::now_s();
        ds.clear();
        cuts.clear();
        im.cut_block.clear();
        im.cut_wait.clear();
        im.cut_role.clear();
        im.encode_block(ds, cuts, b, *in, *out, T, want_dump);
        im.tm.pf_encode_ms += (mtl::now_s() - t_enc) * 1e3;
        const double g = im.cx->run_graph_seg(ds, cuts, im.cut_wait, cut_tramp, &im) * 1e3;
        im.tm.pf_gpu_ms += g;
        if (im.tm.pf_block_ms.size() != im.blk.size()) im.tm.pf_block_ms.assign(im.blk.size(), 0.0);
        im.tm.pf_block_ms[l] += g;
        if (!im.cut_err.empty()) return im.cut_err;
        if (im.pf.noise > 0) {   // HY4_FLASH_PREFILL_NOISE (forward.h): the GPU is done with `out`, a shared buffer
            float* o = out->as<float>();
            const size_t n = (size_t) T * (size_t) im.hcw;
            const double nz = im.pf.noise;
            for (size_t i = 0; i < n; ++i) {
                uint64_t h = ((uint64_t) (p0 + (int) (i / (size_t) im.hcw)) << 40) ^ ((uint64_t) l << 32) ^
                             (uint64_t) (i % (size_t) im.hcw);
                h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33;
                const double r = (double) (h >> 11) * (2.0 / 9007199254740992.0) - 1.0;
                o[i] = (float) ((double) o[i] * (1.0 + nz * r));
            }
        }
        // The per-block dump needs the block's own buffers, which only this per-block flush leaves alive: the
        // decode path encodes all 42 blocks into one command buffer and the intermediates are gone by the end.
        if (want_dump) im.dump_block(b, T, dump, Inject{}, *in, *out);
        std::swap(in, out);
    }
    if (im.pr_record != nullptr && p0 + T > im.pr_n)
        return "set_prompt_routing was given " + std::to_string(im.pr_n) + " positions but the prompt reached " +
               std::to_string(p0 + T);
    if (im.pr_record != nullptr)   // set_prompt_routing: [ordinal][pos][k]
        for (int o = 0; o < im.m->stream.n_layers(); ++o)
            std::memcpy(&(*im.pr_record)[((size_t) o * im.pr_n + p0) * im.topk], &im.routed[(size_t) o * T * im.topk],
                        4 * (size_t) T * im.topk);
    for (size_t i = 0; i < im.pools_done.size(); ++i) im.pools_done[i] = (p0 + T) / im.idx_kpool;
    im.snap_used = p0 + T;   // the caches now hold positions 0 .. p0 + T - 1 (prefill_snap.h)

    // the head, in its own command buffers: the logits of 1024 positions are 495 MB, so they are handed to the
    // caller a sub-batch at a time (forward.h's PrefillSink)
    const double t_head = mtl::now_s();
    if (sink.hidden) {
        for (int t0 = 0; t0 < T; t0 += im.hsub) {
            const int nt = std::min(im.hsub, T - t0);
            if (const std::string e = im.run_hidden(*in, t0, nt); !e.empty()) return e;
            sink.hidden(p0 + t0, nt, im.h_rnorm.as<float>());
        }
    }
    if (sink.all_positions) {
        for (int t0 = 0; t0 < T; t0 += im.hsub) {
            const int nt = std::min(im.hsub, T - t0);
            if (const std::string e = im.run_head(*in, t0, nt); !e.empty()) return e;
            if (sink.logits) sink.logits(p0 + t0, nt, im.logits_b.as<float>());
        }
    } else {
        if (const std::string e = im.run_head(*in, T - 1, 1); !e.empty()) return e;
        if (sink.logits) sink.logits(p0 + T - 1, 1, im.logits_b.as<float>());
    }
    im.tm.pf_head_ms += (mtl::now_s() - t_head) * 1e3;
    if (want_dump) {
        for (const Impl::BW& b : im.blk) {
            if (b.ordinal < 0) continue;
            const std::string sfx = "-" + std::to_string(b.id);
            dump->add("ffn_moe_topk" + sfx, "i32", {im.topk}, 2,
                      &im.routed[(size_t) b.ordinal * T * im.topk], 4 * (size_t) im.topk, T);
            dump->add("ffn_moe_logits" + sfx, "f32", {im.n_experts}, 2,
                      &im.all_logits[(size_t) b.ordinal * T * im.n_experts], 4 * (size_t) im.n_experts, T);
        }
    }
    im.tm.pf_wall_ms += (mtl::now_s() - t_chunk) * 1e3;
    im.tm.pf_host_ms += im.m->stream.prompt_host_ms() - host0;
    im.tm.pf_wait_ms += im.m->stream.prompt_wait_ms() - wait0;
    im.tm.pf_bytes_read = im.m->stream.bytes_read();
    im.tm.pf_chunks++;
    im.tm.pf_positions += (uint64_t) T;
    return {};
}

std::string Forward::attach_prefill() { return im_ == nullptr ? std::string() : im_->attach_prefill(); }

std::string Forward::prompt(const int32_t* tokens, int n, const PrefillSink& sink, Dump* dump) {
    Impl& im = *im_;
    const int C = im.tcap;
    if (C <= kMaxWindow) return "this engine was opened without prefill (--prefill-chunk 0)";
    struct Prompt {   // the prompt's scratch is in for exactly this call, on every way out
        Impl& im;
        bool on;
        explicit Prompt(Impl& i) : im(i), on(i.split && !i.big.empty() && !i.big_in) {
            if (on) { im.m->stream.hold_region(true); im.swap_scratch(); }
        }
        ~Prompt() {
            if (on) { im.swap_scratch(); im.m->stream.hold_region(false); }
        }
    } prompt_scratch(im);
    for (int p = 0; p < n; p += C) {
        const int nt = std::min(C, n - p);
        if (const std::string e = prefill(tokens, p, nt, sink, dump); !e.empty())
            return "prefill at position " + std::to_string(p) + ": " + e;
    }
    return {};
}

std::string Forward::block_window(int L, const int32_t* all_tokens, int p0, int T, const Inject& inj, Dump* dump) {
    Impl& im = *im_;
    if (L < 0 || (size_t) L >= im.blk.size()) return "block " + std::to_string(L) + " is outside the trunk";
    if (T < 1 || T > kMaxWindow) return "the window is outside 1.." + std::to_string(kMaxWindow);
    const Impl::BW& b = im.blk[(size_t) L];
    if (b.ordinal >= 0 && !im.m->stream.ready()) return "the expert arena was not built";
    im.inj = &inj;
    if (const std::string e = im.prepare_window(all_tokens, p0, T, b.engram_table >= 0); !e.empty()) return e;
    if (b.engram_table >= 0)
        if (const std::string e = im.engram_rows_for(b.id, T); !e.empty()) return e;
    if (inj.hc != nullptr) std::memcpy(im.hcA.ptr, inj.hc, 4 * (size_t) T * (size_t) im.hcw);
    if (inj.sel != nullptr && inj.n_sel != nullptr) {
        std::memcpy(im.sel.ptr, inj.sel, 4 * (size_t) T * (size_t) kSelCap);
        std::memcpy(im.nsel.ptr, inj.n_sel, 4 * (size_t) T);
    }
    im.routed.clear();
    im.predicted.clear();
    std::vector<mtl::Dispatch> ds;
    std::vector<int> cuts;
    im.cut_block.clear();
    im.cut_wait.clear();
    im.cut_role.clear();
    im.encode_block(ds, cuts, b, im.hcA, im.hcB, T, dump != nullptr);
    im.prune_barriers(ds, cuts, im.cut_wait);
    const double t0 = mtl::now_s();
    im.tm.gpu_ms += im.cx->run_graph_seg(ds, cuts, im.cut_wait, cut_tramp, &im) * 1e3;
    im.tm.wall_ms += (mtl::now_s() - t0) * 1e3;
    if (!im.cut_err.empty()) return im.cut_err;
    if (const std::string e = im.nonfinite_check(); !e.empty()) return e;   // only after the buffer ran
    im.dump_block(b, T, dump, inj, im.hcA, im.hcB);
    if (b.has_indexer) im.pools_done[(size_t) b.idx_slot] = (p0 + T) / im.idx_kpool;
    im.tm.windows++;
    im.tm.positions += (uint64_t) T;
    return {};
}

// No `nonfinite_check` here, and that is deliberate: the head writes no f16 cache (`encode_head` has no
// `hy4_flash_store_half` dispatch), so there is nothing for it to catch.
std::string Forward::head_window(const float* hc_in, int T, Dump* dump) {
    Impl& im = *im_;
    if (T < 1 || T > kMaxWindow) return "the window is outside 1.." + std::to_string(kMaxWindow);
    std::memcpy(im.hcA.ptr, hc_in, 4 * (size_t) T * (size_t) im.hcw);
    std::vector<mtl::Dispatch> ds;
    im.encode_head(ds, im.hcA, 0, T);
    im.tm.gpu_ms += im.cx->run(ds) * 1e3;
    if (dump != nullptr && dump->on()) {
        dump->add("hc_head", "f32", {im.n_embd}, 2, im.h_cur.ptr, 4 * (size_t) im.n_embd, T);
        dump->add("result_norm", "f32", {im.n_embd}, 2, im.h_rnorm.ptr, 4 * (size_t) im.n_embd, T);
        dump->add("result_output", "f32", {im.vocab}, 2, im.logits_b.ptr, 4 * (size_t) im.vocab, T);
    }
    return {};
}

}  // namespace strata::hy4_flash
