// engine/metal/src/hy4flash_gguf/vis_tower.cpp - see vis_tower.h.
#include "vis_tower.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::hy4_flash {
namespace {

// host mirrors of the kernels' argument structs (hy4_flash_attn.metal, hy4_flash_norm.metal, hy4_flash_vision.metal)
struct ElemArgs {
    uint32_t n, a_stride, b_stride, y_stride;
};
struct NormArgs {
    uint32_t n, x_stride, y_stride;
    float eps;
};
struct PatchArgs {
    uint32_t width, gw, patch;
};
struct PosArgs {
    uint32_t gh, gw, edge, dim;
    float inv_sy, inv_sx;
};
struct LnArgs {
    uint32_t n, x_stride, y_stride;
    float eps;
};
struct AttnArgs {
    uint32_t n_keys, n_query, q0, heads, head_dim, qkv_stride, s_stride;
    float scale;
};
struct ActArgs {
    uint32_t n;
};
struct MergeArgs {
    uint32_t gw, ow, dim, merge;
};
struct RowsArgs {
    uint32_t oh, ow, dim;
};

constexpr uint32_t kTg = 256;        ///< HY4_FLASH_TG: every row kernel in this tower uses it

/// The score block's byte budget. The score matrix is heads * n_query * n_keys and n_keys is every patch,
/// so for a large picture it is the one array that cannot be held whole; blocking the queries is what keeps
/// it bounded, and this is the bound.
constexpr int64_t kScoreFloats = 16ll << 20;

/// gemv.metal computes `U = n_in / UNIT` and TRUNCATES, so a K that is not a multiple of UNIT silently
/// drops its tail. UNIT is 16 for both of the types this shard uses (quant.metal's fmt<T_BF16> and
/// fmt<T_F32>), and every vision K - 768, 1152, 4304, 4608 - is a multiple of 16.
constexpr uint32_t kGemvUnit = 16;

void elem(std::vector<mtl::Dispatch>& ds, const mtl::Pipeline& p, std::vector<mtl::Arg> args, uint32_t n,
          int64_t rows, const char* where) {
    mtl::Dispatch d;
    d.p = &p;
    d.args = std::move(args);
    d.tg[0] = kTg;
    d.grid[0] = (n + kTg - 1) / kTg;
    d.grid[1] = (uint32_t) rows;
    mtl::require_fit(d, where);
    ds.push_back(d);
}

/// A kernel with one thread group a row and kTg threads in it.
void per_row(std::vector<mtl::Dispatch>& ds, const mtl::Pipeline& p, std::vector<mtl::Arg> args, int64_t rows,
             const char* where) {
    mtl::Dispatch d;
    d.p = &p;
    d.args = std::move(args);
    d.tg[0] = kTg;
    d.grid[0] = (uint32_t) rows;
    mtl::require_fit(d, where);
    ds.push_back(d);
}

int64_t prod(const std::vector<uint64_t>& s) {
    int64_t n = 1;
    for (uint64_t d : s) n *= (int64_t) d;
    return n;
}

}  // namespace

PictureGrid picture_grid(const VisionGeometry& g, int32_t h, int32_t w) {
    PictureGrid p;
    p.gh = h / g.patch;
    p.gw = w / g.patch;
    p.n = (int64_t) p.gh * p.gw;
    p.oh = p.gh / g.merge;
    p.ow = p.gw / g.merge;
    p.cells = (int64_t) p.oh * p.ow;
    p.rows = (int64_t) p.oh * (p.ow + 1);
    return p;
}

const DenseGemvF32* VisionTower::shape(const std::string& tensor) const {
    auto it = dense_.find(tensor);
    return it == dense_.end() ? nullptr : &it->second;
}

std::string VisionTower::need(mtl::Context& cx, mtl::Buffer& b, int64_t floats, const char* what) {
    const size_t want = (size_t) floats * sizeof(float);
    if (b.handle != nullptr && b.bytes >= want) return {};
    if (b.handle != nullptr) cx.free(b);
    b = cx.alloc(want);
    if (b.handle == nullptr) return std::string("the vision tower could not allocate ") + what;
    return {};
}

std::string VisionTower::open(mtl::Context& cx, const VisionShard& s, int lib) {
    geo_ = s.geo;
    lib_ = lib;

    patchify_ = cx.pipeline(lib, "hy4_flash_vis_patchify");
    posemb_ = cx.pipeline(lib, "hy4_flash_vis_posemb");
    ln_ = cx.pipeline(lib, "hy4_flash_vis_ln");
    score_ = cx.pipeline(lib, "hy4_flash_vis_score");
    softmax_ = cx.pipeline(lib, "hy4_flash_vis_softmax");
    pv_ = cx.pipeline(lib, "hy4_flash_vis_pv");
    merge_ = cx.pipeline(lib, "hy4_flash_vis_merge");
    rows_ = cx.pipeline(lib, "hy4_flash_vis_rows");
    // The encoder FFN's tanh and the projector's erf are two pipelines, so each call site names the form it
    // wants (hy4_flash_vision.metal's note on why they are not one kernel with a flag).
    act_tanh_ = cx.pipeline(lib, "hy4_flash_vis_act_tanh");
    act_erf_ = cx.pipeline(lib, "hy4_flash_vis_act_erf");
    rmsnorm_ = cx.pipeline(lib, "hy4_flash_rmsnorm");
    add_row_ = cx.pipeline(lib, "hy4_flash_add_row");
    add_f32_ = cx.pipeline(lib, "hy4_flash_add_f32");

    std::vector<std::string> mats = {"v.patch_embd.weight", "mm.0.weight", "mm.2.weight"};
    for (int32_t l = 0; l < geo_.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_out.weight",
                              "ffn_up.weight", "ffn_down.weight"})
            mats.push_back(p + n);
    }
    for (const std::string& name : mats) {
        const gguf::Tensor* t = s.find(name);
        if (t == nullptr) return "the vision shard lacks " + name;
        if (t->type != 0 && t->type != 30)
            return name + " is ggml type " + std::to_string(t->type) + ", and the tower reads only F32 and BF16";
        const int64_t n_out = (int64_t) t->shape.back(), n_in = prod(t->shape) / n_out;
        if (n_in % kGemvUnit != 0)
            return name + " contracts over " + std::to_string(n_in) + ", which is not a multiple of " +
                   std::to_string(kGemvUnit) + ": the dense GEMV would drop the remainder without failing";
        dense_.emplace(name, DenseGemvF32(cx, lib, t->type, (uint32_t) n_in, (uint32_t) n_out,
                                          (uint32_t) (t->bytes / (uint64_t) n_out), {}, /*allow_mm=*/true));
    }
    return {};
}

void VisionTower::close(mtl::Context& cx) {
    for (mtl::Buffer* b : {&rgb_, &px_, &x_, &t_, &q_, &k_, &v_, &f_, &sc_, &cell_, &mid_, &co_, &out_})
        if (b->handle != nullptr) cx.free(*b);
    dense_.clear();
    sc_floats_ = 0;
}

std::string VisionTower::encode(mtl::Context& cx, const VisionShard& s, const uint8_t* rgb, int32_t h,
                                int32_t w, std::vector<float>& out, const StageSink& sink) {
    if (!s.loaded()) return "the vision shard is not open";
    if (const std::string why = check_picture_size(geo_, h, w, geo_.max_pixels); !why.empty()) return why;

    const PictureGrid g = picture_grid(geo_, h, w);
    const uint32_t D = (uint32_t) geo_.hidden, HD = (uint32_t) geo_.head_dim, FF = (uint32_t) geo_.ffn;
    const uint32_t OD = (uint32_t) geo_.out_hidden, PK = 3u * geo_.patch * geo_.patch;
    const uint32_t CD = D * (uint32_t) (geo_.merge * geo_.merge);
    const uint32_t s_stride = ((uint32_t) g.n + 3u) & ~3u;
    const int64_t qb = std::clamp<int64_t>(kScoreFloats / ((int64_t) geo_.heads * s_stride), 1, g.n);

    struct Want {
        mtl::Buffer* b;
        int64_t floats;
        const char* what;
    };
    const Want wants[] = {{&px_, g.n * PK, "the patch plane"},
                          {&x_, g.n * D, "the residual stream"},
                          {&t_, g.n * D, "the norm output"},
                          {&q_, g.n * D, "the queries"},
                          {&k_, g.n * D, "the keys"},
                          {&v_, g.n * D, "the values"},
                          {&f_, g.n * FF, "the FFN hidden state"},
                          {&sc_, (int64_t) geo_.heads * qb * s_stride, "the score block"},
                          {&cell_, g.cells * CD, "the merged cells"},
                          {&mid_, g.cells * CD, "the projector's hidden state"},
                          {&co_, g.cells * OD, "the projected cells"},
                          {&out_, g.rows * OD, "the output rows"}};
    for (const Want& want : wants)
        if (const std::string why = need(cx, *want.b, want.floats, want.what); !why.empty()) return why;
    const size_t rgb_bytes = (size_t) h * w * 3;
    if (rgb_.handle != nullptr && rgb_.bytes < rgb_bytes) cx.free(rgb_);
    if (rgb_.handle == nullptr) rgb_ = cx.alloc(std::max<size_t>(rgb_bytes, 1));
    if (rgb_.handle == nullptr) return "the vision tower could not allocate the pixel buffer";
    std::memcpy(rgb_.ptr, rgb, rgb_bytes);

    // One graph a stage, committed and waited for by run_graph. The split is the same whether or not a sink
    // is attached, so the dumps describe exactly the tower that runs without them - a sink that changed the
    // batching would be measuring a different tower than the one it is there to localise.
    //
    // run_graph rather than the deferred queue because this file is compiled into the engine AND into the
    // test tree, and mtl_rt's flush_runs / mark / split_runs exist only under STRATA_DS4_RUNTIME.
    std::vector<mtl::Dispatch> ds;
    auto stage = [&](const char* name, const mtl::Buffer& b, int64_t rows, int64_t cols) {
        if (!ds.empty()) cx.run_graph(ds);
        ds.clear();
        if (sink) sink(name, b.as<float>(), rows, cols);
    };

    // ---- the stem ------------------------------------------------------------------------------------
    {
        PatchArgs a{(uint32_t) w, (uint32_t) g.gw, (uint32_t) geo_.patch};
        per_row(ds, patchify_, {mtl::Arg::B(rgb_), mtl::Arg::B(px_), mtl::Arg::V(a)}, g.n, "hy4_flash_vis_patchify");
    }
    shape("v.patch_embd.weight")
        ->append(ds, s.weights, s.find("v.patch_embd.weight")->offset, px_, PK, x_, D, 0, (uint32_t) g.n);
    {
        ElemArgs a{D, D, 0, D};
        elem(ds, add_row_, {mtl::Arg::B(x_), s.arg("v.patch_embd.bias"), mtl::Arg::B(x_), mtl::Arg::V(a)}, D,
             g.n, "hy4_flash_add_row");
        // the 0.1 is the reference's: it is what makes floor(edge * scale) land on the last patch exactly,
        // and the ratio is formed in double because the float quotient would not
        PosArgs p{(uint32_t) g.gh, (uint32_t) g.gw, (uint32_t) geo_.pos_edge, D,
                  (float) ((double) geo_.pos_edge / ((double) g.gh + 0.1)),
                  (float) ((double) geo_.pos_edge / ((double) g.gw + 0.1))};
        per_row(ds, posemb_, {s.arg("v.position_embd.weight"), mtl::Arg::B(x_), mtl::Arg::V(p)}, g.n,
                "hy4_flash_vis_posemb");
    }
    stage("emb0", x_, g.n, D);

    // ---- the encoder ---------------------------------------------------------------------------------
    const LnArgs ln{D, D, D, geo_.eps};
    const ElemArgs res{D, D, D, D};
    const float scale = 1.0f / std::sqrt((float) HD);
    for (int32_t l = 0; l < geo_.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        // t = ln1(x); then q, k, v from t, after which t is dead and the attention output reuses it
        per_row(ds, ln_,
                {mtl::Arg::B(x_), s.arg(p + "ln1.weight"), s.arg(p + "ln1.bias"), mtl::Arg::B(t_),
                 mtl::Arg::V(ln)},
                g.n, "hy4_flash_vis_ln");
        struct Proj {
            const char* w;
            const char* b;
            mtl::Buffer* y;
        };
        for (const Proj& pr : {Proj{"attn_q.weight", "attn_q.bias", &q_},
                               Proj{"attn_k.weight", "attn_k.bias", &k_},
                               Proj{"attn_v.weight", "attn_v.bias", &v_}}) {
            shape(p + pr.w)->append(ds, s.weights, s.find(p + pr.w)->offset, t_, D, *pr.y, D, 0, (uint32_t) g.n);
            ElemArgs a{D, D, 0, D};
            elem(ds, add_row_, {mtl::Arg::B(*pr.y), s.arg(p + pr.b), mtl::Arg::B(*pr.y), mtl::Arg::V(a)}, D,
                 g.n, "hy4_flash_add_row");
        }
        // Full bidirectional attention: every patch is a key. The queries are blocked so that the score
        // matrix stays within kScoreFloats; the keys never are, so each block's softmax is over the whole
        // row and no rescaling across blocks is needed.
        for (int64_t q0 = 0; q0 < g.n; q0 += qb) {
            const uint32_t nq = (uint32_t) std::min<int64_t>(qb, g.n - q0);
            AttnArgs a{(uint32_t) g.n, nq, (uint32_t) q0, (uint32_t) geo_.heads, HD, D, s_stride, scale};
            auto head_block = [&](const mtl::Pipeline& p, std::vector<mtl::Arg> args, const char* where) {
                mtl::Dispatch d;
                d.p = &p;
                d.args = std::move(args);
                d.tg[0] = kTg;
                d.grid[0] = nq;
                d.grid[1] = (uint32_t) geo_.heads;
                mtl::require_fit(d, where);
                ds.push_back(d);
            };
            head_block(score_, {mtl::Arg::B(q_), mtl::Arg::B(k_), mtl::Arg::B(sc_), mtl::Arg::V(a)},
                       "hy4_flash_vis_score");
            head_block(softmax_, {mtl::Arg::B(sc_), mtl::Arg::V(a)}, "hy4_flash_vis_softmax");
            head_block(pv_, {mtl::Arg::B(sc_), mtl::Arg::B(v_), mtl::Arg::B(t_), mtl::Arg::V(a)},
                       "hy4_flash_vis_pv");
        }
        // x += attn_out(t) + bias, with q dead and reused for the product
        shape(p + "attn_out.weight")
            ->append(ds, s.weights, s.find(p + "attn_out.weight")->offset, t_, D, q_, D, 0, (uint32_t) g.n);
        {
            ElemArgs a{D, D, 0, D};
            elem(ds, add_row_, {mtl::Arg::B(q_), s.arg(p + "attn_out.bias"), mtl::Arg::B(q_), mtl::Arg::V(a)},
                 D, g.n, "hy4_flash_add_row");
        }
        elem(ds, add_f32_, {mtl::Arg::B(x_), mtl::Arg::B(q_), mtl::Arg::B(x_), mtl::Arg::V(res)}, D, g.n,
             "hy4_flash_add_f32");

        // t = ln2(x); f = gelu_tanh(up(t) + bias); x += down(f) + bias
        per_row(ds, ln_,
                {mtl::Arg::B(x_), s.arg(p + "ln2.weight"), s.arg(p + "ln2.bias"), mtl::Arg::B(t_),
                 mtl::Arg::V(ln)},
                g.n, "hy4_flash_vis_ln");
        shape(p + "ffn_up.weight")
            ->append(ds, s.weights, s.find(p + "ffn_up.weight")->offset, t_, D, f_, FF, 0, (uint32_t) g.n);
        {
            ElemArgs a{FF, FF, 0, FF};
            elem(ds, add_row_, {mtl::Arg::B(f_), s.arg(p + "ffn_up.bias"), mtl::Arg::B(f_), mtl::Arg::V(a)},
                 FF, g.n, "hy4_flash_add_row");
            ActArgs act{(uint32_t) (g.n * FF)};
            elem(ds, act_tanh_, {mtl::Arg::B(f_), mtl::Arg::V(act)}, act.n, 1, "hy4_flash_vis_act_tanh");
        }
        shape(p + "ffn_down.weight")
            ->append(ds, s.weights, s.find(p + "ffn_down.weight")->offset, f_, FF, q_, D, 0, (uint32_t) g.n);
        {
            ElemArgs a{D, D, 0, D};
            elem(ds, add_row_, {mtl::Arg::B(q_), s.arg(p + "ffn_down.bias"), mtl::Arg::B(q_), mtl::Arg::V(a)},
                 D, g.n, "hy4_flash_add_row");
        }
        elem(ds, add_f32_, {mtl::Arg::B(x_), mtl::Arg::B(q_), mtl::Arg::B(x_), mtl::Arg::V(res)}, D, g.n,
             "hy4_flash_add_f32");
        stage(("L" + std::to_string(l)).c_str(), x_, g.n, D);
    }

    // ---- the projector -------------------------------------------------------------------------------
    // mm.pre_norm is an RMSNorm: a weight and no bias, which is why this is hy4_flash_rmsnorm and not hy4_flash_vis_ln.
    {
        NormArgs a{D, D, D, geo_.eps};
        per_row(ds, rmsnorm_,
                {mtl::Arg::B(x_), s.arg("mm.pre_norm.weight"), mtl::Arg::B(t_), mtl::Arg::V(a)}, g.n,
                "hy4_flash_rmsnorm");
    }
    stage("pre_norm", t_, g.n, D);
    {
        MergeArgs a{(uint32_t) g.gw, (uint32_t) g.ow, D, (uint32_t) geo_.merge};
        per_row(ds, merge_, {mtl::Arg::B(t_), mtl::Arg::B(cell_), mtl::Arg::V(a)}, g.cells, "hy4_flash_vis_merge");
    }
    shape("mm.0.weight")->append(ds, s.weights, s.find("mm.0.weight")->offset, cell_, CD, mid_, CD, 0,
                                 (uint32_t) g.cells);
    {
        ElemArgs a{CD, CD, 0, CD};
        elem(ds, add_row_, {mtl::Arg::B(mid_), s.arg("mm.0.bias"), mtl::Arg::B(mid_), mtl::Arg::V(a)}, CD,
             g.cells, "hy4_flash_add_row");
        ActArgs act{(uint32_t) (g.cells * CD)};
        elem(ds, act_erf_, {mtl::Arg::B(mid_), mtl::Arg::V(act)}, act.n, 1, "hy4_flash_vis_act_erf");
    }
    stage("mm0", mid_, g.cells, CD);
    shape("mm.2.weight")->append(ds, s.weights, s.find("mm.2.weight")->offset, mid_, CD, co_, OD, 0,
                                 (uint32_t) g.cells);
    {
        ElemArgs a{OD, OD, 0, OD};
        elem(ds, add_row_, {mtl::Arg::B(co_), s.arg("mm.2.bias"), mtl::Arg::B(co_), mtl::Arg::V(a)}, OD,
             g.cells, "hy4_flash_add_row");
    }
    stage("mm2", co_, g.cells, OD);
    {
        RowsArgs a{(uint32_t) g.oh, (uint32_t) g.ow, OD};
        per_row(ds, rows_, {mtl::Arg::B(co_), s.arg("v.image_newline"), mtl::Arg::B(out_), mtl::Arg::V(a)},
                g.rows, "hy4_flash_vis_rows");
    }
    cx.run_graph(ds);
    out.assign(out_.as<float>(), out_.as<float>() + g.rows * OD);
    return {};
}

}  // namespace strata::hy4_flash
