// engine/metal/src/hy4flash_gguf/vision.cpp - see vision.h.
#include "vision.h"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace strata::hy4_flash {
namespace {

/// The hy4-flash vocabulary the image token ids were checked against: read from the trunk GGUF's
/// tokenizer.ggml.tokens, and the same in both quantizations of this model and in the pack's vocab.json.
constexpr uint64_t kTrunkVocabSize = 120832;

bool num(const gguf::File& f, const char* key, double& out) {
    auto it = f.meta_num.find(key);
    if (it == f.meta_num.end()) return false;
    out = it->second;
    return true;
}

std::string want(const gguf::File& f, const char* key, int64_t& out) {
    double v = 0;
    if (!num(f, key, v)) return std::string("the vision shard has no ") + key;
    out = (int64_t) std::llround(v);
    return {};
}

/// The tensors every block and the projector need. A shard missing one of these would fail later inside a
/// dispatch, where the name is gone.
std::string check_tensors(const gguf::File& f, const VisionGeometry& g) {
    const char* fixed[] = {"v.patch_embd.weight", "v.patch_embd.bias", "v.position_embd.weight",
                           "v.image_newline", "mm.pre_norm.weight", "mm.0.weight", "mm.0.bias",
                           "mm.2.weight", "mm.2.bias"};
    for (const char* n : fixed)
        if (f.find(n) == nullptr) return std::string("the vision shard lacks ") + n;
    const char* per_block[] = {"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                               "attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias",
                               "attn_v.weight", "attn_v.bias", "attn_out.weight", "attn_out.bias",
                               "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"};
    for (int32_t l = 0; l < g.layers; ++l)
        for (const char* n : per_block) {
            const std::string name = "v.blk." + std::to_string(l) + "." + n;
            if (f.find(name) == nullptr) return "the vision shard lacks " + name;
        }
    return {};
}

/// Shapes, in GGUF order (dim 0 varies fastest). Checked because a transposed or mis-sized matrix is the
/// one error that produces numbers rather than a failure.
std::string check_shapes(const gguf::File& f, const VisionGeometry& g) {
    struct Want {
        const char* name;
        std::vector<uint64_t> ne;
    };
    const uint64_t D = (uint64_t) g.hidden, F = (uint64_t) g.ffn, O = (uint64_t) g.out_hidden;
    const uint64_t M = (uint64_t) g.merge, P = (uint64_t) g.patch, E = (uint64_t) g.pos_edge;
    std::vector<Want> wants = {
        {"v.patch_embd.weight", {P, P, 3, D}},
        {"v.patch_embd.bias", {D}},
        {"v.position_embd.weight", {D, E * E}},
        {"v.image_newline", {O}},
        {"mm.pre_norm.weight", {D}},
        {"mm.0.weight", {M, M, D, D * M * M}},
        {"mm.0.bias", {D * M * M}},
        {"mm.2.weight", {1, 1, D * M * M, O}},
        {"mm.2.bias", {O}},
    };
    for (int32_t l = 0; l < g.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        const std::vector<std::pair<std::string, std::vector<uint64_t>>> per = {
            {p + "ln1.weight", {D}}, {p + "ln1.bias", {D}},
            {p + "ln2.weight", {D}}, {p + "ln2.bias", {D}},
            {p + "attn_q.weight", {D, D}}, {p + "attn_q.bias", {D}},
            {p + "attn_k.weight", {D, D}}, {p + "attn_k.bias", {D}},
            {p + "attn_v.weight", {D, D}}, {p + "attn_v.bias", {D}},
            {p + "attn_out.weight", {D, D}}, {p + "attn_out.bias", {D}},
            {p + "ffn_up.weight", {D, F}}, {p + "ffn_up.bias", {F}},
            {p + "ffn_down.weight", {F, D}}, {p + "ffn_down.bias", {D}},
        };
        for (const auto& kv : per) {
            const gguf::Tensor* t = f.find(kv.first);
            if (t == nullptr) return "the vision shard lacks " + kv.first;
            if (t->shape != kv.second) {
                std::string s = kv.first + " is";
                for (uint64_t d : t->shape) s += " " + std::to_string(d);
                s += ", want";
                for (uint64_t d : kv.second) s += " " + std::to_string(d);
                return s;
            }
        }
    }
    for (const Want& w : wants) {
        const gguf::Tensor* t = f.find(w.name);
        if (t == nullptr) return std::string("the vision shard lacks ") + w.name;
        if (t->shape != w.ne) {
            std::string s = std::string(w.name) + " is";
            for (uint64_t d : t->shape) s += " " + std::to_string(d);
            s += ", want";
            for (uint64_t d : w.ne) s += " " + std::to_string(d);
            return s;
        }
    }
    return {};
}

}  // namespace

mtl::Arg VisionShard::arg(const std::string& name) const {
    const gguf::Tensor* t = find(name);
    return mtl::Arg::B(weights, t == nullptr ? 0 : t->offset);
}

namespace {

/// round(n / f) with Python's half-to-EVEN rule, in integers only: at a tie the quotient goes to the even
/// one, so 80/32 is 2 and not 3. `std::round` would be half away from zero and wrong here.
int64_t round_half_even_div(int64_t n, int64_t f) {
    const int64_t q = n / f, r = n % f;
    if (2 * r < f) return q;
    if (2 * r > f) return q + 1;
    return (q % 2 == 0) ? q : q + 1;
}

}  // namespace

std::string smart_resize_target(int32_t h, int32_t w, int32_t factor, int64_t min_pixels, int64_t max_pixels,
                                int32_t& out_h, int32_t& out_w) {
    if (h <= 0 || w <= 0) return "a picture must have a positive height and width";
    if (factor <= 0) return "the resize factor must be positive";
    const int32_t lo = std::min(h, w), hi = std::max(h, w);
    if ((double) hi / (double) lo > 200.0)
        return "the aspect ratio must be smaller than 200, and this picture is " + std::to_string(hi) + ":" +
               std::to_string(lo);
    int64_t hb = round_half_even_div(h, factor) * factor;
    int64_t wb = round_half_even_div(w, factor) * factor;
    const double px = (double) h * (double) w;
    if (hb * wb > max_pixels) {
        const double beta = std::sqrt(px / (double) max_pixels);
        hb = std::max<int64_t>(factor, (int64_t) std::floor((double) h / beta / factor) * factor);
        wb = std::max<int64_t>(factor, (int64_t) std::floor((double) w / beta / factor) * factor);
    } else if (hb * wb < min_pixels) {
        // the authority applies no max(factor, ...) on this branch, and both branches read the ORIGINAL
        // h and w rather than the rounded ones
        const double beta = std::sqrt((double) min_pixels / px);
        hb = (int64_t) std::ceil((double) h * beta / factor) * factor;
        wb = (int64_t) std::ceil((double) w * beta / factor) * factor;
    }
    out_h = (int32_t) hb;
    out_w = (int32_t) wb;
    return {};
}

std::string patch_grid_for(const VisionGeometry& g, int32_t h, int32_t w, int64_t max_pixels, int32_t& gh,
                           int32_t& gw) {
    int32_t rh = 0, rw = 0;
    const int64_t cap = max_pixels > 0 ? std::min<int64_t>(max_pixels, g.max_pixels) : g.max_pixels;
    if (const std::string why = smart_resize_target(h, w, g.patch * g.merge, g.min_pixels, cap, rh, rw);
        !why.empty())
        return why;
    gh = rh / g.patch;
    gw = rw / g.patch;
    return {};
}

int64_t image_token_count(const VisionGeometry& g, int32_t h, int32_t w) {
    const int64_t cell = (int64_t) g.patch * g.merge;
    return (h / cell) * (w / cell + 1);
}

std::string check_picture_size(const VisionGeometry& g, int32_t h, int32_t w, int64_t max_pixels) {
    const int64_t cell = (int64_t) g.patch * g.merge;
    if (h < cell || w < cell || h % cell != 0 || w % cell != 0)
        return "a picture's sides must be multiples of " + std::to_string(cell) + " and at least that, got " +
               std::to_string(w) + "x" + std::to_string(h);
    const int64_t px = (int64_t) h * w;
    if (px > max_pixels)
        return "a " + std::to_string(w) + "x" + std::to_string(h) + " picture is " + std::to_string(px) +
               " pixels, over the limit of " + std::to_string(max_pixels);
    if (px < g.min_pixels)
        return "a " + std::to_string(w) + "x" + std::to_string(h) + " picture is " + std::to_string(px) +
               " pixels, under the shard's minimum of " + std::to_string(g.min_pixels);
    return {};
}

std::string open_vision(VisionShard& out, mtl::Context& cx, const std::string& mmproj_path,
                        const std::string& trunk_shard1, int32_t trunk_hidden) {
    std::string err;
    if (!out.tensors.open(mmproj_path, err)) return "vision shard: " + err;
    const gguf::File& f = out.tensors;
    out.path = mmproj_path;

    const auto arch = f.meta_str.find("general.architecture");
    if (arch == f.meta_str.end() || arch->second != "clip")
        return "the vision shard's general.architecture is not 'clip'";
    const auto proj = f.meta_str.find("clip.projector_type");
    if (proj == f.meta_str.end() || proj->second != "hy_v4_flash")
        return "hy4-flash requires a hy_v4_flash mmproj GGUF: clip.projector_type is '" +
               (proj == f.meta_str.end() ? std::string("(absent)") : proj->second) + "'";
    double has_vision = 0;
    if (!num(f, "clip.has_vision_encoder", has_vision) || has_vision == 0)
        return "the vision shard does not declare a vision encoder";

    VisionGeometry& g = out.geo;
    int64_t v = 0;
    struct Field { const char* key; int32_t* dst; };
    const Field fields[] = {
        {"clip.vision.embedding_length", &g.hidden},
        {"clip.vision.attention.head_count", &g.heads},
        {"clip.vision.feed_forward_length", &g.ffn},
        {"clip.vision.block_count", &g.layers},
        {"clip.vision.patch_size", &g.patch},
        {"clip.vision.spatial_merge_size", &g.merge},
        {"clip.vision.projection_dim", &g.out_hidden},
    };
    for (const Field& fl : fields) {
        if (const std::string e = want(f, fl.key, v); !e.empty()) return e;
        *fl.dst = (int32_t) v;
    }
    if (const std::string e = want(f, "clip.vision.image_size", v); !e.empty()) return e;
    const int64_t image_size = v;
    double eps = 0;
    if (!num(f, "clip.vision.attention.layer_norm_epsilon", eps))
        return "the vision shard has no clip.vision.attention.layer_norm_epsilon";
    g.eps = (float) eps;
    double mn = 0, mx = 0;
    if (!num(f, "clip.vision.image_min_pixels", mn) || !num(f, "clip.vision.image_max_pixels", mx))
        return "the vision shard has no image_min_pixels / image_max_pixels";
    g.min_pixels = (int64_t) mn;
    g.max_pixels = (int64_t) mx;

    if (g.hidden <= 0 || g.heads <= 0 || g.hidden % g.heads != 0)
        return "the vision shard's embedding_length is not a multiple of its head_count";
    g.head_dim = g.hidden / g.heads;
    if (g.merge != 2)
        return "this tower implements spatial_merge_size 2; the shard declares " + std::to_string(g.merge);
    if (g.patch <= 0 || image_size % g.patch != 0)
        return "the vision shard's image_size is not a multiple of its patch_size";
    g.pos_edge = (int32_t) (image_size / g.patch);
    if (g.out_hidden != trunk_hidden)
        return "the vision shard projects to " + std::to_string(g.out_hidden) +
               " but the trunk's hidden size is " + std::to_string(trunk_hidden);
    if (g.min_pixels <= 0 || g.max_pixels < g.min_pixels)
        return "the vision shard's pixel bounds are not a range";

    // THE CLS-ROW CHECK. The reference's position table carries an unused CLS row that the converter drops,
    // so the grid here is exactly pos_edge^2 rows. A shard that still had the CLS row would be read one row
    // off for every patch - an error that produces plausible embeddings, not a failure - so it is refused.
    const gguf::Tensor* pe = f.find("v.position_embd.weight");
    if (pe == nullptr) return "the vision shard lacks v.position_embd.weight";
    const uint64_t rows = pe->shape.size() == 2 ? pe->shape[1] : 0;
    if (rows != (uint64_t) g.pos_edge * (uint64_t) g.pos_edge)
        return "v.position_embd.weight has " + std::to_string(rows) + " rows but the position grid is " +
               std::to_string(g.pos_edge) + "x" + std::to_string(g.pos_edge) + " = " +
               std::to_string((uint64_t) g.pos_edge * g.pos_edge) + "; a shard that still carries the "
               "unused CLS row would be read one row off for every patch";

    if (const std::string e = check_tensors(f, g); !e.empty()) return e;
    if (const std::string e = check_shapes(f, g); !e.empty()) return e;

    // THE IMAGE TOKEN. The mmproj carries no token id, so it comes from the contract, and what can be
    // checked cheaply here is checked: that the id is inside the trunk's vocabulary.
    //
    // The id's IDENTITY - that it names the image-pad placeholder and not some other token - is NOT
    // checked here, and the reason is worth knowing rather than rediscovering: the shared GGUF reader
    // keeps a 64-item sample of every metadata array by design ("the count is what matters"), so the token
    // text at index 120120 is not available without a second full parse of a 120,832-entry vocabulary that
    // nothing else in the engine needs. The identity is asserted instead against the pack's
    // tokenizer/vocab.json, where the text is present, by a fast test
    // (tmp/tests/hy4_flash/vision/test_image_token.py). Both halves matter: a wrong id does not fail, it makes
    // the picture silently ignored.
    g.image_token = 120120;
    try {
        const strata::GgufFile trunk(trunk_shard1);
        const strata::MetaValue* toks = trunk.get("tokenizer.ggml.tokens");
        if (toks == nullptr || toks->count == 0) return "the trunk shard has no tokenizer.ggml.tokens";
        if ((uint64_t) g.image_token >= toks->count)
            return "the image token id " + std::to_string(g.image_token) + " is outside the trunk's " +
                   std::to_string(toks->count) + "-token vocabulary";
        // THE PIN. The identity check above lives in a test, against the pack's vocabulary; at run time
        // the engine cannot tell whether the pack in front of it is the one that test validated. A
        // different pack where 120120 names some other token clears the bound, loads, and drops pictures
        // in silence. Comparing the vocabulary's exact size - a value this reader does give exactly, even
        // though it keeps only a sample of the entries - does not prove identity, and is not claimed to:
        // it converts "some vocabulary was validated" into "this one is the size of the one that was".
        //
        // A legitimately different vocabulary trips this, and that trade is deliberate rather than
        // inherited: a refusal that names its reason is recovered by changing one constant here, while a
        // silently dropped picture is not diagnosable at all. Where the two failure modes are that
        // asymmetric, the loud one is the right default.
        if (toks->count != kTrunkVocabSize)
            return "the trunk's vocabulary has " + std::to_string(toks->count) + " tokens, not the " +
                   std::to_string(kTrunkVocabSize) + " this vision path's image token id was checked "
                   "against; refusing rather than placing image rows at positions that may name a "
                   "different token (which would drop the picture in silence). If this vocabulary is "
                   "correct, re-check the image token id and update kTrunkVocabSize in "
                   "engine/metal/src/hy4flash_gguf/vision.cpp";
    } catch (const std::exception& e) {
        return std::string("the trunk shard could not be read for the image-token check: ") + e.what();
    }

    // One no-copy wrap of the mapping gguf::File already made: page-aligned and rounded up by `open`.
    out.weights = cx.wrap(f.map, f.map_bytes);
    if (out.weights.handle == nullptr) return "the vision shard could not be wrapped as a Metal buffer";
    return {};
}

void close_vision(VisionShard& s, mtl::Context& cx) {
    if (s.weights.handle != nullptr) cx.free(s.weights);
    s.weights = mtl::Buffer();
}

std::string describe_vision(const VisionShard& s) {
    const VisionGeometry& g = s.geo;
    return "vision shard   " + std::to_string(g.layers) + " blocks, hidden " + std::to_string(g.hidden) +
           " = " + std::to_string(g.heads) + " x " + std::to_string(g.head_dim) + ", ffn " +
           std::to_string(g.ffn) + ", patch " + std::to_string(g.patch) + ", merge " +
           std::to_string(g.merge) + ", position grid " + std::to_string(g.pos_edge) + "^2, out " +
           std::to_string(g.out_hidden) + "; pixels " + std::to_string(g.min_pixels) + ".." +
           std::to_string(g.max_pixels) + ", image token " + std::to_string(g.image_token);
}

}  // namespace strata::hy4_flash
