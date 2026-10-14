// src/artifact/exl3_pack.cpp - build the engine WeightTable from a turboderp EXL3 model (docs/EXL3.md).
#include "strata/artifact/exl3_pack.hpp"

#include "strata/artifact/exl3_model.hpp"
#include "strata/artifact/safetensors.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <ggml.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {
namespace {

enum class Form { F32, BF16, F16 };
enum class Xf { Id, T, Swap, Conv1d, SplitQ, SplitK };

/// The v-head reorder of `llama.cpp`'s converter (`conversion/qwen.py`, `_LinearAttentionVReorderBase`).
///
/// HF stores the v heads grouped by k head, `[G0_v0 G0_v1 G0_v2 G1_v0 ...]`; the engine's GDN kernels pair
/// v head `h` with k head `h % n_k` (`gdn_step_kernel`'s `src = h % h_k`), which is the tiled order
/// `[G0_v0 G1_v0 ... G15_v0 G0_v1 ...]`.  The GGUF converter reorders, so every GDN weight the engine has
/// ever read is tiled; an EXL3 model is a straight quantization of the HF checkpoint, so this loader has to
/// do the same or every GDN layer pairs q/k/v from the wrong heads.
enum class Vr {
    None,   ///< no reorder
    Out,    ///< the whole output dim is v: rows of the projection, or the 1-D A/dt parameters
    OutV,   ///< only the v part of the output dim (q and k stay as stored): in_proj_qkv
    In,     ///< the input dim is v: out_proj's columns
    Conv,   ///< the v channels of the conv kernel, after the q|k channels
};

/// The value transforms the converter applies on top of the name mapping.
enum class Vt {
    None,
    Norm1,    ///< zero-centred gamma: HF stores `w` and the engine's RMSNorm reads `1 + w`
    NegExp,   ///< `A_log` -> `-exp(A_log)`, the decay the GDN gate multiplies the softplus by
};

// One engine role.  `hf` is the module path under `model.language_model.layers.<L>.` unless `global`.
struct RoleSpec {
    const char* engine;   // "{L}" for the layer; else a fixed engine name (blk.1.ple_*)
    const char* hf;
    bool exl3;
    bool qsa_only;
    bool gdn_only;
    Form form;
    Xf xf;
    bool global;
    bool deq = false;   // reconstruct an EXL3 linear and store it as BF16 (the indexer, read directly)
    Vr vr = Vr::None;
    Vt vt = Vt::None;
};

/// Grouped -> tiled: tiled v head `t` is stored at grouped head `(t % n_k) * v_per_k + t / n_k`.
int64_t v_src_head(int64_t t, int64_t n_k, int64_t v_per_k) { return (t % n_k) * v_per_k + t / n_k; }

/// Reorder `n_v` head blocks of `block` elements each, starting at element `off`, from grouped to tiled.
template <typename T>
void reorder_v_blocks(T* p, int64_t off, int64_t n_v, int64_t block, int64_t n_k, int64_t v_per_k) {
    std::vector<T> tmp(p + off, p + off + n_v * block);
    for (int64_t t = 0; t < n_v; ++t)
        std::memcpy(p + off + t * block, tmp.data() + v_src_head(t, n_k, v_per_k) * block, block * sizeof(T));
}

const RoleSpec kCommon[] = {
    {"hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight", false, false, false, Form::BF16, Xf::Swap, false},
    {"hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight", false, false, false, Form::BF16, Xf::Swap, false},
    {"hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", false, false, false, Form::BF16, Xf::Swap, false},
    {"hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight", false, false, false, Form::BF16, Xf::Swap, false},
    {"hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight", false, false, false, Form::BF16, Xf::Swap, false},
    {"hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight", false, false, false, Form::BF16, Xf::Swap, false},
    {"ffn_gate_inp.weight", "mlp.gate.weight", false, false, false, Form::BF16, Xf::Swap, false},
    {"ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj", true, false, false, Form::F16, Xf::Id, false},
    {"ffn_up_shexp.weight", "mlp.shared_expert.up_proj", true, false, false, Form::F16, Xf::Id, false},
    {"ffn_down_shexp.weight", "mlp.shared_expert.down_proj", true, false, false, Form::F16, Xf::Id, false},
    {"hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", false, false, false, Form::F32, Xf::Id, false, false, Vr::None, Vt::Norm1},
    {"hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", false, false, false, Form::F32, Xf::Id, false, false, Vr::None, Vt::Norm1},
    {"ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", false, false, false, Form::BF16, Xf::Id, false},
};

const RoleSpec kGdn[] = {
    {"attn_qkv.weight", "linear_attn.in_proj_qkv", true, false, true, Form::F16, Xf::Id, false, false, Vr::OutV},
    {"attn_gate.weight", "linear_attn.in_proj_z", true, false, true, Form::F16, Xf::Id, false, false, Vr::Out},
    {"ssm_out.weight", "linear_attn.out_proj", true, false, true, Form::F16, Xf::Id, false, false, Vr::In},
    {"ssm_conv1d.weight", "linear_attn.conv1d.weight", false, false, true, Form::F32, Xf::Conv1d, false, false, Vr::Conv},
    {"ssm_alpha.weight", "linear_attn.in_proj_a.weight", false, false, true, Form::BF16, Xf::Swap, false, false, Vr::Out},
    {"ssm_beta.weight", "linear_attn.in_proj_b.weight", false, false, true, Form::BF16, Xf::Swap, false, false, Vr::Out},
    {"ssm_a", "linear_attn.A_log", false, false, true, Form::F32, Xf::Id, false, false, Vr::Out, Vt::NegExp},
    {"ssm_dt.bias", "linear_attn.dt_bias", false, false, true, Form::F32, Xf::Id, false, false, Vr::Out},
    {"ssm_norm.weight", "linear_attn.norm.weight", false, false, true, Form::F32, Xf::Id, false},
};

const RoleSpec kQsa[] = {
    {"attn_q.weight", "self_attn.q_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_k.weight", "self_attn.k_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_v.weight", "self_attn.v_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_output.weight", "self_attn.o_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_q_norm.weight", "self_attn.q_norm.weight", false, true, false, Form::F32, Xf::Id, false, false, Vr::None, Vt::Norm1},
    {"attn_k_norm.weight", "self_attn.k_norm.weight", false, true, false, Form::F32, Xf::Id, false, false, Vr::None, Vt::Norm1},
    {"indexer.q_proj.weight", "self_attn.indexer.index_qk_proj", false, true, false, Form::BF16, Xf::SplitQ, false, true},
    {"indexer.k_proj.weight", "self_attn.indexer.index_qk_proj", false, true, false, Form::BF16, Xf::SplitK, false, true},
    {"indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", false, true, false, Form::F32, Xf::Id, false, false, Vr::None, Vt::Norm1},
    {"indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", false, true, false, Form::F32, Xf::Id, false, false, Vr::None, Vt::Norm1},
};

const RoleSpec kGlobal[] = {
    {"token_embd.weight", "model.language_model.embed_tokens.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"output.weight", "lm_head", true, false, false, Form::F16, Xf::Id, true},
    {"output_hc_norm.weight", "model.language_model.hyper_connection_mixer.hc_norm.weight", false, false, false, Form::F32, Xf::Id, true, false, Vr::None, Vt::Norm1},
    {"output_hc_down.weight", "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"output_hc_up.weight", "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", false, false, false, Form::BF16, Xf::Swap, true},
};

const RoleSpec kPle[] = {
    {"blk.1.ple_key.weight", "model.language_model.layers.1.ple.key_proj.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"blk.1.ple_value.weight", "model.language_model.layers.1.ple.value_proj.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"blk.1.ple_norm_key.weight", "model.language_model.layers.1.ple.norm_key.weight", false, false, false, Form::F32, Xf::Id, true, false, Vr::None, Vt::Norm1},
    {"blk.1.ple_norm_query.weight", "model.language_model.layers.1.ple.norm_query.weight", false, false, false, Form::F32, Xf::Id, true, false, Vr::None, Vt::Norm1},
    {"blk.1.ple_norm_conv.weight", "model.language_model.layers.1.ple.norm_conv.weight", false, false, false, Form::F32, Xf::Id, true, false, Vr::None, Vt::Norm1},
    {"blk.1.ple_conv1d.weight", "model.language_model.layers.1.ple.conv1d.weight", false, false, false, Form::F16, Xf::Conv1d, true},
};

const char* kLayerPrefix = "model.language_model.layers.";

float bf16_to_f32(uint16_t h) {
    uint32_t f = (uint32_t) h << 16;
    float o; std::memcpy(&o, &f, 4); return o;
}
float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) { if (!man) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); } }
    else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float o; std::memcpy(&o, &f, 4); return o;
}
uint16_t f32_to_bf16(float v) {
    uint32_t f; std::memcpy(&f, &v, 4);
    uint32_t r = (f + 0x8000u) & 0xFFFF0000u;   // round-to-nearest
    return (uint16_t)(r >> 16);
}

}  // namespace

struct Exl3Pack::Impl {
    std::string dir;
    Exl3Model model;
    std::map<std::string, std::string> weight_map;
    std::map<std::string, std::unique_ptr<SafetensorsFile>> shards;
    std::vector<void*> dev_allocs;   // Exl3Mat uploads + arena

    explicit Impl(const std::string& d) : dir(d), model(d) {
        std::ifstream in(d + "/model.safetensors.index.json", std::ios::binary);
        std::ostringstream ss; ss << in.rdbuf();
        weight_map = exl3_detail::parse_weight_map(ss.str());
    }
    ~Impl() { for (void* p : dev_allocs) (void) cudaFree(p); }

    const SafetensorsFile& shard(const std::string& name) {
        auto it = shards.find(name);
        if (it != shards.end()) return *it->second;
        auto f = std::make_unique<SafetensorsFile>(dir + "/" + name);
        auto* raw = f.get();
        shards.emplace(name, std::move(f));
        return *raw;
    }
    const StTensor* raw(const std::string& name) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) return nullptr;
        return shard(it->second).find(name);
    }
    const uint8_t* raw_data(const std::string& name, const StTensor** shape_out) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) return nullptr;
        const SafetensorsFile& sf = shard(it->second);
        const StTensor* t = sf.find(name);
        if (!t) return nullptr;
        if (shape_out) *shape_out = t;
        return sf.data(*t);
    }

    void* alloc(size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) throw std::runtime_error("exl3: cudaMalloc failed");
        dev_allocs.push_back(p);
        return p;
    }

    // Read one HF tensor as float (handles F16/F32/BF16).  `n` elements.
    void read_f32(const std::string& name, std::vector<float>& out) {
        const StTensor* t = nullptr;
        const uint8_t* p = raw_data(name, &t);
        if (!p) throw std::runtime_error("exl3: no tensor " + name);
        const size_t n = t->nelem();
        out.resize(n);
        if (t->dtype == "F32") { std::memcpy(out.data(), p, n * 4); }
        else if (t->dtype == "F16" || t->dtype == "BF16") {
            const uint16_t* h = (const uint16_t*) p;
            const bool bf = t->dtype == "BF16";
            for (size_t i = 0; i < n; ++i) out[i] = bf ? bf16_to_f32(h[i]) : f16_to_f32(h[i]);
        } else throw std::runtime_error("exl3: unsupported dtype " + t->dtype + " for " + name);
    }
};

Exl3Pack::Exl3Pack(const std::string& dir) : impl_(new Impl(dir)) {}
Exl3Pack::~Exl3Pack() = default;

namespace {

// Stage one non-EXL3 role into `dst` (arena cursor) in engine form; returns the WeightRef and advances dst.
bool stage_role(Exl3Pack::Impl& im, const RoleSpec& r, const ModelGeometry& g, const std::string& hf_name,
                const std::string& engine_name, uint8_t*& dst, WeightRef& out, std::string& err) {
    int64_t out_ne0 = 0, out_ne1 = 0;
    if (r.deq) {
        // the engine reads the indexer projections directly as BF16; reconstruct the EXL3 index_qk_proj
        // (out = q(512) | k(128)) and split it into the two engine [in, out] matrices.
        Exl3Linear L = im.model.linear(hf_name);
        std::vector<uint16_t> w((size_t) L.in() * L.out());
        strata::kernels::cpu::exl3_reconstruct_weight(L.trellis, L.ki, L.nj, L.bits,
                                                      (strata::kernels::cpu::Exl3Codebook) L.cb, L.suh, L.svh, w.data());
        const bool q = r.xf == Xf::SplitQ;
        const int oc = q ? L.out() * 4 / 5 : L.out() / 5;   // 512 / 128
        const int c0 = q ? 0 : oc * 4;
        // `bf16_gemv` walks a row of `n_in` per output (`row = w + o * n_in`), so the engine form is
        // [out][in]: row j is HF's output j, the input index contiguous inside it.  The reconstruct gives
        // [in][out], so this is where the transpose is taken - a [in][out] buffer reads as the transpose
        // and nothing faults.
        std::vector<float> eng((size_t) L.in() * oc);
        for (int i = 0; i < L.in(); ++i)
            for (int j = 0; j < oc; ++j)
                eng[(size_t) j * L.in() + i] = f16_to_f32(w[(size_t) i * L.out() + c0 + j]);
        out_ne0 = L.in(); out_ne1 = oc;
        const int64_t elems = (int64_t) eng.size();
        const size_t bytes = (size_t) elems * 2;
        for (int64_t i = 0; i < elems; ++i) ((uint16_t*) dst)[i] = f32_to_bf16(eng[i]);
        out.data = dst; out.bytes = bytes; out.ne0 = out_ne0; out.ne1 = out_ne1; out.elements = elems;
        out.kind = WeightKind::Bf16InF32;
        dst += (bytes + 255) & ~(size_t) 255;
        return true;
    }
    std::vector<float> f;
    try { im.read_f32(hf_name, f); } catch (const std::exception& e) { err = e.what(); return false; }
    // the converter's value transforms (elementwise, so before the shape handling)
    if (r.vt == Vt::Norm1) for (float& x : f) x += 1.0f;
    else if (r.vt == Vt::NegExp) for (float& x : f) x = -std::exp(x);
    const StTensor* t = nullptr;
    im.raw_data(hf_name, &t);
    const std::vector<int64_t>& sh = t->shape;
    // engine shape + element order
    int64_t ne0 = 0, ne1 = 0;
    std::vector<float> eng;
    if (r.xf == Xf::Swap) {
        // HF [a, b] is already [ne1=a][ne0=b] with b contiguous, which is exactly the engine's convention
        // (ne0 the contiguous axis).  Swap the shape, keep the data.
        ne0 = sh[1]; ne1 = sh[0];
        eng = f;
    } else if (r.xf == Xf::Conv1d) {
        // HF [C, 1, K] flattens to [c][k] with k contiguous, which is exactly the engine's ne0=K (the
        // CONTIGUOUS axis is the kernel).  No reorder.
        const int64_t C = sh[0], K = sh[sh.size() - 1];
        ne0 = K; ne1 = C;
        eng = f;
    } else {   // Id
        ne0 = sh[0]; ne1 = sh.size() > 1 ? sh[1] : 0;
        eng = f;
    }
    const int64_t elems = (int64_t) eng.size();
    // the v-head reorder, on the staged values: `Vr::Out` blocks are one head of the output dim (a whole
    // row of the [n_v, n_embd] gates, one element of the 1-D A/dt), `Vr::Conv` is one head of the v
    // channels, after the q|k ones.
    if (r.vr != Vr::None) {
        const int64_t n_k = g.ssm_k_heads, n_v = g.ssm_v_heads, v_per_k = n_v / n_k;
        int64_t off = 0, block = 0;
        if (r.vr == Vr::Out) {
            block = elems / n_v;
        } else if (r.vr == Vr::Conv) {
            const int64_t K = elems / g.ssm_conv_channels;
            block = (g.ssm_value_dim / n_v) * K;
            off = (g.ssm_conv_channels - g.ssm_value_dim) * K;
        } else {
            err = engine_name + ": this role's v reorder is EXL3-only";
            return false;
        }
        if (block <= 0 || elems != off + n_v * block) {
            err = engine_name + ": the v reorder does not tile the staged tensor";
            return false;
        }
        reorder_v_blocks(eng.data(), off, n_v, block, n_k, v_per_k);
    }
    size_t bytes;
    if (r.form == Form::F32) {
        bytes = (size_t) elems * 4;
        std::memcpy(dst, eng.data(), bytes);
    } else if (r.form == Form::BF16) {
        bytes = (size_t) elems * 2;
        for (int64_t i = 0; i < elems; ++i) ((uint16_t*) dst)[i] = f32_to_bf16(eng[i]);
    } else {   // F16
        bytes = (size_t) elems * 2;
        for (int64_t i = 0; i < elems; ++i) {
            // f32 -> f16 (round to nearest even)
            uint32_t b; std::memcpy(&b, &eng[i], 4);
            uint32_t sign = (b >> 16) & 0x8000u, m = b & 0x7FFFFFu; int e = (int)((b >> 23) & 0xFF) - 127 + 15;
            uint16_t hh;
            if (e <= 0) hh = (uint16_t) sign;
            else if (e >= 0x1F) hh = (uint16_t)(sign | 0x7C00u);
            else hh = (uint16_t)(sign | (e << 10) | (m >> 13));
            ((uint16_t*) dst)[i] = hh;
        }
    }
    out.data = dst;
    out.bytes = bytes;
    out.ne0 = ne0;
    out.ne1 = ne1;
    out.elements = elems;
    out.kind = r.form == Form::F32 ? WeightKind::F32 : (r.form == Form::BF16 ? WeightKind::Bf16InF32 : WeightKind::F16InF32);
    dst += (bytes + 255) & ~(size_t) 255;
    (void) engine_name;
    return true;
}

// Upload one EXL3 linear and attach an Exl3Mat.
bool stage_exl3(Exl3Pack::Impl& im, const RoleSpec& r, const ModelGeometry& g, const std::string& hf_base,
                const std::string& engine_name, WeightTable& wt, std::string& err, uint64_t& bytes_out) {
    if (!im.model.has(hf_base)) { err = "exl3: no EXL3 linear " + hf_base; return false; }
    Exl3Linear L;
    try { L = im.model.linear(hf_base); } catch (const std::exception& e) { err = e.what(); return false; }
    const int words = 256 * L.bits / 16;
    int nj = L.nj;
    const uint16_t* svh = L.svh;
    uint16_t* d_trellis = nullptr; uint16_t* d_suh = nullptr; uint16_t* d_svh = nullptr;
    if (r.xf == Xf::SplitQ || r.xf == Xf::SplitK) {
        // index_qk_proj: out 640 = q(512, nj=32) | k(128, nj=8).  Repack the trellis (stride nj changes).
        const bool q = r.xf == Xf::SplitQ;
        const int nj_new = q ? L.nj * 4 / 5 : L.nj / 5;   // 40 -> 32 / 8
        const int j0 = q ? 0 : nj_new * 4;
        d_trellis = (uint16_t*) im.alloc((size_t) L.ki * nj_new * words * 2);
        std::vector<uint16_t> tmp((size_t) L.ki * nj_new * words);
        for (int i = 0; i < L.ki; ++i)
            for (int j = 0; j < nj_new; ++j)
                std::memcpy(&tmp[((size_t) i * nj_new + j) * words], L.trellis + ((size_t) i * L.nj + j0 + j) * words, (size_t) words * 2);
        (void) cudaMemcpy(d_trellis, tmp.data(), tmp.size() * 2, cudaMemcpyHostToDevice);
        d_suh = (uint16_t*) im.alloc((size_t) L.in() * 2);
        (void) cudaMemcpy(d_suh, L.suh, (size_t) L.in() * 2, cudaMemcpyHostToDevice);
        d_svh = (uint16_t*) im.alloc((size_t) nj_new * 16 * 2);
        (void) cudaMemcpy(d_svh, svh + (size_t) j0 * 16, (size_t) nj_new * 16 * 2, cudaMemcpyHostToDevice);
        nj = nj_new;
        svh = d_svh;
    } else {
        // host copies, permuted where the converter reorders: the trellis carries the weights, so the
        // reorder is done on it, at whole 16-row tiles (a v head is 128 = 8 tiles, so a head boundary is
        // always a tile boundary).
        std::vector<uint16_t> h_trellis(L.trellis, L.trellis + (size_t) L.ki * L.nj * words);
        std::vector<uint16_t> h_suh(L.suh, L.suh + (size_t) L.in());
        std::vector<uint16_t> h_svh(L.svh, L.svh + (size_t) L.out());
        const int64_t n_k = g.ssm_k_heads, n_v = g.ssm_v_heads, v_per_k = n_v / n_k;
        const int64_t tiles_per_head = g.ssm_state_size / 16;
        if (r.vr == Vr::Out || r.vr == Vr::OutV) {
            // the output dim: `Vr::OutV` keeps q and k as stored and moves only the v tiles after them
            const int64_t j0 = r.vr == Vr::OutV ? (g.ssm_conv_channels - g.ssm_value_dim) / 16 : 0;
            for (int i = 0; i < L.ki; ++i)
                reorder_v_blocks(h_trellis.data() + (size_t) i * L.nj * words, j0 * words, n_v,
                                 tiles_per_head * words, n_k, v_per_k);
            reorder_v_blocks(h_svh.data(), j0 * 16, n_v, g.ssm_state_size, n_k, v_per_k);
        } else if (r.vr == Vr::In) {
            // the input dim: whole groups of `tiles_per_head` i-tiles, and the input scales
            reorder_v_blocks(h_trellis.data(), 0, n_v, tiles_per_head * (int64_t) L.nj * words, n_k, v_per_k);
            reorder_v_blocks(h_suh.data(), 0, n_v, g.ssm_state_size, n_k, v_per_k);
        } else if (r.vr != Vr::None) {
            err = engine_name + ": this role's v reorder is not implemented for an EXL3 linear";
            return false;
        }
        d_trellis = (uint16_t*) im.alloc((size_t) L.ki * L.nj * words * 2);
        d_suh = (uint16_t*) im.alloc((size_t) L.in() * 2);
        d_svh = (uint16_t*) im.alloc((size_t) L.out() * 2);
        (void) cudaMemcpy(d_trellis, h_trellis.data(), h_trellis.size() * 2, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_suh, h_suh.data(), h_suh.size() * 2, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_svh, h_svh.data(), h_svh.size() * 2, cudaMemcpyHostToDevice);
    }
    bytes_out += (uint64_t) L.ki * nj * words * 2 + (uint64_t) L.in() * 2 + (uint64_t) nj * 16 * 2;
    auto* m = new strata::kernels::Exl3Mat;
    m->trellis = d_trellis; m->suh = d_suh; m->svh = d_svh;
    m->ki = L.ki; m->nj = nj; m->bits = L.bits; m->cb = L.cb;
    WeightRef w;
    w.exl3 = m;
    w.ne0 = L.in();
    w.ne1 = nj * 16;
    w.elements = (int64_t) L.in() * nj * 16;
    w.kind = WeightKind::Verbatim;
    wt.set(engine_name, w);
    return true;
}

}  // namespace

// ================================ the dense linears: a ggml block quant, made at load ================================
//
// **A TRELLIS WEIGHT COSTS ABOUT TWO LOADS AND ELEVEN ALU OPS TO DECODE; A GGML BLOCK QUANT'S `vec_dot` COSTS
// ABOUT ONE LOAD PER FOUR OR FIVE WEIGHTS.**  Measured on the W7800, a 2560x10240 dense linear: the EXL3
// decode-GEMV kernel 106.8 us against the native Q8_0 MMVQ kernel's 31.8 us, 3.4x.  The dense linears are
// 2117 MiB of the model; as Q8_0 they are ~3.7 GiB, which the expert cache pays for (measured: 24.57 GiB free
// -> 289 slots/layer against 311 at 5 bits), and the dense GEMV is still ~3x faster.
//
// **Q8_0, NOT A 5-BIT BLOCK QUANT.**  The trellis decode's values are ALREADY quantized (a 10-bit code per
// weight), and a 5-bit block quant on top of them measures a 4% round-trip error - enough to turn the model
// into "surface表面的表面的" (measured).  Q8_0's round-trip error is 0.65% (measured, the same way).
//
// **THE REQUANTIZATION STREAMS THROUGH SMALL BUFFERS.**  Holding a whole matrix on the host (7 GB of fp16 for
// the model, one tensor at a time) leaves that much anonymous dirty memory behind, and the kernel evicts the
// model's page cache to make room for it: measured, the expert reads then came off the SATA SSD - 1132 MiB of
// disk reads for 8 tokens, ~500 ms/token of uploads against 14-27 ms.  A row chunk per worker keeps the host
// footprint at a couple of MiB per worker and leaves the cache alone.
//
// **THE V-HEAD REORDER IS PART OF THE WEIGHT.**  The GDN's v projections are staged with their v heads permuted
// - `stage_exl3` reorders the trellis and the scales, and the activations are permuted to match.  A
// requantized copy has to carry the SAME permutation or its rows are in the wrong order: measured, every GDN
// v role (attn_qkv, attn_gate, ssm_out - `Vr::OutV`/`Out`/`In`) then answers nonsense ("surface表面的表面的"),
// while the QSA roles (`Vr::None`) are correct.  The host copies below are the same code `stage_exl3` runs, so
// the two paths cannot drift.
//
// The routed EXPERTS stay EXL3: 45 GB of trellis streamed per token, where the decode's cost is amortized
// against the read and nothing else could hold them.
bool stage_exl3_requantized(Exl3Pack::Impl& im, const RoleSpec& r, const ModelGeometry& g, const std::string& hf_base,
                            const std::string& engine_name, WeightTable& wt, std::string& err, uint64_t& bytes_out,
                            void* q8_scratch) {
    Exl3Linear L;
    try { L = im.model.linear(hf_base); } catch (const std::exception& e) { err = e.what(); return false; }
    const int n_in = L.in(), n_out = L.out();
    const int words = 256 * L.bits / 16;
    auto fail = [&](const char* what, cudaError_t e) {
        err = std::string("exl3: ") + engine_name + ": " + what + ": " + cudaGetErrorString(e);
        return false;
    };
    uint16_t *d_tr = nullptr, *d_suh = nullptr, *d_svh = nullptr, *d_w = nullptr, *d_wt = nullptr, *d_q = nullptr;
    auto alloc = [&](uint16_t** p, size_t bytes) {
        try { *p = (uint16_t*) im.alloc(bytes); } catch (const std::exception& e) { err = e.what(); return false; }
        return true;
    };
    if (!alloc(&d_tr, (size_t) L.ki * L.nj * words * 2) || !alloc(&d_suh, (size_t) n_in * 2) ||
        !alloc(&d_svh, (size_t) n_out * 2) || !alloc(&d_w, (size_t) n_in * n_out * 2) ||
        !alloc(&d_wt, (size_t) n_in * n_out * 2)) return false;
    const uint16_t* up_trellis = L.trellis;
    const uint16_t* up_suh = L.suh;
    const uint16_t* up_svh = L.svh;
    std::vector<uint16_t> h_trellis, h_suh, h_svh;
    if (r.vr != Vr::None) {
        h_trellis.assign(L.trellis, L.trellis + (size_t) L.ki * L.nj * words);
        h_suh.assign(L.suh, L.suh + (size_t) n_in);
        h_svh.assign(L.svh, L.svh + (size_t) n_out);
        const int64_t n_k = g.ssm_k_heads, n_v = g.ssm_v_heads, v_per_k = n_v / n_k;
        const int64_t tiles_per_head = g.ssm_state_size / 16;
        if (r.vr == Vr::Out || r.vr == Vr::OutV) {
            const int64_t j0 = r.vr == Vr::OutV ? (g.ssm_conv_channels - g.ssm_value_dim) / 16 : 0;
            for (int i = 0; i < L.ki; ++i)
                reorder_v_blocks(h_trellis.data() + (size_t) i * L.nj * words, j0 * words, n_v,
                                 tiles_per_head * words, n_k, v_per_k);
            reorder_v_blocks(h_svh.data(), j0 * 16, n_v, g.ssm_state_size, n_k, v_per_k);
        } else if (r.vr == Vr::In) {
            reorder_v_blocks(h_trellis.data(), 0, n_v, tiles_per_head * (int64_t) L.nj * words, n_k, v_per_k);
            reorder_v_blocks(h_suh.data(), 0, n_v, g.ssm_state_size, n_k, v_per_k);
        } else {
            err = "exl3: " + engine_name + ": this role's v reorder is not implemented for a requantized linear";
            return false;
        }
        up_trellis = h_trellis.data(); up_suh = h_suh.data(); up_svh = h_svh.data();
    }
    if (cudaMemcpy(d_tr, up_trellis, (size_t) L.ki * L.nj * words * 2, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("trellis upload", cudaGetLastError());
    if (cudaMemcpy(d_suh, up_suh, (size_t) n_in * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_svh, up_svh, (size_t) n_out * 2, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("scale upload", cudaGetLastError());
    strata::kernels::exl3_reconstruct_weight(d_tr, L.ki, L.nj, L.bits, L.cb, d_suh, d_svh, d_w, nullptr);
    // **THE RECONSTRUCT HANDS BACK [in][out]; A BLOCK QUANT STORES A ROW OF `n_in` PER OUTPUT.**  So the
    // weight is transposed on the GPU (a 32x32 tiled kernel) before it is downloaded and quantized.
    strata::kernels::exl3_transpose_f16(d_w, d_wt, n_in, n_out, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) return fail("reconstruct", cudaGetLastError());
    (void) cudaFree(d_tr); (void) cudaFree(d_suh); (void) cudaFree(d_svh);

    const ggml_type qtype = GGML_TYPE_Q8_0;
    const size_t row_bytes = ggml_row_size(qtype, n_in);
    if (!alloc(&d_q, row_bytes * (size_t) n_out)) return false;
    const int chunk = 32;
    std::atomic<int> next_row{0};
    const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const char* nw_env = std::getenv("STRATA_EXL3_QUANT_THREADS");
    const int n_workers = nw_env != nullptr ? std::max(1, std::atoi(nw_env)) : (int) std::min<unsigned>(hw_threads, 8u);
    std::string werr;
    auto worker = [&] {
        std::vector<uint16_t> h16((size_t) chunk * (size_t) n_in);
        std::vector<float> f32((size_t) chunk * (size_t) n_in);
        std::vector<uint8_t> q((size_t) chunk * row_bytes);
        for (;;) {
            const int r0 = next_row.fetch_add(chunk);
            if (r0 >= n_out) break;
            const int nr = std::min(chunk, n_out - r0);
            const size_t elems = (size_t) nr * (size_t) n_in;
            if (cudaMemcpy(h16.data(), d_wt + (size_t) r0 * (size_t) n_in, elems * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
                werr = "reconstruct download: " + std::string(cudaGetErrorString(cudaGetLastError()));
                return;
            }
            for (size_t i = 0; i < elems; ++i) f32[i] = f16_to_f32(h16[i]);
            // `ggml_quantize_chunk` offsets BOTH sides itself: `src + start` and `dst + start_row*row_size`.
            // So both pointers are shifted back by `start` and the function lands on this chunk's buffers.
            const float* src_base = f32.data() - (size_t) r0 * (size_t) n_in;
            uint8_t* dst_base = q.data() - (size_t) r0 * row_bytes;
            ggml_quantize_chunk(qtype, src_base, dst_base, (int64_t) r0 * n_in, nr, n_in, nullptr);
            if (std::getenv("STRATA_EXL3_Q5_CHECK") != nullptr && r0 == 0) {
                // round trip: dequantize what we just quantized and compare with the source
                std::vector<float> back(elems);
                ggml_get_type_traits(qtype)->to_float(q.data(), back.data(), (int64_t) elems);
                double num = 0, den = 0;
                for (size_t i = 0; i < elems; ++i) { const double d = (double) back[i] - f32[i]; num += d * d; den += (double) f32[i] * f32[i]; }
                std::fprintf(stderr, "exl3 q8 check %s: rel %.3e (first row, %zu elems)\n", engine_name.c_str(),
                             std::sqrt(num / (den + 1e-30)), elems);
            }
            if (cudaMemcpy((uint8_t*) d_q + (size_t) r0 * row_bytes, q.data(), (size_t) nr * row_bytes,
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                werr = "quantized upload: " + std::string(cudaGetErrorString(cudaGetLastError()));
                return;
            }
        }
    };
    {
        std::vector<std::thread> pool;
        pool.reserve((size_t) n_workers - 1);
        for (int i = 0; i < n_workers - 1; ++i) pool.emplace_back(worker);
        worker();
        for (std::thread& t : pool) t.join();
    }
    if (!werr.empty()) { err = "exl3: " + engine_name + ": " + werr; return false; }
    (void) cudaFree(d_wt);

    WeightRef w;
    w.native_data = d_q;
    w.native_type = (int) qtype;
    w.native_q8_1 = q8_scratch;
    w.ne0 = n_in;
    w.ne1 = n_out;
    w.elements = (int64_t) n_in * (int64_t) n_out;
    w.kind = WeightKind::Verbatim;
    wt.set(engine_name, w);
    bytes_out += row_bytes * (size_t) n_out;
    return true;
}

bool Exl3Pack::build(WeightTable& wt, std::string& err, void* stream) {
    (void) stream;
    Impl& im = *impl_;
    try {
        const ModelGeometry g;
        // ---- size the non-EXL3 arena by walking the roles once (all staged tensors), then upload.
        struct Staged { const RoleSpec* r; std::string hf, engine; };
        std::vector<Staged> staged;
        auto add = [&](const RoleSpec& r, const std::string& hf, const std::string& engine) {
            staged.push_back({&r, hf, engine});
        };
        for (const RoleSpec& r : kGlobal)
            if (!r.exl3) add(r, r.hf, r.engine);
        for (int layer = 0; layer < 48; ++layer) {
            const bool qsa = (layer % 4) == 3;
            auto for_each = [&](const RoleSpec* arr, size_t n) {
                for (size_t i = 0; i < n; ++i) {
                    const RoleSpec& r = arr[i];
                    if (r.qsa_only && !qsa) continue;
                    if (r.gdn_only && qsa) continue;
                    if (r.exl3) continue;
                    std::string eng = "blk." + std::to_string(layer) + "." + r.engine;
                    add(r, kLayerPrefix + std::to_string(layer) + "." + r.hf, eng);
                }
            };
            for_each(kCommon, sizeof kCommon / sizeof kCommon[0]);
            for_each(qsa ? kQsa : kGdn, qsa ? sizeof kQsa / sizeof kQsa[0] : sizeof kGdn / sizeof kGdn[0]);
        }
        for (const RoleSpec& r : kPle) if (!r.exl3) add(r, r.hf, r.engine);

        uint64_t need = 0;
        for (const Staged& s : staged) {
            if (s.r->deq) {   // reconstructed to BF16 [in, out]
                Exl3Linear L = im.model.linear(s.hf);
                const uint64_t oc = s.r->xf == Xf::SplitQ ? L.out() * 4 / 5 : L.out() / 5;
                const uint64_t bytes = (uint64_t) L.in() * oc * 2;
                need += (bytes + 255) & ~(uint64_t) 255;
                continue;
            }
            const StTensor* t = nullptr; im.raw_data(s.hf, &t);
            if (!t) { err = "exl3: missing tensor " + s.hf; return false; }
            const uint64_t bytes = t->nelem() * (s.r->form == Form::F32 ? 4 : 2);
            need += (bytes + 255) & ~(uint64_t) 255;
        }
        std::vector<uint8_t> host_arena((size_t) need);
        uint8_t* dst = host_arena.data();
        std::vector<std::pair<std::string, uint64_t>> placed;
        for (const Staged& s : staged) {
            WeightRef w;
            if (!stage_role(im, *s.r, g, s.hf, s.engine, dst, w, err)) return false;
            placed.emplace_back(s.engine, (uint64_t) ((const uint8_t*) w.data - host_arena.data()));
            w.data = nullptr;   // filled in after the upload
            wt.set(s.engine, w);
        }
        uint8_t* arena = (uint8_t*) im.alloc((size_t) need);
        arena_bytes_ = need;
        if (cudaMemcpy(arena, host_arena.data(), (size_t) need, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "exl3: arena upload failed"; return false;
        }
        for (const auto& p : placed) {
            WeightRef w = *wt.find(p.first);
            w.data = arena + p.second;
            wt.set(p.first, w);
        }

        // ---- the dense linears: requantized to a block quant and served by native_mmvq (see above).
        // STRATA_EXL3_DENSE=exl3 keeps the trellis decode instead; STRATA_EXL3_DENSE=only:<role>[,...]
        // requantizes just those roles (a bisect aid).  Both are A/B arms, and the way back if a machine
        // prefers the trellis.  The q8_1 scratch is shared: the projections are serialized, as in NativeDense.
        const char* dense_env = std::getenv("STRATA_EXL3_DENSE");
        const std::string dense_only = dense_env != nullptr && std::strncmp(dense_env, "only:", 5) == 0 ? dense_env + 5 : "";
        const bool dense_q5 = !(dense_env != nullptr && std::strcmp(dense_env, "exl3") == 0);
        auto use_q5 = [&](const RoleSpec& r) {
            if (!dense_q5 || r.xf != Xf::Id) return false;   // the split (indexer) roles keep the trellis
            if (dense_only.empty()) return true;
            return dense_only.find(r.engine) != std::string::npos;
        };
        void* q8_scratch = nullptr;
        {
            int64_t max_in = 0;
            for (const RoleSpec& r : kGlobal)
                if (r.exl3) { try { max_in = std::max(max_in, (int64_t) im.model.linear(r.hf).in()); } catch (...) {} }
            for (int layer = 0; layer < 48; ++layer) {
                const bool qsa = (layer % 4) == 3;
                auto scan = [&](const RoleSpec* arr, size_t n) {
                    for (size_t i = 0; i < n; ++i) {
                        const RoleSpec& r = arr[i];
                        if (!r.exl3 || (r.qsa_only && !qsa) || (r.gdn_only && qsa)) continue;
                        try { max_in = std::max(max_in, (int64_t) im.model.linear(kLayerPrefix + std::to_string(layer) + "." + r.hf).in()); } catch (...) {}
                    }
                };
                scan(kCommon, sizeof kCommon / sizeof kCommon[0]);
                scan(qsa ? kQsa : kGdn, qsa ? sizeof kQsa / sizeof kQsa[0] : sizeof kGdn / sizeof kGdn[0]);
            }
            q8_scratch = im.alloc((size_t) strata::kernels::native_q8_1_bytes((int) max_in));
            if (std::getenv("STRATA_EXL3_DEBUG") != nullptr)
                std::fprintf(stderr, "exl3 dense: q8_1 scratch for max_in=%lld (%zu bytes)\n", (long long) max_in,
                             (size_t) strata::kernels::native_q8_1_bytes((int) max_in));
        }
        for (const RoleSpec& r : kGlobal)
            if (r.exl3 && !(use_q5(r) ? stage_exl3_requantized(im, r, g, r.hf, r.engine, wt, err, exl3_bytes_, q8_scratch)
                                      : stage_exl3(im, r, g, r.hf, r.engine, wt, err, exl3_bytes_))) return false;
        for (int layer = 0; layer < 48; ++layer) {
            const bool qsa = (layer % 4) == 3;
            auto for_each = [&](const RoleSpec* arr, size_t n) -> bool {
                for (size_t i = 0; i < n; ++i) {
                    const RoleSpec& r = arr[i];
                    if (!r.exl3) continue;
                    if (r.qsa_only && !qsa) continue;
                    if (r.gdn_only && qsa) continue;
                    std::string eng = "blk." + std::to_string(layer) + "." + r.engine;
                    if (!(use_q5(r) ? stage_exl3_requantized(im, r, g, kLayerPrefix + std::to_string(layer) + "." + r.hf,
                                                             eng, wt, err, exl3_bytes_, q8_scratch)
                                    : stage_exl3(im, r, g, kLayerPrefix + std::to_string(layer) + "." + r.hf, eng, wt,
                                                 err, exl3_bytes_)))
                        return false;
                }
                return true;
            };
            if (!for_each(kCommon, sizeof kCommon / sizeof kCommon[0])) return false;
            if (!for_each(qsa ? kQsa : kGdn, qsa ? sizeof kQsa / sizeof kQsa[0] : sizeof kGdn / sizeof kGdn[0]))
                return false;
        }

        // ---- validate against the engine's own shape contract
        if (!check_all(wt, g, err)) return false;
        LoadReport rep;
        rep.tensors = wt.all().size();
        rep.arena_bytes = arena_bytes_ + exl3_bytes_;
        wt.finish(rep);
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3: ") + e.what();
        return false;
    }
}

}  // namespace strata::core
