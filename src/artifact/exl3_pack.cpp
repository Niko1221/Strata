// src/artifact/exl3_pack.cpp - build the engine WeightTable from a turboderp EXL3 model (docs/EXL3.md).
#include "strata/artifact/exl3_pack.hpp"

#include "strata/artifact/exl3_model.hpp"
#include "strata/artifact/safetensors.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <cuda_runtime.h>

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
};

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
    {"hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", false, false, false, Form::F32, Xf::Id, false},
    {"hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", false, false, false, Form::F32, Xf::Id, false},
    {"ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", false, false, false, Form::BF16, Xf::Id, false},
};

const RoleSpec kGdn[] = {
    {"attn_qkv.weight", "linear_attn.in_proj_qkv", true, false, true, Form::F16, Xf::Id, false},
    {"attn_gate.weight", "linear_attn.in_proj_z", true, false, true, Form::F16, Xf::Id, false},
    {"ssm_out.weight", "linear_attn.out_proj", true, false, true, Form::F16, Xf::Id, false},
    {"ssm_conv1d.weight", "linear_attn.conv1d.weight", false, false, true, Form::F32, Xf::Conv1d, false},
    {"ssm_alpha.weight", "linear_attn.in_proj_a.weight", false, false, true, Form::BF16, Xf::Swap, false},
    {"ssm_beta.weight", "linear_attn.in_proj_b.weight", false, false, true, Form::BF16, Xf::Swap, false},
    {"ssm_a", "linear_attn.A_log", false, false, true, Form::F32, Xf::Id, false},
    {"ssm_dt.bias", "linear_attn.dt_bias", false, false, true, Form::F32, Xf::Id, false},
    {"ssm_norm.weight", "linear_attn.norm.weight", false, false, true, Form::F32, Xf::Id, false},
};

const RoleSpec kQsa[] = {
    {"attn_q.weight", "self_attn.q_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_k.weight", "self_attn.k_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_v.weight", "self_attn.v_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_output.weight", "self_attn.o_proj", true, true, false, Form::F16, Xf::Id, false},
    {"attn_q_norm.weight", "self_attn.q_norm.weight", false, true, false, Form::F32, Xf::Id, false},
    {"attn_k_norm.weight", "self_attn.k_norm.weight", false, true, false, Form::F32, Xf::Id, false},
    {"indexer.q_proj.weight", "self_attn.indexer.index_qk_proj", false, true, false, Form::BF16, Xf::SplitQ, false, true},
    {"indexer.k_proj.weight", "self_attn.indexer.index_qk_proj", false, true, false, Form::BF16, Xf::SplitK, false, true},
    {"indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", false, true, false, Form::F32, Xf::Id, false},
    {"indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", false, true, false, Form::F32, Xf::Id, false},
};

const RoleSpec kGlobal[] = {
    {"token_embd.weight", "model.language_model.embed_tokens.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"output.weight", "lm_head", true, false, false, Form::F16, Xf::Id, true},
    {"output_hc_norm.weight", "model.language_model.hyper_connection_mixer.hc_norm.weight", false, false, false, Form::F32, Xf::Id, true},
    {"output_hc_down.weight", "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"output_hc_up.weight", "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", false, false, false, Form::BF16, Xf::Swap, true},
};

const RoleSpec kPle[] = {
    {"blk.1.ple_key.weight", "model.language_model.layers.1.ple.key_proj.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"blk.1.ple_value.weight", "model.language_model.layers.1.ple.value_proj.weight", false, false, false, Form::BF16, Xf::Swap, true},
    {"blk.1.ple_norm_key.weight", "model.language_model.layers.1.ple.norm_key.weight", false, false, false, Form::F32, Xf::Id, true},
    {"blk.1.ple_norm_query.weight", "model.language_model.layers.1.ple.norm_query.weight", false, false, false, Form::F32, Xf::Id, true},
    {"blk.1.ple_norm_conv.weight", "model.language_model.layers.1.ple.norm_conv.weight", false, false, false, Form::F32, Xf::Id, true},
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
bool stage_role(Exl3Pack::Impl& im, const RoleSpec& r, const std::string& hf_name, const std::string& engine_name,
                uint8_t*& dst, WeightRef& out, std::string& err) {
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
        std::vector<float> eng((size_t) L.in() * oc);
        for (int i = 0; i < L.in(); ++i)
            for (int j = 0; j < oc; ++j) {
                const uint16_t h = w[(size_t) i * L.out() + c0 + j];
                eng[(size_t) i * oc + j] = f16_to_f32(h);
            }
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
bool stage_exl3(Exl3Pack::Impl& im, const RoleSpec& r, const std::string& hf_base, const std::string& engine_name,
                WeightTable& wt, std::string& err, uint64_t& bytes_out) {
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
        d_trellis = (uint16_t*) im.alloc((size_t) L.ki * L.nj * words * 2);
        d_suh = (uint16_t*) im.alloc((size_t) L.in() * 2);
        d_svh = (uint16_t*) im.alloc((size_t) L.out() * 2);
        (void) cudaMemcpy(d_trellis, L.trellis, (size_t) L.ki * L.nj * words * 2, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_suh, L.suh, (size_t) L.in() * 2, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_svh, L.svh, (size_t) L.out() * 2, cudaMemcpyHostToDevice);
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

bool Exl3Pack::build(WeightTable& wt, std::string& err, void* stream) {
    (void) stream;
    Impl& im = *impl_;
    try {
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
            if (!stage_role(im, *s.r, s.hf, s.engine, dst, w, err)) return false;
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

        // ---- EXL3 linears (attached, not staged)
        for (const RoleSpec& r : kGlobal)
            if (r.exl3 && !stage_exl3(im, r, r.hf, r.engine, wt, err, exl3_bytes_)) return false;
        for (int layer = 0; layer < 48; ++layer) {
            const bool qsa = (layer % 4) == 3;
            auto for_each = [&](const RoleSpec* arr, size_t n) {
                for (size_t i = 0; i < n; ++i) {
                    const RoleSpec& r = arr[i];
                    if (!r.exl3) continue;
                    if (r.qsa_only && !qsa) continue;
                    if (r.gdn_only && qsa) continue;
                    std::string eng = "blk." + std::to_string(layer) + "." + r.engine;
                    if (!stage_exl3(im, r, kLayerPrefix + std::to_string(layer) + "." + r.hf, eng, wt, err, exl3_bytes_))
                        return false;
                }
            };
            for_each(kCommon, sizeof kCommon / sizeof kCommon[0]);
            for_each(qsa ? kQsa : kGdn, qsa ? sizeof kQsa / sizeof kQsa[0] : sizeof kGdn / sizeof kGdn[0]);
        }

        // ---- validate against the engine's own shape contract
        ModelGeometry g;
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
