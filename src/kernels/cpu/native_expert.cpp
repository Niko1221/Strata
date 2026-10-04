// src/kernels/cpu/native_expert.cpp - native (GGUF-form) expert rows on the CPU.
// IQ formats use ggml-cpu's quantizers and dot products; Strata's Hadamard-INT2 extension uses a scalar reference path.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/artifact/dequant.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <mutex>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

uint64_t splitmix(uint64_t seed, uint64_t index) {
    uint64_t z = seed + (index + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void had2_rotate(const float* src, float* dst, int64_t n, uint64_t seed) {
    constexpr float inv = 1.0f / 11.3137084989847603904f;  // 1 / sqrt(128)
    for (int64_t base = 0; base < n; base += 128) {
        float v[128];
        for (int i = 0; i < 128; ++i)
            v[i] = src[base + i] * ((splitmix(seed, (uint64_t) base + (uint64_t) i) & 1u) ? 1.0f : -1.0f);
        for (int stride = 1; stride < 128; stride <<= 1)
            for (int block = 0; block < 128; block += 2 * stride)
                for (int j = 0; j < stride; ++j) {
                    const float a = v[block + j], b = v[block + stride + j];
                    v[block + j] = a + b;
                    v[block + stride + j] = a - b;
                }
        for (int i = 0; i < 128; ++i) dst[base + i] = v[i] * inv;
    }
}

float had2_block_scale(const uint8_t* p) {
    return strata::fp16_to_fp32((uint16_t) (p[0] | ((uint16_t) p[1] << 8)));
}

float had2_level(const uint8_t* block, int lane) {
    static constexpr float levels[4] = {-1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f};
    const uint8_t q = block[2 + lane / 4];
    return levels[(q >> ((lane & 3) * 2)) & 3];
}

float had2_dot(const uint8_t* row, const float* x, int64_t width) {
    float sum = 0.0f;
    const int groups = (int) (width / 128);
    for (int g = 0; g < groups; ++g) {
        const uint8_t* block = row + (size_t) g * 34;
        const float scale = had2_block_scale(block);
        for (int i = 0; i < 128; ++i)
            sum += (scale * had2_level(block, i)) * x[(size_t) g * 128 + i];
    }
    return sum;
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err,
                uint64_t had2_seed) {
    if ((gu_type == 144) != (d_type == 144)) {
        err = "native experts: Hadamard-INT2 requires type 144 for gate/up and down in the same layer";
        return false;
    }
    const ggml_type_traits_cpu* tg = nullptr;
    const ggml_type_traits_cpu* td = nullptr;
    const ggml_type_traits_cpu* ag = nullptr;
    const ggml_type_traits_cpu* ad = nullptr;
    if (gu_type != 144) {
        init_once();
        tg = traits(gu_type);
        td = traits(d_type);
        if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
            err = "native experts: ggml-cpu has no dot product for type " +
                  std::to_string(tg && tg->vec_dot ? d_type : gu_type);
            return false;
        }
        ag = traits(tg->vec_dot_type);
        ad = traits(td->vec_dot_type);
        if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
            err = "native experts: ggml-cpu cannot quantize an activation for this layer";
            return false;
        }
    }
    if ((gu_type == 144 && n_embd % 128) || (d_type == 144 && n_ff % 128) ||
        (gu_type != 144 && (n_embd % ggml_blck_size((ggml_type) gu_type) || n_embd % ggml_blck_size(tg->vec_dot_type))) ||
        (d_type != 144 && (n_ff % ggml_blck_size((ggml_type) d_type) || n_ff % ggml_blck_size(td->vec_dot_type)))) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = gu_type == 144 ? -1 : (int) tg->vec_dot_type;
    f.d_act = d_type == 144 ? -1 : (int) td->vec_dot_type;
    f.had2_seed = had2_seed;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = gu_type == 144 ? (size_t) (n_embd / 128) * 34 : ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = d_type == 144 ? (size_t) (n_ff / 128) * 34 : ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = gu_type == 144 ? (size_t) n_embd * sizeof(float) : ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = d_type == 144 ? (size_t) n_ff * sizeof(float) : ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (f.gu_type == 144) { had2_rotate(x, (float*) dst, f.n_embd, f.had2_seed); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (f.d_type == 144) { had2_rotate(h, (float*) dst, f.n_ff, f.had2_seed); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    if (f.gu_type == 144) {
        const float* const* x = (const float* const*) act;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gr = blob + (size_t) r * f.gu_row;
            const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
            for (int t = 0; t < nt; ++t) {
                const float g = had2_dot(gr, x[t], f.n_embd);
                const float u = had2_dot(ur, x[t], f.n_embd);
                ff[t][r] = (g / (1.f + std::exp(-g))) * u;
            }
        }
        return;
    }
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    static const bool avx2 = std::getenv("STRATA_NO_IQ256") == nullptr;
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any group
    // size, no #152 rule).  Opt-in, STRATA_KQ256=1: measured no faster in the engine (a window's expert groups hold
    // ~1.4 tokens and the weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    if (kq && f.gu_type == 12 && nt >= 2) {   // one token: ggml's own dot below (the same bits, less overhead)
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu for
    // it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    if (nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    if (f.d_type == 144) {
        const float* const* x = (const float* const*) hq;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
            for (int t = 0; t < nt; ++t) out[t][r] = had2_dot(dr, x[t], f.n_ff);
        }
        return;
    }
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    if (kq && nt >= 2 && (f.d_type == 7 || f.d_type == 8)) {   // Q5_1 / Q8_0 down: bit-exact, any group size
        kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (nt >= mt_min && f.d_type == 20 && iq4nl_mt) {   // #152: the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
