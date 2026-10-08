// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <mutex>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

// The three opt-outs, as functions so that `native_rows_sliceable` and the two row functions ask the SAME
// question rather than repeating it.  See the note on the multi-token kernels in `native_gu_rows`.
bool iq512_on() {
    static const bool v = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    return v;
}
bool iq256_on() {
    static const bool v = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    return v;
}
bool kq256_on() {
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any
    // group size, no #152 rule).  Opt-in, STRATA_KQ256=1.
    static const bool v = [] { const char* e = std::getenv("STRATA_KQ256"); return cpu_avx2_ok() && e != nullptr && std::atoi(e) != 0; }();
    return v;
}

/// Which multi-token kernel `native_gu_rows` hands this group to: 5 = the AVX-512 i-quant one, 2 = the AVX-2
/// one, 0 = none, so ggml-cpu's per-token `vec_dot`.  Every one of those kernels addresses the ASSEMBLED blob
/// (`blob + up_off`), which is why a caller holding the three slices must not be given one; `native_rows_sliceable`
/// is this same answer for that caller.
int gu_multi_kind(int gu_type, int nt) {
    if (nt < native_gu_mt_min(gu_type)) return 0;
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps
    // ggml-cpu for it, as before (its rows would round differently).
    const bool cpu512 = cpu_avx512_ok();
    if (!(iq512_supported(gu_type) || (!cpu512 && iq256_supported(gu_type)))) return 0;
    if (iq512_on() && iq512_supported(gu_type)) return 5;
    if (iq256_on() && iq256_supported(gu_type)) return 2;
    return 0;
}

/// As `gu_multi_kind`, for the down rows: 8 = the Q5_1/Q8_0 kernel, 20 = IQ4_NL's, 0 = ggml-cpu's per-token dot.
int down_multi_kind(int d_type, int nt) {
    if (!cpu_avx2_ok()) return 0;   // both kernels below are /arch:AVX2 translation units
    if (kq256_on() && nt >= 2 && (d_type == 7 || d_type == 8)) return 8;
    const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    if (nt >= mt_min && d_type == 20 && iq4nl_mt) return 20;
    return 0;
}

}  // namespace

bool native_experts_available() noexcept { return true; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    if (n_ff > kMaxExpertFF) {
        err = "native experts: the expert intermediate is " + std::to_string(n_ff) + " rows, wider than the " +
              std::to_string(kMaxExpertFF) + " the pool's per-token buffers hold";
        return false;
    }
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or STRATA_FORCE_ISA=avx, keeps ggml's).  STRATA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
}  // namespace

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

int native_gu_mt_min(int gu_type) {
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const char* env = std::getenv("STRATA_IQ_MT_MIN");
    static const int mt_min = env ? std::atoi(env) : 2;
    // IQ3_S where the AVX-2 kernel gathers its grid (cpu_gather_fast, unless STRATA_IQ256_GATHER=0) on a CPU without
    // AVX-512: there that kernel beats ggml's dot for ONE token too (14900KF P-core, an expert's gate/up rows: 0.34 ->
    // 0.19 ms; its E-cores, which keep the scalar decode: 0.71 -> 0.68), and in decode most CPU experts serve one token
    // of the window.  So IQ3_S takes it for every group, and those rows no longer depend on the drafting.  A machine-
    // wide rule, not a per-core one: every core rounds an expert the same.  Every probe here is in expert_layout.cpp
    // and cpu_avx2_ok() comes first, so no AVX2 code runs before the check.  STRATA_IQ_MT_MIN set keeps its rule.
    // Opt-in (STRATA_IQ3S_MT1=1): it changes a lone token's IQ3_S rounding on those CPUs, so the default stays 0.1.39's.
    static const bool iq3s_one = env == nullptr && std::getenv("STRATA_IQ3S_MT1") != nullptr && cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr &&
                                 !cpu_avx512_ok() && iq256_gather_setting() != 0 && cpu_gather_fast();
    return iq3s_one && gu_type == 21 ? 1 : mt_min;
}

// The multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster at
// one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512 first,
// then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the AVX-2 kernel,
// STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with both set.
// Every one of them reads the ASSEMBLED blob's offsets, so `native_gu_rows_ptrs` below cannot use them and asks
// `native_rows_sliceable` first.  `gu_multi_kind` is where that question is answered once.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: bit-exact against ggml's per-token dot (any group size, no #152 rule).
    // Opt-in, STRATA_KQ256=1: measured no faster in the engine (a window's expert groups hold ~1.4 tokens and the
    // weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).  One token takes
    // ggml's own dot below - the same bits, less overhead.
    if (kq256_on() && f.gu_type == 12 && nt >= 2) {
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
        return;
    }
    // Each kernel only for the formats it implements: falling through an empty switch would leave ff unwritten
    // instead of falling back to ggml-cpu.
    if (const int kind = gu_multi_kind(f.gu_type, nt)) {
        if (kind == 5)
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
        else iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
        return;
    }
    // ggml-cpu's vec_dot, reached through the slice form: the blob's two halves ARE its gate and up rows.
    native_gu_rows_ptrs(f, blob, blob + f.up_off, act, nt, ff, r0, r1);
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    if (const int kind = down_multi_kind(f.d_type, nt)) {
        if (kind == 8) kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        else iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    native_down_rows_ptr(f, blob + f.down_off, hq, nt, out, r0, r1);
}

bool native_rows_sliceable(int gu_type, int d_type, int nt) {
    if (kq256_on() && gu_type == 12 && nt >= 2) return false;
    return gu_multi_kind(gu_type, nt) == 0 && down_multi_kind(d_type, nt) == 0;
}

// ================================ the multi-token kernels ON SLICES ================================
//
// **THE BLOB WAS NEVER THE POINT.**  `iq256_gu_rows` addresses the up row as `blob + up_off + r * gu_row`;
// nothing in it wants gate and up ADJACENT, only a fixed distance apart.  Two slices out of one mapping
// (`FileExpertSource::slices` returns three pointers into the same file) therefore ARE a blob, with `up_off`
// spelled `up - gate`.  The same holds for the down rows: `blob + down_off` is just `down`.
//
// So the choice `native_rows_sliceable` forces - assemble the blob, or drop to ggml-cpu's per-token dot - is a
// false one for a caller that already holds the slices.  A chunk wants the multi-token kernel precisely because
// it decodes each weight row ONCE for every token of the group, and the bytes a chunk saves are worthless
// without it: measured on glm5-next, `--prefill 128` read 4.5x fewer expert bytes than `--prefill 1` (1245.3 ->
// 276.6 GiB) and the pool took the SAME 30.4 s, because the per-token dot re-decodes every row per token and
// that decode, not the read, is the pool's limit.
//
// **THE ROUNDING IS THE PRICE, AND IT IS PAID IN THE CHUNK ONLY.**  These kernels differ from ggml-cpu's
// per-token dot by ~3e-8 relative, and glm5-next amplifies that: the geometry's `h` activation is quantized to
// q8_0 between the two projections, so a 3e-8 shift in a gate/up row occasionally crosses a rounding step and
// lands ~1% away.  Measured at the head, `--prefill 128` with these on differs from `--prefill 1` by up to 1.76
// on a logit.
//
// **THAT IS ENOUGH TO MOVE A TOKEN, AND ON A LONG ENOUGH PROMPT IT DOES.**  Over 64 greedy tokens on the short
// prompt this was first measured on, the text was identical at chunk 1, 8, 32, 128 and 512, which is why the
// switch is on by default.  It does not generalise: on a 344-token prompt `--prefill` 1, 128, 256 and 512 give
// four arms that each reproduce themselves byte for byte and disagree with each other from the first token.  The
// shape a token's dot takes is the number of tokens sharing its job, and a job's composition is the chunk's, so
// a chunk-dependent rounding survives into greedy output.  `STRATA_NO_SLICE_MT` goes back to the per-token dot,
// which reproduces `--prefill 1` bit for bit at every chunk size (verified at 1, 128 and 512 on that prompt) and
// costs 2.3x on the pool.  The card side is not implicated: `multi_exact` makes a batched column bitwise equal
// to the single-column call it replaced (`mmvq_multi_parity`).
//
// Decode is untouched either way: at `nt == 1` both halves fall through to the `_ptrs` functions below.
bool slice_mt_on() {
    static const bool v = std::getenv("STRATA_NO_SLICE_MT") == nullptr;
    return v;
}
/// The down rows' half, and it is OPT-IN where the gate/up half is not: `down_multi_kind` deliberately does not
/// name IQ4_XS (the parity test covers IQ4_NL and Q2_0 down rows, not IQ4_XS), so this reaches a kernel the
/// engine's own dispatch would not - and on this pack it is worth about 4%, measured, against the gate/up half's
/// 2.3x.  Off unless STRATA_SLICE_MT_DOWN is set.
bool slice_mt_down_on() {
    static const bool v = std::getenv("STRATA_SLICE_MT_DOWN") != nullptr;
    return v;
}

void native_gu_rows_slice(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act,
                          int nt, float* const* ff, int r0, int r1) {
    if (slice_mt_on()) {
        const size_t up_off = (size_t) (up - gate);
        if (kq256_on() && f.gu_type == 12 && nt >= 2) {
            kq256_gu_rows(f.gu_type, gate, f.gu_row, up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
            return;
        }
        if (const int kind = gu_multi_kind(f.gu_type, nt)) {
            if (kind == 5)
                iq512_gu_rows(f.gu_type, gate, f.gu_row, up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
            else
                iq256_gu_rows(f.gu_type, gate, f.gu_row, up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
            return;
        }
    }
    native_gu_rows_ptrs(f, gate, up, act, nt, ff, r0, r1);
}

void native_down_rows_slice(const NativeFmt& f, const uint8_t* down, const void* const* hq, int nt, float* const* out,
                            int r0, int r1) {
    if (slice_mt_down_on()) {
        if (const int kind = down_multi_kind(f.d_type, nt)) {
            if (kind == 8) kq256_rows(f.d_type, down, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
            else iq4nl256_down_rows(down, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
            return;
        }
        // `down_multi_kind` only names the two kernels someone wired to it (Q5_1/Q8_0 and IQ4_NL).  The row-dot
        // kernels cover IQ4_XS too, and a down row is half a glm5-next expert's bytes, so ask the general one.
        if (iq256_on() && nt >= 2 && iq256_supported(f.d_type)) {
            iq256_rows(f.d_type, down, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
            return;
        }
    }
    native_down_rows_ptr(f, down, hq, nt, out, r0, r1);
}

void native_gu_rows_ptrs(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act,
                         int nt, float* const* ff, int r0, int r1) {
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    // The reference's SwiGLU with a limit: **THE SILU'S OUTPUT IS CLAMPED, ABOVE ONLY** - not the raw gate, which
    // is what DEEPSEEK4's `ggml_swiglu_clamp` does and what this port computed for a while - and the UP IS
    // CLAMPED ON BOTH SIDES.  Both oracles in the engine's header note compute it this way.  `1e-6` is the
    // reference's own guard, so `f.swiglu_limit == 0` (the first family's pack) takes the same branch it always
    // took and this loop computes exactly its old arithmetic.
    const float lim = f.swiglu_limit;
    const bool clamp = lim > 1e-6f;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate + (size_t) r * f.gu_row;
        const uint8_t* ur = up + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            const float h = g / (1.f + std::exp(-g));
            ff[t][r] = (clamp ? std::fmin(h, lim) : h) * (clamp ? std::fmin(std::fmax(u, -lim), lim) : u);
        }
    }
}

void native_down_rows_ptr(const NativeFmt& f, const uint8_t* down, const void* const* hq, int nt, float* const* out,
                          int r0, int r1) {
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = down + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
