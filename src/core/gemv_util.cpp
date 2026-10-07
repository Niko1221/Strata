// src/core/gemv_util.cpp - see the header.  Moved out of `src/core/layer.cpp` when the second architecture's
// layer file needed the same rules; the comments below are the ones that were paid for there, kept with the
// code they explain rather than summarised at the new call site.
#include "gemv_util.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cstdio>
#include <exception>
#include <string>

namespace strata::core {
namespace gemv {

uint64_t q8k_bytes(int64_t n) { return (uint64_t) (n / Q8K_ELEMS_PER_BLOCK) * Q8K_BYTES_PER_BLOCK; }
/// `s_gemv_q8k` takes the canonical-form attributes; a `WeightRef` carries them, and a tensor that is NOT
/// quantized has none.  Returns false and names the tensor rather than building a form out of zeroes - which
/// would decode every code as `0 + bias` and produce a perfectly finite wrong answer.
bool sform_of(const WeightRef& r, strata::kernels::SForm& f, const std::string& name, std::string& err) {    if (!r.quantized()) {        err = name + " is not a quantized tensor, so it has no S-form";        return false;    }    f.code_bits = r.code_bits;    f.code_bias = r.code_bias;    f.group_elems = r.group_elems;    f.codebook = r.codebook_iq4nl ? strata::kernels::Codebook::Iq4Nl : strata::kernels::Codebook::Affine;    f.has_offset = r.has_offset;    f.act_kind = r.act_kind;
// carried, not derived - see the note on `SForm::act_kind`
return true;}
/// The three canonical planes of a quantized tensor, located INSIDE the loaded region.
//
//
// **THIS FUNCTION USED TO RE-DERIVE THE LAYOUT, AND THAT IS HOW A WIDTH BUG SURVIVED A ROUND.**  It computed
/// `n_groups = n_in / group_elems` and then `scales_bytes = n_out * n_groups * sizeof(float)` - 4 bytes per
/// scale for every tensor.  90 of the 303 quantized tensors hold fp16 scales in the pack (all 58 Q2_0 expert
/// tensors, plus Q4_0, Q5_0, Q8_0 and IQ4_NL), so for those the computed plane was TWICE the real one and the
/// three planes did not add up to the tensor.  The first diagnosis was "the manifest's `group_elems` must be
/// wrong, the real group is 64"; it is not, and an audit of all 303 tensors against `scales_fp16` found zero
/// inconsistencies.  **The unexamined input was the scale WIDTH, not the group COUNT.**
//
//
// The loader now widens fp16 scales to f32 on the way into the arena and records the resulting plane sizes in
/// the `WeightRef`, so this function has nothing left to derive - it reads what was loaded.  The redundancy is
/// deliberate: the sizes come from ONE place (the index, checked against the manifest by `pack_index.py` and
/// against the span by the loader), and this function only checks that they describe the tensor it was given.
//
//
// **THE PLANE KEY IS `offsets`, NOT `mins`, AND THAT COST 136 MiB.**  `tools/pack_index.py` and
/// `tools/pack_budget.py` both looked for `mins` and both therefore omitted the offset plane of 54 Q4_K/Q5_K
/// tensors - so the arena was sized without it AND the bytes were never copied.  Two tools agreeing is only
/// evidence when they do not share the assumption that is wrong.  This function refuses rather than guessing,
/// which is what caught it.
bool plane_ptrs(const WeightRef& r, const std::string& name, Planes& out, std::string& err) {
// S2, S4 AND S8 ALL SPLIT THE SAME WAY.  The plane LOCATION does not depend on the code width - the three
// sizes come from the index and are checked against the tensor below - so the guard is here to catch a
// tensor that is not quantized at all, not to pick a decoder.  WHICH KERNEL reads the planes is the
// caller's choice and the two differ: an S2 tensor's activation contract is Q8_0 (`s2_gemv_q8`) while a
// K-quant's is Q8_K (`s_gemv_q8k`).  This used to accept only 4 and 8, which refused `attn_q` - a Q2_0
// tensor and the reason the QSA layer could not be composed at all.
if (r.code_bits != 2 && r.code_bits != 4 && r.code_bits != 8) {        err = name + ": code_bits " + std::to_string(r.code_bits) + " is not an S2/S4/S8 form";        return false;    }    const uint64_t end = r.codes_bytes + r.scales_bytes + r.offset_bytes;    if (r.codes_bytes == 0 || r.scales_bytes == 0 || end != r.bytes) {        char buf[320];        std::snprintf(buf, sizeof buf,                      "%s: the planes add up to %llu B but the tensor is %llu B (codes %llu, scales %llu, "                      "offsets %llu) - what the loader recorded is not the layout that was loaded",                      name.c_str(), (unsigned long long) end, (unsigned long long) r.bytes,                      (unsigned long long) r.codes_bytes, (unsigned long long) r.scales_bytes,                      (unsigned long long) r.offset_bytes);        err = buf;        return false;    }    if (r.has_offset != (r.offset_bytes != 0)) {        err = name + ": has_offset is " + std::to_string(r.has_offset ? 1 : 0) + " but the offset plane is " +              std::to_string(r.offset_bytes) + " B";        return false;    }    const uint8_t* base = (const uint8_t*) r.data;    out.codes = base;    out.scales = (const float*) (base + r.codes_bytes);    out.offset = r.offset_bytes ? (const float*) (base + r.codes_bytes + r.scales_bytes) : nullptr;    return true;}
constexpr int TPR = 32;
bool native_bf16_projections = false;
bool native_flash_attn_short = false;

void project_bf16(const float* x, const uint16_t* x_bf16, const uint16_t* weights, float* out,
                  int64_t n_in, int64_t n_out, bool split, void* stream) {
    using namespace strata::kernels;
    if (native_bf16_projections) bf16_gemv_fp32_mmvf(x, weights, out, n_in, n_out, stream);
    else if (split) bf16_gemv_split(x_bf16, weights, out, n_in, n_out, TPR, stream);
    else bf16_gemv(x_bf16, weights, out, n_in, n_out, stream);
}
///< threads per row for the row-split GEMVs.
/// **MEASURED, NOT ASSUMED, AND 64 IS NOT BETTER.**  `dense_pass.exe` drives the same split kernels at
/// threads_per_row 64 and reports 248.5 GB/s, so matching it looked like free performance.  It is not:
/// TPR=64 gives 11.75 tok/s against 32's 11.99 with experts, and 20.19 against 20.26 without - identical
/// inside noise and marginally worse on both.  That is also what the stage table predicts, because the GEMVs
/// are only ~16% of a token, so a few percent there cannot move the total.  Kept at 32; the sweep that would
/// actually settle it is per-tensor, not global, and belongs with the fusion work rather than before it.
///< threads per row for the row-split GEMVs
/// Writes the sequence number into MAPPED PINNED memory.  A KERNEL, not cudaEventRecord - round 199 found
/// that an event record inside a capture is silently dropped, while this is captured normally and the host can
/// read its result MID-GRAPH.  One thread: it is a store.
// (the ring kernel lives in src/kernels/cuda/elementwise.cu; this file is HOST code)
/// WHICH ACTIVATION A QUANTIZED WEIGHT WANTS, AND IT IS A PROPERTY OF THE **TENSOR**, NOT OF ITS ROLE.
//
//
// Rounds 205-215 ran `gdn_layer` and `qsa_layer` on the assumption that a name implies a family: `attn_qkv` is
/// a K-quant, `attn_q` is Q2_0, and so on.  **The pack does not work that way.**  The same name is quantized
/// per LAYER, and the spread is wide:
//
//
//     attn_qkv     IQ4_XS x13, Q3_K x18, Q2_0 x1,  Q4_K x4      (36 GDN layers)
///     attn_gate    Q3_K x25,  IQ4_XS x4, Q4_K x3,  Q2_0 x4
///     attn_q       Q3_K x6,   IQ4_XS x4, Q2_0 x2                (12 QSA layers)
//
//
// `docs/activation-contract.md` fixes the rule per TYPE, and it is short:
//
//
//     code_bits == 2           ->  Q8_0,  via `quantize_q8_0` + `s2_gemv_q8`
///     anything else quantized  ->  Q8_K,  via `quantize_q8_K` + `s_gemv_q8k_split`
//
//
// Getting it wrong is worth 0.6-1.4% on the GEMV - a plausible vector, not a broken one - which is exactly
/// what Gate C1 exists to find and what nothing before C1 would have.  `s_gemv_q8k`'s own guard is what caught
/// it here (`code_bits 2 has no Q8_K contract`).
//
//
// `x80` and `xq8k` are the two quantized images of the SAME activation; a caller produces both once and this
/// picks.  Producing only the one it thinks it needs is how the assumption gets baked in again.
bool gemv_quantized(const WeightRef& w, const Planes& p, const strata::kernels::SForm& f, const uint8_t* x80, const uint8_t* xq8k,                    float* y, int64_t n_in, int64_t n_out, const std::string& name, void* stream,                    std::string& err, const float* x_f32, bool x_q8_1_ready, int ncols) {
    using namespace strata::kernels;
    if (ncols < 1 || ncols > NATIVE_MMVQ_MAX_NCOLS) {
        err = name + ": a projection may carry 1.." + std::to_string(NATIVE_MMVQ_MAX_NCOLS) + " columns, not " +
              std::to_string(ncols);
        return false;
    }
    if (w.native_data) {
        if (!x_f32 || !w.native_q8_1 || !stream || n_in != w.ne0 || n_out != w.ne1) {
            err = name + ": native projection requires matching FP32 input and session scratch";
            return false;
        }
        try {
            // Plan v0.3 P3: a caller whose previous native projection quantized the SAME x into the shared
            // scratch, with no native projection in between, passes x_q8_1_ready and the quantize is skipped.
            // The hand-off is only meaningful for one column at a time: a batched caller that changed `ncols`
            // is describing a different scratch CONTENT, so it must quantize.
            if (!x_q8_1_ready || ncols != 1) native_quantize_q8_1(x_f32, w.native_q8_1, (int) n_in, ncols, stream);
            native_mmvq(w.native_type, w.native_data, w.native_q8_1, y,
                        (int) n_in, (int) n_out, ncols, stream);
        } catch (const std::exception& error) {
            err = name + ": " + error.what();
            return false;
        }
        return true;
    }
    // THE CANONICAL PATH IS PER COLUMN.  Its kernels take one activation vector and the two image kinds have
    // fixed per-column strides (`q8k_bytes` / 34 bytes per 32), so a batch is the same call with the column
    // bases walked - there is nothing to fuse and nothing to get subtly wrong.  The native path is where the
    // batching pays, because there the weight read is what costs.
    if (ncols > 1) {
        const int64_t q8k_row = (int64_t) q8k_bytes(n_in);
        const int64_t q8_0_row = (n_in / 32) * 34;
        for (int64_t c = 0; c < ncols; ++c) {
            if (!gemv_quantized(w, p, f, x80 ? x80 + c * q8_0_row : nullptr, xq8k ? xq8k + c * q8k_row : nullptr,
                                y + c * n_out, n_in, n_out, name, stream, err, x_f32 ? x_f32 + c * n_in : nullptr,
                                false, 1)) {
                return false;
            }
        }
        return true;
    }
    if ((w.code_bits == 2 || !w.wants_q8k()) ? !x80 : !xq8k) {
        err = name + ": missing canonical quantized activation";
        return false;
    }
    if (w.code_bits == 2) {
// S2 carries no offset plane and no SForm: its attributes are fixed (2 bits, group 64, bias -1), which
// is why `s2_gemv_q8` takes neither.
if (w.has_offset || p.offset != nullptr) {            err = name + ": an S2 form must have no offset plane";            return false;        }        if (w.group_elems != 64 || w.code_bias != -1) {            err = name + ": an S2 form must be group 64 with bias -1, this one is group " +                  std::to_string(w.group_elems) + " with bias " + std::to_string(w.code_bias);            return false;        }        s2_gemv_q8(x80, p.codes, p.scales, y, n_in, n_out, TPR, stream);        return true;    }
// **A LEGACY 4/8-BIT FORM WANTS Q8_0, NOT Q8_K, AND `code_bits` CANNOT TELL YOU WHICH.**  This used to be
// `code_bits == 2 ? Q8_0 : Q8_K`, which is CORRECT ONLY BY LUCK: every legacy tensor in this pack apart
// from `ffn_down_shexp` is Q2_0, whose attributes happen to be S2's.  Q5_0 and Q5_K are both 8-bit with
// bias -16 and differ only in `has_offset`; IQ4_NL and IQ4_XS differ in NOTHING the S-form carries.  The
// kind therefore comes from the manifest's `source_type`, through the index (LEDGER L54/L55).
if (!w.wants_q8k()) {        s_gemv_q8_0_split(x80, p.codes, p.scales, p.offset, y, n_in, n_out, f, stream);        return true;    }    s_gemv_q8k_split(xq8k, p.codes, p.scales, p.offset, y, n_in, n_out, f, stream);    return true;}

bool project_rows(const WeightRef& w, const std::string& name, const float* x_f32, const uint8_t* x80,
                  const uint8_t* xq8k, float* y, int64_t n_in, int64_t row0, int64_t rows, void* stream,
                  std::string& err, int64_t ncols) {
    using namespace strata::kernels;
    if (rows <= 0) { err = name + ": no rows to project"; return false; }
    if (n_in != w.ne0 || row0 < 0 || row0 + rows > w.ne1) {
        char buf[224];
        std::snprintf(buf, sizeof buf,
                      "%s: rows [%lld, %lld) of a %lldx%lld weight, asked for an input of %lld",
                      name.c_str(), (long long) row0, (long long) (row0 + rows), (long long) w.ne0,
                      (long long) w.ne1, (long long) n_in);
        err = buf;
        return false;
    }
    WeightRef s = w;
    s.ne1 = rows;
    if (w.native_data != nullptr) {
        std::size_t row_bytes = 0;
        try {
            row_bytes = native_mmvq_weight_bytes(w.native_type, (int) n_in, 1);
        } catch (const std::exception& e) {
            err = name + ": " + e.what();
            return false;
        }
        s.native_data = (const uint8_t*) w.native_data + (std::size_t) row0 * row_bytes;
        // The planes are unused on this path and deliberately left empty rather than pointed at a base that
        // would be wrong if the weight ever stopped being served natively.
        return gemv_quantized(s, Planes{}, SForm{}, x80, xq8k, y, n_in, rows, name, stream, err, x_f32, false,
                              (int) ncols);
    }
    if (!w.quantized()) {
        err = name + ": a row-sliced projection is only defined for a quantized or native weight";
        return false;
    }
    const uint64_t ne1 = (uint64_t) w.ne1;
    if (w.codes_bytes % ne1 != 0 || w.scales_bytes % ne1 != 0 || (w.offset_bytes != 0 && w.offset_bytes % ne1 != 0)) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "%s: the planes do not divide into %llu rows (codes %llu, scales %llu, offsets %llu), so a "
                      "row band is not a fixed offset and this weight cannot be sliced",
                      name.c_str(), (unsigned long long) ne1, (unsigned long long) w.codes_bytes,
                      (unsigned long long) w.scales_bytes, (unsigned long long) w.offset_bytes);
        err = buf;
        return false;
    }
    const uint8_t* base = (const uint8_t*) w.data;
    const uint64_t cr = w.codes_bytes / ne1, sr = w.scales_bytes / ne1, orr = w.offset_bytes / ne1;
    Planes p;
    p.codes = base + (uint64_t) row0 * cr;
    p.scales = (const float*) (base + w.codes_bytes + (uint64_t) row0 * sr);
    p.offset = w.offset_bytes ? (const float*) (base + w.codes_bytes + w.scales_bytes + (uint64_t) row0 * orr)
                              : nullptr;
    SForm f;
    if (!sform_of(s, f, name, err)) return false;
    s.codes_bytes = (uint64_t) rows * cr;
    s.scales_bytes = (uint64_t) rows * sr;
    s.offset_bytes = (uint64_t) rows * orr;
    return gemv_quantized(s, p, f, x80, xq8k, y, n_in, rows, name, stream, err, x_f32, false, (int) ncols);
}

}  // namespace gemv
}  // namespace strata::core
