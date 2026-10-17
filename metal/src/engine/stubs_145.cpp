// metal/src/engine/stubs_145.cpp - the 0.1.45 PC entry points the Metal engine does not implement (#2860): new functions
// and new overloads that stubs.cpp (generated) and stubs_144.cpp do not know.  Each either stops the engine with its name
// (like stubs.cpp) or, where the caller has a documented fallback, answers "not available".  None of the stopping ones is
// reached with the Metal defaults: the batch rows need batch_rec_ (no --batch on Metal, #2715), the layer-split hand-off
// and the mapped plan ring need a second GPU, the q8_1 interleave needs sm_75 or STRATA_QIL=1, STRATA_HC_REQ8 and the
// FP16 prompt images are the PC prompt path's, and the exact-one CPU rows need AVX2.

#include <cstddef>
#include <cstdint>

#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {
// STRATA_HC_PERSIST=1 (HIP): the verifier calls the init on every card; nothing to set up on Metal
void fused_gr_persist_init() {}
bool fused_gr_read_persist(const FusedGrArgs*, int, void*) { return false; }   // "did not run": the caller reads with fused_gr_read_multi
bool qsa_decode_attn_split(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*,
                           uint64_t, float*, int64_t, void*) {
    return false;   // STRATA_QSA_SPLIT_ATTN=1 (int8 KV, sm_80+): "not applied", the caller runs the plain attention
}

void gdn_step_norm_groups(const GdnGroups&, const float*, int, const float*, const float*, const float*, const float*, float, float*,
                          int, int, bool, void*) {
    strata::metal_engine::missing("strata::kernels::gdn_step_norm_groups");
}
void qsa_block_scores_rows(const QsaScoreRows&, const float*, const int32_t*, int64_t, int64_t, const QsaShapes&, float*, void*) {
    strata::metal_engine::missing("strata::kernels::qsa_block_scores_rows(rows)");
}
void qsa_decode_attn_batch_rows(const float*, const QsaAttnPoolsRows&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                                float*, float*, int64_t, void*) {
    strata::metal_engine::missing("strata::kernels::qsa_decode_attn_batch_rows");
}
void handoff_publish(float*, const float*, int64_t, const float*, int64_t, const float*, int64_t, uint32_t*, uint32_t*, void*,
                     const uint32_t*) {
    strata::metal_engine::missing("strata::kernels::handoff_publish");
}
void wait_handoff(const uint32_t*, uint32_t, uint32_t*, unsigned long long, void*) {
    strata::metal_engine::missing("strata::kernels::wait_handoff");
}
void wait_copy_i32_from_mapped(const uint32_t*, uint32_t, int32_t*, const int32_t*, int64_t, void*) {
    strata::metal_engine::missing("strata::kernels::wait_copy_i32_from_mapped");
}
void wait_flag_ge_stamped(const uint32_t*, uint32_t, unsigned long long*, unsigned long long*, void*) {
    strata::metal_engine::missing("strata::kernels::wait_flag_ge_stamped");
}
void copy_rows_from_mapped_after(const uint32_t*, uint32_t, float*, const float*, int64_t, int64_t, const int32_t*, const int32_t*,
                                 void*) {
    strata::metal_engine::missing("strata::kernels::copy_rows_from_mapped_after");
}
void native_quantize_q8_1_il(const float*, void*, void*, int, int, void*) {
    strata::metal_engine::missing("strata::kernels::native_quantize_q8_1_il");
}
void hc_q8_to_bf16(const void*, const float*, uint16_t*, long long, void*) {
    strata::metal_engine::missing("strata::kernels::hc_q8_to_bf16");
}
void res_patch(int32_t*, const int32_t*, int, void*) { strata::metal_engine::missing("strata::kernels::res_patch"); }
}  // namespace strata::kernels

namespace strata::kernels::cpu {
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::iq256_gu_rows_exact_one");
}
}  // namespace strata::kernels::cpu

namespace strata::prefill {
// STRATA_F16_RANGE: the FP16 range report of the PC prompt path; "nothing recorded" here
bool f16_range_read(float&, unsigned long long&, bool) { return false; }
bool gemm_f16_range_read(float*, unsigned long long*, bool) { return false; }
// the FP16 image overloads (f16img; the old signatures are in stubs.cpp)
void gr_norm(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*, bool) {
    strata::metal_engine::missing("strata::prefill::gr_norm(f16img)");
}
void gr_norm_rs(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*, int64_t, bool) {
    strata::metal_engine::missing("strata::prefill::gr_norm_rs(f16img)");
}
void gr_write_norm_rs(float*, const float*, const float*, int64_t, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*,
                      int64_t, bool) {
    strata::metal_engine::missing("strata::prefill::gr_write_norm_rs(f16img)");
}
void gr_write_cvec_norm_rs(float*, const float*, const float*, int64_t, const float*, const float*, const int*, int, const float*,
                           float, float*, uint16_t*, int64_t, void*, uint16_t*, int64_t, bool) {
    strata::metal_engine::missing("strata::prefill::gr_write_cvec_norm_rs(f16img)");
}
}  // namespace strata::prefill
