// metal/src/engine/stubs_kv_q2.cpp - the q2 / q2k KV formats, the FP16 window overlay and the 5-argument ring table
// (include/strata/kernels/kv_q2.hpp, kv_window.hpp, kv_stream.hpp) on Metal: not ported yet.
//
// The shared host sources (src/core/layer.cpp, verify.cpp, mtp.cpp, conversation_snapshot.cpp, src/prefill/prefill.cpp)
// call these, so the Metal link needs a definition for each.  Each stops the engine with its name, as stubs.cpp does.
// They are reached only with --kv q2 or q2k, or with the FP16 window (--kv-window / STRATA_KV_WINDOW) over a quantized
// cache, and the Metal defaults take none of those paths.
//
// Hand-written, not generated: stubs.cpp comes from a link log (metal/tools/gen_stubs.py, run on a Mac) and does not
// know these names.  A regeneration after this file exists finds nothing undefined here and drops the old 4-argument
// kv_ring_table from stubs.cpp.

#include <cstdint>

#include "strata/kernels/kv_q2.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_window.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {

void kv_append_q2_step(uint8_t* p0, uint8_t* p1, const int32_t* p2, const int32_t* p3, const float* p4, const float* p5,
                       const QsaShapes& p6, void* p7, const KvHostPools* p8, bool p9, float p10) {
    strata::metal_engine::missing("strata::kernels::kv_append_q2_step");
}

void kv_append_q2_steps(uint8_t* p0, uint8_t* p1, const int32_t* p2, const int32_t* p3, int p4, int p5, const float* p6,
                        const float* p7, const QsaShapes& p8, void* p9, const KvHostPools* p10, bool p11, float p12) {
    strata::metal_engine::missing("strata::kernels::kv_append_q2_steps");
}

void kv_append_q2(uint8_t* p0, uint8_t* p1, const int32_t* p2, int64_t p3, int64_t p4, const float* p5, const float* p6,
                  const QsaShapes& p7, void* p8, const KvHostPools* p9, const KvHostPools* p10, bool p11, float p12) {
    strata::metal_engine::missing("strata::kernels::kv_append_q2");
}

void kv_gather_q2_step(const uint8_t* p0, const uint8_t* p1, const int32_t* p2, const int32_t* p3, const int32_t* p4,
                       int64_t p5, const QsaShapes& p6, uint16_t* p7, uint16_t* p8, void* p9, bool p10) {
    strata::metal_engine::missing("strata::kernels::kv_gather_q2_step");
}

void kv_window_write_step(const KvWindowPools& p0, const KvWindowShape& p1, const int32_t* p2, const float* p3,
                          const float* p4, const QsaShapes& p5, void* p6) {
    strata::metal_engine::missing("strata::kernels::kv_window_write_step");
}

void kv_window_write_steps(const KvWindowPools& p0, const KvWindowShape& p1, const int32_t* p2, int p3, int p4,
                           const float* p5, const float* p6, const QsaShapes& p7, void* p8) {
    strata::metal_engine::missing("strata::kernels::kv_window_write_steps");
}

void kv_window_write_batch(const KvWindowPools& p0, const KvWindowShape& p1, int64_t p2, int64_t p3, const float* p4,
                           const float* p5, const QsaShapes& p6, void* p7) {
    strata::metal_engine::missing("strata::kernels::kv_window_write_batch");
}

void kv_window_restore(const KvWindowPools& p0, const KvWindowShape& p1, const QsaAttnPools& p2, int p3, int64_t p4,
                       const QsaShapes& p5, void* p6) {
    strata::metal_engine::missing("strata::kernels::kv_window_restore");
}

// Declared in kv_stream.hpp and called by src/prefill/prefill.cpp (the streamed prompt path's write-back); the base
// tree had the call without a stub, so this is the same gap, closed here.
void kv_unstage_to_host(const QsaAttnPools& p0, const KvHostPools& p1, int p2, int64_t p3, int64_t p4,
                        const QsaShapes& p5, void* p6) {
    strata::metal_engine::missing("strata::kernels::kv_unstage_to_host");
}

// The 5-argument form the shared header declares (kv_stream.hpp); the old 4-argument one in stubs.cpp is not called.
void kv_ring_table(int32_t* p0, int64_t p1, int64_t p2, void* p3, int p4) {
    strata::metal_engine::missing("strata::kernels::kv_ring_table");
}

}  // namespace strata::kernels
