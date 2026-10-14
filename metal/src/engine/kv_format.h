// metal/src/engine/kv_format.h - the decode window's K/V cache format wiring (M1-b: --kv q4_0 real on Metal) as one
// plain, testable place: the pool sizes by format, the qsa pipeline keys / function-constant sets, the startup
// format check and the prompt-path choice.  window.cpp allocates its pools and builds its dispatches through these
// (no literal 144 or "#q4" outside this file); tests/kv_format_test.cpp pins them, including the stale-key trap.
//
// The format state itself is the shared host code's (strata::core::qsa_kv_q4 / qsa_kv_int8 / qsa_kv_hybrid, set by
// generate.cpp's --kv before any window comes up).  Window::init reads it ONCE: on Metal nothing toggles it while
// the window lives (the PC's MTP drafter flips the flags around its own init on CUDA only - src/core/mtp.cpp hands
// over to this backend's hook before that code runs), so a window's pools and its pipelines cannot disagree.
#pragma once

#include "mtl_rt.h"

#include <string>

namespace strata::metal_engine {

// the qsa K/V pool bytes per cell, per pool (K and V each take one).  int8/fp16: today's layout, codes
// [cell][2 kv heads][256] plus one fp16 scale per 64 values ([cell][2 heads][4] halves).  q4_0: the pool row of
// qsa.metal's function-constant-25 mode, [cell][2 kv heads][144] (8 blocks of 18 bytes, the fp16 scale inside each
// block - qsa.metal's Q4_ROW), and NO scale pool: the fc-25 kernels never read ksc/vsc, so their buffer slots bind
// a token-sized placeholder instead of a real allocation.  The indexer pools (tail / dead / pooled) are the same
// bytes in every format and are not described here.
struct KvPoolBytes {
    size_t pool;   // kq / vq each, per cell
    size_t scale;  // ksc / vsc each, per cell; 0 in q4
};
constexpr KvPoolBytes kv_pool_bytes(bool q4) { return q4 ? KvPoolBytes{2 * 144, 0} : KvPoolBytes{2 * 256, 2 * 4 * 2}; }

// the engine's startup format check (Window::init): "" when this backend runs the combination.  fp16 / int8 are
// today's path (the window's pools are the int8 ones either way), q4_0 takes the fused fc-25 path of this PR;
// k8v4 is not implemented on Metal and is REFUSED here - before M1-b the flag was silently ignored (the int8 pools
// were allocated whatever --kv said, M0's finding), and an explicit error beats a wrong-but-running engine.
inline std::string kv_format_check(bool q4, bool hybrid) {
    (void) q4;   // generate.cpp's --kv sets q4 and hybrid mutually exclusive; hybrid is simply not built here
    if (hybrid) return "metal window: --kv k8v4 is not implemented on Metal (use fp16, int8 or q4_0)";
    return "";
}

// qsa_prep's Constants and pipeline key: every function constant the specialization compiles in must be in BOTH -
// Window::P() looks the key up FIRST and only creates the pipeline (with its Constants) on a miss, so a key that
// left a constant out would hand back an EARLIER specialization of the same function name (the int8 one under q4,
// the plain-RoPE one under M-RoPE): a stale pipeline that writes or reads the pools in the wrong format, silently.
// fc 24 = M-RoPE (qsa_prep only; attn/merge have no such arm), fc 25 = the q4 KV mode (qsa_prep / qsa_attn /
// qsa_merge; qsa_score and qsa_topk have no q4 arm and keep their plain keys).
inline mtl::Constants qsa_prep_constants(bool mrope, bool q4) {
    mtl::Constants c;
    if (mrope) c.b(24, true);
    if (q4) c.b(25, true);
    return c;
}
inline std::string qsa_prep_key(bool mrope, bool q4) {
    std::string k = "qsa_prep";
    if (mrope) k += "#mrope";
    if (q4) k += "#q4";
    return k;
}

// qsa_attn and qsa_merge (decode, MTP and both prompt chains alike): their only format-sensitive arm is q4 (fc 25).
inline mtl::Constants qsa_q4_constants(bool q4) { return q4 ? mtl::Constants().b(25, true) : mtl::Constants(); }
inline std::string qsa_q4_key(const char* fn, bool q4) {
    std::string k = fn;
    if (q4) k += "#q4";
    return k;
}

// the prompt chains' matrix-unit attention (qsa_prompt.metal's qsa_attn_mm / qsa_flash, M2-b): fc 7 is the M5's
// right-input layout, the SITE's own choice (attn_mm's arm is the k4 one, flash as probed) - bound unconditionally,
// for is_function_constant_defined defaults it to true when left out; fc 25 is the KV format.  One Constants set,
// and the key is the site's today-literal plus "#q4" (fc 7 stays out of flash's key as today: one process probes
// one layout, so that constant cannot go stale in the cache - the format can, which is what the #q4 tag fixes).
inline mtl::Constants qsa_prompt_mm_constants(bool fc7, bool q4) {
    mtl::Constants c;
    c.b(7, fc7);
    if (q4) c.b(25, true);
    return c;
}
inline std::string qsa_prompt_mm_key(const char* today, bool q4) {
    return q4 ? std::string(today) + "#q4" : std::string(today);
}

// the prompt fast path runs in EVERY format (M2-b): the chains' prep / attn / attn_mm / flash / merge dispatches
// are format-aware through this file (the format in the key and the Constants both), over the fc-25 kernel arms.
// Kept as the one switch point the arena and prompt_run_rows read; STRATA_METAL_PROMPT=window still forces the
// decode window's path for any format.
inline bool prompt_fast_path(bool q4) {
    (void) q4;
    return true;
}

}  // namespace strata::metal_engine
