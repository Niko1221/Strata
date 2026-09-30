// s2_avx512_stub.cpp - TEST STUB, never linked into the shipped library.
//
// The dispatchers in expert_layout.cpp reference the AVX-512 originals in expert.cpp.  That file is
// compiled with /arch:AVX512, so it CANNOT be compiled on the target CPU at all - which is the entire
// reason the dispatchers exist.  For the parity test the linker still has to resolve those symbols, so
// they are stubbed here.
//
// Every stub ABORTS rather than returning a plausible number.  On this CPU `cpu_avx512_ok()` is false, so
// the dispatchers must never reach these; if one ever did - a broken feature probe, say - the test would
// otherwise compare a real result against a fake one and pass, which is the worst possible failure mode.
// Turning it into a loud abort means a wrong dispatch is impossible to miss.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::kernels::cpu {
namespace {
[[noreturn]] void unreachable_avx512(const char* who) {
    std::fprintf(stderr,
                 "s2_avx1_parity: %s was called on a CPU with no AVX-512.\n"
                 "        That means the dispatch in expert_layout.cpp picked the wrong branch - cpu_avx512_ok()\n"
                 "        returned true on a CPUID probe that should have said no.  The result would be an\n"
                 "        illegal instruction, so this aborts rather than returning something plausible.\n",
                 who);
    std::abort();
}
}  // namespace

void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { unreachable_avx512("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { unreachable_avx512("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) {
    unreachable_avx512("s2_expert_down_rows");
}
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unreachable_avx512("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unreachable_avx512("s2_expert_down_rows_multi");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    unreachable_avx512("s2_expert_vnni_multi");
}
bool expert_oracle_q8_0_enabled() { return false; }

// The native-layout Q2_0 entry points and the AVX-512 activation quantizer, also from expert.cpp.
// `q2_rows_any` / `act_quant_any` in expert_layout.cpp dispatch to these, and on this CPU the probes send
// both to the AVX1 port in q2_avx1.cpp.  Same reasoning as above: they must be unreachable here.
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unreachable_avx512("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unreachable_avx512("q2_0_gguf_rows_multi_avx2");
}
void act_quant_q8_1(const float*, int, ActQ&) { unreachable_avx512("act_quant_q8_1"); }
void act_quant_q8_1_avx2(const float*, int, ActQ&) { unreachable_avx512("act_quant_q8_1_avx2"); }

// ---- the NATIVE-pack entry points (native_expert.cpp -> iq_avx2.cpp) --------------------------
//
// Stubbed for a second reason, and this one is not about the dispatch.  `expert_layout_load` in
// expert_layout.cpp calls `native_fmt`, so merely LINKING this test against strata_kernels_cpu drags
// native_expert.cpp in, and that calls iq_avx2.cpp - which is compiled /arch:AVX2 and therefore
// contains BMI2 (`shlx`).  On the target CPU (Sandy Bridge-E, no BMI2) that faults.
//
// The fault happens in a STATIC INITIALIZER, before main() runs, so the test died with 0xC000001D
// before printing its first line.  This was found after the rebase onto 0.1.28: iq_avx2.cpp did not
// exist at our 0.1.20 base (upstream added it in df6980d, "E-2: prefetch the i-quant expert rows"),
// so the same test passed there and fails here.  Stubbing native_fmt breaks the link edge, iq_avx2.cpp
// is never pulled in, and the BMI2 initializer never executes.
//
// Returning false with a reason, rather than aborting, is deliberate and is the ONE exception to the
// rule above: `expert_layout_load` legitimately CALLS native_fmt during normal setup, so an aborting
// stub would fire on a correct run.  The other five are never reached on the AVX1 path and still abort.
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err) {
    err = "not built with native experts (parity test stub)";
    return false;
}
[[noreturn]] void unreachable_native(const char* who) {
    std::fprintf(stderr,
                 "s2_avx1_parity: %s was called, but the native (iq) path is not part of this test.\n"
                 "        It lives in iq_avx2.cpp, which is compiled /arch:AVX2 and uses BMI2, which the\n"
                 "        target CPU does not have - reaching it is an illegal instruction by construction.\n",
                 who);
    std::abort();
}
bool native_experts_available() noexcept { return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { unreachable_native("native_quant_act"); }
void native_quant_h(const NativeFmt&, const float*, void*) { unreachable_native("native_quant_h"); }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    unreachable_native("native_gu_rows");
}
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    unreachable_native("native_down_rows");
}

}  // namespace strata::kernels::cpu
