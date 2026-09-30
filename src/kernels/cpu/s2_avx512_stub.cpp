// s2_avx512_stub.cpp - LOCAL TEST STUB, never linked into the shipped library.
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

#include <cstdio>
#include <cstdlib>

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

}  // namespace strata::kernels::cpu
