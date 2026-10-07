// Windows ARM64 supports native/IQ expert packs through ggml-cpu.  The legacy Q2_0 pack has only x86 kernels.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {
[[noreturn]] void unsupported() {
    std::fprintf(stderr, "strata: the legacy Q2_0 CPU expert path is not supported on ARM64; use an IQ model\n");
    std::abort();
}
}

void act_quant_q8_1(const float*, int, ActQ&) { unsupported(); }
void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { unsupported(); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { unsupported(); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported(); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported(); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) { unsupported(); }
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) { unsupported(); }
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) { unsupported(); }
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { unsupported(); }
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported();
}
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported();
}
void q2_0_gguf_rows_multi_avx2_v(bool, const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported();
}
void act_quant_q8_1_avx2(const float*, int, ActQ&) { unsupported(); }
bool q2_bitplane_enabled() { return false; }
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { unsupported(); }

void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported();
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported();
}

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        for (int t = 0; t < nt; ++t) {
            const float* xr = x + (size_t) t * (size_t) cols;
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const uint32_t bits = (uint32_t) wr[c] << 16;
                float wf;
                std::memcpy(&wf, &bits, sizeof wf);
                sum += wf * xr[c];
            }
            out[(size_t) t * (size_t) rows + r] = sum;
        }
    }
}

void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, nt, out);
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, 1, out);
}

}  // namespace strata::kernels::cpu
