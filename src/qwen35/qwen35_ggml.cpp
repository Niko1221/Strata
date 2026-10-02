// src/qwen35/qwen35_ggml.cpp - the quantized matvec/row-dequant hooks, backed by ggml-cpu's type traits.
//
// This is the ONLY place the Qwen35 forward pass touches a quantized format.  The arithmetic is ggml-cpu's own
// (`vec_dot` on the weight type's `vec_dot_type` activation, `to_float` for a single row), so an IQ4_XS or Q4_K
// block means exactly what it means in llama.cpp.  `qwen35.cpp` calls through the two function pointers so the
// float-only unit tests link without ggml.
#include "strata/qwen35/qwen35.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::qwen35 {
namespace {

void quant_matvec(int type, const void* w, int64_t n_in, int64_t n_out, const float* x, float* y) {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
    const auto* t = ggml_get_type_traits_cpu((ggml_type) type);
    if (!t || !t->vec_dot) {
        std::fprintf(stderr, "qwen35: ggml-cpu has no vec_dot for type %d\n", type);
        std::exit(1);
    }
    const ggml_type vdt = t->vec_dot_type;
    const auto* at = ggml_get_type_traits_cpu(vdt);
    static thread_local std::vector<uint8_t> scratch;
    scratch.resize(ggml_row_size(vdt, n_in));
    at->from_float(x, scratch.data(), n_in);
    const size_t rb = ggml_row_size((ggml_type) type, n_in);
    const int n = (int) n_in;
    for (int64_t o = 0; o < n_out; ++o) {
        float s = 0.0f;
        t->vec_dot(n, &s, 0, (const char*) w + (size_t) o * rb, 0, scratch.data(), 0, 1);
        y[o] = s;
    }
}

void row_dequant(int type, const void* row, int64_t n, float* out) {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
    const auto* t = ggml_get_type_traits((ggml_type) type);
    if (!t || !t->to_float) {
        std::fprintf(stderr, "qwen35: ggml has no to_float for type %d\n", type);
        std::exit(1);
    }
    t->to_float(row, out, n);
}

}  // namespace

void qwen35_enable_ggml() {
    g_quant_matvec = quant_matvec;
    g_row_dequant = row_dequant;
}

}  // namespace strata::qwen35
