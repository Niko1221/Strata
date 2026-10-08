#include "strata/kernels/cpu/kq_avx512.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    namespace kc = strata::kernels::cpu;
    if (!kc::kq_type_ok(10) || !kc::kq_type_ok(11) || kc::kq_type_ok(12)) return 1;
    if (!kc::kq_avx512_ok()) { std::puts("kq_avx512_test: SKIP arithmetic (CPU lacks enabled AVX-512)"); return 0; }
    ggml_cpu_init();
    constexpr int n = 768, nt = 13, rows = 3;
    std::vector<float> values(n), results(nt * rows);
    const auto* qa = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t ar = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<unsigned char> acts(nt * ar);
    std::vector<const void*> ap(nt);
    std::vector<float*> op(nt);
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < n; ++i) values[i] = std::sin(float(i * 17 + t * 23)) * (t + 1) / 13;
        ap[t] = acts.data() + t * ar;
        qa->from_float(values.data(), acts.data() + t * ar, n);
        op[t] = results.data() + t * rows;
    }
    for (int type : {10, 11}) {
        const auto gt = static_cast<ggml_type>(type);
        const auto* traits = ggml_get_type_traits_cpu(gt);
        const size_t wr = ggml_row_size(gt, n);
        std::vector<unsigned char> weights(rows * wr);
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < n; ++i) values[i] = std::cos(float(i * 7 + r * 31)) * (r + 1);
            traits->from_float(values.data(), weights.data() + r * wr, n);
        }
        kc::kq_rows(type, weights.data(), wr, n, ap.data(), nt, op.data(), 0, rows);
        for (int t = 0; t < nt; ++t) for (int r = 0; r < rows; ++r) {
            float ref = 0;
            traits->vec_dot(n, &ref, 0, weights.data() + r * wr, 0, ap[t], 0, 1);
            if (std::abs(results[t * rows + r] - ref) > 0.0002f * std::max(1.0f, std::abs(ref))) {
                std::fprintf(stderr, "kq_avx512_test: type %d token %d row %d got %g expected %g\n", type, t, r, results[t * rows + r], ref);
                return 1;
            }
        }
    }
    std::puts("kq_avx512_test: PASS (78 Q2_K/Q3_K multi-token row comparisons)");
}
