// metal/tests/ds4_dq_parity.cpp - the DeepSeek engine's dense-matrix dequantizers to F16 rows (ds4_init.metal: ds4_dq_*; the host
// side is dequant_rows in ds4_finish.cpp, reached from ds4_gguf.cu rows() for the prompt chunk's dense GEMM operands) against ggml's
// dequantize_row_* (ref_quant.cpp).  Q4_K had none: a dense Q4_K matrix fell through to iq::dequant_rows, whose CUDA launch has no
// Metal dispatch and ended the engine.  Rows are padded (ld above the row size) so the stride is tested too.
//   cmake --build <build-metal> --target ds4_dq_parity && <build-metal>/ds4_dq_parity
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"../src/engine/msl/ds4_init.metal"}, "", false);
    struct Case { int type; const char* name; const char* fn; int k; };
    const Case cases[] = {{ref::Q4_K, "Q4_K", "ds4_dq_q4k", 768}, {ref::Q4_K, "Q4_K", "ds4_dq_q4k", 256}, {ref::Q5_K, "Q5_K", "ds4_dq_q5k", 768},
                          {ref::Q6_K, "Q6_K", "ds4_dq_q6k", 768}, {ref::Q8_0, "Q8_0", "ds4_dq_q8", 640}, {ref::BF16, "BF16", "ds4_dq_bf16", 640}};
    const int n = 37;
    int fails = 0;
    for (const Case& c : cases) {
        mtl::Pipeline pl = cx.pipeline(lib, c.fn);
        const size_t row = ref::row_bytes(c.type, c.k);
        const size_t ld = row + 64;   // bytes
        std::vector<uint8_t> packed((size_t) n * row);
        ref::random_blocks(c.type, packed.data(), n, c.k, 1234 + (uint64_t) c.type, 0.02f);
        mtl::Buffer bw = cx.alloc(ld * (size_t) n), bo = cx.alloc((size_t) n * c.k * 2);
        std::memset(bw.ptr, 0, ld * (size_t) n);
        for (int r = 0; r < n; ++r) std::memcpy(bw.as<uint8_t>() + (size_t) r * ld, packed.data() + (size_t) r * row, row);
        const bool bf = c.type == ref::BF16;
        const unsigned long ldk = bf ? (unsigned long) (ld / 2) : (unsigned long) ld;   // BF16: values a row
        size_t items;
        switch (c.type) {
            case ref::Q4_K: case ref::Q5_K: items = (size_t) n * (c.k / 256) * 8; break;
            case ref::Q6_K: items = (size_t) n * (c.k / 256) * 64; break;
            case ref::Q8_0: items = (size_t) n * (c.k / 32); break;
            default: items = (size_t) n * c.k; break;
        }
        mtl::Dispatch d;
        d.p = &pl;
        d.args = {mtl::Arg::B(bw), mtl::Arg::V(ldk), mtl::Arg::V((uint32_t) c.k), mtl::Arg::V((uint32_t) n), mtl::Arg::B(bo)};
        d.grid[0] = (uint32_t) ((items + 255) / 256);
        d.tg[0] = 256;
        cx.run({d});
        std::vector<float> want((size_t) c.k);
        double worst = 0;
        int bad = 0;
        for (int r = 0; r < n; ++r) {
            ref::dequant(c.type, packed.data() + (size_t) r * row, want.data(), (size_t) c.k);
            for (int i = 0; i < c.k; ++i) {
                const float got = ref::f16_to_f32(bo.as<uint16_t>()[(size_t) r * c.k + i]);
                const double err = std::fabs((double) got - (double) want[(size_t) i]);
                const double tol = 1.5e-3 * std::fabs((double) want[(size_t) i]) + 1e-6;   // a half's rounding of the float result (2^-11 = 4.9e-4) and an fma's
                worst = std::max(worst, err / (std::fabs((double) want[(size_t) i]) + 1e-6));
                if (!(err <= tol)) ++bad;
            }
        }
        std::printf("%-5s n=%d k=%d ld=%zu: %d of %d values outside half rounding of ggml's dequantize_row (worst rel %.2e)  %s\n", c.name, n,
                    c.k, ld, bad, n * c.k, worst, bad == 0 ? "PASS" : "FAIL");
        fails += bad != 0;
    }
    std::printf("%s\n", fails ? "ds4_dq_parity: FAILED" : "ds4_dq_parity: all ok");
    return fails ? 1 : 0;
}
