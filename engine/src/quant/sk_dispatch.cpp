// src/quant/sk_dispatch.cpp - the public strata::quant::sk codecs on x86, forwarding to the ISA tier the running CPU can execute
// (sk_codecs.cpp compiled as base / avx2 / avx512, see sk_tiers.hpp).  Built with the baseline flags: it runs on every CPU, so it holds no wide SIMD.
// The tier is chosen once from strata::quant::caps() (which STRATA_FORCE_ISA=avx2|avx|sse caps for tests and older-CPU runs):
//   AVX-512 F/DQ/BW/VL + VNNI -> avx512,  AVX2 + FMA + F16C -> avx2,  else base (plain C++).
#include "sk_tiers.hpp"
#include "strata/quant/isa.hpp"

namespace strata::quant::sk {
namespace {

struct Table {
    void (*quantize_q8_1_g64)(const float*, BlockQ8_1_G64*, size_t) noexcept;
    void (*fwht64_f32)(const float*, float*, size_t) noexcept;
    void (*dequant_sk_row)(const uint8_t*, SkCode, float*, size_t) noexcept;
    float (*dot_sk_q8_1_row)(const uint8_t*, SkCode, const BlockQ8_1_G64*, size_t) noexcept;
    void (*gemm_sk_q8_1)(const uint8_t*, SkCode, const BlockQ8_1_G64*, float*, size_t, size_t, size_t) noexcept;
    void (*dequant_sf4_row)(const uint8_t*, const uint8_t*, float, float*, size_t) noexcept;
    float (*dot_sf4_f32_row)(const uint8_t*, const uint8_t*, float, const float*, size_t) noexcept;
    void (*dequant_mx8_row)(const uint8_t*, const uint8_t*, float*, size_t) noexcept;
    float (*dot_mx8_f32_row)(const uint8_t*, const uint8_t*, const float*, size_t) noexcept;
    float (*dot_f32)(const float*, const float*, size_t) noexcept;
    void (*dequant_paired2_sk_rows)(const uint8_t*, SkCode, float*, float*, size_t) noexcept;
    void (*gemv_paired2_sk_q8_1)(const uint8_t*, SkCode, const BlockQ8_1_G64*, float*, size_t, size_t) noexcept;
    void (*gemm_paired2_sk_q8_1)(const uint8_t*, SkCode, const BlockQ8_1_G64*, float*, size_t, size_t, size_t) noexcept;
    const char* isa;
};

#define STRATA_SK_TABLE(ns, name)                                                                                                    \
    Table {                                                                                                                          \
        &ns::quantize_q8_1_g64, &ns::fwht64_f32, &ns::dequant_sk_row, &ns::dot_sk_q8_1_row, &ns::gemm_sk_q8_1,                       \
        &ns::dequant_sf4_row, &ns::dot_sf4_f32_row, &ns::dequant_mx8_row, &ns::dot_mx8_f32_row, &ns::dot_f32, &ns::dequant_paired2_sk_rows,  \
        &ns::gemv_paired2_sk_q8_1, &ns::gemm_paired2_sk_q8_1, name                                                                   \
    }

const Table& table() noexcept {
    static const Table t = []() {
        const auto& c = strata::quant::caps();
        if (c.avx512 && c.avx512_vnni) return STRATA_SK_TABLE(avx512, "avx512-vnni");
        if (c.avx2) return STRATA_SK_TABLE(avx2, "avx2");
        return STRATA_SK_TABLE(base, "scalar");
    }();
    return t;
}

}  // namespace

void quantize_q8_1_g64(const float* x, BlockQ8_1_G64* out, size_t n) noexcept { table().quantize_q8_1_g64(x, out, n); }
void fwht64_f32(const float* in, float* out, size_t n) noexcept { table().fwht64_f32(in, out, n); }
void dequant_sk_row(const uint8_t* raw, SkCode code, float* out, size_t n) noexcept { table().dequant_sk_row(raw, code, out, n); }
float dot_sk_q8_1_row(const uint8_t* raw, SkCode code, const BlockQ8_1_G64* x, size_t n) noexcept {
    return table().dot_sk_q8_1_row(raw, code, x, n);
}
void gemm_sk_q8_1(const uint8_t* weights, SkCode code, const BlockQ8_1_G64* xs, float* out, size_t nrows, size_t ncols,
                  size_t m) noexcept {
    table().gemm_sk_q8_1(weights, code, xs, out, nrows, ncols, m);
}
const char* active_sk_isa() noexcept { return table().isa; }
void dequant_sf4_row(const uint8_t* packed_e2m1, const uint8_t* scales_e4m3, float outer_scale, float* out, size_t ncols) noexcept {
    table().dequant_sf4_row(packed_e2m1, scales_e4m3, outer_scale, out, ncols);
}
float dot_sf4_f32_row(const uint8_t* packed_e2m1, const uint8_t* scales_e4m3, float outer_scale, const float* x, size_t ncols) noexcept {
    return table().dot_sf4_f32_row(packed_e2m1, scales_e4m3, outer_scale, x, ncols);
}
void dequant_mx8_row(const uint8_t* weights_e4m3, const uint8_t* scales_ue8m0, float* out, size_t ncols) noexcept {
    table().dequant_mx8_row(weights_e4m3, scales_ue8m0, out, ncols);
}
float dot_mx8_f32_row(const uint8_t* weights_e4m3, const uint8_t* scales_ue8m0, const float* x, size_t ncols) noexcept {
    return table().dot_mx8_f32_row(weights_e4m3, scales_ue8m0, x, ncols);
}
float dot_f32(const float* a, const float* b, size_t n) noexcept { return table().dot_f32(a, b, n); }
void dequant_paired2_sk_rows(const uint8_t* pair_raw, SkCode code, float* out_row0, float* out_row1, size_t ncols) noexcept {
    table().dequant_paired2_sk_rows(pair_raw, code, out_row0, out_row1, ncols);
}
void gemv_paired2_sk_q8_1(const uint8_t* weights, SkCode code, const BlockQ8_1_G64* x, float* out, size_t nrows, size_t ncols) noexcept {
    table().gemv_paired2_sk_q8_1(weights, code, x, out, nrows, ncols);
}
void gemm_paired2_sk_q8_1(const uint8_t* weights, SkCode code, const BlockQ8_1_G64* xs, float* out, size_t nrows, size_t ncols,
                          size_t m) noexcept {
    table().gemm_paired2_sk_q8_1(weights, code, xs, out, nrows, ncols, m);
}

}  // namespace strata::quant::sk
