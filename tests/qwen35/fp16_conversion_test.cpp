#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "fp16 fallback test: %s\n", message);
        std::exit(1);
    }
}
struct Vector { float input; uint16_t bits; };
}  // namespace

int main() {
    constexpr std::array<Vector, 12> vectors{{
        {0.f, 0x0000}, {-0.f, 0x8000}, {1.f, 0x3c00}, {-2.f, 0xc000},
        {0.33333334f, 0x3555}, {65504.f, 0x7bff}, {65520.f, 0x7c00},
        {5.960464477539063e-8f, 0x0001}, {2.9802322387695312e-8f, 0x0000},
        {-5.960464477539063e-8f, 0x8001}, {1.00048828125f, 0x3c00}, {1.00146484375f, 0x3c02}
    }};
    for (const auto& vector : vectors) {
        const uint16_t bits = strata::kernels::f16_from_f32(vector.input);
        require(bits == vector.bits, "round-to-nearest-even mismatch");
        require(strata::kernels::f16_from_f32(strata::kernels::f32_from_f16(bits)) == bits,
                "FP16 round trip changed bits");
    }

    require(strata::kernels::f16_from_f32(std::numeric_limits<float>::infinity()) == 0x7c00,
            "infinity encoding");
    const uint16_t nan = strata::kernels::f16_from_f32(std::numeric_limits<float>::quiet_NaN());
    require((nan & 0x7c00) == 0x7c00 && (nan & 0x03ff) != 0, "NaN encoding");
    std::puts("q36_fp16_fallback_test OK");
}
