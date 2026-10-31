// Decode real GGUF tensors with Strata's existing scalar references.
#include "strata/artifact/dequant.hpp"

extern "C" int strata_moe_decode(const uint8_t* raw, float* out,
                                uint64_t elements, uint32_t type) {
    uint64_t n = 0, bytes = 0;
    switch (type) {
    case 0: strata::dequantize_f32(raw, out, elements); return 0;
    case 30: strata::dequantize_bf16(raw, out, elements); return 0;
    case 6: n = 32; bytes = 22; break;
    case 11: n = 256; bytes = 110; break;
    case 23: n = 256; bytes = 136; break;
    case 42: n = 64; bytes = 18; break;
    default: return 1;
    }
    if (elements % n) return 2;
    for (uint64_t i = 0; i < elements / n; ++i) {
        switch (type) {
        case 6: strata::dequantize_q5_0(raw + i * bytes, out + i * n); break;
        case 11: strata::dequantize_q3_K(raw + i * bytes, out + i * n); break;
        case 23: strata::dequantize_iq4_xs(raw + i * bytes, out + i * n); break;
        case 42: strata::dequantize_q2_0(raw + i * bytes, out + i * n); break;
        }
    }
    return 0;
}
