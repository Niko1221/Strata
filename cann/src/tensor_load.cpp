#include "strata/cann/tensor_load.hpp"

#include <cstring>
#include <stdexcept>

#include "strata/cann/quants.hpp"

namespace strata::cann {
namespace {

constexpr uint32_t kF32 = 0;
constexpr uint32_t kF16 = 1;
constexpr uint32_t kQ8_0 = 8;
constexpr uint32_t kBF16 = 30;
constexpr uint32_t kQ4_K = 12;
constexpr uint32_t kQ5_K = 13;
constexpr uint32_t kQ6_K = 14;

float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    float v;
    if (exp == 0) {
        v = (float) man * (1.0f / 16777216.0f);
    } else if (exp == 31) {
        v = man ? __builtin_nanf("") : __builtin_inff();
    } else {
        v = (float) (man | 0x400u) * (1.0f / 33554432.0f) * (float) (1 << exp);
    }
    return sign ? -v : v;
}

} // namespace

uint64_t tensor_bytes_on_disk(const GgufTensor & t) {
    const uint64_t n = t.elements();
    switch (t.type) {
        case kF32:
            return n * 4;
        case kF16:
        case kBF16:
            return n * 2;
        case kQ8_0:
            return n / 32 * 34;
        case kQ4_K:
        case kQ5_K:
        case kQ6_K: {
            const int bs = quant_type_size((int) t.type);
            return bs ? n / QK_K * (uint64_t) bs : 0;
        }
        default:
            return 0;
    }
}

std::vector<float> load_tensor(const Gguf & g, const GgufTensor & t) {
    const uint64_t bytes = tensor_bytes_on_disk(t);
    if (bytes == 0) {
        throw std::runtime_error("tensor_load: unsupported type " + std::to_string(t.type) + " for " + t.name);
    }
    const std::vector<uint8_t> raw = g.tensor_bytes(t, 0, bytes);
    const uint64_t n = t.elements();
    std::vector<float> out((size_t) n);

    switch (t.type) {
        case kF32:
            std::memcpy(out.data(), raw.data(), (size_t) bytes);
            break;
        case kF16: {
            const uint16_t * h = (const uint16_t *) raw.data();
            for (uint64_t i = 0; i < n; ++i) {
                out[(size_t) i] = half_to_float(h[i]);
            }
            break;
        }
        case kBF16: {
            const uint16_t * h = (const uint16_t *) raw.data();
            for (uint64_t i = 0; i < n; ++i) {
                uint32_t bits = (uint32_t) h[i] << 16;
                float f;
                std::memcpy(&f, &bits, 4);
                out[(size_t) i] = f;
            }
            break;
        }
        default:
            if (!dequantize((int) t.type, raw.data(), out.data(), (int64_t) n)) {
                throw std::runtime_error("tensor_load: dequantize failed for " + t.name);
            }
            break;
    }
    return out;
}

} // namespace strata::cann
