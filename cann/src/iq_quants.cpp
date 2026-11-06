// IQ arithmetic follows ggml-quants.c at Strata's pinned llama.cpp revision
// 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT, The ggml authors).
// Codebooks and the other scalar decoders are Strata's existing vendored data.
#include "strata/cann/quants.hpp"
#include "strata/artifact/dequant.hpp"

#include <cstring>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../../third_party/ggml/ggml-common.h"
#undef QK_K

namespace strata::cann {

bool dequantize_iq(int type, const void * raw, float * y, int64_t count) {
    const auto * bytes = static_cast<const uint8_t *>(raw);
    const int elements = quant_block_elements(type);
    const int block_bytes = quant_type_size(type);
    if (!elements || count <= 0 || count % elements) return false;
    for (int64_t b = 0; b < count / elements; ++b) {
        const uint8_t * p = bytes + b * block_bytes;
        float * out = y + b * elements;
        const float d = strata::fp16_to_fp32(strata::read_u16(p));
        if (type == 20) {
            strata::dequantize_iq4_nl(p, out);
        } else if (type == 23) {
            strata::dequantize_iq4_xs(p, out);
        } else if (type == 42) {
            strata::dequantize_q2_0(p, out);
        } else if (type == 16) {
            for (int chunk = 0; chunk < 8; ++chunk) {
                uint32_t aux[2];
                std::memcpy(aux, p + 2 + chunk * 8, 8);
                const auto * indices = reinterpret_cast<const uint8_t *>(aux);
                const float scale = d * (0.5f + (aux[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const auto * grid = reinterpret_cast<const uint8_t *>(iq2xxs_grid + indices[l]);
                    const uint8_t signs = ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j)
                        out[chunk * 32 + l * 8 + j] =
                            scale * grid[j] * ((signs & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        } else if (type == 17) {
            for (int chunk = 0; chunk < 8; ++chunk) {
                const uint8_t sc = p[66 + chunk];
                const float scales[2] = {d * (0.5f + (sc & 15)) * 0.25f,
                                         d * (0.5f + (sc >> 4)) * 0.25f};
                for (int l = 0; l < 4; ++l) {
                    const uint16_t index = strata::read_u16(p + 2 + (4 * chunk + l) * 2);
                    const auto * grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + (index & 511));
                    const uint8_t signs = ksigns_iq2xs[index >> 9];
                    for (int j = 0; j < 8; ++j)
                        out[chunk * 32 + l * 8 + j] =
                            scales[l / 2] * grid[j] * ((signs & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        } else if (type == 22) {
            for (int chunk = 0; chunk < 8; ++chunk) {
                const uint8_t sc = p[74 + chunk];
                const float scales[2] = {d * (0.5f + (sc & 15)) * 0.25f,
                                         d * (0.5f + (sc >> 4)) * 0.25f};
                for (int l = 0; l < 4; ++l) {
                    const int index = p[2 + chunk * 4 + l] | ((p[66 + chunk] << (8 - 2 * l)) & 0x300);
                    const auto * grid = reinterpret_cast<const uint8_t *>(iq2s_grid + index);
                    const uint8_t signs = p[34 + chunk * 4 + l];
                    for (int j = 0; j < 8; ++j)
                        out[chunk * 32 + l * 8 + j] =
                            scales[l / 2] * grid[j] * ((signs & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        } else if (type == 18) {
            for (int chunk = 0; chunk < 8; ++chunk) {
                uint32_t aux;
                std::memcpy(&aux, p + 66 + 4 * chunk, 4);
                const float scale = d * (0.5f + (aux >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                    const uint8_t * grids[2] = {
                        reinterpret_cast<const uint8_t *>(iq3xxs_grid + p[2 + chunk * 8 + 2 * l]),
                        reinterpret_cast<const uint8_t *>(iq3xxs_grid + p[3 + chunk * 8 + 2 * l])};
                    for (int half = 0; half < 2; ++half)
                        for (int j = 0; j < 4; ++j)
                            out[chunk * 32 + l * 8 + half * 4 + j] =
                                scale * grids[half][j] *
                                ((signs & (1u << (half * 4 + j))) ? -1.0f : 1.0f);
                }
            }
        } else if (type == 21) {
            for (int chunk = 0; chunk < 8; ++chunk) {
                const uint8_t sc = p[106 + chunk / 2];
                const float scale = d * (1 + 2 * ((sc >> (4 * (chunk % 2))) & 15));
                const uint8_t high = p[66 + chunk];
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = p[74 + chunk * 4 + l];
                    for (int half = 0; half < 2; ++half) {
                        const int index = p[2 + chunk * 8 + 2 * l + half] |
                                          ((high << (8 - 2 * l - half)) & 256);
                        const auto * grid = reinterpret_cast<const uint8_t *>(iq3s_grid + index);
                        for (int j = 0; j < 4; ++j)
                            out[chunk * 32 + l * 8 + half * 4 + j] =
                                scale * grid[j] *
                                ((signs & (1u << (half * 4 + j))) ? -1.0f : 1.0f);
                    }
                }
            }
        } else {
            return false;
        }
    }
    return true;
}
} // namespace strata::cann
