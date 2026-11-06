// Small C ABI for the CANN Python backend. Only weight decoding runs on the host;
// model projections and recurrent state updates run through torch_npu / CANN.
#include "strata/cann/quants.hpp"
#include "strata/artifact/dequant.hpp"

#include <cstring>

extern "C" int strata_cann_decode(int type, const void * input, float * output, int64_t count) {
    if (!input || !output || count <= 0) return -1;
    const auto * p = static_cast<const uint8_t *>(input);
    if (type == 0) {
        std::memcpy(output, input, static_cast<size_t>(count) * sizeof(float));
    } else if (type == 1 || type == 30) {
        for (int64_t i = 0; i < count; ++i) {
            const uint16_t bits = strata::read_u16(p + i * 2);
            if (type == 1) {
                output[i] = strata::fp16_to_fp32(bits);
            } else {
                const uint32_t wide = static_cast<uint32_t>(bits) << 16;
                std::memcpy(output + i, &wide, 4);
            }
        }
    } else if (!strata::cann::dequantize(type, input, output, count)) {
        return -2;
    }
    return 0;
}
