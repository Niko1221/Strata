// Use the existing upstream reference rather than writing a second decoder.
#include "strata/artifact/dequant.hpp"

extern "C" void strata_q2_reference(const uint8_t* raw, float* output, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; ++b)
        strata::dequantize_q2_0(raw + b * 18, output + b * 64);
}
