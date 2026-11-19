#include "spin8/spin8.h"

extern "C" spin8_status spin8_get_capabilities(spin8_capabilities* out) {
    if (!out) return SPIN8_INVALID_ARGUMENT;
    *out = spin8_capabilities{1, (1u << SPIN8_F32) | (1u << SPIN8_F64), 8, 1024};
    return SPIN8_OK;
}
