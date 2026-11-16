#include "strata/prefill/esimd_moe.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const bool expected = argc > 1 && std::atoi(argv[1]) != 0;
    if (strata::prefill::i8::iq2_xxs_enabled() != expected) return 1;
    if (!expected) {
        strata::prefill::i8::Layout layout;
        const int32_t count = 1;
        strata::prefill::i8::make_layout(layout, &count, 1);
        // A disabled format must be refused before touching a stream or pointer.
        if (strata::prefill::i8::gate_up(16, nullptr, 422400, 640, layout, 2560,
                                        nullptr, nullptr, nullptr, nullptr)) return 2;
    }
    std::printf("IQ2_XXS opt-in expected=%d: passed\n", expected);
    return 0;
}
