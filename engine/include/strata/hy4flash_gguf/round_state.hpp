#pragma once
#include <cstdint>

namespace strata::hy4_flash {
// One accepted prefix. count includes the bonus/rejection token; accepted counts
// matching draft tokens only. The KV input prefix has count rows, while tokens
// contains outputs one position ahead. No token after stop/error is published.
struct RoundVerdict {
    int32_t count=0, accepted=0, next_token=-1, next_position=0;
    int32_t stopped=0, nonfinite=0;
    int32_t tokens[8]={};
};
struct DeviceLoopState {
    int32_t base=0,remaining=0,pending=-1,produced=0;
    int32_t accepted=0,offered=0,windows=0,stopped=0,nonfinite=0;
};
struct LoopOutput {
    volatile uint32_t count=0,cancel=0,error=0;
    int32_t tokens[8192]={};
};
}
