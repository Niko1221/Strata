// Experimental Ascend C correctness baseline for Strata's canonical S2 planes.
// This deliberately serializes each block's scalar unpack and DMA. It is not an
// optimized implementation and is not linked into the Strata engine.
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void strata_s2_kernel(
    GM_ADDR codes, GM_ADDR scales, GM_ADDR output, uint64_t n_blocks) {
    GlobalTensor<uint8_t> c;
    GlobalTensor<float> s, y;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(codes));
    s.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    TPipe pipe;
    TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, 64 * sizeof(float));
    LocalTensor<float> row = scratch.Get<float>();
    for (uint64_t b = GetBlockIdx(); b < n_blocks; b += GetBlockNum()) {
        const float d = s.GetValue(b);
        for (uint32_t j = 0; j < 64; ++j) {
            const uint8_t packed = c.GetValue(b * 16 + j / 4);
            const int code = (packed >> ((j % 4) * 2)) & 3;
            row.SetValue(j, static_cast<float>(code - 1) * d);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(y[b * 64], row, 64);
        PipeBarrier<PIPE_ALL>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void strata_s2_launch(void* codes, void* scales, void* output,
                                  uint64_t n_blocks, void* stream) {
    if (n_blocks == 0) return;
    const uint32_t cores = n_blocks < 32 ? static_cast<uint32_t>(n_blocks) : 32;
    strata_s2_kernel<<<cores, nullptr, stream>>>(
        static_cast<uint8_t*>(codes), static_cast<uint8_t*>(scales),
        static_cast<uint8_t*>(output), n_blocks);
}
#endif
