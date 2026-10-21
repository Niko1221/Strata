#include "kernel_operator.h"

using namespace AscendC;

// Experimental projection only. Weight codes are unchanged Q2_0 code-1;
// each 64-weight block consumes TWO independent 32-element activation scales.
// Eight output rows form one aligned 32-byte store, including zero tail padding.
extern "C" __global__ __aicore__ void strata_s2_projection_kernel(
    GM_ADDR codes_ptr, GM_ADDR weights_ptr, GM_ADDR q_ptr, GM_ADDR acts_ptr,
    GM_ADDR out_ptr, uint32_t rows, uint32_t columns, uint32_t tokens) {
    GlobalTensor<uint8_t> codes;
    GlobalTensor<float> weights, acts, out;
    GlobalTensor<int8_t> q;
    codes.SetGlobalBuffer((__gm__ uint8_t*)codes_ptr);
    weights.SetGlobalBuffer((__gm__ float*)weights_ptr);
    q.SetGlobalBuffer((__gm__ int8_t*)q_ptr);
    acts.SetGlobalBuffer((__gm__ float*)acts_ptr);
    out.SetGlobalBuffer((__gm__ float*)out_ptr);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer, activationBuffer;
    pipe.InitBuffer(buffer, 8 * sizeof(float));
    pipe.InitBuffer(activationBuffer, 8 * 2560 * sizeof(int8_t));
    auto tile = buffer.Get<float>();
    auto cachedQ = activationBuffer.Get<int8_t>();
    DataCopy(cachedQ, q, tokens * columns);
    PipeBarrier<PIPE_ALL>();
    const uint32_t groups = (rows + 7) / 8;
    const uint32_t blocks = columns / 64;
    for (uint32_t group = GetBlockIdx(); group < groups; group += GetBlockNum()) {
        float values[8][8] = {};
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t row = group * 8 + lane;
            if (row >= rows) continue;
            for (uint32_t b = 0; b < blocks; ++b) {
                int32_t dot[8][2] = {};
                for (uint32_t byteIndex = 0; byteIndex < 16; ++byteIndex) {
                    const uint32_t byte = codes.GetValue((row * blocks + b) * 16 + byteIndex);
                    for (uint32_t sub = 0; sub < 4; ++sub) {
                        const uint32_t j = byteIndex * 4 + sub;
                        const int32_t weight = int32_t((byte >> (2 * sub)) & 3) - 1;
                        for (uint32_t t = 0; t < tokens; ++t)
                            dot[t][j / 32] += weight * int32_t(cachedQ.GetValue(t * columns + b * 64 + j));
                    }
                }
                const float d = weights.GetValue(row * blocks + b);
                for (uint32_t t = 0; t < tokens; ++t) {
                    const uint32_t chunk = t * (columns / 32) + b * 2;
                    float block = acts.GetValue(chunk) * float(dot[t][0]);
                    block += acts.GetValue(chunk + 1) * float(dot[t][1]);
                    values[t][lane] += d * block;
                }
            }
        }
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t lane = 0; lane < 8; ++lane) tile.SetValue(lane, values[t][lane]);
            PipeBarrier<PIPE_ALL>();
            DataCopy(out[t * groups * 8 + group * 8], tile, 8);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

extern "C" int strata_s2_projection_launch(void* codes, void* weights, void* q,
    void* acts, void* out, uint32_t rows, uint32_t columns, uint32_t tokens, void* stream) {
    if (columns == 0 || columns > 2560 || columns % 64 || tokens == 0 || tokens > 8) return 1;
    if (rows == 0) return 0;
    const uint32_t groups = (rows + 7) / 8;
    const uint32_t cores = groups < 32 ? groups : 32;
    strata_s2_projection_kernel<<<cores, nullptr, stream>>>(codes, weights, q, acts, out, rows, columns, tokens);
    return 0;
}
