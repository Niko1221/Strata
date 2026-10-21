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
    TBuf<TPosition::VECCALC> buffer, activationBuffer, halfBuffer, floatBuffer;
    TBuf<TPosition::VECCALC> codeBuffer, productBuffer, sumBuffer;
    pipe.InitBuffer(buffer, 8 * sizeof(float));
    pipe.InitBuffer(activationBuffer, 8 * 2560 * sizeof(int8_t));
    pipe.InitBuffer(halfBuffer, 8 * 2560 * sizeof(half));
    pipe.InitBuffer(floatBuffer, 8 * 2560 * sizeof(float));
    pipe.InitBuffer(codeBuffer, 2560 * sizeof(float));
    pipe.InitBuffer(productBuffer, 2560 * sizeof(float));
    pipe.InitBuffer(sumBuffer, 80 * sizeof(float));
    auto tile = buffer.Get<float>();
    auto cachedQ = activationBuffer.Get<int8_t>();
    auto halfQ = halfBuffer.Get<half>();
    auto floatQ = floatBuffer.Get<float>();
    auto rowCodes = codeBuffer.Get<float>();
    auto products = productBuffer.Get<float>();
    auto sums = sumBuffer.Get<float>();
    DataCopy(cachedQ, q, tokens * columns);
    PipeBarrier<PIPE_ALL>();
    // INT8 -> half -> FP32 is exact over the entire INT8 range. Products of
    // code-1 and q, and sums of 32 such integers, are exact in FP32 as well.
    Cast(halfQ, cachedQ, RoundMode::CAST_NONE, tokens * columns);
    PipeBarrier<PIPE_V>();
    Cast(floatQ, halfQ, RoundMode::CAST_NONE, tokens * columns);
    PipeBarrier<PIPE_ALL>();
    const uint32_t groups = (rows + 7) / 8;
    const uint32_t blocks = columns / 64;
    for (uint32_t group = GetBlockIdx(); group < groups; group += GetBlockNum()) {
        float values[8][8] = {};
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t row = group * 8 + lane;
            if (row >= rows) continue;
            for (uint32_t b = 0; b < blocks; ++b) {
                for (uint32_t byteIndex = 0; byteIndex < 16; ++byteIndex) {
                    const uint32_t byte = codes.GetValue((row * blocks + b) * 16 + byteIndex);
                    for (uint32_t sub = 0; sub < 4; ++sub) {
                        const uint32_t j = byteIndex * 4 + sub;
                        const int32_t weight = int32_t((byte >> (2 * sub)) & 3) - 1;
                        rowCodes.SetValue(b * 64 + j, float(weight));
                    }
                }
            }
            PipeBarrier<PIPE_ALL>();
            for (uint32_t t = 0; t < tokens; ++t) {
                Mul(products, rowCodes, floatQ[t * columns], columns);
                PipeBarrier<PIPE_V>();
                // mask=32, source repeat stride=4 x 32 bytes. Each repeat
                // yields one consecutive FP32 integer dot for a Q8 chunk.
                WholeReduceSum(sums, products, 32, columns / 32, 1, 1, 4);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t b = 0; b < blocks; ++b) {
                    const float d = weights.GetValue(row * blocks + b);
                    const uint32_t chunk = t * (columns / 32) + b * 2;
                    float block = acts.GetValue(chunk) * sums.GetValue(2 * b);
                    block += acts.GetValue(chunk + 1) * sums.GetValue(2 * b + 1);
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
