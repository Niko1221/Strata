// hy4-flash image tensors use patch-major rows, with contiguous channels in each row.
#pragma once
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>

namespace strata::hy4_flash::vision {
constexpr int hidden = 1152, heads = 16, ffn = 4304, output = 3072;
constexpr int layers = 27, edge = 128, threads = 256;

template <class T> __global__ void as_half(const T* src, __half* dst, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dst[i] = __float2half(static_cast<float>(src[i]));
}

__global__ void patchify(const unsigned char* rgb, __half* patches, int width) {
    int p = blockIdx.x, stride = width / 16;
    for (int j = threadIdx.x; j < 768; j += blockDim.x) {
        int y = p / stride * 16 + j / 16 % 16, x = p % stride * 16 + j % 16;
        patches[p * 768 + j] = __float2half(rgb[(y * width + x) * 3 + j / 256] / 127.5f - 1.f);
    }
}

__global__ void add_position(float* x, const float* grid, int height, int width) {
    int p = blockIdx.x;
    float y = fminf(edge - 1, fmaxf(0, (p / width + 0.5f) * edge / (height + 0.1f) - 0.5f));
    float u = fminf(edge - 1, fmaxf(0, (p % width + 0.5f) * edge / (width + 0.1f) - 0.5f));
    int top = int(y), left = int(u), right = min(left + 1, edge - 1), bottom = min(top + 1, edge - 1);
    y -= top;
    u -= left;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float* col = grid + d;
        float a = (1 - u) * col[(top * edge + left) * hidden] + u * col[(top * edge + right) * hidden];
        float b = (1 - u) * col[(bottom * edge + left) * hidden] + u * col[(bottom * edge + right) * hidden];
        x[p * hidden + d] += (1 - y) * a + y * b;
    }
}

// LayerNorm in the transformer, RMSNorm followed by 2x2 gather in the projector.
__global__ void normalize(const float* x, const float* weight, const float* bias, __half* out, float eps,
                          int patch_width) {
    using Reduce = cub::BlockReduce<float, threads>;
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ float mean, scale;
    int p = blockIdx.x;
    x += p * hidden;
    if (bias) {
        float sum = 0;
        for (int j = threadIdx.x; j < hidden; j += blockDim.x)
            sum += x[j];
        float total = Reduce(scratch).Sum(sum);
        if (!threadIdx.x)
            mean = total / hidden;
    } else if (!threadIdx.x)
        mean = 0;
    __syncthreads();
    float squares = 0;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float z = x[j] - mean;
        squares += z * z;
    }
    float total = Reduce(scratch).Sum(squares);
    if (!threadIdx.x)
        scale = rsqrtf(total / hidden + eps);
    __syncthreads();
    int base = p * hidden, stride = 1;
    if (!bias) {
        int y = p / patch_width, xcoord = p % patch_width;
        base = (y / 2 * (patch_width / 2) + xcoord / 2) * hidden * 4 + (y % 2) * 2 + xcoord % 2;
        stride = 4;
    }
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        out[base + j * stride] = __float2half((x[j] - mean) * scale * weight[j] + (bias ? bias[j] : 0));
}

enum class Activation { linear, gelu_tanh, gelu_erf };
__global__ void epilogue(float* x, const float* bias, int size, int channels, Activation act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    float v = x[i] + bias[i % channels];
    if (act == Activation::gelu_tanh)
        v *= 0.5f * (1 + tanhf(0.79788456f * v * (1 + 0.044715f * v * v)));
    if (act == Activation::gelu_erf)
        v *= 0.5f * (1 + erff(0.70710678f * v));
    x[i] = v;
}

__global__ void softmax(float* scores, int keys) {
    using Reduce = cub::BlockReduce<float, threads>;
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ float peak, sum;
    float* row = scores + size_t(blockIdx.x) * keys;
    float v = -CUDART_INF_F;
    for (int j = threadIdx.x; j < keys; j += blockDim.x)
        v = fmaxf(v, row[j]);
    float mx = Reduce(scratch).Reduce(v, cub::Max());
    if (!threadIdx.x)
        peak = mx;
    __syncthreads();
    float subtotal = 0;
    for (int j = threadIdx.x; j < keys; j += blockDim.x) {
        row[j] = expf(row[j] - peak);
        subtotal += row[j];
    }
    float total = Reduce(scratch).Sum(subtotal);
    if (!threadIdx.x)
        sum = total;
    __syncthreads();
    for (int j = threadIdx.x; j < keys; j += blockDim.x)
        row[j] /= sum;
}

__global__ void add_newlines(const float* cells, const float* newline, float* rows, int width) {
    int row = blockIdx.x, col = row % (width + 1);
    const float* src = col == width ? newline : cells + (row - row / (width + 1)) * output;
    for (int j = threadIdx.x; j < output; j += blockDim.x)
        rows[row * output + j] = src[j];
}

__global__ void embed_rows(float* state, const float* rows, const int* indices) {
    int token = blockIdx.x, row = indices[token];
    if (row < 0)
        return;
    for (int j = threadIdx.x; j < output; j += blockDim.x) {
        float v = rows[row * output + j];
        for (int hc = 0; hc < 4; ++hc)
            state[(token * 4 + hc) * output + j] = v;
    }
}
} // namespace strata::hy4_flash::vision
