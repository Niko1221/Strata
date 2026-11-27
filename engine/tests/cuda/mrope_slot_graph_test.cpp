#include "strata/core/on_device.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cstdio>
#include <stdexcept>

#define CUDA_OK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 1; } } while (0)

int check_device(int device) {
    const strata::core::OnDevice on(device);
    using namespace strata::kernels;
    constexpr int cells = 64, dim = 128, count = 3;
    float *input = nullptr, *output = nullptr;
    int32_t *table_a = nullptr, *table_b = nullptr, *pos = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_OK(cudaMalloc(&input, count * dim * sizeof(float)));
    CUDA_OK(cudaMalloc(&output, count * dim * sizeof(float)));
    CUDA_OK(cudaMalloc(&table_a, cells * 3 * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&table_b, cells * 3 * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&pos, sizeof(int32_t)));
    struct Cleanup {
        float *x, *y; int32_t *a, *b, *p; cudaStream_t s;
        ~Cleanup() { mrope_table_set(nullptr); cudaStreamSynchronize(s); cudaFree(x); cudaFree(y);
            cudaFree(a); cudaFree(b); cudaFree(p); cudaStreamDestroy(s); }
    } cleanup{input, output, table_a, table_b, pos, stream};
    std::vector<float> x(count * dim), expected(count * dim), actual(count * dim);
    for (int i = 0; i < count * dim; ++i) x[i] = (i % 31 - 15) * 0.0625f;
    std::vector<int32_t> a(cells * 3), b(cells * 3);
    for (int i = 0; i < cells; ++i) {
        a[i*3]=2; a[i*3+1]=i/4+2; a[i*3+2]=i%4+2;
        b[i*3]=7; b[i*3+1]=i/8+7; b[i*3+2]=i%8+7;
    }
    int32_t position = 5;
    CUDA_OK(cudaMemcpyAsync(input, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(table_a, a.data(), a.size()*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(table_b, b.data(), b.size()*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(pos, &position, sizeof(position), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    const int32_t* tables[] = {table_a, table_b, nullptr};
    RopeScaling scaling;
    auto record = [&] {
        for (int i = 0; i < count; ++i) {
            MropeScope positions(tables[i]);
            native_rope_apply(input+i*dim, output+i*dim, 1, dim, 64, scaling, pos, stream);
        }
    };
    record();
    CUDA_OK(cudaMemcpyAsync(expected.data(), output, expected.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    mrope_table_set(table_b);
    try { MropeScope scope(table_a); throw std::runtime_error("test"); }
    catch (const std::runtime_error&) {}
    if (mrope_table() != table_b) return 2;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    record();
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
    cudaGraphDestroy(graph);
    struct Destroy { cudaGraphExec_t e; ~Destroy() { cudaGraphExecDestroy(e); } } destroy{exec};
    if (mrope_table() != table_b) return 3;
    // Later admissions may change the device-wide selection. Existing graphs keep each slot's table.
    for (const int32_t* foreground : tables) {
        mrope_table_set(foreground);
        CUDA_OK(cudaGraphLaunch(exec, stream));
        CUDA_OK(cudaMemcpyAsync(actual.data(), output, actual.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        if (std::memcmp(expected.data(), actual.data(), actual.size()*sizeof(float)) != 0) return 4;
    }
    // Negative control: selecting the wrong table must be numerically distinguishable.
    mrope_table_set(table_b);
    native_rope_apply(input, output, 1, dim, 64, scaling, pos, stream);
    CUDA_OK(cudaMemcpyAsync(actual.data(), output, dim*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    if (std::memcmp(expected.data(), actual.data(), dim*sizeof(float)) == 0) return 5;
    return 0;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    for (int d = 0; d < devices; ++d) { int result = check_device(d); if (result) return result; }
    return 0;
}
