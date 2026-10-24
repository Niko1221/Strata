// BF16 draft-head opt-in: unchanged payload, F32 activations, CPU dot-product reference.
#include "gguf_fixture.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <filesystem>
#include <vector>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr int N = 2560, V = 19, K = 7;
    const auto dir = std::filesystem::temp_directory_path() / "strata-dflash-head-bf16-test";
    std::filesystem::create_directories(dir);
    std::vector<uint16_t> weights(N * V);
    std::vector<float> x(N * K), expected(V * K), actual(V * K);
    for (int i = 0; i < N * V; ++i) {
        const float value = (float) ((i * 7) % 17 - 8) / 32;
        uint32_t bits; std::memcpy(&bits, &value, 4); weights[i] = (uint16_t) (bits >> 16);
    }
    for (int i = 0; i < N * K; ++i) x[i] = (float) ((i * 13) % 23 - 11) / 19;
    for (int k = 0; k < K; ++k)
        for (int v = 0; v < V; ++v) {
            double sum = 0;
            for (int i = 0; i < N; ++i)
                sum += (double) ((float) (( (v * N + i) * 7) % 17 - 8) / 32) * x[k * N + i];
            expected[k * V + v] = (float) sum;
        }
    std::vector<uint8_t> raw(weights.size() * 2);
    std::memcpy(raw.data(), weights.data(), raw.size());
    const auto path = dir / "head.gguf";
    const auto written = fixture::write(path, {fixture::str("general.architecture", "qwen4exp"),
        fixture::u32("qwen4exp.block_count", 48), fixture::u32("qwen4exp.embedding_length", N),
        fixture::u32("qwen4exp.expert_count", 512), fixture::u32("qwen4exp.expert_used_count", 10),
        fixture::u32("qwen4exp.attention.head_count", 24), fixture::u32("qwen4exp.attention.head_count_kv", 2)},
        {{"output.weight", {N, V}, 30}});
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(written.data_start); file.write((const char*) raw.data(), raw.size());
    }
    std::string err;
    strata::core::NativeHead head;
    if (head.load({path.string()}, N, V, err)) return 1; // default rejects BF16
    if (!head.load({path.string()}, N, V, err, true)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (head.type() != 30 || head.weight_bytes() != raw.size()) return 1;
    float *dx = nullptr, *dy = nullptr;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMalloc(&dx, x.size() * sizeof(float)); cudaMalloc(&dy, actual.size() * sizeof(float));
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    bool ok = head.run(dx, dy, stream, err);
    cudaStreamSynchronize(stream);
    cudaMemcpy(actual.data(), dy, V * sizeof(float), cudaMemcpyDeviceToHost);
    for (int v = 0; v < V; ++v) ok = ok && std::abs(actual[v] - expected[v]) < 1e-4f;
    strata::kernels::bf16_gemv_fp32_mmvf_multi(dx, N, (const uint16_t*) head.weights(), dy, V, N, V, K, stream);
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;
    cudaMemcpy(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < actual.size(); ++i) ok = ok && std::abs(actual[i] - expected[i]) < 1e-4f;
    cudaFree(dx); cudaFree(dy); cudaStreamDestroy(stream);
    std::filesystem::remove_all(dir);
    if (!ok) std::fprintf(stderr, "BF16 head mismatch: %s\n", err.c_str());
    return ok ? 0 : 1;
}
