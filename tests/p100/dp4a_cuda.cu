#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"
#include "dp4a_cases.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
__device__ __forceinline__ int scalar_dot(int a, int b, int c) {
    const int8_t *av = reinterpret_cast<const int8_t *>(&a);
    const int8_t *bv = reinterpret_cast<const int8_t *>(&b);
    return c + av[0]*bv[0] + av[1]*bv[1] + av[2]*bv[2] + av[3]*bv[3];
}
__global__ void parity(const p100_test::Case *cases, int *output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto v = cases[i];
    output[5*i] = STRATA_DP4A(v.a, v.b, v.c);
    output[5*i+1] = STRATA_DP4A(v.a, v.b, v.a);
    output[5*i+2] = STRATA_DP4A(v.a, v.b, v.b);
    output[5*i+3] = STRATA_DP4A(v.a, v.a, v.c);
    output[5*i+4] = STRATA_DP4A(v.a, v.a, v.a);
}
// Every partial sum stays inside int32: 4096 dots * 65536 < 2^31.
template<bool Candidate, int Chains>
__global__ void timing(const p100_test::Case *cases, int *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int sums[Chains] = {};
    const auto v = cases[i];
    for (int j = 0; j < 4096; ++j) {
#pragma unroll
        for (int k = 0; k < Chains; ++k) {
            int a = v.a ^ (j * 257 + k);
            int b = v.b ^ (j * 17 + k * 31);
            sums[k] = Candidate ? STRATA_DP4A(a, b, sums[k]) : scalar_dot(a, b, sums[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < Chains; ++k) output[Chains*i+k] = sums[k];
}
template<int Chains>
static void benchmark(const p100_test::Case *input, int n) {
    int *base, *candidate;
    check(cudaMalloc(&base, sizeof(int)*n*Chains));
    check(cudaMalloc(&candidate, sizeof(int)*n*Chains));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    auto launch = [&](bool arm) {
        if (arm) timing<true, Chains><<<(n+255)/256, 256>>>(input, candidate, n);
        else timing<false, Chains><<<(n+255)/256, 256>>>(input, base, n);
        check(cudaGetLastError());
    };
    launch(false); launch(true); check(cudaDeviceSynchronize());
    std::vector<int> a(n*Chains), b(n*Chains);
    check(cudaMemcpy(a.data(), base, sizeof(int)*a.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), candidate, sizeof(int)*b.size(), cudaMemcpyDeviceToHost));
    if (a != b) { std::fprintf(stderr, "benchmark arithmetic mismatch\n"); std::exit(1); }
    for (int round = 0; round < 8; ++round) for (int slot = 0; slot < 4; ++slot) {
        bool arm = (slot == 1 || slot == 2) ^ bool(round & 1);
        check(cudaEventRecord(start)); launch(arm); check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms; check(cudaEventElapsedTime(&ms, start, stop));
        std::printf("%d,%d,%d,%s,%d,4096,%.9g\n", round, slot, Chains,
                    arm ? "vmad" : "scalar", n, ms);
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaFree(base)); check(cudaFree(candidate));
}
int main(int argc, char **argv) {
    if (argc != 2 || std::string(argv[1]) != "--run-on-authorized-p100") {
        std::fprintf(stderr, "Deferred GPU test. After authorization: %s --run-on-authorized-p100\n", argv[0]);
        return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
    }
    // All GPU API calls are below the explicit execution opt-in.
    cudaDeviceProp p; check(cudaGetDeviceProperties(&p, 0));
    if (p.major != 6 || p.minor != 0 || std::string(p.name).find("P100") == std::string::npos) {
        std::fprintf(stderr, "Requires Tesla P100 sm_60; got %s\n", p.name); return 2;
    }
    int driver, runtime;
    check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::fprintf(stderr, "gpu=%s cc=%d.%d driver_api=%d runtime=%d\n", p.name, p.major, p.minor, driver, runtime);
    const auto cases = p100_test::cases();
    p100_test::Case *device_cases; int *device_out;
    check(cudaMalloc(&device_cases, cases.size()*sizeof(cases[0])));
    check(cudaMalloc(&device_out, cases.size()*5*sizeof(int)));
    check(cudaMemcpy(device_cases, cases.data(), cases.size()*sizeof(cases[0]), cudaMemcpyHostToDevice));
    parity<<<(cases.size()+255)/256, 256>>>(device_cases, device_out, cases.size());
    check(cudaGetLastError());
    std::vector<int> output(cases.size()*5);
    check(cudaMemcpy(output.data(), device_out, output.size()*sizeof(int), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto v = cases[i];
        const p100_test::Case aliases[] = {v, {v.a,v.b,v.a}, {v.a,v.b,v.b}, {v.a,v.a,v.c}, {v.a,v.a,v.a}};
        for (int k = 0; k < 5; ++k) if (output[5*i+k] != p100_test::oracle(aliases[k].a, aliases[k].b, aliases[k].c)) {
            std::fprintf(stderr, "device parity mismatch at case %zu variant %d\n", i, k); return 1;
        }
    }
    std::fprintf(stderr, "PASS: %zu device cases\n", output.size());
    std::puts("round,slot,chains,arm,threads,dots_per_chain,milliseconds");
    benchmark<1>(device_cases, 4096); benchmark<4>(device_cases, 4096);
    check(cudaFree(device_out)); check(cudaFree(device_cases));
}
