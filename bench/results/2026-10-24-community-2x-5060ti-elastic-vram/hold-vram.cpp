#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto check = [](cudaError_t result) {
        if (result != cudaSuccess) {
            std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
            std::exit(1);
        }
    };
    check(cudaSetDevice(0));
    void* memory = nullptr;
    check(cudaMalloc(&memory, std::strtoull(argv[1], nullptr, 10) * 1024 * 1024));
    std::puts("READY");
    std::fflush(stdout);
    std::getchar();
    check(cudaFree(memory));
}
