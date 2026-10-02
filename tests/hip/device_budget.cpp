#include <cuda_runtime.h>
#include <cstdio>

int main() {
    void* rejected = nullptr;
    // 10 GiB is denied by the reservation policy BEFORE asking HIP to allocate.
    if (cudaMalloc(&rejected, 10240ull << 20) != hipErrorOutOfMemory || rejected != nullptr) return 1;
    (void) cudaGetLastError();
    if (hipMallocManaged(&rejected, 65536, hipMemAttachGlobal) != hipErrorNotSupported) return 7;
    if (hipMallocAsync(&rejected, 65536, nullptr) != hipErrorNotSupported) return 8;
    size_t before = 0, total = 0, during = 0, after = 0;
    if (cudaMemGetInfo(&before, &total) != cudaSuccess) return 2;
    void* small = nullptr;
    if (cudaMalloc(&small, 1ull << 20) != cudaSuccess) return 3;
    if (cudaMemGetInfo(&during, &total) != cudaSuccess || during >= before) return 4;
    if (cudaFree(small) != cudaSuccess) return 5;
    if (cudaMemGetInfo(&after, &total) != cudaSuccess || after != before) return 6;
    std::puts("HIP budget refused oversized allocation and restored headroom after free");
}
