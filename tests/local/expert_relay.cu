// Exercises the prefill relay's slot reuse, stream fences, and copied bytes on the supported NVLink pair.
#include "strata/core/local_expert_relay.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
bool check(cudaError_t rc, const char* what) {
    if (rc == cudaSuccess) return true;
    std::fprintf(stderr, "expert_relay_test: %s: %s\n", what, cudaGetErrorString(rc));
    return false;
}
#define CHECK(call) do { if (!check((call), #call)) return 1; } while (0)
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count != 2) {
        std::puts("SKIP: relay test requires exactly two visible GPUs");
        return 77;
    }
    if (setenv("STRATA_LOCAL_PREFILL_RELAY_TEST", "1", 1) != 0) return 1;

    strata::core::LocalExpertRelay relay;
    const cudaError_t open = relay.open("STRATA_LOCAL_PREFILL_RELAY_TEST");
    if (open == cudaErrorNotSupported || open == cudaErrorPeerAccessUnsupported) {
        std::puts("SKIP: relay test requires two RTX 2080 Ti GPUs with active bidirectional NVLink");
        return 77;
    }
    CHECK(open);
    if (!relay.active()) return 1;

    constexpr size_t sizes[] = {64, 1024 * 1024, 3 * 1024 * 1024 + 128,
                                5 * 1024 * 1024 + 16, 1024 * 1024 + 256};
    constexpr int batches = 4;
    size_t total = 0;
    for (const size_t n : sizes) total += n;
    unsigned char *src = nullptr, *back = nullptr, *dst = nullptr;
    cudaStream_t stream = nullptr;
    CHECK(cudaSetDevice(0));
    CHECK(cudaHostAlloc(&src, batches * total, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&back, batches * total, cudaHostAllocPortable));
    CHECK(cudaMalloc(&dst, total));
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (size_t i = 0; i < batches * total; ++i) src[i] = (unsigned char) ((i * 131 + (i >> 11) * 17) % 251 + 1);

    for (int batch = 0; batch < batches; ++batch) {
        CHECK(cudaMemsetAsync(dst, 0, total, stream));
        CHECK(relay.begin(stream, 0));
        size_t offset = 0;
        for (const size_t n : sizes) {
            CHECK(relay.copy(dst + offset, src + (size_t) batch * total + offset, n, stream, 0));
            offset += n;
        }
        CHECK(cudaMemcpyAsync(back + (size_t) batch * total, dst, total, cudaMemcpyDeviceToHost, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(src, back, batches * total) != 0) {
        std::fprintf(stderr, "expert_relay_test: relay output differs from source\n");
        return 2;
    }
    if (relay.copies() != (size_t) batches * (sizeof(sizes) / sizeof(sizes[0])) ||
        relay.bytes() != (size_t) batches * total) {
        std::fprintf(stderr, "expert_relay_test: accounting mismatch: %zu copies, %zu bytes\n",
                     relay.copies(), relay.bytes());
        return 3;
    }
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(dst));
    CHECK(cudaFreeHost(back));
    CHECK(cudaFreeHost(src));
    std::printf("PASS: %zu payloads, %zu bytes, ring reuse and receiver-stream ordering verified\n",
                relay.copies(), relay.bytes());
    return 0;
}
