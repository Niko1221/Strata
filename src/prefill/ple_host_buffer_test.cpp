// No model or pack needed. Exercise real DMA while growing/reusing PLE host storage.
#include "ple_host_buffer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

static void CUDART_CB delay(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(15)); }

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::puts("SKIP: CUDA device unavailable");
        return 77;
    }
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    float* device = nullptr;
    CHECK(cudaMalloc((void**) &device, 1024 * 2560 * sizeof(float)) == cudaSuccess);
    int cases = 0;
    for (int mode = 0; mode < 4; ++mode) {
        float* data[2] = {};
        size_t capacity[2] = {};
        std::vector<float> pageable[2];
        cudaEvent_t copied[2] = {};
        for (auto& e : copied) CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess);
        const size_t rows[] = {32, 16, 64, 64, 128, 32, 512, 256};
        for (int step = 0; step < 8; ++step) for (int b = 0; b < 2; ++b) {
            const bool fallback = mode == 1 || (mode == 2 && step % 2 == 0) || (mode == 3 && step % 2 != 0);
            const auto ensure = [&](size_t n) {
                return strata::prefill::detail::ensure_ple_host(data[b], pageable[b], capacity[b], n, copied[b], fallback);
            };
            CHECK(ensure(rows[step] * 2560));
            float* old = data[b];
            const size_t old_count = capacity[b];
            const float value = float(1 + mode * 100 + step * 2 + b);
            std::fill(old, old + old_count, value);
            CHECK(cudaLaunchHostFunc(stream, delay, nullptr) == cudaSuccess);
            CHECK(cudaMemcpyAsync(device, old, old_count * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
            CHECK(cudaEventRecord(copied[b], stream) == cudaSuccess);
            // A growing allocation must wait for the deliberately delayed DMA.
            // A retained allocation is not modified again until the stream is done.
            const size_t need = old_count + (step % 2);
            CHECK(ensure(need));
            CHECK(capacity[b] >= need);
            if (need == old_count) CHECK(old == data[b]);
            else CHECK(pageable[b].empty() != fallback);
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
            std::vector<float> out(old_count);
            CHECK(cudaMemcpy(out.data(), device, old_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(std::all_of(out.begin(), out.end(), [&](float x) { return x == value; }));
            ++cases;
        }
        for (int b = 0; b < 2; ++b) {
            CHECK(cudaEventSynchronize(copied[b]) == cudaSuccess);
            if (pageable[b].empty()) CHECK(cudaFreeHost(data[b]) == cudaSuccess);
            CHECK(cudaEventDestroy(copied[b]) == cudaSuccess);
        }
    }
    CHECK(cudaFree(device) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::printf("PASS: %d delayed-DMA growth/reuse cases, pinned/pageable transitions\n", cases);
}
