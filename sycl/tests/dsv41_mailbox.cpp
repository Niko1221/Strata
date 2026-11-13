// Exercise the running-kernel / CPU expert mailbox, including routed payloads.
#include "ds4_kernels.dp.hpp"
#include "strata/sycl_queue.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <thread>

int main() try {
    auto& q = dpct::get_in_order_queue();
    using namespace strata::ds4;
    auto* request = static_cast<Req*>(strata::host_malloc_polled(sizeof(Req), q));
    auto* response = static_cast<Resp*>(strata::host_malloc_polled(sizeof(Resp), q));
    if (!request || !response) return 2;
    std::memset(request, 0, sizeof(Req)); std::memset(response, 0, sizeof(Resp));
    auto* logits = sycl::malloc_shared<float>(1024, q);
    auto* bias = sycl::malloc_shared<float>(384, q);
    auto* vram = sycl::malloc_shared<unsigned long long>(384, q);
    auto* input = sycl::malloc_shared<float>(5120, q);
    auto* indices = sycl::malloc_shared<int>(8, q);
    auto* weights = sycl::malloc_shared<float>(8, q);
    auto* plan = sycl::malloc_shared<unsigned long long>(8, q);
    auto* result = sycl::malloc_shared<float>(1, q);
    for (int i = 0; i < 1024; ++i) logits[i] = float(i) / 64;
    for (int i = 0; i < 384; ++i) bias[i] = 0, vram[i] = 0;
    for (int i = 0; i < 5120; ++i) input[i] = float(i % 31) / 32;
    size_t failures = 0;
    std::thread cpu([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (*reinterpret_cast<volatile unsigned*>(&request->seq) != 7 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        std::atomic_thread_fence(std::memory_order_acquire);
        if (request->seq != 7 || request->T != 1) ++failures;
        for (int i = 0; i < 6; ++i) if (request->ids[0][i] != 383 - i) ++failures;
        for (int i = 0; i < 5120; ++i) if (request->x[0][i] != input[i]) ++failures;
        response->part[0][0] = 123.5f;
        std::atomic_thread_fence(std::memory_order_release);
        *reinterpret_cast<volatile unsigned*>(&response->seq) = 7;
    });
    route_publish(logits, bias, 384, 6, 1, 1.5f, vram, input, 5120, indices, weights, plan, request, 7, &q);
    wait_seq(&response->seq, 7, &q);
    q.single_task([=] { *result = response->part[0][0]; });
    q.wait_and_throw();
    cpu.join();
    if (*result != 123.5f) ++failures;
    std::printf("DeepSeek CPU/GPU mailbox: top-6, 5120-value request, response visibility; %zu failures\n", failures);
    strata::host_free_polled(request, q); strata::host_free_polled(response, q);
    sycl::free(logits, q); sycl::free(bias, q); sycl::free(vram, q); sycl::free(input, q);
    sycl::free(indices, q); sycl::free(weights, q); sycl::free(plan, q); sycl::free(result, q);
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what()); return 2;
}
