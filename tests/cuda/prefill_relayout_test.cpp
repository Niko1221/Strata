#include "strata/prefill/prefill.hpp"
#include "strata/core/live_prefill.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
class UnusedSource final : public strata::core::ExpertSource {
public:
    const uint8_t* blob(int64_t, int64_t) override { throw std::runtime_error("allocator fixture must not request model weights"); }
};
int checks = 0;
void require(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
}

int main() {
    cudaStream_t stream = nullptr;
    void* backing = nullptr;
    try {
        strata::core::ModelGeometry geometry;
        strata::core::WeightTable weights;
        strata::core::SessionState session;
        strata::core::QsaState qsa{};
        qsa.max_cells = 65536;
        session.qsa_states = &qsa;
        // No model weights or inference. This is the real borrowed Prefill allocator
        // with normal geometry and a fixed-KV descriptor, followed by real relayouts.
        UnusedSource supplied;
        for (bool with_source : {false, true}) {
            strata::core::ExpertSource* src = with_source ? &supplied : nullptr;
            const uint64_t large = strata::prefill::Prefill::bytes_needed(geometry, session, 1024, with_source);
            const uint64_t small = strata::prefill::Prefill::bytes_needed(geometry, session, 256, with_source);
            require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
                    cudaMalloc(&backing, large) == cudaSuccess, "allocate bounded Prefill fixture");
            {
                strata::prefill::Prefill prefill;
                std::string err;
                const bool initialized = prefill.init(weights, geometry, session, src, nullptr, nullptr, 1024,
                                                      stream, err, backing, large);
                require(initialized, err.c_str());
                require(prefill.chunk() == 1024 && cudaDeviceSynchronize() == cudaSuccess, "valid old Prefill layout");
                size_t before = 0, after = 0, total = 0;
                require(cudaMemGetInfo(&before, &total) == cudaSuccess, "measure valid allocator state");
                for (uint64_t short_bytes : std::array<uint64_t, 4>{0, 256, 8ull << 20, small - 1}) {
                    require(!prefill.relayout(256, backing, short_bytes, err), "undersized real carve rejected");
                    require(err == "prefill: relayout loan is smaller than its counted buffers" && prefill.chunk() == 1024,
                            "rejection occurs before the actual carved chunk/view state is published");
                }
                require(cudaDeviceSynchronize() == cudaSuccess && cudaMemGetInfo(&after, &total) == cudaSuccess &&
                        after + (8ull << 20) >= before, "invalid borrowed carves cannot leak nested owned allocations");
                const auto result = strata::core::live_prefill_rebind(
                    [&](std::string& e) { return prefill.relayout(256, backing, small - 1, e); },
                    [&](std::string& e) { return prefill.relayout(1024, backing, large, e); }, err);
                require(result == strata::core::LivePrefillRebind::restored && prefill.chunk() == 1024,
                        "actual failed chosen layout followed by actual valid old relayout");
                require(prefill.relayout(256, backing, small, err) && prefill.chunk() == 256,
                        "valid minimum loan still carves successfully after rejection");
                require(prefill.relayout(1024, backing, large, err) && prefill.chunk() == 1024,
                        "actual Prefill recovers its originally initialized chunk");
            }
            cudaFree(backing); cudaStreamDestroy(stream);
            backing = nullptr; stream = nullptr;
        }
        std::printf("prefill relayout: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        if (backing) cudaFree(backing);
        if (stream) cudaStreamDestroy(stream);
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
