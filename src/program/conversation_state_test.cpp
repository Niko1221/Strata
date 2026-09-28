// Synthetic CUDA + host round trip; no weights required. Checks remain active in Release builds.
#include "strata/program/conversation_state.hpp"
#include <array>
#include <cstdio>
#include <limits>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    using namespace strata::program;
    ConversationState cache;
    CHECK(!cache.restore());
    CHECK(!cache.save({}, 256, 4096, 256));
    CHECK(!cache.save({{nullptr, 1, false}}, 256, 4096, 256));
    unsigned char* device = nullptr;
    CHECK(cudaMalloc((void**)&device, 128) == cudaSuccess);
    std::array<unsigned char, 128> host{}, out{};
    host.fill(19);
    CHECK(cudaMemset(device, 37, 128) == cudaSuccess);
    std::vector<ConversationRange> ranges = {{device, 128, true}, {nullptr, 0, false}, {host.data(), 128, false}};
    CHECK(!cache.save(ranges, 255, 4096, 256) && cache.bytes() == 0);
    CHECK(!cache.save(ranges, 256, 511, 256) && cache.bytes() == 0);
    CHECK(!cache.save(ranges, 256, 255, 256) && cache.bytes() == 0);
    const auto max = std::numeric_limits<uint64_t>::max();
    CHECK(!cache.save({{host.data(), (size_t) max, false}, {host.data(), 1, false}}, max, max, 0));
    CHECK(cache.save(ranges, 256, 512, 256) && cache.bytes() == 256);
    CHECK(cudaMemset(device, 99, 128) == cudaSuccess);
    host.fill(77);
    CHECK(cache.restore());
    CHECK(cudaMemcpy(out.data(), device, 128, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (auto value : out) CHECK(value == 37);
    for (auto value : host) CHECK(value == 19);
    // A new snapshot replaces the old one, including the saved values.
    host.fill(42);
    CHECK(cache.save(ranges, 256, 512, 256));
    host.fill(0);
    CHECK(cache.restore());
    for (auto value : host) CHECK(value == 42);
    // Failed admission must not leave an earlier snapshot available to restore.
    CHECK(!cache.save(ranges, 255, 4096, 256));
    CHECK(cache.bytes() == 0 && !cache.restore());
    cache.clear();
    CHECK(cache.bytes() == 0 && !cache.restore());
    CHECK(cudaFree(device) == cudaSuccess);
    std::puts("PASS: host/device restoration, replacement, admission bounds and invalidation");
    return 0;
}
