#include "strata/core/branch_sha256.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <exception>
#include <limits>

namespace strata::core {
namespace {
struct Counters {
    std::atomic<uint64_t> device_hashed_bytes{0}, digest_copy_bytes{0}, kernel_calls{0};
    std::atomic<uint64_t> workspace_allocations{0}, workspace_releases{0}, workspace_live_bytes{0};
};
Counters& counters() { static auto* value = new Counters; return *value; }
uint64_t tiles(uint64_t bytes) { return bytes ? (bytes - 1) / branch_digest_tile_bytes + 1 : 1; }
__global__ void hash_tiles(const uint8_t* input, uint64_t bytes, uint64_t count, BranchDigest* result) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto offset = index * branch_digest_tile_bytes;
    const auto remaining = bytes - offset;
    const auto length = remaining < branch_digest_tile_bytes ? remaining : branch_digest_tile_bytes;
    branch_sha256(input + offset, length, result[index]);
}
}
BranchDigestCounters branch_digest_counters() {
    const auto& c = counters();
    return {c.device_hashed_bytes.load(), c.digest_copy_bytes.load(), c.kernel_calls.load(),
            c.workspace_allocations.load(), c.workspace_releases.load(), c.workspace_live_bytes.load()};
}
BranchDigestWorkspace::~BranchDigestWorkspace() {
    if (!device_) return;
    if (cudaFree(device_) != cudaSuccess) std::terminate();
    ++counters().workspace_releases;
    counters().workspace_live_bytes -= capacity_ * sizeof(BranchDigest);
}
bool BranchDigestWorkspace::reserve(uint64_t bytes) {
    const auto count = tiles(bytes);
    if (count <= capacity_) return true;
    // Bound the launch dimension and byte multiplication before any allocation.
    if (count > uint64_t(std::numeric_limits<int>::max()) * 128 ||
        count > std::numeric_limits<size_t>::max() / sizeof(BranchDigest)) return false;
    BranchDigest* next = nullptr;
    if (cudaMalloc(&next, count * sizeof(BranchDigest)) != cudaSuccess) return false;
    if (device_) {
        if (cudaFree(device_) != cudaSuccess) std::terminate();
        ++counters().workspace_releases;
        counters().workspace_live_bytes -= capacity_ * sizeof(BranchDigest);
    }
    device_ = next; capacity_ = count;
    ++counters().workspace_allocations;
    counters().workspace_live_bytes += count * sizeof(BranchDigest);
    return true;
}
bool BranchDigestWorkspace::digest(const uint8_t* input, uint64_t bytes, std::vector<BranchDigest>& result) {
    const auto count = tiles(bytes);
    if (!input || !device_ || count > capacity_) return false;
    std::vector<BranchDigest> output;
    try { output.resize(count); } catch (...) { return false; }
    hash_tiles<<<unsigned((count + 127) / 128), 128>>>(input, bytes, count, device_);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
    ++counters().kernel_calls;
    counters().device_hashed_bytes += bytes;
    if (cudaMemcpy(output.data(), device_, count * sizeof(BranchDigest), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    counters().digest_copy_bytes += count * sizeof(BranchDigest);
    result.swap(output);
    return true;
}
} // namespace strata::core
