// CUDA's logical 32-lane warp contract on a physical AMD wave64, including
// gfx900's software dp4a and overflow. No model or GGUF fixture needed.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

struct Result { int dot, shuffle, down, exchange; unsigned ballot, perm; };

__global__ void probe(Result* out) {
    const int lane = threadIdx.x;
    const unsigned a = 0x80ff017fu ^ (unsigned(lane) * 0x01020408u);
    const unsigned b = 0x0203fe81u ^ (unsigned(lane) * 0x08040201u);
    __shared__ int exchange[64];
    exchange[lane] = lane * 3;
    __syncwarp();
    out[lane] = {__dp4a(int(a), int(b), 0x7fffffff - lane),
                 __shfl_sync(0xffffffffu, lane, 0),
                 __shfl_down_sync(0xffffffffu, lane, 1), exchange[lane ^ 1],
                 __ballot_sync(0xffffffffu, lane == 3 || lane == 37),
                 __byte_perm(a, b, 0x6420)};
}

int main() {
    Result* device = nullptr;
    if (cudaMalloc(&device, 64 * sizeof(Result)) != cudaSuccess) return 2;
    probe<<<1, 64>>>(device);
    Result got[64];
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(got, device, sizeof(got), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
    if (cudaFree(device) != cudaSuccess) return 2;
    for (int lane = 0; lane < 64; ++lane) {
        const unsigned a = 0x80ff017fu ^ (unsigned(lane) * 0x01020408u);
        const unsigned b = 0x0203fe81u ^ (unsigned(lane) * 0x08040201u);
        unsigned sum = unsigned(0x7fffffff - lane);
        unsigned perm = 0;
        for (int i = 0; i < 4; ++i) {
            const int x = int((a >> (8 * i)) & 255u) - ((a >> (8 * i)) & 128u ? 256 : 0);
            const int y = int((b >> (8 * i)) & 255u) - ((b >> (8 * i)) & 128u ? 256 : 0);
            sum += unsigned(x * y);
            const int byte = (0x6420 >> (4 * i)) & 7;
            perm |= (((byte < 4 ? a : b) >> (8 * (byte & 3))) & 255u) << (8 * i);
        }
        unsigned dot;
        std::memcpy(&dot, &got[lane].dot, sizeof(dot));
        if (dot != sum || got[lane].shuffle != (lane & ~31) ||
            got[lane].down != ((lane & 31) == 31 ? lane : lane + 1) ||
            got[lane].exchange != (lane ^ 1) * 3 ||
            got[lane].ballot != (lane < 32 ? 8u : 32u) || got[lane].perm != perm) {
            std::fprintf(stderr, "FAIL wave64 intrinsics lane %d\n", lane);
            return 1;
        }
    }
    std::puts("PASS wave64: signed dp4a overflow, half-wave shuffles/ballots, LDS barrier, byte permutation");
    return 0;
}
