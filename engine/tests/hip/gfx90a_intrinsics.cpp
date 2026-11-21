// gfx90a primitive checks through the production wave64 compatibility layer.
// Full logical32 masks only. Half-wave cases retire a complete logical warp;
// they never shuffle from an inactive source or promise arbitrary partial masks.
#include "strata/platform/hip_compat/strata_hip.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(STRATA_HIP_GFX90A) || !defined(STRATA_HIP_WAVE64) || defined(STRATA_HIP_GFX906)
#error "Build this test with the gfx90a production target's wave64 definitions"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)
#error "The intrinsics probe must contain gfx90a device code"
#endif

#define HIP_CHECK(call) do { \
    const hipError_t error = (call); \
    if (error != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
        return 2; \
    } \
} while (0)

namespace {
constexpr unsigned kFull = 0xffffffffu;
constexpr int kMaxThreads = 256, kMaxWidth = 64, kRounds = 64, kWordCases = 4096;

struct Votes {
    int physical_lane, wave_size, all, any, lds_sum, sleep_done;
    unsigned ballot, active;
};
struct Shuffles {
    int broadcast, xored, up, down;
    float float_broadcast;
    double double_xor;
};
struct Words { uint32_t a, b, c, selector; };
struct WordResults { uint32_t signed_dot, unsigned_dot, perm, sub, sat_sub, equal, not_equal; };

__global__ void votes_and_wave_lds(Votes* out, int live_half) {
    const int tid = threadIdx.x, lane = tid & 31;
    if (live_half >= 0 && ((tid & 63) / 32) != live_half) return;
    Votes r{};
    r.physical_lane = __lane_id();
    r.wave_size = warpSize;
    // The two halves deliberately disagree: a physical-wave vote is wrong here.
    r.all = __all_sync(kFull, (tid & 32) == 0);
    r.any = __any_sync(kFull, (tid & 32) != 0 && lane == 7);
    r.ballot = __ballot_sync(kFull, (tid & 32) ? lane % 5 == 2 : lane % 3 == 1);
    r.active = __activemask();
    __shared__ volatile int exchange[kMaxThreads];
    const int peer = (tid & ~31) + ((lane + 17) & 31);
    for (int i = 0; i < kRounds; ++i) {
        exchange[tid] = 1000 + tid + i;
        __syncwarp(kFull);
        r.lds_sum += exchange[peer];
        __syncwarp(kFull); // no lane overwrites the next round before its peer reads
    }
    __nanosleep(100); // Completion only; this is not a clock-rate measurement.
    r.sleep_done = 1;
    out[tid] = r;
}

__global__ void shuffles(Shuffles* out, int width, int live_half) {
    const int tid = threadIdx.x;
    if (live_half >= 0 && ((tid & 63) / 32) != live_half) return;
    const int count = width ? width : 32;
    for (int source = 0; source < count; ++source) {
        Shuffles r{};
        if (width == 0) { // Exercise the production shim's default, not just width=32.
            r.broadcast = __shfl_sync(kFull, 1000 + tid, source);
            r.xored = __shfl_xor_sync(kFull, 1000 + tid, source);
            r.up = __shfl_up_sync(kFull, 1000 + tid, source);
            r.down = __shfl_down_sync(kFull, 1000 + tid, source);
            r.float_broadcast = __shfl_sync(kFull, tid + 0.5f, source);
            r.double_xor = __shfl_xor_sync(kFull, tid + 0.25, source);
        } else {
            // width64 is tested separately, only with complete physical waves.
            r.broadcast = __shfl_sync(kFull, 1000 + tid, source, width);
            r.xored = __shfl_xor_sync(kFull, 1000 + tid, source, width);
            r.up = __shfl_up_sync(kFull, 1000 + tid, source, width);
            r.down = __shfl_down_sync(kFull, 1000 + tid, source, width);
            r.float_broadcast = __shfl_sync(kFull, tid + 0.5f, source, width);
            r.double_xor = __shfl_xor_sync(kFull, tid + 0.25, source, width);
        }
        out[tid * kMaxWidth + source] = r;
    }
}

__global__ void block_lds(int* out) {
    __shared__ volatile int exchange[kMaxThreads];
    const int tid = threadIdx.x, peer = (tid + blockDim.x / 2) % blockDim.x;
    int sum = 0;
    for (int i = 0; i < kRounds; ++i) {
        exchange[tid] = 512 * i + tid;
        __syncthreads(); // Unlike __syncwarp, this exchange can cross physical waves.
        sum += exchange[peer];
        __syncthreads();
    }
    out[tid] = sum;
}

__global__ void word_intrinsics(const Words* in, WordResults* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kWordCases) return;
    const Words w = in[i];
    WordResults r{};
    r.signed_dot = (uint32_t) __dp4a((int) w.a, (int) w.b, (int) w.c);
    r.unsigned_dot = __dp4a(w.a, w.b, w.c);
    r.perm = __byte_perm(w.a, w.b, w.selector);
    r.sub = (uint32_t) __vsub4(w.a, w.b);
    r.sat_sub = (uint32_t) __vsubss4(w.a, w.b);
    r.equal = __vcmpeq4(w.a, w.b);
    r.not_equal = __vcmpne4(w.a, w.b);
    out[i] = r;
}

bool equal(const char* what, double got, double want, int lane, int width = 0, int source = -1) {
    if (got == want) return true;
    std::fprintf(stderr, "FAIL %s lane/case=%d width=%d source=%d got=%.17g want=%.17g\n",
                 what, lane, width, source, got, want);
    return false;
}

bool untouched(const void* p, size_t bytes) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < bytes; ++i) if (b[i] != 0xa5) return false;
    return true;
}

int signed_byte(uint32_t word, int byte) {
    const int v = (word >> (8 * byte)) & 255;
    return v < 128 ? v : v - 256;
}

WordResults word_reference(const Words& w) {
    // Unsigned arithmetic expresses the CUDA dot contract modulo 2^32 without
    // signed C++ overflow. No device decoder/helper is used by this oracle.
    WordResults r{w.c, w.c, 0, 0, 0, 0, 0};
    const uint64_t pair = (uint64_t(w.b) << 32) | w.a;
    for (int byte = 0; byte < 4; ++byte) {
        const unsigned shift = 8 * byte;
        const unsigned a = (w.a >> shift) & 255, b = (w.b >> shift) & 255;
        r.signed_dot += uint32_t(signed_byte(w.a, byte) * signed_byte(w.b, byte));
        r.unsigned_dot += a * b;
        const unsigned index = (w.selector >> (4 * byte)) & 7;
        r.perm |= uint32_t((pair >> (8 * index)) & 255) << shift;
        r.sub |= ((a - b) & 255) << shift;
        int sat = signed_byte(w.a, byte) - signed_byte(w.b, byte);
        if (sat < -128) sat = -128;
        if (sat > 127) sat = 127;
        r.sat_sub |= (uint32_t(sat) & 255) << shift;
        r.equal |= uint32_t(a == b ? 255 : 0) << shift;
        r.not_equal |= uint32_t(a != b ? 255 : 0) << shift;
    }
    return r;
}
} // namespace

int main() {
    int count = 0;
    const hipError_t discovery = hipGetDeviceCount(&count);
    if (discovery == hipErrorNoDevice || (discovery == hipSuccess && count == 0)) {
        std::puts("SKIP gfx90a_intrinsics: no HIP device (M2 requires execution on MI210)");
        return 77;
    }
    HIP_CHECK(discovery); // Driver/runtime failures are errors, not missing hardware.
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    if (std::strncmp(prop.gcnArchName, "gfx90a", 6) != 0 ||
        (prop.gcnArchName[6] != '\0' && prop.gcnArchName[6] != ':') || prop.warpSize != 64) {
        std::fprintf(stderr, "FAIL expected gfx90a wave64, got %s wave%d\n", prop.gcnArchName, prop.warpSize);
        return 1;
    }
    std::printf("gfx90a_intrinsics: device=%d arch=%s wave=%d\n", device, prop.gcnArchName, prop.warpSize);

    Votes* dv = nullptr;
    Shuffles* ds = nullptr;
    int* dl = nullptr;
    Words* dw = nullptr;
    WordResults* dr = nullptr;
    HIP_CHECK(hipMalloc((void**) &dv, kMaxThreads * sizeof(Votes)));
    HIP_CHECK(hipMalloc((void**) &ds, kMaxThreads * kMaxWidth * sizeof(Shuffles)));
    HIP_CHECK(hipMalloc((void**) &dl, kMaxThreads * sizeof(int)));
    HIP_CHECK(hipMalloc((void**) &dw, kWordCases * sizeof(Words)));
    HIP_CHECK(hipMalloc((void**) &dr, kWordCases * sizeof(WordResults)));
    std::vector<Votes> votes(kMaxThreads);
    std::vector<Shuffles> shuffled(kMaxThreads * kMaxWidth);
    int launches = 0, checked_lanes = 0, shuffle_values = 0;
    const struct { int threads, live_half; } cases[] = {{32, -1}, {64, -1}, {128, -1}, {256, -1}, {64, 0}, {64, 1}};
    for (const auto c : cases) {
        std::printf("  threads=%d live_half=%d (-1: all)\n", c.threads, c.live_half);
        HIP_CHECK(hipMemset(dv, 0xa5, votes.size() * sizeof(Votes)));
        hipLaunchKernelGGL(votes_and_wave_lds, dim3(1), dim3(c.threads), 0, 0, dv, c.live_half);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        HIP_CHECK(hipMemcpy(votes.data(), dv, votes.size() * sizeof(Votes), hipMemcpyDeviceToHost));
        for (int tid = 0; tid < kMaxThreads; ++tid) {
            const bool active = tid < c.threads && (c.live_half < 0 || (tid & 63) / 32 == c.live_half);
            if (!active) {
                if (!untouched(&votes[tid], sizeof(Votes))) { std::fprintf(stderr, "FAIL inactive vote lane %d written\n", tid); return 1; }
                continue;
            }
            unsigned ballot = 0;
            for (int lane = 0; lane < 32; ++lane)
                if ((tid & 32) ? lane % 5 == 2 : lane % 3 == 1) ballot |= 1u << lane;
            const int peer = (tid / 32) * 32 + ((tid % 32 + 17) % 32);
            const Votes& v = votes[tid];
            if (!equal("physical lane", v.physical_lane, tid % 64, tid) || !equal("device wave size", v.wave_size, 64, tid) ||
                !equal("all logical32", v.all, (tid & 32) == 0, tid) || !equal("any logical32", v.any, (tid & 32) != 0, tid) ||
                !equal("ballot", v.ballot, ballot, tid) || !equal("activemask", v.active, kFull, tid) ||
                !equal("wave LDS", v.lds_sum, kRounds * (1000 + peer) + kRounds * (kRounds - 1) / 2, tid) ||
                !equal("sleep completion", v.sleep_done, 1, tid)) return 1;
            ++checked_lanes;
        }
        for (const int width : {0, 2, 4, 8, 16, 32, 64}) {
            if (width == 64 && (c.threads < 64 || c.live_half >= 0)) continue;
            const int actual_width = width ? width : 32;
            HIP_CHECK(hipMemset(ds, 0xa5, shuffled.size() * sizeof(Shuffles)));
            hipLaunchKernelGGL(shuffles, dim3(1), dim3(c.threads), 0, 0, ds, width, c.live_half);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            ++launches;
            HIP_CHECK(hipMemcpy(shuffled.data(), ds, shuffled.size() * sizeof(Shuffles), hipMemcpyDeviceToHost));
            for (int tid = 0; tid < kMaxThreads; ++tid) for (int source = 0; source < kMaxWidth; ++source) {
                const Shuffles& s = shuffled[tid * kMaxWidth + source];
                const bool active = tid < c.threads && (c.live_half < 0 || (tid & 63) / 32 == c.live_half);
                if (!active || source >= actual_width) {
                    if (!untouched(&s, sizeof(s))) { std::fprintf(stderr, "FAIL inactive shuffle lane %d source %d written\n", tid, source); return 1; }
                    continue;
                }
                const int lane = tid % actual_width, base = tid - lane;
                const int up = lane >= source ? tid - source : tid;
                const int down = lane + source < actual_width ? tid + source : tid;
                if (!equal("broadcast", s.broadcast, 1000 + base + source, tid, width, source) ||
                    !equal("xor", s.xored, 1000 + base + (lane ^ source), tid, width, source) ||
                    !equal("up boundary", s.up, 1000 + up, tid, width, source) ||
                    !equal("down boundary", s.down, 1000 + down, tid, width, source) ||
                    !equal("float broadcast", s.float_broadcast, base + source + 0.5, tid, width, source) ||
                    !equal("double xor", s.double_xor, base + (lane ^ source) + 0.25, tid, width, source)) return 1;
                shuffle_values += 6;
            }
        }
    }
    for (const int threads : {32, 64, 128, 256}) {
        HIP_CHECK(hipMemset(dl, 0xa5, kMaxThreads * sizeof(int)));
        hipLaunchKernelGGL(block_lds, dim3(1), dim3(threads), 0, 0, dl);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        int sums[kMaxThreads];
        HIP_CHECK(hipMemcpy(sums, dl, sizeof(sums), hipMemcpyDeviceToHost));
        for (int tid = 0; tid < kMaxThreads; ++tid) {
            if (tid >= threads) {
                if (!untouched(&sums[tid], sizeof(int))) { std::fprintf(stderr, "FAIL block LDS guard written\n"); return 1; }
            } else if (!equal("block LDS", sums[tid], kRounds * ((tid + threads / 2) % threads) + 512 * kRounds * (kRounds - 1) / 2, tid)) return 1;
        }
    }

    std::vector<Words> words(kWordCases);
    std::vector<WordResults> results(kWordCases);
    uint32_t state = 0x9e3779b9u;
    const auto next = [&state]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (Words& w : words) w = {next(), next(), next(), next()};
    // Signed limits, unsigned limits, accumulator wrap, and both saturation directions.
    words[0] = {0x80808080u, 0x80808080u, 0x7fffffffu, 0xffff3210u};
    words[1] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffff7654u};
    words[2] = {0x7f7f7f7fu, 0x80808080u, 0x80000000u, 0x8888fedcu};
    words[3] = {0x807f0080u, 0x017f0180u, 7u, 0x00007160u};
    words[4] = {0x01234567u, 0x01234567u, 0u, 0u};
    words[5] = {0u, 0xffffffffu, 0u, 0xfffffff8u};
    HIP_CHECK(hipMemcpy(dw, words.data(), words.size() * sizeof(Words), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dr, 0xa5, results.size() * sizeof(WordResults)));
    hipLaunchKernelGGL(word_intrinsics, dim3((kWordCases + 255) / 256), dim3(256), 0, 0, dw, dr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    ++launches;
    HIP_CHECK(hipMemcpy(results.data(), dr, results.size() * sizeof(WordResults), hipMemcpyDeviceToHost));
    for (int i = 0; i < kWordCases; ++i) {
        const WordResults got = results[i], ref = word_reference(words[i]);
        if (!equal("signed dp4a modulo 2^32", got.signed_dot, ref.signed_dot, i) ||
            !equal("unsigned dp4a modulo 2^32", got.unsigned_dot, ref.unsigned_dot, i) ||
            !equal("byte_perm", got.perm, ref.perm, i) || !equal("packed subtract", got.sub, ref.sub, i) ||
            !equal("packed saturating subtract", got.sat_sub, ref.sat_sub, i) ||
            !equal("packed equal", got.equal, ref.equal, i) || !equal("packed not_equal", got.not_equal, ref.not_equal, i)) return 1;
    }
    HIP_CHECK(hipFree(dr)); HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dl)); HIP_CHECK(hipFree(ds)); HIP_CHECK(hipFree(dv));
    std::printf("PASS gfx90a_intrinsics: launches=%d vote_lanes=%d shuffle_values=%d word_cases=%d skipped=0\n",
                launches, checked_lanes, shuffle_values, kWordCases);
    return 0;
}
