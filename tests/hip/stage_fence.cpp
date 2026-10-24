// #1237: a page-locked stage buffer must not be recycled while a queued copy is still reading it.
//
// A pageable stage buffer is staged by the driver before cudaMemcpyAsync returns, so handing it out again as
// soon as the copy is *queued* is harmless.  A PAGE-LOCKED one (STRATA_STAGE_PIN) is read by the DMA after the
// call returned: recycling it while that copy is in flight makes the copy land whatever bytes the refill wrote -
// the corrupted IQ3_S decode of #1237 (48 tokens of id 0 after a 4096-token prompt).
//
// The engine fences a queued copy with an event (`note_async_read`) and waits for every such event before the
// buffer's next fill.  This test makes the race deterministic instead of hoping for a 4096-token prompt:
//
//   begin_layer(0); blob(0, 0)   a pinned stage buffer, filled with the file's (0,0) bytes
//   stream A:  64 x 64 MiB device copies, then a copy of that buffer out, event recorded   (pending, ~ms)
//   stream B:  a copy of the same buffer out, event recorded                              (completes at once)
//   begin_layer(1..3)            kStageAge layers pass: the buffer may be recycled now
//   blob(3, 1)                   the refill must wait for BOTH copies, then overwrite the buffer
//   read back                    both copies must hold the (0,0) bytes
//
// Two streams, not one: with a single event per buffer the second record (stream B, already complete) would hide
// stream A's, the refill would not wait, and the readback of A would be the (3,1) bytes.
//
// Needs a device (hipHostMalloc for STRATA_STAGE_PIN, and the copies above): exits 77 without one.

#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) {
        std::fprintf(stderr, "stage_fence: %s\n", message.c_str());
        std::fflush(stderr);
        std::exit(1);
    }
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

/// The pack lives in the working directory, not the temp directory: a tmpfs /tmp has no O_DIRECT, and this test
/// needs the unbuffered file tier (the only tier that assembles blobs into stage buffers).
struct TempDirectory {
    fs::path path;
    explicit TempDirectory(const fs::path& base) {
        path = base / ("strata-stage-fence-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

}  // namespace

int main() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::printf("stage_fence: no device, skipped\n");
        return 77;   // ctest "skipped"
    }
    // The pin mode is read at the first stage buffer this process makes: both knobs have to be set before any
    // blob() call.  STRATA_UNBUFFERED_LOAD forces the file tier unbuffered whatever the RAM looks like.
    set_env("STRATA_STAGE_PIN", "1");
    set_env("STRATA_UNBUFFERED_LOAD", "1");

    constexpr int64_t layers = 4, experts = 2;
    const uint64_t blob_bytes = (uint64_t) BLOB;
    const uint64_t total = (uint64_t) layers * (uint64_t) experts * blob_bytes;
    TempDirectory dir(fs::current_path());

    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "expert_layout_load: " + err);
    // deterministic bytes, and each blob starts with its own marker (1 for (0,0), 8 for (3,1))
    std::vector<uint8_t> file_bytes((size_t) total);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint8_t& b : file_bytes) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = (uint8_t) (x >> 56);
    }
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < experts; ++e)
            file_bytes[(size_t) (((uint64_t) l * experts + (uint64_t) e) * blob_bytes)] = (uint8_t) (1 + l * experts + e);
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
        out.write((const char*) file_bytes.data(), (std::streamsize) file_bytes.size());
        require((bool) out, "could not write the synthetic experts.bin");
    }

    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), "open: " + err);
    std::string why;
    if (!source.set_unbuffered(0, why)) {
        std::printf("stage_fence: unbuffered reads unavailable, skipped (%s)\n", why.c_str());
        source.close();
        return 77;
    }
    require(source.unbuffered(), "the source is not unbuffered: no stage buffers to test");

    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    require(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking) == cudaSuccess &&
                cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking) == cudaSuccess,
            "could not create the two test streams");
    const size_t chunk = 64ull << 20;
    void* pad_a = nullptr;
    void* pad_b = nullptr;
    void* out_a = nullptr;
    void* out_b = nullptr;
    require(cudaMalloc(&pad_a, chunk) == cudaSuccess && cudaMalloc(&pad_b, chunk) == cudaSuccess &&
                cudaMalloc(&out_a, (size_t) blob_bytes) == cudaSuccess && cudaMalloc(&out_b, (size_t) blob_bytes) == cudaSuccess,
            "could not allocate the test buffers");

    // the pinned stage buffer, holding (0,0)
    source.begin_layer(0, nullptr, 0);
    const uint8_t* p0 = source.blob(0, 0);
    require(p0 != nullptr, "blob(0, 0) produced no buffer");
    require(p0[0] == 1, "blob(0, 0) did not come from the file's (0,0) blob");

    // stream A: the buffer's copy goes behind ~4 GiB of device copies, so the DMA cannot reach it yet
    for (int i = 0; i < 64; ++i)
        require(cudaMemcpyAsync(i % 2 == 0 ? pad_a : pad_b, i % 2 == 0 ? pad_b : pad_a, chunk, cudaMemcpyDeviceToDevice,
                                stream_a) == cudaSuccess,
                "could not queue the padding copies");
    require(cudaMemcpyAsync(out_a, p0, (size_t) blob_bytes, cudaMemcpyHostToDevice, stream_a) == cudaSuccess,
            "could not queue the stream A copy");
    source.note_async_read(p0, (void*) stream_a);
    // stream B: the same buffer, on a stream with nothing in front of it (this is the record that a single
    // shared event per buffer would keep, losing A's)
    require(cudaMemcpyAsync(out_b, p0, (size_t) blob_bytes, cudaMemcpyHostToDevice, stream_b) == cudaSuccess,
            "could not queue the stream B copy");
    source.note_async_read(p0, (void*) stream_b);

    // kStageAge layer changes: the buffer is recyclable now
    source.begin_layer(1, nullptr, 0);
    source.begin_layer(2, nullptr, 0);
    source.begin_layer(3, nullptr, 0);
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t waits_before = source.stage_fence_waits();
    const uint8_t* p31 = source.blob(3, 1);
    const double recycle_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    require(p31 != nullptr, "blob(3, 1) produced no buffer");
    require(p31[0] == 8, "blob(3, 1) did not come from the file's (3,1) blob");
    require(source.stage_fence_waits() > waits_before, "the refill did not wait for the queued copy (fence never held)");

    // and the copies themselves: both must hold the (0,0) bytes the buffer had when they were queued
    require(cudaStreamSynchronize(stream_a) == cudaSuccess && cudaStreamSynchronize(stream_b) == cudaSuccess,
            "the test streams did not finish");
    std::vector<uint8_t> got_a((size_t) blob_bytes), got_b((size_t) blob_bytes);
    require(cudaMemcpy(got_a.data(), out_a, (size_t) blob_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(got_b.data(), out_b, (size_t) blob_bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
            "could not read the copied blobs back");
    for (int which = 0; which < 2; ++which) {
        const std::vector<uint8_t>& got = which == 0 ? got_a : got_b;
        if (std::memcmp(got.data(), file_bytes.data(), (size_t) blob_bytes) != 0) {
            std::fprintf(stderr, "stage_fence: the stream %c copy landed the REFILLED bytes (marker %u, expected 1): a\n"
                                 "page-locked stage buffer was recycled under a queued copy (#1237)\n",
                         which == 0 ? 'A' : 'B', (unsigned) got[0]);
            std::fflush(stderr);
            return 1;
        }
    }

    std::printf("stage_fence: two queued copies, one refill, %lld fence wait(s), %.1f ms; both copies hold the pre-refill bytes\n",
                (long long) (source.stage_fence_waits() - waits_before), recycle_ms);

    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    cudaFree(pad_a);
    cudaFree(pad_b);
    cudaFree(out_a);
    cudaFree(out_b);
    source.close();
    set_env("STRATA_STAGE_PIN", nullptr);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
    return 0;
}
