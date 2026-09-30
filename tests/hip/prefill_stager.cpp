#include "stager.hpp"
#include <cstdio>
#include <string>

namespace {
void env(const char* name, int value) {
    const std::string text = std::to_string(value);
#if defined(_WIN32)
    _putenv_s(name, text.c_str());
#else
    setenv(name, text.c_str(), 1);
#endif
}

bool check(cudaError_t status) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "HIP staging test: %s\n", cudaGetErrorString(status));
    return false;
}
}  // namespace

int main() {
    using strata::prefill::detail::Stager;
    constexpr size_t bytes = 64 * 1024;
    cudaStream_t copy = nullptr;
    uint8_t* dst = nullptr;
    if (!check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking)) ||
        !check(cudaMalloc((void**) &dst, 65 * bytes))) return 1;
    env("STRATA_PREFILL_TIMING", 1);
    int cases = 0;
    for (int ring : {16, 17}) for (int requested_batch : {1, 2, 4, 8, 16}) {
        env("STRATA_STAGER_RING", ring);
        env("STRATA_HIP_STAGE_BATCH", requested_batch);
        Stager stage;
        if (!stage.init(bytes, 3)) return 1;
        for (int count : {1, 15, 16, 17, 31, 32, 33, 65}) {
            for (int stop : {-1, 0, 1, 3, 15, 17, 31, 64}) {
                if (stop >= count) continue;
                // Reuse the same stager across generations, including after abort.
                for (int generation = 0; generation < 2; ++generation) {
                    const int issued = stop < 0 ? count : stop;
                    std::vector<uint8_t> source((size_t) count * bytes), got((size_t) issued * bytes);
                    for (size_t i = 0; i < source.size(); ++i)
                        source[i] = (uint8_t) (i * 17 + i / bytes * 71 + generation * 23);
                    std::vector<Stager::Job> jobs;
                    for (int j = 0; j < count; ++j)
                        jobs.push_back({source.data() + (size_t) j * bytes, bytes});
                    stage.start(std::move(jobs), true);
#if defined(STRATA_USE_HIP) && defined(_WIN32)
                    if (stage.fence_batch != requested_batch) return 1;
#else
                    if (stage.fence_batch != 1) return 1;
#endif
                    for (int j = 0; j < issued; ++j) {
                        if (!check(cudaMemcpyAsync(dst + (size_t) j * bytes, stage.wait(j), bytes,
                                                   cudaMemcpyHostToDevice, copy))) return 1;
                        stage.issued_one(j, copy);
                    }
                    stage.finish();
                    const uint64_t records = stage.fence_records;
                    stage.finish();
                    if (stage.fence_records != records || stage.fence_errors.load()) return 1;
                    const int batch = stage.fence_batch;
                    const uint64_t expected = (issued / ring) * ((ring + batch - 1) / batch) +
                                              ((issued % ring) + batch - 1) / batch;
                    if (records != expected || !check(cudaStreamSynchronize(copy))) return 1;
                    if (issued && (!check(cudaMemcpy(got.data(), dst, got.size(), cudaMemcpyDeviceToHost)) ||
                                   std::memcmp(source.data(), got.data(), got.size()) != 0)) {
                        std::fprintf(stderr, "staging mismatch: ring=%d batch=%d jobs=%d issued=%d\n",
                                     ring, batch, count, issued);
                        return 1;
                    }
                    ++cases;
                }
            }
        }
    }
    if (!check(cudaFree(dst)) || !check(cudaStreamDestroy(copy))) return 1;
    std::printf("HIP staging ownership: %d cases, wraps, partial abort, repeated finish and generations passed\n", cases);
    return 0;
}
