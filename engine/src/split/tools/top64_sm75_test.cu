#include "kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <future>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
static bool measure = false;
static bool dispatch = false;
static thread_local cudaStream_t copy_stream = nullptr;
cudaError_t test_copy(void *destination, const void *source, size_t bytes, cudaMemcpyKind kind) {
    const auto status = cudaMemcpyAsync(destination, source, bytes, kind, copy_stream);
    return status == cudaSuccess ? cudaStreamSynchronize(copy_stream) : status;
}
cudaError_t test_zero(void *destination, int value, size_t bytes) {
    const auto status = cudaMemsetAsync(destination, value, bytes, copy_stream);
    return status == cudaSuccess ? cudaStreamSynchronize(copy_stream) : status;
}
namespace fnx {
void top_tokens_reference_sm75(float *, int, int, int, const int *, const int *, const int *, int, const float *,
                               float *, int *, unsigned *, void *);
void top_tokens_parallel_sm75(float *, int, int, int, const int *, const int *, const int *, int, const float *,
                              float *, int *, unsigned *, void *);
} // namespace fnx
void ck(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
template <class T> T *upload(const std::vector<T> &values) {
    T *result;
    ck(cudaMalloc(&result, values.size() * sizeof(T)));
    ck(test_copy(result, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return result;
}
unsigned order(float value) {
    unsigned bits;
    std::memcpy(&bits, &value, 4);
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
template <class F> double microseconds(F function, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    for (int i = 0; i < 16; ++i)
        function();
    ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&executable, graph, 0));
    for (int i = 0; i < 2; ++i)
        ck(cudaGraphLaunch(executable, stream));
    ck(cudaStreamSynchronize(stream));
    cudaEvent_t start, end;
    ck(cudaEventCreate(&start));
    ck(cudaEventCreate(&end));
    ck(cudaEventRecord(start, stream));
    for (int i = 0; i < 4; ++i)
        ck(cudaGraphLaunch(executable, stream));
    ck(cudaEventRecord(end, stream));
    ck(cudaEventSynchronize(end));
    float ms;
    ck(cudaEventElapsedTime(&ms, start, end));
    ck(cudaEventDestroy(start));
    ck(cudaEventDestroy(end));
    ck(cudaGraphExecDestroy(executable));
    ck(cudaGraphDestroy(graph));
    return ms * 1000 / 64;
}
void run(int device, int n, int T, int mode, int v0) {
    ck(cudaSetDevice(device));
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    copy_stream = stream;
    constexpr int ps = 18, K = fnx::TOPK_N;
    std::mt19937 random(2739 + n + T + mode);
    std::vector<float> logits((size_t)n * T);
    for (size_t i = 0; i < logits.size(); ++i)
        logits[i] = mode == 0   ? ((int)(random() % 2000001) - 1000000) / 65536.f
                    : mode == 1 ? (int)(i % 7) - 3
                                : (i % 2 ? -0.f : 0.f);
    std::vector<int> pn(T, mode ? ps : 0), pt(T * ps), pc(T * ps);
    std::vector<float> pp(3 * T);
    for (int t = 0; t < T; ++t) {
        pp[3 * t] = mode == 2 ? .875f : 1.125f;
        pp[3 * t + 1] = t % 2 ? -.03125f : .03125f;
        pp[3 * t + 2] = t % 2 ? -.25f : .25f;
        for (int j = 0; j < ps; ++j) {
            pt[t * ps + j] = j == ps - 1 ? v0 + n : j == ps - 2 ? v0 - 1 : v0 + (j * 997 + t * 31) % n;
            pc[t * ps + j] = j % 5 == 0 ? 0 : j % 7 == 0 ? -1 : j % 3 + 1;
        }
    }
    float *reference = upload(logits), *candidate = upload(logits), *par = upload(pp);
    int *nt = upload(pn), *tok = upload(pt), *cnt = upload(pc);
    auto rv_memory = upload(std::vector<float>(T * K + 2, 31337.5f));
    auto cv_memory = upload(std::vector<float>(T * K + 2, 31337.5f));
    auto ri_memory = upload(std::vector<int>(T * K + 2, -987654));
    auto ci_memory = upload(std::vector<int>(T * K + 2, -987654));
    auto rv = rv_memory + 1, cv = cv_memory + 1;
    auto ri = ri_memory + 1, ci = ci_memory + 1;
    auto err = upload(std::vector<unsigned>(1));
    auto base = [&] {
        fnx::top_tokens_reference_sm75(reference, n, T, v0, tok, cnt, nt, ps, par, rv, ri, err, stream);
    };
    base();
    ck(cudaStreamSynchronize(stream));
    std::vector<float> expected(T * K), actual(T * K), penalized(logits.size()), changed(logits.size());
    std::vector<int> expected_ids(T * K), actual_ids(T * K);
    ck(test_copy(expected.data(), rv, expected.size() * 4, cudaMemcpyDeviceToHost));
    ck(test_copy(expected_ids.data(), ri, expected_ids.size() * 4, cudaMemcpyDeviceToHost));
    ck(test_copy(penalized.data(), reference, penalized.size() * 4, cudaMemcpyDeviceToHost));
    // Independent CPU selection over the actual penalized values, including all ties and signed zeros.
    for (int t = 0; t < T; ++t) {
        std::vector<int> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + K, ids.end(), [&](int a, int b) {
            const unsigned x = order(penalized[(size_t)t * n + a]), y = order(penalized[(size_t)t * n + b]);
            return x > y || (x == y && a < b);
        });
        for (int j = 0; j < K; ++j)
            if (expected_ids[t * K + j] != v0 + ids[j])
                throw std::runtime_error("baseline CPU oracle mismatch");
    }
    ck(test_copy(candidate, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    ck(test_copy(nt, pn.data(), T * 4, cudaMemcpyHostToDevice));
    auto test = [&] {
        if (dispatch)
            fnx::top_tokens(candidate, n, T, v0, tok, cnt, nt, ps, par, cv, ci, err, stream);
        else
            fnx::top_tokens_parallel_sm75(candidate, n, T, v0, tok, cnt, nt, ps, par, cv, ci, err, stream);
    };
    test();
    ck(cudaStreamSynchronize(stream));
    ck(test_copy(actual.data(), cv, actual.size() * 4, cudaMemcpyDeviceToHost));
    ck(test_copy(actual_ids.data(), ci, actual_ids.size() * 4, cudaMemcpyDeviceToHost));
    ck(test_copy(changed.data(), candidate, changed.size() * 4, cudaMemcpyDeviceToHost));
    unsigned error;
    ck(test_copy(&error, err, 4, cudaMemcpyDeviceToHost));
    if (error || actual_ids != expected_ids || std::memcmp(actual.data(), expected.data(), actual.size() * 4) ||
        std::memcmp(changed.data(), penalized.data(), changed.size() * 4))
        throw std::runtime_error("parallel topk changed candidates/penalties");
    // The same captured graph sees new input data on each replay. Penalties
    // remain enabled and are applied once to freshly restored logits.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    test();
    ck(cudaStreamEndCapture(stream, &graph));
    size_t node_count = 0;
    ck(cudaGraphGetNodes(graph, nullptr, &node_count));
    const bool tiled =
        n >= 65536 && (!dispatch || std::getenv("FNX_SM75_TOP64") && std::atoi(std::getenv("FNX_SM75_TOP64")) != 0);
    if (node_count != (tiled ? 4u : 1u))
        throw std::runtime_error("Top64 graph changed the expected dispatch");
    ck(cudaGraphInstantiate(&executable, graph, 0));
    for (int replay = 0; replay < 3; ++replay) {
        auto changed_input = logits;
        changed_input[0] = 100.f + replay;
        ck(test_copy(reference, changed_input.data(), changed_input.size() * 4, cudaMemcpyHostToDevice));
        ck(test_copy(candidate, changed_input.data(), changed_input.size() * 4, cudaMemcpyHostToDevice));
        base();
        ck(cudaGraphLaunch(executable, stream));
        ck(cudaStreamSynchronize(stream));
        ck(test_copy(expected.data(), rv, expected.size() * 4, cudaMemcpyDeviceToHost));
        ck(test_copy(actual.data(), cv, actual.size() * 4, cudaMemcpyDeviceToHost));
        ck(test_copy(expected_ids.data(), ri, expected_ids.size() * 4, cudaMemcpyDeviceToHost));
        ck(test_copy(actual_ids.data(), ci, actual_ids.size() * 4, cudaMemcpyDeviceToHost));
        ck(test_copy(penalized.data(), reference, penalized.size() * 4, cudaMemcpyDeviceToHost));
        ck(test_copy(changed.data(), candidate, changed.size() * 4, cudaMemcpyDeviceToHost));
        if (actual_ids != expected_ids || std::memcmp(actual.data(), expected.data(), actual.size() * 4) ||
            std::memcmp(changed.data(), penalized.data(), changed.size() * 4))
            throw std::runtime_error("captured Top64 changed candidates or penalties");
    }
    cudaGraph_t replacement;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    test();
    ck(cudaStreamEndCapture(stream, &replacement));
    cudaGraphExecUpdateResultInfo info{};
    if (cudaGraphExecUpdate(executable, replacement, &info) != cudaSuccess) {
        ck(cudaGraphExecDestroy(executable));
        ck(cudaGraphInstantiate(&executable, replacement, 0));
    }
    ck(cudaGraphDestroy(replacement));
    ck(test_copy(reference, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    ck(test_copy(candidate, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    base();
    ck(cudaGraphLaunch(executable, stream));
    ck(cudaStreamSynchronize(stream));
    ck(test_copy(expected.data(), rv, expected.size() * 4, cudaMemcpyDeviceToHost));
    ck(test_copy(actual.data(), cv, actual.size() * 4, cudaMemcpyDeviceToHost));
    ck(test_copy(expected_ids.data(), ri, expected_ids.size() * 4, cudaMemcpyDeviceToHost));
    ck(test_copy(actual_ids.data(), ci, actual_ids.size() * 4, cudaMemcpyDeviceToHost));
    if (actual_ids != expected_ids || std::memcmp(actual.data(), expected.data(), actual.size() * 4))
        throw std::runtime_error("graph replacement changed Top64 candidates");
    ck(cudaGraphExecDestroy(executable));
    ck(cudaGraphDestroy(graph));
    ck(test_copy(reference, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    ck(test_copy(candidate, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    if (measure) {
        // No penalties during repeated timing: their correct in-place application is checked above.
        ck(test_zero(nt, 0, T * 4));
        std::vector<std::pair<double, double>> timings;
        for (int repeat = 0; repeat < 3; ++repeat) {
            double a, b;
            if (repeat % 2) {
                b = microseconds(test, stream);
                a = microseconds(base, stream);
            } else {
                a = microseconds(base, stream);
                b = microseconds(test, stream);
            }
            timings.emplace_back(a, b);
        }
        std::sort(timings.begin(), timings.end(),
                  [](const auto &a, const auto &b) { return a.first / a.second < b.first / b.second; });
        const auto [old, new_time] = timings[1];
        std::printf("PASS dev=%d n=%d T=%d mode=%d old_us=%.4f new_us=%.4f speedup=%.5f "
                    "exact_candidates=1 exact_penalties=1 cpu_oracle=1\n",
                    device, n, T, mode, old, new_time, old / new_time);
        std::fflush(stdout);
    }
    // Both implementations must refuse nonfinite head logits (the caller checks error bit 16).
    for (float nan : {NAN, INFINITY, -INFINITY}) {
        ck(test_copy(reference, &nan, 4, cudaMemcpyHostToDevice));
        ck(test_copy(candidate, &nan, 4, cudaMemcpyHostToDevice));
        for (int implementation = 0; implementation < 2; ++implementation) {
            ck(test_zero(err, 0, 4));
            if (!implementation)
                base();
            else if (dispatch)
                fnx::top_tokens(candidate, n, T, v0, tok, cnt, nt, ps, par, cv, ci, err, stream);
            else
                fnx::top_tokens_parallel_sm75(candidate, n, T, v0, tok, cnt, nt, ps, par, cv, ci, err, stream);
            ck(cudaStreamSynchronize(stream));
            unsigned error;
            ck(test_copy(&error, err, 4, cudaMemcpyDeviceToHost));
            if (error != 16u)
                throw std::runtime_error("nonfinite head was not detected");
        }
    }
    for (float *pointer : {rv_memory, cv_memory}) {
        float guard[2];
        ck(test_copy(guard, pointer, sizeof(float), cudaMemcpyDeviceToHost));
        ck(test_copy(guard + 1, pointer + T * K + 1, sizeof(float), cudaMemcpyDeviceToHost));
        if (guard[0] != 31337.5f || guard[1] != 31337.5f)
            throw std::runtime_error("Top64 values crossed output guards");
    }
    for (int *pointer : {ri_memory, ci_memory}) {
        int guard[2];
        ck(test_copy(guard, pointer, sizeof(int), cudaMemcpyDeviceToHost));
        ck(test_copy(guard + 1, pointer + T * K + 1, sizeof(int), cudaMemcpyDeviceToHost));
        if (guard[0] != -987654 || guard[1] != -987654)
            throw std::runtime_error("Top64 indices crossed output guards");
    }
    ck(cudaStreamDestroy(stream));
    for (void *pointer : {(void *)reference, (void *)candidate, (void *)par, (void *)nt, (void *)tok, (void *)cnt,
                          (void *)rv_memory, (void *)cv_memory, (void *)ri_memory, (void *)ci_memory, (void *)err})
        ck(cudaFree(pointer));
}
int main(int argc, char **argv) {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0)
            return 77;
        const bool sweep = argc > 1 && !std::strcmp(argv[1], "--sweep");
        measure = sweep || (argc > 1 && !std::strcmp(argv[1], "--benchmark"));
        dispatch = argc > 1 && !std::strcmp(argv[1], "--dispatch");
        for (int device = 0; device < devices; ++device) {
            cudaDeviceProp properties{};
            ck(cudaGetDeviceProperties(&properties, device));
            if (properties.major != 7 || properties.minor != 5)
                return 77;
            int cases = 0;
            const std::vector<int> sizes =
                sweep     ? std::vector<int>{2048, 4096, 8192, 16384, 32768, 53149, 65536, 124160}
                : measure ? std::vector<int>{2048, 124160}
                          : std::vector<int>{64, 65, 1024, 2048, 2049, 2079, 53149, 65535, 65536, 65537, 124160};
            const std::vector<int> rows = sweep     ? std::vector<int>{1, 2, 3, 4, 7, 8, 16}
                                          : measure ? std::vector<int>{1, 3, 4, 7}
                                                    : std::vector<int>{1, 2, 3, 4, 7, 8, 16};
            const std::vector<int> modes = sweep ? std::vector<int>{0} : std::vector<int>{0, 1, 2};
            const std::vector<int> offsets = measure ? std::vector<int>{124160} : std::vector<int>{0, 124160};
            for (int n : sizes)
                for (int T : rows)
                    for (int mode : modes) {
                        for (int offset : offsets) {
                            std::printf("CASE device=%d n=%d rows=%d mode=%d first_token=%d\n", device, n, T, mode,
                                        offset);
                            std::fflush(stdout);
                            run(device, n, T, mode, offset);
                            ++cases;
                        }
                    }
            if (!measure) {
                auto first = std::async(std::launch::async, [=] { run(device, 65537, 3, 1, 0); });
                auto second = std::async(std::launch::async, [=] { run(device, 124160, 4, 2, 124160); });
                first.get();
                second.get();
            }
            std::printf("PASS device=%d cases=%d exact_values=1 exact_ids=1 exact_penalties=1 nonfinite=1 "
                        "graph_replays=3 concurrent_streams=%d\n",
                        device, cases, !measure);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
