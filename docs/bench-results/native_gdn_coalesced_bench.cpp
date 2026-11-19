// Production-linked GDN test. No model files, services or downloads.
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int S = 128;
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool ok, const char *m) {
    if (!ok)
        throw std::runtime_error(m);
}
struct Stream {
    cudaStream_t p{};
    Stream() { ck(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking)); }
    ~Stream() {
        if (p)
            cudaStreamDestroy(p);
    }
};
struct Buffer {
    float *p{};
    size_t count;
    explicit Buffer(size_t n) : count(n) { ck(cudaMalloc(&p, n * sizeof(float))); }
    ~Buffer() {
        if (p)
            cudaFree(p);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void put(const std::vector<float> &x) {
        require(x.size() == count, "upload size");
        ck(cudaMemcpy(p, x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> get() const {
        std::vector<float> x(count);
        ck(cudaMemcpy(x.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost));
        return x;
    }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    ~Graph() {
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
    }
    void end(cudaStream_t st) {
        ck(cudaStreamEndCapture(st, &graph));
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
};
struct Event {
    cudaEvent_t p{};
    Event() { ck(cudaEventCreate(&p)); }
    ~Event() {
        if (p)
            cudaEventDestroy(p);
    }
};
struct Fixture {
    int hk, hv;
    std::vector<float> state, q, k, v, gate, beta, z, gamma;
    Fixture(int kh, int vh, unsigned seed)
        : hk(kh), hv(vh), state(size_t(S) * S * hv), q(hk * S), k(hk * S), v(hv * S), gate(hv), beta(hv), z(hv * S),
          gamma(S) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto *a : {&state, &q, &k, &v, &z})
            for (float &x : *a)
                x = dist(rng);
        for (auto *a : {&q, &k})
            for (int h = 0; h < hk; ++h) {
                double norm = 1e-6;
                for (int i = 0; i < S; ++i)
                    norm += double((*a)[h * S + i]) * (*a)[h * S + i];
                for (int i = 0; i < S; ++i)
                    (*a)[h * S + i] = float((*a)[h * S + i] / std::sqrt(norm));
            }
        for (int h = 0; h < hv; ++h) {
            const float gates[] = {0.0f, -0.001f, -0.1f, -2.0f, -80.0f};
            const float betas[] = {0.0f, 1.0f, 0.5f, 0.001f, 0.999f};
            gate[h] = gates[(h + seed) % 5];
            beta[h] = betas[(h + seed / 5) % 5];
        }
        for (float &x : gamma)
            x = 1.0f + dist(rng);
    }
};
struct Inputs {
    Buffer q, k, v, gate, beta, z, gamma;
    explicit Inputs(const Fixture &f)
        : q(f.q.size()), k(f.k.size()), v(f.v.size()), gate(f.gate.size()), beta(f.beta.size()), z(f.z.size()),
          gamma(f.gamma.size()) {
        put(f);
        z.put(f.z);
        gamma.put(f.gamma);
    }
    void put(const Fixture &f) {
        q.put(f.q);
        k.put(f.k);
        v.put(f.v);
        gate.put(f.gate);
        beta.put(f.beta);
    }
};
void launch(bool fast, Buffer &state, Buffer &out, Inputs &in, int hk, int hv, cudaStream_t st) {
    strata::kernels::native_gdn_set_coalesced(fast);
    strata::kernels::native_gdn_step(state.p, in.q.p, in.k.p, in.v.p, in.gate.p, in.beta.p, out.p, {S, hk, hv}, st);
}
void equal(const std::vector<float> &a, const std::vector<float> &b, const char *what) {
    require(a.size() == b.size(), "comparison size");
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::memcmp(&a[i], &b[i], sizeof(float))) {
            std::fprintf(stderr, "%s mismatch index=%zu a=%.9g b=%.9g\n", what, i, a[i], b[i]);
            throw std::runtime_error(what);
        }
    }
}
void oracle(std::vector<double> &state, std::vector<double> &output, const Fixture &f) {
    for (int h = 0; h < f.hv; ++h)
        for (int j = 0; j < S; ++j) {
            double kv = 0;
            for (int i = 0; i < S; ++i)
                kv += state[(size_t(i) * f.hv + h) * S + j] * double(f.k[(h % f.hk) * S + i]);
            const double g = std::exp(double(f.gate[h]));
            const double d = (double(f.v[h * S + j]) - g * kv) * double(f.beta[h]);
            double o = 0;
            for (int i = 0; i < S; ++i) {
                double &s = state[(size_t(i) * f.hv + h) * S + j];
                s = g * s + double(f.k[(h % f.hk) * S + i]) * d;
                o += s * double(f.q[(h % f.hk) * S + i]);
            }
            output[h * S + j] = o / std::sqrt(double(S));
        }
}
double near(const std::vector<float> &actual, const std::vector<double> &ref, const char *what) {
    require(actual.size() == ref.size(), "oracle size");
    double maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::abs(double(actual[i]) - ref[i]);
        maxerr = std::max(maxerr, e);
        if (!std::isfinite(actual[i]) || !std::isfinite(ref[i]) || e > 2e-5 + 2e-5 * std::abs(ref[i]))
            throw std::runtime_error(what);
    }
    return maxerr;
}
void correctness(int hk, int hv, int steps, unsigned seed) {
    Stream st;
    Fixture f(hk, hv, seed);
    Inputs in(f);
    Buffer a(f.state.size()), b(f.state.size()), oa(f.v.size()), ob(f.v.size());
    a.put(f.state);
    b.put(f.state);
    std::vector<double> ref(f.state.begin(), f.state.end()), ro(f.v.size());
    double max_state = 0, max_output = 0;
    for (int t = 0; t < steps; ++t) {
        Fixture next(hk, hv, seed + unsigned(t));
        in.put(next);
        launch(false, a, oa, in, hk, hv, st.p);
        launch(true, b, ob, in, hk, hv, st.p);
        ck(cudaStreamSynchronize(st.p));
        equal(a.get(), b.get(), "state");
        equal(oa.get(), ob.get(), "output");
        if (t < 16) {
            oracle(ref, ro, next);
            max_state = std::max(max_state, near(b.get(), ref, "FP64 state"));
            max_output = std::max(max_output, near(ob.get(), ro, "FP64 output"));
        }
    }
    // Capture choice must survive subsequent host-flag changes.
    a.put(f.state);
    b.put(f.state);
    in.put(f);
    Graph ga, gb;
    ck(cudaStreamBeginCapture(st.p, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 7; ++i)
        launch(false, a, oa, in, hk, hv, st.p);
    ga.end(st.p);
    ck(cudaStreamBeginCapture(st.p, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 7; ++i)
        launch(true, b, ob, in, hk, hv, st.p);
    gb.end(st.p);
    strata::kernels::native_gdn_set_coalesced(false);
    for (int r = 0; r < 3; ++r) {
        ck(cudaGraphLaunch(ga.exec, st.p));
        ck(cudaGraphLaunch(gb.exec, st.p));
    }
    ck(cudaStreamSynchronize(st.p));
    equal(a.get(), b.get(), "graph state");
    equal(oa.get(), ob.get(), "graph output");
    for (bool fast : {false, true}) {
        strata::kernels::native_gdn_set_coalesced(fast);
        auto bad = [&](float *state, float *out, strata::kernels::GdnShapes shape, void *stream) {
            bool rejected = false;
            try {
                strata::kernels::native_gdn_step(state, in.q.p, in.k.p, in.v.p, in.gate.p, in.beta.p, out, shape,
                                                 stream);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            require(rejected, "invalid arguments accepted");
        };
        bad(a.p, oa.p, {S, hk, hv}, nullptr);
        bad(a.p, a.p, {S, hk, hv}, st.p);
        bad(a.p, in.q.p, {S, hk, hv}, st.p);
        bad(a.p, oa.p, {64, hk, hv}, st.p);
        bad(a.p, oa.p, {S, 2, 3}, st.p);
        bad(nullptr, oa.p, {S, hk, hv}, st.p);
        bad(reinterpret_cast<float *>(reinterpret_cast<char *>(a.p) + 1), oa.p, {S, hk, hv}, st.p);
    }
    std::printf("{\"test\":\"correctness\",\"hk\":%d,\"hv\":%d,\"steps\":%d,\"seed\":%u,"
                "\"bitwise\":true,\"graph\":true,\"invalid_inputs\":true,\"fp64_steps\":%d,"
                "\"max_state_abs\":%.9g,\"max_output_abs\":%.9g}\n",
                hk, hv, steps, seed, std::min(steps, 16), max_state, max_output);
}
void benchmark(int layers, int samples, int repeats) {
    const int hk = 16, hv = 48;
    Fixture f(hk, hv, 101);
    Inputs in(f);
    Stream st;
    Buffer state(f.state.size() * layers), out(f.v.size()), norm(f.v.size());
    std::vector<float> initial(state.count);
    for (int l = 0; l < layers; ++l)
        std::copy(f.state.begin(), f.state.end(), initial.begin() + l * f.state.size());
    const char *names[] = {"original_recurrence", "coalesced_recurrence", "original_separate_norm",
                           "coalesced_separate_norm", "existing_fused_norm"};
    Graph graphs[5];
    for (int mode = 0; mode < 5; ++mode) {
        strata::kernels::native_gdn_set_coalesced(mode == 1 || mode == 3);
        ck(cudaStreamBeginCapture(st.p, cudaStreamCaptureModeThreadLocal));
        for (int r = 0; r < repeats; ++r)
            for (int l = 0; l < layers; ++l) {
                float *s = state.p + size_t(l) * f.state.size();
                if (mode == 4) {
                    strata::kernels::fused_gdn_step_norm(s, in.q.p, in.k.p, in.v.p, in.gate.p, in.beta.p, in.z.p,
                                                         in.gamma.p, 1e-6f, norm.p, hk, hv, st.p);
                } else {
                    strata::kernels::native_gdn_step(s, in.q.p, in.k.p, in.v.p, in.gate.p, in.beta.p, out.p,
                                                     {S, hk, hv}, st.p);
                    if (mode >= 2)
                        strata::kernels::native_gdn_out_norm(out.p, in.z.p, in.gamma.p, norm.p, hv, S, 1e-6f, st.p);
                }
            }
        graphs[mode].end(st.p);
    }
    Event start, stop;
    for (int mode = 0; mode < 5; ++mode) {
        state.put(initial);
        for (int w = 0; w < 3; ++w)
            ck(cudaGraphLaunch(graphs[mode].exec, st.p));
    }
    ck(cudaStreamSynchronize(st.p));
    for (int round = 0; round < samples; ++round)
        for (int pos = 0; pos < 5; ++pos) {
            int mode = round % 2 ? 4 - pos : pos;
            state.put(initial);
            ck(cudaEventRecord(start.p, st.p));
            ck(cudaGraphLaunch(graphs[mode].exec, st.p));
            ck(cudaEventRecord(stop.p, st.p));
            ck(cudaEventSynchronize(stop.p));
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, start.p, stop.p));
            std::printf("{\"test\":\"timing\",\"layers\":%d,\"state_bytes\":%zu,\"round\":%d,"
                        "\"mode\":\"%s\",\"calls\":%d,\"us_per_layer\":%.9g}\n",
                        layers, state.count * sizeof(float), round, names[mode], layers * repeats,
                        double(ms) * 1000 / (layers * repeats));
        }
}
} // namespace
int main(int argc, char **argv) {
    try {
        bool test_only = false, quick = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--selftest") == 0)
                test_only = true;
            else if (std::strcmp(argv[i], "--quick") == 0)
                quick = true;
            else {
                std::fprintf(stderr, "usage: %s [--selftest] [--quick]\n", argv[0]);
                return 2;
            }
        }
        cudaDeviceProp p{};
        ck(cudaGetDeviceProperties(&p, 0));
        int runtime = 0, driver = 0;
        ck(cudaRuntimeGetVersion(&runtime));
        ck(cudaDriverGetVersion(&driver));
        std::printf("{\"test\":\"device\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
                    "\"l2_bytes\":%d,\"persisting_l2_max_bytes\":%d,\"runtime\":%d,\"driver\":%d}\n",
                    p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize, p.persistingL2CacheMaxSize, runtime,
                    driver);
        require(!strata::kernels::native_gdn_coalesced(), "default is not off");
        strata::kernels::native_gdn_set_enabled(true);
        const char *env = std::getenv("STRATA_NATIVE_GDN_COALESCED");
        require(strata::kernels::native_gdn_coalesced() == (env && std::strcmp(env, "1") == 0),
                "environment selection");
        strata::kernels::native_gdn_set_enabled(false);
        require(!strata::kernels::native_gdn_coalesced(), "disable did not reset choice");
        correctness(1, 1, quick ? 4 : 32, 19);
        correctness(2, 6, quick ? 4 : 32, 27);
        correctness(16, 48, quick ? 8 : 256, 42);
        correctness(17, 51, quick ? 4 : 32, 53);
        if (!test_only) {
            benchmark(1, quick ? 3 : 15, 64);
            benchmark(48, quick ? 3 : 15, 4);
        }
        ck(cudaDeviceSynchronize());
        std::printf("{\"test\":\"result\",\"pass\":true}\n");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
}
