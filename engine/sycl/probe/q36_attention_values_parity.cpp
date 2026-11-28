// Link with q36_engine compiled using STRATA_Q36_TEST_HOOKS. This verifies the
// real attention-value GEMM helper, including noncompact interleaved output.
// -DSTRATA_Q36_VALUES_CPU_ONLY builds the fixture/oracle checks without engine.
#include "../src/qwen35/q36_test_hooks.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int HD = 256, H = 16, G = 8, guard = 32;
constexpr float sentinel = -12345.0f;
void require(bool yes, const std::string& message) {
    if (!yes) throw std::runtime_error(message);
}
int value_numerator(int p, int d) { return (p * 17 + d * 13 + 5) % 255 - 127; }
int probability_numerator(int g, int t, int p) { return (g * 11 + t * 7 + p * 3) % 32 + 1; }
struct Fixture {
    int nt, P, ldp;
    bool dyadic;
    std::vector<sycl::half> values, probabilities;
    Fixture(int nt, int P, bool dyadic)
        : nt(nt), P(P), ldp((P + 7) / 8 * 8), dyadic(dyadic),
          values(size_t(P) * HD + 2 * guard, sycl::half(-17.0f)),
          probabilities(size_t(G) * nt * ldp + 2 * guard, sycl::half(-19.0f)) {
        for (int p = 0; p < P; ++p)
            for (int d = 0; d < HD; ++d)
                v()[size_t(p) * HD + d] = sycl::half(value_numerator(p,d) / (dyadic ? 256.0f : 263.0f));
        for (int g = 0; g < G; ++g)
            for (int t = 0; t < nt; ++t)
                for (int p = 0; p < ldp; ++p)
                    // A wrong leading dimension consumes NaN padding, not zeros.
                    s()[(size_t(g) * nt + t) * ldp + p] = p < P
                        ? sycl::half(probability_numerator(g,t,p) / (dyadic ? 1024.0f : float(P * 32)))
                        : sycl::half(std::numeric_limits<float>::quiet_NaN());
    }
    sycl::half* v() { return values.data() + guard; }
    sycl::half* s() { return probabilities.data() + guard; }
    float reference(int g, int t, int d) const {
        double sum = 0;
        for (int p = 0; p < P; ++p)
            sum += double(float(values[guard + size_t(p) * HD + d])) *
                   double(float(probabilities[guard + (size_t(g) * nt + t) * ldp + p]));
        return float(sum);
    }
    float integer_reference(int g, int t, int d) const {
        int64_t numerator = 0;
        for (int p = 0; p < P; ++p)
            numerator += int64_t(value_numerator(p,d)) * probability_numerator(g,t,p);
        // Every product and partial sum is exactly representable in FP32 for
        // these bounded fixtures. Therefore the GPU result must match bits.
        return float(numerator) / 262144.0f;
    }
};
void cpu_checks() {
    size_t checked = 0;
    for (auto shape : {std::pair<int,int>{1,1}, {1,257}, {3,27}, {3,271}, {27,27}, {27,283}}) {
        Fixture f(shape.first,shape.second,true);
        for (int g = 0; g < G; ++g) for (int t = 0; t < f.nt; ++t) for (int d = 0; d < HD; ++d) {
            float a = f.reference(g,t,d), b = f.integer_reference(g,t,d);
            require(std::memcmp(&a,&b,sizeof(float)) == 0, "CPU floating/integer oracle mismatch");
            ++checked;
        }
    }
    std::cout << "CPU oracle passed " << checked << " output elements; GPU not selected\n";
}
#ifndef STRATA_Q36_VALUES_CPU_ONLY
template<class T> struct Buffer {
    sycl::queue& q;
    T* p;
    size_t n;
    Buffer(sycl::queue& q, size_t n) : q(q), p(sycl::malloc_shared<T>(n,q)), n(n) {
        require(p != nullptr, "USM allocation failed");
    }
    ~Buffer() { sycl::free(p,q); }
    Buffer(const Buffer&) = delete;
};
void gpu_case(sycl::queue& q, int nt, int P, int first_head, bool dyadic) {
    Fixture f(nt,P,dyadic);
    Buffer<sycl::half> values(q,f.values.size()), probabilities(q,f.probabilities.size());
    const size_t nout = size_t(nt) * H * HD;
    Buffer<float> output(q,nout + 2 * guard);
    std::copy(f.values.begin(),f.values.end(),values.p);
    std::copy(f.probabilities.begin(),f.probabilities.end(),probabilities.p);
    std::fill(output.p,output.p + output.n,sentinel);
    strata::qwen35::test::attention_values_gemm(
        &q,values.p+guard,probabilities.p+guard,output.p+guard+first_head*HD,
        HD,nt,P,f.ldp,H,G);
    q.wait_and_throw();
    require(std::memcmp(values.p,f.values.data(),values.n*sizeof(sycl::half)) == 0,"value input/canary modified");
    require(std::memcmp(probabilities.p,f.probabilities.data(),probabilities.n*sizeof(sycl::half)) == 0,"probability input/padding/canary modified");
    for (int i = 0; i < guard; ++i)
        require(output.p[i] == sentinel && output.p[guard+nout+i] == sentinel,"output boundary canary modified");
    double max_abs = 0;
    size_t exact = 0;
    for (int t = 0; t < nt; ++t) for (int h = 0; h < H; ++h) for (int d = 0; d < HD; ++d) {
        float actual = output.p[guard + (size_t(t) * H + h) * HD + d];
        if (h < first_head || h >= first_head + G) {
            require(actual == sentinel,"GEMM overwrote another KV head group's interleaved output");
            continue;
        }
        float expected = f.reference(h-first_head,t,d);
        require(std::isfinite(actual),"GEMM produced nonfinite output (probability ldp/padding)");
        max_abs = std::max(max_abs,std::abs(double(actual)-expected));
        if (dyadic) {
            require(std::memcmp(&actual,&expected,sizeof(float)) == 0,
                    "exact value GEMM mismatch at t="+std::to_string(t)+" h="+std::to_string(h)+" d="+std::to_string(d));
            ++exact;
        } else {
            require(std::abs(actual-expected) <= 2e-5f,"value GEMM exceeds FP32-accumulation tolerance");
        }
    }
    std::cout << "values GEMM HD256 H16 G8 nt=" << nt << " P=" << P << " ldp=" << f.ldp
              << " first_head=" << first_head << " dyadic=" << dyadic << " exact=" << exact
              << " max_abs=" << max_abs << "; input/padding/output canaries passed\n";
}
void gpu_checks() {
    sycl::queue q(sycl::gpu_selector_v,[](sycl::exception_list es) {
        for (auto& e : es) std::rethrow_exception(e);
    },sycl::property::queue::in_order{});
    require(q.get_device().has(sycl::aspect::fp16),"probe requires fp16 GPU");
    std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    for (auto shape : {std::pair<int,int>{1,1}, {1,257}, {3,27}, {3,271}, {27,27}, {27,283}})
        for (int first_head : {0,8}) gpu_case(q,shape.first,shape.second,first_head,true);
    for (int nt : {1,3,27}) gpu_case(q,nt,283,8,false);
    std::cout << "All 15 attention-values GPU cases passed\n";
}
#endif
}
int main() {
    try {
        cpu_checks();
#ifndef STRATA_Q36_VALUES_CPU_ONLY
        gpu_checks();
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
