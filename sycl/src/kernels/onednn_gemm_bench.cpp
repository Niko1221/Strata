// onednn_gemm_bench: the prompt path's GEMM shapes in oneMKL fp16 (the path in use: dequantized fp16 weights, fp32
// out, as prefill::Gemm::f16), oneDNN fp16, and oneDNN int8 (s8 x s8, scales per tensor / per output column, fp32
// out): the matrix engine's rate for each, to see whether oneDNN or int8 are worth porting (sycl/TODO.md, 4).
//   onednn_gemm_bench [reps=20]
// Not in the CMake build: it needs oneDNN's headers, and the dev image carries only oneDNN's runtime. Where oneDNN
// (3.x) is installed in <dnnl>:
//   icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xs "-device bmg-g31" onednn_gemm_bench.cpp -I<dnnl>/include
//     -L<dnnl>/lib -Wl,-rpath,<dnnl>/lib -ldnnl -qmkl -lmkl_sycl_blas -o onednn_gemm_bench
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>

using half = sycl::half;
namespace mkl = oneapi::mkl;

struct Shape { const char* what; int64_t M, K, N; };

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

    // n_embd 2560, n_ff 640 (gate+up 1280), 24 q heads x 256, 2 KV heads, DeltaNet 10240 conv channels / 6144 values,
    // hyper-connections 4 x 2560 -> 320
    std::vector<Shape> shapes;
    for (int64_t m : {16, 32, 64, 128, 256, 512}) shapes.push_back({"expert gate/up", m, 2560, 1280});
    for (int64_t m : {16, 64, 256, 512}) shapes.push_back({"expert down", m, 640, 2560});
    shapes.push_back({"dense 2560->12288", 4096, 2560, 12288});
    shapes.push_back({"dense 2560->10240", 4096, 2560, 10240});
    shapes.push_back({"dense 2560->6144", 4096, 2560, 6144});
    shapes.push_back({"dense 6144->2560", 4096, 6144, 2560});
    shapes.push_back({"hc 10240->320", 4096, 10240, 320});
    shapes.push_back({"hc 320->10240", 4096, 320, 10240});

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::printf("%-20s %6s %6s %6s | %16s %16s %16s | %s\n", "shape", "M", "K", "N", "oneMKL f16", "oneDNN f16",
                "oneDNN int8", "TFLOP/s (ms)");
    for (const Shape& s : shapes) {
        const size_t xa = (size_t) s.M * s.K, wa = (size_t) s.N * s.K, ya = (size_t) s.M * s.N;
        std::vector<half> hx(xa), hw(wa);
        std::vector<int8_t> bx(xa), bw(wa);
        for (size_t i = 0; i < xa; ++i) { const float v = nd(rng); hx[i] = (half) v; bx[i] = (int8_t) std::max(-127.f, std::min(127.f, v * 32.f)); }
        for (size_t i = 0; i < wa; ++i) { const float v = nd(rng) * 0.05f; hw[i] = (half) v; bw[i] = (int8_t) std::max(-127.f, std::min(127.f, v * 640.f)); }
        half* X = sycl::malloc_device<half>(xa, q);
        half* W = sycl::malloc_device<half>(wa, q);
        int8_t* Xi = sycl::malloc_device<int8_t>(xa, q);
        int8_t* Wi = sycl::malloc_device<int8_t>(wa, q);
        float* Y = sycl::malloc_device<float>(ya, q);
        float* sw = sycl::malloc_device<float>((size_t) s.N, q);
        float* sx = sycl::malloc_device<float>(1, q);
        q.memcpy(X, hx.data(), xa * 2); q.memcpy(W, hw.data(), wa * 2);
        q.memcpy(Xi, bx.data(), xa); q.memcpy(Wi, bw.data(), wa);
        q.fill(sw, 1.0f / 640.f, s.N); q.fill(sx, 1.0f / 32.f, 1).wait();

        auto time = [&](auto&& fn) {
            for (int i = 0; i < 3; ++i) fn();
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            q.wait();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        // oneMKL, as prefill::Gemm::f16: Y[M,N] = X[M,K] . W[N,K]^T, column-major view (N x M = W^T-op . X)
        const float alpha = 1.f, beta = 0.f;
        const double t_mkl = time([&] {
            mkl::blas::column_major::gemm(q, mkl::transpose::trans, mkl::transpose::nontrans, s.N, s.M, s.K, alpha, W,
                                          s.K, X, s.K, beta, Y, s.N);
        });
        // oneDNN matmul: src [M,K] row-major, weights [K,N] given as W[N,K] (format "ba"), dst [M,N] f32
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        auto run_dnnl = [&](dt in_t, void* src, void* wei, bool int8) {
            dnnl::memory::desc src_md({s.M, s.K}, in_t, tag::ab), wei_md({s.K, s.N}, in_t, tag::ba),
                dst_md({s.M, s.N}, dt::f32, tag::ab);
            dnnl::primitive_attr attr;
            if (int8) {
                attr.set_scales_mask(DNNL_ARG_SRC, 0);
                attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
            }
            dnnl::matmul::primitive_desc pd(eng, src_md, wei_md, dst_md, attr);
            dnnl::matmul mm(pd);
            auto m_src = dnnl::sycl_interop::make_memory(src_md, eng, dnnl::sycl_interop::memory_kind::usm, src);
            auto m_wei = dnnl::sycl_interop::make_memory(wei_md, eng, dnnl::sycl_interop::memory_kind::usm, wei);
            auto m_dst = dnnl::sycl_interop::make_memory(dst_md, eng, dnnl::sycl_interop::memory_kind::usm, Y);
            std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, m_src}, {DNNL_ARG_WEIGHTS, m_wei}, {DNNL_ARG_DST, m_dst}};
            if (int8) {
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({1}, dt::f32, tag::a), eng, dnnl::sycl_interop::memory_kind::usm, sx);
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({s.N}, dt::f32, tag::a), eng, dnnl::sycl_interop::memory_kind::usm, sw);
            }
            return time([&] { mm.execute(strm, args); });
        };
        const double t_df = run_dnnl(dt::f16, X, W, false);
        const double t_di = run_dnnl(dt::s8, Xi, Wi, true);
        const double flop = 2.0 * s.M * s.N * s.K;
        auto tf = [&](double ms) { return flop / (ms * 1e-3) / 1e12; };
        std::printf("%-20s %6lld %6lld %6lld | %7.1f (%6.3f) %7.1f (%6.3f) %7.1f (%6.3f) | dnnl/mkl %.2fx, int8/mkl %.2fx\n",
                    s.what, (long long) s.M, (long long) s.K, (long long) s.N, tf(t_mkl), t_mkl, tf(t_df), t_df, tf(t_di),
                    t_di, t_mkl / t_df, t_mkl / t_di);
        for (void* p : {(void*) X, (void*) W, (void*) Xi, (void*) Wi, (void*) Y, (void*) sw, (void*) sx}) sycl::free(p, q);
    }
    return 0;
}
