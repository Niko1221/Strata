#include "native_gdn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(int card, int rows, bool benchmark) {
    std::cout << "SETUP card=" << card << " rows=" << rows << std::endl;
    dpct::select_device(card);
    sycl::queue q(dpct::get_current_device(),
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}});
    constexpr int nk = fnx::NativeGdn::nk, nv = fnx::NativeGdn::nv;
    constexpr int hs = fnx::NativeGdn::hs, ch = (2 * nk + nv) * hs;
    const int groups = (rows + fnx::MAXT - 1) / fnx::MAXT;
    std::vector<void*> buffers;
    std::mt19937 rng(17);
    auto random = [&](size_t n, float amplitude) {
        auto* p = sycl::malloc_device<float>(n, q);
        if (!p) throw std::bad_alloc();
        buffers.push_back(p);
        auto* h = sycl::malloc_host<float>(n, q);
        if (!h) throw std::bad_alloc();
        for (size_t i = 0; i < n; ++i)
            h[i] = (int(rng() % 201) - 100) * amplitude / 100.f;
        std::cout << "UPLOAD floats=" << n << std::endl;
        try {
            q.memcpy(p, h, n * sizeof(float)).wait_and_throw();
        } catch (...) {
            sycl::free(h, q);
            throw;
        }
        sycl::free(h, q);
        return p;
    };
    auto clone = [&](float* p, size_t n) {
        auto* c = sycl::malloc_device<float>(n, q);
        if (!c) throw std::bad_alloc();
        buffers.push_back(c);
        q.memcpy(c, p, n * sizeof(float)).wait_and_throw();
        return c;
    };
    try {
        auto* qkvz = random((size_t)rows * (ch + nv * hs), 0.3f);
        auto* ab = random((size_t)rows * 2 * nv, 0.2f);
        auto* conv = random(ch * 4, 0.2f);
        auto* dt = random(nv, 0.1f);
        auto* a = random(nv, 0);
        std::vector<float> ha(nv, -0.03f);
        q.memcpy(a, ha.data(), nv * sizeof(float)).wait_and_throw();
        auto* norm = random(hs, 0.9f);
        auto* state = random(nv * hs * hs, 0.001f);
        auto* native_state = clone(state, nv * hs * hs);
        auto* history = random(6 * ch, 0.3f);
        auto* native_history = clone(history, 6 * ch);
        auto* xc = random((size_t)rows * ch, 0);
        auto* sq = random((size_t)rows * nv * 16, 0);
        auto* out = random((size_t)rows * nv * hs, 0);
        auto* native_out = clone(out, (size_t)rows * nv * hs);
        auto* work = random(fnx::gdn_prompt_work_floats(rows, nv), 0);
        std::vector<int> hctl(2 + groups * 2), hmap(nv);
        for (int g = 0; g < groups; ++g) {
            hctl[2 + g * 2] = g * fnx::MAXT;
            hctl[3 + g * 2] = 11 + g;
        }
        for (int h = 0; h < nv; ++h) hmap[h] = h % nk | (h < nk ? 256 : 0);
        auto* ctl = sycl::malloc_device<int>(hctl.size(), q);
        auto* map = sycl::malloc_device<int>(nv, q);
        if (!ctl || !map) throw std::bad_alloc();
        buffers.push_back(ctl); buffers.push_back(map);
        q.memcpy(ctl, hctl.data(), hctl.size() * sizeof(int));
        q.memcpy(map, hmap.data(), nv * sizeof(int)).wait_and_throw();
        std::cout << "RUN legacy GDN" << std::endl;
        fnx::gdn_prompt(qkvz, ab, history, conv, dt, a, norm, state,
                         ctl, groups, nk, nv, map, rows, 1e-6f, xc, sq, out, work, &q);
        q.wait_and_throw();
        std::cout << "RUN PP-source GDN" << std::endl;
        fnx::NativeGdn native(&q, rows);
        native.run(qkvz, ab, native_history, conv, dt, a, norm, native_state,
                     ctl, groups, rows, 1e-6f, native_out);
        auto compare = [&](float* expected, float* actual, size_t n,
                           float tolerance, const char* name) {
            std::vector<float> h1(n), h2(n);
            q.memcpy(h1.data(), expected, n * sizeof(float));
            q.memcpy(h2.data(), actual, n * sizeof(float)).wait_and_throw();
            float worst = 0, scale = 0;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(h2[i])) throw std::runtime_error("Nonfinite native GDN result");
                worst = std::max(worst, std::fabs(h1[i] - h2[i]));
                scale = std::max(scale, std::fabs(h1[i]));
            }
            if (worst > tolerance * (1.f + scale))
                throw std::runtime_error(std::string("Native GDN mismatch: ") + name);
            std::cout << "PASS card=" << card << " rows=" << rows
                      << " " << name << "_max_error=" << worst << std::endl;
        };
        compare(out, native_out, (size_t)rows * nv * hs, 2e-4f, "output");
        compare(state, native_state, nv * hs * hs, 2e-5f, "state");
        compare(history, native_history, 6 * ch, 0, "history");
        if (benchmark) {
            auto timed = [&](bool use_native) {
                std::vector<double> times;
                for (int rep = 0; rep < 4; ++rep) {
                    const auto start = std::chrono::steady_clock::now();
                    if (use_native)
                        native.run(qkvz, ab, native_history, conv, dt, a, norm,
                                   native_state, ctl, groups, rows, 1e-6f, native_out);
                    else
                        fnx::gdn_prompt(qkvz, ab, history, conv, dt, a, norm, state,
                            ctl, groups, nk, nv, map, rows, 1e-6f, xc, sq, out, work, &q);
                    q.wait_and_throw();
                    if (rep) times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count());
                }
                std::sort(times.begin(), times.end());
                return times[1];
            };
            const double legacy = timed(false), adapter = timed(true);
            std::cout << "GDN_BENCH card=" << card << " rows=" << rows
                      << " legacy_ms=" << legacy << " PP_adapter_ms=" << adapter << std::endl;
        }
        q.wait_and_throw();
    } catch (...) {
        q.wait_and_throw();
        for (void* p : buffers) sycl::free(p, q);
        throw;
    }
    for (void* p : buffers) sycl::free(p, q);
}
}

int main(int argc, char** argv) {
    try {
        if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two B70 GPUs");
        const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
        for (int card = 0; card < 2; ++card) {
            for (int rows : {1, 17, 65}) check(card, rows, false);
            if (benchmark) check(card, 4096, true);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
