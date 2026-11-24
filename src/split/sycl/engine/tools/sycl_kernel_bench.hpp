#pragma once
#include "kernels.hpp"
#include "strata/kprof.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel_bench {
struct Fixture {
    sycl::queue& q;
    std::vector<void*> buffers;
    explicit Fixture(sycl::queue& queue) : q(queue) {}
    ~Fixture() {
        q.wait();
        for (void* p : buffers) sycl::free(p, q);
    }
    template<class T> T* alloc(size_t n) {
        auto* p = sycl::malloc_device<T>(n, q);
        if (!p) throw std::bad_alloc();
        buffers.push_back(p);
        return p;
    }
    template<class T> T* upload(const std::vector<T>& values) {
        std::cout << "UPLOAD bytes=" << values.size() * sizeof(T) << std::endl;
        auto* p = alloc<T>(values.size());
        auto* h = sycl::malloc_host<T>(values.size(), q);
        if (!h) throw std::bad_alloc();
        std::copy(values.begin(), values.end(), h);
        try {
            q.memcpy(p, h, values.size() * sizeof(T)).wait_and_throw();
        } catch (...) {
            sycl::free(h, q);
            throw;
        }
        sycl::free(h, q);
        return p;
    }
    template<class T> std::vector<T> read(const T* p, size_t n) {
        auto* h = sycl::malloc_host<T>(n, q);
        if (!h) throw std::bad_alloc();
        try {
            q.memcpy(h, p, n * sizeof(T)).wait_and_throw();
        } catch (...) {
            sycl::free(h, q);
            throw;
        }
        std::vector<T> result(h, h + n);
        sycl::free(h, q);
        return result;
    }
    float* random(size_t n, float amplitude, unsigned seed) {
        std::mt19937 rng(seed);
        std::vector<float> h(n);
        for (float& v : h) v = (int(rng() % 201) - 100) * amplitude / 100.f;
        return upload(h);
    }
};

inline void clear_events() {
    auto& state = strata::kprof::st();
    std::lock_guard<std::mutex> lock(state.mu);
    state.recs.clear();
}

struct Sample {
    double gpu_ms = 0, tag_ms = 0, span_ms = 0, submit_ms = 0, wall_ms = 0;
    int kernels = 0, sections = 0;
    std::map<std::string, double> sites;
};

inline Sample measure(sycl::queue& q, const std::function<void()>& run) {
    q.wait_and_throw();
    clear_events();
    Sample result;
    auto start = std::chrono::steady_clock::now();
    run();
    auto submitted = std::chrono::steady_clock::now();
    q.wait_and_throw();
    auto done = std::chrono::steady_clock::now();
    result.submit_ms = std::chrono::duration<double, std::milli>(submitted - start).count();
    result.wall_ms = std::chrono::duration<double, std::milli>(done - start).count();
    std::vector<strata::kprof::Rec> records;
    {
        auto& state = strata::kprof::st();
        std::lock_guard<std::mutex> lock(state.mu);
        records.swap(state.recs);
    }
    uint64_t first = UINT64_MAX, last = 0;
    for (const auto& rec : records) {
        uint64_t begin, end;
        std::string name;
        if (rec.section) {
            begin = rec.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            end = rec.ev1.get_profiling_info<sycl::info::event_profiling::command_end>();
            name = "section " + rec.label;
            ++result.sections;
        } else {
            begin = rec.ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            end = rec.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            name = std::string(strata::kprof::base(rec.file)) + ":" + rec.fn +
                   ":" + std::to_string(rec.line);
            ++result.kernels;
        }
        if (end < begin) throw std::runtime_error("Invalid profiling timestamps");
        double ms = (end - begin) / 1e6;
        (rec.section ? result.tag_ms : result.gpu_ms) += ms;
        result.sites[name] += ms;
        first = std::min(first, begin);
        last = std::max(last, end);
    }
    if (records.empty()) throw std::runtime_error("No profiling events recorded");
    result.span_ms = (last - first) / 1e6;
    return result;
}

inline void emit(int card, const std::string& name, const char* mode, int iteration,
                 const Sample& s) {
    std::cout << std::setprecision(9)
              << "SAMPLE," << card << ',' << name << ',' << mode << ',' << iteration
              << ',' << s.gpu_ms << ',' << s.tag_ms << ',' << s.span_ms
              << ',' << s.submit_ms << ',' << s.wall_ms << ',' << s.kernels
              << ',' << s.sections << '\n';
    for (const auto& [site, ms] : s.sites)
        std::cout << "SITE," << card << ',' << name << ',' << mode << ','
                  << iteration << ',' << site << ',' << ms << '\n';
}

inline void paired(sycl::queue& q, int card, const std::string& name,
                   const std::function<void()>& pp, const std::function<void()>& split,
                   int warmup, int repeats, const std::function<void()>& prepare = [] {}) {
    for (int i = 0; i < warmup; ++i) {
        prepare(); pp(); q.wait_and_throw();
        prepare(); split(); q.wait_and_throw();
        clear_events();
    }
    for (int i = 0; i < repeats; ++i) {
        auto run = [&](bool allcards) {
            prepare();
            emit(card, name, allcards ? "allcards" : "pp2", i,
                 measure(q, allcards ? split : pp));
        };
        // Alternate AB / BA to reduce ordering and clock drift bias.
        run(i % 2 != 0);
        run(i % 2 == 0);
    }
    std::cout << "DONE card=" << card << " case=" << name << std::endl;
}

inline void compare(const std::vector<float>& a, const std::vector<float>& b,
                    float abs_tolerance, float relative_l2, const std::string& name) {
    if (a.size() != b.size()) throw std::runtime_error("Different parity shapes");
    double error2 = 0, norm2 = 0;
    float worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            throw std::runtime_error("Nonfinite parity output: " + name);
        double error = double(a[i]) - b[i];
        error2 += error * error;
        norm2 += double(a[i]) * a[i];
        worst = std::max(worst, float(std::fabs(error)));
    }
    double rel = std::sqrt(error2 / std::max(norm2, 1e-30));
    if (worst > abs_tolerance || rel > relative_l2)
        throw std::runtime_error("Parity mismatch: " + name +
                                 " max_abs=" + std::to_string(worst) +
                                 " relative_l2=" + std::to_string(rel));
    std::cout << "PARITY " << name << " max_abs=" << worst
              << " relative_l2=" << rel << " PASS" << std::endl;
}
} // namespace kernel_bench

namespace fnx {
void check_attention_copy(bool direct, int repeats, int rows);
void check_attention_landing_reuse(bool ack, int sender, int repeats, int rows);
void check_hc_swap(const std::string& mode, int repeats, int rows, int cols, int pieces);
void benchmark_hc(sycl::queue& q, int card, int rows, int warmup, int repeats,
                  bool update_only = false);
}
