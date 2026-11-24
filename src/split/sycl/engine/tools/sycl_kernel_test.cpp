#include "kernels.hpp"

#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check_router(int card) {
    dpct::select_device(card);
    auto& q = dpct::get_in_order_queue();
    std::vector<void*> allocations;
    auto alloc = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    };
    try {
        auto* weights = static_cast<float*>(alloc(size_t(fnx::NEXP + 1) * fnx::N * sizeof(float)));
        auto* input = static_cast<float*>(alloc(fnx::MAXT * fnx::N * sizeof(float)));
        auto* logits = static_cast<float*>(alloc(fnx::MAXT * (fnx::NEXP + 1) * sizeof(float)));
        auto* indices = static_cast<int*>(alloc(fnx::MAXT * fnx::TOPK * sizeof(int)));
        auto* probabilities = static_cast<float*>(alloc(fnx::MAXT * fnx::TOPK * sizeof(float)));
        auto* gu = static_cast<fnx::Job*>(alloc((fnx::MAXT * fnx::TOPK + 1) * sizeof(fnx::Job)));
        auto* down = static_cast<fnx::Job*>(alloc((fnx::MAXT * fnx::TOPK + 1) * sizeof(fnx::Job)));
        auto* counts = static_cast<int*>(alloc(2 * sizeof(int)));
        q.memset(weights, 0, size_t(fnx::NEXP + 1) * fnx::N * sizeof(float));
        q.parallel_for(sycl::range<1>(fnx::NEXP + 1), [=](sycl::id<1> id) {
            weights[size_t(id[0]) * fnx::N] = float(id[0]) * 0.01f;
        });
        q.memset(counts, 0, 2 * sizeof(int));
        double normalizer = 0;
        for (int k = 0; k < fnx::TOPK; ++k) normalizer += std::exp(-0.01 * k);
        for (int tokens : {1, 2, 8, fnx::MAXT})
            for (int round = 0; round < 64; ++round) {
                const int sign = round & 1 ? -1 : 1;
                q.parallel_for(sycl::range<1>(fnx::MAXT * fnx::N), [=](sycl::id<1> id) {
                    input[id] = id[0] % fnx::N == 0 ? float(sign) : 0.f;
                });
                fnx::router_route(weights, input, logits, tokens, indices, probabilities,
                                  gu, down, counts, &q);
                std::vector<int> got(tokens * fnx::TOPK);
                std::vector<float> values(got.size());
                q.memcpy(got.data(), indices, got.size() * sizeof(int));
                q.memcpy(values.data(), probabilities, values.size() * sizeof(float)).wait_and_throw();
                for (int t = 0; t < tokens; ++t)
                    for (int k = 0; k < fnx::TOPK; ++k)
                        if (got[t * fnx::TOPK + k] != (sign > 0 ? fnx::NEXP - 1 - k : k) ||
                            std::fabs(values[t * fnx::TOPK + k] -
                                      std::exp(-0.01 * k) / normalizer) > 0.00001)
                            throw std::runtime_error("Router top-k disagrees with completed-logit reference");
            }
        std::cout << "PASS router card=" << card
                  << " tokens=1,2,8,16 alternating_rounds=64 cpu_topk=PASS" << std::endl;
    } catch (...) {
        q.wait_and_throw();
        for (void* p : allocations) sycl::free(p, q);
        throw;
    }
    for (void* p : allocations) sycl::free(p, q);
}

struct Q8 {
    sycl::half d, s;
    int8_t q[32];
};
static_assert(sizeof(Q8) == 36);

void check_hc_stages(int card) {
    dpct::select_device(card);
    auto& q = dpct::get_in_order_queue();
    std::vector<void*> allocations;
    std::vector<std::pair<void*, size_t>> outputs;
    auto alloc = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    };
    auto guarded = [&](size_t bytes) {
        auto* p = static_cast<unsigned char*>(alloc(bytes + 64));
        outputs.emplace_back(p, bytes + 64);
        return p + 32;
    };
    try {
        auto* R = reinterpret_cast<float*>(guarded(fnx::MAXT * fnx::D * sizeof(float)));
        auto* mixed = reinterpret_cast<float*>(guarded(fnx::MAXT * fnx::N * sizeof(float)));
        auto* inject = reinterpret_cast<float*>(guarded(fnx::MAXT * fnx::HC * sizeof(float)));
        auto* quantized = guarded(fnx::MAXT * fnx::N / 32 * sizeof(Q8));
        auto* bo = static_cast<float*>(alloc(fnx::MAXT * fnx::N * sizeof(float)));
        auto* previous = static_cast<float*>(alloc(fnx::MAXT * fnx::HC * sizeof(float)));
        auto* norm = static_cast<float*>(alloc(fnx::D * sizeof(float)));
        const size_t weight_count = size_t(fnx::LR) * fnx::D;
        auto* weights = static_cast<int8_t*>(alloc(weight_count));
        auto* scales = static_cast<sycl::half*>(alloc(weight_count / 32 * sizeof(sycl::half)));
        auto* inj_weights = static_cast<uint16_t*>(alloc(fnx::HC * fnx::D * sizeof(uint16_t)));
        auto* scratch = static_cast<float*>(alloc(fnx::hc_scratch_floats(fnx::MAXT) * sizeof(float)));
        auto* error = static_cast<unsigned*>(alloc(sizeof(unsigned)));
        q.parallel_for(sycl::range<1>(weight_count), [=](sycl::id<1> i) {
            weights[i] = int(i[0] * 13 % 23) - 11;
        });
        q.fill(scales, sycl::half(0.002f), weight_count / 32);
        q.parallel_for(sycl::range<1>(fnx::HC * fnx::D), [=](sycl::id<1> i) {
            inj_weights[i] = i[0] % 2 ? 0x3c00 : 0xbc00;
        });
        q.parallel_for(sycl::range<1>(fnx::D), [=](sycl::id<1> i) {
            norm[i] = 0.9f + float(i[0] % 11) * 0.02f;
        });
        q.parallel_for(sycl::range<1>(fnx::MAXT * fnx::N), [=](sycl::id<1> i) {
            bo[i] = float(int(i[0] * 17 % 31) - 15) * 0.005f;
        });
        q.fill(previous, 0.3f, fnx::MAXT * fnx::HC);
        fnx::hc_prepare(scratch);
        const fnx::Q8S w{weights, reinterpret_cast<uint16_t*>(scales)};
        double milliseconds[2] = {};
        int cases = 0;
        for (int tokens : {1, 2, 8, fnx::MAXT})
            for (int variant = 0; variant < 4; ++variant)
                for (int round = 0; round < 4; ++round) {
                    std::vector<float> initial(fnx::MAXT * fnx::D);
                    for (size_t i = 0; i < initial.size(); ++i)
                        initial[i] = float(int((i * 19 + round * 7) % 127) - 63) * 0.01f;
                    std::vector<unsigned char> snapshots[2];
                    for (int order = 0; order < 2; ++order) {
                        const int split = (order + round) % 2;
                        setenv("FNX_HC_STAGE_SPLIT", split ? "1" : "0", 1);
                        for (const auto& buffer : outputs) q.memset(buffer.first, 0xa5, buffer.second);
                        q.memcpy(R, initial.data(), initial.size() * sizeof(float));
                        q.memset(scratch, 0, fnx::hc_scratch_floats(fnx::MAXT) * sizeof(float));
                        q.memset(error, 0, sizeof(unsigned)).wait_and_throw();
                        const auto begin = std::chrono::steady_clock::now();
                        fnx::hc_read(R, variant & 1 ? bo : nullptr, previous, norm, w, w,
                                     variant & 2 ? inj_weights : nullptr, 1e-6f,
                                     mixed, inject, scratch, error, tokens, &q, quantized);
                        q.wait_and_throw();
                        if (round) milliseconds[split] +=
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - begin).count();
                        unsigned errors = 0;
                        q.memcpy(&errors, error, sizeof(errors)).wait_and_throw();
                        if (errors) throw std::runtime_error("HC error word set");
                        for (const auto& buffer : outputs) {
                            std::vector<unsigned char> got(buffer.second);
                            q.memcpy(got.data(), buffer.first, got.size()).wait_and_throw();
                            for (size_t i = 0; i < 32; ++i)
                                if (got[i] != 0xa5 || got[got.size() - 1 - i] != 0xa5)
                                    throw std::runtime_error("HC output guard overwritten");
                            snapshots[split].insert(snapshots[split].end(), got.begin(), got.end());
                        }
                    }
                    if (snapshots[0] != snapshots[1])
                        throw std::runtime_error("HC staged output differs from single-group reference");
                    ++cases;
                }
        unsetenv("FNX_HC_STAGE_SPLIT");
        std::cout << "PASS hc_stages card=" << card << " cases=" << cases
                  << " tokens=1,2,8,16 folded_write_and_inject=all guards=PASS"
                  << " host_wait_total_off_ms=" << milliseconds[0]
                  << " on_ms=" << milliseconds[1] << std::endl;
    } catch (...) {
        unsetenv("FNX_HC_STAGE_SPLIT");
        q.wait_and_throw();
        for (void* p : allocations) sycl::free(p, q);
        throw;
    }
    for (void* p : allocations) sycl::free(p, q);
}

void check_card(int card) {
    dpct::select_device(card);
    auto& queue = dpct::get_in_order_queue();
    constexpr int cols = fnx::N, rows = 32, tokens = fnx::MAXT;
    std::vector<float> x(cols * tokens);
    std::vector<int8_t> weights(rows * cols);
    std::vector<sycl::half> scales(rows * cols / 32, sycl::half(0.125f));
    for (size_t i = 0; i < x.size(); ++i) x[i] = (int(i * 17 % 255) - 127) * 0.0625f;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = int(i * 13 % 31) - 15;
    auto* dx = sycl::malloc_device<float>(x.size(), queue);
    auto* dw = sycl::malloc_device<int8_t>(weights.size(), queue);
    auto* ds = sycl::malloc_device<sycl::half>(scales.size(), queue);
    auto* dq = sycl::malloc_device<Q8>(cols * tokens / 32, queue);
    auto* dy = sycl::malloc_device<float>(rows * tokens, queue);
    if (!dx || !dw || !ds || !dq || !dy) throw std::bad_alloc();
    try {
        queue.memcpy(dx, x.data(), x.size() * sizeof(float)).wait_and_throw();
        queue.memcpy(dw, weights.data(), weights.size()).wait_and_throw();
        queue.memcpy(ds, scales.data(), scales.size() * sizeof(sycl::half)).wait_and_throw();
        fnx::quantize_q8_1(dx, dq, cols, tokens, &queue);
        std::vector<Q8> quantized(cols * tokens / 32);
        queue.memcpy(quantized.data(), dq, quantized.size() * sizeof(Q8)).wait_and_throw();
        for (const int count : {1, 2, tokens}) {
            fnx::Q8S w{dw, reinterpret_cast<uint16_t*>(ds)};
            fnx::q8s_gemv(w, rows, cols, dq, dy, rows, count, &queue);
            std::vector<float> got(rows * count);
            queue.memcpy(got.data(), dy, got.size() * sizeof(float)).wait_and_throw();
            float worst = 0;
            for (int t = 0; t < count; ++t)
                for (int r = 0; r < rows; ++r) {
                    float expected = 0;
                    for (int b = 0; b < cols / 32; ++b) {
                        const Q8& q = quantized[t * cols / 32 + b];
                        int dot = 0;
                        for (int j = 0; j < 32; ++j)
                            dot += int(weights[r * cols + b * 32 + j]) * int(q.q[j]);
                        expected += float(dot) * float(q.d) * 0.125f;
                    }
                    const float error = std::fabs(got[t * rows + r] - expected);
                    worst = std::max(worst, error);
                    if (!std::isfinite(got[t * rows + r]) ||
                        error > 0.002f + std::fabs(expected) * 1e-5f)
                        throw std::runtime_error("Q8S GEMV disagrees with quantized CPU reference");
                }
            std::cout << "PASS card=" << card << " tokens=" << count
                      << " q8s_gemv_max_error=" << worst << std::endl;
        }
        auto* products = sycl::malloc_shared<int>(32 * 4 * 8, queue);
        if (!products) throw std::bad_alloc();
        try {
            queue.parallel_for(sycl::nd_range<1>(256, 32),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    const int test = item.get_group(0), lane = item.get_local_id(0);
                    const int av = test % 4 == 0 ? 255 : test % 4 == 1 ? 128 :
                                   test % 4 == 2 ? 127 : 0;
                    const int bv = test < 4 ? -128 : 127;
                    const uint32_t ap = uint32_t(av) * 0x01010101;
                    const uint32_t bp = uint32_t(uint8_t(bv)) * 0x01010101;
                    uint32_t a[4]{ap, ap, ap, ap}, b[2]{bp, bp};
                    int c[4]{lane, lane + 1, lane + 2, lane + 3};
                    fnx::sycl_port::mma_u8s8(c, a, b);
                    for (int i = 0; i < 4; ++i)
                        products[(test * 32 + lane) * 4 + i] = c[i];
                }).wait_and_throw();
            for (int test = 0; test < 8; ++test) {
                const int av = test % 4 == 0 ? 255 : test % 4 == 1 ? 128 :
                               test % 4 == 2 ? 127 : 0;
                const int bv = test < 4 ? -128 : 127;
                for (int lane = 0; lane < 32; ++lane)
                    for (int i = 0; i < 4; ++i)
                        if (products[(test * 32 + lane) * 4 + i] != 32 * av * bv + lane + i)
                            throw std::runtime_error("Unsigned/signed attention MMA disagrees with CPU reference");
            }
        } catch (...) {
            sycl::free(products, queue);
            throw;
        }
        sycl::free(products, queue);
        std::cout << "PASS card=" << card << " attention_mma_u8s8 cases=8" << std::endl;
    } catch (...) {
        queue.wait_and_throw();
        sycl::free(dx, queue); sycl::free(dw, queue); sycl::free(ds, queue);
        sycl::free(dq, queue); sycl::free(dy, queue);
        throw;
    }
    sycl::free(dx, queue); sycl::free(dw, queue); sycl::free(ds, queue);
    sycl::free(dq, queue); sycl::free(dy, queue);
}
}

int main() {
    try {
        if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two GPUs");
        if (!std::getenv("FNX_ROUTER_SPLIT")) setenv("FNX_ROUTER_SPLIT", "1", 1);
        for (int card = 0; card < 2; ++card) {
            check_card(card);
            check_router(card);
            if (std::getenv("FNX_TEST_HC_STAGE_SPLIT")) check_hc_stages(card);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
