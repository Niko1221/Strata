#include "native_qsa.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(int card, int position, int rows, int max_pools, bool ties) {
    dpct::select_device(card);
    sycl::queue queue(dpct::get_current_device(),
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}});
    const int pages = (max_pools + fnx::KVP_BLOCKS - 1) / fnx::KVP_BLOCKS;
    fnx::KvPages layout{};
    layout.page = fnx::KVP_BLOCKS * 128 + 256;
    std::vector<int> table(pages);
    std::iota(table.rbegin(), table.rend(), 0);
    std::vector<uint16_t> pooled((size_t)pages * layout.page, 0);
    std::vector<float> query((size_t)rows * 512);
    std::mt19937 rng(91);
    for (auto& v : query) v = ties ? 1.f : (int(rng() % 65) - 32) / 32.f;
    for (int b = 0; b < max_pools; ++b)
        for (int d = 0; d < 128; ++d) {
            const float v = ties ? 1.f : (int(rng() % 65) - 32) / 32.f;
            pooled[(size_t)table[b / fnx::KVP_BLOCKS] * layout.page +
                   b % fnx::KVP_BLOCKS * 128 + d] = sycl::bit_cast<uint16_t>(sycl::half(v));
        }
    std::vector<void*> buffers;
    auto alloc = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, queue);
        if (!p) throw std::bad_alloc();
        buffers.push_back(p);
        return p;
    };
    auto upload = [&](const void* src, size_t bytes) {
        void* dst = alloc(bytes);
        void* pinned = sycl::malloc_host(bytes, queue);
        if (!pinned) throw std::bad_alloc();
        std::memcpy(pinned, src, bytes);
        try {
            queue.memcpy(dst, pinned, bytes).wait_and_throw();
        } catch (...) {
            sycl::free(pinned, queue);
            throw;
        }
        sycl::free(pinned, queue);
        return dst;
    };
    std::cout << "BEGIN QSA card=" << card << " pos=" << position
              << " rows=" << rows << " ties=" << ties << std::endl;
    try {
        auto* dq = static_cast<float*>(upload(query.data(), query.size() * sizeof(float)));
        auto* dk = static_cast<uint16_t*>(upload(pooled.data(), pooled.size() * sizeof(uint16_t)));
        auto* pos = static_cast<int*>(upload(&position, sizeof(position)));
        layout.pt = static_cast<int*>(upload(table.data(), table.size() * sizeof(int)));
        auto* scores = static_cast<float*>(alloc((size_t)rows * max_pools * sizeof(float)));
        auto* cells = static_cast<int*>(alloc((size_t)rows * fnx::qsa_maxsel() * sizeof(int)));
        auto* counts = static_cast<int*>(alloc((size_t)rows * sizeof(int)));
        fnx::NativeQsa scorer(&queue, rows, max_pools);
        fnx::qsa_select(dq, dk, layout, scores, cells, counts, max_pools, pos,
                        rows, card, &queue, &scorer, (position + rows) / 4);
        std::vector<float> actual_scores((size_t)rows * max_pools);
        std::vector<int> actual_cells((size_t)rows * fnx::qsa_maxsel()), actual_counts(rows);
        queue.memcpy(actual_scores.data(), scores, actual_scores.size() * sizeof(float));
        queue.memcpy(actual_cells.data(), cells, actual_cells.size() * sizeof(int));
        queue.memcpy(actual_counts.data(), counts, actual_counts.size() * sizeof(int)).wait_and_throw();
        for (int r = 0; r < rows; ++r) {
            const int nv = (position + r + 1) / 4, tail = (position + r + 1) % 4;
            if (nv <= fnx::QSA_TOP_K / 4) {
                if (actual_counts[r] != -1) throw std::runtime_error("QSA dense sentinel mismatch");
                continue;
            }
            std::vector<double> reference(nv, 0);
            for (int b = 0; b < nv; ++b)
                for (int h = 0; h < 4; ++h) {
                    double dot = 0;
                    for (int d = 0; d < 128; ++d) {
                        const size_t off = (size_t)table[b / fnx::KVP_BLOCKS] * layout.page +
                                           b % fnx::KVP_BLOCKS * 128 + d;
                        dot += double(float(sycl::bit_cast<sycl::half>(pooled[off]))) *
                               query[(size_t)r * 512 + h * 128 + d];
                    }
                    reference[b] += std::max(dot, 0.0);
                }
            for (int b = 0; b < nv; ++b)
                if (std::fabs(actual_scores[(size_t)r * max_pools + b] - reference[b]) > 1e-5)
                    throw std::runtime_error("PP GEMM QSA scorer disagrees with FP64 reference");
            std::vector<int> order(nv);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return reference[a] > reference[b]; });
            order.resize(fnx::QSA_TOP_K / 4);
            std::sort(order.begin(), order.end());
            std::vector<int> expected;
            for (int b : order)
                if (b % fnx::NCARD == card)
                    for (int t = 0; t < 4; ++t) expected.push_back(b / fnx::NCARD * 4 + t);
            if (nv % fnx::NCARD == card)
                for (int t = 0; t < tail; ++t) expected.push_back(nv / fnx::NCARD * 4 + t);
            if (actual_counts[r] != int(expected.size()) ||
                !std::equal(expected.begin(), expected.end(), actual_cells.begin() + (size_t)r * fnx::qsa_maxsel()))
                throw std::runtime_error("QSA causal/tie/card-local selection mismatch");
        }
        std::cout << "PASS QSA FP64 scores, causal selection, lower-index ties, permuted pages and tail ownership" << std::endl;
        queue.wait_and_throw();
    } catch (...) {
        queue.wait_and_throw();
        for (void* p : buffers) sycl::free(p, queue);
        throw;
    }
    for (void* p : buffers) sycl::free(p, queue);
}
}

int main() {
    try {
        if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two B70 GPUs");
        for (int card = 0; card < 2; ++card) {
            check(card, 0, 5, 64, false);
            check(card, 2035, 33, 8320, false);
            check(card, 2048, 5, 8320, true);
            check(card, 32760, 33, 8320, false);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
