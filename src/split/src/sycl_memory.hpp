// The SYCL split's expert placement, using the loader's actual per-card strides.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fnx {

struct SyclMemoryBudget {
    size_t free_bytes;
    size_t fixed_bytes;
    size_t reserve_bytes;

    size_t expert_bytes() const {
        if (fixed_bytes > free_bytes || reserve_bytes > free_bytes - fixed_bytes)
            throw std::runtime_error("SYCL split: fixed buffers and reserve exceed free VRAM");
        return free_bytes - fixed_bytes - reserve_bytes;
    }
};

struct SyclExpertPlacement {
    std::vector<std::vector<char>> resident;
    std::vector<size_t> device_bytes;
    // The cache keeps a host copy of resident experts too, for later replacement.
    std::vector<size_t> host_bytes;
    std::vector<size_t> nonresident_bytes;
    size_t evicted = 0;
};

inline size_t sycl_size_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error("SYCL split: allocation size overflow");
    return a + b;
}

// Each pool must keep two windows of slots on EVERY card. A pool's identity includes
// both strides on every card, not just its unsliced size or the first card's size.
inline SyclExpertPlacement sycl_expert_placement(
    const std::vector<std::vector<std::pair<size_t, size_t>>>& sizes,
    const std::vector<double>& score, int nexp, int floor,
    const std::vector<SyclMemoryBudget>& budgets) {
    if (sizes.size() < 2 || sizes.size() != budgets.size() || sizes[0].empty() ||
        nexp <= 0 || floor <= 0)
        throw std::runtime_error("SYCL split: invalid expert placement geometry");
    const size_t nl = sizes[0].size(), nc = sizes.size();
    if (nl > std::numeric_limits<size_t>::max() / (size_t) nexp ||
        score.size() != nl * (size_t) nexp)
        throw std::runtime_error("SYCL split: expert scores do not match the model");
    for (double s : score)
        if (!std::isfinite(s)) throw std::runtime_error("SYCL split: non-finite expert score");
    std::vector<size_t> capacity;
    for (size_t c = 0; c < nc; ++c) {
        if (sizes[c].size() != nl)
            throw std::runtime_error("SYCL split: inconsistent layer counts");
        capacity.push_back(budgets[c].expert_bytes());
    }
    std::vector<std::vector<std::pair<size_t, size_t>>> pools;
    std::vector<size_t> pool_of(nl), left;
    for (size_t l = 0; l < nl; ++l) {
        std::vector<std::pair<size_t, size_t>> key;
        for (size_t c = 0; c < nc; ++c) {
            if (!sizes[c][l].first || !sizes[c][l].second)
                throw std::runtime_error("SYCL split: empty expert slice");
            key.push_back(sizes[c][l]);
        }
        auto it = std::find(pools.begin(), pools.end(), key);
        if (it == pools.end()) {
            pools.push_back(key);
            left.push_back(0);
            it = pools.end() - 1;
        }
        pool_of[l] = (size_t) (it - pools.begin());
        left[pool_of[l]] = sycl_size_add(left[pool_of[l]], (size_t) nexp);
    }
    for (size_t n : left)
        if (n < (size_t) floor)
            throw std::runtime_error("SYCL split: expert pool cannot hold two windows");
    SyclExpertPlacement plan;
    plan.resident.assign(nl, std::vector<char>((size_t) nexp, 1));
    plan.device_bytes.assign(nc, 0);
    plan.host_bytes.assign(nc, 0);
    plan.nonresident_bytes.assign(nc, 0);
    for (size_t c = 0; c < nc; ++c)
        for (size_t l = 0; l < nl; ++l) {
            const size_t stride = sycl_size_add(sizes[c][l].first, sizes[c][l].second);
            if (stride > std::numeric_limits<size_t>::max() / (size_t) nexp)
                throw std::runtime_error("SYCL split: expert allocation size overflow");
            plan.device_bytes[c] = sycl_size_add(plan.device_bytes[c], stride * (size_t) nexp);
        }
    plan.host_bytes = plan.device_bytes;
    const auto fits = [&] {
        for (size_t c = 0; c < nc; ++c)
            if (plan.device_bytes[c] > capacity[c]) return false;
        return true;
    };
    std::vector<size_t> order(score.size());
    std::iota(order.begin(), order.end(), (size_t) 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return score[a] < score[b]; });
    for (size_t i : order) {
        if (fits()) break;
        const size_t l = i / (size_t) nexp, p = pool_of[l];
        if (left[p] == (size_t) floor) continue;
        --left[p];
        plan.resident[l][i % (size_t) nexp] = 0;
        ++plan.evicted;
        for (size_t c = 0; c < nc; ++c) {
            const size_t stride = sycl_size_add(sizes[c][l].first, sizes[c][l].second);
            plan.device_bytes[c] -= stride;
            plan.nonresident_bytes[c] = sycl_size_add(plan.nonresident_bytes[c], stride);
        }
    }
    if (!fits())
        throw std::runtime_error("SYCL split: insufficient VRAM for fixed buffers, reserve and expert window floors");
    return plan;
}

}  // namespace fnx
