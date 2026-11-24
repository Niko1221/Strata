// src/hy4flash_gguf/sampler.cpp - see include/strata/hy4flash_gguf/sampler.hpp.
#include "strata/hy4flash_gguf/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

namespace strata::hy4_flash {
namespace {

const char* arg(int argc, char** argv, const char* flag) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}

bool has(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

/// splitmix64. A named, fixed generator rather than `std::mt19937_64`, whose stream is not promised to be
/// the same across standard libraries - and a seed is only worth reporting if it reproduces elsewhere.
std::uint64_t next_u64(std::uint64_t& s) {
    s += 0x9e3779b97f4a7c15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

float next_unit(std::uint64_t& s) {
    // 53 bits into [0, 1), the usual construction: exactly representable and never 1.0.
    return (float) ((double) (next_u64(s) >> 11) * (1.0 / 9007199254740992.0));
}

}  // namespace

bool sampler_config_from_args(int argc, char** argv, SamplerConfig& out, std::string& err) {
    out = SamplerConfig{};
    if (has(argc, argv, "--greedy")) return true;   // every parameter stays off
    if (const char* v = arg(argc, argv, "--temp")) out.temperature = (float) std::atof(v);
    if (const char* v = arg(argc, argv, "--top-k")) out.top_k = std::atoi(v);
    if (const char* v = arg(argc, argv, "--top-p")) out.top_p = (float) std::atof(v);
    if (const char* v = arg(argc, argv, "--repeat-penalty")) out.repeat_penalty = (float) std::atof(v);
    if (const char* v = arg(argc, argv, "--repeat-last")) out.repeat_last = std::atoi(v);
    if (const char* v = arg(argc, argv, "--seed")) out.seed = std::strtoull(v, nullptr, 10);
    if (out.temperature < 0.0f) { err = "--temp cannot be negative"; return false; }
    if (out.top_k < 0) { err = "--top-k cannot be negative"; return false; }
    if (out.top_p < 0.0f || out.top_p > 1.0f) { err = "--top-p must be in [0, 1]"; return false; }
    if (out.repeat_last < 0) { err = "--repeat-last cannot be negative"; return false; }
    return true;
}

std::string sampler_describe(const SamplerConfig& c) {
    if (c.greedy()) return "greedy (the argmax; bit for bit what this engine produced before a sampler)";
    char b[512];
    std::snprintf(b, sizeof b, "temp %.3g, top-k %d, top-p %.3g, repeat %.3g over %d",
                  c.temperature <= 0.0f ? 1.0 : (double) c.temperature, c.top_k, (double) c.top_p,
                  (double) c.repeat_penalty, c.repeat_last);
    return b;
}

Sampler::Sampler(const SamplerConfig& c) : cfg_(c) {
    seed_ = c.seed;
    if (seed_ == 0)
        seed_ = (std::uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() * 0x2545f491ull;
    state_ = seed_;
}

std::int64_t Sampler::pick(const float* logits, std::int64_t n_vocab,
                           const std::vector<std::int32_t>& history) {
    if (logits == nullptr || n_vocab <= 0) return -1;

    // Check for nonfinite logits while finding the greedy token.
    std::int64_t best = cfg_.excluded_token == 0 ? 1 : 0;
    if (best >= n_vocab) return -1;
    for (std::int64_t v = 0; v < n_vocab; ++v) {
        if (!std::isfinite(logits[v])) return -1;
        if (v != cfg_.excluded_token && logits[v] > logits[best]) best = v;
    }
    if (cfg_.greedy() && cfg_.repeat_penalty == 1.f) return best;

    cand_.clear();
    cand_.reserve((std::size_t) n_vocab);
    for (std::int64_t v = 0; v < n_vocab; ++v)
        if (v != cfg_.excluded_token) cand_.emplace_back(logits[v], (std::int32_t) v);

    // ---- the repetition penalty, over the last `repeat_last` positions
    if (cfg_.repeat_penalty != 1.0f && cfg_.repeat_last > 0 && !history.empty()) {
        const std::size_t from =
            history.size() > (std::size_t) cfg_.repeat_last ? history.size() - (std::size_t) cfg_.repeat_last : 0;
        std::unordered_set<std::int32_t> seen(history.begin() + (long) from, history.end());
        for (auto& c : cand_)
            if (seen.count(c.second) != 0) {
                // llama.cpp's rule: DIVIDE a positive logit and MULTIPLY a negative one. Dividing a
                // negative logit would make it larger, i.e. reward the repetition.
                c.first = c.first > 0.0f ? c.first / cfg_.repeat_penalty : c.first * cfg_.repeat_penalty;
            }
    }

    if (cfg_.greedy()) {
        return std::max_element(cand_.begin(), cand_.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        })->second;
    }

    // ---- temperature
    const float temp = cfg_.temperature <= 0.0f ? 1.0f : cfg_.temperature;
    if (temp != 1.0f)
        for (auto& c : cand_) c.first /= temp;

    // ---- top-k: a partial sort is enough, and it keeps the descending order the next two steps want
    const std::size_t k = cfg_.top_k > 0 ? std::min<std::size_t>((std::size_t) cfg_.top_k, cand_.size())
                                         : cand_.size();
    const auto order = [](const std::pair<float, std::int32_t>& a, const std::pair<float, std::int32_t>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    // A full partial_sort builds/drains a heap for the entire vocabulary. sort
    // produces exactly the same total order (including ties), without changing
    // top-k/top-p, softmax accumulation order or the random stream.
    if (k == cand_.size()) std::sort(cand_.begin(), cand_.end(), order);
    else std::partial_sort(cand_.begin(), cand_.begin() + (long) k, cand_.end(), order);
    cand_.resize(k);

    // ---- softmax over what is left, then top-p over the cumulative mass
    float mx = cand_[0].first;
    double sum = 0.0;
    for (auto& c : cand_) {
        c.first = std::exp(c.first - mx);
        sum += (double) c.first;
    }
    if (!(sum > 0.0)) return best;   // every candidate underflowed: the argmax is the only defensible pick
    for (auto& c : cand_) c.first = (float) ((double) c.first / sum);

    if (cfg_.top_p < 1.0f) {
        double acc = 0.0;
        std::size_t keep = 0;
        for (; keep < cand_.size(); ++keep) {
            acc += (double) cand_[keep].first;
            if (acc >= (double) cfg_.top_p) { ++keep; break; }
        }
        if (keep == 0) keep = 1;   // top_p of 0 still has to choose something
        cand_.resize(keep);
        double renorm = 0.0;
        for (const auto& c : cand_) renorm += (double) c.first;
        if (!(renorm > 0.0)) return best;
        for (auto& c : cand_) c.first = (float) ((double) c.first / renorm);
    }

    // ---- the draw. The cumulative walk is in the descending order above, so the same seed and the same
    // candidate set always give the same id whatever the vocabulary's own order is.
    const float u = next_unit(state_);
    double acc = 0.0;
    for (const auto& c : cand_) {
        acc += (double) c.first;
        if ((double) u < acc) return c.second;
    }
    return cand_.back().second;   // rounding left `u` past the end
}

}  // namespace strata::hy4_flash
