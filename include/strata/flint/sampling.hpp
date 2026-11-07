// include/strata/flint/sampling.hpp - Flint 40B sampler settings.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace strata::flint {

struct Sampling {
    float temperature = 0.f, top_p = 0.95f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

inline void apply_penalties(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            if (id < 0 || (size_t) id >= lg.size()) continue;
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
}

}  // namespace strata::flint
