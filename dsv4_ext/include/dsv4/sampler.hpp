// dsv4/sampler.hpp - token selection from the logits: greedy, temperature, top-k, top-p, min_p, penalties.
// Same key spellings as Strata's engine protocol (temperature= top_p= top_k= min_p= penalty_repeat=
// penalty_freq= penalty_present= penalty_last_n= seed=), so serve/server.py's sampling_keys() works unchanged.
#pragma once

#include <cstdint>
#include <vector>

namespace dsv4 {

struct SampleOpts {
    float temperature = 0.0f;      // 0 = greedy argmax (the engine's default, as Strata's)
    int top_k = 0;                 // 0 = off; the sampled path keeps at most this many candidates
    float top_p = 1.0f;            // 1 = off; nucleus
    float min_p = 0.0f;            // 0 = off; relative to the best candidate's probability
    float penalty_repeat = 1.0f;   // 1 = off (divides/multiplies the logit, as llama.cpp)
    float penalty_freq = 0.0f;     // subtracted once per occurrence in the window
    float penalty_present = 0.0f;  // subtracted once when the token appears at all
    int penalty_last_n = 64;       // window the penalties count over (0 = no penalties)
    uint64_t seed = 0;             // 0 = seed from the clock
};

/// One request's sampler: keeps the RNG state across tokens of the same request.
class Sampler {
public:
    explicit Sampler(const SampleOpts& opts) : o_(opts) { init_rng(); }

    /// Picks one id from `logits` (n entries). `history` is every id seen so far (prompt + generated);
    /// only its last penalty_last_n entries matter. Always returns a valid id in [0, n).
    int sample(const float* logits, int n, const std::vector<int>& history);

    const SampleOpts& opts() const { return o_; }

private:
    void init_rng();
    float uniform();

    SampleOpts o_;
    uint64_t st_ = 0;
};

}  // namespace dsv4
