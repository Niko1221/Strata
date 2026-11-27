// Host sampler for a single row of logits. Apply repetition penalty, temperature,
// top-k and top-p before drawing a token. Greedy mode returns the argmax.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash {

struct SamplerConfig {
    /// <= 0: greedy. Above 0, each logit is divided by it before the draw;
    /// a small value sharpens, a large one flattens.
    float temperature = 0.0f;
    /// 0: every token is a candidate. Otherwise only the `top_k` highest logits are.
    int top_k = 0;
    /// >= 1: off. Otherwise the smallest set of candidates whose probabilities sum to at least this,
    /// taken in descending order - so a confident step keeps one candidate and an uncertain one keeps many.
    float top_p = 1.0f;
    /// 1.0: off. A token that appeared in the last `repeat_last` positions has its logit divided by this
    /// when positive and MULTIPLIED when negative, which is llama.cpp's rule and not the same as dividing
    /// a negative logit.
    float repeat_penalty = 1.0f;
    int repeat_last = 64;
    /// The draw's seed. The same seed, prompt and parameters give the same tokens; 0 asks for a random one,
    /// which is reported so a run can be repeated.
    std::uint64_t seed = 0;
    /// Raw HTTP ignore_eos suppresses the model EOS in the target distribution.
    std::int32_t excluded_token = -1;

    /// True when every parameter is off, i.e. this is the argmax.
    bool greedy() const {
        return temperature <= 0.0f;
    }
};

/// `--temp` / `--top-k` / `--top-p` / `--repeat-penalty` / `--repeat-last` / `--seed`, and
/// `--greedy` which resets all of them. False with `err` set for a value that cannot be meant
/// (a negative temperature, a top_p outside [0, 1]).
bool sampler_config_from_args(int argc, char** argv, SamplerConfig& out, std::string& err);

/// One line describing what the sampler will do, for the engine's startup block.
std::string sampler_describe(const SamplerConfig& c);

class Sampler {
public:
    explicit Sampler(const SamplerConfig& c);
    /// `logits` is one row of `n_vocab`. `history` is every token so far, by absolute position, for the
    /// repetition penalty. Returns the chosen id, or -1 when the row has a non-finite value - which the
    /// caller must report rather than paper over, because it means the pass produced nonsense.
    std::int64_t pick(const float* logits, std::int64_t n_vocab, const std::vector<std::int32_t>& history);
    std::uint64_t seed() const { return seed_; }

private:
    SamplerConfig cfg_;
    std::uint64_t seed_ = 0;
    std::uint64_t state_ = 0;
    std::vector<std::pair<float, std::int32_t>> cand_;   ///< reused, so a step allocates nothing
};

}  // namespace strata::hy4_flash
