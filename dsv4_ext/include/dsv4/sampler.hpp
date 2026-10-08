/**
 * @file sampler.hpp
 * @brief Token selection from logits: greedy, temperature, top-k, top-p, min_p, and repetition penalties.
 *
 * Implements a full sampling pipeline matching llama.cpp's `sample.cpp` semantics:
 *   1. Penalties applied first (on raw logits)
 *   2. Temperature scaling
 *   3. Top-k filtering
 *   4. Nucleus (top-p) and min_p pruning
 *   5. Roulette selection over remaining candidates
 *
 * With temperature == 0, falls back to greedy argmax (the engine's default, matching Strata).
 * Key names use Strata's spelling (`temperature=`, `top_p=`, etc.) so `serve/server.py`'s
 * `sampling_keys()` works unchanged.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace dsv4 {

/**
 * @struct SampleOpts
 * @brief Sampling configuration for token selection.
 *
 * All parameters match Strata's engine protocol key names for seamless integration with the serve loop.
 */
struct SampleOpts {
    float temperature = 0.0f;       ///< Sampling temperature: 0 = greedy argmax (engine default).
    int top_k = 0;                  ///< Keep top-k candidates: 0 = off (keep all).
    float top_p = 1.0f;             ///< Nucleus sampling threshold: 1 = off (keep all).
    float min_p = 0.0f;             ///< Min-p threshold relative to best candidate's probability: 0 = off.
    float penalty_repeat = 1.0f;    ///< Repetition penalty divisor/multiplier: 1 = off (like llama.cpp).
    float penalty_freq = 0.0f;      ///< Frequency penalty: subtracted once per occurrence in the window.
    float penalty_present = 0.0f;   ///< Presence penalty: subtracted once when token appears at all.
    int penalty_last_n = 64;        ///< Penalty window size: 0 = no penalties, >0 = last N tokens.
    uint64_t seed = 0;              ///< RNG seed: 0 = auto-seed from clock + address entropy.
};

/**
 * @class Sampler
 * @brief Per-request token sampler with persistent RNG state across generated tokens.
 *
 * Each inference request gets its own Sampler instance, which maintains xorshift64* RNG state
 * between calls to `sample()`. This ensures reproducible sampling from a given seed — critical
 * for debugging and benchmarking.
 *
 * @par Sampling pipeline
 *   1. Apply penalties (repeat/freq/presence) on raw logits within the last N tokens
 *   2. If temperature == 0: return argmax (greedy)
 *   3. Keep top-k candidates (if k > 0)
 *   4. Softmax over scaled logits to get probabilities
 *   5. Prune by top-p cumulative probability and min_p absolute threshold
 *   6. Roulette selection from remaining candidates using xorshift64* RNG
 *
 * @par Random number generation
 *   Uses xorshift64* (12 ns per draw, reproducible from seed). A warm-up loop of 8 iterations
 *   ensures that a seed of 1 does not start near 0. The uniform() method extracts 53 bits of
 *   mantissa precision for [0, 1) floating-point values.
 */
class Sampler {
public:
    /**
     * @brief Construct a sampler with the given sampling options.
     *
     * Initializes RNG state from `opts.seed` (or auto-seeds if seed == 0).
     * The warm-up loop ensures good initial state distribution.
     *
     * @param opts  Sampling configuration (temperature, top-k, penalties, etc.).
     */
    explicit Sampler(const SampleOpts& opts) : o_(opts) { init_rng(); }

    /**
     * @brief Select one token id from the logit distribution.
     *
     * Applies the full sampling pipeline: penalties → temperature → top-k → top-p/min_p → roulette.
     * Always returns a valid id in [0, n), even on edge cases (NaN logits, empty candidates).
     *
     * @param logits    Vocabulary-sized logit vector [n].
     * @param n         Vocabulary size (number of entries in logits).
     * @param history   All token ids seen so far (prompt + generated); only last penalty_last_n matter.
     * @return Selected token id in range [0, n).
     */
    int sample(const float* logits, int n, const std::vector<int>& history);

    /**
     * @brief Get the current sampling options.
     * @return Const reference to SampleOpts used at construction.
     */
    const SampleOpts& opts() const { return o_; }

private:
    /** Initialize RNG state from seed or system clock + address entropy. */
    void init_rng();

    /** Generate one uniform random float in [0, 1) with 53-bit mantissa precision. */
    float uniform();

    SampleOpts o_;              ///< Sampling configuration (immutable after construction).
    uint64_t st_ = 0;           ///< xorshift64* RNG state (persistent across sample() calls).
};

}  // namespace dsv4
