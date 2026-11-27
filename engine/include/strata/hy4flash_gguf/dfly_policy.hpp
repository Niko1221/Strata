#pragma once
// Choose how many proposed DFly tokens the target verifies in the current round.
#include <array>
#include <string>

namespace strata::hy4_flash {

constexpr int kDflyPolicySlots = 8;   ///< the drafter's block size; the loop's own ceiling is lower today

/// `--draft-window`: how many of the drafted rows a round verifies.
///  - fixed: all of them, every round (the default, and the path before this switch existed);
///  - confidence: the prefix the drafter is sure of (DflyConfidenceGate).
enum class DflyWindow { fixed, confidence };
const char* dfly_window_name(DflyWindow w);
bool dfly_window_parse(const std::string& s, DflyWindow& out);

/// How `DflyWindowCosts` prices a width.
struct DflyCostConfig {
    /// Cost of a round verifying kv drafts relative to kv = 0, used until kv has been measured: the H20 W1..W4
    /// verify sweep at depth 128 (15.7, 21.4, 25.0, 29.5 ms), extended linearly past kv = 3.
    std::array<double, kDflyPolicySlots + 1> cost_ratio{1.0, 1.36, 1.59, 1.88, 2.17, 2.46, 2.75, 3.04, 3.33};
    /// Weight faster and slower timing samples separately to limit the effect of transient stalls.
    double cost_alpha = 0.2, cost_alpha_up = 0.05;
    /// A sample is clamped to [lo, hi] x the current estimate, so one slow round moves the estimate by at most
    /// cost_alpha_up * (hi - 1).
    double cost_clamp_lo = 0.67, cost_clamp_hi = 1.5;
};

/// The measured cost of a round by width, kept across requests (it describes the card, not the text).
struct DflyWindowCosts {
    std::array<double, kDflyPolicySlots + 1> ema{};   ///< ms; 0 until the width has been measured
    std::array<long, kDflyPolicySlots + 1> n{};
    double scale = 0.0;                                 ///< ms per unit of cost_ratio, over every sample
    long n_scale = 0;

    void observe(const DflyCostConfig& c, int kv, double ms);
    /// In ms once anything has been measured; before that in units of cost_ratio (only the ratios matter).
    double estimate(const DflyCostConfig& c, int kv) const;
};

struct DflyGateConfig {
    /// Per-slot Platt calibration on logit(confidence).
    std::array<double, kDflyPolicySlots> a{0.896, 0.857, 0.776, 0.776, 0.776, 0.776, 0.776, 0.776};
    std::array<double, kDflyPolicySlots> b{-0.655, -0.553, -0.224, -0.224, -0.224, -0.224, -0.224, -0.224};
    double eta = 0.05;          ///< learning rate of the per-request bias; 0: the offline fit only
    double bias_limit = 2.0;    ///< |bias| stays below this (in logits)
    double rate_decay = 0.9;    ///< lambda: decayed tokens over decayed ms, per round
    DflyCostConfig costs;
};

/// `STRATA_HY4_FLASH_DFLY_GATE="eta=0,a=1.2/1.1/1.0,b=-0.3/0/0.2"`: overrides for experiments (a and b take one value
/// per slot, '/' between; the last one repeats; alpha / alpha_up set the cost EMA's weights). False (and `err`
/// says why) on an unknown key or a value out of range; `c` is then unchanged.
bool dfly_gate_config_parse(const std::string& spec, DflyGateConfig& c, std::string& err);

class DflyConfidenceGate {
public:
    /// `verify_costs` outlives the gate and is shared by the process's requests.
    DflyConfidenceGate(const DflyGateConfig& c, DflyWindowCosts* verify_costs);

    /// How many of the k drafts to verify, 0..k, from their confidences. The request's first round verifies
    /// all k: it sets lambda.
    int choose(const float* conf, int k);
    /// After the rewind: kv drafts verified, m outputs, r matched; `rest_ms` the round minus the proposal,
    /// `round_ms` the whole round. `conf` the same values `choose` saw.
    void observe(const float* conf, int kv, int m, int r, double rest_ms, double round_ms);

    double p(int slot, float conf) const;   ///< calibrated, with this request's bias
    double rate() const;                    ///< lambda in tokens per ms; 0 before the first round
    double bias() const { return bias_; }

private:
    DflyGateConfig c_;
    DflyWindowCosts* costs_;
    double bias_ = 0.0, tok_ = 0.0, ms_ = 0.0;
};

}  // namespace strata::hy4_flash
