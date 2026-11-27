// src/hy4flash_gguf/dfly_policy.cpp - the confidence-gated Dfly window. See include/strata/hy4flash_gguf/dfly_policy.hpp.
#include "strata/hy4flash_gguf/dfly_policy.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace strata::hy4_flash {

namespace {

bool parse_number(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (end != s.c_str() + s.size() || errno != 0 || !(v == v)) return false;
    out = v;
    return true;
}

}  // namespace

const char* dfly_window_name(DflyWindow w) {
    switch (w) {
    case DflyWindow::fixed: return "fixed";
    case DflyWindow::confidence: return "confidence";
    }
    return "?";
}

bool dfly_window_parse(const std::string& s, DflyWindow& out) {
    for (DflyWindow w : {DflyWindow::fixed, DflyWindow::confidence})
        if (s == dfly_window_name(w)) { out = w; return true; }
    return false;
}

void DflyWindowCosts::observe(const DflyCostConfig& c, int kv, double ms) {
    if (kv < 0 || kv > kDflyPolicySlots || !(ms > 0.0)) return;
    const double ratio = c.cost_ratio[(std::size_t) kv];
    if (n_scale == 0) {
        scale = ms / ratio;   // the first sample sets the units; every estimate scales with it alike
    } else {
        const double s = std::clamp(ms / ratio, c.cost_clamp_lo * scale, c.cost_clamp_hi * scale);
        scale += (s > scale ? c.cost_alpha_up : c.cost_alpha) * (s - scale);
    }
    ++n_scale;
    double& e = ema[(std::size_t) kv];
    if (n[(std::size_t) kv] == 0) e = ratio * scale;   // start from the prior, then follow the samples
    const double v = std::clamp(ms, c.cost_clamp_lo * e, c.cost_clamp_hi * e);
    e += (v > e ? c.cost_alpha_up : c.cost_alpha) * (v - e);
    ++n[(std::size_t) kv];
}

double DflyWindowCosts::estimate(const DflyCostConfig& c, int kv) const {
    if (n[(std::size_t) kv] > 0) return ema[(std::size_t) kv];
    const double ratio = c.cost_ratio[(std::size_t) kv];
    return n_scale > 0 ? ratio * scale : ratio;
}

bool dfly_gate_config_parse(const std::string& spec, DflyGateConfig& c, std::string& err) {
    DflyGateConfig t = c;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const std::size_t eq = item.find('=');
        if (eq == std::string::npos) { err = "'" + item + "' is not key=value"; return false; }
        const std::string key = item.substr(0, eq), val = item.substr(eq + 1);
        if (key == "a" || key == "b") {
            std::array<double, kDflyPolicySlots>& dst = key == "a" ? t.a : t.b;
            std::stringstream vs(val);
            std::string one;
            int i = 0;
            while (std::getline(vs, one, '/')) {
                double v = 0.0;
                if (i >= kDflyPolicySlots || !parse_number(one, v) || std::fabs(v) > 100.0) {
                    err = "'" + item + "': up to " + std::to_string(kDflyPolicySlots) + " numbers, '/' between";
                    return false;
                }
                dst[(std::size_t) i++] = v;
            }
            if (i == 0) { err = "'" + item + "': no value"; return false; }
            for (; i < kDflyPolicySlots; ++i) dst[(std::size_t) i] = dst[(std::size_t) i - 1];   // last one repeats
            continue;
        }
        double v = 0.0;
        if (!parse_number(val, v)) { err = "'" + item + "': not a number"; return false; }
        if (key == "eta") {
            if (v < 0.0 || v > 1.0) { err = "eta must be in [0, 1]"; return false; }
            t.eta = v;
        } else if (key == "bias_limit") {
            if (v < 0.0 || v > 20.0) { err = "bias_limit must be in [0, 20]"; return false; }
            t.bias_limit = v;
        } else if (key == "rate_decay") {
            if (v <= 0.0 || v > 1.0) { err = "rate_decay must be in (0, 1]"; return false; }
            t.rate_decay = v;
        } else if (key == "alpha" || key == "alpha_up") {
            if (v <= 0.0 || v > 1.0) { err = key + " must be in (0, 1]"; return false; }
            (key == "alpha" ? t.costs.cost_alpha : t.costs.cost_alpha_up) = v;
        } else {
            err = "unknown key '" + key + "' (a, b, eta, bias_limit, rate_decay, alpha, alpha_up)";
            return false;
        }
    }
    c = t;
    return true;
}

DflyConfidenceGate::DflyConfidenceGate(const DflyGateConfig& c, DflyWindowCosts* verify_costs)
    : c_(c), costs_(verify_costs) {}

double DflyConfidenceGate::p(int slot, float conf) const {
    const double x = std::clamp((double) conf, 1e-6, 1.0 - 1e-6);
    const double z = c_.a[(std::size_t) slot] * std::log(x / (1.0 - x)) + c_.b[(std::size_t) slot] + bias_;
    return 1.0 / (1.0 + std::exp(-z));
}

double DflyConfidenceGate::rate() const { return ms_ > 0.0 ? tok_ / ms_ : 0.0; }

int DflyConfidenceGate::choose(const float* conf, int k) {
    k = std::clamp(k, 0, kDflyPolicySlots);
    const double lambda = rate();
    if (k == 0 || !(lambda > 0.0)) return k;   // the request's first round verifies everything: it sets lambda
    const auto cost = [&](int kv) {
        return costs_ != nullptr ? costs_->estimate(c_.costs, kv) : c_.costs.cost_ratio[(std::size_t) kv];
    };
    int best = 0;
    double tau = 1.0, s = 1.0;
    double best_value = tau - lambda * cost(0);
    for (int j = 1; j <= k; ++j) {
        s *= p(j - 1, conf[j - 1]);
        tau += s;
        const double v = tau - lambda * cost(j);
        if (v > best_value) { best = j; best_value = v; }
    }
    return best;
}

void DflyConfidenceGate::observe(const float* conf, int kv, int m, int r, double rest_ms, double round_ms) {
    const int compared = std::min(kv, m);   // slot i was compared iff row i was picked
    for (int i = 0; i < compared; ++i) {
        const double y = i < r ? 1.0 : 0.0;
        bias_ = std::clamp(bias_ + c_.eta * (y - p(i, conf[i])), -c_.bias_limit, c_.bias_limit);
    }
    tok_ = c_.rate_decay * tok_ + (double) m;
    ms_ = c_.rate_decay * ms_ + round_ms;
    if (costs_ != nullptr) costs_->observe(c_.costs, kv, rest_ms);
}

}  // namespace strata::hy4_flash
