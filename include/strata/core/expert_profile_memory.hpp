#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// Opt-in long-term routing memory for --expert-profile-save. This is a gated running statistic, not a neural network
/// or a change to the model. The forget gate is time-based, the input gate admits only new heat since the previous
/// consolidation, and the output mixes that long-term cell with the recent distribution. All times are Unix seconds.
class ExpertProfileMemory {
public:
    bool load(const std::string& path, int64_t n_layers, int64_t n_expert, std::string& err);
    bool save(const std::string& path, std::string& err) const;

    void reset(int64_t n_layers, int64_t n_expert, double now);
    bool consolidate(const std::vector<double>& heat, double now, double half_life_hours, double long_term_weight,
                     std::string& err);
    const std::vector<double>& scores() const { return scores_; }
    const std::vector<double>& cell() const { return cell_; }
    double updated_at() const { return updated_at_; }

private:
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    double updated_at_ = 0.0;
    std::vector<double> cell_;
    std::vector<double> committed_heat_;
    std::vector<double> scores_;
};

}  // namespace strata::core
