// include/strata/spec/nvfp4_ngram_drafter.hpp - DSpark confidence-scheduled suffix drafter with hardware-aware prefix scheduling.
#pragma once

#include "strata/artifact/qwen_tokenizer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::spec {

class Nvfp4NgramDrafter {
public:
    explicit Nvfp4NgramDrafter(const strata::artifact::QwenTokenizer& tokenizer);

    void clear_rejections();
    void record_rejection(int32_t prev2_t, int32_t prev_t, int32_t bad_cand);

    // Online hardware-aware latency & acceptance feedback (DSpark Algorithm 1 calibration).
    // drafted == 0 records baseline single-token step_token latency.
    // drafted >= 1 records verification batch M = 1 + drafted latency and accepted prefix count.
    void observe_step(int32_t drafted, int32_t accepted, double step_ms);
    void observe_target_confidence(float prob, float margin);

    void draft(int32_t cur_tok, size_t prompt_len, const std::vector<int32_t>& history,
               bool enabled, std::vector<int32_t>& draft_out);

private:
    struct StepStats {
        int32_t cand = -1;
        int32_t agree = 0;
        int32_t total = 0;
        int32_t runner_up = 0;
        int32_t match_order = 0;
        int32_t bucket = 0;
        double confidence = 0.0;
    };

    static uint64_t make_rej_key(int32_t p2, int32_t p1, int32_t cand);
    bool is_rejected(int32_t p2, int32_t p1, int32_t cand) const;
    const std::string& get_tok_str(int32_t id);

    bool is_clause_break(int32_t cand_id, int32_t next_id);
    bool is_subword_cont(int32_t prev_id, int32_t cand_id, int32_t next_id);
    bool is_digit_tok(int32_t id);
    bool is_entity_or_num(int32_t id);
    bool is_new_number_head(int32_t prev2_id, int32_t prev_id, int32_t cand_id);
    bool is_content_token(int32_t id);

    double estimate_verify_ms(int32_t drafted) const;
    double estimate_step_confidence(const StepStats& st, int32_t prev2_id, int32_t prev_id, int32_t next_id);
    int32_t schedule_prefix(const std::vector<StepStats>& steps) const;

    const strata::artifact::QwenTokenizer& tokenizer_;
    std::unordered_map<uint64_t, uint8_t> rejected_set_;
    std::unordered_map<int32_t, std::string> tok_str_cache_;

    // DSpark hardware cost model (EMA wall-clock ms for drafted = 0, 1, 2, ..., dynamically extended)
    // On R9700 + 7950X3D with zero-copy GDN ping-pong + hybrid PCIe MoE verification:
    std::vector<double> ema_step_ms_{24.2, 29.5, 31.0, 32.5};
    double ema_rollback_ms_ = 2.2;
    float last_target_prob_ = 1.0f;
    float last_target_margin_ = 10.0f;

    // Online empirical survival rate per Markov order bucket:
    // bucket 0: L=1, bucket 1: L=2, bucket 2: L=3, bucket 3: L=4, bucket 4: L>=5
    std::array<double, 5> bucket_emp_survival_{{0.58, 0.72, 0.84, 0.91, 0.96}};
    std::vector<int32_t> last_draft_buckets_;
    int32_t last_draft_len_ = 0;
};

}  // namespace strata::spec
