// src/spec/nvfp4_ngram_drafter.cpp - DSpark confidence-scheduled suffix drafter with hardware-aware prefix scheduling.
//
// Adapts the Calibrated Confidence Estimator (Section 3.3.1, Eq. 11-13) and
// Hardware-Aware Prefix Length Scheduler (Section 3.3.2, Algorithm 1) from:
//   - DeepSeek-AI, "DSpark: Confidence-Scheduled Speculative Decoding with
//     Semi-Autoregressive Generation" (arXiv:2607.05147, CC BY 4.0)
//   - deepseek-ai/DeepSpec (MIT License, Copyright (c) 2026 DeepSeek)
#include "strata/spec/nvfp4_ngram_drafter.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace strata::spec {

Nvfp4NgramDrafter::Nvfp4NgramDrafter(const strata::artifact::QwenTokenizer& tokenizer)
    : tokenizer_(tokenizer) {}

void Nvfp4NgramDrafter::clear_rejections() {
    rejected_set_.clear();
    last_draft_len_ = 0;
}

uint64_t Nvfp4NgramDrafter::make_rej_key(int32_t p2, int32_t p1, int32_t cand) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(p2) & 0x1FFFFFu) << 42) |
           (static_cast<uint64_t>(static_cast<uint32_t>(p1) & 0x1FFFFFu) << 21) |
           static_cast<uint64_t>(static_cast<uint32_t>(cand) & 0x1FFFFFu);
}

void Nvfp4NgramDrafter::record_rejection(int32_t prev2_t, int32_t prev_t, int32_t bad_cand) {
    rejected_set_[make_rej_key(prev2_t, prev_t, bad_cand)] = 1;
    rejected_set_[make_rej_key(-1, prev_t, bad_cand)] = 1;
}

bool Nvfp4NgramDrafter::is_rejected(int32_t p2, int32_t p1, int32_t cand) const {
    if (rejected_set_.find(make_rej_key(p2, p1, cand)) != rejected_set_.end()) return true;
    if (rejected_set_.find(make_rej_key(-1, p1, cand)) != rejected_set_.end()) return true;
    return false;
}

const std::string& Nvfp4NgramDrafter::get_tok_str(int32_t id) {
    auto it = tok_str_cache_.find(id);
    if (it != tok_str_cache_.end()) return it->second;
    return tok_str_cache_.emplace(id, tokenizer_.decode_token(id)).first->second;
}

bool Nvfp4NgramDrafter::is_clause_break(int32_t cand_id, int32_t next_id) {
    if (cand_id < 0 || cand_id >= 248040) return true;
    const std::string& s = get_tok_str(cand_id);
    if (s.empty()) return true;
    char c0 = s[0];
    if (c0 == '\n' || c0 == '\r' || c0 == ',' || c0 == ';' || c0 == ':' ||
        c0 == ')' || c0 == ']' || c0 == '}' || c0 == '(' || c0 == '[' || c0 == '"') {
        return true;
    }
    if (c0 == ' ' && s.size() >= 2) {
        char c1 = s[1];
        if (c1 == '(' || c1 == '[' || c1 == '"' || c1 == '*') return true;
    }
    if (c0 == '.') {
        if (s.size() == 1 && next_id >= 0 && next_id < 248040) {
            const std::string& ns = get_tok_str(next_id);
            if (!ns.empty() && std::isdigit(static_cast<unsigned char>(ns[0]))) return false;
        }
        return true;
    }
    return false;
}

bool Nvfp4NgramDrafter::is_subword_cont(int32_t prev_id, int32_t cand_id, int32_t next_id) {
    if (prev_id < 0 || cand_id < 0 || cand_id >= 248040) return false;
    const std::string& ps = get_tok_str(prev_id);
    const std::string& cs = get_tok_str(cand_id);
    if (ps.empty() || cs.empty()) return false;
    char pb = ps.back();
    if (!std::isalnum(static_cast<unsigned char>(pb)) && pb != '-') return false;
    char c0 = cs[0];
    if (std::isalnum(static_cast<unsigned char>(c0))) return true;
    if ((c0 == '-' || c0 == '/') && cs.size() >= 2 && std::isalnum(static_cast<unsigned char>(cs[1]))) return true;
    if ((cs == "-" || cs == ".") && next_id >= 0 && next_id < 248040) {
        const std::string& ns = get_tok_str(next_id);
        if (!ns.empty() && std::isalnum(static_cast<unsigned char>(ns[0]))) return true;
    }
    return false;
}

bool Nvfp4NgramDrafter::is_digit_tok(int32_t id) {
    if (id < 0 || id >= 248040) return false;
    const std::string& s = get_tok_str(id);
    size_t pos = (!s.empty() && s[0] == ' ') ? 1 : 0;
    return pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]));
}

bool Nvfp4NgramDrafter::is_entity_or_num(int32_t id) {
    if (id < 0 || id >= 248040) return false;
    const std::string& s = get_tok_str(id);
    if (s.empty()) return false;
    size_t pos = (s[0] == ' ') ? 1 : 0;
    if (s.size() - pos < 2) return false;
    unsigned char c0 = static_cast<unsigned char>(s[pos]);
    if (std::isdigit(c0)) return true;
    // Also detect mixed-case/numeric technical tokens like "gfx", "nvfp", "fp4", "x3d"
    bool has_digit_or_inner_upper = false;
    for (size_t k = pos + 1; k < s.size(); ++k) {
        unsigned char ck = static_cast<unsigned char>(s[k]);
        if (std::isdigit(ck) || std::isupper(ck)) {
            has_digit_or_inner_upper = true;
            break;
        }
    }
    if (has_digit_or_inner_upper) return true;
    if (!std::isupper(c0)) return false;
    std::string w = s.substr(pos);
    static const char* const kCapStop[] = {
        "The", "A", "An", "In", "On", "At", "To", "For", "Of", "With", "By", "From", "As",
        "Is", "Are", "Was", "Were", "Be", "This", "That", "These", "Those", "It", "Its",
        "They", "We", "You", "And", "Or", "But", "If", "When", "While", "Where", "Why",
        "How", "What", "Which", "Who", "Here", "There", "Each", "Both", "All", "Some",
        "No", "Not", "Only", "Also", "However", "Therefore", "Thus", "Summarize",
        "Explain", "Describe", "List", "Compare", "Key", "Summary", "Overview", "Note",
        "Contains", "Equipped", "Uses", "Based", "Running", "Routes", "Total", "Step",
        "Modern", "First", "Second", "Third", "Finally", "When", "Once", "After", "Before"
    };
    for (const char* sw : kCapStop) {
        if (w == sw) return false;
    }
    return true;
}

bool Nvfp4NgramDrafter::is_new_number_head(int32_t /*prev2_id*/, int32_t prev_id, int32_t cand_id) {
    if (!is_digit_tok(cand_id) || prev_id < 0) return false;
    const std::string& ps = get_tok_str(prev_id);
    if (ps.empty()) return false;
    char pb = ps.back();
    return (pb == '(' || pb == '*' || pb == '[' || pb == '\n');
}

bool Nvfp4NgramDrafter::is_content_token(int32_t id) {
    if (is_entity_or_num(id)) return true;
    if (id < 0 || id >= 248040) return false;
    const std::string& s = get_tok_str(id);
    size_t pos = (!s.empty() && s[0] == ' ') ? 1 : 0;
    if (s.size() - pos < 4) return false;
    if (!std::isalpha(static_cast<unsigned char>(s[pos]))) return false;
    std::string w = s.substr(pos);
    static const char* const kLowStop[] = {
        "that", "this", "with", "from", "have", "will", "what", "when", "where", "which",
        "while", "their", "there", "these", "those", "about", "into", "over", "after",
        "before", "under", "between", "through", "during", "without", "within", "because",
        "should", "would", "could", "other", "such", "more", "most", "some", "many",
        "much", "very", "also", "only", "even", "just", "than", "then", "them", "they",
        "were", "been", "being", "does", "make", "made", "used", "using", "uses", "each", "both"
    };
    for (const char* sw : kLowStop) {
        if (w == sw) return false;
    }
    return true;
}

void Nvfp4NgramDrafter::observe_target_confidence(float prob, float margin) {
    last_target_prob_ = std::clamp(prob, 0.0f, 1.0f);
    last_target_margin_ = std::max(0.0f, margin);
}

double Nvfp4NgramDrafter::estimate_verify_ms(int32_t drafted) const {
    const double t_base = std::max(5.0, ema_step_ms_[0]);
    if (drafted <= 0) return t_base;
    // Physical cost floor & ceiling relative to current-context base latency t_base:
    // fixed batch-verification setup (1.13x) + marginal per-token cost (0.040x)
    // + extra 16-row LM-head pass every 16 tokens (0.05x * (drafted / 16)).
    const double min_ver = t_base * (1.13 + 0.040 * static_cast<double>(drafted) +
                                     0.05 * static_cast<double>(drafted / 16));
    const double max_ver = t_base * (1.28 + 0.062 * static_cast<double>(drafted) +
                                     0.06 * static_cast<double>(drafted / 16));
    if (static_cast<size_t>(drafted) < ema_step_ms_.size() &&
        ema_step_ms_[static_cast<size_t>(drafted)] > 0.0) {
        return std::clamp(ema_step_ms_[static_cast<size_t>(drafted)], min_ver, max_ver);
    }
    return min_ver;
}

void Nvfp4NgramDrafter::observe_step(int32_t drafted, int32_t accepted, double step_ms) {
    if (step_ms >= 5.0 && step_ms <= 300.0) {
        if (drafted == 0) {
            ema_step_ms_[0] = 0.85 * ema_step_ms_[0] + 0.15 * step_ms;
            for (size_t k = 1; k < ema_step_ms_.size(); ++k) {
                const double min_ver = estimate_verify_ms(static_cast<int32_t>(k));
                if (ema_step_ms_[k] < min_ver) ema_step_ms_[k] = min_ver;
            }
        } else if (drafted >= 1) {
            while (ema_step_ms_.size() <= static_cast<size_t>(drafted)) {
                int32_t next_k = static_cast<int32_t>(ema_step_ms_.size());
                ema_step_ms_.push_back(estimate_verify_ms(next_k));
            }
            const double factor = 1.13 + 0.040 * static_cast<double>(drafted) +
                                  0.05 * static_cast<double>(drafted / 16);
            if (accepted == drafted) {
                // Full acceptance: zero GDN rollback occurred, direct measurement of T_verify(1 + drafted).
                // Also update ema_step_ms_[0] so base latency tracks context growth when MTP handles fallback steps.
                const double implied_base = std::clamp(step_ms / factor, 12.0, 60.0);
                ema_step_ms_[0] = 0.88 * ema_step_ms_[0] + 0.12 * implied_base;
                ema_step_ms_[static_cast<size_t>(drafted)] =
                    0.80 * ema_step_ms_[static_cast<size_t>(drafted)] + 0.20 * step_ms;
            } else {
                // Partial/zero acceptance: step_ms includes GDN recurrent state restore + replay
                double est_verify = std::max(ema_step_ms_[0] * 1.20, step_ms - ema_rollback_ms_);
                const double implied_base = std::clamp(est_verify / factor, 12.0, 60.0);
                ema_step_ms_[0] = 0.92 * ema_step_ms_[0] + 0.08 * implied_base;
                ema_step_ms_[static_cast<size_t>(drafted)] =
                    0.85 * ema_step_ms_[static_cast<size_t>(drafted)] + 0.15 * est_verify;
            }
        }
    }

    // DSpark causal prefix survival calibration (Eq. 11):
    // Update empirical survival probability only for positions up to the first rejected draft token.
    if (drafted >= 1 && drafted == last_draft_len_) {
        const int32_t tested = std::min<int32_t>(
            {drafted, accepted + 1, static_cast<int32_t>(last_draft_buckets_.size())});
        for (int32_t j = 0; j < tested; ++j) {
            const int32_t b = std::clamp(last_draft_buckets_[static_cast<size_t>(j)], 0, 4);
            const double survived = (j < accepted) ? 1.0 : 0.0;
            bucket_emp_survival_[static_cast<size_t>(b)] =
                0.92 * bucket_emp_survival_[static_cast<size_t>(b)] + 0.08 * survived;
        }
    }
    last_draft_len_ = 0;
}

double Nvfp4NgramDrafter::estimate_step_confidence(const StepStats& st, int32_t prev2_id,
                                                   int32_t prev_id, int32_t next_id) {
    if (st.cand < 0 || st.total <= 0 || st.agree <= 0) return 0.0;

    // DSpark distribution summary statistics (Section 3.3.1, Eq. 12):
    // top-1 empirical probability pi_1, runner-up pi_2, and margin (pi_1 - pi_2).
    const double pi_1 = static_cast<double>(st.agree) / static_cast<double>(st.total);
    const double pi_2 = static_cast<double>(st.runner_up) / static_cast<double>(st.total);
    const double margin = std::max(0.0, pi_1 - pi_2);

    // Base survival prior indexed by effective Markov suffix order L_eff
    double prior = 0.50;
    if (st.match_order >= 5) {
        prior = 0.94 + 0.02 * std::min(st.agree - 1, 2);
    } else if (st.match_order == 4) {
        prior = 0.88 + 0.04 * std::min(st.agree - 1, 2);
    } else if (st.match_order == 3) {
        prior = 0.75 + 0.07 * std::min(st.agree - 1, 2);
    } else if (st.match_order == 2) {
        prior = 0.64 + 0.10 * std::min(st.agree - 1, 2);
    } else {
        prior = 0.56 + 0.12 * std::min(st.agree - 1, 2);
    }

    // Blend prior with online calibrated bucket survival rate
    const int32_t b = std::clamp(st.bucket, 0, 4);
    double conf = 0.75 * prior + 0.25 * bucket_emp_survival_[static_cast<size_t>(b)];

    // Modulate by empirical consensus and top-2 margin
    conf *= pi_1 * (0.55 + 0.45 * margin);

    // Modulate by target model's softmax top-1 probability & top-2 logit margin (DSpark Eq. 12):
    // High target certainty on cur_tok reinforces continuation; high target entropy penalizes weak suffixes.
    if (st.match_order < 4) {
        const double target_mod = std::clamp(0.84 + 0.16 * static_cast<double>(last_target_prob_) +
                                             0.015 * std::min<double>(4.0, last_target_margin_),
                                             0.78, 1.06);
        conf *= target_mod;
    }

    // Lexical / structural features
    if (is_subword_cont(prev_id, st.cand, next_id)) {
        const std::string& cs = get_tok_str(st.cand);
        if (!cs.empty() && std::isalnum(static_cast<unsigned char>(cs[0]))) {
            conf += 0.10;
        } else {
            conf += 0.05;
        }
    } else if (is_entity_or_num(prev_id) && is_entity_or_num(st.cand)) {
        conf += 0.08;
    } else if (prev2_id >= 0 && is_content_token(prev_id) && is_content_token(st.cand)) {
        conf += 0.05;
    }

    if (is_clause_break(st.cand, next_id) && st.match_order < 5) {
        conf -= 0.20;
    }
    return std::clamp(conf, 0.0, 0.99);
}

int32_t Nvfp4NgramDrafter::schedule_prefix(const std::vector<StepStats>& steps) const {
    const int32_t num_cand = static_cast<int32_t>(steps.size());
    if (num_cand <= 0) return 0;

    // DSpark Algorithm 1: Hardware-Aware Prefix Length Scheduler
    // Baseline throughput without speculation (l = 0):
    //   Theta(0) = 1.0 / T_verify(1)
    // Speculative throughput for prefix length l in {1, ..., num_cand}:
    //   a_l      = prod_{i=1}^l c_i
    //   A(l)     = 1.0 + sum_{j=1}^l a_j
    //   C(l)     = T_verify(1 + l) + (1.0 - a_l) * T_rollback
    //   Theta(l) = A(l) / C(l)
    const double t_base = std::max(5.0, ema_step_ms_[0]);
    const double theta_0 = 1.0 / t_base;

    // Require >= 15% expected speedup over single-token decode to justify batch verification
    double best_theta = theta_0 * 1.15;
    int32_t best_len = 0;

    double a_l = 1.0;
    double expected_adv = 1.0;
    for (int32_t l = 1; l <= num_cand; ++l) {
        const double c_l = steps[static_cast<size_t>(l - 1)].confidence;
        if (c_l <= 0.0) break;
        a_l *= c_l;
        expected_adv += a_l;

        const double t_ver = estimate_verify_ms(l);
        const double expected_cost = t_ver + (1.0 - a_l) * ema_rollback_ms_;
        const double theta_l = expected_adv / expected_cost;

        if (theta_l > best_theta) {
            best_theta = theta_l * 1.015;  // Causal greedy margin for extending prefix
            best_len = l;
        } else {
            // Because l=1 -> l=2 amortizes the fixed batch verification setup cost across 2 tokens,
            // allow evaluating l=2 when l=1 has strong confidence (c_1 >= 0.58)
            if (l == 1 && num_cand >= 2 && c_l >= 0.58) {
                continue;
            }
            break;
        }
    }
    return best_len;
}

void Nvfp4NgramDrafter::draft(int32_t cur_tok, size_t prompt_len, const std::vector<int32_t>& history,
                              bool enabled, std::vector<int32_t>& draft_out) {
    draft_out.clear();
    last_draft_len_ = 0;
    if (!enabled || prompt_len < 8 || history.size() < 4) return;
    const size_t n = history.size();
    const int32_t w0 = history[n - 3];
    const int32_t w1 = history[n - 2];
    const int32_t w2 = history[n - 1];
    const int32_t w3 = cur_tok;

    // If cur_tok is inside a multi-digit number (Qwen tokenizes each digit 0..9 separately),
    // require any history match to align with the full digit prefix from the start of the number
    // (prevents e.g. "mid-20" from matching the middle two digits of "gfx1201").
    size_t cur_digit_run = 0;
    if (is_digit_tok(w3)) {
        cur_digit_run = 1;
        while (cur_digit_run <= n && is_digit_tok(history[n - cur_digit_run])) {
            cur_digit_run++;
        }
    }

    bool matched_higher_order = false;

    // Multi-level Markov suffix matcher: L = 4 (with extension up to L = 32), 3, 2, 1
    for (int L = 4; L >= 1; --L) {
        if (L == 1 && matched_higher_order) {
            // Never fall through to an uncalibrated 1-gram if a >=2-gram suffix matched in history
            break;
        }

        std::vector<int32_t> cands;
        std::vector<int> totals;
        std::vector<int> agrees;
        std::vector<int> runners;
        int32_t best_match_L = L;

        const size_t min_i = std::max(static_cast<size_t>(L - 1), cur_digit_run);
        if (n < 2 || (n - 2) < min_i) continue;
        for (size_t i = n - 2;; --i) {
            bool match = (history[i] == w3);
            if (match && L >= 2) match = (history[i - 1] == w2);
            if (match && L >= 3) match = (history[i - 2] == w1);
            if (match && L >= 4) match = (history[i - 3] == w0);
            if (match && cur_digit_run > 0) {
                for (size_t d = 1; d < cur_digit_run; ++d) {
                    if (history[i - d] != history[n - d]) {
                        match = false;
                        break;
                    }
                }
                if (match && is_digit_tok(history[i - cur_digit_run])) {
                    match = false;
                }
            }
            if (match && (i + 1) < n) {
                int32_t cand1 = history[i + 1];
                if (cand1 >= 0 && cand1 < 248040 && !is_rejected(w2, w3, cand1)) {
                    int32_t cur_match_L = L;
                    if (L == 4) {
                        while (cur_match_L < 32 && i >= static_cast<size_t>(cur_match_L) &&
                               n >= static_cast<size_t>(cur_match_L) &&
                               history[i - static_cast<size_t>(cur_match_L)] ==
                                   history[n - static_cast<size_t>(cur_match_L)]) {
                            cur_match_L++;
                        }
                    }

                    if (cands.empty() || (L == 4 && cur_match_L > best_match_L && cand1 != cands[0])) {
                        int prev_runner0 = 0;
                        int prev_total0 = 0;
                        if (!cands.empty()) {
                            prev_total0 = totals[0];
                            prev_runner0 = (cand1 != cands[0]) ? std::max(runners[0], agrees[0]) : runners[0];
                        }
                        cands.assign(1, cand1);
                        totals.assign(1, prev_total0);
                        agrees.assign(1, 0);
                        runners.assign(1, prev_runner0);
                        best_match_L = std::max(best_match_L, cur_match_L);
                    }

                    totals[0]++;
                    if (cand1 == cands[0]) {
                        agrees[0]++;
                        best_match_L = std::max(best_match_L, cur_match_L);
                        const bool can_grow = (cands.size() == 1);
                        const size_t max_ext = 28;
                        for (size_t s = 1; (i + 1 + s) < n && s < max_ext; ++s) {
                            int32_t cs = history[i + 1 + s];
                            if (cs < 0 || cs >= 248040) break;
                            if (s == cands.size()) {
                                if (!can_grow) break;
                                int32_t p2 = (s == 1) ? w3 : cands[s - 2];
                                int32_t p1 = cands[s - 1];
                                if (is_rejected(p2, p1, cs) || is_new_number_head(p2, p1, cs)) break;
                                int32_t cs_next_peek = ((i + 2 + s) < n && history[i + 2 + s] < 248040)
                                                           ? history[i + 2 + s]
                                                           : -1;
                                if (L < 4 && is_clause_break(cs, cs_next_peek)) break;
                                if (L <= 2 && !is_subword_cont(p1, cs, cs_next_peek) &&
                                    !is_entity_or_num(cs) && !is_digit_tok(cs) &&
                                    !(s == 1 && is_content_token(cs))) {
                                    break;
                                }
                                cands.push_back(cs);
                                totals.push_back(0);
                                agrees.push_back(0);
                                runners.push_back(0);
                            }
                            totals[s]++;
                            if (cs == cands[s]) {
                                agrees[s]++;
                            } else {
                                runners[s]++;
                                break;
                            }
                        }
                    } else {
                        runners[0]++;
                    }
                }
            }
            if (i == min_i) break;
        }
        if (cands.empty() || totals[0] == 0 || cands[0] < 0) continue;

        const int32_t c1 = cands[0];
        const int32_t c2 = (cands.size() > 1) ? cands[1] : -1;
        const int total1 = totals[0];
        const int agree1 = agrees[0];
        const int runner1 = runners[0];

        if (L >= 3 || (L == 2 && total1 >= 2)) {
            matched_higher_order = true;
        }

        // Subword-chain word-boundary normalization:
        // If w3 completes a multi-token hyphenated/numeric word (e.g. "out-of-order" = 5 tokens)
        // and c1 starts a new word (!is_subword_cont(w3, c1, c2)), count how many tokens belong
        // to the single word ending at w3. If the entire suffix match (best_match_L) is just that
        // single word (the word preceding it did not match), the inter-word Markov order is 1 word.
        if (!is_subword_cont(w3, c1, c2) && is_subword_cont(w2, w3, c1)) {
            int32_t tail_word_toks = 2;
            while (tail_word_toks < best_match_L && n >= static_cast<size_t>(tail_word_toks)) {
                int32_t p_tok = history[n - static_cast<size_t>(tail_word_toks)];
                int32_t c_tok = history[n - static_cast<size_t>(tail_word_toks) + 1];
                int32_t n_tok = (tail_word_toks == 2) ? w3 : history[n - static_cast<size_t>(tail_word_toks) + 2];
                if (!is_subword_cont(p_tok, c_tok, n_tok)) break;
                tail_word_toks++;
            }
            if (best_match_L <= tail_word_toks && agree1 < 2 &&
                !(is_entity_or_num(w3) && is_entity_or_num(c1))) {
                continue;
            }
        }

        auto space_ok = [&](int32_t ck, int32_t ck_next, int ag_next, int tot_next, int32_t prev_t) -> bool {
            if (get_tok_str(ck) != " ") return true;
            return (ck_next >= 0 && tot_next > 0 && ag_next == tot_next && !is_new_number_head(prev_t, ck, ck_next));
        };

        auto cand_at = [&](size_t idx) -> int32_t {
            return (idx < cands.size()) ? cands[idx] : -1;
        };
        auto tot_at = [&](size_t idx) -> int {
            return (idx < totals.size()) ? totals[idx] : 0;
        };
        auto agr_at = [&](size_t idx) -> int {
            return (idx < agrees.size()) ? agrees[idx] : 0;
        };

        std::vector<StepStats> steps;
        const int32_t base_bucket = (best_match_L >= 5) ? 4 : (L - 1);

        if (L == 4) {
            if (is_new_number_head(w2, w3, c1) && agree1 < 2 && best_match_L < 5) continue;
            if (is_clause_break(c1, c2) && agree1 < 2 && best_match_L < 6) continue;
            if (agree1 <= runner1) continue;

            StepStats st0{c1, agree1, total1, runner1, best_match_L, base_bucket, 0.0};
            st0.confidence = estimate_step_confidence(st0, w2, w3, c2);
            steps.push_back(st0);

            for (size_t s = 1; s < cands.size(); ++s) {
                int32_t cs = cands[s];
                int32_t cs_next = cand_at(s + 1);
                int32_t prev1 = cands[s - 1];
                int32_t prev2 = (s == 1) ? w3 : cands[s - 2];
                int tot_s = totals[s];
                int agr_s = agrees[s];
                int run_s = runners[s];
                int32_t eff_L = best_match_L + static_cast<int32_t>(s);
                if (cs < 0 || tot_s <= 0 || agr_s <= run_s) break;
                if (is_rejected(prev2, prev1, cs) || is_new_number_head(prev2, prev1, cs)) break;
                if (is_clause_break(cs, cs_next) && eff_L < 6 && agr_s < 2) break;
                StepStats sts{cs, agr_s, tot_s, run_s, eff_L, std::min(4, base_bucket + 1), 0.0};
                sts.confidence = estimate_step_confidence(sts, prev2, prev1, cs_next);
                if (sts.confidence <= 0.0) break;
                steps.push_back(sts);
            }
        } else if (L == 3) {
            int n_content3 = (is_content_token(w1) ? 1 : 0) + (is_content_token(w2) ? 1 : 0) +
                             (is_content_token(w3) ? 1 : 0);
            bool ok3 = (agree1 == total1) && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                       space_ok(c1, c2, agr_at(1), tot_at(1), w3) &&
                       (agree1 >= 2 || is_subword_cont(w3, c1, c2) || is_subword_cont(w2, w3, c1) ||
                        is_content_token(c1) || n_content3 >= 2);
            if (ok3) {
                StepStats st0{c1, agree1, total1, runner1, 3, 2, 0.0};
                st0.confidence = estimate_step_confidence(st0, w2, w3, c2);
                steps.push_back(st0);

                for (size_t s = 1; s < cands.size(); ++s) {
                    int32_t cs = cands[s];
                    int32_t cs_next = cand_at(s + 1);
                    int32_t prev1 = cands[s - 1];
                    int32_t prev2 = (s == 1) ? w3 : cands[s - 2];
                    int tot_s = totals[s];
                    int agr_s = agrees[s];
                    int run_s = runners[s];
                    int32_t eff_L = 3 + static_cast<int32_t>(s);
                    if (cs < 0 || tot_s <= 0 || agr_s != tot_s) break;
                    if (is_rejected(prev2, prev1, cs) || is_clause_break(cs, cs_next) ||
                        is_new_number_head(prev2, prev1, cs)) {
                        break;
                    }
                    if (!space_ok(cs, cs_next, agr_at(s + 1), tot_at(s + 1), prev1)) break;
                    StepStats sts{cs, agr_s, tot_s, run_s, eff_L, std::min(4, 2 + static_cast<int32_t>(s)), 0.0};
                    sts.confidence = estimate_step_confidence(sts, prev2, prev1, cs_next);
                    if (sts.confidence <= 0.0) break;
                    steps.push_back(sts);
                }
            }
        } else if (L == 2) {
            bool ok2 = (agree1 == total1) && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                       space_ok(c1, c2, agr_at(1), tot_at(1), w3) &&
                       (is_subword_cont(w3, c1, c2) || is_subword_cont(w2, w3, c1) || is_content_token(w3) ||
                        (is_content_token(w2) && is_content_token(c1)));
            if (ok2) {
                StepStats st0{c1, agree1, total1, runner1, 2, 1, 0.0};
                st0.confidence = estimate_step_confidence(st0, w2, w3, c2);
                steps.push_back(st0);

                for (size_t s = 1; s < cands.size(); ++s) {
                    int32_t cs = cands[s];
                    int32_t cs_next = cand_at(s + 1);
                    int32_t prev1 = cands[s - 1];
                    int32_t prev2 = (s == 1) ? w3 : cands[s - 2];
                    int tot_s = totals[s];
                    int agr_s = agrees[s];
                    int run_s = runners[s];
                    int32_t eff_L = 2 + static_cast<int32_t>(s);
                    if (cs < 0 || tot_s <= 0 || agr_s != tot_s) break;
                    if (is_rejected(prev2, prev1, cs) || is_clause_break(cs, cs_next) ||
                        is_new_number_head(prev2, prev1, cs)) {
                        break;
                    }
                    if (!space_ok(cs, cs_next, agr_at(s + 1), tot_at(s + 1), prev1)) break;
                    bool struct_ok = is_subword_cont(prev1, cs, cs_next) ||
                                     (s == 1 ? is_content_token(cs) : is_entity_or_num(cs)) ||
                                     total1 >= 2;
                    if (!struct_ok) break;
                    StepStats sts{cs, agr_s, tot_s, run_s, eff_L, std::min(4, 1 + static_cast<int32_t>(s)), 0.0};
                    sts.confidence = estimate_step_confidence(sts, prev2, prev1, cs_next);
                    if (sts.confidence <= 0.0) break;
                    steps.push_back(sts);
                }
            }
        } else if (L == 1) {
            // Strict calibrated gating at L = 1 to prevent unigram false positives:
            // Do not trigger on single-occurrence lowercase common words (e.g. " out" -> "-of")
            // or short lowercase prefixes (e.g. " re", " mis").
            const std::string& w3_s = get_tok_str(w3);
            const std::string& c1_s = get_tok_str(c1);
            size_t w3_pos = (!w3_s.empty() && w3_s[0] == ' ') ? 1 : 0;
            size_t w3_len = (w3_s.size() > w3_pos) ? (w3_s.size() - w3_pos) : 0;
            bool w3_ent = is_entity_or_num(w3);
            bool c1_alnum_head = !c1_s.empty() && std::isalnum(static_cast<unsigned char>(c1_s[0]));

            bool sub1 = is_subword_cont(w3, c1, c2) &&
                        (w3_ent || is_digit_tok(c1) ||
                         (c1_alnum_head && ((agree1 >= 2 && w3_len >= 4) || w3_len >= 6)));

            bool ent1 = (total1 <= 6) && w3_ent &&
                        (is_entity_or_num(c1) || (c1_s == " " && space_ok(c1, c2, agr_at(1), tot_at(1), w3)));

            if (agree1 == total1 && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                space_ok(c1, c2, agr_at(1), tot_at(1), w3) && (sub1 || ent1)) {
                StepStats st0{c1, agree1, total1, runner1, 1, 0, 0.0};
                st0.confidence = estimate_step_confidence(st0, w2, w3, c2);
                steps.push_back(st0);

                for (size_t s = 1; s < cands.size(); ++s) {
                    int32_t cs = cands[s];
                    int32_t cs_next = cand_at(s + 1);
                    int32_t prev1 = cands[s - 1];
                    int32_t prev2 = (s == 1) ? w3 : cands[s - 2];
                    int tot_s = totals[s];
                    int agr_s = agrees[s];
                    int run_s = runners[s];
                    int32_t eff_L = 1 + static_cast<int32_t>(s);
                    if (cs < 0 || tot_s <= 0 || agr_s != tot_s) break;
                    if (is_rejected(prev2, prev1, cs) || is_clause_break(cs, cs_next) ||
                        is_new_number_head(prev2, prev1, cs)) {
                        break;
                    }
                    if (!space_ok(cs, cs_next, agr_at(s + 1), tot_at(s + 1), prev1)) break;
                    bool cont_ok = is_subword_cont(prev1, cs, cs_next) ||
                                   (s == 1 && (is_entity_or_num(cs) || (c1_s == " " && is_digit_tok(cs)))) ||
                                   (is_digit_tok(prev1) && is_digit_tok(cs));
                    if (!cont_ok) break;
                    StepStats sts{cs, agr_s, tot_s, run_s, eff_L, std::min(4, static_cast<int32_t>(s)), 0.0};
                    sts.confidence = estimate_step_confidence(sts, prev2, prev1, cs_next);
                    if (sts.confidence <= 0.0) break;
                    steps.push_back(sts);
                }
            }
        }

        if (!steps.empty()) {
            int32_t sched_len = schedule_prefix(steps);
            // Never end a draft on a bare space token " "
            while (sched_len > 0 && get_tok_str(steps[static_cast<size_t>(sched_len - 1)].cand) == " ") {
                sched_len--;
            }
            // At L = 1 with only a single occurrence in history (agree1 == 1), require either
            // a 2-token verified entity/subword continuation or an intra-word alphanumeric subword
            if (L == 1 && agree1 == 1 && sched_len == 1) {
                const std::string& c1_s = get_tok_str(steps[0].cand);
                bool strong_subword = is_subword_cont(w3, steps[0].cand, c2) &&
                                      !c1_s.empty() && std::isalnum(static_cast<unsigned char>(c1_s[0]));
                if (!strong_subword && !(is_entity_or_num(w3) && is_entity_or_num(steps[0].cand))) {
                    sched_len = 0;
                }
            }
            if (sched_len > 0) {
                last_draft_buckets_.resize(static_cast<size_t>(sched_len));
                for (int32_t k = 0; k < sched_len; ++k) {
                    draft_out.push_back(steps[static_cast<size_t>(k)].cand);
                    last_draft_buckets_[static_cast<size_t>(k)] = steps[static_cast<size_t>(k)].bucket;
                }
                last_draft_len_ = sched_len;
                return;
            }
        }
    }
}

}  // namespace strata::spec
