// Opt-in native numerical evidence, never a client request parameter. Raw head
// rows are buffered, then written once after generation. Off: no logits copy.
#pragma once
#include "strata/core/verify.hpp"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace strata::program {

class GrammarDiagnostics {
    struct Row {
        int T, row, kept, selected;
        int64_t position;
        std::vector<float> logits;
        std::vector<int32_t> mask, history, inputs;
    };
    const char* path_;
    uint64_t request_;
    int dropped_ = 0;
    strata::kernels::SamplerParams params_;
    std::vector<Row> rows_;
    static uint64_t next_request() { static uint64_t n = 0; return ++n; }
    static size_t& total() { static size_t n = 0; return n; }
public:
    explicit GrammarDiagnostics(bool constrained, const strata::kernels::SamplerParams& params)
        : path_(constrained ? std::getenv("STRATA_GRAMMAR_LOGITS") : nullptr),
          request_(path_ ? next_request() : 0), params_(params) {}

    void capture(const strata::core::Verifier& verifier, int T, int64_t p, int kept,
                 const int32_t* inputs, const int32_t* selected, const int32_t* masks,
                 const int32_t* history, int history_len) {
        if (!path_) return;
        const size_t nv = (size_t)verifier.vocab(), words = (nv + 31) / 32;
        if (nv > 262144) throw std::runtime_error("grammar diagnostics vocabulary bound exceeded");
        for (int row = 0; row < T; ++row) {
            if (rows_.size() >= 32 || total() >= 128) { ++dropped_; continue; }
            Row sample{T, row, kept, selected[row], p + row, std::vector<float>(nv),
                       {masks + row * words, masks + (row + 1) * words}, {}, {inputs, inputs + T}};
            if (history_len) sample.history.assign(history + row * history_len, history + (row + 1) * history_len);
            if (!verifier.copy_logits(row, sample.logits.data())) throw std::runtime_error("grammar diagnostics logits copy failed");
            rows_.push_back(std::move(sample)); ++total();
        }
    }

    void flush() {
        if (!path_) return;
        std::FILE* file = std::fopen(path_, "ab");
        if (!file) throw std::runtime_error("grammar diagnostics cannot open output file");
        bool ok = true;
        for (const auto& row : rows_) {
            // G5ROW1: ASCII header, then native little-endian f32 logits and
            // i32 mask/history/input arrays, then LF. Qualified x86 host only.
            ok &= std::fprintf(file, "G5ROW1 %llu %d %d %d %lld %d %zu %zu %zu %llu %d %.9g %.9g %.9g %.9g %.9g %.9g %d\n",
                (unsigned long long)request_, row.T, row.row, row.kept, (long long)row.position,
                row.selected, row.logits.size(), row.mask.size(), row.history.size(),
                (unsigned long long)params_.seed, params_.top_k, params_.top_p, params_.min_p,
                params_.temperature, params_.penalty_repeat, params_.penalty_freq, params_.penalty_present,
                params_.greedy) > 0;
            ok &= std::fwrite(row.logits.data(), sizeof(float), row.logits.size(), file) == row.logits.size();
            for (const auto* v : {&row.mask, &row.history, &row.inputs})
                ok &= std::fwrite(v->data(), sizeof(int32_t), v->size(), file) == v->size();
            ok &= std::fputc('\n', file) != EOF;
        }
        ok &= std::fprintf(file, "G5END1 %llu %zu %d\n", (unsigned long long)request_, rows_.size(), dropped_) > 0;
        ok &= std::fclose(file) == 0;
        if (!ok) throw std::runtime_error("grammar diagnostics write failed");
        rows_.clear();
    }
};

} // namespace strata::program
