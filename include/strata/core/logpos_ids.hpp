// include/strata/core/logpos_ids.hpp - measurement hooks beside STRATA_LOGPOS (strata-mesh stage A; off by default).
//
// Both are read once from the environment; unset, nothing here runs and the STRATA_LOGPOS output is byte-identical to
// a build without them.  Header-only and plain C++, so the CUDA/HIP tree and the SYCL port (which keeps its own
// verify.cpp and generate.cpp) share one implementation.
//
// STRATA_LOGPOS_SPLIT_AT=<n> (n > 0): a prompt's read is cut at token n - [0, n) goes through the batched prompt path
//   (prompt chunks) and [n, end) through the verify windows, which write the STRATA_LOGPOS rows.  The KL check of the
//   prompt-chunk hand-off (DESIGN.md 6.3): a long preamble read as chunks, the positions after it scored.
// STRATA_LOGPOS_IDS=<baseline dump>: every STRATA_LOGPOS row gets, after its own columns, a "\t|" and then
//   "\t<id>:<logprob>" for each id that the baseline dump lists at the same text and position (its STRATA_LOGPOS_TOPK
//   columns, in the baseline's order) - the log-probabilities this run gives the baseline's top ids, for a KL over
//   them.  Format and the text/position matching: ~/strata-mesh/IMPL-PLAN-stage-a.md, "STRATA_LOGPOS_IDS".
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core::logpos {

/// STRATA_LOGPOS_SPLIT_AT: the token the prompt read is cut at; -1 when unset (or not positive).
inline int64_t split_at() {
    static const int64_t v = [] {
        const char* e = std::getenv("STRATA_LOGPOS_SPLIT_AT");
        return e != nullptr && *e != '\0' ? (int64_t) std::atoll(e) : (int64_t) -1;
    }();
    return v > 0 ? v : -1;
}

/// The baseline dump: (text number, pos) -> the ids of its `id:lp` columns, in the baseline's order.  A text ends
/// when a row's pos is not above the previous row's (the same rule on the writing side, `next_row`).
class BaselineIds {
public:
    /// Read `path`.  Columns 0 (pos) and 8.. (`id:lp`, up to a bare `|`) of each row are used; a row with fewer than
    /// 8 columns or a column that is not `id:lp` fails with the line number in `err`.  Blank lines are skipped.
    bool load(const std::string& path, std::string& err) {
        texts_.clear();
        rows_ = 0;
        cur_ = -1;
        last_pos_ = 0;
        std::ifstream in(path);
        if (!in) {
            err = path + ": cannot open the baseline dump";
            return false;
        }
        std::string line;
        int64_t lineno = 0, prev_pos = 0;
        bool have_prev = false;
        while (std::getline(in, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::vector<std::string> cols;
            size_t b = 0;
            for (;;) {
                const size_t t = line.find('\t', b);
                if (t == std::string::npos) {
                    cols.push_back(line.substr(b));
                    break;
                }
                cols.push_back(line.substr(b, t - b));
                b = t + 1;
            }
            if (cols.size() < 8) {
                err = path + ":" + std::to_string(lineno) + ": expected at least 8 tab-separated columns, got " +
                      std::to_string(cols.size());
                return false;
            }
            int64_t pos = 0;
            if (!parse_int(cols[0], pos)) {
                err = path + ":" + std::to_string(lineno) + ": the pos column is not an integer: '" + cols[0] + "'";
                return false;
            }
            std::vector<int32_t> ids;
            for (size_t i = 8; i < cols.size(); ++i) {
                if (cols[i] == "|") break;
                int32_t id = 0;
                if (!parse_id_lp(cols[i], id)) {
                    err = path + ":" + std::to_string(lineno) + ": column " + std::to_string(i + 1) +
                          " is not 'id:logprob': '" + cols[i] + "'";
                    return false;
                }
                ids.push_back(id);
            }
            if (!have_prev || pos <= prev_pos) texts_.emplace_back();
            have_prev = true;
            prev_pos = pos;
            texts_.back()[pos] = std::move(ids);
            ++rows_;
        }
        return true;
    }

    /// For the writing side, one call per STRATA_LOGPOS row: the baseline ids of this text and `pos`, or nullptr when
    /// the baseline has no such row.  A `pos` not above the previous call's starts the next text.
    const std::vector<int32_t>* next_row(int64_t pos) {
        if (cur_ < 0 || pos <= last_pos_) ++cur_;
        last_pos_ = pos;
        if ((size_t) cur_ >= texts_.size()) return nullptr;
        const auto it = texts_[(size_t) cur_].find(pos);
        return it == texts_[(size_t) cur_].end() ? nullptr : &it->second;
    }

    size_t texts() const { return texts_.size(); }
    size_t rows() const { return rows_; }
    /// The number of texts `next_row` has counted so far (the text of the last call is `texts_seen() - 1`).
    int64_t texts_seen() const { return cur_ + 1; }

private:
    static bool parse_int(const std::string& s, int64_t& out) {
        if (s.empty()) return false;
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(s.c_str(), &end, 10);
        if (errno != 0 || end != s.c_str() + s.size()) return false;
        out = (int64_t) v;
        return true;
    }
    // `id:logprob`; the logprob is any number strtod accepts (nan and inf included) and is not kept
    static bool parse_id_lp(const std::string& s, int32_t& id) {
        const size_t c = s.find(':');
        if (c == std::string::npos || c == 0 || c + 1 >= s.size()) return false;
        int64_t v = 0;
        if (!parse_int(s.substr(0, c), v) || v < INT32_MIN || v > INT32_MAX) return false;
        const std::string lp = s.substr(c + 1);
        char* end = nullptr;
        std::strtod(lp.c_str(), &end);
        if (end == lp.c_str() || end != lp.c_str() + lp.size()) return false;
        id = (int32_t) v;
        return true;
    }

    std::vector<std::unordered_map<int64_t, std::vector<int32_t>>> texts_;
    size_t rows_ = 0;
    int64_t cur_ = -1;
    int64_t last_pos_ = 0;
};

/// Write "\t|" and then "\t<id>:<row[id] - lse>" for the baseline ids of this row (nothing after the bar when the
/// baseline has no row here; "nan" for an id outside [0, n_vocab)).  False when the write fails.
inline bool write_ids_with(BaselineIds& b, std::FILE* out, int64_t pos, const float* row, int64_t n_vocab,
                           double lse) {
    const std::vector<int32_t>* ids = b.next_row(pos);
    if (std::fputs("\t|", out) < 0) return false;
    if (ids == nullptr) return true;
    for (const int32_t id : *ids) {
        const int r = id >= 0 && (int64_t) id < n_vocab
                          ? std::fprintf(out, "\t%d:%.6f", (int) id, (double) row[id] - lse)
                          : std::fprintf(out, "\t%d:nan", (int) id);
        if (r < 0) return false;
    }
    return true;
}

/// STRATA_LOGPOS_IDS: append the baseline's ids for the row at `pos` (`row`: the row's n_vocab logits, `lse`: their
/// log-sum-exp).  True and nothing written when unset; false with `err` when the baseline cannot be used (every call
/// then fails, so a measurement does not go on with its columns missing).  One thread (the request thread) only.
inline bool write_ids(std::FILE* out, int64_t pos, const float* row, int64_t n_vocab, double lse, std::string& err) {
    struct State {
        bool enabled = false;
        bool ok = true;
        bool warned = false;   // the baseline ran out of texts (warned once)
        std::string error;
        BaselineIds ids;
        State() {
            const char* path = std::getenv("STRATA_LOGPOS_IDS");
            if (path == nullptr || *path == '\0') return;
            enabled = true;
            ok = ids.load(path, error);
            if (ok)
                std::fprintf(stderr, "strata: STRATA_LOGPOS_IDS: %s: %zu texts, %zu rows\n", path, ids.texts(),
                             ids.rows());
            else
                error = "STRATA_LOGPOS_IDS: " + error;
        }
    };
    static State st;
    if (!st.enabled) return true;
    if (!st.ok) {
        err = st.error;
        return false;
    }
    if (!write_ids_with(st.ids, out, pos, row, n_vocab, lse)) {
        err = "STRATA_LOGPOS_IDS: writing the ids failed";
        return false;
    }
    // This run reads more texts than the baseline holds: their rows get a bare "|".  Say so once (stderr only), since
    // a baseline taken from a run that was not finished looks the same as a good one in the output.
    if (!st.warned && (size_t) st.ids.texts_seen() > st.ids.texts()) {
        st.warned = true;
        std::fprintf(stderr,
                     "strata: STRATA_LOGPOS_IDS: this run reached text %lld but the baseline has only %zu texts: "
                     "its rows (and the rest) get no ids\n",
                     (long long) st.ids.texts_seen(), st.ids.texts());
    }
    return true;
}

}  // namespace strata::core::logpos
