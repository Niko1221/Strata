// engine/src/inkling2/ink_serve.h - Continuous-Batching (1..64 slots) `--serve` Protocol Engine for strata-inkling
// Speaks Strata's native line protocol (INFO / READY / GEN / BGEN / BCO / STOP / BSTOP / BYIELD / QUIT -> PP / T / DONE / BADM / BT / BDONE / BCOK / IMG / ERR)
// PLUS extended protocol commands (HELLO / GENERATE <req_id> / CANCEL <req_id> / STATS) on the same non-blocking stdin loop.
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <algorithm>
#include <climits>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"
#include "ink_vision.h"

namespace ink2 {

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#ifndef _MSC_VER
inline void proto(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif
inline void proto(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

struct SampleOpts {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 64;
    float min_p = 0.0f;
    float rep = 1.0f;
    float freq = 0.0f;
    float pres = 0.0f;
    int last_n = 64;
    uint64_t seed = 0;
    int hold_budget = 0;       // hold=N:ID reasoning budget
    int32_t hold_token_id = -1;
    int pen_from = 0;

    bool greedy() const noexcept { return temperature <= 0.0f; }
    bool penalties() const noexcept { return rep != 1.0f || freq != 0.0f || pres != 0.0f; }
};

// The K best token ids of lg[0..V-1], best first: larger logit first, the smaller id first on equal logits.  Exactly what
// std::partial_sort over an iota with that order leaves in idx[0..K-1], but one pass over the row (about 0.1 ms for V = 200058
// instead of a few ms), because every sampled token of every slot pays it on the host (#6779: 16..64 slots, temperature > 0).
inline void ink_top_k_ids(const float* lg, int V, int K, std::vector<int>& idx) {
    K = std::max(0, std::min(K, V));
    idx.assign((size_t)K, 0);
    if (K == 0) return;
    std::vector<float> val((size_t)K);
    int n = 0;
    for (int i = 0; i < V; ++i) {
        const float v = lg[i];
        if (n == K && !(v > val[K - 1])) continue;   // a later id with an equal logit sorts after the kept ones
        int j = (n < K) ? n++ : K - 1;
        while (j > 0 && v > val[j - 1]) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; --j; }
        val[j] = v;
        idx[j] = i;
    }
}

// The sampling step once the K best candidates are known: tv[i] is the logit of token ti[i], best first (larger logit first, the
// smaller id first on equal logits), K = min(top_k, vocab, 128).  Shared by sample_logits (candidates found on the host) and the
// candidate rows a backend returns (forward_batch_cand), so both give the same token for the same seed.
inline int32_t ink_sample_top(const float* tv, const int* ti, int K, const SampleOpts& s, std::mt19937_64& rng) {
    std::vector<double> p(K);
    double mx = tv[0], sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((tv[i] - mx) / static_cast<double>(s.temperature));
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.0f) {
        const double thr = static_cast<double>(s.min_p) * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= static_cast<double>(s.top_p)) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    std::uniform_real_distribution<double> u(0.0, z);
    const double r = u(rng);
    double c = 0.0;
    for (int i = 0; i < keep; ++i) {
        c += p[i];
        if (r <= c) return ti[i];
    }
    return ti[keep - 1];
}

// Sampling from a candidate row (ink2::CandRows, `cand` = the row's kInkCandidates entries).  Only for a slot without penalties:
// a penalty changes logits anywhere in the row before the best are picked, so such a slot needs its full row.
inline int32_t sample_candidates(const float* cv, const int32_t* ci, int V, const SampleOpts& s, std::mt19937_64& rng) {
    if (s.greedy()) return ci[0];
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    return ink_sample_top(cv, ci, K, s, rng);
}

inline int32_t sample_logits(float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.penalties() && s.last_n > 0) {
        size_t start = static_cast<size_t>(std::max(0, s.pen_from));
        if (hist.size() > start) {
            size_t from = (hist.size() - start > static_cast<size_t>(s.last_n)) ? (hist.size() - static_cast<size_t>(s.last_n)) : start;
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
            for (const auto& kv : cnt) {
                if (kv.first < 0 || kv.first >= V) continue;
                float& x = lg[kv.first];
                if (s.rep != 1.0f) x = (x > 0.0f) ? (x / s.rep) : (x * s.rep);
                x -= static_cast<float>(kv.second) * s.freq + (s.pres != 0.0f ? s.pres : 0.0f);
            }
        }
    }
    if (s.greedy()) {
        int32_t best = 0;
        float mx = lg[0];
        for (int i = 1; i < V; ++i) {
            if (lg[i] > mx) { mx = lg[i]; best = i; }
        }
        return best;
    }
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    std::vector<int> idx;
    ink_top_k_ids(lg, V, K, idx);
    std::vector<float> tv((size_t)K);
    for (int i = 0; i < K; ++i) tv[(size_t)i] = lg[idx[(size_t)i]];
    return ink_sample_top(tv.data(), idx.data(), K, s, rng);
}

inline bool finite_logits(const float* lg, int V) noexcept {
    for (int i = 0; i < V; i += 67) if (!std::isfinite(lg[i])) return false;
    return true;
}

// Exact lossless speculative verification sampler for DSpark (Leviathan et al., 2023):
// Given target logits `lg[0..V-1]` and a deterministic draft token `draft_tok`:
//   - If greedy: picks argmax and accepts iff argmax == draft_tok.
//   - If non-greedy (temperature > 0 / top_k / min_p / top_p / penalties):
//     Computes exact target distribution p(x), accepts `draft_tok` with probability p(draft_tok),
//     else samples from exact residual p(x) / (1 - p(draft_tok)) over x != draft_tok.
// Marginal distribution of emitted tokens is 100% mathematically identical to sample_logits()!
inline int32_t verify_draft_logits(
    float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist,
    std::mt19937_64& rng, int32_t draft_tok, bool& accepted) {
    accepted = false;
    if (s.greedy()) {
        const int32_t tok = sample_logits(lg, V, s, hist, rng);
        accepted = (tok == draft_tok);
        return tok;
    }
    if (s.penalties() && s.last_n > 0) {
        size_t start = static_cast<size_t>(std::max(0, s.pen_from));
        if (hist.size() > start) {
            size_t from = (hist.size() - start > static_cast<size_t>(s.last_n)) ? (hist.size() - static_cast<size_t>(s.last_n)) : start;
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
            for (const auto& kv : cnt) {
                if (kv.first < 0 || kv.first >= V) continue;
                float& x = lg[kv.first];
                if (s.rep != 1.0f) x = (x > 0.0f) ? (x / s.rep) : (x * s.rep);
                x -= static_cast<float>(kv.second) * s.freq + (s.pres != 0.0f ? s.pres : 0.0f);
            }
        }
    }
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    std::vector<int> idx;
    ink_top_k_ids(lg, V, K, idx);
    std::vector<double> p(K);
    double mx = lg[idx[0]], sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((lg[idx[i]] - mx) / static_cast<double>(s.temperature));
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.0f) {
        const double thr = static_cast<double>(s.min_p) * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= static_cast<double>(s.top_p)) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    int draft_pos = -1;
    for (int i = 0; i < keep; ++i) {
        z += p[i];
        if (idx[i] == draft_tok) draft_pos = i;
    }
    if (z <= 0.0) return idx[0];
    const double p_draft = (draft_pos >= 0) ? (p[draft_pos] / z) : 0.0;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (draft_pos >= 0 && (keep == 1 || u01(rng) < p_draft)) {
        accepted = true;
        return draft_tok;
    }
    // Rejected: sample from residual p(x) over i != draft_pos
    const double z_rem = z - (draft_pos >= 0 ? p[draft_pos] : 0.0);
    if (z_rem <= 0.0) {
        accepted = (draft_pos >= 0);
        return (draft_pos >= 0) ? draft_tok : idx[0];
    }
    const double r = u01(rng) * z_rem;
    double c = 0.0;
    int last_valid = idx[0];
    for (int i = 0; i < keep; ++i) {
        if (i == draft_pos) continue;
        last_valid = idx[i];
        c += p[i];
        if (r <= c) return idx[i];
    }
    return last_valid;
}

// Anti-loop repetition detector: detects pathological n-gram stalling while avoiding false positives on structured YAML/JSON (#6620)
inline bool detect_repetition_loop(const std::vector<int32_t>& gen_ids) noexcept {
    const size_t n = gen_ids.size();
    if (n < 96) return false;
    for (size_t period = 1; period <= 32; ++period) {
        const size_t need_reps = (period <= 4) ? 24 : 10;
        if (n < period * need_reps) continue;
        bool match = true;
        for (size_t k = 0; k < period * (need_reps - 1); ++k) {
            if (gen_ids[n - 1 - k] != gen_ids[n - 1 - k - period]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

struct ParsedGen {
    std::string cmd;      // GEN, BGEN, GENERATE
    int slot = 0;         // for BGEN <slot> ...
    std::string req_id;   // for GENERATE <req_id> ...
    int max_new = 128;
    SampleOpts s;
    std::vector<int32_t> ids;
    std::vector<std::string> images;
    int pin = 0;          // shared-prefix checkpoint boundary (tokens)
    bool prefill_only = false;
    bool slot_read = false;
    bool no_ckpt = false; // ckpt=0: save no prompt-end checkpoint for this request and keep the slot's (a continuation pass)
};

inline bool parse_id_list(const std::string& s, std::vector<int32_t>& out, std::string& err) {
    out.clear();
    size_t p = 0;
    while (p < s.size()) {
        char* ee = nullptr;
        long v = std::strtol(s.c_str() + p, &ee, 10);
        if (ee == s.c_str() + p) {
            err = "bad token id list";
            return false;
        }
        out.push_back(static_cast<int32_t>(v));
        p = static_cast<size_t>(ee - s.c_str());
        if (p < s.size() && s[p] == ',') ++p;
    }
    return !out.empty();
}

inline bool parse_gen_cmd(const std::string& line, ParsedGen& r, std::string& err) {
    std::vector<std::string> w;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        size_t j = i;
        while (j < line.size() && line[j] != ' ') ++j;
        if (j > i) w.push_back(line.substr(i, j - i));
        i = j;
    }
    if (w.empty()) return false;
    r.cmd = w[0];
    size_t arg_idx = 1;
    if (r.cmd == "BGEN") {
        if (w.size() < 4) { err = "expected: BGEN <slot> <max_new> [k=v ...] <ids>"; return false; }
        r.slot = std::atoi(w[1].c_str());
        arg_idx = 2;
    } else if (r.cmd == "GENERATE") {
        if (w.size() < 4) { err = "expected: GENERATE <req_id> <max_new> [k=v ...] <ids>"; return false; }
        r.req_id = w[1];
        arg_idx = 2;
    } else if (r.cmd == "GEN") {
        if (w.size() < 3) { err = "expected: GEN <max_new> [k=v ...] <ids>"; return false; }
        r.slot = 0;
        arg_idx = 1;
    } else {
        err = "unknown command " + r.cmd;
        return false;
    }
    char* e = nullptr;
    long mn = std::strtol(w[arg_idx].c_str(), &e, 10);
    if (*e || mn < 0) { err = "bad max_new"; return false; }
    r.max_new = static_cast<int>(mn);

    for (size_t k = arg_idx + 1; k + 1 < w.size(); ++k) {
        size_t eq = w[k].find('=');
        if (eq == std::string::npos) { err = "bad key " + w[k]; return false; }
        std::string key = w[k].substr(0, eq), val = w[k].substr(eq + 1);
        double d = std::atof(val.c_str());
        if (key == "temperature") r.s.temperature = static_cast<float>(d);
        else if (key == "top_p") r.s.top_p = static_cast<float>(d);
        else if (key == "top_k") r.s.top_k = std::max(1, std::min(128, static_cast<int>(d)));
        else if (key == "min_p") r.s.min_p = static_cast<float>(d);
        else if (key == "penalty_repeat") r.s.rep = static_cast<float>(d);
        else if (key == "penalty_freq") r.s.freq = static_cast<float>(d);
        else if (key == "penalty_present") r.s.pres = static_cast<float>(d);
        else if (key == "penalty_last_n") r.s.last_n = static_cast<int>(d);
        else if (key == "pen_from") r.s.pen_from = static_cast<int>(d);
        else if (key == "seed") r.s.seed = std::strtoull(val.c_str(), nullptr, 10);
        else if (key == "pin") {
            char* end = nullptr;
            const long n = std::strtol(val.c_str(), &end, 10);
            if (end == val.c_str() || *end || n <= 0 || n > INT32_MAX) {
                err = "bad pin boundary";
                return false;
            }
            r.pin = static_cast<int>(n);
        }
        else if (key == "prefill") r.prefill_only = (d != 0.0);
        else if (key == "slot_read") r.slot_read = (d != 0.0);
        else if (key == "ckpt") r.no_ckpt = (d == 0.0);
        else if (key == "image") r.images.push_back(pct_decode(val));
        else if (key == "hold") {
            size_t col = val.find(':');
            if (col != std::string::npos) {
                r.s.hold_budget = std::atoi(val.substr(0, col).c_str());
                r.s.hold_token_id = std::atoi(val.substr(col + 1).c_str());
            }
        } else if (key == "pcie_frac" || key == "spec_min_p" || key == "prefill_pipe") {
            continue;
        } else {
            err = "unsupported key " + key;
            return false;
        }
    }
    return parse_id_list(w.back(), r.ids, err);
}

// ---- Non-blocking stdin reader thread ----
class ServeInput {
public:
    std::atomic<bool> solo_stop{false};
    void start() {
        th_ = std::thread([this] {
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (line == "STOP") {
                    solo_stop.store(true);
                    std::lock_guard<std::mutex> lk(m_);
                    q_.push_back(line);
                    cv_.notify_one();
                    continue;
                }
                std::lock_guard<std::mutex> lk(m_);
                q_.push_back(std::move(line));
                cv_.notify_one();
            }
            std::lock_guard<std::mutex> lk(m_);
            eof_ = true;
            cv_.notify_all();
        });
        th_.detach();
    }

    bool poll(std::string& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool wait_next(std::string& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return !q_.empty() || eof_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool is_eof() {
        std::lock_guard<std::mutex> lk(m_);
        return eof_ && q_.empty();
    }

private:
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    bool eof_ = false;
};

struct ActiveReq {
    bool active = false;
    bool is_solo = false;       // GEN vs BGEN / GENERATE
    bool is_ext_gen = false;    // GENERATE <req_id>
    std::string req_id;
    int slot = 0;
    int max_new = 0;
    int generated = 0;
    int prompt_len = 0;
    int prefill_pos = 0;        // next prompt token index to feed
    int cur_pos = 0;            // next decode position in slot
    int32_t next_tok = -1;      // token ready to feed in decode step
    SampleOpts s;
    std::mt19937_64 rng;
    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> prompt_img_idx;
    std::vector<float> img_embeds;
    std::vector<int32_t> hist;
    std::vector<int32_t> gen_ids;
    std::vector<int32_t> cached_prefix;       // tokens currently resident in slot KV
    std::vector<int32_t> prompt_ckpt_prefix;  // prompt-end checkpoint prefix (#6627): used on a backend without ckpt_save
    int reused_tokens = 0;                    // prefix tokens reused for this request
    std::vector<int32_t> shared_prefix;
    int pin = 0;
    bool shared_pending = false;
    bool no_ckpt = false;                     // ckpt=0: this request takes no mark at its prompt's end
    bool ckpt_pending = false;                // true while prefilling up to prompt_len - 1 to save that checkpoint
    // A picture's positions hold the same pad token whatever the picture is, so the tokens alone cannot say that two prompts agree
    // there: position -> the picture's content hash (mixed with the pad's index in it), for what the slot holds and for this prompt.
    std::unordered_map<int, uint64_t> cached_sig;
    std::unordered_map<int, uint64_t> prompt_sig;
    struct Ckpt { int pos; int handle; };
    std::vector<Ckpt> ckpts;             // the slot's checkpoints, ascending by position (InkBackend::ckpt_save)
    int turn_mark = 0;                   // the position of the mark before the last token of the prompt read last (0: none)
    // `slot` is the request's name on the protocol lines.  `phys` is the backend slot (KV store) it runs in: the engine picks, for
    // each request, the idle store that already holds the most of its prompt, so every conversation keeps its own cache whatever
    // slot number the server names (it sends every request that runs alone as GEN = slot 0).  -1: holds no store.
    int phys = -1;
    double t_start = 0.0;
    double prompt_ms = 0.0;
    double decode_t0 = 0.0;
    int drafts_offered = 0;
    int drafts_accepted = 0;
    // drafting under sampling (temperature > 0): how many tokens to draft, plain steps left before the next try,
    // and the measured cost of a plain step and of a drafted token
    int sp_len = 3;
    int sp_cool = 0;
    int sp_steps = 0;
    int sp_cools = 0;
    int sp_fail = 0;
    int sp_win_tok = 0;
    double sp_win_ms = 0.0;
    double sp_plain_ms = 0.0;
    double sp_tok_ms = 0.0;
    bool cancelled = false;
    bool admitted = false;
    bool slot_read = false;
    // STRATA_INK_SPEC=adapt governor state (measurement experiment, see run_serve_loop)
    int gov_plain_left = 0;    // plain decode steps still to run before the next speculative probe
    int gov_calib_left = 0;    // plain steps to run first to learn what one plain step costs
    int gov_win_steps = 0;     // speculative steps in the current window
    int gov_win_emit = 0;      // tokens those steps produced
    double gov_gain = -1.0;    // smoothed gain of the last windows (< 0: none yet)
};

struct ServeConfig {
    std::string model_path;
    std::string vision_path;
    BackendOpts bopts;
    std::string draft_mode = "auto";  // auto | dspark | off
    bool force_cpu = false;
    bool anti_loop = true;
    std::string version = "0.1.69";
};

inline uint64_t ink_fnv1a(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

inline bool ink_hash_file(const std::string& path, uint64_t& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint64_t h = 1469598103934665603ULL;
    static uint8_t buf[1 << 16];
    size_t n, total = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) { h = ink_fnv1a(buf, n, h); total += n; }
    std::fclose(f);
    out = h ^ (static_cast<uint64_t>(total) * 0x9E3779B97F4A7C15ULL);
    if (out == 0) out = 1;
    return true;
}

inline int ink_env_int(const char* name, int dflt, int lo, int hi) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::max(lo, std::min(hi, std::atoi(v)));
}

inline int run_serve_loop(const ServeConfig& cfg, InkModelFile& mf, InkBackend& backend) {
    const InkCfg& c = mf.m.c;
    const int V = c.vocab_unpadded;
    const int P = std::max(1, std::min(kMaxParallelSlots, cfg.bopts.parallel));
    const int chunk_sz = std::max(1, backend.effective_chunk() > 0 ? backend.effective_chunk() : cfg.bopts.chunk);
    // STRATA_INK_PREFILL_CAP=N: tokens of prompt read per step while another slot decodes (default 0 = no cap, as before).
    // It only moves time between the request that reads and the ones that decode (measured on an RTX 5090, 4096-token prompt read
    // while another request decodes: cap 64 / 128 / 256 / none = decode 24 / 21 / 17 / 12.6 tok/s, prompt first token after
    // 2.75 / 1.63 / 1.05 / 0.88 s), so it is opt-in; answers are identical for every value.
    int prefill_cap = 0;
    // STRATA_INK_NO_CAND=1: always copy the full logit rows to the host (the old way) instead of the backend's candidate rows.
    bool cand_ok = !(std::getenv("STRATA_INK_NO_CAND") && std::getenv("STRATA_INK_NO_CAND")[0] == '1');
    if (const char* e = std::getenv("STRATA_INK_PREFILL_CAP"); e && *e) prefill_cap = std::max(0, std::atoi(e));
    const bool use_dspark = (cfg.draft_mode != "off") && mf.m.dspark.present && backend.dspark_ready();
    // STRATA_INK_SPEC=block: measurement switch, greedy requests use the linear block-8 drafter + batch verify instead of the beam tree
    const char* spec_env = std::getenv("STRATA_INK_SPEC");
    const bool spec_tree = !(spec_env && std::string(spec_env) == "block");
    // STRATA_INK_SPEC=adapt: greedy drafting stays on only while one verify window is worth its cost. Per request, after every 12 speculative
    // steps: the smoothed tokens per step / (spec step ms / plain step ms) below 0.93 turns drafting off for 40 plain steps, then it probes again. The plain step
    // time is measured on the first 4 decode steps of the first request and refreshed by every plain step after that.
    const bool spec_adapt = (spec_env && std::string(spec_env) == "adapt") || cfg.bopts.spec_adapt;
    double gov_t_plain = 0.0, gov_t_spec = 0.0;
    // one speculative step (the greedy tree step or the sampled block step) took `ms` and produced `n_emit` tokens
    auto gov_step = [&](ActiveReq& rq, int n_emit, double ms) {
        gov_t_spec = (gov_t_spec <= 0.0) ? ms : 0.8 * gov_t_spec + 0.2 * ms;
        rq.gov_win_steps += 1;
        rq.gov_win_emit += n_emit;
        if (rq.gov_win_steps >= 12) {
            const double per_step = static_cast<double>(rq.gov_win_emit) / rq.gov_win_steps;
            const double gain = (gov_t_plain > 0.0 && gov_t_spec > 0.0) ? per_step / (gov_t_spec / gov_t_plain) : 2.0;
            rq.gov_gain = (rq.gov_gain < 0.0) ? gain : 0.5 * rq.gov_gain + 0.5 * gain;
            if (rq.gov_gain < 0.93) { rq.gov_plain_left = 40; rq.gov_gain = -1.0; }
            rq.gov_win_steps = 0;
            rq.gov_win_emit = 0;
        }
    };
    const bool cache_trace = std::getenv("STRATA_INK_CACHE_TRACE") != nullptr;
    const bool sample_draft = [] { const char* e = std::getenv("STRATA_INK_SAMPLE_DRAFT"); return !e || std::atoi(e) != 0; }();

    // Probe Vision shard lazily (zero VRAM/RAM until first image request)
    bool images_ok = false;
    VisionShardInfo vprobe;
    CpuVisionEncoder cpu_vit;
    if (!cfg.vision_path.empty()) {
        std::string verr;
        if (vision_probe(cfg.vision_path, vprobe, verr)) images_ok = true;
        else fprintf(stderr, "[inkling2-vision] disabled: %s\n", verr.c_str());
    }

    // Allocate all P slots (lazy paged KV uses near-zero memory until populated!)
    for (int s = 0; s < P; ++s) {
        std::string aerr;
        if (!backend.alloc_slot(s, cfg.bopts.max_ctx, cfg.bopts.kv_dtype, aerr)) {
            proto("ERR alloc_slot %d: %s", s, aerr.c_str());
            return 1;
        }
    }

    std::string base = cfg.model_path;
    size_t sl = base.find_last_of("/\\");
    if (sl != std::string::npos) base = base.substr(sl + 1);

    const int eff_max_ctx = (backend.effective_max_ctx() > 0) ? backend.effective_max_ctx() : cfg.bopts.max_ctx;
    const auto kv_str = [&]() -> const char* {
        const KvDtype kd = backend.stats().kv_dtype;
        if (kd == KvDtype::FP8_E4M3) return "fp8";
        if (kd == KvDtype::K8V8)     return "k8v8";
        if (kd == KvDtype::K8V4)     return "k8v4";
        return "fp16";
    }();
    proto("INFO engine=strata-inkling-%s arch=inkling2_loop backend=%s model=%s layers=%d visits=%d vocab=%d chunk=%d batch_slots=%d slot_read=1 shared_prefix=%d kv_dtype=%s drafter=%s drafting=%s images=%d image_pad=%d parallel=%d ckpt_key=1",
          cfg.version.c_str(), backend.backend_name(), base.c_str(), c.n_phys, c.n_visits, V, chunk_sz,
          P, backend.supports_shared_checkpoint() ? 1 : 0, kv_str,
          mf.m.dspark.present ? "dspark" : "none",
          use_dspark ? "dspark" : "off",
          images_ok ? 1 : 0, c.image_token_id, kMaxParallelSlots);
    proto("READY %d stop hold holdb", eff_max_ctx);

    std::vector<ActiveReq> reqs(P);
    std::unordered_map<std::string, int> req_id_to_slot;
    ServeInput in;
    in.start();

    auto is_stop_tok = [&](int32_t tok) -> bool {
        return tok == c.eos_token_id || tok == c.eot_token_id;
    };

    // ---- checkpoints.  A prompt that does not continue a slot's tokens exactly used to be read from token 0 (the sliding-window
    // layers cannot be rolled back).  The client of an agent rewrites the last assistant turn on every request (it drops the
    // thinking, re-serializes the call), so that was the usual case: a 60k-token conversation was read again on every step.
    // Now a slot keeps checkpoints - one before the last token of each prompt (the next request shares at least that much), and one every
    // STRATA_INK_CKPT_EVERY tokens of a long read - and a prompt that diverges is resumed from the nearest one before the divergence.
    // The periodic ones sit on a fixed grid (every multiple of STRATA_INK_CKPT_EVERY tokens: the read stops there), so the cache is
    // a chunked prefix cache: two conversations that start the same way have their checkpoints at the same places, and one can
    // start from the other's (see store_acquire: the shared tools and system prompt are read once, whichever store read them).
    // A checkpoint is ~164 MiB of host memory (ckpt_bytes), so the number kept per store is also bounded by a budget for all stores
    // together (STRATA_INK_CKPT_MB, 6 GiB): 16 each up to two stores, 12 with three, 4 with eight.  A backend without checkpoints
    // (ckpt_bytes 0) gets none, and then no read is split for them.
    int ckpt_max = ink_env_int("STRATA_INK_CKPT_MAX", 16, 0, 256);                  // per store; 0 turns checkpoints off
    {
        const size_t one = backend.ckpt_bytes();
        const size_t budget = static_cast<size_t>(ink_env_int("STRATA_INK_CKPT_MB", 6144, 0, 1 << 20)) << 20;
        if (one == 0) ckpt_max = 0;
        else ckpt_max = std::min(ckpt_max, static_cast<int>(budget / one / static_cast<size_t>(P)));
        if (one != 0 && ckpt_max == 1) ckpt_max = 2;       // (one grid mark and the mark before the prompt's end)
    }
    // The grid: a mark every ckpt_every tokens (0: none).  A store keeps ckpt_max marks, so on a long prompt the grid thins itself
    // (the mark whose neighbours are closest goes first).  Where prompts differ in practice is the end - the last turn is edited or
    // sent again - so the last ckpt_fine_span tokens of every read are marked every ckpt_fine tokens, and half of the marks are
    // kept for those (the newest ones).  A mark is the whole ring state (ckpt_bytes), not a block of position-indexed KV, which is
    // why the fine step cannot be used for the whole context.
    const int ckpt_every = ink_env_int("STRATA_INK_CKPT_EVERY", 2048, 0, 1 << 24);
    const int ckpt_fine = ink_env_int("STRATA_INK_CKPT_FINE", 256, 0, 1 << 24);     // 0: no fine marks
    const int ckpt_fine_span = ink_env_int("STRATA_INK_CKPT_FINE_SPAN", 2048, 0, 1 << 24);
    auto on_grid = [&](int pos) { return ckpt_every > 0 && pos % ckpt_every == 0; };
    auto fine_mark = [&](const ActiveReq& rq, int pos) {      // is `pos` one of the fine marks of the prompt being read?
        return ckpt_fine > 0 && pos % ckpt_fine == 0 && pos >= rq.prompt_len - 1 - ckpt_fine_span;
    };
    const int fork_min = ink_env_int("STRATA_INK_FORK_MIN", 2048, 1, 1 << 24);      // a prefix of another store shorter than this is read again
    auto ckpt_drop_from = [&](ActiveReq& rq, int pos) {         // free the checkpoints at positions >= pos
        while (!rq.ckpts.empty() && rq.ckpts.back().pos >= pos) {
            backend.ckpt_free(rq.ckpts.back().handle);
            rq.ckpts.pop_back();
        }
        if (rq.turn_mark >= pos) rq.turn_mark = 0;
    };
    auto ckpt_take = [&](ActiveReq& rq, int pos) {
        if (ckpt_max <= 0 || pos <= 0) return;
        if (!rq.ckpts.empty() && rq.ckpts.back().pos >= pos) return;
        if (static_cast<int>(rq.ckpts.size()) >= ckpt_max) {
            // full.  The tail marks - those off the grid (fine marks, prompt ends) and any mark within the fine span of the new one,
            // which is where the next prompt is most likely to part - have half of the places: past that the oldest of them goes.
            // Else the grid mark whose neighbours are closest together goes (the first, nearest the shared system prompt, stays).
            const int fine_keep = std::max(1, ckpt_max / 2);
            auto is_tail = [&](int p) { return !on_grid(p) || p >= pos - ckpt_fine_span; };
            int n_fine = 0;
            size_t oldest_fine = rq.ckpts.size();
            for (size_t i = 0; i < rq.ckpts.size(); ++i) {
                if (!is_tail(rq.ckpts[i].pos)) continue;
                if (n_fine++ == 0) oldest_fine = i;
            }
            size_t drop = rq.ckpts.size();
            if (oldest_fine < rq.ckpts.size() && n_fine >= fine_keep) drop = oldest_fine;
            if (drop == rq.ckpts.size()) {
                int best = INT_MAX;
                for (size_t i = 1; i < rq.ckpts.size(); ++i) {
                    if (is_tail(rq.ckpts[i].pos)) continue;
                    const int next = (i + 1 < rq.ckpts.size()) ? rq.ckpts[i + 1].pos : pos;
                    const int gap = next - rq.ckpts[i - 1].pos;
                    if (gap < best) { best = gap; drop = i; }
                }
            }
            if (drop == rq.ckpts.size()) drop = (oldest_fine < rq.ckpts.size()) ? oldest_fine : rq.ckpts.size() - 1;
            backend.ckpt_free(rq.ckpts[drop].handle);
            rq.ckpts.erase(rq.ckpts.begin() + static_cast<long>(drop));
        }
        const int h = backend.ckpt_save(rq.phys);
        if (h >= 0) rq.ckpts.push_back({pos, h});
    };
    auto turn_mark_take = [&](ActiveReq& rq, int pos) {        // the mark before the prompt's last token: this turn's
        ckpt_take(rq, pos);
        if (!rq.ckpts.empty() && rq.ckpts.back().pos == pos) rq.turn_mark = pos;
    };
    auto slot_forget = [&](ActiveReq& rq) {                    // the slot holds nothing any more
        ckpt_drop_from(rq, 0);
        rq.turn_mark = 0;
        rq.cached_prefix.clear();
        rq.cached_sig.clear();
        rq.prompt_ckpt_prefix.clear();
        rq.shared_prefix.clear();
    };
    // ---- KV stores.  While a request runs, the store's tokens / picture marks / checkpoints live in its ActiveReq (cached_prefix,
    // cached_sig, ckpts); between requests they rest here.
    struct KvStore {
        std::vector<int32_t> prefix;
        std::unordered_map<int, uint64_t> sig;
        std::vector<ActiveReq::Ckpt> ckpts;
        int turn_mark = 0;               // (ActiveReq::turn_mark)
        std::vector<int32_t> pckpt;      // the prompt-end checkpoint's tokens (ActiveReq::prompt_ckpt_prefix)
        std::vector<int32_t> shared;     // the shared-prefix checkpoint's tokens (ActiveReq::shared_prefix)
        bool busy = false;
        uint64_t used = 0;
    };
    std::vector<KvStore> stores(static_cast<size_t>(P));
    uint64_t store_tick = 0;
    auto store_release = [&](ActiveReq& rq) {
        if (rq.phys < 0) return;
        KvStore& st = stores[static_cast<size_t>(rq.phys)];
        st.prefix.swap(rq.cached_prefix);
        st.sig.swap(rq.cached_sig);
        st.ckpts.swap(rq.ckpts);
        st.turn_mark = rq.turn_mark;
        rq.turn_mark = 0;
        st.pckpt.swap(rq.prompt_ckpt_prefix);
        st.shared.swap(rq.shared_prefix);
        rq.prompt_ckpt_prefix.clear();
        rq.shared_prefix.clear();
        rq.cached_prefix.clear();
        rq.cached_sig.clear();
        rq.ckpts.clear();
        st.busy = false;
        st.used = ++store_tick;
        rq.phys = -1;
    };
    // how many leading positions of the prompt a store can serve: all it holds when the prompt continues it, else its last
    // checkpoint at or before the place where they part
    // (ckpt_handle, when asked for: the checkpoint that serves it, and then only a checkpoint counts - a state that can be copied)
    auto held_usable = [&](const std::vector<int32_t>& prefix, const std::unordered_map<int, uint64_t>& sig,
                           const std::vector<ActiveReq::Ckpt>& ckpts, const ActiveReq& rq, int* ckpt_handle) -> int {
        const int held = static_cast<int>(prefix.size());
        if (held == 0) return 0;
        const int lim = std::min(held, rq.prompt_len - 1);
        int agree = 0;
        while (agree < lim && prefix[static_cast<size_t>(agree)] == rq.prompt_ids[static_cast<size_t>(agree)]) ++agree;
        for (const auto& kv : sig) {
            if (kv.first >= agree) continue;
            auto it = rq.prompt_sig.find(kv.first);
            if (it == rq.prompt_sig.end() || it->second != kv.second) agree = kv.first;
        }
        for (const auto& kv : rq.prompt_sig) {
            if (kv.first >= agree) continue;
            auto it = sig.find(kv.first);
            if (it == sig.end() || it->second != kv.second) agree = kv.first;
        }
        if (!ckpt_handle && agree == held) return held;
        int best = 0;
        for (const auto& ck : ckpts) if (ck.pos <= agree) { best = ck.pos; if (ckpt_handle) *ckpt_handle = ck.handle; }
        return best;
    };
    // the prompt-end checkpoint (#6627) serves a prompt without pictures that begins with all of its tokens
    auto pckpt_usable = [&](const std::vector<int32_t>& pk, const ActiveReq& rq) -> int {
        const int plen = static_cast<int>(pk.size());
        if (plen == 0 || plen >= rq.prompt_len || !rq.prompt_sig.empty()) return 0;
        for (int i = 0; i < plen; ++i) if (pk[static_cast<size_t>(i)] != rq.prompt_ids[static_cast<size_t>(i)]) return 0;
        return plen;
    };
    // the shared-prefix checkpoint (pin=N) serves a prompt without pictures that names the same boundary and begins with its tokens
    auto shared_usable = [&](const std::vector<int32_t>& sh, const ActiveReq& rq) -> int {
        if (rq.pin <= 0 || static_cast<int>(sh.size()) != rq.pin || rq.pin >= rq.prompt_len || !rq.prompt_sig.empty()) return 0;
        return std::equal(sh.begin(), sh.end(), rq.prompt_ids.begin()) ? rq.pin : 0;
    };
    auto store_acquire = [&](ActiveReq& rq) {
        // 1. the store that holds this conversation: the prompt continues it, or reaches the mark before the last token of the
        //    prompt it read last (the next turn, a retry).  That mark is not always the newest: a continuation pass (ckpt=0)
        //    leaves grid and fine marks past it, and so does a prompt whose length is itself on the fine grid; the client's
        //    next turn parts from the store before those.  A store that shares less - the same tools and system prompt, an
        //    earlier part of the history - keeps its conversation: this request starts from its marks in another store (4).
        int pick = -1, pick_use = 0;
        for (int i = 0; i < P; ++i) {
            const KvStore& st = stores[static_cast<size_t>(i)];
            if (st.busy || st.prefix.empty()) continue;
            // (a backend without ckpt_save: what its prompt-end or shared-prefix checkpoint serves counts as reaching the newest mark)
            const int held_use = held_usable(st.prefix, st.sig, st.ckpts, rq, nullptr);
            const int use = held_use > 0 ? held_use : std::max(pckpt_usable(st.pckpt, rq), shared_usable(st.shared, rq));
            int reach = held_use == 0 ? 0 : st.ckpts.empty() ? static_cast<int>(st.prefix.size()) : st.ckpts.back().pos;
            // (the turn's mark while the store still has it; a rewind or the thinning of a full store may have taken it)
            for (const auto& ck : st.ckpts) if (held_use > 0 && ck.pos == st.turn_mark) reach = ck.pos;
            if (use > 0 && use >= reach &&
                (use > pick_use || (use == pick_use && pick >= 0 && st.used > stores[static_cast<size_t>(pick)].used))) {
                pick = i;
                pick_use = use;
            }
        }
        // 2. else an empty store, 3. else the one used longest ago (the other conversations keep theirs)
        if (pick < 0) for (int i = 0; i < P; ++i) if (!stores[static_cast<size_t>(i)].busy && stores[static_cast<size_t>(i)].prefix.empty()) { pick = i; break; }
        if (pick < 0) for (int i = 0; i < P; ++i) {
            if (stores[static_cast<size_t>(i)].busy) continue;
            if (pick < 0 || stores[static_cast<size_t>(i)].used < stores[static_cast<size_t>(pick)].used) pick = i;
        }
        if (pick < 0) pick = rq.slot;                     // cannot happen: at most P requests, one store each
        KvStore& st = stores[static_cast<size_t>(pick)];
        st.busy = true;
        rq.phys = pick;
        rq.cached_prefix.swap(st.prefix);
        rq.cached_sig.swap(st.sig);
        rq.ckpts.swap(st.ckpts);
        rq.turn_mark = st.turn_mark;
        st.turn_mark = 0;
        rq.prompt_ckpt_prefix.swap(st.pckpt);
        rq.shared_prefix.swap(st.shared);
        st.pckpt.clear();
        st.shared.clear();
        st.prefix.clear();
        st.sig.clear();
        st.ckpts.clear();
        // 4. the chunked prefix cache across stores: when another store - idle, or running another request - has a checkpoint
        //    deeper into this prompt than this store can serve, start from that one (its KV up to there is copied over)
        const int own = held_usable(rq.cached_prefix, rq.cached_sig, rq.ckpts, rq, nullptr);
        int src_phys = -1, src_pos = 0, src_handle = -1;
        const std::vector<int32_t>* src_prefix = nullptr;
        const std::unordered_map<int, uint64_t>* src_sig = nullptr;
        auto consider = [&](int phys, const std::vector<int32_t>& pf, const std::unordered_map<int, uint64_t>& sg,
                            const std::vector<ActiveReq::Ckpt>& ck) {
            int h = -1;
            const int use = held_usable(pf, sg, ck, rq, &h);
            if (h >= 0 && use > src_pos) { src_phys = phys; src_pos = use; src_handle = h; src_prefix = &pf; src_sig = &sg; }
        };
        for (int i = 0; i < P; ++i) {
            if (i == pick || stores[static_cast<size_t>(i)].busy) continue;
            consider(i, stores[static_cast<size_t>(i)].prefix, stores[static_cast<size_t>(i)].sig, stores[static_cast<size_t>(i)].ckpts);
        }
        for (const auto& q : reqs) {
            if (&q == &rq || !q.active || q.phys < 0 || q.phys == pick) continue;
            consider(q.phys, q.cached_prefix, q.cached_sig, q.ckpts);
        }
        if (src_phys >= 0 && src_pos >= own + fork_min) {
          if (!(backend.kv_copy_prefix(pick, src_phys, src_pos) && backend.ckpt_restore(pick, src_handle))) {
            backend.reset_slot(pick);            // a copy that stopped half way: this store's KV is no longer what its tokens say
            slot_forget(rq);
          } else {
            ckpt_drop_from(rq, 0);
            rq.turn_mark = 0;
            rq.cached_prefix.assign(src_prefix->begin(), src_prefix->begin() + src_pos);
            rq.cached_sig.clear();
            for (const auto& kv : *src_sig) if (kv.first < src_pos) rq.cached_sig[kv.first] = kv.second;
            const int h = backend.ckpt_save(pick);
            if (h >= 0) rq.ckpts.push_back({src_pos, h});
            fprintf(stderr, "[inkling2] prefix cache: %d tokens taken from store %d into store %d\n", src_pos, src_phys, pick);
          }
        }
    };

    // a picture seen before is not encoded again (3 to 6 s each; an agent sends the same screenshot with every later request)
    struct ImgCacheEnt { uint64_t hash; ImageRec meta; std::vector<float> emb; uint64_t used; };
    std::vector<ImgCacheEnt> img_cache;
    uint64_t img_cache_tick = 0;
    const int img_cache_max = ink_env_int("STRATA_INK_IMAGE_CACHE", 16, 0, 1024);

    auto finish_req = [&](ActiveReq& rq, const char* reason) {
        if (std::string(reason) == "nonfinite" && rq.phys >= 0) {
            // the state has taken in a token that gave non-finite logits: nothing of this store may be continued or copied
            backend.reset_slot(rq.phys);
            slot_forget(rq);
        }
        const double dec_ms = (rq.decode_t0 > 0.0) ? (now_s() - rq.decode_t0) * 1000.0 : 0.0;
        if (cache_trace && !rq.s.greedy() && rq.generated > 0)
            fprintf(stderr, "[ink_serve] sampled drafting: generated=%d drafts=%d:%d cooldowns=%d plain=%.1fms drafted_tok=%.1fms len=%d\n",
                    rq.generated, rq.drafts_offered, rq.drafts_accepted, rq.sp_cools, rq.sp_plain_ms, rq.sp_tok_ms, rq.sp_len);
        if (rq.is_solo) {
            // Emit full 9-field DONE line so server.py _parse_done() records drafts_accepted, drafts_offered, reused (#6633, #6627)!
            proto("DONE %d %d %.1f %.1f %s %d %d %d",
                  rq.generated, rq.prompt_len, rq.prompt_ms, dec_ms, reason,
                  rq.drafts_accepted, rq.drafts_offered, rq.reused_tokens);
        } else if (rq.is_ext_gen) {
            proto("DONE %s %d %d %.1f %.1f %s %d %d %d",
                  rq.req_id.c_str(), rq.generated, rq.prompt_len, rq.prompt_ms, dec_ms, reason,
                  rq.drafts_accepted, rq.drafts_offered, rq.reused_tokens);
            req_id_to_slot.erase(rq.req_id);
        } else {
            if (!rq.admitted) {
                proto("DONE %d %d %.1f 0.0 %s %d %d %d",
                      rq.generated, rq.prompt_len, rq.prompt_ms, reason,
                      rq.drafts_accepted, rq.drafts_offered, rq.reused_tokens);
                proto("BADM %d 0", rq.slot);
            } else {
                char dstr[64] = "";
                if (rq.drafts_offered > 0) snprintf(dstr, sizeof(dstr), " drafts=%d:%d", rq.drafts_offered, rq.drafts_accepted);
                proto("BDONE %d %d %s %.1f read=%.1f%s", rq.slot, rq.generated, reason, dec_ms, rq.prompt_ms, dstr);
            }
        }
        rq.active = false;
    };

    auto prepare_req = [&](const ParsedGen& pg, ActiveReq& rq, std::string& err) -> bool {
        for (auto& q : reqs) if (!q.active) store_release(q);     // the stores of the requests that have ended are free again
        rq.active = true;
        rq.is_solo = (pg.cmd == "GEN");
        rq.is_ext_gen = (pg.cmd == "GENERATE");
        rq.req_id = pg.req_id;
        rq.slot = pg.slot;
        rq.max_new = pg.max_new;
        rq.generated = 0;
        rq.s = pg.s;
        rq.rng.seed(pg.s.seed ? pg.s.seed : 20261121ULL);
        rq.cancelled = false;
        rq.admitted = false;
        rq.slot_read = pg.slot_read;
        rq.drafts_offered = 0;
        rq.drafts_accepted = 0;
        rq.sp_len = 3; rq.sp_cool = 0; rq.sp_steps = 0; rq.sp_cools = 0; rq.sp_fail = 0; rq.sp_win_tok = 0; rq.sp_win_ms = 0.0; rq.sp_plain_ms = 0.0; rq.sp_tok_ms = 0.0;
        rq.t_start = now_s();
        rq.prompt_ms = 0.0;
        rq.decode_t0 = 0.0;
        rq.gen_ids.clear();
        rq.img_embeds.clear();
        rq.prompt_img_idx.clear();
        rq.prompt_sig.clear();

        if (!pg.images.empty()) {
            if (!images_ok) {
                err = "image: engine started without --vision Inkling-2-Mini-Vision.strata";
                rq.active = false;
                return false;
            }
            std::vector<ImageRec> recs;
            std::vector<uint64_t> hashes;
            for (const auto& f : pg.images) {
                uint64_t hsh = 0;
                ImgCacheEnt* hit = nullptr;
                if (!ink_hash_file(f, hsh)) hsh = 0;         // (the hash also says which picture a position shows: taken with the cache off too)
                if (img_cache_max > 0 && hsh != 0) {
                    for (auto& e : img_cache) if (e.hash == hsh) { hit = &e; break; }
                }
                if (hit) hit->used = ++img_cache_tick;       // not the one a new picture of this request pushes out
                ImageRec rc;
                if (hit) {
                    rc = hit->meta;                  // the sizes only: the pixels are not needed again
                    rc.file = f;
                } else if (!prepare_picture(f, vprobe.c, rc, err)) { rq.active = false; return false; }
                recs.push_back(std::move(rc));
                hashes.push_back(hsh);
            }
            if (!expand_image_pads(pg.ids, c.image_token_id, recs, rq.prompt_ids, rq.prompt_img_idx, err)) {
                rq.active = false;
                return false;
            }
            // Encode images (try GPU backend first, fallback to CPU ViT)
            for (size_t k = 0; k < recs.size(); ++k) {
                std::vector<float> emb;
                std::string verr;
                ImgCacheEnt* hit = nullptr;
                if (hashes[k] != 0) for (auto& e : img_cache) if (e.hash == hashes[k]) { hit = &e; break; }
                if (hit) {
                    hit->used = ++img_cache_tick;
                    emb = hit->emb;
                } else {
                    if (recs[k].pixel_values.empty()) {      // it was a hit when the request was read, and has been pushed out since
                        ImageRec again;
                        if (!prepare_picture(pg.images[k], vprobe.c, again, err)) { rq.active = false; return false; }
                        recs[k] = std::move(again);
                    }
                    if (!backend.encode_image(recs[k], emb, verr)) {
                        if (!cpu_vit.loaded && !cpu_vit.load(cfg.vision_path, err)) { rq.active = false; return false; }
                        if (!cpu_vit.encode(recs[k], emb, err)) { rq.active = false; return false; }
                    }
                    if (img_cache_max > 0 && hashes[k] != 0) {
                        if (static_cast<int>(img_cache.size()) >= img_cache_max) {
                            size_t old = 0;
                            for (size_t e = 1; e < img_cache.size(); ++e) if (img_cache[e].used < img_cache[old].used) old = e;
                            img_cache.erase(img_cache.begin() + static_cast<long>(old));
                        }
                        ImageRec meta = recs[k];
                        std::vector<float>().swap(meta.pixel_values);
                        img_cache.push_back({hashes[k], std::move(meta), emb, ++img_cache_tick});
                    }
                }
                rq.img_embeds.insert(rq.img_embeds.end(), emb.begin(), emb.end());
                proto("IMG %zu %d %d %d", k, recs[k].grid_h, recs[k].grid_w, recs[k].num_lm_tokens());
            }
            // which picture each pad position shows (prompt_img_idx counts the pads over all pictures of the prompt, in order)
            {
                std::vector<uint64_t> row_hash;
                for (size_t k = 0; k < recs.size(); ++k) {
                    const uint64_t hk = hashes[k] ? hashes[k] : (0xD1B54A32D192ED03ULL + static_cast<uint64_t>(rq.t_start * 1e6) + k);
                    for (int j = 0; j < recs[k].num_lm_tokens(); ++j)
                        row_hash.push_back((hk ^ (static_cast<uint64_t>(j) * 0x9E3779B97F4A7C15ULL)) | 1ULL);
                }
                for (size_t i = 0; i < rq.prompt_img_idx.size(); ++i) {
                    const int r = rq.prompt_img_idx[i];
                    if (r >= 0) rq.prompt_sig[static_cast<int>(i)] = (static_cast<size_t>(r) < row_hash.size()) ? row_hash[static_cast<size_t>(r)] : 1ULL;
                }
            }
        } else {
            rq.prompt_ids = pg.ids;
            rq.prompt_img_idx.assign(rq.prompt_ids.size(), -1);
        }

        rq.prompt_len = static_cast<int>(rq.prompt_ids.size());
        if (rq.prompt_len <= 0 || rq.prompt_len >= eff_max_ctx) {
            err = "prompt length out of range (1.." + std::to_string(eff_max_ctx - 1) + ")";
            rq.active = false;
            return false;
        }
        if (pg.images.empty() && pg.pin >= rq.prompt_len) {
            err = "bad pin boundary";
            rq.active = false;
            return false;
        }
        rq.pin = pg.images.empty() ? pg.pin : 0;
        rq.no_ckpt = pg.no_ckpt;
        store_acquire(rq);
        // Prefix reuse: how far this prompt agrees with what the store holds (tokens, and the same picture at every picture position)
        int reuse = 0;
        if (!rq.cached_prefix.empty()) {
            const int held = static_cast<int>(rq.cached_prefix.size());
            const int lim = std::min(held, rq.prompt_len - 1);
            int agree = 0;
            while (agree < lim && rq.cached_prefix[agree] == rq.prompt_ids[agree]) ++agree;
            for (const auto& kv : rq.cached_sig) {
                if (kv.first >= agree) continue;
                auto it = rq.prompt_sig.find(kv.first);
                if (it == rq.prompt_sig.end() || it->second != kv.second) agree = kv.first;
            }
            for (const auto& kv : rq.prompt_sig) {
                if (kv.first >= agree) continue;
                auto it = rq.cached_sig.find(kv.first);
                if (it == rq.cached_sig.end() || it->second != kv.second) agree = kv.first;
            }
            if (agree == held) {
                reuse = held;                    // the prompt continues the slot: nothing to restore
            } else {
                // the slot went on past where the prompt leaves it (SWA/ShortConv state): back to the nearest checkpoint before that
                int pick = -1;
                for (size_t i = 0; i < rq.ckpts.size(); ++i) if (rq.ckpts[i].pos <= agree) pick = static_cast<int>(i);
                if (pick >= 0 && backend.ckpt_restore(rq.phys, rq.ckpts[static_cast<size_t>(pick)].handle)) {
                    reuse = rq.ckpts[static_cast<size_t>(pick)].pos;
                    ckpt_drop_from(rq, reuse + 1);
                    rq.cached_prefix.resize(static_cast<size_t>(reuse));
                    for (auto it = rq.cached_sig.begin(); it != rq.cached_sig.end();) {
                        if (it->first >= reuse) it = rq.cached_sig.erase(it); else ++it;
                    }
                }
            }
        }
        // A backend without ckpt_save (ckpt_max == 0) may still have the prompt-end checkpoint (#6627): the prompt this store
        // read last, up to its last token.  It serves the next turn and the same prompt sent again.  And the shared-prefix
        // checkpoint (pin=N): the first N tokens, kept apart for a new conversation that starts the same way.  The deeper one first.
        if (reuse == 0 && ckpt_max == 0) {
            const int plen = pckpt_usable(rq.prompt_ckpt_prefix, rq);
            const int shared = shared_usable(rq.shared_prefix, rq);
            if (plen > 0 && plen >= shared && backend.restore_prompt_checkpoint(rq.phys, plen)) {
                reuse = plen;
                rq.cached_prefix = rq.prompt_ckpt_prefix;
                rq.cached_sig.clear();
            } else if (shared > 0 && backend.restore_shared_checkpoint(rq.phys, shared)) {
                reuse = shared;
                rq.cached_prefix = rq.shared_prefix;
                rq.cached_sig.clear();
                rq.prompt_ckpt_prefix.clear();       // the next branch overwrites the global KV after the shared boundary
            } else if (plen > 0 && plen < shared && backend.restore_prompt_checkpoint(rq.phys, plen)) {
                reuse = plen;
                rq.cached_prefix = rq.prompt_ckpt_prefix;
                rq.cached_sig.clear();
            }
        }
        if (cache_trace) {
            std::string cks;
            for (const auto& ck : rq.ckpts) cks += " " + std::to_string(ck.pos);
            fprintf(stderr, "[inkling2] cache: slot %d store %d prompt %d reuse %d held %zu marks:%s\n", rq.slot, rq.phys, rq.prompt_len, reuse,
                    rq.cached_prefix.size(), cks.c_str());
        }
        if (reuse == 0) {
            backend.reset_slot(rq.phys);
            slot_forget(rq);
        } else {
            proto("RESUME %d", reuse);
            if (reuse == rq.prompt_len - 1 && !rq.no_ckpt) turn_mark_take(rq, reuse);   // no read stops here any more: the mark for this prompt sent again
        }
        if (reuse < static_cast<int>(rq.shared_prefix.size())) rq.shared_prefix.clear();
        rq.shared_pending = ckpt_max == 0 && rq.pin > reuse && static_cast<int>(rq.shared_prefix.size()) != rq.pin;
        rq.reused_tokens = reuse;
        // ckpt=0 (a later pass of one request: the thinking budget's wrap-up, a reply written again): the checkpoint of the
        // request's first pass, at the end of the prompt the client sends back, stays; this pass's longer prompt would replace it.
        rq.ckpt_pending = (ckpt_max == 0 && pg.images.empty() && !pg.no_ckpt && rq.prompt_len >= 16 && reuse < rq.prompt_len - 1);
        rq.prefill_pos = reuse;
        rq.cur_pos = reuse;
        rq.hist = rq.prompt_ids;
        rq.next_tok = -1;
        rq.gov_plain_left = 0; rq.gov_win_steps = 0; rq.gov_win_emit = 0; rq.gov_gain = -1.0;
        rq.gov_calib_left = (spec_adapt && gov_t_plain <= 0.0) ? 4 : 0;
        return true;
    };

    bool quit_requested = false;
    auto handle_line = [&](const std::string& line) -> bool {
        if (line.empty()) return true;
        if (line == "QUIT") { quit_requested = true; return false; }
        if (line == "HELLO") {
            proto("HELLO ok engine=strata-inkling-%s arch=inkling2_loop backend=%s max_ctx=%d parallel=%d",
                  cfg.version.c_str(), backend.backend_name(), eff_max_ctx, P);
            return true;
        }
        if (line == "STATS") {
            auto st = backend.stats();
            proto("STATS active_slots=%d max_slots=%d vram_experts=%zu ram_experts=%zu total_experts=%zu kv=%s expert_selections=%llu expert_misses=%llu",
                  st.active_slots, st.max_slots, st.vram_experts_loaded, st.ram_experts_loaded, st.total_experts,
                  st.kv_dtype == KvDtype::FP8_E4M3 ? "fp8" : "fp16",
                  (unsigned long long)st.expert_selections, (unsigned long long)st.expert_misses);
            return true;
        }
        if (line == "STOP") {
            for (auto& rq : reqs) if (rq.active && rq.is_solo) rq.cancelled = true;
            return true;
        }
        if (line.rfind("BSTOP ", 0) == 0) {
            int s = std::atoi(line.c_str() + 6);
            if (s >= 0 && s < P && reqs[s].active) reqs[s].cancelled = true;
            return true;
        }
        if (line.rfind("CANCEL ", 0) == 0) {
            std::string rid = line.substr(7);
            auto it = req_id_to_slot.find(rid);
            if (it != req_id_to_slot.end() && reqs[it->second].active) reqs[it->second].cancelled = true;
            return true;
        }
        if (line.rfind("BYIELD ", 0) == 0) {
            // Interleaved continuous batching already prevents long-prompt head-of-line blocking!
            return true;
        }
        if (line.rfind("BCO ", 0) == 0) {
            // BCO <slot> <ids>: pre-register coalesced prompt read into free slot
            size_t sp = line.find(' ', 4);
            if (sp != std::string::npos) {
                int s = std::atoi(line.substr(4, sp - 4).c_str());
                std::vector<int32_t> cids;
                std::string cerr;
                if (s >= 0 && s < P && !reqs[s].active && parse_id_list(line.substr(sp + 1), cids, cerr)) {
                    // (nothing is read here, and the store a request runs in is chosen when it comes: no store is reset for it)
                    proto("BCOK %d %zu", s, cids.size());
                }
            }
            return true;
        }
        if (line.rfind("GEN ", 0) == 0 || line.rfind("BGEN ", 0) == 0 || line.rfind("GENERATE ", 0) == 0) {
            ParsedGen pg;
            std::string perr;
            if (!parse_gen_cmd(line, pg, perr)) {
                proto("ERR %s", perr.c_str());
                return true;
            }
            if (pg.cmd == "GENERATE") {
                int free_s = -1;
                for (int s = 0; s < P; ++s) if (!reqs[s].active) { free_s = s; break; }
                if (free_s < 0) {
                    proto("ERR busy: all %d parallel slots active", P);
                    return true;
                }
                pg.slot = free_s;
            }
            if (pg.slot < 0 || pg.slot >= P) {
                proto("ERR BGEN: slot %d out of range 0..%d", pg.slot, P - 1);
                return true;
            }
            if (reqs[pg.slot].active) {
                proto("ERR BGEN: slot %d is busy", pg.slot);
                return true;
            }
            if (!prepare_req(pg, reqs[pg.slot], perr)) {
                proto("ERR %s", perr.c_str());
                return true;
            }
            if (pg.cmd == "GENERATE") {
                req_id_to_slot[pg.req_id] = pg.slot;
            } else if (pg.cmd == "BGEN" && pg.slot_read && pg.max_new > 0) {
                // Zero-HoL admission: immediately release server.py control lines so concurrent
                // HTTP requests 1..64 all prefill and decode together in the coalesced StepBatch!
                proto("DONE 0 %d 0.0 0.0 admit 0 0 %d", reqs[pg.slot].prompt_len, reqs[pg.slot].reused_tokens);
                proto("BADM %d 1", pg.slot);
                reqs[pg.slot].admitted = true;
            }
            return true;
        }
        proto("ERR unknown command");
        return true;
    };

    // Main continuous batching event loop!
    StepBatch batch;
    std::vector<float> merged_img_embeds;
    std::vector<int> logit_slots;
    std::vector<float> logits_buf;

    while (true) {
        // 1. Drain all queued stdin commands without blocking while any slot is active
        bool any_active = false;
        for (const auto& rq : reqs) if (rq.active) { any_active = true; break; }

        std::string line;
        if (!any_active) {
            if (quit_requested) break;
            if (!in.wait_next(line)) break;
            handle_line(line);
            while (!quit_requested && in.poll(line)) {
                handle_line(line);
            }
        } else {
            while (!quit_requested && in.poll(line)) {
                handle_line(line);
            }
        }

        // Handle cancellations immediately
        for (auto& rq : reqs) {
            if (rq.active && rq.cancelled) finish_req(rq, "cancel");
        }

        // Count active slots
        int n_active = 0;
        int sole_slot = -1;
        for (int s = 0; s < P; ++s) {
            if (reqs[s].active) { ++n_active; sole_slot = s; }
        }
        if (n_active == 0) {
            if (quit_requested || in.is_eof()) break;
            continue;
        }

        // 2. Fast-path: Single-stream DSpark Adaptive Beam Tree / Block-8 speculative step (when decoding)
        // Under sampling the drafted tokens are checked against a token sampled from the model's own distribution at each
        // position (accepted while they are equal), so the output distribution is the model's.  It pays only where the
        // model is confident, so a request first times plain steps, then drafts while a drafted token costs less.
        // (STRATA_INK_SPEC=adapt / --draft-adapt: no drafting either while the governor runs its plain steps)
        bool spec_now = use_dspark && n_active == 1 && reqs[sole_slot].prefill_pos >= reqs[sole_slot].prompt_len &&
            reqs[sole_slot].generated + 2 <= reqs[sole_slot].max_new && reqs[sole_slot].cur_pos + 8 < eff_max_ctx &&
            !(spec_adapt && (reqs[sole_slot].gov_plain_left > 0 || reqs[sole_slot].gov_calib_left > 0));
        if (spec_now && !reqs[sole_slot].s.greedy()) {
            auto& rq = reqs[sole_slot];
            if (!sample_draft || rq.s.hold_budget > 0 || rq.sp_plain_ms <= 0.0) spec_now = false;
            else if (rq.sp_cool > 0) { --rq.sp_cool; spec_now = false; }
        }
        if (spec_now) {
            auto& rq = reqs[sole_slot];
            const bool sampled = !rq.s.greedy();
            const double t_spec0 = now_s();
            // 2a. Zero-D2H fused greedy Beam Tree speculative step (when no penalties and headroom >= 12)
            if (spec_tree && !sampled && !rq.s.penalties() && rq.cur_pos + 12 < eff_max_ctx) {
                int32_t emitted[16]{};
                int drafted = 0;
                std::string serr;
                const int max_emit = rq.max_new - rq.generated;
                const double gov_t0 = now_s();
                const int n_emit = backend.dspark_spec_step_greedy(
                    rq.phys, rq.next_tok, rq.cur_pos, max_emit, emitted, &drafted, serr);
                if (n_emit < 0) {
                    proto("ERR fatal GPU error: %s", serr.c_str());
                    return 1;
                }
                if (n_emit > 0) {
                    rq.drafts_offered += drafted;
                    if (spec_adapt) gov_step(rq, n_emit, (now_s() - gov_t0) * 1000.0);
                    int kept = 0;
                    bool done = false;
                    const char* fin = nullptr;
                    for (int i = 0; i < n_emit; ++i) {
                        const int32_t tok = emitted[i];
                        rq.hist.push_back(tok);
                        rq.gen_ids.push_back(tok);
                        rq.cached_prefix.push_back(i == 0 ? rq.next_tok : emitted[i - 1]);
                        ++rq.generated;
                        ++kept;
                        if (i + 1 < n_emit) ++rq.drafts_accepted;
                        if (rq.is_solo) proto("T %d", tok);
                        else if (rq.is_ext_gen) proto("T %s %d", rq.req_id.c_str(), tok);
                        else proto("BT %d %d", rq.slot, tok);

                        if (is_stop_tok(tok)) { fin = "stop"; done = true; break; }
                        if (rq.generated >= rq.max_new) { fin = "length"; done = true; break; }
                        if (cfg.anti_loop && detect_repetition_loop(rq.gen_ids)) { fin = "stop"; done = true; break; }
                    }
                    if (!done) {
                        rq.next_tok = emitted[n_emit - 1];
                        rq.cur_pos += n_emit;
                    } else {
                        if (kept < n_emit) {
                            backend.rollback_slot(rq.phys, rq.cur_pos + kept);
                        }
                        rq.cur_pos += kept;
                        if (fin) finish_req(rq, fin);
                    }
                    continue;
                }
            }
            int32_t draft_ids[8]{};
            std::string derr;
            const int max_d = std::min(sampled ? rq.sp_len : 7, rq.max_new - rq.generated - 1);
            const double gov_b0 = now_s();
            int nd = backend.dspark_draft(rq.phys, rq.next_tok, rq.cur_pos, max_d, draft_ids, derr);
            if (nd > 0) {
                backend.snapshot_slot(rq.phys);
                batch.clear();
                batch.capture_taps = true;
                batch.push(rq.phys, rq.cur_pos, rq.next_tok, true);
                for (int i = 0; i < nd; ++i) {
                    batch.push(rq.phys, rq.cur_pos + 1 + i, draft_ids[i], true);
                }
                logits_buf.resize(static_cast<size_t>(nd + 1) * V);
                std::string ferr;
                if (!backend.forward_batch(batch, logits_buf.data(), nullptr, ferr)) {
                    proto("ERR fatal GPU error: %s", ferr.c_str());
                    return 1;
                }
                rq.drafts_offered += nd;
                int accepted = 0;
                bool done = false;
                const char* fin = nullptr;
                for (int i = 0; i <= nd; ++i) {
                    float* lg = logits_buf.data() + static_cast<size_t>(i) * V;
                    if (!finite_logits(lg, V)) {
                        finish_req(rq, "nonfinite");
                        done = true;
                        break;
                    }
                    bool draft_ok = false;
                    int32_t tok = (i < nd)
                        ? verify_draft_logits(lg, V, rq.s, rq.hist, rq.rng, draft_ids[i], draft_ok)
                        : sample_logits(lg, V, rq.s, rq.hist, rq.rng);
                    rq.hist.push_back(tok);
                    rq.gen_ids.push_back(tok);
                    rq.cached_prefix.push_back(i == 0 ? rq.next_tok : draft_ids[i - 1]);
                    ++rq.generated;
                    if (rq.is_solo) proto("T %d", tok);
                    else if (rq.is_ext_gen) proto("T %s %d", rq.req_id.c_str(), tok);
                    else proto("BT %d %d", rq.slot, tok);

                    if (is_stop_tok(tok)) { fin = "stop"; done = true; ++accepted; break; }
                    if (rq.generated >= rq.max_new) { fin = "length"; done = true; ++accepted; break; }
                    if (cfg.anti_loop && detect_repetition_loop(rq.gen_ids)) { fin = "stop"; done = true; ++accepted; break; }
                    if (i < nd && draft_ok) {
                        ++accepted;
                        ++rq.drafts_accepted;
                    } else {
                        rq.next_tok = tok;
                        ++accepted;
                        break;
                    }
                }
                if (!done) {
                    if (accepted <= nd) {
                        // Rolled back to anchor + accepted tokens
                        backend.rollback_slot(rq.phys, rq.cur_pos + accepted);
                    }
                    rq.cur_pos += accepted;
                    if (spec_adapt) gov_step(rq, accepted, (now_s() - gov_b0) * 1000.0);
                    if (sampled) {
                        // accepted = tokens this step produced.  Draft longer after a full accept, shorter after a miss,
                        // and go back to plain steps for a while when a drafted token costs more than a plain one.
                        const double spec_ms = (now_s() - t_spec0) * 1000.0;
                        rq.sp_win_ms += spec_ms; rq.sp_win_tok += accepted; ++rq.sp_steps;
                        rq.sp_len = (accepted > nd) ? std::min(7, nd + 1) : std::max(1, std::min(7, accepted));
                        if (rq.sp_steps >= 6) {      // judged over a window: one missed draft says little
                            rq.sp_tok_ms = rq.sp_win_ms / std::max(1, rq.sp_win_tok);
                            if (rq.sp_tok_ms > rq.sp_plain_ms) {
                                rq.sp_cool = std::min(48, 8 << std::min(3, rq.sp_fail)); ++rq.sp_fail; ++rq.sp_cools; rq.sp_len = 2;
                            } else {
                                rq.sp_fail = 0;
                            }
                            rq.sp_steps = 0; rq.sp_win_ms = 0.0; rq.sp_win_tok = 0;
                        }
                    }
                } else if (fin) {
                    // the positions past the last kept token hold drafted tokens that were never accepted
                    if (accepted <= nd) backend.rollback_slot(rq.phys, rq.cur_pos + accepted);
                    rq.cur_pos += accepted;
                    finish_req(rq, fin);
                }
                continue;
            }
        }

        // 3. Continuous Batched 48-Visit Step across ALL active slots (B = 1..64)!
        // Coalesces chunked-prefill tokens + decode tokens from every active request into ONE StepBatch!
        batch.clear();
        batch.capture_taps = use_dspark;
        merged_img_embeds.clear();
        logit_slots.clear();
        int token_budget = std::max(chunk_sz, n_active);
        std::vector<int> prefill_fed(P, 0);
        double t_step0 = now_s();

        // First schedule 1 decode token for every decoding slot (zero head-of-line blocking!)
        for (int s = 0; s < P; ++s) {
            auto& rq = reqs[s];
            if (!rq.active || rq.prefill_pos < rq.prompt_len) continue;
            if (rq.cur_pos >= eff_max_ctx) {
                finish_req(rq, "length");
                continue;
            }
            batch.push(rq.phys, rq.cur_pos, rq.next_tok, true);
            logit_slots.push_back(s);
            --token_budget;
        }
        // While some slot decodes, a prompt is read at most `prefill_cap` tokens per step: one step then costs the decode tokens plus
        // that chunk, so the decoding slots keep their cadence while a long prompt is read (a 2048-token chunk was ~0.5 s per decode token).
        if (prefill_cap > 0 && !logit_slots.empty()) token_budget = std::min(token_budget, prefill_cap);

        // Then fill remaining budget with chunked-prefill tokens across prefilling slots (round-robin!)
        if (token_budget > 0) {
            std::vector<int> pf_slots;
            for (int s = 0; s < P; ++s) if (reqs[s].active && reqs[s].prefill_pos < reqs[s].prompt_len) pf_slots.push_back(s);
            if (!pf_slots.empty()) {
                const int per_slot = std::max(1, token_budget / static_cast<int>(pf_slots.size()));
                for (int s : pf_slots) {
                    auto& rq = reqs[s];
                    const int rem = rq.prompt_len - rq.prefill_pos;
                    int take = std::min(rem, per_slot);
                    if (ckpt_max > 0) {
                        // the prompt's last token is read in a step of its own: the checkpoint taken before it serves the next turn
                        // and also this same prompt sent again (a retry, a regenerate), which has to read at least one token
                        if (rem > 1 && take == rem) take = rem - 1;
                        // ... and a read stops at every multiple of the chunk, where a checkpoint is taken
                        if (ckpt_every > 0) {
                            const int next_mark = (rq.prefill_pos / ckpt_every + 1) * ckpt_every;
                            if (rq.prefill_pos + take > next_mark) take = next_mark - rq.prefill_pos;
                        }
                        // ... and, near the prompt's end, at every multiple of the fine step
                        if (ckpt_fine > 0) {
                            const int from = std::max(0, rq.prompt_len - 1 - ckpt_fine_span);
                            int next_fine = (rq.prefill_pos / ckpt_fine + 1) * ckpt_fine;
                            if (next_fine < from) next_fine = (from + ckpt_fine - 1) / ckpt_fine * ckpt_fine;
                            if (rq.prefill_pos + take > next_fine) take = next_fine - rq.prefill_pos;
                        }
                    } else if (rq.ckpt_pending && rq.prefill_pos < rq.prompt_len - 1 && rq.prefill_pos + take >= rq.prompt_len) {
                        // #6627: Pause 1 token before prompt_len so we save the exact prompt-end checkpoint at prompt_len - 1!
                        take = (rq.prompt_len - 1) - rq.prefill_pos;
                    }
                    if (rq.shared_pending && rq.prefill_pos < rq.pin) {
                        take = std::min(take, rq.pin - rq.prefill_pos);
                    }
                    const int img_base = static_cast<int>(merged_img_embeds.size()) / c.hidden;
                    if (!rq.img_embeds.empty()) {
                        merged_img_embeds.insert(merged_img_embeds.end(), rq.img_embeds.begin(), rq.img_embeds.end());
                    }
                    for (int k = 0; k < take; ++k) {
                        const int p_i = rq.prefill_pos + k;
                        const bool is_last = (p_i == rq.prompt_len - 1);
                        int img_idx = (rq.prompt_img_idx[p_i] >= 0) ? (img_base + rq.prompt_img_idx[p_i]) : -1;
                        batch.push(rq.phys, p_i, rq.prompt_ids[p_i], is_last, img_idx);
                        if (is_last) logit_slots.push_back(s);
                    }
                    prefill_fed[s] = take;
                }
            }
        }
        if (!merged_img_embeds.empty()) batch.img_embeds = merged_img_embeds.data();

        const int n_want = batch.count_logits();
        // Candidate rows (the 128 best logits of each wanted row, exact) instead of the full 200058-float rows, when the backend can and
        // no wanted slot needs its whole row (a penalty changes logits anywhere in it).  Many slots: 64 rows are 51 MB per step otherwise.
        bool use_cand = cand_ok && n_want > 0 && backend.supports_candidates();
        for (int wi = 0; use_cand && wi < n_want; ++wi) if (reqs[logit_slots[wi]].s.penalties()) use_cand = false;
        std::string ferr;
        CandRows cand;
        bool fwd_ok = true;
        if (use_cand) {
            fwd_ok = backend.forward_batch_cand(batch, cand, ferr);
            if (!fwd_ok && cand.rows == 0) {   // nothing ran (no room for the candidate scratch): this and every later step takes the full-row path
                cand_ok = false;
                use_cand = false;
                fwd_ok = true;
            }
        }
        if (!use_cand) {
            logits_buf.resize(static_cast<size_t>(std::max(1, n_want)) * V);
            fwd_ok = backend.forward_batch(batch, n_want > 0 ? logits_buf.data() : nullptr, nullptr, ferr);
        }
        if (!fwd_ok) {
            // The global K/V could not grow (no room in the card, none to spill to): the engine is fine. End the requests whose positions
            // reach past what the K/V holds (reason "length": the context this card can serve), keep every other request, and go on (#6905).
            if (backend.last_failure_is_kv_growth()) {
                const int cap = backend.kv_capacity_tokens();
                bool ended = false;
                for (int i = 0; i < batch.n_tokens; ++i) {
                    int s = -1;       // the batch names the store (phys); the request that runs in it
                    for (int q = 0; q < P; ++q) if (reqs[q].active && reqs[q].phys == batch.slot_ids[i]) { s = q; break; }
                    if (batch.pos[i] >= cap && s >= 0) {
                        std::fprintf(stderr, "[ink-serve] slot %d ended at %d tokens: %s\n", s, batch.pos[i], ferr.c_str());
                        finish_req(reqs[s], "length");
                        ended = true;
                    }
                }
                if (ended) continue;
            }
            proto("ERR fatal GPU error: %s", ferr.c_str());
            return 1;
        }
        const double step_ms = (now_s() - t_step0) * 1000.0;
        if (n_active == 1 && batch.n_tokens == 1 && !reqs[sole_slot].s.greedy()) {
            auto& rq = reqs[sole_slot];
            rq.sp_plain_ms = rq.sp_plain_ms <= 0.0 ? step_ms : 0.8 * rq.sp_plain_ms + 0.2 * step_ms;
        }

        // Update prefill progress & emit PP lines (+ save prompt-end checkpoint at prompt_len - 1 for #6627!)
        for (int s = 0; s < P; ++s) {
            if (prefill_fed[s] > 0) {
                auto& rq = reqs[s];
                for (int k = 0; k < prefill_fed[s]; ++k) {
                    rq.cached_prefix.push_back(rq.prompt_ids[rq.prefill_pos + k]);
                    if (!rq.prompt_sig.empty()) {
                        auto it = rq.prompt_sig.find(rq.prefill_pos + k);
                        if (it != rq.prompt_sig.end()) rq.cached_sig[rq.prefill_pos + k] = it->second;
                    }
                }
                rq.prefill_pos += prefill_fed[s];
                rq.cur_pos = rq.prefill_pos;
                rq.prompt_ms += step_ms;
                if (rq.shared_pending && rq.prefill_pos == rq.pin) {
                    rq.shared_pending = false;
                    rq.shared_prefix.clear();
                    if (backend.save_shared_checkpoint(rq.phys, rq.pin)) {
                        rq.shared_prefix.assign(rq.prompt_ids.begin(), rq.prompt_ids.begin() + rq.pin);
                    }
                }
                // a checkpoint where the prompt ends (before anything is generated), and every ckpt_every tokens on the way there
                // (ckpt=0, a continuation pass: no mark at this prompt's end, the marks of the request's first pass stay the newest)
                if (ckpt_max > 0) {
                    if (rq.prefill_pos == rq.prompt_len - 1 && !rq.no_ckpt) turn_mark_take(rq, rq.prefill_pos);
                    else if (on_grid(rq.prefill_pos) || fine_mark(rq, rq.prefill_pos)) ckpt_take(rq, rq.prefill_pos);
                } else if (rq.ckpt_pending && rq.prefill_pos == rq.prompt_len - 1) {
                    rq.ckpt_pending = false;
                    if (backend.save_prompt_checkpoint(rq.phys, rq.prefill_pos)) {
                        rq.prompt_ckpt_prefix.assign(rq.prompt_ids.begin(), rq.prompt_ids.begin() + rq.prefill_pos);
                    } else {
                        rq.prompt_ckpt_prefix.clear();
                    }
                }
                if (rq.is_solo || !rq.admitted) {
                    const double ppts = step_ms > 0.0 ? (prefill_fed[s] * 1000.0 / step_ms) : 0.0;
                    proto("PP %d %d %.1f %.1f", rq.prefill_pos, rq.prompt_len, rq.prompt_ms, ppts);
                }
            }
        }

        struct SampleProf {
            bool on; double t0, fwd_ms; int rows;
            ~SampleProf() {
                if (!on || rows == 0) return;
                static double s_fwd = 0, s_smp = 0; static long s_n = 0, s_rows = 0;
                s_fwd += fwd_ms; s_smp += (now_s() - t0) * 1000.0; ++s_n; s_rows += rows;
                if (s_n % 64 == 0) fprintf(stderr, "[ink_serve PROF] %ld steps, %.1f logit rows avg: forward_batch %.2f ms, sampling+protocol %.2f ms per step\n",
                                           s_n, (double)s_rows / s_n, s_fwd / s_n, s_smp / s_n);
            }
        } smp_prof{std::getenv("STRATA_INK_PROF") != nullptr, now_s(), step_ms, n_want};

        // Sample next token for every slot that produced logits in this batch!
        for (int wi = 0; wi < n_want; ++wi) {
            const int s = logit_slots[wi];
            auto& rq = reqs[s];
            float* lg = use_cand ? nullptr : logits_buf.data() + static_cast<size_t>(wi) * V;
            if (use_cand ? cand.finite[static_cast<size_t>(wi)] == 0 : !finite_logits(lg, V)) {
                // Purge poisoned KV/ShortConv state so subsequent retries start clean (#879)!
                finish_req(rq, "nonfinite");
                continue;
            }
            const bool just_finished_prefill = (prefill_fed[s] > 0);
            if (!just_finished_prefill) {
                rq.cached_prefix.push_back(rq.next_tok);
                ++rq.cur_pos;
            }
            if (rq.max_new == 0) {
                finish_req(rq, "length");
                continue;
            }
            int32_t tok = use_cand
                ? sample_candidates(cand.val.data() + static_cast<size_t>(wi) * InkBackend::kInkCandidates,
                                    cand.idx.data() + static_cast<size_t>(wi) * InkBackend::kInkCandidates, V, rq.s, rq.rng)
                : sample_logits(lg, V, rq.s, rq.hist, rq.rng);
            rq.hist.push_back(tok);
            rq.gen_ids.push_back(tok);
            rq.next_tok = tok;
            ++rq.generated;

            const bool stop = is_stop_tok(tok);
            const bool hit_len = (rq.generated >= rq.max_new) || (rq.cur_pos + 1 >= eff_max_ctx);
            const bool hit_hold = (rq.s.hold_budget > 0 && rq.generated >= rq.s.hold_budget && tok != rq.s.hold_token_id);
            const bool hit_loop = cfg.anti_loop && detect_repetition_loop(rq.gen_ids);

            if (rq.is_solo) {
                if (just_finished_prefill) rq.decode_t0 = now_s();
                proto("T %d", tok);
                if (stop) finish_req(rq, "stop");
                else if (hit_hold) finish_req(rq, "hold");
                else if (hit_len) finish_req(rq, "length");
                else if (hit_loop) finish_req(rq, "stop");
            } else if (rq.is_ext_gen) {
                if (just_finished_prefill) rq.decode_t0 = now_s();
                proto("T %s %d", rq.req_id.c_str(), tok);
                if (stop || hit_loop) finish_req(rq, "stop");
                else if (hit_len) finish_req(rq, "length");
            } else {
                // Native Strata BGEN slot protocol: first token goes on control lines (T + DONE + BADM <slot> <cont>),
                // subsequent tokens stream asynchronously on slot lines (BT <slot> <id> ... BDONE <slot> ...)
                if (!rq.admitted) {
                    rq.decode_t0 = now_s();
                    const bool cont = !(stop || hit_len || hit_hold || hit_loop);
                    if (!rq.slot_read) proto("T %d", tok);
                    const char* fin = stop ? "stop" : hit_hold ? "hold" : hit_len ? "length" : hit_loop ? "stop" : "admit";
                    proto("DONE %d %d %.1f 0.0 %s 0 0 %d", rq.slot_read ? 0 : 1, rq.prompt_len, rq.prompt_ms, fin, rq.reused_tokens);
                    proto("BADM %d %d", rq.slot, cont ? 1 : 0);
                    if (rq.slot_read) proto("BT %d %d", rq.slot, tok);
                    rq.admitted = true;
                    if (!cont) {
                        if (rq.slot_read) proto("BDONE %d %d %s 0.0 read=%.1f", rq.slot, rq.generated, fin, rq.prompt_ms);
                        rq.active = false;
                    }
                } else {
                    if (just_finished_prefill) rq.decode_t0 = now_s();
                    proto("BT %d %d", rq.slot, tok);
                    if (stop) finish_req(rq, "stop");
                    else if (hit_hold) finish_req(rq, "hold");
                    else if (hit_len) finish_req(rq, "length");
                    else if (hit_loop) finish_req(rq, "stop");
                }
            }
        }
        if (spec_adapt && use_dspark && n_active == 1 && n_want == 1 && prefill_fed[sole_slot] == 0) {
            // one plain decode step: its whole cost, forward + host sampling + output
            const double ms = (now_s() - t_step0) * 1000.0;
            auto& rg = reqs[sole_slot];
            if (rg.gov_calib_left > 0) {
                --rg.gov_calib_left;
                if (rg.gov_calib_left < 3) gov_t_plain = (gov_t_plain <= 0.0) ? ms : std::min(gov_t_plain, ms);  // the first step may capture the graph
            } else {
                if (rg.gov_plain_left > 0) --rg.gov_plain_left;
                gov_t_plain = (gov_t_plain <= 0.0) ? ms : 0.9 * gov_t_plain + 0.1 * ms;
            }
        }
    }
    return 0;
}

}  // namespace ink2
