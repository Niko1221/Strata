// dsv4/serve_loop.cpp - the resident loop: see dsv4/serve_loop.hpp.
//
// stdout is the protocol and NOTHING ELSE: every human line of this build goes to stderr, because
// Strata's serve/server.py reads stdout line by line (READY / INFO / T / PP / DONE / ERR).
//
// stdin is read with raw read(2) through our own buffer rather than std::cin: that keeps a single owner of
// the pipe and lets a STOP be noticed BETWEEN TOKENS (select() with a zero timeout), which a buffered
// std::cin cannot promise.
#include "dsv4/serve_loop.hpp"

#include "dsv4/gpu.hpp"
#include "dsv4/sampler.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <sys/select.h>
#include <unistd.h>

namespace dsv4 {

namespace {

/// Line reader over fd 0 (a pipe from serve/server.py).
class LineReader {
public:
    /// Blocks for the next command line. "" on EOF.
    bool next(std::string& out) {
        for (;;) {
            if (take(out)) return true;
            if (!fill(true)) return false;
        }
    }
    /// Non-blocking: pulls one complete line if it is already there (used for STOP mid-request).
    bool try_next(std::string& out) {
        for (;;) {
            if (take(out)) return true;
            if (!fill(false)) return false;
        }
    }

private:
    bool take(std::string& out) {
        const size_t nl = buf_.find('\n');
        if (nl == std::string::npos) return false;
        out = buf_.substr(0, nl);
        buf_.erase(0, nl + 1);
        while (!out.empty() && out.back() == '\r') out.pop_back();
        return true;
    }
    /// `block`: wait for data. False: poll only. False also on EOF or error.
    bool fill(bool block) {
        if (!block) {
            fd_set rs;
            FD_ZERO(&rs);
            FD_SET(0, &rs);
            timeval tv{0, 0};
            if (select(1, &rs, nullptr, nullptr, &tv) <= 0) return false;
        }
        char tmp[4096];
        const ssize_t n = ::read(0, tmp, sizeof tmp);
        if (n <= 0) { eof_ = true; return false; }
        buf_.append(tmp, (size_t) n);
        return true;
    }
    std::string buf_;
    bool eof_ = false;
};

bool starts(const std::string& s, const char* p) {
    const size_t n = std::strlen(p);
    return s.size() >= n && std::memcmp(s.data(), p, n) == 0;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) ++a;
    while (b > a && std::isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}

/// "GEN 64 temperature=0.7 top_k=64 seed=1 5,23,1" -> max_new, the sampling keys, the prompt ids.
/// Unknown keys are ignored on purpose: Strata sends engine settings this build does not have yet
/// (ckpt=, cvec=, pcie_frac=, spec_min_p=), and refusing them would break every web-app request.
bool parse_gen(const std::string& line, int& max_new, SampleOpts& so, std::vector<int>& ids, std::string& err) {
    std::istringstream ss(line);
    std::string head;
    if (!(ss >> head) || head != "GEN") { err = "expected GEN"; return false; }
    if (!(ss >> max_new)) { err = "GEN: missing max_new"; return false; }
    std::vector<std::string> rest;
    std::string tok;
    while (ss >> tok) rest.push_back(tok);
    if (rest.empty()) { err = "GEN: no prompt token ids"; return false; }
    // The ids are the LAST field (one comma-separated list); everything before them is key=value.
    const std::string id_field = rest.back();
    for (size_t i = 0; i + 1 < rest.size(); ++i) {
        const std::string& kv = rest[i];
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq);
        const double v = std::atof(kv.substr(eq + 1).c_str());
        if (k == "temperature") so.temperature = (float) v;
        else if (k == "top_p") so.top_p = (float) v;
        else if (k == "top_k") so.top_k = (int) v;
        else if (k == "min_p") so.min_p = (float) v;
        else if (k == "penalty_repeat") so.penalty_repeat = (float) v;
        else if (k == "penalty_freq") so.penalty_freq = (float) v;
        else if (k == "penalty_present") so.penalty_present = (float) v;
        else if (k == "penalty_last_n") so.penalty_last_n = (int) v;
        else if (k == "seed") so.seed = (uint64_t) std::strtoull(kv.c_str() + eq + 1, nullptr, 10);
    }
    std::istringstream cs(id_field);
    std::string part;
    while (std::getline(cs, part, ',')) {
        if (part.empty()) continue;
        char* end = nullptr;
        const long t = std::strtol(part.c_str(), &end, 10);
        if (!end || *end) { err = "GEN: bad token id '" + part + "'"; return false; }
        ids.push_back((int) t);
    }
    if (ids.empty()) { err = "GEN: no prompt token ids"; return false; }
    return true;
}

}  // namespace

int serve_loop(Model& m, const RunOpts& o, const std::vector<int>& eos_ids) {
    const Dsv4Config& c = m.config();
    std::vector<int> stops = eos_ids;
    if (stops.empty() && c.eos >= 0) stops.push_back(c.eos);

    // INFO: what the web app's Monitor tab shows. Integers stay integers (server.py parses them that way).
    std::printf("INFO engine=dsv4_ext\n");
    std::printf("INFO architecture=%s\n", c.arch.c_str());
    std::printf("INFO layers=%d\n", c.n_layer);
    std::printf("INFO n_expert=%d\n", c.n_expert);
    std::printf("INFO n_expert_used=%d\n", c.n_expert_used);
    std::printf("INFO vocab=%d\n", (int) c.vocab);
    std::printf("INFO device=%s\n", o.gpu ? (gpu::is_emulated() ? "cpu-emulated" : "cuda") : "cpu");
    // One request at a time: no batch slots, so server.py keeps its FIFO (batch stays 0).
    // "stop" says a STOP line is honoured, which the LineReader below does between tokens.
    std::printf("READY %d stop\n", o.ctx);
    std::fflush(stdout);
    std::fprintf(stderr, "dsv4_run: serving on stdin/stdout (READY %d)\n", o.ctx);
    std::fflush(stderr);

    LineReader in;
    std::string line;
    while (in.next(line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line == "QUIT") break;
        if (line == "STOP") continue;              // nothing running: it arrived after the DONE
        if (!starts(line, "GEN")) {
            std::printf("ERR unknown command: %s\n", line.substr(0, 64).c_str());
            std::fflush(stdout);
            continue;
        }

        int max_new = 0;
        SampleOpts so;
        std::vector<int> ids;
        std::string err;
        if (!parse_gen(line, max_new, so, ids, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (max_new < 0) max_new = 0;
        const int max_new_asked = max_new;
        // The vocabulary size comes from the header (tokenizer.ggml.tokens' length), which is known even
        // when the strings themselves were not kept.
        const int vocab = c.vocab > 0 ? (int) c.vocab : (int) m.tokens().size();
        bool bad_id = false;
        for (int t : ids) {
            if (t < 0 || (vocab > 0 && t >= vocab)) {
                std::printf("ERR token id %d out of range (vocab %d)\n", t, vocab);
                std::fflush(stdout);
                bad_id = true;
                break;
            }
        }
        if (bad_id) continue;
        if ((int) ids.size() >= o.ctx) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%d)\n", ids.size(), o.ctx);
            std::fflush(stdout);
            continue;
        }
        if ((int) ids.size() + max_new > o.ctx) max_new = o.ctx - (int) ids.size();
        {   // what the server really asked for, and how the rendered prompt ends (the chat template / think tag)
            const auto& tk = m.tokens();
            std::string tail;
            for (size_t i = ids.size() > 12 ? ids.size() - 12 : 0; i < ids.size(); ++i)
                tail += (ids[i] >= 0 && (size_t) ids[i] < tk.size() ? tk[(size_t) ids[i]] : std::string("?")) + " ";
            std::fprintf(stderr, "dsv4_run: GEN max_new=%d (asked %d) prompt=%zu temp=%.2f top_p=%.2f top_k=%d min_p=%.2f rep=%.2f last_n=%d | prompt ends: %s\n",
                         max_new, max_new_asked, ids.size(), so.temperature, so.top_p, so.top_k, so.min_p, so.penalty_repeat, so.penalty_last_n, tail.c_str());
            std::fflush(stderr);
        }

        m.reset();
        std::vector<float> logits;
        std::vector<int> history = ids;            // the penalties count over prompt + generated
        std::fill(m.last_routing.begin(), m.last_routing.end(), -1);

        const auto t0 = std::chrono::steady_clock::now();
        int pos = 0;
        bool stopped = false;
        // Prefill: one PP per chunk. server.py's silence watchdog (#481) needs a line often enough and the
        // Monitor shows the progress; a chunk of 8 keeps both happy without flooding the pipe.
        const size_t chunk = 8;
        for (size_t i = 0; i < ids.size(); ++i) {
            const bool last = i + 1 == ids.size();
            m.forward(ids[i], pos, last ? &logits : nullptr);   // logits only for the token that starts the answer
            m.end_token();
            ++pos;
            if (last || (i + 1) % chunk == 0) {
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                const double rate = (double) (i + 1) * 1000.0 / std::max(ms, 1e-9);
                std::printf("PP %zu %zu %.1f %.2f\n", i + 1, ids.size(), ms, rate);
                std::fflush(stdout);
            }
            std::string junk;
            while (in.try_next(junk)) {            // a STOP during the prompt: end the request at once
                if (trim(junk) == "STOP") { stopped = true; break; }
            }
            if (stopped) break;
        }
        const double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (stopped) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel\n", ids.size(), prefill_ms);
            std::fflush(stdout);
            continue;
        }
        if (logits.empty()) {
            std::printf("ERR the model produced no logits\n");
            std::fflush(stdout);
            continue;
        }

        Sampler sampler(so);
        const auto tg = std::chrono::steady_clock::now();
        int generated = 0;
        std::string finish = "length";
        for (int i = 0; i < max_new; ++i) {
            const int tok = sampler.sample(logits.data(), (int) logits.size(), history);
            std::printf("T %d\n", tok);
            std::fflush(stdout);
            history.push_back(tok);
            ++generated;
            if (std::find(stops.begin(), stops.end(), tok) != stops.end()) { finish = "stop"; break; }
            std::string junk;
            while (in.try_next(junk)) {
                if (trim(junk) == "STOP") { stopped = true; break; }
            }
            if (stopped) { finish = "cancel"; break; }
            if (i + 1 >= max_new) break;
            m.forward(tok, pos, &logits);
            m.end_token();
            ++pos;
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg).count();
        {   // the last generated tokens, decoded: was the end an EOS, or the max_new cap?
            const auto& tk = m.tokens();
            std::string g;
            for (size_t i = history.size() > ids.size() + 10 ? history.size() - 10 : ids.size(); i < history.size(); ++i)
                g += (history[i] >= 0 && (size_t) history[i] < tk.size() ? tk[(size_t) history[i]] : std::string("?")) + " ";
            std::fprintf(stderr, "dsv4_run: finish=%s generated=%d max_new=%d | last tokens: %s\n", finish.c_str(), generated, max_new, g.c_str());
            std::fflush(stderr);
        }

        // DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>: the five fields server.py's
        // _parse_done always reads. The later fields it only reads when present are left out on purpose.
        std::printf("DONE %d %zu %.1f %.1f %s\n", generated, ids.size(), prefill_ms, decode_ms, finish.c_str());
        std::fflush(stdout);
        std::fprintf(stderr, "dsv4_run: %zu prompt -> %d tokens, %s (%.0f ms prefill, %.0f ms decode)\n",
                     ids.size(), generated, finish.c_str(), prefill_ms, decode_ms);
        std::fflush(stderr);
    }
    // A profile written at QUIT is what makes the SECOND start fast: the first run learns which experts the
    // prompts actually route to, --expert-ram profile then keeps only those in RAM instead of nothing at all.
    if (!o.profile_out.empty()) {
        std::string perr;
        if (m.save_profile(perr)) std::fprintf(stderr, "dsv4_run: profile saved: %s\n", o.profile_out.c_str());
        else std::fprintf(stderr, "dsv4_run: profile save failed: %s\n", perr.c_str());
        std::fflush(stderr);
    }
    return 0;
}

}  // namespace dsv4
