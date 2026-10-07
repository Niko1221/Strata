// src/glm/main.cpp - `strata-glm`: the GLM-5.3 engine program.
//
// Three modes:
//   strata-glm --model DIR --tokens 1,2,3 --gen 16     greedy generation from token ids (prints the ids and timing)
//   strata-glm --model DIR --tokens 1,2,3 --tf          one forward over the ids: the argmax and top-5 at every
//                                                        position (the comparison against colibri, docs/GLM53.md)
//   strata-glm --model DIR --serve                      the line protocol serve/server.py speaks to `strata --serve`:
//       in:  GEN <max_new> [temperature=F top_p=F top_k=N min_p=F seed=N ...] <id,id,...>  |  STOP  |  QUIT
//       out: INFO k=v ... / READY <ctx> stop, then per request RESUME <reused>, PP <read> <total> <ms> <tok/s>,
//            T <id> per token, and DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel>
//            <drafts accepted> <drafts offered> <reused> <hits> <lookups> <ram blobs> <file blobs> <file MB>
//            <prompt read> <offloaded>;  ERR <why> when a request cannot run.
// The conversation's KV stays between requests: a request that starts with the tokens of the last one reads only
// its new tokens (RESUME says how many it reused).
#include "strata/glm/model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "0.0.0"
#endif

using namespace strata::glm;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

bool parse_ids(const std::string& s, std::vector<int>& out, std::string& err) {
    out.clear();
    std::string t = s;
    for (char& ch : t)
        if (ch == ',') ch = ' ';
    std::istringstream in(t);
    std::string tok;
    while (in >> tok) {
        char* end = nullptr;
        const long long v = std::strtoll(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0' || v < 0 || v > 0x7fffffff) { err = "invalid token id '" + tok + "'"; return false; }
        out.push_back((int) v);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int penalty_last_n = 0;
    unsigned long long seed = 0;
};

int argmax(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

/// Greedy at temperature 0; otherwise penalties, temperature, top-k, top-p, min-p, then a draw.
int sample(std::vector<float>& logits, const Sampling& sp, const std::vector<int>& recent, std::mt19937_64& rng) {
    const int V = (int) logits.size();
    if (sp.penalty_last_n != 0 && (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f)) {
        const size_t n = sp.penalty_last_n < 0 ? recent.size() : std::min(recent.size(), (size_t) sp.penalty_last_n);
        std::vector<std::pair<int, int>> counts;
        for (size_t i = recent.size() - n; i < recent.size(); ++i) {
            const int t = recent[i];
            auto it = std::find_if(counts.begin(), counts.end(), [&](auto& p) { return p.first == t; });
            if (it == counts.end()) counts.emplace_back(t, 1); else ++it->second;
        }
        for (auto [t, cnt] : counts) {
            if (t < 0 || t >= V) continue;
            float& l = logits[t];
            l = l > 0 ? l / sp.penalty_repeat : l * sp.penalty_repeat;
            l -= cnt * sp.penalty_freq + sp.penalty_present;
        }
    }
    if (sp.temperature <= 0.0f) return argmax(logits.data(), V);
    std::vector<int> ord(V);
    std::iota(ord.begin(), ord.end(), 0);
    int k = sp.top_k > 0 ? std::min(sp.top_k, V) : V;
    if (k < V) {
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        ord.resize(k);
    } else {
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    }
    std::vector<double> p(ord.size());
    const double mx = logits[ord[0]];
    double sum = 0.0;
    for (size_t i = 0; i < ord.size(); ++i) { p[i] = std::exp((logits[ord[i]] - mx) / sp.temperature); sum += p[i]; }
    for (double& v : p) v /= sum;
    size_t keep = p.size();
    if (sp.top_p < 1.0f) {
        double cum = 0.0;
        for (size_t i = 0; i < p.size(); ++i) {
            cum += p[i];
            if (cum >= sp.top_p) { keep = i + 1; break; }
        }
    }
    if (sp.min_p > 0.0f)
        for (size_t i = 1; i < keep; ++i)
            if (p[i] < sp.min_p * p[0]) { keep = i; break; }
    double tot = 0.0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        r -= p[i];
        if (r <= 0.0) return ord[i];
    }
    return ord[keep - 1];
}

struct Options {
    std::string model;
    ModelOptions mo;
    std::string tokens;
    int gen = 0;
    bool tf = false;
    bool serve = false;
    int chunk = 1024;   // a chunk reads each layer's routed experts once: larger chunks share that read (docs/GLM53.md)
    bool layer_major = true, ignore_eos = false;
    std::string tokens_file, logits_out;
    int repeat = 1;
};

void usage() {
    std::fprintf(stderr,
                 "usage: strata-glm --model DIR [--threads N] [--io-threads N] [--max-context N] [--ram-gb G]\n"
                 "                  [--expert-ram-gb G] [--prefill-chunk N] [--prefill layer|chunk]\n"
                 "                  [--kv f32|bf16|fp8] [--prefetch] [--gpu N] [--vram-gb G]\n"
                 "                  [--vram-reserve-mib N] [--promote-after N] [--cpu-prefill]\n"
                 "                  (--tokens ID,ID,.. | --tokens-file FILE) [--gen N | --tf] [--logits-out FILE]\n"
                 "                  [--repeat N] [--ignore-eos] | --serve\n");
}

/// One prompt read through the model.  `on_chunk(read, ms, cached)` reports progress: `read` is how far the prompt
/// is, and `cached` says those tokens are now in the KV.  Layer-major (the whole prompt through one layer at a time)
/// reports after every layer, with `read` the prompt's share of the layers done; only the last report is cached.
bool prefill(GlmModel& m, const std::vector<int>& ids, size_t from, int chunk, std::vector<float>& logits,
             const std::function<void(size_t, double, bool)>& on_chunk, const std::atomic<bool>* stop,
             std::string& err, bool layer_major) {
    const double t0 = now_ms();
    if (layer_major) {
        const int n_layers = m.config().n_layers;
        const size_t n = ids.size() - from;
        const std::function<void(int)> on_layer = [&](int done) {
            if (on_chunk && done < n_layers) on_chunk(from + n * (size_t) done / (size_t) n_layers, now_ms() - t0, false);
        };
        if (!m.forward_prefill(ids.data() + from, (int) n, logits.data(), err, stop, chunk, &on_layer)) return false;
        if (on_chunk) on_chunk(ids.size(), now_ms() - t0, true);
        return true;
    }
    for (size_t b = from; b < ids.size();) {
        const size_t n = std::min<size_t>(chunk, ids.size() - b);
        const bool last = b + n == ids.size();
        if (!m.forward(ids.data() + b, (int) n, last ? logits.data() : nullptr, nullptr, err, stop, chunk)) return false;
        b += n;
        if (on_chunk) on_chunk(b, now_ms() - t0, true);
        if (stop && stop->load() && !last) { err = "cancelled"; return false; }
    }
    return true;
}

void dump_logits(const std::string& path, const float* values, size_t count) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)values, count * sizeof(float));
    if (!f) throw std::runtime_error("cannot write logits to " + path);
}

int run_cli(GlmModel& m, const Options& o) {
    std::vector<int> ids;
    std::string err;
    if (!parse_ids(o.tokens, ids, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    const int V = m.config().vocab;
    const ExpertStats es0 = m.experts().stats();
    const ForwardStats fs0 = m.stats();
    const uint64_t io0 = m.io_bytes(), gpu0 = m.gpu_hits();
    if (o.tf) {
        std::vector<float> all((size_t) ids.size() * V);
        const double t0 = now_ms();
        if (!m.forward(ids.data(), (int) ids.size(), nullptr, all.data(), err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        dump_logits(o.logits_out, all.data(), all.size());
        std::fprintf(stderr, "forward of %zu tokens in %.1f s\n", ids.size(), (now_ms() - t0) / 1e3);
        for (size_t s = 0; s < ids.size(); ++s) {
            const float* lo = &all[s * V];
            std::vector<int> ord(V);
            std::iota(ord.begin(), ord.end(), 0);
            std::partial_sort(ord.begin(), ord.begin() + 5, ord.end(), [&](int a, int b) { return lo[a] > lo[b]; });
            std::printf("pos %zu in %d argmax %d top5", s, ids[s], ord[0]);
            for (int k = 0; k < 5; ++k) std::printf(" %d:%.4f", ord[k], lo[ord[k]]);
            std::printf("\n");
        }
        return 0;
    }
    std::vector<float> logits(V);
    double t0 = now_ms();
    if (!prefill(m, ids, 0, o.chunk, logits,
                 [&](size_t done, double ms, bool) { std::fprintf(stderr, "prompt %zu/%zu in %.1f s\n", done, ids.size(), ms / 1e3); },
                 nullptr, err, o.layer_major)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const double pp_ms = now_ms() - t0;
    dump_logits(o.logits_out, logits.data(), logits.size());
    t0 = now_ms();
    std::vector<int> out;
    const auto& eos = m.config().eos;
    for (int i = 0; i < o.gen; ++i) {
        const int t = argmax(logits.data(), V);
        out.push_back(t);
        std::printf("%d%s", t, i + 1 < o.gen ? " " : "\n");
        std::fflush(stdout);
        if (!o.ignore_eos && std::find(eos.begin(), eos.end(), t) != eos.end()) { std::printf("\n"); break; }
        if (i + 1 == o.gen) break;
        if (!m.forward(&t, 1, logits.data(), nullptr, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    }
    const double tg_ms = now_ms() - t0;
    const ExpertStats& es = m.experts().stats();
    const ForwardStats& fs = m.stats();
    std::fprintf(stderr,
                 "prompt %zu tokens in %.1f s | %zu generated in %.1f s (%.2f tok/s) | expert hits %llu of %llu (%.1f%%), "
                 "read %.1f GB | attn %.1f s, moe %.1f s (waiting for reads %.1f s), dense %.1f s, head %.1f s\n",
                 ids.size(), pp_ms / 1e3, out.size(), tg_ms / 1e3, tg_ms > 0 ? 1000.0 * (out.size() > 0 ? out.size() - 1 : 0) / tg_ms : 0.0,
                 (unsigned long long) (es.hits - es0.hits), (unsigned long long) (es.requests - es0.requests),
                 es.requests > es0.requests ? 100.0 * (es.hits - es0.hits) / (es.requests - es0.requests) : 0.0,
                 (double)(m.io_bytes() - io0) / 1e9,
                 (fs.attn_ms - fs0.attn_ms) / 1e3, (fs.moe_ms - fs0.moe_ms) / 1e3,
                 (fs.expert_wait_ms - fs0.expert_wait_ms) / 1e3, (fs.dense_ms - fs0.dense_ms) / 1e3, (fs.head_ms - fs0.head_ms) / 1e3);
    std::fprintf(stderr, "strata-glm: GPU hits %llu, prefetch used %llu of %llu, KV %s (%llu bytes)\n",
                 (unsigned long long)(m.gpu_hits() - gpu0), (unsigned long long)(es.prefetch_used - es0.prefetch_used),
                 (unsigned long long)(es.prefetched - es0.prefetched), m.kv_type(), (unsigned long long)m.kv_bytes());
    return 0;
}

// ---------------------------------------------------------------------------------------------------- serve

class LineReader {
    struct Command { std::string line; std::shared_ptr<std::atomic<bool>> cancel; };
    struct State {
        std::mutex m;
        std::condition_variable cv;
        std::deque<Command> q;
        std::shared_ptr<std::atomic<bool>> active;
        bool eof = false;
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
    std::shared_ptr<std::atomic<bool>> active_;
public:
    LineReader() {
        // The stdin thread owns shared state, never the stack-allocated reader. QUIT can return while stdin
        // remains open without leaving a detached thread referring to destroyed mutexes.
        std::thread([s = state_] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                {
                    std::lock_guard<std::mutex> lk(s->m);
                    if (line == "STOP" || line == "QUIT") {
                        if (s->active) s->active->store(true);
                        else if (line == "STOP" && !s->q.empty()) s->q.front().cancel->store(true);
                        if (line == "STOP") continue;
                    }
                    s->q.push_back({line, std::make_shared<std::atomic<bool>>(false)});
                }
                s->cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(s->m);
            s->eof = true;
            s->cv.notify_all();
        }).detach();
    }
    /// The next command line; false at end of input.
    bool next(std::string& line) {
        std::unique_lock<std::mutex> lk(state_->m);
        state_->active.reset();
        state_->cv.wait(lk, [&] { return state_->eof || !state_->q.empty(); });
        if (state_->q.empty()) return false;
        line = std::move(state_->q.front().line);
        active_ = state_->active = state_->q.front().cancel;
        state_->q.pop_front();
        return true;
    }
    bool stop_requested() const { return active_ && active_->load(); }
    const std::atomic<bool>* cancel_flag() const { return active_.get(); }
};

int run_serve(GlmModel& m, const Options& o) {
    const GlmConfig& c = m.config();
    const int V = c.vocab;
    std::printf("INFO context=%d kv=%s gpu_expert_slots=%d expert_slots=%d expert_cache_mib=%lld pool_workers=%d model=glm-5.3 "
                "dense_mib=%lld engine=" STRATA_VERSION "-glm\n",
                m.max_context(), m.kv_type(), m.gpu_slots(), m.experts().slots(),
                (long long) ((uint64_t) m.experts().slots() * m.experts().slot_bytes() >> 20), m.pool().size(),
                (long long) (m.dense_bytes() >> 20));
    std::printf("READY %d stop\n", m.max_context());
    std::fflush(stdout);
    LineReader in;
    std::vector<int> history;   // the tokens whose KV rows are in the cache, in order
    std::vector<float> logits(V);
    std::string line;
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        // GEN <max_new> [keys] <ids>
        std::istringstream ls(line.substr(4));
        long long max_new = 0;
        ls >> max_new;
        Sampling sp;
        std::string tok, ids_text;
        while (ls >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) {
                ids_text = tok;
                std::string rest;
                std::getline(ls, rest);
                ids_text += rest;
                break;
            }
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "temperature") sp.temperature = std::strtof(v.c_str(), nullptr);
            else if (k == "top_p") sp.top_p = std::strtof(v.c_str(), nullptr);
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = std::strtof(v.c_str(), nullptr);
            else if (k == "penalty_last_n") sp.penalty_last_n = std::atoi(v.c_str());
            else if (k == "penalty_repeat") sp.penalty_repeat = std::strtof(v.c_str(), nullptr);
            else if (k == "penalty_freq") sp.penalty_freq = std::strtof(v.c_str(), nullptr);
            else if (k == "penalty_present") sp.penalty_present = std::strtof(v.c_str(), nullptr);
            else if (k == "seed") sp.seed = std::strtoull(v.c_str(), nullptr, 10);
            // other keys (cvec, ckpt, pcie_frac, spec_min_p) belong to the Qwen engine: skipped
        }
        std::vector<int> ids;
        std::string err;
        if (max_new < 1 || !parse_ids(ids_text, ids, err)) {
            std::printf("ERR bad request: %s\n", err.empty() ? "max_new" : err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int t) { return t >= V; })) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        if ((long long) ids.size() + max_new > m.max_context()) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new,
                        m.max_context());
            std::fflush(stdout);
            continue;
        }
        // reuse the longest common prefix with what the cache holds (at least one token is read: it gives the logits)
        size_t reuse = 0;
        while (reuse < history.size() && reuse < ids.size() && history[reuse] == ids[reuse]) ++reuse;
        if (reuse == ids.size()) --reuse;
        m.truncate((int) reuse);
        history.resize(reuse);
        std::printf("RESUME %zu\n", reuse);
        std::fflush(stdout);
        const ExpertStats e0 = m.experts().stats();
        const double t0 = now_ms();
        bool cancelled = false;
        size_t read_to = reuse;
        const bool ok = prefill(m, ids, reuse, o.chunk, logits,
                                [&](size_t done, double ms, bool cached) {
                                    if (cached) read_to = done;
                                    const double rate = ms > 0 ? 1000.0 * (done - reuse) / ms : 0.0;
                                    std::printf("PP %zu %zu %.0f %.1f\n", done, ids.size(), ms, rate);
                                    std::fflush(stdout);
                                },
                                in.cancel_flag(), err, o.layer_major);
        history.assign(ids.begin(), ids.begin() + (ptrdiff_t) read_to);
        const double prompt_ms = now_ms() - t0;
        int produced = 0;
        const char* finish = "length";
        double decode_ms = 0.0;
        if (!ok && err != "cancelled") {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            m.truncate(0);
            history.clear();
            continue;
        }
        if (!ok) {
            cancelled = true;
            finish = "cancel";
        } else {
            std::mt19937_64 rng(sp.seed ? sp.seed : (unsigned long long) std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::vector<int> recent(ids.begin(), ids.end());
            const double t1 = now_ms();
            for (;;) {
                if (in.stop_requested()) { finish = "cancel"; cancelled = true; break; }
                const int t = sample(logits, sp, recent, rng);
                ++produced;
                recent.push_back(t);
                std::printf("T %d\n", t);
                std::fflush(stdout);
                if (std::find(c.eos.begin(), c.eos.end(), t) != c.eos.end()) { finish = "stop"; break; }
                if (produced >= max_new) break;
                if (in.stop_requested()) { finish = "cancel"; cancelled = true; break; }
                if (!m.forward(&t, 1, logits.data(), nullptr, err, in.cancel_flag())) {
                    if (err == "cancelled") { finish = "cancel"; cancelled = true; break; }
                    std::printf("ERR %s\n", err.c_str());
                    std::fflush(stdout);
                    finish = nullptr;
                    break;
                }
                history.push_back(t);
            }
            decode_ms = now_ms() - t1;
            if (!finish) { m.truncate(0); history.clear(); continue; }
        }
        const ExpertStats& e1 = m.experts().stats();
        const long long hits = (long long) (e1.hits - e0.hits), look = (long long) (e1.requests - e0.requests);
        const long long misses = (long long) (e1.misses - e0.misses);
        const size_t read_n = read_to - reuse;
        std::printf("DONE %d %zu %.1f %.1f %s 0 0 %zu %lld %lld %lld %lld %.1f %zu 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, reuse, hits, look, hits, misses,
                    (double) misses * m.experts().slot_bytes() / 1e6, read_n);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "strata serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %d generated in "
                     "%.0f ms (%.1f tok/s), drafts accepted 0 of 0, 0 checkpoints%s\n",
                     ids.size(), reuse, read_n, prompt_ms, prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * produced / decode_ms : 0.0, cancelled ? " (cancelled)" : "");
        std::fprintf(stderr, "strata-glm: expert hits %lld of %lld (%.1f%%), %.2f GB read from the SSD\n", hits, look,
                     look ? 100.0 * hits / look : 0.0, (double) misses * m.experts().slot_bytes() / 1e9);
        std::fflush(stderr);
    }
    return 0;
}

}  // namespace

int program(int argc, char** argv) {
    Options o;
    auto number = [](const std::string& flag, const char* s, double hi, bool whole = true) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (!end || end == s || *end || !std::isfinite(v) || v < 0 || v > hi || (whole && std::floor(v) != v))
            throw std::invalid_argument(flag + ": invalid nonnegative number '" + s + "'");
        return v;
    };
    if (const char* s = std::getenv("STRATA_WATCHDOG_S")) o.mo.watchdog_seconds = number("STRATA_WATCHDOG_S", s, 86400, false);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") o.model = next();
        else if (a == "--features") {
#ifdef STRATA_GLM_CUDA
            std::puts("{\"family\":\"glm\",\"cuda\":true}");
#else
            std::puts("{\"family\":\"glm\",\"cuda\":false}");
#endif
            return 0;
        }
        else if (a == "--threads") o.mo.threads = (int)number(a, next(), 1024);
        else if (a == "--io-threads") o.mo.io_threads = (int)number(a, next(), 128);
        else if (a == "--max-context") o.mo.max_context = (int)number(a, next(), 131072);
        else if (a == "--ram-gb") o.mo.ram_budget = (uint64_t)(number(a, next(), 1000000, false) * 1e9);
        else if (a == "--expert-ram-gb") o.mo.expert_ram = (uint64_t)(number(a, next(), 1000000, false) * 1e9);
        else if (a == "--prefill-chunk") o.chunk = (int)number(a, next(), 131072);
        else if (a == "--prefill") {
            const std::string v = next();
            if (v != "layer" && v != "chunk") throw std::invalid_argument("--prefill must be layer or chunk");
            o.layer_major = v == "layer";
        }
        else if (a == "--kv") o.mo.kv = kv_format(next());
        else if (a == "--prefetch") o.mo.prefetch = true;
        else if (a == "--gpu") { o.mo.gpu = true; o.mo.device = (int)number(a, next(), 1024); }
        else if (a == "--vram-gb") o.mo.vram_budget = (uint64_t)(number(a, next(), 1000000, false) * 1e9);
        else if (a == "--vram-reserve-mib") o.mo.vram_reserve = (uint64_t)number(a, next(), 1ull << 30) << 20;
        else if (a == "--promote-after") o.mo.promote_after = (int)number(a, next(), 1000000);
        else if (a == "--cpu-prefill") o.mo.gpu_prefill = false;
        else if (a == "--tokens") o.tokens = next();
        else if (a == "--tokens-file") o.tokens_file = next();
        else if (a == "--logits-out") o.logits_out = next();
        else if (a == "--repeat") o.repeat = (int)number(a, next(), 1000000);
        else if (a == "--ignore-eos") o.ignore_eos = true;
        else if (a == "--gen") o.gen = (int)number(a, next(), 131072);
        else if (a == "--tf") o.tf = true;
        else if (a == "--serve") o.serve = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
    if (!o.tokens_file.empty()) {
        std::ifstream f(o.tokens_file);
        if (!f) throw std::invalid_argument("cannot open token file " + o.tokens_file);
        o.tokens.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    if (o.mo.max_context < 16 || o.mo.max_context > 131072 || o.mo.threads < 0 || o.mo.threads > 1024 ||
        o.mo.io_threads < 1 || o.mo.io_threads > 128 || o.gen < 0 || o.repeat < 1 || o.chunk < 1 || o.mo.device < 0 || o.mo.promote_after < 1)
        throw std::invalid_argument("invalid context, thread count, generation length, repeat count or GPU option");
    if (o.model.empty() || (!o.serve && o.tokens.empty())) { usage(); return 2; }
    GlmModel m;
    std::string err;
    if (!m.load(o.model, o.mo, err)) {
        if (o.serve) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); }
        std::fprintf(stderr, "strata-glm: %s\n", err.c_str());
        return 1;
    }
    try {
        if (o.serve) return run_serve(m, o);
        for (int r = 0; r < o.repeat; ++r) {
            m.truncate(0);
            if (o.repeat > 1) std::fprintf(stderr, "run %d/%d\n", r + 1, o.repeat);
            const int rc = run_cli(m, o);
            if (rc) return rc;
        }
        return 0;
    } catch (const std::exception& e) {
        std::printf("ERR %s\n", e.what());
        std::fflush(stdout);
        std::fprintf(stderr, "strata-glm: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    try { return program(argc, argv); }
    catch (const std::exception& e) { std::fprintf(stderr, "strata-glm: %s\n", e.what()); return 1; }
}
