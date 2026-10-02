// src/qwen35/qwen35_main.cpp - the Qwen35MoE (Ornith) entry point: a one-shot CLI and Strata's `--serve` protocol.
//
//   strata-qwen35 --model M.gguf --tokens "1,2,3" [--max-new N]      one greedy generation, prints ids
//   strata-qwen35 --model M.gguf --check                            load + validate only
//   strata-qwen35 --serve --model M.gguf [--max-context N]          the server's stdio protocol:
//
//     READY <ctx> stop
//     < GEN <max_new> [key=value ...] <id,id,...>
//     > T <id>                  (one per generated token)
//     > DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>
//     < STOP | QUIT
//
// This is the same protocol `serve/server.py` speaks to `strata --serve`, so the Python server, the OpenAI API
// and run3.sh need no change to drive this binary.  Sampling keys the server sends are honoured where they are
// implemented and ignored otherwise.
#include "strata/qwen35/qwen35.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace q = strata::qwen35;

namespace {

struct Sampling {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;
    uint64_t seed = 0;
};

/// Greedy when temperature <= 0; otherwise temperature + top_k + top_p + min_p over the full vocabulary.
int64_t sample(const std::vector<float>& logits, const Sampling& s, std::mt19937_64& rng) {
    const int64_t V = (int64_t) logits.size();
    if (s.temperature <= 0.0f) {
        int64_t best = 0;
        for (int64_t v = 1; v < V; ++v) if (logits[(size_t) v] > logits[(size_t) best]) best = v;
        return best;
    }
    const float inv_t = 1.0f / s.temperature;
    std::vector<float> p((size_t) V);
    float mx = -INFINITY;
    for (int64_t v = 0; v < V; ++v) { p[(size_t) v] = logits[(size_t) v] * inv_t; mx = std::max(mx, p[(size_t) v]); }
    double sum = 0.0;
    for (int64_t v = 0; v < V; ++v) { p[(size_t) v] = std::exp(p[(size_t) v] - mx); sum += p[(size_t) v]; }
    for (int64_t v = 0; v < V; ++v) p[(size_t) v] = (float) (p[(size_t) v] / sum);
    std::vector<int64_t> idx((size_t) V);
    for (int64_t v = 0; v < V; ++v) idx[(size_t) v] = v;
    if (s.top_k > 0 && s.top_k < V) {
        std::partial_sort(idx.begin(), idx.begin() + s.top_k, idx.end(),
                          [&](int64_t a, int64_t b) { return p[(size_t) a] > p[(size_t) b]; });
        idx.resize((size_t) s.top_k);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return p[(size_t) a] > p[(size_t) b]; });
    }
    std::vector<int64_t> keep;
    double acc = 0.0;
    const float pmax = p[(size_t) idx[0]];
    for (int64_t v : idx) {
        if (acc >= s.top_p) break;
        if (s.min_p > 0.0f && p[(size_t) v] < s.min_p * pmax) break;
        keep.push_back(v);
        acc += p[(size_t) v];
    }
    if (keep.empty()) return idx[0];
    double tot = 0.0;
    for (int64_t v : keep) tot += p[(size_t) v];
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng) * tot, c = 0.0;
    for (int64_t v : keep) { c += p[(size_t) v]; if (r <= c) return v; }
    return keep.back();
}

std::vector<int64_t> parse_ids(const std::string& s) {
    std::vector<int64_t> ids;
    for (size_t b = 0; b < s.size();) {
        size_t e = s.find(',', b);
        if (e == std::string::npos) e = s.size();
        if (e > b) ids.push_back(std::strtoll(s.substr(b, e - b).c_str(), nullptr, 10));
        b = e + 1;
    }
    return ids;
}

int run_serve(q::TrunkWeights& w, strata::core::Qwen35Geometry& g, int64_t max_context) {
    q::TrunkState st;
    st.reset(g);
    std::vector<float> logits((size_t) g.n_vocab);
    std::printf("READY %lld stop\n", (long long) max_context);
    std::fflush(stdout);

    std::string line;
    std::mt19937_64 rng(0x5EED);
    while (std::getline(std::cin, line)) {
        if (line.rfind("GEN ", 0) == 0 || line.rfind("GENI ", 0) == 0) {
            std::vector<std::string> f;
            for (size_t b = 0; b < line.size();) {
                size_t e = line.find(' ', b);
                if (e == std::string::npos) e = line.size();
                if (e > b) f.push_back(line.substr(b, e - b));
                b = e + 1;
            }
            if (f.size() < 3) { std::printf("ERR malformed GEN\n"); std::fflush(stdout); continue; }
            const int64_t max_new = std::strtoll(f[1].c_str(), nullptr, 10);
            const std::string ids_field = f.back();
            Sampling s;
            for (size_t i = 2; i + 1 < f.size(); ++i) {
                const size_t eq = f[i].find('=');
                if (eq == std::string::npos) continue;
                const std::string k = f[i].substr(0, eq);
                const std::string v = f[i].substr(eq + 1);
                if (k == "temperature") s.temperature = std::strtof(v.c_str(), nullptr);
                else if (k == "top_p") s.top_p = std::strtof(v.c_str(), nullptr);
                else if (k == "top_k") s.top_k = std::atoi(v.c_str());
                else if (k == "min_p") s.min_p = std::strtof(v.c_str(), nullptr);
                else if (k == "seed") s.seed = std::strtoull(v.c_str(), nullptr, 10);
            }
            if (s.seed) rng.seed(s.seed);
            const std::vector<int64_t> ids = parse_ids(ids_field);
            if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); continue; }

            st.zero(g);
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t t : ids) q::trunk_forward(g, w, st, t, logits.data());
            const auto t1 = std::chrono::steady_clock::now();
            const double pms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int64_t generated = 0;
            std::string finish = "length";
            const auto d0 = std::chrono::steady_clock::now();
            for (int64_t n = 0; n < max_new; ++n) {
                const int64_t best = sample(logits, s, rng);
                if (w.eos_token >= 0 && best == w.eos_token) { finish = "stop"; break; }
                std::printf("T %lld\n", (long long) best);
                std::fflush(stdout);
                ++generated;
                q::trunk_forward(g, w, st, best, logits.data());
            }
            const auto d1 = std::chrono::steady_clock::now();
            const double dms = std::chrono::duration<double, std::milli>(d1 - d0).count();
            std::printf("DONE %lld %zu %.3f %.3f %s\n", (long long) generated, ids.size(), pms, dms, finish.c_str());
            std::fflush(stdout);
        } else if (line.rfind("STOP", 0) == 0) {
            // one request at a time: STOP arrives only while a request runs, which this synchronous loop cannot see.
        } else if (line.rfind("QUIT", 0) == 0) {
            break;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string path;
    std::vector<int64_t> tokens;
    int64_t max_new = 8;
    int64_t max_context = 0;
    bool check_only = false, serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") path = next();
        else if (a == "--tokens") tokens = parse_ids(next());
        else if (a == "--max-new") max_new = std::strtoll(next(), nullptr, 10);
        else if (a == "--max-context") max_context = std::strtoll(next(), nullptr, 10);
        else if (a == "--serve") serve = true;
        else if (a == "--check") check_only = true;
        else if (a == "--capabilities") { std::printf("qwen35moe\n"); return 0; }
        else if (a == "--mtp" || a == "--spec" || a == "--spec-min-p" || a == "--kv" ||
                 a == "--prefill" || a == "--pool-workers" || a == "--expert-cache" ||
                 a == "--adapt-every" || a == "--pcie-frac" || a == "--vram-reserve-mib") {
            next();   // accepted for the shared launcher, not used by this CPU path
        }
        else { std::fprintf(stderr, "strata-qwen35: unknown option %s\n", a.c_str()); return 2; }
    }
    if (path.empty()) {
        std::fprintf(stderr, "usage: strata-qwen35 --model M.gguf [--serve] [--tokens \"1,2\"] [--max-new N]\n");
        return 2;
    }

    q::qwen35_enable_ggml();
    strata::core::Qwen35Geometry g;
    q::TrunkWeights w;
    std::string err;
    if (!q::load_trunk(path, g, w, err)) {
        std::fprintf(stderr, "strata-qwen35: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata-qwen35: %lld layers, %lld wide, %lld experts top-%lld, vocab %lld, eos %lld\n",
                 (long long) g.n_layers, (long long) g.n_embd, (long long) g.n_expert,
                 (long long) g.n_expert_used, (long long) g.n_vocab, (long long) w.eos_token);
    if (check_only) { std::printf("check ok\n"); return 0; }
    if (serve) return run_serve(w, g, max_context > 0 ? max_context : g.context_length);
    if (tokens.empty()) { std::fprintf(stderr, "strata-qwen35: --tokens is required without --serve\n"); return 2; }

    q::TrunkState st;
    st.reset(g);
    std::vector<float> logits((size_t) g.n_vocab);
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t tok : tokens) q::trunk_forward(g, w, st, tok, logits.data());
    const auto t1 = std::chrono::steady_clock::now();
    const double pms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "prefill %zu tokens in %.0f ms (%.1f tok/s)\n", tokens.size(), pms,
                 pms > 0 ? 1000.0 * (double) tokens.size() / pms : 0.0);
    std::printf("prompt:");
    for (int64_t t : tokens) std::printf(" %lld", (long long) t);
    std::printf("\ngen:");
    Sampling s;
    std::mt19937_64 rng(0);
    for (int64_t n = 0; n < max_new; ++n) {
        const int64_t best = sample(logits, s, rng);
        std::printf(" %lld", (long long) best);
        std::fflush(stdout);
        q::trunk_forward(g, w, st, best, logits.data());
    }
    std::printf("\n");
    return 0;
}
