// src/program/dense_main.cpp - `strata-dense --serve --native <model.gguf> --context N`
//
// The dense-model twin of `strata --serve`.  It speaks the SAME line protocol on stdin/stdout, so serve/server.py
// drives it through its unchanged StrataEngine (tokenizer, chat template, thinking levels, tools, OpenAI and
// Anthropic endpoints, web app, Monitor):
//
//   out   INFO key=value ...                      (optional facts for the Monitor)
//   out   READY <context> stop                    ("stop": this engine honours STOP)
//   in    GEN <max_new> [key=value ...] id,id,... (keys: temperature top_p top_k min_p penalty_last_n penalty_repeat
//                                                  penalty_freq penalty_present seed; unknown keys are skipped)
//   out   PP <position> <prompt tokens> <ms> <tok/s>          every few dozen prompt tokens
//   out   T <id>                                              one per generated token
//   out   DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused>
//   in    STOP | QUIT
//
// Conversation cache: the recurrent GDN state cannot be rolled back, so a request reuses the state only when
// EVERYTHING the state has consumed is a prefix of the new prompt (a chat that only appends).  Otherwise the
// state is reset and the prompt is read again.
//
// The prompt is read one token at a time (the same step as decoding).  Correct, and the simplest thing that can
// be checked against llama.cpp; a batched prefill is the next optimisation, not a prerequisite.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/dense_model.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Lines {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false};
    bool eof = false;
    void push(std::string s) {
        std::lock_guard<std::mutex> l(m);
        q.push_back(std::move(s));
        cv.notify_one();
    }
    bool pop(std::string& out) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

bool parse_ids(const char* p, std::vector<int32_t>& ids) {
    ids.clear();
    while (*p == ' ') ++p;
    while (*p) {
        char* end = nullptr;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        ids.push_back((int32_t) v);
        p = end;
        if (*p == ',') ++p;
        else if (*p != '\0' && *p != '\r' && *p != '\n') return false;
        else break;
    }
    return !ids.empty();
}

void usage() {
    std::fprintf(stderr,
                 "usage: strata-dense --serve --native <model.gguf> [--context N]\n"
                 "  a dense qwen35 GGUF (Qwen3.8-27B) behind Strata's server protocol; the server runs it with\n"
                 "  `serve/server.py --engine strata --config <run config>` whose `exe` is this program.\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf;
    long long context = 32768;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-dense: %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--serve") serve = true;
        else if (a == "--native" || a == "--model" || a == "--gguf") gguf = next("--native");
        else if (a == "--context" || a == "-c") context = std::atoll(next("--context"));
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        // the flags setup passes to `strata` (--gpu-layers, --cache, ...) mean nothing here and are not an error
    }
    if (!serve || gguf.empty()) { usage(); return 2; }

    strata::core::DenseModel model;
    std::string err;
    std::fprintf(stderr, "strata-dense: loading %s ...\n", gguf.c_str());
    if (!model.load(gguf, context, err)) {
        std::fprintf(stderr, "strata-dense: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        return 1;
    }
    const auto& cfg = model.config();

    // the ids that end an answer, from the model's own vocabulary
    std::vector<int32_t> stop_ids;
    if (cfg.eos_id >= 0) stop_ids.push_back(cfg.eos_id);
    try {
        strata::GgufFile g(gguf);
        if (const auto* toks = g.get("tokenizer.ggml.tokens"))
            for (size_t i = 0; i < toks->items.size(); ++i)
                if (toks->items[i].s == "<|im_end|>" || toks->items[i].s == "<|endoftext|>") stop_ids.push_back((int32_t) i);
    } catch (const std::exception&) {}
    std::sort(stop_ids.begin(), stop_ids.end());
    stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());
    auto is_stop = [&](int32_t t) { return std::binary_search(stop_ids.begin(), stop_ids.end(), t); };

    int* d_next = nullptr;
    int* d_hist = nullptr;
    constexpr int kMaxHist = 4096;
    if (cudaMalloc((void**) &d_next, sizeof(int)) != cudaSuccess || cudaMalloc((void**) &d_hist, kMaxHist * sizeof(int)) != cudaSuccess) {
        std::printf("ERR cannot allocate the sampler buffers\n");
        return 1;
    }

    std::printf("INFO engine=dense-0.1 model=%s layers=%d n_embd=%d weights_mib=%llu kv_mib=%llu context=%lld\n",
                cfg.arch.c_str(), cfg.n_layer, cfg.n_embd, (unsigned long long) (model.weight_bytes() >> 20),
                (unsigned long long) (model.kv_bytes() >> 20), (long long) model.max_context());
    std::printf("READY %lld stop\n", (long long) model.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "strata-dense: ready - %.2f GiB of weights, %.2f GiB of KV cache, context %lld\n",
                 model.weight_bytes() / 1073741824.0, model.kv_bytes() / 1073741824.0, (long long) model.max_context());

    Lines in;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line == "STOP") { in.stop.store(true); continue; }
            in.push(line);
        }
        std::lock_guard<std::mutex> l(in.m);
        in.eof = true;
        in.cv.notify_all();
    });
    reader.detach();

    std::vector<int32_t> fed;                      // every token the state has consumed, in order
    uint64_t draw_counter = 0;
    std::string line;
    while (in.pop(line)) {
        if (line == "QUIT") break;
        in.stop.store(false);                       // a STOP between requests is stale
        if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n"); std::fflush(stdout); continue; }

        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        strata::kernels::SamplerParams sp;
        sp.top_k = 20; sp.top_p = 1.0f; sp.temperature = 0.0f; sp.greedy = true;
        unsigned long long seed = 0;
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq);
            const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
            if (key == "temperature") { sp.temperature = fv; sp.greedy = fv <= 0.0f; }
            else if (key == "top_p") sp.top_p = fv;
            else if (key == "top_k") sp.top_k = std::atoi(tok.c_str() + eq + 1);
            else if (key == "min_p") sp.min_p = fv;
            else if (key == "penalty_last_n") sp.penalty_last_n = std::min(std::atoi(tok.c_str() + eq + 1), kMaxHist);
            else if (key == "penalty_repeat") sp.penalty_repeat = fv;
            else if (key == "penalty_freq") sp.penalty_freq = fv;
            else if (key == "penalty_present") sp.penalty_present = fv;
            else if (key == "seed") seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
        }
        std::vector<int32_t> ids;
        if (max_new < 1 || !parse_ids(endp, ids)) { std::printf("ERR bad request\n"); std::fflush(stdout); continue; }
        if ((long long) ids.size() + 1 > model.max_context()) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) model.max_context());
            std::fflush(stdout);
            continue;
        }
        sp.seed = seed;

        // ---- what of the state can be kept
        size_t resume = 0;
        if (!fed.empty() && fed.size() < ids.size() && std::equal(fed.begin(), fed.end(), ids.begin())) resume = fed.size();
        if (resume == 0) { model.reset(); fed.clear(); }
        const int64_t n = (int64_t) ids.size();
        const long long budget = std::min<long long>(max_new, model.max_context() - n);

        // ---- read the prompt
        bool cancelled = false, failed = false;
        const auto t_prompt = Clock::now();
        auto t_pp = Clock::now();
        for (int64_t i = (int64_t) resume; i < n; ++i) {
            const bool last = (i == n - 1);
            if (!model.step(ids[(size_t) i], last, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
            fed.push_back(ids[(size_t) i]);
            if (in.stop.load()) { cancelled = true; model.sync(err); break; }
            if ((i + 1 - (int64_t) resume) % 32 == 0 || last) {
                if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                const double ms = ms_since(t_pp);
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) (i + 1), (long long) n, ms_since(t_prompt),
                            ms > 0 ? 1000.0 * 32 / ms : 0.0);
                std::fflush(stdout);
                t_pp = Clock::now();
            }
        }
        if (failed) { model.reset(); fed.clear(); std::fflush(stdout); continue; }
        if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); model.reset(); fed.clear(); std::fflush(stdout); continue; }
        const double prompt_ms = ms_since(t_prompt);

        // ---- decode
        long long produced = 0;
        const char* finish = "length";
        const auto t_dec = Clock::now();
        std::vector<int32_t> hist;                  // penalty window, most recent last
        if (sp.penalty_last_n > 0) {
            const size_t take = std::min<size_t>((size_t) sp.penalty_last_n, fed.size());
            hist.assign(fed.end() - (std::ptrdiff_t) take, fed.end());
        }
        while (!cancelled && produced < budget) {
            if (in.stop.load()) { cancelled = true; break; }
            sp.counter = draw_counter;
            const int* hp = nullptr;
            int hl = 0;
            if (sp.penalty_last_n > 0) {
                std::vector<int32_t> row((size_t) sp.penalty_last_n, -1);
                std::copy(hist.begin(), hist.end(), row.end() - (std::ptrdiff_t) hist.size());
                cudaMemcpyAsync(d_hist, row.data(), row.size() * sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t) model.stream());
                hp = d_hist;
                hl = sp.penalty_last_n;
            }
            try {
                strata::kernels::sample_tokens(model.logits(), 1, cfg.n_vocab, hp, hl, sp, d_next, model.stream());
            } catch (const std::exception& e) { std::printf("ERR sampler: %s\n", e.what()); failed = true; break; }
            int next = -1;
            cudaMemcpyAsync(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t) model.stream());
            if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
            ++draw_counter;
            ++produced;
            std::printf("T %d\n", next);
            std::fflush(stdout);
            if (sp.penalty_last_n > 0) { hist.push_back(next); if ((int) hist.size() > sp.penalty_last_n) hist.erase(hist.begin()); }
            if (is_stop(next)) { finish = "stop"; break; }
            if (produced >= budget) break;
            if (!model.step(next, true, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
            fed.push_back(next);
        }
        if (failed) { model.reset(); fed.clear(); std::fflush(stdout); continue; }
        if (cancelled) finish = "cancel";
        const double decode_ms = ms_since(t_dec);
        std::printf("DONE %lld %lld %.1f %.1f %s 0 0 %lld\n", produced, (long long) n, prompt_ms, decode_ms, finish, (long long) resume);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-dense: prompt %lld tokens = %zu reused + %lld read in %.0f ms, %lld generated in %.0f ms (%.1f tok/s)%s\n",
                     (long long) n, resume, (long long) (n - (long long) resume), prompt_ms, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * produced / decode_ms : 0.0, cancelled ? " (cancelled)" : "");
    }
    cudaFree(d_next);
    cudaFree(d_hist);
    return 0;
}
