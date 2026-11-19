// llama_vulkan_engine.cpp - a STRATA PROTOCOL BRIDGE, not a Strata engine.
//
// It makes `terminal/serve/server.py --engine strata` drive llama.cpp's Vulkan
// backend: it speaks the tiny line protocol the server expects (READY / GEN /
// T / DONE / STOP / QUIT) and does the actual inference with libllama.
//
// This is the "bridge first" step of the plan in engine/vulkan/README.md. It is
// intentionally separate from the native Vulkan engine in engine/vulkan/src:
// llama.cpp generates the tokens today; our streaming tier does not yet. The
// native engine will replace this binary once its kernels are ported.
//
// Build: see build.sh.  Use: see setup_bridge.py / README.md.

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string backend_dir;      // directory holding libggml-vulkan.so
    int ctx = 32768;
    int threads = 8;
    int n_batch = 512;
    uint32_t seed = 42;
    int lp = 0;  // --lp K: emit top-K logprobs per generated token for parity checks
};

bool take_int(const std::string& s, const std::string& key, int& out) {
    if (s.rfind(key + "=", 0) != 0) return false;
    try { out = std::stoi(s.substr(key.size() + 1)); }
    catch (...) { return false; }
    return true;
}

bool take_float(const std::string& s, const std::string& key, float& out) {
    if (s.rfind(key + "=", 0) != 0) return false;
    try { out = std::stof(s.substr(key.size() + 1)); }
    catch (...) { return false; }
    return true;
}

struct GenRequest {
    int max_new = 0;
    std::vector<llama_token> prompt;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;
    uint32_t seed = 42;
    bool greedy = false;
};

// Parse "GEN <max_new> [key=value ...] <id,id,...>". The last field is always
// the comma separated token list; everything between is optional sampling.
bool parse_gen(const std::string& line, GenRequest& req) {
    std::vector<std::string> f;
    size_t pos = 0;
    while (pos < line.size()) {
        size_t end = line.find(' ', pos);
        if (end == std::string::npos) end = line.size();
        if (end > pos) f.push_back(line.substr(pos, end - pos));
        pos = end + 1;
    }
    if (f.size() < 2) return false;
    try { req.max_new = std::stoi(f[1]); }
    catch (...) { return false; }
    if (req.max_new < 0) req.max_new = 0;

    const std::string& ids = f.back();
    size_t start = 0;
    while (start <= ids.size()) {
        size_t comma = ids.find(',', start);
        std::string tok = ids.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) {
            try { req.prompt.push_back(static_cast<llama_token>(std::stoi(tok))); }
            catch (...) { /* ignore malformed id */ }
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }

    for (size_t i = 2; i + 1 < f.size(); ++i) {
        int k = 0;
        float v = 0.0f;
        if (take_float(f[i], "temperature", v)) { req.temperature = v; req.greedy = v <= 0.0001f; }
        else if (take_float(f[i], "top_p", v)) req.top_p = v;
        else if (take_int(f[i], "top_k", k)) req.top_k = k;
        else if (take_int(f[i], "seed", k)) req.seed = static_cast<uint32_t>(k);
    }
    return true;
}

void emit(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::cout << std::flush;
}

std::vector<llama_token> parse_ids(const std::string& line) {
    std::vector<llama_token> ids;
    size_t pos = line.find(' ');
    if (pos == std::string::npos) return ids;
    const std::string s = line.substr(pos + 1);
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) { try { ids.push_back(static_cast<llama_token>(std::stoi(tok))); } catch (...) {} }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return ids;
}

int argmax_of(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

// Strata's LP wire format: LP <logprob of token T> <tok>:<logprob> ...
void emit_top_logprobs(const float* logits, int n, int k, int chosen) {
    if (k <= 0 || n <= 0) return;
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) if (!std::isnan(logits[i])) mx = std::max(mx, logits[i]);
    double z = 0.0;
    for (int i = 0; i < n; ++i) z += std::exp(double(logits[i] - mx));
    const double lz = std::log(z);
    const int keep = std::min(k, n);
    std::vector<int> top(n);
    std::iota(top.begin(), top.end(), 0);
    std::partial_sort(top.begin(), top.begin() + keep, top.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });
    emit("LP %.6f", double(logits[chosen] - mx) - lz);
    for (int i = 0; i < keep; ++i) {
        int t = top[i];
        emit(" %d:%.6f", t, double(logits[t] - mx) - lz);
    }
    emit("\n");
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) opt.model = argv[++i];
        else if (a == "--backend-dir" && i + 1 < argc) opt.backend_dir = argv[++i];
        else if (a == "--ctx" && i + 1 < argc) opt.ctx = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) opt.threads = std::atoi(argv[++i]);
        else if (a == "--n-batch" && i + 1 < argc) opt.n_batch = std::atoi(argv[++i]);
        else if (a == "--lp" && i + 1 < argc) opt.lp = std::atoi(argv[++i]);
        // --serve and everything else from the Strata config is ignored.
    }
    if (opt.model.empty()) {
        std::fprintf(stderr, "usage: %s --model FILE.gguf [--ctx N] [--threads N] "
                             "[--backend-dir DIR] [--n-batch N]\n", argv[0]);
        return 2;
    }
    if (opt.ctx < 512) opt.ctx = 32768;
    if (opt.threads < 1) opt.threads = 8;
    if (opt.n_batch < 1) opt.n_batch = 512;

    // Find libggml-vulkan.so even when this binary lives outside llama.cpp's
    // build dir. The config can also set lib_dirs (LD_LIBRARY_PATH).
    if (!opt.backend_dir.empty()) ggml_backend_load_all_from_path(opt.backend_dir.c_str());
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;                 // all layers on the (integrated) Vulkan GPU
    llama_model* model = llama_model_load_from_file(opt.model.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "llama_vulkan_engine: cannot load model %s\n", opt.model.c_str());
        llama_backend_free();
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(opt.ctx);
    cp.n_batch = static_cast<uint32_t>(opt.n_batch);
    cp.n_threads = opt.threads;
    cp.n_threads_batch = opt.threads;
    llama_context* lctx = llama_init_from_model(model, cp);
    if (!lctx) {
        std::fprintf(stderr, "llama_vulkan_engine: cannot create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // The only startup line the server requires. "stop" means it may send STOP.
    emit("INFO engine=llama.cpp-vulkan-bridge\n");
    emit("READY %d stop\n", llama_n_ctx(lctx));

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        if (line.rfind("VRAM", 0) == 0) {
            // server.py waits for a reply after POST /v1/vram. We expose none.
            emit("VRAM expert_slots=0 vram_free_mib=0\n");
            continue;
        }
        if (line.rfind("EVAL", 0) == 0) {
            // Teacher forcing: decode the given tokens and emit top-K logprobs
            // after every position, without sampling (see the native engine).
            std::vector<llama_token> ids = parse_ids(line);
            if (ids.empty()) { emit("ERR bad EVAL\n"); continue; }
            llama_memory_clear(llama_get_memory(lctx), true);
            bool ok = true;
            for (size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] < 0 || ids[i] >= n_vocab) {
                    emit("ERR EVAL token out of range\n"); ok = false; break;
                }
                llama_batch batch = llama_batch_get_one(&ids[i], 1);
                if (llama_decode(lctx, batch) != 0) {
                    emit("ERR EVAL decode failed\n"); ok = false; break;
                }
                const float* lg = llama_get_logits_ith(lctx, -1);
                emit_top_logprobs(lg, n_vocab, opt.lp, argmax_of(lg, n_vocab));
            }
            emit("DONE %d %zu 0.0 0.0 %s\n", ok ? int(ids.size()) : 0, ids.size(),
                 ok ? "length" : "nonfinite");
            continue;
        }
        if (line.rfind("GEN", 0) != 0) continue;   // STOP, SAVE, ... not needed yet

        GenRequest req;
        if (!parse_gen(line, req) || req.prompt.empty()) {
            emit("ERR bad GEN line\n");
            continue;
        }
        bool bad_id = false;
        for (llama_token t : req.prompt)
            if (t < 0 || t >= n_vocab) { bad_id = true; break; }
        if (bad_id) {
            emit("ERR prompt token out of range\n");
            emit("DONE 0 %d 0.0 0.0 nonfinite\n", static_cast<int>(req.prompt.size()));
            continue;
        }

        // Each GEN is an independent prompt. Clear the KV cache from any prior
        // GEN in this session, so a second prompt is not contaminated by the
        // first (the native engine does the same with forward->reset()).
        llama_memory_clear(llama_get_memory(lctx), true);

        const auto t_prompt0 = std::chrono::steady_clock::now();
        const int prompt_n = static_cast<int>(req.prompt.size());
        int rc = 0;
        for (int i = 0; i < prompt_n && rc == 0; i += opt.n_batch) {
            int n = std::min(opt.n_batch, prompt_n - i);
            llama_batch batch = llama_batch_get_one(req.prompt.data() + i, n);
            rc = llama_decode(lctx, batch);
        }
        const double prompt_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_prompt0).count();
        emit("PP %d %d %.1f %.1f\n", prompt_n, prompt_n, prompt_ms,
             prompt_ms > 0 ? 1000.0 * prompt_n / prompt_ms : 0.0);
        if (rc != 0) {
            emit("ERR decode failed: %d\n", rc);
            emit("DONE 0 %d %.1f 0.0 nonfinite\n", prompt_n, prompt_ms);
            continue;
        }

        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        llama_sampler* smpl = llama_sampler_chain_init(sp);
        if (req.greedy) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            if (req.top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(req.top_k));
            if (req.top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(req.top_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(req.temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(req.seed));
        }

        const auto t_gen0 = std::chrono::steady_clock::now();
        int generated = 0;
        const char* finish = "length";
        int used = prompt_n;
        for (int i = 0; i < req.max_new; ++i) {
            if (used >= static_cast<int>(llama_n_ctx(lctx))) { finish = "length"; break; }
            llama_token next = llama_sampler_sample(smpl, lctx, -1);
            llama_sampler_accept(smpl, next);
            if (llama_vocab_is_eog(vocab, next)) { finish = "stop"; break; }
            emit("T %d\n", static_cast<int>(next));
            emit_top_logprobs(llama_get_logits_ith(lctx, -1), n_vocab, opt.lp, static_cast<int>(next));
            ++generated;
            llama_batch batch = llama_batch_get_one(&next, 1);
            rc = llama_decode(lctx, batch);
            if (rc != 0) { finish = "nonfinite"; break; }
            ++used;
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_gen0).count();
        emit("DONE %d %d %.1f %.1f %s\n", generated, prompt_n, prompt_ms, decode_ms, finish);
        llama_sampler_free(smpl);
    }

    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
