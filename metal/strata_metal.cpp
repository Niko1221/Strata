// strata-metal: Strata's engine line protocol (serve/server.py <-> engine) over llama.cpp's Metal backend.
//
// macOS / Apple Silicon only.  The server starts it as `strata-metal --serve <args>` and talks to it over stdin/stdout
// exactly as it talks to the CUDA engine (src/program/generate.cpp): INFO/READY at start, then GEN, STOP, SAVE,
// RESTORE, VRAM and QUIT.  docs/MACOS_PLAN.md says why this is a protocol adapter and not a port of the CUDA engine.
//
// The model is the original GGUF (no Strata pack): llama.cpp's qwen4exp architecture, every tensor in unified memory.
//
// Conversation reuse.  The gated-delta-net layers keep a recurrent state that cannot be cut back to any position, so a
// request whose ids leave the held ones goes back to a checkpoint: the recurrent part of the state (the attention KV
// is cut with seq_rm), taken while reading a prompt just before each <|im_start|> - the message boundaries a chat
// template's re-render leaves intact.  Never a state that does not match the ids: no checkpoint, then the whole prompt.
//
// Not here (phase A): MTP drafts, --batch slots, images (GENI), the expert cache (VRAM).
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "0"
#endif
#ifndef STRATA_LLAMA_COMMIT
#define STRATA_LLAMA_COMMIT "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---------------------------------------------------------------------------------------------- stdin
// STOP has to reach a running request, so stdin is read on its own thread.  A GEN line clears the stop flag before
// it is queued: a STOP the server sent for an earlier request (after that one's DONE) cannot cancel the next.
std::atomic<bool> g_stop{false};
std::mutex g_mu;
std::condition_variable g_cv;
std::deque<std::string> g_lines;

void read_stdin() {
    std::string line;
    int c;
    for (;;) {
        line.clear();
        while ((c = std::fgetc(stdin)) != EOF && c != '\n') line.push_back((char) c);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (c == EOF && line.empty()) line = "QUIT";      // the server is gone
        if (line == "STOP") { g_stop = true; continue; }
        if (line.rfind("GEN", 0) == 0) g_stop = false;
        { std::lock_guard<std::mutex> lk(g_mu); g_lines.push_back(line); }
        g_cv.notify_one();
        if (c == EOF) return;
    }
}

std::string next_line() {
    std::unique_lock<std::mutex> lk(g_mu);
    g_cv.wait(lk, [] { return !g_lines.empty(); });
    std::string s = std::move(g_lines.front());
    g_lines.pop_front();
    return s;
}

void out(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void out(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------- options
struct Options {
    std::string gguf;
    int64_t max_context = 32768;
    std::string kv = "int8";
    std::vector<llama_token> eos = {248044, 248046};   // <|endoftext|>, <|im_end|>: the CUDA engine's --eos-ids
    int n_batch = 2048;
    int checkpoints = 8;
    int threads = 0;
};

bool parse_ids(const char* s, std::vector<llama_token>& v) {
    v.clear();
    while (*s) {
        char* e = nullptr;
        long x = std::strtol(s, &e, 10);
        if (e == s || x < 0) return false;
        v.push_back((llama_token) x);
        s = *e == ',' ? e + 1 : e;
        if (*e && *e != ',') return false;
    }
    return !v.empty();
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-metal: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--serve") continue;
        else if (a == "--gguf") o.gguf = val("--gguf");
        else if (a == "--max-context") o.max_context = std::atoll(val("--max-context"));
        else if (a == "--kv") o.kv = val("--kv");
        else if (a == "--prefill-batch") o.n_batch = std::atoi(val("--prefill-batch"));
        else if (a == "--checkpoints") o.checkpoints = std::max(0, std::atoi(val("--checkpoints")));
        else if (a == "--threads") o.threads = std::atoi(val("--threads"));
        else if (a == "--eos-ids") {
            if (!parse_ids(val("--eos-ids"), o.eos)) { std::fprintf(stderr, "strata-metal: bad --eos-ids\n"); return false; }
        } else if (a.rfind("--", 0) == 0) {
            // a flag of the CUDA engine (a shared config): said once, not fatal.  Its value, if any, is skipped too.
            std::fprintf(stderr, "strata-metal: %s is not used by the Metal engine (ignored)\n", a.c_str());
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) ++i;
        }
    }
    if (o.gguf.empty()) { std::fprintf(stderr, "strata-metal: --gguf <the model's first GGUF shard> is required\n"); return false; }
    if (o.max_context < 512) { std::fprintf(stderr, "strata-metal: --max-context must be 512 or more\n"); return false; }
    if (o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4" && o.kv != "f16") {
        std::fprintf(stderr, "strata-metal: --kv takes int8, q4_0, k8v4 or f16\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- the engine
struct Checkpoint {
    size_t n;                    // tokens the state covers: live[0, n)
    std::vector<uint8_t> data;   // the recurrent part (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
};

struct Engine {
    Options o;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_token im_start = -1;
    std::vector<llama_token> live;          // what sequence 0 holds
    std::deque<Checkpoint> ckpts;           // ascending n

    llama_memory_t mem() const { return llama_get_memory(ctx); }

    bool load(std::string& err) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = -1;               // everything in unified memory
        model = llama_model_load_from_file(o.gguf.c_str(), mp);
        if (!model) { err = "could not load " + o.gguf; return false; }
        vocab = llama_model_get_vocab(model);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) o.max_context;
        cp.n_batch = (uint32_t) o.n_batch;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.type_k = o.kv == "f16" ? GGML_TYPE_F16 : o.kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        cp.type_v = o.kv == "f16" ? GGML_TYPE_F16 : o.kv == "int8" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
        if (o.threads > 0) cp.n_threads = cp.n_threads_batch = o.threads;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { err = "could not create the context (too long a --max-context for this Mac's memory?)"; return false; }
        llama_token t[4];
        const char* s = "<|im_start|>";
        if (llama_tokenize(vocab, s, (int32_t) std::strlen(s), t, 4, false, true) == 1) im_start = t[0];
        return true;
    }

    void checkpoint(size_t n) {
        if (o.checkpoints == 0) return;
        Checkpoint c{n, {}};
        c.data.resize(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (llama_state_seq_get_data_ext(ctx, c.data.data(), c.data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0)
            return;
        while (!ckpts.empty() && ckpts.back().n >= n) ckpts.pop_back();
        ckpts.push_back(std::move(c));
        while ((int) ckpts.size() > o.checkpoints) ckpts.pop_front();
    }

    // Back to at most p held tokens: the latest checkpoint at or below p, else nothing held.
    void rollback(size_t p) {
        while (!ckpts.empty() && ckpts.back().n > p) ckpts.pop_back();
        if (!ckpts.empty()) {
            const Checkpoint& c = ckpts.back();
            if (llama_state_seq_set_data_ext(ctx, c.data.data(), c.data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) &&
                llama_memory_seq_rm(mem(), 0, (llama_pos) c.n, -1)) {
                live.resize(c.n);
                return;
            }
            std::fprintf(stderr, "strata-metal: checkpoint at %zu could not be restored; reading the prompt again\n", c.n);
        }
        llama_memory_clear(mem(), true);
        live.clear();
        ckpts.clear();
    }

    // Feed toks at the end of the sequence.  false: llama_decode failed (err).
    bool feed(llama_token* toks, int n, std::string& err) {
        const int rc = llama_decode(ctx, llama_batch_get_one(toks, n));
        if (rc != 0) { err = "llama_decode failed (" + std::to_string(rc) + ")"; return false; }
        live.insert(live.end(), toks, toks + n);
        return true;
    }

    llama_sampler* sampler(const std::string& keys) {
        float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f, rep = 1.0f, freq = 0.0f, pres = 0.0f;
        int top_k = 20, last_n = 64;
        unsigned long long seed = LLAMA_DEFAULT_SEED;
        size_t i = 0;
        while (i < keys.size()) {
            size_t j = keys.find(' ', i);
            if (j == std::string::npos) j = keys.size();
            std::string kv = keys.substr(i, j - i);
            i = j + 1;
            size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            std::string k = kv.substr(0, eq);
            const char* v = kv.c_str() + eq + 1;
            if (k == "temperature") temperature = std::strtof(v, nullptr);
            else if (k == "top_p") top_p = std::strtof(v, nullptr);
            else if (k == "top_k") top_k = std::atoi(v);
            else if (k == "min_p") min_p = std::strtof(v, nullptr);
            else if (k == "penalty_repeat") rep = std::strtof(v, nullptr);
            else if (k == "penalty_freq") freq = std::strtof(v, nullptr);
            else if (k == "penalty_present") pres = std::strtof(v, nullptr);
            else if (k == "penalty_last_n") last_n = std::atoi(v);
            else if (k == "seed") seed = std::strtoull(v, nullptr, 10);
        }
        llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (rep != 1.0f || freq != 0.0f || pres != 0.0f)
            llama_sampler_chain_add(s, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), last_n, rep, freq, pres));
        if (temperature <= 0.0f) {          // absent keys: greedy, as the CUDA engine
            llama_sampler_chain_add(s, llama_sampler_init_greedy());
            return s;
        }
        llama_sampler_chain_add(s, llama_sampler_init_top_k(top_k > 0 ? top_k : 64));
        if (top_p < 1.0f) llama_sampler_chain_add(s, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(s, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(s, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist((uint32_t) seed));
        return s;
    }

    // GEN <max_new> [key=value ...] <id,id,...>
    void gen(const std::string& line) {
        const size_t sp = line.rfind(' ');
        std::vector<llama_token> ids;
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        if (sp == std::string::npos || sp < 4 || max_new < 0 || !parse_ids(line.c_str() + sp + 1, ids)) {
            out("ERR expected: GEN <max_new> <id,id,...>\n");
            return;
        }
        const std::string keys(static_cast<const char*>(endp), line.c_str() + sp);
        const int64_t n = (int64_t) ids.size();
        if (n >= o.max_context) {
            out("ERR the prompt (%lld tokens) does not fit the context (%lld)\n", (long long) n, (long long) o.max_context);
            return;
        }
        const llama_token n_vocab = llama_vocab_n_tokens(vocab);
        for (llama_token t : ids)
            if (t >= n_vocab) { out("ERR token id %d is outside the vocabulary (%d)\n", t, n_vocab); return; }

        // reuse: the held ids that start the prompt, at least one token left to read (it gives the logits)
        size_t p = 0;
        while (p < live.size() && p < ids.size() && live[p] == ids[p]) ++p;
        p = std::min(p, ids.size() - 1);
        if (p < live.size()) rollback(p);
        const size_t resume = live.size();
        out("RESUME %zu\n", resume);

        std::string err;
        const auto t0 = Clock::now();
        const char* finish = "length";
        bool cancelled = false;
        // the prompt in chunks of n_batch, each also ending at a message boundary (a checkpoint there)
        size_t at = resume;
        while (at < ids.size()) {
            size_t end = std::min(ids.size(), at + (size_t) o.n_batch);
            size_t b = at + 1;
            while (b < end && ids[b] != im_start) ++b;
            const bool boundary = b < end;
            if (boundary) end = b;
            if (!feed(ids.data() + at, (int) (end - at), err)) { rollback(0); out("ERR %s\n", err.c_str()); return; }
            at = end;
            if (boundary) checkpoint(at);
            const double ms = ms_since(t0);
            out("PP %zu %lld %.0f %.1f\n", at, (long long) n, ms, (at - resume) / std::max(1e-3, ms / 1000.0));
            if (g_stop && at < ids.size()) { cancelled = true; finish = "cancel"; break; }
        }
        const double prompt_ms = ms_since(t0);
        const size_t read_n = at - resume;

        int64_t produced = 0;
        const auto t1 = Clock::now();
        if (!cancelled) {
            llama_sampler* s = sampler(keys);
            while (produced < max_new) {
                if (g_stop) { finish = "cancel"; break; }
                if ((int64_t) live.size() >= o.max_context) { finish = "length"; break; }
                llama_token t = llama_sampler_sample(s, ctx, -1);
                out("T %d\n", t);
                ++produced;
                // the token is fed even when it ends the turn: the next request's prompt carries it, and a sequence
                // that already holds it needs no checkpoint to go on
                if (!feed(&t, 1, err)) { llama_sampler_free(s); rollback(0); out("ERR %s\n", err.c_str()); return; }
                if (std::find(o.eos.begin(), o.eos.end(), t) != o.eos.end() || llama_vocab_is_eog(vocab, t)) {
                    finish = "stop";
                    break;
                }
            }
            llama_sampler_free(s);
        }
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      <hits> <lookups> <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
        out("DONE %lld %lld %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n", (long long) produced, (long long) n, prompt_ms,
            ms_since(t1), finish, resume, read_n);
    }

    void save(const std::string& path) {
        const auto t0 = Clock::now();
        const size_t bytes = llama_state_seq_save_file(ctx, path.c_str(), 0, live.data(), live.size());
        if (bytes == 0) { out("SERR io 0 could not write %s\n", path.c_str()); return; }
        out("SAVED %zu %zu %.1f\n", live.size(), bytes, ms_since(t0));
    }

    void restore(const std::string& path) {
        const auto t0 = Clock::now();
        std::vector<llama_token> toks((size_t) o.max_context);
        size_t n = 0;
        llama_memory_clear(mem(), true);
        live.clear();
        ckpts.clear();
        const size_t bytes = llama_state_seq_load_file(ctx, path.c_str(), 0, toks.data(), toks.size(), &n);
        if (bytes == 0) {
            llama_memory_clear(mem(), true);
            out("SERR invalid 0 could not read a session for this model and context from %s\n", path.c_str());
            return;
        }
        live.assign(toks.begin(), toks.begin() + (long) n);
        out("RESTORED %zu %zu %.1f\n", n, bytes, ms_since(t0));
    }
};

}  // namespace

int main(int argc, char** argv) {
    Engine e;
    if (!parse_args(argc, argv, e.o)) return 2;
    llama_backend_init();
    std::string err;
    if (!e.load(err)) {
        out("ERR %s\n", err.c_str());
        return 1;
    }
    const char* kv = e.o.kv == "int8" ? "int8" : e.o.kv.c_str();
    out("INFO engine=" STRATA_VERSION "-metal backend=metal llama=" STRATA_LLAMA_COMMIT " context=%lld kv=%s "
        "batch_slots=0 checkpoints=%d im_start=%d\n", (long long) e.o.max_context, kv, e.o.checkpoints, e.im_start);
    out("READY %lld stop\n", (long long) e.o.max_context);
    std::thread(read_stdin).detach();
    for (;;) {
        const std::string line = next_line();
        if (line == "QUIT") break;
        if (line.rfind("GENI ", 0) == 0) out("ERR images are not supported by the Metal engine yet\n");
        else if (line.rfind("GEN ", 0) == 0) e.gen(line);
        else if (line.rfind("SAVE ", 0) == 0) e.save(line.substr(5));
        else if (line.rfind("RESTORE ", 0) == 0) e.restore(line.substr(8));
        else if (line.rfind("VRAM", 0) == 0) out("ERR the Metal engine has no expert cache to resize\n");
        else if (!line.empty()) out("ERR unknown command: %.40s\n", line.c_str());
    }
    llama_free(e.ctx);
    llama_model_free(e.model);
    llama_backend_free();
    return 0;
}
