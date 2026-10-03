// strata-qwen36 - Qwen3.6-35B-A3B engine that speaks the same stdin/stdout protocol as Strata's own engine, so
// serve/server.py (tokenizer, template, API, web app, MCP) drives it unchanged:
//
//   strata-qwen36 --serve --native model.gguf --max-context 32768 [--eos-ids 248046,248044]
//   strata-qwen36 --native tiny.gguf --selftest expected.txt          (end-to-end check against HuggingFace logits)
//
// stdout carries ONLY protocol lines (INFO / READY / PP / T / DONE / ERR); every log goes to stderr.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "qwen36_model.hpp"
#include "sampler.hpp"

namespace {

struct Args {
    std::string model, selftest;
    int batch_test = 0;
    int dump_n = 0;
    std::string dump_path;
    int max_ctx = 32768;
    std::vector<int> eos;
    bool serve = false;
    bool mtp = false;
    int mtp_k = 3, mtp_variant = 1;
    int kv_mode = 0;
    int ckpt_max = 12, ckpt_every = 8192;   // prompt snapshots: how many to keep, one every N prompt tokens
    bool kv_unified = false;
    std::string probe_prompt;   // --selftest-mtp file with comma-separated prompt token ids
    int probe_gen = 0;
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--serve") a.serve = true;
        else if (k == "--native" || k == "--model" || k == "-m") a.model = val();
        else if (k == "--max-context") a.max_ctx = std::atoi(val().c_str());
        else if (k == "--eos-ids") a.eos = parse_ids(val());
        else if (k == "--selftest") a.selftest = val();
        else if (k == "--selftest-batch") a.batch_test = std::atoi(val().c_str());
        else if (k == "--selftest-dump") { a.dump_n = std::atoi(val().c_str()); a.dump_path = val(); }
        else if (k == "--kv") {
            const std::string v = val();
            if (v == "fp16" || v == "f16") a.kv_mode = 0;
            else if (v == "int8" || v == "q8") a.kv_mode = 1;
            else { std::fprintf(stderr, "--kv must be fp16 or int8 (got '%s')\n", v.c_str()); std::exit(2); }
        }
        else if (k == "--kv-unified") a.kv_unified = true;
        else if (k == "--ckpt-max") a.ckpt_max = std::atoi(val().c_str());
        else if (k == "--ckpt-every") a.ckpt_every = std::max(256, std::atoi(val().c_str()));
        else if (k == "--mtp") a.mtp = true;
        else if (k == "--mtp-k") a.mtp_k = std::atoi(val().c_str());
        else if (k == "--mtp-variant") a.mtp_variant = std::atoi(val().c_str());
        else if (k == "--selftest-mtp") { a.probe_gen = std::atoi(val().c_str()); a.probe_prompt = val(); a.mtp = true; }
        // anything else is a Strata flag this engine does not need: ignored
    }
    return a;
}

// ------------------------------------------------------------------------------------------------ stdin plumbing
struct Commands {
    std::deque<std::string> q;
    std::mutex m;
    std::condition_variable cv;
    bool eof = false;
};
std::atomic<bool> g_stop{false};

void reader(Commands* c) {
    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line == "STOP") { g_stop = true; continue; }
        std::lock_guard<std::mutex> lk(c->m);
        c->q.push_back(line);
        c->cv.notify_one();
    }
    std::lock_guard<std::mutex> lk(c->m);
    c->eof = true;
    c->cv.notify_one();
}

void emit(const std::string& s) { std::fwrite(s.data(), 1, s.size(), stdout); std::fputc('\n', stdout); std::fflush(stdout); }

// ------------------------------------------------------------------------------------------------ one request
struct Engine {
    q36::Model& model;
    std::set<int> eos;
    std::vector<int> cached;  // tokens whose effect is in the model's state (prompt + generated, minus the last sample)
    int mtp_k = 3;
    int ckpt_every = 8192;

    void run(const std::string& line) {
        std::vector<std::string> f;
        {
            std::stringstream ss(line);
            std::string t;
            while (ss >> t) f.push_back(t);
        }
        if (f.size() < 3) { emit("ERR malformed GEN line"); return; }
        const int max_new = std::atoi(f[1].c_str());
        q36::SamplingParams sp;
        for (size_t i = 2; i + 1 < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq == std::string::npos) continue;
            const std::string k = f[i].substr(0, eq), v = f[i].substr(eq + 1);
            if (k == "temperature") sp.temperature = (float)std::atof(v.c_str());
            else if (k == "top_p") sp.top_p = (float)std::atof(v.c_str());
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = (float)std::atof(v.c_str());
            else if (k == "penalty_repeat") sp.repeat_penalty = (float)std::atof(v.c_str());
            else if (k == "penalty_freq") sp.freq_penalty = (float)std::atof(v.c_str());
            else if (k == "penalty_present") sp.presence_penalty = (float)std::atof(v.c_str());
            else if (k == "penalty_last_n") sp.penalty_last_n = std::atoi(v.c_str());
            else if (k == "seed") sp.seed = (uint64_t)std::strtoull(v.c_str(), nullptr, 10);
            // other keys (pcie_frac, spec_min_p, cvec, ...) belong to Strata's own engine
        }
        std::vector<int> prompt = parse_ids(f.back());
        const int n = (int)prompt.size();
        if (n == 0) { emit("ERR empty prompt"); return; }
        if (n + 1 > model.max_ctx()) { emit("ERR prompt is longer than the context"); return; }
        g_stop = false;

        // Reuse only when the new prompt EXTENDS what is already computed: the GDN state cannot be rewound.
        // Otherwise the longest saved snapshot that is a prefix of the new prompt (a template that rewrote the previous turn, an
        // edited history), else everything is recomputed.
        size_t start = 0;
        if (cached.size() < prompt.size() && std::equal(cached.begin(), cached.end(), prompt.begin()))
            start = cached.size();
        else {
            const int r = model.restore_best(prompt);
            if (r > 0) {
                start = (size_t)r;
                cached.assign(prompt.begin(), prompt.begin() + r);
                std::fprintf(stderr, "[qwen36] prefix cache: restored a snapshot at %d of %d prompt tokens\n", r, n);
            } else { model.reset(); cached.clear(); }
        }
        model.invalidate_snapshots(prompt, (int)start);
        const int reused = (int)start;
        int last_snap = (int)start;

        const double t0 = now_ms();
        double last_pp = t0;
        bool stopped = false;
        for (size_t i = start; i < prompt.size();) {
            const int nb = (int)std::min<size_t>(q36::Model::kMaxBatch, prompt.size() - i);
            const bool last = i + nb == prompt.size();
            if (nb == 1) model.forward(prompt[i], last);
            else model.forward_batch(&prompt[i], nb, last);   // block prefill
            cached.insert(cached.end(), prompt.begin() + i, prompt.begin() + i + nb);
            i += nb;
            if ((int)i - last_snap >= ckpt_every && i < prompt.size()) { model.snapshot(prompt.data(), (int)i); last_snap = (int)i; }
            if (!last) {
                const double t = now_ms();
                char b[160];
                std::snprintf(b, sizeof b, "PP %zu %d %.0f %.1f", i, n, t - t0, 1000.0 * (double)(i - start) / std::max(1.0, t - t0));
                emit(b);
                last_pp = t;
                if (g_stop) { stopped = true; break; }
            }
        }
        (void)last_pp;
        if (!stopped && (int)prompt.size() > last_snap) model.snapshot(prompt.data(), (int)prompt.size());   // end of the prompt
        const double prompt_ms = now_ms() - t0;
        {
            char b[160];
            std::snprintf(b, sizeof b, "PP %d %d %.0f %.1f", n, n, prompt_ms, 1000.0 * (double)(n - (int)start) / std::max(1.0, prompt_ms));
            emit(b);
        }

        int gen = 0, draft_n = 0, accepted = 0;
        std::string finish = "length";
        const double t1 = now_ms();
        if (stopped) finish = "stop";
        else {
            q36::Sampler sampler(sp);
            std::vector<int> hist = prompt;
            std::vector<float> lg;
            const bool spec = model.mtp_enabled();
            // emits one token; true when generation must end
            auto push = [&](int tok) {
                emit("T " + std::to_string(tok));
                ++gen;
                hist.push_back(tok);
                if (eos.count(tok)) { finish = "stop"; return true; }
                if (gen >= max_new) { finish = "length"; return true; }
                if (g_stop) { finish = "stop"; return true; }
                return false;
            };
            if (max_new > 0) {
                lg = model.logits();
                int tok = sampler.sample(lg, hist);
                bool done = push(tok);
                while (!done) {
                    if (model.pos() >= model.max_ctx()) { finish = "length"; break; }
                    if (!spec) {
                        model.forward(tok, true);
                        cached.push_back(tok);
                        lg = model.logits();
                        tok = sampler.sample(lg, hist);
                        done = push(tok);
                        continue;
                    }
                    // speculative round: tok is sampled and emitted but not yet fed to the model
                    std::vector<int> drafts;
                    model.mtp_draft(tok, std::min(mtp_k, max_new - gen), drafts);
                    std::vector<int> vt;
                    vt.push_back(tok);
                    vt.insert(vt.end(), drafts.begin(), drafts.end());
                    model.forward_verify(vt.data(), (int)vt.size());
                    draft_n += (int)drafts.size();
                    int keep = 0;
                    for (int i = 0; i < (int)vt.size(); ++i) {
                        lg.assign(model.verify_logits(i), model.verify_logits(i) + model.cfg().n_vocab);
                        const int s_i = sampler.sample(lg, hist);       // what the target itself would emit after vt[0..i]
                        keep = i + 1;
                        done = push(s_i);
                        tok = s_i;
                        if (done) break;
                        if (i + 1 < (int)vt.size() && s_i == vt[i + 1]) { ++accepted; continue; }
                        break;
                    }
                    model.commit(keep);
                    cached.insert(cached.end(), vt.begin(), vt.begin() + keep);
                }
            }
        }
        const double decode_ms = now_ms() - t1;
        char b[200];
        std::snprintf(b, sizeof b, "DONE %d %d %.1f %.1f %s %d %d %d", gen, n, prompt_ms, decode_ms, finish.c_str(), draft_n, accepted, reused);
        emit(b);
    }
};

int serve(q36::Model& model, const Args& a) {
    Engine eng{model, std::set<int>(a.eos.begin(), a.eos.end()), {}, a.mtp_k, a.ckpt_every};
    char info[256];
    std::snprintf(info, sizeof info, "INFO engine=qwen36 version=0.1 ctx=%d vram_mib=%zu layers=%d experts=%d",
                  model.max_ctx(), model.vram_bytes() >> 20, model.cfg().n_layer, model.cfg().n_expert);
    emit(info);
    emit("READY " + std::to_string(model.max_ctx()) + " stop");
    Commands cmds;
    std::thread th(reader, &cmds);
    th.detach();
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(cmds.m);
            cmds.cv.wait(lk, [&] { return !cmds.q.empty() || cmds.eof; });
            if (cmds.q.empty()) break;  // stdin closed
            line = cmds.q.front();
            cmds.q.pop_front();
        }
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) == 0) {
            try {
                eng.run(line);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[qwen36] error: %s\n", e.what());
                emit(std::string("ERR ") + e.what());
                try { model.reset(); } catch (...) {}
                eng.cached.clear();
            }
        } else if (line.rfind("GENI", 0) == 0 || line.rfind("ENC", 0) == 0) {
            emit("ERR images are not supported by strata-qwen36 yet");
        } else if (!line.empty()) {
            emit("ERR unknown command");
        }
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ selftest
int selftest(q36::Model& model, const std::string& path) {
    std::ifstream f(path);
    int n = 0, V = 0;
    if (!(f >> n >> V) || n <= 0 || V != model.cfg().n_vocab) {
        std::fprintf(stderr, "selftest: bad header in %s (vocab %d vs model %d)\n", path.c_str(), V, model.cfg().n_vocab);
        return 2;
    }
    std::vector<int> toks(n);
    for (int& t : toks) f >> t;
    std::vector<std::vector<float>> ref(n, std::vector<float>(V));
    for (auto& row : ref) for (float& x : row) f >> x;
    if (!f) { std::fprintf(stderr, "selftest: %s is truncated\n", path.c_str()); return 2; }
    model.reset();
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        model.forward(toks[i], true);
        const auto& lg = model.logits();
        double maxd = 0;
        int worst = 0, am = 0, amr = 0;
        for (int j = 0; j < V; ++j) {
            const double d = std::fabs((double)lg[j] - ref[i][j]);
            if (d > maxd) { maxd = d; worst = j; }
            if (lg[j] > lg[am]) am = j;
            if (ref[i][j] > ref[i][amr]) amr = j;
        }
        const bool ok = maxd <= 2e-3 + 2e-3 * std::fabs(ref[i][worst]) && am == amr;
        std::printf("pos %2d  token %5d  max|diff| %.3e (at %d)  argmax %s  %s\n", i, toks[i], maxd, worst,
                    am == amr ? "same" : "DIFFERENT", ok ? "OK" : "FAIL");
        bad += !ok;
    }
    std::printf(bad ? "SELFTEST FAILED (%d of %d positions)\n" : "SELFTEST PASSED (%d of %d bad)\n", bad, n);
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ batch self-consistency
// Comparison of two logit vectors: cosine, top-20 overlap, KL(a||b), max relative difference.
struct Cmp { double cos, kl, maxrel; int top20; bool same_arg; };
static Cmp compare_logits(const std::vector<float>& a, const std::vector<float>& b) {
    const int V = (int)a.size();
    double dot = 0, na = 0, nb = 0, maxrel = 0;
    for (int j = 0; j < V; ++j) {
        dot += (double)a[j] * b[j]; na += (double)a[j] * a[j]; nb += (double)b[j] * b[j];
        maxrel = std::max(maxrel, std::fabs((double)a[j] - b[j]) / (1.0 + std::fabs((double)a[j])));
    }
    auto softmax = [&](const std::vector<float>& l) {
        const double m = *std::max_element(l.begin(), l.end());
        std::vector<double> p(l.size()); double s = 0;
        for (size_t i = 0; i < l.size(); ++i) { p[i] = std::exp((double)l[i] - m); s += p[i]; }
        for (double& x : p) x /= s;
        return p;
    };
    const auto pa = softmax(a), pb = softmax(b);
    double kl = 0;
    for (int j = 0; j < V; ++j) if (pa[j] > 1e-12) kl += pa[j] * std::log(pa[j] / std::max(pb[j], 1e-300));
    auto top = [&](const std::vector<float>& l) {
        std::vector<int> idx(V); std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int x, int y) { return l[x] > l[y]; });
        idx.resize(20); return idx;
    };
    auto ta = top(a), tb = top(b);
    int ov = 0;
    for (int x : ta) ov += std::find(tb.begin(), tb.end(), x) != tb.end();
    return {dot / std::sqrt(na * nb + 1e-30), kl, maxrel, ov, ta[0] == tb[0]};
}

// Block prefill vs token-by-token on the SAME model.  Both are "correct", so the noise floor is measured with two block
// prefills of different splits; the verdict compares sequential-vs-batch against that floor.
int selftest_batch(q36::Model& model, int N) {
    const int V = model.cfg().n_vocab;
    N = std::max(2, std::min(N, model.max_ctx() - 48));
    std::vector<int> toks(N);
    uint64_t s = 88172645463325252ull;
    for (int& t : toks) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; t = (int)(s % (uint64_t)V); }
    auto argmax = [](const std::vector<float>& l) { return (int)(std::max_element(l.begin(), l.end()) - l.begin()); };
    struct Run { std::vector<float> after_prefill, after_decode; std::vector<int> greedy; double ms; };
    auto run = [&](const std::vector<int>& chunks) {
        Run r;
        model.reset();
        const double t0 = now_ms();
        for (int i = 0, ci = 0; i < N;) {
            const int want = chunks.empty() ? 1 : chunks[ci++ % chunks.size()];
            const int nb = std::min({want, N - i, (int)q36::Model::kMaxBatch});
            if (nb == 1) model.forward(toks[i], i + 1 == N);
            else model.forward_batch(&toks[i], nb, i + nb == N);
            i += nb;
        }
        r.ms = now_ms() - t0;
        r.after_prefill = model.logits();
        for (int i = 0; i < 32; ++i) { const int t = argmax(model.logits()); r.greedy.push_back(t); model.forward(t, true); }
        r.after_decode = model.logits();
        return r;
    };
    const Run seq = run({});
    const Run bA = run({1, 7, 128, 33, 128});
    const Run bB = run({16});
    const Run bC = run({128});
    auto show = [&](const char* name, const Cmp& c) {
        std::printf("  %-26s cos %.6f  KL %.3e  top20 %2d/20  maxrel %.3e  argmax %s\n", name, c.cos, c.kl, c.top20, c.maxrel,
                    c.same_arg ? "same" : "DIFFERENT");
    };
    std::printf("prompt %d token: sequenziale %.0f ms (%.1f tok/s) | blocchi irregolari %.0f ms (%.1f tok/s, %.2fx)\n", N,
                seq.ms, 1000.0 * N / seq.ms, bA.ms, 1000.0 * N / bA.ms, seq.ms / bA.ms);
    std::printf("dopo il prefill:\n");
    const Cmp s_A = compare_logits(seq.after_prefill, bA.after_prefill), A_B = compare_logits(bA.after_prefill, bB.after_prefill),
              B_C = compare_logits(bB.after_prefill, bC.after_prefill);
    show("sequenziale vs batch A", s_A);
    show("batch A vs batch B (rumore)", A_B);
    show("batch B vs batch C (rumore)", B_C);
    std::printf("dopo 32 passi di decode:\n");
    show("sequenziale vs batch A", compare_logits(seq.after_decode, bA.after_decode));
    show("batch A vs batch B (rumore)", compare_logits(bA.after_decode, bB.after_decode));
    int agree = 0;
    for (int i = 0; i < 32; ++i) agree += seq.greedy[i] == bA.greedy[i];
    std::printf("token greedy uguali tra sequenziale e batch A: %d/32\n", agree);
    const double noise_kl = std::max({A_B.kl, B_C.kl, 1e-6});
    const bool dist_ok = s_A.kl < 5e-2 && s_A.top20 >= 17 && s_A.same_arg && s_A.cos > 0.995;
    const bool vs_noise = s_A.kl <= 10.0 * noise_kl;
    std::printf("rumore KL tra batch: %.3e  |  seq-vs-batch KL: %.3e  (%s)\n", noise_kl, s_A.kl,
                vs_noise ? "dello stesso ordine del rumore" : "MOLTO PEGGIORE del rumore");
    const bool ok = dist_ok && vs_noise && agree >= 28;
    std::printf(ok ? "SELFTEST-BATCH PASSED\n" : "SELFTEST-BATCH FAILED\n");
    return ok ? 0 : 1;
}

// Writes the logits after an N-token block prefill and 16 greedy decode steps to a file (float32, V values per stage) plus
// the greedy tokens, so two runs of the same binary with different settings (Q36_GROUPED=0 / 1) can be compared with
// tests/compare_dumps.py.  Uses real-looking structure: a fixed pseudo-random prompt, the same for every run.
int selftest_dump(q36::Model& model, int N, const std::string& path) {
    const int V = model.cfg().n_vocab;
    N = std::max(2, std::min(N, model.max_ctx() - 32));
    std::vector<int> toks(N);
    uint64_t s = 88172645463325252ull;
    for (int& t : toks) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; t = (int)(s % (uint64_t)V); }
    model.reset();
    const double t0 = now_ms();
    for (int i = 0; i < N;) {
        const int nb = std::min(N - i, (int)q36::Model::kMaxBatch);
        if (nb == 1) model.forward(toks[i], i + 1 == N);
        else model.forward_batch(&toks[i], nb, i + nb == N);
        i += nb;
    }
    const double pre_ms = now_ms() - t0;
    std::vector<float> out = model.logits();
    std::vector<int> greedy;
    const double t1 = now_ms();
    for (int i = 0; i < 16; ++i) {
        const auto& l = model.logits();
        const int t = (int)(std::max_element(l.begin(), l.end()) - l.begin());
        greedy.push_back(t);
        model.forward(t, true);
    }
    const double dec_ms = now_ms() - t1;
    out.insert(out.end(), model.logits().begin(), model.logits().end());
    std::ofstream f(path, std::ios::binary);
    const int32_t hdr[2] = {V, (int32_t)greedy.size()};
    f.write((const char*)hdr, sizeof hdr);
    f.write((const char*)greedy.data(), greedy.size() * sizeof(int));
    f.write((const char*)out.data(), out.size() * sizeof(float));
    std::printf("dump %s: prefill %d tokens %.0f ms (%.1f tok/s), 16 decode steps %.0f ms (%.1f tok/s)\n", path.c_str(), N, pre_ms,
                1000.0 * N / pre_ms, dec_ms, 16000.0 / dec_ms);
    return f ? 0 : 1;
}


// MTP convention probe: greedy-generates `gen` tokens after the prompt, then measures for each of the 8 input variants how often
// the head's argmax equals the real next-next token.  The right variant scores far above the others (typically > 50%).
int selftest_mtp(q36::Model& model, int gen, const std::string& prompt_file) {
    if (!model.mtp_enabled()) { std::fprintf(stderr, "selftest-mtp: the MTP head is not loaded\n"); return 2; }
    std::ifstream f(prompt_file);
    std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int> seq = parse_ids(txt);
    if (seq.size() < 4) { std::fprintf(stderr, "selftest-mtp: %s must hold >= 4 comma-separated token ids\n", prompt_file.c_str()); return 2; }
    const int plen = (int)seq.size();
    gen = std::max(32, std::min(gen, model.max_ctx() - plen - 8));
    model.reset();
    for (int i = 0; i < plen;) {
        const int nb = std::min(plen - i, (int)q36::Model::kMaxBatch);
        if (nb == 1) model.forward(seq[i], true); else model.forward_batch(&seq[i], nb, i + nb == plen);
        i += nb;
    }
    for (int i = 0; i < gen; ++i) {
        const auto& l = model.logits();
        const int t = (int)(std::max_element(l.begin(), l.end()) - l.begin());
        seq.push_back(t);
        model.forward(t, true);
    }
    std::printf("sequenza: %d token di prompt + %d generati (greedy)\n", plen, gen);
    int best = 0; double best_acc = -1;
    for (int v = 0; v < 8; ++v) {
        model.set_mtp_variant(v);
        int cnt = 0;
        const double acc = model.mtp_probe(seq, plen - 1, &cnt);
        std::printf("  variante %d (hidden %s, ordine %s, norm %s): accuratezza %.1f%% su %d posizioni\n", v,
                    (v & 1) ? "post-norm" : "pre-norm", (v & 2) ? "[h|e]" : "[e|h]", (v & 4) ? "+1" : "cosi' com'e'", 100.0 * acc, cnt);
        if (acc > best_acc) { best_acc = acc; best = v; }
    }
    std::printf("MIGLIORE: --mtp-variant %d (%.1f%%)%s\n", best, 100.0 * best_acc,
                best_acc < 0.2 ? "  ATTENZIONE: nessuna variante e' buona, la testa MTP o i tensori non corrispondono" : "");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.model.empty() || (!a.serve && a.selftest.empty() && a.batch_test == 0 && a.dump_n == 0 && a.probe_gen == 0)) {
        std::fprintf(stderr, "usage: %s --serve --native model.gguf [--max-context N] [--eos-ids a,b]\n"
                             "       %s --native model.gguf --selftest expected.txt\n"
                             "       %s --native model.gguf --selftest-batch 300\n"
                             "       %s --native model.gguf --selftest-mtp 200 prompt_ids.txt   (pick the MTP variant)\n"
                             "  options: --mtp [--mtp-k 3] [--mtp-variant N]  --kv fp16|int8  --kv-unified  --ckpt-max 12  --ckpt-every 8192\n", argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    try {
        if (a.eos.empty() && a.serve) {
            strata::GgufModel g = strata::GgufModel::open(a.model);
            if (const strata::MetaValue* v = g.meta().get("tokenizer.ggml.eos_token_id")) a.eos.push_back((int)v->u);
            if (a.eos.empty()) std::fprintf(stderr, "[qwen36] no EOS id known (pass --eos-ids): generation stops only on length or STOP\n");
        }
        q36::Model model(a.model, a.max_ctx, a.mtp, a.mtp_k, a.mtp_variant, a.kv_mode, a.kv_unified);
        model.set_snapshot_limit(a.ckpt_max);
        if (a.probe_gen > 0) return selftest_mtp(model, a.probe_gen, a.probe_prompt);
        if (a.batch_test > 0) return selftest_batch(model, a.batch_test);
        if (a.dump_n > 0) return selftest_dump(model, a.dump_n, a.dump_path);
        return a.selftest.empty() ? serve(model, a) : selftest(model, a.selftest);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[qwen36] fatal: %s\n", e.what());
        std::printf("ERR %s\n", e.what());
        return 1;
    }
}
