// engine/src/inkling2/ink_tail_kl.h - teacher-forced KL harness for the opt-in router tail skip (lane TAILSKIP, 2026-11-26).
//
//   strata-inkling --native MODEL --max-context N --tail-kl CORPUS.tsv --tail-cfgs CFGS.tsv --tail-out DIR [--tail-chunk 2048] [--tail-only NAME,...]
//
// CORPUS.tsv: one sequence per line:  name <TAB> stride <TAB> ids (comma separated) [<TAB> score ranges "a-b,c-d"]
//   The rows scored are the input positions i (the logits that predict token i + 1): every stride-th position from 16 on, or, when ranges are
//   given, every position inside them (stride then thins them: every stride-th of the range positions).
// CFGS.tsv: one configuration per line: name <TAB> spec [<TAB> chunk]; spec is a set_tail_skip() spec ("off" = exact). The baseline (exact,
//   chunk = --tail-chunk) is run first for every sequence on the same engine, in the same process, and every configuration is compared against it.
// One pass per (sequence, configuration): slot 0 reset, the sequence fed in chunks, the logits of the scored rows compared with the baseline's.
// Output DIR/rows.tsv (sequence, name, number of rows, tokens, offset of its rows in the per-config files) and DIR/cfg_<name>.bin
// (records of 12 bytes per scored row: float KL, float dNLL, uint32 top1-match) and DIR/summary.jsonl (per config and sequence: mean KL, tail-skip
// counters). Nothing here changes the default path.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"
#include "cuda/ink_tail_skip.cuh"

namespace ink2 {

struct TailKlSeq {
    std::string name;
    int stride = 1;
    std::vector<int32_t> ids;
    std::vector<int> rows;              // input positions scored (ascending)
};

struct TailKlRef {                      // per scored row of the baseline
    std::vector<float> p;               // [V] probabilities (float)
    double S = 1.0;                     // sum of p as stored
    double H = 0.0;                     // sum p ln p
    double nll = 0.0;                   // -ln p[target]
    int top1 = 0;
};

static inline bool tkl_split(const std::string& s, char sep, std::vector<std::string>& out) {
    out.clear();
    size_t i = 0;
    while (true) {
        size_t j = s.find(sep, i);
        if (j == std::string::npos) { out.push_back(s.substr(i)); break; }
        out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return true;
}

static inline bool tkl_load_corpus(const std::string& path, std::vector<TailKlSeq>& seqs, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open corpus " + path; return false; }
    std::string line;
    std::vector<std::string> cols;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        tkl_split(line, '\t', cols);
        if (cols.size() < 3) { err = "corpus line needs name, stride, ids"; return false; }
        TailKlSeq s;
        s.name = cols[0];
        s.stride = std::max(1, std::atoi(cols[1].c_str()));
        {
            const char* p = cols[2].c_str();
            char* e = nullptr;
            while (*p) {
                long v = std::strtol(p, &e, 10);
                if (e == p) break;
                s.ids.push_back((int32_t)v);
                p = (*e == ',') ? e + 1 : e;
            }
        }
        const int n = (int)s.ids.size();
        if (n < 32) { err = "sequence too short: " + s.name; return false; }
        if (cols.size() >= 4 && !cols[3].empty()) {
            std::vector<std::string> rg;
            tkl_split(cols[3], ',', rg);
            // a range is "a-b" (every stride-th position of it) or "a-b/s" (every s-th)
            for (const auto& r : rg) {
                int a = 0, b = 0, st = s.stride;
                if (std::sscanf(r.c_str(), "%d-%d/%d", &a, &b, &st) < 2) continue;
                st = std::max(1, st);
                for (int i = std::max(a, 0), k = 0; i <= b && i < n - 1; ++i, ++k)
                    if (k % st == 0) s.rows.push_back(i);
            }
            std::sort(s.rows.begin(), s.rows.end());
            s.rows.erase(std::unique(s.rows.begin(), s.rows.end()), s.rows.end());
        } else {
            for (int i = 16; i < n - 1; i += s.stride) s.rows.push_back(i);
        }
        seqs.push_back(std::move(s));
    }
    if (seqs.empty()) { err = "empty corpus"; return false; }
    return true;
}

template <class F>
static inline void tkl_parallel_for(int n, F&& fn) {
    const int nt = (int)std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    if (n <= 1 || nt == 1) { for (int i = 0; i < n; ++i) fn(i); return; }
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < std::min(nt, n); ++t)
        th.emplace_back([&] { for (int i = next.fetch_add(1); i < n; i = next.fetch_add(1)) fn(i); });
    for (auto& t : th) t.join();
}

// One pass of a sequence on slot 0: chunked feed, logits only for the scored rows. cb(first_row_index, count, logits) is called per chunk.
template <class CB>
static inline bool tkl_run_seq(InkBackend& be, const TailKlSeq& s, int chunk, std::vector<float>& buf, int V, std::string& err, CB&& cb) {
    be.reset_slot(0);
    const int n = (int)s.ids.size();
    StepBatch batch;
    size_t r0 = 0;
    // a small chunk (one-token decode steps, verify-window sized chunks) is only used from the first scored row on; the prompt before it goes in
    // 2048-token chunks, as a real run prefills its prompt before it decodes
    const int first_row = s.rows.empty() ? n : s.rows.front();
    for (int i0 = 0, i1 = 0; i0 < n; i0 = i1) {
        const bool bulk = chunk < 64 && i0 < first_row;
        i1 = std::min(n, i0 + (bulk ? 2048 : chunk));
        if (bulk) i1 = std::min(i1, first_row);
        batch.clear();
        size_t r1 = r0;
        for (int i = i0; i < i1; ++i) {
            const bool want = r1 < s.rows.size() && s.rows[r1] == i;
            if (want) ++r1;
            batch.push(0, i, s.ids[i], want);
        }
        const size_t nw = r1 - r0;
        if (nw > 0 && buf.size() < nw * (size_t)V) buf.resize(nw * (size_t)V);
        if (!be.forward_batch(batch, nw ? buf.data() : nullptr, nullptr, err)) return false;
        if (nw) cb(r0, (int)nw, buf.data());
        r0 = r1;
    }
    return true;
}

static inline int cmd_tail_kl(const std::string& model_path, const std::string& corpus_path, const std::string& cfgs_path, const std::string& out_dir,
                              int base_chunk, const std::string& only, const BackendOpts& bopts, bool force_cpu) {
    std::string err;
    std::vector<TailKlSeq> seqs;
    if (!tkl_load_corpus(corpus_path, seqs, err)) { fprintf(stderr, "tail-kl: %s\n", err.c_str()); return 2; }
    struct Cfg { std::string name, spec; int chunk; };
    std::vector<Cfg> cfgs;
    {
        std::ifstream f(cfgs_path);
        if (!f) { fprintf(stderr, "tail-kl: cannot open %s\n", cfgs_path.c_str()); return 2; }
        std::string line;
        std::vector<std::string> cols;
        std::vector<std::string> only_names;
        if (!only.empty()) tkl_split(only, ',', only_names);
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            tkl_split(line, '\t', cols);
            if (cols.size() < 2) continue;
            if (!only_names.empty() && std::find(only_names.begin(), only_names.end(), cols[0]) == only_names.end()) continue;
            cfgs.push_back({cols[0], cols[1], cols.size() >= 3 ? std::max(1, std::atoi(cols[2].c_str())) : base_chunk});
        }
    }
    if (cfgs.empty()) { fprintf(stderr, "tail-kl: no configurations\n"); return 2; }

    InkModelFile mf;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int V = mf.m.c.vocab_unpadded;
    size_t max_len = 0;
    for (const auto& s : seqs) max_len = std::max(max_len, s.ids.size());
    if ((int)max_len > bopts.max_ctx) { fprintf(stderr, "tail-kl: a sequence of %zu tokens needs --max-context >= %zu (now %d)\n", max_len, max_len, bopts.max_ctx); return 2; }
    if (!backend->set_tail_skip("off", err)) { fprintf(stderr, "tail-kl: %s\n", err.c_str()); return 1; }

    std::string mk = "mkdir -p \"" + out_dir + "\"";
    if (std::system(mk.c_str()) != 0) {}
    FILE* fsum = std::fopen((out_dir + "/summary.jsonl").c_str(), "ab");
    FILE* frows = std::fopen((out_dir + "/rows.tsv").c_str(), "wb");
    if (!fsum || !frows) { fprintf(stderr, "tail-kl: cannot write into %s\n", out_dir.c_str()); return 1; }
    std::vector<FILE*> fbin(cfgs.size(), nullptr);
    for (size_t c = 0; c < cfgs.size(); ++c) {
        fbin[c] = std::fopen((out_dir + "/cfg_" + cfgs[c].name + ".bin").c_str(), "ab");
        if (!fbin[c]) { fprintf(stderr, "tail-kl: cannot write cfg_%s.bin\n", cfgs[c].name.c_str()); return 1; }
    }
    // the counters need stats=1 in the spec; every spec of a configuration gets it
    std::vector<float> buf;
    std::vector<uint64_t> st;
    size_t row_offset = 0;
    for (const auto& s : seqs) {
        const int R = (int)s.rows.size();
        std::vector<TailKlRef> ref(R);
        auto t0 = std::chrono::steady_clock::now();
        if (!backend->set_tail_skip("off", err)) { fprintf(stderr, "tail-kl: %s\n", err.c_str()); return 1; }
        // baseline pass
        bool ok = tkl_run_seq(*backend, s, base_chunk, buf, V, err, [&](size_t r0, int nw, const float* lg) {
            tkl_parallel_for(nw, [&](int k) {
                const float* q = lg + (size_t)k * V;
                TailKlRef& rf = ref[r0 + k];
                float m = q[0]; int b = 0;
                for (int v = 1; v < V; ++v) if (q[v] > m) { m = q[v]; b = v; }
                double z = 0.0;
                for (int v = 0; v < V; ++v) z += std::exp((double)q[v] - m);
                rf.p.resize(V);
                double S = 0.0, H = 0.0;
                const double lz = std::log(z);
                for (int v = 0; v < V; ++v) { const double pv = std::exp((double)q[v] - m - lz); rf.p[v] = (float)pv; }
                for (int v = 0; v < V; ++v) { const double pv = rf.p[v]; S += pv; if (pv > 0.0) H += pv * std::log(pv); }
                rf.S = S; rf.H = H; rf.top1 = b;
                const int tgt = s.ids[s.rows[r0 + k] + 1];
                const double pt = rf.p[tgt];
                rf.nll = -std::log(std::max(pt, 1e-30));
            });
        });
        if (!ok) { fprintf(stderr, "tail-kl: baseline %s: %s\n", s.name.c_str(), err.c_str()); return 1; }
        const double t_base = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(frows, "%s\t%d\t%zu\t%zu\n", s.name.c_str(), R, s.ids.size(), row_offset);
        fflush(frows);
        fprintf(stderr, "[tail-kl] %s: %zu tokens, %d rows, baseline %.1fs\n", s.name.c_str(), s.ids.size(), R, t_base);

        for (size_t c = 0; c < cfgs.size(); ++c) {
            const Cfg& cf = cfgs[c];
            std::string spec = cf.spec;
            if (spec != "off" && spec.find("stats=") == std::string::npos) spec += ",stats=1";
            if (spec == "off") spec = "off";
            if (!backend->set_tail_skip(spec, err)) { fprintf(stderr, "tail-kl: spec %s: %s\n", cf.spec.c_str(), err.c_str()); return 1; }
            if (spec != "off") { std::vector<uint64_t> dummy; backend->tail_stats(dummy, true); }
            std::vector<float> kl(R), dn(R);
            std::vector<uint32_t> t1(R);
            auto tc0 = std::chrono::steady_clock::now();
            ok = tkl_run_seq(*backend, s, cf.chunk, buf, V, err, [&](size_t r0, int nw, const float* lg) {
                tkl_parallel_for(nw, [&](int k) {
                    const float* q = lg + (size_t)k * V;
                    const TailKlRef& rf = ref[r0 + k];
                    float m = q[0]; int b = 0;
                    for (int v = 1; v < V; ++v) if (q[v] > m) { m = q[v]; b = v; }
                    double z = 0.0, dot = 0.0;
                    const float* p = rf.p.data();
                    for (int v = 0; v < V; ++v) { z += (double)expf(q[v] - m); dot += (double)p[v] * ((double)q[v] - m); }
                    const double lz = std::log(z);
                    // sum p (ln p - (q - m - lz)) = H - dot + S lz
                    const double klv = rf.H - dot + rf.S * lz;
                    kl[r0 + k] = (float)(klv / rf.S);
                    const int tgt = s.ids[s.rows[r0 + k] + 1];
                    const double nq = -((double)q[tgt] - m - lz);
                    dn[r0 + k] = (float)(nq - rf.nll);
                    t1[r0 + k] = (b == rf.top1) ? 1u : 0u;
                });
            });
            if (!ok) { fprintf(stderr, "tail-kl: %s / %s: %s\n", cf.name.c_str(), s.name.c_str(), err.c_str()); return 1; }
            const double t_c = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
            for (int r = 0; r < R; ++r) {
                float rec[3] = {kl[r], dn[r], 0.0f};
                std::memcpy(&rec[2], &t1[r], 4);
                std::fwrite(rec, 4, 3, fbin[c]);
            }
            std::fflush(fbin[c]);
            double mk = 0, md = 0, t1s = 0;
            for (int r = 0; r < R; ++r) { mk += kl[r]; md += dn[r]; t1s += t1[r]; }
            st.assign((size_t)strata::ink::kInkTailVisits * strata::ink::kInkTailStatW, 0);
            if (spec != "off") backend->tail_stats(st, true);
            std::ostringstream js;
            js << "{\"cfg\":\"" << cf.name << "\",\"seq\":\"" << s.name << "\",\"rows\":" << R << ",\"tokens\":" << s.ids.size()
               << ",\"mean_kl\":" << (R ? mk / R : 0.0) << ",\"mean_dnll\":" << (R ? md / R : 0.0) << ",\"top1\":" << (R ? t1s / R : 0.0)
               << ",\"secs\":" << t_c << ",\"stat\":[";
            for (size_t i = 0; i < st.size(); ++i) js << (i ? "," : "") << st[i];
            js << "]}\n";
            std::fputs(js.str().c_str(), fsum);
            std::fflush(fsum);
            fprintf(stderr, "[tail-kl]   %-28s mean KL %.6f top1 %.4f dNLL %+.5f (%.1fs)\n", cf.name.c_str(), R ? mk / R : 0.0, R ? t1s / R : 0.0, R ? md / R : 0.0, t_c);
        }
        row_offset += R;
    }
    backend->set_tail_skip("off", err);
    for (FILE* f : fbin) std::fclose(f);
    std::fclose(fsum);
    std::fclose(frows);
    printf("[tail-kl] done: %zu sequences, %zu configurations, results in %s\n", seqs.size(), cfgs.size(), out_dir.c_str());
    return 0;
}

// strata-inkling --native MODEL --max-context N --draft off --tail-gen PROMPTS.tsv --tail-gen-out OUT.tsv [--tail-stop id,id]
// PROMPTS.tsv: name <TAB> ids <TAB> n_gen. The exact model answers each prompt greedily (stops at a stop id, which is kept, or after n_gen tokens);
// OUT.tsv: name <TAB> prompt + answer ids <TAB> score range "P-1-(n-2)" (the positions that predict the answer tokens), for --tail-kl.
static inline int cmd_tail_gen(const std::string& model_path, const std::string& in_path, const std::string& out_path, const std::string& stop_str,
                               const BackendOpts& bopts, bool force_cpu) {
    std::string err;
    std::vector<int32_t> stop;
    if (!stop_str.empty()) {
        std::vector<std::string> parts;
        tkl_split(stop_str, ',', parts);
        for (const auto& p : parts) if (!p.empty()) stop.push_back(std::atoi(p.c_str()));
    }
    std::ifstream f(in_path, std::ios::binary);
    if (!f) { fprintf(stderr, "tail-gen: cannot open %s\n", in_path.c_str()); return 2; }
    InkModelFile mf;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int V = mf.m.c.vocab_unpadded;
    FILE* fo = std::fopen(out_path.c_str(), "wb");
    if (!fo) { fprintf(stderr, "tail-gen: cannot write %s\n", out_path.c_str()); return 1; }
    std::string line;
    std::vector<std::string> cols;
    std::vector<float> one(V);
    StepBatch batch;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        tkl_split(line, '\t', cols);
        if (cols.size() < 3) continue;
        std::vector<int32_t> ids;
        {
            const char* p = cols[1].c_str();
            char* e = nullptr;
            while (*p) { long v = std::strtol(p, &e, 10); if (e == p) break; ids.push_back((int32_t)v); p = (*e == ',') ? e + 1 : e; }
        }
        const int n_gen = std::max(1, std::atoi(cols[2].c_str()));
        const int P = (int)ids.size();
        if (P < 2 || P + n_gen >= bopts.max_ctx) { fprintf(stderr, "tail-gen: %s skipped (prompt %d + gen %d vs ctx %d)\n", cols[0].c_str(), P, n_gen, bopts.max_ctx); continue; }
        backend->reset_slot(0);
        auto t0 = std::chrono::steady_clock::now();
        for (int i0 = 0; i0 < P; i0 += 2048) {
            const int i1 = std::min(P, i0 + 2048);
            batch.clear();
            for (int i = i0; i < i1; ++i) batch.push(0, i, ids[i], i == P - 1);
            if (!backend->forward_batch(batch, one.data(), nullptr, err)) { fprintf(stderr, "tail-gen: %s\n", err.c_str()); return 1; }
        }
        std::vector<int32_t> out = ids;
        for (int g = 0; g < n_gen; ++g) {
            int b = 0;
            for (int v = 1; v < V; ++v) if (one[v] > one[b]) b = v;
            out.push_back(b);
            if (std::find(stop.begin(), stop.end(), b) != stop.end()) break;
            batch.clear();
            batch.push(0, P + g, b, true);
            if (!backend->forward_batch(batch, one.data(), nullptr, err)) { fprintf(stderr, "tail-gen: %s\n", err.c_str()); return 1; }
        }
        const int n = (int)out.size();
        fprintf(fo, "%s\t", cols[0].c_str());
        for (int i = 0; i < n; ++i) fprintf(fo, "%s%d", i ? "," : "", out[i]);
        fprintf(fo, "\t%d-%d\n", P - 1, n - 2);
        fflush(fo);
        fprintf(stderr, "[tail-gen] %s: prompt %d, generated %d tokens in %.1fs\n", cols[0].c_str(), P, n - P,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::fclose(fo);
    return 0;
}

// strata-inkling --native MODEL --max-context N --draft off --tail-calib CORPUS.tsv --tail-cfgs SPEC --tail-out OUT.bin
// Runs the corpus (chunked prefill) with the spec (a visit mask and thresholds; the engine adds calib=1: nothing is skipped) and writes the per-layer mean of the weighted
// output of the experts the skip would have dropped, [24 x 2304] float32 = the sq.tail_comp table. The mean is per dropped slot: the router applies it once per skipped slot.
static inline int cmd_tail_calib(const std::string& model_path, const std::string& corpus_path, const std::string& spec_in, const std::string& out_path, int chunk,
                                 const BackendOpts& bopts, bool force_cpu) {
    std::string err;
    std::vector<TailKlSeq> seqs;
    if (!tkl_load_corpus(corpus_path, seqs, err)) { fprintf(stderr, "tail-calib: %s\n", err.c_str()); return 2; }
    for (auto& s : seqs) s.rows.clear();   // no logits are needed
    InkModelFile mf;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    std::string spec = spec_in + ",calib=1";
    if (!backend->set_tail_skip(spec, err)) { fprintf(stderr, "tail-calib: %s\n", err.c_str()); return 1; }
    std::vector<float> buf;
    size_t ntok = 0;
    for (const auto& s : seqs) {
        if (!tkl_run_seq(*backend, s, chunk, buf, mf.m.c.vocab_unpadded, err, [](size_t, int, const float*) {})) { fprintf(stderr, "tail-calib: %s: %s\n", s.name.c_str(), err.c_str()); return 1; }
        ntok += s.ids.size();
    }
    std::vector<double> sum;
    std::vector<uint64_t> cnt;
    if (!backend->tail_calib(sum, cnt, true)) { fprintf(stderr, "tail-calib: no calibration sums\n"); return 1; }
    const int H = 2304;
    std::vector<float> comp((size_t)24 * H, 0.0f);
    for (int l = 0; l < 24; ++l)
        if (cnt[l] > 0) for (int d = 0; d < H; ++d) comp[(size_t)l * H + d] = (float)(sum[(size_t)l * H + d] / (double)cnt[l]);
    FILE* fo = std::fopen(out_path.c_str(), "wb");
    if (!fo) { fprintf(stderr, "tail-calib: cannot write %s\n", out_path.c_str()); return 1; }
    std::fwrite(comp.data(), 4, comp.size(), fo);
    std::fclose(fo);
    printf("[tail-calib] %zu tokens; dropped slots per layer:", ntok);
    for (int l = 0; l < 24; ++l) printf(" %llu", (unsigned long long)cnt[l]);
    printf("\n[tail-calib] mean |bias| per layer:");
    for (int l = 0; l < 24; ++l) { double a = 0; for (int d = 0; d < H; ++d) a += std::fabs(comp[(size_t)l * H + d]); printf(" %.3g", a / H); }
    printf("\n[tail-calib] wrote %s\n", out_path.c_str());
    backend->set_tail_skip("off", err);
    return 0;
}

}  // namespace ink2
