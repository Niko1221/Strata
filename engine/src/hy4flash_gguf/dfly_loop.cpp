// src/hy4flash_gguf/dfly_loop.cpp - the one Dfly request loop. See include/strata/hy4flash_gguf/dfly_loop.hpp.
#include "strata/hy4flash_gguf/dfly_loop.hpp"

#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/indexer.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/mat.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace strata::hy4_flash {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool log_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_LOG");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

// Override width-policy inputs only when running the diagnostic round sequence.
struct KSeq {
    std::vector<int> ks;
    bool ok = true;
    int at(long i) const { return ks[(std::size_t) (i % (long) ks.size())]; }
};
KSeq parse_k_seq(const char* name) {
    KSeq r;
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return r;
    const std::string s(v);
    std::size_t i = 0;
    while (i <= s.size()) {
        const std::size_t j = std::min(s.find(',', i), s.size());
        const std::string t = s.substr(i, j - i);
        char* end = nullptr;
        const long k = std::strtol(t.c_str(), &end, 10);
        if (t.empty() || end != t.c_str() + t.size() || k < 0 || k > DflyDrafter::kMaxK) { r.ok = false; return r; }
        r.ks.push_back((int) k);
        i = j + 1;
    }
    return r;
}
const KSeq& forced_k_seq() { static const KSeq s = parse_k_seq("STRATA_HY4_FLASH_DFLY_K_SEQ"); return s; }
const KSeq& forced_kv_seq() { static const KSeq s = parse_k_seq("STRATA_HY4_FLASH_DFLY_KV_SEQ"); return s; }

}  // namespace

const char* dfly_finish_name(DflyFinish f) {
    switch (f) {
    case DflyFinish::stop: return "stop";
    case DflyFinish::length: return "length";
    case DflyFinish::nonfinite: return "nonfinite";
    case DflyFinish::cancel: return "cancel";
    case DflyFinish::error: return "error";
    case DflyFinish::restart_worker: return "restart_worker";
    }
    return "?";
}

bool dfly_runtime_init(DflyRuntime& rt, const Geometry& g, std::string& err) {
    const int n_blk = (int) g.layers.size() - g.n_mtp_blocks;
    for (DflyCapture* c : {&rt.tail, &rt.win}) {
        c->sel_blocks.clear();
        for (int b = 0; b < n_blk; ++b)
            if (g.layers[(std::size_t) b].attention != Attention::Sliding) c->sel_blocks.push_back(b);
    }
    rt.tail.mode = DflyCapture::Mode::prefill_tail;
    rt.win.mode = DflyCapture::Mode::verify_window;
    return rt.tail.init(g, 1, err) && rt.win.init(g, DflyDrafter::kMaxK + 1, err);
}

DflyFinish dfly_generate(const std::string& gguf, const Geometry& g, const std::string& pack,
                         const ForwardOptions& fo, DenseResidency& dense, KvCache& cache, DflyRuntime& rt,
                         const DflyRequest& req, std::vector<int32_t>& produced, DflyStats& st, std::string& err) {
    produced.clear();
    st = DflyStats{};
    st.reach.assign((std::size_t) DflyDrafter::kMaxK, 0);
    st.hit.assign((std::size_t) DflyDrafter::kMaxK, 0);
    st.k_rounds.assign((std::size_t) DflyDrafter::kMaxK + 1, 0);
    // A failure after the target may have been written: drop any pending proposal, force the target back to a
    // clean state, and never publish what the failed step produced.
    const auto fail = [&](const std::string& why) {
        rt.drafter.drop_pending();
        rt.trusted = false;
        std::string e2;
        const DflyResetResult rr = dfly_abort_reset(dense, cache, e2);
        if (rr == DflyResetResult::clean) {
            err = why + " (the target was reset)";
            return DflyFinish::error;
        }
        err = why + "; the reset failed too (" + e2 + ") - restart the worker";
        return DflyFinish::restart_worker;
    };

    // ---- what this request may be
    if (req.prompt == nullptr || req.prompt->empty()) { err = "dfly: an empty prompt"; return DflyFinish::error; }
    if (req.max_new < 1) { err = "dfly: max_new must be >= 1"; return DflyFinish::error; }
    if (req.k_max < 1 || req.k_max > DflyDrafter::kMaxK) {
        err = "dfly: k_max must be 1.." + std::to_string(DflyDrafter::kMaxK);
        return DflyFinish::error;
    }
    const KSeq& kseq = forced_k_seq();
    const KSeq& kvseq = forced_kv_seq();
    if (!kseq.ok || !kvseq.ok) {
        err = "dfly: STRATA_HY4_FLASH_DFLY_K_SEQ / _KV_SEQ must be a comma list of 0.." + std::to_string(DflyDrafter::kMaxK);
        return DflyFinish::error;
    }
    // The confidence window's experiment overrides, read only when it is on.
    DflyGateConfig gcfg;
    if (req.window == DflyWindow::confidence) {
        std::string perr;
        const char* gv = std::getenv("STRATA_HY4_FLASH_DFLY_GATE");
        if (gv != nullptr && !dfly_gate_config_parse(gv, gcfg, perr)) {
            err = "dfly: STRATA_HY4_FLASH_DFLY_GATE: " + perr;
            return DflyFinish::error;
        }
    }
    DflyConfidenceGate gate(gcfg, &rt.verify_costs);
    if (!rt.trusted && cache.len != 0) {
        err = "dfly: the cache is not trusted after an earlier failure; start from position 0";
        return DflyFinish::error;
    }
    if (kv_format(cache) == kKvF32) { err = "dfly: needs F16 or Q8_0 KV"; return DflyFinish::error; }
    const std::vector<int32_t>& prompt = *req.prompt;
    const int N = (int) prompt.size();
    const int reuse = cache.len;
    const int kp = g.idx_kpool > 0 ? g.idx_kpool : 1;
    if (reuse < 0 || reuse >= N || reuse % kp != 0) {
        err = "dfly: a reused prefix must be a multiple of " + std::to_string(kp) + " and shorter than the prompt (" +
              std::to_string(reuse) + " of " + std::to_string(N) + ")";
        return DflyFinish::error;
    }
    rt.trusted = true;
    if (N + req.max_new > cache.max_tokens) {
        err = "dfly: prompt + max_new exceeds the " + std::to_string(cache.max_tokens) + " reserved positions";
        return DflyFinish::error;
    }
    const int64_t vocab = g.vocab_size;
    const auto is_stop = [&](int32_t t) {
        return req.stop_tokens != nullptr &&
               std::find(req.stop_tokens->begin(), req.stop_tokens->end(), t) != req.stop_tokens->end();
    };
    Sampler host_sampler(req.sampler);
    const auto pick = [&](const float* row, const std::vector<int32_t>& h) -> int64_t {
        return req.pick ? req.pick(row, h) : host_sampler.pick(row, vocab, h);
    };
    std::vector<int32_t> history = prompt;
    std::vector<float> lg;
    const float* dlg = nullptr;   // the pass's logits on the device, with `device_logits`
    // Row j of the last pass's logits, where `pick` and `emit` want it.
    const auto row_of = [&](std::size_t j) -> const float* {
        return req.device_logits ? dlg + j * (std::size_t) vocab : lg.data() + j * (std::size_t) vocab;
    };

    // ---- the prompt, chunked EXACTLY as the caller's plain path does, so the cache and the first row are the
    // same: whole chunks while more than a chunk is left, then the rest with logits and the tail capture.
    const auto t_pf = std::chrono::steady_clock::now();
    const int chunk = prefill_chunk_max();
    int done = reuse;
    while (N - done > chunk) {
        if (req.should_stop && req.should_stop()) return DflyFinish::cancel;   // the cache holds [0, done)
        const std::vector<int32_t> part(prompt.begin() + done, prompt.begin() + done + chunk);
        ForwardOptions co = fo;
        co.history = &history;
        co.want_logits = false;
        if (req.tail_last_row_only) co.logits_last_only = true;   // serve's plain chunks: the head runs one row
        co.dfly = nullptr;
        co.exact_columns = false;
        if (!forward_prefill(gguf, g, pack, part, lg, err, co)) return fail("dfly: prompt chunk at " + std::to_string(done) + ": " + err);
        done += chunk;
        if (req.on_prompt_chunk) req.on_prompt_chunk(done, N);
    }
    if (req.should_stop && req.should_stop()) return DflyFinish::cancel;   // before any output, as the plain loops
    const std::vector<int32_t> rest(prompt.begin() + done, prompt.end());
    const bool tail_one_row = req.tail_last_row_only || (int) rest.size() > kGemmT;
    {
        ForwardOptions co = fo;
        co.history = &history;
        co.want_logits = true;
        co.logits_last_only = tail_one_row;
        co.device_logits_out = req.device_logits ? &dlg : nullptr;
        co.dfly = &rt.tail;
        co.exact_columns = false;
        rt.tail.mode = DflyCapture::Mode::prefill_tail;
        if (!forward_prefill(gguf, g, pack, rest, lg, err, co) || !rt.tail.valid || (req.device_logits && dlg == nullptr))
            return fail("dfly: the prompt's last chunk: " + err);
    }
    st.prefill_ms = ms_since(t_pf);
    const float* first_row = row_of(tail_one_row ? 0 : rest.size() - 1);

    // ---- the drafter, once the target is warm: the expert tier has been sized from the VRAM that was free.
    // Checked before anything is output, so a request that cannot draft fails without a partial answer.
    const bool offload_verify = !dense_experts_complete(dense);
    if (offload_verify && !expert_window_enabled()) {
        err = "dfly: a partial expert tier requires the offload verification window "
              "(STRATA_HY4_FLASH_WINDOW=1, STRATA_HY4_FLASH_ASYNC_MOE=1, STRATA_DF_PDL=0)";
        return DflyFinish::error;
    }
    if (!rt.drafter.loaded() && !rt.drafter.load(req.draft_path, g, err)) return DflyFinish::error;

    if (req.k_max > rt.drafter.max_drafts()) {
        err = "dfly: --draft-max exceeds the GGUF block size minus the anchor";
        return DflyFinish::error;
    }

    // ---- the first output, from the prompt's last row; it is output but not yet in the cache. Picked BEFORE
    // the drafter is bound: device logits are only valid until the next device work on the target.
    const int64_t first = pick(first_row, history);
    if (first < 0) { err = "dfly: the prompt's logits are not all finite"; return DflyFinish::nonfinite; }
    int32_t xa = (int32_t) first;
    history.push_back(xa);
    produced.push_back(xa);
    if (req.emit && !req.emit(xa, first_row)) return DflyFinish::cancel;
    if (is_stop(xa)) return DflyFinish::stop;
    if ((int) produced.size() >= req.max_new) return DflyFinish::length;
    if (!rt.drafter.bound() && !rt.drafter.bind(dense, cache, err)) return DflyFinish::error;
    const std::array<int, 5> layers = rt.drafter.target_layers();
    int a = N;
    DflyContext ctx;
    if (!dfly_context_from_capture(rt.tail, 0, layers, ctx, err)) return fail("dfly: " + err);

    const auto t_dec = std::chrono::steady_clock::now();
    int32_t draft[DflyDrafter::kMaxK];
    float conf[DflyDrafter::kMaxK];
    long round_index = 0;
    for (;;) {
        if (req.should_stop && req.should_stop()) { st.decode_ms = ms_since(t_dec); return DflyFinish::cancel; }
        if (cache.len != a || (int) history.size() != a + 1 || ctx.source_pos != a - 1)
            return fail("dfly: internal state out of step at " + std::to_string(a));
        const auto t_round = std::chrono::steady_clock::now();
        const int remaining = req.max_new - (int) produced.size();
        int cap = std::min(req.k_max, remaining - 1);
        // One indexer plan for the whole window; otherwise shorten it (a = 2047 drafts nothing).
        while (cap > 0 && indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, a, 1).engaged !=
                              indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, a, cap + 1).engaged)
            --cap;
        // The legal maximum is cap; every window drafts all of it (the test hook may draft less).
        int k = cap;
        if (!kseq.ks.empty()) k = std::min(cap, kseq.at(round_index));
        bool fell_back = false;
        double propose_ms = 0.0;
        if (k > 0) {
            const auto tp = std::chrono::steady_clock::now();
            const DflyResult r = rt.drafter.propose(xa, a, ctx, k, draft, conf, err);
            propose_ms = ms_since(tp);
            st.propose_ms += propose_ms;
            if (r == DflyResult::safe_fallback) {
                ++st.fallbacks;
                if (log_on()) std::fprintf(stderr, "DFLY fallback a=%d: %s\n", a, err.c_str());
                k = 0;
                fell_back = true;
            } else if (r != DflyResult::ok) {
                return fail("dfly: propose at " + std::to_string(a) + " (" + dfly_result_name(r) + "): " + err);
            }
        }
        // How many of the k drafts the verify takes: all of them, or the confidence gate's (or the hook's) prefix.
        int kv = k;
        if (k > 0 && !kvseq.ks.empty()) kv = std::min(k, kvseq.at(round_index));
        else if (k > 0 && req.window == DflyWindow::confidence) kv = gate.choose(conf, k);
        ++round_index;
        const int w = kv + 1;
        std::vector<int32_t> inputs{xa};
        for (int j = 0; j < kv; ++j) inputs.push_back(draft[j]);
        // The verify consumes the proposal: the same anchor and context, a prefix of the rows it wrote.
        if (k > 0 && !rt.drafter.consume_pending(a, w, ctx.generation, err))
            return fail("dfly: the verify does not match the pending proposal: " + err);

        // ---- the verify: one exact pass, every row's logits, the window captured
        ForwardOptions co = fo;
        co.history = &history;   // the committed sequence; the window's own tokens come from `inputs`
        co.want_logits = true;
        co.logits_last_only = false;
        co.device_logits_out = req.device_logits ? &dlg : nullptr;
        co.dfly = &rt.win;
        // With offloaded experts use the target's ordinary speculative verification window.
        // The capture still records all candidate rows for Dfly's next proposal and rewind.
        co.speculative = offload_verify;
        co.decode_equivalent = offload_verify;
        co.exact_columns = true;
        rt.win.mode = DflyCapture::Mode::verify_window;
        dlg = nullptr;
        const auto tv = std::chrono::steady_clock::now();
        const bool ok = forward_prefill(gguf, g, pack, inputs, lg, err, co);
        const double verify_ms = ms_since(tv);
        st.verify_ms += verify_ms;
        const bool rows = req.device_logits ? dlg != nullptr : lg.size() >= (std::size_t) w * (std::size_t) vocab;
        if (!ok || !rt.win.valid || !rows)
            return fail("dfly: verify at " + std::to_string(a) + " (w " + std::to_string(w) + "): " + err);

        // ---- row by row: sample-and-match. Each pick sees the picks before it in `history`, as the plain path's
        // would; a row past a pick that differs from its draft is never read.
        std::vector<int32_t> out;
        int m = 0, r = 0;
        DflyFinish fin = DflyFinish::error;
        bool finished = false;
        for (int j = 0; j < w; ++j) {
            const int64_t y = pick(row_of((std::size_t) j), history);
            if (y < 0) { fin = DflyFinish::nonfinite; finished = true; break; }
            const bool match = j < kv && y == draft[j];
            out.push_back((int32_t) y);
            history.push_back((int32_t) y);
            ++m;
            if (match) ++r;
            if (is_stop((int32_t) y)) { fin = DflyFinish::stop; finished = true; break; }
            if ((int) produced.size() + m >= req.max_new) { fin = DflyFinish::length; finished = true; break; }
            if (!match) break;   // the correction, or the bonus when j == kv
        }
        const int new_len = a + m;
        if (!dfly_rewind(dense, cache, rt.win, new_len, err))
            return fail("dfly: rewind from " + std::to_string(a + w) + " to " + std::to_string(new_len) + ": " + err);
        const double round_ms = ms_since(t_round);
        // Only a round that ran what the window chose teaches it; a fallback paid for a proposal it did not use,
        // and a forced (test hook) round was not the window's choice.
        if (req.window == DflyWindow::confidence && k > 0 && !fell_back && kseq.ks.empty() && kvseq.ks.empty())
            gate.observe(conf, kv, m, r, round_ms - propose_ms, round_ms);

        // ---- committed: only now statistics and output (`history` took each pick as it was made)
        st.drafted += k;
        if (kv < k) ++st.gated;
        if (kv > 0) {
            ++st.rounds;
            st.offered += kv;
            st.accepted += r;
            st.round_outputs += m;
            ++st.kr[{kv, r}];
            for (int i = 0; i < std::min(kv, m); ++i) {
                ++st.reach[(std::size_t) i];
                if (i < r) ++st.hit[(std::size_t) i];
            }
        } else {
            ++st.ar_steps;
            if (k > 0) ++st.kv0_chosen;
        }
        ++st.k_rounds[(std::size_t) kv];
        if (log_on()) {
            // After verify=: the legal cap and what limited it, why kv is what it is, the drafter's confidence per
            // drafted slot (k of them; the verify took w - 1) and the round's wall time.
            char cs[64] = "-";
            int o = 0;
            for (int j = 0; j < k && o < (int) sizeof cs - 8; ++j)
                o += std::snprintf(cs + o, sizeof cs - (std::size_t) o, j ? ",%.4f" : "%.4f", (double) conf[j]);
            const char* lim = cap == req.k_max ? "max" : (remaining - 1 <= cap ? "budget" : "boundary");
            const char* why = fell_back                                ? "fallback"
                              : !kseq.ks.empty() || !kvseq.ks.empty()  ? "seq"
                              : kv < k                                 ? "gate"
                              : req.window != DflyWindow::fixed        ? "policy"
                                                                       : "fixed";
            std::fprintf(stderr, "DFLY a=%d k=%d w=%d m=%d r=%d fin=%s propose=%.2fms verify=%.2fms cap=%d lim=%s "
                         "why=%s conf=%s round=%.2fms\n", a, k, w, m, r, finished ? dfly_finish_name(fin) : "-",
                         propose_ms, verify_ms, cap, lim, why, cs, round_ms);
        }
        if (m > 0) {
            a = new_len;
            xa = out.back();
            if (!finished && !dfly_context_from_capture(rt.win, m - 1, layers, ctx, err)) return fail("dfly: " + err);
        }
        for (int j = 0; j < m; ++j) {
            produced.push_back(out[(std::size_t) j]);
            if (req.emit && !req.emit(out[(std::size_t) j], row_of((std::size_t) j))) {
                st.decode_ms = ms_since(t_dec);
                return DflyFinish::cancel;
            }
        }
        if (finished) {
            st.decode_ms = ms_since(t_dec);
            if (fin == DflyFinish::nonfinite) err = "dfly: the logits are not all finite at position " + std::to_string(a + 1);
            return fin;
        }
    }
}

}  // namespace strata::hy4_flash
