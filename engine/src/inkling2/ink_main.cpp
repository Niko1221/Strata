// engine/src/inkling2/ink_main.cpp - Standalone `strata-inkling` Shape-Engine Binary for Inkling-2-Mini (.strata v1.1)
// Supports: --version, --info, --export-tokenizer, --selfcheck, --bench (--parallel 1|4|8|32|64), --kld-eval, --serve, and direct CLI generation.
// MIT License - Copyright (c) 2026 Strata contributors
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"
#include "ink_serve.h"
#include "ink_tail_kl.h"
#include "ink_vision.h"
#include "strata/core/device_error_exit.hpp"
#include "strata/core/volta_forever.hpp"
#if defined(__linux__) && defined(INKLING_ENABLE_HIP)
#include "strata/platform/numa_balancing.hpp"
#endif

using namespace ink2;

#ifndef INKLING_VERSION
#define INKLING_VERSION "0.1.70"
#endif

static int cmd_export_tokenizer(const std::string& file, const std::string& dir) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(file, err, /*keep_tokenizer=*/true)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    std::filesystem::create_directories(dir);
    auto write_str = [&](const std::string& name, const std::string& content) -> bool {
        if (content.empty()) return true;
        std::string path = dir + "/" + name;
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(content.data(), 1, content.size(), f) != content.size()) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            if (f) std::fclose(f);
            return false;
        }
        std::fclose(f);
        return true;
    };
    if (!write_str("chat_template.jinja", mf.m.chat_template)) return 1;
    if (!write_str("tokenizer.hf.json", mf.m.tokenizer_json)) return 1;
    printf("format=%s chat_template: %zu bytes, tokenizer_json: %zu bytes\n",
           source_kind_name(mf.kind), mf.m.chat_template.size(), mf.m.tokenizer_json.size());
    return 0;
}

static int cmd_info(const std::string& file, const std::string& vision, const std::string& dspark_path) {
    std::string err;
    // Check VisionShard (.strata) first
    if (!SafetensorsAdapter::is_safetensors_path(file) && !GgufAdapter::is_gguf_path(file)) {
        strata::artifact::StrataFileReader rd;
        if (!rd.open(file, err)) {
            fprintf(stderr, "open: %s\n", err.c_str());
            return 1;
        }
        if (rd.is_vision_shard()) {
            rd.close();
            VisionShardInfo vi;
            if (!vision_probe(file, vi, err)) {
                fprintf(stderr, "vision shard: %s\n", err.c_str());
                return 1;
            }
            printf("arch=inkling2_vision layers=%d hidden=%d out_hidden=%d patch=%d merge=%d tensors=%zu image_pad=%d\n",
                   vi.c.layers, vi.c.hidden, vi.c.out_hidden, vi.c.patch, vi.c.merge, vi.T.size(), vi.c.image_token);
            return 0;
        }
    }
    InkModelFile mf;
    if (!mf.open(file, err, /*keep_tokenizer=*/false, dspark_path)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    const InkManifest& m = mf.m;
    int images = 1;
    if (!vision.empty()) {
        VisionShardInfo vi;
        std::string verr;
        images = vision_probe(vision, vi, verr) ? 1 : 0;
        if (!images) fprintf(stderr, "vision shard: %s\n", verr.c_str());
    }
    printf("arch=inkling2_loop format=%s build=%s layers=%d visits=%d hidden=%d vocab=%d experts=%d eos=%d drafter=%s images=%d image_pad=%d parallel=%d spec_opts=1 dspark_f32_widen=1\n",
           source_kind_name(mf.kind),
           m.build.empty() ? "unknown" : m.build.c_str(),
           m.c.n_phys, m.c.n_visits, m.c.hidden, m.c.vocab_unpadded, m.c.n_experts, m.c.eos_token_id,
           m.dspark.present ? "dspark" : "none", images, m.c.image_token_id, kMaxParallelSlots);
    return 0;
}

static int cmd_selfcheck(const std::string& model_path, bool force_cpu, bool verify_hash, const BackendOpts& bopts) {
    auto backend = make_default_backend(force_cpu, &bopts);
    std::string err;
    InkModelFile mf;
    if (!model_path.empty()) {
        if (!mf.open(model_path, err, false, bopts.dspark_path)) {
            fprintf(stderr, "[selfcheck] FAIL open: %s\n", err.c_str());
            return 1;
        }
        if (verify_hash) {
            printf("[selfcheck] Verifying SHA-256 of Sections 0..5...\n");
            if (!mf.verify_sections(err, true)) {
                fprintf(stderr, "[selfcheck] FAIL hash: %s\n", err.c_str());
                return 1;
            }
        }
        if (!backend->init(mf, bopts, err)) {
            fprintf(stderr, "[selfcheck] FAIL init: %s\n", err.c_str());
            return 1;
        }
    }
    if (!backend->selfcheck(err)) {
        fprintf(stderr, "[selfcheck] FAIL: %s\n", err.c_str());
        return 1;
    }
    printf("[selfcheck] PASS backend=%s (%s)\n", backend->backend_name(), backend->device_info().c_str());
    return 0;
}

static int cmd_bench(const std::string& model_path, const BackendOpts& bopts, int prompt_len, int gen_len, bool force_cpu, bool use_dspark) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }
    BackendOpts bo = bopts;
    bo.enable_dspark = use_dspark;
    if (prompt_len + gen_len > bo.max_ctx) {
        bo.max_ctx = prompt_len + gen_len + 16;
    }
    auto backend = make_default_backend(force_cpu, &bo);
    if (!backend->init(mf, bo, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    const int eff_ctx = (backend->effective_max_ctx() > 0) ? backend->effective_max_ctx() : bo.max_ctx;
    if (prompt_len >= eff_ctx) {
        fprintf(stderr, "[bench] clamping prompt_len %d -> %d to fit effective max_ctx=%d\n", prompt_len, std::max(1, eff_ctx - 1), eff_ctx);
        prompt_len = std::max(1, eff_ctx - 1);
    }
    const int max_dec_steps = std::max(0, eff_ctx - prompt_len);
    if (gen_len > max_dec_steps) {
        fprintf(stderr, "[bench] clamping gen_len %d -> %d to fit effective max_ctx=%d (prompt=%d)\n",
                gen_len, max_dec_steps, eff_ctx, prompt_len);
        gen_len = max_dec_steps;
    }
    const int P = std::max(1, std::min(kMaxParallelSlots, bo.parallel));
    const int V = mf.m.c.vocab_unpadded;
    for (int s = 0; s < P; ++s) {
        if (!backend->alloc_slot(s, eff_ctx, bo.kv_dtype, err)) {
            fprintf(stderr, "alloc_slot %d: %s\n", s, err.c_str());
            return 1;
        }
    }
    // Synthetic prompt per slot
    std::vector<int32_t> next_tok(P, 100);
    std::vector<float> logits(static_cast<size_t>(P) * V);
    StepBatch batch;
    double t_pf0 = now_s();
    const int chunk_run = std::max(1, backend->effective_chunk() > 0 ? backend->effective_chunk() : bo.chunk);
    for (int p0 = 0; p0 < prompt_len; p0 += chunk_run) {
        int len = std::min(chunk_run, prompt_len - p0);
        batch.clear();
        batch.capture_taps = use_dspark;
        for (int s = 0; s < P; ++s) {
            for (int k = 0; k < len; ++k) {
                bool is_last = (p0 + k == prompt_len - 1);
                batch.push(s, p0 + k, 100 + ((s * 31 + p0 + k) % 1000), is_last);
            }
        }
        if (!backend->forward_batch(batch, (p0 + len == prompt_len) ? logits.data() : nullptr, nullptr, err)) {
            fprintf(stderr, "bench prefill err: %s\n", err.c_str());
            return 1;
        }
    }
    double pf_s = now_s() - t_pf0;
    for (int s = 0; s < P; ++s) {
        const float* lg = logits.data() + static_cast<size_t>(s) * V;
        int32_t best = 0;
        for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
        next_tok[s] = best;
    }

    double t_dec0 = now_s();
    int total_gen = 0;
    if (use_dspark && P == 1) {
        int cur_pos = prompt_len;
        while (total_gen < gen_len && cur_pos < eff_ctx) {
            const int rem = std::min(gen_len - total_gen, eff_ctx - cur_pos);
            if (rem >= 2 && cur_pos + 12 < eff_ctx) {
                int32_t emitted[32]{};
                int drafted = 0;
                const int n_emit = backend->dspark_spec_step_greedy(
                    0, next_tok[0], cur_pos, rem, emitted, &drafted, err);
                if (n_emit < 0) { fprintf(stderr, "bench spec err: %s\n", err.c_str()); return 1; }
                if (n_emit > 0) {
                    next_tok[0] = emitted[n_emit - 1];
                    cur_pos += n_emit;
                    total_gen += n_emit;
                    continue;
                }
            }
            batch.clear();
            batch.capture_taps = true;
            batch.push(0, cur_pos++, next_tok[0], true);
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
                fprintf(stderr, "bench decode err: %s\n", err.c_str());
                return 1;
            }
            int32_t best = 0;
            for (int i = 1; i < V; ++i) if (logits[i] > logits[best]) best = i;
            next_tok[0] = best;
            ++total_gen;
        }
    } else if (use_dspark && P >= 2 && backend->supports_batched_dspark()) {
        // One multi-user drafting hook for every RTX 50 card (ink_backend.h): C2..C64 batched speculative decode, then the tail one token at a time.
        std::vector<int32_t> slot_pos(P, prompt_len), slot_gen(P, 0);
        while (true) {
            std::vector<int32_t> b_slots, b_toks, b_pos, b_maxe;
            for (int s = 0; s < P; ++s) {
                const int rem = std::min(gen_len - slot_gen[s], eff_ctx - slot_pos[s] - 12);
                if (rem >= 2) {
                    b_slots.push_back(s);
                    b_toks.push_back(next_tok[s]);
                    b_pos.push_back(slot_pos[s]);
                    b_maxe.push_back(rem);
                }
            }
            if (b_slots.empty()) break;
            const int B = static_cast<int>(b_slots.size());
            std::vector<int32_t> emitted((size_t)B * 8, 0), n_emit(B, 0), drafted(B, 0);
            int tot = 0;
            if (B >= 2) {
                tot = backend->dspark_spec_step_batched_greedy(
                    B, b_slots.data(), b_toks.data(), b_pos.data(), b_maxe.data(),
                    8, emitted.data(), n_emit.data(), drafted.data(), err);
                if (tot < 0) { fprintf(stderr, "bench batched spec err: %s\n", err.c_str()); return 1; }
            }
            if (tot > 0) {
                for (int bi = 0; bi < B; ++bi) {
                    const int s = b_slots[bi];
                    const int ne = n_emit[bi];
                    next_tok[s] = emitted[(size_t)bi * 8 + ne - 1];
                    slot_pos[s] += ne;
                    slot_gen[s] += ne;
                    total_gen += ne;
                }
                continue;
            }
            batch.clear();
            batch.capture_taps = use_dspark;
            for (int bi = 0; bi < B; ++bi) batch.push(b_slots[bi], slot_pos[b_slots[bi]], next_tok[b_slots[bi]], true);
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) { fprintf(stderr, "bench decode err: %s\n", err.c_str()); return 1; }
            for (int bi = 0; bi < B; ++bi) {
                const int s = b_slots[bi];
                const float* lg = logits.data() + static_cast<size_t>(bi) * V;
                int32_t best = 0;
                for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
                next_tok[s] = best;
                ++slot_pos[s];
                ++slot_gen[s];
                ++total_gen;
            }
        }
        for (;;) {
            batch.clear();
            batch.capture_taps = false;
            std::vector<int> tail_s;
            for (int s = 0; s < P; ++s) {
                if (slot_gen[s] < gen_len && slot_pos[s] < eff_ctx) {
                    batch.push(s, slot_pos[s], next_tok[s], true);
                    tail_s.push_back(s);
                }
            }
            if (tail_s.empty()) break;
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) { fprintf(stderr, "bench decode err: %s\n", err.c_str()); return 1; }
            for (size_t bi = 0; bi < tail_s.size(); ++bi) {
                const int s = tail_s[bi];
                const float* lg = logits.data() + bi * V;
                int32_t best = 0;
                for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
                next_tok[s] = best;
                ++slot_pos[s];
                ++slot_gen[s];
                ++total_gen;
            }
        }
    } else {
        // The shipped decode loop. On a card whose table row asks for it (fast mode), the 128 best candidates per row come back instead of the full
        // vocab-wide logits (what --serve does; 64 rows are 51 MB a step otherwise). The backend decides, there is no environment read here.
        const bool bench_cand = backend->prefers_candidate_rows() && backend->supports_candidates();
        CandRows cand;
        for (int step = 0; step < gen_len; ++step) {
            batch.clear();
            batch.capture_taps = use_dspark;
            for (int s = 0; s < P; ++s) batch.push(s, prompt_len + step, next_tok[s], true);
            if (bench_cand) {
                if (!backend->forward_batch_cand(batch, cand, err)) {
                    fprintf(stderr, "bench decode err: %s\n", err.c_str());
                    return 1;
                }
                for (int s = 0; s < P; ++s) {
                    next_tok[s] = cand.idx[static_cast<size_t>(s) * InkBackend::kInkCandidates];
                    ++total_gen;
                }
                continue;
            }
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
                fprintf(stderr, "bench decode err: %s\n", err.c_str());
                return 1;
            }
            for (int s = 0; s < P; ++s) {
                const float* lg = logits.data() + static_cast<size_t>(s) * V;
                int32_t best = 0;
                for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
                next_tok[s] = best;
                ++total_gen;
            }
        }
    }
    double dec_s = now_s() - t_dec0;
    double ppts = (P * prompt_len) / std::max(pf_s, 1e-9);
    double tps_agg = total_gen / std::max(dec_s, 1e-9);
    printf("[bench] backend=%s format=%s parallel=%d gpus=%d kv=%s prompt=%d gen=%d | prefill=%.1f ppts (%.2fs) | decode=%.1f tps agg (%.1f tps/slot, %.2fs)\n",
           backend->backend_name(), source_kind_name(mf.kind), P, bo.gpus, kv_dtype_name(backend->stats().kv_dtype),
           prompt_len, gen_len, ppts, pf_s, tps_agg, tps_agg / P, dec_s);
    return 0;
}

// Teacher-forced KLD/PPL/Top-1 evaluation against a binary reference file:
// Header: int32 n_tokens, int32 vocab_unpadded, int32 tokens[n_tokens], float ref_logits[n_tokens * vocab_unpadded]
static int cmd_kld_eval(const std::string& model_path, const std::string& ref_path, const BackendOpts& bopts, bool force_cpu) {
    FILE* f = std::fopen(ref_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "kld-eval: cannot open %s\n", ref_path.c_str()); return 1; }
    int32_t n_tok = 0, V_ref = 0;
    if (std::fread(&n_tok, 4, 1, f) != 1 || std::fread(&V_ref, 4, 1, f) != 1 || n_tok <= 1) {
        fprintf(stderr, "kld-eval: invalid reference header\n");
        std::fclose(f);
        return 1;
    }
    std::vector<int32_t> ids(n_tok);
    std::vector<float> ref_lg(static_cast<size_t>(n_tok) * V_ref);
    if (std::fread(ids.data(), 4, n_tok, f) != static_cast<size_t>(n_tok) ||
        std::fread(ref_lg.data(), 4, ref_lg.size(), f) != ref_lg.size()) {
        fprintf(stderr, "kld-eval: truncated reference file\n");
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    const int V = mf.m.c.vocab_unpadded;
    if (V != V_ref) { fprintf(stderr, "vocab mismatch %d vs %d\n", V, V_ref); return 1; }

    std::vector<float> got_lg(static_cast<size_t>(n_tok) * V);
    StepBatch batch;
    // INK_KLD_CHUNK=n feeds the tokens n at a time (as INK_DUMP_CHUNK does for --dump-ref; 16 = the verify-width kernels, 1 = the one-token decode kernels), default: all in one batch
    int kld_chunk = n_tok;
    if (const char* kc = std::getenv("INK_KLD_CHUNK"); kc && std::atoi(kc) > 0) kld_chunk = std::min<int>(std::atoi(kc), n_tok);
    for (int i0 = 0; i0 < n_tok; i0 += kld_chunk) {
        const int i1 = std::min<int>(n_tok, i0 + kld_chunk);
        batch.clear();
        for (int i = i0; i < i1; ++i) batch.push(0, i, ids[i], true);
        if (!backend->forward_batch(batch, got_lg.data() + static_cast<size_t>(i0) * V, nullptr, err)) {
            fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
    }

    double kld_sum = 0.0, nll_sum = 0.0;
    int top1_ok = 0;
    for (int t = 0; t < n_tok - 1; ++t) {
        const float* p_lg = ref_lg.data() + static_cast<size_t>(t) * V;
        const float* q_lg = got_lg.data() + static_cast<size_t>(t) * V;
        float mp = p_lg[0], mq = q_lg[0];
        int bp = 0, bq = 0;
        for (int v = 1; v < V; ++v) {
            if (p_lg[v] > mp) { mp = p_lg[v]; bp = v; }
            if (q_lg[v] > mq) { mq = q_lg[v]; bq = v; }
        }
        if (bp == bq) ++top1_ok;
        double zp = 0.0, zq = 0.0;
        for (int v = 0; v < V; ++v) { zp += std::exp(p_lg[v] - mp); zq += std::exp(q_lg[v] - mq); }
        double lzp = mp + std::log(zp), lzq = mq + std::log(zq);
        double kld = 0.0;
        for (int v = 0; v < V; ++v) {
            double lp = p_lg[v] - lzp;
            double lq = q_lg[v] - lzq;
            double pv = std::exp(lp);
            kld += pv * (lp - lq);
        }
        kld_sum += kld;
        nll_sum += -(q_lg[ids[t + 1]] - lzq);
    }
    const int n_eval = n_tok - 1;
    printf("[kld-eval] n=%d | Mean KLD=%.6f | Top-1=%.2f%% | PPL=%.3f\n",
           n_eval, kld_sum / n_eval, 100.0 * top1_ok / n_eval, std::exp(nll_sum / n_eval));
    return 0;
}

// Dump teacher-forced FP32 reference logits for --kld-eval parity checks across formats & backends
static int cmd_dump_ref(const std::string& model_path, const std::string& out_path,
                        const std::string& prompt_ids_str, const BackendOpts& bopts, bool force_cpu) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) {
        fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    if (!prompt_ids_str.empty()) {
        if (!parse_id_list(prompt_ids_str, ids, err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
    } else {
        ids = {1, 2, 3, 4, 5, 6, 7, 8};
    }
    const int32_t n_tok = static_cast<int32_t>(ids.size());
    const int32_t V = mf.m.c.vocab_unpadded;
    std::vector<float> logits(static_cast<size_t>(n_tok) * V);
    StepBatch batch;
    // INK_DUMP_CHUNK=n feeds the tokens n at a time (1 = the single-token decode kernels), default: all in one batch
    int dump_chunk = n_tok;
    if (const char* dc = std::getenv("INK_DUMP_CHUNK"); dc && std::atoi(dc) > 0) dump_chunk = std::min<int>(std::atoi(dc), n_tok);
    for (int i0 = 0; i0 < n_tok; i0 += dump_chunk) {
        const int i1 = std::min<int>(n_tok, i0 + dump_chunk);
        batch.clear();
        for (int i = i0; i < i1; ++i) batch.push(0, i, ids[i], true);
        if (!backend->forward_batch(batch, logits.data() + static_cast<size_t>(i0) * V, nullptr, err)) {
            fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
    }
    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "dump-ref: cannot write %s\n", out_path.c_str()); return 1; }
    std::fwrite(&n_tok, 4, 1, f);
    std::fwrite(&V, 4, 1, f);
    std::fwrite(ids.data(), 4, n_tok, f);
    std::fwrite(logits.data(), 4, logits.size(), f);
    std::fclose(f);
    printf("[dump-ref] wrote %s (n_tok=%d, V=%d, backend=%s)\n", out_path.c_str(), n_tok, V, backend->backend_name());
    return 0;
}

// #6878: a prompt run as batches must give the logits the one-token steps give, at every position (the SWA K/V ring must not let a later token of
// a batch overwrite a key an earlier query of that batch still reads).  --ring-check STRIDE: feed the prompt ids (a) in batches of INK_RING_CHECK_BATCH
// tokens (default 4096, one forward_batch each) and (b) one token per forward, and compare the logits of every STRIDE-th position (and the last 8):
// KL of the one-token distribution from the batch one, top-1 agreement, and the same for the rows after position swa_win only.  The two paths are
// different kernels, so they differ by rounding (KL below 1e-3 on SQ-class packs); a ring overwrite gives KL of order 1 and a top-1 agreement far below.
// Exit 0 when the median KL after the window is below 0.02 and the top-1 agreement above 90%.
static int32_t argmax_f32(const float* lg, int V);
static int cmd_ring_check(const std::string& model_path, const BackendOpts& bopts, bool force_cpu, const std::vector<int32_t>& ids, int stride) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    auto backend = make_default_backend(force_cpu, &bopts);
    if (!backend->init(mf, bopts, err) || !backend->alloc_slot(0, bopts.max_ctx, bopts.kv_dtype, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int n = static_cast<int>(ids.size());
    const int V = mf.m.c.vocab_unpadded;
    const int win = mf.m.c.swa_win;
    int piece = 4096;
    if (const char* e = std::getenv("INK_RING_CHECK_BATCH"); e && std::atoi(e) > 0) piece = std::atoi(e);
    stride = std::max(1, stride);
    std::vector<int> rows;
    for (int i = 0; i < n; ++i) if (i % stride == 0 || i >= n - 8) rows.push_back(i);
    std::vector<float> A(rows.size() * static_cast<size_t>(V)), B(rows.size() * static_cast<size_t>(V));
    StepBatch batch;
    std::vector<float> lg(static_cast<size_t>(piece) * V);
    for (int i0 = 0; i0 < n; i0 += piece) {
        const int i1 = std::min(n, i0 + piece);
        batch.clear();
        for (int i = i0; i < i1; ++i) batch.push(0, i, ids[i], true);
        if (!backend->forward_batch(batch, lg.data(), nullptr, err)) { fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
        for (size_t r = 0; r < rows.size(); ++r)
            if (rows[r] >= i0 && rows[r] < i1) std::memcpy(A.data() + r * V, lg.data() + static_cast<size_t>(rows[r] - i0) * V, sizeof(float) * V);
    }
    backend->reset_slot(0);
    size_t r = 0;
    for (int i = 0; i < n; ++i) {
        batch.clear();
        batch.push(0, i, ids[i], true);
        if (!backend->forward_batch(batch, lg.data(), nullptr, err)) { fprintf(stderr, "step: %s\n", err.c_str()); return 1; }
        if (r < rows.size() && rows[r] == i) { std::memcpy(B.data() + r * V, lg.data(), sizeof(float) * V); ++r; }
    }
    auto logsm = [&](const float* x, std::vector<double>& out) {
        double m = x[0];
        for (int i = 1; i < V; ++i) m = std::max<double>(m, x[i]);
        double s = 0;
        for (int i = 0; i < V; ++i) s += std::exp(static_cast<double>(x[i]) - m);
        const double ls = std::log(s);
        out.resize(V);
        for (int i = 0; i < V; ++i) out[i] = static_cast<double>(x[i]) - m - ls;
    };
    std::vector<double> la, lb;
    std::vector<double> kl_all, kl_late;
    int top_all = 0, top_late = 0, n_late = 0;
    for (size_t k = 0; k < rows.size(); ++k) {
        const float* a = A.data() + k * V;
        const float* b = B.data() + k * V;
        logsm(b, la);   // reference: the one-token steps
        logsm(a, lb);   // batch
        double kl = 0;
        for (int i = 0; i < V; ++i) kl += std::exp(la[i]) * (la[i] - lb[i]);
        const bool same_top = argmax_f32(a, V) == argmax_f32(b, V);
        kl_all.push_back(kl);
        top_all += same_top;
        if (rows[k] >= win) { kl_late.push_back(kl); top_late += same_top; ++n_late; }
    }
    auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto mx = [](const std::vector<double>& v) { double m = 0; for (double x : v) m = std::max(m, x); return m; };
    printf("[ring-check] %d tokens, batches of %d, %zu rows compared (stride %d), swa_win %d, backend %s\n", n, piece, rows.size(), stride, win, backend->backend_name());
    printf("[ring-check] all rows: KL median %.3g max %.3g, top-1 agreement %.1f%%\n", med(kl_all), mx(kl_all), 100.0 * top_all / std::max<size_t>(1, rows.size()));
    printf("[ring-check] rows at position >= %d (%d rows): KL median %.3g max %.3g, top-1 agreement %.1f%%\n", win, n_late, med(kl_late), mx(kl_late),
           n_late ? 100.0 * top_late / n_late : 100.0);
    const bool ok = n_late == 0 || (med(kl_late) < 0.02 && 100.0 * top_late / n_late > 90.0);
    printf("[ring-check] RESULT: %s\n", ok ? "ok" : "BATCH LOGITS DIFFER FROM THE ONE-TOKEN STEPS AFTER THE WINDOW");
    return ok ? 0 : 1;
}

static int32_t argmax_f32(const float* lg, int V);
// --batch-check N [--parallel K] --prompt-ids ...: the same request alone in slot 0 versus inside a batch of K slots (K - 1 other
// requests of the same length, made from the same ids rotated), scheduled the way ink_serve.h does it (decode tokens first, the rest of
// the chunk budget split round-robin over the prefilling slots).  Slot 0 is teacher-forced with the tokens of its solo greedy run, so
// both runs see the same positions; per step it prints the KL of the batched logits against the solo logits, whether the top-1 agrees
// and the logit gap of the solo top-2 (a near-tie flips on rounding, a wide gap does not).  Exit 1 when a step's KL is above 1.0 (garbage logits; the Metal prefill path alone differs by about 0.1 between chunk shapes).
static int cmd_batch_check(const std::string& model_path, const BackendOpts& bopts, bool force_cpu,
                           const std::vector<int32_t>& ids, int n_gen) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    BackendOpts bo = bopts;
    bo.enable_dspark = false;
    const int K = std::max(1, bo.parallel);
    if (static_cast<int>(ids.size()) + n_gen + 64 > bo.max_ctx) bo.max_ctx = static_cast<int>(ids.size()) + n_gen + 64;
    auto backend = make_default_backend(force_cpu, &bo);
    if (!backend->init(mf, bo, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int eff_ctx = (backend->effective_max_ctx() > 0) ? backend->effective_max_ctx() : bo.max_ctx;
    for (int s = 0; s < K; ++s)
        if (!backend->alloc_slot(s, eff_ctx, bo.kv_dtype, err)) { fprintf(stderr, "alloc_slot: %s\n", err.c_str()); return 1; }
    const int V = mf.m.c.vocab_unpadded;
    const int P = static_cast<int>(ids.size());
    const int chunk = std::max(1, bo.chunk);
    const char* sc_env = std::getenv("INK_BC_SOLO_CHUNK");    // the solo run's chunk size (default: the serve chunk)
    const int solo_chunk = (sc_env && std::atoi(sc_env) > 0) ? std::atoi(sc_env) : chunk;
    StepBatch batch;

    // ---- solo: slot 0 only, the same chunking as the serve loop with one request
    std::vector<std::vector<float>> solo;       // solo[0] = logits after the prompt, solo[k] after decode step k
    std::vector<int32_t> toks;                  // the greedy tokens fed back
    {
        std::vector<float> lg(V);
        int pos = 0;
        while (pos < P) {
            const int take = std::min(solo_chunk, P - pos);
            batch.clear();
            for (int k = 0; k < take; ++k) batch.push(0, pos + k, ids[pos + k], pos + k == P - 1);
            if (!backend->forward_batch(batch, (pos + take == P) ? lg.data() : nullptr, nullptr, err)) { fprintf(stderr, "solo prefill: %s\n", err.c_str()); return 1; }
            pos += take;
        }
        solo.push_back(lg);
        for (int g = 0; g < n_gen; ++g) {
            const int32_t t = argmax_f32(solo.back().data(), V);
            toks.push_back(t);
            batch.clear();
            batch.push(0, P + g, t, true);
            if (!backend->forward_batch(batch, lg.data(), nullptr, err)) { fprintf(stderr, "solo decode: %s\n", err.c_str()); return 1; }
            solo.push_back(lg);
        }
    }
    for (int s = 0; s < K; ++s) backend->reset_slot(s);

    // ---- batched: K slots, slot 0 the same request (teacher-forced), the others rotated copies that decode greedily
    std::vector<std::vector<int32_t>> pr(K);
    for (int s = 0; s < K; ++s) {
        pr[s].resize(P);
        for (int i = 0; i < P; ++i) pr[s][i] = ids[(i + s * 997) % P];
    }
    std::vector<int> fed(K, 0), npos(K, 0);
    std::vector<int32_t> next(K, -1);
    std::vector<int> gdone(K, 0);
    std::vector<std::vector<float>> bat(1 + n_gen);
    std::vector<float> lgbuf;
    int s0_have = 0;                            // solo-indexed logits of slot 0 collected so far
    int guard = 0;
    while (s0_have < 1 + n_gen && ++guard < 1000000) {
        batch.clear();
        std::vector<int> want_slots;
        int budget = std::max(chunk, K);
        std::vector<int> take(K, 0);
        for (int s = 0; s < K; ++s) {
            if (fed[s] < P || gdone[s] >= n_gen + 1) continue;
            const int32_t t = (s == 0) ? toks[gdone[0] - 1] : next[s];
            batch.push(s, P + gdone[s] - 1, t, true);
            want_slots.push_back(s);
            --budget;
        }
        if (budget > 0) {
            std::vector<int> pf;
            for (int s = 0; s < K; ++s) if (fed[s] < P) pf.push_back(s);
            if (!pf.empty()) {
                const int per = std::max(1, budget / static_cast<int>(pf.size()));
                for (int s : pf) {
                    take[s] = std::min(P - fed[s], per);
                    for (int k = 0; k < take[s]; ++k) {
                        const bool last = fed[s] + k == P - 1;
                        batch.push(s, fed[s] + k, pr[s][fed[s] + k], last);
                        if (last) want_slots.push_back(s);
                    }
                }
            }
        }
        const int n_want = batch.count_logits();
        lgbuf.assign(static_cast<size_t>(std::max(1, n_want)) * V, 0.0f);
        if (!backend->forward_batch(batch, n_want > 0 ? lgbuf.data() : nullptr, nullptr, err)) { fprintf(stderr, "batched: %s\n", err.c_str()); return 1; }
        // the rows come back in batch order
        for (size_t r = 0; r < want_slots.size(); ++r) {
            const int s = want_slots[r];
            const float* row = lgbuf.data() + r * static_cast<size_t>(V);
            if (s == 0) {
                bat[gdone[0]].assign(row, row + V);
                ++s0_have;
            }
            next[s] = argmax_f32(row, V);
            ++gdone[s];
        }
        for (int s = 0; s < K; ++s) fed[s] += take[s];
        for (int s = 0; s < K; ++s) if (take[s] > 0 && fed[s] == P) { /* its first logits row was taken above */ }
    }
    if (s0_have < 1 + n_gen) { fprintf(stderr, "batch-check: scheduler did not finish\n"); return 1; }

    int rc = 0, flips = 0, wide_flips = 0;
    double kld_max = 0.0, kld_sum = 0.0;
    int first_flip = -1;
    for (int g = 0; g <= n_gen; ++g) {
        const float* p_lg = solo[g].data();
        const float* q_lg = bat[g].data();
        int bp = 0, bq = 0;
        for (int v = 1; v < V; ++v) { if (p_lg[v] > p_lg[bp]) bp = v; if (q_lg[v] > q_lg[bq]) bq = v; }
        float second = -1e30f;
        for (int v = 0; v < V; ++v) if (v != bp && p_lg[v] > second) second = p_lg[v];
        const float gap = p_lg[bp] - second;
        double zp = 0.0, zq = 0.0;
        for (int v = 0; v < V; ++v) { zp += std::exp(p_lg[v] - p_lg[bp]); zq += std::exp(q_lg[v] - q_lg[bq]); }
        const double lzp = p_lg[bp] + std::log(zp), lzq = q_lg[bq] + std::log(zq);
        double kld = 0.0;
        for (int v = 0; v < V; ++v) { const double lp = p_lg[v] - lzp, lq = q_lg[v] - lzq; kld += std::exp(lp) * (lp - lq); }
        kld_sum += kld;
        kld_max = std::max(kld_max, kld);
        if (bp != bq) { ++flips; if (first_flip < 0) first_flip = g; if (gap > 0.5f) ++wide_flips; }
        if (g < 4 || bp != bq || kld > 0.01)
            printf("[batch-check] step %d: KL %.6f top1 %s (solo %d, batched %d) solo top1-top2 gap %.4f\n", g, kld, bp == bq ? "same" : "DIFFERENT", bp, bq, gap);
        if (kld > 1.0) rc = 1;
    }
    printf("[batch-check] K=%d slots, prompt %d tokens, %d steps: mean KL %.6f, max KL %.6f, top-1 flips %d (first at step %d, with a solo gap above 0.5: %d)\n",
           K, P, n_gen + 1, kld_sum / (n_gen + 1), kld_max, flips, first_flip, wide_flips);
    return rc;
}

// #6671: the drafter must be lossless.  --spec-check N: greedy-generate N tokens (a) one token per forward (B = 1), twice,
// (b) through dspark_spec_step_greedy exactly as the serve loop calls it (with the same fallbacks), and (c) compare the
// logits of an M-token verify window (forward_batch with M rows of one slot) bitwise against the B = 1 logits of the same
// positions, for M = 2, 3, 4, 8.  Prints the first token where (a) and (b) differ and, for every window size, which rows
// differ.  Exit 0: all identical, 1: something differs.
static uint64_t fnv_f32(const float* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n * 4; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static int32_t argmax_f32(const float* lg, int V) {
    int32_t best = 0;
    for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
    return best;
}
static int cmd_spec_check(const std::string& model_path, const BackendOpts& bopts, bool force_cpu,
                          const std::vector<int32_t>& ids, int n_gen) {
    InkModelFile mf;
    std::string err;
    if (!mf.open(model_path, err, false, bopts.dspark_path)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    BackendOpts bo = bopts;
    bo.enable_dspark = true;
    bo.parallel = 1;
    if (static_cast<int>(ids.size()) + n_gen + 64 > bo.max_ctx) bo.max_ctx = static_cast<int>(ids.size()) + n_gen + 64;
    auto backend = make_default_backend(force_cpu, &bo);
    if (!backend->init(mf, bo, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int eff_ctx = (backend->effective_max_ctx() > 0) ? backend->effective_max_ctx() : bo.max_ctx;
    if (!backend->alloc_slot(0, eff_ctx, bo.kv_dtype, err)) { fprintf(stderr, "alloc_slot: %s\n", err.c_str()); return 1; }
    const int V = mf.m.c.vocab_unpadded;
    const int P = static_cast<int>(ids.size());
    std::vector<float> logits(static_cast<size_t>(V) * 9);
    StepBatch batch;
    auto prefill = [&]() -> bool {
        backend->reset_slot(0);
        for (int p0 = 0; p0 < P; p0 += bo.chunk) {
            const int len = std::min(bo.chunk, P - p0);
            batch.clear();
            batch.capture_taps = true;
            for (int k = 0; k < len; ++k) batch.push(0, p0 + k, ids[p0 + k], p0 + k == P - 1);
            if (!backend->forward_batch(batch, (p0 + len == P) ? logits.data() : nullptr, nullptr, err)) {
                fprintf(stderr, "prefill: %s\n", err.c_str());
                return false;
            }
        }
        return true;
    };
    const int kKeep = 16;                      // B = 1 logits kept for the window comparison
    std::vector<std::vector<float>> keep;      // keep[k] = logits after feeding token k of the generated sequence
    auto run_plain = [&](std::vector<int32_t>& out, std::vector<uint64_t>* hashes, bool keep_logits) -> bool {
        if (!prefill()) return false;
        int32_t next = argmax_f32(logits.data(), V);
        out.assign(1, next);
        int pos = P;
        static const bool dump_steps = [] { const char* e = std::getenv("STRATA_SPEC_CHECK_STEPS"); return e && e[0] == '1'; }();
        // STRATA_SPEC_CHECK_SNAP=1: value-neutral work between the prompt and the first decode step (a ShortConv snapshot: allocations
        // and two read-only kernels, what a prompt checkpoint save does); the answer must not change
        static const bool neutral_snap = [] { const char* e = std::getenv("STRATA_SPEC_CHECK_SNAP"); return e && e[0] == '1'; }();
        if (neutral_snap) backend->snapshot_slot(0);
        if (hashes && dump_steps) printf("[steps] prefill %016llx top1 %d\n", static_cast<unsigned long long>(fnv_f32(logits.data(), V)), next);
        while (static_cast<int>(out.size()) < n_gen) {
            batch.clear();
            batch.capture_taps = true;
            batch.push(0, pos++, next, true);
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) { fprintf(stderr, "decode: %s\n", err.c_str()); return false; }
            if (hashes && dump_steps) printf("[steps] %zu %016llx top1 %d\n", hashes->size(), static_cast<unsigned long long>(fnv_f32(logits.data(), V)), argmax_f32(logits.data(), V));
            if (hashes) hashes->push_back(fnv_f32(logits.data(), V));
            if (keep_logits && static_cast<int>(keep.size()) < kKeep) keep.emplace_back(logits.begin(), logits.begin() + V);
            next = argmax_f32(logits.data(), V);
            out.push_back(next);
        }
        return true;
    };
    std::vector<int32_t> a1, a2, b;
    std::vector<uint64_t> h1, h2;
    if (!run_plain(a1, &h1, true) || !run_plain(a2, &h2, false)) return 1;
    int rc = 0;
    size_t plain_diff = 0;
    for (size_t i = 0; i < h1.size(); ++i) if (h1[i] != h2[i]) ++plain_diff;
    printf("[spec-check] plain B=1 twice: tokens %s, logits hashes differ at %zu of %zu steps\n",
           a1 == a2 ? "identical" : "DIFFER", plain_diff, h1.size());
    if (a1 != a2 || plain_diff) rc = 1;
    {   // one number per run for the repeat-run check: the logits hashes of every plain step and the token ids
        uint64_t fp = 1469598103934665603ull;
        for (uint64_t h : h1) { fp ^= h; fp *= 1099511628211ull; }
        for (int32_t t : a1) { fp ^= static_cast<uint64_t>(static_cast<uint32_t>(t)); fp *= 1099511628211ull; }
        printf("[spec-check] fingerprint (plain logits + ids) %016llx\n", static_cast<unsigned long long>(fp));
    }

    // (b) the speculative loop, the way ink_serve.h drives it
    {
        if (!prefill()) return 1;
        int32_t next = argmax_f32(logits.data(), V);
        b.assign(1, next);
        int pos = P;
        long steps = 0, spec_steps = 0, draft_steps = 0, single_steps = 0, drafted_sum = 0, emitted_sum = 0;
        std::vector<float> lg9(static_cast<size_t>(V) * 9);
        while (static_cast<int>(b.size()) < n_gen) {
            ++steps;
            const int rem = n_gen - static_cast<int>(b.size());
            if (rem >= 2 && pos + 12 < eff_ctx) {
                int32_t emitted[32]{};
                int drafted = 0;
                const int n_emit = backend->dspark_spec_step_greedy(0, next, pos, rem, emitted, &drafted, err);
                if (n_emit < 0) { fprintf(stderr, "spec: %s\n", err.c_str()); return 1; }
                if (n_emit > 0) {
                    ++spec_steps;
                    drafted_sum += drafted;
                    emitted_sum += n_emit;
                    for (int i = 0; i < n_emit && static_cast<int>(b.size()) < n_gen; ++i) b.push_back(emitted[i]);
                    next = emitted[n_emit - 1];
                    pos += n_emit;
                    continue;
                }
                int32_t draft_ids[8]{};
                const int nd = backend->dspark_draft(0, next, pos, std::min(7, rem - 1), draft_ids, err);
                if (nd > 0) {
                    ++draft_steps;
                    backend->snapshot_slot(0);
                    batch.clear();
                    batch.capture_taps = true;
                    batch.push(0, pos, next, true);
                    for (int i = 0; i < nd; ++i) batch.push(0, pos + 1 + i, draft_ids[i], true);
                    if (!backend->forward_batch(batch, lg9.data(), nullptr, err)) { fprintf(stderr, "verify: %s\n", err.c_str()); return 1; }
                    int accepted = 0;
                    for (int i = 0; i <= nd; ++i) {
                        const int32_t tok = argmax_f32(lg9.data() + static_cast<size_t>(i) * V, V);
                        b.push_back(tok);
                        ++accepted;
                        next = tok;
                        if (i < nd && tok == draft_ids[i]) continue;
                        break;
                    }
                    if (accepted <= nd) backend->rollback_slot(0, pos + accepted);
                    pos += accepted;
                    drafted_sum += nd;
                    emitted_sum += accepted;
                    continue;
                }
            }
            batch.clear();
            batch.capture_taps = true;
            batch.push(0, pos++, next, true);
            if (!backend->forward_batch(batch, logits.data(), nullptr, err)) { fprintf(stderr, "decode: %s\n", err.c_str()); return 1; }
            next = argmax_f32(logits.data(), V);
            b.push_back(next);
            ++single_steps;
        }
        b.resize(std::min<size_t>(b.size(), static_cast<size_t>(n_gen)));
        size_t first = 0;
        while (first < b.size() && first < a1.size() && b[first] == a1[first]) ++first;
        const bool same = first == b.size() && b.size() == a1.size();
        printf("[spec-check] speculative vs plain over %zu tokens: %s", b.size(), same ? "identical" : "DIFFER");
        if (!same) printf(" (first difference at generated token %zu: plain %d, speculative %d)", first,
                          first < a1.size() ? a1[first] : -1, first < b.size() ? b[first] : -1);
        {
            uint64_t fb = 1469598103934665603ull;
            for (int32_t t : b) { fb ^= static_cast<uint64_t>(static_cast<uint32_t>(t)); fb *= 1099511628211ull; }
            printf("\n[spec-check] speculative ids fingerprint %016llx", static_cast<unsigned long long>(fb));
        }
        printf("\n[spec-check] steps=%ld (fused %ld, draft+verify %ld, single %ld), drafted %ld, emitted %ld\n", steps, spec_steps,
               draft_steps, single_steps, drafted_sum, emitted_sum);
        if (const char* e = std::getenv("STRATA_SPEC_CHECK_IDS"); e && e[0] == '1') {   // the generated ids, for a quality check of the answer text
            printf("[spec-check] IDS plain:");
            for (size_t i = 0; i < a1.size(); ++i) printf("%c%d", i ? ',' : ' ', a1[i]);
            printf("\n[spec-check] IDS drafter:");
            for (size_t i = 0; i < b.size(); ++i) printf("%c%d", i ? ',' : ' ', b[i]);
            printf("\n");
        }
        if (!same) rc = 1;
        if (const char* q = std::getenv("STRATA_SPEC_CHECK_QUICK"); q && q[0] == '1') {   // skip the verify-window comparisons
            printf("[spec-check] RESULT: %s\n", rc ? "DIFFERS" : "identical");
            return rc;
        }
    }

    // (c) verify windows: M rows of one slot in one forward_batch against the B = 1 logits of the same positions
    for (int M : {2, 3, 4, 8}) {
        if (M > static_cast<int>(keep.size()) || M > static_cast<int>(a1.size())) continue;
        if (!prefill()) return 1;
        // keep[k] is the logits after feeding token a1[k] at position P + k (token a1[0] is fed first)
        backend->snapshot_slot(0);   // as the serve loop does before a verify window
        batch.clear();
        batch.capture_taps = true;
        for (int k = 0; k < M; ++k) batch.push(0, P + k, a1[k], true);
        std::vector<float> lgm(static_cast<size_t>(V) * M);
        if (!backend->forward_batch(batch, lgm.data(), nullptr, err)) { fprintf(stderr, "window: %s\n", err.c_str()); return 1; }
        std::string rows;
        int nd = 0;
        double worst = 0;
        for (int k = 0; k < M; ++k) {
            const float* x = lgm.data() + static_cast<size_t>(k) * V;
            const float* y = keep[k].data();
            double mx = 0;
            for (int i = 0; i < V; ++i) mx = std::max(mx, static_cast<double>(std::fabs(x[i] - y[i])));
            const bool eq = std::memcmp(x, y, static_cast<size_t>(V) * 4) == 0;
            if (!eq) { ++nd; rows += " " + std::to_string(k); worst = std::max(worst, mx); }
        }
        printf("[spec-check] window M=%d: %d of %d rows differ from B=1 (rows:%s) max|dlogit|=%.3g\n", M, nd, M, nd ? rows.c_str() : " none", worst);
        if (nd) rc = 1;
    }
    // (d) where does a window row first leave the B = 1 values?  The trunk hidden state after visits 5, 23, 41 and 48
    // (the taps the drafter reads), 8 rows against 8 single steps
    if (static_cast<int>(a1.size()) >= 8) {
        const int M = 8, D = mf.m.c.hidden;
        std::vector<std::vector<float>> single_taps[4];
        for (int t = 0; t < 4; ++t) single_taps[t].resize(M);
        if (!prefill()) return 1;
        for (int k = 0; k < M; ++k) {
            batch.clear();
            batch.capture_taps = true;
            batch.push(0, P + k, a1[k], true);
            TapCapture tc;
            if (!backend->forward_batch(batch, logits.data(), &tc, err)) { fprintf(stderr, "taps single: %s\n", err.c_str()); return 1; }
            for (int t = 0; t < 4; ++t) single_taps[t][k].assign(tc.taps[t].begin(), tc.taps[t].begin() + D);
        }
        if (!prefill()) return 1;
        backend->snapshot_slot(0);
        batch.clear();
        batch.capture_taps = true;
        for (int k = 0; k < M; ++k) batch.push(0, P + k, a1[k], true);
        std::vector<float> lgm(static_cast<size_t>(V) * M);
        TapCapture tcw;
        if (!backend->forward_batch(batch, lgm.data(), &tcw, err)) { fprintf(stderr, "taps window: %s\n", err.c_str()); return 1; }
        for (int t = 0; t < 4; ++t) {
            std::string rows;
            double worst = 0;
            for (int k = 0; k < M; ++k) {
                const float* x = tcw.taps[t].data() + static_cast<size_t>(k) * D;
                const float* y = single_taps[t][k].data();
                if (std::memcmp(x, y, static_cast<size_t>(D) * 4) != 0) {
                    rows += " " + std::to_string(k);
                    for (int i = 0; i < D; ++i) worst = std::max(worst, static_cast<double>(std::fabs(x[i] - y[i])));
                }
            }
            printf("[spec-check] tap %d (after visit %d), window of %d vs single steps: rows differing:%s max|d|=%.3g\n", t,
                   t == 0 ? 5 : t == 1 ? 23 : t == 2 ? 41 : 48, M, rows.empty() ? " none" : rows.c_str(), worst);
        }
    }
    printf("[spec-check] RESULT: %s\n", rc ? "DIFFERS" : "identical");
    return rc;
}

int main(int argc, char** argv) {
#if defined(__linux__) && defined(INKLING_ENABLE_HIP)
    if (const std::string nb = strata::platform::numa_balancing::keep_scanner_off(); !nb.empty()) {
        std::fprintf(stderr, "%s\n", nb.c_str());
    }
#endif
#if defined(INKLING_ENABLE_CUDA) || defined(INKLING_ENABLE_EP_CUDA)
    if (strata::core::apply_default_cuda_module_loading()) {
        std::fprintf(stderr, "[inkling] volta-forever detected: setting CUDA_MODULE_LOADING=LAZY\n");
    }
#endif
    ServeConfig cfg;
    cfg.version = INKLING_VERSION;
    bool mode_serve = false, mode_bench = false, mode_selfcheck = false, mode_info = false, mode_export = false;
    bool verify_hash = false;
    bool vision_reader_given = false;
    std::string info_file, export_file, export_dir, kld_ref, dump_ref, prompt_ids_str;
    std::string tkl_corpus, tkl_cfgs, tkl_out, tkl_only, tgen_in, tgen_out, tgen_stop, tkl_calib, turbo_preset;
    int tkl_chunk = 2048;
    int bench_prompt = 64, bench_gen = 32, spec_check_n = 0, ring_check_stride = 0, batch_check_n = 0;
    float cli_temp = 0.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--version") {
            printf("strata-inkling %s\n", INKLING_VERSION);
            return 0;
        } else if (a == "--info") {
            mode_info = true; info_file = next();
        } else if (a == "--export-tokenizer") {
            mode_export = true; export_file = next(); export_dir = next();
        } else if (a == "--serve") {
            mode_serve = true;
        } else if (a == "--bench") {
            mode_bench = true;
        } else if (a == "--selfcheck") {
            mode_selfcheck = true;
        } else if (a == "--verify-hash") {
            verify_hash = true;
        } else if (a == "--kld-eval") {
            kld_ref = next();
        } else if (a == "--dump-ref") {
            dump_ref = next();
        } else if (a == "--tail-kl") {
            tkl_corpus = next();
        } else if (a == "--tail-gen") {
            tgen_in = next();
        } else if (a == "--tail-gen-out") {
            tgen_out = next();
        } else if (a == "--tail-stop") {
            tgen_stop = next();
        } else if (a == "--tail-cfgs") {
            tkl_cfgs = next();
        } else if (a == "--tail-out") {
            tkl_out = next();
        } else if (a == "--tail-chunk") {
            tkl_chunk = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--turbo") {
            turbo_preset = "turbo";
        } else if (a == "--turbo-max") {
            turbo_preset = "turbo-max";
        } else if (a == "--tail-calib") {
            tkl_calib = next();
        } else if (a == "--tail-only") {
            tkl_only = next();
        } else if (a == "--native" || a == "--model") {
            cfg.model_path = next();
        } else if (a == "--vision") {
            cfg.vision_path = next();
            cfg.bopts.vision_path = cfg.vision_path;
        } else if (a == "--vision-reader") {
            const std::string v = next();
            if (v != "auto" && v != "gpu" && v != "cpu") {
                fprintf(stderr, "--vision-reader takes auto, gpu or cpu\n");
                return 2;
            }
            cfg.bopts.vision_reader = v;
            vision_reader_given = true;
        } else if (a == "--vision-idle-s") {
            cfg.bopts.vision_idle_s = std::max(0, std::atoi(next().c_str()));
        } else if (a == "--max-context" || a == "--ctx") {
            cfg.bopts.max_ctx = std::atoi(next().c_str());
        } else if (a == "--parallel" || a == "--batch" || a == "--slots") {
            cfg.bopts.parallel = std::max(1, std::min(kMaxParallelSlots, std::atoi(next().c_str())));
        } else if (a == "--tail-role-token") {
            // #7241 (opt-in): the id of <|message_system|>; with --serve, the prompt checkpoint goes in front of a trailing level turn (ink_serve.h)
            cfg.tail_role_token = std::atoll(next().c_str());
        } else if (a == "--chunk") {
            cfg.bopts.chunk = std::max(1, std::atoi(next().c_str()));
            cfg.bopts.chunk_auto = false;
        } else if (a == "--kv-dtype" || a == "--kv") {
            std::string kv = next();
            if (kv == "fp8" || kv == "fp8_e4m3") { cfg.bopts.kv_dtype = KvDtype::FP8_E4M3; cfg.bopts.kv_dtype_auto = false; }
            else if (kv == "fp16" || kv == "f16") { cfg.bopts.kv_dtype = KvDtype::FP16; cfg.bopts.kv_dtype_auto = false; }
            else if (kv == "k8v8" || kv == "i8" || kv == "q8") { cfg.bopts.kv_dtype = KvDtype::K8V8; cfg.bopts.kv_dtype_auto = false; }
            else if (kv == "k8v4") { cfg.bopts.kv_dtype = KvDtype::K8V4; cfg.bopts.kv_dtype_auto = false; }
            else if (kv == "auto") { cfg.bopts.kv_dtype_auto = true; }
            else {
                fprintf(stderr, "unknown --kv value '%s' (expected: auto, fp8, fp16, k8v8, k8v4)\n", kv.c_str());
                std::exit(2);
            }
        } else if (a == "--kv-offload") {
            cfg.bopts.kv_offload = next();
        } else if (a == "--draft") {
            cfg.draft_mode = next();
            cfg.bopts.enable_dspark = (cfg.draft_mode != "off");
            cfg.bopts.draft_forced = (cfg.draft_mode == "on");
        } else if (a == "--draft-beam") {
            cfg.bopts.spec_beam = std::atoi(next().c_str());
        } else if (a == "--draft-cost") {
            cfg.bopts.spec_cost = std::atof(next().c_str());
        } else if (a == "--draft-adapt") {
            cfg.bopts.spec_adapt = true;
        } else if (a == "--draft-plain") {
            cfg.bopts.spec_plain = std::atof(next().c_str());
        } else if (a == "--dspark") {
            cfg.bopts.dspark_path = next();
        } else if (a == "--vram-experts-gb") {
            cfg.bopts.vram_experts_gb = std::atof(next().c_str());
        } else if (a == "--ram-experts-gb") {
            cfg.bopts.ram_experts_gb = std::atof(next().c_str());
        } else if (a == "--ssd-experts") {
            cfg.bopts.ssd_experts = true;
        } else if (a == "--vram-reserve-mib") {
            cfg.bopts.vram_reserve_mib = std::atoi(next().c_str());
        } else if (a == "--cpu-threads") {
            cfg.bopts.cpu_threads = std::atoi(next().c_str());
        } else if (a == "--gpu") {
            cfg.bopts.gpu_index = std::atoi(next().c_str());
        } else if (a == "--gpus" || a == "--devices") {
            std::string spec = next();
            if (spec.find(',') != std::string::npos || a == "--devices") {
                cfg.bopts.gpu_ids.clear();
                for (size_t pos = 0; pos < spec.size();) {
                    size_t q = spec.find(',', pos);
                    if (q == std::string::npos) q = spec.size();
                    if (q > pos) {
                        int id = std::atoi(spec.substr(pos, q - pos).c_str());
                        if (id >= 0 && std::find(cfg.bopts.gpu_ids.begin(), cfg.bopts.gpu_ids.end(), id) == cfg.bopts.gpu_ids.end()) {
                            cfg.bopts.gpu_ids.push_back(id);
                        }
                    }
                    pos = q + 1;
                }
                if (!cfg.bopts.gpu_ids.empty()) {
                    cfg.bopts.gpu_index = cfg.bopts.gpu_ids[0];
                    cfg.bopts.gpus = static_cast<int>(cfg.bopts.gpu_ids.size());
                } else {
                    cfg.bopts.gpus = 1;
                }
            } else {
                cfg.bopts.gpu_ids.clear();
                cfg.bopts.gpus = std::max(1, std::atoi(spec.c_str()));
            }
        } else if (a == "--ep-split") {
            cfg.bopts.ep_split = std::atof(next().c_str());
        } else if (a == "--layer-split") {
            cfg.bopts.layer_split = next();
        } else if (a == "--threads") {
            cfg.bopts.threads = std::atoi(next().c_str());
        } else if (a == "--cpu") {
            cfg.force_cpu = true;
        } else if (a == "--prompt-len") {
            bench_prompt = std::atoi(next().c_str());
        } else if (a == "--gen") {
            bench_gen = std::atoi(next().c_str());
        } else if (a == "--prompt-ids") {
            prompt_ids_str = next();
            if (!prompt_ids_str.empty() && prompt_ids_str[0] == '@') {   // @file: the ids in a text file
                std::ifstream pf(prompt_ids_str.substr(1), std::ios::binary);
                if (!pf) { fprintf(stderr, "cannot read %s\n", prompt_ids_str.c_str() + 1); return 2; }
                prompt_ids_str.assign(std::istreambuf_iterator<char>(pf), std::istreambuf_iterator<char>());
            }
        } else if (a == "--batch-check") {
            batch_check_n = std::atoi(next().c_str());
        } else if (a == "--spec-check") {
            spec_check_n = std::atoi(next().c_str());
        } else if (a == "--ring-check") {
            ring_check_stride = std::atoi(next().c_str());
        } else if (a == "--temp") {
            cli_temp = static_cast<float>(std::atof(next().c_str()));
        } else if (a == "--no-anti-loop") {
            cfg.anti_loop = false;
        } else if (a == "--help" || a == "-h") {
            printf("strata-inkling %s - Unified Inkling-2-Mini (.strata v1.1 / HF safetensors / .gguf) Shape Engine\n"
                   "  --version\n"
                   "  --info PATH [--vision VISION] [--dspark PATH]\n"
                   "  --export-tokenizer PATH OUT_DIR\n"
                   "  --selfcheck [--native PATH] [--verify-hash] [--cpu] [--gpu N] [--gpus 1|2] [--kv auto|fp16|fp8|k8v8|k8v4]\n"
                   "  --bench --native PATH [--parallel 1..64] [--gpu N] [--gpus 1..4] [--layer-split auto|K1,K2,...] [--ep-split F] [--vram-experts-gb F] [--ram-experts-gb F] [--ssd-experts] [--vram-reserve-mib N] [--cpu-threads N] [--kv auto|fp16|fp8|k8v8|k8v4] [--kv-offload auto|on|off] [--prompt-len N] [--gen N]\n"
                   "  --turbo | --turbo-max   opt-in router tail skip presets (answers differ slightly; CUDA and HIP; see docs/INKLING_TURBO.md)\n"
                   "  --kld-eval REF.bin --native PATH\n"
                   "  --serve --native PATH [--vision VISION] [--vision-reader auto|gpu|cpu] [--vision-idle-s N] [--dspark PATH] [--max-context N] [--parallel 1..64] [--gpu N] [--gpus 1..4] [--layer-split auto|K1,K2,...] [--kv auto|fp16|fp8|k8v8|k8v4] [--kv-offload auto|on|off] [--draft auto|dspark|off]\n",
                   INKLING_VERSION);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }

    // Read optional IK2_EP_SPLIT and IK2_GPUS env overrides (#6506)
    if (const char* ep_env = std::getenv("IK2_EP_SPLIT"); ep_env && *ep_env) {
        double v = std::atof(ep_env);
        if (v > 0.0 && v < 1.0) cfg.bopts.ep_split = v;
    }
    if (cfg.bopts.gpus == 1 && cfg.bopts.gpu_ids.empty()) {
        if (const char* g_env = std::getenv("IK2_GPUS"); g_env && *g_env) {
            std::string spec(g_env);
            if (spec.find(',') != std::string::npos) {
                for (size_t pos = 0; pos < spec.size();) {
                    size_t q = spec.find(',', pos);
                    if (q == std::string::npos) q = spec.size();
                    if (q > pos) {
                        int id = std::atoi(spec.substr(pos, q - pos).c_str());
                        if (id >= 0 && std::find(cfg.bopts.gpu_ids.begin(), cfg.bopts.gpu_ids.end(), id) == cfg.bopts.gpu_ids.end()) {
                            cfg.bopts.gpu_ids.push_back(id);
                        }
                    }
                    pos = q + 1;
                }
                if (!cfg.bopts.gpu_ids.empty()) {
                    cfg.bopts.gpu_index = cfg.bopts.gpu_ids[0];
                    cfg.bopts.gpus = static_cast<int>(cfg.bopts.gpu_ids.size());
                }
            } else {
                cfg.bopts.gpus = std::max(1, std::atoi(g_env));
            }
        }
    }

    // --turbo / --turbo-max: the preset goes through the same environment switch as STRATA_INK_TAIL_SKIP=turbo (a value the person set wins)
    if (!turbo_preset.empty()) {
        if (const char* have = std::getenv("STRATA_INK_TAIL_SKIP"); have && have[0]) {
            fprintf(stderr, "[inkling] --%s ignored: STRATA_INK_TAIL_SKIP=%s is set and wins\n", turbo_preset.c_str(), have);
        } else {
#ifdef _WIN32
            _putenv_s("STRATA_INK_TAIL_SKIP", turbo_preset.c_str());
#else
            setenv("STRATA_INK_TAIL_SKIP", turbo_preset.c_str(), 1);
#endif
        }
    }

    // Auto-select FP8_E4M3 KV when parallel >= 16 unless user explicitly forced a format
    if (cfg.bopts.kv_dtype_auto && cfg.bopts.parallel >= 16) {
        cfg.bopts.kv_dtype = KvDtype::FP8_E4M3;
    }

    // The picture reader (ink_vision_policy.h): --vision-reader wins over STRATA_INK_VISION_READER; both default to "auto" (the 24 GB rule).
    if (!vision_reader_given) {
        if (const char* e = std::getenv("STRATA_INK_VISION_READER"); e && (std::string(e) == "gpu" || std::string(e) == "cpu")) cfg.bopts.vision_reader = e;
    }
    if (cfg.bopts.vision_idle_s < 0) {
        if (const char* e = std::getenv("STRATA_INK_VISION_IDLE_S"); e && *e) cfg.bopts.vision_idle_s = std::max(0, std::atoi(e));
    }

    if (mode_serve) {
        strata::core::set_serve_error_line(true);
    }

#ifdef INKLING_ENABLE_CUDA
    // One CUDA GPU (sm_80 or newer: the int8 tensor-core GEMM): a GGUF's Q8_0 dense matrices stay as stored (half the bytes of rounded BF16, the file's own weights).
    // STRATA_INK_Q8_0_DENSE=0 goes back to BF16 (for A/B runs). Several GPUs and the CPU backend keep BF16.
    if (!cfg.force_cpu && cfg.bopts.gpus == 1 && cfg.bopts.layer_split.empty() && ink2::ink_cuda_q8f_supported(cfg.bopts.gpu_index)) {
        const char* q8e = std::getenv("STRATA_INK_Q8_0_DENSE");
        ink2::gguf_detail::keep_q8_0_dense() = !(q8e && q8e[0] == '0');
    }
    // One CUDA GPU (sm_61 or newer: the Q8_ROW kernels): a GSQ-RCO file may compact its trunk (the loader decides per file; STRATA_INK_GGUF_COMPACT=0 / 1 overrides)
    if (!cfg.force_cpu && cfg.bopts.gpus == 1 && cfg.bopts.layer_split.empty() && ink2::ink_cuda_compact_trunk_supported(cfg.bopts.gpu_index)) {
        ink2::gguf_detail::compact_trunk_allowed() = true;
    }
#endif
#if defined(INKLING_ENABLE_HIP) && !defined(INKLING_ENABLE_CUDA)
    // One HIP GPU: a GSQ-RCO file may compact its trunk (Q8_ROW unembed, embedding in pinned host RAM; the loader decides per file, STRATA_INK_GGUF_COMPACT=0 / 1 overrides)
    if (!cfg.force_cpu && cfg.bopts.gpus == 1 && cfg.bopts.layer_split.empty()) {
        ink2::gguf_detail::compact_trunk_allowed() = true;
    }
#endif

    if (mode_info) return cmd_info(info_file, cfg.vision_path, cfg.bopts.dspark_path);
    if (mode_export) return cmd_export_tokenizer(export_file, export_dir);
    if (mode_selfcheck) return cmd_selfcheck(cfg.model_path, cfg.force_cpu, verify_hash, cfg.bopts);
    if (!tkl_calib.empty()) return cmd_tail_calib(cfg.model_path, tkl_calib, tkl_cfgs, tkl_out, tkl_chunk, cfg.bopts, cfg.force_cpu);
    if (!tgen_in.empty()) return cmd_tail_gen(cfg.model_path, tgen_in, tgen_out, tgen_stop, cfg.bopts, cfg.force_cpu);
    if (!tkl_corpus.empty()) return cmd_tail_kl(cfg.model_path, tkl_corpus, tkl_cfgs, tkl_out, tkl_chunk, tkl_only, cfg.bopts, cfg.force_cpu);
    if (!kld_ref.empty()) return cmd_kld_eval(cfg.model_path, kld_ref, cfg.bopts, cfg.force_cpu);
    if (!dump_ref.empty()) return cmd_dump_ref(cfg.model_path, dump_ref, prompt_ids_str, cfg.bopts, cfg.force_cpu);
    if (ring_check_stride > 0) {
        std::vector<int32_t> rc_ids;
        std::string rc_err;
        if (prompt_ids_str.empty() || !parse_id_list(prompt_ids_str, rc_ids, rc_err)) {
            fprintf(stderr, "--ring-check needs --prompt-ids (a list or @file)%s%s\n", rc_err.empty() ? "" : ": ", rc_err.c_str());
            return 2;
        }
        return cmd_ring_check(cfg.model_path, cfg.bopts, cfg.force_cpu, rc_ids, ring_check_stride);
    }
    if (batch_check_n > 0) {
        std::vector<int32_t> bc_ids;
        std::string bc_err;
        if (prompt_ids_str.empty() || !parse_id_list(prompt_ids_str, bc_ids, bc_err)) {
            fprintf(stderr, "--batch-check needs --prompt-ids (a list or @file)\n");
            return 2;
        }
        return cmd_batch_check(cfg.model_path, cfg.bopts, cfg.force_cpu, bc_ids, batch_check_n);
    }
    if (spec_check_n > 0) {
        std::vector<int32_t> sc_ids;
        std::string sc_err;
        if (prompt_ids_str.empty() || !parse_id_list(prompt_ids_str, sc_ids, sc_err)) {
            fprintf(stderr, "--spec-check needs --prompt-ids (a list or @file)%s%s\n", sc_err.empty() ? "" : ": ", sc_err.c_str());
            return 2;
        }
        return cmd_spec_check(cfg.model_path, cfg.bopts, cfg.force_cpu, sc_ids, spec_check_n);
    }
    if (mode_bench) return cmd_bench(cfg.model_path, cfg.bopts, bench_prompt, bench_gen, cfg.force_cpu, cfg.bopts.enable_dspark);

    if (cfg.model_path.empty()) {
        fprintf(stderr, "error: --native <model.strata|hf_dir|model.gguf> is required\n");
        return 2;
    }

    InkModelFile mf;
    std::string err;
    if (!mf.open(cfg.model_path, err, false, cfg.bopts.dspark_path)) {
        if (mode_serve) proto("ERR open: %s", err.c_str());
        else fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }

    auto backend = make_default_backend(cfg.force_cpu, &cfg.bopts);
    if (!backend->init(mf, cfg.bopts, err)) {
        if (mode_serve) proto("ERR init: %s", err.c_str());
        else fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }

    if (const char* ts = std::getenv("STRATA_INK_TAIL_SKIP"); ts && ts[0] && std::string(ts) != "off" && std::string(ts) != "0") {
        // the opt-in tail skip (--turbo, STRATA_INK_TAIL_SKIP): only the CUDA and HIP backends have it; say so where it has no effect
        std::string terr;
        if (backend->set_tail_skip(ts, terr)) fprintf(stderr, "[inkling] tail skip on (%s): answers differ slightly from exact\n", ts);
        else fprintf(stderr, "[inkling] STRATA_INK_TAIL_SKIP=%s has no effect on the %s backend (%s): answers are exact\n", ts, backend->backend_name(), terr.c_str());
    }

    if (mode_serve) {
        const int rc = run_serve_loop(cfg, mf, *backend);
        // QUIT: free the backend here, as the old end of main did, then leave without exit()'s own teardown (leave_after_quit, ink_serve.h)
        backend.reset();
        leave_after_quit(rc);
    }

    // Direct CLI generation mode
    if (!backend->alloc_slot(0, cfg.bopts.max_ctx, cfg.bopts.kv_dtype, err)) {
        fprintf(stderr, "alloc_slot: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    if (!prompt_ids_str.empty()) {
        if (!parse_id_list(prompt_ids_str, ids, err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
    } else {
        ids = {200002, 200004, 100, 200010, 200000, 200004, 101, 200010, 200001, 200004};
    }
    const int V = mf.m.c.vocab_unpadded;
    std::vector<float> logits(V);
    StepBatch batch;
    batch.capture_taps = cfg.bopts.enable_dspark;
    for (size_t i = 0; i < ids.size(); ++i) batch.push(0, static_cast<int>(i), ids[i], i + 1 == ids.size());
    if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
        fprintf(stderr, "prefill: %s\n", err.c_str());
        return 1;
    }
    SampleOpts so;
    so.temperature = cli_temp;
    std::mt19937_64 rng(42);
    std::vector<int32_t> hist = ids;
    int pos = static_cast<int>(ids.size());
    // INK_LOGITS_HASH=1: FNV-1a of the raw logits bytes of every step on stderr (repeat-run fingerprint, bitwise).
    const bool hash_logits = std::getenv("INK_LOGITS_HASH") && std::getenv("INK_LOGITS_HASH")[0] == '1';
    auto logits_hash = [&](int step) {
        if (!hash_logits) return;
        uint64_t h = 1469598103934665603ull;
        const unsigned char* p = reinterpret_cast<const unsigned char*>(logits.data());
        for (size_t i = 0; i < logits.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
        fprintf(stderr, "[logits-hash] step %d %016llx\n", step, static_cast<unsigned long long>(h));
    };
    logits_hash(-1);
    for (int g = 0; g < bench_gen; ++g) {
        int32_t tok = sample_logits(logits.data(), V, so, hist, rng);
        printf("%d%s", tok, (g + 1 == bench_gen) ? "\n" : ",");
        fflush(stdout);
        hist.push_back(tok);
        if (tok == mf.m.c.eos_token_id || tok == mf.m.c.eot_token_id) break;
        batch.clear();
        batch.push(0, pos++, tok, true);
        if (!backend->forward_batch(batch, logits.data(), nullptr, err)) {
            fprintf(stderr, "decode: %s\n", err.c_str());
            return 1;
        }
        logits_hash(g);
    }
    return 0;
}
