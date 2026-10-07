// src/core/glm_pack_test.cpp - the pack-backed Glm5Model against the REAL UD-IQ1_S weights.
//
//   glm_pack_test <pack_dir> <n_tokens> [out.bin]
//
// Teacher-forces token ids 1..n (streaming, one forward per token) through the pack-backed
// runner and writes the (T, vocab) float32 logits.  The cross-check is llama.cpp on the same
// GGUF (llm_logits_dump) - docs/GLM5-FLASH.md §9.
#include "strata/core/glm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <pack_dir> <n_tokens> [out.bin]\n", argv[0]);
        return 2;
    }
    const std::string pack = argv[1];
    const int n_tok = std::atoi(argv[2]);
    const char* out_path = argc > 3 ? argv[3] : nullptr;

    strata::core::Glm5Model model;
    std::string err;
    if (!model.load_pack_env(pack, 4096, err)) {
        std::fprintf(stderr, "glm_pack_test: %s\n", err.c_str());
        return 1;
    }
    const size_t V = (size_t) model.geometry().n_vocab;
    std::fprintf(stderr, "glm_pack_test: loaded (vocab %zu, layers %d)\n", V, model.geometry().n_layers);

    if (getenv("STRATA_GLM_TEST_PIPE")) {
        // ONE forward for all tokens: with STRATA_GLM_SPLIT this exercises the cross-token pipeline,
        // and the single output row must be BITWISE equal to the serial run's LAST row
        std::vector<int32_t> all((size_t) n_tok);
        for (int t = 0; t < n_tok; ++t) all[(size_t) t] = (int32_t) ((t % 511) + 1);
        std::vector<float> lg;
        if (!model.forward(all, lg, err)) {
            std::fprintf(stderr, "glm_pack_test: pipe forward: %s\n", err.c_str());
            return 1;
        }
        if (out_path) {
            FILE* f = std::fopen(out_path, "wb");
            if (f) {
                std::fwrite(lg.data(), sizeof(float), lg.size(), f);
                std::fclose(f);
            }
        }
        int am = 0;
        for (size_t v = 1; v < V; ++v)
            if (lg[v] > lg[(size_t) am]) am = (int) v;
        std::printf("pipe: last-position argmax %d (%.6f)\n", am, lg[(size_t) am]);
        return 0;
    }

    // GLM_TEST_IDS=<file>: teacher-force those ids (comma/space separated) instead of 1..n - real text
    // is what the fast-vs-reference comparison needs (synthetic ids sit in degenerate near-ties)
    std::vector<int32_t> ids;
    if (const char* fp = getenv("GLM_TEST_IDS")) {
        FILE* f = std::fopen(fp, "r");
        if (f) {
            long v = 0;
            int c = 0;
            bool in = false;
            while ((c = std::fgetc(f)) != EOF) {
                if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); in = true; }
                else if (in) { ids.push_back((int32_t) v); v = 0; in = false; }
            }
            if (in) ids.push_back((int32_t) v);
            std::fclose(f);
        }
    }
    // GLM_TEST_WARM=N: run the sequence N times first (then reset) - the fast path's expert tiers fill, so the
    // recorded pass runs its experts where a warm decode would (VRAM), not on the cold CPU path
    if (const char* wv = getenv("GLM_TEST_WARM")) {
        for (int r = 0; r < std::atoi(wv); ++r) {
            for (int t = 0; t < n_tok; ++t) {
                std::vector<float> lg;
                const int32_t tok = !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1);
                if (!model.forward({tok}, lg, err)) {
                    std::fprintf(stderr, "glm_pack_test: warm forward at %d: %s\n", t, err.c_str());
                    return 1;
                }
            }
            model.reset();
        }
        std::fprintf(stderr, "glm_pack_test: warmed (%s passes)\n", wv);
    }
    std::vector<float> all;
    // GLM_TEST_PREFILL=k: the first k positions through the batched prompt path (model.prefill), the rest through
    // the token path as usual - the recorded rows then start at position k (compare against a plain run's tail)
    int t_first = 0;
    if (const char* pv = getenv("GLM_TEST_PREFILL")) {
        const int k = std::min(std::atoi(pv), getenv("GLM_TEST_DUMP") ? n_tok : n_tok - 1);
        if (k > 0) {
            std::vector<int32_t> pre((size_t) k);
            for (int t = 0; t < k; ++t) pre[(size_t) t] = !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1);
            const int32_t nxt = k < n_tok ? (!ids.empty() ? ids[(size_t) k % ids.size()] : (int32_t) ((k % 511) + 1)) : -1;
            if (!model.prefill(pre, err, nxt)) {
                std::fprintf(stderr, "glm_pack_test: prefill of %d: %s\n", k, err.empty() ? "unavailable" : err.c_str());
                return 1;
            }
            t_first = k;
            std::fprintf(stderr, "glm_pack_test: prefilled %d positions\n", k);
        }
    }
    // GLM_TEST_MTP=1: after every position t the NextN block drafts t + 2 from (h_t, token t + 1); the next position's
    // argmax is what a greedy speculative decode would compare it with
    const bool mtp_probe = getenv("GLM_TEST_MTP") != nullptr && model.has_mtp();
    int draft_prev = -1, mtp_n = 0, mtp_ok = 0;
    for (int t = t_first; t < n_tok; ++t) {
        std::vector<float> lg;
        const int32_t tok = !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1);
        if (!model.forward({tok}, lg, err)) {
            std::fprintf(stderr, "glm_pack_test: forward at %d: %s\n", t, err.c_str());
            return 1;
        }
        all.insert(all.end(), lg.begin(), lg.end());
        int am = 0;
        for (size_t v = 1; v < V; ++v)
            if (lg[v] > lg[am]) am = (int) v;
        if (mtp_probe) {
            static const int mtp_from = getenv("GLM_TEST_MTP_FROM") ? std::atoi(getenv("GLM_TEST_MTP_FROM")) : 0;
            if (draft_prev >= 0 && t >= mtp_from) {
                ++mtp_n;
                mtp_ok += draft_prev == am;
            }
            draft_prev = -1;
            if (t + 1 < n_tok) {
                const int32_t nxt = !ids.empty() ? ids[(size_t) (t + 1) % ids.size()] : (int32_t) (((t + 1) % 511) + 1);
                draft_prev = model.mtp_draft(nxt, err);
                if (draft_prev < 0 && !err.empty()) {
                    std::fprintf(stderr, "glm_pack_test: mtp at %d: %s\n", t, err.c_str());
                    return 1;
                }
            }
        }
        std::printf("pos %d tok %d argmax %d (%.6f)\n", t, tok, am, lg[(size_t) am]);
    }
    // GLM_TEST_DUMP=<path>: the sequence state after the run (Glm5Model::dump_state), for per-layer comparisons
    if (const char* dp = getenv("GLM_TEST_DUMP"))
        if (!model.dump_state(dp)) std::fprintf(stderr, "glm_pack_test: state dump failed\n");
    if (out_path) {
        FILE* f = std::fopen(out_path, "wb");
        if (f) {
            std::fwrite(all.data(), sizeof(float), all.size(), f);
            std::fclose(f);
        }
    }
    if (mtp_probe)
        std::printf("mtp: draft == next argmax %d/%d (%.1f%%)\n", mtp_ok, mtp_n, 100.0 * mtp_ok / std::max(1, mtp_n));
    std::fprintf(stderr, "glm_pack_test: PASS (%d tokens streamed)\n", n_tok);
    return 0;
}
