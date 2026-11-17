// tests/program/grammar_decode_test.cpp - Task 6's closed loop: accept -> mask -> sample never surfaces a
// token the grammar rejects.  NO MODEL: the GPU only hosts the mask, the logits and the sampling kernel, the
// same host framework `sampler_parity` uses.  The state machine itself is pinned by grammar_test; what this
// test pins is the WIRE between it and the kernel - the mask the host builds must be the mask the kernel
// samples from, step after step, with the sampler's randomness (Philox, (seed, counter)) driving the walk.
//
// The grammar is ("ab" | "ba")* eog: an alternating two-letter walk that may end on a complete pair.  A
// sampled sequence must never repeat a letter, and eog must never land after a dangling one.  100 rounds x
// up to 200 steps: every sampled token is checked against the host mask (a kernel that ignored the mask
// would pick a banned letter within a step or two at uniform 3-way odds) and against the sequence rule
// (which catches a mask that bans the wrong letters in a way the kernel mask check cannot see, e.g. a
// stale stack).
#include "strata/grammar/grammar.hpp"
#include "strata/grammar/grammar_registry.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int kRounds = 100;
constexpr int kSteps = 200;
constexpr int kVocab = 3;                       // "a", "b", "<|eog|>"
constexpr int kWords = (kVocab + 31) / 32;

}   // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }

    auto vocab = std::make_shared<strata::grammar::GrammarVocab>();
    vocab->pieces = {"a", "b", "<|eog|>"};
    vocab->control = {0, 0, 1};
    vocab->eog = {0, 0, 1};

    strata::grammar::GrammarRegistry registry(vocab);
    std::string err;
    const char* text = "root ::= pair*\npair ::= \"ab\" | \"ba\"\n";
    if (!registry.register_grammar("walk", text, err)) {
        std::fprintf(stderr, "register: %s\n", err.c_str());
        return 1;
    }
    const auto entry = registry.find("walk");
    if (!entry) {
        std::fprintf(stderr, "grammar vanished from the registry\n");
        return 1;
    }

    float* d_l = nullptr;
    uint32_t* d_mask = nullptr;
    int* d_o = nullptr;
    check(cudaMalloc(&d_l, kVocab * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_mask, kWords * sizeof(uint32_t)), "malloc mask");
    check(cudaMalloc(&d_o, sizeof(int)), "malloc out");
    const std::vector<float> flat((size_t) kVocab, 100.0f);   // uniform; the mask alone shapes the choice
    check(cudaMemcpy(d_l, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice), "copy logits");

    strata::kernels::SamplerParams p;
    p.temperature = 0.8f;            // sampled, not greedy: Philox drives the walk
    p.top_k = 0;
    p.top_p = 1.0f;
    p.seed = 20261024;
    p.grammar_mask = d_mask;
    p.mask_stride_words = kWords;

    int failures = 0, tokens = 0, rounds_done = 0;
    for (int round = 0; round < kRounds && !failures; ++round) {
        strata::grammar::SlotGrammar sg(entry);   // a fresh walk from the grammar's root
        std::string seq;
        bool done = false;
        for (int step = 0; step < kSteps && !done; ++step) {
            check(cudaMemcpy(d_mask, sg.mask.data(), (size_t) kWords * sizeof(uint32_t),
                             cudaMemcpyHostToDevice), "upload mask");
            p.counter = (uint64_t) round * kSteps + (uint64_t) step;
            strata::kernels::sample_tokens(d_l, 1, kVocab, nullptr, 0, p, d_o, nullptr);
            int y = -1;
            check(cudaMemcpy(&y, d_o, sizeof(int), cudaMemcpyDeviceToHost), "read pick");
            ++tokens;
            if (y < 0 || y >= kVocab) {
                std::printf("round %d step %d: pick %d outside the vocabulary\n", round, step, y);
                ++failures;
                break;
            }
            if (sg.mask[(size_t) (y >> 5)] & (1u << (y & 31))) {
                std::printf("round %d step %d: sampled the BANNED id %d\n", round, step, y);
                ++failures;
                break;
            }
            if (y == 2) {           // eog: the walk is over; the mask allowed it, so the pair is complete
                done = true;
                ++rounds_done;
                break;
            }
            seq += (char) ('a' + y);
            if (!sg.advance(y, err)) {
                std::printf("round %d step %d: advance failed: %s\n", round, step, err.c_str());
                ++failures;
                break;
            }
        }
        // The sampled sequence, independent of the mask: every PAIR alternates (a doubling is legal only at
        // a pair boundary, "ba" + "ab" = "baab"), and eog may only close a complete pair.
        for (size_t i = 0; i + 1 < seq.size(); i += 2) {
            if (seq[i] == seq[i + 1]) {
                std::printf("round %d: pair \"%c%c\" in \"%s\"\n", round, seq[i], seq[i + 1], seq.c_str());
                ++failures;
                break;
            }
        }
        if (done && (seq.size() % 2) != 0) {
            std::printf("round %d: eog after the dangling pair \"%s\"\n", round, seq.c_str());
            ++failures;
        }
    }

    cudaFree(d_mask);
    cudaFree(d_o);
    if (failures) {
        cudaFree(d_l);
        std::printf("grammar decode: %d failures\n", failures);
        return 1;
    }
    std::printf("grammar decode: %d tokens over %d rounds, all legal\n", tokens, rounds_done);

    // #3862 phase 2: the drafted window's mask chain. A window of T rows samples row t after the
    // drafts window[1..t], so row t's mask is the state advanced through them; a draft the grammar
    // cannot take ends the chain there, the row that would emit it refuses it (its mask bans the
    // draft), and the tail rows - never emitted - reuse the last good row's mask. The harness drives
    // the sampler over the chain the way the engine does (one sample_tokens call, the ring base at
    // row 0, a stride per row) with flat logits and greedy picks, so every row's pick is the lowest
    // allowed id and each case's acceptance is deterministic.
    constexpr int kT = 4;
    uint32_t* d_ring = nullptr;
    int* d_out4 = nullptr;
    check(cudaMalloc(&d_ring, (size_t) kT * kWords * sizeof(uint32_t)), "malloc chain ring");
    check(cudaMalloc(&d_out4, kT * sizeof(int)), "malloc chain outs");

    strata::kernels::SamplerParams cp;
    cp.greedy = true;                 // flat logits: the pick is the lowest allowed id, case by case
    cp.temperature = 0.0f;
    cp.grammar_mask = d_ring;
    cp.mask_stride_words = kWords;

    auto staged = [&](const strata::grammar::SlotGrammar& live, const int32_t* window, int& chain_t,
                      std::vector<std::vector<uint32_t>>& rows) {
        // the engine's chain (generate.cpp, the window assembly): a scratch clone walks the drafts,
        // each row's mask is staged from the clone's state, a refused draft stops the walk and the
        // tail rows reuse the row before it
        rows.assign((size_t) kT, {});
        rows[0] = live.mask;
        chain_t = kT;
        std::shared_ptr<strata::grammar::grammar> scratch(strata::grammar::grammar_clone(*live.g),
                                                          [](strata::grammar::grammar* gp) { strata::grammar::grammar_free(gp); });
        for (int t = 1; t < kT; ++t) {
            try {
                strata::grammar::grammar_accept(*scratch, window[t], entry->vocab->token_to_piece(window[t]));
            } catch (const std::exception&) {
                chain_t = t;
                break;
            }
            rows[(size_t) t] = entry->cache->mask_for(strata::grammar::grammar_stacks(scratch.get()),
                                                      strata::grammar::grammar_get_rules(scratch.get()));
        }
        for (int t = chain_t; t < kT; ++t) rows[(size_t) t] = rows[(size_t) chain_t - 1];
        for (int t = 0; t < kT; ++t)
            check(cudaMemcpy(d_ring + (size_t) t * kWords, rows[(size_t) t].data(),
                             (size_t) kWords * sizeof(uint32_t), cudaMemcpyHostToDevice), "upload chain row");
    };

    auto run_window = [&](const int32_t* window, const std::vector<std::vector<uint32_t>>& rows,
                          int32_t* outv) {
        (void) window;
        strata::kernels::sample_tokens(d_l, kT, kVocab, nullptr, 0, cp, d_out4, nullptr);
        check(cudaGetLastError(), "sample chain window");   // (nullptr stream: the call syncs inside)
        check(cudaMemcpy(outv, d_out4, kT * sizeof(int32_t), cudaMemcpyDeviceToHost), "read chain picks");
        int bad = 0;
        for (int t = 0; t < kT; ++t) {
            const int y = outv[t];
            if (y < 0 || y >= kVocab) { std::printf("chain: row %d pick %d outside the vocabulary\n", t, y); ++bad; continue; }
            // the rows the engine would EMIT are 0..a (set by the caller); the pick itself must sit in
            // the mask staged for its row regardless - a row sampled under a mask that bans it is a
            // wire bug even when the row is discarded
            if (rows[(size_t) t].empty()) continue;
            if (rows[(size_t) t][(size_t) (y >> 5)] & (1u << (y & 31))) {
                std::printf("chain: row %d sampled the BANNED id %d\n", t, y);
                ++bad;
            }
        }
        return bad;
    };

    int chain_failures = 0;
    {   // all drafts legal: the walk a, b, a rides the whole window
        strata::grammar::SlotGrammar live(entry);
        const int32_t window[kT] = {0, 0, 1, 0};   // row 0's input, then drafts a, b, a
        int chain_t = kT;
        std::vector<std::vector<uint32_t>> rows;
        staged(live, window, chain_t, rows);
        if (chain_t != kT) {
            std::printf("chain: a legal draft chain truncated at row %d\n", chain_t);
            ++chain_failures;
        } else {
            int32_t outv[kT] = {};
            chain_failures += run_window(window, rows, outv);
            int a = 0;
            while (a < kT - 1 && window[a + 1] == outv[a]) ++a;
            if (a != kT - 1) {
                std::printf("chain: the legal window accepted only %d of %d drafts\n", a, kT - 1);
                ++chain_failures;
            }
            std::string seq;
            for (int i = 0; i <= a; ++i) {
                std::string aerr;
                if (!live.advance(outv[i], aerr)) {
                    std::printf("chain: the emitted walk died: %s\n", aerr.c_str());
                    ++chain_failures;
                    break;
                }
                seq += (char) ('a' + outv[i]);
            }
            if (seq.size() >= 2 && seq[0] == seq[1]) {
                std::printf("chain: the emitted walk \"%s\" repeats a letter inside a pair\n", seq.c_str());
                ++chain_failures;
            }
        }
    }
    {   // a draft the grammar refuses: the chain truncates there, the acceptance breaks at or before
        // its row, and the tail rows reuse the last good mask
        strata::grammar::SlotGrammar live(entry);
        const int32_t window[kT] = {0, 0, 0, 1};   // row 0's input, drafts a, a (refused after a), b
        int chain_t = kT;
        std::vector<std::vector<uint32_t>> rows;
        staged(live, window, chain_t, rows);
        if (chain_t != 2) {
            std::printf("chain: the refused draft did not end the chain at row 2 (at %d)\n", chain_t);
            ++chain_failures;
        }
        int32_t outv[kT] = {};
        chain_failures += run_window(window, rows, outv);
        int a = 0;
        while (a < kT - 1 && window[a + 1] == outv[a]) ++a;
        if (a > chain_t - 1) {
            std::printf("chain: acceptance reached %d past the refused draft's row %d\n", a, chain_t - 1);
            ++chain_failures;
        }
        if (a < 1 || outv[0] != 0) {
            std::printf("chain: the legal first draft was not taken (a=%d, outv[0]=%d)\n", a, outv[0]);
            ++chain_failures;
        }
        for (int t = 2; t < kT; ++t)
            if (rows[(size_t) t] != rows[(size_t) chain_t - 1]) {
                std::printf("chain: tail row %d did not reuse the last good mask\n", t);
                ++chain_failures;
            }
        std::string aerr;
        for (int i = 0; i <= a; ++i)
            if (!live.advance(outv[i], aerr)) {
                std::printf("chain: the emitted walk died: %s\n", aerr.c_str());
                ++chain_failures;
                break;
            }
    }

    cudaFree(d_ring);
    cudaFree(d_out4);
    if (chain_failures) {
        std::printf("grammar chain: %d failures\n", chain_failures);
        return 1;
    }
    std::printf("grammar chain: legal and refused-draft windows behave\n");
    return failures ? 1 : 0;
}
