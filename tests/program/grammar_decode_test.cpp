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

    cudaFree(d_l);
    cudaFree(d_mask);
    cudaFree(d_o);
    if (failures) {
        std::printf("grammar decode: %d failures\n", failures);
        return 1;
    }
    std::printf("grammar decode: %d tokens over %d rounds, all legal\n", tokens, rounds_done);
    return 0;
}
