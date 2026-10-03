// Fixed synthetic target logits; the production CUDA selector, masks, history
// construction and equality/commit policy. This is not a model-quality test.
#include "strata/core/grammar.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/program/speculative_window.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

using namespace strata::grammar;
using namespace strata::kernels;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
constexpr int nv = 8192, words = nv / 32, max_rows = 8, hlen = 5, eos = 256;

struct Device {
    float *logits = nullptr, *scratch = nullptr;
    int32_t *masks = nullptr, *history = nullptr, *status = nullptr, *out = nullptr;
    cudaStream_t stream{};
    Device() {
        check(cudaMalloc(&logits, max_rows * nv * sizeof(float)));
        check(cudaMalloc(&scratch, max_rows * nv * sizeof(float)));
        check(cudaMalloc(&masks, max_rows * words * sizeof(int32_t)));
        check(cudaMalloc(&history, max_rows * hlen * sizeof(int32_t)));
        check(cudaMalloc(&status, max_rows * sizeof(int32_t)));
        check(cudaMalloc(&out, max_rows * sizeof(int32_t)));
        check(cudaStreamCreate(&stream));
    }
    ~Device() {
        for (void* p : {(void*)logits, (void*)scratch, (void*)masks, (void*)history, (void*)status, (void*)out}) cudaFree(p);
        cudaStreamDestroy(stream);
    }
};
struct Row { std::vector<int32_t> mask, history; uint64_t counter; int selected; };
struct Result { std::vector<int32_t> tokens; std::vector<Row> rows; };

static Result run(Device& dev, std::shared_ptr<const Compiled> compiled, SamplerParams params,
                  int width, int cap, int variant, int corrupt, const Result* reference, bool capture) {
    Matcher matcher(compiled);
    // A real prompt is intentionally absent: arbitrary fixed prompt IDs test
    // both prompt-offset counters and penalties across the feedback boundary.
    const std::vector<int32_t> prompt{97,98,257,97,17};
    std::vector<int32_t> consumed(prompt.begin(), prompt.end() - 1);
    int32_t pending = prompt.back();
    Result result;
    while ((int)result.tokens.size() < cap) {
        const size_t offset = result.tokens.size();
        int T = offset ? std::min(width, cap - (int)offset) : 1;
        std::vector<int32_t> inputs(T, 97);
        inputs[0] = pending;
        for (int i = 1; i < T; ++i) {
            // An exact reference proposal, then an illegal, legal-mismatch or
            // end-token proposal at every possible position. All are synthetic.
            if (reference && offset + i - 1 < reference->tokens.size()) inputs[i] = reference->tokens[offset + i - 1];
            if (i - 1 == corrupt && variant == 1) inputs[i] = 'z';
            if (i - 1 == corrupt && variant == 2) inputs[i] = inputs[i] == 'a' ? 'b' : 'a';
            if (i - 1 == corrupt && variant == 3) inputs[i] = eos;
        }
        PrefixMasks masks;
        matcher.prefix_masks(inputs.data() + 1, T - 1, masks);
        T = masks.rows;
        std::vector<int32_t> history(T * hlen), picked(T), status(T);
        penalty_rows(consumed.data(), (int64_t)consumed.size(), inputs.data(), T, hlen, history.data());
        std::vector<float> logits(T * nv, 30.0f); // high illegal scores must not survive
        for (int row = 0; row < T; ++row) {
            const int at = (int)offset + row;
            logits[row * nv + 'a'] = 2.9f + (at % 3) * 0.13f;
            logits[row * nv + 'b'] = 3.0f - (at % 2) * 0.21f;
            logits[row * nv + 257] = 2.5f + (at % 5) * 0.19f;
            logits[row * nv + eos] = at >= 8 ? 20.0f : -4.0f;
        }
        check(cudaMemcpy(dev.logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dev.masks, masks.bits.data(), masks.bits.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dev.history, history.data(), history.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        params.counter = consumed.size();
        TokenMask mask{dev.masks, dev.scratch, dev.status};
        auto select = [&] { sample_tokens(dev.logits, T, nv, dev.history, hlen, params, dev.out, dev.stream, &mask); };
        if (capture) {
            cudaGraph_t graph; cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(dev.stream, cudaStreamCaptureModeGlobal)); select();
            check(cudaStreamEndCapture(dev.stream, &graph));
            check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            check(cudaGraphLaunch(exec, dev.stream)); check(cudaStreamSynchronize(dev.stream));
            check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
        } else { select(); check(cudaStreamSynchronize(dev.stream)); }
        check(cudaMemcpy(picked.data(), dev.out, T * sizeof(int32_t), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(status.data(), dev.status, T * sizeof(int32_t), cudaMemcpyDeviceToHost));
        for (int s : status) REQUIRE(s == 0);
        const auto kept = strata::program::retained_window(inputs.data(), picked.data(), T, cap - offset, {eos});
        for (int i = 0; i < kept.count; ++i) {
            Row row{{masks.bits.begin() + i * words, masks.bits.begin() + (i + 1) * words},
                    {history.begin() + i * hlen, history.begin() + (i + 1) * hlen}, params.counter + i, picked[i]};
            if (reference) {
                REQUIRE(offset + i < reference->rows.size());
                const auto& ref = reference->rows[offset + i];
                REQUIRE(row.mask == ref.mask && row.history == ref.history && row.counter == ref.counter);
                REQUIRE(row.selected == ref.selected);
            }
            REQUIRE(matcher.accept(picked[i]));
            result.rows.push_back(std::move(row)); result.tokens.push_back(picked[i]);
            consumed.push_back(inputs[i]);
        }
        pending = picked[kept.count - 1];
        REQUIRE(consumed.size() == prompt.size() + result.tokens.size() - 1);
        std::vector<int32_t> expected = prompt;
        expected.insert(expected.end(), result.tokens.begin(), result.tokens.end() - 1);
        REQUIRE(consumed == expected && matcher.tokens() == result.tokens);
        if (kept.eos) { REQUIRE(matcher.terminated() && matcher.complete()); break; }
    }
    if (reference) REQUIRE(result.tokens == reference->tokens);
    return result;
}

static void scoped_window(Device& dev) {
    std::vector<std::string> bytes(nv);
    for (int i = 0; i < 256; ++i) bytes[i] = std::string(1, (char) i);
    Compiler compiler(Vocabulary::from_bytes(std::move(bytes), {eos}, {257, 258, 259}));
    Matcher matcher(compiler.compile("root ::= \"OK\""), 2000000, {true, true});
    const int32_t expected[] = {'x', 257, 258, 'y', 259, 'O', 'K', eos};
    const char* channels[] = {"reasoning", "control", "tool", "tool", "tool", "answer", "answer", "control"};
    PrefixMasks masks;
    matcher.prefix_masks(expected, 7, masks);
    REQUIRE(masks.rows == 8 && matcher.tokens().empty());
    std::vector<float> logits(8 * nv, -10.0f);
    for (int row = 0; row < 8; ++row) {
        logits[row * nv + expected[row]] = 20;
        logits[row * nv + 260] = 100; // unavailable special in every phase
        if (row >= 5) logits[row * nv + 'z'] = 100; // not an answer continuation
    }
    check(cudaMemcpy(dev.logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dev.masks, masks.bits.data(), masks.bits.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    SamplerParams params; params.greedy = true; params.temperature = 0;
    TokenMask mask{dev.masks, dev.scratch, dev.status};
    sample_tokens(dev.logits, 8, nv, nullptr, 0, params, dev.out, dev.stream, &mask);
    check(cudaStreamSynchronize(dev.stream));
    int32_t picked[8], status[8];
    check(cudaMemcpy(picked, dev.out, sizeof picked, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(status, dev.status, sizeof status, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) {
        REQUIRE(status[i] == 0 && picked[i] == expected[i]);
        REQUIRE(matcher.accept(picked[i]) && std::string(matcher.channel()) == channels[i]);
    }
    REQUIRE(matcher.terminated());
    std::cout << "PASS: CUDA selection across reasoning/tool/answer phases in one 8-row speculative window. "
                 "Synthetic logits; real masks and sampler.\n";
}

int main() {
    try {
        std::vector<std::string> bytes(nv);
        bytes['a'] = "a"; bytes['b'] = "b"; bytes['z'] = "z"; bytes[257] = "ab";
        Compiler compiler(Vocabulary::from_bytes(bytes, {eos}));
        auto compiled = compiler.compile("root ::= [ab]+");
        Device dev;
        scoped_window(dev);
        int cases = 0;
        for (bool capture : {false, true}) for (int chain = 0; chain < 3; ++chain)
            for (int cap : {1,2,6,12}) for (int seed : {4,49}) {
                SamplerParams p;
                p.greedy = chain == 0; p.temperature = p.greedy ? 0.0f : 0.8f;
                p.top_k = chain == 1 ? 2 : 20; p.top_p = 0.92f; p.min_p = 0.04f; p.seed = seed;
                p.penalty_last_n = chain == 2 ? hlen : 0;
                p.penalty_repeat = 1.1f; p.penalty_freq = 0.12f; p.penalty_present = 0.07f;
                const auto reference = run(dev, compiled, p, 1, cap, 0, -1, nullptr, capture);
                for (int width : {2,4,8}) for (int variant = 0; variant < 4; ++variant)
                    for (int at = 0; at < (variant ? width - 1 : 1); ++at) {
                        run(dev, compiled, p, width, cap, variant, at, &reference, capture);
                        ++cases;
                    }
            }
        std::cout << "PASS: " << cases << " exact fixed-logit/counter speculative decodes; captured and uncaptured, "
                     "greedy/sampled/penalized, 2/4/8 rows, every rejection position. Synthetic target logits/proposals.\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
