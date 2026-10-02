// src/qwen35/qwen35_main.cpp - a minimal CLI to exercise the Qwen35MoE trunk on a real Ornith GGUF.
//
//   strata-qwen35 --model Ornith-....gguf --tokens "1,2,3" --max-new 16
//
// It prints the generated token IDs and the per-token time.  This is the correctness/debug entry point; the
// server wiring and the tokenizer/text path come next.  No GPU: ggml-cpu runs the quantized matvecs.
#include "strata/qwen35/qwen35.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string path;
    std::vector<int64_t> tokens;
    int64_t max_new = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") path = next();
        else if (a == "--tokens") {
            std::string s = next();
            for (size_t b = 0; b < s.size();) {
                size_t e = s.find(',', b);
                if (e == std::string::npos) e = s.size();
                tokens.push_back(std::strtoll(s.substr(b, e - b).c_str(), nullptr, 10));
                b = e + 1;
            }
        } else if (a == "--max-new") max_new = std::strtoll(next(), nullptr, 10);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (path.empty() || tokens.empty()) {
        std::fprintf(stderr, "usage: strata-qwen35 --model M.gguf --tokens \"1,2,3\" [--max-new N]\n");
        return 2;
    }

    strata::qwen35::qwen35_enable_ggml();
    strata::core::Qwen35Geometry g;
    strata::qwen35::TrunkWeights w;
    std::string err;
    if (!strata::qwen35::load_trunk(path, g, w, err)) {
        std::fprintf(stderr, "strata-qwen35: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata-qwen35: %lld layers, %lld wide, %lld experts top-%lld, vocab %lld\n",
                 (long long) g.n_layers, (long long) g.n_embd, (long long) g.n_expert,
                 (long long) g.n_expert_used, (long long) g.n_vocab);

    strata::qwen35::TrunkState st;
    st.reset(g);
    std::vector<float> logits((size_t) g.n_vocab);

    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t tok : tokens) strata::qwen35::trunk_forward(g, w, st, tok, logits.data());
    const auto t1 = std::chrono::steady_clock::now();
    const double pms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "prefill %zu tokens in %.0f ms (%.1f tok/s)\n", tokens.size(), pms,
                 pms > 0 ? 1000.0 * (double) tokens.size() / pms : 0.0);

    std::printf("prompt:");
    for (int64_t t : tokens) std::printf(" %lld", (long long) t);
    std::printf("\ngen:");
    for (int64_t n = 0; n < max_new; ++n) {
        int64_t best = 0;
        for (int64_t v = 1; v < g.n_vocab; ++v) if (logits[(size_t) v] > logits[(size_t) best]) best = v;
        std::printf(" %lld", (long long) best);
        std::fflush(stdout);
        const auto a = std::chrono::steady_clock::now();
        strata::qwen35::trunk_forward(g, w, st, best, logits.data());
        const auto b = std::chrono::steady_clock::now();
        std::fprintf(stderr, "token %lld: %.0f ms\n", (long long) n,
                     std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::printf("\n");
    return 0;
}
