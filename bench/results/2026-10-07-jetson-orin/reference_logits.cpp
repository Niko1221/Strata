// CPU llama.cpp reference for the exact IDs supplied to Strata --tokens.
// Writes Strata's dump format: int32 vocabulary, int32 rows, row-major FP32.
// Use only after freeing enough shared RAM for model validation.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: reference-logits SHARD1 OUTPUT TOKEN_ID...\n");
        return 2;
    }
    std::vector<llama_token> ids;
    for (int i = 3; i < argc; ++i) {
        char * end = nullptr;
        const long value = std::strtol(argv[i], &end, 10);
        if (end == argv[i] || *end || value < 0 || value > INT32_MAX) return 2;
        ids.push_back(static_cast<llama_token>(value));
    }
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) return 1;
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for (auto id : ids) if (id >= vocab) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max<uint32_t>(256, ids.size() + 64);
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) return 1;
    std::unique_ptr<std::FILE, decltype(&std::fclose)> out(
        std::fopen(argv[2], "wb"), std::fclose);
    if (!out) return 1;
    const int32_t header[] = {vocab, static_cast<int32_t>(ids.size())};
    if (std::fwrite(header, sizeof header, 1, out.get()) != 1) return 1;
    for (auto id : ids) {
        auto batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx.get(), batch)) return 1;
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) return 1;
        for (int32_t i = 0; i < vocab; ++i) if (!std::isfinite(logits[i])) return 1;
        if (std::fwrite(logits, sizeof(float), vocab, out.get()) != size_t(vocab)) return 1;
        std::printf("input=%d argmax=%lld\n", id,
                    static_cast<long long>(std::max_element(logits, logits + vocab) - logits));
    }
    return std::fflush(out.get()) == 0 ? 0 : 1;
}
