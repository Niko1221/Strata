// Independent llama.cpp oracle for the optional CANN text backend.
// Build against the pinned llama.cpp checkout, never link it into the backend:
// g++ -O2 -std=c++17 qwen38_oracle.cpp -I<source>/include
//     -I<source>/ggml/include -L<build>/bin -lllama -lggml -lggml-base
//     -Wl,-rpath,<build>/bin -o qwen38_oracle
// Usage: qwen38_oracle MODEL OUTPUT_PREFIX TOKENS N_GENERATE
// TOKENS is a comma-separated list. Dumps are float32 [position,layer,4,2560]
// and [generation_step,248320]. The prompt is evaluated one token at a time
// so every layer is available at every position.
// STRATA_ORACLE_FLOAT_MATMUL=1 replaces matmul outputs through the scheduler
// callback with ggml's own dequantized weights and float32 activations. This
// removes CPU Q8 activation rounding when comparing an FP16 device backend.
// Compile with -fopenmp; set OMP_NUM_THREADS to bound this diagnostic's CPU use.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct Capture {
    std::map<int, std::vector<float>> layers;
    bool float_matmul = false;
};

static void float_matmul(ggml_tensor * tensor) {
    const auto * weight = tensor->src[0];
    const auto * input = tensor->src[1];
    const auto * ids = tensor->op == GGML_OP_MUL_MAT_ID ? tensor->src[2] : nullptr;
    if (input->type != GGML_TYPE_F32 || tensor->type != GGML_TYPE_F32 ||
        input->ne[2] != 1 || input->ne[3] != 1 ||
        (!ids && (input->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1))) {
        std::fprintf(stderr, "unsupported diagnostic matmul shape: %s\n", ggml_get_name(tensor));
        std::exit(4);
    }
    const auto * traits = ggml_get_type_traits(weight->type);
    const int64_t rows = weight->ne[1];
    const int64_t width = weight->ne[0];
    const int64_t experts = ids ? ids->ne[0] : 1;
    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < rows * experts; ++task) {
        const int64_t row = task % rows;
        const int64_t selected = task / rows;
        const int32_t expert = ids ?
            *reinterpret_cast<const int32_t *>(static_cast<const char *>(ids->data) + selected * ids->nb[0]) : 0;
        const char * source = static_cast<const char *>(weight->data) + row * weight->nb[1] +
                              expert * weight->nb[2];
        thread_local std::vector<float> decoded;
        decoded.resize(width);
        if (weight->type == GGML_TYPE_F32) {
            std::memcpy(decoded.data(), source, width * sizeof(float));
        } else {
            if (!traits->to_float) {
                std::abort();
            }
            traits->to_float(source, decoded.data(), width);
        }
        const auto * values = reinterpret_cast<const float *>(
            static_cast<const char *>(input->data) + (selected % input->ne[1]) * input->nb[1]);
        double sum = 0;
        for (int64_t k = 0; k < width; ++k) {
            sum += static_cast<double>(decoded[k]) * values[k];
        }
        auto * result = reinterpret_cast<float *>(
            static_cast<char *>(tensor->data) + row * tensor->nb[0] + selected * tensor->nb[1]);
        *result = static_cast<float>(sum);
    }
}

static bool capture(ggml_tensor * tensor, bool ask, void * opaque) {
    auto * state = static_cast<Capture *>(opaque);
    if (state->float_matmul &&
        (tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) &&
        std::strstr(ggml_get_name(tensor->src[0]), ".weight")) {
        if (!ask) {
            float_matmul(tensor);
        }
        return true;
    }
    const char * name = ggml_get_name(tensor);
    if (std::strncmp(name, "l_last-", 7) != 0) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    if (tensor->type != GGML_TYPE_F32 || ggml_nelements(tensor) != 4 * 2560) {
        std::fprintf(stderr, "unexpected residual geometry for %s\n", name);
        std::exit(3);
    }
    auto & values = state->layers[std::atoi(name + 7)];
    values.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL OUTPUT_PREFIX TOKENS N_GENERATE\n", argv[0]);
        return 2;
    }
    std::vector<llama_token> prompt;
    std::string input(argv[3]);
    for (size_t start = 0; start < input.size();) {
        size_t end = input.find(',', start);
        prompt.push_back(std::stoi(input.substr(start, end - start)));
        start = end == std::string::npos ? input.size() : end + 1;
    }
    const int count = std::stoi(argv[4]);
    if (prompt.empty() || count < 1) {
        return 2;
    }
    llama_backend_init();
    const bool precise = std::getenv("STRATA_ORACLE_FLOAT_MATMUL") != nullptr;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    if (precise) {
        mp.use_extra_bufts = false;
    }
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model || llama_model_n_embd(model) != 2560 || llama_model_n_layer(model) != 48) {
        std::fprintf(stderr, "expected the original 48-layer Flash-Next GGUF\n");
        return 1;
    }
    Capture captured;
    captured.float_matmul = precise;
    std::printf("oracle float matmul: %s\n", precise ? "enabled" : "disabled");
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 12;
    cp.op_offload = false;
    cp.cb_eval = capture;
    cp.cb_eval_user_data = &captured;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 1;
    }
    const auto * vocab = llama_model_get_vocab(model);
    const int vocabulary = llama_vocab_n_tokens(vocab);
    const std::string prefix(argv[2]);
    FILE * layers = std::fopen((prefix + ".layers.f32").c_str(), "wb");
    FILE * logits = std::fopen((prefix + ".logits.f32").c_str(), "wb");
    if (!layers || !logits) {
        std::perror("open dump");
        return 1;
    }
    auto step = [&](llama_token token) {
        if (llama_decode(ctx, llama_batch_get_one(&token, 1))) {
            std::fprintf(stderr, "llama_decode failed\n");
            std::exit(1);
        }
        for (int layer = 0; layer < 48; ++layer) {
            const auto & values = captured.layers.at(layer);
            std::fwrite(values.data(), sizeof(float), values.size(), layers);
            std::printf("layer %d: %.9g %.9g %.9g %.9g %.9g %.9g\n", layer,
                        values[0], values[1], values[2], values[3], values[4], values[5]);
        }
        captured.layers.clear();
        std::fflush(layers);
        std::fflush(stdout);
    };
    for (auto token : prompt) {
        step(token);
    }
    for (int generation = 0; generation < count; ++generation) {
        const float * values = llama_get_logits_ith(ctx, -1);
        const int token = std::max_element(values, values + vocabulary) - values;
        std::fwrite(values, sizeof(float), vocabulary, logits);
        std::fflush(logits);
        std::printf("generated %d: %d\n", generation, token);
        std::fflush(stdout);
        if (generation + 1 == count || llama_vocab_is_eog(vocab, token)) {
            break;
        }
        step(token);
    }
    std::fclose(layers);
    std::fclose(logits);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
