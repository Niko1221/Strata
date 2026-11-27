#include "llama.h"
#include "ggml-backend.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Capture v1.1.0 logits after the prefix and each fixed continuation token.
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    auto device = ggml_backend_dev_by_name("CUDA0");
    if (!device) return 2;
    llama_model_tensor_buft_override overrides[] = {
        {"^token_embd[.]weight$", ggml_backend_dev_buffer_type(device)}, {nullptr, nullptr}};
    model_params.tensor_buft_overrides = overrides;
    model_params.n_gpu_layers = 99;
    auto model = llama_model_load_from_file(argv[1], model_params);
    if (!model) return 3;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (vocab != 120832) return 4;
    std::ifstream list(argv[2]);
    if (!list) return 5;
    int prefix, rows;
    std::string token_file, output_file;
    while (list >> prefix >> rows >> token_file >> output_file) {
        std::ifstream in(token_file);
        std::vector<llama_token> ids;
        std::string field;
        while (std::getline(in, field, ',')) ids.push_back(std::stoi(field));
        if (ids.size() != size_t(prefix + rows)) return 5;

        auto context_params = llama_context_default_params();
        context_params.n_ctx = prefix == 128 ? 512 : 8192;
        context_params.n_batch = context_params.n_ubatch = prefix == 128 ? 128 : 4096;
        context_params.n_threads = context_params.n_threads_batch = 16;
        context_params.type_k = context_params.type_v = GGML_TYPE_F16;
        auto ctx = llama_init_from_model(model, context_params);
        if (!ctx) return 6;
        auto* batch = llama_batch_ext_init(ctx);
        if (!batch) return 7;
        std::ofstream out(output_file, std::ios::binary);
        for (int offset = 0; offset < prefix + rows - 1;) {
            const int take = offset ? 1 : prefix;
            llama_batch_ext_clear(batch);
            for (int j = 0; j < take; ++j) {
                const int idx = llama_batch_ext_add_token(batch, 0, ids[offset + j]);
                const llama_pos pos = offset + j;
                if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos) ||
                    !llama_batch_ext_set_output_logits(batch, idx, j + 1 == take)) return 7;
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return 7;
            const float* logits = llama_get_logits_ith(ctx, take - 1);
            if (!logits) return 8;
            out.write(reinterpret_cast<const char*>(logits), sizeof(float) * vocab);
            offset += take;
        }
        out.close();
        if (!out) return 9;
        std::cerr << "CAPTURED " << output_file << "\n";
        llama_batch_ext_free(batch);
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return list.eof() ? 0 : 10;
}
