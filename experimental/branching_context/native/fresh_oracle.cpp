// Independent oracle: each forward allocates a new context and consumes all inputs.
#include "llama.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void send(T value) {
    std::cout.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected one model path");
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Model load failed");
        if (llama_model_is_hybrid(model) || llama_model_is_recurrent(model) || llama_model_has_encoder(model)) {
            throw std::runtime_error("Oracle profile supports a causal, non-hybrid decoder only");
        }
        auto * vocab = llama_model_get_vocab(model);
        const int32_t nv = llama_vocab_n_tokens(vocab);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.rfind("T ", 0) == 0) {
                auto text = line.substr(2);
                int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, false);
                std::vector<llama_token> ids(n);
                n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), true, false);
                if (n <= 0) throw std::runtime_error("Tokenization failed");
                send<int32_t>(n);
                std::cout.write(reinterpret_cast<const char *>(ids.data()), n*sizeof(llama_token));
                std::cout.flush();
                continue;
            }
            std::istringstream request(line);
            char command;
            int chunk, count;
            request >> command >> chunk >> count;
            if (command != 'F' || count < 1 || count > 2048 || chunk < 1) throw std::runtime_error("Invalid forward request");
            std::vector<llama_token> tokens(count);
            for (auto & token : tokens) {
                if (!(request >> token) || token < 0 || token >= nv) throw std::runtime_error("Invalid token");
            }
            auto cp = llama_context_default_params();
            cp.n_ctx = 2048;
            cp.n_batch = 2048;
            cp.n_ubatch = 2048;
            cp.n_seq_max = 1;
            cp.n_threads = cp.n_threads_batch = 2;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = cp.type_v = GGML_TYPE_F32;
            cp.no_perf = false;
            auto * ctx = llama_init_from_model(model, cp);
            if (!ctx) throw std::runtime_error("Fresh context allocation failed");
            if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != -1) throw std::runtime_error("Oracle context was not empty");
            uint64_t processed = 0;
            for (int start = 0; start < count; start += chunk) {
                const int n = std::min(chunk, count-start);
                auto batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (int j=0; j<n; ++j) {
                    batch.token[j] = tokens[start+j];
                    batch.pos[j] = start+j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j] = (j == n-1);
                }
                int rc = llama_decode(ctx, batch);
                llama_batch_free(batch);
                if (rc != 0) throw std::runtime_error("Oracle decode failed");
                processed += n;
            }
            const int32_t position = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            auto * logits = llama_get_logits_ith(ctx, -1);
            if (!logits || position != count-1) throw std::runtime_error("Oracle frontier mismatch");
            send<int32_t>(nv);
            send<uint64_t>(processed);
            send<int32_t>(position);
            std::cout.write(reinterpret_cast<const char *>(logits), nv*sizeof(float));
            std::cout.flush();
            llama_free(ctx);
        }
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ORACLE ERROR: " << error.what() << '\n';
        return 2;
    }
}
