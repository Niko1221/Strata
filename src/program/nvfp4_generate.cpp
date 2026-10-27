// src/program/nvfp4_generate.cpp - RDNA4 (gfx1201) NVFP4 CLI and --serve driver.
#include "strata/program/nvfp4_generate.hpp"

#include "strata/artifact/qwen_tokenizer.hpp"
#include "strata/core/nvfp4_engine.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/spec/nvfp4_ngram_drafter.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace strata::program {

int run_nvfp4(int argc, char** argv) {
    std::string model_dir;
    if (const char* env_md = std::getenv("STRATA_MODEL_DIR")) {
        if (env_md[0] != '\0') model_dir = env_md;
    }
    std::string cache_dir; // Empty by default: uses embedded gfx1201 .hsaco kernels
    std::string expert_profile_path;
    if (const char* env_ep = std::getenv("STRATA_EXPERT_PROFILE")) {
        if (env_ep[0] != '\0') expert_profile_path = env_ep;
    }
    std::string prompt = "The capital of France is";
    std::string tokens_csv;
    std::string dump_logits_path;
    int max_new_tokens = 16;
    int vram_expert_layers = 14;
    int vram_reserve_mib = 1536; // Default 1.5 GiB reserved for display / compositor + elastic KV headroom on R9700
    if (const char* env_vr = std::getenv("STRATA_NVFP4_RESERVE_VRAM_MB")) {
        vram_reserve_mib = std::max(256, std::atoi(env_vr));
    }
    int max_seq = 1048576; // 1M context window by default (grows elastically from 6,553 tokens via VmmRange + lazy host slabs)
    int kv_resident_tokens = 32768; // 32K tokens resident in VRAM (28,672 prefix + 4,096 sliding window ring = ~240 MB max)
    bool kv_grow = true;
    std::string rope_scaling_str = "auto";
    float rope_scale_arg = 0.0f;
    int yarn_orig_ctx = 262144;
    bool chat_mode = false;
    bool think_mode = false;
    bool serve_mode = false;
    bool spec_ngram = true;
    if (const char* env_sp = std::getenv("STRATA_NVFP4_SPEC")) {
        spec_ngram = (std::atoi(env_sp) != 0);
    }
    strata::core::Nvfp4SamplingParams cli_sp{};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gfx1201-nvfp4" || a == "--nvfp4") continue;
        if (a == "--serve") { serve_mode = true; continue; }
        if (a == "--think") { think_mode = true; chat_mode = true; continue; }
        if (a == "--no-think") { think_mode = false; continue; }
        if (a == "--spec-ngram") { spec_ngram = true; continue; }
        if (a == "--no-spec-ngram") { spec_ngram = false; continue; }
        if (a == "--kv-grow") { kv_grow = true; continue; }
        if (a == "--no-kv-grow") { kv_grow = false; continue; }
        if ((a == "--model" || a == "--pack" || a == "--native" || a == "--safetensors" || a == "-m") && i + 1 < argc) {
            model_dir = argv[++i];
            // If user passed a shard file path inside the directory, strip to parent directory
            struct stat st {};
            if (::stat(model_dir.c_str(), &st) == 0 && !S_ISDIR(st.st_mode)) {
                auto slash = model_dir.find_last_of('/');
                if (slash != std::string::npos) model_dir = model_dir.substr(0, slash);
            }
        } else if ((a == "--prompt" || a == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((a == "--prompt-file" || a == "-f") && i + 1 < argc) {
            std::ifstream pf(argv[++i]);
            if (pf) {
                prompt.assign((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
            }
        } else if (a == "--tokens" && i + 1 < argc) {
            tokens_csv = argv[++i];
        } else if ((a == "--n-predict" || a == "-n" || a == "--max-new" || a == "--tokens-out") && i + 1 < argc) {
            max_new_tokens = std::atoi(argv[++i]);
        } else if (a == "--vram-expert-layers" && i + 1 < argc) {
            vram_expert_layers = std::atoi(argv[++i]);
        } else if ((a == "--vram-reserve-mib" || a == "--vram-reserve") && i + 1 < argc) {
            vram_reserve_mib = std::max(256, std::atoi(argv[++i]));
        } else if (a == "--cache-dir" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (a == "--expert-profile" && i + 1 < argc) {
            expert_profile_path = argv[++i];
        } else if ((a == "--max-context" || a == "-c" || a == "--ctx-size") && i + 1 < argc) {
            max_seq = std::atoi(argv[++i]);
        } else if ((a == "--kv-resident" || a == "--vram-kv-tokens") && i + 1 < argc) {
            kv_resident_tokens = std::atoi(argv[++i]);
        } else if (a == "--rope-scaling" && i + 1 < argc) {
            rope_scaling_str = argv[++i];
        } else if (a == "--rope-scale" && i + 1 < argc) {
            rope_scale_arg = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--yarn-orig-ctx" && i + 1 < argc) {
            yarn_orig_ctx = std::atoi(argv[++i]);
        } else if (a == "--dump-logits" && i + 1 < argc) {
            dump_logits_path = argv[++i];
        } else if (a == "--chat") {
            chat_mode = true;
        } else if ((a == "--temp" || a == "--temperature") && i + 1 < argc) {
            cli_sp.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--top-p" && i + 1 < argc) {
            cli_sp.top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--top-k" && i + 1 < argc) {
            cli_sp.top_k = std::atoi(argv[++i]);
        } else if (a == "--presence-penalty" && i + 1 < argc) {
            cli_sp.presence_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--frequency-penalty" && i + 1 < argc) {
            cli_sp.frequency_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--repeat-penalty" && i + 1 < argc) {
            cli_sp.repeat_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--seed" && i + 1 < argc) {
            cli_sp.seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        }
    }

    if (model_dir.empty()) {
        std::fprintf(stderr, "[gfx1201-nvfp4] Error: model directory required (pass --model <dir> or set STRATA_MODEL_DIR).\n");
        if (serve_mode) {
            std::printf("ERR model directory required\n");
            std::fflush(stdout);
        }
        return 2;
    }

    // Startup GPU architecture check: embedded NVFP4 kernels are compiled specifically for RDNA4 (gfx1200 / gfx1201)
    {
        int dev_count = 0;
        hipError_t herr = hipGetDeviceCount(&dev_count);
        if (herr != hipSuccess || dev_count <= 0) {
            std::fprintf(stderr, "[gfx1201-nvfp4] Error: No HIP GPU devices detected (%s).\n",
                         hipGetErrorString(herr));
            if (serve_mode) {
                std::printf("ERR No HIP GPU devices detected\n");
                std::fflush(stdout);
            }
            return 1;
        }
        hipDeviceProp_t prop{};
        herr = hipGetDeviceProperties(&prop, 0);
        if (herr != hipSuccess) {
            std::fprintf(stderr, "[gfx1201-nvfp4] Error: hipGetDeviceProperties failed (%s).\n",
                         hipGetErrorString(herr));
            if (serve_mode) {
                std::printf("ERR hipGetDeviceProperties failed\n");
                std::fflush(stdout);
            }
            return 1;
        }
        std::string arch(prop.gcnArchName);
        if (arch.rfind("gfx120", 0) != 0) {
            std::fprintf(stderr,
                         "[gfx1201-nvfp4] Error: Unsupported GPU architecture '%s' on device '%s'. "
                         "The gfx1201-nvfp4 engine requires an RDNA4 (gfx1200 / gfx1201) GPU.\n",
                         arch.c_str(), prop.name);
            if (serve_mode) {
                std::printf("ERR Unsupported GPU architecture %s (requires gfx1200/gfx1201)\n", arch.c_str());
                std::fflush(stdout);
            }
            return 1;
        }
    }

    strata::artifact::QwenTokenizer tokenizer;
    const bool tok_ok = tokenizer.load(model_dir);
    if (!tok_ok && !serve_mode && tokens_csv.empty()) {
        std::fprintf(stderr, "[gfx1201-nvfp4] Failed to load tokenizer from %s (pass --tokens or include vocab.json + merges.txt)\n",
                     model_dir.c_str());
        return 1;
    }

    std::vector<int32_t> prompt_ids;
    if (!serve_mode) {
        if (!tokens_csv.empty()) {
            std::stringstream ss(tokens_csv);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) prompt_ids.push_back(std::stoi(item));
            }
        } else {
            std::string full_prompt = prompt;
            if (think_mode) {
                full_prompt = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
            } else if (chat_mode) {
                full_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            }
            prompt_ids = tokenizer.encode(full_prompt);
        }

        if (prompt_ids.empty()) {
            std::fprintf(stderr, "[gfx1201-nvfp4] Prompt produced 0 tokens.\n");
            return 1;
        }

        std::printf("[gfx1201-nvfp4] Prompt tokens (%zu):", prompt_ids.size());
        for (int32_t id : prompt_ids) std::printf(" %d", id);
        std::printf("\n");
    }

    strata::kernels::RopeScaling rope_cfg{};
    rope_cfg.freq_base = 1.0e7;
    rope_cfg.orig_ctx = static_cast<double>(std::max(4096, yarn_orig_ctx));
    bool rope_auto_yarn = false;
    if (rope_scaling_str == "none") {
        rope_cfg.type = strata::kernels::RopeScalingType::None;
    } else if (rope_scaling_str == "linear") {
        rope_cfg.type = strata::kernels::RopeScalingType::Linear;
        rope_cfg.factor = (rope_scale_arg > 0.0f)
                              ? static_cast<double>(rope_scale_arg)
                              : std::max(1.0, static_cast<double>(max_seq) / rope_cfg.orig_ctx);
        rope_cfg.ext_factor = 0.0;
    } else if (rope_scaling_str == "yarn") {
        rope_cfg.type = strata::kernels::RopeScalingType::YaRN;
        rope_cfg.factor = (rope_scale_arg > 0.0f)
                              ? static_cast<double>(rope_scale_arg)
                              : std::max(1.0, static_cast<double>(max_seq) / rope_cfg.orig_ctx);
        rope_cfg.ext_factor = 1.0;
    } else {
        // "auto": Dynamic YaRN — exact unscaled RoPE up to orig_ctx (262,144), and YaRN beyond 262,144 up to 1M+
        rope_auto_yarn = true;
        rope_cfg.type = strata::kernels::RopeScalingType::YaRN;
        rope_cfg.factor = (rope_scale_arg > 0.0f)
                              ? static_cast<double>(rope_scale_arg)
                              : std::max(1.0, static_cast<double>(max_seq) / rope_cfg.orig_ctx);
        rope_cfg.ext_factor = 1.0;
    }

    strata::core::Qwen4ExpGfx1201Engine engine(model_dir, cache_dir, expert_profile_path, vram_expert_layers, max_seq,
                                               kv_resident_tokens, kv_grow, rope_cfg, rope_auto_yarn, vram_reserve_mib);
    engine.init(serve_mode);

    std::vector<int32_t> history;
    int32_t next_tok = -1;
    strata::spec::Nvfp4NgramDrafter drafter(tokenizer);
    float mtp_min_prob = 0.35f;
    if (const char* emp = std::getenv("STRATA_MTP_MIN_PROB")) {
        mtp_min_prob = std::clamp(static_cast<float>(std::atof(emp)), 0.0f, 1.0f);
    }
    int32_t mtp_max_k = 2;
    if (const char* emk = std::getenv("STRATA_MTP_K")) {
        mtp_max_k = std::clamp(std::atoi(emk), 0, 3);
    }
    const bool mtp_only = (std::getenv("STRATA_MTP_ONLY") != nullptr &&
                           std::atoi(std::getenv("STRATA_MTP_ONLY")) != 0);

    if (serve_mode) {
        std::printf("INFO kv_resident=%d expert_slots=%d vram_reserve_mib=%d batch_slots=0 engine=gfx1201-nvfp4\n",
                    engine.vram_cap(), engine.total_vram_experts(), vram_reserve_mib);
        std::printf("READY %d stop\n", engine.max_seq());
        std::fflush(stdout);

        std::mutex q_mtx;
        std::condition_variable q_cv;
        std::deque<std::string> cmd_q;
        std::atomic<bool> stop_req{false};
        std::atomic<bool> stdin_eof{false};

        std::thread reader_thr([&]() {
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
                    line.pop_back();
                }
                if (line == "STOP") {
                    stop_req.store(true, std::memory_order_relaxed);
                    continue;
                }
                if (line.empty()) continue;
                const bool is_quit = (line == "QUIT");
                {
                    std::lock_guard<std::mutex> lk(q_mtx);
                    cmd_q.push_back(std::move(line));
                }
                q_cv.notify_one();
                if (is_quit) break;
            }
            stdin_eof.store(true, std::memory_order_relaxed);
            q_cv.notify_all();
        });

        while (true) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(q_mtx);
                q_cv.wait(lk, [&]() { return !cmd_q.empty() || stdin_eof.load(std::memory_order_relaxed); });
                if (cmd_q.empty()) break;
                line = std::move(cmd_q.front());
                cmd_q.pop_front();
            }
            if (line == "QUIT") break;
            if (line.rfind("VRAM", 0) == 0) {
                size_t free_v = 0, total_v = 0;
                (void) hipMemGetInfo(&free_v, &total_v);
                std::printf("VRAM reserve_mib=%d expert_slots=%d vram_free_mib=%zu\n",
                            vram_reserve_mib, engine.total_vram_experts(),
                            static_cast<size_t>(free_v / (1024ULL * 1024ULL)));
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("GEN ", 0) == 0) {
                stop_req.store(false, std::memory_order_relaxed);
                drafter.clear_rejections();

                // Format: GEN <max_new> [key=val ...] <id0,id1,...>
                std::istringstream iss(line);
                std::string cmd;
                int req_max_new = max_new_tokens;
                iss >> cmd >> req_max_new;
                std::vector<std::string> fields;
                std::string tok_field;
                while (iss >> tok_field) {
                    fields.push_back(std::move(tok_field));
                }
                if (fields.empty()) {
                    std::printf("DONE 0 0 0 0 stop 0 0 0\n");
                    std::fflush(stdout);
                    continue;
                }
                strata::core::Nvfp4SamplingParams req_sp = cli_sp;
                for (size_t fi = 0; fi + 1 < fields.size(); ++fi) {
                    const std::string& kv = fields[fi];
                    auto eq = kv.find('=');
                    if (eq == std::string::npos) continue;
                    std::string k = kv.substr(0, eq);
                    std::string v = kv.substr(eq + 1);
                    if (k == "temp" || k == "temperature") {
                        req_sp.temperature = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "top_p") {
                        req_sp.top_p = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "top_k") {
                        req_sp.top_k = std::atoi(v.c_str());
                    } else if (k == "presence_penalty") {
                        req_sp.presence_penalty = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "frequency_penalty") {
                        req_sp.frequency_penalty = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "repeat_penalty") {
                        req_sp.repeat_penalty = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "seed") {
                        req_sp.seed = static_cast<uint64_t>(std::strtoull(v.c_str(), nullptr, 10));
                    }
                }
                std::mt19937_64 req_rng(req_sp.seed);

                const std::string& ids_csv = fields.back();
                std::vector<int32_t> req_ids;
                req_ids.reserve(1024);
                {
                    size_t pos = 0;
                    while (pos < ids_csv.size()) {
                        size_t comma = ids_csv.find(',', pos);
                        if (comma == std::string::npos) comma = ids_csv.size();
                        if (comma > pos) {
                            req_ids.push_back(static_cast<int32_t>(std::strtol(ids_csv.c_str() + pos, nullptr, 10)));
                        }
                        pos = comma + 1;
                    }
                }
                const size_t N = req_ids.size();
                if (N == 0) {
                    std::printf("DONE 0 0 0 0 stop 0 0 0\n");
                    std::fflush(stdout);
                    continue;
                }

                // Check prefix match against current `history` or previous turn's `turn_ckpt_ids()`
                size_t reused = 0;
                if (!history.empty() && N > history.size()) {
                    bool match = true;
                    for (size_t k = 0; k < history.size(); ++k) {
                        if (req_ids[k] != history[k]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        reused = history.size();
                    }
                }
                if (reused == 0) {
                    const auto& ckpt_ids = engine.turn_ckpt_ids();
                    if (!ckpt_ids.empty() && N > ckpt_ids.size()) {
                        bool match = true;
                        for (size_t k = 0; k < ckpt_ids.size(); ++k) {
                            if (req_ids[k] != ckpt_ids[k]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            engine.restore_turn_ckpt(history, next_tok);
                            reused = history.size();
                        }
                    }
                }
                if (reused == 0) {
                    engine.reset_session();
                    history.clear();
                    next_tok = -1;
                } else {
                    std::printf("RESUME %zu\n", reused);
                    std::fflush(stdout);
                }

                auto t_pf0 = std::chrono::steady_clock::now();
                bool cancelled = false;
                for (size_t i = reused; i < N;) {
                    if (stop_req.load(std::memory_order_relaxed)) {
                        cancelled = true;
                        break;
                    }
                    size_t chunk = std::min<size_t>(4080, N - i);
                    std::vector<int32_t> prev0s(chunk);
                    std::vector<int32_t> prev1s(chunk);
                    for (size_t t = 0; t < chunk; ++t) {
                        prev0s[t] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                        prev1s[t] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                        history.push_back(req_ids[i + t]);
                    }
                    bool is_last = (i + chunk == N);
                    next_tok = engine.step_prefill_batch(&req_ids[i], prev0s.data(), prev1s.data(),
                                                         static_cast<int32_t>(chunk), static_cast<int32_t>(i),
                                                         is_last, nullptr, nullptr,
                                                         &req_sp, &history, &req_rng);
                    if (spec_ngram > 0 && engine.has_mtp()) {
                        engine.mtp_catchup_kv(history, static_cast<int32_t>(history.size()) - 1);
                    }
                    i += chunk;
                    double pf_ms_so_far = std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - t_pf0)
                                              .count();
                    double pf_tok_s = (pf_ms_so_far > 0.0) ? ((i - reused) * 1000.0 / pf_ms_so_far) : 0.0;
                    std::printf("PP %zu %zu %.0f %.1f\n", i, N, pf_ms_so_far, pf_tok_s);
                    std::fflush(stdout);
                }
                double prefill_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t_pf0)
                                        .count();

                if (!cancelled) {
                    engine.save_turn_ckpt(history, next_tok);
                }

                auto t_dec0 = std::chrono::steady_clock::now();
                int generated = 0;
                int spec_accepted_total = 0;
                int spec_drafted_total = 0;
                int spec_max_accepted_k = 0;
                int spec_max_drafted_k = 0;
                const char* finish_reason = cancelled ? "cancel" : "length";

                for (int step = 0; !cancelled && step < req_max_new;) {
                    if (stop_req.load(std::memory_order_relaxed)) {
                        finish_reason = "cancel";
                        break;
                    }
                    if (next_tok < 0) {
                        finish_reason = "stop";
                        break;
                    }
                    if (next_tok == 248044 || next_tok == 248046) {
                        std::printf("T %d\n", next_tok);
                        std::fflush(stdout);
                        generated++;
                        finish_reason = "stop";
                        break;
                    }
                    std::printf("T %d\n", next_tok);
                    std::fflush(stdout);
                    generated++;
                    step++;
                    if (step >= req_max_new) {
                        finish_reason = "length";
                        break;
                    }
                    if (stop_req.load(std::memory_order_relaxed)) {
                        finish_reason = "cancel";
                        break;
                    }

                    std::vector<int32_t> draft;
                    bool used_mtp = false;
                    int max_rem = req_max_new - step - 1;
                    int32_t eff_k = std::min<int32_t>(mtp_max_k, std::max(0, max_rem));
                    if (spec_ngram && max_rem > 0) {
                        drafter.observe_target_confidence(engine.last_target_prob(), engine.last_target_margin());
                        if (!mtp_only) {
                            drafter.draft(next_tok, N, history, spec_ngram, draft);
                        }
                        if (draft.empty() && engine.has_mtp() && eff_k > 0) {
                            engine.draft_mtp(next_tok, history, eff_k, mtp_min_prob, draft);
                            used_mtp = !draft.empty();
                        }
                    }
                    if (static_cast<int>(draft.size()) > max_rem) {
                        draft.resize(std::max(0, max_rem));
                    }

                    if (!draft.empty()) {
                        auto t_ver0 = std::chrono::steady_clock::now();
                        const int32_t K = static_cast<int32_t>(draft.size());
                        const int32_t M = 1 + K;
                        spec_drafted_total += K;
                        engine.save_recurrent_ckpt();

                        int32_t pos0 = static_cast<int32_t>(history.size());
                        std::vector<int32_t> b_toks(static_cast<size_t>(M));
                        std::vector<int32_t> b_p0(static_cast<size_t>(M));
                        std::vector<int32_t> b_p1(static_cast<size_t>(M));
                        std::vector<int32_t> preds(static_cast<size_t>(M), 0);
                        for (int32_t j = 0; j < M; ++j) {
                            int32_t tk = (j == 0) ? next_tok : draft[j - 1];
                            b_toks[j] = tk;
                            b_p0[j] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                            b_p1[j] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                            history.push_back(tk);
                        }
                        (void) engine.step_prefill_batch(b_toks.data(), b_p0.data(), b_p1.data(), M, pos0, true,
                                                         nullptr, preds.data(), &req_sp, &history, &req_rng,
                                                         draft.data());

                        int32_t accepted = 0;
                        while (accepted < K && preds[accepted] == draft[accepted]) {
                            accepted++;
                        }
                        spec_accepted_total += accepted;
                        spec_max_accepted_k = std::max(spec_max_accepted_k, accepted);
                        spec_max_drafted_k = std::max(spec_max_drafted_k, K);
                        const int32_t n_adv = 1 + accepted;
                        engine.finish_verify_step(M, n_adv);
                        double ver_ms = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - t_ver0)
                                            .count();
                        if (!used_mtp) {
                            drafter.observe_step(K, accepted, ver_ms);
                        } else if (ver_ms > 5.0) {
                            const double mtp_overhead = 4.2 * static_cast<double>(K);
                            const double pure_ver = std::max(12.0, ver_ms - mtp_overhead);
                            drafter.observe_step(0, 0, pure_ver / (1.18 + 0.05 * static_cast<double>(K)));
                        }

                        if (accepted < K) {
                            if (!used_mtp) {
                                int32_t prev_t = b_toks[accepted];
                                int32_t prev2_t = b_p1[accepted];
                                int32_t bad_cand = draft[accepted];
                                drafter.record_rejection(prev2_t, prev_t, bad_cand);
                            }
                            history.resize(static_cast<size_t>(pos0 + n_adv));
                        }
                        bool hit_eos = false;
                        for (int32_t j = 0; j < accepted; ++j) {
                            if (draft[j] == 248044 || draft[j] == 248046) {
                                std::printf("T %d\n", draft[j]);
                                std::fflush(stdout);
                                generated++;
                                history.resize(static_cast<size_t>(pos0 + 1 + j));
                                finish_reason = "stop";
                                hit_eos = true;
                                break;
                            }
                            std::printf("T %d\n", draft[j]);
                            std::fflush(stdout);
                            generated++;
                            step++;
                        }
                        if (hit_eos) break;
                        next_tok = preds[accepted];
                    } else {
                        auto t_st0 = std::chrono::steady_clock::now();
                        int32_t prev0 = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                        int32_t prev1 = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                        int32_t pos = static_cast<int32_t>(history.size());
                        history.push_back(next_tok);
                        next_tok = engine.step_token(history.back(), prev0, prev1, pos, true, nullptr,
                                                     &req_sp, &history, &req_rng);
                        double st_ms = std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - t_st0)
                                           .count();
                        drafter.observe_step(0, 0, st_ms);
                    }
                }
                if (!cancelled) {
                    if (spec_ngram > 0 && engine.has_mtp()) {
                        engine.mtp_catchup_kv(history, static_cast<int32_t>(history.size()) - 1);
                    }
                    engine.save_turn_ckpt(history, next_tok);
                }
                double decode_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - t_dec0)
                                       .count();
                std::printf("DONE %d %zu %.0f %.0f %s %d %d %zu\n",
                            generated, N, prefill_ms, decode_ms, finish_reason,
                            spec_accepted_total, spec_drafted_total, reused);
                std::fflush(stdout);
                std::fprintf(stderr,
                             "[gfx1201-nvfp4-serve] prompt=%zu (reused=%zu, new=%zu) in %.1f ms (%.1f tok/s) | "
                             "gen=%d in %.1f ms (%.2f tok/s, spec=%d/%d, max_k=%d/%d) | finish=%s\n",
                             N, reused, N - reused, prefill_ms,
                             (N > reused && prefill_ms > 0.0) ? ((N - reused) * 1000.0 / prefill_ms) : 0.0,
                             generated, decode_ms,
                             (generated > 0 && decode_ms > 0.0) ? (generated * 1000.0 / decode_ms) : 0.0,
                             spec_accepted_total, spec_drafted_total, spec_max_accepted_k, spec_max_drafted_k, finish_reason);
                std::fflush(stderr);
            }
        }
        if (reader_thr.joinable()) reader_thr.join();
        return 0;
    }

    std::mt19937_64 cli_rng(cli_sp.seed);
    std::vector<float> dumped_logits;
    auto t_prefill_start = std::chrono::steady_clock::now();

    const size_t total_prompt = prompt_ids.size();
    for (size_t i = 0; i < total_prompt;) {
        size_t chunk = std::min<size_t>(4080, total_prompt - i);
        std::vector<int32_t> prev0s(chunk);
        std::vector<int32_t> prev1s(chunk);
        for (size_t t = 0; t < chunk; ++t) {
            prev0s[t] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
            prev1s[t] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
            history.push_back(prompt_ids[i + t]);
        }
        bool is_last = (i + chunk == total_prompt);
        next_tok = engine.step_prefill_batch(&prompt_ids[i], prev0s.data(), prev1s.data(),
                                             static_cast<int32_t>(chunk), static_cast<int32_t>(i), is_last,
                                             (is_last && !dump_logits_path.empty()) ? &dumped_logits : nullptr,
                                             nullptr, &cli_sp, &history, &cli_rng);
        if (spec_ngram > 0 && engine.has_mtp()) {
            engine.mtp_catchup_kv(history, static_cast<int32_t>(history.size()) - 1);
        }
        i += chunk;
    }
    double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill_start).count();

    if (!dump_logits_path.empty() && !dumped_logits.empty()) {
        std::ofstream lf(dump_logits_path, std::ios::binary);
        lf.write(reinterpret_cast<const char*>(dumped_logits.data()),
                 static_cast<std::streamsize>(dumped_logits.size() * sizeof(float)));
        std::printf("[gfx1201-nvfp4] Dumped %zu float32 logits to %s\n", dumped_logits.size(), dump_logits_path.c_str());
    }

    std::string display_prompt;
    if (tokens_csv.empty()) {
        display_prompt = prompt;
    } else {
        size_t start_idx = prompt_ids.size() > 64 ? prompt_ids.size() - 64 : 0;
        if (start_idx > 0) display_prompt = "... ";
        for (size_t k = start_idx; k < prompt_ids.size(); ++k) {
            display_prompt += tokenizer.decode_token(prompt_ids[k]);
        }
    }
    if (display_prompt.size() > 240) {
        display_prompt = "... " + display_prompt.substr(display_prompt.size() - 240);
    }
    std::printf("\n--- Output ---\n%s", display_prompt.c_str());
    auto t_decode_start = std::chrono::steady_clock::now();
    int generated = 0;
    int spec_accepted_total = 0;
    int spec_drafted_total = 0;
    int spec_max_accepted_k = 0;
    int spec_max_drafted_k = 0;

    for (int step = 0; step < max_new_tokens;) {
        if (next_tok < 0) break;
        std::string piece = tokenizer.decode_token(next_tok);
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
        generated++;
        step++;
        if (next_tok == 248044 || next_tok == 248046 || step >= max_new_tokens) break;

        std::vector<int32_t> draft;
        bool used_mtp = false;
        int max_rem = max_new_tokens - step - 1;
        int32_t eff_k = std::min<int32_t>(mtp_max_k, std::max(0, max_rem));
        if (spec_ngram && max_rem > 0) {
            drafter.observe_target_confidence(engine.last_target_prob(), engine.last_target_margin());
            if (!mtp_only) {
                drafter.draft(next_tok, total_prompt, history, spec_ngram, draft);
            }
            if (draft.empty() && engine.has_mtp() && eff_k > 0) {
                engine.draft_mtp(next_tok, history, eff_k, mtp_min_prob, draft);
                used_mtp = !draft.empty();
            }
        }
        if (static_cast<int>(draft.size()) > max_rem) {
            draft.resize(std::max(0, max_rem));
        }

        if (!draft.empty()) {
            auto t_ver0 = std::chrono::steady_clock::now();
            const int32_t K = static_cast<int32_t>(draft.size());
            const int32_t M = 1 + K;
            spec_drafted_total += K;
            engine.save_recurrent_ckpt();

            int32_t pos0 = static_cast<int32_t>(history.size());
            std::vector<int32_t> b_toks(static_cast<size_t>(M));
            std::vector<int32_t> b_p0(static_cast<size_t>(M));
            std::vector<int32_t> b_p1(static_cast<size_t>(M));
            std::vector<int32_t> preds(static_cast<size_t>(M), 0);
            for (int32_t j = 0; j < M; ++j) {
                int32_t tk = (j == 0) ? next_tok : draft[j - 1];
                b_toks[j] = tk;
                b_p0[j] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                b_p1[j] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                history.push_back(tk);
            }
            (void) engine.step_prefill_batch(b_toks.data(), b_p0.data(), b_p1.data(), M, pos0, true,
                                             nullptr, preds.data(), &cli_sp, &history, &cli_rng,
                                             draft.data());

            int32_t accepted = 0;
            while (accepted < K && preds[accepted] == draft[accepted]) {
                accepted++;
            }
            spec_accepted_total += accepted;
            spec_max_accepted_k = std::max(spec_max_accepted_k, accepted);
            spec_max_drafted_k = std::max(spec_max_drafted_k, K);
            const int32_t n_adv = 1 + accepted;
            engine.finish_verify_step(M, n_adv);
            double ver_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_ver0)
                                .count();
            if (!used_mtp) {
                drafter.observe_step(K, accepted, ver_ms);
            } else if (ver_ms > 5.0) {
                const double mtp_overhead = 4.2 * static_cast<double>(K);
                const double pure_ver = std::max(12.0, ver_ms - mtp_overhead);
                drafter.observe_step(0, 0, pure_ver / (1.18 + 0.05 * static_cast<double>(K)));
            }

            if (accepted < K) {
                // Record rejected transition so it is never re-drafted in this context
                if (!used_mtp) {
                    int32_t prev_t = b_toks[accepted];
                    int32_t prev2_t = b_p1[accepted];
                    int32_t bad_cand = draft[accepted];
                    drafter.record_rejection(prev2_t, prev_t, bad_cand);
                }
                history.resize(static_cast<size_t>(pos0 + n_adv));
            }
            for (int32_t j = 0; j < accepted; ++j) {
                std::string dp = tokenizer.decode_token(draft[j]);
                std::printf("%s", dp.c_str());
                std::fflush(stdout);
                generated++;
                step++;
            }
            next_tok = preds[accepted];
        } else {
            auto t_st0 = std::chrono::steady_clock::now();
            int32_t prev0 = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
            int32_t prev1 = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
            int32_t pos = static_cast<int32_t>(history.size());
            history.push_back(next_tok);
            next_tok = engine.step_token(history.back(), prev0, prev1, pos, true, nullptr,
                                         &cli_sp, &history, &cli_rng);
            double st_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t_st0)
                               .count();
            drafter.observe_step(0, 0, st_ms);
        }
    }
    double decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_decode_start).count();
    int decode_steps = std::max(1, generated - 1);
    std::printf("\n--------------\n");
    if (spec_drafted_total > 0) {
        std::printf("[gfx1201-nvfp4] spec-ngram: accepted %d / %d drafted tokens (%.1f%%, max_k=%d/%d)\n",
                    spec_accepted_total, spec_drafted_total, 100.0 * spec_accepted_total / spec_drafted_total,
                    spec_max_accepted_k, spec_max_drafted_k);
    }
    std::printf("[gfx1201-nvfp4] prefill: %zu tokens in %.3f s (%.2f tok/s) | decode: %d steps in %.3f s (%.2f tok/s, %.2f ms/tok)\n",
                prompt_ids.size(), prefill_s, prompt_ids.size() / prefill_s,
                decode_steps, decode_s, decode_steps / std::max(1e-6, decode_s),
                1000.0 * decode_s / decode_steps);
    return 0;
}

}  // namespace strata::program
