#include "engine.hpp"
#include "strata/kprof.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4 && argc != 5 && argc != 6)
            throw std::runtime_error("usage: split_sycl_model_probe MODEL.gguf [--startup-init reused|fresh | CONTEXT MTP_DIR|- [PROMPT_IDS|- [SLOTS]]]");
        const bool startup_init = argc == 4 && std::string(argv[2]) == "--startup-init";
        const std::string mode = startup_init ? argv[3] : "";
        if (startup_init && mode != "reused" && mode != "fresh")
            throw std::invalid_argument("startup-init expects reused or fresh");
        fnx::Model model(argv[1]);
        size_t host = 0;
        for (const auto& card : model.cards) {
            host += card.host_bytes;
            std::cout << "MODEL card=" << card.dev
                      << " pci=" << dpct::get_device(card.dev)
                          .get_info<sycl::ext::intel::info::device::pci_address>()
                      << " ff=[" << card.ff0 << "," << card.ff1 << ")"
                      << " device_bytes=" << card.bytes
                      << " host_expert_bytes=" << card.host_bytes << std::endl;
        }
        std::cout << "MODEL_LOADED cards=" << model.cards.size()
                  << " full_host_expert_bytes=" << host << std::endl;
        if (startup_init) {
            for (const auto& card : model.cards) {
                dpct::select_device(card.dev);
                auto& dev = dpct::get_current_device();
                auto* compute = dev.create_queue(false, true);
                auto* auxiliary = dev.create_queue();
                auto* fresh = mode == "fresh" ? dev.create_queue(true) : nullptr;
                auto& q = mode == "fresh" ? *fresh : dpct::get_in_order_queue();
                constexpr size_t elements = 0x28000;
                auto* data = sycl::malloc_device<float>(elements, q);
                if (!data) throw std::bad_alloc();
                std::cout << "INIT_BEGIN card=" << card.dev << " mode=" << mode
                          << " bytes=" << elements * sizeof(float) << std::endl;
                try {
                    auto event = q.memset(data, 0, elements * sizeof(float));
                    std::cout << "INIT_SUBMITTED card=" << card.dev << std::endl;
                    event.wait_and_throw();
                    std::vector<float> values(elements, 1.f);
                    q.memcpy(values.data(), data, elements * sizeof(float)).wait_and_throw();
                    if (std::any_of(values.begin(), values.end(), [](float v) { return v != 0.f; }))
                        throw std::runtime_error("First allocation was not cleared");
                } catch (...) {
                    sycl::free(data, q);
                    throw;
                }
                sycl::free(data, q);
                dev.destroy_queue(fresh);
                dev.destroy_queue(auxiliary);
                dev.destroy_queue(compute);
                std::cout << "PASS first_allocation card=" << card.dev << " mode=" << mode << std::endl;
            }
            return 0;
        }
        if (argc >= 4) {
            std::unique_ptr<fnx::Mtp> mtp;
            if (std::string(argv[3]) != "-")
                mtp = std::make_unique<fnx::Mtp>(model, argv[3]);
            const int slots = argc == 6 ? std::stoi(argv[5]) : 1;
            if (slots < 1 || slots > fnx::MAXT) throw std::runtime_error("Invalid slot count");
            fnx::Engine engine(model, std::stoi(argv[2]), slots);
            if (mtp) engine.mtp_attach(*mtp);
            for (int i = 1; i < slots; ++i) engine.add_lane();
            const auto lent = engine.arena_lent();
            if (!lent.empty()) {
                engine.arena_reclaim(0);
                if (engine.arena_lent() != lent)
                    throw std::runtime_error("Arena reclaimed permanent lane-state blocks");
                engine.arena_churn();
                if (engine.arena_lent() != lent)
                    throw std::runtime_error("Arena churn reclaimed permanent lane-state blocks");
                std::cout << "PASS arena_lane_blocks_preserved slots=" << slots << std::endl;
            }
            for (const auto free : engine.free_vram_mib())
                std::cout << "ENGINE free_vram_mib=" << free << std::endl;
            std::cout << "ENGINE_LOADED context=" << argv[2]
                      << " slots=" << engine.lanes() << std::endl;
            if (argc >= 5 && std::string(argv[4]) != "-") {
                std::ifstream input(argv[4]);
                if (!input) throw std::runtime_error("Cannot open prompt IDs");
                std::vector<int> ids;
                int id;
                while (input >> id) {
                    if (id < 0 || id >= model.vocab) throw std::runtime_error("Invalid prompt ID");
                    ids.push_back(id);
                }
                if (!input.eof() || ids.size() < 2) throw std::runtime_error("Invalid prompt IDs file");
                const int last = ids.back();
                ids.pop_back();
                if (const char* compare = std::getenv("FNX_PROBE_BATCH_COMPARE")) {
                    const int batch = std::stoi(compare);
                    const char* steps_env = std::getenv("FNX_PROBE_STEPS");
                    const int steps = steps_env ? std::stoi(steps_env) : 64;
                    if (batch < 2 || batch > slots || steps < 1 || steps > 256)
                        throw std::invalid_argument("Invalid batch comparison or step count");
                    engine.reset();
                    engine.prefill(ids, last);
                    const auto state = engine.save_state(true);
                    std::vector<std::vector<float>> reference;
                    std::vector<int> forced;
                    bool identical = true;
                    std::map<std::string, std::vector<float>> layer_reference;
                    bool trace_initial = false, layer_difference = false;
                    int trace_count = 0, trace_step = 0;
                    if (std::getenv("FNX_PROBE_BATCH_EAGER"))
                        engine.dump = [&](const std::string& name, int layer, int, int width,
                                          const std::vector<float>& values) {
                            if (trace_step != 0) return;
                            const std::string key = name + ":" + std::to_string(layer);
                            if (trace_initial) {
                                layer_reference[key] = std::vector<float>(values.begin(), values.begin() + width);
                            } else if (!layer_difference) {
                                const auto& expected = layer_reference.at(key);
                                double error = 0;
                                for (int v = 0; v < width; ++v)
                                    error = std::max(error, std::fabs(double(values[v]) - expected[v]));
                                if (error != 0) {
                                    std::cout << "BATCH_FIRST_LAYER_DIFFERENCE count=" << trace_count
                                              << " name=" << name << " layer=" << layer
                                              << " max_error=" << error << std::endl;
                                    layer_difference = true;
                                }
                            }
                        };
                    auto decode = [&](int count, bool initial) {
                        trace_initial = initial;
                        trace_count = count;
                        layer_difference = false;
                        std::vector<int> lanes;
                        for (int r = 0; r < count; ++r) {
                            engine.use_lane(r);
                            engine.load_state(*state);
                            lanes.push_back(r);
                        }
                        engine.use_lane(0);
                        int token = last;
                        for (int step = 0; step < steps; ++step) {
                            trace_step = step;
                            const auto next = engine.step(lanes, std::vector<int>(count, token));
                            const auto logits = engine.last_logits(count);
                            if (logits.size() != size_t(count) * model.vocab)
                                throw std::runtime_error("Invalid batch logits size");
                            if (initial) {
                                reference.push_back(logits);
                                forced.push_back(next[0]);
                            }
                            for (int r = 0; r < count; ++r) {
                                size_t changed = 0;
                                double max_error = 0, squared = 0;
                                for (int v = 0; v < model.vocab; ++v) {
                                    const float value = logits[size_t(r) * model.vocab + v];
                                    if (!std::isfinite(value))
                                        throw std::runtime_error("Non-finite batch probe logits");
                                    const double delta = std::fabs(double(value) - reference[step][v]);
                                    changed += delta != 0;
                                    max_error = std::max(max_error, delta);
                                    squared += delta * delta;
                                }
                                identical = identical && changed == 0;
                                std::cout << "BATCH_COMPARE initial=" << initial
                                          << " count=" << count << " row=" << r
                                          << " step=" << step << " token=" << next[r]
                                          << " reference_token=" << forced[step]
                                          << " changed=" << changed << " max_error=" << max_error
                                          << " rms=" << std::sqrt(squared / model.vocab)
                                          << " wall_ms=" << engine.last_ms << std::endl;
                            }
                            token = forced[step];
                        }
                    };
                    decode(1, true);
                    for (int count = 1; count <= batch; count *= 2) decode(count, false);
                    if ((batch & (batch - 1)) != 0) decode(batch, false);
                    return identical ? 0 : 2;
                }
                if (const char* repeat = std::getenv("FNX_PROBE_REPEAT")) {
                    const int rounds = std::stoi(repeat);
                    if (rounds < 2 || rounds > 10)
                        throw std::invalid_argument("FNX_PROBE_REPEAT must be2..10");
                    std::map<std::string, std::vector<float>> layer_reference;
                    std::string label;
                    int trace_step = 0;
                    bool layer_difference = false;
                    if (std::getenv("FNX_PROBE_EAGER"))
                        engine.dump = [&](const std::string& name, int layer, int, int,
                                          const std::vector<float>& values) {
                            if (trace_step != 0) return;
                            const std::string key = name + ":" + std::to_string(layer);
                            if (label == "initial0") layer_reference[key] = values;
                            else if (!layer_difference) {
                                const auto& expected = layer_reference.at(key);
                                double error = 0;
                                for (size_t i = 0; i < values.size(); ++i)
                                    error = std::max(error, std::fabs(double(values[i]) - expected[i]));
                                if (error != 0) {
                                    std::cout << "FIRST_LAYER_DIFFERENCE run=" << label
                                              << " name=" << name << " layer=" << layer
                                              << " max_error=" << error << std::endl;
                                    layer_difference = true;
                                }
                            }
                        };
                    std::vector<std::vector<float>> reference;
                    std::vector<int> forced;
                    bool identical = true;
                    const bool profile_rows = std::getenv("FNX_PROBE_PROFILE");
                    auto decode = [&](const char* mode, int round) {
                        int token = last;
                        label = std::string(mode) + std::to_string(round);
                        layer_difference = false;
                        if (std::getenv("FNX_PROBE_HC_COMPARE"))
                            setenv("FNX_HC_STAGE_SPLIT", mode == std::string("initial") ? "0" : "1", 1);
                        const std::string phase = std::string("probe-") + mode;
                        strata::kprof::phase(phase.c_str());
                        for (int step = 0; step < 16; ++step) {
                            trace_step = step;
                            const bool initial = reference.size() == size_t(step);
                            const auto next = !profile_rows ? engine.verify({token})
                                : initial ? engine.step({0}, {token})
                                          : engine.verify_rows({0}, {1}, {token});
                            const auto logits = engine.last_logits(1);
                            size_t changed = 0;
                            double max_error = 0, squared = 0;
                            for (size_t i = 0; i < logits.size(); ++i) {
                                if (!std::isfinite(logits[i]))
                                    throw std::runtime_error("Non-finite probe logits");
                                if (reference.size() > size_t(step)) {
                                    const double delta = std::fabs(double(logits[i]) - reference[step][i]);
                                    changed += delta != 0;
                                    max_error = std::max(max_error, delta);
                                    squared += delta * delta;
                                }
                            }
                            std::cout << "DETERMINISM mode=" << mode << " round=" << round
                                      << " step=" << step << " token=" << next[0]
                                      << " changed=" << changed << " max_error=" << max_error
                                      << " rms=" << std::sqrt(squared / logits.size())
                                      << " wall_ms=" << engine.last_ms
                                      << " gpu_ms=" << engine.last_gpu_ms << std::endl;
                            identical = identical && changed == 0;
                            if (reference.size() == size_t(step)) {
                                reference.push_back(logits);
                                forced.push_back(next[0]);
                            }
                            if (!profile_rows) engine.commit(1);
                            else if (!initial) engine.commit_rows({1});
                            strata::kprof::window_end(phase.c_str());
                            token = forced[step];
                        }
                        strata::kprof::phase("probe-state");
                    };
                    engine.reset();
                    engine.prefill(ids, last);
                    const auto state = engine.save_state(true);
                    std::cout << "STATE_SAVED bytes=" << state->total() << std::endl;
                    decode("initial", 0);
                    for (int round = 1; round < rounds; ++round) {
                        engine.load_state(*state);
                        decode("restore", round);
                    }
                    for (int round = 1; round < rounds; ++round) {
                        engine.reset();
                        engine.prefill(ids, last);
                        const auto fresh = engine.save_state(true);
                        for (size_t c = 0; c < state->bytes.size(); ++c) {
                            if (fresh->bytes[c] != state->bytes[c])
                                throw std::runtime_error("Probe state layout changed");
                            const auto* a = static_cast<const unsigned char*>(state->host[c]);
                            const auto* b = static_cast<const unsigned char*>(fresh->host[c]);
                            size_t first = 0;
                            while (first < state->bytes[c] && a[first] == b[first]) ++first;
                            std::cout << "STATE_COMPARE round=" << round << " card=" << c
                                      << " first_difference_byte=" << first
                                      << " total_bytes=" << state->bytes[c] << std::endl;
                        }
                        decode("reprefill", round);
                    }
                    return identical ? 0 : 2;
                }
                engine.reset();
                engine.prefill(ids, last);
                const auto next = engine.verify({last});
                const auto logits = engine.last_logits(1);
                for (const float v : logits)
                    if (!std::isfinite(v)) throw std::runtime_error("Non-finite model logits");
                const auto range = std::minmax_element(logits.begin(), logits.end());
                if (*range.first == *range.second) throw std::runtime_error("Constant model logits");
                std::cout << "INFERENCE next_token=" << next.at(0)
                          << " logits=" << logits.size() << " min=" << *range.first
                          << " max=" << *range.second << std::endl;
                engine.commit(1);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
