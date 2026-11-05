// cann/tests/model_seq.cpp - the whole model over a token sequence: parity probe, generator, server.
//
// The engine lives in src/engine.cpp; this file is only the three ways of driving it:
//
//   model_seq <model.gguf> <tokens comma separated>          print the next-token distribution
//   model_seq <model.gguf> <tokens> <n> [temperature]        also generate n tokens greedily
//   model_seq <model.gguf> 0 serve                           answer RUN lines on stdin
//
// Serving exists so the port can be exercised as a service, not only as a test: it is what
// cann/tests/serve_engine.py puts an OpenAI-compatible endpoint in front of.
#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/engine.hpp"

using namespace strata::cann;

namespace {

std::vector<int64_t> parse_ids(const std::string & list) {
    std::vector<int64_t> out;
    size_t s = 0;
    while (s <= list.size()) {
        const size_t comma = list.find(',', s);
        const std::string item = list.substr(s, comma == std::string::npos ? std::string::npos : comma - s);
        if (!item.empty()) {
            out.push_back(std::atoll(item.c_str()));
        }
        if (comma == std::string::npos) {
            break;
        }
        s = comma + 1;
    }
    return out;
}

int sample(const std::vector<std::pair<float, int>> & best, float temp, std::mt19937 & rng) {
    if (temp <= 0.0f || best.size() < 2) {
        return best[0].second;
    }
    std::vector<double> p(best.size());
    double sum = 0.0;
    for (size_t i = 0; i < best.size(); ++i) {
        p[i] = std::exp((double) (best[i].first - best[0].first) / temp);
        sum += p[i];
    }
    std::uniform_real_distribution<double> draw(0.0, sum);
    double r = draw(rng);
    for (size_t i = 0; i < p.size(); ++i) {
        r -= p[i];
        if (r <= 0.0) {
            return best[i].second;
        }
    }
    return best[0].second;
}

// One request: prefill, then generate, printing the same lines the scripts parse.
void run(Model & model, const std::vector<int64_t> & prompt, int64_t n_gen, float temp, bool verbose) {
    model.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> hidden;
    for (size_t i = 0; i < prompt.size(); ++i) {
        hidden = model.step(prompt[i], (int64_t) i);
        if (verbose) {
            printf("token %lld/%lld done  %.1f s\n", (long long) (i + 1), (long long) prompt.size(),
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }

    Model::Top t = model.top(hidden, n_gen > 0 ? 32 : 5);
    printf("LOGP %.6f\n", t.log_z);
    if (verbose) {
        printf("\nnext token from position %lld:\n", (long long) (prompt.size() - 1));
        for (size_t i = 0; i < std::min<size_t>(5, t.best.size()); ++i) {
            printf("  token %d  logit %.4f\n", t.best[i].second, (double) t.best[i].first);
        }
    }

    std::mt19937 rng(1234u);
    int64_t pos = (int64_t) prompt.size();
    for (int64_t step = 0; step < n_gen; ++step) {
        if (pos >= model.capacity()) {
            printf("GEN_STOP context full\n");
            break;
        }
        const int pick = sample(t.best, temp, rng);
        printf("GEN %d %.6f\n", pick, (double) t.best[0].first);
        fflush(stdout);
        if (step + 1 >= n_gen) {
            break;
        }
        hidden = model.step(pick, pos);
        ++pos;
        t = model.top(hidden, 32);
    }
    if (verbose) {
        printf("done in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        printf("usage: model_seq <model.gguf> <tokens> [n_predict [temperature]]\n");
        printf("       model_seq <model.gguf> 0 serve\n");
        return 2;
    }
    const std::string path = argv[1];
    const bool serve = argc > 3 && std::string(argv[3]) == "serve";
    const int64_t n_predict = argc > 3 && !serve ? std::atoll(argv[3]) : 0;
    const float gen_temp = argc > 4 ? (float) std::atof(argv[4]) : 0.0f;

    std::vector<int64_t> toks = parse_ids(argv[2]);
    if (toks.empty() && !serve) {
        printf("no tokens\n");
        return 2;
    }
    const int64_t kCap = serve ? 512 : (int64_t) toks.size() + std::max<int64_t>(0, n_predict);

    try {
        Model model(path, kCap);
        if (serve) {
            printf("READY\n");
            fflush(stdout);
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.rfind("QUIT", 0) == 0) {
                    break;
                }
                if (line.rfind("RUN ", 0) != 0) {
                    printf("ERR bad request\n");
                    fflush(stdout);
                    continue;
                }
                std::istringstream is(line.substr(4));
                int64_t n = 0;
                double temp = 0.0;
                std::string ids;
                is >> n >> temp >> ids;
                const std::vector<int64_t> prompt = parse_ids(ids);
                if (prompt.empty()) {
                    printf("ERR no prompt\n");
                    fflush(stdout);
                    continue;
                }
                run(model, prompt, n, (float) temp, false);
                printf("MEM %.2f GB\n", (double) model.cache_bytes() / 1073741824.0);
                printf("END\n");
                fflush(stdout);
            }
            Model::print_slab_stats();
            shutdown();
            return 0;
        }

        run(model, toks, n_predict, gen_temp, true);
        Model::print_slab_stats();
        shutdown();
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
