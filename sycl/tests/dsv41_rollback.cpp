// Full-model integration check. A deliberately wrong first draft must rewind
// every window and preserve the subsequent target model's greedy trajectory.
#include "ds4_model.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dsv41_rollback_parity PACK PROMPT_IDS_FILE\n"); return 2;
    }
    std::ifstream file(argv[2]);
    std::stringstream text; text << file.rdbuf();
    std::string input = text.str(); std::replace(input.begin(), input.end(), ',', ' ');
    std::istringstream stream(input);
    std::vector<int> ids;
    for (int id; stream >> id;) ids.push_back(id);
    if (ids.empty()) return 2;
    strata::ds4::Options options;
    options.pack = argv[1]; options.gpus = {0}; options.max_context = 2048;
    options.reserve_mib = 2048; options.cpu_threads = 16;
    options.experts = 6; options.vision = false; options.prefill_chunk = 128;
    options.mmap_experts = 1; options.ssd_cache_gb = 0;
    options.dspark = true; options.dspark_auto = false;
    options.profile_readonly = true; options.expert_cache = false;
    strata::ds4::Model model;
    std::string error;
    auto check = [&](bool ok) { if (!ok) throw std::runtime_error(error.empty() ? "model operation failed" : error); };
    check(model.load(options, error));
    constexpr int steps = 16;
    std::vector<int> baseline;
    std::vector<std::vector<float>> reference;
    check(model.feed(ids.data(), int(ids.size()), 1, error));
    for (int i = 0; i <= steps + 1; ++i) {
        baseline.push_back(model.sample(0, 1, 0, 0, 0, 0));
        std::vector<float> logits;
        check(model.logits(logits)); reference.push_back(std::move(logits));
        if (i <= steps) check(model.step(baseline.back(), true, error));
    }
    model.reset();
    check(model.feed(ids.data(), int(ids.size()), 1, error));
    size_t mismatches = 0, bit_equal = 0, values = 0;
    double max_error = 0;
    int y = model.sample(0, 1, 0, 0, 0, 0);
    for (int i = 0; i < steps; ++i) {
        if (y != baseline[i]) ++mismatches;
        std::vector<int> draft;
        std::vector<float> confidence;
        check(model.draft(y, draft, confidence, error));
        if (draft.empty()) throw std::runtime_error("drafter returned no candidates");
        draft[0] = (baseline[i + 1] + 1) % model.vocab();
        int accepted = -1, next = -1;
        check(model.verify(y, draft, 0, 1, 0, 0, 0, 0, accepted, next, error));
        if (accepted != 0 || next != baseline[i + 1] || model.pos() != int64_t(ids.size()) + i + 1) ++mismatches;
        std::vector<float> actual;
        check(model.logits(actual));
        // verify() retains all window logit rows. Row zero predicts the next
        // target token after y; later rows consumed deliberately wrong drafts.
        if (actual.size() < reference[i + 1].size()) throw std::runtime_error("logit shape changed");
        for (size_t j = 0; j < reference[i + 1].size(); ++j) {
            if (!std::isfinite(actual[j])) ++mismatches;
            bit_equal += std::bit_cast<uint32_t>(actual[j]) == std::bit_cast<uint32_t>(reference[i + 1][j]);
            max_error = std::max(max_error, std::fabs(double(actual[j]) - reference[i + 1][j]));
            ++values;
        }
        std::printf("forced reject %d: accepted=%d next=%d expected=%d pos=%lld\n", i, accepted, next, baseline[i + 1], (long long)model.pos());
        std::fflush(stdout);
        y = next;
    }
    check(model.step(y, true, error));
    if (model.sample(0, 1, 0, 0, 0, 0) != baseline[steps + 1]) ++mismatches;
    std::printf("DeepSeek rollback: %d forced rejections, %zu semantic mismatches; logits %zu/%zu bit equal, max abs error %.9g\n",
        steps, mismatches, bit_equal, values, max_error);
    return mismatches ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what()); return 2;
}
