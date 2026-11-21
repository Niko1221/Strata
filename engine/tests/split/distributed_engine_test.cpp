#include "engine.hpp"
#include "distributed.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void save(const std::filesystem::path& path, const std::vector<float>& values) {
    require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }), "non-finite head logits");
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    require(bool(out), "cannot save teacher-forced logits");
}
void compare_batches(const std::vector<float>& a, const std::vector<float>& b, int vocab) {
    require(a.size() == b.size(), "batch head shape changed");
    double worst = 0;
    for (size_t row = 0; row < a.size(); row += vocab) {
        const auto ma = std::max_element(a.begin() + row, a.begin() + row + vocab);
        const auto mb = std::max_element(b.begin() + row, b.begin() + row + vocab);
        require(ma - a.begin() == mb - b.begin(), "batch changed the next token");
        double za = 0, zb = 0;
        for (int i = 0; i < vocab; ++i) { za += std::exp(double(a[row+i] - *ma)); zb += std::exp(double(b[row+i] - *mb)); }
        double kl = 0;
        for (int i = 0; i < vocab; ++i) {
            const double la = double(a[row+i]) - *ma - std::log(za), lb = double(b[row+i]) - *mb - std::log(zb);
            kl += std::exp(la) * (la - lb);
        }
        worst = std::max(worst, kl);
    }
    // The shipped cache check uses the same 1.5e-3 KL bound for reusing an engine state.
    require(worst <= 1.5e-3, "batch changed the next-token distribution beyond the cache-check bound");
    std::printf("batch versus isolated lanes: maximum KL %.9g, next tokens agree\n", worst);
}
}

int main(int argc, char** argv) try {
    require(argc == 4, "expected MODEL CASES OUTPUT_DIRECTORY");
    fnx::distributed::Runtime runtime(argc, argv);
    try {
    runtime.initialize();
    fnx::Model model(argv[1], fnx::Load::PLAIN, nullptr, nullptr, fnx::distributed::first_card());
    fnx::Engine engine(model, 1024, 2);
    runtime.run(engine);
    if (!fnx::distributed::coordinator()) return 0;
    const std::filesystem::path directory(argv[3]);
    std::filesystem::create_directories(directory);
    std::ifstream input(argv[2]);
    require(bool(input), "cannot open test cases");
    const int other = engine.add_lane();
    engine.warm_graphs();
    std::vector<std::vector<int>> batch_prompts;
    int index = 0, count = 0, steps = 0;
    while (input >> count >> steps) {
        require(count > 1 && count < 8192 && steps > 0 && steps <= fnx::MAXT, "invalid fixture shape");
        std::vector<int> prompt(count), continuation(steps);
        for (int& id : prompt) require(bool(input >> id), "truncated fixture prompt");
        for (int& id : continuation) require(bool(input >> id), "truncated fixture continuation");
        if (batch_prompts.size() < 2) batch_prompts.push_back(prompt);
        engine.use_lane(0);
        engine.reset();
        engine.prefill(std::vector<int>(prompt.begin(), prompt.end() - 1), prompt.back());
        engine.verify({prompt.back()});
        save(directory / (std::to_string(index) + ".prompt.f32"), engine.last_logits(1));
        engine.commit(1);
        const auto state = engine.save_state(true);
        engine.verify(continuation);
        const auto expected = engine.last_logits(steps);
        save(directory / (std::to_string(index) + ".continuation.f32"), expected);
        engine.commit(steps);
        engine.use_lane(other);
        engine.reset();
        engine.window({prompt.front(), prompt.back()});
        engine.use_lane(0);
        engine.load_state(*state);
        engine.verify(continuation);
        const auto restored = engine.last_logits(steps);
        require(restored.size() == expected.size() && std::memcmp(restored.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "another lane or snapshot restore changed teacher-forced logits");
        engine.commit(steps);
        const fnx::Engine::Penalties penalties{{{1.1f, .2f, .1f, {{prompt.back(), 3}, {continuation.back(), 1}}}}};
        engine.verify({continuation.back()});
        const auto top = engine.top_tokens(1, penalties);
        require(top.size() == 1 && top[0].size() == size_t(fnx::TOPK_N), "distributed sampler lost candidates");
        require(std::all_of(top[0].begin(), top[0].end(), [&](const auto& p) {
            return std::isfinite(p.first) && p.second >= 0 && p.second < model.vocab;
        }), "distributed sampler returned invalid candidates");
        engine.commit(1);
        std::printf("case %d: full logits, parked-state restore, lane isolation and penalties passed\n", index++);
        std::fflush(stdout);
    }
    require(index > 0 && input.eof(), "no complete test cases");
    require(batch_prompts.size() == 2, "need two prompts for the batch check");
    {
        const std::vector<int> lanes{0, other}, counts{2, 3}, tokens{198, 17, 198, 18, 13};
        std::vector<std::shared_ptr<fnx::Engine::SeqState>> states;
        std::vector<float> expected;
        int from = 0;
        engine.set_pass_rows(8192);
        for (int i = 0; i < 2; ++i) {
            engine.use_lane(lanes[i]); engine.reset();
            const auto& p = batch_prompts[i];
            engine.prefill(std::vector<int>(p.begin(), p.end()-1), p.back()); engine.window({p.back()});
            states.push_back(engine.save_state(true));
            engine.verify(std::vector<int>(tokens.begin()+from, tokens.begin()+from+counts[i]));
            const auto rows = engine.last_logits(counts[i]); expected.insert(expected.end(), rows.begin(), rows.end());
            engine.commit(counts[i]); from += counts[i];
        }
        for (int i = 0; i < 2; ++i) { engine.use_lane(lanes[i]); engine.load_state(*states[i]); }
        engine.verify_rows(lanes, counts, tokens);
        const auto batch = engine.last_logits(5);
        save(directory / "batch.f32", batch); compare_batches(expected, batch, model.vocab);
        engine.commit_rows(counts);
        expected.clear();
        for (int i = 0; i < 2; ++i) {
            engine.use_lane(lanes[i]); engine.load_state(*states[i]); engine.window({tokens[i]});
            const auto row = engine.last_logits(1); expected.insert(expected.end(), row.begin(), row.end());
        }
        for (int i = 0; i < 2; ++i) { engine.use_lane(lanes[i]); engine.load_state(*states[i]); }
        engine.step(lanes, {tokens[0], tokens[1]});
        const auto step = engine.last_logits(2);
        save(directory / "step.f32", step); compare_batches(expected, step, model.vocab);
    }
    engine.use_lane(other);
    engine.reset();
    constexpr int cells = 30 * 30;
    std::vector<float> rows(size_t(cells) * fnx::N);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = float(int(i % 29) - 14) / 16;
    engine.set_images({{1, cells, 30, 30, rows.data()}});
    std::vector<int> image_prompt{248045};
    for (int i = 0; i < cells; ++i) image_prompt.push_back(fnx::Engine::IMAGE_TOK + i);
    engine.prefill(image_prompt, 198);
    engine.verify({198});
    save(directory / "image.f32", engine.last_logits(1));
    engine.commit(1);
    engine.reset();
    std::printf("image payload %zu bytes: chunked broadcast and complete prefill passed\n", rows.size() * sizeof(float));
    return 0;
    } catch (const std::exception& e) {
        // Report before unwinding the runtime: its unfinished MPI job must abort.
        std::fprintf(stderr, "distributed engine test: %s\n", e.what());
        std::fflush(stderr);
        throw;
    }
} catch (const std::exception& e) {
    std::fprintf(stderr, "distributed engine test: %s\n", e.what());
    return 1;
}
