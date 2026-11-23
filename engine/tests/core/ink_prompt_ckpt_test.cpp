// engine/tests/core/ink_prompt_ckpt_test.cpp - Unit test for where strata-inkling saves its prompt checkpoint (#6627; no GPU)
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_prompt_ckpt.h"

#include <cstdio>
#include <vector>

using namespace ink2;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)

constexpr int32_t MODEL = 200001;     // <|message_model|>
constexpr int32_t TOOL = 200003;      // <|message_tool|>
constexpr int32_t TEXT = 200004;      // <|content_text|>
constexpr int32_t SAMPLING = 200006;  // <|content_model_end_sampling|>
constexpr int32_t THINK = 200008;     // <|content_thinking|>
constexpr int32_t END = 200010;       // <|end_message|>
constexpr int32_t INVOKE = 200049;    // <|content_invoke_tool_json|>

std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

bool is_prefix(const std::vector<int32_t>& p, const std::vector<int32_t>& of) {
    if (p.size() > of.size()) return false;
    for (size_t i = 0; i < p.size(); ++i) if (p[i] != of[i]) return false;
    return true;
}
}  // namespace

int main() {
    // The rendered history: tool declaration, system, effort and a user turn (any tokens ending in <|end_message|>).
    const std::vector<int32_t> history = {11, 12, 13, END, 21, TEXT, 22, 23, END, 31, TEXT, 32, 33, 34, END};
    const int h = static_cast<int>(history.size());

    // Without tools: <|message_model|><|content_text|> -> in front of the opener.
    const std::vector<int32_t> plain = cat(history, {MODEL, TEXT});
    CHECK(prompt_ckpt_pos(plain, MODEL) == h);

    // With tools: the empty thinking turn -> in front of the opener, not at prompt_len - 1.
    const std::vector<int32_t> tools = cat(history, {MODEL, THINK, END});
    CHECK(prompt_ckpt_pos(tools, MODEL) == h);

    // The next request of a tool-call loop starts with the checkpointed tokens (the old prompt_len - 1 did not).
    const std::vector<int32_t> next = cat(history, {MODEL, 41, INVOKE, 42, 43, END, SAMPLING, TOOL, 41, TEXT, 44, END, MODEL, THINK, END});
    const std::vector<int32_t> ckpt(tools.begin(), tools.begin() + prompt_ckpt_pos(tools, MODEL));
    const std::vector<int32_t> old_ckpt(tools.begin(), tools.end() - 1);
    CHECK(is_prefix(ckpt, next));
    CHECK(!is_prefix(old_ckpt, next));
    // ... and its own checkpoint is in front of its own opener.
    CHECK(prompt_ckpt_pos(next, MODEL) == static_cast<int>(next.size()) - 3);

    // A multi-turn text follow-up also starts with the checkpointed tokens.
    const std::vector<int32_t> follow = cat(history, {MODEL, TEXT, 51, 52, END, SAMPLING, 61, TEXT, 62, END, MODEL, TEXT});
    const std::vector<int32_t> plain_ckpt(plain.begin(), plain.begin() + prompt_ckpt_pos(plain, MODEL));
    CHECK(is_prefix(plain_ckpt, follow));

    // No opener near the end (a raw prompt): prompt_len - 1, as before.
    CHECK(prompt_ckpt_pos(history, MODEL) == h - 1);
    // An earlier assistant turn far from the end does not count.
    const std::vector<int32_t> far = cat({MODEL, TEXT, 1, END}, history);
    CHECK(prompt_ckpt_pos(far, MODEL) == static_cast<int>(far.size()) - 1);
    // Edge cases.
    CHECK(prompt_ckpt_pos({}, MODEL) == 0);
    CHECK(prompt_ckpt_pos({MODEL}, MODEL) == 0);
    CHECK(prompt_ckpt_pos({7, MODEL}, MODEL) == 1);

    if (failed != 0) {
        std::fprintf(stderr, "ink_prompt_ckpt_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("ink_prompt_ckpt_test: OK\n");
    return 0;
}
