// engine/src/inkling2/ink_prompt_ckpt.h - Where strata-inkling saves its prompt checkpoint (#6627; no GPU)
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>
#include <vector>

namespace ink2 {

// The server ends every rendered prompt with an assistant opener (serve/inkling_dialect.py finish_prompt):
//   <|message_model|><|content_text|>                       without tools
//   <|message_model|><|content_thinking|><|end_message|>    with tools (an empty thinking turn)
// The next request renders the model's reply in its own way after <|message_model|> (a tool call is
// <|message_model|>NAME<|content_invoke_tool_json|>..., and the empty thinking turn is never in the history),
// so only the tokens in front of that opener are sure to come back. A checkpoint at prompt_len - 1 kept the
// opener and was never reused by a tool-call loop. The checkpoint goes in front of the last <|message_model|>
// when it is one of the last kMaxOpenerTokens tokens; otherwise (no opener: a raw prompt) at prompt_len - 1.
constexpr int kMaxOpenerTokens = 4;

inline int prompt_ckpt_pos(const std::vector<int32_t>& prompt_ids, int32_t message_model_token_id) {
    const int n = static_cast<int>(prompt_ids.size());
    if (n <= 0) return 0;
    for (int k = n - 1; k >= 1 && k >= n - kMaxOpenerTokens; --k) {
        if (prompt_ids[k] == message_model_token_id) return k;
    }
    return n - 1;
}

}  // namespace ink2
