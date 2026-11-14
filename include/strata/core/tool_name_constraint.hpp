#pragma once
// include/strata/core/tool_name_constraint.hpp
//
// Constrained decoding for tool-call names, scoped to a real call envelope.  A model
// that writes `<tool_call><function=NAME>` can hallucinate a NAME no declared tool has;
// the parser then drops the call and the raw text leaks into the reply.  This constraint
// removes the possibility at the sampler: only while the reply is *inside* a `<tool_call>`
// envelope (and past the `<think>` reasoning block) does a `<function=` trigger, and only
// tokens that continue one of the declared names (followed by its `>` terminator) are
// sampled; a complete name releases the constraint.
//
// The envelope is what makes this safe to leave on: a `<function=` written anywhere else —
// in the answer's prose, or in the reasoning while the model *talks about* a call ("I will
// use `<function=foo>` …") — is plain text and is never constrained.  The reasoning block
// (`<think>…</think>`) is excluded as well, so the model may freely discuss the call format
// while it thinks.
//
// The engine streams token ids and has no detokenizer, so the server tokenizes the pieces
// and sends ids (see `parse_tool_name_constraint` for the wire format).  The mask is a
// dense `-inf`/`0` vector, exactly what `SamplerParams::logit_bias` takes (sampler.cu applies
// `l[v] + bias[v]`, so `-inf` is a hard exclusion).  A request that does not send the key
// pays nothing: the whole path is behind `req_tnc.valid()`.
//
// The constraint is stateful per request: `update()` feeds the committed reply tokens as the
// decode loop commits them (the loop already passes `consumed[reply0..]`, which only grows by
// accepted tokens) and returns the mask for the next token.  Scanning is O(1) amortized per
// token (a bounded rolling tail + a trie walk), so an enabled-but-idle constraint is as cheap
// as the static logit_bias it rides.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

class ToolNameConstraint {
public:
    struct Node {
        std::unordered_map<int32_t, std::unique_ptr<Node>> next;
        bool terminal = false;   // a complete "name + terminator" sequence ends here
    };

    // --- config (immutable after parse) ---
    std::vector<int32_t> trigger;       // token ids of "<function="
    std::vector<int32_t> think_open;    // token ids of "<think>"  (empty: no reasoning block)
    std::vector<int32_t> think_close;   // token ids of "</think>"
    std::vector<int32_t> call_open;     // token ids of "<tool_call>"
    std::vector<int32_t> call_close;    // token ids of "</tool_call>"
    Node root;                          // trie over the token-id sequences of "name + >"
    int max_seq = 0;                    // the longest name sequence, in tokens
    bool initial_reasoning = false;     // the generation prompt opens <think> (thinking enabled)

    // --- state (mutable, per request) ---
    int64_t scanned = 0;                // reply tokens already fed to update()
    bool in_reasoning = false;          // inside <think> (or started there)
    bool in_call = false;               // inside an unmatched <tool_call>
    const Node* at = nullptr;           // the active name-trie node while a name is being written
    std::vector<int32_t> buf;           // rolling tail for tag detection
    size_t cap = 0;                     // the longest tag sequence, in tokens

    bool valid() const { return !trigger.empty() && !call_open.empty() && !call_close.empty() && !root.next.empty(); }

    void reset_state() {
        scanned = 0;
        in_reasoning = initial_reasoning;
        in_call = false;
        at = nullptr;
        buf.clear();
        cap = std::max({trigger.size(), think_open.size(), think_close.size(), call_open.size(), call_close.size()});
    }

    /// Feed the reply committed so far (`tokens[0..n)`, generation order, the most recent token
    /// last) and return the dense mask the sampler should apply for the next token: `vocab`
    /// floats, `0` for the allowed continuation ids and `-inf` everywhere else.  Empty when the
    /// constraint is not active (not inside a call's name right now).
    std::vector<float> update(const int32_t* tokens, int64_t n, int vocab) {
        if (!valid() || vocab <= 0) return {};
        if (n < scanned) reset_state();          // the reply was rewound (defensive): re-derive from scratch
        for (int64_t i = scanned; i < n; ++i) feed(tokens[i]);
        scanned = n;
        if (at == nullptr) return {};
        std::vector<float> m((size_t) vocab, -std::numeric_limits<float>::infinity());
        for (const auto& [id, child] : at->next) m[(size_t) id] = 0.0f;
        return m;
    }

private:
    void append(int32_t t) {
        buf.push_back(t);
        if (buf.size() > cap) buf.erase(buf.begin());
    }

    bool ends_with(const std::vector<int32_t>& seq) const {
        if (seq.empty() || buf.size() < seq.size()) return false;
        return std::equal(seq.begin(), seq.end(), buf.end() - (int64_t) seq.size());
    }

    void feed(int32_t t) {
        append(t);
        if (at != nullptr) {                     // inside a tool name: the token continues (or ends) it
            auto it = at->next.find(t);
            at = (it == at->next.end()) ? nullptr : it->second.get();
            if (at != nullptr && at->terminal) at = nullptr;   // "name>" complete -> release
            return;
        }
        if (ends_with(think_open))  in_reasoning = true;
        if (ends_with(think_close)) in_reasoning = false;
        if (ends_with(call_open))   in_call = true;
        if (ends_with(call_close))  in_call = false;
        if (in_call && !in_reasoning && ends_with(trigger)) at = &root;   // begin a tool name
    }
};

/// Parse the wire format:
///
///     tool_name_constraint=<initial>|<trigger>|<think_open>|<think_close>|<call_open>|<call_close>|<name+term>|...
///
/// `<initial>` is `0` or `1`: whether the generation prompt opens the `<think>` block (thinking
/// enabled).  `<trigger>` is the comma-separated token ids of `<function=`.  The four markers
/// are the comma-separated token ids of `<think>`, `</think>`, `<tool_call>` and `</tool_call>`
/// (the reasoning pair may be empty to disable reasoning tracking; a model with no reasoning
/// block sends `0` for `<initial>` and empty think markers).  Each following `|`-separated record
/// is the comma-separated token ids of a declared tool name with its `>` terminator appended.
/// Returns false and sets `error` on malformed input, ids outside [0, vocab), or fewer than one
/// name.
inline bool parse_tool_name_constraint(const std::string& text, int vocab,
                                       ToolNameConstraint& out, std::string& error) {
    out = ToolNameConstraint{};
    if (text.empty() || vocab <= 0) { error = "empty tool_name_constraint or invalid vocabulary"; return false; }
    auto parse_ids = [&](const std::string& s, std::vector<int32_t>& ids) -> bool {
        ids.clear();
        if (s.empty()) return false;
        size_t p = 0;
        for (;;) {
            const size_t c = s.find(',', p);
            const std::string one = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (one.empty()) return false;
            char* e = nullptr;
            const unsigned long v = std::strtoul(one.c_str(), &e, 10);
            if (e == one.c_str() || *e != '\0' || v >= (unsigned long) vocab) return false;
            ids.push_back((int32_t) v);
            if (c == std::string::npos) break;
            p = c + 1;
        }
        return true;
    };
    std::vector<std::string> recs;
    size_t p = 0;
    for (;;) {
        const size_t c = text.find('|', p);
        recs.push_back(text.substr(p, c == std::string::npos ? std::string::npos : c - p));
        if (c == std::string::npos) break;
        p = c + 1;
    }
    // <initial>|<trigger>|<think_open>|<think_close>|<call_open>|<call_close>|<name...>
    if (recs.size() < 7) { error = "tool_name_constraint needs initial, five markers and at least one name"; return false; }
    if (recs[0] == "0") out.initial_reasoning = false;
    else if (recs[0] == "1") out.initial_reasoning = true;
    else { error = "tool_name_constraint: the initial flag must be 0 or 1"; return false; }
    if (!parse_ids(recs[1], out.trigger))    { error = "tool_name_constraint: invalid trigger token ids"; return false; }
    // the think pair may be empty (a model with no reasoning block): then reasoning tracking is off
    if (!recs[2].empty() && !parse_ids(recs[2], out.think_open))  { error = "tool_name_constraint: invalid think-open token ids"; return false; }
    if (!recs[3].empty() && !parse_ids(recs[3], out.think_close)) { error = "tool_name_constraint: invalid think-close token ids"; return false; }
    if (!parse_ids(recs[4], out.call_open))  { error = "tool_name_constraint: invalid call-open token ids"; return false; }
    if (!parse_ids(recs[5], out.call_close)) { error = "tool_name_constraint: invalid call-close token ids"; return false; }
    if (out.trigger.empty() || out.call_open.empty() || out.call_close.empty()) {
        error = "tool_name_constraint: trigger, call-open and call-close must be non-empty"; return false;
    }
    if (out.think_open.empty() != out.think_close.empty()) {
        error = "tool_name_constraint: the think markers must be both present or both absent"; return false;
    }
    for (size_t i = 6; i < recs.size(); ++i) {
        std::vector<int32_t> seq;
        if (!parse_ids(recs[i], seq)) { error = "tool_name_constraint: invalid name token ids"; return false; }
        ToolNameConstraint::Node* node = &out.root;
        for (int32_t id : seq)
            node = node->next.emplace(id, std::make_unique<ToolNameConstraint::Node>()).first->second.get();
        node->terminal = true;
        out.max_seq = std::max(out.max_seq, (int) seq.size());
    }
    out.reset_state();
    return true;
}

}  // namespace strata::core
