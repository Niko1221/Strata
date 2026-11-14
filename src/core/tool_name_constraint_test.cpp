#include "strata/core/tool_name_constraint.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool is_neg_inf(float v) { return std::isinf(v) && v < 0; }

// A mask that allows exactly the ids in `allowed`: non-empty, 0 for those ids, -inf elsewhere.
static bool mask_allows(const std::vector<float>& m, const std::vector<int32_t>& allowed) {
    if (m.empty()) return false;
    std::vector<bool> ok(m.size(), false);
    for (int32_t id : allowed) { if (id < 0 || id >= (int32_t) m.size()) return false; ok[(size_t) id] = true; }
    for (size_t v = 0; v < m.size(); ++v) {
        if (ok[v] ? m[v] != 0.0f : !is_neg_inf(m[v])) return false;
    }
    return true;
}

// Drives the stateful constraint one token at a time (simulating the decode loop's committed
// reply growing monotonically).
struct Sim {
    strata::core::ToolNameConstraint c;
    std::vector<int32_t> reply;
    int vocab;
    Sim(const char* cfg, int v) : vocab(v) {
        std::string e;
        if (!strata::core::parse_tool_name_constraint(cfg, v, c, e)) {
            std::fprintf(stderr, "tool_name_constraint parse: %s\n", e.c_str());
            std::exit(1);
        }
    }
    std::vector<float> feed(std::initializer_list<int32_t> ts) {
        // Mirrors the decode loop: everything but the last token is the committed prefix, the
        // last token is the head the sampler continues from, and update() returns the mask for
        // the token AFTER that head.
        for (int32_t t : ts) reply.push_back(t);
        const int32_t head = reply.back();
        return c.update(reply.data(), (int64_t) (reply.size() - 1), head, vocab);
    }
};

int main() {
    auto check = [](bool ok) { if (!ok) { std::fprintf(stderr, "tool_name_constraint regression\n"); std::exit(1); } };
    // Token ids: trigger <function= = 1, <think> = 2, </think> = 3, <tool_call> = 4, </tool_call> = 5,
    // names [10,100] and [11,100] and [12,101] (name token + ">" terminator).
    const char* CFG = "1|1|2|3|4|5|10,100|11,100|12,101";

    // ---- parse: valid and malformed ----
    strata::core::ToolNameConstraint c0;
    std::string e0;
    check(strata::core::parse_tool_name_constraint(CFG, 102, c0, e0));
    check(c0.valid() && c0.trigger == std::vector<int32_t>{1} && c0.max_seq == 2 && c0.initial_reasoning);
    check(strata::core::parse_tool_name_constraint("0|1|||4|5|10,100", 102, c0, e0));   // no reasoning block
    check(c0.valid() && !c0.initial_reasoning && c0.think_open.empty() && c0.think_close.empty());
    for (const char* bad : {"", "1|1|2|3|4|5", "1|1|2|3|4|5|", "2|1|2|3|4|5|10", "1||2|3|4|5|10",
                            "1|1|2|3|4|5|10,", "1|1|2|3|4|5|,10", "1|1|2|3|4|5|x", "1|1|2|3|4|5|200",
                            "1|1|2||4|5|10", "1|1||3|4|5|10"}) {
        strata::core::ToolNameConstraint cb;
        std::string eb;
        check(!strata::core::parse_tool_name_constraint(bad, 102, cb, eb));
    }
    check(!strata::core::parse_tool_name_constraint(CFG, 0, c0, e0));   // invalid vocab

    // ---- reasoning blocks the trigger ----
    {
        Sim s(CFG, 102);
        check(s.feed({4}).empty());                                 // <tool_call> while reasoning
        check(s.feed({1}).empty());                                 // <function= while reasoning: not constrained
        check(s.feed({3}).empty());                                 // </think> ends the reasoning
        check(mask_allows(s.feed({1}), {10, 11, 12}));              // now inside the call: constrain
        check(mask_allows(s.feed({10}), {100}));                    // first name token: only ">" next
        check(s.feed({100}).empty());                               // ">" completes the name: released
    }

    // ---- "talking about" tools: <function= outside <tool_call> is plain text ----
    {
        Sim s(CFG, 102);
        check(s.feed({3}).empty());                                 // </think>
        check(s.feed({1}).empty());                                 // <function= with no envelope
        check(s.feed({10}).empty());                                // ... and its name stays text too
    }

    // ---- thinking disabled (initial 0): no reasoning gate, but still needs the envelope ----
    {
        Sim s("0|1|2|3|4|5|10,100", 102);
        check(s.feed({1}).empty());                                 // <function= with no <tool_call>
        check(mask_allows(s.feed({4, 1}), {10}));                   // <tool_call><function= constrains
        check(s.feed({99}).empty());                                // off-path token releases (defensive)
    }

    // ---- shared prefix between two names ----
    {
        Sim s("0|1|2|3|4|5|10,100|10,101", 102);                    // [10,100] and [10,101] share "10"
        check(mask_allows(s.feed({4, 1}), {10}));
        check(mask_allows(s.feed({10}), {100, 101}));
        check(s.feed({100}).empty());
    }

    // ---- a new <tool_call> re-arms after a released name; </tool_call> disarms ----
    {
        Sim s("0|1|2|3|4|5|10,100", 102);
        check(mask_allows(s.feed({4, 1}), {10}));
        check(s.feed({10, 100}).empty());                           // full name+">"
        check(mask_allows(s.feed({5, 4, 1}), {10}));                // </tool_call><tool_call><function= re-arms
        check(s.feed({10, 100}).empty());                           // the second name completes
        check(s.feed({5}).empty());                                 // </tool_call> disarms
        check(s.feed({1}).empty());                                 // <function= outside an envelope: text
    }

    // ---- multi-token trigger ----
    {
        Sim s("0|1,2|30|31|4|5|10,100", 102);                       // trigger = [1,2]
        check(s.feed({4, 1}).empty());                              // partial trigger not yet complete
        check(mask_allows(s.feed({2}), {10}));                      // completes
    }

    // ---- window_masks: per-row masks equal sequential single-token update() ----
    // A multi-token verify window feeds its drafts through a COPY of the scan state, so each row's mask is the
    // mask that row actually needs (row t constrains the token after window[t] = the next draft).
    {
        strata::core::ToolNameConstraint c;
        std::string e;
        check(strata::core::parse_tool_name_constraint(CFG, 102, c, e));
        std::vector<int32_t> consumed;
        auto step = [&](std::initializer_list<int32_t> ts) {
            for (int32_t t : ts) consumed.push_back(t);
            const int32_t head = consumed.back();                   // the decode loop's head = last produced token
            return c.update(consumed.data(), (int64_t) (consumed.size() - 1), head, 102);
        };
        step({4});                                                  // <tool_call> (reasoning)
        step({3});                                                  // </think> ends reasoning
        step({1});                                                  // <function= arms the trie
        auto head_mask = step({10});                                // head 10 = first name token -> at = node(10)
        check(mask_allows(head_mask, {100}));                       // only ">" (100) follows "10"

        // window = [10, 100]: row 1 after ">" releases the name
        std::vector<int32_t> window = {10, 100};
        auto masks = c.window_masks(window.data(), 2, 102, head_mask);
        check(masks.size() == 2 && masks[0] == head_mask);
        check(masks[1].empty());                                    // ">" completes the name

        // a hallucinated continuation: 999 is not ">" -> the row is off-path (released), so the verifier's
        // unconstrained argmax will not equal 999 and the draft is rejected by the acceptance loop
        std::vector<int32_t> w2 = {10, 999};
        auto m2 = c.window_masks(w2.data(), 2, 102, head_mask);
        check(m2.size() == 2 && m2[0] == head_mask);
        check(m2[1].empty());
    }
    {
        // The trigger's own mask constrains the FIRST name token across a multi-token window.
        strata::core::ToolNameConstraint c;
        std::string e;
        check(strata::core::parse_tool_name_constraint(CFG, 102, c, e));
        std::vector<int32_t> consumed;
        auto step = [&](std::initializer_list<int32_t> ts) {
            for (int32_t t : ts) consumed.push_back(t);
            const int32_t head = consumed.back();
            return c.update(consumed.data(), (int64_t) (consumed.size() - 1), head, 102);
        };
        step({4});                                                  // <tool_call> (reasoning)
        step({3});                                                  // </think>
        auto hm = step({1});                                        // head <function= -> at = root
        check(mask_allows(hm, {10, 11, 12}));                       // first name token constrained

        std::vector<int32_t> w3 = {1, 10, 100};                     // head <function=, drafts 10 ("name"), 100 (">")
        auto m3 = c.window_masks(w3.data(), 3, 102, hm);
        check(m3.size() == 3 && m3[0] == hm);
        check(mask_allows(m3[1], {100}));                           // after "10": only ">"
        check(m3[2].empty());                                       // after ">": released

        std::vector<int32_t> w4 = {1, 999};                         // hallucinated first name token
        auto m4 = c.window_masks(w4.data(), 2, 102, hm);
        check(m4.size() == 2 && m4[0] == hm);
        check(m4[1].empty());                                       // 999 off-path -> released
    }

    std::puts("tool_name_constraint_test: passed");
    return 0;
}
