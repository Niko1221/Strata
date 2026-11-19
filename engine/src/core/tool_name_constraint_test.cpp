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
#define check(ok) do { if (!(ok)) { std::fprintf(stderr, "tool_name_constraint regression at line %d\n", __LINE__); std::exit(1); } } while (0)
    // Token ids: <function= = 1, <think> = 2, </think> = 3, <tool_call> = 4, </tool_call> = 5,
    // names [10,100] and [11,100] and [12,101] (name token + ">" terminator).
    const char* CFG = "v2|1|2|3|4|5|#1|10,100|11,100|12,101";
    // Two forms: <function= = 1 (terminator ">" = 100) and <invoke name=" = 6,7 (terminator "\">" = 102),
    // both with the names 10 and 11.
    const char* CFG2 = "v2|0|2|3|4|5|#1|10,100|11,100|#6,7|10,102|11,102";

    // ---- parse: valid and malformed ----
    strata::core::ToolNameConstraint c0;
    std::string e0;
    check(strata::core::parse_tool_name_constraint(CFG, 103, c0, e0));
    check(c0.valid() && c0.forms.size() == 1 && c0.forms[0].trigger == std::vector<int32_t>{1} && c0.max_seq == 2 &&
          c0.initial_reasoning);
    check(strata::core::parse_tool_name_constraint("v2|0|||4|5|#1|10,100", 103, c0, e0));   // no reasoning block
    check(c0.valid() && !c0.initial_reasoning && c0.think_open.empty() && c0.think_close.empty());
    check(strata::core::parse_tool_name_constraint(CFG2, 103, c0, e0));
    check(c0.forms.size() == 2 && c0.forms[1].trigger == (std::vector<int32_t>{6, 7}));
    for (const char* bad : {"", "v2|1|2|3|4|5", "v2|1|2|3|4|5|#1", "v2|1|2|3|4|5|#1|", "v2|2|2|3|4|5|#1|10",
                            "v2|1|2|3|4|5|10", "v2|1|2|3|4|5|#|10", "v2|1|2|3|4|5|#1|10,", "v2|1|2|3|4|5|#1|x",
                            "v2|1|2|3|4|5|#1|200", "v2|1|2||4|5|#1|10", "v2|1||3|4|5|#1|10",
                            "v2|1|2|3|4|5|#1|#6|10", "v2|1|2|3|4|5|#1|10|#6",
                            "1|1|2|3|4|5|10,100",                       // the v1 format: refused, not misread
                            "v3|1|2|3|4|5|#1|10"}) {
        strata::core::ToolNameConstraint cb;
        std::string eb;
        check(!strata::core::parse_tool_name_constraint(bad, 103, cb, eb));
    }
    check(!strata::core::parse_tool_name_constraint(CFG, 0, c0, e0));   // invalid vocab

    // ---- reasoning blocks the trigger ----
    {
        Sim s(CFG, 103);
        check(s.feed({4}).empty());                                 // <tool_call> while reasoning
        check(s.feed({1}).empty());                                 // <function= while reasoning: not constrained
        check(s.feed({3}).empty());                                 // </think> ends the reasoning
        check(mask_allows(s.feed({1}), {10, 11, 12}));              // now inside the call: constrain
        check(mask_allows(s.feed({10}), {100}));                    // first name token: only ">" next
        check(s.feed({100}).empty());                               // ">" completes the name: released
    }

    // ---- "talking about" tools: <function= outside <tool_call> is plain text ----
    {
        Sim s(CFG, 103);
        check(s.feed({3}).empty());                                 // </think>
        check(s.feed({1}).empty());                                 // <function= with no envelope
        check(s.feed({10}).empty());                                // ... and its name stays text too
    }

    // ---- thinking disabled (initial 0): no reasoning gate, but still needs the envelope ----
    {
        Sim s("v2|0|2|3|4|5|#1|10,100", 103);
        check(s.feed({1}).empty());                                 // <function= with no <tool_call>
        check(mask_allows(s.feed({4, 1}), {10}));                   // <tool_call><function= constrains
        check(s.feed({99}).empty());                                // off-path token releases (defensive)
    }

    // ---- shared prefix between two names ----
    {
        Sim s("v2|0|2|3|4|5|#1|10,100|10,101", 103);                // [10,100] and [10,101] share "10"
        check(mask_allows(s.feed({4, 1}), {10}));
        check(mask_allows(s.feed({10}), {100, 101}));
        check(s.feed({100}).empty());
    }

    // ---- a new <tool_call> re-arms after a released name; </tool_call> disarms ----
    {
        Sim s("v2|0|2|3|4|5|#1|10,100", 103);
        check(mask_allows(s.feed({4, 1}), {10}));
        check(s.feed({10, 100}).empty());                           // full name+">"
        check(mask_allows(s.feed({5, 4, 1}), {10}));                // </tool_call><tool_call><function= re-arms
        check(s.feed({10, 100}).empty());                           // the second name completes
        check(s.feed({5}).empty());                                 // </tool_call> disarms
        check(s.feed({1}).empty());                                 // <function= outside an envelope: text
    }

    // ---- multi-token trigger ----
    {
        Sim s("v2|0|30|31|4|5|#1,2|10,100", 103);                   // trigger = [1,2]
        check(s.feed({4, 1}).empty());                              // partial trigger not yet complete
        check(mask_allows(s.feed({2}), {10}));                      // completes
    }

    // ---- the invoke form: its own trigger and terminator ----
    {
        Sim s(CFG2, 103);
        check(s.feed({4, 6}).empty());                              // half of <invoke name="
        check(mask_allows(s.feed({7}), {10, 11}));                  // <invoke name=" complete: only declared names
        check(mask_allows(s.feed({11}), {102}));                    // then only its "\"> terminator, not ">"
        check(s.feed({102}).empty());                               // released
        check(mask_allows(s.feed({6, 7}), {10, 11}));               // a second <invoke name=" in the same envelope re-arms
    }
    {
        // the #6022 shape: <tool_call><function=function_calls> then <invoke name="NAME"> in the same envelope
        Sim s("v2|0|2|3|4|5|#1|10,100|13,100|#6,7|10,102", 103);   // 13 = the reserved "function_calls"
        check(mask_allows(s.feed({4, 1}), {10, 13}));
        check(s.feed({13, 100}).empty());                           // <function=function_calls> released
        check(mask_allows(s.feed({6, 7}), {10}));                   // the invoke inside it is masked to the declared name
        check(s.feed({10, 102}).empty());
        check(s.feed({5}).empty());
        check(s.feed({6, 7}).empty());                              // <invoke name=" outside an envelope: text
    }
    {
        // the invoke trigger is not armed inside the reasoning either
        Sim s("v2|1|2|3|4|5|#1|10,100|#6,7|10,102", 103);
        check(s.feed({4, 6, 7}).empty());
        check(s.feed({3, 4, 6}).empty());
        check(mask_allows(s.feed({7}), {10}));
    }

    // ---- window_masks: per-row masks equal sequential single-token update() ----
    // A multi-token verify window feeds its drafts through a COPY of the scan state, so each row's mask is the
    // mask that row actually needs (row t constrains the token after window[t] = the next draft).
    {
        strata::core::ToolNameConstraint c;
        std::string e;
        check(strata::core::parse_tool_name_constraint(CFG, 103, c, e));
        std::vector<int32_t> consumed;
        auto step = [&](std::initializer_list<int32_t> ts) {
            for (int32_t t : ts) consumed.push_back(t);
            const int32_t head = consumed.back();                   // the decode loop's head = last produced token
            return c.update(consumed.data(), (int64_t) (consumed.size() - 1), head, 103);
        };
        step({4});                                                  // <tool_call> (reasoning)
        step({3});                                                  // </think> ends reasoning
        step({1});                                                  // <function= arms the trie
        auto head_mask = step({10});                                // head 10 = first name token -> at = node(10)
        check(mask_allows(head_mask, {100}));                       // only ">" (100) follows "10"

        std::vector<int32_t> window = {10, 100};                    // row 1 after ">" releases the name
        auto masks = c.window_masks(window.data(), 2, 103, head_mask);
        check(masks.size() == 2 && masks[0] == head_mask);
        check(masks[1].empty());

        std::vector<int32_t> w2 = {10, 999 % 103};                  // a hallucinated continuation: off-path, released
        auto m2 = c.window_masks(w2.data(), 2, 103, head_mask);
        check(m2.size() == 2 && m2[0] == head_mask);
        check(m2[1].empty());
    }
    {
        // The trigger's own mask constrains the FIRST name token across a multi-token window.
        strata::core::ToolNameConstraint c;
        std::string e;
        check(strata::core::parse_tool_name_constraint(CFG, 103, c, e));
        std::vector<int32_t> consumed;
        auto step = [&](std::initializer_list<int32_t> ts) {
            for (int32_t t : ts) consumed.push_back(t);
            const int32_t head = consumed.back();
            return c.update(consumed.data(), (int64_t) (consumed.size() - 1), head, 103);
        };
        step({4});
        step({3});
        auto hm = step({1});                                        // head <function= -> at = root
        check(mask_allows(hm, {10, 11, 12}));

        std::vector<int32_t> w3 = {1, 10, 100};                     // head <function=, drafts 10 ("name"), 100 (">")
        auto m3 = c.window_masks(w3.data(), 3, 103, hm);
        check(m3.size() == 3 && m3[0] == hm);
        check(mask_allows(m3[1], {100}));
        check(m3[2].empty());
    }
    {
        // A drafted <invoke name="NAME"> inside one window: each row gets the invoke form's mask.
        strata::core::ToolNameConstraint c;
        std::string e;
        check(strata::core::parse_tool_name_constraint(CFG2, 103, c, e));
        std::vector<int32_t> consumed = {4};
        auto hm = c.update(consumed.data(), 0, 4, 103);             // head <tool_call>
        check(hm.empty());
        std::vector<int32_t> w = {4, 6, 7, 11, 102, 50};
        auto m = c.window_masks(w.data(), 6, 103, hm);
        check(m[1].empty());                                        // after "<invoke": trigger not complete
        check(mask_allows(m[2], {10, 11}));                         // after `name="`: the names
        check(mask_allows(m[3], {102}));                            // after the name: its "\"> terminator
        check(m[4].empty() && m[5].empty());                        // released: the parameters are free
        check(c.st.at == nullptr && c.st.buf.back() == 4);          // the canonical state was not touched
    }

    // ---- a --batch slot: ids hold the prompt and the tokens its windows fed (never the head x), each window calls update() +
    // window_masks() over ids[from..) with its own state; a rejected draft leaves the state at what was kept ----
    {
        strata::core::ToolNameConstraint c;
        std::string e;
        check(strata::core::parse_tool_name_constraint("v2|0|2|3|4|5|#1|10,100|11,100|12,101", 103, c, e));   // thinking off
        std::vector<int32_t> ids = {50, 51};                        // the prompt (thinking off here: no <think> token), the reply starts at 2
        const int64_t from = 2;
        auto window = [&](std::vector<int32_t> w) {                 // w[0] = the slot's x, the rest its drafts
            auto head = c.update(ids.data() + from, (int64_t) ids.size() - from, w[0], 103);
            return c.window_masks(w.data(), (int) w.size(), 103, head);
        };
        // window 1: x = <tool_call>, nothing drafted; kept
        auto m1 = window({4});
        check(m1.size() == 1 && m1[0].empty());
        ids.push_back(4);
        // window 2: x = <function=, drafts 77 (a name no tool has) and ">"; the target picks 10 under row 0's mask, so only x is kept
        auto m2 = window({1, 77, 100});
        check(m2.size() == 3 && mask_allows(m2[0], {10, 11, 12}));
        check(m2[1].empty());                                       // 77 is off the trie: the trie releases, later rows are free
        ids.push_back(1);                                           // kept: x only
        // window 3: x = 10 (the target's pick), drafts ">" (100): the head is inside the name, only ">" follows
        auto m3 = window({10, 100});
        check(m3.size() == 2 && mask_allows(m3[0], {100}));
        check(m3[1].empty());
        ids.push_back(10);
        ids.push_back(100);                                         // both kept
        // window 4: x = the next token after the name: released, nothing masked
        auto m4 = window({60});
        check(m4.size() == 1 && m4[0].empty());
        ids.push_back(60);
        // a second call in the same reply arms again
        auto m5 = window({4});
        check(m5[0].empty());
        ids.push_back(4);
        auto m6 = window({1});
        check(mask_allows(m6[0], {10, 11, 12}));
    }
    {
        // a slot that reads its prompt first (ids shorter than from): nothing is fed, nothing is masked
        strata::core::ToolNameConstraint c;
        std::string e;
        check(strata::core::parse_tool_name_constraint(CFG, 103, c, e));
        std::vector<int32_t> ids = {50};
        const int64_t from = std::min<int64_t>(5, (int64_t) ids.size());   // as batch_tnc clamps it
        auto head = c.update(ids.data() + from, (int64_t) ids.size() - from, 4, 103);
        check(head.empty());
    }

    std::puts("tool_name_constraint_test: passed");
    return 0;
}
