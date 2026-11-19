#pragma once
// engine/include/strata/core/tool_name_constraint.hpp
//
// Constrained decoding for tool-call names, scoped to a real call envelope.  A model
// that writes `<tool_call><function=NAME>` (or `<tool_call><invoke name="NAME">`) can
// hallucinate a NAME no declared tool has; the parser then drops the call and the raw text
// leaks into the reply.  This constraint removes the possibility at the sampler: only while
// the reply is *inside* a `<tool_call>` envelope (and past the `<think>` reasoning block) does
// a name opener trigger, and only tokens that continue one of the declared names (followed by
// that form's terminator) are sampled; a complete name releases the constraint.
//
// The envelope is what makes this safe to leave on: an opener written anywhere else — in the
// answer's prose, or in the reasoning while the model *talks about* a call ("I will use
// `<function=foo>` …") — is plain text and is never constrained.  The reasoning block
// (`<think>…</think>`) is excluded as well, so the model may freely discuss the call format
// while it thinks.
//
// Several name forms can be armed at once (`<function=NAME>` and `<invoke name="NAME">`), each
// with its own trigger and its own trie; the first trigger the committed tail ends with picks
// the trie.
//
// The engine streams token ids and has no detokenizer, so the server tokenizes the pieces
// and sends ids (see `parse_tool_name_constraint` for the wire format).  The mask is a
// dense `-inf`/`0` vector, exactly what `SamplerParams::logit_bias` takes (sampler.cu applies
// `l[v] + bias[v]`, so `-inf` is a hard exclusion).  A request that does not send the key
// pays nothing: the whole path is behind `req_tnc.valid()`.
//
// The constraint is stateful per request: `update()` feeds the committed reply tokens as the
// decode loop commits them and returns the mask for the next token.  Scanning is O(1) amortized
// per token (a bounded rolling tail + a trie walk).  A multi-token spec window is validated under
// a PER-ROW mask: `window_masks()` walks a copy of the state over the drafted rows, so a drafted
// name no tool declares is not the verifier's pick under its row's mask and the acceptance loop
// stops there; the drafter stays on through the whole reply.
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
    struct Form {
        std::vector<int32_t> trigger;          // token ids of the opener ("<function=", "<invoke name=\"")
        std::unique_ptr<Node> root;            // trie over the token-id sequences of "name + terminator"
    };

    // --- config (immutable after parse) ---
    std::vector<Form> forms;
    std::vector<int32_t> think_open;    // token ids of "<think>"  (empty: no reasoning block)
    std::vector<int32_t> think_close;   // token ids of "</think>"
    std::vector<int32_t> call_open;     // token ids of "<tool_call>"
    std::vector<int32_t> call_close;    // token ids of "</tool_call>"
    int max_seq = 0;                    // the longest name sequence, in tokens
    bool initial_reasoning = false;     // the generation prompt opens <think> (thinking enabled)

    // --- state (mutable, per request) ---
    // The mutable scan state, grouped so `window_masks` can walk a COPY of it over the drafted
    // prefix without touching the canonical state (a rejected draft must leave it at the committed
    // prefix).  `scanned` stays outside: it counts committed reply tokens fed to update().
    struct Snapshot {
        bool in_reasoning = false;      // inside <think> (or started there)
        bool in_call = false;           // inside an unmatched <tool_call>
        const Node* at = nullptr;       // the active name-trie node while a name is being written
        std::vector<int32_t> buf;       // rolling tail for tag detection
        size_t cap = 0;                 // the longest tag sequence, in tokens
    };
    int64_t scanned = 0;                // reply tokens already fed to update()
    Snapshot st;                        // the canonical scan state

    bool valid() const { return !forms.empty() && !call_open.empty() && !call_close.empty(); }

    void reset_state() {
        scanned = 0;
        st.in_reasoning = initial_reasoning;
        st.in_call = false;
        st.at = nullptr;
        st.buf.clear();
        st.cap = std::max({think_open.size(), think_close.size(), call_open.size(), call_close.size()});
        for (const Form& f : forms) st.cap = std::max(st.cap, f.trigger.size());
    }

    /// Feed the reply committed so far (`tokens[0..n)`, generation order, the most recent token
    /// last) plus the fed-back head `head` — the token the sampler is now continuing from — and
    /// return the dense mask for the token AFTER `head`: `vocab` floats, `0` for the allowed
    /// continuation ids and `-inf` everywhere else.  Empty when the constraint is not active
    /// (not inside a call's name right now).
    ///
    /// `head` matters: the decode loop samples the token after `window[0]` (the head), which is
    /// one token AHEAD of `consumed.back()`.  Feeding only `tokens` would make the mask lag the
    /// sampler by one token, so `<function=NAME>` names got their first letter constrained one
    /// step late (`B=ash` instead of `Bash`).  The head is fed here and counted in `scanned`, so
    /// the next call (whose `tokens` then include it) does not feed it twice.
    std::vector<float> update(const int32_t* tokens, int64_t n, int32_t head, int vocab) {
        if (!valid() || vocab <= 0) return {};
        if (n < scanned) reset_state();          // the reply was rewound (defensive): re-derive from scratch
        for (int64_t i = scanned; i < n; ++i) feed(st, tokens[i]);
        scanned = n;
        feed(st, head);                          // the token the sampler continues from (window[0])
        ++scanned;                               // it lands in `tokens` next round: do not refeed
        if (st.at == nullptr) return {};
        std::vector<float> m((size_t) vocab, -std::numeric_limits<float>::infinity());
        for (const auto& [id, child] : st.at->next) m[(size_t) id] = 0.0f;
        return m;
    }

    /// AFTER update() has fed the head (`window[0]`), the masks for a window's rows 0..T-1.
    /// Row 0 is `head_mask` (what update() returned).  Rows 1.. are derived from a COPY of the
    /// canonical state feeding `window[1..t]`, so a rejected draft leaves the state exactly at
    /// the committed prefix and the next update() re-derives it from `consumed` alone.  An empty
    /// mask means "no constraint on that row" (outside a call name).
    std::vector<std::vector<float>> window_masks(const int32_t* window, int T, int vocab,
                                                 const std::vector<float>& head_mask) const {
        std::vector<std::vector<float>> out((size_t) (T > 0 ? T : 0));
        if (T <= 0) return out;
        out[0] = head_mask;
        if (T == 1 || !valid() || vocab <= 0) return out;
        Snapshot s = st;                         // already includes window[0] (the head, fed by update())
        for (int t = 1; t < T; ++t) {
            feed(s, window[t]);
            if (s.at == nullptr) continue;       // not inside a name: the row is unconstrained
            std::vector<float> m((size_t) vocab, -std::numeric_limits<float>::infinity());
            for (const auto& [id, child] : s.at->next) m[(size_t) id] = 0.0f;
            out[(size_t) t] = std::move(m);
        }
        return out;
    }

private:
    static void append(Snapshot& s, int32_t t) {
        s.buf.push_back(t);
        if (s.buf.size() > s.cap) s.buf.erase(s.buf.begin());
    }

    static bool ends_with(const Snapshot& s, const std::vector<int32_t>& seq) {
        if (seq.empty() || s.buf.size() < seq.size()) return false;
        return std::equal(seq.begin(), seq.end(), s.buf.end() - (int64_t) seq.size());
    }

    void feed(Snapshot& s, int32_t t) const {
        append(s, t);
        if (s.at != nullptr) {                     // inside a tool name: the token continues (or ends) it
            auto it = s.at->next.find(t);
            s.at = (it == s.at->next.end()) ? nullptr : it->second.get();
            if (s.at != nullptr && s.at->terminal) s.at = nullptr;   // "name" + terminator complete -> release
            return;
        }
        if (ends_with(s, think_open))  s.in_reasoning = true;
        if (ends_with(s, think_close)) s.in_reasoning = false;
        if (ends_with(s, call_open))   s.in_call = true;
        if (ends_with(s, call_close))  s.in_call = false;
        if (!s.in_call || s.in_reasoning) return;
        for (const Form& f : forms)
            if (ends_with(s, f.trigger)) { s.at = f.root.get(); return; }   // begin a tool name
    }
};

/// Parse the wire format:
///
///     tool_name_constraint=v2|<initial>|<think_open>|<think_close>|<call_open>|<call_close>|#<trigger>|<name+term>|...[|#<trigger>|<name+term>|...]
///
/// `v2` is the format version (any other first record is refused, so a server and engine from
/// different releases fail loudly instead of misreading each other).  `<initial>` is `0` or `1`:
/// whether the generation prompt opens the `<think>` block (thinking enabled).  The four markers
/// are the comma-separated token ids of `<think>`, `</think>`, `<tool_call>` and `</tool_call>`
/// (the reasoning pair may be empty to disable reasoning tracking).  Then one or more name forms:
/// a record starting with `#` is the comma-separated token ids of that form's opener (the trigger),
/// and each following record (up to the next `#`) is the token ids of one declared name with the
/// form's terminator appended.  Returns false and sets `error` on malformed input, ids outside
/// [0, vocab), or a form with no name.
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
    if (recs[0] != "v2") { error = "tool_name_constraint: unsupported format (expected v2)"; return false; }
    // v2|<initial>|<think_open>|<think_close>|<call_open>|<call_close>|#<trigger>|<name...>
    if (recs.size() < 8) { error = "tool_name_constraint needs initial, four markers and at least one form with a name"; return false; }
    if (recs[1] == "0") out.initial_reasoning = false;
    else if (recs[1] == "1") out.initial_reasoning = true;
    else { error = "tool_name_constraint: the initial flag must be 0 or 1"; return false; }
    // the think pair may be empty (a model with no reasoning block): then reasoning tracking is off
    if (!recs[2].empty() && !parse_ids(recs[2], out.think_open))  { error = "tool_name_constraint: invalid think-open token ids"; return false; }
    if (!recs[3].empty() && !parse_ids(recs[3], out.think_close)) { error = "tool_name_constraint: invalid think-close token ids"; return false; }
    if (!parse_ids(recs[4], out.call_open))  { error = "tool_name_constraint: invalid call-open token ids"; return false; }
    if (!parse_ids(recs[5], out.call_close)) { error = "tool_name_constraint: invalid call-close token ids"; return false; }
    if (out.think_open.empty() != out.think_close.empty()) {
        error = "tool_name_constraint: the think markers must be both present or both absent"; return false;
    }
    ToolNameConstraint::Form* form = nullptr;
    size_t form_names = 0;
    for (size_t i = 6; i < recs.size(); ++i) {
        const std::string& r = recs[i];
        if (!r.empty() && r[0] == '#') {
            if (form != nullptr && form_names == 0) { error = "tool_name_constraint: a form without names"; return false; }
            out.forms.push_back(ToolNameConstraint::Form{{}, std::make_unique<ToolNameConstraint::Node>()});
            form = &out.forms.back();
            form_names = 0;
            if (!parse_ids(r.substr(1), form->trigger)) { error = "tool_name_constraint: invalid trigger token ids"; return false; }
            continue;
        }
        if (form == nullptr) { error = "tool_name_constraint: a name before the first #trigger"; return false; }
        std::vector<int32_t> seq;
        if (!parse_ids(r, seq)) { error = "tool_name_constraint: invalid name token ids"; return false; }
        ToolNameConstraint::Node* node = form->root.get();
        for (int32_t id : seq)
            node = node->next.emplace(id, std::make_unique<ToolNameConstraint::Node>()).first->second.get();
        node->terminal = true;
        out.max_seq = std::max(out.max_seq, (int) seq.size());
        ++form_names;
    }
    if (form == nullptr || form_names == 0) { error = "tool_name_constraint: a form without names"; return false; }
    out.reset_state();
    return true;
}

}  // namespace strata::core
