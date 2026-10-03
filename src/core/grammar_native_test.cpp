#include "strata/core/grammar.hpp"
#include "strata/core/grammar_budget.hpp"
#include <picojson.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace strata::grammar;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string("failed: ") + #x); } while (false)

static std::string read(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read native grammar fixture: " + path.string());
    return {std::istreambuf_iterator<char>(in), {}};
}

template<class Fn> void fails(Fn fn, const std::string& message = "") {
    try { fn(); }
    catch (const std::exception& error) {
        REQUIRE(std::string(error.what()).find(message) != std::string::npos);
        return;
    }
    throw std::runtime_error("expected a grammar failure: " + message);
}

static void corpus(const std::filesystem::path& cases, const std::shared_ptr<const Vocabulary>& vocab) {
    picojson::value spec;
    REQUIRE(picojson::parse(spec, read(cases)).empty());
    Compiler compiler(vocab);
    std::unordered_map<std::string, int32_t> byte_id;
    for (size_t id = 0; id < vocab->bytes.size(); ++id)
        if (vocab->bytes[id].size() == 1) byte_id[vocab->bytes[id]] = (int32_t) id;
    REQUIRE(byte_id.size() == 256);
    size_t checks = 0;
    for (const auto& value : spec.get<picojson::array>()) {
        const auto& item = value.get<picojson::object>();
        const std::string filename = item.at("file").get<std::string>();
        auto compiled = compiler.compile(read(cases.parent_path() / filename));
        for (const std::string key : {"accept", "reject"}) {
            for (const auto& example : item.at(key).get<picojson::array>()) {
                Matcher matcher(compiled);
                bool accepted = true;
                for (char c : example.get<std::string>()) {
                    if (!matcher.accept(byte_id.at(std::string(1, c)))) { accepted = false; break; }
                }
                accepted = accepted && matcher.complete();
                REQUIRE(accepted == (key == "accept"));
                if (accepted) {
                    for (int32_t stop : vocab->stop_ids) {
                        auto ended = matcher.fork();
                        REQUIRE(ended.allows(stop) && ended.accept(stop) && ended.terminated());
                    }
                    REQUIRE(matcher.accept(vocab->stop_ids[0]));
                    REQUIRE(matcher.terminated());
                    fails([&] { matcher.mask(); }, "terminal");
                }
                ++checks;
            }
        }
        Matcher initial(compiled);
        for (size_t id = 0; id < vocab->bytes.size(); ++id)
            if (vocab->bytes[id].empty() &&
                std::find(vocab->stop_ids.begin(), vocab->stop_ids.end(), (int32_t) id) == vocab->stop_ids.end())
                REQUIRE(!initial.allows((int32_t) id));
        std::cout << "corpus " << filename << " passed\n";
    }
    std::cout << "corpus cases=" << checks << " vocabulary=" << vocab->bytes.size()
              << " identity=" << vocab->identity << "\n";
}

static void unit(const std::shared_ptr<const Vocabulary>& vocab) {
    Compiler c(vocab, 2, 64 * 1024 * 1024);
    const auto overlap = c.compile("root ::= \"a\" | \"ab\"");
    Matcher parent(overlap);
    REQUIRE(!parent.allows(256) && !parent.allows(257)); // stop and special
    REQUIRE(!parent.accept('x') && parent.tokens().empty());
    REQUIRE(parent.accept('a') && parent.complete());
    REQUIRE(parent.allows('b') && parent.allows(256));
    const auto snapshot = parent.checkpoint();
    auto left = parent.fork(), right = parent.fork();
    REQUIRE(left.accept('b') && left.accept(256));
    REQUIRE(right.accept(256));
    REQUIRE(parent.tokens().size() == 1 && !parent.terminated());
    left.restore(snapshot);
    REQUIRE(left.tokens() == parent.tokens() && !left.terminated());
    Matcher wrong(c.compile("root ::= \"z\""));
    REQUIRE(parent.identity() != wrong.identity());
    fails([&] { wrong.restore(snapshot); }, "identity mismatch");
    auto different_bytes = vocab->bytes;
    different_bytes['a'] = "q";
    Compiler other(Vocabulary::from_bytes(std::move(different_bytes), {256}));
    Matcher mismatch(other.compile("root ::= \"a\" | \"ab\""));
    fails([&] { mismatch.restore(snapshot); }, "identity mismatch");
    REQUIRE(c.compile("root ::= \"a\" | \"ab\"") == overlap);
    c.compile("root ::= \"b\"");
    c.compile("root ::= \"c\"");
    REQUIRE(c.cache_entries() == 2 && c.cache_bytes() <= 64 * 1024 * 1024);
    REQUIRE(parent.accept('b')); // eviction cannot mutate an active sequence
    REQUIRE(parent.complete());
    Matcher multi(c.compile("root ::= \"a\" \"b\""));
    REQUIRE(multi.accept(259) && multi.complete());
    Matcher utf(c.compile("root ::= \"caf\xc3\xa9\""));
    REQUIRE(utf.accept('c') && utf.accept('a') && utf.accept('f'));
    REQUIRE(utf.accept(0xc3) && !utf.complete());
    REQUIRE(utf.allows(0xa9) && !utf.allows('x'));
    REQUIRE(utf.accept(0xa9) && utf.complete());
    Matcher loop(c.compile("root ::= \"(\" root \")\" root | \"\""));
    for (int i = 0; i < 32; ++i) REQUIRE(loop.accept('('));
    for (int i = 0; i < 32; ++i) REQUIRE(loop.accept(')'));
    REQUIRE(loop.complete());
    Matcher epsilon(c.compile("root ::= \"\""));
    REQUIRE(epsilon.complete() && epsilon.allows(256) && !epsilon.allows('a'));
    Matcher ambiguous(c.compile("root ::= root root | \"a\""));
    for (int i = 0; i < 24; ++i) REQUIRE(ambiguous.accept('a') && ambiguous.complete());

    fails([&] { c.compile("root ::= root"); });
    fails([&] { c.compile("root ::= missing"); });
    fails([&] { c.compile("root ::= \"unterminated"); });
    fails([&] { c.compile("root ::= )[a]("); }, "closing group");
    fails([&] { c.compile("root ::= \"a\"{100000000}"); }, "unsupported");
    fails([&] { c.compile("root[temperature=0.1] ::= \"a\""); }, "attributes");
    fails([&] { c.compile(std::string(kMaxSourceBytes + 1, 'a')); }, "8192");
    fails([&] { c.compile(std::string("root ::= \"a\"\0", 13)); }, "NUL");
    REQUIRE(!valid_utf8("\xc0\x80") && !valid_utf8("\xed\xa0\x80") && !valid_utf8("\xf4\x90\x80\x80"));
    REQUIRE(valid_utf8("\xe7\x8c\xab"));
    Compiler no_cache(vocab, 0, 0);
    fails([&] { no_cache.compile("root ::= \"bounded\"", 0); }, "work budget");
    fails([&] { no_cache.compile("root ::= \"" + std::string(512, 'x') + "\"", 100); }, "work budget");
    const auto bounded = no_cache.compile("root ::= [ab]*");
    Matcher limited(bounded, 48);
    fails([&] { for (int i = 0; i < 100; ++i) REQUIRE(limited.accept('a')); }, "work budget");
    REQUIRE(limited.failed());
    fails([&] { limited.mask(); }, "discard");
    // A failed operation does not poison the compiler or a new sequence.
    Matcher fresh(no_cache.compile("root ::= \"ok\""));
    REQUIRE(fresh.accept('o') && fresh.accept('k') && fresh.complete());
    Compiler impossible(Vocabulary::from_bytes({"a", ""}, {1}));
    fails([&] { impossible.compile("root ::= \"z\""); }, "no tokenizer-realizable continuation");
    detail::WorkBudget deadline{10};
    detail::WorkScope scope(deadline, std::chrono::milliseconds(100));
    deadline.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    fails([&] { detail::work(); }, "deadline");
    std::cout << "native masks, UTF-8 fragments, recursion, accepting prefixes, stop/special IDs, forks, "
                 "checkpoints, cache isolation, syntax and resource failures passed\n";
}

static void dump_vocab(const Vocabulary& vocab, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) out.put((char) (v >> (i * 8))); };
    out.write("SVOC1", 5); u32((uint32_t) vocab.bytes.size());
    for (const auto& token : vocab.bytes) { u32((uint32_t) token.size()); out.write(token.data(), token.size()); }
    if (!out) throw std::runtime_error("vocabulary audit write failed");
}

static void scoped_unit() {
    // Synthetic protocol IDs, with ordinary one-byte text tokens. The actual
    // native matcher and speculative-prefix routine enforce every assertion.
    std::vector<std::string> bytes(260);
    for (int i = 0; i < 256; ++i) bytes[i] = std::string(1, (char) i);
    constexpr int stop = 256, think_end = 257, call_start = 258, call_end = 259;
    auto vocab = Vocabulary::from_bytes(std::move(bytes), {stop}, {think_end, call_start, call_end});
    Compiler compiler(vocab);
    auto compiled = compiler.compile("root ::= \"OK\"");
    Matcher m(compiled, 2000000, {true, true});
    REQUIRE(m.allows('x') && !m.allows(stop) && !m.allows(call_start));
    REQUIRE(m.accept('x') && std::string(m.channel()) == "reasoning");
    const auto saved = m.checkpoint();
    const int32_t drafts[] = {think_end, call_start, 'x', call_end, 'O', 'K', stop};
    PrefixMasks prefix;
    m.prefix_masks(drafts, 7, prefix);
    REQUIRE(prefix.rows == 7 && prefix.end_draft && prefix.blocked_draft == -1);
    REQUIRE(m.tokens() == saved.tokens && std::string(m.phase()) == "reasoning");
    REQUIRE(m.accept(think_end) && std::string(m.channel()) == "control");
    REQUIRE(!m.allows('x') && !m.allows(stop) && m.allows(call_start));
    REQUIRE(m.accept('\n') && std::string(m.channel()) == "control");
    REQUIRE(m.accept('\n') && !m.allows('\n')); // formatting cannot consume the entire output budget
    auto after_separator = m.fork();
    REQUIRE(!after_separator.allows('\n') && after_separator.allows('O'));
    REQUIRE(m.accept(call_start) && std::string(m.channel()) == "tool");
    for (char c : std::string("arbitrary parameters: purple 999, not OK")) REQUIRE(m.accept(c));
    REQUIRE(!m.allows(stop) && !m.allows(think_end));
    REQUIRE(m.accept(call_end) && m.complete());
    auto tools_only = m.fork();
    REQUIRE(tools_only.accept(stop) && tools_only.terminated());
    REQUIRE(m.accept('\n') && std::string(m.channel()) == "control");
    REQUIRE(m.accept(call_start) && m.accept('2') && m.accept(call_end));
    REQUIRE(m.accept('O') && std::string(m.channel()) == "answer");
    REQUIRE(!m.allows('x') && !m.allows(call_start) && !m.allows(stop));
    REQUIRE(m.accept('K') && m.complete() && m.accept(stop) && m.terminated());
    m.restore(saved);
    REQUIRE(m.tokens() == saved.tokens && std::string(m.phase()) == "reasoning");
    Matcher raw(compiled);
    fails([&] { raw.restore(saved); }, "identity mismatch");
    REQUIRE(!raw.allows(call_start) && !raw.allows('x')); // no state leak from shared compilation

    Matcher reasoning(compiler.compile("root ::= \"\\n<think>literal</think>\""), 2000000, {true, false});
    REQUIRE(reasoning.accept(think_end));
    for (char c : std::string("\n<think>literal</think>")) {
        REQUIRE(reasoning.accept(c));
        REQUIRE(std::string(reasoning.channel()) == "answer");
    }
    REQUIRE(reasoning.complete() && reasoning.accept(stop));
    Matcher leading(compiler.compile("root ::= \"\\n\\n\\nOK\""), 2000000, {true, false});
    REQUIRE(leading.accept(think_end));
    for (char c : std::string("\n\n\nOK"))
        REQUIRE(leading.accept(c) && std::string(leading.channel()) == "answer");
    REQUIRE(leading.complete()); // answer newlines are not envelope separators
    Matcher empty(compiler.compile("root ::= \"\""), 2000000, {false, true});
    REQUIRE(empty.complete() && empty.accept(stop));
    Matcher limited(compiled, 64, {true, true});
    fails([&] { for (int i = 0; i < 100; ++i) limited.accept('x'); }, "work budget");
    REQUIRE(limited.failed());
    std::cout << "scoped grammar: reasoning, parallel tools, answer bytes, EOS, speculative phase crossings, "
                 "checkpoints, budgets and isolation passed\n";
}

static void json_unit() {
    std::vector<std::string> bytes;
    for (int i = 0; i < 256; ++i) bytes.push_back(std::string(1, (char)i));
    bytes.insert(bytes.end(), {"", "", "", ""});
    auto vocab = Vocabulary::from_bytes(std::move(bytes), {256}, {257, 258, 259});
    Compiler compiler(vocab);
    const std::string schema = R"({"type":"object","properties":{"title":{"type":"string","minLength":1,"maxLength":36}},"required":["title"],"additionalProperties":false})";
    auto compiled = compiler.compile(schema, 5000000, true);
    REQUIRE(compiler.compile(schema, 5000000, true) == compiled);
    fails([&] { compiler.compile(schema); }); // distinct source formats/cache admission
    for (const auto& example : std::vector<std::pair<std::string, bool>>{
            {R"({"title":"Fix addition"})", true}, {R"({"title":""})", false},
            {R"({"title":123})", false}, {R"({"wrong":"x"})", false},
            {R"({"title":"0123456789012345678901234567890123456"})", false},
            {R"({"title":"x","extra":0})", false}}) {
        Matcher m(compiled);
        bool accepted = true;
        for (unsigned char c : example.first) if (!m.accept(c)) { accepted = false; break; }
        REQUIRE((accepted && m.complete()) == example.second);
    }
    Matcher budget(compiled, 2000000, {true, true, 2});
    REQUIRE(budget.accept('a'));
    const auto saved = budget.checkpoint();
    const int32_t drafts[] = {'b', 'c'};
    PrefixMasks masks;
    budget.prefix_masks(drafts, 2, masks);
    REQUIRE(masks.blocked_draft == 1 && budget.tokens() == saved.tokens);
    REQUIRE(budget.accept('b') && !budget.allows('c') && budget.allows(257));
    auto forked = budget.fork();
    REQUIRE(!forked.allows('c') && forked.accept(257));
    REQUIRE(budget.accept(257) && budget.accept(258));
    REQUIRE(budget.accept('x') && budget.accept(259)); // tool text stays unconstrained
    for (unsigned char c : std::string(R"({"title":"After tool"})")) REQUIRE(budget.accept(c));
    REQUIRE(budget.complete() && budget.accept(256));
    budget.restore(saved);
    REQUIRE(budget.accept('b') && !budget.allows('c'));
    Matcher different(compiled, 2000000, {true, true, 3});
    fails([&] { different.restore(saved); }, "identity mismatch");
    auto object = compiler.compile(R"({"type":"object"})", 5000000, true);
    Matcher arbitrary(object);
    for (unsigned char c : std::string(R"({"nested":[true,null,1.5,"ok"]})")) REQUIRE(arbitrary.accept(c));
    REQUIRE(arbitrary.complete());
    // A pattern does not forbid nonmatching keys when additionalProperties is
    // allowed. The pinned backend previously blocked "name" after "count".
    const std::string dynamic_schema = R"({"type":"object","propertyNames":{"pattern":"^[a-z]+$"},"patternProperties":{"^count":{"type":"integer"}},"additionalProperties":{"type":"string"},"minProperties":1,"maxProperties":3})";
    auto dynamic = compiler.compile(dynamic_schema, 5000000, true);
    for (const auto& example : std::vector<std::pair<std::string, bool>>{
            {R"({"count":3,"name":"tests"})", true},
            {R"({"name":"tests","count":3})", true},
            {R"({"name":"tests"})", true}, {R"({"count":3})", true},
            {R"({"name":true})", false}}) {
        Matcher m(dynamic);
        bool accepted = true;
        for (unsigned char c : example.first) if (!m.accept(c)) { accepted = false; break; }
        REQUIRE((accepted && m.complete()) == example.second);
    }
    auto closed_dynamic = compiler.compile(R"({"type":"object","patternProperties":{"^count":{"type":"integer"}},"additionalProperties":false})", 5000000, true);
    Matcher closed(closed_dynamic);
    bool extra_allowed = true;
    for (unsigned char c : std::string(R"({"name":"tests"})"))
        if (!closed.accept(c)) { extra_allowed = false; break; }
    REQUIRE(!extra_allowed);
    std::cout << "JSON schema: title bounds/types/keys, JSON object, native thinking budget, tools, speculation, checkpoints passed\n";
}

static void json_pack(const std::shared_ptr<const Vocabulary>& vocab) {
    Compiler compiler(vocab);
    const std::string schema = R"({"type":"object","properties":{"title":{"type":"string","minLength":1,"maxLength":36}},"required":["title"],"additionalProperties":false})";
    const auto begin = std::chrono::steady_clock::now();
    auto compiled = compiler.compile(schema, Compiler::kJsonWorkLimit, true);
    std::unordered_map<std::string, int32_t> byte_id;
    for (size_t id = 0; id < vocab->bytes.size(); ++id)
        if (vocab->bytes[id].size() == 1) byte_id[vocab->bytes[id]] = (int32_t)id;
    for (const auto& sample : std::vector<std::pair<std::string, bool>>{
            {R"({"title":"Fix addition"})", true}, {R"({"title":""})", false},
            {R"({"title":"012345678901234567890123456789012345"})", true},
            {R"({"title":"0123456789012345678901234567890123456"})", false}}) {
        Matcher matcher(compiled, Matcher::kJsonWorkLimit);
        bool accepted = true;
        for (char c : sample.first)
            if (!matcher.accept(byte_id.at(std::string(1, c)))) { accepted = false; break; }
        REQUIRE((accepted && matcher.complete()) == sample.second);
    }
    std::cout << "JSON title with actual vocabulary: " << vocab->bytes.size() << " tokens, "
              << compiler.cache_bytes() << " compiled bytes, "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
              << " seconds; required/nonempty/length constraints passed\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4) throw std::runtime_error("usage: grammar_native_test CASES [TOKENIZER_DIR VOCAB_AUDIT]");
        std::vector<std::string> bytes;
        for (int i = 0; i < 256; ++i) bytes.push_back(std::string(1, (char) i));
        bytes.insert(bytes.end(), {"", "", "true", "ab", "()"});
        auto vocab = Vocabulary::from_bytes(std::move(bytes), {256});
        corpus(argv[1], vocab);
        unit(vocab);
        scoped_unit();
        json_unit();
        if (argc == 4) {
            auto actual = Vocabulary::from_pack(argv[2], {248044, 248046});
            corpus(argv[1], actual);
            json_pack(actual);
            dump_vocab(*actual, argv[3]);
        }
        std::cout << "grammar native tests passed (" << kBackend << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "grammar native test failed: " << error.what() << '\n';
        return 1;
    }
}
