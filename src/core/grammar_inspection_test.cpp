// Local native debugging/inspection tests, not a production inspector service.
#include "strata/core/grammar.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using namespace strata::grammar;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string("failed: ") + #x); } while (false)

template<class Fn> void fails(Fn fn, const std::string& message) {
    try { fn(); }
    catch (const std::exception& e) { REQUIRE(std::string(e.what()).find(message) != std::string::npos); return; }
    throw std::runtime_error("expected failure: " + message);
}

static void trace(const char* occurrence, const Matcher& matcher, size_t limit = 8) {
    const auto view = matcher.inspect(limit);
    std::cout << "operation=constraint.inspect occurrence=" << occurrence << " definition=" << view.definition
              << " committed_tokens=" << view.committed_tokens << " committed_bytes=" << view.committed_bytes
              << " accepting=" << view.accepting << " terminated=" << view.terminated
              << " legal_tokens=" << view.legal_tokens << " preview_complete=" << view.preview_complete
              << " mask=" << view.mask_fingerprint << "\n";
    for (const auto& token : view.preview) {
        std::ostringstream bytes;
        for (unsigned char ch : token.bytes) bytes << std::hex << std::setfill('0') << std::setw(2) << (int) ch;
        std::cout << "  token=" << token.id << " bytes_hex=" << bytes.str() << " stop=" << token.stop
                  << " bytes_truncated=" << token.bytes_truncated << "\n";
    }
    std::cout << "  hypotheses=backend_union_not_enumerated model_checkpoint=false"
                 " raw_model_probability=not_collected grammar_conditioned_probability=not_collected"
                 " final_sampler_probability=not_collected branch_mass=not_inferred\n";
}

static void inspect(const std::shared_ptr<const Vocabulary>& vocab) {
    std::unordered_map<std::string, int32_t> ids;
    for (size_t i = 0; i < vocab->bytes.size(); ++i)
        if (!vocab->bytes[i].empty()) ids.emplace(vocab->bytes[i], (int32_t) i);
    auto byte = [&](char ch) { return ids.at(std::string(1, ch)); };
    Compiler compiler(vocab);
    const std::string source = "root ::= left | right | alias\nleft ::= \"a\" \"b\"\n"
                               "right ::= \"a\" \"c\"\nalias ::= \"ab\"";
    auto definition = compiler.compile(source);
    Matcher parent(definition);
    const auto start = parent.checkpoint();
    const auto budget_before = parent.work_used();
    auto initial = parent.inspect();
    REQUIRE(!initial.accepting && !initial.terminated && initial.committed_tokens == 0);
    REQUIRE(parent.work_used() == budget_before && parent.tokens().empty());
    REQUIRE(parent.inspect().mask_fingerprint == initial.mask_fingerprint);
    auto zero = parent.inspect(0);
    REQUIRE(zero.preview.empty() && !zero.preview_complete && zero.legal_tokens == initial.legal_tokens);
    fails([&] { parent.inspect(65); }, "64");
    REQUIRE(parent.work_used() == budget_before && !parent.failed());
    trace("parent.initial", parent);

    auto left = parent.fork(), right = parent.fork();
    REQUIRE(left.accept(byte('a')) && !left.complete());
    REQUIRE(left.allows(byte('b')) && left.allows(byte('c')));
    REQUIRE(!left.allows(vocab->stop_ids.front()));
    trace("left.shared-prefix-a", left);
    // The 'a' token is compatible with both alternatives; counting it as the
    // probability mass of either branch would invent a unique branch assignment.
    REQUIRE(left.accept(byte('b')) && left.complete());
    REQUIRE(right.accept(ids.at("ac")) && right.complete());
    REQUIRE(parent.tokens().empty() && parent.inspect().mask_fingerprint == initial.mask_fingerprint);
    REQUIRE(left.tokens().size() == 2 && right.tokens().size() == 1);
    trace("left.ab", left); trace("right.ac", right);
    auto left_checkpoint = left.checkpoint();
    REQUIRE(left.accept(vocab->stop_ids.front()) && left.terminated());
    auto terminal = left.inspect();
    REQUIRE(terminal.accepting && terminal.terminated && terminal.preview.empty() && terminal.legal_tokens == 0);
    trace("left.ended", left);
    left.restore(left_checkpoint);
    REQUIRE(!left.terminated() && left.complete());
    left.restore(start);
    REQUIRE(left.tokens().empty() && left.inspect().mask_fingerprint == initial.mask_fingerprint);
    auto invalid = start; invalid.tokens = {byte('a'), byte('x')};
    fails([&] { left.restore(invalid); }, "invalid grammar checkpoint");
    REQUIRE(left.tokens().empty() && !left.failed());
    trace("left.restored", left);

    Compiler independent(vocab);
    Matcher same_diagnostic(independent.compile(source));
    REQUIRE(same_diagnostic.identity() == parent.identity());
    fails([&] { same_diagnostic.restore(start); }, "identity mismatch");
    auto changed = vocab->bytes; changed[(size_t) byte('a')] = "z";
    Compiler other(Vocabulary::from_bytes(std::move(changed), vocab->stop_ids));
    Matcher mismatch(other.compile(source));
    REQUIRE(mismatch.identity() != parent.identity());
    fails([&] { mismatch.restore(start); }, "identity mismatch");

    Matcher accepting(compiler.compile("root ::= \"a\" | \"ab\""));
    REQUIRE(accepting.accept(byte('a')) && accepting.complete());
    REQUIRE(accepting.allows(byte('b')) && accepting.allows(vocab->stop_ids.front()));
    trace("accepting.with-continuation", accepting);
    Matcher recursive(compiler.compile("root ::= root root | \"a\""));
    for (int i = 0; i < 12; ++i) REQUIRE(recursive.accept(byte('a')) && recursive.complete());
    auto inspected = recursive.inspect();
    REQUIRE(inspected.accepting && !inspected.terminated && recursive.tokens().size() == 12);
    trace("recursive.ambiguous", recursive);
    std::cout << "inspection passed vocabulary=" << vocab->bytes.size() << " identity=" << vocab->identity << "\n";
}

int main(int argc, char** argv) {
    try {
        std::vector<std::string> bytes;
        for (int i = 0; i < 256; ++i) bytes.emplace_back(1, (char) i);
        bytes.insert(bytes.end(), {"", "ab", "ac", std::string(40, 'a')});
        auto toy = Vocabulary::from_bytes(std::move(bytes), {256});
        inspect(toy);
        Compiler compiler(toy);
        Matcher long_token(compiler.compile("root ::= [ab]*"));
        auto bounded = long_token.inspect(64);
        auto found = std::find_if(bounded.preview.begin(), bounded.preview.end(), [](const auto& p) { return p.id == 259; });
        REQUIRE(found != bounded.preview.end() && found->bytes.size() == 32 && found->bytes_truncated);
        if (argc == 2) inspect(Vocabulary::from_pack(argv[1], {248044, 248046}));
        else if (argc != 1) throw std::runtime_error("usage: grammar_inspection_test [trusted-tokenizer-directory]");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
