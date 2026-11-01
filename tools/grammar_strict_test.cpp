// tools/grammar_strict_test.cpp - the strict tool-value grammars, fed to the real engine parser.
//
// The Python side (serve/grammar_tools.py) guards the envelope; the converter guards a parameter
// value. Both must agree with each other AND with the reply parser (serve/frontend.py's param_end),
// which ends a value at any "</parameter>" that a protocol marker follows. A grammar is only useful
// if it refuses that exact text - including the overlap case "<<" ... where a naive exclusion chain
// restarts at zero and lets the second '<' begin a fresh attempt that then completes.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "strata/grammar/grammar.hpp"

#include "json-schema-to-grammar.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::grammar;

static const char * CLOSER = "</parameter>";

// feed the piece code point by code point; the grammar accepts it iff some stack reaches its end
static bool accepts(const std::string & gbnf, const std::string & text) {
    grammar * g = grammar_init(nullptr, gbnf.c_str(), "root");
    if (g == nullptr) {
        std::fprintf(stderr, "grammar_init failed for:\n%s\n", gbnf.c_str());
        std::exit(1);
    }
    for (unsigned char c : text) {  // the corpus is ASCII: one byte, one code point
        grammar_accept(g, (uint32_t) c);
    }
    bool any_empty = false;
    for (const grammar_stack & stack : grammar_stacks(g)) {
        if (stack.empty()) {
            any_empty = true;
            break;
        }
    }
    grammar_free(g);
    return any_empty;
}

static int failures = 0;

static void expect(const std::string & gbnf, const std::string & text, bool wanted, const char * what) {
    const bool got = accepts(gbnf, text);
    if (got != wanted) {
        std::fprintf(stderr, "FAIL %s: %s %s\n", what, wanted ? "should accept" : "should refuse",
                     gbnf_format_literal(text).c_str());
        ++failures;
    }
}

static common_json parse(const char * json_text) {
    try {
        return common_json::parse(json_text);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test schema does not parse: %s\n", e.what());
        std::exit(1);
    }
}

static std::string strict(const char * schema_text) {
    common_grammar_options options;
    options.tool_value = true;
    options.forbidden_sequence = CLOSER;
    return json_schema_to_grammar(parse(schema_text), true, options);
}

// a root string parameter value is raw template text: no quotes, and never the closer
static void test_raw_root_string(void) {
    const std::string gbnf = strict(R"({"type":"string"})");
    expect(gbnf, "", true, "empty value");
    expect(gbnf, "hello", true, "ordinary text");
    expect(gbnf, "a < b", true, "a lone '<' is ordinary text");
    expect(gbnf, "safe </parameter>", false, "the whole closer");
    expect(gbnf, "safe <</parameter>", false, "the closer after an overlap prefix");
    expect(gbnf, "safe <<</parameter>", false, "the closer after a longer overlap prefix");
    expect(gbnf, "</param", true, "a proper prefix of the closer");
    expect(gbnf, "safe </parameter", true, "the closer without its last character");
    expect(gbnf, "safe </parameter> tail", false, "the closer mid-value");
}

// the template writes string parameters raw, so a raw root schema's own bounds ride the same guard
static void test_raw_root_bounds(void) {
    const std::string three = strict(R"({"type":"string","maxLength":3})");
    expect(three, "", true, "bounded empty");
    expect(three, "abc", true, "bounded at the limit");
    expect(three, "abcd", false, "bounded over the limit");
    const std::string at_least = strict(R"({"type":"string","minLength":2})");
    expect(at_least, "a", false, "below the floor");
    expect(at_least, "ab", true, "at the floor");
    expect(at_least, "abcdef", true, "unbounded above the floor");
    expect(at_least, "</parameter>", false, "the closer is refused whatever the length");
}

// a non-string parameter is serialized JSON by the template: quoted, and no RAW '<' may start a closer
static void test_nested_json_string(void) {
    const std::string gbnf = strict(
        R"({"type":"object","properties":{"t":{"type":"string"}},"required":["t"],"additionalProperties":false})");
    expect(gbnf, R"({"t":"hello"})", true, "nested ordinary text");
    expect(gbnf, R"({"t":"a < b"})", false, "nested raw '<' is refused");
    expect(gbnf, R"({"t":"safe </parameter>"})", false, "nested raw closer");
    // the logical '<' survives as a JSON escape: valid JSON, decoded by the client to the same text
    expect(gbnf, R"({"t":"\u003c/parameter>"})", true, "nested closer written as an escape");
    expect(gbnf, R"({"t":"\u003c"})", true, "nested escaped '<'");
    expect(gbnf, R"({"t":"\u003c/parameter>"})", true, "nested escaped '<' before the closer text");
    // the escape branch spells exactly one decoded character, so bounds count characters not bytes
    const std::string bounded = strict(
        R"({"type":"object","properties":{"t":{"type":"string","maxLength":1}},"required":["t"],"additionalProperties":false})");
    expect(bounded, R"({"t":"\u003c"})", true, "one escaped character inside a one-character bound");
    expect(bounded, R"({"t":"\u003c\u003c"})", false, "two escaped characters over a one-character bound");
    expect(bounded, R"({"t":"ab"})", false, "two characters over a one-character bound");
    expect(bounded, R"({"t":"\ud83d\ude00"})", false, "a surrogate pair is not one grammar character");
}

// Client schemas use a huge maxLength to mean "a lot" (65535, 1000000).  The delimiter guarantee holds in every
// state, so such a bound is dropped rather than refused: refusing it failed the whole request.  A minLength the
// chain cannot count still refuses, because dropping it would let a value come out shorter than the schema says.
static void test_large_length_bounds(void) {
    for (const char * bound : {"512", "513", "40000", "1000000"}) {
        const std::string schema = std::string(R"({"type":"string","maxLength":)") + bound + "}";
        bool threw = false;
        std::string gbnf;
        try {
            gbnf = strict(schema.c_str());
        } catch (const std::exception &) {
            threw = true;
        }
        if (threw) {
            std::fprintf(stderr, "FAIL a large maxLength was refused: %s\n", schema.c_str());
            ++failures;
            continue;
        }
        expect(gbnf, "plain text", true, (std::string("text under maxLength ") + bound).c_str());
        expect(gbnf, "safe </parameter>", false, (std::string("closer under maxLength ") + bound).c_str());
        expect(gbnf, "safe <</parameter>", false, (std::string("overlapping closer under maxLength ") + bound).c_str());
    }
    const std::string ranged = strict(R"({"type":"string","minLength":10,"maxLength":40000})");
    expect(ranged, "0123456789", true, "a value at the minLength of a widened bound");
    expect(ranged, "012345678", false, "one characters short of the minLength");
    // the threshold itself: llama.cpp clamps an upper bound there and refuses a lower one above it
    // (src/llama-grammar.cpp MAX_REPETITION_THRESHOLD), and this converter follows those rules
    const std::string at_limit = strict(R"({"type":"string","minLength":2000})");
    expect(at_limit, std::string(2000, 'a'), true, "a value exactly at a 2000-character minLength");
    expect(at_limit, std::string(1999, 'a'), false, "a value one short of a 2000-character minLength");
    bool refused = false;
    try {
        strict(R"({"type":"string","minLength":2001})");
    } catch (const std::exception &) {
        refused = true;
    }
    if (!refused) {
        std::fprintf(stderr, "FAIL a minLength past the repetition threshold was accepted\n");
        ++failures;
    }
}

// a fixed literal spells '<' as a JSON escape inside serialized JSON, so the closer can never be written
// raw there; a RAW root literal that would BE the closer is refused outright rather than emitted
static void test_literals_and_enums(void) {
    const std::string root_literal = strict(R"({"const":"a<b"})");
    expect(root_literal, "a<b", true, "a raw literal with a lone '<'");
    expect(root_literal, "a", false, "outside the literal");

    const std::string nested_literal = strict(
        R"({"type":"object","properties":{"k":{"const":"a<b"}},"required":["k"],"additionalProperties":false})");
    expect(nested_literal, R"({"k":"a\u003cb"})", true, "a nested literal closer-adjacent '<' written escaped");
    expect(nested_literal, R"({"k":"a<b"})", false, "a nested literal '<' written raw");

    const std::string enum_rule = strict(R"({"enum":["x","y"]})");
    expect(enum_rule, "x", true, "raw enum member");
    expect(enum_rule, "z", false, "outside the enum");
}

// anything the converter would otherwise widen must be refused here, not silently unconstrained
static void test_fail_closed(void) {
    const char * refused[] = {
        R"({"$ref":"#/definitions/x"})",
        R"({"type":"string","pattern":"^a$"})",
        R"({"type":"string","format":"date"})",
        R"({"anyOf":[{"type":"string"},{"type":"integer"}]})",
        R"({"type":"object","additionalProperties":true})",
        R"({})",
        R"({"type":"object","properties":{"a":{}},"additionalProperties":false})",
        R"({"type":"object","properties":{"a":{"type":"array","items":{}}},"additionalProperties":false})",
        R"({"type":"integer","minimum":10,"maximum":5})",
        R"({"type":"string","minLength":3,"maxLength":2})",
        R"({"enum":["a",1]})",
        R"({"type":"integer","enum":[1],"minLength":2})",
        R"({"const":"a</parameter>b"})",
    };
    for (const char * schema : refused) {
        bool threw = false;
        try {
            strict(schema);
        } catch (const std::exception &) {
            threw = true;
        }
        if (!threw) {
            std::fprintf(stderr, "FAIL strict mode accepted a schema it cannot honour: %s\n", schema);
            ++failures;
        }
    }
}

int main(void) {
    test_raw_root_string();
    test_raw_root_bounds();
    test_nested_json_string();
    test_large_length_bounds();
    test_literals_and_enums();
    test_fail_closed();
    if (failures != 0) {
        std::fprintf(stderr, "%d strict grammar test(s) failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "All strict grammar tests passed.\n");
    return 0;
}
