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

// A dictionary (an object with typed additionalProperties and no declared properties) is a shape tool
// schemas use for free-form maps, and one a schema generator writes with `propertyNames: {"type": "string"}`
// plus `properties: {}`.  Every JSON member name is a string, so those two keywords add nothing; the key rule
// stays the string primitive, whose `char` already refuses a raw '<'.
// A property name reaches the rule names as written, and '_' is not a GBNF word character: an unsanitized
// "a_b-guard-state-0" makes the whole grammar fail to parse, so a schema from a snake_case tool (Zod writes
// old_string/new_string everywhere) would 400 with "grammar parse failed".
static void test_underscore_rule_names(void) {
    const std::string gbnf = strict(
        R"({"type":"object","additionalProperties":false,"required":["old_string","new_string"],"properties":{"old_string":{"type":"string","minLength":1},"new_string":{"type":"string"},"file-path":{"type":"string"}}})");
    expect(gbnf, R"({"old_string":"a","new_string":"b"})", true, "a snake_case parameter pair with bounds");
    expect(gbnf, R"({"old_string":"","new_string":"b"})", false, "an empty value where minLength is 1");
    expect(gbnf, R"({"old_string":"a</parameter>","new_string":"b"})", false, "a snake_case value spelling the closer");

    // two names that sanitize to the same rule base must stay distinct rules with distinct bodies
    const std::string twins = strict(
        R"({"type":"object","additionalProperties":false,"required":["a_b","a-b"],"properties":{"a_b":{"type":"string"},"a-b":{"type":"string","minLength":3}}})");
    expect(twins, R"({"a_b":"x","a-b":"xyz"})", true, "a snake_case and a hyphenated name side by side");
    expect(twins, R"({"a_b":"x","a-b":"xy"})", false, "the hyphenated twin keeping its own minLength");
}

static void test_dictionaries(void) {
    const std::string dict = strict(R"({"type":"object","propertyNames":{"type":"string"},"additionalProperties":{"type":"string"}})");
    expect(dict, R"({"a":"b"})", true, "a dictionary with one entry");
    expect(dict, R"({"a":"b","c":"d"})", true, "a dictionary with two entries");
    expect(dict, R"({})", true, "an empty dictionary");
    expect(dict, R"({"a":1})", false, "a value outside the value schema");
    expect(dict, R"({"a":"b</parameter>"})", false, "a dictionary value spelling the closer raw");
    expect(dict, R"({"a":"b\u003c/parameter>"})", true, "a dictionary value carrying the closer escaped");
    expect(dict, R"({"\u003c/parameter>":"b"})", true, "a key spelling '<' through the JSON escape");
    expect(dict, R"({"<<":"b"})", false, "a key with a raw '<'");

    // the generator's other spelling: an empty properties object alongside the dictionary
    const std::string with_empty = strict(
        R"({"type":"object","properties":{},"propertyNames":{"type":"string"},"additionalProperties":{"type":"string"}})");
    expect(with_empty, R"({"a":"b"})", true, "a dictionary written with an empty properties object");

    // propertyNames: true and an empty schema both mean "no constraint" in JSON
    const std::string plain_true = strict(R"({"type":"object","propertyNames":true,"additionalProperties":{"type":"integer"}})");
    expect(plain_true, R"({"n":7})", true, "propertyNames true with typed values");
    expect(plain_true, R"({"n":"7"})", false, "a string value where the schema wants an integer");

    // a typed dictionary of objects: the nested value schema is guarded too
    const std::string nested = strict(
        R"({"type":"object","additionalProperties":{"type":"object","additionalProperties":false,"required":["t"],"properties":{"t":{"type":"string"}}}})");
    expect(nested, R"({"k":{"t":"v"}})", true, "a dictionary of closed objects");
    expect(nested, R"({"k":{"t":"v</parameter>"}})", false, "a nested value spelling the closer raw");
}

// A schema generator writes `record<string, unknown>` for a free-form map and `["string", "null"]` (or an
// anyOf) for a nullable field.  Both are expressible: the `value` primitive spells its strings through
// `char`, which refuses a raw '<', and a union is one alternative per branch.
static void test_free_form_values(void) {
    const std::string unknown_map = strict(
        R"({"type":"object","propertyNames":{"type":"string"},"additionalProperties":{}})");
    expect(unknown_map, R"({"a":"b"})", true, "a record<string, unknown> with a string value");
    expect(unknown_map, R"({"a":{"b":[1,2,{"c":null}]}})", true, "a record<string, unknown> with a nested value");
    expect(unknown_map, R"({"a":"b</parameter>"})", false, "a free-form value spelling the closer raw");
    expect(unknown_map, R"({"a":["</parameter>"]})", false, "the closer raw inside a free-form array");

    const std::string nullable_type = strict(
        R"({"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"type":["string","null"]}}})");
    expect(nullable_type, R"({"a":"x"})", true, "a nullable field holding a string");
    expect(nullable_type, R"({"a":null})", true, "a nullable field holding null");
    expect(nullable_type, R"({"a":1})", false, "a nullable field holding a number");

    const std::string nullable_union = strict(
        R"({"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"anyOf":[{"type":"string"},{"type":"null"}]}}})");
    expect(nullable_union, R"({"a":"x"})", true, "an anyOf-nullable field holding a string");
    expect(nullable_union, R"({"a":null})", true, "an anyOf-nullable field holding null");
    expect(nullable_union, R"({"a":"x</parameter>"})", false, "an anyOf-nullable field spelling the closer raw");
}

// A pattern is a common way to bound a string, and it compiles to character classes.  Excluding '<' from
// those classes narrows what the grammar accepts (never cuts a call short), so an ordinary pattern works
// under a protected value; one that NEEDS '<' has nothing left to accept and is reported instead.
static void test_patterns_and_formats(void) {
    const std::string word = strict(
        R"({"type":"object","additionalProperties":false,"required":["w"],"properties":{"w":{"type":"string","pattern":"^[a-z]+$"}}})");
    expect(word, R"({"w":"abc"})", true, "a lowercase word");
    expect(word, R"({"w":"ABC"})", false, "an uppercase word");
    expect(word, R"({"w":"a<c"})", false, "a '<' inside a narrow class");

    const std::string printable = strict(
        R"({"type":"object","additionalProperties":false,"required":["w"],"properties":{"w":{"type":"string","pattern":"^[!-~]+$"}}})");
    expect(printable, R"({"w":"a!b"})", true, "a printable-only string");
    expect(printable, R"({"w":"a;b"})", true, "the class character just below '<'");
    expect(printable, R"({"w":"a=b"})", true, "the class character just above '<'");
    expect(printable, R"({"w":"a<b"})", false, "a '<' inside a range that spans it");

    const std::string negated = strict(
        R"({"type":"object","additionalProperties":false,"required":["w"],"properties":{"w":{"type":"string","pattern":"^[^abc]+$"}}})");
    expect(negated, R"({"w":"xyz"})", true, "a string outside a negated class");
    expect(negated, R"({"w":"xaz"})", false, "a member of a negated class");
    expect(negated, R"({"w":"x<z"})", false, "a '<' excluded from a negated class");

    const std::string uuid = strict(
        R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","format":"uuid"}}})");
    expect(uuid, R"({"id":"123e4567-e89b-12d3-a456-426614174000"})", true, "a uuid");
    expect(uuid, R"({"id":"123e4567"})", false, "a prefix of a uuid");

    // a pattern that EXCLUDES '<' is safe and must not be refused just for naming it
    const std::string not_lt = strict(
        R"({"type":"object","additionalProperties":false,"required":["w"],"properties":{"w":{"type":"string","pattern":"^[^<]+$"}}})");
    expect(not_lt, R"({"w":"ab"})", true, "a string outside a '<'-excluding class");

    // ...and a negated range that spans '<' keeps the range, since dropping it would admit '<'
    const std::string negated_range = strict(
        R"({"type":"object","additionalProperties":false,"required":["w"],"properties":{"w":{"type":"string","pattern":"^[^ -?]+$"}}})");
    expect(negated_range, R"({"w":"zz"})", true, "a string outside a negated range");
    expect(negated_range, R"({"w":"z<z"})", false, "a '<' inside a negated range that spans it");
    expect(negated_range, R"({"w":"z;z"})", false, "a character of the negated range just below '<'");

    // \d \w \s are expanded into classes, so they work where client schemas use them most
    const std::string digits = strict(
        R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","pattern":"^\\d{4}$"}}})");
    expect(digits, R"({"id":"2026"})", true, "four digits");
    expect(digits, R"({"id":"202a"})", false, "a letter among the digits");
    expect(digits, R"({"id":"26"})", false, "too few digits");

    const std::string filename = strict(
        R"({"type":"object","additionalProperties":false,"required":["f"],"properties":{"f":{"type":"string","pattern":"^[\\w.-]+$"}}})");
    expect(filename, R"({"f":"a_b-c.txt"})", true, "a word-class file name");
    expect(filename, R"({"f":"a b"})", false, "a space outside the class");

    // a negated shorthand becomes a negated class, and '<' is still kept out of it
    const std::string not_digit = strict(
        R"({"type":"object","additionalProperties":false,"required":["w"],"properties":{"w":{"type":"string","pattern":"^\\D+$"}}})");
    expect(not_digit, R"({"w":"abc"})", true, "a non-digit string");
    expect(not_digit, R"({"w":"a1c"})", false, "a digit where \D refuses one");
    expect(not_digit, R"({"w":"a<c"})", false, "a '<' where \D is used");

    // a union is one production per branch, and each branch is framed as a nested value
    const std::string root_union = strict(R"({"anyOf":[{"type":"string"},{"type":"integer"}]})");
    expect(root_union, R"("x")", true, "a root union holding a quoted string");
    expect(root_union, "7", true, "a root union holding an integer");
    expect(root_union, "x", false, "an unquoted string where the union branches are JSON");
}

// anything the converter would otherwise widen must be refused here, not silently unconstrained
static void test_fail_closed(void) {
    const char * refused[] = {
        R"({"$ref":"#/definitions/x"})",
        R"({"type":"object","additionalProperties":true})",
        R"({"anyOf":[{"type":"string"},{"type":"integer"}],"type":"string"})",
        R"({"type":"object","additionalProperties":false,"properties":{"a":{"anyOf":[{"type":"string"},{"type":"integer"}],"patternProperties":{}}}})",
        R"({"type":"object","additionalProperties":false,"properties":{"a":{"anyOf":[]}}})",
        R"({"type":["string",1]})",
        R"({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{"type":"string"}})",
        R"({"type":"object","propertyNames":{"type":"string","pattern":"^[a-z]+$"},"additionalProperties":{"type":"string"}})",
        R"({"type":"object","propertyNames":{"enum":["a","b"]},"additionalProperties":{"type":"string"}})",
        R"({"type":"object","propertyNames":{"type":"string","maxLength":3},"additionalProperties":{"type":"string"}})",
        R"({"type":"object","properties":[],"additionalProperties":{"type":"string"}})",
        R"({})",
        R"({"type":"object","properties":{"a":{}},"additionalProperties":false})",
        R"({"type":"string","pattern":"^<+$"})",
        R"({"type":"string","pattern":"^</parameter>$"})",
        R"({"type":"string","pattern":"^a"})",
        R"({"type":"string","format":"hostname"})",
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
    test_dictionaries();
    test_underscore_rule_names();
    test_free_form_values();
    test_patterns_and_formats();
    test_fail_closed();
    if (failures != 0) {
        std::fprintf(stderr, "%d strict grammar test(s) failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "All strict grammar tests passed.\n");
    return 0;
}
