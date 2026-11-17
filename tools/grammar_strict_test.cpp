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
#include <functional>
#include <string>
#include <vector>

using namespace strata::grammar;

static const char * CLOSER = "</parameter>";

// feed the piece code point by code point; the grammar accepts it iff some stack reaches its end
static bool accepts(const std::string & gbnf, const std::string & text, const char * root = "root") {
    grammar * g = grammar_init(nullptr, gbnf.c_str(), root);
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
// The exclusion automaton makes a claim about a language, so it is checked against the language itself:
// every string over an alphabet that can spell both forbidden sequences, up to a length where every
// overlap shape has appeared, must be accepted exactly when it contains neither sequence.  A structural
// bug (a state that forgets its prefix, a completing character that keeps its alternative) shows up here
// as an accepted string that contains the closer, or a refused one that does not.
static void test_exclusion_bruteforce(void) {
    const std::vector<std::string> forbidden = {"</p>", "ab"};
    const std::string gbnf = gbnf_excluding_grammar("x", forbidden);   // its start rule is "x"
    // '<' and '/' can start the closer, 'a' and 'b' the second sequence, 'p' completes one, '-' is noise
    const std::string alphabet = "</pab-";
    const int max_len = 5;
    std::string text;
    long checked = 0;
    long refused = 0;
    std::function<void(size_t)> walk = [&](size_t len) {
        if (!text.empty()) {
            bool contains = false;
            for (const std::string & seq : forbidden) {
                if (text.find(seq) != std::string::npos) {
                    contains = true;
                }
            }
            if (accepts(gbnf, text, "x") == contains) {
                std::fprintf(stderr, "FAIL exclusion: %s %s\n", contains ? "accepted" : "refused",
                             gbnf_format_literal(text).c_str());
                ++failures;
            }
            ++checked;
            if (contains) {
                ++refused;
            }
        }
        if (len == (size_t) max_len) {
            return;
        }
        for (char c : alphabet) {
            text.push_back(c);
            walk(len + 1);
            text.pop_back();
        }
    };
    walk(0);
    // a run that checked nothing (or refused nothing) would pass vacuously
    if (checked < 4000 || refused < 100) {
        std::fprintf(stderr, "FAIL exclusion: %ld strings checked, %ld refused\n", checked, refused);
        ++failures;
    }
    std::printf("exclusion brute force: %ld strings, %ld with a forbidden sequence\n", checked, refused);
}

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

// Zod writes a nullable field as a union, and a `z.union` with `inclusive: false` as `oneOf`; the schema
// generator ZCode uses even RENAMES anyOf to oneOf before sending.  The converter reads both as one union,
// which is exact while the branches are disjoint - that is what a tagged union emits.
static void test_unions_keyword_spellings(void) {
    const std::string one_of = strict(
        R"({"type":"object","additionalProperties":false,"required":["v"],"properties":{"v":{"oneOf":[{"type":"string"},{"type":"integer"}]}}})");
    expect(one_of, R"({"v":"x"})", true, "a oneOf branch holding a string");
    expect(one_of, R"({"v":7})", true, "a oneOf branch holding an integer");
    expect(one_of, R"({"v":true})", false, "a value outside both oneOf branches");

    const std::string nullable = strict(
        R"({"type":"object","additionalProperties":false,"required":["v"],"properties":{"v":{"oneOf":[{"type":"object","additionalProperties":false,"required":["k"],"properties":{"k":{"type":"string"}}},{"type":"null"}]}}})");
    expect(nullable, R"({"v":{"k":"x"}})", true, "the object branch of a nullable oneOf");
    expect(nullable, R"({"v":null})", true, "the null branch of a nullable oneOf");
    expect(nullable, R"({"v":{"k":"x</parameter>"}})", false, "a oneOf branch value spelling the closer");

    // allOf is an intersection: the merged object requires what ANY branch requires, and keeps each
    // branch's optional properties optional
    const std::string all_of = strict(
        R"({"type":"object","additionalProperties":false,"required":["v"],"properties":{"v":{"allOf":[{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]},{"type":"object","properties":{"y":{"type":"integer"}}}]}}})");
    expect(all_of, R"({"v":{"x":"s"}})", true, "an allOf merge with only the required branch property");
    expect(all_of, R"({"v":{"x":"s","y":3}})", true, "an allOf merge with the optional branch property too");
    expect(all_of, R"({"v":{"y":3}})", false, "an allOf merge missing a required branch property");
    expect(all_of, R"({"v":{"x":"s","y":"no"}})", false, "an allOf property outside its own branch's schema");
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

// A constraint this converter cannot spell is dropped and reported, not refused: losing it widens the
// value set and cannot break the delimiter guarantee, which is the only thing strict mode promises.
// Client schemas hit this constantly - Zod writes `maximum` for every `.max()` number - and refusing them
// failed whole turns, where llama.cpp's own conversion drops them silently.
static void test_widened_constraints(void) {
    // the grammar of a width-dropped schema must be the grammar without it, and must still refuse the closer
    const std::string number = strict(R"({"type":"number","minimum":0,"maximum":100,"multipleOf":0.1})");
    expect(number, "50.5", true, "a number inside the dropped bounds");
    expect(number, "1e999", true, "a number outside the dropped bounds - the bound is gone, not narrowed");
    const std::string plain_number = strict(R"({"type":"number"})");
    if (number != plain_number) {
        std::fprintf(stderr, "FAIL a dropped number bound changed the grammar\n");
        ++failures;
    }

    const std::string object = strict(
        R"({"type":"object","minProperties":1,"maxProperties":4,"additionalProperties":false,"required":["a"],"properties":{"a":{"type":"string"}}})");
    expect(object, R"({"a":"x"})", true, "an object under dropped property-count bounds");
    expect(object, R"({"a":"x</parameter>"})", false, "a nested value under dropped bounds still refuses the closer");

    const std::string array = strict(R"({"type":"array","items":{"type":"string"},"uniqueItems":true})");
    expect(array, R"(["a","a"])", true, "a duplicate item where uniqueItems was dropped");
    expect(array, R"(["</parameter>"])", false, "an array item spelling the closer raw");

    // a keyword beside a union, beside $ref, beside allOf, and beside enum is dropped (the converter reads
    // the combinator first), so the request succeeds on the combinator's own terms.  Note the framing: an
    // enum at a parameter root spells bare text (`"a"` in GBNF is the character a), while a union frames each
    // branch as a nested value, so its string branch is quoted JSON - which serve/frontend decodes because a
    // union's declared type is a list, not "string".
    const std::string union_sibling = strict(R"({"anyOf":[{"type":"string"},{"type":"integer"}],"description":"x","minLength":2})");
    expect(union_sibling, R"("a")", true, "a union with a sibling constraint outside its branches");
    expect(union_sibling, "7", true, "the integer branch of a union with a sibling constraint");
    const std::string enum_sibling = strict(R"({"type":"string","enum":["a","b"],"minLength":5})");
    expect(enum_sibling, "a", true, "an enum with a dropped length bound");
    expect(enum_sibling, "c", false, "a value outside an enum whose length bound was dropped");

    // a dictionary beside declared properties: the emitted key rule spells the declared keys and excludes
    // '<' from everything else, so the guarantee holds there too
    const std::string mixed = strict(
        R"({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{"type":"string"}})");
    expect(mixed, R"({"a":"x"})", true, "a declared property with a dictionary beside it");
    expect(mixed, R"({"b":"y"})", true, "an extra key holding the dictionary's value type");
    expect(mixed, R"({"a":"x</parameter>"})", false, "a declared-property value spelling the closer raw");
    expect(mixed, R"({"b":"y</parameter>"})", false, "an extra-key value spelling the closer raw");
    expect(mixed, R"({"<":"y"})", false, "an extra key with a raw '<'");

    const std::string any_value = strict(
        R"({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true})");
    expect(any_value, R"({"b":[1,{"c":null}]})", true, "additionalProperties:true holds any value");
    expect(any_value, R"({"b":"y</parameter>"})", false, "a free value beside a declared property refuses the closer");

    // an unknown format constrains nothing this converter can spell; the string still cannot spell the closer
    const std::string format = strict(R"({"type":"string","format":"hostname"})");
    expect(format, "any-host", true, "a string whose format was dropped");
    expect(format, "safe </parameter>", false, "a value whose format was dropped still refuses the closer");

    // a pattern the converter cannot compile falls back to the guarded string (llama.cpp's warning path), so
    // the value is wider but the closer is still impossible - including a pattern that ASKS for the closer
    for (const char * pattern : {"^a", "^<+$", "^</parameter>$", "[a-z"}) {
        const std::string schema = std::string(R"({"type":"string","pattern":")") + pattern + R"("})";
        bool threw = false;
        std::string gbnf;
        try {
            gbnf = strict(schema.c_str());
        } catch (const std::exception &) {
            threw = true;
        }
        if (threw) {
            std::fprintf(stderr, "FAIL a dropped pattern was refused: %s\n", schema.c_str());
            ++failures;
            continue;
        }
        expect(gbnf, "plain text", true, (std::string("text where the pattern ") + pattern + " was dropped").c_str());
        expect(gbnf, "safe </parameter>", false, (std::string("the closer where the pattern ") + pattern + " was dropped").c_str());
        expect(gbnf, "safe <</parameter>", false, (std::string("the overlapping closer where ") + pattern + " was dropped").c_str());
    }

    // propertyNames with a real constraint is dropped, and the key rule still excludes a raw '<'
    const std::string named = strict(
        R"({"type":"object","propertyNames":{"type":"string","maxLength":3},"additionalProperties":{"type":"integer"}})");
    expect(named, R"({"any":1})", true, "a key over the dropped length bound");
    expect(named, R"({"<":1})", false, "a key with a raw '<' despite the dropped constraint");

    // a tuple IS expressed exactly: this used to be refused for not being an object schema
    const std::string tuple = strict(R"({"type":"array","items":[{"type":"string"},{"type":"integer"}]})");
    expect(tuple, R"(["a",1])", true, "a draft-7 tuple");
    expect(tuple, R"([1,"a"])", false, "a tuple with its items in the wrong order");

    // an enum beside a type is an exact narrowing: a value outside the declared type is unsatisfiable
    const std::string narrowed = strict(R"({"type":"string","enum":["a",1,true]})");
    expect(narrowed, "a", true, "the string branch of a mixed enum");
    expect(narrowed, "1", false, "an integer that the declared string type excludes");

    // an empty schema at a parameter root is "anything", which a generator writes for `z.unknown()`.  The
    // root holds the JSON spelling of the value, so its strings still go through the protected `char`.
    const std::string free_root = strict(R"({})");
    expect(free_root, R"({"a":1})", true, "a free-form object at a parameter root");
    expect(free_root, R"("<")", false, "the closer's first character as an unquoted root value");
    expect(free_root, R"({"a":"safe </parameter>"})", false, "the closer inside a free-form root");
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
    expect(not_digit, R"({"w":"a1c"})", false, "a digit where \\D refuses one");
    expect(not_digit, R"({"w":"a<c"})", false, "a '<' where \\D is used");

    // a union is one production per branch, and each branch is framed as a nested value
    const std::string root_union = strict(R"({"anyOf":[{"type":"string"},{"type":"integer"}]})");
    expect(root_union, R"("x")", true, "a root union holding a quoted string");
    expect(root_union, "7", true, "a root union holding an integer");
    expect(root_union, "x", false, "an unquoted string where the union branches are JSON");
}

// anything the converter would otherwise widen must be refused here, not silently unconstrained
static void test_refs(void) {
    // a $defs target is validated through the reference: what the grammar holds is the target's own node,
    // so a strict target keeps the guarantee and a target strict mode refuses is refused here too
    const std::string nested = strict(
        R"({"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"$ref":"#/$defs/s"}},"$defs":{"s":{"type":"string","minLength":2}}})");
    expect(nested, R"({"a":"xy"})", true, "a $ref to a nested string, at its minimum length");
    expect(nested, R"({"a":"x"})", false, "a $ref'd string one character short");
    expect(nested, R"({"a":"xy</parameter>"})", false, "a $ref'd string spelling the closer");

    // A parameter root holds raw text, so a root $ref must build its target under the root name - aliasing
    // the target's nested rule would wrap the object in quotes.
    const std::string root_ref = strict(
        R"({"$ref":"#/$defs/o","$defs":{"o":{"type":"object","additionalProperties":false,"required":["n"],"properties":{"n":{"type":"integer"}}}}})");
    expect(root_ref, R"({"n":7})", true, "a root $ref holding an object");
    expect(root_ref, R"("{\"n\":7}")", false, "a root $ref spelled as a quoted JSON string");

    // a self-referential $defs entry: the builder shares one rule, so the grammar terminates and stays
    // as strict as its non-recursive part
    const std::string recursive = strict(
        R"({"$ref":"#/$defs/node","$defs":{"node":{"type":"object","additionalProperties":false,"required":["v"],"properties":{"v":{"type":"string"},"child":{"$ref":"#/$defs/node"}}}}})");
    expect(recursive, R"({"v":"a"})", true, "a recursive node without a child");
    expect(recursive, R"({"v":"a","child":{"v":"b"}})", true, "a recursive node with one child");
    expect(recursive, R"({"v":"a","child":{"v":"b","child":{"v":"c"}}})", true, "two levels of recursion");
    expect(recursive, R"({"child":{"v":"b"}})", false, "a recursive node missing its required field");
    expect(recursive, R"({"v":"a","child":{"v":"b</parameter>"}})", false, "a closer inside a nested child");
}

static void test_fail_closed(void) {
    // Refused: the converter would fail on these, or produce a grammar that REFUSES a value the schema
    // allows (a broken call, not a wider one).  Nothing here is a droppable constraint.
    const char * refused[] = {
        R"({"$ref":"#/definitions/x"})",                       // unresolvable: no such definition
        R"({"$ref":"https://example.com/s.json"})",            // only local references into this document
        // a $defs target strict mode refuses, reached through a reference
        R"({"type":"object","additionalProperties":false,"properties":{"a":{"anyOf":[]}}})",
        R"({"type":["string",1]})",
        R"({"type":"object","properties":[],"additionalProperties":{"type":"string"}})",
        R"({"type":"string","minLength":3,"maxLength":2})",
        R"({"enum":["a",1]})",
        R"({"const":"a</parameter>b"})",
        R"({"oneOf":[]})",
        // an allOf merge that cannot be exact: the converter emits the same key twice, or demands a
        // property that another branch forbids - either way the grammar refuses a legal object
        R"({"type":"object","additionalProperties":false,"properties":{"a":{"allOf":[{"type":"object","properties":{"x":{"type":"string"}}},{"type":"object","properties":{"x":{"type":"integer"}}}]}}})",
        R"({"type":"object","additionalProperties":false,"properties":{"a":{"allOf":[{"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":false},{"type":"object","properties":{"y":{"type":"string"}}}]}}})",
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

    // Accepted with a warning: every one of these constrains the VALUE only, so dropping it widens the
    // language and cannot break the delimiter guarantee.  Refusing them failed whole turns in clients that
    // write such schemas by default (Zod emits `maximum` for every `.max()` number).
    const char * widened[] = {
        R"({"type":"object","additionalProperties":true})",
        R"({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{"type":"string"}})",
        R"({"type":"string","format":"hostname"})",
        R"({"type":"integer","enum":[1],"minLength":2})",
        R"({"$ref":"#/$defs/s","maxLength":4,"$defs":{"s":{"type":"string"}}})",
        R"({"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"$ref":"#/$defs/d"}},"$defs":{"d":{"type":"number","minimum":1}}})",
        R"({"anyOf":[{"type":"string"},{"type":"integer"}],"type":"string"})",
        R"({"oneOf":[{"type":"string"}],"pattern":"^a$"})",
        R"({"type":"object","additionalProperties":false,"properties":{"a":{"anyOf":[{"type":"string"},{"type":"integer"}],"patternProperties":{}}}})",
        R"({"type":"object","additionalProperties":false,"properties":{"a":{"allOf":[{"type":"string"}]}}})",
        // the converter's own warn-and-guard fallback: the pattern is dropped and the guarded string takes
        // its place, which still cannot spell the closer - a weaker value constraint, not a broken call
        R"({"type":"string","pattern":"^a"})",
        R"({"type":"string","pattern":"^<+$"})",
        R"({"type":"string","pattern":"^</parameter>$"})",
        R"({"type":"object","propertyNames":{"type":"string","pattern":"^[a-z]+$"},"additionalProperties":{"type":"string"}})",
        R"({"type":"object","propertyNames":{"enum":["a","b"]},"additionalProperties":{"type":"string"}})",
        R"({"type":"object","propertyNames":{"type":"string","maxLength":3},"additionalProperties":{"type":"string"}})",
        R"({"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"$ref":"#/$defs/p"}},"$defs":{"p":{"type":"string","pattern":"^x<y$"}}})",
        R"({"type":"integer","minimum":10,"maximum":5})",
        // a nested empty schema is "anything", spelled by `value` whose every string goes through `char`
        R"({"type":"object","properties":{"a":{}},"additionalProperties":false})",
    };
    for (const char * schema : widened) {
        try {
            strict(schema);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "FAIL strict mode refused a droppable constraint: %s\n  %s\n", schema, e.what());
            ++failures;
        }
    }
}

int main(void) {
    test_exclusion_bruteforce();
    test_raw_root_string();
    test_raw_root_bounds();
    test_nested_json_string();
    test_large_length_bounds();
    test_literals_and_enums();
    test_dictionaries();
    test_unions_keyword_spellings();
    test_underscore_rule_names();
    test_free_form_values();
    test_widened_constraints();
    test_patterns_and_formats();
    test_refs();
    test_fail_closed();
    if (failures != 0) {
        std::fprintf(stderr, "%d strict grammar test(s) failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "All strict grammar tests passed.\n");
    return 0;
}
