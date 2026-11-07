// grammar_schema: read a JSON Schema on stdin, write the equivalent GBNF grammar (root rule "root") on stdout.
// Built from the vendored llama.cpp converter (vendor/json_schema_to_grammar/, MIT).  Exits 1 with a single
// stderr line when the input is not valid JSON or the schema cannot be converted.  force_gbnf=true so the
// converter never emits an %llguidance grammar (Strata's engine has no llguidance backend).
//
// --ac-grammar is the second, independent mode: it reads {"prefix": "...", "forbidden": ["..."]} and writes
// the Aho-Corasick exclusion grammar ported from llama.cpp.  The server uses it instead of building the
// same automaton in Python, so one implementation serves every caller.
#include "json-schema-to-grammar.h"
#include "json.h"

#include <iostream>
#include <string>

static std::string read_stdin() {
    std::string text;
    char buf[65536];
    while (std::cin.read(buf, sizeof buf)) text.append(buf, std::cin.gcount());
    text.append(buf, std::cin.gcount());
    return text;
}

int main(int argc, char ** argv) {
    common_grammar_options options;
    bool ac_mode = false;
    bool with_document = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tool-value") {
            options.tool_value = true;
        } else if (arg == "--forbid-sequence" && i + 1 < argc) {
            options.forbidden_sequence = argv[++i];
        } else if (arg == "--ac-grammar") {
            ac_mode = true;
        } else if (arg == "--with-document") {
            with_document = true;
        } else {
            std::cerr << "grammar_schema: unknown argument: " << arg << "\n";
            return 1;
        }
    }
    const std::string text = read_stdin();

    common_json input;
    try {
        input = common_json::parse(text);
    } catch (const std::exception & e) {
        std::cerr << "grammar_schema: invalid JSON: " << e.what() << "\n";
        return 1;
    }

    // --with-document: {"schema": <the node to spell>, "document": <what $ref resolves against>}.  A JSON
    // Pointer is rooted at the whole document, so a caller that compiles one parameter at a time has to hand
    // over the schema the client sent - otherwise `#/properties/x` cannot reach a sibling.
    common_json document;
    common_json sub_schema;
    const common_json * schema = &input;
    if (with_document) {
        if (!input.is_object() || !input.contains("schema") || !input.contains("document")) {
            std::cerr << "grammar_schema: --with-document needs {\"schema\": ..., \"document\": ...}\n";
            return 1;
        }
        document   = input.at("document");
        sub_schema = input.at("schema");
        options.ref_root = &document;
        schema = &sub_schema;
    }

    try {
        if (ac_mode) {
            if (!schema->contains("prefix") || !schema->contains("forbidden")) {
                std::cerr << "grammar_schema: --ac-grammar needs {\"prefix\": \"...\", \"forbidden\": [...]}\n";
                return 1;
            }
            std::string gbnf = gbnf_excluding_grammar(schema->at("prefix").get<std::string>(),
                                                      schema->at("forbidden").get<std::vector<std::string>>());
            std::cout << gbnf;
            if (!gbnf.empty() && gbnf.back() != '\n') std::cout << "\n";
            return 0;
        }
        if (options.tool_value && options.forbidden_sequence.empty()) {
            std::cerr << "grammar_schema: --tool-value requires --forbid-sequence\n";
            return 1;
        }
        std::string gbnf = json_schema_to_grammar(*schema, /* force_gbnf= */ true, options);
        std::cout << gbnf;
        if (!gbnf.empty() && gbnf.back() != '\n') std::cout << "\n";
    } catch (const std::exception & e) {
        std::cerr << "grammar_schema: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
